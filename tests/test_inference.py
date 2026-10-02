"""Check cache completion and adaptive decoding. Run: python -m unittest discover -s tests."""

import unittest

import torch

from alodlm.inference import DecodeConfig, Decoder, PrefixCache
from helpers import tiny_model, tiny_tokenizer


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cache_completion_preserves_computed_depths(self):
        model = tiny_model()
        for completed in range(1, 5):
            cache = PrefixCache(model.config)
            window = {(1, d): (torch.full((3, 2, 8), float(d)),
                               torch.full((3, 2, 8), float(d + 10))) for d in range(completed)}
            window[0, 0] = torch.ones(3, 2, 8), torch.ones(3, 2, 8)
            cache.complete(window, completed)
            for d in range(4):
                self.assertEqual(float(window[1, d][0][0, 0, 0]), min(d, completed - 1))
            cache.append(window, torch.tensor([2, 0]))
            self.assertEqual(cache.length, 2)
            for key in window:
                torch.testing.assert_close(cache.values[key][0], window[key][0][[2, 0]])

    def test_generation_is_request_independent_and_respects_budget(self):
        decoder = Decoder(tiny_model(), tiny_tokenizer())
        cfg = DecodeConfig(q=.5, tau=.1, max_new_tokens=8, window_size=4)
        first = decoder.generate([4, 7, 9], cfg)
        decoder.generate([4, 7, 10, 11], DecodeConfig(q=.9, max_new_tokens=7, mode="left1"))
        second = decoder.generate([4, 7, 9], cfg)
        self.assertEqual(first["token_ids"], second["token_ids"])
        self.assertEqual(first["exit_depths"], second["exit_depths"])
        self.assertLessEqual(first["generated_tokens"], 8)
        self.assertEqual(len(first["exit_depths"]), len(first["token_ids"]))
        self.assertGreater(first["wall_seconds"], 0)
        self.assertGreaterEqual(first["wall_seconds"], first["prefill_seconds"])

    def test_q_controls_executed_recurrence(self):
        decoder = Decoder(tiny_model(), tiny_tokenizer())
        shallow = decoder.generate([4, 7, 9], DecodeConfig(q=0, tau=0, max_new_tokens=6, window_size=4))
        deep = decoder.generate([4, 7, 9], DecodeConfig(q=1, tau=0, max_new_tokens=6, window_size=4))
        self.assertEqual(shallow["recurrent_passes"], shallow["outer_steps"])
        self.assertEqual(deep["recurrent_passes"], 4 * deep["outer_steps"])

    def test_prefix_cache_matches_observed_token_forward(self):
        # With one recurrent pass, cached causal decoding must have the same
        # prefix keys as a fresh forward over the actual generated tokens.
        model = tiny_model()
        model.config.max_depth = 1
        checked = []

        class CheckingDecoder(Decoder):
            known = {}

            def _window(inner, ids, positions, cache, mask_rows, config, logical):
                for token, position in zip(ids.tolist(), positions.tolist()):
                    if token != model.config.mask_token_id:
                        inner.known[position] = token
                if cache.length > 3:
                    prefix_ids = torch.tensor([inner.known[i] for i in range(cache.length)])
                    empty = torch.empty(0, dtype=torch.long)
                    expected = super()._window(
                        prefix_ids, torch.arange(cache.length), PrefixCache(model.config),
                        empty, config, empty)[0]
                    for key in expected:
                        for actual, reference in zip(cache.values[key], expected[key]):
                            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
                    checked.append(cache.length)
                result = super()._window(ids, positions, cache, mask_rows, config, logical)
                for row, agreed, token in zip(mask_rows, result[1], result[2]):
                    if agreed:
                        inner.known[int(positions[row])] = int(token)
                return result

        result = CheckingDecoder(model, tiny_tokenizer()).generate(
            [4, 7, 9], DecodeConfig(q=0, mode="left1", max_new_tokens=8, window_size=4))
        self.assertEqual(result["generated_tokens"], 8)
        self.assertGreaterEqual(len(checked), 3)

    def test_committed_token_continues_as_latent_after_injection(self):
        from unittest.mock import patch

        model = tiny_model()
        cfg = DecodeConfig(q=1, tau=0, mode="left1", max_new_tokens=4)
        core_inputs = []

        def record_layer(layer, hidden, cos, sin, mask, prefix=None):
            if layer is model.backbone.model.layers[1]:
                core_inputs.append(hidden.clone())
            # Nonconstant transformation lets the test distinguish a recurrent
            # latent from an embedding that was incorrectly reset each pass.
            result = hidden + torch.arange(hidden.shape[-1], dtype=hidden.dtype) / 32
            kv = hidden.unsqueeze(1), hidden.unsqueeze(1)
            return result, kv

        ids = torch.full((4,), model.config.mask_token_id)
        rows = torch.arange(4)
        with torch.inference_mode(), patch("alodlm.inference.layer_forward", record_layer):
            Decoder(model, tiny_tokenizer())._window(
                ids, rows, PrefixCache(model.config), rows, cfg, rows)
        self.assertEqual(len(core_inputs), 4)
        # The first token commits on pass one and gets an embedding at pass two.
        # Pass three must carry its transformed, normalized state forward.
        self.assertFalse(torch.equal(core_inputs[1][0], core_inputs[2][0]))
        expected = model.backbone.model.norm(
            core_inputs[1] + torch.arange(32, dtype=core_inputs[1].dtype) / 32)
        torch.testing.assert_close(core_inputs[2][0], expected[0])

    def test_invalid_decode_config_and_context_fail(self):
        decoder = Decoder(tiny_model(), tiny_tokenizer())
        for cfg in (DecodeConfig(q=1.1), DecodeConfig(tau=float("nan")),
                    DecodeConfig(window_size=0)):
            with self.assertRaises(ValueError):
                decoder.generate([4, 7], cfg)
        with self.assertRaises(ValueError):
            decoder.generate([4] * 100, DecodeConfig(max_new_tokens=40))


if __name__ == "__main__":
    unittest.main()
