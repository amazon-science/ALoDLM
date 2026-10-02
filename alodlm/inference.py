"""Cached adaptive decoding. Run: python -m alodlm.generate --help.

Modified from the WeDLM recurrence and decoding algorithms; see optimized/licenses/WeDLM.txt.
"""

from dataclasses import dataclass
import time

import torch

from .model import layer_forward


@dataclass
class DecodeConfig:
    q: float = 0.5
    tau: float = 0.4
    window_size: int = 16
    max_new_tokens: int = 4096
    position_penalty: float | None = None
    mode: str = "entropy"

    def validate(self):
        if not 0 <= self.q <= 1 or not 0 <= self.tau < float("inf"):
            raise ValueError("q must be in [0,1] and tau must be finite and nonnegative")
        if min(self.window_size, self.max_new_tokens) < 1:
            raise ValueError("Window size and generation budget must be positive")
        if self.mode not in ("entropy", "left1"):
            raise ValueError("mode must be entropy or left1")
        if self.position_penalty is not None and not 0 <= self.position_penalty < float("inf"):
            raise ValueError("position_penalty must be finite and nonnegative")


class PrefixCache:
    """Recurrent layers have depth-specific keys; other layers share one depth."""
    def __init__(self, config):
        self.config = config
        self.values = {}
        self.length = 0

    def key(self, layer, depth):
        c = self.config
        return layer, depth if c.loop_start <= layer < c.loop_end else 0

    def complete(self, window, completed):
        """Fill only unexecuted recurrent depths from the last computed depth."""
        c = self.config
        for layer in range(c.loop_start, c.loop_end):
            last = window[layer, completed - 1]
            for depth in range(completed, c.max_depth):
                window[layer, depth] = last

    def append(self, window, rows):
        if rows.numel() == 0:
            return
        for key, (k, v) in window.items():
            selected = k[rows].clone(), v[rows].clone()
            previous = self.values.get(key)
            self.values[key] = selected if previous is None else (
                torch.cat((previous[0], selected[0]), 0),
                torch.cat((previous[1], selected[1]), 0),
            )
        self.length += len(rows)


class Decoder:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

    def _window(self, ids, positions, cache, mask_rows, config, logical_mask_positions):
        base, c = self.model.backbone.model, self.model.config
        hidden = base.embed_tokens(ids)
        cos, sin = base.rotary_emb(hidden.unsqueeze(0), positions.unsqueeze(0))
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        length = len(ids)
        attention = torch.ones(length, cache.length + length, dtype=torch.bool, device=ids.device)
        attention[:, cache.length:] = torch.ones(
            length, length, dtype=torch.bool, device=ids.device).tril()
        window = {}

        def run(layer_index, h, depth):
            key = cache.key(layer_index, depth)
            h, kv = layer_forward(base.layers[layer_index], h, cos, sin,
                                  attention, cache.values.get(key))
            window[key] = kv
            return h

        for layer in range(c.loop_start):
            hidden = run(layer, hidden, 0)
        count = len(mask_rows)
        committed = torch.zeros(count, dtype=torch.bool, device=ids.device)
        newly_committed = committed.clone()
        tokens = torch.zeros(count, dtype=torch.long, device=ids.device)
        exit_depths = torch.zeros(count, dtype=torch.long, device=ids.device)
        survival = torch.ones(count, device=ids.device)
        first_hazard = torch.empty(count, device=ids.device)
        penalty = config.position_penalty if config.position_penalty is not None else config.tau * 0.05
        relative = logical_mask_positions - logical_mask_positions[:1] if count else logical_mask_positions
        for depth in range(c.max_depth):
            if depth and count:
                replacement = base.embed_tokens(tokens)
                injected = hidden[mask_rows]
                hidden = hidden.clone()
                hidden[mask_rows] = torch.where(newly_committed[:, None], replacement, injected)
            for layer in range(c.loop_start, c.loop_end):
                hidden = run(layer, hidden, depth)
            readout = hidden
            for layer in range(c.loop_end, len(base.layers)):
                readout = run(layer, readout, depth)
            features = base.norm(readout)
            stop = depth == c.max_depth - 1
            if count:
                selected = features[mask_rows]
                logits = self.model.backbone.lm_head(selected).float()
                predicted = logits.argmax(-1)
                logp = logits.log_softmax(-1)
                entropy = -(logp.exp() * logp).sum(-1)
                hazard = self.model.exit_gate(selected, depth).float().sigmoid()
                if depth == 0:
                    first_hazard = hazard.clone()
                survival *= 1 - hazard
                adjusted = entropy + relative * penalty
                if config.mode == "left1":
                    agree = torch.zeros_like(committed)
                    remaining = (~committed).nonzero().flatten()
                    if len(remaining):
                        agree[remaining[0]] = True
                else:
                    agree = (adjusted < config.tau) & ~committed
                tokens = torch.where(agree, predicted, tokens)
                exit_depths = torch.where(agree, depth + 1, exit_depths)
                committed |= agree
                newly_committed = agree
                residual = ~committed
                stop |= not bool(residual.any()) or bool((1 - survival)[residual].mean() >= config.q)
                if stop and not bool(committed.any()):
                    index = adjusted.argmin()
                    tokens[index] = predicted[index]
                    exit_depths[index] = depth + 1
                    committed[index] = True
            if stop:
                cache.complete(window, depth + 1)
                return window, committed, tokens, exit_depths, depth + 1, first_hazard
            hidden = base.norm(hidden)
        raise RuntimeError("Recurrent decoding did not terminate")

    @torch.inference_mode()
    def generate(self, prompt_ids, config=None):
        config = config or DecodeConfig()
        config.validate()
        device = self.model.backbone.device
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        if not len(prompt):
            raise ValueError("Prompt must contain at least one token")
        context = self.model.backbone.config.max_position_embeddings
        if len(prompt) + config.max_new_tokens > context:
            raise ValueError("Prompt and generation budget exceed the model context")
        if bool(((prompt < 0) | (prompt >= self.model.backbone.config.vocab_size)).any()):
            raise ValueError("Prompt token is outside the model vocabulary")
        eos = self.tokenizer.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
            if token in self.tokenizer.get_vocab():
                stop_ids.add(self.tokenizer.convert_tokens_to_ids(token))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        cache = PrefixCache(self.model.config)
        empty = torch.empty(0, dtype=torch.long, device=device)
        prefill = self._window(prompt, torch.arange(len(prompt), device=device),
                               cache, empty, config, empty)[0]
        cache.append(prefill, torch.arange(len(prompt), device=device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_seconds = time.perf_counter() - started
        generated, depths, hazards = [], [], []
        tokens, flags, window_depths, window_hazards = [], [], [], []
        steps, passes, loop_token_passes = 0, 0, 0
        stop_reason = "length"
        while len(generated) < config.max_new_tokens:
            # Refresh the previously committed prefix as observed tokens before
            # retaining its K/V. Newly predicted mask rows are not prefix keys.
            prefix = next((i for i, masked in enumerate(flags) if masked), len(flags))
            finished = (len(generated) + prefix >= config.max_new_tokens
                        or any(t in stop_ids for t in tokens[:prefix]))
            if finished:
                for i in range(prefix):
                    if tokens[i] in stop_ids:
                        stop_reason = "eos"
                        break
                    generated.append(tokens[i])
                    depths.append(window_depths[i])
                    hazards.append(window_hazards[i])
                break
            desired = min(config.window_size,
                          config.max_new_tokens - len(generated) - prefix)
            grow = max(0, prefix + desired - len(tokens))
            tokens.extend([self.model.config.mask_token_id] * grow)
            flags.extend([True] * grow)
            window_depths.extend([0] * grow)
            window_hazards.extend([0.0] * grow)
            observed = [i for i, masked in enumerate(flags) if not masked]
            masked = [i for i, is_mask in enumerate(flags) if is_mask]
            order = observed + masked
            order_tensor = torch.tensor(order, device=device)
            ids = torch.tensor([tokens[i] for i in order], dtype=torch.long, device=device)
            positions = cache.length + order_tensor
            mask_rows = torch.arange(len(observed), len(order), device=device)
            window, agreed, predicted, committed_depth, executed, halt = self._window(
                ids, positions, cache, mask_rows, config, torch.tensor(masked, device=device))
            steps += 1
            passes += executed
            loop_token_passes += len(order) * executed
            inverse = torch.argsort(order_tensor)
            cache.append(window, inverse[:prefix])
            generated.extend(tokens[:prefix])
            depths.extend(window_depths[:prefix])
            hazards.extend(window_hazards[:prefix])
            for j, original in enumerate(masked):
                if bool(agreed[j]):
                    tokens[original] = int(predicted[j])
                    flags[original] = False
                    window_depths[original] = int(committed_depth[j])
                    window_hazards[original] = float(halt[j])
            tokens, flags = tokens[prefix:], flags[prefix:]
            window_depths, window_hazards = window_depths[prefix:], window_hazards[prefix:]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        count = len(generated)
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "token_ids": generated, "prompt_tokens": len(prompt), "generated_tokens": count,
            "wall_seconds": elapsed, "prefill_seconds": prefill_seconds,
            "tokens_per_second": count / elapsed,
            "mean_exit_depth": sum(depths) / count if count else None,
            "loop_token_passes": loop_token_passes,
            "loop_token_passes_per_output_token": loop_token_passes / count if count else None,
            "outer_steps": steps, "recurrent_passes": passes, "exit_depths": depths,
            "first_pass_halt_probabilities": hazards, "stop_reason": stop_reason,
        }
