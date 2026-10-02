"""CPU checks for model-directory compatibility, without importing GPU libraries."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from alodlm_optimized.cli import install_rope_compatibility, main, read_model_metadata


class ModelDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Metadata fixtures only; these files are never loaded as tensors.
        (self.root / "model.safetensors").touch()
        (self.root / "exit_gate.pt").touch()

    def write_metadata(self, layers=36, start=10, end=26, **updates):
        config = {
            "model_type": "qwen3", "num_hidden_layers": layers,
            "max_position_embeddings": 40960, "tie_word_embeddings": False,
        }
        config.update(updates)
        recurrence = {"loop_start": start, "loop_end": end, "max_depth": 4}
        (self.root / "config.json").write_text(json.dumps(config))
        (self.root / "alodlm_config.json").write_text(json.dumps(recurrence))
        return recurrence, config

    def test_accepts_both_recurrence_topologies(self):
        for layers, start, end in ((36, 10, 26), (28, 0, 28)):
            with self.subTest(layers=layers):
                expected = self.write_metadata(layers, start, end)
                self.assertEqual(read_model_metadata(self.root), expected)

    def test_missing_gate_and_missing_weights_are_rejected(self):
        self.write_metadata()
        for filename in ("exit_gate.pt", "model.safetensors"):
            with self.subTest(filename=filename):
                path = self.root / filename
                path.unlink()
                with self.assertRaisesRegex(ValueError, "missing|safetensors"):
                    read_model_metadata(self.root)
                path.touch()

    def test_incompatible_backbone_or_recurrence_is_rejected(self):
        for settings in (
            {"tie_word_embeddings": True},
            {"model_type": "qwen2"},
            {"end": 37},
            {"start": -1},
            {"start": 26},
            {"start": False},
        ):
            with self.subTest(settings=settings):
                self.write_metadata(**settings)
                with self.assertRaises(ValueError):
                    read_model_metadata(self.root)

    def test_invalid_decode_arguments_fail_before_gpu_import(self):
        for arguments in (
            ["--q", "nan"],
            ["--tau", "inf"],
            ["--max-new-tokens", "0"],
            ["--gpu-memory-utilization", "1"],
            ["--nccl-port", "65536"],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(["--prompt", "Hello", *arguments])
                self.assertEqual(error.exception.code, 2)

    def test_default_rope_reads_theta_from_either_config_format(self):
        models = types.ModuleType("wedlm.models")
        models.wedlm = types.SimpleNamespace()
        rotary = types.ModuleType("wedlm.layers.rotary_embedding")
        rotary.get_rope = Mock()
        with patch.dict("sys.modules", {
            "wedlm.models": models,
            "wedlm.layers.rotary_embedding": rotary,
        }):
            install_rope_compatibility()
        adapter = models.wedlm.get_rope
        adapter(128, 128, 40960, 1000000)
        rotary.get_rope.assert_called_with(128, 128, 40960, 1000000, None)
        adapter(128, 128, 40960, 10000,
                {"rope_type": "default", "rope_theta": 1000000})
        rotary.get_rope.assert_called_with(128, 128, 40960, 1000000, None)
        for unsupported in ({"rope_type": "yarn"}, {"rope_type": "default", "factor": 2}):
            with self.assertRaises(ValueError):
                adapter(128, 128, 40960, 1000000, unsupported)


if __name__ == "__main__":
    unittest.main()
