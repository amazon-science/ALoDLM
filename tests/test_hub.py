"""Check inference downloads without contacting a model hub."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from alodlm.hub import resolve_model


class HubTests(unittest.TestCase):
    def test_import_respects_hub_network_mode(self):
        code = (
            "import json; import alodlm.generate; "
            "from huggingface_hub.constants import HF_HUB_OFFLINE; "
            "from transformers.utils import is_offline_mode; "
            "print(json.dumps([HF_HUB_OFFLINE, is_offline_mode()]))"
        )
        for setting, expected in ((None, False), ("0", False), ("1", True)):
            with self.subTest(setting=setting):
                environment = dict(os.environ)
                for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                    if setting is None:
                        environment.pop(key, None)
                    else:
                        environment[key] = setting
                result = subprocess.run(
                    [sys.executable, "-c", code], env=environment, text=True,
                    capture_output=True, check=True, timeout=60)
                self.assertEqual(json.loads(result.stdout), [expected, expected])

    def test_local_directory_stays_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download") as download:
                self.assertEqual(resolve_model(directory), Path(directory).resolve())
                download.assert_not_called()

    def test_missing_absolute_directory_is_not_sent_to_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download") as download:
                with self.assertRaisesRegex(ValueError, "local model directory"):
                    resolve_model(str(Path(directory) / "missing"))
                download.assert_not_called()

    def test_hub_download_selects_inference_files_only(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download", return_value=directory) as download:
                self.assertEqual(resolve_model("example/model"), Path(directory))
                arguments = download.call_args.kwargs
                self.assertEqual(arguments["repo_id"], "example/model")
                patterns = arguments["allow_patterns"]
                self.assertIn("exit_gate.pt", patterns)
                self.assertIn("alodlm_config.json", patterns)
                self.assertIn("*.safetensors", patterns)
                self.assertNotIn("*.pt", patterns)
                self.assertNotIn("*", patterns)


if __name__ == "__main__":
    unittest.main()
