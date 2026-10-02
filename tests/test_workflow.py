"""Test the local training-to-evaluation workflow. Run: python -m unittest discover -s tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml

from alodlm.data import PackedDataset, load_tokenizer, prepare_cache, tokenize_messages
from alodlm.model import ALoDLM
from helpers import write_fixture


class WorkflowTests(unittest.TestCase):
    def test_offline_train_resume_generate_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_fixture(root)
            guard = root / "guard"
            guard.mkdir()
            (guard / "sitecustomize.py").write_text(
                '"""Disable network connections during offline execution checks."""\n'
                "import socket\n"
                "def blocked(*args, **kwargs):\n"
                "    raise RuntimeError('Network access is disabled during this test')\n"
                "socket.socket.connect = blocked\n"
                "socket.socket.connect_ex = blocked\n"
                "socket.create_connection = blocked\n"
            )
            package = Path(__file__).resolve().parents[1]
            env = dict(os.environ, PYTHONPATH=str(guard) + os.pathsep + str(package),
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       TOKENIZERS_PARALLELISM="false")
            def run(module, *args):
                result = subprocess.run([sys.executable, "-m", module, *args],
                                        cwd=root, env=env, text=True, capture_output=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            path = root / "train.yaml"
            path.write_text(yaml.safe_dump(config))
            run("alodlm.train", "--config", str(path))
            final = root / "run/checkpoint-4"
            self.assertTrue((final / "trainer_state.json").is_file())
            saved = ALoDLM.from_pretrained(final)
            self.assertTrue(all(torch.isfinite(t).all() for t in saved.state_dict().values()))
            self.assertGreater(float(saved.exit_gate.net.weight.detach().norm()), 0)
            original = {key: value.clone() for key, value in saved.state_dict().items()}
            config["resume"] = str(root / "run/checkpoint-2")
            config["output_dir"] = str(root / "resumed")
            path.write_text(yaml.safe_dump(config))
            run("alodlm.train", "--config", str(path))
            resumed = ALoDLM.from_pretrained(root / "resumed/checkpoint-4")
            for key, value in resumed.state_dict().items():
                torch.testing.assert_close(value, original[key], atol=0, rtol=0, msg=key)

            text = run("alodlm.generate", "--model", str(final), "--prompt", "Return value one",
                       "--max-new-tokens", "6", "--window-size", "4", "--device", "cpu")
            generation = json.loads(text)
            self.assertLessEqual(generation["generated_tokens"], 6)
            tasks = root / "evaluation.jsonl"
            tasks.write_text(json.dumps({"id": "1", "benchmark": "synthetic", "metric": "exact",
                                         "prompt": "user Return value one </s> assistant ",
                                         "answer": "one ."}) + "\n")
            output = root / "evaluated"
            run("alodlm.evaluate", "--model", str(final), "--input", str(tasks),
                "--output", str(output), "--max-new-tokens", "6", "--window-size", "4", "--device", "cpu")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["benchmark_count"], 1)
            self.assertEqual(summary["benchmarks"]["synthetic"]["examples"], 1)
            rescored = root / "rescored"
            run("alodlm.evaluate", "--input", str(tasks), "--output", str(rescored),
                "--predictions", str(output / "predictions.jsonl"))
            rescored_summary = json.loads((rescored / "summary.json").read_text())
            self.assertEqual(rescored_summary["macro_score_percent"], summary["macro_score_percent"])
            self.assertEqual(rescored_summary["suite_wall_seconds"], summary["suite_wall_seconds"])
            self.assertIsNone(rescored_summary["protocol"]["decode"])
            self.assertIsNone(rescored_summary["protocol"]["device"])
            self.assertIsNone(rescored_summary["protocol"]["precision"])

    def test_cache_binds_data_and_supervises_only_last_assistant(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_fixture(directory)
            tokenizer = load_tokenizer(config["model_dir"])
            filename = prepare_cache(config["train_file"], tokenizer, config["cache_file"], 24, 17)
            dataset = PackedDataset(filename)
            self.assertGreater(len(dataset), 0)
            for pack in dataset:
                self.assertEqual(int(pack["boundaries"][-1]), len(pack["ids"]))
                self.assertLessEqual(len(pack["ids"]), 24)
            with Path(config["train_file"]).open("a") as handle:
                handle.write("\n")
            with self.assertRaises(ValueError):
                prepare_cache(config["train_file"], tokenizer, config["cache_file"], 24, 17)

    def test_multiturn_labels_and_overlength_filter(self):
        from helpers import tiny_tokenizer
        tokenizer = tiny_tokenizer()
        messages = [
            {"role": "user", "content": "Return value one"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "Return value three"},
            {"role": "assistant", "content": "three"},
        ]
        ids, labels = tokenize_messages(messages, tokenizer, 64)
        self.assertTrue(bool((labels[ids == tokenizer.convert_tokens_to_ids("one")] == -100).all()))
        self.assertIn(tokenizer.convert_tokens_to_ids("three"), labels.tolist())
        self.assertIsNone(tokenize_messages(messages, tokenizer, 4))
        _, labels = tokenize_messages(messages[:1], tokenizer, 64)
        self.assertTrue(bool((labels == -100).all()))

    def test_incomplete_conversations_are_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_fixture(directory)
            tokenizer = load_tokenizer(config["model_dir"])
            valid = [
                {"role": "user", "content": "Return value one"},
                {"role": "assistant", "content": "one"},
            ]
            invalid = valid + [{"role": "user", "content": "Return value three"}]
            with self.assertRaisesRegex(ValueError, "assistant"):
                tokenize_messages(invalid, tokenizer, 64)
            Path(config["train_file"]).write_text(
                json.dumps(valid) + "\n" + json.dumps(invalid) + "\n")
            cache = prepare_cache(config["train_file"], tokenizer, config["cache_file"], 64, 17)
            dataset = PackedDataset(cache)
            self.assertEqual(dataset.cache["filtered_records"], 1)
            ids, labels = tokenize_messages(valid, tokenizer, 64)
            torch.testing.assert_close(dataset[0]["ids"], ids.long())
            torch.testing.assert_close(dataset[0]["labels"], labels.long())


if __name__ == "__main__":
    unittest.main()
