"""Evaluate local JSONL tasks. Run: python -m alodlm.evaluate --help."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import torch

from .data import digest_file
from .generate import decode_config, load_decoder, model_arguments
from .scoring import METRICS, score_isolated


def load_tasks(filename):
    tasks, seen = [], set()
    with Path(filename).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["benchmark"], str(record["id"])
            if key in seen:
                raise ValueError("Duplicate benchmark and task ID")
            if record["metric"] not in METRICS:
                raise ValueError("Unsupported metric")
            if not isinstance(record.get("prompt"), str) or not record["prompt"]:
                raise ValueError("Each task needs a fully formatted prompt")
            if record["metric"] not in ("python_tests", "evalplus") and not str(record.get("answer", "")).strip():
                raise ValueError("Each scored task needs a reference answer")
            seen.add(key)
            tasks.append(record)
    if not tasks:
        raise ValueError("Evaluation input is empty")
    return tasks


def aggregate(results):
    groups = defaultdict(list)
    for result in results:
        groups[result["benchmark"]].append(result)
    summary = {}
    for benchmark, rows in sorted(groups.items()):
        seconds = sum(r["wall_seconds"] for r in rows)
        tokens = sum(r["generated_tokens"] for r in rows)
        if seconds <= 0:
            raise ValueError("Evaluation timing must be positive")
        summary[benchmark] = {
            "examples": len(rows), "correct": sum(r["correct"] for r in rows),
            "score_percent": 100 * sum(r["correct"] for r in rows) / len(rows),
            "wall_seconds": seconds, "generated_tokens": tokens,
            "tokens_per_second": tokens / seconds,
        }
    return {
        "benchmarks": summary, "benchmark_count": len(summary),
        "macro_score_percent": sum(r["score_percent"] for r in summary.values()) / len(summary),
        "suite_wall_seconds": sum(r["wall_seconds"] for r in summary.values()),
        "suite_tokens_per_second": sum(r["generated_tokens"] for r in summary.values())
                                  / sum(r["wall_seconds"] for r in summary.values()),
    }


def evaluate(args):
    tasks = load_tasks(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions_file = output / "predictions.jsonl"
    if predictions_file.exists() or (output / "summary.json").exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    config = decode_config(args)
    config.validate()
    if args.predictions:
        predictions = {}
        with Path(args.predictions).open() as handle:
            for line in handle:
                row = json.loads(line)
                key = row["benchmark"], str(row["id"])
                if key in predictions:
                    raise ValueError("Duplicate prediction")
                predictions[key] = row
        if set(predictions) != {(r["benchmark"], str(r["id"])) for r in tasks}:
            raise ValueError("Prediction IDs must match the complete task set")
        decoder = None
    else:
        decoder = load_decoder(args)
        warm_ids = decoder.tokenizer.encode(tasks[0]["prompt"], add_special_tokens=False)
        decoder.generate(warm_ids, config)
    results = []
    with predictions_file.open("x") as handle:
        for task in tasks:
            if decoder is None:
                result = dict(predictions[task["benchmark"], str(task["id"])])
                if (not math.isfinite(result["wall_seconds"]) or result["wall_seconds"] <= 0
                        or not isinstance(result["generated_tokens"], int) or result["generated_tokens"] < 0):
                    raise ValueError("Invalid timing or token count in prediction")
            else:
                ids = decoder.tokenizer.encode(task["prompt"], add_special_tokens=False)
                result = decoder.generate(ids, config)
            correct, extracted = score_isolated(task, result["text"])
            result.update(benchmark=task["benchmark"], id=task["id"], correct=correct, extracted=extracted)
            results.append(result)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
    summary = aggregate(results)
    versions = {}
    for package in ("torch", "transformers", "math-verify", "evalplus"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    summary["protocol"] = {
        "decoder": "pytorch-sdpa-reference" if decoder is not None else "supplied-predictions",
        "input_sha256": digest_file(args.input),
        "decode": asdict(config) if decoder is not None else None,
        "precision": args.precision if decoder is not None else None,
        "device": args.device if decoder is not None else None,
        "python": platform.python_version(), "versions": versions,
        "timing": ("prompt prefill plus decoding; excludes tokenization, warmup, and scoring"
                   if decoder is not None else
                   "copied from supplied predictions; generation protocol not verified"),
        "score_unit": "percent",
    }
    if args.predictions:
        summary["protocol"]["predictions_sha256"] = digest_file(args.predictions)
    if decoder is not None and args.device == "cuda":
        summary["protocol"]["gpu"] = torch.cuda.get_device_name()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser, required=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", help="Score existing local predictions without loading weights")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
