"""Offline benchmark scoring. Run: python -m alodlm.evaluate --help.

Modified from WeDLM answer and code extraction; see optimized/licenses/WeDLM.txt.
"""

import ast
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .extraction import ARCCEvaluator, GPQAEvaluator, MATHEvaluator, MMLUEvaluator


METRICS = {"arc", "mmlu", "mmlu_pro", "gpqa", "math", "exact", "python_tests", "evalplus"}


def extract_code(text, function_prefix=""):
    tail = text.rsplit("</think>", 1)[-1]
    fences = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", tail, re.DOTALL)
    code = next((block.strip() for block in reversed(fences) if "def " in block),
                fences[-1].strip() if fences else tail.strip())
    return code if "def " in code else function_prefix + "\n" + code


def math_answer(text):
    tail = str(text).rsplit("</think>", 1)[-1]
    position = tail.rfind("\\boxed")
    if position >= 0:
        start = tail.find("{", position)
        if start >= 0:
            depth = 0
            for i in range(start, len(tail)):
                depth += (tail[i] == "{") - (tail[i] == "}")
                if depth == 0:
                    value = tail[start + 1:i].strip()
                    if value:
                        return value
                    break
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return numbers[-1].replace(",", "") if numbers else ""


def math_correct(text, reference):
    from math_verify import parse, verify
    prediction = math_answer(text)
    reference = str(reference)
    if "####" in reference:
        gold = reference.rsplit("####", 1)[-1].strip().replace(",", "")
    elif "\\boxed" in reference:
        gold = MATHEvaluator()._extract_answer(reference)
    else:
        gold = reference.strip()
    if not prediction or not gold:
        return False, prediction
    try:
        equivalent = bool(verify(parse(f"${gold}$"), parse(f"${prediction}$")))
    except Exception:
        equivalent = False
    return equivalent or MATHEvaluator()._is_equiv(prediction, gold), prediction


def _python_tests(code, tests, timeout):
    if not isinstance(tests, str) or not tests.strip():
        raise ValueError("python_tests requires a nonempty local test program")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Code timeout must be finite and positive")
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "check.py"
        script.write_text(code + "\n\n" + tests + "\n")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=directory,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=timeout, check=False,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def _evalplus(code, record):
    from evalplus.eval import untrusted_check
    _check_evalplus()
    tests = record["code_tests"]
    dataset = tests["dataset"]
    if dataset not in ("humaneval", "mbpp"):
        raise ValueError("EvalPlus dataset must be humaneval or mbpp")
    entry = tests["entry_point"]
    suites = tests["suites"]
    if not suites:
        raise ValueError("An empty test suite cannot establish correctness")
    atol = float(tests.get("atol", 0.0))
    # The residual oracle accepts any valid polynomial root.
    if dataset == "humaneval" and entry == "find_zero":
        code += (
            "\nfrom evalplus.eval._special_oracle import _poly as _alodlm_poly\n"
            "def _alodlm_root_check(*args):\n"
            f"    return abs(_alodlm_poly(*args, find_zero(*args))) <= {atol!r}\n"
        )
        entry = "_alodlm_root_check"
    for suite in suites:
        inputs = ast.literal_eval(suite["inputs_literal"]) if "inputs_literal" in suite else suite["inputs"]
        expected = ast.literal_eval(suite["expected_literal"]) if "expected_literal" in suite else suite["expected"]
        times = suite["reference_seconds"]
        if not len(inputs) == len(expected) == len(times) or not len(inputs):
            raise ValueError("Test inputs, expected outputs, and reference times must align")
        if any(not math.isfinite(t) or t < 0 for t in times):
            raise ValueError("Reference execution times must be finite and nonnegative")
        if dataset == "humaneval" and tests["entry_point"] == "find_zero":
            expected = [True] * len(inputs)
        status, _ = untrusted_check(
            dataset, code, inputs, entry, expected, atol, times,
            min_time_limit=float(tests.get("min_time_limit", 1.0)),
            gt_time_limit_factor=float(tests.get("time_limit_factor", 4.0)),
        )
        if status != "pass":
            return False
    return True


@lru_cache(maxsize=1)
def _check_evalplus():
    from evalplus.eval import untrusted_check
    if sys.platform == "darwin":
        os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")
    positive, _ = untrusted_check("humaneval", "def identity(x):\n    return x",
                                  [[3]], "identity", [3], 0, [.001])
    negative, _ = untrusted_check("humaneval", "def identity(x):\n    return 0",
                                  [[3]], "identity", [3], 0, [.001])
    if positive != "pass" or negative != "fail":
        raise RuntimeError("Code grader failed its positive/negative execution checks")


def score(record, generation):
    metric = record["metric"]
    if metric not in METRICS:
        raise ValueError("Unsupported evaluation metric")
    reference = str(record.get("answer", "")).strip()
    if metric in ("python_tests", "evalplus"):
        code = extract_code(generation, record.get("function_prefix", ""))
        result = (_evalplus(code, record) if metric == "evalplus"
                  else _python_tests(code, record["tests"], float(record.get("timeout_seconds", 10))))
        return bool(result), None
    if not reference:
        raise ValueError("An evaluation reference is required")
    if metric == "exact":
        return generation.strip() == reference, generation.strip()
    if metric == "math":
        return math_correct(generation, reference)
    options = record.get("available_options", list("ABCDEFGHIJ" if metric == "mmlu_pro" else "ABCD"))
    if reference.upper() not in options:
        raise ValueError("Gold answer is outside the displayed option labels")
    if metric == "arc":
        prediction = ARCCEvaluator()._extract_answer(generation, options)
    elif metric == "mmlu":
        prediction = MMLUEvaluator()._extract_answer(generation)
    elif metric == "gpqa":
        prediction = GPQAEvaluator()._extract_answer(generation)
    else:
        match = re.search(r"(?i)ANSWER\s*:\s*([A-P])", generation)
        if match:
            prediction = match.group(1)
        else:
            matches = list(re.finditer(r"answer is[:\s]*\*{0,2}\(?([A-P])\)?", generation, re.I))
            matches = matches or list(re.finditer(r"\b([A-J])\b", generation.strip()[-30:]))
            prediction = matches[-1].group(1) if matches else ""
    return prediction.upper() == reference.upper(), prediction


def score_isolated(record, generation):
    """Keep code execution outside the model process and its CUDA address space."""
    if record["metric"] not in ("python_tests", "evalplus"):
        return score(record, generation)
    timeout = (30 + 70 * len(record["code_tests"]["suites"]) if record["metric"] == "evalplus"
               else 30 + float(record.get("timeout_seconds", 10)))
    result = subprocess.run(
        [sys.executable, "-m", "alodlm.scoring"],
        input=json.dumps({"record": record, "generation": generation}),
        text=True, capture_output=True, check=False, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError("Isolated code scoring failed: " + result.stderr)
    return tuple(json.loads(result.stdout))


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    print(json.dumps(score(payload["record"], payload["generation"])))
