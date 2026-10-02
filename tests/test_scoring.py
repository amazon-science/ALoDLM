"""Check answer extraction and local execution metrics. Run: python -m unittest discover -s tests."""

import unittest

from alodlm.evaluate import aggregate
from alodlm.scoring import score, score_isolated


class ScoringTests(unittest.TestCase):
    def test_multiple_choice_conclusions(self):
        cases = [
            ("arc", "Based on the calculation, the answer is (C).", "C", True),
            ("arc", "Based on the calculation, the answer is (C).", "B", False),
            ("mmlu", "<think>The answer is A</think>The answer is (D).", "D", True),
            ("gpqa", "<think>B</think>#### C", "C", True),
            ("gpqa", "No option selected.", "D", False),
            ("mmlu_pro", "The answer is **(J)**.", "J", True),
            ("mmlu_pro", "ANSWER: F", "F", True),
        ]
        for metric, text, answer, expected in cases:
            with self.subTest(metric=metric, text=text):
                self.assertEqual(score({"metric": metric, "answer": answer}, text)[0], expected)

    def test_math_equivalence_and_final_answer(self):
        for text, answer, expected in [
            (r"<think>3</think>\boxed{\frac{1}{2}}", "0.5", True),
            (r"\boxed{1,024}", "#### 1024", True),
            (r"\boxed{41}", "42", False),
            ("No result.", "0", False),
        ]:
            self.assertEqual(score({"metric": "math", "answer": answer}, text)[0], expected)

    def test_local_python_suite(self):
        task = {"metric": "python_tests", "tests": "assert identity(3) == 3"}
        self.assertTrue(score(task, "```python\ndef identity(x):\n    return x\n```")[0])
        self.assertFalse(score(task, "def identity(x):\n    return 0")[0])
        task["timeout_seconds"] = .2
        self.assertFalse(score(task, "def identity(x):\n    while True: pass")[0])

    def test_evalplus_root_and_standard_oracles(self):
        task = {"metric": "evalplus", "code_tests": {
            "dataset": "humaneval", "entry_point": "find_zero", "atol": .0001,
            "suites": [{"inputs": [[[-10, -2]]], "expected": [-5.],
                        "reference_seconds": [.001]}],
        }}
        self.assertTrue(score(task, "def find_zero(xs):\n    return -xs[0] / xs[1]")[0])
        self.assertFalse(score(task, "def find_zero(xs):\n    return 0")[0])
        task["code_tests"].update(entry_point="identity", atol=0,
                                 suites=[{"inputs": [[3]], "expected": [3], "reference_seconds": [.001]}])
        self.assertTrue(score(task, "def identity(x):\n    return x")[0])
        self.assertFalse(score(task, "def identity(x):\n    return 1")[0])

    def test_macro_average_is_over_benchmarks(self):
        rows = [
            {"benchmark": "a", "correct": True, "wall_seconds": 2, "generated_tokens": 10},
            {"benchmark": "b", "correct": False, "wall_seconds": 3, "generated_tokens": 20},
            {"benchmark": "b", "correct": False, "wall_seconds": 5, "generated_tokens": 30},
        ]
        result = aggregate(rows)
        self.assertEqual(result["macro_score_percent"], 50)
        self.assertEqual(result["suite_tokens_per_second"], 6)
        self.assertEqual(result["benchmark_count"], 2)

    def test_isolated_code_scorer(self):
        record = {"metric": "python_tests", "tests": "assert identity(3) == 3"}
        self.assertTrue(score_isolated(record, "def identity(x):\n    return x")[0])
        self.assertFalse(score_isolated(record, "def identity(x):\n    return 0")[0])


if __name__ == "__main__":
    unittest.main()
