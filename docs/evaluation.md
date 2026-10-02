# Evaluation guide

[ALoDLM](../README.md)

Install the [portable package with evaluation dependencies](../README.md#portable-decoder-and-training-environment)
before running these commands.

`--model` accepts either a complete local model directory or a Hugging Face
model ID; the command downloads inference files when needed. Evaluation
prompts and references are supplied locally.

Supply a JSONL file containing fully formatted prompts, including any chat
markers, few-shot context, answer instructions, and reference labels. Prompts
are tokenized verbatim without adding special tokens or another chat template.

```json
{"benchmark":"<BENCHMARK>","id":"<ID>","metric":"arc","prompt":"<FORMATTED_PROMPT>","answer":"<LETTER>","available_options":["A","B","C","D"]}
```

Use the following metric mappings:

| Benchmark | Metric | Required reference |
|---|---|---|
| ARC-C, ARC-E | `arc` | Answer letter and available option labels |
| MMLU | `mmlu` | Answer letter |
| MMLU-Pro | `mmlu_pro` | Answer letter and available option labels |
| GPQA-Diamond | `gpqa` | Answer letter matching the displayed option order |
| GSM8K, MATH-500 | `math` | Numeric or LaTeX answer |
| MBPP | `python_tests` | Local test program, including any setup code |
| MBPP+ | `evalplus` | Base and extended tests, expected outputs, reference timings |
| HumanEval | `evalplus` | Base tests, expected outputs, reference timings |
| HumanEval+ | `evalplus` | Base and extended tests, expected outputs, reference timings |

Mathematical scoring uses final-answer extraction, mathematical equivalence,
and normalized comparison. Multiple-choice scorers handle final-answer cues;
failed extraction counts as incorrect. Gold labels must correspond to the
provided prompts, including any option permutation.

For `python_tests`, set `tests` to the complete Python test program.
`timeout_seconds` defaults to 10. Code extraction prefers the last fenced
function after a thinking section. A response containing a full function is
scored as that function; it is not prepended with another signature.
`function_prefix` can supply a signature when a response contains only a body.

For `evalplus`, add a `code_tests` object:

```json
{"dataset":"humaneval","entry_point":"<FUNCTION>","atol":0.0,"suites":[{"inputs":[["<ARGUMENT>"]],"expected":["<EXPECTED_OUTPUT>"],"reference_seconds":[0.001]}]}
```

Each input is an argument list. Include all official base tests for a base
score, and both base and extended suites for a plus score. Provide expected
outputs and reference execution times measured from the trusted reference
implementation. `inputs_literal` and `expected_literal` accept Python literal
lists when tuples or sets must be preserved; they are parsed with
`ast.literal_eval`. MBPP special-oracle expected outputs must follow EvalPlus
conventions. Polynomial-root tasks use residual checking and count each
successful test explicitly.

Code evaluation executes generated programs in child processes. Run untrusted
model outputs in an isolated environment; process timeouts are not a security
sandbox. EvalPlus applies its resource limits on supported systems; its memory
cap is disabled on macOS because that limit interface is unsupported.

```bash
alodlm-evaluate \
  --model "<LOCAL_CHECKPOINT_DIRECTORY>" \
  --input "<LOCAL_EVALUATION_JSONL>" \
  --output outputs/evaluation \
  --device cuda --precision bf16 \
  --q 0.5 --tau 0.4 --max-new-tokens 4096
```

To score existing local predictions without loading a model:

```bash
alodlm-evaluate \
  --input "<LOCAL_EVALUATION_JSONL>" \
  --predictions "<LOCAL_PREDICTIONS_JSONL>" \
  --output outputs/rescored
```

Each prediction needs `benchmark`, `id`, `text`, `generated_tokens`, and
`wall_seconds`. The prediction keys must match the complete task set.
Rescoring preserves the supplied token counts and timing. Its summary records
the prediction file's SHA-256 hash and leaves decoding, device, and precision
metadata unset, since those settings cannot be recovered from these records.

`predictions.jsonl` stores per-example outputs, scores, and timing.
`summary.json` reports percent scores per benchmark, their unweighted macro
average, total suite wall time, and total visible tokens divided by total time.
The benchmark count is explicit; an incomplete suite is not labeled as an
11-benchmark average.

Generation timing includes prompt prefill and decoding. It excludes model
loading, warmup, tokenization, and scoring. CUDA measurements synchronize before
and after generation. `alodlm-evaluate` uses the portable SDPA decoder. The
repository also includes the [optimized CUDA-graph engine](inference.md#optimized-inference),
invoked by `alodlm-generate-optimized`; installing it does not change the
evaluation command's backend. Report the engine used for each throughput result.

Benchmark reproduction additionally requires matching the supplied prompts,
dataset versions, reference labels, test suites, and decoding settings.
No benchmark files or dataset acquisition code are included.
