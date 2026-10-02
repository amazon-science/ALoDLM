# Optimized ALoDLM inference

This directory includes the ALoDLM-aware nano-vLLM/WeDLM engine, its
completed-depth cache repair, and the `alodlm-generate-optimized` command.
No external engine checkout is needed.

From the repository root, install in a dedicated Linux CUDA environment on an
NVIDIA GPU supporting BF16:

```bash
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install --upgrade pip
python -m pip install ./optimized
```

The runtime pins PyTorch 2.10.0, Transformers 5.5.4, and vLLM 0.19.1.
It uses vLLM's FlashAttention 2 kernel bundle when a standalone FlashAttention
installation is absent. This environment is separate from the portable
decoder and trainer, which use their own dependency versions.

```bash
alodlm-generate-optimized \
  --model amazon/ALoDLM-8B \
  --mode entropy --q 0.5 --tau 0.4 \
  --max-new-tokens 512 \
  --prompt "Explain binary search."
```

Use `amazon/ALoDLM-1.7B` for the smaller model or pass a local model directory.
Both need tokenizer/backbone files, safetensors weights, `exit_gate.pt`, and
`alodlm_config.json`. The model's metadata selects its recurrent layer range.

The command runs one request on one GPU with BF16 weights. It enables adaptive
pass CUDA graphs for both architectures and decision graphs for the 8B
architecture. The first generation includes compilation and graph capture.
Generation uses the trained gate and greedy token selection; `entropy` enables
parallel commitment and is the default. `left1` is available for single-token
commitment per recurrent pass.

## Validation

From the repository root, check model-directory compatibility on CPU:

```bash
PYTHONPATH=optimized python -m unittest discover -s optimized/tests -v
```

In the optimized CUDA environment, compare the cache repair with an exact
reference at every stopping depth:

```bash
python -m alodlm_optimized.adaptive_cache_repair
```

Add `--json` to the generation command to include token IDs and CUDA graph
activation status. Select the GPU with `CUDA_VISIBLE_DEVICES`.

See [the inference guide](../docs/inference.md) for runtime settings and
[NOTICE](NOTICE) and [the upstream terms](licenses/WeDLM.txt) for licensing.
