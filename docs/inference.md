# Generation and adaptive decoding

[ALoDLM](../README.md)

## Setup and model choice

Use Python 3.12. Clone the code before running either installation:

```bash
git clone https://github.com/amazon-science/ALoDLM.git
cd ALoDLM
```

Both generation commands accept `amazon/ALoDLM-1.7B`, `amazon/ALoDLM-8B`,
or a local directory containing the complete inference files. Downloads are
cached by the Hugging Face client. Local directories work offline.

The optimized engine runs on Linux with an NVIDIA GPU supporting BF16 and a
compatible NVIDIA driver. Full 8B optimized generation has been checked on an
A100 with 40 GB of GPU memory. The GPU must fit the BF16 weights, KV cache, and
runtime workspace; memory requirements grow with context length. The portable
decoder also supports CPU with FP32, which needs more RAM and runs more slowly.

## Optimized inference

The repository includes the ALoDLM-aware nano-vLLM engine and its completed-depth
cache repair in `optimized/`. Install it from the repository root in a dedicated
Linux CUDA environment on an NVIDIA GPU supporting BF16:

```bash
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install --upgrade pip
python -m pip install ./optimized
```

The optimized runtime pins PyTorch 2.10.0, Transformers 5.5.4, and vLLM 0.19.1.
It uses standalone FlashAttention 2 when available, or vLLM's compatible
FlashAttention 2 kernels. Requests use the included adaptive engine. Keep this
environment separate from the portable decoder and trainer.

```bash
alodlm-generate-optimized \
  --model amazon/ALoDLM-8B \
  --mode entropy \
  --q 0.5 --tau 0.4 --max-new-tokens 512 \
  --prompt "Explain binary search."
```

Use `--model amazon/ALoDLM-1.7B` for the smaller model, or pass a local model
directory. HF model IDs are downloaded through the Hugging Face client, using
its configured credentials when needed. Both models require safetensors
backbone weights, tokenizer/configuration files, `exit_gate.pt`, and
`alodlm_config.json`. No external engine directory is needed.

The command reads the recurrent layer range from model metadata and restores
the trained gate. Greedy token selection and entropy-based parallel commitment
are the defaults, with a 16-token window and position penalty `0.02`.
`--mode left1` selects one token per recurrent pass instead; for the paper's
quality settings use `--q 0.4` at 1.7B or `--q 0.5` at 8B. Entropy selection is
inactive in that mode.

The launcher enables adaptive pass CUDA graphs for both architectures and
additional decision graphs for the 8B architecture. It verifies activation
after generation. In this engine, `enforce_eager=True` disables full-model
capture; the separate pass and decision graph flags remain enabled. The cache
repair fills only uncomputed recurrent depths and preserves computed keys.

Use `CUDA_VISIBLE_DEVICES` to select a GPU. The command serves one request on
one GPU; `--gpu-memory-utilization` defaults to `0.8`, and `--max-model-len`
defaults to 8,192 tokens, bounded by the model configuration. The prompt and
output budget must leave space for the decoding window. An available local
NCCL port is selected automatically, or set one with `--nccl-port`.

The first generation includes compilation and graph capture. The engine's
first-call timing therefore includes startup work and is not steady-state
throughput. `--json` prints the response, token IDs, decoding mode, and graph
activation status. The [example script](../examples/generate_optimized.py)
accepts the same arguments as the installed command.

## Portable decoder

From the repository root, use a separate environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The portable runtime pins PyTorch 2.8.0 and Transformers 4.56.1. It uses the
same model artifacts and entropy default as the optimized command:

```bash
alodlm-generate \
  --model amazon/ALoDLM-8B \
  --device cuda --precision bf16 \
  --mode entropy \
  --prompt "Explain binary search." \
  --q 0.5 --tau 0.4 --max-new-tokens 512
```

Use `--device cpu --precision fp32` for small CPU tests. Generation is greedy.
`--thinking` selects the tokenizer's thinking-enabled chat template when
supported. A checkpoint containing standard Qwen3 weights and `exit_gate.pt`
without recurrence metadata can use `--model-config configs/train.yaml`.
The model section must describe the trained architecture.

`entropy` is the default mode, including when `--mode` is omitted.
In this mode, a token commits when its entropy plus position penalty is
below `tau`. After each pass, the mean gate exit CDF over uncommitted tokens
determines whether to advance the window: advance when it reaches `q`.
If no token qualifies, the least-entropy position is committed. The default
position penalty coefficient is `0.05 * tau`.

`--mode left1` commits the leftmost uncommitted token each pass and uses the
same gate-controlled window stopping rule. Entropy does not select tokens in
this mode. Increasing `q` typically requests more recurrence; accuracy need
not increase monotonically.

The cache retains separate depths for recurrent layers and shared caches for
the prelude and coda. Committed prefix tokens are recomputed as observed inputs
before their keys and values enter the persistent prefix cache. Within a window,
newly committed tokens enter the next recurrent pass as token embeddings, then
continue through later passes as latent states. When a window stops early, uncomputed recurrent cache
depths inherit the last completed depth. Already computed deeper keys are
preserved. Coda prefix keys retain the last completed readout branch.

The decoder returns text, token IDs, visible-token exit depths, first-pass
halting probabilities, wall time, and loop-token execution counts. Exit depth
measures the pass where a token committed in its final window. Total
`loop_token_passes_per_output_token` also counts repeated window processing;
it is a different quantity.

## Download and access troubleshooting

Public models need no login. For a private or gated repository to which you
have access, run `hf auth login` in the selected environment, using your own
Hugging Face access token.

To download once and then work offline:

```bash
hf download amazon/ALoDLM-1.7B --local-dir ./models/ALoDLM-1.7B
alodlm-generate --model ./models/ALoDLM-1.7B \
  --device cuda --precision bf16 --mode entropy \
  --q 0.5 --tau 0.4 --max-new-tokens 512 \
  --prompt "Explain binary search."
```

| Symptom | Action |
|---|---|
| Repository not found, 401, or 403 | Check the organization and model name, repository visibility, and your access. Log in if the repository requires it. |
| CUDA unavailable | Run the optimized command on a supported NVIDIA GPU; use `nvidia-smi` to check the driver. For portable CPU inference, select `--device cpu --precision fp32`. |
| GPU out of memory | Close other GPU workloads or use the smaller model. For optimized inference, reduce `--max-model-len` and the output budget. `--gpu-memory-utilization` controls the engine's memory allocation. |
| Missing gate or recurrence metadata | Download the complete model repository, including `exit_gate.pt` and `alodlm_config.json`. |
| Dependency conflicts | Create the separate environment shown above for the selected runtime. |

The supplied weight repositories contain model artifacts and their model cards;
install the code from this repository to run adaptive decoding. Standard
Transformers `generate()` and `vllm serve` do not provide the included recurrent
decoder and trained gate.
