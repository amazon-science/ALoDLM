# Training guide

[ALoDLM](../README.md)

Install the [portable decoder and training environment](../README.md#portable-decoder-and-training-environment)
before running these commands. The optimized inference environment has its own
dependency versions and entry point.

## Local training input

The configuration defaults to the 8B topology. For 1.7B, set `model.loop_start: 0`
and `model.loop_end: 28`, and use the corresponding model directory. Starting
from released ALoDLM weights restores their trained exit gate. Starting from a
base Qwen3 model requires an untied output projection and creates a fresh gate.

Replace the placeholders in [configs/train.yaml](../configs/train.yaml).
Each line of the supplied JSONL file is a message list:

```json
[{"role":"user","content":"<PROMPT>"},{"role":"assistant","content":"<RESPONSE>"}]
```

The tokenizer's local chat template formats conversations. Supply conversations
ending in an assistant response: labels cover the final turn, and earlier
messages are context. A single-message record is retained with no supervised
labels. Empty, malformed, and overlength records are filtered; overlength
conversations are dropped without truncation. Multi-message records ending in
a non-assistant turn are malformed. Seeded shuffling precedes packing.
An example crossing a pack boundary becomes two independently attended segments.

The cache contains memory-mapped flat token and label tensors plus pack and
segment offsets. Its identity binds the file contents, tokenizer, template,
sequence length, and seed. An incompatible existing cache is rejected.

Prepare the cache explicitly, or let the training entry point prepare it:

```bash
alodlm-prepare \
  --model "<LOCAL_MODEL_DIRECTORY>" \
  --input "<LOCAL_TRAINING_JSONL>" \
  --output "<LOCAL_PACKED_CACHE>" \
  --max-length 4096
```

## Training

```bash
alodlm-train --config configs/train.yaml
```

Choose `epochs` and optionally `max_steps` for your own run. The example uses
AdamW, learning rate `1e-5`, cosine decay,
3% warmup, weight decay `0.01`, and gradient clipping at `1.0`. AdamW uses
betas `(0.9, 0.999)` and epsilon `1e-8`. Parameters with `bias`, `LayerNorm.weight`,
or `layer_norm.weight` in their names are excluded from weight decay.

Each process handles one pack per microbatch. Global batch size is process
count times `gradient_accumulation_steps`; choose accumulation for your data
and hardware. The example settings describe a new run on user-supplied inputs.

Schedule length is derived from the local corpus:
`ceil(ceil(number_of_packs / process_count) / accumulation_steps) * epochs`.
The cosine scheduler advances once per optimizer update.
`max_steps` can cap the schedule for a small test.

For distributed training, install `python -m pip install '.[distributed]'`
and set `use_deepspeed: true` in the configuration. Choose the process count
for the available GPUs. For example, on a machine with eight GPUs:

```bash
torchrun --nproc_per_node=8 -m alodlm.train --config configs/train.yaml
```

The launcher controls the machine count, rank, process count, and rendezvous
address. For multiple machines, set those standard launcher parameters for each
machine. Model, input, cache, and checkpoint paths must be accessible to all
processes. `use_deepspeed: true` enables ZeRO stage 2 directly from the training
configuration. It shards optimizer state, keeps model parameters replicated,
uses 50-million-element reduction/all-gather buckets, and disables communication
overlap and optimizer offload.

`precision` accepts `bf16` or `fp32`. The default `sdpa` backend uses PyTorch
attention and needs no additional attention library. Dense attention uses more
memory at long sequence lengths; reduce `max_seq_length` for initial tests.
The optional `magi` and `magi-fa4` backends require a separately installed,
compatible MagiAttention build and are intended for advanced setups.

Gradient checkpointing is configurable. Set `model.compile_layers: true` to
enable `torch.compile(dynamic=False)` for decoder layers.
Training projects full vocabulary logits at every readout depth. Cross-entropy
and per-segment autoregressive reductions use the parameter dtype; autocast
is disabled inside the training forward. This preserves the BF16 reduction
path. The backbone and gate are trained jointly.

The supplied YAML and Python defaults initialize equal exit mass at each of
the four depths: `[0.25, 0.25, 0.25, 0.25]`. The corresponding conditional
halting probabilities are `1/4`, `1/3`, and `1/2`, with a forced exit at the
final depth. This initialization is separate from the geometric regularization
prior, whose slope remains `c=0.4`. When the input model directory contains
`exit_gate.pt`, its learned parameters replace the fresh initialization.

For a small CPU test, set `use_deepspeed: false`, `precision: fp32`,
`model.attention_backend: sdpa`, and `model.compile_layers: false`, and supply
a small local model with matching recurrence settings.

Each complete checkpoint contains:

```text
config.json
model.safetensors (or indexed shards)
tokenizer files
exit_gate.pt
alodlm_config.json
training_state/
trainer_state.json
```

Set `resume` in the training YAML to a complete checkpoint directory to restore
weights, optimizer, schedule, random states, and data position. Resume requires
the same objective, input identity, process count, and training horizon.
A checkpoint is complete only after `trainer_state.json` exists. `save_steps`
controls checkpoint frequency and `save_total_limit` controls retention;
completion also writes a checkpoint. Training-state resume uses this package's checkpoint
format; weight-only loading from another trainer does not restore its optimizer
or data position.
Logs contain scalar training metrics; examples and generated text are not logged.
