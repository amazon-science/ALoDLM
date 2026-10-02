"""Train with sequence outcome credit. Run: python -m alodlm.train --config configs/train.yaml."""

import argparse
import itertools
import json
import math
from pathlib import Path
import shutil

import torch
from accelerate import Accelerator
from accelerate.data_loader import get_sampler
from accelerate.utils import DeepSpeedPlugin, set_seed
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .config import TrainConfig
from .data import PackedDataset, load_tokenizer, one_pack, prepare_cache
from .model import ALoDLM


def train(config):
    plugin = DeepSpeedPlugin(hf_ds_config=config.deepspeed_config()) if config.use_deepspeed else None
    accelerator = Accelerator(
        cpu=not torch.cuda.is_available(),
        mixed_precision="bf16" if config.precision == "bf16" else "no",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        deepspeed_plugin=plugin,
    )
    set_seed(config.seed)
    tokenizer = load_tokenizer(config.model_dir)
    with accelerator.main_process_first():
        cache = prepare_cache(config.train_file, tokenizer, config.cache_file,
                              config.max_seq_length, config.seed)
    dataset = PackedDataset(cache)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=one_pack, num_workers=0,
                        pin_memory=accelerator.device.type == "cuda")
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float32
    model = ALoDLM.from_pretrained(config.model_dir, config.model, dtype, require_gate=False)
    if config.max_seq_length > model.backbone.config.max_position_embeddings:
        raise ValueError("Training sequence length exceeds the model context")
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(s in n for s in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(s in n for s in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
    steps_per_epoch = math.ceil(math.ceil(len(dataset) / accelerator.num_processes)
                               / config.gradient_accumulation_steps)
    horizon = steps_per_epoch * config.epochs
    if config.max_steps:
        horizon = min(horizon, config.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(horizon * config.warmup_ratio), horizon)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.register_for_checkpointing(scheduler)
    output = Path(config.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "global_batch_size": accelerator.num_processes * config.gradient_accumulation_steps,
            "training_steps": horizon,
        }), flush=True)
    accelerator.wait_for_everyone()
    step, start_epoch, skip_batches, epoch_rng = 0, 0, 0, None
    signature = config.to_dict()
    for key in ("model_dir", "train_file", "cache_file", "output_dir", "resume"):
        signature.pop(key)
    identity = json.loads(json.dumps({
        "config": signature, "cache_identity": dataset.cache["identity"],
        "world_size": accelerator.num_processes, "horizon": horizon,
    }))
    if config.resume:
        resume = Path(config.resume)
        state = json.loads((resume / "trainer_state.json").read_text())
        if state["identity"] != identity:
            raise ValueError("Resume requires the same objective, data, schedule, and process count")
        accelerator.load_state(str(resume / "training_state"))
        step, start_epoch, skip_batches = state["step"], state["epoch"], state["next_batch"]
        epoch_rng = state["epoch_rng"]

    def save(epoch, next_batch):
        destination = output / f"checkpoint-{step}"
        if destination.exists():
            raise FileExistsError("Checkpoint already exists")
        accelerator.wait_for_everyone()
        weights = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(destination, weights)
            tokenizer.save_pretrained(destination)
        accelerator.wait_for_everyone()
        accelerator.save_state(str(destination / "training_state"))
        if accelerator.is_main_process:
            state = {"step": step, "epoch": epoch, "next_batch": next_batch,
                     "identity": identity, "epoch_rng": epoch_rng}
            (destination / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
            completed = sorted(
                (p for p in output.glob("checkpoint-*") if not p.is_symlink()
                 and p.name.removeprefix("checkpoint-").isdigit()
                 and (p / "trainer_state.json").is_file()),
                key=lambda p: int(p.name.removeprefix("checkpoint-")),
            )
            for obsolete in completed[:-config.save_total_limit]:
                shutil.rmtree(obsolete)
        accelerator.wait_for_everyone()

    optimizer.zero_grad()
    model.train()
    last_saved = step if config.resume else -1
    for epoch in range(start_epoch, config.epochs):
        loader.set_epoch(epoch)
        sampler = get_sampler(loader)
        generator = getattr(sampler, "generator", None)
        if skip_batches:
            checkpoint_cpu_rng = torch.get_rng_state()
            torch.set_rng_state(torch.tensor(epoch_rng["cpu"], dtype=torch.uint8))
            if generator is not None and epoch_rng["sampler"] is not None:
                generator.set_state(torch.tensor(epoch_rng["sampler"], dtype=torch.uint8))
            current = iter(accelerator.skip_first_batches(loader, skip_batches))
            first = next(current, None)
            torch.set_rng_state(checkpoint_cpu_rng)
            current = itertools.chain(() if first is None else (first,), current)
        else:
            epoch_rng = {"cpu": torch.get_rng_state().tolist(),
                         "sampler": generator.get_state().tolist() if generator is not None else None}
            current = loader
        first_batch = skip_batches
        skip_batches = 0
        for batch_index, item in enumerate(current, first_batch):
            if step >= horizon:
                break
            with accelerator.accumulate(model):
                loss, metrics = model(**item)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    if norm is not None and not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                if accelerator.sync_gradients:
                    if accelerator.optimizer_step_was_skipped:
                        raise FloatingPointError("Optimizer skipped an update")
                    scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                if step % config.logging_steps == 0 or step == 1:
                    values = {key: float(accelerator.reduce(value.detach().float(), reduction="mean"))
                              for key, value in metrics.items()}
                    if accelerator.is_main_process:
                        record = {"step": step, "learning_rate": scheduler.get_last_lr()[0], **values}
                        print(json.dumps(record), flush=True)
                        with (output / "metrics.jsonl").open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                if step % config.save_steps == 0 or step == horizon:
                    save(epoch, batch_index + 1)
                    last_saved = step
            if step >= horizon:
                break
        if step >= horizon:
            break
    if last_saved != step:
        save(config.epochs, 0)
    accelerator.end_training()
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(TrainConfig.load(args.config))


if __name__ == "__main__":
    main()
