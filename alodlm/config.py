"""Training configuration. Run: python -m alodlm.train --help."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


def local_dir(value):
    path = Path(value)
    if not path.is_dir():
        raise ValueError("Expected an existing local directory")
    return path


@dataclass
class ModelConfig:
    loop_start: int = 10
    loop_end: int = 26
    max_depth: int = 4
    depth_prior_c: float = 0.4
    kl_beta_mi: float = 0.1
    kl_beta_marg: float = 1.0
    # Uniform exit-depth masses for a new gate; conditional hazards differ by depth.
    gate_init_pi: tuple = (0.25, 0.25, 0.25, 0.25)
    mask_token_id: int = 151665
    block_size: int = 32
    gradient_checkpointing: bool = True
    gradient_checkpointing_stride: int = 2
    attention_backend: str = "sdpa"
    compile_layers: bool = False

    def validate(self, backbone=None):
        if not 0 <= self.loop_start < self.loop_end:
            raise ValueError("Require 0 <= loop_start < loop_end")
        if self.max_depth < 2:
            raise ValueError("max_depth must be at least two")
        if self.block_size < 1 or self.gradient_checkpointing_stride < 1:
            raise ValueError("Block size and checkpoint stride must be positive")
        if self.attention_backend not in ("sdpa", "magi", "magi-fa4"):
            raise ValueError("Unsupported attention backend")
        if min(self.depth_prior_c, self.kl_beta_mi, self.kl_beta_marg) < 0:
            raise ValueError("Prior slope and KL coefficients must be nonnegative")
        if len(self.gate_init_pi) != self.max_depth or min(self.gate_init_pi) <= 0:
            raise ValueError("gate_init_pi must contain one positive mass per depth")
        if abs(sum(self.gate_init_pi) - 1) > 1e-6:
            raise ValueError("gate_init_pi must sum to one")
        if backbone is not None:
            if backbone.config.model_type != "qwen3":
                raise ValueError("This implementation supports Qwen3 causal models")
            if backbone.config.tie_word_embeddings:
                raise ValueError("Training requires an untied Qwen3 output projection")
            if self.loop_end > backbone.config.num_hidden_layers:
                raise ValueError("loop_end exceeds the number of decoder layers")
            if not 0 <= self.mask_token_id < backbone.config.vocab_size:
                raise ValueError("mask_token_id must be in the model vocabulary")


@dataclass
class TrainConfig:
    model_dir: str
    train_file: str
    output_dir: str
    cache_file: str
    model: ModelConfig
    max_seq_length: int = 4096
    epochs: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 40
    use_deepspeed: bool = False
    max_steps: int = 0
    resume: str | None = None

    @classmethod
    def load(cls, filename):
        value = yaml.safe_load(Path(filename).read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a YAML mapping")
        allowed = {f.name for f in fields(cls)}
        if set(value) - allowed:
            raise ValueError("Unknown training configuration fields")
        value["model"] = ModelConfig(**value.get("model", {}))
        config = cls(**value)
        config.model.validate()
        if min(config.max_seq_length, config.epochs, config.gradient_accumulation_steps,
               config.logging_steps, config.save_steps, config.save_total_limit) < 1:
            raise ValueError("Training lengths and intervals must be positive")
        if config.max_steps < 0 or not 0 <= config.warmup_ratio < 1:
            raise ValueError("Invalid step budget or warmup ratio")
        if config.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if config.learning_rate <= 0 or config.max_grad_norm <= 0 or config.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        return config

    def to_dict(self):
        return asdict(self)

    def deepspeed_config(self):
        return {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.max_grad_norm,
            "steps_per_print": self.logging_steps,
            "wall_clock_breakdown": False,
            "bf16": {"enabled": self.precision == "bf16"},
            "zero_optimization": {
                "stage": 2, "overlap_comm": False, "contiguous_gradients": True,
                "reduce_bucket_size": 50000000, "allgather_bucket_size": 50000000,
            },
        }
