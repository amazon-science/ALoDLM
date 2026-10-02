"""Recurrent Qwen3 model. Run: python -m alodlm.train --help.

Modified from WeDLM training code; see optimized/licenses/WeDLM.txt.
"""

import json
import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from .attention import sparse_attention
from .batch import build_batch
from .config import ModelConfig, local_dir
from .loss import autoregressive_loss, outcome_loss


class GateHead(nn.Module):
    """Initialize conditional hazards from exit-depth masses; the last exit is forced."""

    def __init__(self, hidden_size, initial_mass):
        super().__init__()
        self.net = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        bias, survival = [], 1.0
        for mass in initial_mass[:-1]:
            hazard = mass / survival
            bias.append(math.log(hazard / (1 - hazard)))
            survival *= 1 - hazard
        self.depth_bias = nn.Parameter(torch.tensor(bias + [0.0], dtype=torch.float32))

    def forward(self, features, depth):
        return self.net(features).squeeze(-1) + self.depth_bias[depth]


def rotary(q, k, cos, sin):
    def rotate(x):
        first, second = x.chunk(2, -1)
        return torch.cat((-second, first), -1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


def layer_forward(layer, hidden, cos, sin, mask, prefix=None):
    """Apply one decoder layer and return its current-token K/V."""
    residual = hidden
    h = layer.input_layernorm(hidden)
    attn = layer.self_attn
    length, dim = len(h), attn.head_dim
    q = attn.q_proj(h).view(length, attn.config.num_attention_heads, dim)
    k = attn.k_proj(h).view(length, attn.config.num_key_value_heads, dim)
    v = attn.v_proj(h).view(length, attn.config.num_key_value_heads, dim)
    q, k = rotary(attn.q_norm(q), attn.k_norm(k), cos, sin)
    current_kv = k, v
    if prefix is not None:
        k, v = torch.cat((prefix[0], k), 0), torch.cat((prefix[1], v), 0)
    if isinstance(mask, dict):
        out = sparse_attention(q, k, v, mask).reshape(length, -1)
    else:
        if q.shape[1] != k.shape[1]:
            repeats = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=mask, dropout_p=0.0, scale=dim ** -0.5,
        ).squeeze(0).transpose(0, 1).reshape(length, -1)
    hidden = residual + attn.o_proj(out)
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, current_kv


def training_layer(layer, hidden, cos, sin, mask):
    return layer_forward(layer, hidden, cos, sin, mask)[0]


class ALoDLM(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        config.validate(backbone)
        self.backbone = backbone
        self.config = config
        self.exit_gate = GateHead(backbone.config.hidden_size, config.gate_init_pi)
        self.exit_gate.to(dtype=backbone.dtype, device=backbone.device)
        self._training_layer = (torch.compile(training_layer, dynamic=False)
                                if config.compile_layers else training_layer)

    @classmethod
    def from_pretrained(cls, directory, config=None, dtype=torch.float32, require_gate=True):
        path = local_dir(directory)
        metadata = path / "alodlm_config.json"
        if config is None:
            if not metadata.is_file():
                raise ValueError("Checkpoint requires alodlm_config.json or an explicit ModelConfig")
            config = ModelConfig(**json.loads(metadata.read_text()))
        backbone = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation="eager",
        )
        model = cls(backbone, config)
        gate_file = path / "exit_gate.pt"
        if gate_file.is_file():
            # Saved learned parameters replace the fresh gate initialization.
            model.exit_gate.load_state_dict(
                torch.load(gate_file, map_location="cpu", weights_only=True), strict=True)
        elif require_gate:
            raise ValueError("Adaptive inference requires trained exit_gate.pt weights")
        return model

    def save_pretrained(self, directory, state_dict=None):
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        state = state_dict if state_dict is not None else self.state_dict()
        backbone = {k.removeprefix("backbone."): v for k, v in state.items()
                    if k.startswith("backbone.")}
        gate = {k.removeprefix("exit_gate."): v for k, v in state.items()
                if k.startswith("exit_gate.")}
        self.backbone.config._name_or_path = ""
        self.backbone.save_pretrained(path, state_dict=backbone, safe_serialization=True)
        torch.save(gate, path / "exit_gate.pt")
        (path / "alodlm_config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")

    def rollout(self, batch):
        """Sample commitments and return per-depth logits, gates, and exits."""
        base, cfg = self.backbone.model, self.config
        hidden = base.embed_tokens(batch.ids)
        cos, sin = base.rotary_emb(hidden.unsqueeze(0), batch.positions.unsqueeze(0))
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        calls = 0

        def run(layer, h):
            nonlocal calls
            use_checkpoint = (cfg.gradient_checkpointing and self.training
                              and calls % cfg.gradient_checkpointing_stride == 0)
            calls += 1
            def block(x):
                return self._training_layer(layer, x, cos, sin, batch.attention_mask)
            return checkpoint(block, h, use_reentrant=False) if use_checkpoint else block(h)

        for layer in base.layers[:cfg.loop_start]:
            hidden = run(layer, hidden)
        gold_embeddings = base.embed_tokens(batch.original)
        exits = torch.full_like(batch.ids, cfg.max_depth)
        logits, gates = [], []
        for depth in range(cfg.max_depth):
            if depth:
                hidden = torch.where((exits <= depth).unsqueeze(-1), gold_embeddings, hidden)
            for layer in base.layers[cfg.loop_start:cfg.loop_end]:
                hidden = run(layer, hidden)
            readout = hidden
            for layer in base.layers[cfg.loop_end:]:
                readout = run(layer, readout)
            readout = base.norm(readout)
            logits.append(self.backbone.lm_head(readout))
            gates.append(self.exit_gate(readout.detach(), depth))
            if depth < cfg.max_depth - 1:
                with torch.no_grad():
                    hazard = gates[-1].float().sigmoid()
                    selected = (torch.rand_like(hazard) < hazard) & batch.masked & (exits == cfg.max_depth)
                exits = torch.where(selected, depth + 1, exits)
                hidden = base.norm(hidden)
        return logits, gates, exits

    def forward(self, ids, labels, boundaries):
        # Parameters determine training precision, including cross-entropy reduction.
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self._forward_loss(ids, labels, boundaries)

    def _forward_loss(self, ids, labels, boundaries):
        batch = build_batch(ids, labels, boundaries, self.config.block_size,
                            self.config.mask_token_id, self.config.attention_backend)
        logits, gates, exits = self.rollout(batch)
        ce = torch.stack([F.cross_entropy(lg[batch.masked], batch.original[batch.masked], reduction="none")
                          for lg in logits])
        denoising_actor, metrics = outcome_loss(ce, gates, exits, batch, self.config)
        ar_terms = [autoregressive_loss(lg, labels, batch.boundaries) for lg in logits]
        ar_loss = torch.stack(ar_terms).mean()
        total = (denoising_actor + ar_loss) / 2
        total = total + 0.0 * sum(t.sum() for t in logits + gates)
        return total, {**metrics, "ar_loss": ar_loss.detach(), "loss": total.detach()}
