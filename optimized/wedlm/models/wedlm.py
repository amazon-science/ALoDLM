# Modified for ALoDLM: adaptive recurrence, depth-aware caching, and release packaging.
# Original notices are retained; see NOTICE and licenses/WeDLM.txt.

# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from wedlm.layers.activation import SiluAndMul
from wedlm.layers.attention import Attention
from wedlm.layers.layernorm import RMSNorm
from wedlm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from wedlm.layers.rotary_embedding import get_rope
from wedlm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class WeDLMAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        wedlm_window_size: int | None = None,
        max_context_len: int | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            wedlm_window_size=wedlm_window_size,
            max_context_len=max_context_len,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class WeDLMMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class WeDLMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        wedlm_window_size = getattr(config, "wedlm_window_size", None)
        max_model_len = getattr(config, "max_model_len", 4096)

        self.self_attn = WeDLMAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            wedlm_window_size=wedlm_window_size,
            max_context_len=max_model_len,
        )
        self.mlp = WeDLMMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class WeDLMModel(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [WeDLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:


        import os
        from wedlm.utils.context import (
            set_loop_depth, reset_perdepth_readouts, stash_perdepth_readout,
            adaptive_depth_enabled,
        )
        from wedlm.utils import effstats
        ls = int(os.environ.get("WEDLM_LOOP_START", "-1"))
        le = int(os.environ.get("WEDLM_LOOP_END", "-1"))
        K = int(os.environ.get("WEDLM_LOOP_K", "1"))
        _T = int(input_ids.numel())          # stream tokens this forward (FLOP accounting)
        _N = len(self.layers)
        effstats.add_forward(_T)
        # Adaptive-depth decode: stash norm(h_d) at EVERY loop depth so the decoder can read
        # the per-depth readout on the masks and commit each token at its own exit depth d*.
        adaptive_depth = adaptive_depth_enabled()
        # Ouro-style between-step carry RMSNorm (mirrors training's loop_carry_norm): renorm the
        # loop-carry after each pass so ‖H‖ stays bounded. Decode MUST match training, so eval
        # sets this whenever the checkpoint was trained with loop_carry_norm=true.
        carry_norm = int(os.environ.get("WEDLM_LOOP_CARRY_NORM", "0")) != 0
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        if ls >= 0 and le > ls and K > 1:
            if adaptive_depth:
                reset_perdepth_readouts()
            # depth 0 for prelude (non-looped; single KV buffer regardless)
            set_loop_depth(0)
            if ls > 0:
                effstats.add_pass(_T, ls)    # prelude, once
            for layer in self.layers[:ls]:
                hidden_states, residual = layer(positions, hidden_states, residual)
            # weight-shared middle block, K passes; depth d selects the depth-d KV
            # buffer in the granular per-depth cache (auto-on while looping)
            for d in range(K):
                set_loop_depth(d)
                effstats.add_pass(_T, le - ls, loop_block=True)   # looped block, this pass
                for layer in self.layers[ls:le]:
                    hidden_states, residual = layer(positions, hidden_states, residual)
                if adaptive_depth and le < len(self.layers):
                    # SANDWICH per-depth readout: branch the depth-d carry through the coda
                    # + final norm (training's per_step semantics), while the loop carries on
                    # pre-coda. Coda layers are non-looped -> single KV buffer (depth 0): each
                    # branch overwrites the window rows' coda KV, so pruned tokens keep the
                    # LAST branch's coda keys ("coda-last-step" approximation, the analogue of
                    # Ouro's near-lossless decode-phase sharing; looped layers stay granular).
                    set_loop_depth(0)
                    effstats.add_pass(_T, _N - le)  # coda readout branch, this pass
                    bh, br = hidden_states, residual
                    for layer in self.layers[le:]:
                        bh, br = layer(positions, bh, br)
                    rd, _ = self.norm(bh, br)
                    stash_perdepth_readout(rd)
                    # Stage-I gate: compute-then-stop — break the shared loop as soon as
                    # every batched sequence's stop rule fired (also records the forced
                    # decisions at d == K-1). No deeper pass is computed after True.
                    from wedlm.engine.gate_decode import (
                        commit_on_agree_apply, commit_on_agree_enabled, loop_pass_check,
                    )
                    stop = loop_pass_check(rd, d, K)
                    if stop or d == K - 1:
                        return rd          # deepest computed branch IS the final output
                    if carry_norm:
                        h = hidden_states + residual if residual is not None else hidden_states
                        hidden_states, residual = self.norm(h), None
                    if commit_on_agree_enabled():
                        # COMMIT-ON-AGREE, sandwich wiring: selection + token choice on the CODA
                        # readout rd (what the head scores), embedding injected into the PRE-CODA
                        # carry -- training's option C exactly, and in training's order (the raw
                        # embedding OVERWRITES the carry-normed stream for committed rows).
                        if residual is not None:
                            # no carry_norm: materialize the fused stream first; (h, None) is
                            # exact -- input_layernorm(h, None) rms-norms h alone next pass.
                            hidden_states, residual = hidden_states + residual, None
                        carry = commit_on_agree_apply(rd, d, K, carry=hidden_states)
                        if carry is not None:
                            hidden_states = carry
                elif adaptive_depth:
                    # all-layer loop (le == N): readout at depth d = norm(carry value). For
                    # d<K-1 this IS the carry input to the next pass (identical to the
                    # carry_norm branch below), so reuse it; the last pass is left to flow
                    # into the final norm below.
                    h_d = hidden_states + residual if residual is not None else hidden_states
                    rd = self.norm(h_d)
                    stash_perdepth_readout(rd)
                    # Stage-I gate compute-then-stop (see sandwich branch above); at
                    # d == K-1 the check still runs to record the forced decisions, then
                    # the pass flows into the (empty) coda + final norm as before.
                    from wedlm.engine.gate_decode import (
                        commit_on_agree_apply, loop_pass_check,
                    )
                    stop = loop_pass_check(rd, d, K)
                    if stop and d < K - 1:
                        return rd
                    # COMMIT-ON-AGREE: rows that just satisfied both criteria are decoded now and
                    # enter the next pass as their token's embedding rather than as a latent. The
                    # returned tensor is a COPY -- rd itself was already stashed as this depth's
                    # readout and must stay untouched. None => feature off or nothing decoded.
                    carry = commit_on_agree_apply(rd, d, K)
                    if carry_norm and d < K - 1:
                        hidden_states, residual = (carry if carry is not None else rd), None
                elif carry_norm and d < K - 1:
                    # Ouro-style between-step renorm, mirroring training's base_model.norm on the
                    # loop-carry. This engine threads a FUSED residual stream (hidden_states,
                    # residual): the true carry value is hidden_states+residual, add-normed lazily
                    # inside each layer's input_layernorm. Materialize it, RMSNorm, and hand the
                    # normed carry to the next pass as a fresh stream (residual=None; feeding a
                    # value as (h, None) is exact — input_layernorm(h, None) rms-norms h alone).
                    # The last pass is left un-normed: its output flows to coda+final-norm, which
                    # matches training's readout on the raw deepest pass. Each pass still writes
                    # its own depth-d KV via set_loop_depth(d), so the per-depth cache is unchanged.
                    h = hidden_states + residual if residual is not None else hidden_states
                    hidden_states, residual = self.norm(h), None
            set_loop_depth(0)
            if le < _N:
                effstats.add_pass(_T, _N - le)   # trailing coda (non-adaptive exit path)
            for layer in self.layers[le:]:
                hidden_states, residual = layer(positions, hidden_states, residual)
        else:
            set_loop_depth(0)
            effstats.add_pass(_T, _N)            # plain single pass (K<=1 / no loop)
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def loop_pass(
        self,
        positions: torch.Tensor,
        carry: torch.Tensor,
        depth: int,
        ls: int,
        le: int,
    ) -> torch.Tensor:
        """ONE weight-shared loop pass + its readout, for the per-pass CUDA-graph path.

        Mirrors the all-layer branch of forward(): run layers[ls:le] at `depth`, then
        materialise the carry and RMSNorm it. Under carry-norm that readout IS both the
        depth-d readout and the next pass's carry, so a captured pass-graph needs exactly
        one tensor crossing its boundary. Deliberately free of env reads, stashes and gate
        calls -- those are host work the driver does BETWEEN replays, which is what a
        whole-forward capture cannot express.
        """
        from wedlm.utils.context import set_loop_depth
        set_loop_depth(depth)
        residual = None
        for layer in self.layers[ls:le]:
            carry, residual = layer(positions, carry, residual)
        h = carry + residual if residual is not None else carry
        return self.norm(h)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def prelude(self, positions: torch.Tensor, input_ids: torch.Tensor, ls: int) -> torch.Tensor:
        """Embed + layers[0:ls) once per step (depth 0), for the sandwich pass-graph path.

        Returns the MATERIALIZED un-normed fused stream (h + residual): handing it to the
        first loop pass as (h, None) is exact -- input_layernorm(h, None) rms-norms h alone,
        which equals input_layernorm(h', r') on the fused pair. ls == 0 reduces to embed().
        """
        from wedlm.utils.context import set_loop_depth
        set_loop_depth(0)
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers[:ls]:
            h, residual = layer(positions, h, residual)
        return h + residual if residual is not None else h

    def loop_pass_raw(
        self, positions: torch.Tensor, carry: torch.Tensor, depth: int, ls: int, le: int
    ) -> torch.Tensor:
        """One weight-shared loop pass, returning the MATERIALIZED PRE-NORM output.

        The sandwich needs both faces of the pass boundary: the coda readout branch reads
        the un-normed fused output (eager: bh, br = hidden_states, residual), while the
        next pass's carry is its RMSNorm. loop_pass() == norm(loop_pass_raw()) exactly.
        """
        from wedlm.utils.context import set_loop_depth
        set_loop_depth(depth)
        residual = None
        for layer in self.layers[ls:le]:
            carry, residual = layer(positions, carry, residual)
        return carry + residual if residual is not None else carry

    def coda_readout(self, positions: torch.Tensor, h: torch.Tensor, le: int) -> torch.Tensor:
        """layers[le:) at depth 0 + final norm on a pre-norm loop output: the per-depth
        sandwich readout (training's per_step semantics; coda KV single-buffer, so each
        depth's replay overwrites the window rows' coda keys == eager's coda-last-step)."""
        from wedlm.utils.context import set_loop_depth
        set_loop_depth(0)
        residual = None
        for layer in self.layers[le:]:
            h, residual = layer(positions, h, residual)
        out, _ = self.norm(h, residual) if residual is not None else (self.norm(h), None)
        return out


class WeDLMForDiffusionLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = WeDLMModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
