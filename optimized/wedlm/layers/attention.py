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
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Optional
from flash_attn import flash_attn_varlen_func
from wedlm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        wedlm_window_size: Optional[int] = None,
        max_context_len: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Granular per-depth KV buffers for looped layers (set by ModelRunner when the
        # middle block loops, K>1); None => single buffer (non-looped / base model).
        # The depth-d loop pass stores into buffer[d] and a depth-d query reads buffer[d],
        # so intermediate depths stay depth-matched (== recompute). A single last-write-
        # wins buffer would freeze the prefix at depth-K and collapse shallow-depth reads.
        self.k_cache_depths = None
        self.v_cache_depths = None
        self.wedlm_window_size = wedlm_window_size
        self.max_context_len = max_context_len if max_context_len is not None else 4096

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        device = q.device

        # 1) Store KV. Looped layers keep a per-depth buffer set: the depth-d pass stores
        #    into buffer[d], preserving every loop depth (granular cache). A query at loop
        #    depth d reads buffer[d] (below); early-exited committed tokens get their deep
        #    buffers back-filled with the exit key by ModelRunner._recache_granular, so the
        #    effective read is buffer[min(d, exit_depth)]. Non-looped layers use one buffer.
        k_cache, v_cache = self.k_cache, self.v_cache
        if self.k_cache_depths is not None:
            from wedlm.utils.context import get_loop_depth
            d = get_loop_depth()
            if d >= len(self.k_cache_depths):
                d = len(self.k_cache_depths) - 1
            k_cache, v_cache = self.k_cache_depths[d], self.v_cache_depths[d]
        if k_cache.numel() and v_cache.numel():
            sm = context.slot_mapping
            if not context.is_prefill:
                # Token-freeze emulation (WEDLM_TOKEN_FREEZE): rows whose gate exited at an
                # earlier loop pass keep their back-filled exit-depth KV — suppress their
                # deeper-pass store so the frozen keys are what attention reads (decode
                # attention reads the window from the cache buffers, never the fresh k/v).
                from wedlm.engine.gate_decode import frozen_store_rows
                fr = frozen_store_rows()
                if fr is not None and fr.numel() == sm.numel():
                    sm = sm.masked_fill(fr, -1)
            store_kvcache(k, v, k_cache, v_cache, sm)

        # 2) Prefill
        if context.is_prefill:
            k_src, v_src = (
                (k_cache, v_cache) if context.block_tables is not None else (k, v)
            )
            return flash_attn_varlen_func(
                q,
                k_src,
                v_src,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )

        if context.per_seq_wedlm_sizes is None:
            raise RuntimeError(
                "context.per_seq_wedlm_sizes is None inside Attention.forward (Decode mode)."
            )

        per_seq_wedlm_sizes = context.per_seq_wedlm_sizes

        cu_seqlens_q = F.pad(per_seq_wedlm_sizes.cumsum(0), (1, 0)).to(dtype=torch.int32)

        if context.max_seqlen_q > 0:
            max_seqlen_q = context.max_seqlen_q
        else:
            max_seqlen_q = torch.max(per_seq_wedlm_sizes).item()

        prefix_lens = context.context_lens
        k_lens = (prefix_lens + per_seq_wedlm_sizes).to(torch.int32)
        cu_seqlens_k = F.pad(k_lens.cumsum(dim=0), (1, 0)).to(torch.int32)

        return flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=self.max_context_len,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )
