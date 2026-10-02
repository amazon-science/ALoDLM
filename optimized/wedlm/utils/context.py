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

import os
from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    per_seq_wedlm_sizes: torch.Tensor | None = None 


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    per_seq_wedlm_sizes=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        per_seq_wedlm_sizes,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


# --- Loop-depth signal for the granular per-depth KV cache ---
# Set by the looped model forward before each middle-block iteration; read by
# Attention to pick the depth-d KV buffer (buffer[d]). A plain module global (not a
# Context field) because it changes WITHIN a single forward; under CUDA-graph capture
# the loop is unrolled so each iteration's depth is baked as a constant per block.
_LOOP_DEPTH = 0


def get_loop_depth() -> int:
    return _LOOP_DEPTH


def set_loop_depth(d: int):
    global _LOOP_DEPTH
    _LOOP_DEPTH = d


# --- Per-depth READOUT stash for the adaptive-depth decode ---
# When the adaptive-depth decode is on, the looped model forward stashes norm(h_d) for EVERY
# loop depth d (not just the final one) so the decoder can read the readout at all K depths
# on the mask positions and commit each token at its own exit depth d*. A plain module list
# (like _LOOP_DEPTH): populated within one forward, drained right after.
_PERDEPTH_READOUTS: list = []


def reset_perdepth_readouts():
    global _PERDEPTH_READOUTS
    _PERDEPTH_READOUTS = []


def stash_perdepth_readout(h):
    _PERDEPTH_READOUTS.append(h)


def get_perdepth_readouts() -> list:
    return _PERDEPTH_READOUTS


# --- Adaptive-depth decode config (single source of truth for the env flags) ---
# The general per-token-depth decode over the granular per-depth KV cache: the looped model
# reads out every loop depth on the masks, the decoder commits each token at its own exit
# depth d* (1 or many tokens per step), and the cache caps each committed token's KV at d*.
# Uniform-K / commit-1 are special cases (all d*=K / threshold=None).


def adaptive_depth_enabled() -> bool:
    """Master switch for the adaptive-depth decode (env WEDLM_ADAPTIVE_DEPTH)."""
    return os.environ.get("WEDLM_ADAPTIVE_DEPTH", "0") != "0"


def exit_cap_enabled() -> bool:
    """Whether committed tokens' KV is capped at their exit depth d* -- the FAITHFUL default
    of the adaptive-depth decode. Set WEDLM_ADAPTIVE_DEPTH_CAP=0 for the no-cap quality
    ceiling (full depth-matched)."""
    return adaptive_depth_enabled() and os.environ.get("WEDLM_ADAPTIVE_DEPTH_CAP", "1") != "0"
