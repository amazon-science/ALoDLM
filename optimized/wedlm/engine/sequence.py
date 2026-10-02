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

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import List, Optional

from wedlm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Status of a sequence in the generation pipeline."""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class WeDLMState:
    """Persistent state for WeDLM sliding window decoding.
    
    This class stores the decoding state for WeDLM, allowing each step()
    to execute only one forward pass. State persists across multiple step() calls.
    
    Note: Metrics tracking (entropy, confidence) has been removed.
    The model_runner now only handles token generation.
    """
    # Current window tokens (list for pickle serialization)
    window_tokens: List[int]
    # Which positions are masks (True = mask, False = filled)
    window_mask_flags: List[bool]
    # Current sequence length committed to KV cache (excluding window tokens)
    current_seq_len: int
    # Whether generation is finished
    is_finished: bool = False
    # Whether initialized (window needs initialization before first forward)
    is_initialized: bool = False
    # Per-window-position exit depth (1-indexed loop depth d* a committed token was
    # read out at; 0 = uncommitted / no cap). Kept in lockstep with window_tokens.
    # Used by the per-token exit-depth KV cap (WEDLM_ADAPTIVE_DEPTH_CAP). Placed last so
    # dataclass default-ordering stays valid (defaulted fields after non-defaulted ones).
    window_exit_depth: List[int] = field(default_factory=list)

    def to_tuple(self):
        """Serialize to tuple for pickle."""
        return (
            self.window_tokens,
            self.window_mask_flags,
            self.current_seq_len,
            self.is_finished,
            self.is_initialized,
            self.window_exit_depth,
        )

    @classmethod
    def from_tuple(cls, data):
        """Deserialize from tuple."""
        if data is None:
            return None
        return cls(
            window_tokens=data[0],
            window_mask_flags=data[1],
            current_seq_len=data[2],
            is_finished=data[3],
            is_initialized=data[4],
            # backward-safe if an older tuple (5-elem) crosses the pickle boundary
            window_exit_depth=data[5] if len(data) > 5 else [0] * len(data[0]),
        )


class Sequence:
    """Represents a single generation sequence.
    
    This class tracks the state of a sequence throughout the generation process,
    including prompt tokens, generated tokens, and WeDLM decoding state.
    
    Note: Per-sequence metrics tracking (entropy, confidence) has been removed.
    Statistics are now computed at the engine level only.
    """
    block_size = None
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.max_tokens = sampling_params.max_tokens

        self.stop_token_ids = (
            set(sampling_params.stop_token_ids)
            if sampling_params.stop_token_ids
            else set()
        )

        # Stop sequences (token ID lists)
        self.stop_sequences = []

        # WeDLM sampling parameters
        self.wedlm_entropy_threshold = sampling_params.wedlm_entropy_threshold
        self.wedlm_pos_penalty_factor = sampling_params.wedlm_pos_penalty_factor

        # KV budget for sliding window
        self.kv_budget: int = 0

        # Persistent state for WeDLM decoding
        self.wedlm_state: Optional[WeDLMState] = None

        if Sequence.block_size is None:
            raise RuntimeError(
                "Sequence.block_size has not been initialized via Config."
            )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        """Check if the sequence has finished generation."""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """Number of tokens generated so far."""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """Get the prompt token IDs."""
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """Get the generated token IDs."""
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        """Number of blocks that have been cached."""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """Total number of blocks needed for the current sequence length."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """Number of tokens in the last block."""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """Get the token IDs in block i."""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """Append a token to the sequence."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def check_stop(self) -> bool:
        """Check if any stop sequence has been generated."""
        for stop_seq in self.stop_sequences:
            L = len(stop_seq)
            if self.num_tokens >= L:
                if self.token_ids[-L:] == stop_seq:
                    return True
        return False

    def reset_wedlm_state(self):
        """Reset WeDLM state for new generation (e.g., after preemption)."""
        self.wedlm_state = None

    def __getstate__(self):
        """Serialize state for pickle (used in multi-GPU communication)."""
        # Serialize wedlm_state
        wedlm_state_data = self.wedlm_state.to_tuple() if self.wedlm_state else None
        
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
            self.stop_token_ids,
            self.max_tokens,
            self.kv_budget,
            self.wedlm_entropy_threshold,
            self.wedlm_pos_penalty_factor,
            self.stop_sequences,
            wedlm_state_data,
            self.top_p,
            self.top_k,
        )

    def __setstate__(self, state):
        """Deserialize state from pickle."""
        self.stop_sequences = []
        self.wedlm_state = None
        self.top_p = 1.0
        self.top_k = 0

        if len(state) == 14:
            # New format with top_p and top_k
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.stop_token_ids,
                self.max_tokens,
                self.kv_budget,
                self.wedlm_entropy_threshold,
                self.wedlm_pos_penalty_factor,
                self.stop_sequences,
                wedlm_state_data,
                self.top_p,
                self.top_k,
            ) = state
        else:
            # Old format without top_p and top_k (backward compatibility)
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.stop_token_ids,
                self.max_tokens,
                self.kv_budget,
                self.wedlm_entropy_threshold,
                self.wedlm_pos_penalty_factor,
                self.stop_sequences,
                wedlm_state_data,
            ) = state

        self.wedlm_state = WeDLMState.from_tuple(wedlm_state_data)

        if self.num_completion_tokens == 0:
            self.token_ids = token_data
        else:
            self.last_token = token_data
