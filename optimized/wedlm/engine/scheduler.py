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

from collections import deque
from typing import List, Optional

from wedlm.config import Config
from wedlm.engine.sequence import Sequence, SequenceStatus
from wedlm.engine.block_manager import BlockManager


class Scheduler:
    """Scheduler for managing sequence execution.
    
    Handles prefill and decode scheduling, block allocation,
    and sequence lifecycle management.
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.wedlm_window_size = config.wedlm_window_size

    def is_finished(self):
        """Check if all sequences have been processed."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """Add a new sequence to the waiting queue."""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule sequences for the next step.
        
        Returns:
            scheduled_seqs: List of sequences to process
            is_prefill: True if this is a prefill step, False for decode
        """
        # Prefill phase - process waiting sequences
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode phase with sliding window support
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # Ensure enough space for sliding window decoding
            budget = self.block_manager.ensure_space_for_sliding_window(
                seq, self.wedlm_window_size
            )

            if budget <= 0:
                # Not enough space, preempt a sequence
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break

            seq.kv_budget = budget

            # Revive sequences whose window emptied under KV-pool pressure
            # (refill capped to 0 while every window token committed): with an
            # empty window the decoder skips them forever — a silent hang. A
            # fresh state re-opens a mask window at the commit point, sized to
            # the budget we just secured.
            state = seq.wedlm_state
            if (
                state is not None
                and not state.is_finished
                and len(state.window_tokens) == 0
            ):
                seq.reset_wedlm_state()

            num_seqs += 1
            scheduled_seqs.append(seq)

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """Preempt a sequence, returning it to the waiting queue."""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        seq.reset_wedlm_state()
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], run_outputs: List[Optional[List[int]]]):
        """Process outputs from one decode step.
        
        This method processes only the generated tokens. Metrics tracking
        has been removed and is no longer handled here.
        
        Args:
            seqs: List of sequences that were processed
            run_outputs: List of generated tokens for each sequence (None if no tokens)
        
        Behavior:
        - run_outputs contains only tokens generated in this step
        - Check wedlm_state.is_finished to determine if sequence is complete
        """
        if not isinstance(run_outputs, list):
            run_outputs = list(run_outputs)

        for seq, toks in zip(seqs, run_outputs):
            # Skip if no tokens were generated
            if toks is None:
                continue

            # Ensure toks is a list
            token_list = toks if isinstance(toks, list) else [toks]

            # Process each generated token
            preempted = False
            for t in token_list:
                # KV-pool exhaustion backstop: appending this token would
                # cross into a new block (len % block_size == 1 after append),
                # but the free pool is empty — may_append() would crash the whole
                # worker with "IndexError: deque index out of range". Instead,
                # drop the rest of this step's tokens and preempt the sequence:
                # it keeps its committed tokens and re-prefills once other
                # sequences finish and release blocks.
                if (
                    (len(seq) + 1) % self.block_manager.block_size == 1
                    and len(seq.block_table) * self.block_manager.block_size
                    < len(seq) + 1
                    and not self.block_manager.free_block_ids
                ):
                    if seq in self.running:
                        self.running.remove(seq)
                    self.preempt(seq)
                    preempted = True
                    break

                seq.append_token(t)

                # Update block manager for new token
                self.block_manager.may_append(seq)

                # Check stop conditions
                is_stop_token = t in seq.stop_token_ids
                is_stop_seq = seq.check_stop()

                if (
                    is_stop_token or is_stop_seq
                ) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                    break

            if preempted:
                # Sequence was re-queued (blocks freed, wedlm_state reset);
                # skip the finished-state check below.
                continue

            # Check if wedlm_state indicates sequence is finished
            if seq.wedlm_state is not None and seq.wedlm_state.is_finished:
                if seq.status != SequenceStatus.FINISHED:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
