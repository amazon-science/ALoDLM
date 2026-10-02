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
import xxhash
import numpy as np

from wedlm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        # A 100% prefix-cache hit would make the prefill compute zero tokens
        # (seqlen_q == 0), which crashes flash_attn_varlen with "CUDA error:
        # invalid configuration argument" (seen on re-prefill of
        # preempted sequences: their length is block-aligned and every block
        # hash-hits). Recompute the last block instead.
        if seq.num_cached_tokens == len(seq):
            seq.num_cached_tokens -= self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # This engine's
        # ensure_space_for_sliding_window() pre-appends EMPTY trailing blocks to
        # block_table, so block_table[-1] is NOT necessarily the block holding
        # the newest token. Allocate only when the table doesn't already cover
        # the position, and hash the block that actually contains it.
        block_table = seq.block_table
        if len(seq) % self.block_size == 1:
            if len(block_table) * self.block_size < len(seq):
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            blk_idx = len(seq) // self.block_size - 1
            block = self.blocks[block_table[blk_idx]]
            if block.hash == -1:
                token_ids = seq.block(blk_idx)
                prefix = (
                    self.blocks[block_table[blk_idx - 1]].hash if blk_idx > 0 else -1
                )
                h = self.compute_hash(token_ids, prefix)
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block.block_id

    def get_allocated_capacity(self, seq: Sequence) -> int:
        return len(seq.block_table) * self.block_size

    def ensure_space_for_sliding_window(
        self, seq: Sequence, wedlm_window_size: int
    ) -> int:
        current_len = len(seq)
        allocated_capacity = self.get_allocated_capacity(seq)

        target_capacity = current_len + 2 * wedlm_window_size

        while allocated_capacity < target_capacity:
            if not self.free_block_ids:
                break

            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
            allocated_capacity += self.block_size

        budget = allocated_capacity - current_len
        return budget
