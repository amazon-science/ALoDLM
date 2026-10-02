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

"""
Model runner for WeDLM decoding.

This module handles model execution, KV cache management, and multi-GPU
coordination. The WeDLM decoding algorithm logic has been moved to
engine/wedlm_decoder.py for better separation of concerns.

Responsibilities:
- Model initialization and lifecycle
- KV cache allocation and management
- Multi-GPU process communication
- Prefill input preparation
- Model forward pass execution (eager and CUDA graph modes)
- Coordination with WeDLMDecoder for decode phase
"""

import pickle
import logging
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional

from wedlm.config import Config
from wedlm.engine.sequence import Sequence
from wedlm.engine.sampler import Sampler
from wedlm.engine.wedlm_decoder import WeDLMDecoder
from wedlm.models.wedlm import WeDLMForDiffusionLM
from wedlm.utils.context import (
    set_context, get_context, reset_context,
    adaptive_depth_enabled, exit_cap_enabled,
)
from wedlm.utils.loader import load_model


logger = logging.getLogger(__name__)


class ModelRunner:
    """Model runner for WeDLM decoding.
    
    Handles model execution, KV cache management, and multi-GPU coordination.
    Delegates WeDLM-specific decoding logic to WeDLMDecoder.
    
    The class is organized into several logical sections:
    - Initialization: Model loading, distributed setup, KV cache allocation
    - Process Management: Multi-GPU communication via shared memory
    - Input Preparation: Building tensors for prefill phase
    - Model Execution: Forward pass with eager or CUDA graph modes
    - Main Entry Point: The run() method that coordinates everything
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.wedlm_window_size = config.wedlm_window_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        Sequence.block_size = self.block_size

        # Setup HF config
        hf_config = config.hf_config
        hf_config.wedlm_window_size = self.wedlm_window_size
        hf_config.max_model_len = config.max_model_len

        # Initialize components
        self._init_distributed(config)
        self._init_model(hf_config)
        self._init_mask_token(config, hf_config)
        self._init_wedlm_decoder()

        # Multi-GPU communication setup
        if self.world_size > 1:
            self._init_shared_memory()

    # ========== Initialization ==========

    def _init_distributed(self, config: Config):
        """Initialize distributed training environment."""
        init_method = f"tcp://localhost:{config.nccl_port}"
        dist.init_process_group(
            "nccl", init_method, world_size=self.world_size, rank=self.rank
        )
        torch.cuda.set_device(self.rank)

    def _init_model(self, hf_config):
        """Initialize and load model along with the sampler."""
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        self.model = WeDLMForDiffusionLM(hf_config)
        load_model(self.model, self.config.model)

        # Stage-I gate decode: the in-loop stop check needs the LM head to compute mask-row
        # entropies pass-by-pass (gate_decode.loop_pass_check).
        from wedlm.engine.gate_decode import gate_enabled, set_logits_fn, set_recache_fn
        if gate_enabled():
            set_logits_fn(self.model.compute_logits)
            from wedlm.engine.gate_decode import set_embed_fn
            set_embed_fn(self.model.model.embed_tokens)
            # token-freeze emulation back-fills a row's exit-depth KV mid-loop via the
            # same granular-recache path the cross-step exit cap uses
            set_recache_fn(self._recache_granular)

        # Initialize the sampler for token sampling
        self.sampler = Sampler()
        
        self.warmup_model()
        self.allocate_kv_cache()
        
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def _init_mask_token(self, config: Config, hf_config):
        """Initialize mask token ID."""
        if config.mask_token_id is not None:
            self.mask_token_id = config.mask_token_id
        else:
            self.mask_token_id = getattr(hf_config, "mask_token_id", None)
            if self.mask_token_id is None:
                self.mask_token_id = 151665

    def _init_wedlm_decoder(self):
        """Initialize the WeDLM decoder for sliding window decoding."""
        self.wedlm_decoder = WeDLMDecoder(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            wedlm_window_size=self.wedlm_window_size,
            sampler=self.sampler,
        )

    def _init_shared_memory(self):
        """Initialize shared memory for multi-GPU communication."""
        if self.rank == 0:
            self.shm = SharedMemory(name="wedlm", create=True, size=2**20)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name="wedlm")
            self.loop()

    # ========== Process Management ==========

    def exit(self):
        """Clean up resources and terminate."""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        if hasattr(self, "_passgraphs"):
            # lazily captured under WEDLM_PASS_GRAPH; must die before the process group
            del self._passgraphs
            self.graph_pool = None
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Worker loop for non-master ranks."""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Read method call from shared memory."""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """Write method call to shared memory."""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """Call method, broadcasting to workers if needed."""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ========== Model Setup ==========

    def warmup_model(self):
        """Warmup model with dummy input to trigger JIT compilation."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """Allocate KV cache based on available GPU memory."""
        config = self.config
        hf_config = config.hf_config
        
        # Calculate available memory
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # Calculate block size in bytes
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )


        import os as _os
        _ls = int(_os.environ.get("WEDLM_LOOP_START", "-1"))
        _le = int(_os.environ.get("WEDLM_LOOP_END", "-1"))
        _K = int(_os.environ.get("WEDLM_LOOP_K", "1"))
        _perdepth = _ls >= 0 and _le > _ls and _K > 1
        _N = hf_config.num_hidden_layers
        self._depth_of = [(_K if (_perdepth and _ls <= i < _le) else 1) for i in range(_N)]
        self._effective_layers = sum(self._depth_of)
        if _perdepth:
            logger.info(
                f"[granular-kv] per-depth ON: layers[{_ls}:{_le}] x K={_K} -> "
                f"effective_layers={self._effective_layers} (base {_N})"
            )

        block_bytes = (
            2
            * self._effective_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        
        # Allocate cache
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        if config.num_kvcache_blocks <= 0:
            logger.warning(
                f"num_kvcache_blocks ({config.num_kvcache_blocks}) is <= 0. "
                "Setting it to 1."
            )
        config.num_kvcache_blocks = max(config.num_kvcache_blocks, 1)
        
        self.kv_cache = torch.empty(
            2,
            self._effective_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        # Assign cache to attention layers. Looped layers (depth_of>1) get a list of
        # K per-depth buffers; the model forward selects buffer[d] on loop-iteration d.
        layer_id = 0
        slot = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                nd = self._depth_of[layer_id]
                if nd > 1:
                    module.k_cache_depths = [self.kv_cache[0, slot + d] for d in range(nd)]
                    module.v_cache_depths = [self.kv_cache[1, slot + d] for d in range(nd)]
                    module.k_cache = module.k_cache_depths[0]
                    module.v_cache = module.v_cache_depths[0]
                else:
                    module.k_cache = self.kv_cache[0, slot]
                    module.v_cache = self.kv_cache[1, slot]
                    module.k_cache_depths = None
                    module.v_cache_depths = None
                slot += nd
                layer_id += 1

    # ========== Input Preparation ==========

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """Prepare block tables for batch processing.
        
        Pads block tables to the same length and converts to tensor.
        
        Args:
            seqs: List of sequences
            
        Returns:
            Padded block tables tensor on GPU
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare inputs for prefill phase.
        
        Builds input tensors and sets up attention context for prefill.
        
        Args:
            seqs: List of sequences to prefill
            
        Returns:
            Tuple of (input_ids, positions) tensors
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    # ========== Model Execution ==========

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ) -> torch.Tensor:
        """Run model forward pass.
        
        Chooses between eager execution and CUDA graph based on configuration
        and input size.
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            is_prefill: Whether this is a prefill (vs decode) step
            
        Returns:
            Model logits
        """
        pass_cfg = None if is_prefill else self._passgraph_config(input_ids.size(0))
        hidden_states = None
        if pass_cfg is not None:
            # PER-PASS graphs: one captured graph per loop depth, the gate stop-check and the
            # readout stash run on the host BETWEEN replays. Overrides enforce_eager on
            # purpose -- enforce_eager exists to dodge the whole-forward capture below.
            # Returns None when this step does not fit the captured buffers -> eager.
            hidden_states = self._run_pass_graphs(input_ids, positions, *pass_cfg)

        if hidden_states is not None:
            logits = self.model.compute_logits(hidden_states)
        elif is_prefill or self.enforce_eager or adaptive_depth_enabled() or input_ids.size(0) > 512:
            # adaptive-depth MUST be eager here: the whole-forward CUDA-graph path replays a
            # single captured final-output buffer, so the per-depth readout stash would never
            # populate (use the per-pass path above instead).
            hidden_states = self.model(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
        else:
            logits = self._run_with_cudagraph(input_ids, positions)

        if self.rank == 0 and is_prefill:
            context = get_context()
            last_indices = context.cu_seqlens_q[1:] - 1
            logits = logits[last_indices]

        return logits

    def _run_with_cudagraph(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run model using CUDA graph for better performance.
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            
        Returns:
            Model logits
        """
        context = get_context()
        real_bs = context.per_seq_wedlm_sizes.size(0)

        try:
            graph_seq_capacity = next(x for x in self.graph_bs if x >= real_bs)
        except StopIteration:
            hidden_states = self.model(input_ids, positions)
            return self.model.compute_logits(hidden_states)

        graph = self.graphs[graph_seq_capacity]
        graph_vars = self.graph_vars[graph_seq_capacity]
        num_tokens = input_ids.size(0)

        if num_tokens > graph_vars["input_ids"].size(0):
            hidden_states = self.model(input_ids, positions)
            return self.model.compute_logits(hidden_states)

        # Copy inputs to graph buffers
        graph_vars["input_ids"][:num_tokens].copy_(input_ids)
        graph_vars["positions"][:num_tokens].copy_(positions)
        graph_vars["per_seq_wedlm_sizes"][:real_bs].copy_(context.per_seq_wedlm_sizes)
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:num_tokens].copy_(context.slot_mapping)
        graph_vars["context_lens"][:real_bs].copy_(context.context_lens)

        if context.block_tables is not None:
            valid_rows = min(real_bs, context.block_tables.size(0))
            valid_cols = min(
                graph_vars["block_tables"].size(1), context.block_tables.size(1)
            )
            graph_vars["block_tables"][:valid_rows, :valid_cols].copy_(
                context.block_tables[:valid_rows, :valid_cols]
            )

        graph.replay()

        hidden_states = graph_vars["outputs"][:num_tokens]
        return self.model.compute_logits(hidden_states)

    # ========== Per-pass CUDA graphs (looped decode) ==========
    # The whole-forward capture above cannot serve the looped decode: it bakes ONE unrolled
    # K into a single graph, its lone output buffer never populates the per-depth readout
    # stash, and the gate's compute-then-stop is host control flow mid-forward. Capturing one
    # graph PER LOOP PASS instead keeps every one of those on the host between replays, so
    # adaptive depth is preserved and only the layer stack (the launch-bound part) is graphed.
    # Env-gated: WEDLM_PASS_GRAPH=1. Capture is LAZY (first qualifying decode step) because
    # the loop topology comes from env that the eval harness sets AFTER engine construction --
    # capturing at init is what silently froze K=1 into the old graphs.

    def _passgraph_config(self, num_tokens: int):
        """(ls, le, K) if the per-pass graph path can serve this step, else None."""
        import os as _os
        if _os.environ.get("WEDLM_PASS_GRAPH", "0") == "0":
            return None
        if _os.environ.get("WEDLM_TOKEN_FREEZE", "0") != "0":
            return None          # mutates slot_mapping mid-forward: capture-hostile


        ls = int(_os.environ.get("WEDLM_LOOP_START", "-1"))
        le = int(_os.environ.get("WEDLM_LOOP_END", "-1"))
        K = int(_os.environ.get("WEDLM_LOOP_K", "1"))
        n_layers = len(self.model.model.layers)
        if not (0 <= ls < le <= n_layers and K > 1):
            return None          # no loop => nothing to capture per pass
        if int(_os.environ.get("WEDLM_LOOP_CARRY_NORM", "0")) == 0:
            return None          # only carry-norm collapses the pass boundary to one tensor
        if num_tokens > int(_os.environ.get("WEDLM_PASS_GRAPH_MAX_TOKENS", "512")):
            return None          # large batches are FLOP-bound; graphs buy latency, not that
        return ls, le, K

    @torch.inference_mode()
    def _passgraph_entry(self, num_seqs: int, ls: int, le: int, K: int):
        """Capture (once) and return the pass-graph family for this seq-count bucket."""
        if not hasattr(self, "_passgraphs"):
            self._passgraphs = {}
            if not hasattr(self, "graph_pool"):
                self.graph_pool = None
        key = (num_seqs, K, ls, le)
        if key in self._passgraphs:
            return self._passgraphs[key]

        config = self.config
        hf_config = config.hf_config
        base = self.model.model
        step_size = 2 * max(1, int(self.wedlm_window_size or 1))
        max_tokens = num_seqs * step_size
        # ensure_space_for_sliding_window pre-allocates len(seq) + 2*window, so a near-max-len
        # sequence needs one more block column than ceil(max_model_len/block_size).
        max_num_blocks = (
            config.max_model_len + 2 * step_size + self.block_size - 1
        ) // self.block_size

        gv = dict(
            input_ids=torch.zeros(max_tokens, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(max_tokens, dtype=torch.int64, device="cuda"),
            slot_mapping=torch.full((max_tokens,), -1, dtype=torch.int32, device="cuda"),
            context_lens=torch.zeros(num_seqs, dtype=torch.int32, device="cuda"),
            block_tables=torch.zeros(num_seqs, max_num_blocks, dtype=torch.int32, device="cuda"),
            per_seq_wedlm_sizes=torch.full((num_seqs,), step_size, dtype=torch.int32, device="cuda"),
        )
        # One readout buffer PER DEPTH: pass d reads readouts[d-1] and writes readouts[d], so
        # every address crossing a pass boundary is static AND each depth's readout survives
        # the whole step for the decoder's logits_by_depth (no ping-pong, nothing clobbered).
        # dtype MUST be named: capture is lazy, so it runs after _init_model restored the
        # ambient default to fp32 -- a default-dtype buffer would feed fp32 back into bf16
        # weights (and RMSNorm's .float() no-op would then clobber it in place).
        readouts = [
            torch.zeros(
                max_tokens, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda"
            )
            for _ in range(K)
        ]
        assert readouts[0].dtype == next(self.model.parameters()).dtype
        gv["readouts"] = readouts

        saved = get_context()
        set_context(
            False,
            slot_mapping=gv["slot_mapping"],          # all -1: capture stores nothing
            context_lens=gv["context_lens"],
            block_tables=gv["block_tables"],
            per_seq_wedlm_sizes=gv["per_seq_wedlm_sizes"],
            max_seqlen_q=step_size,
        )
        n_layers = len(base.layers)
        sandwich = ls > 0 or le < n_layers
        if sandwich:
            # Sandwich family. Buffers: prelude_out (un-normed fused stream, written once per
            # step), and per depth the pre-norm loop output raws[d] (what the coda branch
            # reads) + its RMSNorm carries[d] (the next pass's input; under COA the host
            # writes committed-token embeddings into carries[d] between replays). For
            # le == N the coda is empty and carries[d] doubles as the depth-d readout, so
            # raws/coda graphs are skipped (readout == carry, all-layer semantics).
            gv["prelude_out"] = torch.zeros_like(readouts[0])
            gv["carries"] = readouts          # reuse: carries ARE the readouts when le == N
            has_coda = le < n_layers
            if has_coda:
                gv["raws"] = [torch.zeros_like(readouts[0]) for _ in range(K)]
                gv["rds"] = [torch.zeros_like(readouts[0]) for _ in range(K)]

            def cap(fn):
                fn()                                            # warmup
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, self.graph_pool):
                    fn()
                if self.graph_pool is None:
                    self.graph_pool = g.pool()
                return g

            def _prelude():
                gv["prelude_out"][:] = base.prelude(gv["positions"], gv["input_ids"], ls)
            g_pre = cap(_prelude)
            loop_graphs, coda_graphs = [], []
            for d in range(K):
                src = gv["prelude_out"] if d == 0 else gv["carries"][d - 1]
                if has_coda:
                    def _loop(d=d, src=src):
                        raw = base.loop_pass_raw(gv["positions"], src, d, ls, le)
                        gv["raws"][d][:] = raw
                        gv["carries"][d][:] = base.norm(raw)
                    loop_graphs.append(cap(_loop))
                    def _coda(d=d):
                        gv["rds"][d][:] = base.coda_readout(gv["positions"], gv["raws"][d], le)
                    coda_graphs.append(cap(_coda))
                else:
                    def _loop(d=d, src=src):
                        gv["carries"][d][:] = base.loop_pass(gv["positions"], src, d, ls, le)
                    loop_graphs.append(cap(_loop))


            check_graphs = None
            import os as _os
            _chk_mode = _os.environ.get("WEDLM_GATE_STOP", "consensus")
            _left1 = _chk_mode == "gate_depth_left1"
            if (_os.environ.get("WEDLM_CHECK_GRAPH", "0") == "1"
                    and _chk_mode in ("gate_depth", "gate_depth_left1")
                    and num_seqs == 1 and has_coda):
                from wedlm.engine import gate_decode as _gd
                gate_mod = _gd._ensure_loaded(
                    hf_config.hidden_size, torch.device("cuda"), hf_config.dtype)
                thr_bake = float(_os.environ.get("WEDLM_CHECK_THR", "nan"))
                gv["chk_mask"] = torch.zeros(max_tokens, dtype=torch.bool, device="cuda")
                gv["chk_pen"] = torch.zeros(max_tokens, dtype=torch.float32, device="cuda")
                gv["chk_thr"] = torch.zeros((), dtype=torch.float32, device="cuda")
                gv["chk_surv"] = torch.ones(max_tokens, dtype=torch.float32, device="cuda")
                gv["chk_committed"] = torch.zeros(max_tokens, dtype=torch.bool, device="cuda")
                gv["chk_committed_at"] = torch.full((max_tokens,), -1, dtype=torch.long,
                                                    device="cuda")
                gv["chk_packed"] = torch.zeros(4, dtype=torch.float32, device="cuda")
                # phase 3a: per-depth greedy tokens saved during the pass so the decoder
                # can consume them directly (skips the end-of-step K x lm_head rebuild).
                gv["chk_argmax"] = torch.zeros(K, max_tokens, dtype=torch.long, device="cuda")
                gv["chk_arange"] = torch.arange(max_tokens, dtype=torch.long, device="cuda")
                emb_w = self.model.model.embed_tokens
                logits_fn = self.model.compute_logits
                check_graphs = []
                for d in range(K):
                    def _check(d=d):
                        rows = gv["rds"][d]
                        lam = torch.sigmoid(gate_mod(rows, d).float())
                        lf = logits_fn(rows).float()
                        lse = torch.logsumexp(lf, dim=-1)
                        ent = lse - (torch.softmax(lf, dim=-1) * lf).sum(dim=-1)
                        m = gv["chk_mask"]
                        gv["chk_surv"].mul_(torch.where(m, 1.0 - lam, torch.ones_like(lam)))
                        cdf = 1.0 - gv["chk_surv"]
                        adj = ent + gv["chk_pen"]
                        if _left1:
                            # gate_depth_left1: commit exactly the LEFTMOST uncommitted
                            # mask row (entropy plays no role in selection).
                            avail = m & (~gv["chk_committed"])
                            pos = gv["chk_arange"]
                            idxv = torch.where(avail, pos,
                                               torch.full_like(pos, pos.numel()))
                            agree = avail & (pos == idxv.argmin())
                        else:
                            agree = m & (adj < gv["chk_thr"]) & (~gv["chk_committed"])
                        gv["chk_committed_at"].copy_(
                            torch.where(agree, torch.full_like(gv["chk_committed_at"], d),
                                        gv["chk_committed_at"]))
                        newc = gv["chk_committed"] | agree
                        # COA injection: agree rows' greedy token embedding becomes the
                        # next pass's loop input (write into carries[d], graph d+1's source)
                        tok = lf.argmax(dim=-1)
                        gv["chk_argmax"][d].copy_(tok)
                        emb = emb_w(tok).to(gv["carries"][d].dtype)
                        gv["carries"][d].copy_(
                            torch.where(agree.unsqueeze(-1), emb, gv["carries"][d]))
                        resid = m & (~newc)
                        gv["chk_committed"].copy_(newc)
                        gv["chk_packed"].copy_(torch.stack([
                            agree.sum().float(), resid.sum().float(),
                            (cdf * resid.float()).sum(),
                            torch.where(m, adj, torch.full_like(adj, float("inf"))
                                        ).argmin().float(),
                        ]))
                    check_graphs.append(cap(_check))
                del thr_bake
            torch.cuda.synchronize()
            entry = dict(graphs=loop_graphs, coda_graphs=coda_graphs, prelude_graph=g_pre,
                         vars=gv, sandwich=True, has_coda=has_coda,
                         check_graphs=check_graphs)
            # restore the live context below, then cache
            graphs = loop_graphs
        else:
            graphs = []
            for d in range(K):
                src = base.embed(gv["input_ids"]) if d == 0 else readouts[d - 1]
                readouts[d][:] = base.loop_pass(gv["positions"], src, d, ls, le)   # warmup
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.graph_pool):
                    src_g = base.embed(gv["input_ids"]) if d == 0 else readouts[d - 1]
                    readouts[d][:] = base.loop_pass(gv["positions"], src_g, d, ls, le)
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                graphs.append(graph)
            torch.cuda.synchronize()
            entry = dict(graphs=graphs, vars=gv, sandwich=False, has_coda=False)

        # restore the live step's context (capture ran inside a real decode step)
        set_context(
            saved.is_prefill,
            cu_seqlens_q=saved.cu_seqlens_q,
            cu_seqlens_k=saved.cu_seqlens_k,
            max_seqlen_q=saved.max_seqlen_q,
            max_seqlen_k=saved.max_seqlen_k,
            slot_mapping=saved.slot_mapping,
            context_lens=saved.context_lens,
            block_tables=saved.block_tables,
            per_seq_wedlm_sizes=saved.per_seq_wedlm_sizes,
        )
        self._passgraphs[key] = entry
        return entry

    @torch.inference_mode()
    def _run_pass_graphs(
        self, input_ids: torch.Tensor, positions: torch.Tensor, ls: int, le: int, K: int
    ) -> torch.Tensor:
        """Replay one graph per loop pass; gate stop-check + readout stash on the host between."""
        from wedlm.utils.context import (
            reset_perdepth_readouts, stash_perdepth_readout,
        )
        from wedlm.utils import effstats
        from wedlm.engine.gate_decode import loop_pass_check

        context = get_context()
        real_bs = context.per_seq_wedlm_sizes.size(0)
        num_tokens = input_ids.size(0)
        step_size = 2 * max(1, int(self.wedlm_window_size or 1))
        num_seqs = 1
        while num_seqs < real_bs or num_seqs * step_size < num_tokens:
            num_seqs *= 2

        entry = self._passgraph_entry(num_seqs, ls, le, K)
        gv, graphs = entry["vars"], entry["graphs"]
        if (
            context.block_tables is not None
            and context.block_tables.size(1) > gv["block_tables"].size(1)
        ):
            return None      # wider than captured: refuse rather than silently truncate

        # Step inputs are constant across passes -> copy in ONCE (the eager path re-reads them
        # every pass). Same stale-tail contract as the whole-forward path: only slot_mapping is
        # scrubbed, pad rows can never store (slot -1) and their outputs are sliced off.
        gv["input_ids"][:num_tokens].copy_(input_ids)
        gv["positions"][:num_tokens].copy_(positions)
        gv["per_seq_wedlm_sizes"][:real_bs].copy_(context.per_seq_wedlm_sizes)
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:num_tokens].copy_(context.slot_mapping)
        gv["context_lens"][:real_bs].copy_(context.context_lens)
        if context.block_tables is not None:
            rows = min(real_bs, context.block_tables.size(0))
            cols = min(gv["block_tables"].size(1), context.block_tables.size(1))
            gv["block_tables"][:rows, :cols].copy_(context.block_tables[:rows, :cols])

        adaptive = adaptive_depth_enabled()
        if adaptive:
            reset_perdepth_readouts()
        effstats.add_forward(num_tokens)

        from wedlm.engine.gate_decode import commit_on_agree_enabled, commit_on_agree_apply
        coa = adaptive and commit_on_agree_enabled()
        rd = None
        if entry.get("sandwich"):
            has_coda = entry["has_coda"]
            n_layers = len(self.model.model.layers)
            entry["prelude_graph"].replay()
            if ls > 0:
                effstats.add_pass(num_tokens, ls)
            # infra pass 2b: if the captured check-graph family can serve this step, the
            # whole between-pass eager section (gate+lm_head+entropy+COA injection) runs
            # in-graph and the host reads one packed [4] buffer per pass.
            use_chk = False
            if entry.get("check_graphs") is not None and adaptive and coa:
                from wedlm.engine.gate_decode import graphed_seed
                use_chk = graphed_seed(gv)
            for d in range(K):
                graphs[d].replay()
                effstats.add_pass(num_tokens, le - ls, loop_block=True)
                if has_coda:
                    entry["coda_graphs"][d].replay()
                    effstats.add_pass(num_tokens, n_layers - le)
                    rd = gv["rds"][d][:num_tokens]
                else:
                    rd = gv["carries"][d][:num_tokens]      # empty coda: readout == carry
                if adaptive:
                    stash_perdepth_readout(rd.clone() if coa else rd)
                    if use_chk:
                        entry["check_graphs"][d].replay()
                        from wedlm.engine.gate_decode import graphed_gate_depth_step
                        if graphed_gate_depth_step(gv, d, K) and d < K - 1:
                            break
                        # COA injection for agree rows already happened inside the check
                        # graph (write into carries[d], graph d+1's captured source).
                        continue
                    if loop_pass_check(rd, d, K) and d < K - 1:
                        break
                    if coa and d < K - 1:
                        # selection on the readout, injection into the CARRY buffer -- the
                        # captured source of graph d+1 (mirrors eager's sandwich wiring; for
                        # le == N the carry buffer IS the readout buffer, all-layer style).
                        carry = commit_on_agree_apply(
                            rd, d, K,
                            carry=gv["carries"][d][:num_tokens] if has_coda else None)
                        if carry is not None:
                            gv["carries"][d][:num_tokens].copy_(carry)
            return rd
        for d in range(K):
            graphs[d].replay()
            rd = gv["readouts"][d][:num_tokens]
            effstats.add_pass(num_tokens, le - ls, loop_block=True)
            if adaptive:
                # Under COA the stash must be a CLONE: the stash holds a reference, and the
                # write-back below mutates the static buffer it would otherwise point into
                # (eager gets this for free because commit_on_agree_apply returns a copy).
                stash_perdepth_readout(rd.clone() if coa else rd)
                # compute-then-stop: no deeper pass is replayed once every seq has decided
                if loop_pass_check(rd, d, K) and d < K - 1:
                    break
                if coa:
                    carry = commit_on_agree_apply(rd, d, K)
                    if carry is not None and d < K - 1:
                        # graph d+1's captured source address IS readouts[d]: writing the
                        # embedding-substituted carry here is exactly eager's "use the copy
                        # as the next pass's carry", expressed through the static buffer.
                        gv["readouts"][d][:num_tokens].copy_(carry)
        return rd

    # ========== WeDLM Decode Step ==========

    @torch.inference_mode()
    def _wedlm_decode_one_step(
        self, seqs: List[Sequence]
    ) -> List[Optional[List[int]]]:
        """Execute one step of WeDLM decoding (single forward pass).
        
        This method coordinates with WeDLMDecoder:
        1. WeDLMDecoder initializes states
        2. WeDLMDecoder prepares inputs
        3. ModelRunner runs the model
        4. WeDLMDecoder processes outputs
        
        Args:
            seqs: List of sequences to process
            
        Returns:
            List of generated tokens for each sequence (None if no tokens)
        """
        # Initialize WeDLM states for sequences that don't have one
        self.wedlm_decoder.initialize_states(seqs)

        # Initialize results - one entry per input sequence
        step_results: List[Optional[List[int]]] = [None for _ in seqs]

        # Prepare decode inputs (returns None if no active sequences)
        prepared = self.wedlm_decoder.prepare_decode_inputs(seqs)
        
        if prepared is None:
            return step_results

        # Set context for attention
        context = prepared.context
        set_context(
            False,
            slot_mapping=context.slot_mapping,
            context_lens=context.context_lens,
            block_tables=context.block_tables,
            per_seq_wedlm_sizes=context.per_seq_wedlm_sizes,
            max_seqlen_q=context.max_seqlen_q,
        )

        # Run model forward pass
        logits = self.run_model(prepared.input_ids, prepared.positions, is_prefill=False)


        import os as _os2


        if _os2.environ.get("WEDLM_COMMIT_ON_AGREE", "0") != "0":
            pass
        elif exit_cap_enabled() and getattr(context, "exit_depths", None) is not None:
            # per-token adaptive-depth cap (default when adaptive-depth is on): cap each
            # committed slot's KV at its own recorded exit depth d*.
            self._recache_granular(context.slot_mapping, context.exit_depths)

        # adaptive-depth: gather the per-depth readouts stashed by the looped forward and turn
        # each into logits, so the decoder can commit each token at its own exit depth d*.
        logits_by_depth = None
        if adaptive_depth_enabled():
            # phase 3a: when the check-graph path already recorded every committed token
            # (greedy), the decoder consumes them directly -- skip rebuilding K full-stream
            # lm_heads over the stashed readouts (they would reproduce the same argmaxes).
            from wedlm.engine.gate_decode import graphed_all_tokens_ready
            all_greedy = all(getattr(s, "temperature", 0.0) == 0 for s in seqs)
            if not (all_greedy and graphed_all_tokens_ready()):
                from wedlm.utils.context import get_perdepth_readouts
                readouts = get_perdepth_readouts()
                if readouts:
                    logits_by_depth = [self.model.compute_logits(rd) for rd in readouts]

        # Process outputs through WeDLMDecoder
        step_results = self.wedlm_decoder.process_decode_outputs(
            seqs, prepared, logits, logits_by_depth=logits_by_depth
        )

        reset_context()
        return step_results

    @torch.inference_mode()
    def _recache_granular(self, slot_mapping, commit_depth):
        """Granular cache back-fill: cap the just-processed generated slots at exit depth
        c by copying per-depth buffer[c-1] into buffers[c..K-1] at those slots -> a future
        depth-d query reads buffer[min(d, c)] (accurate key if it reached depth d, else its
        deepest/exit key, never a key deeper than the query).
        The active window was stored per-depth during its own loop (self-attention stays
        correct); this only caps what FUTURE tokens read. c>=K is a no-op (full depth-matched).

        `commit_depth` is either an int (UNIFORM exit depth, scalar fast-path kept for
        tests) OR an int tensor aligned 1:1 with slot_mapping (PER-TOKEN
        exit depth -- WEDLM_ADAPTIVE_DEPTH_CAP; each slot capped at its own d*, sentinel 0 =
        uncommitted/no-cap, c>=K = full-depth no-op). Uses the identical `slot_mapping >= 0`
        mask for both paths, then buckets slots by their exit depth c."""
        if slot_mapping is None:
            return
        valid = slot_mapping >= 0
        slots = slot_mapping[valid]
        if slots.numel() == 0:
            return
        per_slot = torch.is_tensor(commit_depth)
        depths = commit_depth[valid] if per_slot else None  # SAME mask as slots
        for module in self.model.modules():
            kd = getattr(module, "k_cache_depths", None)
            vd = getattr(module, "v_cache_depths", None)
            if kd is None or vd is None:
                continue
            K = len(kd)
            if not per_slot:
                # unchanged scalar fast-path (uniform exit depth)
                c = int(commit_depth)
                if c >= K:
                    continue
                H, D = kd[c - 1].shape[-2], kd[c - 1].shape[-1]
                ksrc = kd[c - 1].view(-1, H, D)[slots]
                vsrc = vd[c - 1].view(-1, H, D)[slots]
                for j in range(c, K):
                    kd[j].view(-1, H, D)[slots] = ksrc
                    vd[j].view(-1, H, D)[slots] = vsrc
            else:
                # per-token: bucket slots by their exit depth c in [1, K-1].
                # c=0 (uncommitted) and c>=K (exited deepest) select nothing -> no-op.
                for c in range(1, K):
                    sel = slots[depths == c]
                    if sel.numel() == 0:
                        continue
                    H, D = kd[c - 1].shape[-2], kd[c - 1].shape[-1]
                    ksrc = kd[c - 1].view(-1, H, D)[sel]
                    vsrc = vd[c - 1].view(-1, H, D)[sel]
                    for j in range(c, K):
                        kd[j].view(-1, H, D)[sel] = ksrc
                        vd[j].view(-1, H, D)[sel] = vsrc

    # ========== Main Entry Point ==========

    def run(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> List[Optional[List[int]]]:
        """Execute one inference step.
        
        This method is responsible only for token generation.
        The outer layer (LLMEngine) can determine the number of active sequences
        by examining the input seqs list.
        
        Args:
            seqs: List of sequences to process
            is_prefill: True for prefill phase, False for decode phase
            
        Returns:
            List of generated tokens for each sequence (None for prefill or
            if no tokens generated)
        """
        if is_prefill:
            # Prefill phase: process prompts and fill KV cache
            input_ids, positions = self.prepare_prefill(seqs)
            _ = self.run_model(input_ids, positions, is_prefill=True)
            reset_context()
            return [None for _ in seqs]

        # Decode phase: generate tokens using WeDLM decoding
        return self._wedlm_decode_one_step(seqs)

    # ========== CUDA Graph Capture ==========

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Capture CUDA graphs for efficient decode execution.
        
        Creates CUDA graphs for various batch sizes to accelerate
        decode-phase model execution.
        """
        config = self.config
        hf_config = config.hf_config
        max_seqs = config.max_num_seqs

        # Determine batch sizes to capture (powers of 2 up to max_seqs)
        self.graph_bs = []
        current_bs = 1
        while current_bs <= max_seqs:
            self.graph_bs.append(current_bs)
            current_bs *= 2

        base_step_size = (
            self.wedlm_window_size if self.wedlm_window_size is not None else 1
        )
        capture_step_size = 2 * max(1, int(base_step_size))
        max_num_blocks = (
            (config.max_model_len + self.block_size - 1) // self.block_size
        )

        self.graphs = {}
        self.graph_vars = {}
        self.graph_pool = None

        for num_seqs in self.graph_bs:
            max_tokens_in_bucket = num_seqs * capture_step_size

            # Allocate buffers for graph capture
            input_ids = torch.zeros(
                max_tokens_in_bucket, dtype=torch.int64, device="cuda"
            )
            positions = torch.zeros(
                max_tokens_in_bucket, dtype=torch.int64, device="cuda"
            )
            slot_mapping = torch.full(
                (max_tokens_in_bucket,), -1, dtype=torch.int32, device="cuda"
            )
            context_lens = torch.zeros(num_seqs, dtype=torch.int32, device="cuda")
            block_tables = torch.zeros(
                num_seqs, max_num_blocks, dtype=torch.int32, device="cuda"
            )
            per_seq_wedlm_sizes = torch.full(
                (num_seqs,), capture_step_size, dtype=torch.int32, device="cuda"
            )
            outputs = torch.zeros(
                max_tokens_in_bucket, hf_config.hidden_size, device="cuda"
            )

            graph = torch.cuda.CUDAGraph()

            set_context(
                False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                per_seq_wedlm_sizes=per_seq_wedlm_sizes,
                max_seqlen_q=capture_step_size,
            )

            # Warmup run before capture
            outputs[:] = self.model(input_ids, positions)

            # Capture the graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:] = self.model(input_ids, positions)

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[num_seqs] = graph
            self.graph_vars[num_seqs] = dict(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                per_seq_wedlm_sizes=per_seq_wedlm_sizes,
                outputs=outputs,
            )

            reset_context()

        torch.cuda.synchronize()
