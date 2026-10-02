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
import atexit
from dataclasses import fields
from time import perf_counter
from typing import Generator, Dict, Any, List, Union

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from wedlm.config import Config
from wedlm.sampling_params import SamplingParams
from wedlm.engine.sequence import Sequence
from wedlm.engine.scheduler import Scheduler
from wedlm.engine.model_runner import ModelRunner


class LLMEngine:
    """Main engine for LLM inference with WeDLM decoding.
    
    This class coordinates the scheduler and model runner, and handles
    statistics tracking including decode forward counts.
    """

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        Sequence.block_size = config.kvcache_block_size

        # Initialize worker processes for tensor parallelism
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # Initialize main model runner
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True, trust_remote_code=True
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        atexit.register(self.exit)

    def exit(self):
        """Clean up resources and terminate worker processes."""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def _create_sequence(
        self, prompt: list[int], sampling_params: SamplingParams
    ) -> Sequence:
        """Create a new sequence with the given prompt and sampling parameters."""
        if sampling_params.stop_token_ids is None:
            sampling_params.stop_token_ids = []

        if (
            self.eos_token_id is not None
            and self.eos_token_id not in sampling_params.stop_token_ids
        ):
            sampling_params.stop_token_ids = list(sampling_params.stop_token_ids) + [
                self.eos_token_id
            ]

        seq = Sequence(prompt, sampling_params)

        # Parse stop strings into token sequences
        if sampling_params.stop:
            for stop_str in sampling_params.stop:
                stop_ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if stop_ids:
                    seq.stop_sequences.append(stop_ids)

        return seq

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Add a new generation request to the engine."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = self._create_sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _count_active_sequences(self, seqs: list[Sequence]) -> int:
        """Count the number of active sequences for decode phase.
        
        An active sequence is one that has a wedlm_state, is not finished,
        and has tokens in its window.
        """
        count = 0
        for seq in seqs:
            state = seq.wedlm_state
            if state is not None and not state.is_finished and len(state.window_tokens) > 0:
                count += 1
        return count

    def step(self):
        """Execute one inference step.
        
        Behavior:
        - Prefill: Process prompt, fill KV cache (one forward pass)
        - Decode: Execute one WeDLM decoding step (one forward pass)
        
        Each step() call executes only one forward pass.
        The number of decode forwards is determined by counting active sequences
        in the outer layer rather than being returned by model_runner.
        
        Returns:
            outputs: List of (seq_id, completion_token_ids) for finished sequences
            num_tokens: Positive for prefill token count, negative for decode token count
            num_forwards: Number of forward passes (1 for prefill, active_seq_count for decode)
        """
        seqs, is_prefill = self.scheduler.schedule()

        # Run model - returns only generated tokens, no metrics
        run_results = self.model_runner.call("run", seqs, is_prefill)

        # Process results through scheduler
        self.scheduler.postprocess(seqs, run_results)

        # Collect finished sequences
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]

        if is_prefill:
            # Prefill: count input tokens, 1 forward pass
            num_tokens = sum(len(seq) for seq in seqs)
            num_forwards = 1
        else:
            # Decode: count generated tokens (negative), count active sequences for forwards
            gen = 0
            if isinstance(run_results, list):
                for item in run_results:
                    if item is None:
                        continue
                    if isinstance(item, list):
                        gen += len(item)
                    else:
                        gen += 1
            num_tokens = -gen
            
            # Count active sequences - this replaces the num_forwards returned by model_runner
            num_forwards = self._count_active_sequences(seqs)

        return outputs, num_tokens, num_forwards

    def is_finished(self):
        """Check if all requests have been processed."""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Generate completions for a batch of prompts.
        
        Args:
            prompts: List of prompts (strings or token ID lists)
            sampling_params: Sampling parameters (single or per-prompt)
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of result dictionaries containing text, token_ids, and stats
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        seq_map = {}

        # Add all requests to the scheduler
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                prompt_ids = self.tokenizer.encode(prompt)
            else:
                prompt_ids = prompt
            seq = self._create_sequence(prompt_ids, sp)
            self.scheduler.add(seq)
            seq_map[seq.seq_id] = seq

        outputs = {}

        # Statistics tracking
        total_prefill_tokens = 0
        total_prefill_time = 0.0
        from wedlm.utils import effstats as _effstats
        _effstats.reset()   # per-generate-call deltas for worker aggregation
        total_decode_tokens = 0
        total_decode_time = 0.0
        total_decode_forwards = 0
        prefill_throughput = decode_throughput = 0.0

        # Main generation loop
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, num_forwards = self.step()
            dt = perf_counter() - t

            if num_tokens > 0:
                # Prefill phase
                total_prefill_tokens += num_tokens
                total_prefill_time += dt
                prefill_throughput = num_tokens / dt
            else:
                # Decode phase
                decoded = -num_tokens
                total_decode_tokens += decoded
                total_decode_time += dt
                total_decode_forwards += num_forwards
                decode_throughput = decoded / dt if dt > 0 else 0.0

            if use_tqdm:
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # Calculate final statistics
        avg_prefill_speed = (
            total_prefill_tokens / total_prefill_time if total_prefill_time > 0 else 0.0
        )
        avg_decode_speed = (
            total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0
        )
        tokens_per_forward = (
            total_decode_tokens / total_decode_forwards
            if total_decode_forwards > 0
            else 0.0
        )

        print(
            f"[Stats] decode_tokens={total_decode_tokens}, decode_forwards={total_decode_forwards}, tokens_per_forward={tokens_per_forward:.4f}"
        )

        from wedlm.utils import effstats
        global_stats = {
            "prefill_speed": avg_prefill_speed,
            "decode_speed": avg_decode_speed,
            "prefill_tokens": total_prefill_tokens,
            "decode_tokens": total_decode_tokens,
            "decode_forwards": total_decode_forwards,
            "tokens_per_forward": tokens_per_forward,
            # FLOP accounting (hardware-agnostic): token-layer-calls incl. loop passes,
            # coda branches, and window recomputation; + wall times for tokens/s.
            "prefill_time_s": total_prefill_time,
            "decode_time_s": total_decode_time,
            **effstats.snapshot(),
        }

        # Build final results
        final_results = []
        sorted_seq_ids = sorted(outputs.keys())
        for seq_id in sorted_seq_ids:
            token_ids = outputs[seq_id]
            final_results.append(
                {
                    "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                    "token_ids": token_ids,
                    "stats": global_stats.copy(),
                }
            )

        if use_tqdm:
            pbar.close()

        return final_results

    def generate_stream(
        self,
        prompts: Union[List[str], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate completions with streaming output.
        
        This method yields token updates as they are generated, allowing
        for real-time display of generated text. Only confirmed prefix
        tokens are streamed (tokens that have been decoded and will not change).
        
        Args:
            prompts: List of prompts (strings or token ID lists)
            sampling_params: Sampling parameters (single or per-prompt)
            
        Yields:
            Dictionary containing:
                - seq_id: Sequence identifier
                - new_token_ids: List of newly generated token IDs
                - new_text: Decoded text for new tokens
                - is_finished: Whether this sequence has finished
                - stats: Statistics (only included when is_finished=True)
        """
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        seq_map = {}
        yielded_tokens = {}  # seq_id -> number of tokens already yielded

        # Add all requests to the scheduler
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                prompt_ids = self.tokenizer.encode(prompt)
            else:
                prompt_ids = prompt
            seq = self._create_sequence(prompt_ids, sp)
            self.scheduler.add(seq)
            seq_map[seq.seq_id] = seq
            yielded_tokens[seq.seq_id] = 0

        finished_seq_ids = set()

        # Statistics tracking
        total_prefill_tokens = 0
        total_prefill_time = 0.0
        from wedlm.utils import effstats as _effstats
        _effstats.reset()   # per-generate-call deltas for worker aggregation
        total_decode_tokens = 0
        total_decode_time = 0.0
        total_decode_forwards = 0

        # Main generation loop
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, num_forwards = self.step()
            dt = perf_counter() - t

            if num_tokens > 0:
                # Prefill phase
                total_prefill_tokens += num_tokens
                total_prefill_time += dt
            else:
                # Decode phase
                decoded = -num_tokens
                total_decode_tokens += decoded
                total_decode_time += dt
                total_decode_forwards += num_forwards

            # Check all sequences for new tokens and yield updates
            for seq_id, seq in seq_map.items():
                if seq_id in finished_seq_ids:
                    continue

                current_tokens = seq.num_completion_tokens
                prev_tokens = yielded_tokens[seq_id]

                if current_tokens > prev_tokens:
                    # New tokens have been generated
                    new_token_ids = seq.completion_token_ids[prev_tokens:current_tokens]
                    new_text = self.tokenizer.decode(
                        new_token_ids, skip_special_tokens=True
                    )
                    yielded_tokens[seq_id] = current_tokens

                    yield {
                        "seq_id": seq_id,
                        "new_token_ids": list(new_token_ids),
                        "new_text": new_text,
                        "is_finished": False,
                    }

            # Handle finished sequences
            for seq_id, token_ids in output:
                if seq_id not in finished_seq_ids:
                    finished_seq_ids.add(seq_id)

                    # Yield any remaining tokens that haven't been yielded yet
                    seq = seq_map[seq_id]
                    current_tokens = seq.num_completion_tokens
                    prev_tokens = yielded_tokens[seq_id]

                    # Calculate final statistics
                    avg_prefill_speed = (
                        total_prefill_tokens / total_prefill_time
                        if total_prefill_time > 0
                        else 0.0
                    )
                    avg_decode_speed = (
                        total_decode_tokens / total_decode_time
                        if total_decode_time > 0
                        else 0.0
                    )
                    tokens_per_forward = (
                        total_decode_tokens / total_decode_forwards
                        if total_decode_forwards > 0
                        else 0.0
                    )

                    stats = {
                        "prefill_speed": avg_prefill_speed,
                        "decode_speed": avg_decode_speed,
                        "prefill_tokens": total_prefill_tokens,
                        "decode_tokens": total_decode_tokens,
                        "decode_forwards": total_decode_forwards,
                        "tokens_per_forward": tokens_per_forward,
                    }

                    if current_tokens > prev_tokens:
                        # There are remaining tokens to yield
                        new_token_ids = seq.completion_token_ids[prev_tokens:current_tokens]
                        new_text = self.tokenizer.decode(
                            new_token_ids, skip_special_tokens=True
                        )
                        yielded_tokens[seq_id] = current_tokens

                        yield {
                            "seq_id": seq_id,
                            "new_token_ids": list(new_token_ids),
                            "new_text": new_text,
                            "is_finished": True,
                            "stats": stats,
                        }
                    else:
                        # No new tokens, just mark as finished
                        yield {
                            "seq_id": seq_id,
                            "new_token_ids": [],
                            "new_text": "",
                            "is_finished": True,
                            "stats": stats,
                        }
