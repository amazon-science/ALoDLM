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
Sampler module for WeDLM decoding.

This module handles all token sampling and position selection logic,
including:
- Temperature-based token sampling
- Top-p (nucleus) sampling
- Top-k sampling
- Entropy computation for position selection
- Position selection using entropy-based parallel decoding
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


# STRICT LEFT-TO-RIGHT commit mode, env-gated (read ONCE at module load, mirroring
# how WEDLM_LOOP_START/END/K are read in model_runner.py). When WEDLM_L2R is set to
# a nonzero int, WeDLM decode commits EXACTLY ONE token per forward -- the leftmost
# still-masked window slot -- instead of the entropy-gated parallel multi-commit.
# Unset / "0" => byte-identical to the default confidence-order decode.
try:
    _WEDLM_L2R = int(os.environ.get("WEDLM_L2R", "0")) != 0
except ValueError:
    _WEDLM_L2R = False

# STRESS TEST for the granular per-token cache: assign each committed token a RANDOM exit
# depth d* (instead of the argmin-entropy depth), reading its value + caching its KV at that
# depth. Selection stays the DEFAULT final-depth entropy decode. If the per-token cache is
# faithful, accuracy stays ~normal (depth is ~flat + capping is near-lossless); a bug in the
# depth-bucketing / slot order / cap direction would collapse it. Read once at module load.
_WEDLM_RANDOM_DEPTH = os.environ.get("WEDLM_RANDOM_DEPTH", "0") != "0"


class Sampler:
    """Handles all token sampling and position selection logic for WeDLM decoding.
    
    This class centralizes all sampling-related operations that were previously
    scattered in model_runner.py, providing a cleaner separation of concerns.
    
    Responsibilities:
    - Sample tokens from logits with temperature scaling
    - Apply top-p (nucleus) and top-k filtering
    - Compute entropy for position selection decisions
    - Select which mask positions to fill based on entropy threshold
    """

    def __init__(self):
        """Initialize the Sampler."""
        pass

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy for each position's probability distribution.
        
        Entropy is used to measure the model's uncertainty at each position.
        Lower entropy indicates higher confidence in the prediction.
        
        Args:
            logits: Raw logits from the model, shape [num_positions, vocab_size]
            
        Returns:
            Entropy values for each position, shape [num_positions]
        """
        return torch.distributions.Categorical(logits=logits).entropy()

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits.
        
        Sets logits of tokens outside top-k to -inf.
        
        Args:
            logits: Raw logits, shape [num_positions, vocab_size]
            top_k: Number of top tokens to keep. 0 means no filtering.
            
        Returns:
            Filtered logits with same shape as input.
        """
        if top_k <= 0:
            return logits
        
        top_k = min(top_k, logits.size(-1))
        # Get the k-th largest value for each position
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits.
        
        Keeps the smallest set of tokens whose cumulative probability exceeds top_p.
        
        Args:
            logits: Raw logits, shape [num_positions, vocab_size]
            top_p: Cumulative probability threshold. 1.0 means no filtering.
            
        Returns:
            Filtered logits with same shape as input.
        """
        if top_p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        # Shift right to keep the first token above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits with temperature, top-p, and top-k.
        
        Supports greedy decoding (temperature=0) and sampling with various
        filtering strategies.
        
        The order of operations is: top-k -> top-p -> temperature scaling -> sample
        
        Args:
            logits: Raw logits from the model, shape [num_positions, vocab_size]
            temperature: Sampling temperature. 0 means greedy decoding.
            top_p: Nucleus sampling threshold. 1.0 means no filtering.
            top_k: Top-k sampling parameter. 0 means no filtering.
            
        Returns:
            Tuple of (sampled_ids, greedy_ids):
            - sampled_ids: Token IDs sampled with the specified strategy
            - greedy_ids: Greedy (argmax) token IDs for reference
        """
        # Compute greedy predictions (always useful for comparison)
        probs = F.softmax(logits, dim=-1)
        greedy_ids = probs.argmax(dim=-1)
        
        if temperature == 0:
            # Greedy decoding - ignore top_p and top_k
            return greedy_ids, greedy_ids
        
        # Apply top-k filtering first
        filtered_logits = self._apply_top_k(logits, top_k)
        
        # Apply top-p filtering
        filtered_logits = self._apply_top_p(filtered_logits, top_p)
        
        # Temperature-scaled sampling
        scaled_logits = filtered_logits / temperature
        sampling_probs = F.softmax(scaled_logits, dim=-1)
        sampled_ids = torch.multinomial(sampling_probs, num_samples=1).squeeze(-1)
        
        return sampled_ids, greedy_ids

    def select_positions_to_fill(
        self,
        entropy: torch.Tensor,
        remaining_mask_indices: List[int],
        entropy_threshold: Optional[float],
        pos_penalty_factor: float
    ) -> List[int]:
        """Select mask positions to fill using entropy-based parallel decoding.
        
        This method implements the core WeDLM position selection algorithm:
        1. Compute position-adjusted entropy by adding a distance penalty
        2. If entropy_threshold is set, select all positions below threshold
        3. Otherwise, select only the position with minimum adjusted entropy
        
        The position penalty encourages the model to decode earlier positions first,
        which helps maintain left-to-right coherence in generation.
        
        Args:
            entropy: Entropy values for each mask position, shape [num_mask_positions]
            remaining_mask_indices: Indices of remaining mask positions in the window
            entropy_threshold: Threshold for parallel decoding. If None, uses greedy
                position selection (single position with min entropy).
            pos_penalty_factor: Factor for position-based penalty. Higher values
                penalize positions further from the start more heavily.
            
        Returns:
            List of indices (into remaining_mask_indices) indicating which
            positions should be filled in this step.
        """
        # STRICT LEFT-TO-RIGHT (env-gated by WEDLM_L2R): commit EXACTLY the
        # leftmost still-masked slot, one token per forward, regardless of
        # entropy/confidence. remaining_mask_indices is in ascending window-slot
        # order, so index 0 is the leftmost mask; returning [0] makes
        # process_mask_positions fill remaining_mask_indices[0] with that row's
        # sampled token (temperature sampling / argmax at temp 0 unchanged).
        # This is a strict special case of the default's contiguous leading-
        # prefix commit, so the granular per-depth KV cache stays exact: a
        # committed token is never rewritten, so its per-depth keys are final
        # when committed. When WEDLM_L2R is unset/0 this branch is skipped and
        # the entropy-gated default below runs byte-identically.
        if _WEDLM_L2R:
            return [0]

        device = entropy.device
        
        # Convert mask indices to tensor for vectorized computation
        mask_indices_tensor = torch.tensor(
            remaining_mask_indices, device=device, dtype=torch.float
        )
        
        # Compute position-based penalty
        # Positions further from the first mask position get higher penalty
        base_pos = mask_indices_tensor[0]
        distances = mask_indices_tensor - base_pos
        position_penalty = distances * pos_penalty_factor
        
        # Add penalty to entropy
        adjusted_entropy = entropy + position_penalty
        
        # Select positions based on threshold
        if entropy_threshold is not None:
            # Parallel decoding: select all positions with low enough entropy
            candidates = (adjusted_entropy < entropy_threshold).nonzero(as_tuple=True)[0]
            if candidates.numel() > 0:
                return candidates.tolist()
        
        # Fallback: greedy position selection (minimum adjusted entropy)
        return [int(adjusted_entropy.argmin().item())]

    def process_mask_positions(
        self,
        mask_logits: torch.Tensor,
        remaining_mask_indices: List[int],
        temperature: float,
        entropy_threshold: Optional[float],
        pos_penalty_factor: float,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[List[int], List[int]]:
        """Process mask positions and return positions to fill with their token IDs.
        
        This is the main entry point for sampling during WeDLM decoding. It combines
        all sampling operations: entropy computation, token sampling, and position
        selection.
        
        Args:
            mask_logits: Logits for mask positions only, shape [num_masks, vocab_size]
            remaining_mask_indices: Window indices of remaining mask positions
            temperature: Sampling temperature
            entropy_threshold: Threshold for parallel position selection
            pos_penalty_factor: Position penalty factor for entropy adjustment
            top_p: Nucleus sampling threshold. 1.0 means no filtering.
            top_k: Top-k sampling parameter. 0 means no filtering.
            
        Returns:
            Tuple of (fill_indices, token_ids):
            - fill_indices: Indices into remaining_mask_indices for positions to fill
            - token_ids: Corresponding token IDs to fill at those positions
        """
        if mask_logits.size(0) == 0:
            return [], []
        
        # Step 1: Compute entropy for position selection
        entropy = self.compute_entropy(mask_logits)
        
        # Step 2: Sample tokens with top_p and top_k filtering
        sampled_ids, _ = self.sample_tokens(mask_logits, temperature, top_p, top_k)
        
        # Step 3: Select positions to fill
        fill_indices = self.select_positions_to_fill(
            entropy,
            remaining_mask_indices,
            entropy_threshold,
            pos_penalty_factor
        )
        
        # Step 4: Get token IDs for selected positions
        token_ids = [int(sampled_ids[k].item()) for k in fill_indices]

        return fill_indices, token_ids

    def process_mask_positions_perdepth(
        self,
        mask_logits_by_depth: torch.Tensor,
        remaining_mask_indices: List[int],
        temperature: float,
        entropy_threshold: Optional[float] = None,
        pos_penalty_factor: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        mask_hidden_by_depth: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Adaptive-DEPTH commit (the general per-token-depth decode; supports MULTI-token).

        Given per-depth logits for every mask position [num_masks, K, vocab], the exit depth
        of position i is the most-confident loop depth  d*_i = argmin_d entropy(logits[i, d]),
        and its confidence is that minimum entropy e_i. (This argmin-over-depth rule is the one
        swappable "signal" -- replace it to try loss-difference / a learned gate.)

        WHICH positions commit this step mirrors select_positions_to_fill:
          - entropy_threshold is None  -> commit-1: the single global lowest-e_i position.
          - else                        -> multi:   every position with e_i (+ pos penalty)
            below the threshold (>=1, greedy fallback so decoding never stalls).
        Each committed position emits the token sampled from its OWN depth-d*_i logits and
        reports d*_i, which the granular cache caps its KV at (WEDLM_ADAPTIVE_DEPTH_CAP).

        Returns:
            (fill_indices, token_ids, exit_depths): parallel lists over the committed
            positions (indices into remaining_mask_indices; 0-indexed loop depths). ([], [], [])
            if there are no mask positions.
        """
        M, K, V = mask_logits_by_depth.shape
        if M == 0:
            return [], [], []
        flat = mask_logits_by_depth.reshape(M * K, V)
        entropy = self.compute_entropy(flat).reshape(M, K)  # [num_masks, K]
        # NOTE: the Stage-I exit-gate decode (WEDLM_GATE) does NOT pass through here — its
        # decisions are made LIVE inside the looped forward (gate_decode.loop_pass_check,
        # compute-then-stop) and consumed by the decoder directly.
        # Learned DEPTH gate: if a controller is loaded (WEDLM_CONTROLLER), it replaces the
        # argmin-entropy exit rule -- picks ONE stop depth s* for the outer step and commits the
        # entropy set B_{s*} at s* (spec: lambda-conditioned optimal stopping). WEDLM_LAMBDA = cost.
        if entropy_threshold is not None and mask_hidden_by_depth is not None:
            from wedlm.engine.controller_gate import controller_enabled, controller_stop_depth
            if controller_enabled():
                bs_idx, s_star = controller_stop_depth(
                    mask_hidden_by_depth, entropy, remaining_mask_indices,
                    entropy_threshold, pos_penalty_factor, K,
                )
                fill_indices, token_ids, exit_depths = [], [], []
                for i in bs_idx:
                    sampled_ids, _ = self.sample_tokens(
                        mask_logits_by_depth[i, s_star].unsqueeze(0), temperature, top_p, top_k
                    )
                    fill_indices.append(i)
                    token_ids.append(int(sampled_ids[0]))
                    exit_depths.append(s_star)
                return fill_indices, token_ids, exit_depths
        if _WEDLM_RANDOM_DEPTH:
            # STRESS TEST (per-STEP random depth): pick ONE loop depth d_t for this step, then
            # do the DEFAULT entropy decode AT d_t -- SELECT positions by entropy at d_t AND
            # commit their values at d_t (consistent, no select/value decoupling). Committed
            # tokens across steps end up at mixed depths -> coherently stress-tests the cache.
            d_t = int(torch.randint(0, K, (1,), device=entropy.device).item())
            d_star = torch.full((M,), d_t, device=entropy.device, dtype=torch.long)
            e_min = entropy[:, d_t]
        else:
            # per-position best (lowest-entropy) depth and its confidence
            e_min, d_star = entropy.min(dim=1)              # [M], [M]
        # position penalty (mirror select_positions_to_fill: prefer earlier masks)
        if pos_penalty_factor and remaining_mask_indices:
            mi = torch.tensor(
                remaining_mask_indices, device=entropy.device, dtype=torch.float
            )
            adjusted = e_min + (mi - mi[0]) * pos_penalty_factor
        else:
            adjusted = e_min

        if entropy_threshold is not None:
            sel = (adjusted < entropy_threshold).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:  # greedy fallback: always commit >=1 to avoid a stall
                sel = adjusted.argmin().reshape(1)
        else:
            sel = adjusted.argmin().reshape(1)  # commit-1

        fill_indices: List[int] = []
        token_ids: List[int] = []
        exit_depths: List[int] = []
        for i in sel.tolist():
            d = int(d_star[i].item())
            sampled_ids, _ = self.sample_tokens(
                mask_logits_by_depth[i, d].unsqueeze(0), temperature, top_p, top_k
            )
            fill_indices.append(i)
            token_ids.append(int(sampled_ids[0].item()))
            exit_depths.append(d)
        return fill_indices, token_ids, exit_depths
