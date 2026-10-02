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

from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.
    
    Attributes:
        temperature: Sampling temperature. 0 means greedy decoding.
        top_p: Nucleus sampling parameter. Only tokens with cumulative
            probability <= top_p are considered. 1.0 means no filtering.
        top_k: Top-k sampling parameter. Only the top k tokens are considered.
            0 means no filtering.
        max_tokens: Maximum number of tokens to generate.
        stop_token_ids: List of token IDs that trigger generation stop.
        stop: String or list of strings that trigger generation stop.
        wedlm_entropy_threshold: Entropy threshold for WeDLM parallel decoding.
            If None, selects the position with minimum adjusted entropy.
            If set, all positions with adjusted entropy below this threshold
            will be decoded in parallel.
        wedlm_pos_penalty_factor: Position penalty factor for WeDLM decoding.
            Higher values penalize positions further from the current position,
            making the model prefer to decode earlier positions first.
    """
    temperature: float = 0.2
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 64

    stop_token_ids: Optional[List[int]] = None
    stop: Optional[Union[str, List[str]]] = None

    # WeDLM decoding parameters with defaults
    wedlm_entropy_threshold: Optional[float] = 0.4
    wedlm_pos_penalty_factor: float = 0.02

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")

        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

        if self.stop_token_ids is None:
            self.stop_token_ids = []

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.wedlm_entropy_threshold is not None:
            if self.wedlm_entropy_threshold < 0.0:
                raise ValueError("wedlm_entropy_threshold must be non-negative")

        if self.wedlm_pos_penalty_factor < 0.0:
            raise ValueError("wedlm_pos_penalty_factor must be non-negative")
