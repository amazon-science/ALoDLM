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
from transformers import AutoConfig


@dataclass
class Config:
    """Engine configuration for WeDLMLLM.
    
    This class contains model and engine-level configurations.
    Sampling-related parameters should be set in SamplingParams.
    """
    model: str

    # Batching configuration
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096

    # Memory configuration
    gpu_memory_utilization: float = 0.9

    # Parallelism configuration
    tensor_parallel_size: int = 1

    # Execution configuration
    enforce_eager: bool = False

    # KV cache configuration
    kvcache_block_size: int = 4096
    num_kvcache_blocks: int = -1

    # WeDLM decoding window size
    wedlm_window_size: int = 16

    # Model-specific configuration
    mask_token_id: int | None = None
    hf_config: AutoConfig | None = None

    # Communication configuration
    nccl_port: int = 2333

    def __post_init__(self):
        # HuggingFace model id support:
        # - If `model` is a local directory, it will be used directly.
        # - If `model` is a HF repo id (e.g. "org/name" or "gpt2"), it will be
        #   downloaded to local cache and then loaded from the downloaded directory.
        # - Cache dir, revision, and token can be configured via environment variables
        #   (HF_HOME, HF_HUB_CACHE, HF_TOKEN, etc.)
        if not os.path.isdir(self.model):
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise ImportError(
                    "huggingface_hub is required to use remote HuggingFace model ids. "
                    "Install it with: pip install huggingface_hub"
                ) from e

            # Download snapshot to local cache and replace `model` with the local directory.
            self.model = snapshot_download(repo_id=self.model)

        assert self.kvcache_block_size % 256 == 0, (
            "kvcache_block_size must be divisible by 256."
        )
        assert 1 <= self.tensor_parallel_size <= 8, (
            "tensor_parallel_size must be between 1 and 8."
        )

        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )

        assert self.max_num_batched_tokens >= self.max_model_len, (
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= max_model_len ({self.max_model_len})"
        )
