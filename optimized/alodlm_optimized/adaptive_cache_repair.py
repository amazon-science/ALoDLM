# Modified for ALoDLM: adaptive recurrence, depth-aware caching, and release packaging.
# Original notices are retained; see NOTICE and licenses/WeDLM.txt.

"""Repair uncomputed recurrent cache depths without replacing valid COA keys.

Install before constructing the inference engine:
    from alodlm_optimized.adaptive_cache_repair import install
    install()

Run the GPU reference comparison:
    python -m alodlm_optimized.adaptive_cache_repair

When the entire window stops at depth d, commit-on-agree has only written cache
depths 1..d. Copy depth d into d+1..K before a future window can read them.
This differs from capping at each token's commit depth: computed later keys,
including keys produced after embedding injection, remain untouched.
"""
import os

import torch
import triton
import triton.language as tl


VERSION = "adaptive-coa-completed-depth-cache-v1"


@triton.jit
def _fill_uncomputed_depths(
    cache, slots, layer_offsets,
    STRIDE_KV: tl.constexpr, STRIDE_LAYER: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    COMPLETED: tl.constexpr, K: tl.constexpr,
):
    row = tl.program_id(0)
    # Real B200 cache allocations exceed 2^31 elements. Promote before products,
    # including layer * STRIDE_LAYER, rather than after an int32 overflow.
    layer = tl.load(layer_offsets + tl.program_id(1)).to(tl.int64)
    kv = tl.program_id(2).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    active = (slot >= 0) & (lane < WIDTH)
    common = kv * STRIDE_KV + slot * WIDTH + lane
    source = common + (layer + COMPLETED - 1) * STRIDE_LAYER
    value = tl.load(cache + source, mask=active, other=0)
    for depth in tl.static_range(COMPLETED, K):
        target = common + (layer + depth) * STRIDE_LAYER
        tl.store(cache + target, value, mask=active)


def fill_uncomputed(cache, slots, offsets, completed, k):
    """One launch fills only unexecuted depths at valid physical token slots."""
    if completed >= k or slots.numel() == 0 or offsets.numel() == 0:
        return
    assert 1 <= completed < k
    assert cache.is_contiguous() and slots.is_contiguous()
    width = cache.shape[-2] * cache.shape[-1]
    _fill_uncomputed_depths[(slots.numel(), offsets.numel(), 2)](
        cache, slots, offsets,
        STRIDE_KV=cache.stride(0), STRIDE_LAYER=cache.stride(1),
        WIDTH=width, BLOCK=triton.next_power_of_2(width),
        COMPLETED=completed, K=k, num_warps=4,
    )


def _initialize(runner):
    offsets, cursor, depths = [], 0, set()
    for n in runner._depth_of:
        if n > 1:
            offsets.append(cursor)
            depths.add(n)
        cursor += n
    assert len(depths) == 1, "Expected one shared recurrent depth"
    k = depths.pop()
    runner._completed_cache_offsets = torch.tensor(
        offsets, device=runner.kv_cache.device, dtype=torch.int32)
    runner._completed_cache_k = k
    # Compile every stopping depth during initial warm-up. Invalid slots make
    # these launches no-ops, so model state is preserved.
    empty = torch.full((32,), -1, device=runner.kv_cache.device, dtype=torch.int32)
    for completed in range(1, k):
        fill_uncomputed(runner.kv_cache, empty, runner._completed_cache_offsets,
                        completed, k)
    torch.cuda.synchronize()


def install():
    """Apply the same completed-depth repair after eager or graphed forwards."""
    from wedlm.engine.model_runner import ModelRunner
    from wedlm.utils.context import get_context, get_perdepth_readouts

    if getattr(ModelRunner, "_completed_cache_repair_version", None) == VERSION:
        return
    original = ModelRunner.run_model

    def run_model(self, input_ids, positions, is_prefill):
        result = original(self, input_ids, positions, is_prefill)
        if (not is_prefill
                and os.environ.get("WEDLM_ADAPTIVE_DEPTH", "0") != "0"
                and os.environ.get("WEDLM_COMMIT_ON_AGREE", "0") != "0"):
            assert os.environ.get("WEDLM_TOKEN_FREEZE", "0") == "0", \
                "Token freezing requires a separate cache policy"
            if not hasattr(self, "_completed_cache_offsets"):
                _initialize(self)
            completed = len(get_perdepth_readouts())
            assert 1 <= completed <= self._completed_cache_k
            fill_uncomputed(self.kv_cache, get_context().slot_mapping,
                            self._completed_cache_offsets, completed,
                            self._completed_cache_k)
        return result

    ModelRunner.run_model = run_model
    ModelRunner._completed_cache_repair_version = VERSION


def selftest():
    """Compare every stopping depth and untouched region against exact copies."""
    torch.manual_seed(123)
    source = torch.randn((2, 14, 2, 16, 4, 8), device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([-1, 0, 5, 16, 31], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([2, 6], device="cuda", dtype=torch.int32)
    selected = slots[slots >= 0].long()
    for depth in range(1, 5):
        actual, expected = source.clone(), source.clone()
        for base in (2, 6):
            for target in range(depth, 4):
                expected[:, base + target].view(2, -1, 4, 8)[:, selected] = \
                    source[:, base + depth - 1].reshape(2, -1, 4, 8)[:, selected]
        fill_uncomputed(actual, slots, offsets, depth, 4)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected), f"Cache copy differs at depth {depth}"
    print("CACHE_REPAIR_KERNEL_REFERENCE_OK", VERSION, flush=True)


if __name__ == "__main__":
    selftest()
