"""Sparse training attention. Run: python -m unittest discover -s tests.

Modified from WeDLM attention code; see optimized/licenses/WeDLM.txt.
"""

from functools import lru_cache
import inspect

import torch


def sparse_plan(boundaries, block_size, backend):
    qr, kr, types = [], [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        length = (end - start) // 2
        blocks = [(start + length + b, start + length + min(b + block_size, length))
                  for b in range(0, length, block_size)]
        for a, z in blocks:
            qr.append([a, z])
            kr.append([a, z])
            types.append(1)
        for b, (a, z) in enumerate(blocks[1:], 1):
            qr.append([a, z])
            kr.append([start, start + b * block_size])
            types.append(0)
        if length:
            qr.append([start, start + length])
            kr.append([start, start + length])
            types.append(1)
    return {"q_ranges": qr, "k_ranges": kr, "attn_type_map": types, "backend": backend}


def plan_to_device(plan, device):
    result = dict(plan)
    result["max_seqlen_q"] = max((b - a for a, b in plan["q_ranges"]), default=0)
    result["max_seqlen_k"] = max((b - a for a, b in plan["k_ranges"]), default=0)
    for name in ("q_ranges", "k_ranges", "attn_type_map"):
        result[name] = torch.tensor(result[name], dtype=torch.int32, device=device)
        if name.endswith("_ranges"):
            result[name] = result[name].reshape(-1, 2)
    return result


@lru_cache(maxsize=2)
def _kernel(backend):
    if backend == "magi-fa4":
        from magi_attention.functional.fa4 import ffa_fa4_func
        return ffa_fa4_func
    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
    return flex_flash_attn_func


@torch.compiler.disable
def sparse_attention(q, k, v, plan):
    if q.device.type != "cuda":
        raise ValueError("Sparse attention requires CUDA; use sdpa for CPU")
    kernel = _kernel(plan["backend"])
    kwargs = {key: plan[key] for key in ("q_ranges", "k_ranges", "attn_type_map")}
    kwargs.update(softmax_scale=q.shape[-1] ** -0.5, softcap=0.0)
    if plan["backend"] == "magi":
        kwargs["max_seqlen_q"] = plan["max_seqlen_q"]
        if "max_seqlen_k" in inspect.signature(kernel).parameters:
            kwargs["max_seqlen_k"] = plan["max_seqlen_k"]
        kwargs["deterministic"] = False
    result = kernel(q.contiguous(), k.contiguous(), v.contiguous(), **kwargs)
    return result[0] if isinstance(result, tuple) else result
