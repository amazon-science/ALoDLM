"""Dual-stream masking. Run: python -m unittest discover -s tests.

Modified from WeDLM training code; see optimized/licenses/WeDLM.txt.
"""

from dataclasses import dataclass

import torch

from .attention import plan_to_device, sparse_plan

@dataclass
class Batch:
    ids: torch.Tensor
    original: torch.Tensor
    positions: torch.Tensor
    masked: torch.Tensor
    mask_probability: torch.Tensor
    boundaries: torch.Tensor
    attention_mask: torch.Tensor | dict


def attention_mask(length, block_size, device):
    q = torch.arange(2 * length, device=device)[:, None]
    k = torch.arange(2 * length, device=device)[None, :]
    qt, kt = q >= length, k >= length
    qb = torch.where(qt, q - length, q) // block_size
    kb = torch.where(kt, k - length, k) // block_size
    within = (qb == kb) & (qt == kt) & (k <= q)
    preceding_clean = (kb < qb) & ~kt
    return within | preceding_clean


@torch.no_grad()
def build_batch(ids, labels, boundaries, block_size, mask_token_id, backend="sdpa"):
    if ids.ndim != 1 or ids.shape != labels.shape or ids.numel() == 0:
        raise ValueError("Expected nonempty flat input and label tensors")
    offsets = boundaries.tolist()
    if offsets[0] != 0 or offsets[-1] != len(ids):
        raise ValueError("Boundaries must cover the packed tokens")
    if any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Packed segments must be nonempty")
    device = ids.device
    parts = [[] for _ in range(5)]
    masks, new_offsets = [], [0]
    for start, end in zip(offsets, offsets[1:]):
        seq, lab = ids[start:end], labels[start:end]
        length = len(seq)
        positions = torch.arange(length, device=device)
        probabilities = torch.empty((length + block_size - 1) // block_size,
                                    device=device).uniform_(0, 1).clamp_min(1e-8)
        noisy, gold, pos, selected, prob = [], [], [], [], []
        for block, left in enumerate(range(0, length, block_size)):
            right = min(left + block_size, length)
            candidate = (lab[left:right] != -100).nonzero().flatten()
            p = float(probabilities[block])
            count = round(len(candidate) * p)
            chosen = torch.zeros(right - left, dtype=torch.bool, device=device)
            if count:
                chosen[candidate[torch.randperm(len(candidate), device=device)[:count]]] = True
            order = torch.cat(((~chosen).nonzero().flatten(), chosen.nonzero().flatten()))
            tok = seq[left:right]
            noisy.append(torch.where(chosen, mask_token_id, tok)[order])
            gold.append(tok[order])
            pos.append(positions[left:right][order])
            selected.append(chosen[order])
            prob.append(torch.where(chosen[order], p, 0.0))
        values = (
            torch.cat((seq, torch.cat(noisy))),
            torch.cat((seq, torch.cat(gold))),
            torch.cat((positions, torch.cat(pos))),
            torch.cat((torch.zeros(length, dtype=torch.bool, device=device), torch.cat(selected))),
            torch.cat((torch.zeros(length, device=device), torch.cat(prob))),
        )
        for dst, value in zip(parts, values):
            dst.append(value)
        if backend == "sdpa":
            masks.append(attention_mask(length, block_size, device))
        new_offsets.append(new_offsets[-1] + 2 * length)
    return Batch(
        *(torch.cat(p) for p in parts[:5]),
        torch.tensor(new_offsets, dtype=torch.long, device=device),
        (torch.block_diag(*masks).contiguous() if backend == "sdpa"
         else plan_to_device(sparse_plan(new_offsets, block_size, backend), device)),
    )
