"""Sequence credit primitives. Run: python -m unittest discover -s tests.

Modified from the training implementation; see optimized/licenses/WeDLM.txt.
"""
import torch

def masked_sequence_ids(masked_indices, cum_seqlens):
    if masked_indices.ndim != 1 or masked_indices.dtype != torch.bool:
        raise ValueError('masked_indices must be a one-dimensional boolean tensor')
    if cum_seqlens is None or cum_seqlens.ndim != 1 or cum_seqlens.numel() < 2:
        raise ValueError('Outcome credit requires packed sequence boundaries')
    if cum_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError('Sequence boundaries must use an integer dtype')
    if cum_seqlens.device != masked_indices.device:
        raise ValueError('Sequence boundaries and mask must share a device')
    if int(cum_seqlens[0]) != 0 or int(cum_seqlens[-1]) != masked_indices.numel():
        raise ValueError('Sequence boundaries must cover the actual packed stream')
    if bool((cum_seqlens[1:] < cum_seqlens[:-1]).any()):
        raise ValueError('Sequence boundaries must be nondecreasing')
    positions = torch.arange(masked_indices.numel(), device=masked_indices.device)
    return torch.bucketize(positions, cum_seqlens[1:], right=True)[masked_indices]

def sequence_score_function(credit, log_prob, sequence_ids, sequence_count):
    if credit.ndim != 1 or log_prob.shape != credit.shape or sequence_ids.shape != credit.shape:
        raise ValueError('Credit, log probabilities, and sequence IDs must be equal-length vectors')
    if sequence_ids.dtype != torch.int64 or sequence_count < 1:
        raise ValueError('Sequence IDs must be int64 and sequence_count must be positive')
    if not credit.device == log_prob.device == sequence_ids.device:
        raise ValueError('Credit, log probabilities, and IDs must share a device')
    if credit.dtype != log_prob.dtype:
        raise ValueError('Credit and log probabilities must share a dtype')
    if credit.numel() == 0:
        return log_prob.sum() * 0
    if bool(((sequence_ids < 0) | (sequence_ids >= sequence_count)).any()):
        raise ValueError('Sequence IDs must be within the declared sequence count')
    order = torch.argsort(sequence_ids, stable=True)
    lengths = torch.bincount(sequence_ids, minlength=sequence_count)
    returns = torch.segment_reduce(credit.detach()[order], 'sum', lengths=lengths)
    scores = torch.segment_reduce(log_prob[order], 'sum', lengths=lengths)
    return torch.dot(returns, scores) / credit.numel()
