"""Outcome-level training loss. Run: python -m unittest discover -s tests.

Modified from the recurrent training objective; see optimized/licenses/WeDLM.txt.
"""

import torch
import torch.nn.functional as F

from .credit import masked_sequence_ids, sequence_score_function


def autoregressive_loss(logits, labels, packed_boundaries):
    """Average next-token losses without crossing independently attended segments."""
    total = torch.tensor(0.0, device=logits.device)
    count = torch.tensor(0.0, device=logits.device)
    for start, end in zip(packed_boundaries[:-1].tolist(), packed_boundaries[1:].tolist()):
        length = (end - start) // 2
        if length < 2:
            continue
        targets = labels[start // 2 + 1:start // 2 + length]
        n = (targets != -100).sum()
        if n == 0:
            continue
        token_loss = F.cross_entropy(
            logits[start:start + length - 1], targets, reduction="none", ignore_index=-100)
        segment_mean = token_loss.sum() / n
        n_float = n.float()
        total = total + segment_mean * n_float
        count = count + n_float
    if bool(count > 0):
        return total / count
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def outcome_loss(ce, gate_logits, depths, batch, config):
    """Return intermediate denoising loss and sequence score-function loss.

ce contains masked-token cross-entropies with shape [depth, masked_tokens].
The gate observes detached features. Sampled costs have no gradient path.
"""
    selected = batch.masked
    count = int(selected.sum())
    if count == 0:
        zero = ce.sum() + sum(g.sum() * 0 for g in gate_logits)
        return zero, {key: zero.detach() for key in (
            "denoising", "actor", "sampled_mean_depth", "prediction_cost",
            "regularization_cost", "profile_mean_depth", "first_pass_halt_probability",
        )}
    steps = config.max_depth
    z = depths[selected].view(1, -1)
    rows = torch.arange(steps, device=ce.device).view(-1, 1)
    raw = torch.stack([g[selected] for g in gate_logits])

    # The denoiser uses a detached stick closed at each sampled exit.
    survival = torch.cat((torch.zeros_like(raw[:1]),
                          torch.cumsum(-F.softplus(raw), 0)[:-1]), 0)
    truncated = torch.where(rows == z - 1, survival, -F.softplus(-raw) + survival)
    truncated = truncated.masked_fill(rows >= z, float("-inf")).exp().detach()
    weights = 1 / (batch.mask_probability[selected] + 1e-8)
    weights = weights / weights.sum()
    denoising = (weights * (truncated * (ce * (rows < z))).sum(0)).sum()

    # Score probabilities use the actual rollout hazards through the sampled exit.
    g = raw.float()
    before = torch.cat((torch.zeros_like(g[:1]),
                        torch.cumsum(-F.softplus(g), 0)[:-1]), 0)
    full_log_pi = torch.where(rows == steps - 1, before, -F.softplus(-g) + before)
    logq = full_log_pi.gather(0, z - 1).squeeze(0)
    with torch.no_grad():
        log_prior = torch.log_softmax(
            -config.depth_prior_c * torch.arange(1, steps + 1, device=g.device).float(), 0)
        qbar = full_log_pi.detach().exp().mean(1).clamp_min(1e-12)
        marginal = qbar.log().gather(0, z.flatten() - 1)
        prior = log_prior.gather(0, z.flatten() - 1)
        prediction = ce.detach().float().gather(0, z - 1).squeeze(0) - ce[0].detach().float()
        mi_cost = config.kl_beta_mi * (logq - marginal)
        budget_cost = config.kl_beta_marg * (marginal - prior)
        credit = (prediction + mi_cost) + budget_cost
    ids = masked_sequence_ids(selected, batch.boundaries)
    actor = sequence_score_function(credit, logq, ids, batch.boundaries.numel() - 1)
    metrics = {
        "denoising": denoising.detach(),
        "actor": actor.detach(),
        "sampled_mean_depth": z.float().mean(),
        "prediction_cost": prediction.mean(),
        "regularization_cost": (mi_cost + budget_cost).mean(),
        "profile_mean_depth": (qbar * torch.arange(1, steps + 1, device=g.device)).sum(),
        "first_pass_halt_probability": g[0].sigmoid().mean().detach(),
    }
    return denoising + actor, metrics
