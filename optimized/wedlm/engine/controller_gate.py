# Modified for ALoDLM: adaptive recurrence, depth-aware caching, and release packaging.
# Original notices are retained; see NOTICE and licenses/WeDLM.txt.

# coding=utf-8
"""Lambda-conditioned DEPTH optimal-stopping controller — engine inference gate.

Drops into process_mask_positions_perdepth: instead of the default per-token exit depth
d*_i = argmin_d entropy, the controller picks ONE stop depth s* for the whole outer step
(commit the entropy set B_{s*} at depth s*). Enabled by env WEDLM_CONTROLLER=<path to the
controller weight file>; loop cost via WEDLM_LAMBDA.
Feature r_s and the B_s (adjusted-entropy) rule are byte-identical to training (variant A, s-only).
When WEDLM_CONTROLLER is unset the engine is untouched (default per-token-depth path runs).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthController(nn.Module):
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, 1))
        self.log_alpha = nn.Parameter(torch.zeros(1))
    def t_hat(self, r):
        return F.softplus(self.mlp(r).squeeze(-1))
    def stop_logit(self, r, lam):
        return F.softplus(self.log_alpha) * (lam - self.t_hat(r))


_CTRL = None
_LAM = 0.5
_LOADED = False

# Compute-axis accounting: mean effective loops/token = mean(s*+1) over committed tokens.
# fixed-K commits every token at depth K-1 (=> mean_K == K); the controller adapts s* per step.
_LOOP_SUM = 0.0
_LOOP_CNT = 0


def _acct(s_star, n):
    global _LOOP_SUM, _LOOP_CNT
    prev = _LOOP_CNT
    _LOOP_SUM += (s_star + 1) * n
    _LOOP_CNT += n
    # throttled running print (robust to ray-worker teardown not firing atexit); last line == aggregate
    if _LOOP_CNT // 5000 != prev // 5000:
        print(f"[controller_gate] MEAN_K={_LOOP_SUM / _LOOP_CNT:.4f} over N={_LOOP_CNT} committed tokens (lambda={_LAM})", flush=True)


def controller_loop_stats():
    return (_LOOP_SUM / _LOOP_CNT if _LOOP_CNT else 0.0), _LOOP_CNT


def _ensure_loaded():
    global _CTRL, _LAM, _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = os.environ.get("WEDLM_CONTROLLER")
    if not path or not os.path.exists(path):
        return
    _LAM = float(os.environ.get("WEDLM_LAMBDA", "0.5"))
    variant = os.environ.get("WEDLM_CONTROLLER_VARIANT", "A")  # A=s-only (chosen by ablation), B=s+dyn
    blob = torch.load(path, map_location="cuda")
    sub = blob[variant]
    c = DepthController(sub["d_in"]).cuda().eval()
    c.load_state_dict(sub["state_dict"])
    global _CTRL
    _CTRL = c
    print(f"[controller_gate] loaded variant={variant} d_in={sub['d_in']} lambda={_LAM}", flush=True)


def controller_enabled():
    _ensure_loaded()
    return _CTRL is not None


@torch.no_grad()
def controller_stop_depth(mask_hidden_by_depth, entropy, remaining_mask_indices, threshold, pos_penalty, K):
    """mask_hidden_by_depth [M,K,d], entropy [M,K]. Runs s=1..K: build B_s (adjusted-entropy rule,
    exactly as training/select_positions_to_fill), feature r_s (variant A), stop at first s* where
    z_s(lambda) >= 0 (or s=K). Returns (B_{s*} indices into the M masks, s_star 0-indexed)."""
    M = entropy.shape[0]
    mi = None
    if pos_penalty and remaining_mask_indices:
        mi = torch.tensor(remaining_mask_indices, device=entropy.device, dtype=torch.float)
    Bs = None
    for s in range(K):
        adj = entropy[:, s] + ((mi - mi[0]) * pos_penalty if mi is not None else 0.0)
        Bs = (adj < threshold).nonzero(as_tuple=True)[0]
        if Bs.numel() == 0:
            Bs = adj.argmin().reshape(1)
        h_ln = F.layer_norm(mask_hidden_by_depth[:, s, :].float(), (mask_hidden_by_depth.shape[-1],))
        pool_U = h_ln.mean(0)                                   # Pool_{U_k} (all M = window masks)
        pool_B = h_ln[Bs].mean(0)                               # Pool_{B_s}
        extra = torch.tensor([entropy[:, s].mean().item(), entropy[Bs, s].mean().item(),
                              Bs.numel() / max(M, 1), (s + 1) / K], device=entropy.device)
        r_s = torch.cat([pool_U, pool_B, extra]).unsqueeze(0)   # variant A: [pool_U, pool_B, Hbar_U, Hbar_B, |B|/|U|, s/K]
        if s == K - 1 or _CTRL.stop_logit(r_s, _LAM).item() >= 0:
            _acct(s, len(Bs))
            return Bs.tolist(), s
    _acct(K - 1, len(Bs))
    return Bs.tolist(), K - 1


import atexit


@atexit.register
def _report_loop_stats():
    if _LOOP_CNT:
        mk, n = controller_loop_stats()
        print(f"[controller_gate] MEAN_K={mk:.4f} over N={n} committed tokens (lambda={_LAM})", flush=True)
