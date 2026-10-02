# Modified for ALoDLM: adaptive recurrence, depth-aware caching, and release packaging.
# Original notices are retained; see NOTICE and licenses/WeDLM.txt.


import atexit
import json
import os

import torch
from torch import nn

_gate = None
_loaded = False
_logits_fn = None      # set once by ModelRunner: model.compute_logits (lm_head)
_embed_fn = None       # set once by ModelRunner: model.model.embed_tokens (COMMIT-ON-AGREE)
_recache_fn = None     # set once by ModelRunner: _recache_granular (token-freeze back-fill)
_plan = None           # active GatePlan for the in-flight decode forward
_frozen = None         # bool [N_stream]: rows KV-frozen for the current forward's later passes
_stats = {"exit_hist_committed": [0] * 8, "exit_hist_window": [0] * 8, "window_stop_hist": [0] * 8,
          "outer_steps": 0, "passes": 0, "committed": 0, "forced_fallback": 0,
          "batch_steps": 0, "batch_passes": 0, "tf_tok_passes": 0, "tf_frozen_tok_passes": 0,
          "tf_stream_tok_passes": 0,
          # entropy_perpass mode: tokens committed at each pass index, and how many windows
          # were still alive (executed) at each pass index
          "perpass_commits": [0] * 8, "perpass_windows_alive": [0] * 8,
          # union_perpass mode: split of per-pass commits by which criterion fired
          "perpass_commits_gate": 0, "perpass_commits_ent": 0, "perpass_commits_both": 0}


def gate_enabled() -> bool:
    return bool(os.environ.get("WEDLM_GATE", ""))


def empty_c_stop_first_exit() -> bool:
    v = os.environ.get("WEDLM_EMPTY_C_STOP", "")
    if v and v != "first_exit_pass":
        raise RuntimeError(f"unknown WEDLM_EMPTY_C_STOP={v!r} (valid: first_exit_pass)")
    return v == "first_exit_pass"


def gate_soft_alpha():
    """union_perpass: WEDLM_GATE_SOFT_ALPHA relaxes the entropy bar in proportion to the gate's
    exit CDF -- commit iff adj < thr*(1 + alpha*CDF). alpha=0 reduces to pure entropy; large
    alpha lets any gate-confident row commit at high entropy. None => hard union (gate OR
    entropy) instead of the soft/modulated bar."""
    v = os.environ.get("WEDLM_GATE_SOFT_ALPHA", "")
    return float(v) if v != "" else None


def union_commit_and_advance() -> bool:
    """union_perpass: WEDLM_UNION_ADVANCE=1 stops the window at the FIRST pass that commits any
    token and advances to the next outer step (commit-and-advance), instead of running until
    all rows commit or K. Keeps passes/forward near 1 (max throughput) while still letting a
    window that commits nothing at pass 0 ride to deeper passes."""
    return os.environ.get("WEDLM_UNION_ADVANCE", "") == "1"


def token_freeze_enabled() -> bool:
    v = os.environ.get("WEDLM_TOKEN_FREEZE", "")
    if v and v not in ("1", "consensus"):
        raise RuntimeError(f"unknown WEDLM_TOKEN_FREEZE={v!r} (valid: 1|consensus; "
                           "window_persist/persist modes are unsupported)")
    return v in ("1", "consensus")


def token_freeze_mode() -> str:
    return os.environ.get("WEDLM_TOKEN_FREEZE", "")


def set_embed_fn(fn):
    """Token-embedding lookup, needed only by COMMIT-ON-AGREE."""
    global _embed_fn
    _embed_fn = fn


def commit_on_agree_enabled() -> bool:
    """WEDLM_COMMIT_ON_AGREE=1: decode a position the MOMENT entropy and gate agree and feed the
    decided token's embedding into the deeper passes, instead of freezing its logits and waiting
    for the whole entropy-selected set to agree. Fuses the inner (depth) loop into the outer
    (diffusion) one. Requires a checkpoint trained with
    commit_depth_prob > 0 -- a model trained without it has never seen an embedding appear
    mid-loop and will be far out of distribution.
    """
    return os.environ.get("WEDLM_COMMIT_ON_AGREE", "0") != "0"


def set_logits_fn(fn):
    global _logits_fn
    _logits_fn = fn


def set_recache_fn(fn):
    global _recache_fn
    _recache_fn = fn


def frozen_store_rows():
    """Attention-side: bool mask over the flat decode stream (None = nothing frozen).
    Frozen rows' stores are suppressed so their back-filled exit-depth KV survives."""
    return _frozen


def _resolve_path() -> str:
    p = os.environ.get("WEDLM_GATE", "")
    if os.path.isdir(p):
        p = os.path.join(p, "exit_gate.pt")
    return p


def _ensure_loaded(hidden_size: int, device, dtype):
    """Lazy-load the trained gate, reconstructing its linear or MLP layout from shapes."""
    global _gate, _loaded
    if _loaded:
        return _gate
    _loaded = True
    sd = torch.load(_resolve_path(), map_location="cpu")
    db = sd.pop("depth_bias", None)   # per-depth bias (uniform-pi init gates); None = legacy
    if "net.weight" in sd:
        net = nn.Linear(hidden_size, 1)
    else:
        hdim = sd["net.0.weight"].shape[0]
        net = nn.Sequential(nn.Linear(hidden_size, hdim), nn.SiLU(), nn.Linear(hdim, 1))

    class _W(nn.Module):
        def __init__(self, n, b):
            super().__init__()
            self.net = n
            self.depth_bias = None if b is None else b

        def forward(self, h, d=None):
            g = self.net(h).squeeze(-1)
            if self.depth_bias is not None and d is not None:
                g = g + self.depth_bias[min(d, self.depth_bias.numel() - 1)]
            return g

    w = _W(net, db)
    w.load_state_dict(sd, strict=False)
    _gate = w.to(device=device, dtype=dtype).eval()
    # depth_bias must stay fp32 AFTER the module-wide dtype cast, not before it
    if w.depth_bias is not None:
        w.depth_bias = w.depth_bias.to(device=device, dtype=torch.float32)
    return _gate


class _SeqState:
    """Per-sequence protocol state, advanced one loop pass at a time."""

    def __init__(self, mask_rows, remaining_mask_indices, threshold, pos_penalty,
                 seq_id=None):
        self.rows = mask_rows                      # flat row indices of this seq's masks
        self.thr = threshold
        self.M = len(remaining_mask_indices)
        if self.M and pos_penalty and remaining_mask_indices:
            mi = torch.tensor(remaining_mask_indices, dtype=torch.float)
            self.pen = (mi - mi[0]) * pos_penalty
        else:
            self.pen = torch.zeros(self.M)
        self.surv = None                           # [M] survival prod(1-lambda)
        self.exit_depth = None                     # [M] int, -1 = not yet exited
        self.frozen_ent = None                     # [M] entropy at exit depth
        self.decision = None                       # (fill_indices, exit_depths, s_stop)
        # HALT TRACE only (WEDLM_DEPTH_TRACE): per-pass hazards / greedy tokens for this
        # window. A GateState is built per window, so these cannot leak across windows.
        self.seq_id = seq_id                       # HALT TRACE only: clustering unit
        self.lam_hist = None                       # [pass] -> [M] lambda
        self.am_hist = None                        # [pass] -> [M] greedy token ids

    def step(self, lam, ent, s, K, q, mode):
        """Advance with pass-s hazards/entropies for this seq's mask rows. Returns True
        once the decision is recorded (seq no longer needs deeper passes)."""
        if self.decision is not None:
            return True
        if self.M == 0:
            self.decision = ([], [], s)
            return True
        dev = lam.device
        if self.surv is None:
            self.surv = torch.ones(self.M, device=dev)
            self.exit_depth = torch.full((self.M,), -1, dtype=torch.long, device=dev)
            self.frozen_ent = torch.zeros(self.M, device=dev)
            self.pen = self.pen.to(dev)
            # COMMIT-ON-AGREE bookkeeping: which of this seq's mask rows have already been
            # decoded mid-loop, and what token each got. Allocated even when the feature is off
            # (M is tiny) so the state shape does not depend on an env var.
            self.committed = torch.zeros(self.M, dtype=torch.bool, device=dev)
            self.committed_tok = torch.zeros(self.M, dtype=torch.long, device=dev)
        if mode == "entropy_perpass":
            return self._step_entropy_perpass(ent, s, K)
        if mode == "union_perpass":
            return self._step_union_perpass(lam, ent, s, K)
        if mode == "gate_depth":
            return self._step_gate_depth(lam, ent, s, K)
        if mode == "gate_depth_left1":
            return self._step_gate_depth_left1(lam, s, K)
        if mode == "gate_depth_min1":
            return self._step_gate_depth_min1(lam, ent, s, K)
        active = self.exit_depth < 0
        # Q-exit: CDF = 1 - surv*(1-lam) crosses q, or forced at the true S_max
        surv_new = self.surv * (1.0 - lam)
        crossed = active & ((1.0 - surv_new >= q) | (s == K - 1))
        self.exit_depth = torch.where(crossed, torch.full_like(self.exit_depth, s), self.exit_depth)
        self.frozen_ent = torch.where(crossed, ent, self.frozen_ent)
        self.surv = torch.where(active, surv_new, self.surv)
        # freeze-at-exit readout: exited tokens keep their exit-depth entropy
        eff_ent = torch.where(self.exit_depth >= 0, self.frozen_ent, ent)
        adj = eff_ent + self.pen
        if self.thr is not None:
            C = (adj < self.thr).nonzero(as_tuple=True)[0]
        else:
            C = adj.argmin().reshape(1)            # commit-1 protocol
        # rows that satisfy BOTH criteria right now and have not been decoded yet. Under
        # commit-on-agree these are decoded immediately instead of waiting for the whole set.
        if self.thr is not None:
            self.agreed_now = (self.exit_depth >= 0) & (adj < self.thr) & (~self.committed)
        else:
            self.agreed_now = torch.zeros_like(self.committed)
        if mode == "allexit":
            ready = bool((self.exit_depth >= 0).all())
        else:                                      # consensus
            ready = C.numel() > 0 and bool((self.exit_depth[C] >= 0).all())
            if (not ready and C.numel() == 0 and empty_c_stop_first_exit()
                    and bool((self.exit_depth >= 0).all())):
                # every row exited with a frozen entropy and none qualifies: deeper passes
                # cannot change C (freeze-at-exit), so the K-1 fallback is decidable NOW
                ready = True
        if ready or s == K - 1:
            if C.numel() == 0:                     # native fallback: commit lowest-entropy 1
                C = adj.argmin().reshape(1)
                _stats["forced_fallback"] += 1
            fill = C.tolist()
            depths = [int(self.exit_depth[i].item()) for i in fill]
            self.decision = (fill, depths, s)
            # exit-depth accounting (token-freeze design study): committed tokens' own exit
            # depths; ALL window tokens' effective depth (un-exited tokens rode to pass s);
            # and the window stop pass. Pure logging — no behavior change.
            for d in depths:
                _stats["exit_hist_committed"][min(max(d, 0), 7)] += 1
            for i in range(self.M):
                e = int(self.exit_depth[i].item())
                _stats["exit_hist_window"][min(e if e >= 0 else s, 7)] += 1
            _stats["window_stop_hist"][min(s, 7)] += 1
            _stats["outer_steps"] += 1
            _stats["passes"] += s + 1
            _stats["committed"] += len(fill)
            if _stats["outer_steps"] % 50 == 0:
                _flush_stats()
            return True
        return False


    def _step_entropy_perpass(self, ent, s, K):
        """WEDLM_GATE_STOP=entropy_perpass: gate ignored; the raw WeDLM entropy rule is applied
        at EVERY pass on the live (un-frozen) readout, qualifying rows are committed on the
        spot (agreed_now -> commit_on_agree_apply injects their embeddings for the next pass),
        and the window stops once every mask row is committed or at pass K-1."""
        if self.thr is None:
            raise RuntimeError("entropy_perpass needs an entropy threshold (commit-1 protocol "
                               "has no qualification rule)")
        if not hasattr(self, "committed_at"):
            self.committed_at = torch.full((self.M,), -1, dtype=torch.long, device=ent.device)
        _stats["perpass_windows_alive"][min(s, 7)] += 1
        adj = ent + self.pen
        agree = (adj < self.thr) & (~self.committed)
        # consumed by commit_on_agree_apply (embedding injection) when a deeper pass follows
        self.agreed_now = agree
        self.committed_at = torch.where(agree, torch.full_like(self.committed_at, s),
                                        self.committed_at)
        _stats["perpass_commits"][min(s, 7)] += int(agree.sum().item())
        will = self.committed | agree
        ready = bool(will.all()) or s == K - 1
        if not ready:
            return False
        fill_mask = will.clone()
        if not bool(fill_mask.any()):              # nothing ever qualified: commit lowest-entropy 1
            j = int(adj.argmin().item())
            fill_mask[j] = True
            self.committed_at[j] = s
            _stats["forced_fallback"] += 1
            _stats["perpass_commits"][min(s, 7)] += 1
        fill = fill_mask.nonzero(as_tuple=True)[0].tolist()
        depths = [int(self.committed_at[i].item()) for i in fill]
        # exit depths: the commit pass for committed rows, the stop pass for the rest (only the
        # fill rows are consumed by the decoder; the exit cap is skipped under COA anyway)
        self.exit_depth = torch.where(fill_mask, self.committed_at,
                                      torch.full_like(self.committed_at, s))
        self.decision = (fill, depths, s)
        for d in depths:
            _stats["exit_hist_committed"][min(max(d, 0), 7)] += 1
        for i in range(self.M):
            _stats["exit_hist_window"][min(int(self.exit_depth[i].item()), 7)] += 1
        _stats["window_stop_hist"][min(s, 7)] += 1
        _stats["outer_steps"] += 1
        _stats["passes"] += s + 1
        _stats["committed"] += len(fill)
        if _stats["outer_steps"] % 50 == 0:
            _flush_stats()
        return True


    def _step_union_perpass(self, lam, ent, s, K):
        """WEDLM_GATE_STOP=union_perpass: entropy and gate are COMBINED per pass and
        qualifying rows commit immediately (commit-on-agree, so WEDLM_COMMIT_ON_AGREE=1 required).
        Two combination rules, selected by WEDLM_GATE_SOFT_ALPHA:
          - alpha is None  -> HARD UNION: commit iff (adj < thr) OR (gate CDF >= q). The gate can
                              commit a row the entropy rule would not, and vice versa.
          - alpha set      -> SOFT/MODULATED: commit iff adj < thr*(1 + alpha*CDF). The gate does
                              not commit on its own; it only widens the entropy bar in proportion
                              to how confident it is that the row has converged.
        The window stops when every row is committed, or at pass K-1, or -- when
        WEDLM_UNION_ADVANCE=1 -- at the first pass that commits anything (commit-and-advance).
        Live (un-frozen) entropy is read every pass; committed rows are excluded via ~committed and
        their embeddings are injected by commit_on_agree_apply for the remaining passes."""
        if self.thr is None:
            raise RuntimeError("union_perpass needs an entropy threshold (commit-1 protocol "
                               "has no qualification rule)")
        q = float(os.environ.get("WEDLM_QEXIT", "0.5"))
        if not hasattr(self, "committed_at"):
            self.committed_at = torch.full((self.M,), -1, dtype=torch.long, device=ent.device)
        _stats["perpass_windows_alive"][min(s, 7)] += 1
        # gate exit CDF (accumulated across passes; committed rows are masked out of `agree` below,
        # so updating surv uniformly for every row is harmless and simpler than the active-mask).
        self.surv = self.surv * (1.0 - lam)
        cdf = 1.0 - self.surv
        gate_pass = cdf >= q
        adj = ent + self.pen
        alpha = gate_soft_alpha()
        if alpha is not None:
            ent_pass = adj < self.thr * (1.0 + alpha * cdf)   # gate-modulated (soft) bar
            crit = ent_pass                                    # gate never commits on its own
        else:
            ent_pass = adj < self.thr
            crit = ent_pass | gate_pass                        # hard union
        agree = crit & (~self.committed)
        self.agreed_now = agree
        self.committed_at = torch.where(agree, torch.full_like(self.committed_at, s),
                                        self.committed_at)
        n_agree = int(agree.sum().item())
        _stats["perpass_commits"][min(s, 7)] += n_agree
        if n_agree:
            _stats["perpass_commits_both"] += int((agree & ent_pass & gate_pass).sum().item())
            _stats["perpass_commits_ent"] += int((agree & ent_pass & ~gate_pass).sum().item())
            _stats["perpass_commits_gate"] += int((agree & gate_pass & ~ent_pass).sum().item())
        will = self.committed | agree
        advance = union_commit_and_advance() and bool(will.any())
        ready = bool(will.all()) or advance or s == K - 1
        if not ready:
            return False
        fill_mask = will.clone()
        if not bool(fill_mask.any()):              # nothing ever qualified: commit lowest-entropy 1
            j = int(adj.argmin().item())
            fill_mask[j] = True
            self.committed_at[j] = s
            _stats["forced_fallback"] += 1
            _stats["perpass_commits"][min(s, 7)] += 1
        fill = fill_mask.nonzero(as_tuple=True)[0].tolist()
        depths = [int(self.committed_at[i].item()) for i in fill]
        self.exit_depth = torch.where(fill_mask, self.committed_at,
                                      torch.full_like(self.committed_at, s))
        self.decision = (fill, depths, s)
        for d in depths:
            _stats["exit_hist_committed"][min(max(d, 0), 7)] += 1
        for i in range(self.M):
            _stats["exit_hist_window"][min(int(self.exit_depth[i].item()), 7)] += 1
        _stats["window_stop_hist"][min(s, 7)] += 1
        _stats["outer_steps"] += 1
        _stats["passes"] += s + 1
        _stats["committed"] += len(fill)
        if _stats["outer_steps"] % 50 == 0:
            _flush_stats()
        return True


    def _step_gate_depth(self, lam, ent, s, K):
        """WEDLM_GATE_STOP=gate_depth: a clean division of labor --
        ENTROPY alone decides commits (live readout every pass, qualifying rows committed
        immediately via commit-on-agree, identical to entropy_perpass), while the GATE alone
        decides DEPTH: after this pass's commits, take the mean exit CDF (1 - prod(1-lambda))
        over the still-uncommitted rows; if it >= q (WEDLM_QEXIT) the window ADVANCES -- the
        gate says those latents have halted, so deeper passes will not change their readouts
        and the next outer step should re-predict them with fresh committed context -- else the
        window runs another pass. q interpolates between single-pass entropy (q=0: always
        advance) and full-K entropy_perpass (q=1: never advance early)."""
        if self.thr is None:
            raise RuntimeError("gate_depth needs an entropy threshold (commit-1 protocol "
                               "has no qualification rule)")
        q = float(os.environ.get("WEDLM_QEXIT", "0.5"))
        if not hasattr(self, "committed_at"):
            self.committed_at = torch.full((self.M,), -1, dtype=torch.long, device=ent.device)
        _stats["perpass_windows_alive"][min(s, 7)] += 1
        # gate exit CDF accumulates every pass; committed rows are excluded from the mean below
        self.surv = self.surv * (1.0 - lam)
        cdf = 1.0 - self.surv
        adj = ent + self.pen
        agree = (adj < self.thr) & (~self.committed)
        self.agreed_now = agree
        self.committed_at = torch.where(agree, torch.full_like(self.committed_at, s),
                                        self.committed_at)
        will = self.committed | agree
        resid = ~will


        n_agree, n_resid, cdf_sum = torch.stack([
            agree.sum().float(), resid.sum().float(), (cdf * resid.float()).sum(),
        ]).tolist()
        _stats["perpass_commits"][min(s, 7)] += int(n_agree)
        advance = (n_resid == 0) or (cdf_sum / n_resid >= q)
        ready = advance or s == K - 1
        if not ready:
            return False
        fill_mask = will.clone()
        # Finalize with BATCHED reads: one .tolist() replaces the per-row .item() loops
        # (2 x M syncs at M~16-32 was 1-2 ms of pure sync per window). ca[i] >= 0 iff the
        # row ever agreed, i.e. exactly `will` (committed rows agreed at an earlier pass).
        ca = self.committed_at.tolist()
        if not any(c >= 0 for c in ca):            # nothing ever qualified: commit lowest-entropy 1
            j = int(adj.argmin().item())
            fill_mask[j] = True
            self.committed_at[j] = s
            ca[j] = s
            _stats["forced_fallback"] += 1
            _stats["perpass_commits"][min(s, 7)] += 1
        fill = [i for i, c in enumerate(ca) if c >= 0]
        depths = [ca[i] for i in fill]
        self.exit_depth = torch.where(fill_mask, self.committed_at,
                                      torch.full_like(self.committed_at, s))
        self.decision = (fill, depths, s)
        for d in depths:
            _stats["exit_hist_committed"][min(max(d, 0), 7)] += 1
        for i in range(self.M):
            _stats["exit_hist_window"][min(ca[i] if ca[i] >= 0 else s, 7)] += 1
        _stats["window_stop_hist"][min(s, 7)] += 1
        _stats["outer_steps"] += 1
        _stats["passes"] += s + 1
        _stats["committed"] += len(fill)
        if _stats["outer_steps"] % 50 == 0:
            _flush_stats()
        return True


    def _step_gate_depth_left1(self, lam, s, K):
        """WEDLM_GATE_STOP=gate_depth_left1: strictly-L2R commits,
        gate-controlled depth. Every pass commits exactly ONE token -- the LEFTMOST not-yet-
        committed mask row (entropy plays no role); then the mean exit CDF over the remaining
        uncommitted rows decides advance (>= q, WEDLM_QEXIT) vs another pass. So a window
        commits between 1 (q=0) and K tokens per forward, left to right, and the gate's halt
        probability is the only depth signal. Requires COMMIT_ON_AGREE (the per-pass commit is
        realized by embedding injection, same as gate_depth)."""
        q = float(os.environ.get("WEDLM_QEXIT", "0.5"))
        if not hasattr(self, "committed_at"):
            self.committed_at = torch.full((self.M,), -1, dtype=torch.long, device=lam.device)
        _stats["perpass_windows_alive"][min(s, 7)] += 1
        self.surv = self.surv * (1.0 - lam)
        cdf = 1.0 - self.surv
        agree = torch.zeros(self.M, dtype=torch.bool, device=lam.device)
        not_comm = (~self.committed).nonzero(as_tuple=True)[0]
        if not_comm.numel():
            agree[not_comm[0]] = True              # leftmost uncommitted row, exactly one
        self.agreed_now = agree
        self.committed_at = torch.where(agree, torch.full_like(self.committed_at, s),
                                        self.committed_at)
        _stats["perpass_commits"][min(s, 7)] += int(agree.sum().item())
        will = self.committed | agree
        resid = ~will
        if bool(resid.any()):
            advance = bool(cdf[resid].float().mean().item() >= q)
        else:
            advance = True
        ready = advance or s == K - 1
        if not ready:
            return False
        fill_mask = will.clone()
        fill = fill_mask.nonzero(as_tuple=True)[0].tolist()
        depths = [int(self.committed_at[i].item()) for i in fill]
        self.exit_depth = torch.where(fill_mask, self.committed_at,
                                      torch.full_like(self.committed_at, s))
        self.decision = (fill, depths, s)
        for d in depths:
            _stats["exit_hist_committed"][min(max(d, 0), 7)] += 1
        for i in range(self.M):
            _stats["exit_hist_window"][min(int(self.exit_depth[i].item()), 7)] += 1
        _stats["window_stop_hist"][min(s, 7)] += 1
        _stats["outer_steps"] += 1
        _stats["passes"] += s + 1
        _stats["committed"] += len(fill)
        if _stats["outer_steps"] % 50 == 0:
            _flush_stats()
        return True

    def _step_gate_depth_min1(self, lam, ent, s, K):
        """WEDLM_GATE_STOP=gate_depth_min1: exactly ONE commit per
        inner-loop pass -- the argmin of (entropy + pos_penalty) over the not-yet-committed
        rows -- while the GATE alone decides depth, exactly as gate_depth/gate_depth_left1 do.

        Confidence-ranked sibling of gate_depth_left1: same one-token-per-pass budget, but the
        token is the most CONFIDENT remaining position under the same left-bias gate_depth
        applies (`pen = (mi - mi[0]) * pos_penalty`, so at the canonical 0.02 it is entropy
        ranking with a mild L2R tie-break) instead of unconditionally the leftmost. It is also
        the gate-path analogue of sampler.py's commit-1 branch (`entropy_threshold is None ->
        sel = adjusted.argmin()`), which gate decoding never reaches because gate commits are
        decided live inside the looped forward.

        NEEDS NO THRESHOLD (unlike gate_depth): argmin always returns exactly one row, so there
        is no qualification rule and self.thr may be None. Consequently the captured
        check-graph family -- which bakes in the `adj < thr` test -- must NOT serve this mode;
        it is deliberately absent from that whitelist and falls back to the eager per-pass check.
        """
        q = float(os.environ.get("WEDLM_QEXIT", "0.5"))
        if not hasattr(self, "committed_at"):
            self.committed_at = torch.full((self.M,), -1, dtype=torch.long, device=lam.device)
        _stats["perpass_windows_alive"][min(s, 7)] += 1
        self.surv = self.surv * (1.0 - lam)
        cdf = 1.0 - self.surv
        agree = torch.zeros(self.M, dtype=torch.bool, device=lam.device)
        if bool((~self.committed).any()):
            # +inf on committed rows so argmin can only land on a live one
            adj = (ent + self.pen).masked_fill(self.committed, float("inf"))
            agree[int(adj.argmin().item())] = True    # most confident remaining, exactly one
        self.agreed_now = agree
        self.committed_at = torch.where(agree, torch.full_like(self.committed_at, s),
                                        self.committed_at)
        _stats["perpass_commits"][min(s, 7)] += int(agree.sum().item())
        will = self.committed | agree
        resid = ~will
        if bool(resid.any()):
            advance = bool(cdf[resid].float().mean().item() >= q)
        else:
            advance = True
        ready = advance or s == K - 1
        if not ready:
            return False
        fill_mask = will.clone()
        fill = fill_mask.nonzero(as_tuple=True)[0].tolist()
        depths = [int(self.committed_at[i].item()) for i in fill]
        self.exit_depth = torch.where(fill_mask, self.committed_at,
                                      torch.full_like(self.committed_at, s))
        self.decision = (fill, depths, s)
        for d in depths:
            _stats["exit_hist_committed"][min(max(d, 0), 7)] += 1
        for i in range(self.M):
            _stats["exit_hist_window"][min(int(self.exit_depth[i].item()), 7)] += 1
        _stats["window_stop_hist"][min(s, 7)] += 1
        _stats["outer_steps"] += 1
        _stats["passes"] += s + 1
        _stats["committed"] += len(fill)
        if _stats["outer_steps"] % 50 == 0:
            _flush_stats()
        return True


class GatePlan:
    """Batch plan built by the decoder before the forward; consumed after it."""

    def __init__(self, seq_states):
        self.seqs = seq_states                     # ordered like prepared.active_seqs

    def all_decided(self):
        return all(s.decision is not None for s in self.seqs)


def set_plan(plan):
    global _plan, _frozen
    _plan = plan
    _frozen = None


def clear_plan():
    global _plan, _frozen
    _plan = None
    _frozen = None


def plan_active() -> bool:
    return _plan is not None


# HALT-PROBABILITY TRACE (piggybacks WEDLM_DEPTH_TRACE, default OFF -> zero cost and the
# decode is untouched). One row per COMMITTED token: its id, the gate hazard lambda at pass 0,
# the hazard at the pass where it committed, and that depth.
#
# Recorded HERE rather than in wedlm_decoder because gate_decode already caches each pass's
# greedy tokens (st.pass_argmax), so the token id is available at the decision without any
# output-position plumbing -- and a per-category mean only needs the token id, not the position.
#
# lam1 is the quantity to compare ACROSS token categories: it is evaluated at the same depth
# for every token, so it carries no depth confound. lam_commit is evaluated at each token's own
# committing pass, so a category's mean is entangled with the depths that category tends to reach.
_HALT_ROWS: list = []


def _halt_trace_enabled() -> bool:
    return bool(os.environ.get("WEDLM_DEPTH_TRACE"))


def _halt_trace_collect(st) -> None:
    fill, depths, _ = st.decision
    if not fill or st.lam_hist is None:
        return
    lam = [t.tolist() for t in st.lam_hist]
    am = [t.tolist() for t in st.am_hist]
    top = len(lam) - 1
    for i, d in zip(fill, depths):
        dd = d if 0 <= d <= top else top          # forced-fallback rows can carry depth -1
        if i < len(lam[0]) and i < len(am[dd]):
            _HALT_ROWS.append((int(am[dd][i]), float(lam[0][i]), float(lam[dd][i]),
                               int(dd) + 1, st.seq_id))


def halt_trace_dump(path: str) -> int:
    """Append one JSON object per committed token; returns the row count written."""
    if not _HALT_ROWS:
        return 0
    with open(path, "a") as fh:
        for tok, l0, lc, d, sid in _HALT_ROWS:
            fh.write(json.dumps({"tok": tok, "lam1": l0, "lam_commit": lc,
                                 "depth": d, "seq_id": sid}) + "\n")
    n = len(_HALT_ROWS)
    _HALT_ROWS.clear()
    return n


def take_decision(active_ordinal: int):
    """Decoder-side: the recorded (fill_indices, exit_depths, s_stop) for active seq j."""
    d = _plan.seqs[active_ordinal].decision
    if d is None:
        raise RuntimeError("gate_decode: seq reached decode without a recorded decision "
                           "(loop_pass_check not called to S_max?)")
    return d


def commit_on_agree_apply(rd: torch.Tensor, s: int, K: int, carry: torch.Tensor = None):
    """Decode every position that agreed on THIS pass and hand back the carry to use next.

    Selection + token choice always read `rd` (the depth-s READOUT -- for a sandwich that is
    the coda-branch output the lm_head scores, matching training's per_step readout). The
    embedding substitution targets the next pass's LOOP INPUT:
      - all-layer (carry=None): rd doubles as the carry -> substitute into a clone of rd;
      - sandwich (carry=the pre-coda loop stream, post carry-norm): substitute into a clone
        of `carry`. Training's ordering is carry-norm first, then the raw embedding
        OVERWRITES the normed carry for committed rows -- pass the normed stream here.

    Returns the NEW carry tensor, or None when nothing was decoded / the feature is off (the
    caller keeps its own carry). Never writes rd in place: the looped forward stashes rd
    BEFORE this hook, and that stash is what the final commit turns into logits.

    The injected value is the RAW embedding, matching the training forward:
    native at all-layer, an embedding substitution under a sandwich.
    """
    if _plan is None or not commit_on_agree_enabled():
        return None
    from wedlm.utils.context import get_context
    if get_context().is_prefill:
        return None
    if _embed_fn is None or _logits_fn is None:
        raise RuntimeError("commit-on-agree needs set_embed_fn/set_logits_fn from ModelRunner")
    base = carry if carry is not None else rd
    out = None
    for st in _plan.seqs:
        ag = getattr(st, "agreed_now", None)
        if ag is None or st.M == 0 or not bool(ag.any()):
            continue
        idx = ag.nonzero(as_tuple=True)[0]
        rows = st.rows[idx] if torch.is_tensor(st.rows) else [st.rows[i] for i in idx.tolist()]
        with torch.no_grad():
            pa = getattr(st, "pass_argmax", None)
            if pa is not None and pa.numel() == st.M:
                # loop_pass_check already ran the lm_head over ALL of this seq's mask rows
                # on this very readout; the agreeing subset's argmax is identical.
                tok = pa[idx]
            else:
                tok = _logits_fn(rd[rows]).argmax(dim=-1)  # greedy; the suite decodes greedily
            emb = _embed_fn(tok).to(base.dtype)
        if out is None:
            out = base.clone()
        out[rows] = emb
        st.committed[idx] = True
        st.committed_tok[idx] = tok
        _stats["coa_committed"] = _stats.get("coa_committed", 0) + int(idx.numel())
        _stats["coa_depth_sum"] = _stats.get("coa_depth_sum", 0) + int(idx.numel()) * s
    return out


def graphed_seed(gv) -> bool:
    """infra pass 2b: seed the check-graph state buffers for THIS outer step. Returns True
    iff the captured check-graph family may serve it: exactly one active seq, gate_depth
    mode, a real threshold, M > 0. Any other shape falls back to the eager check."""
    if _plan is None or len(_plan.seqs) != 1:
        return False
    mode = os.environ.get("WEDLM_GATE_STOP", "consensus")
    if mode not in ("gate_depth", "gate_depth_left1"):
        return False
    st = _plan.seqs[0]
    if st.decision is not None or st.M == 0:
        return False
    if mode == "gate_depth" and st.thr is None:
        return False
    dev = gv["chk_mask"].device
    rows = st.rows if torch.is_tensor(st.rows) else torch.as_tensor(st.rows, device=dev)
    rows = rows.to(dev)
    st._chk_rows = rows
    gv["chk_mask"].zero_()
    gv["chk_mask"][rows] = True
    gv["chk_pen"].zero_()
    gv["chk_pen"][rows] = st.pen.to(dev, dtype=torch.float32)
    gv["chk_thr"].fill_(float(st.thr) if st.thr is not None else 0.0)
    gv["chk_surv"].fill_(1.0)
    gv["chk_committed"].zero_()
    gv["chk_committed_at"].fill_(-1)
    return True


def graphed_gate_depth_step(gv, s: int, K: int) -> bool:
    """infra pass 2b host half: after check-graph s replayed, read the packed decision
    (ONE small D2H sync) and advance the protocol. Bookkeeping mirrors _step_gate_depth
    exactly; the full state readback happens only once, at the deciding pass."""
    st = _plan.seqs[0]
    q = float(os.environ.get("WEDLM_QEXIT", "0.5"))
    _stats["perpass_windows_alive"][min(s, 7)] += 1
    n_agree, n_resid, cdf_sum, fb_idx = gv["chk_packed"].tolist()
    _stats["perpass_commits"][min(s, 7)] += int(n_agree)
    advance = (n_resid == 0) or (cdf_sum / n_resid >= q)
    ready = advance or s == K - 1
    if not ready:
        return False
    rows = st._chk_rows
    ca = gv["chk_committed_at"][rows].tolist()
    if not any(c >= 0 for c in ca):                # forced fallback: lowest adjusted entropy
        gpos = int(fb_idx)
        loc = (rows == gpos).nonzero(as_tuple=True)[0]
        j = int(loc[0].item()) if loc.numel() else 0
        ca[j] = s
        _stats["forced_fallback"] += 1
        _stats["perpass_commits"][min(s, 7)] += 1
    fill = [i for i, c in enumerate(ca) if c >= 0]
    depths = [ca[i] for i in fill]
    st.decision = (fill, depths, s)
    # phase 3a: hand the decoder each committed position's greedy token straight from the
    # check-graph's per-depth argmax buffer (one small [K, M] read) -- the decoder then
    # skips its end-of-step K x lm_head rebuild AND its per-token sample/.item() loop.
    if "chk_argmax" in gv:
        am = gv["chk_argmax"][:, rows].tolist()
        st.decision_tokens = [am[depths[n]][i] for n, i in enumerate(fill)]
    for d in depths:
        _stats["exit_hist_committed"][min(max(d, 0), 7)] += 1
    for i in range(st.M):
        _stats["exit_hist_window"][min(ca[i] if ca[i] >= 0 else s, 7)] += 1
    _stats["window_stop_hist"][min(s, 7)] += 1
    _stats["outer_steps"] += 1
    _stats["passes"] += s + 1
    _stats["committed"] += len(fill)
    if _stats["outer_steps"] % 50 == 0:
        _flush_stats()
    return True


def graphed_tokens(j: int):
    """phase 3a decoder-side: (fill_indices, fill_depths, token_ids) recorded by the
    check-graph path for active seq j, or None when this step wasn't check-graph-served
    (the decoder then falls back to the logits_by_depth route)."""
    if _plan is None:
        return None
    st = _plan.seqs[j]
    toks = getattr(st, "decision_tokens", None)
    if toks is None or st.decision is None:
        return None
    return (st.decision[0], st.decision[1], toks)


def graphed_all_tokens_ready() -> bool:
    """True iff EVERY active seq's decision carries check-graph tokens, i.e. the decoder
    will not need logits_by_depth at all this step."""
    if _plan is None:
        return False
    return all(
        s.decision is not None
        and (s.M == 0 or getattr(s, "decision_tokens", None) is not None)
        for s in _plan.seqs
    )


def loop_pass_check(rd: torch.Tensor, s: int, K: int) -> bool:
    """Called by the looped forward after stashing pass-s readout rd (full token stream).
    Advances every undecided sequence's protocol state; True => every seq decided => the
    shared loop may stop (no deeper pass is computed)."""
    if _plan is None:
        return False
    from wedlm.utils.context import get_context
    if get_context().is_prefill:
        # prefill must run the full K passes: every depth's prompt KV is needed later
        return False
    q = float(os.environ.get("WEDLM_QEXIT", "0.5"))
    mode = os.environ.get("WEDLM_GATE_STOP", "consensus")
    if mode not in ("consensus", "allexit", "entropy_perpass", "union_perpass", "gate_depth",
                    "gate_depth_left1", "gate_depth_min1"):
        raise RuntimeError(f"unknown WEDLM_GATE_STOP={mode!r}")
    if mode in ("entropy_perpass", "union_perpass", "gate_depth", "gate_depth_left1",
                "gate_depth_min1"):
        if not commit_on_agree_enabled():
            raise RuntimeError(f"WEDLM_GATE_STOP={mode} requires WEDLM_COMMIT_ON_AGREE=1 "
                               "(per-pass commits are realized by embedding injection)")
        if token_freeze_enabled():
            raise RuntimeError(f"WEDLM_GATE_STOP={mode} does not support WEDLM_TOKEN_FREEZE")
    tf = token_freeze_enabled()
    if tf:
        # honest denominator: EVERY stream row (masks + non-mask window rows, decided
        # seqs included) is computed each executed pass — a real freeze impl only drops
        # the frozen mask rows, so realized saving = tf_frozen / tf_stream
        _stats["tf_stream_tok_passes"] += rd.shape[0]
    gate = _ensure_loaded(rd.shape[-1], rd.device, rd.dtype)
    for st in _plan.seqs:
        if st.decision is not None or st.M == 0:
            if st.decision is None:
                st.decision = ([], [], s)
            continue
        if tf:
            # savings accounting for this executed pass: rows frozen at earlier passes
            # would be dropped from the stream by a real variable-length implementation
            _stats["tf_tok_passes"] += st.M
            fz = getattr(st, "tf_frozen", None)
            if fz is not None:
                _stats["tf_frozen_tok_passes"] += int(fz.sum().item())
        rows = rd[st.rows]                         # [M, H] this seq's mask readouts
        with torch.no_grad():
            lam = torch.sigmoid(gate(rows, s).float())
            logits = _logits_fn(rows)


            lf = logits.float()
            lse = torch.logsumexp(lf, dim=-1)
            ent = lse - (torch.softmax(lf, dim=-1) * lf).sum(dim=-1)


            st.pass_argmax = logits.argmax(dim=-1)
            if _halt_trace_enabled():
                if st.lam_hist is None:
                    st.lam_hist, st.am_hist = [], []


                st.lam_hist.append(lam.detach().float().cpu())
                st.am_hist.append(st.pass_argmax.detach().cpu())
        st.step(lam, ent, s, K, q, mode)
        # collect AFTER step(), which is what records st.decision for this window
        if st.decision is not None and _halt_trace_enabled():
            _halt_trace_collect(st)
    done = _plan.all_decided()
    if done:
        _stats["batch_steps"] += 1
        _stats["batch_passes"] += s + 1
    if tf and not done and _recache_fn is not None:
        # Freeze the mask rows that EXITED at this pass in still-undecided seqs (decided
        # seqs' deeper passes are dead compute — per-seq varlen attention reads nothing
        # across seq boundaries, so freezing them would change no output). Back-fill each
        # frozen row's depth-s KV into buffers s+1..K-1 NOW so the next pass reads the
        # frozen key from buffer[s+1..]; the attention store for frozen rows is suppressed
        # (frozen_store_rows) so deeper passes never overwrite it.
        global _frozen
        new_rows = []
        consensus = token_freeze_mode() == "consensus"
        for st in _plan.seqs:
            if st.decision is not None or st.exit_depth is None:
                continue
            new_local = st.exit_depth == s
            if consensus:
                # per-token gate∩entropy consensus: freeze only if the frozen exit
                # entropy also commit-qualifies (static after exit, so decidable now);
                # commit-1 protocol (thr None) has no qualification set — no freeze
                if st.thr is None:
                    continue
                new_local = new_local & ((st.frozen_ent + st.pen) < st.thr)
            if not hasattr(st, "tf_frozen"):
                st.tf_frozen = torch.zeros(st.M, dtype=torch.bool, device=rd.device)
            st.tf_frozen |= new_local
            for i in new_local.nonzero(as_tuple=True)[0].tolist():
                new_rows.append(st.rows[i])
        if new_rows:
            if _frozen is None:
                _frozen = torch.zeros(rd.shape[0], dtype=torch.bool, device=rd.device)
            idx = torch.tensor(new_rows, dtype=torch.long, device=rd.device)
            _frozen[idx] = True
            if s + 1 < K:
                from wedlm.utils.context import get_context as _gc
                sm = _gc().slot_mapping
                if sm is not None:
                    _recache_fn(sm[idx], s + 1)
    return done


def _flush_stats():
    sd = os.environ.get("WEDLM_GATE_STATS_DIR", "")
    if not (sd and _stats["outer_steps"]):
        return
    try:
        os.makedirs(sd, exist_ok=True)
        rec = dict(_stats)
        rec["q"] = os.environ.get("WEDLM_QEXIT", "0.5")
        rec["stop"] = os.environ.get("WEDLM_GATE_STOP", "consensus")
        with open(os.path.join(sd, f"gate_stats_{os.getpid()}.json"), "w") as f:
            json.dump(rec, f)
    except Exception:
        pass


@atexit.register
def _report():


    _flush_stats()
    if _stats["outer_steps"]:
        print(
            f"[gate_decode] outer_steps={_stats['outer_steps']} "
            f"mean_passes={_stats['passes'] / _stats['outer_steps']:.3f} "
            f"executed_passes/batch_step="
            f"{_stats['batch_passes'] / max(1, _stats['batch_steps']):.3f} "
            f"mean_commits/step={_stats['committed'] / _stats['outer_steps']:.3f} "
            f"forced_fallback={_stats['forced_fallback']} "
            f"(q={os.environ.get('WEDLM_QEXIT', '0.5')}, "
            f"stop={os.environ.get('WEDLM_GATE_STOP', 'consensus')})",
            flush=True,
        )
    if _stats["tf_tok_passes"]:
        mfrac = _stats["tf_frozen_tok_passes"] / _stats["tf_tok_passes"]
        sfrac = _stats["tf_frozen_tok_passes"] / max(1, _stats["tf_stream_tok_passes"])
        print(f"[gate_decode] token-freeze: frozen {mfrac:.1%} of freeze-eligible mask "
              f"token-passes; {sfrac:.1%} of ALL stream token-passes -> realizable "
              f"loop-compute ratio {1.0 - sfrac:.3f} (speedup x"
              f"{1.0 / max(1e-9, 1.0 - sfrac):.3f})", flush=True)
