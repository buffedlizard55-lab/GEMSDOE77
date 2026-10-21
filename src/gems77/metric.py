"""Official Distance-Weighted Tversky Index (DTI) for the DOE GEMS Prize Challenge.

Verified from official competition problem description (DrivenData Page 967):
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
and official organizer forum clarification on known-fault masking (Thread 11516):
https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516

Mathematical formulation:
- Triangular kernel: k(d) = max(1 - d / R, 0), with R = 300 m = 3 pixels at 100 m resolution.
- TP_w = sum_{g in G} max_{x : d(x,g) <= R} p(x) * k(d(x,g))
- FP_w = sum_{x : p(x) > 0} p(x) * [1 - max_{g in G} k(d(x,g))]
- FN_w = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))] = |G| - TP_w
- DTI(alpha=0.2, beta=0.8) = TP_w / (TP_w + 0.2 * FP_w + 0.8 * FN_w + eps)
                           = TP_w / (0.2 * (TP_w + FP_w) + 0.8 * |G| + eps)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt

ALPHA: float = 0.2
BETA: float = 0.8
RADIUS_PX: float = 3.0
EPS: float = 1e-12


def kernel(d: np.ndarray | float, radius: float = RADIUS_PX) -> np.ndarray:
    """Linear triangular kernel k(d) = max(1 - d / R, 0) where d and radius are in pixels (1 px = 100 m)."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


def _offsets(radius: float = RADIUS_PX) -> list[tuple[int, int, float]]:
    r = int(np.ceil(radius))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = float(kernel(np.hypot(dy, dx), radius))
            if k > 0.0:
                out.append((dy, dx, k))
    return out


_OFFS = _offsets()


def _prepare(pred, truth, valid, known):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("prediction and truth must be equal-shaped 2-D grids")
    valid = np.ones(pred.shape, bool) if valid is None else np.asarray(valid, bool)
    known = np.zeros(pred.shape, bool) if known is None else np.asarray(known, bool)
    if valid.shape != pred.shape or known.shape != pred.shape:
        raise ValueError("mask grid mismatch")
    active = valid & ~known
    vals = pred[active]
    if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
        raise ValueError("predictions inside the scored domain must be finite and in [0, 1]")
    p = np.where(active & np.isfinite(pred), pred, 0.0).astype(np.float64)
    g = active & (np.asarray(truth) > 0)
    return p, g, active


def dti_exact(pred, truth, valid=None, known=None, alpha: float = ALPHA, beta: float = BETA) -> dict:
    """Exact DTI for arbitrary soft or binary predictions in [0, 1]."""
    p, g, _ = _prepare(pred, truth, valid, known)
    H, W = p.shape
    yy, xx = np.nonzero(g)
    n = int(yy.size)
    if n == 0:
        return dict(tp=0.0, fp=float(p.sum()), fn=0.0, n_truth=0, dti=0.0, coverage=0.0)
    credit = np.zeros(n, dtype=np.float64)
    for dy, dx, k in _OFFS:
        ny, nx = yy + dy, xx + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        credit[ok] = np.maximum(credit[ok], p[ny[ok], nx[ok]] * k)
    tp = float(credit.sum())
    fn = float(n) - tp
    d = distance_transform_edt(~g)
    fp = float((p * (1.0 - kernel(d))).sum())
    dti = tp / (tp + alpha * fp + beta * fn + EPS)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=float(dti), coverage=tp / n)


def dti_binary(pred_bool, truth, valid=None, known=None, alpha: float = ALPHA, beta: float = BETA) -> dict:
    """Fast exact DTI when predictions are binary {0, 1} via Euclidean distance transforms."""
    pred_bool = np.asarray(pred_bool, bool)
    truth = np.asarray(truth, bool)
    valid_ = np.ones(pred_bool.shape, bool) if valid is None else np.asarray(valid, bool)
    known_ = np.zeros(pred_bool.shape, bool) if known is None else np.asarray(known, bool)
    active = valid_ & ~known_
    p = pred_bool & active
    g = truth & active
    n = int(g.sum())
    if n == 0:
        return dict(tp=0.0, fp=float(p.sum()), fn=0.0, n_truth=0, dti=0.0, coverage=0.0)
    if not p.any():
        return dict(tp=0.0, fp=0.0, fn=float(n), n_truth=n, dti=0.0, coverage=0.0)
    dp = distance_transform_edt(~p)
    tp = float(kernel(dp[g]).sum())
    fn = float(n) - tp
    dg = distance_transform_edt(~g)
    fp = float((1.0 - kernel(dg[p])).sum())
    dti = tp / (tp + alpha * fp + beta * fn + EPS)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=float(dti), coverage=tp / n)


def dti_bruteforce(pred, truth, alpha: float = ALPHA, beta: float = BETA, radius: float = RADIUS_PX) -> dict:
    """Literal O(|G|*|P|) transcription of the published equations for unit-test verification."""
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, bool)
    gs = np.argwhere(truth)
    xs = np.argwhere(pred > 0)
    tp = fn = 0.0
    for g in gs:
        best = 0.0
        for x in xs:
            d = float(np.hypot(*(x - g)))
            if d <= radius:
                best = max(best, pred[tuple(x)] * max(1.0 - d / radius, 0.0))
        tp += best
        fn += 1.0 - best
    fp = 0.0
    for x in xs:
        kmax = 0.0
        for g in gs:
            kmax = max(kmax, max(1.0 - float(np.hypot(*(x - g))) / radius, 0.0))
        fp += pred[tuple(x)] * (1.0 - kmax)
    return dict(tp=tp, fp=fp, fn=fn, dti=tp / (tp + alpha * fp + beta * fn + EPS))


def marginal_inclusion_threshold(current_dti: float, alpha: float = ALPHA) -> float:
    """Minimum marginal kernel credit k for an added dot to improve DTI at score level s.

    CORRECTED 2026-10-04 (IR-32-BAR-01).  This function previously returned
    ``alpha*s / (1 - alpha*s)`` -- the form quoted in ``docs/research/score-ceiling-analysis.md``
    and propagated into the GEMSDOE28 repository as ``tau_live = 0.05485``.  That form does not
    follow from the metric.  Writing ``DTI = T / (alpha*(T+F) + beta*K)`` and adding one unit of
    mass whose realised kernel weight is k (so dT = k and dF = 1 - k),

        dDTI > 0  <=>  k*D > alpha*T  <=>  k > alpha*DTI        with D = alpha*(T+F) + beta*K,

    because the denominator rises by exactly ``alpha`` regardless of where the mass lands:

        dD = alpha*(dT + dF) = alpha*(k + 1 - k) = alpha.

    The two forms coincide at s = 0 and differ materially above it: at s = 0.2708 the wrong form
    gives 0.05726 and the correct one 0.05416, and the only kernel weight on the 100 m lattice in
    that window is the (2,2)-diagonal 1 - 2*sqrt(2)/3 = 0.05719.  The wrong form therefore
    **rejects a legitimate (2,2)-diagonal emission** at exactly the group's best reported score.
    Verified by brute force in ``scripts/verify_theorems.py``.
    """
    return alpha * current_dti


# ---------------------------------------------------------------------------------------------
# Full-API surface used by scripts/audit_shipped.py, src/gems32/emitter.py and tests/test_*.py.
# Added back 2026-10-04 after the parallel-session rewrite of this module removed it.
# ---------------------------------------------------------------------------------------------

def _offsets_list_as_arrays():
    import numpy as _np
    dy = _np.array([o[0] for o in _OFFS], dtype=_np.int64)
    dx = _np.array([o[1] for o in _OFFS], dtype=_np.int64)
    kw = _np.array([o[2] for o in _OFFS], dtype=_np.float64)
    return dy, dx, kw


_OFF = _offsets_list_as_arrays()


def _shift(arr, dy: int, dx: int):
    """Shift ``arr`` by (dy, dx), filling with 0 (exact here: every shifted array is >= 0)."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = arr[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return out


def components(pred, truth, valid=None, known=None) -> dict:
    """Exact ``TP_w``/``FP_w``/``FN_w`` plus the algebra terms ``K``, ``S``, ``M``."""
    p, g, _ = _prepare(pred, truth, valid, known)
    dy, dx, kw = _OFF
    best = np.zeros(p.shape)
    for d, e, kk in zip(dy, dx, kw):
        np.maximum(best, _shift(p, -int(d), -int(e)) * kk, out=best)
    best = np.clip(best, 0.0, 1.0)
    tp = float(best[g].sum())
    if g.any():
        dist_to_g = distance_transform_edt(~g)
        k_near = kernel(dist_to_g)
    else:
        k_near = np.zeros(p.shape)
    fp = float((p * (1.0 - k_near)).sum())
    fn = float(g.sum()) - tp
    k_total = float(g.sum())
    s = float(p.sum())
    m = float((p * k_near).sum())
    return {"TP_w": tp, "FP_w": fp, "FN_w": fn, "K": k_total, "S": s, "M": m,
            "credit_per_unit_mass": (tp / s) if s else 0.0,
            "marginal_bar": ALPHA * dti(tp, fp, fn)}


def dti(tp: float, fp: float, fn: float) -> float:
    """Distance-weighted Tversky index from the three published components."""
    return float(tp / (tp + ALPHA * fp + BETA * fn + EPS))


def dti_components(tp: float, fp: float, fn: float) -> float:
    return dti(tp, fp, fn)


def dti_algebra(t: float, s: float, m: float, k: float) -> float:
    """Closed form ``T / (0.2*(T + S - M) + 0.8*K)`` (identical to :func:`dti`)."""
    return float(t / (ALPHA * (t + s - m) + BETA * k + EPS))


def score(pred, truth) -> float:
    c = components(pred, truth)
    return dti(c["TP_w"], c["FP_w"], c["FN_w"])


@dataclass(frozen=True)
class CreditAudit:
    n_emitted: int
    total_mass: float
    redundancy_fraction: float
    mean_self_kernel: float


def credit_audit(mask) -> CreditAudit:
    """Kernel-structure audit of an emission mask (no truth required)."""
    m = np.asarray(mask, dtype=np.float64)
    pos = m > 0
    n = int(pos.sum())
    if n == 0:
        return CreditAudit(0, 0.0, 0.0, 0.0)
    best = np.zeros(m.shape)
    dy, dx, kw = _OFF
    for d, e, kk in zip(dy, dx, kw):
        if d == 0 and e == 0:
            continue
        np.maximum(best, _shift(m, -int(d), -int(e)) * kk, out=best)
    red = np.clip(best, 0.0, 1.0)
    return CreditAudit(n_emitted=n, total_mass=float(m.sum()),
                       redundancy_fraction=float((red[pos] > 0).mean()),
                       mean_self_kernel=float(red[pos].mean()))
