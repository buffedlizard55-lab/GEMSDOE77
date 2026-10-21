"""Bayes-calibrated and submodular 300 m kernel coverage dot emitters for H77-HSL.

Why Bayes prior calibration matters for expected_credit(p):
-----------------------------------------------------------
In fold-wise HGB training, positives are sampled at pi_train = 1/3 (40,000 pos vs
80,000 neg), whereas off-catalogue fault pixels occupy pi_true ~ 0.001-0.002 of
the evaluation footprint. Raw tree probabilities p_raw therefore saturate near 1.0
for any likelihood ratio LR > 2. When expected_credit(p) integrates over the 29-cell
2D triangular kernel (sum of weights = 9.934 vs 3.000 along a 1-px line), uncalibrated
p_raw gives diffuse 2D blobs a 3.31x area integration bonus over sharp 1-px fault
traces. Exact Bayes prior-shift calibration:

    LR(x)    = ((1 - pi_train) / pi_train) * (p_raw(x) / (1 - p_raw(x)))
    p_cal(x) = (pi_true * LR(x)) / (pi_true * LR(x) + (1 - pi_true))

restores the true posterior probability scale before 300 m kernel integration and
submodular marginal-coverage selection.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
from gems57.emit import expected_credit  # noqa: E402
from gems57.metric import OFF_DX, OFF_DY, OFF_K  # noqa: E402


def bayes_calibrate(
    p_raw: np.ndarray,
    pi_train: float = 1.0 / 3.0,
    pi_true: float = 0.002,
    eps: float = 1e-6,
) -> np.ndarray:
    """Map oversampled classifier probability p_raw to prior-calibrated posterior p_cal."""
    p = np.clip(np.asarray(p_raw, dtype=np.float32), eps, 1.0 - eps)
    odds_train = np.float32((1.0 - pi_train) / max(pi_train, eps))
    odds_true = np.float32(pi_true / max(1.0 - pi_true, eps))
    scale = odds_train * odds_true
    num = scale * p
    return (num / (num + (1.0 - p))).astype(np.float32)


def emit_dots_standard(
    p: np.ndarray,
    allowed: np.ndarray,
    budget: int,
    min_sep: float = 2.8,
    candidate_factor: int = 10,
) -> np.ndarray:
    """Baseline greedy emitter (matches gems77.emit77.emit_dots)."""
    p = np.clip(np.asarray(p, np.float32), 0.0, 1.0)
    allowed = np.asarray(allowed, bool)
    if budget <= 0 or not allowed.any():
        return np.zeros(p.shape, bool)
    ek = expected_credit(p)
    key = np.where(allowed, ek, np.float32(-np.inf))
    flat = key.ravel()
    n_cand = int(min(flat.size, max(budget * candidate_factor, budget + 1000)))
    cand = np.argpartition(flat, -n_cand)[-n_cand:]
    cand = cand[np.argsort(-flat[cand], kind="stable")]
    cand = cand[np.isfinite(flat[cand])]
    H, W = p.shape
    blocked = np.zeros((H, W), bool)
    out = np.zeros((H, W), bool)
    offs = [
        (dy, dx)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if (dy * dy + dx * dx) ** 0.5 < min_sep
    ]
    n = 0
    for fidx in cand:
        if n >= budget:
            break
        y, x = divmod(int(fidx), W)
        if blocked[y, x] or not allowed[y, x]:
            continue
        out[y, x] = True
        n += 1
        for dy, dx in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                blocked[yy, xx] = True
    return out


def emit_dots_submodular_bayes(
    p_raw: np.ndarray,
    allowed: np.ndarray,
    budget: int,
    min_sep: float = 2.8,
    pi_train: float = 1.0 / 3.0,
    pi_true: float = 0.002,
    candidate_factor: int = 16,
) -> np.ndarray:
    """Bayes-calibrated submodular 300 m kernel marginal-coverage dot emitter.

    Uses lazy submodular maximization (Minoux 1978):
    1. Calibrate p_cal = bayes_calibrate(p_raw, pi_train, pi_true).
    2. Compute initial upper bound U(x) = expected_credit(p_cal)(x).
    3. When considering candidate x: if a nearby dot within 6 px was accepted since
       U(x) was computed, recompute exact marginal coverage gain
           dT(x | M) = sum_{|y-x|<3} p_cal(y) * max(0, k(|y-x|) - M(y))
       and defer x if dT(x | M) falls below the current round threshold.
    """
    allowed = np.asarray(allowed, bool)
    if budget <= 0 or not allowed.any():
        return np.zeros(p_raw.shape, bool)
    p_cal = bayes_calibrate(p_raw, pi_train=pi_train, pi_true=pi_true)
    p_cal = np.where(allowed, p_cal, np.float32(0.0))
    ek = expected_credit(p_cal)
    key = np.where(allowed, ek, np.float32(-np.inf))
    flat = key.ravel()
    n_cand = int(min(flat.size, max(budget * candidate_factor, budget + 2000)))
    cand = np.argpartition(flat, -n_cand)[-n_cand:]
    cand = cand[np.argsort(-flat[cand], kind="stable")]
    cand = cand[np.isfinite(flat[cand]) & (flat[cand] > 0)]
    if cand.size == 0:
        return np.zeros(p_raw.shape, bool)

    H, W = p_cal.shape
    # Pad p_cal and cover map M by 3 px so 7x7 kernel lookups need no boundary checks
    PAD = 3
    p_pad = np.pad(p_cal, PAD, mode="constant", constant_values=0.0)
    M_pad = np.zeros((H + 2 * PAD, W + 2 * PAD), dtype=np.float32)
    dirty = np.zeros((H, W), dtype=bool)
    blocked = np.zeros((H, W), dtype=bool)
    out = np.zeros((H, W), dtype=bool)

    block_offs = [
        (dy, dx)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if (dy * dy + dx * dx) ** 0.5 < min_sep
    ]
    dirty_offs = [
        (dy, dx)
        for dy in range(-5, 6)
        for dx in range(-5, 6)
        if (dy * dy + dx * dx) ** 0.5 < 5.6
    ]
    k_list = list(zip(OFF_DY.tolist(), OFF_DX.tolist(), OFF_K.astype(np.float32).tolist()))

    scores = flat[cand].copy()
    ys_all, xs_all = np.divmod(cand, W)

    n = 0
    # Two-pass lazy submodular selection: Pass 1 accepts clean or high-gain candidates;
    # Pass 2 re-sorts deferred candidates by their updated exact marginal gain.
    deferred_idx = []
    deferred_gain = []
    thresh_ratio = np.float32(0.82)

    for i in range(cand.size):
        if n >= budget:
            break
        y = int(ys_all[i])
        x = int(xs_all[i])
        if blocked[y, x]:
            continue
        ub = scores[i]
        if dirty[y, x]:
            yp, xp = y + PAD, x + PAD
            gain = np.float32(0.0)
            for dy, dx, kw in k_list:
                rem = kw - M_pad[yp + dy, xp + dx]
                if rem > 0.0:
                    gain += p_pad[yp + dy, xp + dx] * rem
            if gain < ub * thresh_ratio:
                deferred_idx.append((y, x))
                deferred_gain.append(float(gain))
                continue
        out[y, x] = True
        n += 1
        yp, xp = y + PAD, x + PAD
        for dy, dx, kw in k_list:
            if kw > M_pad[yp + dy, xp + dx]:
                M_pad[yp + dy, xp + dx] = kw
        for dy, dx in block_offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                blocked[yy, xx] = True
        for dy, dx in dirty_offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                dirty[yy, xx] = True

    if n < budget and deferred_idx:
        order = np.argsort(-np.asarray(deferred_gain, dtype=np.float32), kind="stable")
        for idx in order:
            if n >= budget:
                break
            y, x = deferred_idx[int(idx)]
            if blocked[y, x]:
                continue
            out[y, x] = True
            n += 1
            for dy, dx in block_offs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    blocked[yy, xx] = True
    return out


def emit_dots_unique_stratified(
    p_final: np.ndarray,
    p_phys: np.ndarray,
    allowed: np.ndarray,
    active_halos: np.ndarray,
    budget: int = 37_654,
    b1_struct: int = 23_345,
    overlap_cap_frac: float = 0.655,
    min_sep: float = 2.8,
) -> np.ndarray:
    """Two-pool stratified submodular emitter with explicit per-raster overlap cap (<= 0.655).

    Pool 1 (b1_struct = 62% of budget): Bayes-calibrated submodular 300 m kernel coverage on p_final.
    Pool 2 (remaining 38% of budget): Bayes-calibrated expected credit on 0.70*p_phys + 0.30*p_final
    subject to |D within 3 px of R| <= overlap_cap_frac * budget for every informative raster R.
    """
    from scipy import ndimage as ndi

    H, W = p_final.shape
    d1 = emit_dots_submodular_bayes(p_final, allowed, budget=b1_struct, min_sep=min_sep, candidate_factor=16)
    b2 = budget - int(d1.sum())
    if b2 <= 0:
        return d1

    cap = int(overlap_cap_frac * budget)
    d1_ys, d1_xs = np.nonzero(d1)
    counts = active_halos[:, d1_ys, d1_xs].sum(axis=1).astype(np.int32)

    edt_d1 = ndi.distance_transform_edt(~d1) >= min_sep
    a2 = allowed & edt_d1
    ek2 = expected_credit(bayes_calibrate(np.float32(0.70) * p_phys + np.float32(0.30) * p_final))
    ys2, xs2 = np.nonzero(a2)
    order = np.argsort(-ek2[ys2, xs2], kind="stable")
    ys2 = ys2[order]
    xs2 = xs2[order]

    blocked = ~a2
    d2 = np.zeros((H, W), dtype=bool)
    block_offs = [
        (dy, dx)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if (dy * dy + dx * dx) ** 0.5 < min_sep
    ]
    n2 = 0
    for i in range(ys2.size):
        if n2 >= b2:
            break
        y = int(ys2[i])
        x = int(xs2[i])
        if blocked[y, x]:
            continue
        col_hit = active_halos[:, y, x]
        if np.any(counts[col_hit] >= cap):
            continue
        d2[y, x] = True
        counts[col_hit] += 1
        n2 += 1
        for dy, dx in block_offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                blocked[yy, xx] = True
    return d1 | d2

