"""Metric-exact Flank-Zeroed, Crest-Centered 300 m Submodular Expected-DTI Emitter (`GEMSDOE77`)."""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
from gems57 import metric  # noqa: E402


def peak_nms_emit(
    prob: np.ndarray,
    allowed_mask: np.ndarray,
    n_dots: int = 37654,
    nms_size: int = 7,
) -> np.ndarray:
    """Deterministic 7x7 local-maximum non-maximum suppression control emitter."""
    H, W = prob.shape
    p = np.where(allowed_mask & np.isfinite(prob), prob, -1.0).astype(np.float64)
    tie = np.arange(p.size, dtype=np.float64).reshape(H, W) * 1e-13
    p_tb = np.where(allowed_mask, p + tie, -1.0)

    local_max = maximum_filter(p_tb, size=nms_size, mode="constant", cval=-2.0)
    peaks = allowed_mask & (p_tb >= local_max) & (p > 0.0)
    peak_idx = np.flatnonzero(peaks)

    out = np.zeros((H, W), dtype=np.float32)
    if peak_idx.size >= n_dots:
        top_sub = np.argpartition(p_tb.ravel()[peak_idx], -n_dots)[-n_dots:]
        out.ravel()[peak_idx[top_sub]] = 1.0
        return out

    out.ravel()[peak_idx] = 1.0
    rem = n_dots - int(peak_idx.size)
    rem_idx = np.flatnonzero(allowed_mask & (~peaks))
    top_rem = np.argpartition(p_tb.ravel()[rem_idx], -rem)[-rem:]
    out.ravel()[rem_idx[top_rem]] = 1.0
    return out


def flank_zeroed_crest_submodular_sequence(
    prob: np.ndarray,
    crest_2d: np.ndarray,
    allowed_mask: np.ndarray,
    max_dots: int = 41500,
    support_quantile: float = 0.88,
    gamma: float = 1.85,
    crest_alpha: float = 0.65,
    fp_penalty_weight: float = 0.055,
    candidate_pool: int = 440000,
) -> np.ndarray:
    """Return ordered 1D flat indices ``seq[:max_dots]`` of exact Minoux lazy-greedy 300 m cover.

    Because greedy submodular maximization is strictly nested, ``seq[:B]`` gives the exact
    greedy ``B``-dot allocation for any ``B <= max_dots`` in a single pass!
    """
    H, W = prob.shape
    allow = np.asarray(allowed_mask, dtype=bool)
    vals = prob[allow]
    if vals.size < max_dots:
        raise ValueError(f"Allowed domain ({vals.size}) < max_dots ({max_dots})")

    ranks = (rankdata(vals, method="average") / float(vals.size)).astype(np.float32)
    pos = np.clip((ranks - support_quantile) / max(1.0 - support_quantile, 1e-6), 0.0, 1.0)
    crest_allow = np.clip(crest_2d[allow].astype(np.float32), 0.0, 1.0)
    w_allow = (pos ** gamma) * (1.0 + crest_alpha * crest_allow) + 1e-6 * ranks
    del ranks, pos, crest_allow

    pad = 3
    q_pad = np.zeros((H + 2 * pad, W + 2 * pad), dtype=np.float32)
    ay, ax = np.nonzero(allow)
    # CRITICAL H77-4 INVARIANT: ONLY allowed off-collar cells receive positive target weight!
    q_pad[ay + pad, ax + pad] = w_allow
    del ay, ax, w_allow
    cov_pad = np.zeros_like(q_pad)

    dy = metric.OFF_DY.tolist()
    dx = metric.OFF_DX.tolist()
    kw = [float(k) for k in metric.OFF_K.tolist()]
    n_off = len(kw)

    init_tp = np.zeros((H, W), dtype=np.float32)
    q_self = np.zeros((H, W), dtype=np.float32)
    for oy, ox, k_val in zip(dy, dx, kw):
        sub = q_pad[pad + oy : pad + oy + H, pad + ox : pad + ox + W]
        init_tp += k_val * sub
        np.maximum(q_self, k_val * sub, out=q_self)

    pen_arr = (fp_penalty_weight * (1.0 - np.minimum(q_self, 1.0))).astype(np.float32)
    del q_self
    u0 = init_tp - pen_arr

    valid_idx = np.flatnonzero(allow & (init_tp > 0))
    del init_tp
    if valid_idx.size > candidate_pool:
        u_sub = u0.ravel()[valid_idx]
        keep = np.argpartition(u_sub, -candidate_pool)[-candidate_pool:]
        valid_idx = valid_idx[keep]

    ys = (valid_idx // W + pad).tolist()
    xs = (valid_idx % W + pad).tolist()
    pens = pen_arr.ravel()[valid_idx].tolist()
    u_vals = u0.ravel()[valid_idx].tolist()
    del pen_arr, u0

    heap: list[tuple[float, int]] = [(-u_vals[i], i) for i in range(len(u_vals))]
    heapq.heapify(heap)

    seq = np.empty(max_dots, dtype=np.int64)
    selected = 0

    while selected < max_dots and heap:
        neg_ub, i = heapq.heappop(heap)
        cy = ys[i]
        cx = xs[i]
        gain = 0.0
        for m in range(n_off):
            ny = cy + dy[m]
            nx = cx + dx[m]
            k_m = kw[m]
            cur_c = float(cov_pad[ny, nx])
            if k_m > cur_c:
                gain += float(q_pad[ny, nx]) * (k_m - cur_c)
        exact_u = gain - pens[i]

        if not heap or (-exact_u <= heap[0][0] + 1e-12):
            seq[selected] = (cy - pad) * W + (cx - pad)
            for m in range(n_off):
                ny = cy + dy[m]
                nx = cx + dx[m]
                k_m = kw[m]
                if k_m > cov_pad[ny, nx]:
                    cov_pad[ny, nx] = k_m
            selected += 1
        else:
            heapq.heappush(heap, (-exact_u, i))

    if selected < max_dots:
        used = np.zeros(H * W, dtype=bool)
        used[seq[:selected]] = True
        rem_idx = np.flatnonzero(allow.ravel() & (~used))
        rem = max_dots - selected
        top_rem = np.argpartition(prob.ravel()[rem_idx], -rem)[-rem:]
        top_rem_sorted = top_rem[np.argsort(-prob.ravel()[rem_idx[top_rem]])]
        seq[selected:] = rem_idx[top_rem_sorted]

    return seq


def flank_zeroed_crest_submodular_emit(
    prob: np.ndarray,
    crest_2d: np.ndarray,
    allowed_mask: np.ndarray,
    n_dots: int = 37654,
    **kwargs: float,
) -> np.ndarray:
    """Emit binary float32 array in [0, 1] with exact ``n_dots`` ones via ``flank_zeroed_crest_submodular_sequence``."""
    seq = flank_zeroed_crest_submodular_sequence(prob, crest_2d, allowed_mask, max_dots=n_dots, **kwargs)
    out = np.zeros(prob.shape, dtype=np.float32)
    out.ravel()[seq] = 1.0
    return out
