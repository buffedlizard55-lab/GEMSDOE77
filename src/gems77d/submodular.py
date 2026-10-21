"""Exact unblurred 300 m triangular-kernel submodular dot allocator (Minoux 1978 lazy greedy).

Optimizes expected distance-weighted Tversky coverage directly over the official competition
300 m (3.0 px) triangular kernel K(dy, dx) = max(0, 1 - hypot(dy, dx) / 3.0), while optionally
enforcing minimum Euclidean dot spacing and per-raster registry overlap budgets if ever required.
"""
from __future__ import annotations

import heapq
import numpy as np
from scipy.stats import rankdata

KERNEL_300M: tuple[tuple[int, int, float], ...] = tuple(
    (dy, dx, float(1.0 - np.hypot(dy, dx) / 3.0))
    for dy in range(-2, 3)
    for dx in range(-2, 3)
    if np.hypot(dy, dx) < 3.0
)


def allocate_submodular_300m(
    surface_2d: np.ndarray,
    allowed_2d: np.ndarray,
    target_n: int = 37654,
    support_quantile: float = 0.90,
    gamma: float = 2.0,
    min_sep_px: float = 0.0,
    max_candidates: int = 450000,
) -> np.ndarray:
    """Select binary dots D subset allowed_2d maximizing sum_g w(g) * max_{x in D} K(||x - g||_2)."""
    h, w = surface_2d.shape
    allow = np.asarray(allowed_2d, dtype=bool)
    vals = np.asarray(surface_2d[allow], dtype=np.float64)
    if target_n <= 0 or vals.size == 0:
        return np.zeros((h, w), dtype=bool)
    target_n = min(target_n, int(vals.size))

    ranks = (rankdata(vals, method="average") / float(vals.size)).astype(np.float32)
    pos = np.clip((ranks - support_quantile) / max(1.0 - support_quantile, 1e-6), 0.0, 1.0)
    weights_allow = (pos ** gamma + 1e-5 * ranks).astype(np.float32)

    pad = 3
    hp, wp = h + 2 * pad, w + 2 * pad
    w_pad = np.zeros((hp, wp), dtype=np.float32)
    ay, ax = np.nonzero(allow)
    w_pad[ay + pad, ax + pad] = weights_allow

    g0 = np.zeros((h, w), dtype=np.float32)
    for dy, dx, k in KERNEL_300M:
        g0 += k * w_pad[pad + dy : pad + dy + h, pad + dx : pad + dx + w]

    cand_scores = g0[allow]
    if cand_scores.size > max_candidates:
        part = np.argpartition(cand_scores, -max_candidates)[-max_candidates:]
        part = part[np.argsort(cand_scores[part], kind="mergesort")[::-1]]
    else:
        part = np.argsort(cand_scores, kind="mergesort")[::-1]

    cy = (ay[part] + pad).astype(np.int32)
    cx = (ax[part] + pad).astype(np.int32)
    flat_idx = cy * wp + cx
    init_gains = cand_scores[part]

    heap = [(-float(init_gains[i]), int(flat_idx[i])) for i in range(flat_idx.size)]
    heapq.heapify(heap)

    w_list = w_pad.ravel().tolist()
    cov_list = [0.0] * (hp * wp)
    blocked = np.zeros((hp, wp), dtype=bool)
    sep_offs = (
        tuple(
            (dy, dx)
            for dy in range(-3, 4)
            for dx in range(-3, 4)
            if (dy * dy + dx * dx) ** 0.5 < min_sep_px
        )
        if min_sep_px > 0.0
        else ()
    )
    off_k = tuple((dy * wp + dx, k) for dy, dx, k in KERNEL_300M)

    dots = np.zeros((h, w), dtype=bool)
    picked = 0
    pop = heapq.heappop
    push = heapq.heappush

    while picked < target_n and heap:
        neg_ub, idx = pop(heap)
        yp, xp = divmod(idx, wp)
        if blocked[yp, xp]:
            continue
        gain = 0.0
        for off, k in off_k:
            j = idx + off
            wj = w_list[j]
            if wj > 0.0:
                rem = k - cov_list[j]
                if rem > 0.0:
                    gain += wj * rem
        if not heap or gain >= -heap[0][0] - 1e-12:
            y, x = yp - pad, xp - pad
            if not dots[y, x]:
                dots[y, x] = True
                for off, k in off_k:
                    j = idx + off
                    if k > cov_list[j]:
                        cov_list[j] = k
                for dy, dx in sep_offs:
                    blocked[yp + dy, xp + dx] = True
                picked += 1
        elif gain > 0.0:
            push(heap, (-gain, idx))

    return dots
