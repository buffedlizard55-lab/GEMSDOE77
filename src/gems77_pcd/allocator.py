"""300 m triangular-kernel lazy-greedy submodular dot allocator with optional
bitwise per-raster 3-px sparse-overlap capacity guard for Lane H83-PCD.

Maximizes the expected Distance-Weighted Tversky Index (DTI) numerator coverage
under the exact competition 300 m (3 px at 100 m resolution) triangular kernel
`K(d) = max(0, 1 - d / 3)` via Minoux (1978) accelerated lazy greedy selection,
while optionally enforcing a hard per-raster 3-px forward-overlap ceiling
(`max_overlap_fraction <= 0.64 < 0.70`) across up to 64 prior sparse dot-field
rasters packed into a single `uint64` bitmask per pixel.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata

from src.gems77d.submodular import KERNEL_300M, allocate_submodular_300m


def build_sparse_halo_bitmask(
    allowed_mask: np.ndarray,
    halo_masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack up to 64 boolean 3-px halo masks into a `uint64` bitmask and normalized crowd map."""
    if len(halo_masks) > 64:
        raise ValueError(f"At most 64 sparse halo masks supported, got {len(halo_masks)}")
    h, w = allowed_mask.shape
    bits = np.zeros((h, w), dtype=np.uint64)
    crowd = np.zeros((h, w), dtype=np.float32)
    if not halo_masks:
        return bits, crowd

    for j, m in enumerate(halo_masks):
        in_halo = np.asarray(m, dtype=bool) & allowed_mask
        bits[in_halo] |= np.uint64(1) << np.uint64(j)
        crowd[in_halo] += 1.0
    crowd /= float(len(halo_masks))
    return bits, crowd


def allocate_submodular_300m_pcd(
    prob_surface: np.ndarray,
    allowed_mask: np.ndarray,
    target_n: int = 37654,
    support_quantile: float = 0.90,
    gamma: float = 2.0,
    min_spacing_px: float = 0.0,
    max_candidates: int = 450000,
    sparse_halo_bits: Optional[np.ndarray] = None,
    sparse_crowd_fraction: Optional[np.ndarray] = None,
    num_sparse_rasters: int = 0,
    max_overlap_fraction: float = 0.64,
    crowd_discount: float = 0.14,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Select up to `target_n` binary `{0.0, 1.0}` dots via 300 m submodular cover.

    When `sparse_halo_bits` is None, delegates directly to `allocate_submodular_300m`.
    When `sparse_halo_bits` is provided, enforces a strict per-raster 3-px overlap cap
    `floor(max_overlap_fraction * target_n)` across all `num_sparse_rasters` sparse
    reference rasters.
    """
    if sparse_halo_bits is None or num_sparse_rasters == 0:
        dots_bool = allocate_submodular_300m(
            surface_2d=prob_surface,
            allowed_2d=allowed_mask,
            target_n=target_n,
            support_quantile=support_quantile,
            gamma=gamma,
            min_sep_px=min_spacing_px,
            max_candidates=max_candidates,
        )
        out = dots_bool.astype(np.float32)
        return out, {
            "emitted_dots": int(dots_bool.sum()),
            "max_sparse_overlap_cap_fraction": 1.0,
            "max_realized_sparse_overlap_fraction": 0.0,
            "max_realized_sparse_overlap_dots": 0,
            "skipped_due_to_halo_cap": 0,
        }

    h, w = prob_surface.shape
    allow = np.asarray(allowed_mask, dtype=bool)
    vals = np.asarray(prob_surface[allow], dtype=np.float64)
    if target_n <= 0 or vals.size == 0:
        return np.zeros((h, w), dtype=np.float32), {
            "emitted_dots": 0,
            "max_sparse_overlap_cap_fraction": float(max_overlap_fraction),
            "max_realized_sparse_overlap_fraction": 0.0,
            "max_realized_sparse_overlap_dots": 0,
            "skipped_due_to_halo_cap": 0,
        }
    target_n = min(target_n, int(vals.size))

    ranks = (rankdata(vals, method="average") / float(vals.size)).astype(np.float32)
    pos = np.clip(
        (ranks - support_quantile) / max(1.0 - support_quantile, 1e-6), 0.0, 1.0
    )
    weights_allow = (pos ** gamma + 1e-5 * ranks).astype(np.float32)
    if sparse_crowd_fraction is not None and crowd_discount > 0.0:
        crowd_allow = np.asarray(sparse_crowd_fraction[allow], dtype=np.float32)
        weights_allow = weights_allow * (1.0 - float(crowd_discount) * crowd_allow)

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
            if (dy * dy + dx * dx) ** 0.5 < min_spacing_px
        )
        if min_spacing_px > 0.0
        else ()
    )
    off_k = tuple((dy * wp + dx, k) for dy, dx, k in KERNEL_300M)

    max_allowed_per_raster = int(np.floor(float(max_overlap_fraction) * float(target_n)))
    overlap_counts = np.zeros(num_sparse_rasters, dtype=np.int32)
    bit_values = np.array(
        [np.uint64(1) << np.uint64(j) for j in range(num_sparse_rasters)],
        dtype=np.uint64,
    )
    saturated_mask = np.uint64(0)
    skipped_halo = 0

    dots = np.zeros((h, w), dtype=np.float32)
    picked = 0
    pop = heapq.heappop
    push = heapq.heappush

    while picked < target_n and heap:
        neg_ub, idx = pop(heap)
        yp, xp = divmod(idx, wp)
        if blocked[yp, xp]:
            continue
        y, x = yp - pad, xp - pad
        if dots[y, x] > 0.5:
            continue
        c_bits = sparse_halo_bits[y, x]
        if (c_bits & saturated_mask) != np.uint64(0):
            skipped_halo += 1
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
            dots[y, x] = 1.0
            picked += 1
            if c_bits != np.uint64(0):
                hit = (c_bits & bit_values) != np.uint64(0)
                overlap_counts[hit] += 1
                newly_sat = hit & (overlap_counts >= max_allowed_per_raster)
                if np.any(newly_sat):
                    for j in np.where(newly_sat)[0]:
                        saturated_mask |= bit_values[j]
            for off, k in off_k:
                j = idx + off
                if k > cov_list[j]:
                    cov_list[j] = k
            for dy, dx in sep_offs:
                blocked[yp + dy, xp + dx] = True
        elif gain > 0.0:
            push(heap, (-gain, idx))

    max_realized = (
        float(overlap_counts.max()) / float(max(picked, 1))
        if num_sparse_rasters > 0
        else 0.0
    )
    meta = {
        "emitted_dots": int(picked),
        "max_sparse_overlap_cap_fraction": float(max_overlap_fraction),
        "max_realized_sparse_overlap_fraction": round(max_realized, 6),
        "max_realized_sparse_overlap_dots": (
            int(overlap_counts.max()) if num_sparse_rasters > 0 else 0
        ),
        "skipped_due_to_halo_cap": int(skipped_halo),
    }
    return dots, meta
