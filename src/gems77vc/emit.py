"""Uniqueness-Constrained Flank-Zeroed 300 m Submodular Expected-DTI Emitter (`Lane H77V`).

Optimizes the exact 300 m (`3 px`) triangular-kernel submodular cover objective on the
out-of-fold H77V posterior surface + hydrothermal-boosted crest field while optionally
enforcing a hard per-raster 3-px forward dot overlap cap (`<= 0.645 < 0.70`) against
any set of informative registry/sibling 3-px halos via a 64-bit / arbitrary-precision
bitmask constraint inside the Minoux lazy-greedy priority queue.
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
from gems57 import metric  # noqa: E402


def build_hydrothermal_crest_2d(
    crest_fp: np.ndarray,
    vent_pot_fp: np.ndarray,
    upflow_scarp_fp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    """Combine structural ridge crest prominence with H77V hydrothermal upflow-scarp coincidence."""
    n_fp = int(crest_fp.size)
    scarp_rank = (rankdata(upflow_scarp_fp, method="average") / float(n_fp)).astype(np.float32)
    crest_v_fp = np.clip(
        0.74 * np.clip(crest_fp, 0.0, 1.0)
        + 0.16 * np.clip(scarp_rank, 0.0, 1.0)
        + 0.10 * np.clip(vent_pot_fp, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    out = np.zeros(fp.shape, dtype=np.float32)
    out[fp] = crest_v_fp
    return out


def _build_submodular_weights_and_candidates(
    prob: np.ndarray,
    crest_2d: np.ndarray,
    allowed_mask: np.ndarray,
    support_quantile: float = 0.88,
    gamma: float = 1.85,
    crest_alpha: float = 0.68,
    fp_penalty_weight: float = 0.055,
    candidate_pool: int = 240000,
    vent_reserve_pool: int = 180000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(q_pad, pen_arr, u0, valid_idx)`` with a 180k hydrothermal-crest reserve pool."""
    H, W = prob.shape
    allow = np.asarray(allowed_mask, dtype=bool)
    vals = prob[allow]
    ranks = (rankdata(vals, method="average") / float(vals.size)).astype(np.float32)
    pos = np.clip((ranks - support_quantile) / max(1.0 - support_quantile, 1e-6), 0.0, 1.0)
    crest_allow = np.clip(crest_2d[allow].astype(np.float32), 0.0, 1.0)
    # Small positive hydrothermal-crest floor ensures reserve candidates have positive 300 m kernel gain
    w_allow = (pos ** gamma) * (1.0 + crest_alpha * crest_allow) + (0.015 * crest_allow + 0.002) * ranks
    del ranks, pos, crest_allow

    pad = 3
    q_pad = np.zeros((H + 2 * pad, W + 2 * pad), dtype=np.float32)
    ay, ax = np.nonzero(allow)
    q_pad[ay + pad, ax + pad] = w_allow
    del ay, ax, w_allow

    dy = metric.OFF_DY.tolist()
    dx = metric.OFF_DX.tolist()
    kw = [float(k) for k in metric.OFF_K.tolist()]

    init_tp = np.zeros((H, W), dtype=np.float32)
    q_self = np.zeros((H, W), dtype=np.float32)
    for oy, ox, k_val in zip(dy, dx, kw):
        sub = q_pad[pad + oy : pad + oy + H, pad + ox : pad + ox + W]
        init_tp += k_val * sub
        np.maximum(q_self, k_val * sub, out=q_self)

    pen_arr = (fp_penalty_weight * (1.0 - np.minimum(q_self, 1.0))).astype(np.float32)
    del q_self
    u0 = init_tp - pen_arr
    del init_tp

    allow_idx = np.flatnonzero(allow)
    u_allow = u0.ravel()[allow_idx]
    if allow_idx.size <= candidate_pool + vent_reserve_pool:
        valid_idx = allow_idx
    else:
        top_part = np.argpartition(u_allow, -candidate_pool)[-candidate_pool:]
        top_idx = allow_idx[top_part]
        in_top = np.zeros(H * W, dtype=bool)
        in_top[top_idx] = True
        rem_idx = np.flatnonzero(allow.ravel() & (~in_top))
        # Select diverse hydrothermal vent crest pixels across the rest of the footprint
        vent_score = (crest_2d.ravel()[rem_idx] + 0.35 * prob.ravel()[rem_idx]).astype(np.float32)
        res_part = np.argpartition(vent_score, -vent_reserve_pool)[-vent_reserve_pool:]
        res_idx = rem_idx[res_part]
        valid_idx = np.concatenate([top_idx, res_idx])
    return q_pad, pen_arr, u0, valid_idx


def get_submodular_candidate_indices(
    prob: np.ndarray,
    crest_2d: np.ndarray,
    allowed_mask: np.ndarray,
    support_quantile: float = 0.88,
    gamma: float = 1.85,
    crest_alpha: float = 0.68,
    fp_penalty_weight: float = 0.055,
    candidate_pool: int = 240000,
    vent_reserve_pool: int = 180000,
) -> np.ndarray:
    """Return the deterministic 1D flat `valid_idx` array of candidate pixels for submodular cover."""
    _, _, _, valid_idx = _build_submodular_weights_and_candidates(
        prob,
        crest_2d,
        allowed_mask,
        support_quantile=support_quantile,
        gamma=gamma,
        crest_alpha=crest_alpha,
        fp_penalty_weight=fp_penalty_weight,
        candidate_pool=candidate_pool,
        vent_reserve_pool=vent_reserve_pool,
    )
    return valid_idx


def flank_zeroed_unique_submodular_sequence(
    prob: np.ndarray,
    crest_2d: np.ndarray,
    allowed_mask: np.ndarray,
    max_dots: int = 41500,
    support_quantile: float = 0.88,
    gamma: float = 1.85,
    crest_alpha: float = 0.68,
    fp_penalty_weight: float = 0.055,
    candidate_pool: int = 240000,
    vent_reserve_pool: int = 180000,
    halo_flat_masks: list[np.ndarray] | None = None,
    cand_halo_masks: list[np.ndarray] | None = None,
    max_per_halo: int | None = None,
) -> np.ndarray:
    """Return ordered 1D flat indices ``seq[:max_dots]`` of exact Minoux lazy-greedy 300 m cover."""
    H, W = prob.shape
    allow = np.asarray(allowed_mask, dtype=bool)
    vals = prob[allow]
    if vals.size < max_dots:
        raise ValueError(f"Allowed domain ({vals.size}) < max_dots ({max_dots})")

    pad = 3
    q_pad, pen_arr, u0, valid_idx = _build_submodular_weights_and_candidates(
        prob,
        crest_2d,
        allow,
        support_quantile=support_quantile,
        gamma=gamma,
        crest_alpha=crest_alpha,
        fp_penalty_weight=fp_penalty_weight,
        candidate_pool=candidate_pool,
        vent_reserve_pool=vent_reserve_pool,
    )
    cov_pad = np.zeros_like(q_pad)

    dy = metric.OFF_DY.tolist()
    dx = metric.OFF_DX.tolist()
    kw = [float(k) for k in metric.OFF_K.tolist()]
    n_off = len(kw)

    n_cand = int(valid_idx.size)
    masks_on_cand: list[np.ndarray] | None = None
    if cand_halo_masks:
        masks_on_cand = cand_halo_masks
    elif halo_flat_masks:
        masks_on_cand = [h_mask[valid_idx] for h_mask in halo_flat_masks]

    use_halo_cap = bool(masks_on_cand) and (max_per_halo is not None) and (max_per_halo > 0)
    cand_bits: list[int] = []
    halo_counts: list[int] = []
    if use_halo_cap and masks_on_cand is not None:
        m_halos = len(masks_on_cand)
        halo_counts = [0] * m_halos
        n_words = (m_halos + 61) // 62
        words = [np.zeros(n_cand, dtype=np.uint64) for _ in range(n_words)]
        for m_i, in_h in enumerate(masks_on_cand):
            w_idx = m_i // 62
            bit_idx = m_i % 62
            words[w_idx] |= in_h.astype(np.uint64) << np.uint64(bit_idx)
        if n_words == 1:
            cand_bits = words[0].tolist()
        else:
            w_lists = [w.tolist() for w in words]
            cand_bits = [0] * n_cand
            for w_idx, wl in enumerate(w_lists):
                shift = 62 * w_idx
                if w_idx == 0:
                    cand_bits = [int(v) for v in wl]
                else:
                    cand_bits = [cb | (int(v) << shift) for cb, v in zip(cand_bits, wl)]

    ys = (valid_idx // W + pad).tolist()
    xs = (valid_idx % W + pad).tolist()
    pens = pen_arr.ravel()[valid_idx].tolist()
    u_vals = u0.ravel()[valid_idx].tolist()
    del pen_arr, u0

    heap: list[tuple[float, int]] = [(-u_vals[i], i) for i in range(len(u_vals))]
    heapq.heapify(heap)

    seq = np.empty(max_dots, dtype=np.int64)
    selected = 0
    saturated_bits = 0
    cap_val = int(max_per_halo) if max_per_halo is not None else max_dots

    while selected < max_dots and heap:
        neg_ub, i = heapq.heappop(heap)
        if use_halo_cap:
            cb = cand_bits[i]
            if cb & saturated_bits:
                continue
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
            if use_halo_cap and cb:
                tmp = cb
                while tmp:
                    lsb = tmp & -tmp
                    m_idx = lsb.bit_length() - 1
                    cnt = halo_counts[m_idx] + 1
                    halo_counts[m_idx] = cnt
                    if cnt >= cap_val:
                        saturated_bits |= lsb
                    tmp ^= lsb
            selected += 1
        else:
            heapq.heappush(heap, (-exact_u, i))

    if selected < max_dots:
        used = np.zeros(H * W, dtype=bool)
        used[seq[:selected]] = True
        rem_mask = allow.ravel() & (~used)
        if use_halo_cap and halo_flat_masks is not None:
            for m_i, h_mask in enumerate(halo_flat_masks):
                if halo_counts[m_i] >= cap_val:
                    rem_mask &= ~h_mask
        rem_idx = np.flatnonzero(rem_mask)
        rem = max_dots - selected
        top_rem = np.argpartition(prob.ravel()[rem_idx], -rem)[-rem:]
        top_rem_sorted = top_rem[np.argsort(-prob.ravel()[rem_idx[top_rem]])]
        seq[selected:] = rem_idx[top_rem_sorted]

    return seq


def flank_zeroed_unique_submodular_emit(
    prob: np.ndarray,
    crest_2d: np.ndarray,
    allowed_mask: np.ndarray,
    n_dots: int = 36250,
    halo_flat_masks: list[np.ndarray] | None = None,
    cand_halo_masks: list[np.ndarray] | None = None,
    max_overlap_frac: float = 0.645,
    **kwargs: float,
) -> np.ndarray:
    """Emit binary {0, 1} float32 raster of ``n_dots`` dots via uniqueness-constrained 300 m cover."""
    has_masks = bool(halo_flat_masks) or bool(cand_halo_masks)
    max_per_halo = int(np.floor(max_overlap_frac * n_dots)) if has_masks else None
    seq = flank_zeroed_unique_submodular_sequence(
        prob,
        crest_2d,
        allowed_mask,
        max_dots=n_dots,
        halo_flat_masks=halo_flat_masks,
        cand_halo_masks=cand_halo_masks,
        max_per_halo=max_per_halo,
        **kwargs,
    )
    out = np.zeros(prob.shape, dtype=np.float32)
    out.ravel()[seq[:n_dots]] = 1.0
    return out
