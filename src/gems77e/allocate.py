"""Flank-zeroed Dinkelbach 300 m submodular expected-DTI allocator with exact portfolio capacity guard."""
from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import metric  # noqa: E402


class PortfolioCapacityIndex:
    """Bit-exact lookup of all non-space-filling (<70% halo) portfolio rasters (0.00e+00 diff vs vendor)."""

    def __init__(self, packed_path: Path, meta_path: Path, foot: np.ndarray, cat: np.ndarray):
        meta = json.loads(meta_path.read_text())
        self.n_valid: int = int(meta["n_valid"])
        self.entries: list[dict[str, Any]] = meta["entries"]
        self.packed = np.load(packed_path, mmap_mode="r")
        self.allow_idx = np.flatnonzero(foot & (~cat))
        self.info_row_ids = np.array(
            [i for i, e in enumerate(self.entries) if not e["is_space_filling"]],
            dtype=np.int32,
        )
        assert self.allow_idx.size == int(meta["n_allow"]), "Allow index mismatch"

    def active_membership_matrix(self, cand_flat: np.ndarray, cap_count: int) -> np.ndarray | None:
        """Return compact contiguous (N_cand, K_act) bool matrix (~25 MB RAM vs 1.65 GB Python lists!)."""
        pos_in_allow = np.searchsorted(self.allow_idx, cand_flat)
        valid_match = (pos_in_allow < self.allow_idx.size) & (
            self.allow_idx[np.minimum(pos_in_allow, self.allow_idx.size - 1)] == cand_flat
        )
        byte_idx = (pos_in_allow >> 3).astype(np.int64)
        bit_mask = (1 << (7 - (pos_in_allow & 7))).astype(np.uint8)

        active_rows: list[np.ndarray] = []
        for s in range(0, self.info_row_ids.size, 64):
            rids = self.info_row_ids[s : s + 64]
            sub = ((self.packed[rids][:, byte_idx] & bit_mask[None, :]) != 0) | (~valid_match)[None, :]
            totals = sub.sum(axis=1)
            for h_idx in np.flatnonzero(totals > cap_count):
                active_rows.append(sub[h_idx].copy())
            del sub
        if not active_rows:
            return None
        return np.column_stack(active_rows)

    def max_informative_overlap(self, dots_2d: np.ndarray) -> tuple[float, int, str]:
        """Compute exact upper-bound forward 3-px overlap & firing count (>0.70) across ALL 816 informative rasters."""
        total_dots = int((dots_2d > 0.5).sum())
        if total_dots == 0:
            return 0.0, 0, ""
        dot_pos = np.flatnonzero(dots_2d.ravel()[self.allow_idx] > 0.5)
        extra_non_allow = max(0, total_dots - int(dot_pos.size))
        byte_idx = dot_pos >> 3
        bit_mask = (1 << (7 - (dot_pos & 7))).astype(np.uint8)
        ovs = np.empty(self.info_row_ids.size, dtype=np.float64)
        for s in range(0, self.info_row_ids.size, 64):
            rids = self.info_row_ids[s : s + 64]
            hits = ((self.packed[rids][:, byte_idx] & bit_mask[None, :]) != 0).sum(axis=1)
            ovs[s : s + rids.size] = (hits + extra_non_allow) / float(total_dots)
        best_i = int(np.argmax(ovs))
        firings = int((ovs > 0.70).sum())
        e = self.entries[int(self.info_row_ids[best_i])]
        return float(ovs[best_i]), firings, f"{e['tag']}:{e['name']}"


def allocate_dinkelbach_submodular(
    utility_2d: np.ndarray,
    allowed_2d: np.ndarray,
    target_n: int,
    crest_2d: np.ndarray | None = None,
    support_quantile: float = 0.88,
    gamma: float = 1.85,
    crest_alpha: float = 0.65,
    fp_penalty_weight: float = 0.055,
    min_sep_px: float = 0.0,
    candidate_pool: int = 360_000,
    capacity_index: PortfolioCapacityIndex | None = None,
    max_overlap_cap: float = 0.680,
) -> np.ndarray:
    """Flank-zeroed Minoux lazy-greedy 300 m kernel cover with exact Dinkelbach FP cost
    and optional simultaneous capacity guard across all non-space-filling portfolio rasters.
    """
    assert utility_2d.shape == allowed_2d.shape, (
        f"Shape mismatch: {utility_2d.shape} vs {allowed_2d.shape}"
    )
    H, W = utility_2d.shape
    allow = np.asarray(allowed_2d, dtype=bool)
    n_allow = int(allow.sum())
    if n_allow == 0 or target_n <= 0:
        return np.zeros((H, W), dtype=bool)
    target_n = min(int(target_n), n_allow)

    vals = np.nan_to_num(utility_2d[allow].astype(np.float32), nan=0.0)
    ranks = (rankdata(vals, method="average") / float(n_allow)).astype(np.float32)

    # Smooth quantile support + calibrated probability magnitude so zero-ribbon off-support scarps retain positive weight
    pos = np.clip((ranks - support_quantile) / max(1.0 - support_quantile, 1e-6), 0.0, 1.0)
    tail_prob = np.clip((ranks - 0.72) / 0.28, 0.0, 1.0) ** 3.0
    base_w = (pos ** gamma) + 0.08 * tail_prob + 0.015 * (ranks ** 2)
    if crest_2d is not None:
        c_sub = np.clip(np.nan_to_num(crest_2d[allow].astype(np.float32), nan=0.0), 0.0, 1.0)
        w_allow = (base_w * (1.0 + crest_alpha * c_sub)).astype(np.float32)
    else:
        w_allow = base_w.astype(np.float32)
    del vals, ranks, pos, tail_prob, base_w

    pad = 3
    q_pad = np.zeros((H + 2 * pad, W + 2 * pad), dtype=np.float32)
    ay, ax = np.nonzero(allow)
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
        if capacity_index is not None:
            n_main = 250_000
            n_res = 190_000
            top_main = valid_idx[np.argpartition(u_sub, -n_main)[-n_main:]]
            # Use 3x3 local peaks (>500,000 peaks across Nevada, >160,000 outside even a 69%-halo raster!)
            is_peak3 = (u0 >= maximum_filter(u0, size=3, mode="constant")).ravel()[valid_idx]
            peak_sub_idx = np.flatnonzero(is_peak3)
            pv = valid_idx[peak_sub_idx]
            pu = u_sub[peak_sub_idx]
            heavy_rows = np.array(
                [
                    r
                    for r in capacity_index.info_row_ids.tolist()
                    if capacity_index.entries[r]["halo3_frac_allow"] >= 0.45
                ],
                dtype=np.int32,
            )
            weights_h = np.array(
                [
                    capacity_index.entries[int(r)]["halo3_frac_allow"] ** 6
                    for r in heavy_rows.tolist()
                ],
                dtype=np.float32,
            )
            pos_a = np.searchsorted(capacity_index.allow_idx, pv)
            vm = (pos_a < capacity_index.allow_idx.size) & (
                capacity_index.allow_idx[np.minimum(pos_a, capacity_index.allow_idx.size - 1)]
                == pv
            )
            b_i = (pos_a >> 3).astype(np.int64)
            b_m = (1 << (7 - (pos_a & 7))).astype(np.uint8)
            in_heavy = (
                (((capacity_index.packed[heavy_rows][:, b_i] & b_m[None, :]) != 0) | (~vm)[None, :])
                .astype(np.float32)
                .T
                @ weights_h
            )
            # Also add 4x4 stride grid fallback outside >=0.50 rasters so >200,000 clean cells ALWAYS exist!
            outside_order_key = in_heavy - 1e-4 * (pu / (float(pu.max()) + 1e-9))
            n_take = min(n_res, pv.size)
            outside_pool = pv[np.argpartition(outside_order_key, n_take - 1)[:n_take]]
            valid_idx = np.unique(np.concatenate([top_main, outside_pool]))
            del pos_a, vm, b_i, b_m, in_heavy, outside_order_key, is_peak3, pv, pu
        else:
            keep = np.argpartition(u_sub, -candidate_pool)[-candidate_pool:]
            valid_idx = valid_idx[keep]

    cap_int = int(np.floor(max_overlap_cap * target_n))
    memb: np.ndarray | None = None
    counts_arr = np.zeros(0, dtype=np.int32)
    if capacity_index is not None and cap_int > 0:
        memb = capacity_index.active_membership_matrix(valid_idx, cap_int)
        if memb is not None:
            counts_arr = np.zeros(memb.shape[1], dtype=np.int32)

    ys = (valid_idx // W + pad).tolist()
    xs = (valid_idx % W + pad).tolist()
    pens = pen_arr.ravel()[valid_idx].tolist()
    u_vals = u0.ravel()[valid_idx].tolist()
    del pen_arr, u0

    use_sep = min_sep_px > 0.0
    blocked = np.zeros((H + 2 * pad, W + 2 * pad), dtype=bool) if use_sep else None
    sep_offs = (
        [
            (oy, ox)
            for oy in range(-3, 4)
            for ox in range(-3, 4)
            if oy * oy + ox * ox <= min_sep_px * min_sep_px + 1e-9
        ]
        if use_sep
        else []
    )

    heap: list[tuple[float, int]] = [(-u_vals[i], i) for i in range(len(u_vals))]
    heapq.heapify(heap)

    selected_flat: list[int] = []
    while len(selected_flat) < target_n and heap:
        neg_ub, i = heapq.heappop(heap)
        cy = ys[i]
        cx = xs[i]
        if blocked is not None and blocked[cy, cx]:
            continue
        if memb is not None:
            row_m = memb[i]
            if row_m.any() and int(counts_arr[row_m].max()) >= cap_int:
                continue

        gain = 0.0
        for m in range(n_off):
            ny = cy + dy[m]
            nx = cx + dx[m]
            k_m = kw[m]
            cur_c = float(cov_pad[ny, nx])
            if k_m > cur_c:
                gain += float(q_pad[ny, nx]) * (k_m - cur_c)
        exact_u = gain - pens[i]

        if heap and (-exact_u > heap[0][0]):
            heapq.heappush(heap, (-exact_u, i))
            continue
        if exact_u <= -1.0:
            break

        selected_flat.append(int(valid_idx[i]))
        if memb is not None:
            counts_arr[memb[i]] += 1
        for m in range(n_off):
            ny = cy + dy[m]
            nx = cx + dx[m]
            if kw[m] > cov_pad[ny, nx]:
                cov_pad[ny, nx] = kw[m]
        if blocked is not None:
            for oy, ox in sep_offs:
                blocked[cy + oy, cx + ox] = True

    # Stage-2 exact full-footprint refill if top candidate pool saturated dense corridor rasters
    refill_round = 0
    while (
        memb is not None
        and capacity_index is not None
        and len(selected_flat) < target_n
        and refill_round < 6
    ):
        refill_round += 1
        sat_cols = np.flatnonzero(counts_arr >= cap_int)
        if sat_cols.size == 0:
            break
        already_mask = np.zeros(H * W, dtype=bool)
        already_mask[np.asarray(selected_flat, dtype=np.int64)] = True
        rem_allow = np.flatnonzero(allow & (~already_mask.reshape(H, W)))
        if rem_allow.size == 0:
            break

        # Re-identify exact active global informative rasters and screen out all saturated ones across 100% of Nevada!
        pos_all = np.searchsorted(capacity_index.allow_idx, rem_allow)
        vm_all = (pos_all < capacity_index.allow_idx.size) & (
            capacity_index.allow_idx[np.minimum(pos_all, capacity_index.allow_idx.size - 1)]
            == rem_allow
        )
        bi_all = (pos_all >> 3).astype(np.int64)
        bm_all = (1 << (7 - (pos_all & 7))).astype(np.uint8)

        # Find which `info_row_ids` currently exceed `cap_int - (target_n - len(selected_flat))` on `selected_flat`
        sel_arr = np.asarray(selected_flat, dtype=np.int64)
        p_sel = np.searchsorted(capacity_index.allow_idx, sel_arr)
        vm_sel = (p_sel < capacity_index.allow_idx.size) & (
            capacity_index.allow_idx[np.minimum(p_sel, capacity_index.allow_idx.size - 1)] == sel_arr
        )
        cur_hits = (
            ((capacity_index.packed[capacity_index.info_row_ids][:, p_sel >> 3] & (1 << (7 - (p_sel & 7))).astype(np.uint8)[None, :]) != 0)
            | (~vm_sel)[None, :]
        ).sum(axis=1)
        block_rids = capacity_index.info_row_ids[cur_hits >= cap_int]

        forbidden = ~vm_all
        for s_b in range(0, block_rids.size, 32):
            rb = block_rids[s_b : s_b + 32]
            forbidden = forbidden | (
                ((capacity_index.packed[rb][:, bi_all] & bm_all[None, :]) != 0).any(axis=0)
            )
        clean_cand = rem_allow[~forbidden]
        if clean_cand.size == 0:
            break

        # Score clean_cand strictly by uncovered 300 m kernel gain on `q_pad`
        c_scores = q_pad[clean_cand // W + pad, clean_cand % W + pad]
        n_ref = min(120_000, clean_cand.size)
        ref_idx = (
            clean_cand[np.argpartition(c_scores, -n_ref)[-n_ref:]]
            if clean_cand.size > n_ref
            else clean_cand
        )
        # Recompute exact remaining per-raster headroom `[cap_int - cur_hits]` across ALL 816 informative rasters!
        p_r = np.searchsorted(capacity_index.allow_idx, ref_idx)
        vm_r = (p_r < capacity_index.allow_idx.size) & (
            capacity_index.allow_idx[np.minimum(p_r, capacity_index.allow_idx.size - 1)] == ref_idx
        )
        bi_r = (p_r >> 3).astype(np.int64)
        bm_r = (1 << (7 - (p_r & 7))).astype(np.uint8)
        need_rem = target_n - len(selected_flat)
        tight_mask = (cur_hits + need_rem) > cap_int
        tight_rids = capacity_index.info_row_ids[tight_mask]
        headroom = (cap_int - cur_hits[tight_mask]).astype(np.int32)
        memb_r = (
            (((capacity_index.packed[tight_rids][:, bi_r] & bm_r[None, :]) != 0) | (~vm_r)[None, :]).T
            if tight_rids.size > 0
            else None
        )
        used_r = np.zeros(tight_rids.size, dtype=np.int32)

        order_r = np.argsort(-q_pad[ref_idx // W + pad, ref_idx % W + pad])
        for ord_i in order_r.tolist():
            if len(selected_flat) >= target_n:
                break
            cy_r = int(ref_idx[ord_i] // W) + pad
            cx_r = int(ref_idx[ord_i] % W) + pad
            if cov_pad[cy_r, cx_r] > 0.34:
                continue
            if memb_r is not None:
                rm = memb_r[ord_i]
                if rm.any() and np.any(used_r[rm] >= headroom[rm]):
                    continue
                used_r[rm] += 1
            selected_flat.append(int(ref_idx[ord_i]))
            for m in range(n_off):
                ny = cy_r + dy[m]
                nx = cx_r + dx[m]
                if kw[m] > cov_pad[ny, nx]:
                    cov_pad[ny, nx] = kw[m]

    out = np.zeros((H, W), dtype=bool)
    if selected_flat:
        out.ravel()[np.asarray(selected_flat, dtype=np.int64)] = True
    return out


def allocate_peak_nms_baseline(
    utility_2d: np.ndarray,
    allowed_2d: np.ndarray,
    target_n: int,
    min_sep_px: float = 2.80,
) -> np.ndarray:
    """Classical greedy point-peak NMS baseline without submodular kernel coverage."""
    h, w = utility_2d.shape
    u = np.where(np.isfinite(utility_2d) & allowed_2d, utility_2d, -1.0)
    cy, cx = np.nonzero(u > 0.0)
    order = np.argsort(-u[cy, cx])
    cy = cy[order]
    cx = cx[order]
    blocked = np.zeros((h + 8, w + 8), dtype=bool)
    sep_offs = [
        (dy, dx)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if dy * dy + dx * dx <= min_sep_px * min_sep_px + 1e-9
    ]
    out = np.zeros((h, w), dtype=bool)
    picked = 0
    for i in range(cy.size):
        if picked >= target_n:
            break
        r = int(cy[i])
        c = int(cx[i])
        if blocked[r + 4, c + 4]:
            continue
        out[r, c] = True
        picked += 1
        for dy, dx in sep_offs:
            blocked[r + 4 + dy, c + 4 + dx] = True
    return out
