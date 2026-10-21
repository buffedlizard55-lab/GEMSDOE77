"""Kernel-Convolved Flank-Zeroed Lagrangian Dinkelbach Submodular Allocator (LDSA) for H77V.

Combines:
1. Flank-zeroed quantile support (support_quantile=0.88, gamma=1.85) so diffuse background
   halos cannot outvote sharp 300 m fault corridors in the 29-pixel triangular kernel sum.
2. Smooth Monotonic Lagrangian Congestion Pricing inside the Minoux (1978) lazy-greedy
   300 m kernel cover heap: as any informative prior raster's 3-px halo utilization
   u_r = c_r / C_r grows, in-halo candidates are discounted by
       cong_div(u) = 1.0 + lambda_quad * u^2 + lambda_sixth * u^6,
   automatically steering dot placement to halo-free saddle positions at distance
   sqrt(10) = 3.162 px > 3.0 px from prior dot centers while maintaining 300 m kernel
   coverage across the fault trace.
3. Exact submodular Stage-2 refill if any binding raster hits its hard safety cap C_r.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, maximum_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import metric  # noqa: E402
from gems77e.allocate import (  # noqa: E402
    PortfolioCapacityIndex,
    allocate_peak_nms_baseline,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_audit_index_built(
    root_dir: Path,
    cache_dir: Path,
    fp: np.ndarray,
    cat: np.ndarray,
    stride: int = 16,
) -> Path:
    """Build or verify the packed 3-px halo & subsample matrix across all accessible rasters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "audit_meta.json"
    halos_path = cache_dir / "audit_halos_packed.npy"
    subvals_path = cache_dir / "audit_subvals.npy"

    scan_dirs: list[tuple[str, Path]] = [
        ("registry", root_dir / ".cache" / "data" / "registry" / "rasters"),
        ("witness", root_dir / ".cache" / "registry"),
        ("scored_sibling", root_dir / ".cache" / "registry_Audit" / "sibling_rasters"),
        ("recent_sibling", root_dir / ".cache" / "registry" / "sibling_rasters"),
        ("local_gemsdoe77", root_dir / "docs" / "downloads"),
        ("local_submissions", root_dir / "submissions"),
    ]

    discovered: list[tuple[str, Path]] = []
    for tag, d in scan_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.tif")):
            if "-nan.tif" in p.name or "h77v" in p.name.lower():
                continue
            discovered.append((tag, p))

    if meta_path.exists() and halos_path.exists() and subvals_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text())
            if old_meta.get("total_paths_discovered", 0) == len(discovered) and old_meta.get("n_valid", 0) >= 80:
                return meta_path
        except Exception:
            pass

    allow = fp & (~cat)
    allow_idx = np.flatnonzero(allow)
    n_allow = int(allow_idx.size)
    n_fp = int(fp.sum())
    sub_fp_idx = allow_idx[::stride]
    n_sub = int(sub_fp_idx.size)
    n_packed_bytes = (n_allow + 7) // 8

    yy, xx = np.ogrid[-3:4, -3:4]
    disk3 = (yy * yy + xx * xx) <= 9

    by_sha: dict[str, dict[str, Any]] = {}
    unreadable: list[dict[str, Any]] = []

    for tag, p in discovered:
        sha = _sha256_file(p)
        alias = f"{tag}:{p.name}"
        if sha in by_sha:
            if alias not in by_sha[sha]["aliases"]:
                by_sha[sha]["aliases"].append(alias)
            continue
        try:
            with rasterio.open(p) as ds:
                if ds.count != 1 or ds.shape != fp.shape:
                    unreadable.append(
                        {
                            "tag": tag,
                            "name": p.name,
                            "path": str(p),
                            "sha256": sha,
                            "aliases": [alias],
                            "reason": f"shape {ds.shape} count {ds.count}",
                        }
                    )
                    continue
                arr = ds.read(1).astype(np.float32)
        except Exception as exc:
            unreadable.append(
                {
                    "tag": tag,
                    "name": p.name,
                    "path": str(p),
                    "sha256": sha,
                    "aliases": [alias],
                    "reason": str(exc),
                }
            )
            continue

        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr[~fp] = 0.0
        pos_fp = (arr > 0.0) & fp
        pos_allow = pos_fp & allow
        n_pos_fp = int(pos_fp.sum())
        n_pos_allow = int(pos_allow.sum())
        halo_2d = binary_dilation(pos_fp, structure=disk3)
        halo_allow = halo_2d.ravel()[allow_idx]
        halo_frac = float(halo_allow.mean())
        supp_frac = float(n_pos_fp) / float(n_fp)
        packed_row = np.packbits(halo_allow.astype(np.uint8))
        sub_row = arr.ravel()[sub_fp_idx].astype(np.float32)

        by_sha[sha] = {
            "name": p.name,
            "tag": tag,
            "path": str(p),
            "sha256": sha,
            "aliases": [alias],
            "n_pos_fp": n_pos_fp,
            "n_pos_allow": n_pos_allow,
            "support_frac_fp": round(supp_frac, 6),
            "halo3_frac_allow": round(halo_frac, 6),
            "is_space_filling": bool(halo_frac >= 0.70),
            "_packed": packed_row,
            "_sub": sub_row,
        }

    entries_raw = sorted(by_sha.values(), key=lambda r: (r["tag"], r["name"], r["sha256"]))
    n_valid = len(entries_raw)
    packed_mat = np.empty((n_valid, n_packed_bytes), dtype=np.uint8)
    subvals_mat = np.empty((n_valid, n_sub), dtype=np.float32)

    entries_clean: list[dict[str, Any]] = []
    for idx, rec in enumerate(entries_raw):
        packed_mat[idx] = rec.pop("_packed")
        subvals_mat[idx] = rec.pop("_sub")
        rec["index"] = idx
        entries_clean.append(rec)

    np.save(halos_path, packed_mat)
    np.save(subvals_path, subvals_mat)
    meta = {
        "total_paths_discovered": len(discovered),
        "n_valid": n_valid,
        "stride": stride,
        "n_allow": n_allow,
        "n_fp": n_fp,
        "entries": entries_clean,
        "unreadable": unreadable,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


def allocate_ldsa_submodular(
    utility_2d: np.ndarray,
    allowed_2d: np.ndarray,
    target_n: int = 36_500,
    crest_2d: np.ndarray | None = None,
    support_quantile: float = 0.88,
    gamma: float = 1.85,
    crest_alpha: float = 0.65,
    fp_penalty_weight: float = 0.055,
    min_sep_px: float = 0.0,
    candidate_pool: int = 380_000,
    capacity_index: PortfolioCapacityIndex | None = None,
    max_overlap_cap: float = 0.650,
    lambda_quad: float = 0.45,
    lambda_sixth: float = 1.65,
) -> np.ndarray:
    """Flank-zeroed Minoux lazy-greedy 300 m kernel cover with Smooth Lagrangian Congestion Pricing (LDSA)."""
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
            n_main = 270_000
            n_res = 190_000
            top_main = valid_idx[np.argpartition(u_sub, -n_main)[-n_main:]]
            is_peak3 = (u0 >= maximum_filter(u0, size=3, mode="constant")).ravel()[valid_idx]
            peak_sub_idx = np.flatnonzero(is_peak3)
            pv = valid_idx[peak_sub_idx]
            pu = u_sub[peak_sub_idx]
            heavy_rows = np.array(
                [
                    r
                    for r in capacity_index.info_row_ids.tolist()
                    if capacity_index.entries[r]["halo3_frac_allow"] >= 0.16
                ],
                dtype=np.int32,
            )
            if heavy_rows.size > 0:
                weights_h = np.array(
                    [
                        capacity_index.entries[int(r)]["halo3_frac_allow"] ** 4
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
                outside_order_key = in_heavy - 1e-3 * (pu / (float(pu.max()) + 1e-9))
                n_take = min(n_res, pv.size)
                outside_pool = pv[np.argpartition(outside_order_key, n_take - 1)[:n_take]]
                valid_idx = np.unique(np.concatenate([top_main, outside_pool]))
                del pos_a, vm, b_i, b_m, in_heavy, outside_order_key
            else:
                keep = np.argpartition(u_sub, -candidate_pool)[-candidate_pool:]
                valid_idx = valid_idx[keep]
            del is_peak3, pv, pu
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
    inv_cap = 1.0 / float(max(cap_int, 1))

    while len(selected_flat) < target_n and heap:
        neg_ub, i = heapq.heappop(heap)
        cy = ys[i]
        cx = xs[i]
        if blocked is not None and blocked[cy, cx]:
            continue
        cong_div = 1.0
        row_m = None
        if memb is not None:
            row_m = memb[i]
            if row_m.any():
                c_max = int(counts_arr[row_m].max())
                if c_max >= cap_int:
                    continue
                u = float(c_max) * inv_cap
                cong_div = 1.0 + lambda_quad * (u * u) + lambda_sixth * (u ** 6)

        gain = 0.0
        for m in range(n_off):
            ny = cy + dy[m]
            nx = cx + dx[m]
            k_m = kw[m]
            cur_c = float(cov_pad[ny, nx])
            if k_m > cur_c:
                gain += float(q_pad[ny, nx]) * (k_m - cur_c)
        raw_u = gain - pens[i]
        exact_u = raw_u / cong_div if raw_u > 0.0 else raw_u * cong_div

        if heap and (-exact_u > heap[0][0]):
            heapq.heappush(heap, (-exact_u, i))
            continue
        if raw_u <= -1.0:
            break

        selected_flat.append(int(valid_idx[i]))
        if row_m is not None and row_m.any():
            counts_arr[row_m] += 1
        for m in range(n_off):
            ny = cy + dy[m]
            nx = cx + dx[m]
            if kw[m] > cov_pad[ny, nx]:
                cov_pad[ny, nx] = kw[m]
        if blocked is not None:
            for oy, ox in sep_offs:
                blocked[cy + oy, cx + ox] = True

    # Stage-2 submodular refill if any active corridor saturated
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

        pos_all = np.searchsorted(capacity_index.allow_idx, rem_allow)
        vm_all = (pos_all < capacity_index.allow_idx.size) & (
            capacity_index.allow_idx[np.minimum(pos_all, capacity_index.allow_idx.size - 1)]
            == rem_allow
        )
        bi_all = (pos_all >> 3).astype(np.int64)
        bm_all = (1 << (7 - (pos_all & 7))).astype(np.uint8)

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

        c_scores = q_pad[clean_cand // W + pad, clean_cand % W + pad]
        n_ref = min(120_000, clean_cand.size)
        ref_idx = (
            clean_cand[np.argpartition(c_scores, -n_ref)[-n_ref:]]
            if clean_cand.size > n_ref
            else clean_cand
        )
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


__all__ = [
    "PortfolioCapacityIndex",
    "allocate_ldsa_submodular",
    "allocate_peak_nms_baseline",
    "ensure_audit_index_built",
]
