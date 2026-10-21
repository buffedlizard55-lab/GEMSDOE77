"""300 m triangular-kernel submodular dot allocator with exact informative-registry capacity cap.

Uses the exact 3-px Euclidean disk ``UQ._near`` from ``vendor/gems57/uniqueness.py`` so that
every informative (<70% footprint halo coverage) raster in the checked registry is guaranteed
to have forward 3-px dot overlap <= ``max_overlap_cap`` (default ``0.680 < 0.70``).
"""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import rankdata

from gems57 import uniqueness as UQ
from gems77d.submodular import KERNEL_300M


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_checked_registry(root: Path, raw_dir: Path) -> list[dict[str, Any]]:
    """Collect all accessible prior submission rasters on disk, deduplicated by SHA-256."""
    recs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(repo: str, name: str, path: Path) -> None:
        if not path.is_file():
            return
        s = sha256_file(path)
        if s in seen:
            return
        seen.add(s)
        recs.append(
            {
                "repo_first": repo,
                "sources": [name],
                "cache_file": str(path),
                "sha256": s,
            }
        )

    for p in sorted((root / ".cache" / "registry").glob("*.tif")):
        _add("registry-witness", p.name, p)
    for p in sorted((root / "docs" / "downloads").glob("*.tif")):
        if "h77h" in p.name.lower() or p.name.endswith("-nan.tif"):
            continue
        _add("GEMSDOE77-local", p.name, p)
    for p in sorted((root / ".cache" / "h77e_registry").glob("*.tif")):
        repo_tag = p.name.split("__")[0] if "__" in p.name else "sibling-scored"
        _add(repo_tag, p.name, p)
    h19_path = (
        raw_dir
        / "scored"
        / "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif"
    )
    _add("19GEMSDOE", h19_path.name, h19_path)
    return recs


def build_informative_halo_matrix(
    records: list[dict[str, Any]],
    footprint: np.ndarray,
    catalogue: np.ndarray,
    cache_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str], dict[str, float]]:
    """Build or reload bit-packed 3-px halo matrix for all informative (<70% halo) rasters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    foot = np.asarray(footprint, dtype=bool)
    cat = np.asarray(catalogue, dtype=bool)
    allowed = foot & (~cat)
    n_flat = foot.size
    n_bytes = (n_flat + 7) // 8

    packed_path = cache_dir / "h77h_info_halos_packed.npy"
    meta_path = cache_dir / "h77h_info_halos_meta.json"
    shas_ordered = [r["sha256"] for r in records]

    if packed_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("record_shas") == shas_ordered and meta.get("n_flat") == n_flat:
            return (
                np.load(packed_path),
                meta["info_meta"],
                meta["space_filling_shas"],
                meta["coverage_by_sha"],
            )

    packed_rows: list[np.ndarray] = []
    info_meta: list[dict[str, Any]] = []
    space_filling_shas: list[str] = []
    coverage_by_sha: dict[str, float] = {}

    for r in records:
        with rasterio.open(r["cache_file"]) as src:
            arr = src.read(1)
        sup = np.isfinite(arr) & (arr > 0) & foot
        if sup.any():
            halo = UQ._near(sup)
            cov = float(halo[allowed].mean())
        else:
            halo = np.zeros_like(foot)
            cov = 0.0
        coverage_by_sha[r["sha256"]] = round(cov, 6)
        if cov >= 0.70:
            space_filling_shas.append(r["sha256"])
        else:
            packed_rows.append(np.packbits(halo.ravel()))
            info_meta.append(
                {
                    "sha256": r["sha256"],
                    "repo_first": r["repo_first"],
                    "submission": r["sources"][0],
                    "halo_coverage_allowed": round(cov, 6),
                    "support_pixels": int(sup.sum()),
                }
            )

    if packed_rows:
        packed_arr = np.vstack(packed_rows).astype(np.uint8)
    else:
        packed_arr = np.zeros((0, n_bytes), dtype=np.uint8)

    np.save(packed_path, packed_arr)
    meta_out = {
        "n_flat": n_flat,
        "record_shas": shas_ordered,
        "info_meta": info_meta,
        "space_filling_shas": sorted(space_filling_shas),
        "coverage_by_sha": coverage_by_sha,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2) + "\n")
    return packed_arr, info_meta, sorted(space_filling_shas), coverage_by_sha


def check_informative_overlaps(
    dots_2d: np.ndarray,
    packed_halos: np.ndarray,
    info_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify exact 3-px forward dot overlap against every informative raster."""
    flat_dots = np.flatnonzero(dots_2d.ravel() > 0.5)
    n_dots = int(flat_dots.size)
    if n_dots == 0 or packed_halos.shape[0] == 0:
        return {
            "n_informative": int(packed_halos.shape[0]),
            "worst_forward_overlap_3px": 0.0,
            "worst_overlap_submission": None,
            "informative_firings_gt_0p70": 0,
            "top5_by_overlap": [],
        }

    byte_idx = flat_dots >> 3
    bit_mask = (1 << (7 - (flat_dots & 7))).astype(np.uint8)
    hits = ((packed_halos[:, byte_idx] & bit_mask[None, :]) != 0).sum(axis=1)
    ovs = hits.astype(np.float64) / float(n_dots)
    order = np.argsort(-ovs)
    best_i = int(order[0])
    top5 = [
        {
            "submission": info_meta[int(i)]["submission"],
            "repo": info_meta[int(i)]["repo_first"],
            "sha256": info_meta[int(i)]["sha256"],
            "forward_overlap_3px": round(float(ovs[int(i)]), 6),
        }
        for i in order[:5]
    ]
    return {
        "n_informative": int(packed_halos.shape[0]),
        "worst_forward_overlap_3px": round(float(ovs[best_i]), 6),
        "worst_overlap_submission": info_meta[best_i]["submission"],
        "informative_firings_gt_0p70": int((ovs > 0.70).sum()),
        "top5_by_overlap": top5,
    }


def allocate_h77h_submodular_300m(
    surface_2d: np.ndarray,
    allowed_2d: np.ndarray,
    target_n: int = 36500,
    crest_2d: np.ndarray | None = None,
    support_quantile: float = 0.88,
    gamma: float = 1.85,
    crest_alpha: float = 0.55,
    fp_penalty_weight: float = 0.055,
    min_sep_px: float = 0.0,
    max_candidates: int = 420_000,
    packed_halos: np.ndarray | None = None,
    max_overlap_cap: float = 0.680,
) -> np.ndarray:
    """Flank-aware lazy-greedy 300 m triangular-kernel submodular dot allocator with Dinkelbach
    FP cost and optional exact per-raster informative overlap cap (<= max_overlap_cap < 0.70).
    """
    h, w = surface_2d.shape
    allow = np.asarray(allowed_2d, dtype=bool)
    vals = np.nan_to_num(np.asarray(surface_2d[allow], dtype=np.float32), nan=0.0)
    if target_n <= 0 or vals.size == 0:
        return np.zeros((h, w), dtype=bool)
    target_n = min(int(target_n), int(vals.size))

    ranks = (rankdata(vals, method="average") / float(vals.size)).astype(np.float32)
    pos = np.clip((ranks - support_quantile) / max(1.0 - support_quantile, 1e-6), 0.0, 1.0)
    tail_prob = np.clip((ranks - 0.72) / 0.28, 0.0, 1.0) ** 3.0
    base_w = (pos ** gamma) + 0.08 * tail_prob + 0.015 * (ranks ** 2)
    if crest_2d is not None:
        c_sub = np.clip(np.nan_to_num(crest_2d[allow].astype(np.float32), nan=0.0), 0.0, 1.0)
        weights_allow = (base_w * (1.0 + crest_alpha * c_sub)).astype(np.float32)
    else:
        weights_allow = base_w.astype(np.float32)

    pad = 3
    hp, wp = h + 2 * pad, w + 2 * pad
    w_pad = np.zeros((hp, wp), dtype=np.float32)
    ay, ax = np.nonzero(allow)
    w_pad[ay + pad, ax + pad] = weights_allow

    g0 = np.zeros((h, w), dtype=np.float32)
    q_self = np.zeros((h, w), dtype=np.float32)
    for dy, dx, k in KERNEL_300M:
        sub = w_pad[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
        g0 += k * sub
        np.maximum(q_self, k * sub, out=q_self)

    pen_arr = (fp_penalty_weight * (1.0 - np.minimum(q_self, 1.0))).astype(np.float32)
    u0 = g0 - pen_arr

    cand_scores = u0[allow]
    if packed_halos is not None and packed_halos.shape[0] > 0:
        n_top = min(max_candidates, cand_scores.size)
        top_part = np.argpartition(cand_scores, -n_top)[-n_top:]
        is_peak = (u0 >= ndimage.maximum_filter(u0, size=3, mode="constant"))[allow]
        peak_part = np.flatnonzero(is_peak)
        part = np.unique(np.concatenate([top_part, peak_part]))
        part = part[np.argsort(cand_scores[part], kind="mergesort")[::-1]]
    else:
        if cand_scores.size > max_candidates:
            part = np.argpartition(cand_scores, -max_candidates)[-max_candidates:]
            part = part[np.argsort(cand_scores[part], kind="mergesort")[::-1]]
        else:
            part = np.argsort(cand_scores, kind="mergesort")[::-1]

    cy_orig = ay[part].astype(np.int32)
    cx_orig = ax[part].astype(np.int32)
    orig_flat = cy_orig * w + cx_orig
    cy = cy_orig + pad
    cx = cx_orig + pad
    flat_idx = cy * wp + cx
    init_gains = cand_scores[part]
    pens_cand = pen_arr[allow][part].tolist()

    cap_count = int(np.floor(max_overlap_cap * target_n))
    cand_memb: np.ndarray | None = None
    halo_counts = np.zeros(0, dtype=np.int32)
    if packed_halos is not None and packed_halos.shape[0] > 0 and cap_count > 0:
        b_idx = orig_flat >> 3
        b_mask = (1 << (7 - (orig_flat & 7))).astype(np.uint8)
        cand_memb = np.ascontiguousarray(((packed_halos[:, b_idx] & b_mask[None, :]) != 0).T)
        halo_counts = np.zeros(packed_halos.shape[0], dtype=np.int32)

    heap = [(-float(init_gains[i]), i) for i in range(flat_idx.size)]
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
        neg_ub, c_i = pop(heap)
        yp = int(cy[c_i])
        xp = int(cx[c_i])
        if blocked[yp, xp]:
            continue
        if cand_memb is not None:
            row_m = cand_memb[c_i]
            if row_m.any() and int(halo_counts[row_m].max()) >= cap_count:
                continue

        idx = int(flat_idx[c_i])
        gain = 0.0
        for off, k in off_k:
            j = idx + off
            wj = w_list[j]
            if wj > 0.0:
                rem = k - cov_list[j]
                if rem > 0.0:
                    gain += wj * rem
        exact_u = gain - pens_cand[c_i]

        if not heap or exact_u >= -heap[0][0] - 1e-12:
            y, x = yp - pad, xp - pad
            if not dots[y, x]:
                dots[y, x] = True
                if cand_memb is not None:
                    halo_counts[cand_memb[c_i]] += 1
                for off, k in off_k:
                    j = idx + off
                    if k > cov_list[j]:
                        cov_list[j] = k
                for dy, dx in sep_offs:
                    blocked[yp + dy, xp + dx] = True
                picked += 1
        elif exact_u > -1.0:
            push(heap, (-exact_u, c_i))

    return dots
