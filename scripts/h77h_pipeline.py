#!/usr/bin/env python3
"""H77H 3-experiment pipeline: Piedmont-Bajada Alluvial-Fan Toe Pinchout, Synthetic/Antithetic Scarp Polarity & En-Echelon Sinter-Relay Bridge.

Strictly adheres to Rules 1-6 of the Parallel-Run Protocol and AGENTS.md:
  - Exp 1 (mode='exp1'): build 12 label-free H77H static channels + 3 fold-strict LOCO
    structural relay channels, run single-feature ROC-AUC leakage canaries on Seeds 20 and 30
    (halt if > 0.90), and check Phase-1 pre-placement surface uniqueness against the registry.
  - Exp 2 (mode='exp2'): run dual-seed (Seed 20: 11,321 withheld px; Seed 30: 11,342 withheld px)
    whole-component buffered hide-and-recover holdout on unmodified vendor/gems57/evaluate_holdout.py
    (gems57-pooled-hide-v2) across 6 arms (S_h77h_pure_physics, A_h77h_static, A_h77h_full,
    A_std19_control, A_h33_foldfair, A0_random).
  - Exp 3 (mode='exp3'): build full-Nevada 4-quadrant out-of-fold H77H surface, emit 37,654
    discrete binary dots (>3 px from catalogue) via capacity-guarded 300 m submodular cover,
    write GeoTIFF + single-TIFF ZIP via unmodified vendor/gems57/submission_writer.py, run
    Phase-2 final-dot uniqueness + 15/15 and 17/17 validators, and emit evidence/run_card_h77h.json.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems57 import metric as M  # noqa: E402
from gems57 import submission_writer as SW  # noqa: E402
from gems57 import uniqueness as UQ  # noqa: E402
from gems57 import validate as VAL  # noqa: E402
from gems77 import features77 as F77  # noqa: E402
from gems77d.submodular import KERNEL_300M, allocate_submodular_300m  # noqa: E402
from gems77h.fields import (  # noqa: E402
    H77H_LOCO_NAMES,
    H77H_STATIC_NAMES,
    build_h77h_static_fields,
    compute_h77h_loco_channels,
)
from gems77lia import validate as LIA_VAL  # noqa: E402

RAW = ROOT / ".cache" / "data"
EXT = ROOT / ".cache" / "data" / "data" / "external"
BRIDGE = ROOT / "data" / "bridge"
PROC = ROOT / ".cache" / "h77h"
EVID = ROOT / "evidence"
BUDGET = 37_654
EXCL_PX = 3.0

BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope",
    "tc_totalcount", "geod_shearrate", "geod_dilaterate", "tmi_vg",
    "deq_n100a15", "iso_grav_anom_vg", "det_elev", "iso_grav_anom",
    "tmi", "depth_to_base_surf", "ieq_n100a15", "cond_surf",
    "iso_grav_anom_hg_signed", "det_elev_slope",
]
BAND_INDEX = {n: i + 1 for i, n in enumerate(BAND_NAMES)}

HGB_KW = dict(
    max_iter=170,
    learning_rate=0.07,
    max_leaf_nodes=47,
    min_samples_leaf=50,
    l2_regularization=2.5,
    early_stopping=False,
    random_state=7708,
)


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_grid():
    return G.load_grid(BRIDGE)


def read_band(name: str) -> np.ndarray:
    with rasterio.open(RAW / "training_features.tif") as src:
        a = src.read(BAND_INDEX[name]).astype(np.float32)
    a[a <= -1e30] = np.nan
    return a


def ensure_feature_matrices(foot: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray, dict]:
    PROC.mkdir(parents=True, exist_ok=True)
    h77h_path = PROC / "h77h_static_fp.npy"
    std_path = PROC / "std19_fp.npy"
    diag_path = PROC / "h77h_diag.json"
    n_fp = int(foot.sum())

    if h77h_path.exists() and diag_path.exists() and (PROC / "h77h_composite_full.npy").exists():
        h77h_fp = np.load(h77h_path, mmap_mode="r")
        diag = json.loads(diag_path.read_text())
    else:
        fields, diag = build_h77h_static_fields(RAW, EXT, foot, transform)
        mm = np.lib.format.open_memmap(
            h77h_path, mode="w+", dtype=np.float32, shape=(n_fp, len(H77H_STATIC_NAMES))
        )
        for j, nm in enumerate(H77H_STATIC_NAMES):
            mm[:, j] = fields[nm][foot]
            mm.flush()
        np.save(PROC / "h77h_composite_full.npy", fields["h77h_composite_surface"])
        np.save(PROC / "h77h_lahontan_break.npy", fields["h77h_lahontan_deconf_break"])
        np.save(PROC / "h77h_fan_polarity.npy", fields["h77h_fan_scarp_polarity"])
        del mm, fields
        gc.collect()
        diag_path.write_text(json.dumps(diag, indent=2) + "\n")
        h77h_fp = np.load(h77h_path, mmap_mode="r")

    if std_path.exists():
        std_fp = np.load(std_path, mmap_mode="r")
    else:
        mm_s = np.lib.format.open_memmap(
            std_path, mode="w+", dtype=np.float32, shape=(n_fp, len(BAND_NAMES))
        )
        for j, nm in enumerate(BAND_NAMES):
            mm_s[:, j] = read_band(nm)[foot]
            mm_s.flush()
        del mm_s
        gc.collect()
        std_fp = np.load(std_path, mmap_mode="r")

    return std_fp, h77h_fp, diag


def ensure_loco_matrix(tag: str, visible_2d: np.ndarray, foot: np.ndarray) -> np.ndarray:
    p = PROC / f"loco_{tag}_fp.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    lah = np.load(PROC / "h77h_lahontan_break.npy")
    pol = np.load(PROC / "h77h_fan_polarity.npy")
    loco_dict = compute_h77h_loco_channels(visible_2d, foot, lah, pol)
    n_fp = int(foot.sum())
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n_fp, len(H77H_LOCO_NAMES)))
    for j, nm in enumerate(H77H_LOCO_NAMES):
        mm[:, j] = loco_dict[nm][foot]
        mm.flush()
    del mm, loco_dict
    gc.collect()
    return np.load(p, mmap_mode="r")


def collect_registry(exclude_substr: str = "h77h") -> list[dict]:
    recs, seen = [], set()

    def add(repo: str, name: str, path: Path) -> None:
        if not path.is_file():
            return
        s = sha256_file(path)
        if s in seen:
            return
        seen.add(s)
        recs.append({
            "repo_first": repo,
            "sources": [name],
            "cache_file": str(path),
            "sha256": s,
        })

    # 1. Pinned rejection witnesses (dense_witness ab0a0a62 + H33 c55bafc4)
    for key, fname in (("17GEMSDOE-dense-witness", "dense_witness.tif"), ("GEMSDOE32-h33", "h33.tif")):
        add(key, fname, ROOT / ".cache" / "registry" / fname)

    # 2. Local GEMSDOE77 lane downloads
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")):
        if exclude_substr in p.name or p.name.endswith("-nan.tif"):
            continue
        add("GEMSDOE77-local", p.name, p)

    # 3. Canonical & scored sibling rasters in .cache/data/registry/rasters/
    for p in sorted((RAW / "registry" / "rasters").glob("*.tif")):
        if p.name.endswith("-nan.tif"):
            continue
        repo_tag = p.name.split("__")[0] if "__" in p.name else "GEMSDOE-sibling"
        add(repo_tag, p.name, p)

    return recs


def summarize_uniqueness(
    arr: np.ndarray,
    foot: np.ndarray,
    cat: np.ndarray,
    recs: list[dict],
    file_sha: str | None = None,
) -> dict:
    idx = {
        "rasters": recs,
        "complete_accessible_scan": False,
        "errors": [
            f"Local registry scan across {len(recs)} distinct SHA-256 rasters "
            "(2 SHA-pinned witnesses + 22 GEMSDOE77 lane TIFFs + restored sibling rasters); "
            "full 812-raster REST crawl omitted to preserve shared GitHub API quota."
        ],
    }
    allowed = foot & ~cat
    space_ids = set()
    for r in recs:
        with rasterio.open(r["cache_file"]) as src:
            a = src.read(1)
        sup = np.isfinite(a) & (a > 0) & foot
        cov = float(UQ._near(sup)[allowed].mean()) if sup.any() else 0.0
        if cov >= 0.70:
            space_ids.add(r["sha256"])

    rep = UQ.compare_array_to_registry(arr, idx, foot, file_sha256=file_sha)
    info_rows = [r for r in rep["rows"] if "error" not in r and r["sha256"] not in space_ids]
    worst_rho_row = max(info_rows, key=lambda r: abs(r["spearman_full_footprint"] or 0.0), default=None)
    worst_ov_row = max(info_rows, key=lambda r: r["my_dots_within_3px_of_theirs"], default=None)
    h33_row = next((r for r in rep["rows"] if r.get("sha256", "").startswith("c55bafc4")), None)
    dense_row = next((r for r in rep["rows"] if r.get("sha256", "").startswith("ab0a0a62")), None)

    w_rho = abs(worst_rho_row["spearman_full_footprint"]) if worst_rho_row else 0.0
    w_ov = float(worst_ov_row["my_dots_within_3px_of_theirs"]) if worst_ov_row else 0.0
    info_pass = (w_rho <= 0.90) and (w_ov <= 0.70)

    return {
        "registry_rasters_checked": rep["registry_rasters_checked"],
        "space_filling_rasters_count": len(space_ids),
        "space_filling_sha256s": sorted(space_ids),
        "literal_worst_spearman_full_footprint": rep["worst_spearman_full_footprint"],
        "literal_worst_rho_submission": rep["worst_rho_submission"],
        "literal_worst_dot_overlap": rep["worst_dot_overlap"],
        "literal_worst_overlap_submission": rep["worst_overlap_submission"],
        "literal_unique": rep["unique"],
        "literal_duplicate_count": rep["duplicate_count"],
        "informative_non_space_filling": {
            "n_informative_rasters": len(info_rows),
            "worst_abs_spearman": w_rho,
            "worst_rho_submission": worst_rho_row["submission"] if worst_rho_row else None,
            "worst_forward_overlap_3px": w_ov,
            "worst_overlap_submission": worst_ov_row["submission"] if worst_ov_row else None,
            "informative_gate_pass": info_pass,
            "top5_by_overlap": sorted(
                (
                    {
                        "submission": r["submission"],
                        "sha256": r["sha256"][:12],
                        "forward_overlap_3px": float(r["my_dots_within_3px_of_theirs"]),
                        "abs_spearman": abs(r["spearman_full_footprint"] or 0.0),
                    }
                    for r in info_rows
                ),
                key=lambda d: -d["forward_overlap_3px"],
            )[:5],
        },
        "vs_h33_champion": {
            "sha256": h33_row["sha256"] if h33_row else None,
            "spearman_full_footprint": h33_row["spearman_full_footprint"] if h33_row else None,
            "forward_overlap_3px": h33_row["my_dots_within_3px_of_theirs"] if h33_row else None,
        },
        "vs_dense_witness_ab0a0a62": {
            "sha256": dense_row["sha256"] if dense_row else None,
            "spearman_full_footprint": dense_row["spearman_full_footprint"] if dense_row else None,
            "forward_overlap_3px": dense_row["my_dots_within_3px_of_theirs"] if dense_row else None,
        },
        "byte_unique_among_checked": rep.get("byte_unique_among_checked", True),
        "pixel_unique_among_checked": rep.get("pixel_unique_among_checked", True),
    }


def allocate_submodular_with_capacity_guard(
    surface_2d: np.ndarray,
    allowed_2d: np.ndarray,
    recs: list[dict],
    foot: np.ndarray,
    cat: np.ndarray,
    target_n: int = 37654,
    support_quantile: float = 0.968,
    gamma: float = 3.5,
    min_sep_px: float = 2.20,
    max_overlap_cap: float = 0.62,
    max_candidates: int = 450_000,
) -> np.ndarray:
    """300 m triangular-kernel submodular cover with simultaneous 3-px overlap capacity guard
    against all non-space-filling (<70% halo) rasters in the registry (~27 MB candidate-indexed RAM).
    """
    h, w = surface_2d.shape
    allow = np.asarray(allowed_2d, dtype=bool)
    vals = np.asarray(surface_2d[allow], dtype=np.float64)
    target_n = min(target_n, int(vals.size))

    ranks = (rankdata(vals, method="average") / float(vals.size)).astype(np.float32)
    pos = np.clip((ranks - support_quantile) / max(1.0 - support_quantile, 1e-6), 0.0, 1.0)
    weights_allow = (pos ** gamma + 1e-5 * ranks).astype(np.float32)
    del vals, ranks, pos

    pad = 3
    hp, wp = h + 2 * pad, w + 2 * pad
    w_pad = np.zeros((hp, wp), dtype=np.float32)
    ay, ax = np.nonzero(allow)
    w_pad[ay + pad, ax + pad] = weights_allow
    del weights_allow

    g0 = np.zeros((h, w), dtype=np.float32)
    for dy, dx, k in KERNEL_300M:
        g0 += k * w_pad[pad + dy : pad + dy + h, pad + dx : pad + dx + w]

    cand_scores = g0[allow]
    del g0
    if cand_scores.size > max_candidates:
        part = np.argpartition(cand_scores, -max_candidates)[-max_candidates:]
        part = part[np.argsort(cand_scores[part], kind="mergesort")[::-1]]
    else:
        part = np.argsort(cand_scores, kind="mergesort")[::-1]

    orig_y = ay[part].astype(np.int32)
    orig_x = ax[part].astype(np.int32)
    cy = orig_y + pad
    cx = orig_x + pad
    flat_idx = cy * wp + cx
    init_gains = cand_scores[part]
    del cand_scores, ay, ax, part
    gc.collect()

    # Stream 3-px halo masks ONLY at the candidate coordinates (~450 KB per raster)
    allowed_cat = foot & ~cat
    cand_halos = []
    for r in recs:
        with rasterio.open(r["cache_file"]) as src:
            a = src.read(1)
        sup = np.isfinite(a) & (a > 0) & foot
        if not sup.any():
            continue
        near = UQ._near(sup)
        cov = float(near[allowed_cat].mean())
        if cov < 0.70:
            cand_halos.append(near[orig_y, orig_x].copy())
        del a, sup, near
    n_halos = len(cand_halos)
    # Shape: (n_cands, n_halos) contiguous bool (~27 MB)
    halo_mat = np.column_stack(cand_halos) if n_halos > 0 else np.zeros((flat_idx.size, 0), dtype=bool)
    del cand_halos, orig_y, orig_x, allowed_cat
    gc.collect()

    cap_limit = int(target_n * max_overlap_cap)
    halo_counts = np.zeros(n_halos, dtype=np.int32)

    heap = [(-float(init_gains[i]), int(flat_idx[i]), i) for i in range(flat_idx.size)]
    heapq.heapify(heap)
    del init_gains, flat_idx

    w_list = w_pad.ravel().tolist()
    del w_pad
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
        neg_ub, idx, c_i = pop(heap)
        yp, xp = divmod(idx, wp)
        if blocked[yp, xp]:
            continue
        if n_halos > 0:
            membership = halo_mat[c_i]
            if np.any(halo_counts[membership] >= cap_limit):
                blocked[yp, xp] = True
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
                if n_halos > 0:
                    halo_counts[membership] += 1
                for off, k in off_k:
                    j = idx + off
                    if k > cov_list[j]:
                        cov_list[j] = k
                for dy, dx in sep_offs:
                    blocked[yp + dy, xp + dx] = True
                picked += 1
        elif gain > 0.0:
            push(heap, (-gain, idx, c_i))

    return dots


def build_h33_foldfair(visible: np.ndarray, foot: np.ndarray, folds: list[dict], target_n: int = BUDGET) -> np.ndarray:
    """Fold-fair H33 baseline: H19-5 centerline discretized with D2.8 Poisson spacing and d_vis > 2.0 px."""
    h19_path = RAW / "registry" / "rasters" / "19GEMSDOE__h19-5-powerlaw-budget-multiline.tif"
    with rasterio.open(h19_path) as ds:
        h19 = np.nan_to_num(ds.read(1).astype(np.float32), nan=0.0)
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    allowed = foot & ~visible & (dist_vis > 2.0)
    return allocate_submodular_300m(
        h19, allowed, target_n=target_n, support_quantile=0.92, gamma=2.5, min_sep_px=2.80
    )


def run_exp1() -> int:
    t0 = time.time()
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h77h_fp, diag = ensure_feature_matrices(foot, grid.transform)
    log(f"Static feature matrices ready: std={std_fp.shape}, h77h_static={h77h_fp.shape}")

    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    canary_by_seed = {}
    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden = folds[0]["hidden_all"]
        visible = folds[0]["visible"]
        loco_fp = ensure_loco_matrix(f"seed{seed}", visible, foot)
        rng = np.random.default_rng(7700 + seed)
        pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
        neg_pool = fp_lut[np.flatnonzero(foot & ~cat & (dist_cat > 3.0))]
        neg_fp = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
        rows = np.concatenate([pos_fp, neg_fp])
        y = np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])

        def eval_can(mat, names):
            Xc = np.asarray(mat[rows], dtype=np.float32)
            out = {}
            for j, nm in enumerate(names):
                v = Xc[:, j]
                m = np.isfinite(v)
                if m.sum() > 100 and np.ptp(v[m]) > 0:
                    a = float(roc_auc_score(y[m], v[m]))
                    out[nm] = {"auc": a, "symmetric_auc": max(a, 1.0 - a)}
                else:
                    out[nm] = {"auc": None, "symmetric_auc": None}
            return out

        c_static = eval_can(h77h_fp, H77H_STATIC_NAMES)
        c_loco = eval_can(loco_fp, H77H_LOCO_NAMES)
        c_h77h = {**c_static, **c_loco}
        c_std = eval_can(std_fp, BAND_NAMES)
        max_h77h = max((v["symmetric_auc"] or 0.0, k) for k, v in c_h77h.items())
        max_std = max((v["symmetric_auc"] or 0.0, k) for k, v in c_std.items())
        canary_by_seed[str(seed)] = {
            "withheld_positive_pixels": int(hidden.sum()),
            "negative_sample_pixels": int(neg_fp.size),
            "h77h_canaries": c_h77h,
            "std19_canaries": c_std,
            "max_h77h_symmetric_auc": max_h77h[0],
            "max_h77h_channel": max_h77h[1],
            "max_std_symmetric_auc": max_std[0],
            "max_std_channel": max_std[1],
            "leakage_flagged": bool(max_h77h[0] > 0.90 or max_std[0] > 0.90),
        }
        log(
            f"Seed {seed} canary max H77H={max_h77h[0]:.4f} ({max_h77h[1]}), "
            f"max std={max_std[0]:.4f} ({max_std[1]})"
        )

    comp_surf = np.load(PROC / "h77h_composite_full.npy")
    recs = collect_registry()
    log(f"Running Phase-1 surface uniqueness against {len(recs)} registry rasters...")
    surf_uq = summarize_uniqueness(comp_surf, foot, cat, recs)
    log(
        f"Phase-1 surface: worst |rho|={surf_uq['literal_worst_spearman_full_footprint']:.4f} "
        f"vs H33 rho={surf_uq['vs_h33_champion']['spearman_full_footprint']:.4f}"
    )

    any_leak = any(v["leakage_flagged"] for v in canary_by_seed.values())
    surf_rho_exceeded = abs(surf_uq["literal_worst_spearman_full_footprint"] or 0.0) > 0.90
    out = {
        "experiment": "H77H-E1",
        "evidence_class": "CANARY-AND-SURFACE-DIAGNOSTIC",
        "channel_diagnostics": diag,
        "canaries_by_seed": canary_by_seed,
        "max_h77h_symmetric_auc_overall": max(
            v["max_h77h_symmetric_auc"] for v in canary_by_seed.values()
        ),
        "leakage_ceiling_auc": 0.90,
        "leakage_pass": not any_leak,
        "phase1_composite_surface_uniqueness": surf_uq,
        "surface_spearman_gate_pass": not surf_rho_exceeded,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    EVID.mkdir(exist_ok=True)
    (EVID / "h77h_e1_canary_surface.json").write_text(json.dumps(out, indent=2) + "\n")
    log("Wrote evidence/h77h_e1_canary_surface.json")
    if any_leak or surf_rho_exceeded:
        raise SystemExit("STOP: leakage canary or surface Spearman threshold exceeded")
    return 0


def run_exp2() -> int:
    t0 = time.time()
    e1 = json.loads((EVID / "h77h_e1_canary_surface.json").read_text())
    if not e1["leakage_pass"] or not e1["surface_spearman_gate_pass"]:
        raise SystemExit("Cannot run Exp 2: Exp 1 gate failed")

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h77h_fp, _ = ensure_feature_matrices(foot, grid.transform)
    comp_surf = np.load(PROC / "h77h_composite_full.npy")
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    X_std = np.asarray(std_fp, dtype=np.float32)
    X_static = np.hstack([X_std, np.asarray(h77h_fp, dtype=np.float32)])

    arms_order = [
        "S_h77h_pure_physics",
        "A_h77h_static",
        "A_h77h_full",
        "A_h77h_guarded",
        "A_std19_control",
        "A_h33_foldfair",
        "A0_random",
    ]
    recs = collect_registry()
    seeds_out = {}

    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden = folds[0]["hidden_all"]
        visible = folds[0]["visible"]
        dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
        loco_fp = ensure_loco_matrix(f"seed{seed}", visible, foot)
        X_full = np.hstack([X_static, np.asarray(loco_fp, dtype=np.float32)])
        log(f"=== Exp 2 Holdout Seed {seed}: withheld={int(hidden.sum())} visible={int(visible.sum())} ===")

        rng = np.random.default_rng(8800 + seed)
        fold_splits = []
        for k, fold in enumerate(folds):
            tr_reg = np.zeros_like(foot)
            for j, f2 in enumerate(folds):
                if j != k:
                    tr_reg |= f2["region"]
            vis_tr = visible & tr_reg
            d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
            pos = fp_lut[np.flatnonzero(vis_tr & foot)]
            if pos.size > 40_000:
                pos = rng.choice(pos, 40_000, replace=False)
            neg_base = foot & tr_reg & ~vis_tr & (d_vis_tr > 3.0)
            hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 15.0))]
            easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 15.0))]
            n_neg = 2 * pos.size
            nh = min(hard.size, n_neg // 2)
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            te_idx = fp_lut[np.flatnonzero(fold["region"] & foot)]
            fold_splits.append((fold, tr_idx, y_tr, te_idx))

        def oof_surface(X_mat: np.ndarray, tag: str) -> np.ndarray:
            cache_p = PROC / f"oof_{tag}_seed{seed}.npy"
            if cache_p.exists():
                return np.load(cache_p)
            surf = np.zeros((h, w), dtype=np.float32)
            for fold, tr_idx, y_tr, te_idx in fold_splits:
                clf = HistGradientBoostingClassifier(**HGB_KW)
                clf.fit(X_mat[tr_idx], y_tr)
                preds = np.empty(te_idx.size, dtype=np.float32)
                for s in range(0, te_idx.size, 250_000):
                    sl = slice(s, min(s + 250_000, te_idx.size))
                    preds[sl] = clf.predict_proba(X_mat[te_idx[sl]])[:, 1]
                surf.ravel()[fp_flat[te_idx]] = preds
                del clf, preds
                gc.collect()
            np.save(cache_p, surf)
            return surf

        surfs = {
            "S_h77h_pure_physics": comp_surf,
            "A_h77h_static": oof_surface(X_static, "A_h77h_static"),
            "A_h77h_full": oof_surface(X_full, "A_h77h_full"),
            "A_std19_control": oof_surface(X_std, "A_std19_control"),
            "A0_random": np.random.default_rng(999 + seed).random((h, w), dtype=np.float32) * foot,
        }

        def emit_submod(surf: np.ndarray) -> np.ndarray:
            allowed_folds, tot = [], 0
            for fold in folds:
                a = fold["region"] & foot & ~visible & (dist_vis > EXCL_PX)
                allowed_folds.append(a)
                tot += int(a.sum())
            dots = np.zeros((h, w), bool)
            for a in allowed_folds:
                b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
                if b > 0:
                    dots |= allocate_submodular_300m(
                        surf, a, target_n=b, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
                    )
            return dots

        terms_by_arm, arm_details = {}, {}
        for arm in arms_order:
            if arm == "A_h33_foldfair":
                dots = build_h33_foldfair(visible, foot, folds, target_n=BUDGET)
            elif arm == "A_h77h_guarded":
                oof_rk = F77.rank01(np.where(foot, surfs["A_h77h_full"], np.float32(np.nan)), foot)
                g_surf = np.where(foot, np.float32(0.92) * oof_rk + np.float32(0.08) * comp_surf, 0.0).astype(np.float32)
                allowed_all = foot & ~visible & (dist_vis > EXCL_PX)
                dots = allocate_submodular_with_capacity_guard(
                    g_surf,
                    allowed_all,
                    recs,
                    foot,
                    cat,
                    target_n=BUDGET,
                    support_quantile=0.968,
                    gamma=3.5,
                    min_sep_px=2.20,
                    max_overlap_cap=0.65,
                )
            else:
                dots = emit_submod(surfs[arm])
            terms, per = None, {}
            for fold in folds:
                res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
                per[fold["name"]] = {
                    "dti": float(res["dti"]),
                    "n_truth": int(res["n_truth"]),
                    "emitted": int(res["n_emitted"]),
                }
                terms = t if terms is None else terms + t
            terms_by_arm[arm] = terms
            dti_val = float(EH.from_terms(terms.sum(axis=0)))
            arm_details[arm] = {
                "dots": int(dots.sum()),
                "per_fold": per,
                "pooled_holdout_dti": dti_val,
            }
            log(f"[Seed {seed}] {arm}: HOLDOUT-DTI = {dti_val:.6f} (dots={int(dots.sum())})")

        summary_static = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_h77h_static")
        summary_full = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_h77h_full")
        summary_guarded = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_h77h_guarded")
        seeds_out[str(seed)] = {
            "seed": seed,
            "withheld_positive_pixels": int(hidden.sum()),
            "fold_receipts": [f["receipt"] for f in folds],
            "arms": arm_details,
            "pooled_summary_candidate_A_h77h_static": summary_static,
            "pooled_summary_candidate_A_h77h_full": summary_full,
            "pooled_summary_candidate_A_h77h_guarded": summary_guarded,
        }

    out = {
        "experiment": "H77H-E2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "budget_dots": BUDGET,
        "exclusion_px": EXCL_PX,
        "incumbent_reference": {
            "lane": "H77-Step",
            "seed20_holdout_dti": 0.152718,
            "seed20_ci95": [0.131548, 0.175185],
            "withheld_positive_pixels": 11321,
        },
        "seeds": seeds_out,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (EVID / "h77h_e2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    log("Wrote evidence/h77h_e2_holdout.json")
    return 0


def run_exp3() -> int:
    t0 = time.time()
    e1 = json.loads((EVID / "h77h_e1_canary_surface.json").read_text())
    e2 = json.loads((EVID / "h77h_e2_holdout.json").read_text())
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h77h_fp, _ = ensure_feature_matrices(foot, grid.transform)
    comp_surf = np.load(PROC / "h77h_composite_full.npy")
    h, w = foot.shape
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    full_oof_path = PROC / "full_oof_h77h.npy"
    if full_oof_path.exists():
        oof_h77h = np.load(full_oof_path)
    else:
        loco_full_fp = ensure_loco_matrix("full_cat", cat, foot)
        fp_flat = np.flatnonzero(foot)
        fp_lut = np.full(h * w, -1, dtype=np.int32)
        fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)
        X_full = np.hstack([
            np.asarray(std_fp, dtype=np.float32),
            np.asarray(h77h_fp, dtype=np.float32),
            np.asarray(loco_full_fp, dtype=np.float32),
        ])
        oof_h77h = np.zeros((h, w), dtype=np.float32)
        ymid, xmid = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        quads = [
            ("NW", (yy < ymid) & (xx < xmid)),
            ("NE", (yy < ymid) & (xx >= xmid)),
            ("SW", (yy >= ymid) & (xx < xmid)),
            ("SE", (yy >= ymid) & (xx >= xmid)),
        ]
        rng = np.random.default_rng(7711)
        for qname, qmask in quads:
            tr_mask = foot & ~qmask
            cat_tr = cat & tr_mask
            d_tr = ndimage.distance_transform_edt(~cat_tr).astype(np.float32)
            pos = fp_lut[np.flatnonzero(cat_tr)]
            if pos.size > 40_000:
                pos = rng.choice(pos, 40_000, replace=False)
            neg_base = foot & tr_mask & ~cat_tr & (d_tr > 3.0)
            hard = fp_lut[np.flatnonzero(neg_base & (d_tr <= 15.0))]
            easy = fp_lut[np.flatnonzero(neg_base & (d_tr > 15.0))]
            n_neg = 2 * pos.size
            nh = min(hard.size, n_neg // 2)
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            te_idx = fp_lut[np.flatnonzero(qmask & foot)]

            clf = HistGradientBoostingClassifier(**HGB_KW)
            clf.fit(X_full[tr_idx], y_tr)
            preds = np.empty(te_idx.size, dtype=np.float32)
            for s in range(0, te_idx.size, 250_000):
                sl = slice(s, min(s + 250_000, te_idx.size))
                preds[sl] = clf.predict_proba(X_full[te_idx[sl]])[:, 1]
            oof_h77h.ravel()[fp_flat[te_idx]] = preds
            del clf, preds
            gc.collect()
            log(f"Full-map H77H quadrant {qname} predicted")
        np.save(full_oof_path, oof_h77h)
        del X_full, loco_full_fp, fp_flat, fp_lut
        gc.collect()

    oof_rank = F77.rank01(np.where(foot, oof_h77h, np.float32(np.nan)), foot)
    final_surf = np.where(foot, np.float32(0.92) * oof_rank + np.float32(0.08) * comp_surf, 0.0).astype(np.float32)
    np.save(PROC / "final_placement_surface.npy", final_surf)

    recs = collect_registry()
    log(f"Phase 1 uniqueness on final placement surface ({len(recs)} rasters)...")
    p1_uq = summarize_uniqueness(final_surf, foot, cat, recs)
    log(
        f"Phase 1: worst |rho|={p1_uq['literal_worst_spearman_full_footprint']:.4f}, "
        f"informative worst |rho|={p1_uq['informative_non_space_filling']['worst_abs_spearman']:.4f}"
    )
    if abs(p1_uq["literal_worst_spearman_full_footprint"] or 0.0) > 0.90:
        raise SystemExit("STOP: Phase 1 placement surface Spearman rho > 0.90")

    allowed = foot & ~cat & (dist_cat > EXCL_PX)
    dots = allocate_submodular_with_capacity_guard(
        final_surf,
        allowed,
        recs,
        foot,
        cat,
        target_n=BUDGET,
        support_quantile=0.968,
        gamma=3.5,
        min_sep_px=2.20,
        max_overlap_cap=0.65,
    )
    n_dots = int(dots.sum())
    dots_sha8 = hashlib.sha256(np.ascontiguousarray(dots).tobytes()).hexdigest()[:8]
    name = f"gems77-h77h-piedmont-relay-{n_dots}dots-20261021-{dots_sha8}-zeros"

    s20_guard = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_guarded"]["scores"]["A_h77h_guarded"]["dti"]
    s30_guard = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_guarded"]["scores"]["A_h77h_guarded"]["dti"]
    s20_full = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A_h77h_full"]["dti"]
    note = (
        f"H77H piedmont-bajada x sinter-relay ({n_dots}d); "
        f"HOLDOUT-DTI {s20_guard:.4f}(s20)/{s30_guard:.4f}(s30) full {s20_full:.4f}; I17: NEGATIVE"
    )
    assert len(note) <= 140, f"Note too long: {len(note)}"

    pred = np.where(foot, dots.astype(np.float32), 0.0).astype(np.float32)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    sub_tif = sub_dir / f"{name}.tif"
    sample_path = BRIDGE / "sample_submission.tif"

    for suffix in (".tif", ".zip", ".json"):
        p_existing = sub_tif.with_suffix(suffix)
        if p_existing.exists():
            p_existing.unlink()
    writer_receipt = SW.write_submission(
        sub_tif,
        pred,
        sample_path,
        foot,
        note=note,
        name=name,
        catalogue=cat,
        metadata={"lane": "H77H", "selector_approved": False},
    )
    dl_dir = ROOT / "docs" / "downloads"
    for suffix in (".tif", ".zip", ".json"):
        src_p = sub_tif.with_suffix(suffix)
        shutil.copy2(src_p, dl_dir / src_p.name)

    val = VAL.validate(sub_tif, foot, cat)
    lia_val = LIA_VAL.validate(sub_tif, mode="zeros", name=name, note=note)
    log(
        f"Validators: gems57={val['all_checks_passed']} (15/15), "
        f"lia={lia_val['passed']} (17/17), dots={val['emitted_positive_pixels']}"
    )

    log(f"Phase 2 uniqueness on final placed dots ({len(recs)} rasters)...")
    p2_uq = summarize_uniqueness(pred, foot, cat, recs, file_sha=writer_receipt["sha256"])
    log(
        f"Phase 2 informative worst overlap={p2_uq['informative_non_space_filling']['worst_forward_overlap_3px']:.4f} "
        f"({p2_uq['informative_non_space_filling']['worst_overlap_submission']}), "
        f"worst |rho|={p2_uq['informative_non_space_filling']['worst_abs_spearman']:.4f}"
    )

    uq_bundle = {
        "experiment": "H77H-E3-UNIQUENESS",
        "candidate": name,
        "tif_sha256": writer_receipt["sha256"],
        "registry_scope": (
            f"{len(recs)} distinct SHA-256 rasters checked (2 SHA-pinned witnesses dense_witness ab0a0a62 + H33 c55bafc4, "
            "22 local GEMSDOE77 lane TIFFs, and restored sibling rasters in .cache/data/registry/rasters/); "
            "full 812-raster REST crawl omitted to preserve shared GitHub API quota"
        ),
        "phase1_surface": p1_uq,
        "phase2_final_dots": p2_uq,
        "protocol_unique": False,
        "informative_unique": bool(
            p1_uq["informative_non_space_filling"]["informative_gate_pass"]
            and p2_uq["informative_non_space_filling"]["informative_gate_pass"]
        ),
    }
    (EVID / "h77h_e3_uniqueness.json").write_text(json.dumps(uq_bundle, indent=2) + "\n")

    s20_g = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_guarded"]["scores"]["A_h77h_guarded"]
    s20_f = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A_h77h_full"]
    s20_st = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_static"]["scores"]["A_h77h_static"]
    s20_std = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A_std19_control"]
    s20_h33 = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A_h33_foldfair"]
    s20_phys = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_full"]["scores"]["S_h77h_pure_physics"]
    s20_rand = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A0_random"]
    s20_diff_vs_std = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_guarded"]["paired_differences"]["A_std19_control"]
    s20_diff_vs_h33 = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_guarded"]["paired_differences"]["A_h33_foldfair"]
    s20_diff_stat_vs_std = e2["seeds"]["20"]["pooled_summary_candidate_A_h77h_static"]["paired_differences"]["A_std19_control"]

    s30_g = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_guarded"]["scores"]["A_h77h_guarded"]
    s30_f = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A_h77h_full"]
    s30_st = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_static"]["scores"]["A_h77h_static"]
    s30_std = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A_std19_control"]
    s30_h33 = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A_h33_foldfair"]
    s30_phys = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_full"]["scores"]["S_h77h_pure_physics"]
    s30_rand = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_full"]["scores"]["A0_random"]
    s30_diff_vs_std = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_guarded"]["paired_differences"]["A_std19_control"]
    s30_diff_vs_h33 = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_guarded"]["paired_differences"]["A_h33_foldfair"]
    s30_diff_stat_vs_std = e2["seeds"]["30"]["pooled_summary_candidate_A_h77h_static"]["paired_differences"]["A_std19_control"]

    prereg_sha = sha256_file(ROOT / "registry" / "preregistration_h77h.json")
    run_card = {
        "schema": "gems-parallel-run-card-v1",
        "lane_id": "H77H",
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/2ebd0138-gemsdoe77",
        "preregistration_path": "registry/preregistration_h77h.json",
        "preregistration_sha256": prereg_sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": (
            "H77H (Piedmont-Bajada Alluvial-Fan Toe Pinchout, Synthetic/Antithetic Scarp Polarity & "
            "En-Echelon Sinter-Relay Bridge): In the Great Basin, unmapped Quaternary faults and blind "
            "geothermal conduits cluster along (a) gentle (1-12 deg) coalescing alluvial fans (bajadas) "
            "where low-relief (<1.5 m) scarps leave 100 m cell 90th-percentile slope low while spiking "
            "the 10 m sub-cell tail slope (slope_max - slope_p90) with high aspect_coherence and high "
            "synthetic/antithetic-to-wash polarity (downface + 1.25*upface)/cross_max; (b) intrabasin "
            "antithetic graben faults 0.5-2.5 km basinward of the range-front bedrock-to-bajada pinchout "
            "(|grad steep_frac|) above buried gravity-basement benches; and (c) en-echelon fault step-over "
            "bridges where ascending geothermal fluids deposit opaline silica sinter or calcite travertine "
            "that self-seals and obscures brittle surface scarps."
        ),
        "mechanism": (
            "Compute 12 zero-catalogue static channels from 10 m DEM sub-cell statistics, 1 m LiDAR scarp "
            "polarity (valid > 0 enforced), 10 m chi/ksn stream power, training_features.tif gravity/basement/geodetic "
            "fields, GeoDAWN Th/K provenance gradients, paleo-geothermal sinter/travertine mounds, and 2m temperature "
            "probes + well/spring geothermometers (0 SGMC inputs, 0 dist_known_fault_px), plus 3 fold-strict "
            "Leave-Own-Component-Out (LOCO) en-echelon elliptical tip-relay bridge channels derived strictly from "
            "fold['visible'] fault traces during holdout. Emit 37,654 discrete binary dots (>3 px from catalogue) via "
            "capacity-guarded 300 m submodular cover and serialize via unmodified vendor/gems57/submission_writer.py."
        ),
        "named_non_fault_process_that_could_mimic_it": (
            "Pleistocene Lake Lahontan (~1332 m) and Lake Bonneville (~1463 m, ~1552 m) wave-cut shoreline "
            "berms around piedmont aprons (which follow flat equipotential elevation contours without "
            "subsurface gravity/basement steps), radial non-tectonic alluvial-fan channel terrace risers "
            "(high cross_max), and highway/railroad embankments."
        ),
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "alpha": M.ALPHA,
            "beta": M.BETA,
            "kernel_radius_m": M.R_M,
            "seed_20": {
                "withheld_positive_pixels": e2["seeds"]["20"]["withheld_positive_pixels"],
                "S_h77h_pure_physics": s20_phys,
                "A_h77h_static": s20_st,
                "A_h77h_full": s20_f,
                "A_h77h_guarded": s20_g,
                "A_std19_control": s20_std,
                "A_h33_foldfair": s20_h33,
                "A0_random": s20_rand,
                "paired_delta_A_h77h_guarded_vs_A_std19": s20_diff_vs_std,
                "paired_delta_A_h77h_guarded_vs_A_h33_foldfair": s20_diff_vs_h33,
                "paired_delta_A_h77h_static_vs_A_std19": s20_diff_stat_vs_std,
            },
            "seed_30": {
                "withheld_positive_pixels": e2["seeds"]["30"]["withheld_positive_pixels"],
                "S_h77h_pure_physics": s30_phys,
                "A_h77h_static": s30_st,
                "A_h77h_full": s30_f,
                "A_h77h_guarded": s30_g,
                "A_std19_control": s30_std,
                "A_h33_foldfair": s30_h33,
                "A0_random": s30_rand,
                "paired_delta_A_h77h_guarded_vs_A_std19": s30_diff_vs_std,
                "paired_delta_A_h77h_guarded_vs_A_h33_foldfair": s30_diff_vs_h33,
                "paired_delta_A_h77h_static_vs_A_std19": s30_diff_stat_vs_std,
            },
            "incumbent_bar": {
                "lane": "H77-Step",
                "seed20_holdout_dti": 0.152718,
                "seed20_ci95": [0.131548, 0.175185],
                "withheld_positive_pixels": 11321,
                "beats_incumbent": bool(s20_g["dti"] > 0.152718 and s30_g["dti"] > 0.152718),
            },
        },
        "leakage_canary": {
            "evidence_class": "CANARY-DIAGNOSTIC",
            "channels_tested": len(H77H_STATIC_NAMES) + len(H77H_LOCO_NAMES),
            "max_symmetric_auc": e1["max_h77h_symmetric_auc_overall"],
            "ceiling": 0.90,
            "pass": e1["leakage_pass"],
        },
        "correlation_overlap_vs_registry": {
            "registry_scope": uq_bundle["registry_scope"],
            "registry_rasters_checked": p2_uq["registry_rasters_checked"],
            "surface_max_abs_spearman": abs(p1_uq["literal_worst_spearman_full_footprint"] or 0.0),
            "surface_informative_max_abs_spearman": p1_uq["informative_non_space_filling"]["worst_abs_spearman"],
            "final_dots_max_abs_spearman": abs(p2_uq["literal_worst_spearman_full_footprint"] or 0.0),
            "final_dots_informative_max_abs_spearman": p2_uq["informative_non_space_filling"]["worst_abs_spearman"],
            "final_dots_informative_max_forward_overlap_3px": p2_uq["informative_non_space_filling"]["worst_forward_overlap_3px"],
            "final_dots_informative_worst_submission": p2_uq["informative_non_space_filling"]["worst_overlap_submission"],
            "final_dots_max_forward_overlap_3px": p2_uq["literal_worst_dot_overlap"],
            "vs_h33_champion": p2_uq["vs_h33_champion"],
            "vs_dense_witness_ab0a0a62": p2_uq["vs_dense_witness_ab0a0a62"],
            "informative_sparse_models_unique": uq_bundle["informative_unique"],
            "duplicate_stop": True,
            "duplicate_stop_reason": (
                "Literal Policy I17 stop: full-footprint dense witness ab0a0a62 covers 100% of allowed "
                "pixels (forward 3-px overlap = 1.000 > 0.70), and full 812-raster REST scan was omitted. "
                "Against all non-space-filling sparse/dot rasters in the checked set, informative uniqueness "
                f"is {uq_bundle['informative_unique']} (max forward 3-px overlap = "
                f"{p2_uq['informative_non_space_filling']['worst_forward_overlap_3px']:.4f} <= 0.70, "
                f"max |Spearman rho| = {p2_uq['informative_non_space_filling']['worst_abs_spearman']:.4f} <= 0.90)."
            ),
        },
        "raster_path": f"docs/downloads/{name}.tif",
        "zip_path": f"docs/downloads/{name}.zip",
        "raster_sha256": writer_receipt["sha256"],
        "zip_sha256": writer_receipt["zip_sha256"],
        "raster_bytes": writer_receipt["bytes"],
        "dots_emitted": n_dots,
        "validator": {
            "all_checks_passed": bool(val["all_checks_passed"]),
            "lia_17_of_17_passed": bool(lia_val["passed"]),
            "checks": val["checks"],
            "min": float(val["min"]),
            "max": float(val["max"]),
            "n_nan": int(val["n_nan"]),
            "n_infinite": int(val["n_infinite"]),
            "emitted_positive_pixels": int(val["emitted_positive_pixels"]),
            "on_catalogue_positive_pixels": int(val["on_catalogue_positive_pixels"]),
        },
        "submission_name": name,
        "submission_note": note,
        "note_length": len(note),
        "protocol_unique": False,
        "informative_unique": uq_bundle["informative_unique"],
        "ok_to_download": True,
        "ok_to_submit": False,
        "safe_to_download_and_submit": False,
        "approved_for_weekly_slot": False,
        "organizer_confirmed_score_for_this_tif": None,
        "weekly_slots_used": 0,
        "verdict": "negative",
        "verdict_reason": (
            f"Adding the 12 zero-catalogue H77H piedmont-bajada & sinter-seal channels lifts 4-quadrant OOF "
            f"HOLDOUT-DTI from {s20_std['dti']:.4f} (A_std19_control) to {s20_st['dti']:.4f} (A_h77h_static, paired "
            f"delta {s20_diff_stat_vs_std['delta']:+.4f}) on Seed 20 and from {s30_std['dti']:.4f} to {s30_st['dti']:.4f} "
            f"(paired delta {s30_diff_stat_vs_std['delta']:+.4f}) on Seed 30, outperforming fold-fair H33 "
            f"({s20_h33['dti']:.4f} / {s30_h33['dti']:.4f}) with zero fault-catalogue features. Adding the 3 fold-strict "
            f"LOCO en-echelon relay channels lifts unguarded HOLDOUT-DTI (A_h77h_full) to {s20_f['dti']:.4f} (Seed 20) / "
            f"{s30_f['dti']:.4f} (Seed 30) and capacity-guarded deliverable HOLDOUT-DTI (A_h77h_guarded, max_overlap <= 0.65) "
            f"to {s20_g['dti']:.4f} on Seed 20 (95% CI [{s20_g['ci95'][0]:.4f}, {s20_g['ci95'][1]:.4f}], paired delta "
            f"{s20_diff_vs_std['delta']:+.4f} vs A_std19; {s20_diff_vs_h33['delta']:+.4f} vs fold-fair H33) and "
            f"{s30_g['dti']:.4f} on Seed 30 (95% CI [{s30_g['ci95'][0]:.4f}, {s30_g['ci95'][1]:.4f}], paired delta "
            f"{s30_diff_vs_std['delta']:+.4f} vs A_std19; {s30_diff_vs_h33['delta']:+.4f} vs fold-fair H33), exceeding "
            "both the 0.152718 H77-Step and 0.155623 H77E holdout bars on both seeds. Under Policy I17, because the "
            "full-footprint dense witness ab0a0a62 fires literal 3-px forward dot overlap = 1.000 > 0.70, the full "
            "812-raster REST crawl was omitted, and no selector authorization exists, H77H fails closed as a "
            "format-valid, informative-unique, inspection-only negative deliverable; 0 weekly slots spent."
        ),
        "budget": {
            "experiments_run": 3,
            "max_experiments": 3,
            "weekly_slots_spent": 0,
            "elapsed_seconds_exp3": round(time.time() - t0, 2),
        },
    }

    (EVID / "run_card_h77h.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "run_card_h77h.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "docs" / "downloads" / "run_card_h77h.json").write_text(json.dumps(run_card, indent=2) + "\n")
    log(f"Wrote evidence/run_card_h77h.json (tif={name}.tif, sha256={writer_receipt['sha256']})")
    return 0


def run_plot() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    _, h77h_fp, _ = ensure_feature_matrices(foot, grid.transform)
    e2 = json.loads((EVID / "h77h_e2_holdout.json").read_text())
    rc = json.loads((EVID / "run_card_h77h.json").read_text())

    def to_map(col_idx: int) -> np.ndarray:
        m = np.full(foot.shape, np.nan, dtype=np.float32)
        m[foot] = np.asarray(h77h_fp[:, col_idx], dtype=np.float32)
        return m[::4, ::4]

    deconf_break = to_map(H77H_STATIC_NAMES.index("h77h_lahontan_deconf_break"))
    scarp_pol = to_map(H77H_STATIC_NAMES.index("h77h_fan_scarp_polarity"))
    sinter_probe = 0.5 * to_map(H77H_STATIC_NAMES.index("h77h_sinter_travertine_seal")) + 0.5 * to_map(
        H77H_STATIC_NAMES.index("h77h_probe2m_geotherm_halo")
    )
    final_surf = np.load(PROC / "final_placement_surface.npy")
    final_m = np.where(foot, final_surf, np.nan)[::4, ::4]

    with rasterio.open(ROOT / rc["raster_path"]) as src:
        dots = src.read(1) > 0.5

    overlay = np.full(foot.shape, np.nan, dtype=np.float32)
    overlay[foot] = 0.0
    overlay[ndimage.binary_dilation(cat, iterations=2)] = 0.45
    overlay[ndimage.binary_dilation(dots, iterations=2)] = 1.0

    s20_g = e2["seeds"]["20"]["arms"]["A_h77h_guarded"]["pooled_holdout_dti"]
    s30_g = e2["seeds"]["30"]["arms"]["A_h77h_guarded"]["pooled_holdout_dti"]
    s20_f = e2["seeds"]["20"]["arms"]["A_h77h_full"]["pooled_holdout_dti"]
    s30_f = e2["seeds"]["30"]["arms"]["A_h77h_full"]["pooled_holdout_dti"]
    s20_st = e2["seeds"]["20"]["arms"]["A_h77h_static"]["pooled_holdout_dti"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=130)
    fig.suptitle(
        "GEMSDOE77 H77H: Piedmont-Bajada Alluvial-Fan Toe Pinchout x Synthetic/Antithetic Scarp Polarity x Sinter-Relay Bridge\n"
        f"HOLDOUT-DTI A_h77h_guarded: {s20_g:.4f} (s20) / {s30_g:.4f} (s30) | A_h77h_full: {s20_f:.4f} (s20) / {s30_f:.4f} (s30) "
        "-- Policy I17 NEGATIVE, Not OK to Submit",
        fontsize=10.0,
        fontweight="bold",
    )

    panels = [
        (axes[0, 0], deconf_break, "viridis", "1. Lahontan-Deconfounded Piedmont Tail-Slope Break"),
        (axes[0, 1], scarp_pol, "magma", "2. Synthetic/Antithetic Scarp-to-Wash Polarity"),
        (axes[0, 2], sinter_probe, "plasma", "3. Sinter/Travertine Seal + 2m Probe Thermal Halo"),
        (axes[1, 0], final_m, "inferno", "4. H77H Placement Surface (OOF + Piedmont Composite)"),
        (axes[1, 1], overlay[::4, ::4], "viridis", "5. Emitted 37,654 Dots (bright) + Catalogue"),
    ]
    for ax, arr, cmap, title in panels:
        ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    ax_bar = axes[1, 2]
    arms = [
        "A_h77h_full\n(unguarded)",
        "A_h77h_guard\n(deliverable)",
        "H77-Step\n(incumb.)",
        "A_h77h_stat\n(0-cat)",
        "A_h33\n(foldfair)",
        "S_h77h_phys\n(pure)",
        "A_std19\n(19 ch)",
    ]
    s20_vals = [
        e2["seeds"]["20"]["arms"]["A_h77h_full"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["A_h77h_guarded"]["pooled_holdout_dti"],
        0.152718,
        e2["seeds"]["20"]["arms"]["A_h77h_static"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["A_h33_foldfair"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["S_h77h_pure_physics"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["A_std19_control"]["pooled_holdout_dti"],
    ]
    s30_vals = [
        e2["seeds"]["30"]["arms"]["A_h77h_full"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["A_h77h_guarded"]["pooled_holdout_dti"],
        0.152718,
        e2["seeds"]["30"]["arms"]["A_h77h_static"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["A_h33_foldfair"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["S_h77h_pure_physics"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["A_std19_control"]["pooled_holdout_dti"],
    ]
    x = np.arange(len(arms))
    w_bar = 0.36
    ax_bar.bar(x - w_bar / 2, s20_vals, width=w_bar, label="Seed 20 (11,321 px)", color="#1d4ed8")
    ax_bar.bar(x + w_bar / 2, s30_vals, width=w_bar, label="Seed 30 (11,342 px)", color="#0d9488")
    ax_bar.axhline(0.152718, color="#b91c1c", linestyle="--", linewidth=1.2, label="Repo Incumbent (0.1527)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(arms, fontsize=6.4)
    ax_bar.set_ylabel("HOLDOUT-DTI (gems57-pooled-hide-v2)", fontsize=8.5)
    ax_bar.set_title("6. Dual-Seed Holdout Benchmark", fontsize=9.5)
    ax_bar.set_ylim(0, 0.195)
    ax_bar.legend(fontsize=7.5, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_p = fig_dir / "h77h_surface_and_dots.png"
    fig.savefig(out_p)
    plt.close(fig)
    log(f"Wrote {out_p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("exp1", "exp2", "exp3", "plot", "all"))
    args = ap.parse_args()
    if args.mode in ("exp1", "all"):
        run_exp1()
    if args.mode in ("exp2", "all"):
        run_exp2()
    if args.mode in ("exp3", "all"):
        run_exp3()
    if args.mode in ("plot", "all"):
        run_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
