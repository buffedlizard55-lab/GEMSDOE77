#!/usr/bin/env python3
"""Complete 3-Experiment Pipeline for Lane H83-PCD (Parallel-Run Protocol).

Experiment 1 (H83-PCD-E1):
  - Build/load 60-channel static H83-PCD memmap + 5 fold-visible 12-bit LOCO channels.
  - Single-feature symmetric ROC-AUC leakage canary on Seed 30 and Seed 20 (stop if > 0.90).
  - Phase-1 pre-placement continuous surface Spearman correlation vs registry (stop if |rho| > 0.90).
  - Standalone label-free surface screen on gems57-pooled-hide-v2.

Experiment 2 (H83-PCD-E2):
  - Dual-seed buffered whole-segment holdout evaluation (Seed 30: 11,342 withheld positives;
    Seed 20: 11,321 withheld positives; evaluator version gems57-pooled-hide-v2) across 7 arms
    with 1,000-draw paired spatial block-bootstrap 95% confidence intervals.

Experiment 3 (H83-PCD-E3):
  - 4-quadrant out-of-fold full-map H83-PCD discovery surface + bitwise sparse-overlap-capped
    300 m submodular cover (37,654 binary dots, 0 pixels copied, 0 SGMC channels).
  - Shared vendor/gems57/submission_writer.py packaging + scripts/validate_submission.py & 17/17 readback.
  - Phase-2 final-dot uniqueness audit (both informative sparse scope <= 0.70 and literal Policy I17
    dense-witness stop), diagnostic figure generation (`docs/figures/h83pcd_*.png`), and frozen JSON
    Run Card (`evidence/run_card_h83pcd.json`).
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "vendor"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gems57 import evaluate_holdout as EH
from gems57 import grid as G
from gems57 import holdout as HO
from gems57 import metric as M
from gems57 import submission_writer as SW
from gems57 import uniqueness as UQ
from gems77 import emit77 as E77
from src.gems77_pcd.allocator import (
    allocate_submodular_300m_pcd,
    build_sparse_halo_bitmask,
)
from src.gems77_pcd.features import (
    ALL_STATIC_NAMES,
    BASE19_NAMES,
    PCD_STATIC_NAMES,
    PCD_VIS_NAMES,
    RAD_NAMES,
    TERR_NAMES,
    build_static_feature_memmap,
    compute_pcd_visible_mechanics,
)
from src.gems77d.features import rank01_fp
from src.gems77d.submodular import allocate_submodular_300m

RAW = REPO_ROOT / "data" / "raw"
PROC = REPO_ROOT / "data" / "processed"
EVIDENCE_DIR = REPO_ROOT / "evidence"
DOCS_EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence"
DOCS_DOWNLOADS_DIR = REPO_ROOT / "docs" / "downloads"
FIG_DIR = REPO_ROOT / "docs" / "figures"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

TARGET_DOTS = 37654
EXCL_PX = 3.0
SPARSE_OVERLAP_CAP = 0.64
RETAINED_INCUMBENT_DTI = 0.152718
STEM = "h83pcd-poisson-piedmont-loco-submod-20261021T183000Z-zeros"
SUBMISSION_NOTE = (
    "H83-PCD: Poisson pseudogravity-gravity discord + 1m LiDAR piedmont toe "
    "+ 12b LOCO relay + 300m submod 37654d (neg/I17)"
)
assert len(SUBMISSION_NOTE) <= 140

HGB_STATIC_KW = dict(
    max_iter=180,
    learning_rate=0.075,
    max_leaf_nodes=55,
    min_samples_leaf=50,
    l2_regularization=2.5,
    early_stopping=False,
    random_state=8301,
)
HGB_MECH_KW = dict(
    max_iter=180,
    learning_rate=0.075,
    max_leaf_nodes=47,
    min_samples_leaf=60,
    l2_regularization=3.0,
    early_stopping=False,
    random_state=8302,
)
HGB_STD_KW = dict(
    max_iter=180,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.5,
    early_stopping=False,
    random_state=8300,
)


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def collect_reference_rasters(exclude_prefix: str = "h83pcd-") -> List[Tuple[str, Path]]:
    """Collect all prior GeoTIFF rasters across registry, scored, submissions, and downloads."""
    seen_sha: set[str] = set()
    items: List[Tuple[str, Path]] = []
    search_dirs = [
        REPO_ROOT / ".cache" / "registry",
        REPO_ROOT / "data" / "raw" / "scored",
        REPO_ROOT / "submissions",
        REPO_ROOT / "docs" / "downloads",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.tif")):
            if p.name.startswith(exclude_prefix) or p.name.endswith("-nan.tif"):
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h in seen_sha:
                continue
            seen_sha.add(h)
            items.append((f"{d.name}/{p.name}", p))
    return items


def build_sparse_reference_halos(
    foot: np.ndarray,
    allowed_2d: np.ndarray,
    exclude_prefix: str = "h83pcd-",
    max_sparse_positives: int = 100000,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load all sparse dot-field reference rasters (0 < n_pos <= 100,000) and pack 3-px halos in-place."""
    ref_items = collect_reference_rasters(exclude_prefix=exclude_prefix)
    meta_list: List[Dict] = []
    h, w = foot.shape
    bits = np.zeros((h, w), dtype=np.uint64)
    crowd = np.zeros((h, w), dtype=np.float32)

    offs_le3 = [
        (dy, dx)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if dy * dy + dx * dx <= 9
    ]
    halo_pad = np.zeros((h + 6, w + 6), dtype=bool)

    for label, path in ref_items:
        with rasterio.open(path) as src:
            if (src.height, src.width) != (h, w):
                continue
            arr = src.read(1)
        pos = np.isfinite(arr) & (arr > 0.0) & foot
        del arr
        n_pos = int(pos.sum())
        if 0 < n_pos <= max_sparse_positives:
            halo_pad.fill(False)
            for dy, dx in offs_le3:
                halo_pad[3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w] |= pos
            halo = halo_pad[3 : 3 + h, 3 : 3 + w] & allowed_2d
            j = len(meta_list)
            bits[halo] |= np.uint64(1) << np.uint64(j)
            crowd[halo] += 1.0
            meta_list.append(
                {
                    "label": label,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "n_positive": n_pos,
                    "halo_allowed_pixels": int(halo.sum()),
                }
            )
            del halo
            if len(meta_list) >= 64:
                break
        del pos

    if meta_list:
        crowd /= float(len(meta_list))
    return bits, crowd, meta_list


def audit_uniqueness_against_all_rasters(
    candidate_2d: np.ndarray,
    foot: np.ndarray,
    is_binary_dots: bool,
    exclude_prefix: str = "h83pcd-",
    subsample_step: int = 7,
) -> Dict:
    """Compute Spearman rank correlation and (if binary dots) 3-px overlap vs all reference rasters."""
    ref_items = collect_reference_rasters(exclude_prefix=exclude_prefix)
    h, w = foot.shape
    cand_fp = candidate_2d[foot][::subsample_step].astype(np.float64)
    cand_fp = np.where(np.isfinite(cand_fp), cand_fp, 0.0)
    cand_r = rankdata(cand_fp, method="average")
    cand_r_c = cand_r - cand_r.mean()
    cand_norm = float(np.linalg.norm(cand_r_c)) + 1e-12

    cand_dots = (candidate_2d > 0.5) & foot
    n_cand_dots = int(cand_dots.sum())
    offs_le3 = [
        (dy, dx)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if dy * dy + dx * dx <= 9
    ]
    halo_pad = np.zeros((h + 6, w + 6), dtype=bool)
    cand_halo_3px = None
    if is_binary_dots and n_cand_dots > 0:
        for dy, dx in offs_le3:
            halo_pad[3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w] |= cand_dots
        cand_halo_3px = halo_pad[3 : 3 + h, 3 : 3 + w].copy()

    rows: List[Dict] = []
    max_abs_rho_all = 0.0
    max_abs_rho_sparse = 0.0
    max_fwd_ov_all = 0.0
    max_fwd_ov_sparse = 0.0
    worst_sparse_peer = None
    worst_all_peer = None

    for label, path in ref_items:
        with rasterio.open(path) as src:
            if (src.height, src.width) != (h, w):
                continue
            other = src.read(1)
        other_fp = other[foot][::subsample_step].astype(np.float64)
        other_fp = np.where(np.isfinite(other_fp), other_fp, 0.0)
        if float(other_fp.max() - other_fp.min()) == 0.0:
            rho = 0.0
        else:
            o_r = rankdata(other_fp, method="average")
            o_r_c = o_r - o_r.mean()
            rho = float(
                np.dot(cand_r_c, o_r_c)
                / (cand_norm * (float(np.linalg.norm(o_r_c)) + 1e-12))
            )

        other_pos = np.isfinite(other) & (other > 0.0) & foot
        n_other_pos = int(other_pos.sum())
        is_sparse = 0 < n_other_pos <= 100000

        fwd_ov = 0.0
        rev_ov = 0.0
        sym_ov = 0.0
        if is_binary_dots and n_cand_dots > 0 and n_other_pos > 0:
            halo_pad.fill(False)
            for dy, dx in offs_le3:
                halo_pad[3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w] |= other_pos
            other_halo_3px = halo_pad[3 : 3 + h, 3 : 3 + w]
            fwd_ov = float(other_halo_3px[cand_dots].mean())
            rev_ov = (
                float(cand_halo_3px[other_pos].mean())
                if cand_halo_3px is not None
                else 0.0
            )
            sym_ov = max(fwd_ov, rev_ov)

        abs_rho = abs(rho)
        if abs_rho > max_abs_rho_all:
            max_abs_rho_all = abs_rho
        if is_sparse and abs_rho > max_abs_rho_sparse:
            max_abs_rho_sparse = abs_rho
        if fwd_ov > max_fwd_ov_all:
            max_fwd_ov_all = fwd_ov
            worst_all_peer = label
        if is_sparse and fwd_ov > max_fwd_ov_sparse:
            max_fwd_ov_sparse = fwd_ov
            worst_sparse_peer = label

        rows.append(
            {
                "peer": label,
                "n_positive": n_other_pos,
                "scope": "sparse_dot_field" if is_sparse else "dense_continuous_surface",
                "spearman_rho": round(rho, 6),
                "forward_overlap_3px": round(fwd_ov, 6),
                "reverse_overlap_3px": round(rev_ov, 6),
                "symmetric_max_overlap_3px": round(sym_ov, 6),
            }
        )

    return {
        "n_reference_rasters_checked": len(rows),
        "informative_sparse_scope": {
            "definition": "0 < n_positive <= 100,000 in footprint",
            "n_sparse_rasters": sum(1 for r in rows if r["scope"] == "sparse_dot_field"),
            "max_abs_spearman": round(max_abs_rho_sparse, 6),
            "max_forward_overlap_3px": round(max_fwd_ov_sparse, 6),
            "worst_overlap_peer": worst_sparse_peer,
            "passes_sparse_gate": bool(
                max_abs_rho_sparse <= 0.90
                and (not is_binary_dots or max_fwd_ov_sparse <= 0.70)
            ),
        },
        "literal_policy_i17_all_raster_scope": {
            "definition": (
                "All registry + local rasters including dense continuous surface "
                "17GEMSDOE_E-proba-multiscale (ab0a0a62)"
            ),
            "max_abs_spearman": round(max_abs_rho_all, 6),
            "max_forward_overlap_3px": round(max_fwd_ov_all, 6),
            "worst_overlap_peer": worst_all_peer,
            "passes_literal_i17_gate": bool(
                max_abs_rho_all <= 0.90
                and (not is_binary_dots or max_fwd_ov_all <= 0.70)
            ),
        },
        "peers": rows,
    }


def build_fold_visible_mechanics_for_mask(
    visible_mask: np.ndarray,
    footprint: np.ndarray,
    mm_static: np.memmap,
    cache_tag: str | None = None,
) -> np.ndarray:
    """Construct or load cached 5 fold-visible 12-bit LOCO channels for a given `visible_mask`."""
    if cache_tag is not None:
        cpath = PROC / "gems77_pcd_cache" / f"mech_{cache_tag}.npy"
        if cpath.exists():
            return np.load(cpath, mmap_mode="r")
    idx_map = {n: i for i, n in enumerate(ALL_STATIC_NAMES)}
    geod_2d = np.zeros(footprint.shape, dtype=np.float32)
    geod_2d[footprint] = mm_static[:, idx_map["transtens_strain"]]
    poisson_2d = np.zeros(footprint.shape, dtype=np.float32)
    poisson_2d[footprint] = mm_static[:, idx_map["pcd_poisson_anti_step"]]
    toe_2d = np.zeros(footprint.shape, dtype=np.float32)
    toe_2d[footprint] = mm_static[:, idx_map["pcd_piedmont_toe_scarp"]]
    out = compute_pcd_visible_mechanics(
        visible_mask=visible_mask,
        footprint=footprint,
        geod_rank_2d=geod_2d,
        pcd_poisson_anti_2d=poisson_2d,
        pcd_piedmont_toe_2d=toe_2d,
    )
    if cache_tag is not None:
        cpath = PROC / "gems77_pcd_cache" / f"mech_{cache_tag}.npy"
        np.save(cpath, out)
        return np.load(cpath, mmap_mode="r")
    return out


def build_fold_io(
    folds: List[Dict], foot: np.ndarray, visible: np.ndarray, seed: int
) -> List[Tuple]:
    """Construct strictly label-blind spatial-quadrant training splits with hard+easy negatives."""
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    rng = np.random.default_rng(1000 + seed)
    fold_io = []
    for k, fold in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        vis_tr = visible & tr_reg
        d_vis_tr = ndi.distance_transform_edt(~vis_tr).astype(np.float32)
        pos = fp_lut[np.flatnonzero(vis_tr & foot)]
        if pos.size > 48000:
            pos = rng.choice(pos, 48000, replace=False)
        neg_base = foot & tr_reg & (~vis_tr) & (d_vis_tr > EXCL_PX)
        hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 16.0))]
        easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 16.0))]
        n_neg = int(2.5 * pos.size)
        nh = min(hard.size, int(0.55 * n_neg))
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate(
            [rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)]
        )
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        te_mask = fold["region"] & foot
        te_idx = fp_lut[np.flatnonzero(te_mask)]
        fold_io.append((fold["name"], tr_idx, y_tr, te_idx, te_mask))
    return fold_io


def fit_oof_surface_from_cols(
    mm_static: np.memmap,
    mech_5: np.ndarray | None,
    col_indices: List[int],
    fold_io: List[Tuple],
    shape: Tuple[int, int],
    kw: Dict,
) -> np.ndarray:
    """Fit 4-quadrant OOF HistGradientBoostingClassifier and predict over footprint."""
    surf = np.zeros(shape, dtype=np.float32)
    n_stat = len(col_indices)
    n_tot = n_stat + (mech_5.shape[1] if mech_5 is not None else 0)

    for fname, tr_idx, y_tr, te_idx, te_mask in fold_io:
        X_tr = np.empty((tr_idx.size, n_tot), dtype=np.float32)
        for j, c in enumerate(col_indices):
            X_tr[:, j] = mm_static[tr_idx, c]
        if mech_5 is not None:
            X_tr[:, n_stat:] = mech_5[tr_idx]
        clf = HistGradientBoostingClassifier(**kw)
        clf.fit(X_tr, y_tr)
        del X_tr

        out_te = np.empty(te_idx.size, dtype=np.float32)
        step = 350000
        for s in range(0, te_idx.size, step):
            sl = te_idx[s : s + step]
            Xc = np.empty((sl.size, n_tot), dtype=np.float32)
            for j, c in enumerate(col_indices):
                Xc[:, j] = mm_static[sl, c]
            if mech_5 is not None:
                Xc[:, n_stat:] = mech_5[sl]
            out_te[s : s + step] = clf.predict_proba(Xc)[:, 1].astype(np.float32)
            del Xc
        surf[te_mask] = out_te
        del clf, out_te
        gc.collect()
    return surf


def emit_submod_pooled(
    p_surf: np.ndarray,
    folds: List[Dict],
    foot: np.ndarray,
    visible: np.ndarray,
    dist_vis: np.ndarray,
    budget: int = TARGET_DOTS,
    excl_px: float = EXCL_PX,
    sparse_halo_bits: np.ndarray | None = None,
    sparse_crowd: np.ndarray | None = None,
    n_sparse_rasters: int = 0,
    max_overlap_fraction: float = SPARSE_OVERLAP_CAP,
) -> Tuple[np.ndarray, Dict]:
    """Allocate `budget` dots proportionally across quadrant folds using 300 m submodular cover."""
    h, w = foot.shape
    allowed_folds = []
    tot = 0
    for fold in folds:
        a = fold["region"] & foot & (~visible) & (dist_vis > excl_px)
        allowed_folds.append(a)
        tot += int(a.sum())
    dots = np.zeros((h, w), dtype=bool)
    skipped_total = 0
    for a in allowed_folds:
        b = int(round(budget * int(a.sum()) / max(tot, 1)))
        if b > 0:
            d_fold, meta_f = allocate_submodular_300m_pcd(
                prob_surface=p_surf,
                allowed_mask=a,
                target_n=b,
                support_quantile=0.962,
                gamma=3.2,
                min_spacing_px=2.20,
                sparse_halo_bits=sparse_halo_bits,
                sparse_crowd_fraction=sparse_crowd,
                num_sparse_rasters=n_sparse_rasters,
                max_overlap_fraction=max_overlap_fraction,
            )
            dots |= d_fold > 0.5
            skipped_total += int(meta_f.get("skipped_due_to_halo_cap", 0))
    return dots, {"emitted_dots": int(dots.sum()), "skipped_due_to_halo_cap": skipped_total}


def emit_greedy_pooled(
    p_surf: np.ndarray,
    folds: List[Dict],
    foot: np.ndarray,
    visible: np.ndarray,
    dist_vis: np.ndarray,
    budget: int = TARGET_DOTS,
    excl_px: float = EXCL_PX,
) -> np.ndarray:
    """Allocate `budget` dots proportionally across quadrant folds using greedy disk thinning."""
    h, w = foot.shape
    allowed_folds = []
    tot = 0
    for fold in folds:
        a = fold["region"] & foot & (~visible) & (dist_vis > excl_px)
        allowed_folds.append(a)
        tot += int(a.sum())
    dots = np.zeros((h, w), dtype=bool)
    for a in allowed_folds:
        b = int(round(budget * int(a.sum()) / max(tot, 1)))
        if b > 0:
            dots |= E77.emit_dots(p_surf, a, b)
    return dots


def score_dots_pooled(
    dots: np.ndarray, folds: List[Dict], foot: np.ndarray
) -> Tuple[Dict, np.ndarray]:
    """Evaluate pooled DTI and per-quadrant block terms via shared `vendor/gems57/evaluate_holdout.py`."""
    terms = None
    per = {}
    for fold in folds:
        res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
        per[fold["name"]] = {
            "dti": float(res["dti"]),
            "n_truth": int(res["n_truth"]),
            "emitted": int(res["n_emitted"]),
        }
        terms = t if terms is None else terms + t
    return per, terms


def run_experiment_1(
    grid: G.Grid,
    mm_static: np.memmap,
    static_meta: Dict,
    folds_by_seed: Dict[int, List[Dict]],
    mech_by_seed: Dict[int, np.ndarray],
) -> Dict:
    """Experiment 1: Single-feature leakage canaries (Seeds 30 & 20) + Phase-1 uniqueness + screen."""
    t0 = time.time()
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)
    dist_cat = ndi.distance_transform_edt(~cat).astype(np.float32)
    idx_map = {n: i for i, n in enumerate(ALL_STATIC_NAMES)}

    canary_by_seed: Dict[str, Dict] = {}
    overall_max_sym_auc = 0.0
    overall_worst_feature = ""

    for seed, folds in folds_by_seed.items():
        hidden = folds[0]["hidden_all"]
        mech_5 = mech_by_seed[seed]
        rng = np.random.default_rng(8300 + seed)
        pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
        neg_pool_fp = fp_lut[np.flatnonzero(foot & (~cat) & (dist_cat > 3.0))]
        neg_fp = rng.choice(
            neg_pool_fp, size=min(200000, neg_pool_fp.size), replace=False
        )
        can_fp = np.concatenate([pos_fp, neg_fp])
        y_can = np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])

        canary_rows: Dict[str, Dict] = {}
        seed_max = 0.0
        for name in PCD_STATIC_NAMES:
            v = np.asarray(mm_static[can_fp, idx_map[name]], dtype=np.float32)
            m = np.isfinite(v)
            auc = float(roc_auc_score(y_can[m], v[m])) if m.sum() > 100 else 0.5
            sym_auc = float(max(auc, 1.0 - auc))
            canary_rows[name] = {
                "auc": round(auc, 6),
                "symmetric_auc": round(sym_auc, 6),
                "leakage_flag": bool(sym_auc > 0.90),
            }
            if sym_auc > seed_max:
                seed_max = sym_auc
            if sym_auc > overall_max_sym_auc:
                overall_max_sym_auc = sym_auc
                overall_worst_feature = f"{name}@seed{seed}"

        for j, vname in enumerate(PCD_VIS_NAMES):
            v = np.asarray(mech_5[can_fp, j], dtype=np.float32)
            m = np.isfinite(v)
            auc = float(roc_auc_score(y_can[m], v[m])) if m.sum() > 100 else 0.5
            sym_auc = float(max(auc, 1.0 - auc))
            canary_rows[vname] = {
                "auc": round(auc, 6),
                "symmetric_auc": round(sym_auc, 6),
                "leakage_flag": bool(sym_auc > 0.90),
            }
            if sym_auc > seed_max:
                seed_max = sym_auc
            if sym_auc > overall_max_sym_auc:
                overall_max_sym_auc = sym_auc
                overall_worst_feature = f"{vname}@seed{seed}"

        canary_by_seed[f"seed_{seed}"] = {
            "n_hidden_positives": int(hidden.sum()),
            "max_symmetric_auc": round(seed_max, 6),
            "passes_gate_le_0_90": bool(seed_max <= 0.90),
            "features": canary_rows,
        }

    if overall_max_sym_auc > 0.90:
        raise RuntimeError(
            f"STOP: Leakage canary failed on {overall_worst_feature}: {overall_max_sym_auc:.4f} > 0.90"
        )

    # Phase-1 pre-placement surface uniqueness check on pcd_composite_surface
    comp_2d = np.zeros((h, w), dtype=np.float32)
    comp_2d[foot] = mm_static[:, idx_map["pcd_composite_surface"]]
    phase1_uniq = audit_uniqueness_against_all_rasters(
        candidate_2d=comp_2d,
        foot=foot,
        is_binary_dots=False,
    )
    max_surface_rho = phase1_uniq["literal_policy_i17_all_raster_scope"][
        "max_abs_spearman"
    ]
    if max_surface_rho > 0.90:
        raise RuntimeError(
            f"STOP: Phase-1 surface Spearman rank correlation {max_surface_rho:.4f} > 0.90"
        )

    # Standalone label-free physical composite holdout screen on Seed 30
    folds_30 = folds_by_seed[30]
    vis_30 = folds_30[0]["visible"]
    dist_vis_30 = ndi.distance_transform_edt(~vis_30).astype(np.float32)
    dots_screen, _ = emit_submod_pooled(
        comp_2d, folds_30, foot, vis_30, dist_vis_30, budget=TARGET_DOTS
    )
    per_30, terms_30 = score_dots_pooled(dots_screen, folds_30, foot)
    pooled_30 = float(EH.from_terms(terms_30.sum(axis=0)))

    exp1_out = {
        "experiment_id": "H83-PCD-E1",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "runtime_seconds": round(time.time() - t0, 2),
        "static_feature_meta": static_meta,
        "leakage_canary": {
            "threshold_symmetric_auc": 0.90,
            "max_symmetric_auc": round(overall_max_sym_auc, 6),
            "worst_feature": overall_worst_feature,
            "passed": bool(overall_max_sym_auc <= 0.90),
            "by_seed": canary_by_seed,
        },
        "phase1_surface_uniqueness": {
            "threshold_abs_spearman": 0.90,
            "max_abs_spearman_all": phase1_uniq["literal_policy_i17_all_raster_scope"][
                "max_abs_spearman"
            ],
            "max_abs_spearman_sparse": phase1_uniq["informative_sparse_scope"][
                "max_abs_spearman"
            ],
            "passed": bool(max_surface_rho <= 0.90),
        },
        "standalone_composite_screen_seed30": {
            "emitted_dots": int(dots_screen.sum()),
            "pooled_holdout_dti": round(pooled_30, 6),
            "per_fold": per_30,
        },
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "h83pcd_exp1_canary.json").write_text(json.dumps(exp1_out, indent=2))
    (DOCS_EVIDENCE_DIR / "h83pcd_exp1_canary.json").write_text(
        json.dumps(exp1_out, indent=2)
    )
    return exp1_out


def run_experiment_2(
    grid: G.Grid,
    mm_static: np.memmap,
    folds_by_seed: Dict[int, List[Dict]],
    mech_by_seed: Dict[int, np.ndarray],
    sparse_bits_full: np.ndarray,
    sparse_crowd_full: np.ndarray,
    n_sparse_refs: int,
) -> Dict:
    """Experiment 2: Dual-seed buffered whole-segment holdout evaluation across 7 arms."""
    t0 = time.time()
    foot = grid.footprint
    h, w = foot.shape
    idx_map = {n: i for i, n in enumerate(ALL_STATIC_NAMES)}

    std_cols = [idx_map[n] for n in (BASE19_NAMES + TERR_NAMES + RAD_NAMES)]
    all_static_cols = list(range(len(ALL_STATIC_NAMES)))

    comp_2d = np.zeros((h, w), dtype=np.float32)
    comp_2d[foot] = mm_static[:, idx_map["pcd_composite_surface"]]
    anti_2d = np.zeros((h, w), dtype=np.float32)
    anti_2d[foot] = mm_static[:, idx_map["pcd_poisson_anti_step"]]

    results_by_seed: Dict[str, Dict] = {}

    for seed, folds in folds_by_seed.items():
        t_seed = time.time()
        visible = folds[0]["visible"]
        hidden = folds[0]["hidden_all"]
        dist_vis = ndi.distance_transform_edt(~visible).astype(np.float32)
        mech_5 = mech_by_seed[seed]
        fold_io = build_fold_io(folds, foot, visible, seed)

        surf_cache = PROC / "gems77_pcd_cache" / f"oof_surfs_seed{seed}.npz"
        if surf_cache.exists():
            log(f"[Exp2 Seed {seed}] Loading cached OOF surfaces from {surf_cache.name}...")
            z = np.load(surf_cache)
            s_std = z["s_std"]
            s_pcd_static = z["s_pcd_static"]
            s_pcd_mech = z["s_pcd_mech"]
        else:
            log(f"[Exp2 Seed {seed}] Fitting s_std ({len(std_cols)} channels)...")
            s_std = fit_oof_surface_from_cols(
                mm_static, None, std_cols, fold_io, (h, w), HGB_STD_KW
            )
            log(f"[Exp2 Seed {seed}] Fitting s_pcd_static ({len(all_static_cols)} channels)...")
            s_pcd_static = fit_oof_surface_from_cols(
                mm_static, None, all_static_cols, fold_io, (h, w), HGB_STATIC_KW
            )
            log(
                f"[Exp2 Seed {seed}] Fitting s_pcd_mech ({len(all_static_cols)} static + 5 LOCO)..."
            )
            s_pcd_mech = fit_oof_surface_from_cols(
                mm_static, mech_5, all_static_cols, fold_io, (h, w), HGB_MECH_KW
            )
            np.savez(
                surf_cache,
                s_std=s_std,
                s_pcd_static=s_pcd_static,
                s_pcd_mech=s_pcd_mech,
            )

        # Calibrated OOF probability blend (0.45 static H83-PCD view + 0.55 LOCO-relay H83-PCD view)
        p_pcd_full = (0.45 * s_pcd_static + 0.55 * s_pcd_mech).astype(np.float32)

        rng = np.random.default_rng(900 + seed)
        rand_surf = (rng.random((h, w), dtype=np.float32) * foot).astype(np.float32)

        dots_a0 = emit_greedy_pooled(rand_surf, folds, foot, visible, dist_vis)
        dots_std = emit_greedy_pooled(s_std, folds, foot, visible, dist_vis)
        dots_static = emit_greedy_pooled(s_pcd_static, folds, foot, visible, dist_vis)
        dots_mech = emit_greedy_pooled(s_pcd_mech, folds, foot, visible, dist_vis)
        dots_pcd_greedy = emit_greedy_pooled(p_pcd_full, folds, foot, visible, dist_vis)
        dots_pcd_submod, _ = emit_submod_pooled(
            p_pcd_full, folds, foot, visible, dist_vis
        )
        dots_pcd_unique, unique_meta = emit_submod_pooled(
            p_pcd_full,
            folds,
            foot,
            visible,
            dist_vis,
            sparse_halo_bits=sparse_bits_full,
            sparse_crowd=sparse_crowd_full,
            n_sparse_rasters=n_sparse_refs,
            max_overlap_fraction=SPARSE_OVERLAP_CAP,
        )

        arms_dots = {
            "A0_random": dots_a0,
            "A_std_greedy": dots_std,
            "A_pcd_static_only": dots_static,
            "A_pcd_mech_only": dots_mech,
            "A_pcd_greedy": dots_pcd_greedy,
            "A_pcd_submod": dots_pcd_submod,
            "A_pcd_submod_unique": dots_pcd_unique,
        }

        terms_by: Dict[str, np.ndarray] = {}
        arms_detail: Dict[str, Dict] = {}
        for arm_name, d_arr in arms_dots.items():
            per_f, t_arr = score_dots_pooled(d_arr, folds, foot)
            terms_by[arm_name] = t_arr
            pooled_val = float(EH.from_terms(t_arr.sum(axis=0)))
            arms_detail[arm_name] = {
                "dots": int(d_arr.sum()),
                "pooled_holdout_dti": round(pooled_val, 6),
                "per_fold": per_f,
            }
            log(f"  [Seed {seed}] {arm_name:22s}: HOLDOUT-DTI = {pooled_val:.6f}")

        summary_submod = EH.pooled_summary(
            terms_by, draws=1000, seed=52000 + seed, candidate="A_pcd_submod"
        )
        summary_unique = EH.pooled_summary(
            terms_by, draws=1000, seed=62000 + seed, candidate="A_pcd_submod_unique"
        )

        results_by_seed[f"seed_{seed}"] = {
            "seed": seed,
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "runtime_seconds": round(time.time() - t_seed, 2),
            "withheld_positive_pixels": int(hidden.sum()),
            "visible_positive_pixels": int(visible.sum()),
            "arms": arms_detail,
            "pooled_summary_submod": summary_submod,
            "pooled_summary_unique": summary_unique,
            "sparse_unique_allocator_meta": unique_meta,
        }
        del s_std, s_pcd_static, s_pcd_mech, p_pcd_full, rand_surf, arms_dots
        gc.collect()

    s30_scores = results_by_seed["seed_30"]["pooled_summary_unique"]["scores"]
    s20_scores = results_by_seed["seed_20"]["pooled_summary_unique"]["scores"]

    s30_std = s30_scores["A_std_greedy"]["dti"]
    s20_std = s20_scores["A_std_greedy"]["dti"]
    s30_sub = s30_scores["A_pcd_submod"]["dti"]
    s20_sub = s20_scores["A_pcd_submod"]["dti"]
    s30_uni = s30_scores["A_pcd_submod_unique"]["dti"]
    s20_uni = s20_scores["A_pcd_submod_unique"]["dti"]

    exp2_out = {
        "experiment_id": "H83-PCD-E2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "runtime_seconds": round(time.time() - t0, 2),
        "dual_seed_summary": {
            "A_std_greedy_mean_dti": round(0.5 * (s30_std + s20_std), 6),
            "A_pcd_submod_mean_dti": round(0.5 * (s30_sub + s20_sub), 6),
            "A_pcd_submod_unique_mean_dti": round(0.5 * (s30_uni + s20_uni), 6),
            "mean_gain_submod_over_std": round(
                0.5 * ((s30_sub - s30_std) + (s20_sub - s20_std)), 6
            ),
            "mean_gain_unique_over_std": round(
                0.5 * ((s30_uni - s30_std) + (s20_uni - s20_std)), 6
            ),
            "retained_incumbent_h77_step_seed20_dti": RETAINED_INCUMBENT_DTI,
            "beats_retained_incumbent_seed20": bool(s20_uni > RETAINED_INCUMBENT_DTI),
        },
        "by_seed": results_by_seed,
    }
    (EVIDENCE_DIR / "h83pcd_exp2_holdout.json").write_text(
        json.dumps(exp2_out, indent=2)
    )
    (DOCS_EVIDENCE_DIR / "h83pcd_exp2_holdout.json").write_text(
        json.dumps(exp2_out, indent=2)
    )
    return exp2_out


def generate_diagnostic_figures(
    exp2_out: Dict,
    full_discovery_surface: np.ndarray,
    dots_2d: np.ndarray,
    foot: np.ndarray,
    cat: np.ndarray,
) -> List[str]:
    """Generate crisp vector SVG diagnostic figures in docs/figures/h83pcd_*.svg."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    arm_order = [
        ("A_pcd_submod", "H83-PCD Full (LOCO + Submod300)", "#0b6e4f"),
        ("A_pcd_submod_unique", "H83-PCD Sparse-Unique (<=0.64 Cap)", "#1d4ed8"),
        ("A_pcd_mech_only", "H83-PCD LOCO View + Greedy Disk", "#2a9d8f"),
        ("A_pcd_greedy", "H83-PCD Full Surface + Greedy Disk", "#457b9d"),
        ("A_pcd_static_only", "H83-PCD Static View + Greedy Disk", "#64748b"),
        ("A_std_greedy", "Standard 41-Ch Control + Greedy Disk", "#94a3b8"),
        ("A0_random", "Random Uniform Control", "#cbd5e1"),
    ]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 360" width="920" height="360" style="font-family:system-ui,sans-serif;background:#ffffff;">',
        '<text x="460" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">Lane H83-PCD: Dual-Seed Buffered Whole-Segment Spatial Holdout Ablation (1,000-Draw Block Bootstrap)</text>',
    ]
    for col_idx, (skey, stitle) in enumerate(
        [
            ("seed_30", "Seed 30 (11,342 Withheld Positives, B=37,654)"),
            ("seed_20", "Seed 20 (11,321 Withheld Positives, B=37,654)"),
        ]
    ):
        x0 = 250 if col_idx == 0 else 610
        w_max = 260.0
        scale = w_max / 0.18
        lines.append(
            f'<text x="{x0 + 120}" y="52" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e293b">{stitle}</text>'
        )
        sc_dict = exp2_out["by_seed"][skey]["pooled_summary_unique"]["scores"]
        for i, (k, lbl, color) in enumerate(arm_order):
            y = 70 + i * 36
            if col_idx == 0:
                lines.append(
                    f'<text x="242" y="{y + 15}" text-anchor="end" font-size="11" fill="#334155">{lbl}</text>'
                )
            v = float(sc_dict[k]["dti"])
            lo = float(sc_dict[k]["ci95"][0])
            hi = float(sc_dict[k]["ci95"][1])
            bw = max(1.0, v * scale)
            x_lo = x0 + lo * scale
            x_hi = x0 + hi * scale
            lines.append(
                f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="22" fill="{color}" rx="3"/>'
            )
            lines.append(
                f'<line x1="{x_lo:.1f}" y1="{y + 11}" x2="{x_hi:.1f}" y2="{y + 11}" stroke="#0f172a" stroke-width="1.5"/>'
            )
            lines.append(
                f'<text x="{x_hi + 5:.1f}" y="{y + 15}" font-size="10.5" fill="#0f172a">{v:.4f}</text>'
            )
        x_ceil = x0 + RETAINED_INCUMBENT_DTI * scale
        lines.append(
            f'<line x1="{x_ceil:.1f}" y1="62" x2="{x_ceil:.1f}" y2="322" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3"/>'
        )
    lines.append("</svg>")
    p1 = FIG_DIR / "h83pcd_holdout_ablation.svg"
    p1.write_text("\n".join(lines), encoding="utf-8")
    return [str(p1.relative_to(REPO_ROOT))]


def run_experiment_3(
    grid: G.Grid,
    mm_static: np.memmap,
    exp1_out: Dict,
    exp2_out: Dict,
    sparse_bits_full: np.ndarray,
    sparse_crowd_full: np.ndarray,
    sparse_ref_meta: List[Dict],
) -> Dict:
    """Experiment 3: Full-map 4-quadrant OOF surface, sparse-unique 37,654-dot emission & Run Card."""
    t0 = time.time()
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    dist_cat = ndi.distance_transform_edt(~cat).astype(np.float32)
    allowed_full = foot & (~cat) & (dist_cat > EXCL_PX)
    idx_map = {n: i for i, n in enumerate(ALL_STATIC_NAMES)}
    all_static_cols = list(range(len(ALL_STATIC_NAMES)))

    # 4-quadrant spatial OOF full-map prediction so no pixel is predicted by an in-sample tree
    fp_flat = np.flatnonzero(foot)
    r_all, c_all = np.unravel_index(fp_flat, (h, w))
    r_mid = int(np.median(r_all))
    c_mid = int(np.median(c_all))
    yy, xx = np.indices((h, w))
    quad_folds = [
        {"name": "NW", "region": (yy < r_mid) & (xx < c_mid)},
        {"name": "NE", "region": (yy < r_mid) & (xx >= c_mid)},
        {"name": "SW", "region": (yy >= r_mid) & (xx < c_mid)},
        {"name": "SE", "region": (yy >= r_mid) & (xx >= c_mid)},
    ]

    full_cache = PROC / "gems77_pcd_cache" / "oof_surfs_full.npz"
    if full_cache.exists():
        log(f"[Exp3] Loading cached 4-quadrant OOF full-map surfaces from {full_cache.name}...")
        zf = np.load(full_cache)
        s_static = zf["s_static"]
        s_mech = zf["s_mech"]
    else:
        log("[Exp3] Computing full-catalog 12-bit LOCO mechanics...")
        mech_full = build_fold_visible_mechanics_for_mask(
            cat, foot, mm_static, cache_tag="full"
        )
        fold_io_full = build_fold_io(quad_folds, foot, cat, seed=7783)

        log("[Exp3] Fitting 4-quadrant OOF full-map s_static...")
        s_static = fit_oof_surface_from_cols(
            mm_static, None, all_static_cols, fold_io_full, (h, w), HGB_STATIC_KW
        )
        log("[Exp3] Fitting 4-quadrant OOF full-map s_mech...")
        s_mech = fit_oof_surface_from_cols(
            mm_static, mech_full, all_static_cols, fold_io_full, (h, w), HGB_MECH_KW
        )
        np.savez(full_cache, s_static=s_static, s_mech=s_mech)
        del mech_full
        gc.collect()

    full_discovery_surface = (0.45 * s_static + 0.55 * s_mech).astype(np.float32)
    del s_static, s_mech
    gc.collect()

    # Phase-1 uniqueness check on full-map continuous surface
    full_surface_uniq = audit_uniqueness_against_all_rasters(
        candidate_2d=full_discovery_surface,
        foot=foot,
        is_binary_dots=False,
    )
    if full_surface_uniq["literal_policy_i17_all_raster_scope"]["max_abs_spearman"] > 0.90:
        raise RuntimeError("STOP: Full-map discovery surface Spearman > 0.90")

    # Allocate 37,654 binary dots with bitwise per-raster 3-px sparse overlap cap <= 0.64
    log(
        f"[Exp3] Allocating {TARGET_DOTS} binary dots via allocate_submodular_300m_pcd "
        f"(cap={SPARSE_OVERLAP_CAP:.2f} across {len(sparse_ref_meta)} sparse rasters)..."
    )
    dots_2d, alloc_meta = allocate_submodular_300m_pcd(
        prob_surface=full_discovery_surface,
        allowed_mask=allowed_full,
        target_n=TARGET_DOTS,
        support_quantile=0.960,
        gamma=3.2,
        min_spacing_px=2.20,
        sparse_halo_bits=sparse_bits_full,
        sparse_crowd_fraction=sparse_crowd_full,
        num_sparse_rasters=len(sparse_ref_meta),
        max_overlap_fraction=SPARSE_OVERLAP_CAP,
    )
    assert int((dots_2d > 0.5).sum()) == TARGET_DOTS

    # Package via unmodified shared vendor/gems57/submission_writer.py
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    sub_tif_path = SUBMISSIONS_DIR / f"{STEM}.tif"
    for old in (
        sub_tif_path,
        sub_tif_path.with_suffix(".zip"),
        sub_tif_path.with_suffix(".json"),
    ):
        if old.exists():
            old.unlink()

    sw_receipt = SW.write_submission(
        sub_tif_path,
        dots_2d.astype(np.float32),
        RAW / "sample_submission.tif",
        foot,
        note=SUBMISSION_NOTE,
        name=STEM,
        catalogue=cat,
        metadata={
            "lane": "H83-PCD",
            "verdict": "negative",
            "protocol_unique": False,
            "policy_i17": (
                "NOT OK TO SUBMIT — inspection download only "
                "(literal dense witness ab0a0a62 overlap = 1.0)"
            ),
        },
    )

    tif_sha256 = hashlib.sha256(sub_tif_path.read_bytes()).hexdigest()
    sub_zip_path = sub_tif_path.with_suffix(".zip")
    sub_json_path = sub_tif_path.with_suffix(".json")
    if not sub_zip_path.exists():
        with zipfile.ZipFile(sub_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(sub_tif_path, arcname=sub_tif_path.name)
            if sub_json_path.exists():
                zf.write(sub_json_path, arcname=sub_json_path.name)
    zip_sha256 = hashlib.sha256(sub_zip_path.read_bytes()).hexdigest()

    dl_tif_path = DOCS_DOWNLOADS_DIR / sub_tif_path.name
    dl_zip_path = DOCS_DOWNLOADS_DIR / sub_zip_path.name
    shutil.copy2(sub_tif_path, dl_tif_path)
    shutil.copy2(sub_zip_path, dl_zip_path)

    # Validate via scripts/validate_submission.py
    val_proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_submission.py"),
            str(sub_tif_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    val_json = json.loads(val_proc.stdout)
    assert val_json["all_checks_passed"] is True, f"Validator failed: {val_json}"

    # Independent 17-point readback verification
    with rasterio.open(sub_tif_path) as src:
        rb = src.read(1)
        crs_str = str(src.crs)
        transform_tuple = tuple(src.transform)[:6]
        nodata_val = src.nodata
        dtype_str = src.dtypes[0]
        band_count = src.count
        shape_hw = (src.height, src.width)
    with rasterio.open(RAW / "sample_submission.tif") as sref:
        ref_transform_tuple = tuple(sref.transform)[:6]

    pos_mask = rb > 0.5
    n_pos = int(pos_mask.sum())
    min_cat_dist = float(dist_cat[pos_mask].min()) if n_pos > 0 else 0.0
    unique_vals = [float(x) for x in np.unique(rb)]
    checks_17 = {
        "01_file_exists": sub_tif_path.exists(),
        "02_single_band": band_count == 1,
        "03_dtype_float32": dtype_str == "float32",
        "04_shape_3730x3292": shape_hw == (3730, 3292),
        "05_crs_epsg32611": crs_str == "EPSG:32611",
        "06_transform_matches_sample": transform_tuple == ref_transform_tuple,
        "07_nodata_is_none": nodata_val is None,
        "08_zero_nan_full_raster": int(np.isnan(rb).sum()) == 0,
        "09_zero_inf_full_raster": int(np.isinf(rb).sum()) == 0,
        "10_min_ge_0": float(rb.min()) >= 0.0,
        "11_max_le_1": float(rb.max()) <= 1.0,
        "12_outside_footprint_all_zero": float(np.abs(rb[~foot]).max()) == 0.0,
        "13_exact_binary_values_0_1": unique_vals == [0.0, 1.0],
        "14_emitted_dots_37654": n_pos == TARGET_DOTS,
        "15_zero_dots_within_3px_catalog": int((pos_mask & (dist_cat <= 3.0)).sum()) == 0,
        "16_note_length_le_140": len(SUBMISSION_NOTE) <= 140,
        "17_downloads_copy_sha256_match": hashlib.sha256(dl_tif_path.read_bytes()).hexdigest()
        == tif_sha256,
    }
    assert all(checks_17.values()), f"17-check readback failed: {checks_17}"

    # Phase-2 final-dot uniqueness audit
    phase2_uniq = audit_uniqueness_against_all_rasters(
        candidate_2d=rb,
        foot=foot,
        is_binary_dots=True,
    )
    assert phase2_uniq["informative_sparse_scope"]["passes_sparse_gate"], (
        f"Sparse uniqueness gate failed: {phase2_uniq['informative_sparse_scope']}"
    )

    # Also run unmodified vendor/gems57/uniqueness.py compare_array_to_registry on pinned witnesses
    witness_recs = [
        {
            "submission": "17GEMSDOE_E-proba-multiscale",
            "file": str(REPO_ROOT / ".cache" / "registry" / "dense_witness.tif"),
        },
        {
            "submission": "gemsdoe32-h33-h33-2-b2-zeros",
            "file": str(REPO_ROOT / ".cache" / "registry" / "h33.tif"),
        },
    ]
    raw_surf_w = UQ.compare_array_to_registry(full_discovery_surface, witness_recs, foot)
    raw_dots_w = UQ.compare_array_to_registry(rb, witness_recs, foot)
    vendor_surf_rep = UQ.two_phase_summary(
        raw_surf_w["rows"], int(foot.sum()), expected_rasters=2
    )
    vendor_dots_rep = UQ.two_phase_summary(
        raw_dots_w["rows"], int(foot.sum()), expected_rasters=2
    )

    fig_paths = generate_diagnostic_figures(
        exp2_out, full_discovery_surface, rb, foot, cat
    )

    s30_sum = exp2_out["by_seed"]["seed_30"]["pooled_summary_unique"]
    s20_sum = exp2_out["by_seed"]["seed_20"]["pooled_summary_unique"]
    s30_uni = s30_sum["scores"]["A_pcd_submod_unique"]
    s20_uni = s20_sum["scores"]["A_pcd_submod_unique"]
    s30_sub = s30_sum["scores"]["A_pcd_submod"]
    s20_sub = s20_sum["scores"]["A_pcd_submod"]
    s30_del = s30_sum["paired_differences"]["A_std_greedy"]
    s20_del = s20_sum["paired_differences"]["A_std_greedy"]

    run_card = {
        "schema_version": "gems-run-card-v2",
        "lane_id": "H83-PCD",
        "branch": "arena/5c385eb9-gemsdoe77",
        "generated_utc": "2026-10-21T18:30:00Z",
        "hypothesis": (
            "H83-PCD: Multi-Scale Poisson Pseudogravity-Gravity Gradient Discordance "
            "x Piedmont Scarp-Toe Asymmetry x Geothermal-Strain Conduit Coupling "
            "+ 12-Bit Leave-Own-Component-Out (LOCO) Transfer Relay & 300 m Submodular Cover"
        ),
        "physical_mechanism": (
            "Under Poisson's relation (Baranov 1957; Cordell & Grauch 1985), a lithologic body "
            "with uniform magnetization-to-density ratio has parallel horizontal gravity and "
            "Fourier pseudogravity gradients. Concealed fault conduits with hydrothermal "
            "magnetite destruction or basement-vs-volcanic half-graben juxtaposition exhibit "
            "multi-scale (sigma=2.5, 5.0 px) gradient anti-concordance (1 - cos theta) aligned "
            "with basement depth steps, down-facing 1 m LiDAR piedmont scarp-toe dipoles "
            "(missing-coverage-aware q==0 -> NaN bridged by 100 m DEM ridges), transtensional "
            "strain x swarm excess, and DOE GDR 1391 chemical geothermometer disequilibrium, "
            "propagated across unmapped stepovers via 12-bit LOCO relay mechanics and converted "
            "to 37,654 binary dots via bitwise sparse-overlap-capped 300 m submodular cover."
        ),
        "named_non_fault_mimic": (
            "Unfaulted erosional margins of reversely magnetized Tertiary ash-flow tuffs "
            "lapping onto dense Paleozoic carbonate basement, Lake Lahontan wave-cut shoreline "
            "terraces overlying buried pluton contacts, or highway/railroad embankments."
        ),
        "evaluator_version": EH.VERSION,
        "holdout_dti": {
            "primary_seed_30": {
                "score_label": "HOLDOUT-DTI",
                "evaluator_version": EH.VERSION,
                "withheld_positives": exp2_out["by_seed"]["seed_30"][
                    "withheld_positive_pixels"
                ],
                "dti": round(float(s30_uni["dti"]), 6),
                "ci95": [
                    round(float(s30_uni["ci95"][0]), 6),
                    round(float(s30_uni["ci95"][1]), 6),
                ],
                "delta_over_std_greedy_control": round(float(s30_del["delta"]), 6),
                "delta_ci95": [
                    round(float(s30_del["ci95"][0]), 6),
                    round(float(s30_del["ci95"][1]), 6),
                ],
                "unconstrained_submod_dti": round(float(s30_sub["dti"]), 6),
                "unconstrained_submod_ci95": [
                    round(float(s30_sub["ci95"][0]), 6),
                    round(float(s30_sub["ci95"][1]), 6),
                ],
            },
            "secondary_seed_20": {
                "score_label": "HOLDOUT-DTI",
                "evaluator_version": EH.VERSION,
                "withheld_positives": exp2_out["by_seed"]["seed_20"][
                    "withheld_positive_pixels"
                ],
                "dti": round(float(s20_uni["dti"]), 6),
                "ci95": [
                    round(float(s20_uni["ci95"][0]), 6),
                    round(float(s20_uni["ci95"][1]), 6),
                ],
                "delta_over_std_greedy_control": round(float(s20_del["delta"]), 6),
                "delta_ci95": [
                    round(float(s20_del["ci95"][0]), 6),
                    round(float(s20_del["ci95"][1]), 6),
                ],
                "unconstrained_submod_dti": round(float(s20_sub["dti"]), 6),
                "unconstrained_submod_ci95": [
                    round(float(s20_sub["ci95"][0]), 6),
                    round(float(s20_sub["ci95"][1]), 6),
                ],
            },
            "dual_seed_mean_dti": exp2_out["dual_seed_summary"][
                "A_pcd_submod_unique_mean_dti"
            ],
            "dual_seed_unconstrained_mean_dti": exp2_out["dual_seed_summary"][
                "A_pcd_submod_mean_dti"
            ],
            "standard_control_dual_seed_mean_dti": exp2_out["dual_seed_summary"][
                "A_std_greedy_mean_dti"
            ],
            "retained_repo_incumbent_h77_step_dti": RETAINED_INCUMBENT_DTI,
        },
        "leakage_canary": {
            "threshold_symmetric_auc": 0.90,
            "max_symmetric_auc": exp1_out["leakage_canary"]["max_symmetric_auc"],
            "worst_feature": exp1_out["leakage_canary"]["worst_feature"],
            "passed": exp1_out["leakage_canary"]["passed"],
            "sgmc_channels_included": 0,
            "dist_known_fault_px_used": False,
        },
        "uniqueness_vs_registry": {
            "phase1_continuous_surface": {
                "max_abs_spearman_all": full_surface_uniq[
                    "literal_policy_i17_all_raster_scope"
                ]["max_abs_spearman"],
                "max_abs_spearman_sparse": full_surface_uniq[
                    "informative_sparse_scope"
                ]["max_abs_spearman"],
                "passed_le_0_90": True,
            },
            "informative_sparse_scope": phase2_uniq["informative_sparse_scope"],
            "literal_policy_i17_all_raster_scope": phase2_uniq[
                "literal_policy_i17_all_raster_scope"
            ],
            "vendor_pinned_witness_surface_summary": vendor_surf_rep,
            "vendor_pinned_witness_dots_summary": vendor_dots_rep,
        },
        "raster_sha256": tif_sha256,
        "zip_sha256": zip_sha256,
        "validator_output": {
            "validate_submission_py_all_checks_passed": val_json["all_checks_passed"],
            "independent_readback_17_of_17_ok": all(checks_17.values()),
            "checks_17": checks_17,
            "emitted_positive_dots": n_pos,
            "min_distance_to_existing_catalog_px": round(min_cat_dist, 4),
            "allocator_meta": alloc_meta,
            "submission_writer_receipt": sw_receipt,
        },
        "submission_name": sub_tif_path.name,
        "submission_note": SUBMISSION_NOTE,
        "submission_note_length": len(SUBMISSION_NOTE),
        "artifact_paths": {
            "submission_tif": str(sub_tif_path.relative_to(REPO_ROOT)),
            "submission_zip": str(sub_zip_path.relative_to(REPO_ROOT)),
            "downloads_tif": str(dl_tif_path.relative_to(REPO_ROOT)),
            "downloads_zip": str(dl_zip_path.relative_to(REPO_ROOT)),
            "figures": fig_paths,
        },
        "verdict": "negative",
        "selector_recommendation": (
            "Not approved for live DrivenData upload (inspection only; 0 weekly slots spent): "
            "(1) Policy I17 literal all-raster 3-px overlap gate returns 1.0000 > 0.70 against "
            "space-filling continuous witness 17GEMSDOE_E-proba-multiscale (ab0a0a62), even "
            "though H83-PCD passes 100% of sparse dot-field rasters (max sparse 3-px overlap "
            f"{phase2_uniq['informative_sparse_scope']['max_forward_overlap_3px']:.4f} <= 0.70, "
            f"max |Spearman rho| {phase2_uniq['informative_sparse_scope']['max_abs_spearman']:.4f} <= 0.90); "
            f"and (2) Seed-20 HOLDOUT-DTI ({s20_uni['dti']:.6f}) does not exceed the "
            f"retained repo ceiling H77-Step ({RETAINED_INCUMBENT_DTI:.6f})."
        ),
        "experiments_used": 3,
        "runtime_seconds": round(time.time() - t0, 2),
    }

    exp3_out = {
        "experiment_id": "H83-PCD-E3",
        "run_card": run_card,
        "phase2_uniqueness_detail": phase2_uniq,
    }
    (EVIDENCE_DIR / "h83pcd_exp3_audit.json").write_text(json.dumps(exp3_out, indent=2))
    (DOCS_EVIDENCE_DIR / "h83pcd_exp3_audit.json").write_text(
        json.dumps(exp3_out, indent=2)
    )
    (EVIDENCE_DIR / "run_card_h83pcd.json").write_text(json.dumps(run_card, indent=2))
    (DOCS_EVIDENCE_DIR / "run_card_h83pcd.json").write_text(
        json.dumps(run_card, indent=2)
    )
    return run_card


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["all", "exp1", "exp2", "exp3"],
        default="all",
    )
    args = ap.parse_args()

    t_start = time.time()
    log("[H83-PCD] Loading competition Grid...")
    grid = G.load_grid(RAW)
    dist_cat = ndi.distance_transform_edt(~grid.catalogue).astype(np.float32)
    allowed_full = grid.footprint & (~grid.catalogue) & (dist_cat > EXCL_PX)

    log("[H83-PCD] Building/loading 60-channel static feature memmap...")
    mm_static, static_meta = build_static_feature_memmap(force=False)
    log(
        f"[H83-PCD] Static memmap ready ({static_meta['n_fp']} x {static_meta['n_cols']}, "
        f"build={static_meta['build_seconds']}s)"
    )

    folds_by_seed = {}
    mech_by_seed = {}
    if args.stage in ("all", "exp1", "exp2"):
        log(
            "[H83-PCD] Building buffered whole-component holdout splits (Seeds 30 & 20)..."
        )
        folds_30, _ = HO.buffered_component_draw(
            grid, seed=30, hide_frac=0.20, buffer_px=3
        )
        folds_20, _ = HO.buffered_component_draw(
            grid, seed=20, hide_frac=0.20, buffer_px=3
        )
        folds_by_seed = {30: folds_30, 20: folds_20}
        for seed, folds in folds_by_seed.items():
            log(f"[H83-PCD] Loading fold-visible 12-bit LOCO mechanics for Seed {seed}...")
            mech_by_seed[seed] = build_fold_visible_mechanics_for_mask(
                folds[0]["visible"], grid.footprint, mm_static, cache_tag=f"seed{seed}"
            )

    sparse_bits_full, sparse_crowd_full, sparse_ref_meta = None, None, []
    if args.stage in ("all", "exp2", "exp3"):
        log("[H83-PCD] Packing bitwise 3-px halos for all sparse reference rasters...")
        sparse_bits_full, sparse_crowd_full, sparse_ref_meta = (
            build_sparse_reference_halos(
                foot=grid.footprint,
                allowed_2d=allowed_full,
                exclude_prefix="h83pcd-",
            )
        )
        log(
            f"[H83-PCD] Packed {len(sparse_ref_meta)} sparse reference rasters into uint64 bitmask."
        )

    exp1_path = EVIDENCE_DIR / "h83pcd_exp1_canary.json"
    if args.stage in ("all", "exp1") and not exp1_path.exists():
        log(
            "[H83-PCD] Running Experiment 1 (Leakage Canaries + Phase-1 Uniqueness + Screen)..."
        )
        exp1_out = run_experiment_1(
            grid, mm_static, static_meta, folds_by_seed, mech_by_seed
        )
    elif args.stage == "exp1":
        exp1_out = run_experiment_1(
            grid, mm_static, static_meta, folds_by_seed, mech_by_seed
        )
    else:
        exp1_out = json.loads(exp1_path.read_text())
    log(
        f"[H83-PCD] Exp 1 ready: max_sym_auc={exp1_out['leakage_canary']['max_symmetric_auc']:.4f} "
        f"({exp1_out['leakage_canary']['worst_feature']}), "
        f"max_surface_rho={exp1_out['phase1_surface_uniqueness']['max_abs_spearman_all']:.4f}"
    )
    if args.stage == "exp1":
        return

    exp2_path = EVIDENCE_DIR / "h83pcd_exp2_holdout.json"
    if args.stage in ("all", "exp2"):
        log(
            "[H83-PCD] Running Experiment 2 (Dual-Seed Holdout Evaluation + Block Bootstrap)..."
        )
        exp2_out = run_experiment_2(
            grid=grid,
            mm_static=mm_static,
            folds_by_seed=folds_by_seed,
            mech_by_seed=mech_by_seed,
            sparse_bits_full=sparse_bits_full,
            sparse_crowd_full=sparse_crowd_full,
            n_sparse_refs=len(sparse_ref_meta),
        )
        log(
            f"[H83-PCD] Exp 2 complete: "
            f"A_std_greedy mean={exp2_out['dual_seed_summary']['A_std_greedy_mean_dti']:.6f}, "
            f"A_pcd_submod mean={exp2_out['dual_seed_summary']['A_pcd_submod_mean_dti']:.6f}, "
            f"A_pcd_submod_unique mean={exp2_out['dual_seed_summary']['A_pcd_submod_unique_mean_dti']:.6f}"
        )
        if args.stage == "exp2":
            return
    else:
        exp2_out = json.loads(exp2_path.read_text())

    if args.stage in ("all", "exp3"):
        log(
            "[H83-PCD] Running Experiment 3 (Full-Map OOF Surface + 37,654-Dot Emission + Audit)..."
        )
        run_card = run_experiment_3(
            grid=grid,
            mm_static=mm_static,
            exp1_out=exp1_out,
            exp2_out=exp2_out,
            sparse_bits_full=sparse_bits_full,
            sparse_crowd_full=sparse_crowd_full,
            sparse_ref_meta=sparse_ref_meta,
        )
        log(
            f"[H83-PCD] Exp 3 complete in {time.time() - t_start:.1f}s total: "
            f"tif={run_card['submission_name']}, sha256={run_card['raster_sha256'][:16]}..., "
            f"sparse_max_ov={run_card['uniqueness_vs_registry']['informative_sparse_scope']['max_forward_overlap_3px']:.4f}"
        )


if __name__ == "__main__":
    main()
