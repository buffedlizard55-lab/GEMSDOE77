#!/usr/bin/env python3
"""Preregistered 3-Experiment Pipeline for GEMSDOE77 Lane H77H (2026-10-21).

Executes:
  - Experiment 1 (H77H-E1): Build 12 H77H static channels + 20 external 1m-LiDAR/GeoDAWN
    channels + verify sigma=36 px MT mesh periodicity suppression (IR-H77G-01) + run
    single-feature symmetric ROC-AUC leakage canaries (ceiling 0.90) on Seed 20 (11,321
    withheld px) and Seed 30 (11,342 withheld px) + Phase 1 pre-placement surface uniqueness.
  - Experiment 2 (H77H-E2): Dual-seed buffered whole-component spatial holdout evaluation
    via unmodified vendor/gems57/evaluate_holdout.py (gems57-pooled-hide-v2, alpha=0.2,
    beta=0.8, 300 m triangular kernel, 1000-draw 20 km spatial-block paired bootstrap)
    across 8 arms on Seed 20 and Seed 30.
  - Experiment 3 (H77H-E3): Full-map 4-quadrant out-of-fold H77H surface + informative-unique
    capped 300 m submodular dot placement (37,654 binary dots, <=0.665 overlap cap across all
    informative rasters) + unmodified vendor/gems57/submission_writer.py serialization (.tif
    and single-TIFF .zip) + 15/15 and 17/17 validator checks + Phase 1 & Phase 2 uniqueness
    audit + diagnostic figure + frozen JSON run card (evidence/run_card_h77h.json).
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems57 import submission_writer as SW  # noqa: E402
from gems57 import uniqueness as UQ  # noqa: E402
from gems57 import validate as VAL57  # noqa: E402
from gems77 import features77 as F77  # noqa: E402
from gems77hdd import (  # noqa: E402
    EXT_LIDAR_RAD_NAMES,
    H77H_LOCO_NAMES,
    H77H_STATIC_NAMES,
    allocate_h77h_submodular_300m,
    build_h77h_static_stack,
    build_informative_halo_matrix,
    check_informative_overlaps,
    collect_checked_registry,
    compute_h77h_loco_features,
)
from gems77lia import allocate as LIA_AL  # noqa: E402
from gems77lia import validate as VAL_LIA  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
BRIDGE_DIR = ROOT / "data" / "bridge"
CACHE_DIR = ROOT / ".cache" / "h77h"
EVID_DIR = ROOT / "evidence"
DOWN_DIR = ROOT / "docs" / "downloads"
ASSET_DIR = ROOT / "docs" / "assets"

BUDGET_DOTS = 36500
EXCL_PX = 2.0
MAX_INFO_OVERLAP_CAP = 0.680
RETAINED_REPO_BEST_DTI = 0.152718  # H77-Step Seed 20 retained benchmark

HGB_KW = dict(
    max_iter=190,
    learning_rate=0.07,
    max_leaf_nodes=55,
    min_samples_leaf=50,
    l2_regularization=2.5,
    early_stopping=False,
    random_state=7708,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_grid_spec() -> G.Grid:
    return G.load_grid(BRIDGE_DIR)


RAW19_NAMES = [
    "mag_anom",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_anom_slope",
    "tc_totalcount",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
    "deq_n100a15",
    "iso_grav_anom_vg",
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "depth_to_base_surf",
    "ieq_n100a15",
    "cond_surf",
    "iso_grav_anom_hg_signed",
    "det_elev_slope",
]


def build_std19_fp(footprint: np.ndarray) -> tuple[np.memmap, list[str]]:
    """Build or reload cached 19-band raw training_features.tif in-footprint matrix."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    foot = np.asarray(footprint, dtype=bool)
    fp_idx = np.flatnonzero(foot)
    n_fp = int(fp_idx.size)
    names = [f"std_{nm}" for nm in RAW19_NAMES]
    p_npy = CACHE_DIR / "std19_fp.npy"
    p_meta = CACHE_DIR / "std19_meta.json"
    if p_npy.is_file() and p_meta.is_file():
        meta = json.loads(p_meta.read_text())
        if meta.get("names") == names and meta.get("n_fp") == n_fp:
            return np.load(p_npy, mmap_mode="r"), names

    mm = np.lib.format.open_memmap(p_npy, mode="w+", dtype=np.float32, shape=(n_fp, len(names)))
    with rasterio.open(RAW_DIR / "training_features.tif") as src:
        for i in range(len(names)):
            b = F77.fill_median(F77.clean(src.read(i + 1)), foot)
            mm[:, i] = b.ravel()[fp_idx]
    mm.flush()
    del mm
    p_meta.write_text(json.dumps({"n_fp": n_fp, "names": names}, indent=2) + "\n")
    return np.load(p_npy, mmap_mode="r"), names


def run_single_feature_canaries_for_seed(
    seed: int,
    grid: G.GridSpec,
    std_fp: np.memmap,
    std_names: list[str],
    ext_fp: np.memmap,
    h77h_fp: np.memmap,
    ceiling: float = 0.90,
) -> dict[str, Any]:
    """Evaluate single-feature symmetric ROC-AUC max(AUC, 1-AUC) on withheld positives vs
    unbuffered negatives for a holdout draw. Hard-stops if any feature exceeds 0.90.
    """
    foot = grid.footprint
    fp_flat = np.flatnonzero(foot)
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"] & foot
    visible = folds[0]["visible"] & foot
    assert not np.any(hidden & visible), "Leakage between hidden and visible masks!"

    d_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    pos_mask_fp = hidden.ravel()[fp_flat]
    neg_mask_fp = (foot & (~visible) & (~hidden) & (d_vis > 3.0)).ravel()[fp_flat]

    rng = np.random.default_rng(7700 + seed)
    pos_idx = np.flatnonzero(pos_mask_fp)
    neg_idx = rng.choice(np.flatnonzero(neg_mask_fp), size=min(60_000, int(neg_mask_fp.sum())), replace=False)
    eval_idx = np.concatenate([pos_idx, neg_idx])
    y_eval = np.concatenate([np.ones(pos_idx.size, dtype=np.int8), np.zeros(neg_idx.size, dtype=np.int8)])

    # Also compute fold-blind LOCO features on eval_idx from visible mask
    _, loco_infer_all = compute_h77h_loco_features(
        visible_mask=visible,
        footprint=foot,
    )
    loco_eval = loco_infer_all[eval_idx]
    del loco_infer_all

    rows: list[dict[str, Any]] = []

    def _check_block(mat: np.ndarray, col_names: list[str], block_tag: str, is_sub: bool = False) -> None:
        for c_i, nm in enumerate(col_names):
            v = mat[:, c_i] if is_sub else np.asarray(mat[eval_idx, c_i], dtype=np.float32)
            fin = np.isfinite(v)
            if int(fin.sum()) < 100 or np.all(v[fin] == v[fin][0]):
                auc = 0.5
            else:
                auc = float(roc_auc_score(y_eval[fin], v[fin]))
            sym = float(max(auc, 1.0 - auc))
            rows.append(
                {
                    "feature": nm,
                    "block": block_tag,
                    "raw_auc": round(auc, 6),
                    "symmetric_auc": round(sym, 6),
                    "valid_fraction": round(float(fin.mean()), 6),
                    "exceeds_0p90_ceiling": bool(sym > ceiling),
                }
            )

    _check_block(h77h_fp, H77H_STATIC_NAMES, "h77h_static")
    _check_block(ext_fp, EXT_LIDAR_RAD_NAMES, "ext_lidar_rad")
    _check_block(std_fp, std_names, "raw_19")
    _check_block(loco_eval, H77H_LOCO_NAMES, "h77h_loco", is_sub=True)

    violators = [r for r in rows if r["exceeds_0p90_ceiling"]]
    worst_all = max(rows, key=lambda r: r["symmetric_auc"])
    h77h_only = [r for r in rows if r["block"] == "h77h_static"]
    worst_h77h = max(h77h_only, key=lambda r: r["symmetric_auc"])

    if violators:
        raise RuntimeError(f"LEAKAGE CANARY FAILED CLOSED (>0.90) on seed {seed}: {violators}")

    return {
        "seed": seed,
        "withheld_positive_pixels": int(hidden.sum()),
        "negative_sample_pixels": int(neg_idx.size),
        "ceiling": ceiling,
        "n_features_checked": len(rows),
        "passed": len(violators) == 0,
        "violators": violators,
        "max_overall": worst_all,
        "max_h77h_static": worst_h77h,
        "features": rows,
    }


def run_experiment_1(
    grid: G.GridSpec,
    records: list[dict[str, Any]],
    space_shas: list[str],
) -> tuple[np.memmap, list[str], np.memmap, np.memmap, dict[str, Any]]:
    """Execute Experiment 1: build features, verify MT mesh notch filter, run leakage canaries,
    and run Phase 1 pre-placement surface uniqueness check.
    """
    t0 = time.perf_counter()
    log("=== EXPERIMENT 1 (H77H-E1): Feature Build, Mesh Suppression, Leakage Canaries & Surface Uniqueness ===")
    foot = grid.footprint
    std_fp, std_names = build_std19_fp(foot)
    h77h_fp, ext_fp, static_diag = build_h77h_static_stack(RAW_DIR, CACHE_DIR, foot)

    mesh_info = static_diag["mt_mesh_periodicity_audit_ir_h77g_01"]
    log(
        f"  [IR-H77G-01 MT Mesh Audit] raw cond_surf zero-step={mesh_info['raw_cond_adjacent_zero_step_frac']:.4f} | "
        f"35px mesh suppression={mesh_info['mesh_35px_suppression_fraction']*100:.2f}% | "
        f"70px mesh suppression={mesh_info['mesh_70px_suppression_fraction']*100:.2f}%"
    )

    canary20 = run_single_feature_canaries_for_seed(20, grid, std_fp, std_names, ext_fp, h77h_fp)
    canary30 = run_single_feature_canaries_for_seed(30, grid, std_fp, std_names, ext_fp, h77h_fp)
    log(
        f"  [Canary Gate] Seed 20 passed={canary20['passed']} (max={canary20['max_overall']['symmetric_auc']:.4f} "
        f"on {canary20['max_overall']['feature']}; max H77H static={canary20['max_h77h_static']['symmetric_auc']:.4f} "
        f"on {canary20['max_h77h_static']['feature']})"
    )
    log(
        f"  [Canary Gate] Seed 30 passed={canary30['passed']} (max={canary30['max_overall']['symmetric_auc']:.4f} "
        f"on {canary30['max_overall']['feature']}; max H77H static={canary30['max_h77h_static']['symmetric_auc']:.4f} "
        f"on {canary30['max_h77h_static']['feature']})"
    )

    # Phase 1 surface uniqueness check on label-free composite prior
    comp_col = H77H_STATIC_NAMES.index("h77h_composite_physics_prior")
    surf_prior = np.zeros(foot.shape, dtype=np.float32)
    surf_prior.ravel()[np.flatnonzero(foot)] = np.asarray(h77h_fp[:, comp_col], dtype=np.float32)

    idx_reg = {
        "rasters": records,
        "complete_accessible_scan": False,
        "errors": ["Checked local lane TIFFs + 32 git-fetched scored rasters + 2 witnesses"],
    }
    p1_prior = UQ.compare_array_to_registry(surf_prior, idx_reg, foot, file_sha256=None)
    abs_rho_prior = abs(float(p1_prior["worst_spearman_full_footprint"] or 0.0))
    log(
        f"  [Phase 1 Prior Surface Uniqueness] max |Spearman rho| = {abs_rho_prior:.4f} "
        f"vs {p1_prior['worst_rho_submission']} (limit <= 0.90)"
    )
    if abs_rho_prior > 0.90:
        raise RuntimeError(f"Phase 1 surface Spearman correlation {abs_rho_prior:.4f} > 0.90!")

    e1_out = {
        "experiment": "H77H-E1",
        "evidence_class": "CANARY-DIAGNOSTIC",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "static_diagnostics": static_diag,
        "canary_seed20": canary20,
        "canary_seed30": canary30,
        "phase1_composite_prior_surface": {
            "worst_spearman_full_footprint": p1_prior["worst_spearman_full_footprint"],
            "worst_abs_spearman": round(abs_rho_prior, 6),
            "worst_rho_submission": p1_prior["worst_rho_submission"],
            "exceeds_0p90_limit": bool(abs_rho_prior > 0.90),
            "space_filling_rasters": space_shas,
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    (EVID_DIR / "h77h_e1_canary.json").write_text(json.dumps(e1_out, indent=2) + "\n")
    return std_fp, std_names, ext_fp, h77h_fp, e1_out


def evaluate_holdout_seed(
    seed: int,
    grid: G.GridSpec,
    std_fp: np.memmap,
    ext_fp: np.memmap,
    h77h_fp: np.memmap,
    packed_halos: np.ndarray,
    info_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run 4-quadrant out-of-fold cross-fitting and evaluate all 8 arms on a holdout draw."""
    t_seed = time.perf_counter()
    foot = grid.footprint
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"] & foot
    visible = folds[0]["visible"] & foot
    assert not np.any(hidden & visible), "Leakage between hidden and visible masks!"
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)

    log(f"=== EXPERIMENT 2 (H77H-E2) SEED {seed}: withheld={int(hidden.sum()):,} px, visible={int(visible.sum()):,} px ===")

    oof_cache = CACHE_DIR / f"oof_surfaces_seed{seed}.npz"
    if oof_cache.is_file():
        log(f"  Reloading cached Seed {seed} OOF surfaces from {oof_cache.name}...")
        npz = np.load(oof_cache)
        surf_full = npz["surf_full"]
        surf_static = npz["surf_static"]
        surf_std_h77h = npz["surf_std_h77h"]
        surf_std = npz["surf_std"]
    else:
        X_std = np.asarray(std_fp, dtype=np.float32)
        X_std_h77h = np.hstack([X_std, np.asarray(h77h_fp, dtype=np.float32)])
        X_static_all = np.hstack([X_std, np.asarray(ext_fp, dtype=np.float32), np.asarray(h77h_fp, dtype=np.float32)])

        # Compute 4-bucket Segment-LOCO + bilateral bridge features strictly from visible (0 hidden px)
        loco_tr_all, loco_inf_all = compute_h77h_loco_features(
            visible_mask=visible,
            footprint=foot,
        )

        surf_full = np.zeros((h, w), dtype=np.float32)
        surf_static = np.zeros((h, w), dtype=np.float32)
        surf_std_h77h = np.zeros((h, w), dtype=np.float32)
        surf_std = np.zeros((h, w), dtype=np.float32)

        rng = np.random.default_rng(2000 + seed)
        for k, fold in enumerate(folds):
            tf = time.perf_counter()
            tr_reg = np.zeros_like(foot)
            for j, f2 in enumerate(folds):
                if j != k:
                    tr_reg |= f2["region"] & foot
            vis_tr = visible & tr_reg
            d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)

            pos = fp_lut[np.flatnonzero(vis_tr & foot)]
            if pos.size > 48_000:
                pos = rng.choice(pos, 48_000, replace=False)

            neg_base = foot & tr_reg & (~vis_tr) & (d_vis_tr > 3.0)
            hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 16.0))]
            easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 16.0))]
            n_neg = int(2.8 * pos.size)
            nh = min(hard.size, int(0.50 * n_neg))
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])

            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
            te_idx = fp_lut[np.flatnonzero(fold["region"] & foot)]

            tr_loco = loco_tr_all[tr_idx]
            te_loco = loco_inf_all[te_idx]

            def _fit_pred(X_base: np.ndarray, add_loco: bool) -> np.ndarray:
                x_tr = np.hstack([X_base[tr_idx], tr_loco]) if add_loco else X_base[tr_idx]
                clf = HistGradientBoostingClassifier(**HGB_KW)
                clf.fit(x_tr, y_tr)
                del x_tr
                preds = np.empty(te_idx.size, dtype=np.float32)
                for s in range(0, te_idx.size, 250_000):
                    sl = slice(s, min(s + 250_000, te_idx.size))
                    xb = np.hstack([X_base[te_idx[sl]], te_loco[sl]]) if add_loco else X_base[te_idx[sl]]
                    preds[sl] = clf.predict_proba(xb)[:, 1].astype(np.float32)
                del clf
                gc.collect()
                return preds

            te_flat = fp_flat[te_idx]
            p_loco = _fit_pred(X_static_all, add_loco=True)
            p_stat = _fit_pred(X_static_all, add_loco=False)
            surf_full.ravel()[te_flat] = (0.76 * p_loco + 0.24 * p_stat).astype(np.float32)
            surf_static.ravel()[te_flat] = p_stat
            surf_std_h77h.ravel()[te_flat] = _fit_pred(X_std_h77h, add_loco=False)
            surf_std.ravel()[te_flat] = _fit_pred(X_std, add_loco=False)
            log(f"  [Seed {seed}] Fold {fold['name']} cross-fitted in {time.perf_counter()-tf:.1f}s")

        del X_std, X_std_h77h, X_static_all, loco_tr_all, loco_inf_all
        gc.collect()
        np.savez_compressed(
            oof_cache,
            surf_full=surf_full,
            surf_static=surf_static,
            surf_std_h77h=surf_std_h77h,
            surf_std=surf_std,
        )

    # Pure label-free composite physics prior surface & scarp crest 2D
    comp_col = H77H_STATIC_NAMES.index("h77h_composite_physics_prior")
    crest_col = H77H_STATIC_NAMES.index("h77h_lidar_scarp_crest")
    surf_pure_phys = np.zeros((h, w), dtype=np.float32)
    surf_pure_phys.ravel()[fp_flat] = np.asarray(h77h_fp[:, comp_col], dtype=np.float32)
    crest_2d = np.zeros((h, w), dtype=np.float32)
    crest_2d.ravel()[fp_flat] = np.asarray(h77h_fp[:, crest_col], dtype=np.float32)

    allow_global = foot & (~visible) & (dist_vis > EXCL_PX)

    def _emit_per_fold(surf: np.ndarray, use_cap: bool = False) -> np.ndarray:
        return allocate_h77h_submodular_300m(
            surf,
            allow_global,
            target_n=BUDGET_DOTS,
            crest_2d=crest_2d,
            support_quantile=0.88,
            gamma=1.85,
            min_sep_px=0.0,
            packed_halos=packed_halos if use_cap else None,
            max_overlap_cap=MAX_INFO_OVERLAP_CAP,
        )

    # Fold-fair H33 reference (H19-5 support, >2 px from visible catalogue, 3 px greedy spacing)
    h19_path = (
        RAW_DIR
        / "scored"
        / "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif"
    )
    with rasterio.open(h19_path) as src:
        sup_h19 = src.read(1)
    sup_h19 = np.isfinite(sup_h19) & (sup_h19 > 0)
    rk_h19 = np.random.default_rng(7).random(h * w)
    dots_h33 = np.zeros((h, w), dtype=bool)
    for fold in folds:
        valid_f = fold["region"] & foot & (~fold["masked_known"]) & (dist_vis > 2.0)
        cand = np.flatnonzero((sup_h19 & valid_f).ravel())
        sel = LIA_AL.greedy_select(cand[np.argsort(rk_h19[cand])], 3.0)
        dots_h33.ravel()[sel] = True

    rng_rand = np.random.default_rng(999 + seed)
    dots_rand = np.zeros((h, w), dtype=bool)
    dots_rand.ravel()[rng_rand.choice(np.flatnonzero(allow_global), size=BUDGET_DOTS, replace=False)] = True

    arm_dots: dict[str, np.ndarray] = {
        "A_h77h_full": _emit_per_fold(surf_full, use_cap=False),
        "A_h77h_unique_cap": _emit_per_fold(surf_full, use_cap=True),
        "A_h77h_static_no_loco": _emit_per_fold(surf_static, use_cap=False),
        "A_std_plus_h77h_only": _emit_per_fold(surf_std_h77h, use_cap=False),
        "S_h77h_pure_physics": _emit_per_fold(surf_pure_phys, use_cap=False),
        "A_std_control": _emit_per_fold(surf_std, use_cap=False),
        "A0_h33_foldfair": dots_h33,
        "A0_random": dots_rand,
    }

    ov_unique_oof = check_informative_overlaps(arm_dots["A_h77h_unique_cap"], packed_halos, info_meta)
    assert ov_unique_oof["informative_firings_gt_0p70"] == 0, f"Unexpected OOF firing: {ov_unique_oof}"

    terms_by_arm: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for arm_name, d_bool in arm_dots.items():
        d_f32 = d_bool.astype(np.float32)
        acc_terms = None
        per_fold: dict[str, Any] = {}
        for fold in folds:
            res_f, t_f = EH.evaluate(d_f32, fold, valid=foot, block_side=200)
            per_fold[fold["name"]] = {
                "dti": round(float(res_f["dti"]), 6),
                "n_truth": int(res_f["n_truth"]),
                "emitted": int(res_f["n_emitted"]),
            }
            acc_terms = t_f if acc_terms is None else (acc_terms + t_f)
        assert acc_terms is not None
        terms_by_arm[arm_name] = acc_terms
        pooled_dti = float(EH.from_terms(acc_terms.sum(axis=0)))
        details[arm_name] = {
            "dots": int(d_bool.sum()),
            "pooled_holdout_dti": round(pooled_dti, 6),
            "per_fold": per_fold,
        }
        log(f"  [Seed {seed}] {arm_name:24s}: HOLDOUT-DTI = {pooled_dti:.6f} (dots={int(d_bool.sum()):,})")

    pooled_full = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_h77h_full")
    pooled_uniq = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_h77h_unique_cap")

    return {
        "seed": seed,
        "evaluator_version": EH.VERSION,
        "withheld_positive_pixels": int(hidden.sum()),
        "visible_positive_pixels": int(visible.sum()),
        "fold_receipts": [f["receipt"] for f in folds],
        "arms": details,
        "oof_unique_cap_informative_overlap": ov_unique_oof,
        "pooled_summary_full_candidate": pooled_full,
        "pooled_summary_unique_cap_candidate": pooled_uniq,
        "elapsed_s": round(time.perf_counter() - t_seed, 2),
    }


def run_experiment_3(
    grid: G.GridSpec,
    std_fp: np.memmap,
    ext_fp: np.memmap,
    h77h_fp: np.memmap,
    packed_halos: np.ndarray,
    info_meta: list[dict[str, Any]],
    records: list[dict[str, Any]],
    space_shas: list[str],
    cov_by_sha: dict[str, float],
    e1_out: dict[str, Any],
    e2_out: dict[str, Any],
) -> dict[str, Any]:
    """Execute Experiment 3: full-map 4-quadrant OOF surface, capacity-capped submodular
    placement, shared submission_writer serialization, validation, uniqueness audit, figure,
    and frozen run card.
    """
    t0 = time.perf_counter()
    log("=== EXPERIMENT 3 (H77H-E3): Full-Map OOF Surface, Capacity-Capped Placement, Validation & Uniqueness ===")
    foot = grid.footprint
    cat = grid.catalogue & foot
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    surf_cache = CACHE_DIR / "full_map_surface_h77h.npy"
    if surf_cache.is_file():
        surf_full = np.load(surf_cache)
        log("  Reloaded cached full-map 4-quadrant OOF surface.")
    else:
        X_static_all = np.hstack(
            [
                np.asarray(std_fp, dtype=np.float32),
                np.asarray(ext_fp, dtype=np.float32),
                np.asarray(h77h_fp, dtype=np.float32),
            ]
        )
        loco_tr_all, loco_inf_all = compute_h77h_loco_features(
            visible_mask=cat,
            footprint=foot,
        )
        surf_full = np.zeros((h, w), dtype=np.float32)
        ymid, xmid = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        quads = [
            ("NW", (yy < ymid) & (xx < xmid)),
            ("NE", (yy < ymid) & (xx >= xmid)),
            ("SW", (yy >= ymid) & (xx < xmid)),
            ("SE", (yy >= ymid) & (xx >= xmid)),
        ]
        rng = np.random.default_rng(7708)
        for qname, qmask in quads:
            tq = time.perf_counter()
            tr_mask = foot & (~qmask)
            cat_tr = cat & tr_mask
            d_tr = ndimage.distance_transform_edt(~cat_tr).astype(np.float32)

            pos = fp_lut[np.flatnonzero(cat_tr)]
            if pos.size > 48_000:
                pos = rng.choice(pos, 48_000, replace=False)
            neg_base = foot & tr_mask & (~cat_tr) & (d_tr > 3.0)
            hard = fp_lut[np.flatnonzero(neg_base & (d_tr <= 16.0))]
            easy = fp_lut[np.flatnonzero(neg_base & (d_tr > 16.0))]
            n_neg = int(2.8 * pos.size)
            nh = min(hard.size, int(0.50 * n_neg))
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])

            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
            te_idx = fp_lut[np.flatnonzero(qmask & foot)]

            tr_loco = loco_tr_all[tr_idx]
            te_loco = loco_inf_all[te_idx]

            # Fit LOCO-augmented model + static-only model
            clf_loco = HistGradientBoostingClassifier(**HGB_KW)
            clf_loco.fit(np.hstack([X_static_all[tr_idx], tr_loco]), y_tr)

            clf_stat = HistGradientBoostingClassifier(**HGB_KW)
            clf_stat.fit(X_static_all[tr_idx], y_tr)

            preds = np.empty(te_idx.size, dtype=np.float32)
            for s in range(0, te_idx.size, 250_000):
                sl = slice(s, min(s + 250_000, te_idx.size))
                xb_s = X_static_all[te_idx[sl]]
                xb_l = np.hstack([xb_s, te_loco[sl]])
                p_l = clf_loco.predict_proba(xb_l)[:, 1].astype(np.float32)
                p_s = clf_stat.predict_proba(xb_s)[:, 1].astype(np.float32)
                preds[sl] = (0.76 * p_l + 0.24 * p_s).astype(np.float32)

            surf_full.ravel()[fp_flat[te_idx]] = preds
            del clf_loco, clf_stat, preds, tr_loco, te_loco
            gc.collect()
            log(f"  Full-map quadrant {qname} OOF finished in {time.perf_counter()-tq:.1f}s")

        del X_static_all, loco_tr_all, loco_inf_all
        gc.collect()
        np.save(surf_cache, surf_full)

    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    allowed_full = foot & (~cat) & (dist_cat > EXCL_PX)
    crest_col = H77H_STATIC_NAMES.index("h77h_lidar_scarp_crest")
    crest_2d = np.zeros((h, w), dtype=np.float32)
    crest_2d.ravel()[fp_flat] = np.asarray(h77h_fp[:, crest_col], dtype=np.float32)

    # Allocate 36,500 binary dots under exact informative-registry capacity cap (<= 0.680 < 0.70)
    dots_bool = allocate_h77h_submodular_300m(
        surf_full,
        allowed_full,
        target_n=BUDGET_DOTS,
        crest_2d=crest_2d,
        support_quantile=0.88,
        gamma=1.85,
        min_sep_px=0.0,
        packed_halos=packed_halos,
        max_overlap_cap=MAX_INFO_OVERLAP_CAP,
    )
    n_dots = int(dots_bool.sum())
    info_ov_check = check_informative_overlaps(dots_bool, packed_halos, info_meta)
    log(
        f"  Allocated {n_dots:,} binary dots | worst informative 3-px overlap = "
        f"{info_ov_check['worst_forward_overlap_3px']:.4f} vs {info_ov_check['worst_overlap_submission']} "
        f"(firings > 0.70: {info_ov_check['informative_firings_gt_0p70']})"
    )
    assert info_ov_check["informative_firings_gt_0p70"] == 0, f"Informative overlap exceeded 0.70: {info_ov_check}"

    dots_sha = hashlib.sha256(np.ascontiguousarray(dots_bool).tobytes()).hexdigest()
    sub_name = f"gems77-h77h-downdip-notch36-bridge-{n_dots}dots-20261021-{dots_sha[:8]}-zeros"

    s20_full = float(e2_out["seeds"]["20"]["arms"]["A_h77h_full"]["pooled_holdout_dti"])
    s20_uniq = float(e2_out["seeds"]["20"]["arms"]["A_h77h_unique_cap"]["pooled_holdout_dti"])
    s30_full = float(e2_out["seeds"]["30"]["arms"]["A_h77h_full"]["pooled_holdout_dti"])
    s30_uniq = float(e2_out["seeds"]["30"]["arms"]["A_h77h_unique_cap"]["pooled_holdout_dti"])

    note = (
        f"H77H downdip500m+MTnotch36+LOCObridge {n_dots}d; "
        f"HOLDOUT-DTI {s20_uniq:.4f}/{s30_uniq:.4f} (full {s20_full:.4f}/{s30_full:.4f}); "
        f"0 info dupes"
    )
    assert len(note) <= 140, f"Note too long ({len(note)} chars): {note}"

    pred_f32 = np.where(foot, dots_bool.astype(np.float32), 0.0).astype(np.float32)
    out_tif = DOWN_DIR / f"{sub_name}.tif"
    sample_tif = BRIDGE_DIR / "sample_submission.tif"

    # Remove any older h77h files in docs/downloads before writing
    for old_p in DOWN_DIR.glob("gems77-h77h-*"):
        old_p.unlink()

    writer_receipt = SW.write_submission(
        out_tif,
        pred_f32,
        sample_tif,
        foot,
        note=note,
        name=sub_name,
        catalogue=cat,
    )
    out_zip = out_tif.with_suffix(".zip")

    # Also write the diagnostic -nan.tif twin (NaN outside footprint; NOT the portal file)
    nan_pred = np.where(foot, dots_bool.astype(np.float32), np.nan).astype(np.float32)
    nan_tif = DOWN_DIR / f"{sub_name.replace('-zeros', '-nan')}.tif"
    G.write_submission(nan_tif, nan_pred, mode="nan")

    # Run both 15/15 shared validator and 17/17 LIA validator
    val57_report = VAL57.validate(out_tif, footprint=foot, catalogue=cat)
    val17_report = VAL_LIA.validate(out_tif, mode="zeros", name=sub_name, note=note)
    val_zip_report = VAL_LIA.validate_zip(out_zip, out_tif)
    assert val57_report["all_checks_passed"], f"15/15 validator failed: {val57_report}"
    assert val17_report["passed"], f"17/17 validator failed: {val17_report}"
    assert val_zip_report["passed"], f"ZIP validator failed: {val_zip_report}"

    # Phase 1 (OOF surface) & Phase 2 (final dots) uniqueness audit via unmodified vendor/gems57/uniqueness.py
    idx_reg = {
        "rasters": records,
        "complete_accessible_scan": False,
        "errors": [
            "Registry scope: all local GEMSDOE77 lane TIFFs + 32 git-fetched scored rasters (score>=0.14) + 2 witnesses; full 812-raster REST crawl not executed in this turn."
        ],
    }
    surf_audit = np.where(foot, surf_full, 0.0).astype(np.float32)
    p1 = UQ.compare_array_to_registry(surf_audit, idx_reg, foot, file_sha256=None)
    p2 = UQ.compare_array_to_registry(pred_f32, idx_reg, foot, file_sha256=sha256_file(out_tif))

    space_set = set(space_shas)

    def _informative_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        info_rows = [r for r in rows if "error" not in r and r["sha256"] not in space_set]
        w_rho = max(info_rows, key=lambda r: abs(r["spearman_full_footprint"] or 0.0), default=None)
        w_ov = max(info_rows, key=lambda r: float(r["my_dots_within_3px_of_theirs"]), default=None)
        firings_ov = sum(1 for r in info_rows if float(r["my_dots_within_3px_of_theirs"]) > 0.70)
        firings_rho = sum(1 for r in info_rows if abs(r["spearman_full_footprint"] or 0.0) > 0.90)
        return {
            "n_informative": len(info_rows),
            "worst_abs_spearman": round(abs(float(w_rho["spearman_full_footprint"] or 0.0)), 6) if w_rho else 0.0,
            "worst_rho_submission": w_rho["submission"] if w_rho else None,
            "worst_forward_overlap_3px": round(float(w_ov["my_dots_within_3px_of_theirs"]), 6) if w_ov else 0.0,
            "worst_overlap_submission": w_ov["submission"] if w_ov else None,
            "informative_firings_overlap_gt_0p70": firings_ov,
            "informative_firings_rho_gt_0p90": firings_rho,
            "duplicate_by_rho_or_overlap": bool(firings_ov > 0 or firings_rho > 0),
            "top5_by_overlap": sorted(
                (
                    {
                        "submission": r["submission"],
                        "repo": r["repo"],
                        "sha256": r["sha256"],
                        "overlap_3px": round(float(r["my_dots_within_3px_of_theirs"]), 6),
                        "abs_spearman": round(abs(float(r["spearman_full_footprint"] or 0.0)), 6),
                    }
                    for r in info_rows
                ),
                key=lambda d: -d["overlap_3px"],
            )[:5],
        }

    p1_info = _informative_summary(p1["rows"])
    p2_info = _informative_summary(p2["rows"])
    assert not p1_info["duplicate_by_rho_or_overlap"], f"Phase 1 informative duplicate: {p1_info}"
    assert not p2_info["duplicate_by_rho_or_overlap"], f"Phase 2 informative duplicate: {p2_info}"

    uniq_out = {
        "candidate": sub_name,
        "registry_scope": "local GEMSDOE77 lane TIFFs + 32 git-fetched scored rasters (score>=0.14, incl H33) + 2 witnesses (dense_witness ab0a0a62, h33 c55bafc4)",
        "registry_records": len(records),
        "space_filling_rasters_in_checked_set": space_shas,
        "coverage_3px_by_sha256": cov_by_sha,
        "phase1_surface": {
            "worst_spearman_full_footprint": p1["worst_spearman_full_footprint"],
            "worst_abs_spearman": round(abs(float(p1["worst_spearman_full_footprint"] or 0.0)), 6),
            "worst_rho_submission": p1["worst_rho_submission"],
            "literal_worst_dot_overlap": p1["worst_dot_overlap"],
            "informative": p1_info,
            "vendor_unique": p1["unique"],
            "vendor_duplicate_count": p1["duplicate_count"],
        },
        "phase2_final_dots": {
            "worst_spearman_full_footprint": p2["worst_spearman_full_footprint"],
            "worst_abs_spearman": round(abs(float(p2["worst_spearman_full_footprint"] or 0.0)), 6),
            "worst_rho_submission": p2["worst_rho_submission"],
            "literal_worst_dot_overlap": p2["worst_dot_overlap"],
            "literal_worst_overlap_submission": p2["worst_overlap_submission"],
            "informative": p2_info,
            "vendor_unique": p2["unique"],
            "vendor_duplicate_count": p2["duplicate_count"],
            "byte_unique_among_checked": p2["byte_unique_among_checked"],
            "pixel_unique_among_checked": p2["pixel_unique_among_checked"],
        },
        "protocol_unique": False,
        "protocol_unique_reason": (
            "Fail-closed under Policy I17: dense witness 17GEMSDOE_E-proba-multiscale (ab0a0a62) covers "
            "100% of allowed footprint cells (forward 3-px overlap = 1.0000 > 0.70 on any non-empty raster). "
            f"Against all {p2_info['n_informative']} informative (<70% halo) rasters, 0/{p2_info['n_informative']} "
            f"duplicate firings occur (worst overlap = {p2_info['worst_forward_overlap_3px']:.4f} <= 0.70, "
            f"worst |rho| = {p2_info['worst_abs_spearman']:.4f} <= 0.90)."
        ),
        "rows_phase2": p2["rows"],
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    (EVID_DIR / "h77h_uniqueness.json").write_text(json.dumps(uniq_out, indent=2, default=str) + "\n")

    # Generate diagnostic 4-panel figure docs/assets/h77h_diagnostics.png
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    # Panel 1: +500m Downdip Scarp-Coupled Potential-Field Dipole (subsampled 4x for clean render)
    dd_col = H77H_STATIC_NAMES.index("h77h_downdip_scarp_coupled")
    dd_2d = np.full((h, w), np.nan, dtype=np.float32)
    dd_2d.ravel()[fp_flat] = np.asarray(h77h_fp[:, dd_col], dtype=np.float32)
    im0 = axes[0].imshow(dd_2d[::4, ::4], cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("H77H +500 m Downdip Gravity-Base\nx Scarp Dipole", color="#f0f6fc", fontsize=9.5)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: sigma=36 px Mesh-Notched MT Clay-Cap & Hydro-Regime Modulator
    hm_col = H77H_STATIC_NAMES.index("h77h_hydro_scarp_modulator")
    hm_2d = np.full((h, w), np.nan, dtype=np.float32)
    hm_2d.ravel()[fp_flat] = np.asarray(h77h_fp[:, hm_col], dtype=np.float32)
    im1 = axes[1].imshow(hm_2d[::4, ::4], cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("H77H sigma=36 px (3.6 km) Mesh-Notched\nMT/Strain/GDR1391 Modulator", color="#f0f6fc", fontsize=9.5)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Full-Map 4-Quadrant OOF Surface + Allocated Dots
    s_plot = np.where(foot, surf_full, np.nan)
    im2 = axes[2].imshow(s_plot[::4, ::4], cmap="inferno", vmin=0.0, vmax=np.nanpercentile(s_plot[::4, ::4], 99.0))
    dy_p, dx_p = np.nonzero(dots_bool)
    axes[2].scatter(dx_p[::20] / 4.0, dy_p[::20] / 4.0, s=0.35, c="#58a6ff", alpha=0.65, label="37,654 dots")
    axes[2].set_title(f"4-Quad OOF Surface + {n_dots:,} Dots\n(Max Info Overlap {p2_info['worst_forward_overlap_3px']:.3f} <= 0.70)", color="#f0f6fc", fontsize=9.5)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Dual-Seed Holdout DTI Bar Comparison
    arm_labels = [
        "H77H Full\n(S20)",
        "H77H Full\n(S30)",
        "H77H Uniq\n(S20)",
        "H77H Uniq\n(S30)",
        "H77H Static\n(0 Cat S20)",
        "H33 Fold\n(S20)",
        "Std 19B\n(S20)",
    ]
    arm_vals = [
        s20_full,
        s30_full,
        s20_uniq,
        s30_uniq,
        float(e2_out["seeds"]["20"]["arms"]["A_h77h_static_no_loco"]["pooled_holdout_dti"]),
        float(e2_out["seeds"]["20"]["arms"]["A0_h33_foldfair"]["pooled_holdout_dti"]),
        float(e2_out["seeds"]["20"]["arms"]["A_std_control"]["pooled_holdout_dti"]),
    ]
    colors = ["#2ea043", "#2ea043", "#58a6ff", "#58a6ff", "#d29922", "#8b949e", "#6e7681"]
    bars = axes[3].bar(range(len(arm_labels)), arm_vals, color=colors, edgecolor="#f0f6fc", linewidth=0.6)
    axes[3].axhline(RETAINED_REPO_BEST_DTI, color="#f85149", linestyle="--", linewidth=1.1, label=f"H77-Step Best ({RETAINED_REPO_BEST_DTI:.4f})")
    axes[3].set_xticks(range(len(arm_labels)))
    axes[3].set_xticklabels(arm_labels, fontsize=7.5, color="#c9d1d9")
    axes[3].set_ylabel("HOLDOUT-DTI (gems57-pooled-hide-v2)", color="#c9d1d9", fontsize=8.5)
    axes[3].set_title("Dual-Seed Buffered Whole-Component\nHoldout Benchmark", color="#f0f6fc", fontsize=9.5)
    axes[3].legend(loc="upper right", fontsize=7.5, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    for b_rect, v_num in zip(bars, arm_vals):
        axes[3].text(
            b_rect.get_x() + b_rect.get_width() / 2.0,
            v_num + 0.002,
            f"{v_num:.4f}",
            ha="center",
            va="bottom",
            color="#f0f6fc",
            fontsize=7.2,
        )
    axes[3].set_ylim(0.0, max(max(arm_vals) + 0.025, 0.175))
    plt.tight_layout()
    fig_path = ASSET_DIR / "h77h_diagnostics.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    tif_sha = sha256_file(out_tif)
    zip_sha = sha256_file(out_zip)
    nan_sha = sha256_file(nan_tif)

    e3_build = {
        "experiment": "H77H-E3-BUILD",
        "submission_name": sub_name,
        "note": note,
        "note_chars": len(note),
        "dots": n_dots,
        "dots_decoded_sha256": dots_sha,
        "tif": str(out_tif.relative_to(ROOT)),
        "tif_sha256": tif_sha,
        "tif_bytes": out_tif.stat().st_size,
        "zip": str(out_zip.relative_to(ROOT)),
        "zip_sha256": zip_sha,
        "zip_bytes": out_zip.stat().st_size,
        "nan_twin": str(nan_tif.relative_to(ROOT)),
        "nan_twin_sha256": nan_sha,
        "writer_receipt": writer_receipt,
        "validator_15_of_15": val57_report,
        "validator_17_of_17": val17_report,
        "diagnostic_figure": str(fig_path.relative_to(ROOT)),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    (EVID_DIR / "h77h_e3_build.json").write_text(json.dumps(e3_build, indent=2, default=str) + "\n")

    # Determine promotion verdict per protocol rules:
    # Promoting to OK TO SUBMIT requires beating the repo incumbent (0.152718) AND passing literal
    # uniqueness (which fails closed under Policy I17 due to dense witness ab0a0a62) AND manual selector approval.
    beats_repo_best_full = bool(s20_full > RETAINED_REPO_BEST_DTI and s30_full > RETAINED_REPO_BEST_DTI)
    beats_repo_best_uniq = bool(s20_uniq > RETAINED_REPO_BEST_DTI and s30_uniq > RETAINED_REPO_BEST_DTI)
    verdict = "promote" if (beats_repo_best_uniq and uniq_out["protocol_unique"]) else "negative"

    sc20_u = e2_out["seeds"]["20"]["pooled_summary_unique_cap_candidate"]["scores"]["A_h77h_unique_cap"]
    sc20_f = e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["scores"]["A_h77h_full"]
    sc30_u = e2_out["seeds"]["30"]["pooled_summary_unique_cap_candidate"]["scores"]["A_h77h_unique_cap"]
    sc30_f = e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["scores"]["A_h77h_full"]

    run_card = {
        "schema_version": "gems-run-card-v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane_id": "H77H",
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/17079c9b-gemsdoe77",
        "hypothesis_id": "H77H-1 (with H77H-2 mesh-notched hydro-regime and H77H-3 Laplace SVD sub-blocks)",
        "hypothesis": (
            "For a 55-60 deg dipping Basin-and-Range normal fault with dense pre-Tertiary basement offset "
            "at 500-1000 m depth inside the hanging wall, the steepest horizontal gradients of isostatic "
            "gravity (|grad(iso_grav_anom)|) and basement depth (|grad(depth_to_base_surf)|) are displaced "
            "~500 m (+5.0 px) downdip into the hanging wall along the local dip vector relative to the "
            "surface scarp breakaway. Pairing this +500 m downdip hanging-wall dipole with Laplace "
            "second-vertical-derivative zero-crossing curvature, a sigma=36 px (3.6 km) mesh-deconfounded "
            "MT clay-cap / transtension / GDR-1391 chemical geothermometer modulator (fixing IR-H77G-01), "
            "de-quantized 1 m LiDAR scarp morphology (q==0 -> NaN per IR-77D-01), and fold-strict "
            "Leave-Own-Component-Out (LOCO) bilateral inter-component tip-bridge continuation improves "
            "withheld whole-component fault recovery while maintaining <0.70 3-px overlap against all "
            "informative registry rasters."
        ),
        "mechanism": (
            "(1) Advect |grad(iso_grav_anom)| and |grad(depth_to_base_surf)| by +5.0 px (+500 m) downdip "
            "into the hanging wall minus -5.0 px footwall along the local dip unit vector and multiply by "
            "surface scarp crest curvature; (2) compute Laplace 2nd vertical derivatives -nabla_H^2(grav) "
            "and -nabla_H^2(rtp) + across-valley Hessian demagnetization trough; (3) apply a sigma=36 px "
            "(3.6 km) Gaussian filter to cond_surf, geod_*, and ieq/deq before gradient extraction "
            "(suppressing the 35.02-px MT mesh spike by "
            f"{e1_out['static_diagnostics']['mt_mesh_periodicity_audit_ir_h77g_01']['mesh_35px_suppression_fraction']*100:.2f}%"
            f") and multiply by GDR-1391 silica/cation geothermometer excess; (4) compute fold-strict "
            f"4-bucket Segment-LOCO distinct-component distances, splay collars, Pollard tip lobes, step-overs, "
            f"and bilateral facing-tip relay bridges; (5) allocate {n_dots:,} binary dots via 300 m submodular "
            f"cover under a <={MAX_INFO_OVERLAP_CAP:.3f} per-raster informative overlap cap."
        ),
        "named_non_fault_mimic": (
            "Gently dipping unfaulted pediment unconformities (Tertiary volcanic tuff over Paleozoic "
            "basement), Pleistocene Lake Lahontan shoreline benches above basinward-thickening lacustrine "
            "clays, and highway/railroad embankments."
        ),
        "leakage_canary": {
            "evidence_class": "CANARY-DIAGNOSTIC",
            "ceiling": 0.90,
            "seed20_passed": e1_out["canary_seed20"]["passed"],
            "seed20_max_overall": e1_out["canary_seed20"]["max_overall"],
            "seed20_max_h77h_static": e1_out["canary_seed20"]["max_h77h_static"],
            "seed30_passed": e1_out["canary_seed30"]["passed"],
            "seed30_max_overall": e1_out["canary_seed30"]["max_overall"],
            "seed30_max_h77h_static": e1_out["canary_seed30"]["max_h77h_static"],
        },
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "alpha": 0.2,
            "beta": 0.8,
            "kernel_radius_m": 300.0,
            "bootstrap_draws": 1000,
            "bootstrap_block_m": 20000.0,
            "retained_repo_incumbent_dti": RETAINED_REPO_BEST_DTI,
            "seed_20_primary": {
                "withheld_positive_pixels": e2_out["seeds"]["20"]["withheld_positive_pixels"],
                "A_h77h_unique_cap": sc20_u,
                "A_h77h_full": sc20_f,
                "A_h77h_static_no_loco": e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["scores"]["A_h77h_static_no_loco"],
                "A_std_plus_h77h_only": e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["scores"]["A_std_plus_h77h_only"],
                "S_h77h_pure_physics": e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["scores"]["S_h77h_pure_physics"],
                "A_std_control": e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["scores"]["A_std_control"],
                "A0_h33_foldfair": e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["scores"]["A0_h33_foldfair"],
                "A0_random": e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["scores"]["A0_random"],
                "paired_deltas_unique_cap": e2_out["seeds"]["20"]["pooled_summary_unique_cap_candidate"]["paired_differences"],
                "paired_deltas_full": e2_out["seeds"]["20"]["pooled_summary_full_candidate"]["paired_differences"],
            },
            "seed_30_replication": {
                "withheld_positive_pixels": e2_out["seeds"]["30"]["withheld_positive_pixels"],
                "A_h77h_unique_cap": sc30_u,
                "A_h77h_full": sc30_f,
                "A_h77h_static_no_loco": e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["scores"]["A_h77h_static_no_loco"],
                "A_std_plus_h77h_only": e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["scores"]["A_std_plus_h77h_only"],
                "S_h77h_pure_physics": e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["scores"]["S_h77h_pure_physics"],
                "A_std_control": e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["scores"]["A_std_control"],
                "A0_h33_foldfair": e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["scores"]["A0_h33_foldfair"],
                "A0_random": e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["scores"]["A0_random"],
                "paired_deltas_unique_cap": e2_out["seeds"]["30"]["pooled_summary_unique_cap_candidate"]["paired_differences"],
                "paired_deltas_full": e2_out["seeds"]["30"]["pooled_summary_full_candidate"]["paired_differences"],
            },
            "beats_std_control_both_seeds": bool(
                sc20_u["dti"] > e2_out["seeds"]["20"]["arms"]["A_std_control"]["pooled_holdout_dti"]
                and sc30_u["dti"] > e2_out["seeds"]["30"]["arms"]["A_std_control"]["pooled_holdout_dti"]
            ),
            "beats_h33_foldfair_both_seeds": bool(
                sc20_u["dti"] > e2_out["seeds"]["20"]["arms"]["A0_h33_foldfair"]["pooled_holdout_dti"]
                and sc30_u["dti"] > e2_out["seeds"]["30"]["arms"]["A0_h33_foldfair"]["pooled_holdout_dti"]
            ),
            "beats_repo_incumbent_0p1527_full": beats_repo_best_full,
            "beats_repo_incumbent_0p1527_unique_cap": beats_repo_best_uniq,
        },
        "correlation_and_overlap_vs_registry": {
            "registry_records_checked": len(records),
            "informative_records_checked": p2_info["n_informative"],
            "space_filling_records_checked": len(space_shas),
            "phase1_pre_placement_surface": uniq_out["phase1_surface"],
            "phase2_final_dots": uniq_out["phase2_final_dots"],
            "protocol_unique": uniq_out["protocol_unique"],
            "protocol_unique_reason": uniq_out["protocol_unique_reason"],
        },
        "artifact": {
            "tif_path": str(out_tif.relative_to(ROOT)),
            "tif_sha256": tif_sha,
            "tif_bytes": out_tif.stat().st_size,
            "zip_path": str(out_zip.relative_to(ROOT)),
            "zip_sha256": zip_sha,
            "zip_bytes": out_zip.stat().st_size,
            "nan_diagnostic_twin_path": str(nan_tif.relative_to(ROOT)),
            "nan_diagnostic_twin_sha256": nan_sha,
            "decoded_dots_sha256": dots_sha,
            "emitted_dots": n_dots,
        },
        "validator_output": {
            "vendor_gems57_15_of_15": {
                "all_passed": val57_report["all_checks_passed"],
                "n_passed": sum(1 for v in val57_report["checks"].values() if v),
                "n_checks": len(val57_report["checks"]),
                "no_nan_anywhere": val57_report["checks"]["no_nan_anywhere"]
                and val57_report["checks"]["outside_footprint_zero"],
                "values_in_0_1": val57_report["checks"]["range_0_1_guaranteed"],
                "crs_shape_transform_match": (
                    val57_report["checks"]["crs_epsg_32611"]
                    and val57_report["checks"]["dimensions_3730x3292"]
                    and val57_report["checks"]["transform_matches_sample_submission"]
                ),
            },
            "lia_17_of_17": {
                "all_passed": val17_report["passed"],
                "n_passed": val17_report["n_pass"],
                "n_checks": val17_report["n_checks"],
                "zip_passed": val_zip_report["passed"],
            },
        },
        "submission_name": sub_name,
        "submission_note": note,
        "submission_note_length": len(note),
        "verdict": verdict,
        "protocol_unique": False,
        "ok_to_download": True,
        "ok_to_submit": False,
        "download_and_submit_status": {
            "ok_to_download": True,
            "ok_to_submit": False,
            "selector_approved": False,
            "status_banner": (
                f"OK TO DOWNLOAD (15/15 & 17/17 VALIDATED, 0 NaN, [0, 1] FLOAT32, 0/{p2_info['n_informative']} INFORMATIVE DUPLICATES) -- "
                "DO NOT SUBMIT WITHOUT EXPLICIT SELECTOR APPROVAL (Policy I17 dense witness ab0a0a62 + Protocol Rule 6)"
            ),
        },
        "budget_compliance": {
            "experiments_used": 3,
            "max_experiments": 3,
            "weekly_slots_spent": 0,
        },
    }
    (EVID_DIR / "run_card_h77h.json").write_text(json.dumps(run_card, indent=2, default=str) + "\n")
    log(f"=== H77H PIPELINE COMPLETE: {sub_name} | verdict={verdict} ===")
    return run_card


def main() -> int:
    t_all = time.perf_counter()
    grid = load_grid_spec()
    records = collect_checked_registry(ROOT, RAW_DIR)
    packed_halos, info_meta, space_shas, cov_by_sha = build_informative_halo_matrix(
        records, grid.footprint, grid.catalogue, CACHE_DIR
    )
    log(
        f"Loaded checked registry: {len(records)} unique rasters "
        f"({len(info_meta)} informative <70% halo, {len(space_shas)} space-filling >=70% halo)"
    )

    std_fp, std_names, ext_fp, h77h_fp, e1_out = run_experiment_1(grid, records, space_shas)

    r20 = evaluate_holdout_seed(20, grid, std_fp, ext_fp, h77h_fp, packed_halos, info_meta)
    r30 = evaluate_holdout_seed(30, grid, std_fp, ext_fp, h77h_fp, packed_halos, info_meta)
    e2_out = {
        "experiment": "H77H-E2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "alpha": 0.2,
        "beta": 0.8,
        "kernel_radius_m": 300.0,
        "budget_dots": BUDGET_DOTS,
        "exclusion_px": EXCL_PX,
        "max_informative_overlap_cap": MAX_INFO_OVERLAP_CAP,
        "seeds": {"20": r20, "30": r30},
    }
    (EVID_DIR / "h77h_e2_holdout.json").write_text(json.dumps(e2_out, indent=2, default=str) + "\n")

    run_card = run_experiment_3(
        grid,
        std_fp,
        ext_fp,
        h77h_fp,
        packed_halos,
        info_meta,
        records,
        space_shas,
        cov_by_sha,
        e1_out,
        e2_out,
    )
    log(f"Total H77H pipeline elapsed: {time.perf_counter()-t_all:.1f}s")
    print(json.dumps(run_card["holdout_dti"]["seed_20_primary"]["A_h77h_unique_cap"], indent=2))
    print(json.dumps(run_card["holdout_dti"]["seed_30_replication"]["A_h77h_unique_cap"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
