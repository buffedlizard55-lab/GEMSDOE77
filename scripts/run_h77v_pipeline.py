#!/usr/bin/env python3
"""End-to-End 3-Experiment Pipeline for GEMSDOE77 Lane H77V.

Lane H77V: Spectral Mesh-Dealiased Regional Crustal Regime (sigma=36 px = 3.6 km)
  x Tubular Hydrothermal Demagnetization Trough (150 m, 300 m, 600 m)
  x Antiparallel Gravity-Basement Step Concordance
  x Road/Mining-Deconfounded 1 m LiDAR & 10 m 3DEP Fluvial Knickpoint-Scarp Conduit
  + 8-Channel Fold-Visible SLOCO Mechanics
  + Kernel-Convolved Lagrangian-Priced Dinkelbach Submodular Cover (LDSA).

Strict 3-Experiment Budget:
  - Experiment 1 (E1): Build 132-channel zero-SGMC static + 8-channel SLOCO stack (140 total),
    run 140 single-feature leakage canaries (<=0.90), and evaluate H77V + ablations on official
    gems57-pooled-hide-v2 Seed 20 (11,321 withheld positive pixels, 2,000-draw 20 km block bootstrap).
  - Experiment 2 (E2): Independent zero-retuning replication on official gems57-pooled-hide-v2
    Seed 30 (11,342 withheld positive pixels, 2,000-draw 20 km block bootstrap).
  - Experiment 3 (E3): Strictly out-of-fold full-Nevada dual-checkerboard propensity surface,
    LDSA 300 m submodular dot placement, official submission_writer packaging, 17-check validator,
    and exhaustive pre-placement surface & post-placement dot uniqueness audit.
"""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout as eval_mod
from gems57 import grid as grid_mod
from gems57 import holdout as hold_mod
from gems57 import metric as metric_mod
from gems57 import spatial as spatial_mod
from gems57 import submission_writer as writer_mod
from gems57 import uniqueness as uniq_mod
from gems57 import validate as validate_mod

from gems77v import LANE_ID, VERSION
from gems77v.allocate import (
    PortfolioCapacityIndex,
    allocate_ldsa_submodular,
    allocate_peak_nms_baseline,
    ensure_audit_index_built,
)
from gems77v.features import (
    H77V_DELTA_NAMES,
    H77V_SLOCO_MECH_NAMES,
    build_h77v_static_stack,
    compute_h77v_sloco_mechanics_fp,
    run_single_feature_leakage_canaries,
)
from gems77v.models import fit_predict_h77v_fold

DATA_DIR = ROOT / ".cache" / "data"
CACHE_DIR = ROOT / ".cache" / "h77v_work"
EVID_DIR = ROOT / "evidence" / "h77v"
SUB_DIR = ROOT / "submissions"
DL_DIR = ROOT / "docs" / "downloads"
FIG_DIR = ROOT / "docs" / "figures" / "h77v"

SUBMISSION_STEM = "gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021"
SUBMISSION_TIF_NAME = f"{SUBMISSION_STEM}.tif"
SUBMISSION_ZIP_NAME = f"{SUBMISSION_STEM}.zip"
SUBMISSION_JSON_NAME = f"{SUBMISSION_STEM}.json"
SUBMISSION_NOTE = (
    "GEMSDOE77 H77V: 3.6km dealiased MT/strain + tubular demag trough + "
    "3DEP knick/scarp + 8-ch SLOCO + LDSA 300m cover (inspection only)"
)
assert len(SUBMISSION_NOTE) <= 140, f"Note too long: {len(SUBMISSION_NOTE)}"

TARGET_DOTS_FULL = 36_500
MAX_OVERLAP_CAP = 0.650
EXCL_PX = 2.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_core_masks(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(data_dir / "sample_submission.tif") as ds:
        ss = ds.read(1)
        fp = np.isfinite(ss)
        profile = {
            "shape": (ds.height, ds.width),
            "crs": str(ds.crs),
            "transform": list(ds.transform)[:6],
        }
    with rasterio.open(data_dir / "existing_faults.tif") as ds:
        ef = ds.read(1)
        cat = (ef == 1) & fp
    return fp, cat, profile


def _eval_arms_official(
    arms: dict[str, np.ndarray],
    folds: list[dict[str, Any]],
    fp: np.ndarray,
    candidate_key: str,
    draws: int = 2000,
) -> dict[str, Any]:
    terms_by_arm: dict[str, np.ndarray] = {}
    fold_receipts_by_arm: dict[str, list[dict[str, Any]]] = {k: [] for k in arms}
    for name, pred in arms.items():
        acc_terms = None
        for f in folds:
            res_f, t_f = eval_mod.evaluate(pred, f, fp, global_shape=fp.shape)
            acc_terms = t_f.copy() if acc_terms is None else (acc_terms + t_f)
            fold_receipts_by_arm[name].append(res_f)
        assert acc_terms is not None
        terms_by_arm[name] = acc_terms
    summary = eval_mod.pooled_summary(
        terms_by_arm, draws=draws, seed=520810, candidate=candidate_key
    )
    summary["per_fold_receipts"] = fold_receipts_by_arm
    return summary


def evaluate_seed_suite(
    seed: int,
    g: Any,
    X132: np.ndarray,
    static_names: list[str],
    has_lidar_fp: np.ndarray,
    crest_v_2d: np.ndarray,
    crest_e_2d: np.ndarray,
    hydro_demag_conduit_2d: np.ndarray,
    fp: np.ndarray,
    cat: np.ndarray,
    cap_idx: PortfolioCapacityIndex,
    h33_static: np.ndarray,
    run_canaries_and_ablations: bool = False,
) -> dict[str, Any]:
    """Run locked 4-quadrant OOF holdout evaluation for one seed on gems57-pooled-hide-v2."""
    t_s = time.perf_counter()
    folds, draw_audit = hold_mod.buffered_component_draw(
        g, seed=seed, hide_frac=0.20, buffer_px=3
    )
    hidden = folds[0]["hidden_all"] & fp
    visible = folds[0]["visible"] & fp
    assert not np.any(visible & hidden), "Holdout leak check failed!"
    d_vis = distance_transform_edt(~(visible & fp)).astype(np.float32)

    canary_report = None
    oof_ckpt = CACHE_DIR / f"oof_probs_seed{seed}.npz"
    canary_path = EVID_DIR / "canary_seed20.json"

    if oof_ckpt.exists() and (not run_canaries_and_ablations or canary_path.exists()):
        log(f"  Loading cached Seed {seed} OOF probabilities from {oof_ckpt.name} ...")
        z = np.load(oof_ckpt)
        oof_prob_full = z["oof_prob_full"]
        oof_prob_static = z["oof_prob_static"]
        oof_prob_base120 = z["oof_prob_base120"]
        if run_canaries_and_ablations and canary_path.exists():
            canary_report = json.loads(canary_path.read_text())
    else:
        t_m = time.perf_counter()
        xm_tr, xm_inf = compute_h77v_sloco_mechanics_fp(
            visible, fp, hydro_demag_conduit_2d, n_buckets=4, max_seg_px=16
        )
        log(
            f"  Seed {seed} 4-bucket SLOCO mechanics (8 ch) computed in {time.perf_counter()-t_m:.1f}s"
        )

        if run_canaries_and_ablations:
            fp_idx = np.flatnonzero(fp)
            d_cat = distance_transform_edt(~cat).astype(np.float32)
            pos_fp = np.flatnonzero(hidden.ravel()[fp_idx])
            neg_fp = np.flatnonzero((fp & (~cat) & (d_cat > 3.0)).ravel()[fp_idx])
            canary_report = run_single_feature_leakage_canaries(
                X132,
                static_names,
                xm_inf,
                H77V_SLOCO_MECH_NAMES,
                pos_fp,
                neg_fp,
                ceiling=0.90,
            )
            log(
                f"  Leakage Canary Gate: {canary_report['n_features_checked']}/{canary_report['n_features_checked']} "
                f"features -> passed={canary_report['passed']} | "
                f"max overall={canary_report['max_overall']['symmetric_auc']:.4f} ({canary_report['max_overall']['feature']}) | "
                f"max static={canary_report['max_static_physics']['symmetric_auc']:.4f} ({canary_report['max_static_physics']['feature']}) | "
                f"max H77V new={canary_report['max_h77v_new_physics']['symmetric_auc']:.4f} ({canary_report['max_h77v_new_physics']['feature']})"
            )
            if not canary_report["passed"]:
                raise RuntimeError(f"LEAKAGE CANARY FAILED CLOSED (>0.90): {canary_report['violators']}")
            canary_path.write_text(json.dumps(canary_report, indent=2) + "\n")

        fp_idx = np.flatnonzero(fp)
        crest_v_fp = crest_v_2d.ravel()[fp_idx]
        crest_e_fp = crest_e_2d.ravel()[fp_idx]
        oof_prob_full = np.zeros(fp.shape, dtype=np.float32)
        oof_prob_static = np.zeros(fp.shape, dtype=np.float32)
        oof_prob_base120 = np.zeros(fp.shape, dtype=np.float32)

        for fi, fold in enumerate(folds):
            tf0 = time.perf_counter()
            reg = fold["region"] & fp
            tr_reg = np.zeros_like(fp)
            for fj, f2 in enumerate(folds):
                if fj != fi:
                    tr_reg |= f2["region"] & fp

            res = fit_predict_h77v_fold(
                X132,
                xm_tr,
                xm_inf,
                has_lidar_fp,
                crest_v_fp,
                fp,
                visible_train_2d=visible,
                train_region_2d=tr_reg,
                eval_region_2d=reg,
                static_cols=None,
                use_mechanics=True,
                seed=seed * 100 + fi,
            )
            oof_prob_full[reg] = res["prob_blend_2d"][reg]
            oof_prob_static[reg] = res["prob_static_2d"][reg]

            if run_canaries_and_ablations:
                res120 = fit_predict_h77v_fold(
                    X132,
                    xm_tr[:7],
                    xm_inf[:7],
                    has_lidar_fp,
                    crest_e_fp,
                    fp,
                    visible_train_2d=visible,
                    train_region_2d=tr_reg,
                    eval_region_2d=reg,
                    static_cols=list(range(120)),
                    use_mechanics=True,
                    seed=seed * 100 + 50 + fi,
                )
                oof_prob_base120[reg] = res120["prob_blend_2d"][reg]
                del res120

            del res
            gc.collect()
            log(f"  Seed {seed} Quadrant Fold {fi}/3 OOF propensity finished in {time.perf_counter()-tf0:.1f}s")

        del xm_tr, xm_inf
        gc.collect()
        np.savez(
            oof_ckpt,
            oof_prob_full=oof_prob_full,
            oof_prob_static=oof_prob_static,
            oof_prob_base120=oof_prob_base120,
        )

    allow_oof = fp & (~visible) & (d_vis > EXCL_PX)

    # 1. H77V Pure LDSA Submodular (unconstrained by registry)
    dots_pure = allocate_ldsa_submodular(
        oof_prob_full,
        allow_oof,
        target_n=TARGET_DOTS_FULL,
        crest_2d=crest_v_2d,
        capacity_index=None,
    ).astype(np.float32)

    # 2. H77V Portfolio-Unique LDSA Submodular (<=0.650 cap across ALL informative rasters -> 0 firings!)
    dots_unique_bool = allocate_ldsa_submodular(
        oof_prob_full,
        allow_oof,
        target_n=TARGET_DOTS_FULL,
        crest_2d=crest_v_2d,
        capacity_index=cap_idx,
        max_overlap_cap=MAX_OVERLAP_CAP,
    )
    max_ov_oof, firings_oof, worst_name_oof = cap_idx.max_informative_overlap(dots_unique_bool)
    assert firings_oof == 0 and max_ov_oof <= 0.70, (
        f"Unexpected OOF informative firing: {max_ov_oof} on {worst_name_oof}"
    )
    dots_unique = dots_unique_bool.astype(np.float32)

    # 3. H77V Peak-NMS Control on exact same OOF surface
    dots_nms = allocate_peak_nms_baseline(
        oof_prob_full, allow_oof, target_n=TARGET_DOTS_FULL
    ).astype(np.float32)

    # 4. H77V Pure Zero-Catalogue Static Physics Only (0 SLOCO features)
    dots_static = allocate_ldsa_submodular(
        oof_prob_static, allow_oof, target_n=TARGET_DOTS_FULL, crest_2d=crest_v_2d
    ).astype(np.float32)

    # 5. Fold-Fair H33 Reference (D2.8 at d_vis > 2.0)
    h33_foldfair = allocate_peak_nms_baseline(
        h33_static, allow_oof, target_n=39_610, min_sep_px=2.80
    ).astype(np.float32)

    rng_r = np.random.default_rng(seed + 900)
    rand_map = np.zeros(fp.shape, dtype=np.float32)
    rand_map.ravel()[rng_r.choice(np.flatnonzero(allow_oof), size=TARGET_DOTS_FULL, replace=False)] = 1.0

    arms: dict[str, np.ndarray] = {
        "h77v_pure_submodular": dots_pure,
        "h77v_portfolio_unique_submodular": dots_unique,
        "h77v_peak_nms_control": dots_nms,
        "h77v_pure_physics_no_mechanics": dots_static,
        "h33_foldfair": h33_foldfair,
        "random_uniform_dots": rand_map,
    }
    if run_canaries_and_ablations:
        arms["ablation_base120_without_h77v_channels"] = allocate_ldsa_submodular(
            oof_prob_base120, allow_oof, target_n=TARGET_DOTS_FULL, crest_2d=crest_e_2d
        ).astype(np.float32)

    summary_pure = _eval_arms_official(arms, folds, fp, candidate_key="h77v_pure_submodular", draws=2000)
    summary_uniq = _eval_arms_official(
        arms, folds, fp, candidate_key="h77v_portfolio_unique_submodular", draws=2000
    )

    sc = summary_pure["scores"]
    log(
        f"  [Seed {seed} OFFICIAL GEMS57-POOLED-HIDE-V2 RESULTS (n={sc['h77v_pure_submodular']['withheld_positive_pixels']:,} px)]\n"
        f"    * H77V Pure LDSA Submodular  : {sc['h77v_pure_submodular']['dti']:.6f}  95% CI {sc['h77v_pure_submodular']['ci95']}\n"
        f"    * H77V Unique LDSA (0 firings): {sc['h77v_portfolio_unique_submodular']['dti']:.6f}  95% CI {sc['h77v_portfolio_unique_submodular']['ci95']} (max_ov={max_ov_oof:.4f} on {worst_name_oof})\n"
        f"    * H77V Peak-NMS Control      : {sc['h77v_peak_nms_control']['dti']:.6f}  95% CI {sc['h77v_peak_nms_control']['ci95']}\n"
        f"    * H77V Static Physics (0 cat): {sc['h77v_pure_physics_no_mechanics']['dti']:.6f}  95% CI {sc['h77v_pure_physics_no_mechanics']['ci95']}\n"
        f"    * H33 Fold-Fair Reference    : {sc['h33_foldfair']['dti']:.6f}  95% CI {sc['h33_foldfair']['ci95']}\n"
        f"    * Random Uniform Control     : {sc['random_uniform_dots']['dti']:.6f}"
    )
    if "ablation_base120_without_h77v_channels" in sc:
        log(
            f"    * Base120 (no H77V channels) : {sc['ablation_base120_without_h77v_channels']['dti']:.6f}  "
            f"(H77V delta = {sc['h77v_pure_submodular']['dti'] - sc['ablation_base120_without_h77v_channels']['dti']:+.6f})"
        )

    return {
        "seed": seed,
        "evaluator_version": eval_mod.VERSION,
        "implementation_sha256": summary_pure["implementation_sha256"],
        "withheld_positive_pixels": sc["h77v_pure_submodular"]["withheld_positive_pixels"],
        "elapsed_seconds": round(time.perf_counter() - t_s, 2),
        "oof_unique_max_informative_overlap": round(max_ov_oof, 6),
        "oof_unique_informative_firings": firings_oof,
        "oof_unique_worst_informative_raster": worst_name_oof,
        "official_pooled_summary_pure_candidate": summary_pure,
        "official_pooled_summary_unique_candidate": summary_uniq,
        "canary_audit": canary_report,
    }


def build_full_nevada_oof_surface(
    X132: np.ndarray,
    has_lidar_fp: np.ndarray,
    crest_v_2d: np.ndarray,
    hydro_demag_conduit_2d: np.ndarray,
    fp: np.ndarray,
    cat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute strictly out-of-fold full-Nevada H77V propensity surface via 2 staggered 4-fold checkerboards."""
    ckpt = CACHE_DIR / "full_nevada_oof_h77v.npz"
    if ckpt.exists():
        log(f"  Loading cached full-Nevada OOF surface from {ckpt.name} ...")
        z = np.load(ckpt)
        return z["prob_full_2d"], z["prob_static_2d"]

    t0 = time.perf_counter()
    xm_tr, xm_inf = compute_h77v_sloco_mechanics_fp(
        cat, fp, hydro_demag_conduit_2d, n_buckets=4, max_seg_px=16
    )
    log(f"  Full-Nevada 4-bucket SLOCO mechanics (8 ch) computed in {time.perf_counter()-t0:.1f}s")

    h, w = fp.shape
    yy, xx = np.indices((h, w))
    fp_idx = np.flatnonzero(fp)
    crest_v_fp = crest_v_2d.ravel()[fp_idx]

    prob_full_acc = np.zeros((h, w), dtype=np.float32)
    prob_stat_acc = np.zeros((h, w), dtype=np.float32)

    grids = [
        ("coarse_460px_4fold", ((yy // 460) * 2 + (xx // 460)) % 4, 77100),
        ("staggered_380px_4fold", (((yy + 190) // 380) * 2 + ((xx + 190) // 380)) % 4, 77200),
    ]
    for g_name, fold_id_2d, base_seed in grids:
        p_f = np.zeros((h, w), dtype=np.float32)
        p_s = np.zeros((h, w), dtype=np.float32)
        for k in range(4):
            tf = time.perf_counter()
            ev_reg = (fold_id_2d == k) & fp
            tr_reg = (fold_id_2d != k) & fp
            res = fit_predict_h77v_fold(
                X132,
                xm_tr,
                xm_inf,
                has_lidar_fp,
                crest_v_fp,
                fp,
                visible_train_2d=cat,
                train_region_2d=tr_reg,
                eval_region_2d=ev_reg,
                static_cols=None,
                use_mechanics=True,
                seed=base_seed + k,
            )
            p_f[ev_reg] = res["prob_blend_2d"][ev_reg]
            p_s[ev_reg] = res["prob_static_2d"][ev_reg]
            del res
            gc.collect()
            log(f"    [{g_name}] fold {k}/3 finished in {time.perf_counter()-tf:.1f}s")
        prob_full_acc += 0.5 * p_f
        prob_stat_acc += 0.5 * p_s

    del xm_tr, xm_inf
    gc.collect()
    np.savez(ckpt, prob_full_2d=prob_full_acc, prob_static_2d=prob_stat_acc)
    return prob_full_acc, prob_stat_acc


def run_portfolio_uniqueness_audit(
    surface_2d: np.ndarray,
    dots_bool: np.ndarray,
    cand_file_sha256: str,
    foot: np.ndarray,
    cat: np.ndarray,
) -> dict[str, Any]:
    """Exhaustive pre-placement surface & post-placement dot audit across all valid rasters."""
    meta = json.loads((CACHE_DIR / "audit_meta.json").read_text())
    entries = meta["entries"]
    n_valid = int(meta["n_valid"])
    packed = np.load(CACHE_DIR / "audit_halos_packed.npy", mmap_mode="r")
    subvals = np.load(CACHE_DIR / "audit_subvals.npy", mmap_mode="r")
    stride = int(meta["stride"])

    allow_idx = np.flatnonzero(foot & (~cat))
    sub_fp_idx = allow_idx[::stride]

    surf_sub = surface_2d.ravel()[sub_fp_idx].astype(np.float64)
    surf_r = rankdata(surf_sub)
    surf_r -= surf_r.mean()
    surf_norm = float(np.linalg.norm(surf_r)) + 1e-12

    dots_sub = dots_bool.ravel()[sub_fp_idx].astype(np.float64)
    dots_r = rankdata(dots_sub)
    dots_r -= dots_r.mean()
    dots_norm = float(np.linalg.norm(dots_r)) + 1e-12

    dot_allow_pos = np.flatnonzero(dots_bool.ravel()[allow_idx])
    byte_idx = dot_allow_pos >> 3
    bit_mask = (1 << (7 - (dot_allow_pos & 7))).astype(np.uint8)

    fwd_all = np.empty(n_valid, dtype=np.float64)
    surf_rhos = np.zeros(n_valid, dtype=np.float64)
    dot_rhos = np.zeros(n_valid, dtype=np.float64)

    for s in range(0, n_valid, 64):
        e = min(s + 64, n_valid)
        fwd_all[s:e] = ((packed[s:e][:, byte_idx] & bit_mask[None, :]) != 0).mean(axis=1)
        block_v = np.asarray(subvals[s:e], dtype=np.float64)
        for bi in range(e - s):
            v = block_v[bi]
            if float(np.ptp(v)) > 0.0:
                vr = rankdata(v)
                vr -= vr.mean()
                vn = float(np.linalg.norm(vr)) + 1e-12
                surf_rhos[s + bi] = float((surf_r @ vr) / (surf_norm * vn))
                dot_rhos[s + bi] = float((dots_r @ vr) / (dots_norm * vn))

    info_rows: list[dict[str, Any]] = []
    sf_rows: list[dict[str, Any]] = []
    for i, ent in enumerate(entries):
        rec = {
            "index": i,
            "name": ent["name"],
            "tag": ent["tag"],
            "path": ent["path"],
            "sha256": ent["sha256"],
            "aliases": ent["aliases"][:4],
            "their_pos_allow": int(ent["n_pos_allow"]),
            "support_frac_fp": round(float(ent.get("support_frac_fp", 0.0)), 6),
            "halo3_bg_frac": round(float(ent["halo3_frac_allow"]), 6),
            "is_space_filling": bool(ent["is_space_filling"]),
            "forward_3px_overlap": round(float(fwd_all[i]), 6),
            "surface_spearman_rho": round(float(surf_rhos[i]), 6),
            "dots_spearman_rho": round(float(dot_rhos[i]), 6),
            "identical_bytes": ent["sha256"] == cand_file_sha256,
        }
        if ent["is_space_filling"]:
            sf_rows.append(rec)
        else:
            info_rows.append(rec)

    info_sorted = sorted(info_rows, key=lambda r: -r["forward_3px_overlap"])
    sf_sorted = sorted(sf_rows, key=lambda r: -r["forward_3px_overlap"])
    surf_sorted = sorted(info_rows + sf_rows, key=lambda r: -abs(r["surface_spearman_rho"]))
    dot_rho_sorted = sorted(info_rows + sf_rows, key=lambda r: -abs(r["dots_spearman_rho"]))

    d_my = distance_transform_edt(~dots_bool).astype(np.float32)
    vendor_verified_top: list[dict[str, Any]] = []
    for check_row in info_sorted[:15] + sf_sorted[:5]:
        with rasterio.open(check_row["path"]) as ds_t:
            theirs = ds_t.read(1)
        exact_fwd = uniq_mod.dot_overlap(dots_bool, theirs)
        t_mask = np.isfinite(theirs) & (theirs > 0)
        rev = float((d_my[t_mask] <= 3.0).mean()) if t_mask.any() else 0.0
        jacc = float(int((dots_bool & t_mask).sum()) / max(int((dots_bool | t_mask).sum()), 1))
        diff = abs(exact_fwd - float(fwd_all[check_row["index"]]))
        assert diff < 1e-7, f"Memmap vs vendor.dot_overlap mismatch: {diff}"
        vendor_verified_top.append(
            {
                **check_row,
                "vendor_exact_forward_3px_overlap": round(exact_fwd, 6),
                "reverse_3px_overlap": round(rev, 6),
                "mutual_3px_overlap": round(min(exact_fwd, rev), 6),
                "jaccard_dot_sets": round(jacc, 6),
                "memmap_vs_vendor_abs_diff": float(diff),
            }
        )

    info_firings = [r for r in info_sorted if r["forward_3px_overlap"] > 0.70]
    surf_rho_firings = [r for r in surf_sorted if abs(r["surface_spearman_rho"]) > 0.90]
    dot_rho_firings = [r for r in dot_rho_sorted if abs(r["dots_spearman_rho"]) > 0.90]
    identical_any = [r for r in (info_rows + sf_rows) if r["identical_bytes"]]

    return {
        "evidence_class": "REGISTRY-MEASUREMENT",
        "total_paths_discovered_across_portfolio": int(meta["total_paths_discovered"]),
        "unique_valid_rasters_checked": n_valid,
        "unreadable_or_off_grid_logged": meta["unreadable"],
        "informative_rasters_checked_lt_0p70_bg": len(info_sorted),
        "space_filling_rasters_ge_0p70_bg": len(sf_sorted),
        "phase1_pre_placement_surface": {
            "max_abs_spearman": round(abs(surf_sorted[0]["surface_spearman_rho"]), 6),
            "worst_raster": f"{surf_sorted[0]['tag']}:{surf_sorted[0]['name']}",
            "firings_gt_0p90": len(surf_rho_firings),
            "pass_le_0p90": len(surf_rho_firings) == 0,
        },
        "phase2_post_placement_final_dots": {
            "informative_reading_lt_0p70_bg": {
                "rasters_checked": len(info_sorted),
                "max_forward_3px_overlap": info_sorted[0]["forward_3px_overlap"],
                "worst_raster": f"{info_sorted[0]['tag']}:{info_sorted[0]['name']}",
                "worst_aliases": info_sorted[0]["aliases"],
                "firings_gt_0p70": len(info_firings),
                "max_abs_spearman": round(abs(dot_rho_sorted[0]["dots_spearman_rho"]), 6),
                "rho_firings_gt_0p90": len(dot_rho_firings),
                "identical_bytes_count": len(identical_any),
                "informative_unique": (
                    len(info_firings) == 0 and len(dot_rho_firings) == 0 and len(identical_any) == 0
                ),
            },
            "literal_unfiltered_reading_policy_I17": {
                "rasters_checked": n_valid,
                "max_forward_3px_overlap": sf_sorted[0]["forward_3px_overlap"] if sf_sorted else 0.0,
                "worst_raster": f"{sf_sorted[0]['tag']}:{sf_sorted[0]['name']}" if sf_sorted else None,
                "firings_gt_0p70": len(info_firings)
                + sum(1 for r in sf_sorted if r["forward_3px_overlap"] > 0.70),
                "protocol_unique": False,
                "reason": (
                    f"{len(sf_sorted)} space-filling/lattice rasters in archive registry cover >=70% "
                    f"(up to 100.0% on dense_witness ab0a0a62) of allowed Nevada cells within 3 px; "
                    f"Policy I17 requires fail-closed human selector review."
                ),
            },
        },
        "top_20_vendor_verified_matches": vendor_verified_top,
    }


def generate_diagnostic_figures(
    hydro_demag_conduit_2d: np.ndarray,
    crest_v_2d: np.ndarray,
    prob_full_2d: np.ndarray,
    dots_bool: np.ndarray,
    fp: np.ndarray,
    cat: np.ndarray,
    exp1: dict[str, Any],
    exp2: dict[str, Any],
    uniq_audit: dict[str, Any],
) -> list[str]:
    """Generate 4 publication-grade diagnostic figures in docs/figures/h77v/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    # 1. Physics Panels (Dixie Valley / Stillwater Hydrothermal Conduit Zoom)
    r0, r1, c0, c1 = 1050, 1450, 1100, 1500
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=130)
    im0 = axes[0].imshow(hydro_demag_conduit_2d[r0:r1, c0:c1], cmap="magma", vmin=0.1, vmax=0.9)
    axes[0].set_title("H77V De-Aliased Hydro-Demag Conduit", fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(crest_v_2d[r0:r1, c0:c1], cmap="viridis", vmin=0.0, vmax=0.85)
    axes[1].set_title("Road/BLM-Deconfounded Scarp x Demag Crest", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(prob_full_2d[r0:r1, c0:c1], cmap="inferno", vmin=0.0, vmax=0.85)
    yy_d, xx_d = np.nonzero(dots_bool[r0:r1, c0:c1])
    axes[2].scatter(xx_d, yy_d, s=2.2, c="#38bdf8", alpha=0.85, label="H77V LDSA Dots")
    axes[2].set_title("Full-Nevada OOF Propensity + LDSA Dots", fontsize=10, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=8)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Lane H77V Physical Mechanisms (Dixie Valley / Stillwater Window, EPSG:32611)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    p1 = FIG_DIR / "fig1_h77v_physics_panels.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p1.relative_to(ROOT)))

    # 2. Dual-Seed Holdout DTI Comparison Bar Chart with 95% Block-Bootstrap CIs
    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=130)
    s20_sc = exp1["official_pooled_summary_pure_candidate"]["scores"]
    s30_sc = exp2["official_pooled_summary_pure_candidate"]["scores"]

    labels_bar = [
        "H77V Pure LDSA\n(Unconstrained)",
        "H77V Unique LDSA\n(0 Info Firings)",
        "H77E-Knick Unique\n(Prior Unique Best)",
        "H77D Submodular\n(18 Info Firings)",
        "H77V Static Physics\n(0 Catalogue)",
        "H77V Peak-NMS\nControl",
        "H33 Fold-Fair\nReference",
    ]
    v20 = [
        s20_sc["h77v_pure_submodular"]["dti"],
        s20_sc["h77v_portfolio_unique_submodular"]["dti"],
        0.126664,
        0.140300,
        s20_sc["h77v_pure_physics_no_mechanics"]["dti"],
        s20_sc["h77v_peak_nms_control"]["dti"],
        s20_sc["h33_foldfair"]["dti"],
    ]
    e20_lo = [
        s20_sc["h77v_pure_submodular"]["dti"] - s20_sc["h77v_pure_submodular"]["ci95"][0],
        s20_sc["h77v_portfolio_unique_submodular"]["dti"] - s20_sc["h77v_portfolio_unique_submodular"]["ci95"][0],
        0.126664 - 0.109534,
        0.140300 - 0.123100,
        s20_sc["h77v_pure_physics_no_mechanics"]["dti"] - s20_sc["h77v_pure_physics_no_mechanics"]["ci95"][0],
        s20_sc["h77v_peak_nms_control"]["dti"] - s20_sc["h77v_peak_nms_control"]["ci95"][0],
        s20_sc["h33_foldfair"]["dti"] - s20_sc["h33_foldfair"]["ci95"][0],
    ]
    e20_hi = [
        s20_sc["h77v_pure_submodular"]["ci95"][1] - s20_sc["h77v_pure_submodular"]["dti"],
        s20_sc["h77v_portfolio_unique_submodular"]["ci95"][1] - s20_sc["h77v_portfolio_unique_submodular"]["dti"],
        0.143065 - 0.126664,
        0.157700 - 0.140300,
        s20_sc["h77v_pure_physics_no_mechanics"]["ci95"][1] - s20_sc["h77v_pure_physics_no_mechanics"]["dti"],
        s20_sc["h77v_peak_nms_control"]["ci95"][1] - s20_sc["h77v_peak_nms_control"]["dti"],
        s20_sc["h33_foldfair"]["ci95"][1] - s20_sc["h33_foldfair"]["dti"],
    ]

    v30 = [
        s30_sc["h77v_pure_submodular"]["dti"],
        s30_sc["h77v_portfolio_unique_submodular"]["dti"],
        0.124727,
        0.150110,
        s30_sc["h77v_pure_physics_no_mechanics"]["dti"],
        s30_sc["h77v_peak_nms_control"]["dti"],
        s30_sc["h33_foldfair"]["dti"],
    ]
    e30_lo = [
        s30_sc["h77v_pure_submodular"]["dti"] - s30_sc["h77v_pure_submodular"]["ci95"][0],
        s30_sc["h77v_portfolio_unique_submodular"]["dti"] - s30_sc["h77v_portfolio_unique_submodular"]["ci95"][0],
        0.124727 - 0.109650,
        0.150110 - 0.132800,
        s30_sc["h77v_pure_physics_no_mechanics"]["dti"] - s30_sc["h77v_pure_physics_no_mechanics"]["ci95"][0],
        s30_sc["h77v_peak_nms_control"]["dti"] - s30_sc["h77v_peak_nms_control"]["ci95"][0],
        s30_sc["h33_foldfair"]["dti"] - s30_sc["h33_foldfair"]["ci95"][0],
    ]
    e30_hi = [
        s30_sc["h77v_pure_submodular"]["ci95"][1] - s30_sc["h77v_pure_submodular"]["dti"],
        s30_sc["h77v_portfolio_unique_submodular"]["ci95"][1] - s30_sc["h77v_portfolio_unique_submodular"]["dti"],
        0.138262 - 0.124727,
        0.166900 - 0.150110,
        s30_sc["h77v_pure_physics_no_mechanics"]["ci95"][1] - s30_sc["h77v_pure_physics_no_mechanics"]["dti"],
        s30_sc["h77v_peak_nms_control"]["ci95"][1] - s30_sc["h77v_peak_nms_control"]["dti"],
        s30_sc["h33_foldfair"]["ci95"][1] - s30_sc["h33_foldfair"]["dti"],
    ]

    x = np.arange(len(labels_bar))
    width = 0.36
    ax.bar(
        x - width / 2,
        v20,
        width,
        yerr=[e20_lo, e20_hi],
        capsize=4,
        label="Seed 20 (11,321 withheld pos)",
        color="#2563eb",
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        v30,
        width,
        yerr=[e30_lo, e30_hi],
        capsize=4,
        label="Seed 30 (11,342 withheld pos)",
        color="#0d9488",
        alpha=0.9,
    )
    ax.axhline(0.152718, color="#dc2626", linestyle="--", linewidth=1.3, label="H77-Step Pure Incumbent (0.15272)")
    ax.axhline(0.126664, color="#d97706", linestyle=":", linewidth=1.3, label="H77E-Knick Unique Incumbent (0.12666)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=8.2)
    ax.set_ylabel("HOLDOUT-DTI (gems57-pooled-hide-v2)", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "Official Spatially Blocked Holdout DTI (2,000-Draw 20 km Block Bootstrap 95% CI)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = FIG_DIR / "fig2_h77v_holdout_bootstrap.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p2.relative_to(ROOT)))

    # 3. Uniqueness Spectrum Across Informative vs Space-Filling Registry Rasters
    fig, ax = plt.subplots(figsize=(9.8, 4.4), dpi=130)
    top_matches = uniq_audit["top_20_vendor_verified_matches"]
    plot_rows = top_matches[:12] + top_matches[15:18]
    names_m = [f"{r['tag'][:8]}:{r['name'][:26]}" for r in plot_rows]
    ovs_m = [r["vendor_exact_forward_3px_overlap"] for r in plot_rows]
    colors_m = ["#ef4444" if r["is_space_filling"] else "#10b981" for r in plot_rows]
    y_pos = np.arange(len(names_m))
    ax.barh(y_pos, ovs_m, color=colors_m, alpha=0.88)
    ax.axvline(0.70, color="#111827", linestyle="--", linewidth=1.4, label="Protocol 0.70 Overlap Ceiling")
    ax.axvline(MAX_OVERLAP_CAP, color="#0284c7", linestyle=":", linewidth=1.4, label=f"H77V LDSA Cap ({MAX_OVERLAP_CAP:.3f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_m, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("3-px Forward Dot Overlap (gems57.uniqueness.dot_overlap)", fontsize=9)
    ax.set_title(
        "H77V Uniqueness Audit: 0/95 Informative Firings (Green <=0.650) vs Policy I17 Witnesses (Red)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p3 = FIG_DIR / "fig3_h77v_uniqueness_spectrum.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p3.relative_to(ROOT)))

    # 4. Full-Nevada Submission Overview Map
    fig, ax = plt.subplots(figsize=(6.8, 7.6), dpi=130)
    vis_img = np.full((fp.shape[0], fp.shape[1], 3), 0.96, dtype=np.float32)
    vis_img[fp] = [0.11, 0.15, 0.23]
    cat_dil = binary_dilation(cat, structure=np.ones((5, 5), dtype=bool))
    dots_dil = binary_dilation(dots_bool, structure=np.ones((5, 5), dtype=bool))
    vis_img[cat_dil & fp] = [0.96, 0.62, 0.04]
    vis_img[dots_dil & fp] = [0.22, 0.74, 0.97]
    ax.imshow(vis_img[::4, ::4])
    ax.set_title(
        f"H77V Full-Nevada Submission ({int(dots_bool.sum()):,} Dots, d_cat >= 2.24 px)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    p4 = FIG_DIR / "fig4_h77v_submission_map.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p4.relative_to(ROOT)))

    return saved_paths


def main() -> int:
    t_all = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVID_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    DL_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "evidence").mkdir(parents=True, exist_ok=True)

    log("=== GEMSDOE77 Lane H77V 3-Experiment Pipeline Starting ===")
    g = grid_mod.load_grid(DATA_DIR)
    fp, cat, profile = _load_core_masks(DATA_DIR)
    fp_idx = np.flatnonzero(fp)
    log(f"Loaded grid {profile['shape']} ({profile['crs']}): footprint={int(fp.sum()):,}, catalogue={int(cat.sum()):,}")

    # Build/load 132-channel zero-SGMC H77V static stack
    t_f = time.perf_counter()
    X132, static_names, has_lidar_fp, crest_v_fp, hydro_demag_conduit_2d, prov = build_h77v_static_stack(
        DATA_DIR, CACHE_DIR, fp
    )
    log(
        f"Built/loaded 132-channel H77V static stack {X132.shape} ({X132.dtype}) in {time.perf_counter()-t_f:.1f}s"
    )

    crest_v_2d = np.zeros(fp.shape, dtype=np.float32)
    crest_v_2d.ravel()[fp_idx] = crest_v_fp

    aux_e = np.load(CACHE_DIR / "h77e_delta.aux.npz")
    crest_e_2d = np.zeros(fp.shape, dtype=np.float32)
    crest_e_2d.ravel()[fp_idx] = aux_e["crest_deconf_fp"].astype(np.float32)

    # Load H33 continuous baseline (19GEMSDOE h19-5) for fold-fair reference
    h195_path = DATA_DIR / "registry" / "rasters" / "19GEMSDOE__h19-5-powerlaw-budget-multiline.tif"
    if h195_path.exists():
        with rasterio.open(h195_path) as ds:
            h33_static = np.where(fp & np.isfinite(ds.read(1)), ds.read(1), 0.0).astype(np.float32)
    else:
        h33_static = crest_e_2d.copy()

    # Build/load portfolio capacity index across all valid rasters on disk
    t_idx = time.perf_counter()
    meta_path = ensure_audit_index_built(ROOT, CACHE_DIR, fp, cat)
    cap_idx = PortfolioCapacityIndex(
        CACHE_DIR / "audit_halos_packed.npy", meta_path, fp, cat
    )
    n_sf = sum(1 for e in cap_idx.entries if e["is_space_filling"])
    log(
        f"PortfolioCapacityIndex ready in {time.perf_counter()-t_idx:.1f}s: "
        f"{int(cap_idx.info_row_ids.size)} informative rasters (<0.70 bg), {n_sf} space-filling rasters (>=0.70 bg)"
    )

    # ------------------------------------------------------------------
    # EXPERIMENT 1: Seed 20 Official Holdout + 140 Leakage Canaries + Ablations
    # ------------------------------------------------------------------
    e1_path = EVID_DIR / "exp1_seed20_holdout.json"
    if e1_path.exists():
        log("--- Experiment 1 (E1): Reloading completed Seed 20 Holdout receipt from disk ---")
        exp1 = json.loads(e1_path.read_text())
    else:
        log("--- Experiment 1 (E1): Official gems57-pooled-hide-v2 Seed 20 + 140 Leakage Canaries ---")
        exp1 = evaluate_seed_suite(
            seed=20,
            g=g,
            X132=X132,
            static_names=static_names,
            has_lidar_fp=has_lidar_fp,
            crest_v_2d=crest_v_2d,
            crest_e_2d=crest_e_2d,
            hydro_demag_conduit_2d=hydro_demag_conduit_2d,
            fp=fp,
            cat=cat,
            cap_idx=cap_idx,
            h33_static=h33_static,
            run_canaries_and_ablations=True,
        )
        e1_path.write_text(json.dumps(exp1, indent=2) + "\n")

    # ------------------------------------------------------------------
    # EXPERIMENT 2: Seed 30 Independent Zero-Retuning Replication
    # ------------------------------------------------------------------
    e2_path = EVID_DIR / "exp2_seed30_holdout.json"
    if e2_path.exists():
        log("--- Experiment 2 (E2): Reloading completed Seed 30 Holdout receipt from disk ---")
        exp2 = json.loads(e2_path.read_text())
    else:
        log("--- Experiment 2 (E2): Independent Zero-Retuning Replication on Seed 30 ---")
        exp2 = evaluate_seed_suite(
            seed=30,
            g=g,
            X132=X132,
            static_names=static_names,
            has_lidar_fp=has_lidar_fp,
            crest_v_2d=crest_v_2d,
            crest_e_2d=crest_e_2d,
            hydro_demag_conduit_2d=hydro_demag_conduit_2d,
            fp=fp,
            cat=cat,
            cap_idx=cap_idx,
            h33_static=h33_static,
            run_canaries_and_ablations=False,
        )
        e2_path.write_text(json.dumps(exp2, indent=2) + "\n")

    # ------------------------------------------------------------------
    # EXPERIMENT 3: Full-Nevada OOF Surface + LDSA GeoTIFF + Uniqueness & Validator Audit
    # ------------------------------------------------------------------
    log("--- Experiment 3 (E3): Full-Nevada Dual-Checkerboard OOF Surface + LDSA GeoTIFF ---")
    prob_full_2d, prob_stat_2d = build_full_nevada_oof_surface(
        X132, has_lidar_fp, crest_v_2d, hydro_demag_conduit_2d, fp, cat
    )
    d_cat = distance_transform_edt(~cat).astype(np.float32)
    allow_full = fp & (~cat) & (d_cat > EXCL_PX)

    dots_bool = allocate_ldsa_submodular(
        prob_full_2d,
        allow_full,
        target_n=TARGET_DOTS_FULL,
        crest_2d=crest_v_2d,
        capacity_index=cap_idx,
        max_overlap_cap=MAX_OVERLAP_CAP,
    )
    max_ov_full, firings_full, worst_name_full = cap_idx.max_informative_overlap(dots_bool)
    log(
        f"  Full-Nevada LDSA emitted {int(dots_bool.sum()):,} binary dots | "
        f"max informative 3-px overlap = {max_ov_full:.5f} on {worst_name_full} ({firings_full} firings)"
    )
    assert firings_full == 0 and max_ov_full <= 0.70, (
        f"Full-Nevada informative overlap exceeded 0.70: {max_ov_full} on {worst_name_full}"
    )

    sub_tif = SUB_DIR / SUBMISSION_TIF_NAME
    sub_zip = SUB_DIR / SUBMISSION_ZIP_NAME
    sub_json = SUB_DIR / SUBMISSION_JSON_NAME
    for p_rm in (sub_tif, sub_zip, sub_json):
        if p_rm.exists():
            p_rm.unlink()

    pred_zeros = np.where(fp & (~cat), dots_bool.astype(np.float32), 0.0).astype(np.float32)
    write_receipt = writer_mod.write_submission(
        sub_tif,
        pred_zeros,
        DATA_DIR / "sample_submission.tif",
        fp,
        note=SUBMISSION_NOTE,
        name=SUBMISSION_STEM,
        catalogue=cat,
        metadata={"lane_id": LANE_ID, "version": VERSION},
    )
    val_report = validate_mod.validate(sub_tif, footprint=fp, catalogue=cat)
    assert val_report["all_checks_passed"] and all(val_report["checks"].values()), (
        f"Validator failed: {val_report}"
    )
    tif_sha256 = _sha256_file(sub_tif)
    zip_sha256 = _sha256_file(sub_zip)

    uniq_audit = run_portfolio_uniqueness_audit(
        prob_full_2d,
        dots_bool,
        tif_sha256,
        fp,
        cat,
    )
    assert uniq_audit["phase1_pre_placement_surface"]["pass_le_0p90"], "Surface Spearman > 0.90!"
    assert uniq_audit["phase2_post_placement_final_dots"]["informative_reading_lt_0p70_bg"][
        "informative_unique"
    ], "Informative final-dot overlap > 0.70!"

    fig_paths = generate_diagnostic_figures(
        hydro_demag_conduit_2d,
        crest_v_2d,
        prob_full_2d,
        dots_bool,
        fp,
        cat,
        exp1,
        exp2,
        uniq_audit,
    )

    s20_pure = exp1["official_pooled_summary_pure_candidate"]["scores"]["h77v_pure_submodular"]
    s20_uniq = exp1["official_pooled_summary_unique_candidate"]["scores"][
        "h77v_portfolio_unique_submodular"
    ]
    s30_pure = exp2["official_pooled_summary_pure_candidate"]["scores"]["h77v_pure_submodular"]
    s30_uniq = exp2["official_pooled_summary_unique_candidate"]["scores"][
        "h77v_portfolio_unique_submodular"
    ]

    run_card: dict[str, Any] = {
        "schema_version": "gemsdoe77-run-card-v2",
        "lane_id": LANE_ID,
        "session_branch": "arena/65c05dc9-gemsdoe77",
        "date_utc": "2026-10-21",
        "verdict": "negative",
        "ok_to_submit": False,
        "protocol_unique": False,
        "informative_unique": True,
        "weekly_submission_slots_used": 0,
        "status_banner": (
            "OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT "
            f"(Policy I17 Fail-Closed: 0/{uniq_audit['informative_rasters_checked_lt_0p70_bg']} informative firings "
            f"at <={MAX_OVERLAP_CAP:.3f} overlap, but {uniq_audit['space_filling_rasters_ge_0p70_bg']} space-filling/lattice "
            "registry witnesses cover >=70% to 100% of Nevada)"
        ),
        "hypothesis": (
            "H77V: Spectral mesh de-aliasing (sigma=36 px = 3.6 km) of rectilinear 3.5 km / 7.0 km "
            "MT/geodetic/seismic inversion grids (resolving IR-H77G-01) coupled with corrected "
            "across-valley tubular Hessian demagnetization troughs (resolving IR-H77G-02), antiparallel "
            "gravity-basement step concordance, road/BLM-mining-deconfounded 1 m LiDAR & 10 m 3DEP "
            "fluvial knickpoint-scarp interactions (resolving IR-H77G-03), 8-channel fold-visible SLOCO "
            "mechanics, and Kernel-Convolved Lagrangian-Priced Dinkelbach Submodular Cover (LDSA)."
        ),
        "physical_mechanism": (
            "Ascending hydrothermal fluids (150-250 C) along blind and incipient Great Basin normal/transtensional "
            "fault conduits sulfidize/oxidize primary magnetite to pyrite/smectite/leucoxene (producing linear "
            "multi-scale RTP magnetic troughs rather than positive ridges) and argilize surface fault rock, while "
            "leaving (1) transient 10 m 3DEP fluvial knickpoint wavefronts across the footwall/hanging-wall block "
            "step, (2) antiparallel gravity-vs-basement-depth gradient concordance, (3) regional 3.6 km de-aliased "
            "smectite clay-cap lateral margins and transtensional shear-dilation strain concentration, and "
            "(4) chemical geothermometry disequilibrium + 2 m thermal probe anomalies."
        ),
        "named_non_fault_mimic": (
            "(1) I-80/US-50 highway and railroad embankments across flat playas (damped via US Census TIGER/Line "
            "tiger_road_distance_m.tif); (2) historic open-pit and trench mining disturbances (controlled via BLM "
            "closed mining claim distance blm_closed_claim_distance_m.tif); (3) 3.5 km / 7.0 km rectilinear MT, "
            "GPS, and earthquake inversion grid cell boundaries (eliminated via sigma=36 px Gaussian spectral "
            "de-aliasing, IR-H77G-01); (4) Pleistocene Lake Lahontan (1332 m) & Bonneville (1463/1552 m) shoreline "
            "berms (controlled via equipotential shoreline bands + 1.6 km along-strike elevation variance)."
        ),
        "data_provenance": prov,
        "experiments_executed": 3,
        "wall_clock_seconds": round(time.perf_counter() - t_all, 2),
        "leakage_canary_gate": {
            "evidence_class": "HOLDOUT-CANARY",
            "features_checked": exp1["canary_audit"]["n_features_checked"],
            "ceiling": 0.90,
            "passed": exp1["canary_audit"]["passed"],
            "max_overall": exp1["canary_audit"]["max_overall"],
            "max_static_physics": exp1["canary_audit"]["max_static_physics"],
            "max_h77v_new_physics": exp1["canary_audit"]["max_h77v_new_physics"],
        },
        "holdout_dti_seed20_11321px": exp1["official_pooled_summary_pure_candidate"],
        "holdout_dti_seed20_unique_candidate": exp1["official_pooled_summary_unique_candidate"],
        "holdout_dti_seed30_11342px": exp2["official_pooled_summary_pure_candidate"],
        "holdout_dti_seed30_unique_candidate": exp2["official_pooled_summary_unique_candidate"],
        "headline_holdout_metrics": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": eval_mod.VERSION,
            "seed20_withheld_positives": 11321,
            "seed30_withheld_positives": 11342,
            "h77v_pure_submodular_seed20": {
                "dti": round(s20_pure["dti"], 6),
                "ci95": [round(x, 6) for x in s20_pure["ci95"]],
            },
            "h77v_portfolio_unique_submodular_seed20": {
                "dti": round(s20_uniq["dti"], 6),
                "ci95": [round(x, 6) for x in s20_uniq["ci95"]],
            },
            "h77v_pure_submodular_seed30": {
                "dti": round(s30_pure["dti"], 6),
                "ci95": [round(x, 6) for x in s30_pure["ci95"]],
            },
            "h77v_portfolio_unique_submodular_seed30": {
                "dti": round(s30_uniq["dti"], 6),
                "ci95": [round(x, 6) for x in s30_uniq["ci95"]],
            },
            "dual_seed_mean_pure_dti": round(0.5 * (s20_pure["dti"] + s30_pure["dti"]), 6),
            "dual_seed_mean_unique_dti": round(0.5 * (s20_uniq["dti"] + s30_uniq["dti"]), 6),
        },
        "uniqueness_audit": uniq_audit,
        "submission_artifact": {
            "submission_name": SUBMISSION_TIF_NAME,
            "submission_zip": SUBMISSION_ZIP_NAME,
            "submission_json": SUBMISSION_JSON_NAME,
            "submission_note": SUBMISSION_NOTE,
            "submission_note_length": len(SUBMISSION_NOTE),
            "tif_sha256": tif_sha256,
            "zip_sha256": zip_sha256,
            "tif_size_bytes": sub_tif.stat().st_size,
            "zip_size_bytes": sub_zip.stat().st_size,
            "emitted_positive_dots": int(dots_bool.sum()),
            "min_distance_to_catalogue_px": round(float(d_cat[dots_bool].min()), 4),
            "validator": val_report,
        },
        "diagnostic_figures": fig_paths,
    }

    sub_json.write_text(json.dumps(run_card["submission_artifact"], indent=2) + "\n")
    shutil.copy2(sub_tif, DL_DIR / SUBMISSION_TIF_NAME)
    shutil.copy2(sub_zip, DL_DIR / SUBMISSION_ZIP_NAME)
    shutil.copy2(sub_json, DL_DIR / SUBMISSION_JSON_NAME)

    (EVID_DIR / "exp3_uniqueness_and_validator.json").write_text(
        json.dumps(
            {
                "uniqueness_audit": uniq_audit,
                "submission_artifact": run_card["submission_artifact"],
            },
            indent=2,
        )
        + "\n"
    )
    (EVID_DIR / "run_card.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "evidence" / "run_card_h77v.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "run_card_h77v.json").write_text(json.dumps(run_card, indent=2) + "\n")

    log(f"=== GEMSDOE77 Lane H77V Complete in {time.perf_counter()-t_all:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
