#!/usr/bin/env python3
"""Experiments 1 and 2 of 3 (H77-BGB):
  - Experiment 1: Build 10-channel H77-BGB feature stack, run single-feature ROC-AUC leakage
    canaries on withheld fault segments (seed 30 & seed 20), run Phase 1 pre-placement surface
    uniqueness check against the 812-raster registry + local sibling TIFFs, and evaluate the
    standalone label-free physical surfaces (S_growth_wedge, S_strain_part, S_upflow_breach,
    S_bgb_composite) vs A0_random on vendor/gems57/evaluate_holdout.py (gems57-pooled-hide-v2).
  - Experiment 2: Out-of-fold 4-quadrant spatially blocked holdout evaluation comparing:
      * A0_random (uniform random dots control)
      * A_bgb_surface (label-free S_bgb_composite + 300 m submodular 37,654 dots)
      * A_bgb_hgb_only (OOF HistGradientBoosting on the 10 H77-BGB channels alone)
      * A_bgb_lane (OOF HistGradientBoosting on the 10 H77-BGB channels blended with S_bgb_composite)
      * A_std (OOF HistGradientBoosting on shared 33-channel stack.npy baseline: raw 19 + terrain 8 + rad 6)
      * A_std_plus_bgb (OOF HistGradientBoosting on 33-channel baseline + 10 H77-BGB channels = 43 static channels)
      * A_bgb_loco (43 static channels + 4 strictly fold-visible LOCO structural mechanics channels)
    evaluated on both Seed 30 (11,342 withheld positives) and Seed 20 (11,321 withheld positives).
"""
from __future__ import annotations

import gc
import json
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
sys.path.insert(0, str(ROOT / "scripts"))

from gems57 import evaluate_holdout as EH, grid as G, holdout as HO
from gems77 import features77 as F77
from gems77_bgb import features as FBGB
from gems77d import features as F77D
from gems77d.submodular import allocate_submodular_300m

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
EVID = ROOT / "evidence"
BGB_STACK_PATH = PROC / "bgb_stack.npy"
STD_STACK_PATH = PROC / "stack.npy"
STD_META_PATH = PROC / "stack.meta.json"

TARGET_DOTS = 37654
LEAKAGE_AUC_MAX = 0.90

HGB_KW = dict(
    max_iter=220,
    learning_rate=0.055,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=2.0,
    random_state=7701,
)


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run_leakage_canaries(
    bgb_stack: np.ndarray,
    names: tuple[str, ...],
    folds: list[dict],
    foot: np.ndarray,
) -> dict[str, dict]:
    """Evaluate every single H77-BGB channel alone on withheld fault segments across all 4 folds."""
    rng = np.random.default_rng(7701)
    vis = folds[0]["visible"]
    d_vis = ndimage.distance_transform_edt(~vis).astype(np.float32)

    results: dict[str, dict] = {}
    for c_idx, cname in enumerate(names):
        ch = bgb_stack[:, :, c_idx]
        fold_aucs: dict[str, float] = {}
        fold_sym_aucs: dict[str, float] = {}
        pooled_pos = []
        pooled_neg = []
        for f in folds:
            fname = f["name"]
            eval_dom = foot & f["region"] & ~vis & (d_vis > 2.0)
            truth = f["truth"] & eval_dom
            pos_v = ch[truth]
            neg_all = ch[eval_dom & ~truth]
            if neg_all.size > 80_000:
                neg_v = rng.choice(neg_all, size=80_000, replace=False)
            else:
                neg_v = neg_all
            y = np.concatenate([np.ones(pos_v.size, dtype=np.int8), np.zeros(neg_v.size, dtype=np.int8)])
            s = np.concatenate([pos_v, neg_v])
            auc = float(roc_auc_score(y, s))
            sym = float(max(auc, 1.0 - auc))
            fold_aucs[fname] = round(auc, 5)
            fold_sym_aucs[fname] = round(sym, 5)
            pooled_pos.append(pos_v)
            pooled_neg.append(neg_v)

        p_pos = np.concatenate(pooled_pos)
        p_neg = np.concatenate(pooled_neg)
        y_all = np.concatenate([np.ones(p_pos.size, dtype=np.int8), np.zeros(p_neg.size, dtype=np.int8)])
        s_all = np.concatenate([p_pos, p_neg])
        pooled_auc = float(roc_auc_score(y_all, s_all))
        pooled_sym = float(max(pooled_auc, 1.0 - pooled_auc))
        worst_sym = max(pooled_sym, max(fold_sym_aucs.values()))
        flagged = bool(worst_sym > LEAKAGE_AUC_MAX)
        results[cname] = {
            "channel_index": c_idx,
            "pooled_roc_auc": round(pooled_auc, 5),
            "pooled_symmetric_auc": round(pooled_sym, 5),
            "fold_roc_aucs": fold_aucs,
            "fold_symmetric_aucs": fold_sym_aucs,
            "max_symmetric_auc": round(worst_sym, 5),
            "threshold_auc": LEAKAGE_AUC_MAX,
            "flagged_leakage": flagged,
        }
        log(
            f"  Canary [{c_idx:02d}] {cname:<34s}: pooled_AUC={pooled_auc:.4f} "
            f"max_sym_AUC={worst_sym:.4f} flagged={flagged}"
        )
        assert not flagged, f"LEAKAGE CANARY FIRED on {cname}: {worst_sym:.4f} > {LEAKAGE_AUC_MAX}"
    return results


def phase1_surface_uniqueness_scan(
    surf: np.ndarray,
    foot: np.ndarray,
) -> dict:
    """Scan continuous pre-placement surface against all 812 archive registry rasters + local sibling TIFFs."""
    reg_dir = ROOT / ".cache" / "registry"
    dl_dir = ROOT / "docs" / "downloads"
    records = []
    for p in sorted(reg_dir.glob("*.tif")):
        if p.name in ("dense_witness.tif", "h33.tif"):
            continue
        records.append({"source": "archive_registry", "name": p.stem, "path": p})
    for p in sorted(dl_dir.glob("*.tif")):
        if "bgb" in p.name.lower():
            continue
        records.append({"source": "local_sibling", "name": p.stem, "path": p})

    fp_flat = np.flatnonzero(foot)
    rng = np.random.default_rng(7701)
    sub_fp = rng.choice(fp_flat, size=200_000, replace=False)
    sr = rankdata(surf.ravel()[sub_fp]).astype(np.float64)
    sr -= sr.mean()
    snorm = float(np.linalg.norm(sr)) + 1e-12

    worst_rho = 0.0
    worst_name = ""
    top_rhos = []
    unreadable = 0
    h, w = foot.shape

    for rec in records:
        try:
            with rasterio.open(rec["path"]) as s:
                if s.shape != (h, w):
                    unreadable += 1
                    continue
                arr = s.read(1)
        except Exception:
            unreadable += 1
            continue
        v = np.where(np.isfinite(arr.ravel()[sub_fp]), arr.ravel()[sub_fp], 0.0)
        if float(v.max() - v.min()) <= 0.0:
            rho = 0.0
        else:
            vr = rankdata(v).astype(np.float64)
            vr -= vr.mean()
            rho = abs(float((sr @ vr) / (snorm * (float(np.linalg.norm(vr)) + 1e-12))))
        if rho > worst_rho:
            worst_rho = rho
            worst_name = f"{rec['source']}:{rec['name']}"
        top_rhos.append({"source": rec["source"], "name": rec["name"], "abs_spearman": round(rho, 5)})

    top_rhos.sort(key=lambda x: x["abs_spearman"], reverse=True)
    log(
        f"  Phase 1 surface scan across {len(top_rhos)} readable rasters ({unreadable} unreadable): "
        f"max |rho| = {worst_rho:.4f} ({worst_name}) [gate <= 0.90]"
    )
    assert worst_rho <= 0.90, f"STOP-DUPLICATE Phase 1 surface |rho|={worst_rho:.4f} > 0.90"
    return {
        "rasters_scanned": len(records),
        "rasters_readable": len(top_rhos),
        "rasters_unreadable": unreadable,
        "worst_abs_spearman": round(worst_rho, 5),
        "worst_spearman_match": worst_name,
        "spearman_threshold": 0.90,
        "spearman_duplicate": bool(worst_rho > 0.90),
        "top_10_by_abs_spearman": top_rhos[:10],
    }


def allocate_fold_dots(
    surf: np.ndarray,
    folds: list[dict],
    foot: np.ndarray,
    target_n: int = TARGET_DOTS,
    min_sep_px: float = 2.20,
    excl_px: float = 3.0,
) -> np.ndarray:
    """Allocate target_n dots across the 4 test quadrants strictly outside fold['visible'] (d_vis > excl_px)."""
    h, w = foot.shape
    vis = folds[0]["visible"]
    d_vis = ndimage.distance_transform_edt(~vis).astype(np.float32)
    allowed_folds = []
    tot = 0
    for f in folds:
        a = f["region"] & foot & ~vis & (d_vis > excl_px)
        allowed_folds.append(a)
        tot += int(a.sum())
    dots = np.zeros((h, w), dtype=bool)
    for a in allowed_folds:
        b_q = int(round(target_n * int(a.sum()) / max(tot, 1)))
        if b_q > 0:
            dots |= allocate_submodular_300m(
                surf, a, target_n=b_q, support_quantile=0.968, gamma=3.5, min_sep_px=min_sep_px
            )
    return dots.astype(np.float32)


def score_dots_on_folds(dots: np.ndarray, folds: list[dict], foot: np.ndarray) -> tuple[dict, np.ndarray]:
    terms = None
    per = {}
    for fold in folds:
        res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
        per[fold["name"]] = {
            "dti": round(float(res["dti"]), 6),
            "n_truth": int(res["n_truth"]),
            "emitted": int(res["n_emitted"]),
        }
        terms = t if terms is None else terms + t
    return per, terms


def build_in_footprint_matrices(
    foot: np.ndarray,
    bgb_stack: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int], list[str], np.ndarray]:
    """Build compact in-footprint float32 matrix `(N_fp, 43)` combining 33 standard channels from
    `data/processed/stack.npy` (`raw` 19 + `terrain` 8 + `rad` 6) with the 10 `H77-BGB` channels.
    """
    std_sel_names = list(F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["rad"])
    std_idx = [F77.IDX[nm] for nm in std_sel_names]

    mm = np.load(STD_STACK_PATH, mmap_mode="r")
    n_fp = int(foot.sum())
    n_std = len(std_idx)
    n_bgb = bgb_stack.shape[2]
    X_fp = np.empty((n_fp, n_std + n_bgb), dtype=np.float32)

    for j, c_i in enumerate(std_idx):
        ch = np.asarray(mm[:, :, c_i], dtype=np.float32)
        X_fp[:, j] = ch[foot]
    for j in range(n_bgb):
        X_fp[:, n_std + j] = bgb_stack[:, :, j][foot]

    geod_2d = np.zeros(foot.shape, dtype=np.float32)
    geod_2d[foot] = FBGB.rank_in_footprint(np.asarray(mm[:, :, 3], dtype=np.float32), foot)[foot]

    cols_std = list(range(n_std))
    cols_bgb = list(range(n_std, n_std + n_bgb))
    all_names = std_sel_names + list(FBGB.BGB_NAMES)
    return X_fp, cols_std, cols_bgb, all_names, geod_2d


def fit_oof_arm(
    X_fp: np.ndarray,
    cols: list[int],
    folds: list[dict],
    foot: np.ndarray,
    fp_lut: np.ndarray,
    geod_2d: np.ndarray | None = None,
    use_loco_mech: bool = False,
    seed: int = 7701,
) -> np.ndarray:
    """Fit out-of-fold HistGradientBoostingClassifier across the 4 spatial quadrants."""
    h, w = foot.shape
    surf = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(seed)
    vis = folds[0]["visible"]

    for k, f in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        te_reg = f["region"]
        vis_tr = vis & tr_reg
        d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)

        pos = fp_lut[np.flatnonzero(vis_tr & foot)]
        if pos.size > 48_000:
            pos = rng.choice(pos, size=48_000, replace=False)
        neg_pool = foot & tr_reg & ~vis_tr & (d_vis_tr > 3.0)
        hard = fp_lut[np.flatnonzero(neg_pool & (d_vis_tr <= 16.0))]
        easy = fp_lut[np.flatnonzero(neg_pool & (d_vis_tr > 16.0))]
        n_neg = int(2.5 * pos.size)
        nh = min(hard.size, int(0.55 * n_neg))
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([
            rng.choice(hard, size=nh, replace=False),
            rng.choice(easy, size=ne, replace=False),
        ])
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
        te_mask = te_reg & foot
        te_idx = fp_lut[np.flatnonzero(te_mask)]

        if use_loco_mech and geod_2d is not None:
            mech_4 = F77D.compute_visible_mechanics(vis_tr, foot, geod_2d)
            X_tr = np.hstack([X_fp[tr_idx][:, cols], mech_4[tr_idx]])
            X_te = np.hstack([X_fp[te_idx][:, cols], mech_4[te_idx]])
            del mech_4
        else:
            X_tr = X_fp[tr_idx][:, cols]
            X_te = X_fp[te_idx][:, cols]

        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1].astype(np.float32)
        surf[te_mask] = p_te
        del X_tr, X_te, clf
        gc.collect()

    return surf


def main() -> int:
    t0 = time.time()
    EVID.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    grid = G.load_grid(RAW)
    foot = grid.footprint
    cat = grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    log("=== EXPERIMENT 1: Build 10-channel H77-BGB Stack + Leakage Canaries + Surface Audit ===")
    bgb_stack, bgb_names = FBGB.build_bgb_features(RAW, foot, cache_path=BGB_STACK_PATH)
    surfaces = FBGB.compute_bgb_surfaces(bgb_stack, foot)

    folds_s30, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    folds_s20, _ = HO.buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)

    n_withheld_s30 = int(sum(int(f["truth"].sum()) for f in folds_s30))
    n_withheld_s20 = int(sum(int(f["truth"].sum()) for f in folds_s20))
    log(f"Seed 30 withheld positives = {n_withheld_s30}; Seed 20 withheld positives = {n_withheld_s20}")

    exp1_path = EVID / "bgb_exp1_canary_and_surface.json"
    if exp1_path.is_file():
        log("Loaded cached Experiment 1 receipt from evidence/bgb_exp1_canary_and_surface.json")
        rng_r = np.random.default_rng(7700)
        rand_surf = np.zeros((h, w), dtype=np.float32)
        rand_surf[foot] = rng_r.random(int(foot.sum()), dtype=np.float32)
    else:
        log("Running single-feature ROC-AUC leakage canaries on Seed 30...")
        canaries_s30 = run_leakage_canaries(bgb_stack, bgb_names, folds_s30, foot)
        log("Running single-feature ROC-AUC leakage canaries on Seed 20...")
        canaries_s20 = run_leakage_canaries(bgb_stack, bgb_names, folds_s20, foot)

        highest_canary = max(
            max(v["max_symmetric_auc"] for v in canaries_s30.values()),
            max(v["max_symmetric_auc"] for v in canaries_s20.values()),
        )
        log(f"All 10 channels passed leakage canary: highest symmetric AUC = {highest_canary:.4f} <= 0.90")

        log("Running Phase 1 pre-placement surface uniqueness scan on S_bgb_composite...")
        phase1_scan = phase1_surface_uniqueness_scan(surfaces["S_bgb_composite"], foot)

        # Random baseline A0_random
        rng_r = np.random.default_rng(7700)
        rand_surf = np.zeros((h, w), dtype=np.float32)
        rand_surf[foot] = rng_r.random(int(foot.sum()), dtype=np.float32)

        exp1_terms = {}
        exp1_details = {}
        for arm_name, s_arr in [
            ("A0_random", rand_surf),
            ("S_growth_wedge", surfaces["S_growth_wedge"]),
            ("S_strain_part", surfaces["S_strain_part"]),
            ("S_upflow_breach", surfaces["S_upflow_breach"]),
            ("S_bgb_composite", surfaces["S_bgb_composite"]),
        ]:
            dots_arm = allocate_fold_dots(s_arr, folds_s30, foot, target_n=TARGET_DOTS)
            per, t = score_dots_on_folds(dots_arm, folds_s30, foot)
            exp1_terms[arm_name] = t
            exp1_details[arm_name] = {
                "dots": int((dots_arm > 0).sum()),
                "per_fold": per,
                "pooled_holdout_dti": round(float(EH.from_terms(t.sum(axis=0))), 6),
            }

        exp1_pooled = EH.pooled_summary(exp1_terms, draws=1000, seed=520810, candidate="S_bgb_composite")
        for arm_name in exp1_details:
            s_arm = exp1_pooled["scores"][arm_name]
            exp1_details[arm_name]["ci95"] = [round(float(s_arm["ci95"][0]), 6), round(float(s_arm["ci95"][1]), 6)]
            log(
                f"  [Exp1 Seed30] {arm_name:<18s}: HOLDOUT-DTI = {exp1_details[arm_name]['pooled_holdout_dti']:.6f} "
                f"95% CI [{exp1_details[arm_name]['ci95'][0]:.6f}, {exp1_details[arm_name]['ci95'][1]:.6f}] "
                f"(withheld={n_withheld_s30})"
            )

        exp1_receipt = {
            "experiment": "Exp1_BGB_Canaries_PrePlacementSurface_And_StandaloneHoldout",
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "seed": 30,
            "withheld_positive_pixels": n_withheld_s30,
            "leakage_canary": {
                "threshold_auc": LEAKAGE_AUC_MAX,
                "highest_symmetric_auc": round(highest_canary, 5),
                "flagged": False,
                "channels_seed30": canaries_s30,
                "channels_seed20": canaries_s20,
            },
            "phase1_pre_placement_surface_uniqueness": phase1_scan,
            "standalone_surface_holdout_seed30": exp1_details,
            "pooled_summary_seed30": exp1_pooled,
        }
        exp1_path.write_text(json.dumps(exp1_receipt, indent=2) + "\n")

    log("=== EXPERIMENT 2: 4-Quadrant Out-of-Fold Spatial Holdout (Seed 30 & Seed 20) ===")
    X_fp, cols_std, cols_bgb, all_names, geod_2d = build_in_footprint_matrices(foot, bgb_stack)
    cols_all = cols_std + cols_bgb

    exp2_results = {}
    for seed_val, folds, n_withheld in [(30, folds_s30, n_withheld_s30)]:
        skey = f"seed{seed_val}"
        oof_cache = PROC / f"bgb_oof_{skey}.npz"
        if oof_cache.is_file():
            log(f"--- Loaded cached out-of-fold surfaces for {skey} ---")
            z = np.load(oof_cache)
            oof_bgb_only = z["oof_bgb_only"]
            oof_bgb_lane = z["oof_bgb_lane"]
            oof_std = z["oof_std"]
            oof_std_bgb = z["oof_std_bgb"]
            oof_bgb_loco_ens = z["oof_bgb_loco_ens"]
        else:
            log(f"--- Fitting out-of-fold models on {skey} (withheld={n_withheld}) ---")
            oof_bgb_only = fit_oof_arm(X_fp, cols_bgb, folds, foot, fp_lut, seed=7700 + seed_val)
            oof_bgb_lane = (0.55 * FBGB.rank_in_footprint(oof_bgb_only, foot) + 0.45 * surfaces["S_bgb_composite"]).astype(np.float32)
            oof_std = fit_oof_arm(X_fp, cols_std, folds, foot, fp_lut, seed=7700 + seed_val)
            oof_std_bgb = fit_oof_arm(X_fp, cols_all, folds, foot, fp_lut, seed=7700 + seed_val)
            oof_bgb_loco = fit_oof_arm(
                X_fp, cols_all, folds, foot, fp_lut, geod_2d=geod_2d, use_loco_mech=True, seed=7700 + seed_val
            )
            oof_bgb_loco_ens = (0.45 * oof_std_bgb + 0.55 * oof_bgb_loco).astype(np.float32)
            np.savez(
                oof_cache,
                oof_bgb_only=oof_bgb_only,
                oof_bgb_lane=oof_bgb_lane,
                oof_std=oof_std,
                oof_std_bgb=oof_std_bgb,
                oof_bgb_loco_ens=oof_bgb_loco_ens,
            )

        terms_by_arm = {}
        arm_details = {}
        for arm_name, s_arr in [
            ("A0_random", rand_surf),
            ("A_bgb_surface", surfaces["S_bgb_composite"]),
            ("A_bgb_hgb_only", oof_bgb_only),
            ("A_bgb_lane", oof_bgb_lane),
            ("A_std", oof_std),
            ("A_std_plus_bgb", oof_std_bgb),
            ("A_bgb_loco", oof_bgb_loco_ens),
        ]:
            dots_arm = allocate_fold_dots(s_arr, folds, foot, target_n=TARGET_DOTS)
            per, t = score_dots_on_folds(dots_arm, folds, foot)
            terms_by_arm[arm_name] = t
            arm_details[arm_name] = {
                "dots": int((dots_arm > 0).sum()),
                "per_fold": per,
                "pooled_holdout_dti": round(float(EH.from_terms(t.sum(axis=0))), 6),
            }

        pooled_lane = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_bgb_lane")
        pooled_loco = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_bgb_loco")
        pooled_stdbgb = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_std_plus_bgb")

        for arm_name in arm_details:
            s_arm = pooled_lane["scores"][arm_name]
            arm_details[arm_name]["ci95"] = [round(float(s_arm["ci95"][0]), 6), round(float(s_arm["ci95"][1]), 6)]
            arm_details[arm_name]["withheld_positive_pixels"] = n_withheld
            arm_details[arm_name]["evaluator_version"] = EH.VERSION
            arm_details[arm_name]["evidence_class"] = "HOLDOUT-DTI"
            log(
                f"  [Exp2 {skey}] {arm_name:<18s}: HOLDOUT-DTI = {arm_details[arm_name]['pooled_holdout_dti']:.6f} "
                f"95% CI [{arm_details[arm_name]['ci95'][0]:.6f}, {arm_details[arm_name]['ci95'][1]:.6f}] "
                f"(withheld={n_withheld})"
            )

        exp2_results[skey] = {
            "seed": seed_val,
            "withheld_positive_pixels": n_withheld,
            "evaluator_version": EH.VERSION,
            "evidence_class": "HOLDOUT-DTI",
            "arms": arm_details,
            "pooled_summary_candidate_lane": pooled_lane,
            "pooled_summary_candidate_std_plus_bgb": pooled_stdbgb,
            "pooled_summary_candidate_loco": pooled_loco,
        }

    exp2_receipt = {
        "experiment": "Exp2_BGB_Spatially_Blocked_Holdout",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "repository_holdout_best_reference": {
            "lane": "H77-Step",
            "dti": 0.152718,
            "ci95": [0.131548, 0.175185],
            "withheld_positive_pixels": 11321,
            "evaluator_version": "gems57-pooled-hide-v2",
        },
        "results_by_seed": exp2_results,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (EVID / "bgb_exp2_holdout.json").write_text(json.dumps(exp2_receipt, indent=2) + "\n")
    log(f"Saved evidence/bgb_exp1_canary_and_surface.json and evidence/bgb_exp2_holdout.json in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
