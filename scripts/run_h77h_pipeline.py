#!/usr/bin/env python3
"""Master Preregistered 3-Experiment Pipeline for GEMSDOE77 Lane H77H (`2026-10-21`).

Directly tests Hypothesis `H77H-1` (`registry/preregistration_h77h.json`) and resolves
`IR-H77G-01`, `IR-H77G-02`, and `IR-H77G-03` from `docs/review-2026-10-21-h77g.md`:
  - Purges short-wavelength (<35 px) MT conductance and seismicity derivative channels
    (`cond_grad`, `hydro_chem_claycap_coupled`, `seism_updip_proj_35px`, `seism_updip_proj_50px`)
    that extract rectilinear 3.5 km / 7.0 km inversion-mesh boundaries (`IR-H77G-01`).
  - Filters `cond_surf`, geodetic strain rates, and seismicity strictly at sigma >= 35 px (3.5 km)
    as regional crustal-regime modulators.
  - Couples across-valley Hessian RTP demagnetization troughs (`hessian_trough`, `IR-H77G-02`),
    oriented gravity-basement steps, and basement relay saddles directly with TIGER-deconfounded
    1m/10m 3DEP scarp crests, 10m fluvial knickpoints, and fold-pure SLOCO fracture-mechanics
    continuations (`IR-H77G-03`).

Uses ONLY unmodified `vendor/gems57/{evaluate_holdout,holdout,metric,submission_writer,uniqueness,validate}.py`.
"""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout as eval_mod  # noqa: E402
from gems57 import grid as grid_mod  # noqa: E402
from gems57 import holdout as holdout_mod  # noqa: E402
from gems57 import submission_writer as writer_mod  # noqa: E402
from gems57 import uniqueness as uniq_mod  # noqa: E402
from gems57 import validate as validate_mod  # noqa: E402

from gems77e.allocate import (  # noqa: E402
    PortfolioCapacityIndex,
    allocate_dinkelbach_submodular,
    allocate_peak_nms_baseline,
)
from gems77h.audit_index import ensure_h77h_audit_index  # noqa: E402
from gems77h.features import (  # noqa: E402
    EXCLUDED_MESH_CONTAMINATED_FEATURES,
    H77H_DELTA_NAMES,
    H77H_SLOCO_MECH_NAMES,
    build_h77h_static_stack,
    compute_h77h_sloco_mechanics_fp,
)
from gems77h.models import (  # noqa: E402
    fit_predict_h77h_fold,
    run_single_feature_leakage_canaries,
)

CACHE_DIR = ROOT / ".cache"
DATA_DIR = CACHE_DIR / "data"
EXT_DIR = DATA_DIR / "data" / "external"
EVID_DIR = ROOT / "evidence" / "h77h"
ASSETS_DIR = ROOT / "docs" / "assets" / "h77h"
DOWNLOADS_DIR = ROOT / "docs" / "downloads"
SUBMISSIONS_DIR = ROOT / "submissions"

TARGET_DOTS_FULL = 36_800
EXCL_PX = 2.0
MAX_OVERLAP_CAP = 0.665


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval_arms_official(
    arms: dict[str, np.ndarray],
    folds: list[dict[str, Any]],
    fp: np.ndarray,
    candidate_key: str,
    draws: int = 2000,
) -> dict[str, Any]:
    """Evaluate all prediction arms strictly via unmodified `vendor/gems57/evaluate_holdout.py`."""
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
    fp: np.ndarray,
    cat: np.ndarray,
    X16: np.ndarray,
    static_names: list[str],
    has_lidar_fp: np.ndarray,
    crest_deconf_2d: np.ndarray,
    coupled_fields_2d: dict[str, np.ndarray],
    cap_idx: PortfolioCapacityIndex,
    h33_static: np.ndarray,
    run_ablations: bool = False,
) -> dict[str, Any]:
    """Run strictly fold-blind evaluation on official unmodified `vendor/gems57/evaluate_holdout.py`."""
    t_s = time.perf_counter()
    folds, _ = holdout_mod.buffered_component_draw(g, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"] & fp
    visible = folds[0]["visible"] & fp
    assert not np.any(visible & hidden), "Holdout leak check failed!"
    d_vis = distance_transform_edt(~visible).astype(np.float32)
    n_withheld = sum(int((f["truth"] & fp & f["region"] & (~visible)).sum()) for f in folds)
    log(
        f"--- Official gems57-pooled-hide-v2 Seed {seed} "
        f"(withheld={n_withheld:,} px, visible={int(visible.sum()):,} px) ---"
    )

    oof_ckpt = CACHE_DIR / f"h77h_seed{seed}_3spec_oof.npz"
    legacy_ckpt = CACHE_DIR / f"h77h_seed{seed}_oof.npz"
    canary_report = None
    if oof_ckpt.is_file():
        log(f"  Reloading checkpointed Seed {seed} OOF surfaces from {oof_ckpt.name}...")
        ck = np.load(oof_ckpt)
        oof_prob_full = ck["oof_prob_full"]
        oof_prob_static = ck["oof_prob_static"]
        oof_prob_base116 = ck["oof_prob_base116"]
        if (EVID_DIR / f"canary_seed{seed}.json").is_file():
            canary_report = json.loads((EVID_DIR / f"canary_seed{seed}.json").read_text())
    else:
        t_m = time.perf_counter()
        xm_tr, xm_inf = compute_h77h_sloco_mechanics_fp(
            visible, fp, coupled_fields_2d, n_buckets=4, max_seg_px=16
        )
        log(f"  Seed {seed} 4-bucket H77H SLOCO mechanics computed strictly from `visible` in {time.perf_counter()-t_m:.1f}s")

        fp_idx = np.flatnonzero(fp)
        d_cat = distance_transform_edt(~cat).astype(np.float32)
        pos_fp = np.flatnonzero(hidden.ravel()[fp_idx])
        neg_fp = np.flatnonzero((fp & (~cat) & (d_cat > 3.0)).ravel()[fp_idx])
        canary_report = run_single_feature_leakage_canaries(
            X16, static_names, xm_inf, H77H_SLOCO_MECH_NAMES, pos_fp, neg_fp, ceiling=0.90
        )
        log(
            f"  Leakage Canary Gate (Seed {seed}): {canary_report['n_features_checked']}/{canary_report['n_features_checked']} "
            f"features -> passed={canary_report['passed']} | "
            f"max overall={canary_report['max_overall']['symmetric_auc']:.4f} ({canary_report['max_overall']['feature']}) | "
            f"max static physics={canary_report['max_static_physics']['symmetric_auc']:.4f} ({canary_report['max_static_physics']['feature']})"
        )
        if not canary_report["passed"]:
            raise RuntimeError(f"LEAKAGE CANARY FAILED CLOSED (>0.90): {canary_report['violators']}")
        (EVID_DIR / f"canary_seed{seed}.json").write_text(json.dumps(canary_report, indent=2) + "\n")

        crest_deconf_fp = crest_deconf_2d.ravel()[np.flatnonzero(fp)]
        oof_prob_full = np.zeros(fp.shape, dtype=np.float32)
        oof_prob_static = np.zeros(fp.shape, dtype=np.float32)
        if run_ablations and legacy_ckpt.is_file():
            oof_prob_base116 = np.load(legacy_ckpt)["oof_prob_base116"]
            need_base116 = False
        else:
            oof_prob_base116 = np.zeros(fp.shape, dtype=np.float32)
            need_base116 = run_ablations

        for fi, fold in enumerate(folds):
            tf0 = time.perf_counter()
            reg = fold["region"] & fp
            tr_reg = np.zeros_like(fp)
            for fj, f2 in enumerate(folds):
                if fj != fi:
                    tr_reg |= f2["region"] & fp

            res = fit_predict_h77h_fold(
                X16,
                xm_tr,
                xm_inf,
                has_lidar_fp,
                crest_deconf_fp,
                fp,
                visible_train_2d=visible,
                train_region_2d=tr_reg,
                eval_region_2d=reg,
                static_names=static_names,
                static_cols=None,
                use_mechanics=True,
                seed=seed * 100 + fi,
            )
            oof_prob_full[reg] = res["prob_blend_2d"][reg]
            oof_prob_static[reg] = res["prob_static_2d"][reg]

            if need_base116:
                res116 = fit_predict_h77h_fold(
                    X16,
                    xm_tr[:7],
                    xm_inf[:7],
                    has_lidar_fp,
                    crest_deconf_fp,
                    fp,
                    visible_train_2d=visible,
                    train_region_2d=tr_reg,
                    eval_region_2d=reg,
                    static_names=static_names[:116],
                    static_cols=list(range(116)),
                    use_mechanics=True,
                    seed=seed * 100 + 50 + fi,
                )
                oof_prob_base116[reg] = res116["prob_blend_2d"][reg]
                del res116

            del res
            gc.collect()
            log(f"  Seed {seed} Quadrant Fold {fi}/3 OOF propensity finished in {time.perf_counter()-tf0:.1f}s")

        del xm_tr, xm_inf
        gc.collect()
        np.savez(
            oof_ckpt,
            oof_prob_full=oof_prob_full,
            oof_prob_static=oof_prob_static,
            oof_prob_base116=oof_prob_base116,
        )

    allow_oof = fp & (~visible) & (d_vis > EXCL_PX)

    # 1. H77H Pure Submodular (unconstrained 300 m Dinkelbach facility location)
    dots_pure = allocate_dinkelbach_submodular(
        oof_prob_full,
        allow_oof,
        target_n=TARGET_DOTS_FULL,
        crest_2d=crest_deconf_2d,
        capacity_index=None,
    ).astype(np.float32)

    # 2. H77H Strict Portfolio-Unique Submodular (<=0.665 cap across all informative rasters -> 0 firings!)
    dots_unique_bool = allocate_dinkelbach_submodular(
        oof_prob_full,
        allow_oof,
        target_n=TARGET_DOTS_FULL,
        crest_2d=crest_deconf_2d,
        capacity_index=cap_idx,
        max_overlap_cap=MAX_OVERLAP_CAP,
    )
    max_ov_oof, firings_oof, worst_name_oof = cap_idx.max_informative_overlap(dots_unique_bool)
    assert firings_oof == 0 and max_ov_oof <= 0.70, (
        f"Unexpected OOF informative firing: {max_ov_oof} on {worst_name_oof}"
    )
    dots_unique = dots_unique_bool.astype(np.float32)

    # 3. H77H Peak-NMS Control on exact same OOF surface
    dots_nms = allocate_peak_nms_baseline(
        oof_prob_full, allow_oof, target_n=TARGET_DOTS_FULL
    ).astype(np.float32)

    # 4. H77H Pure Zero-Catalogue Static Physics Only (0 SLOCO features)
    dots_static = allocate_dinkelbach_submodular(
        oof_prob_static, allow_oof, target_n=TARGET_DOTS_FULL, crest_2d=crest_deconf_2d
    ).astype(np.float32)

    # 5. True Fold-Fair H33 (`19GEMSDOE` H19-5 continuous centerline discretized to D2.8 dots at `d_vis > 2.0`)
    h33_foldfair = allocate_peak_nms_baseline(
        h33_static, allow_oof, target_n=39_610, min_sep_px=2.80
    ).astype(np.float32)

    rng_r = np.random.default_rng(seed + 900)
    rand_map = np.zeros(fp.shape, dtype=np.float32)
    rand_map.ravel()[rng_r.choice(np.flatnonzero(allow_oof), size=TARGET_DOTS_FULL, replace=False)] = 1.0

    arms: dict[str, np.ndarray] = {
        "h77h_pure_submodular": dots_pure,
        "h77h_portfolio_unique_submodular": dots_unique,
        "h77h_peak_nms_control": dots_nms,
        "h77h_pure_physics_no_mechanics": dots_static,
        "h33_foldfair": h33_foldfair,
        "random_uniform_dots": rand_map,
    }
    if run_ablations:
        arms["ablation_base116_without_h77h_channels"] = allocate_dinkelbach_submodular(
            oof_prob_base116, allow_oof, target_n=TARGET_DOTS_FULL, crest_2d=crest_deconf_2d
        ).astype(np.float32)

    summary_pure = _eval_arms_official(arms, folds, fp, candidate_key="h77h_pure_submodular", draws=2000)
    summary_uniq = _eval_arms_official(
        arms, folds, fp, candidate_key="h77h_portfolio_unique_submodular", draws=2000
    )

    sc = summary_pure["scores"]
    log(
        f"  [Seed {seed} OFFICIAL GEMS57-POOLED-HIDE-V2 RESULTS (n={sc['h77h_pure_submodular']['withheld_positive_pixels']:,} px)]\n"
        f"    * H77H Pure Submodular        : {sc['h77h_pure_submodular']['dti']:.5f}  95% CI {sc['h77h_pure_submodular']['ci95']}\n"
        f"    * H77H Unique (0 info firings): {sc['h77h_portfolio_unique_submodular']['dti']:.5f}  95% CI {sc['h77h_portfolio_unique_submodular']['ci95']} (max_ov={max_ov_oof:.4f})\n"
        f"    * H77H Peak-NMS Control       : {sc['h77h_peak_nms_control']['dti']:.5f}  95% CI {sc['h77h_peak_nms_control']['ci95']}\n"
        f"    * H77H Static Physics (0 cat) : {sc['h77h_pure_physics_no_mechanics']['dti']:.5f}  95% CI {sc['h77h_pure_physics_no_mechanics']['ci95']}\n"
        f"    * H33 Fold-Fair Reference     : {sc['h33_foldfair']['dti']:.5f}  95% CI {sc['h33_foldfair']['ci95']}\n"
        f"    * Random Uniform Control      : {sc['random_uniform_dots']['dti']:.5f}"
    )
    if "ablation_base116_without_h77h_channels" in sc:
        log(
            f"    * Base116 (no H77H coupling)  : {sc['ablation_base116_without_h77h_channels']['dti']:.5f}  "
            f"(H77H delta gain = {sc['h77h_pure_submodular']['dti'] - sc['ablation_base116_without_h77h_channels']['dti']:+.5f})"
        )

    return {
        "seed": seed,
        "evaluator_version": eval_mod.VERSION,
        "implementation_sha256": summary_pure["implementation_sha256"],
        "withheld_positive_pixels": sc["h77h_pure_submodular"]["withheld_positive_pixels"],
        "elapsed_seconds": round(time.perf_counter() - t_s, 2),
        "oof_unique_max_informative_overlap": round(max_ov_oof, 6),
        "oof_unique_informative_firings": firings_oof,
        "oof_unique_worst_informative_raster": worst_name_oof,
        "official_pooled_summary_pure_candidate": summary_pure,
        "official_pooled_summary_unique_candidate": summary_uniq,
        "canary_audit": canary_report,
    }


def run_full_portfolio_uniqueness_audit(
    surface_2d: np.ndarray,
    dots_bool: np.ndarray,
    cand_file_sha256: str,
    foot: np.ndarray,
    cat: np.ndarray,
) -> dict[str, Any]:
    """Exhaustive pre-placement surface & post-placement dot audit across all valid rasters in the index,
    cross-verified bit-for-bit against `gems57.uniqueness.dot_overlap`.
    """
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
    for check_row in info_sorted[:12] + sf_sorted[:3]:
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
                "max_forward_3px_overlap": sf_sorted[0]["forward_3px_overlap"],
                "worst_raster": f"{sf_sorted[0]['tag']}:{sf_sorted[0]['name']}",
                "firings_gt_0p70": len(info_firings) + sum(1 for r in sf_sorted if r["forward_3px_overlap"] > 0.70),
                "protocol_unique": False,
                "reason": "Space-filling heatmaps in archive registry (including dense_witness ab0a0a62...) cover >=70% (up to 100.0%) of allowed Nevada cells within 3 px; Policy I17 requires fail-closed human selector review.",
            },
        },
        "top_15_vendor_verified_matches": vendor_verified_top,
        "top_25_informative_rasters": info_sorted[:25],
    }


def generate_h77h_figures(
    e1: dict[str, Any],
    e2: dict[str, Any],
    surf_2d: np.ndarray,
    dots_bool: np.ndarray,
    uniq: dict[str, Any],
    fp: np.ndarray,
    cat: np.ndarray,
) -> list[str]:
    """Generate 3 publication diagnostic figures for README & GitHub Pages."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    sc20 = e1["official_pooled_summary_pure_candidate"]["scores"]
    sc30 = e2["official_pooled_summary_pure_candidate"]["scores"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=165)
    labels = [
        "Random\nControl",
        "H77G\n(19-Band)",
        "H33 Live\nFold-Fair",
        "H77H Phys.\n(0 Cat.)",
        "H77H\nPeak-NMS",
        "H77D\n(PR #12)",
        "H77H Unique\n(0 Info Fire.)",
        "H77H Pure\nSubmodular",
    ]
    s20_vals = [
        sc20["random_uniform_dots"]["dti"],
        0.06190,
        sc20["h33_foldfair"]["dti"],
        sc20["h77h_pure_physics_no_mechanics"]["dti"],
        sc20["h77h_peak_nms_control"]["dti"],
        0.14030,
        sc20["h77h_portfolio_unique_submodular"]["dti"],
        sc20["h77h_pure_submodular"]["dti"],
    ]
    s30_vals = [
        sc30["random_uniform_dots"]["dti"],
        0.06176,
        sc30["h33_foldfair"]["dti"],
        sc30["h77h_pure_physics_no_mechanics"]["dti"],
        sc30["h77h_peak_nms_control"]["dti"],
        0.15011,
        sc30["h77h_portfolio_unique_submodular"]["dti"],
        sc30["h77h_pure_submodular"]["dti"],
    ]
    x = np.arange(len(labels))
    w_bar = 0.35
    b1 = ax1.bar(x - w_bar / 2, s20_vals, w_bar, label="Seed 20 (11,321 px)", color="#1d4ed8")
    b2 = ax1.bar(x + w_bar / 2, s30_vals, w_bar, label="Seed 30 (11,342 px)", color="#0d9488")
    for bars in (b1, b2):
        for b in bars:
            h_v = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2,
                h_v + 0.0028,
                f"{h_v:.4f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                rotation=62,
                fontweight="bold",
            )
    ax1.axhline(sc20["h33_foldfair"]["dti"], ls="--", lw=1.1, color="#dc2626", alpha=0.7)
    ax1.set_xticks(x, labels, fontsize=8.0)
    ax1.set_ylim(0.0, max(max(s20_vals), max(s30_vals)) * 1.25)
    ax1.set_ylabel("Official HOLDOUT-DTI (gems57-pooled-hide-v2)")
    ax1.set_title("A. Dual-Seed Official Holdout Benchmark Across GEMSDOE77", fontweight="bold", fontsize=10.5)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="y", alpha=0.25)

    comp_names = [
        "H77-Step\n(PR #9: 49 firings)",
        "H77D\n(PR #12: 18 firings)",
        "H77E-Knick\n(0 info firings)",
        "H77H Deliverable\n(This PR: 0 info firings!)",
    ]
    max_info_ov = [
        0.9573,
        0.8949,
        0.6696,
        uniq["phase2_post_placement_final_dots"]["informative_reading_lt_0p70_bg"]["max_forward_3px_overlap"],
    ]
    bars_ov = ax2.bar(np.arange(4), max_info_ov, width=0.52, color=["#dc2626", "#f59e0b", "#0284c7", "#059669"])
    ax2.axhline(0.70, color="#dc2626", ls="--", lw=1.8, label="Protocol 3-px Overlap Gate (0.70)")
    for b in bars_ov:
        hv = b.get_height()
        y_txt = hv - 0.065 if hv < 0.70 else hv + 0.020
        c_txt = "#ffffff" if hv < 0.70 else "#111827"
        ax2.text(
            b.get_x() + b.get_width() / 2,
            y_txt,
            f"{hv:.4f}",
            ha="center",
            fontsize=8.8,
            color=c_txt,
            fontweight="bold",
        )
    ax2.set_xticks(np.arange(4), comp_names, fontsize=8.4)
    ax2.set_ylim(0.0, 1.09)
    ax2.set_ylabel("Max Forward 3-px Overlap Across Informative Rasters (<70% Halo)")
    ax2.set_title("B. Informative Portfolio Uniqueness Audit", fontweight="bold", fontsize=10.5)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p1 = ASSETS_DIR / "fig1_h77h_holdout_and_uniqueness.png"
    fig.savefig(p1)
    plt.close(fig)
    written.append(str(p1.relative_to(ROOT)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.0), dpi=160)
    disp = np.where(fp[::4, ::4], surf_2d[::4, ::4], np.nan)
    im = ax1.imshow(disp, cmap="inferno", vmin=0.0, vmax=float(np.nanpercentile(disp, 99.3)))
    ax1.set_title("A. H77H Cross-Fitted Out-of-Fold Propensity Surface (Nevada)", fontweight="bold", fontsize=10.2)
    ax1.set_xlabel("Column / 4 (EPSG:32611 100 m Grid)")
    ax1.set_ylabel("Row / 4")
    fig.colorbar(im, ax=ax1, fraction=0.042, pad=0.03, label="OOF Fault Propensity")

    r0, r1, c0, c1 = 1050, 1500, 1150, 1600
    sub_bg = surf_2d[r0:r1, c0:c1]
    ax2.imshow(sub_bg, cmap="Greys", vmin=0.0, vmax=float(np.percentile(sub_bg, 99.0)))
    cy, cx = np.nonzero(cat[r0:r1, c0:c1])
    dy, dx = np.nonzero(dots_bool[r0:r1, c0:c1])
    ax2.scatter(cx, cy, s=0.8, c="#ef4444", alpha=0.65, label="Mapped USGS/INGENIOUS Faults (0.0 Masked)")
    ax2.scatter(dx, dy, s=3.4, c="#10b981", alpha=0.92, label=f"H77H Discrete 300 m Dots (N={int(dots_bool.sum()):,})")
    ax2.set_title("B. 45x45 km Central Nevada Basin-and-Range Zoom", fontweight="bold", fontsize=10.2)
    ax2.set_xlabel("Local Column Offset (100 m px)")
    ax2.set_ylabel("Local Row Offset (100 m px)")
    ax2.legend(loc="upper right", fontsize=8.0, markerscale=2.8)
    fig.tight_layout()
    p2 = ASSETS_DIR / "fig2_h77h_nevada_surface_and_zoom.png"
    fig.savefig(p2)
    plt.close(fig)
    written.append(str(p2.relative_to(ROOT)))

    return written


def main() -> int:
    t_start = time.perf_counter()
    EVID_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    log("=== GEMSDOE77 Lane H77H: Starting Preregistered 3-Experiment Pipeline ===")
    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    fp_idx = np.flatnonzero(fp)
    h, w = fp.shape
    cat = g.catalogue & fp
    d_cat = distance_transform_edt(~cat).astype(np.float32)

    X16, static_names, has_lidar_fp, crest_deconf_fp, coupled_fields_2d, feat_prov = build_h77h_static_stack(
        CACHE_DIR, fp, g.transform, DATA_DIR, EXT_DIR
    )
    crest_deconf_2d = np.zeros((h, w), dtype=np.float32)
    crest_deconf_2d.ravel()[fp_idx] = crest_deconf_fp
    log(f"Loaded {len(static_names)}-channel mesh-safe H77H float16 matrix ({X16.nbytes / 1e6:.1f} MB RAM)")

    ensure_h77h_audit_index(CACHE_DIR, fp, cat, exclude_substr="h77h")
    cap_idx = PortfolioCapacityIndex(
        CACHE_DIR / "audit_halos_packed.npy", CACHE_DIR / "audit_meta.json", fp, cat
    )
    log(
        f"Loaded PortfolioCapacityIndex: {cap_idx.n_valid} unique rasters "
        f"({cap_idx.info_row_ids.size} informative <70% halo, {cap_idx.n_valid - cap_idx.info_row_ids.size} space-filling >=70% halo)"
    )

    h19_path = DATA_DIR / "registry" / "rasters" / "19GEMSDOE__h19-5-powerlaw-budget-multiline.tif"
    with rasterio.open(h19_path) as ds:
        a = ds.read(1)
        h33_static = np.where(fp & np.isfinite(a), np.clip(a, 0.0, 1.0), 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # EXPERIMENT 1 (`E1`): Official Seed 20 Holdout (11,321 px) + Canary + Ablations
    # ------------------------------------------------------------------
    e1_path = EVID_DIR / "exp1_seed20_holdout.json"
    if e1_path.is_file():
        log("Reloading completed Experiment 1 receipt from disk...")
        e1 = json.loads(e1_path.read_text())
    else:
        log("=== EXPERIMENT 1 / 3: Seed 20 Official `gems57-pooled-hide-v2` (11,321 px) + Canaries + Ablations ===")
        e1 = evaluate_seed_suite(
            seed=20,
            g=g,
            fp=fp,
            cat=cat,
            X16=X16,
            static_names=static_names,
            has_lidar_fp=has_lidar_fp,
            crest_deconf_2d=crest_deconf_2d,
            coupled_fields_2d=coupled_fields_2d,
            cap_idx=cap_idx,
            h33_static=h33_static,
            run_ablations=True,
        )
        e1_path.write_text(json.dumps(e1, indent=2) + "\n")

    # ------------------------------------------------------------------
    # EXPERIMENT 2 (`E2`): Official Seed 30 Holdout (11,342 px, Zero Retuning)
    # ------------------------------------------------------------------
    e2_path = EVID_DIR / "exp2_seed30_holdout.json"
    if e2_path.is_file():
        log("Reloading completed Experiment 2 receipt from disk...")
        e2 = json.loads(e2_path.read_text())
    else:
        log("=== EXPERIMENT 2 / 3: Seed 30 Independent Replication (11,342 px, Zero Retuning) ===")
        e2 = evaluate_seed_suite(
            seed=30,
            g=g,
            fp=fp,
            cat=cat,
            X16=X16,
            static_names=static_names,
            has_lidar_fp=has_lidar_fp,
            crest_deconf_2d=crest_deconf_2d,
            coupled_fields_2d=coupled_fields_2d,
            cap_idx=cap_idx,
            h33_static=h33_static,
            run_ablations=False,
        )
        e2_path.write_text(json.dumps(e2, indent=2) + "\n")

    # ------------------------------------------------------------------
    # EXPERIMENT 3 (`E3`): Full-Nevada Dual-Grid OOF Surface + Uniqueness Audit + Packaging
    # ------------------------------------------------------------------
    log("=== EXPERIMENT 3 / 3: Full-Nevada Dual-Grid OOF Build + Portfolio Uniqueness Audit + Packaging ===")
    surf_cache = CACHE_DIR / "h77h_fullmap_oof.npy"
    if surf_cache.is_file():
        surf_full = np.load(surf_cache)
    else:
        xm_full_tr, xm_full_inf = compute_h77h_sloco_mechanics_fp(
            cat, fp, coupled_fields_2d, n_buckets=4, max_seg_px=16
        )
        ymid, xmid = h // 2, w // 2
        yy, xx = np.indices((h, w))
        quads_axis = [
            ("NW", (yy < ymid) & (xx < xmid)),
            ("NE", (yy < ymid) & (xx >= xmid)),
            ("SW", (yy >= ymid) & (xx < xmid)),
            ("SE", (yy >= ymid) & (xx >= xmid)),
        ]
        del yy, xx

        oof_axis = np.zeros((h, w), dtype=np.float32)
        crest_fp = crest_deconf_2d.ravel()[fp_idx]

        for qi, (qname, qmask) in enumerate(quads_axis):
            res_q = fit_predict_h77h_fold(
                X16,
                xm_full_tr,
                xm_full_inf,
                has_lidar_fp,
                crest_fp,
                fp,
                visible_train_2d=cat,
                train_region_2d=(~qmask) & fp,
                eval_region_2d=qmask & fp,
                static_names=static_names,
                use_mechanics=True,
                seed=9100 + qi,
            )
            oof_axis[qmask & fp] = res_q["prob_blend_2d"][qmask & fp]
            log(f"  Full-Nevada OOF Axis-Fold {qname} completed")

        surf_full = oof_axis.astype(np.float32)
        np.save(surf_cache, surf_full)
        del xm_full_tr, xm_full_inf, oof_axis
        gc.collect()

    allowed_full = fp & (~cat) & (d_cat > EXCL_PX)
    dots_full = allocate_dinkelbach_submodular(
        surf_full,
        allowed_full,
        target_n=TARGET_DOTS_FULL,
        crest_2d=crest_deconf_2d,
        capacity_index=cap_idx,
        max_overlap_cap=MAX_OVERLAP_CAP,
    )
    n_emitted = int(dots_full.sum())
    assert n_emitted == TARGET_DOTS_FULL, f"Expected {TARGET_DOTS_FULL}, got {n_emitted}"

    pred_zeros = np.where(fp & (~cat), dots_full.astype(np.float32), 0.0).astype(np.float32)
    decoded_sha = hashlib.sha256(pred_zeros.tobytes()).hexdigest()
    short_hash = decoded_sha[:8]

    sc20_p = e1["official_pooled_summary_pure_candidate"]["scores"]
    sc30_p = e2["official_pooled_summary_pure_candidate"]["scores"]

    sub_name = f"gems77-h77h-meshsafe-hydrostruct-submod-{n_emitted}dots-20261021-{short_hash}"
    sub_note = (
        f"H77H: sigma>=35px regime x RTP demag trough + grav step + 3DEP scarp/knick + SLOCO "
        f"({n_emitted} dots; S20={sc20_p['h77h_portfolio_unique_submodular']['dti']:.4f})"
    )
    assert len(sub_name) <= 140 and len(sub_note) <= 140, f"Length overflow: {len(sub_note)}"

    primary_tif = DOWNLOADS_DIR / f"{sub_name}-zeros.tif"
    sample_path = ROOT / "data" / "bridge" / "sample_submission.tif"
    for ext in ("", ".zip", ".json"):
        p_rm = primary_tif if not ext else primary_tif.with_suffix(ext)
        if p_rm.exists():
            p_rm.unlink()

    write_receipt = writer_mod.write_submission(
        primary_tif,
        pred_zeros,
        sample_path,
        fp,
        note=sub_note,
        name=sub_name,
        catalogue=cat,
        metadata={
            "lane_id": "H77H-MESHSAFE-HYDRO-STRUCT-SUBMOD",
            "evaluator": eval_mod.VERSION,
            "holdout_dti_seed20_unique": sc20_p["h77h_portfolio_unique_submodular"]["dti"],
            "holdout_dti_seed30_unique": sc30_p["h77h_portfolio_unique_submodular"]["dti"],
            "holdout_dti_seed20_pure": sc20_p["h77h_pure_submodular"]["dti"],
            "holdout_dti_seed30_pure": sc30_p["h77h_pure_submodular"]["dti"],
        },
    )
    val_report = validate_mod.validate(primary_tif, footprint=fp, catalogue=cat)
    assert val_report["all_checks_passed"] and all(val_report["checks"].values()), (
        f"Validation failed: {val_report}"
    )

    shutil.copy2(primary_tif, SUBMISSIONS_DIR / f"{sub_name}-zeros.tif")
    shutil.copy2(primary_tif.with_suffix(".zip"), SUBMISSIONS_DIR / f"{sub_name}-zeros.zip")
    shutil.copy2(primary_tif.with_suffix(".json"), SUBMISSIONS_DIR / f"{sub_name}-zeros.json")

    file_sha = sha256_file(primary_tif)
    zip_sha = sha256_file(primary_tif.with_suffix(".zip"))

    uniq_audit = run_full_portfolio_uniqueness_audit(surf_full, dots_full, file_sha, fp, cat)
    (EVID_DIR / "uniqueness_audit.json").write_text(json.dumps(uniq_audit, indent=2) + "\n")

    figs = generate_h77h_figures(e1, e2, surf_full, dots_full, uniq_audit, fp, cat)

    run_card = {
        "schema_version": "1.0",
        "lane_id": "H77H",
        "lane_name": "H77H-MESHSAFE-HYDRO-STRUCT-SUBMOD",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": "arena/46c3997d-gemsdoe77",
        "preregistration_path": "registry/preregistration_h77h.json",
        "preregistration_sha256": sha256_file(ROOT / "registry" / "preregistration_h77h.json"),
        "hypothesis": (
            "Filtering MT conductance (cond_surf), geodetic strain rates, and seismicity strictly at "
            "sigma >= 35 px (3.5 km) eliminates rectilinear 35 px / 70 px inversion-mesh boundary "
            "artifacts (IR-H77G-01), while coupling across-valley Hessian RTP demagnetization troughs "
            "(IR-H77G-02), oriented gravity-basement steps, and basement relay saddles directly with "
            "TIGER-deconfounded 1m/10m 3DEP scarp crests, 10m fluvial knickpoints, and fold-pure SLOCO "
            "fracture-mechanics continuations (IR-H77G-03) inside a 300 m Dinkelbach Tversky submodular "
            "allocator recovers blind and poorly exposed geothermal fault segments."
        ),
        "physical_mechanism": (
            "Hydrothermal fluids circulating along steeply dipping (55-65 deg) normal/transtensional "
            "fault damage zones alter ferromagnetic magnetite to paramagnetic clay/pyrite (across-valley "
            "RTP Hessian trough lambda_1 > 0) along a co-aligned subsurface basement-depth and gravity "
            "step, accompanied at the surface by subtle TIGER-deconfounded 1m/10m scarp crests or 10m "
            "fluvial knickpoints along the projected strike of visible fault tips and relay breaches "
            "within a regional (sigma >= 35 px) transtensional, microseismically active clay-cap province."
        ),
        "named_non_fault_mimic_and_controls": (
            "(1) Rectilinear 3.5 km / 7.0 km MT & geodetic inversion-mesh boundaries (IR-H77G-01: "
            "eliminated by filtering cond_surf, geodetic strain, and seismicity strictly at sigma >= 35 px "
            "and removing short-wavelength cond_grad and hydro_chem_claycap_coupled); "
            "(2) reverse-polarized volcanic flows and equant plutonic stocks (suppressed via across-valley "
            "Hessian tubularity, oriented gravity-basement step co-alignment, and Gallardo-Meju "
            "cross-gradient parallelism); (3) highway/railroad embankments and Pleistocene Lake "
            "Lahontan/Bonneville paleoshorelines (suppressed via US Census TIGER road-distance Gaussian "
            "damping and 1.6 km along-strike elevation variance)."
        ),
        "feature_provenance": feat_prov,
        "excluded_mesh_contaminated_features": sorted(EXCLUDED_MESH_CONTAMINATED_FEATURES),
        "novel_h77h_features": H77H_DELTA_NAMES,
        "sloco_mechanics_features": H77H_SLOCO_MECH_NAMES,
        "leakage_canary_audit": {
            "ceiling_symmetric_auc": 0.90,
            "seed20": e1["canary_audit"],
            "seed30": json.loads((EVID_DIR / "canary_seed30.json").read_text()) if (EVID_DIR / "canary_seed30.json").is_file() else None,
        },
        "holdout_evaluation": {
            "evidence_label": "HOLDOUT-DTI",
            "evaluator_module": "vendor/gems57/evaluate_holdout.py",
            "evaluator_version": eval_mod.VERSION,
            "implementation_sha256": e1["implementation_sha256"],
            "metric_parameters": {"alpha": 0.2, "beta": 0.8, "radius_m": 300.0, "radius_px": 3.0},
            "seed20": {
                "withheld_positive_pixels": e1["withheld_positive_pixels"],
                "scores": sc20_p,
                "paired_delta_pure_vs_baselines": e1["official_pooled_summary_pure_candidate"]["paired_differences"],
                "paired_delta_unique_vs_baselines": e1["official_pooled_summary_unique_candidate"]["paired_differences"],
            },
            "seed30": {
                "withheld_positive_pixels": e2["withheld_positive_pixels"],
                "scores": sc30_p,
                "paired_delta_pure_vs_baselines": e2["official_pooled_summary_pure_candidate"]["paired_differences"],
                "paired_delta_unique_vs_baselines": e2["official_pooled_summary_unique_candidate"]["paired_differences"],
            },
        },
        "uniqueness_audit": uniq_audit,
        "candidate_artifact": {
            "submission_name": sub_name,
            "submission_note": sub_note,
            "submission_note_length": len(sub_note),
            "tif_path": str(primary_tif.relative_to(ROOT)),
            "tif_sha256": file_sha,
            "tif_bytes": primary_tif.stat().st_size,
            "zip_path": str(primary_tif.with_suffix(".zip").relative_to(ROOT)),
            "zip_sha256": zip_sha,
            "zip_bytes": primary_tif.with_suffix(".zip").stat().st_size,
            "decoded_float32_sha256": decoded_sha,
            "emitted_positive_dots": n_emitted,
            "min_distance_to_catalogue_px": round(float(d_cat[dots_full].min()), 4),
            "writer_receipt": write_receipt,
            "validator_report": val_report,
        },
        "diagnostic_figures": figs,
        "experiments_run": 3,
        "elapsed_seconds_total": round(time.perf_counter() - t_start, 2),
        "weekly_slots_spent": 0,
        "ok_to_download": True,
        "ok_to_submit": False,
        "selector_status": "Not approved",
        "verdict": "negative",
        "verdict_reason": (
            "H77H achieves strong dual-seed HOLDOUT-DTI on gems57-pooled-hide-v2 and passes all "
            "validator checks ([0, 1], 0 NaN, EPSG:32611, 3730x3292) and informative uniqueness "
            f"(0 informative firings > 0.70 across all {uniq_audit['informative_rasters_checked_lt_0p70_bg']} informative rasters; max informative "
            f"3-px overlap = {uniq_audit['phase2_post_placement_final_dots']['informative_reading_lt_0p70_bg']['max_forward_3px_overlap']:.4f} <= 0.70; "
            f"pre-placement surface |rho| = {uniq_audit['phase1_pre_placement_surface']['max_abs_spearman']:.4f} <= 0.90), "
            "but under binding Policy I17 the literal unfiltered registry includes space-filling "
            "witnesses (dense_witness.tif ab0a0a62... covering 100.0% of allowed footprint within 3 px), "
            "and no human selector has authorized a weekly DrivenData submission slot."
        ),
    }
    (EVID_DIR / "run_card.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "evidence" / "run_card_h77h.json").write_text(json.dumps(run_card, indent=2) + "\n")
    log(
        f"=== H77H Pipeline Complete in {run_card['elapsed_seconds_total']}s: "
        f"{primary_tif.name} (sha256={file_sha[:12]}...) ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
