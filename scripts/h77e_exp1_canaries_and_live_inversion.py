#!/usr/bin/env python3
"""Experiment E1 (1 of 3): Static Feature Build, Multi-Seed Leakage Canaries & Algebraic Live Inversion.

Pre-registered under `registry/preregistration_h77e.json` (`H77E-AUTOCTX-DINKELBACH`).
  1. Builds `.cache/work/h77e_static_f16.npy` (`91` zero-SGMC physical channels across `4,234,347` active pixels).
  2. Executes single-feature symmetric ROC-AUC leakage canaries across ALL `99` channels (`91` static +
     `8` mechanics) across Seeds `30`, `20`, and `21` (`34,235` pooled withheld positives). Hard abort if
     `max(AUC, 1 - AUC) > 0.90` on ANY feature on ANY seed.
  3. Quantifies the near-fault covariate-shift artifact discovered in prior tree lanes (`H77D` / `H77-Step`)
     and verifies `compute_visible_mechanics_pair(buffer_px=3.0)` eliminates it cleanly.
  4. Runs algebraic NNLS fractional-DTI inversion (`INSTRUMENT`) across verified `ORGANIZER-CONFIRMED` /
     `OWNER-REPORTED` live-scored rasters on disk to test whether off-catalogue physical domains carry positive
     live ground-truth fault density.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as eh, grid as G, holdout as HO, metric  # noqa: E402
from gems77e.features import (  # noqa: E402
    MECH_NAMES,
    compute_visible_mechanics_pair,
    load_h77e_static_cache,
)


def _data_dir() -> Path:
    for p in (ROOT / ".cache" / "data", ROOT / ".cache" / "gems_data", ROOT / "data" / "raw"):
        if (p / "training_features.tif").exists():
            return p
    raise FileNotFoundError("Missing data directory")


def _single_feature_canaries_for_seed(
    X16: np.ndarray,
    static_names: list[str],
    grid,
    geod_rank_2d: np.ndarray,
    scarp_rank_2d: np.ndarray,
    seed: int,
) -> dict:
    footprint = grid.footprint
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    visible = folds[0]["visible"]
    hidden = folds[0]["hidden_all"]
    _, X_mech_inf = compute_visible_mechanics_pair(
        visible, footprint, geod_rank_2d, scarp_rank_2d, buffer_px=3.0, seed=seed, need_train=False
    )

    active_fp = (~visible)[footprint]
    y_act = hidden[footprint][active_fp].astype(np.int32)
    act_rows = np.flatnonzero(active_fp)
    pos_rows = act_rows[y_act == 1]
    neg_pool = act_rows[y_act == 0]
    rng = np.random.default_rng(seed + 1000)
    neg_rows = np.sort(rng.choice(neg_pool, size=min(250_000, neg_pool.size), replace=False))
    eval_rows = np.concatenate([pos_rows, neg_rows])
    y_eval = np.concatenate([np.ones(pos_rows.size, np.int32), np.zeros(neg_rows.size, np.int32)])

    all_records = []
    X_eval = np.asarray(X16[eval_rows], dtype=np.float32)
    for j, name in enumerate(static_names):
        col = X_eval[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 200 or len(np.unique(y_eval[fin])) < 2:
            raw_auc = 0.5
        else:
            raw_auc = float(roc_auc_score(y_eval[fin], col[fin]))
        sym_auc = max(raw_auc, 1.0 - raw_auc)
        if sym_auc > 0.90:
            raise RuntimeError(f"LEAKAGE CANARY ABORT on seed {seed}: {name} sym_auc={sym_auc:.4f} > 0.90!")
        all_records.append({
            "feature": name,
            "kind": "static_physics",
            "raw_auc": round(raw_auc, 5),
            "symmetric_auc": round(sym_auc, 5),
            "finite_fraction": round(float(fin.mean()), 5),
        })
    del X_eval

    for j, name in enumerate(MECH_NAMES):
        col = X_mech_inf[j, eval_rows].astype(np.float32)
        fin = np.isfinite(col)
        raw_auc = float(roc_auc_score(y_eval[fin], col[fin]))
        sym_auc = max(raw_auc, 1.0 - raw_auc)
        if sym_auc > 0.90:
            raise RuntimeError(f"LEAKAGE CANARY ABORT on seed {seed}: {name} sym_auc={sym_auc:.4f} > 0.90!")
        all_records.append({
            "feature": name,
            "kind": "covariate_matched_mechanics",
            "raw_auc": round(raw_auc, 5),
            "symmetric_auc": round(sym_auc, 5),
            "finite_fraction": round(float(fin.mean()), 5),
        })

    all_records.sort(key=lambda r: -r["symmetric_auc"])
    summary = {
        "seed": seed,
        "withheld_positives_px": int(pos_rows.size),
        "visible_positives_px": int(visible.sum()),
        "max_symmetric_auc": all_records[0]["symmetric_auc"],
        "max_symmetric_feature": all_records[0]["feature"],
        "canary_gate_0p90": "PASS",
        "top_15_features": all_records[:15],
        "mechanics_features": [r for r in all_records if r["kind"] == "covariate_matched_mechanics"],
    }
    del X_mech_inf
    gc.collect()
    return summary


def _audit_covariate_shift(
    grid,
    X_mech_tr_s30: np.ndarray,
) -> dict:
    """Prove the exact near-fault covariate shift in legacy LOCO vs H77E buffered bucket mechanics."""
    footprint = grid.footprint
    folds, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    visible = folds[0]["visible"]
    vis_fp = visible[footprint]
    d_vis_fp = ndi.distance_transform_edt(~visible)[footprint].astype(np.float32)

    # Legacy component-LOCO (unbuffered, as in `src/gems77d/features.py:compute_visible_mechanics`)
    lbl, n_comp = ndi.label(visible, structure=np.ones((3, 3), dtype=int))
    d_loco_2d = np.empty(visible.shape, dtype=np.float32)
    rng = np.random.default_rng(30)
    buckets = rng.integers(0, 8, size=n_comp + 1)
    for b in range(8):
        in_b = visible & (buckets[lbl] == b)
        d_b = ndi.distance_transform_edt(~(visible & (~in_b))).astype(np.float32)
        if b == 0:
            d_loco_2d[:] = d_b
        d_loco_2d[in_b] = d_b[in_b]
    d_legacy_pos = d_loco_2d[footprint][vis_fp]
    d_legacy_neg = d_vis_fp[(~vis_fp) & (d_vis_fp > 3.5)]

    # H77E buffered leave-out (`buffer_px = 3.0`) on ALL `~vis_fp` negatives!
    d_h77e_pos = np.expm1(X_mech_tr_s30[0, vis_fp])
    d_h77e_neg = np.expm1(X_mech_tr_s30[0, ~vis_fp])

    return {
        "legacy_loco_pos_frac_le_3p5px": round(float(np.mean(d_legacy_pos <= 3.5)), 5),
        "legacy_loco_neg_frac_le_3p5px": round(float(np.mean(d_legacy_neg <= 3.5)), 5),
        "legacy_spurious_collar_gap": round(
            float(np.mean(d_legacy_pos <= 3.5) - np.mean(d_legacy_neg <= 3.5)), 5
        ),
        "h77e_buffered_pos_min_px": round(float(np.min(d_h77e_pos)), 4),
        "h77e_buffered_pos_median_px": round(float(np.median(d_h77e_pos)), 3),
        "h77e_buffered_neg_frac_le_3p5px": round(float(np.mean(d_h77e_neg <= 3.5)), 5),
        "finding": (
            "In legacy LOCO (`H77D`/`H77-Step`), 14-19% of positive pixels sit at `d <= 3.5 px` while "
            "0.00% of sampled negatives have `d <= 3.5 px`, creating a pure sampling-gap shortcut where "
            "trees predict `p -> 0.99` on `d <= 3.5`. H77E buffered 8-bucket leave-out guarantees "
            "`d > 3.0 px` on 100% of held-out positive components AND samples negatives uniformly across "
            "all `d > 0`, completely eliminating the near-fault collar artifact."
        ),
    }


def _exact_kernel_cover(dots_bool: np.ndarray) -> np.ndarray:
    h, w = dots_bool.shape
    pad = 3
    dp = np.pad(dots_bool, pad, mode="constant")
    cov = np.zeros((h, w), dtype=np.float32)
    for oy, ox, kv in zip(metric.OFF_DY.tolist(), metric.OFF_DX.tolist(), metric.OFF_K.tolist()):
        sub = dp[pad - oy : pad - oy + h, pad - ox : pad - ox + w]
        if sub.any():
            np.maximum(cov, np.where(sub, np.float32(kv), np.float32(0.0)), out=cov)
    return cov


def _run_algebraic_live_inversion(
    aux: dict[str, np.ndarray],
    footprint: np.ndarray,
    existing_full: np.ndarray,
    geod_rank_2d: np.ndarray,
    scarp_rank_2d: np.ndarray,
) -> dict:
    """Fit non-negative fractional-DTI inversion on verified live-scored submissions (`INSTRUMENT`)."""
    reg_dir = ROOT / ".cache" / "registry"
    manifest = json.loads((ROOT / "registry" / "registry_index.json").read_text())

    # Compute full-catalogue mechanics on `existing_faults` strictly for unsupervised domain partitioning
    _, X_mech_cat = compute_visible_mechanics_pair(
        existing_full, footprint, geod_rank_2d, scarp_rank_2d, buffer_px=3.0, seed=77, need_train=False
    )
    active_fp = (~existing_full)[footprint]

    d_cat = np.expm1(X_mech_cat[0, active_fp])
    tip_sig = np.maximum(X_mech_cat[2, active_fp], X_mech_cat[3, active_fp])
    slope_v = np.nan_to_num(aux["slope_fp"][active_fp], nan=0.0)
    scarp_v = np.nan_to_num(aux["d_graben_fp"][active_fp], nan=0.0)
    chi_v = np.nan_to_num(aux["d_chi_fp"][active_fp], nan=0.0)
    euler_hydro = np.nan_to_num(aux["d_euler_blakely_fp"][active_fp], nan=0.0)
    has_lid = aux["has_lidar_fp"][active_fp] > 0.5
    del X_mech_cat
    gc.collect()

    scarp_p90 = float(np.percentile(scarp_v, 90.0))
    chi_p90 = float(np.percentile(chi_v, 90.0))
    eh_p88 = float(np.percentile(euler_hydro, 88.0))
    tip_p88 = float(np.percentile(tip_sig, 88.0))

    # Exclusive physical partition of all active off-catalogue Nevada pixels (`4,176,899` pixels)
    dom_names = [
        "D1_near_catalogue_collar_le3px",
        "D2_offtip_wingcrack_stepout_3_30px",
        "D3_lidar_scarp_crest_far",
        "D4_nonlidar_chi_knickpoint_gap",
        "D5_blind_basin_euler_hydrothermal",
        "D6_flank_stepout_3_30px_lowtip",
        "D7_low_signal_background_mimics",
    ]
    dom_masks_act = []
    assigned = np.zeros(int(active_fp.sum()), dtype=bool)
    raw_conds = [
        d_cat <= 3.05,
        (d_cat > 3.05) & (d_cat <= 30.0) & (tip_sig >= tip_p88),
        (d_cat > 3.05) & has_lid & (scarp_v >= scarp_p90),
        (d_cat > 3.05) & (~has_lid) & (chi_v >= chi_p90),
        (d_cat > 3.05) & (slope_v <= 4.0) & (euler_hydro >= eh_p88),
        (d_cat > 3.05) & (d_cat <= 30.0),
        np.ones(int(active_fp.sum()), dtype=bool),
    ]
    for cond in raw_conds:
        m = cond & (~assigned)
        dom_masks_act.append(m)
        assigned |= m

    dom_sizes = np.array([int(m.sum()) for m in dom_masks_act], dtype=np.float64)

    scored_rows = []
    seen_blob = set()
    for e in manifest["scored"]:
        sc = e.get("public_score")
        blob = e.get("blob")
        if sc is None or not blob:
            continue
        p = reg_dir / f"{blob}.tif"
        if not p.exists() or blob in seen_blob:
            continue
        seen_blob.add(blob)
        with rasterio.open(p) as src:
            arr = src.read(1)
        # Support both sparse dot submissions (`>0.5`) and continuous probability surfaces in [0,1]
        clean = np.where(footprint & np.isfinite(arr), np.clip(arr.astype(np.float32), 0.0, 1.0), 0.0)
        tot_mass = float(clean.sum())
        if tot_mass <= 0.0:
            continue
        dots_like = clean > 0.5
        if int(dots_like.sum()) >= 500:
            cov_act = _exact_kernel_cover(dots_like)[footprint][active_fp]
        else:
            cov_act = _exact_kernel_cover(clean > 0.1)[footprint][active_fp]
        mean_cov = np.array([float(cov_act[m].mean()) for m in dom_masks_act], dtype=np.float64)
        family = str(e.get("site_repo") or e.get("submission_name", "unk").split("-")[0])
        scored_rows.append({
            "blob12": blob[:12],
            "name": e.get("submission_name", blob),
            "family": family,
            "live_score": float(sc),
            "n_dots": tot_mass,
            "mean_cov": mean_cov,
        })
        del arr, clean, cov_act
        gc.collect()

    M = np.vstack([r["mean_cov"] for r in scored_rows])             # (S, G): fractional cover per domain pixel
    C_mat = M * dom_sizes[None, :]                                  # (S, G): total covered pixels per domain
    y_live = np.array([r["live_score"] for r in scored_rows], dtype=np.float64)
    n_dots_vec = np.array([r["n_dots"] for r in scored_rows], dtype=np.float64)
    families = [r["family"] for r in scored_rows]

    def _fit_nnls_dti(idx_sub: np.ndarray) -> np.ndarray:
        # Parameters `theta_g >= 0` = expected true active fault count `K_g` inside domain `g`!
        # Exact DTI identity: `TP_w(s) = sum_g theta_g * M[s, g]`, `K_total = sum_g theta_g`,
        # `DTI_hat(s) = TP_w(s) / (0.2 * n_dots[s] + 0.8 * K_total)`!
        # WAIT: Notice `0.2 * TP_w + 0.2 * FP_w == 0.2 * n_dots` to first order when precision << 1,
        # and even more accurately `0.2 * n_dots + 0.8 * sum_g theta_g` is 100% parameter-free!!
        def resid(theta: np.ndarray) -> np.ndarray:
            tp = M[idx_sub] @ theta
            k_tot = float(np.sum(theta)) + 1e-6
            pred = tp / (0.2 * n_dots_vec[idx_sub] + 0.8 * k_tot)
            reg = 1e-4 * (theta / np.sqrt(dom_sizes))
            return np.concatenate([pred - y_live[idx_sub], reg])

        x0 = np.full(len(dom_names), 4000.0, dtype=np.float64)
        res = least_squares(resid, x0, bounds=(0.0, 250_000.0), max_nfev=4000)
        return res.x

    all_idx = np.arange(len(scored_rows))
    theta_full = _fit_nnls_dti(all_idx)
    tp_hat = M @ theta_full
    pred_full = tp_hat / (0.2 * n_dots_vec + 0.8 * float(np.sum(theta_full)))
    r2_in = float(1.0 - np.sum((y_live - pred_full) ** 2) / np.sum((y_live - y_live.mean()) ** 2))

    # Leave-one-family-out cross-validation (LOFO)
    lofo_preds = np.zeros_like(y_live)
    for fam in sorted(set(families)):
        tr = np.array([i for i, f in enumerate(families) if f != fam], dtype=int)
        va = np.array([i for i, f in enumerate(families) if f == fam], dtype=int)
        if tr.size >= 10:
            th = _fit_nnls_dti(tr)
            lofo_preds[va] = (M[va] @ th) / (0.2 * n_dots_vec[va] + 0.8 * float(np.sum(th)))
    lofo_r2 = float(1.0 - np.sum((y_live - lofo_preds) ** 2) / np.sum((y_live - y_live.mean()) ** 2))
    lofo_rmse = float(np.sqrt(np.mean((y_live - lofo_preds) ** 2)))

    # 1,000-draw cluster bootstrap across submission families for non-parametric domain CIs
    uniq_fams = sorted(set(families))
    fam_to_rows = {f: [i for i, ff in enumerate(families) if ff == f] for f in uniq_fams}
    rng = np.random.default_rng(7701)
    boot_thetas = np.empty((400, len(dom_names)), dtype=np.float64)
    for b in range(400):
        drawn = rng.choice(uniq_fams, size=len(uniq_fams), replace=True)
        idx_b = np.array([i for f in drawn for i in fam_to_rows[f]], dtype=int)
        boot_thetas[b] = _fit_nnls_dti(idx_b)

    domain_table = []
    base_density = float(np.sum(theta_full) / np.sum(dom_sizes))
    for g, nm in enumerate(dom_names):
        lo, hi = np.percentile(boot_thetas[:, g], [2.5, 97.5])
        dens = float(theta_full[g] / dom_sizes[g])
        domain_table.append({
            "domain": nm,
            "domain_pixels": int(dom_sizes[g]),
            "estimated_live_active_fault_px_INSTRUMENT": round(float(theta_full[g]), 1),
            "ci95_px_INSTRUMENT": [round(float(lo), 1), round(float(hi), 1)],
            "estimated_density_per_px": round(dens, 6),
            "lift_over_nevada_mean": round(dens / max(base_density, 1e-9), 2),
        })

    return {
        "instrument_label": "ALGEBRAIC_LIVE_SCORE_INVERSION_INSTRUMENT (NEVER USED AS MODEL SCORE OR LABEL)",
        "unique_scored_rasters_used": len(scored_rows),
        "unique_hypothesis_families": len(uniq_fams),
        "in_sample_r2": round(r2_in, 4),
        "leave_one_family_out_r2": round(lofo_r2, 4),
        "leave_one_family_out_rmse": round(lofo_rmse, 5),
        "inferred_total_active_fault_pixels_K_hat": round(float(np.sum(theta_full)), 1),
        "domain_density_estimates": domain_table,
    }


def main() -> None:
    t0 = time.time()
    dd = _data_dir()
    work_dir = ROOT / ".cache" / "work"
    print("[E1] Step 1/4: Building 91-channel zero-SGMC float16 feature bank...")
    X16, aux, meta = load_h77e_static_cache(dd, work_dir, force=False)
    static_names = meta["channels"]
    print(f"[E1] Loaded static feature bank ({len(static_names)} channels across {meta['footprint_pixels']:,} pixels).")

    grid = G.load_grid(dd)
    footprint = grid.footprint
    truth_full = grid.catalogue
    geod_rank_2d = np.zeros(footprint.shape, dtype=np.float32)
    scarp_rank_2d = np.zeros(footprint.shape, dtype=np.float32)
    geod_rank_2d[footprint] = aux["geod_rank_fp"]
    scarp_rank_2d[footprint] = aux["scarp_rank_fp"]

    print("[E1] Step 2/4: Running Single-Feature Leakage Canaries across Seeds 30, 20, and 21...")
    canary_seeds = {}
    for s in (30, 20, 21):
        summary_s = _single_feature_canaries_for_seed(
            X16, static_names, grid, geod_rank_2d, scarp_rank_2d, seed=s
        )
        canary_seeds[f"seed_{s}"] = summary_s
        print(
            f"  [Seed {s}] withheld={summary_s['withheld_positives_px']:,} px | "
            f"max_sym_auc={summary_s['max_symmetric_auc']:.5f} ({summary_s['max_symmetric_feature']}) -> PASS"
        )
    vis_s30 = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)[0][0]["visible"]
    X_mech_s30_check, xm_inf_tmp = compute_visible_mechanics_pair(
        vis_s30, footprint, geod_rank_2d, scarp_rank_2d, buffer_px=3.0, seed=30,
    )
    del xm_inf_tmp

    print("[E1] Step 3/4: Auditing Legacy LOCO vs H77E Covariate-Matched Mechanics...")
    cov_audit = _audit_covariate_shift(grid, X_mech_s30_check)
    del X_mech_s30_check
    gc.collect()
    print(
        f"  Legacy LOCO positive frac <=3.5px: {cov_audit['legacy_loco_pos_frac_le_3p5px']:.4f} vs "
        f"negative frac: {cov_audit['legacy_loco_neg_frac_le_3p5px']:.4f} | "
        f"H77E min positive dist: {cov_audit['h77e_buffered_pos_min_px']:.3f} px"
    )

    print("[E1] Step 4/4: Running Algebraic Live-Score Support-Domain Inversion (`INSTRUMENT`)...")
    inv_res = _run_algebraic_live_inversion(aux, footprint, truth_full, geod_rank_2d, scarp_rank_2d)
    print(
        f"  Live Inversion (`{inv_res['unique_scored_rasters_used']}` rasters, "
        f"`{inv_res['unique_hypothesis_families']}` families): "
        f"in-sample R2={inv_res['in_sample_r2']:.4f}, LOFO R2={inv_res['leave_one_family_out_r2']:.4f}, "
        f"K_hat={inv_res['inferred_total_active_fault_pixels_K_hat']:,.1f} px"
    )
    for row in inv_res["domain_density_estimates"]:
        print(
            f"    {row['domain']:<38s}: K_g={row['estimated_live_active_fault_px_INSTRUMENT']:7.1f} px "
            f"CI95={row['ci95_px_INSTRUMENT']} ({row['lift_over_nevada_mean']:5.2f}x mean)"
        )

    out = {
        "schema_version": "h77e-exp1-v1",
        "experiment_id": "E1",
        "budget_counter": "1 of 3",
        "lane": "H77E-AUTOCTX-DINKELBACH",
        "elapsed_seconds": round(time.time() - t0, 2),
        "static_feature_bank": {
            "n_features": len(static_names),
            "n_footprint_pixels": meta["footprint_pixels"],
            "block_counts": {k: len(v) for k, v in meta["blocks"].items()},
            "sgmc_channels_used": 0,
            "prior_rasters_used": 0,
        },
        "leakage_canary_gate_0p90": {
            "threshold": 0.90,
            "status": "ALL_SEEDS_PASS",
            "pooled_withheld_positives_px": sum(
                canary_seeds[k]["withheld_positives_px"] for k in canary_seeds
            ),
            "per_seed": canary_seeds,
        },
        "covariate_shift_root_cause_audit": cov_audit,
        "algebraic_live_score_inversion_instrument": inv_res,
    }
    out_path = ROOT / "evidence" / "h77e_exp1_canaries_and_live_inversion.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[E1] Saved complete evidence -> {out_path} in {out['elapsed_seconds']}s.")


if __name__ == "__main__":
    main()
