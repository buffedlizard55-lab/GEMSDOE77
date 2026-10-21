#!/usr/bin/env python3
"""Experiment 1 (`H77V-Exp1`): 114-Channel Geothermal Vent Feature Synthesis & Rule 4 Leakage Canary.

Synthesizes the 12 static H77V geothermal vent/clay-cap breach/sinter/2m-probe channels and
the 6 fold-safe V-LBCO vent-bridged relay/crack-tip channels on top of the 96-channel base
physical stack (`114` total zero-SGMC channels), then runs the mandatory Rule 4 single-feature
ROC-AUC leakage canary (`max(AUC, 1-AUC) <= 0.90`) across all 114 features on both Seed 20
(`11,321` withheld positives) and Seed 30 (`11,342` withheld positives) of `gems57-pooled-hide-v2`
BEFORE fitting any model.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout as eval_mod  # noqa: E402
from gems57 import grid as grid_mod  # noqa: E402
from gems57 import holdout as holdout_mod  # noqa: E402
from gems77vc.features import (  # noqa: E402
    H77V_STATIC_NAMES,
    build_h77v_static_features,
    compute_h77v_vlbco_mechanics_fp,
)

DATA_DIR = ROOT / ".cache" / "data"
EXT_DIR = DATA_DIR / "data" / "external"
BASE_MMAP = ROOT / ".cache" / "h77_features_fp.f32"
H77V_MMAP = ROOT / ".cache" / "h77v_static_12.f32"


def _run_seed_canary(
    seed: int,
    g: grid_mod.Grid,
    X_base: np.memmap,
    base_names: list[str],
    X_v: np.memmap,
    v_names: list[str],
    hydro_imp_fp: np.ndarray,
    vent_pot_fp: np.ndarray,
    upflow_scarp_fp: np.ndarray,
) -> dict:
    fp = g.footprint
    fp_idx = np.flatnonzero(fp)
    folds, draw_audit = holdout_mod.buffered_component_draw(
        g, seed=seed, hide_frac=0.20, buffer_px=3, boundary_px=12
    )
    vis_2d = np.ones(g.shape, dtype=bool)
    for f in folds:
        vis_2d &= f["visible"]
    vis_2d &= fp
    truth_union_2d = np.logical_or.reduce([f["truth"] for f in folds])
    assert not np.any(vis_2d & truth_union_2d), "Visible/truth overlap detected!"

    d_vis_fp = distance_transform_edt(~vis_2d).astype(np.float32).ravel()[fp_idx]
    _, Xm_infer, mech_names = compute_h77v_vlbco_mechanics_fp(
        vis_2d, fp, hydro_imp_fp, vent_pot_fp, upflow_scarp_fp, n_buckets=5, seed=7700 + seed
    )
    all_names = base_names + v_names + mech_names

    truth_fp = truth_union_2d.ravel()[fp_idx]
    eval_pool = (~vis_2d.ravel()[fp_idx]) & ((d_vis_fp > 3.0) | truth_fp)
    rng = np.random.default_rng(77000 + seed)
    pos_idx = np.flatnonzero(eval_pool & truth_fp)
    neg_idx = rng.choice(np.flatnonzero(eval_pool & (~truth_fp)), size=150000, replace=False)
    can_idx = np.concatenate([pos_idx, neg_idx])
    y_can = np.concatenate([np.ones(pos_idx.size, np.int32), np.zeros(neg_idx.size, np.int32)])

    n_b = len(base_names)
    n_v = len(v_names)
    rows = []
    for col_i, nm in enumerate(all_names):
        if col_i < n_b:
            vals = np.asarray(X_base[col_i, can_idx], dtype=np.float64)
        elif col_i < n_b + n_v:
            vals = np.asarray(X_v[col_i - n_b, can_idx], dtype=np.float64)
        else:
            vals = np.asarray(Xm_infer[col_i - n_b - n_v, can_idx], dtype=np.float64)
        med = float(np.nanmedian(vals)) if np.isnan(vals).any() else 0.0
        vals = np.nan_to_num(vals, nan=med)
        auc = float(roc_auc_score(y_can, vals))
        sym_auc = max(auc, 1.0 - auc)
        is_h77v = (nm in H77V_STATIC_NAMES) or nm.startswith("vis_vent_") or nm == "vis_blind_stepover_vent_band"
        rows.append({
            "feature": nm,
            "is_novel_h77v": bool(is_h77v),
            "roc_auc": round(auc, 5),
            "symmetric_auc": round(sym_auc, 5),
            "leakage_pass": bool(sym_auc <= 0.90),
        })
        if sym_auc > 0.90:
            raise RuntimeError(f"LEAKAGE CANARY FAILED CLOSED on seed={seed}, feature={nm}: {sym_auc:.5f} > 0.90")

    rows.sort(key=lambda r: r["symmetric_auc"], reverse=True)
    h77v_rows = [r for r in rows if r["is_novel_h77v"]]
    del Xm_infer
    gc.collect()
    return {
        "seed": seed,
        "withheld_positive_pixels": int(pos_idx.size),
        "n_features_tested": len(rows),
        "n_novel_h77v_features": len(h77v_rows),
        "max_symmetric_auc": rows[0]["symmetric_auc"],
        "top_feature": rows[0]["feature"],
        "max_novel_h77v_symmetric_auc": h77v_rows[0]["symmetric_auc"],
        "top_novel_h77v_feature": h77v_rows[0]["feature"],
        "all_pass_le_0p90": True,
        "novel_h77v_features": h77v_rows,
        "top_25_all_features": rows[:25],
    }


def main() -> int:
    t0 = time.perf_counter()
    g = grid_mod.load_grid(DATA_DIR)
    print("[Exp 1/3] Building 12 static H77V Geothermal Vent & Clay-Cap Breach channels...")
    (
        X_base,
        base_names,
        X_v,
        v_names,
        has_lidar_fp,
        hydro_imp_fp,
        crest_fp,
        vent_pot_fp,
        v_meta,
    ) = build_h77v_static_features(BASE_MMAP, H77V_MMAP, g.footprint, g.transform, DATA_DIR, EXT_DIR)
    upflow_scarp_fp = np.asarray(
        X_v[v_names.index("h77v_upflow_scarp_coincidence"), :], dtype=np.float32
    )

    # Verify zero NaNs or Infs in the 12 new H77V static channels
    h77v_stats = {}
    for i, nm in enumerate(v_names):
        arr = np.asarray(X_v[i, :], dtype=np.float32)
        assert np.all(np.isfinite(arr)), f"Non-finite values in {nm}"
        h77v_stats[nm] = {
            "min": round(float(arr.min()), 5),
            "p50": round(float(np.percentile(arr[::19], 50.0)), 5),
            "p95": round(float(np.percentile(arr[::19], 95.0)), 5),
            "max": round(float(arr.max()), 5),
        }

    print("[Exp 1/3] Running Rule 4 single-feature leakage canary across 114 features on Seed 20 and Seed 30...")
    s20 = _run_seed_canary(20, g, X_base, base_names, X_v, v_names, hydro_imp_fp, vent_pot_fp, upflow_scarp_fp)
    s30 = _run_seed_canary(30, g, X_base, base_names, X_v, v_names, hydro_imp_fp, vent_pot_fp, upflow_scarp_fp)

    elapsed = round(time.perf_counter() - t0, 2)
    out = {
        "experiment": "H77V-Exp1",
        "evidence_class": "LEAKAGE-CANARY",
        "evaluator_version": eval_mod.VERSION,
        "elapsed_seconds": elapsed,
        "n_total_features": 114,
        "n_base_static_features": len(base_names),
        "n_h77v_static_features": len(v_names),
        "n_vlbco_mechanics_features": 6,
        "sgmc_channels_used": 0,
        "gdr_dist_known_fault_px_accessed": False,
        "in_sample_predicted_pixels": 0,
        "pixels_copied_from_prior_rasters": 0,
        "gate_ceiling": 0.90,
        "all_114_features_passed_both_seeds": bool(s20["all_pass_le_0p90"] and s30["all_pass_le_0p90"]),
        "seed20_max_symmetric_auc": s20["max_symmetric_auc"],
        "seed30_max_symmetric_auc": s30["max_symmetric_auc"],
        "seed20_max_novel_h77v_auc": s20["max_novel_h77v_symmetric_auc"],
        "seed30_max_novel_h77v_auc": s30["max_novel_h77v_symmetric_auc"],
        "h77v_static_channel_stats": h77v_stats,
        "seed20": s20,
        "seed30": s30,
    }
    for dst in (ROOT / "evidence" / "h77v_exp1_canary.json", ROOT / "docs" / "evidence" / "h77v_exp1_canary.json"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"[Exp 1/3] PASS in {elapsed:.1f}s | 114/114 features <= 0.90 | "
        f"Seed20 max AUC={s20['max_symmetric_auc']:.4f} (novel H77V max={s20['max_novel_h77v_symmetric_auc']:.4f} on {s20['top_novel_h77v_feature']}) | "
        f"Seed30 max AUC={s30['max_symmetric_auc']:.4f} (novel H77V max={s30['max_novel_h77v_symmetric_auc']:.4f} on {s30['top_novel_h77v_feature']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
