#!/usr/bin/env python3
"""GEMSDOE77 Lane H77E — Experiment 1 (of max 3):
  1. Single-feature symmetric ROC-AUC leakage canaries across ALL 107 channels
     (96 static + 11 fold-visible dynamic channels; hard stop if > 0.90).
  2. Full 8-arm causal ablation on Seed 30 (`11,342` withheld fault pixels) using
     unmodified `vendor/gems57/evaluate_holdout.py` (`gems57-pooled-hide-v2`,
     `alpha=0.2, beta=0.8`, 300 m kernel, 2,000 paired 20 km spatial-cluster draws).

Strict zero-leakage invariant:
  - Hidden fault pixels (`fold['hidden']`) never touch features, training masks,
    negative sampling, calibration, or witness halos.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems77 import features77 as F0  # noqa: E402
from gems77pd import emit as EM  # noqa: E402
from gems77pd import features as FE  # noqa: E402
from gems77pd import models as MD  # noqa: E402

SEED = 30
FULL_BUDGET = 37_654


def log(*args) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def emit_over_folds(folds: list[dict], foot: np.ndarray, fn) -> np.ndarray:
    """Apply emitter per fold with area-proportional dot allocation totalling `FULL_BUDGET`."""
    H, W = foot.shape
    out = np.zeros((H, W), dtype=bool)
    areas = [int((f["region"] & foot & (~f["visible"])).sum()) for f in folds]
    tot = max(sum(areas), 1)
    used = 0
    for k, f in enumerate(folds):
        allow = f["region"] & foot & (~f["visible"])
        b = (FULL_BUDGET - used) if k == len(folds) - 1 else int(round(FULL_BUDGET * areas[k] / tot))
        used += b
        if b > 0:
            out |= fn(f, allow, b)
    return out


def build_fold_fair_witness_halos(
    X_fp: np.ndarray,
    phys_prob: np.ndarray,
    folds: list[dict],
    visible: np.ndarray,
    foot: np.ndarray,
) -> np.ndarray:
    """Construct strictly fold-fair 3-px witness halos using ONLY `visible` (ZERO leakage!).

    Simulates the two dominant competitive families in the 873-raster archive:
      1. Champion H33 / H19-5 multi-scale ridge + potential-field rule (excl 3 px of visible)
      2. Champion H33 / H19-5 rule without collar exclusion (excl 0 px of visible)
      3. Scarp-propensity rank^4 submodular rule (H77-Step / H77D style on visible)
    """
    H, W = foot.shape
    fp_idx = np.flatnonzero(foot)

    def col_grid(name: str) -> np.ndarray:
        g = np.full((H, W), np.nan, dtype=np.float32)
        g.ravel()[fp_idx] = X_fp.read_channel(FE.STATIC_IDX[name])
        return g

    r2 = col_grid("dem_ridge2")
    r35 = col_grid("dem_ridge3p5")
    mag = col_grid("mag_as_log")
    grv = col_grid("grav_thdr_log")
    champ_score = (
        np.float32(0.40) * F0.rank01(np.nan_to_num(r2, nan=-1e30), foot)
        + np.float32(0.25) * F0.rank01(np.nan_to_num(r35, nan=-1e30), foot)
        + np.float32(0.20) * F0.rank01(np.nan_to_num(mag, nan=-1e30), foot)
        + np.float32(0.15) * F0.rank01(np.nan_to_num(grv, nan=-1e30), foot)
    ).astype(np.float32)
    dist_vis = ndi.distance_transform_edt(~visible).astype(np.float32)

    w_dots = []
    # Witness 1: H33-style 300m exclusion
    w1 = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_greedy(champ_score, al & (dist_vis > 3.0), b, min_sep=3.0)
    )
    w_dots.append(w1)
    # Witness 2: H19-5-style 0m exclusion
    w2 = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_greedy(champ_score, al, b, min_sep=3.0)
    )
    w_dots.append(w2)
    # Witness 3: H77D / H77-Step style rank^4 scarp propensity
    w3 = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_submodular_rank(phys_prob, al, b, gamma=4.0, min_sep=2.8)
    )
    w_dots.append(w3)

    halos = np.stack([ndi.distance_transform_edt(~d) <= 3.0 for d in w_dots], axis=0)
    return halos, {"H33_excl3_vis": w1, "H19_excl0_vis": w2, "Step_rank4_vis": w3}


def run_leakage_canaries(
    X_fp: np.ndarray,
    V_vis: np.ndarray,
    hidden: np.ndarray,
    visible: np.ndarray,
    foot: np.ndarray,
) -> dict:
    """Single-feature symmetric ROC-AUC leakage canaries across ALL 107 channels on `foot & ~visible`."""
    fp_idx = np.flatnonzero(foot)
    eval_fp = np.flatnonzero((foot & (~visible)).ravel()[fp_idx])
    rng = np.random.default_rng(20261021)
    sub = np.sort(rng.choice(eval_fp, size=min(400_000, eval_fp.size), replace=False))
    y_sub = hidden.ravel()[fp_idx][sub].astype(np.int8)

    canaries = {}
    max_sym = 0.0
    max_name = ""

    for j, nm in enumerate(FE.STATIC_NAMES):
        v = X_fp.read_channel(j)[sub]
        fin = np.isfinite(v)
        if int(fin.sum()) < 500 or y_sub[fin].sum() < 10:
            sym = 0.5
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_sub[fin], v[fin]))
            sym = float(max(auc, 1.0 - auc))
        canaries[nm] = {"auc": round(auc, 5), "sym_auc": round(sym, 5), "valid_frac": round(float(fin.mean()), 4)}
        if sym > max_sym:
            max_sym, max_name = sym, nm

    for j, nm in enumerate(FE.B4_VIS_FRACTURE_NAMES):
        v = np.asarray(V_vis[sub, j], dtype=np.float32)
        fin = np.isfinite(v)
        auc = float(roc_auc_score(y_sub[fin], v[fin]))
        sym = float(max(auc, 1.0 - auc))
        canaries[nm] = {"auc": round(auc, 5), "sym_auc": round(sym, 5), "valid_frac": round(float(fin.mean()), 4)}
        if sym > max_sym:
            max_sym, max_name = sym, nm

    if max_sym > 0.90:
        raise RuntimeError(f"LEAKAGE CANARY TRIPPED: {max_name} sym_auc={max_sym:.5f} > 0.90! STOPPING.")

    return {
        "status": "PASS",
        "threshold": 0.90,
        "tested_channels": len(canaries),
        "n_eval_sample": int(sub.size),
        "n_hidden_in_sample": int(y_sub.sum()),
        "max_symmetric_auc": round(max_sym, 5),
        "max_channel": max_name,
        "top_12_channels_by_sym_auc": sorted(
            [{"channel": k, **val} for k, val in canaries.items()],
            key=lambda d: -d["sym_auc"],
        )[:12],
        "all_channels": canaries,
    }


def main() -> int:
    t0 = time.time()
    data_dir = ROOT / ".cache" / "data"
    grid = G.load_grid(data_dir)
    foot = grid.footprint

    X_fp, dirs, static_meta = FE.build_or_load_static_stack(ROOT, force=False)

    folds, _ = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    log(f"Seed {SEED}: withheld_pos={int(hidden.sum())}, visible_pos={int(visible.sum())}")

    # 1. Single-feature leakage canaries across ALL 107 channels BEFORE trusting any model!
    V_vis_canary = FE.compute_visible_fracture_features(visible & foot, foot, dirs)
    canary_report = run_leakage_canaries(X_fp, V_vis_canary, hidden, visible, foot)
    del V_vis_canary
    gc.collect()
    log(
        f"Leakage canary PASS across {canary_report['tested_channels']} channels: "
        f"max_sym_auc={canary_report['max_symmetric_auc']} ({canary_report['max_channel']})"
    )

    # 2. Nested Physical Hypothesis Ablation Arms (Seed 30)
    idx_b0 = [FE.STATIC_IDX[n] for n in FE.B0_CORE_NAMES]
    idx_b01 = idx_b0 + [FE.STATIC_IDX[n] for n in FE.B1_FLUVIAL_CHI_NAMES]
    idx_b012 = idx_b01 + [FE.STATIC_IDX[n] for n in FE.B2_DESEAM_SUBSURF_NAMES]
    idx_all_static = list(range(len(FE.STATIC_NAMES)))
    idx_partner = [FE.STATIC_IDX[n] for n in FE.STRUCT_PARTNER_NAMES]

    phys_cache = ROOT / ".cache" / "h77e_work" / "seed30_phys_cache.npz"
    if phys_cache.is_file():
        c = np.load(phys_cache)
        p_b0, p_b01, p_b012, p_phys = c["p_b0"], c["p_b01"], c["p_b012"], c["p_phys"]
        log("Loaded cached Seed-30 static physical OOF surfaces.")
    else:
        p_b0 = MD.train_physical_oof(X_fp, idx_b0, folds, visible, foot, use_lgbm=True, tag="Arm0_B0_core")
        p_b01 = MD.train_physical_oof(X_fp, idx_b01, folds, visible, foot, use_lgbm=True, tag="Arm1_plus_H77E1_fluvial")
        p_b012 = MD.train_physical_oof(X_fp, idx_b012, folds, visible, foot, use_lgbm=True, tag="Arm2_plus_H77E2_deseam")
        p_phys = MD.train_physical_oof(X_fp, idx_all_static, folds, visible, foot, use_lgbm=True, tag="Arm3_plus_H77E3_step_gap")
        np.savez(phys_cache, p_b0=p_b0, p_b01=p_b01, p_b012=p_b012, p_phys=p_phys)

    p_ens, p_struct_only = MD.train_structural_oof(
        X_fp, idx_partner, p_phys, folds, visible, foot, dirs,
        n_inner_splits=3, use_lgbm=True, blend_phys_weight=0.42, seed=400, tag="Arm4_H77E4_full",
    )
    p_logodds = MD.logodds_blend(p_phys, p_struct_only, w_a=0.42)

    # Build strictly fold-fair witness halos (zero access to `hidden` or full-catalogue files!)
    witness_halos, witness_dots = build_fold_fair_witness_halos(X_fp, p_phys, folds, visible, foot)

    # Emit dots across all arms
    dots_map: dict[str, np.ndarray] = {}
    dots_map["Reference_H33_vis_excl3"] = witness_dots["H33_excl3_vis"]
    dots_map["Arm0_B0_core_greedy"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_greedy(p_b0, al, b)
    )
    dots_map["Arm1_plus_H77E1_fluvial_greedy"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_greedy(p_b01, al, b)
    )
    dots_map["Arm2_plus_H77E2_deseam_greedy"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_greedy(p_b012, al, b)
    )
    dots_map["Arm3_plus_H77E3_step_gap_greedy"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_greedy(p_phys, al, b)
    )
    dots_map["Arm3b_phys_only_bayes_submod"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_submodular_bayes(p_phys, al, b)
    )
    dots_map["Arm4_plus_H77E4_ensemble_greedy"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_greedy(p_ens, al, b)
    )
    dots_map["Arm5_H77E_rank4_submod"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_submodular_rank(p_ens, al, b, gamma=4.0)
    )
    dots_map["Arm6_H77E_bayes_submod_unconstrained"] = emit_over_folds(
        folds, foot, lambda f, al, b: EM.emit_dots_submodular_bayes(p_ens, al, b)
    )

    allow_all = foot & (~visible)
    dots_lag, lag_meta = EM.emit_dots_lagrangian_submodular(
        p_ens, allow_all, witness_halos, budget=FULL_BUDGET, cap_frac=0.665
    )
    dots_map["Arm7_H77E_lagrangian_knapsack_submod"] = dots_lag
    dots_logodds, lag_logodds_meta = EM.emit_dots_lagrangian_submodular(
        p_logodds, allow_all, witness_halos, budget=FULL_BUDGET, cap_frac=0.665
    )
    dots_map["Arm8_H77E_logodds_lagrangian_submod"] = dots_logodds
    log(f"Lagrangian dual emitter (linear): {lag_meta}")
    log(f"Lagrangian dual emitter (logodds): {lag_logodds_meta}")

    # Evaluate every arm via unmodified vendor/gems57/evaluate_holdout.py!
    terms_by_arm = {}
    fold_details = {}
    for arm_name, dmask in dots_map.items():
        pred_f32 = dmask.astype(np.float32)
        t_sum = None
        f_list = []
        for f in folds:
            res, t_arr = EH.evaluate(pred_f32, f, foot)
            t_sum = t_arr.copy() if t_sum is None else (t_sum + t_arr)
            f_list.append({
                "fold": f["name"],
                "dti": round(float(res["dti"]), 6),
                "tpw": round(float(res["tpw"]), 3),
                "fpw": round(float(res["fpw"]), 3),
                "fnw": round(float(res["fnw"]), 3),
                "n_truth": int(res["n_truth"]),
                "n_emitted": int(res["n_emitted"]),
            })
        terms_by_arm[arm_name] = t_sum
        fold_details[arm_name] = f_list

    # Save exact spatial-cluster terms so Experiment 2 can pool Seed 30 + Seed 20 + Seed 21 directly!
    np.savez(ROOT / ".cache" / "h77e_work" / "seed30_terms.npz", **terms_by_arm)

    summary = EH.pooled_summary(
        terms_by_arm, draws=2000, seed=520810, candidate="Arm7_H77E_lagrangian_knapsack_submod"
    )
    stepwise_pairs = []
    ordered_chain = [
        ("Arm0_B0_core_greedy", "Arm1_plus_H77E1_fluvial_greedy", "H77E-1: +10m Fluvial Knickpoint & Footwall Chi Uplift"),
        ("Arm1_plus_H77E1_fluvial_greedy", "Arm2_plus_H77E2_deseam_greedy", "H77E-2: +GeoDAWN Seam, BLM/TIGER & MT/Mag/GDR"),
        ("Arm2_plus_H77E2_deseam_greedy", "Arm3_plus_H77E3_step_gap_greedy", "H77E-3: +1m LiDAR Polarity + 13-ch 10m Step Purity"),
        ("Arm3_plus_H77E3_step_gap_greedy", "Arm4_plus_H77E4_ensemble_greedy", "H77E-4a: +Hard-Collar Multi-Split Fracture Continuation"),
        ("Arm4_plus_H77E4_ensemble_greedy", "Arm6_H77E_bayes_submod_unconstrained", "H77E-4b: +Exact Bayes Submodular 300 m Cover"),
        ("Arm5_H77E_rank4_submod", "Arm6_H77E_bayes_submod_unconstrained", "Bayes Calibration vs Uniform Rank^4 Submodular"),
        ("Arm6_H77E_bayes_submod_unconstrained", "Arm7_H77E_lagrangian_knapsack_submod", "Lagrangian Knapsack Uniqueness Cap Cost"),
    ]
    for a_ctrl, a_cand, desc in ordered_chain:
        ps = EH.pooled_summary(
            {a_ctrl: terms_by_arm[a_ctrl], a_cand: terms_by_arm[a_cand]},
            draws=2000,
            seed=520810,
            candidate=a_cand,
        )
        diff = ps["paired_differences"][a_ctrl]
        stepwise_pairs.append({
            "step": desc,
            "from": a_ctrl,
            "to": a_cand,
            "delta_dti": round(float(diff["delta"]), 6),
            "ci95": [round(float(x), 6) for x in diff["ci95"]],
        })

    for arm_name, info in summary["scores"].items():
        log(
            f"HOLDOUT-DTI [{arm_name}]: {info['dti']:.6f} "
            f"(95% CI [{info['ci95'][0]:.5f}, {info['ci95'][1]:.5f}]), "
            f"withheld={info['withheld_positive_pixels']}, dots={int(dots_map[arm_name].sum())}"
        )

    out_payload = {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "lane": "H77E",
        "experiment": "Exp1_Seed30_LeakageCanary_and_CausalAblation",
        "seed": SEED,
        "withheld_positive_pixels": int(hidden.sum()),
        "visible_positive_pixels": int(visible.sum()),
        "leakage_canary": canary_report,
        "lagrangian_emitter_diagnostics": lag_meta,
        "stepwise_hypothesis_deltas": stepwise_pairs,
        "per_fold_results": fold_details,
        "pooled_summary": summary,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out_path = ROOT / "evidence" / "h77pd_exp1_ablation_holdout.json"
    out_path.write_text(json.dumps(out_payload, indent=2) + "\n")
    log(f"Saved {out_path} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
