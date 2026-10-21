#!/usr/bin/env python3
"""GEMSDOE77 Lane H77E — Experiment 2 (of max 3):
Zero-tuning multi-seed & blind province transfer replication of frozen `Arm7`:
  1. `Seed 20` (`11,321` untouched withheld positives) — identical split to
     `H77-Step` (`0.152718`) and `H77D` (`0.140298`).
  2. `Seed 21` (`11,572` untouched withheld positives) — identical split to
     `PR #11 H77-HSL` (`0.1480` unique emitted).
  3. **3-Seed Pooled Spatial-Cluster Benchmark** (`Seed 30 + Seed 20 + Seed 21` =
     `34,235` withheld fault pixels merged by physical 20 km cluster before bootstrap!).
  4. **Blind 4-Quadrant Province Transfer** (`57,448` withheld fault pixels — 100% of
     Nevada fault catalogue held out one province at a time with zero visible faults
     inside the test province, compared directly against `H77-Step = 0.185834`!).

Zero hyperparameters, channels, weights, or emitter rules are altered after Exp 1.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems77pd import emit as EM  # noqa: E402
from gems77pd import features as FE  # noqa: E402
from gems77pd import models as MD  # noqa: E402

from h77e_exp1_ablation_holdout import (  # noqa: E402
    FULL_BUDGET,
    build_fold_fair_witness_halos,
    emit_over_folds,
)


def log(*args) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def evaluate_seed(seed: int, grid, X_fp: FE.StaticFeatureStore, dirs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    cache_p = ROOT / ".cache" / "h77e_work" / f"seed{seed}_bundle.npz"
    meta_p = ROOT / ".cache" / "h77e_work" / f"seed{seed}_rec.json"
    if cache_p.is_file() and meta_p.is_file():
        log(f"Loaded cached Seed {seed} replication bundle.")
        return json.loads(meta_p.read_text()), dict(np.load(cache_p))

    t_s = time.time()
    foot = grid.footprint
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    log(f"=== Replication Seed {seed}: withheld_pos={int(hidden.sum())}, visible_pos={int(visible.sum())} ===")

    idx_b0 = [FE.STATIC_IDX[n] for n in FE.B0_CORE_NAMES]
    idx_all_static = list(range(len(FE.STATIC_NAMES)))
    idx_partner = [FE.STATIC_IDX[n] for n in FE.STRUCT_PARTNER_NAMES]

    p_b0 = MD.train_physical_oof(X_fp, idx_b0, folds, visible, foot, use_lgbm=True, tag=f"S{seed}_B0_core")
    p_phys = MD.train_physical_oof(X_fp, idx_all_static, folds, visible, foot, use_lgbm=True, tag=f"S{seed}_Phys96")
    p_ens, _ = MD.train_structural_oof(
        X_fp, idx_partner, p_phys, folds, visible, foot, dirs,
        n_inner_splits=3, use_lgbm=True, blend_phys_weight=0.42, seed=400, tag=f"S{seed}_H77E_full",
    )

    witness_halos, witness_dots = build_fold_fair_witness_halos(X_fp, p_phys, folds, visible, foot)
    allow_all = foot & (~visible)
    dots_h77e, lag_meta = EM.emit_dots_lagrangian_submodular(
        p_ens, allow_all, witness_halos, budget=FULL_BUDGET, cap_frac=0.665
    )

    dots_map = {
        "Reference_H33_vis_excl3": witness_dots["H33_excl3_vis"],
        "Arm0_B0_core_greedy": emit_over_folds(folds, foot, lambda f, al, b: EM.emit_dots_greedy(p_b0, al, b)),
        "Arm3b_phys_only_bayes_submod": emit_over_folds(
            folds, foot, lambda f, al, b: EM.emit_dots_submodular_bayes(p_phys, al, b)
        ),
        "Arm6_H77E_bayes_submod_unconstrained": emit_over_folds(
            folds, foot, lambda f, al, b: EM.emit_dots_submodular_bayes(p_ens, al, b)
        ),
        "Arm7_H77E_lagrangian_knapsack_submod": dots_h77e,
    }

    terms_by_arm = {}
    for arm_name, dmask in dots_map.items():
        pred_f32 = dmask.astype(np.float32)
        t_sum = None
        for f in folds:
            _, t_arr = EH.evaluate(pred_f32, f, foot)
            t_sum = t_arr.copy() if t_sum is None else (t_sum + t_arr)
        terms_by_arm[arm_name] = t_sum

    summary = EH.pooled_summary(
        terms_by_arm, draws=2000, seed=520810, candidate="Arm7_H77E_lagrangian_knapsack_submod"
    )
    for arm_name, info in summary["scores"].items():
        log(f"Seed {seed} [{arm_name}]: {info['dti']:.6f} (95% CI [{info['ci95'][0]:.5f}, {info['ci95'][1]:.5f}])")

    del p_b0, p_phys, p_ens, witness_halos
    gc.collect()
    rec_out = {
        "seed": seed,
        "withheld_positive_pixels": int(hidden.sum()),
        "lagrangian_diagnostics": lag_meta,
        "pooled_summary": summary,
        "runtime_seconds": round(time.time() - t_s, 2),
    }
    np.savez(cache_p, **terms_by_arm)
    meta_p.write_text(json.dumps(rec_out, indent=2) + "\n")
    return rec_out, terms_by_arm


def make_quadrant_folds(cat: np.ndarray, fp: np.ndarray) -> list[dict]:
    """Pixel-identical 4-quadrant blind province folds (`57,448` px, 3 km guard band) matching H77-Step."""
    from scipy.ndimage import distance_transform_edt

    H, W = fp.shape
    rmid, cmid = H // 2, W // 2
    yy, xx = np.indices((H, W))
    quad_2d = ((yy >= rmid).astype(np.int8) * 2 + (xx >= cmid).astype(np.int8))
    guard_2d = (np.abs(yy - rmid) < 30) | (np.abs(xx - cmid) < 30)

    q_folds = []
    for q_id, q_name in enumerate(("NW", "NE", "SW", "SE")):
        in_q = (quad_2d == q_id) & fp & (~guard_2d)
        vis = cat & fp & (~(quad_2d == q_id)) & (~guard_2d)
        d_vis = distance_transform_edt(~vis)
        truth = cat & in_q & (d_vis > 3.0)
        q_folds.append({"name": q_name, "fold": q_name, "region": in_q, "visible": vis, "truth": truth})
    return q_folds


def evaluate_blind_quadrants(grid, X_fp: FE.StaticFeatureStore) -> dict:
    """Blind 4-Quadrant Province Transfer (57,448 withheld positives, 0 in-region visible faults!)."""
    t_q = time.time()
    foot = grid.footprint
    q_folds = make_quadrant_folds(grid.catalogue, foot)
    idx_b0 = [FE.STATIC_IDX[n] for n in FE.B0_CORE_NAMES]
    idx_all = list(range(len(FE.STATIC_NAMES)))

    # On blind province folds, each fold `k` has its own `visible` (the other 3 quadrants' faults)!
    H, W = foot.shape
    fp_idx = np.flatnonzero(foot)
    N_fp = int(fp_idx.size)
    full_to_fp = np.full(H * W, -1, dtype=np.int32)
    full_to_fp[fp_idx] = np.arange(N_fp, dtype=np.int32)

    def run_quad_phys(cols: list[int], tag: str) -> np.ndarray:
        tr_fp_list, y_list = [], []
        fold_of_fp = np.full(N_fp, -1, dtype=np.int8)
        for k, f in enumerate(q_folds):
            rng = np.random.default_rng(900 + k)
            vis_k = f["visible"]
            train_reg = np.zeros((H, W), dtype=bool)
            for j, f2 in enumerate(q_folds):
                if j != k:
                    train_reg |= f2["region"]
            pos_full = np.flatnonzero(train_reg & foot & vis_k)
            if pos_full.size > 40_000:
                pos_full = rng.choice(pos_full, 40_000, replace=False)
            neg_full = np.flatnonzero(train_reg & foot & (~vis_k))
            neg_full = rng.choice(neg_full, min(neg_full.size, 2 * pos_full.size), replace=False)
            tr_fp_list.append(full_to_fp[np.concatenate([pos_full, neg_full])])
            y_list.append(np.concatenate([np.ones(pos_full.size, np.int8), np.zeros(neg_full.size, np.int8)]))
            fold_of_fp[full_to_fp[np.flatnonzero(f["region"] & foot)]] = k

        X_tr_list = [np.empty((idx.size, len(cols)), dtype=np.float32) for idx in tr_fp_list]
        for j, c in enumerate(cols):
            vec = X_fp.read_channel(c)
            for k, idx in enumerate(tr_fp_list):
                X_tr_list[k][:, j] = vec[idx]
            del vec
        clfs = [MD._fit_pair(X_tr_list[k], y_list[k], use_lgbm=True, seed=910 + k) for k in range(len(q_folds))]
        del X_tr_list
        gc.collect()

        preds_fp = np.zeros(N_fp, dtype=np.float32)
        for start in range(0, N_fp, 650_000):
            end = min(N_fp, start + 650_000)
            blk = X_fp.extract_slice(start, end, cols)
            f_sub = fold_of_fp[start:end]
            for k in range(len(q_folds)):
                m = f_sub == k
                if m.any():
                    preds_fp[start:end][m] = MD._predict_pair(clfs[k], blk[m])
            del blk
        out = np.zeros((H, W), dtype=np.float32)
        out.ravel()[fp_idx] = preds_fp
        log(f"Blind quadrant {tag} finished.")
        return out

    p_q_b0 = run_quad_phys(idx_b0, "B0_core")
    p_q_h77e = run_quad_phys(idx_all, "H77E_Phys96")

    # Emit 37,654 dots globally across Nevada (and area-proportional control) with `visible = False` inside each quadrant!
    allow_nv = foot
    dots_q_b0 = emit_over_folds(q_folds, foot, lambda f, al, b: EM.emit_dots_greedy(p_q_b0, al, b))
    dots_q_prop = emit_over_folds(q_folds, foot, lambda f, al, b: EM.emit_dots_submodular_bayes(p_q_h77e, al, b))
    dots_q_glob = EM.emit_dots_submodular_bayes(p_q_h77e, allow_nv, budget=FULL_BUDGET)

    q_terms = {}
    q_per_fold = {}
    for arm_name, dm in {
        "Quad_B0_core_greedy": dots_q_b0,
        "Quad_H77E_Phys96_area_prop_submod": dots_q_prop,
        "Quad_H77E_Phys96_global_bayes_submod": dots_q_glob,
    }.items():
        pf = dm.astype(np.float32)
        t_sum = None
        fl = []
        for f in q_folds:
            res, ta = EH.evaluate(pf, f, foot)
            t_sum = ta.copy() if t_sum is None else (t_sum + ta)
            fl.append({"fold": f["name"], "dti": round(float(res["dti"]), 6), "n_truth": int(res["n_truth"]), "n_emitted": int(res["n_emitted"])})
        q_terms[arm_name] = t_sum
        q_per_fold[arm_name] = fl

    q_summary = EH.pooled_summary(
        q_terms, draws=2000, seed=520810, candidate="Quad_H77E_Phys96_global_bayes_submod"
    )
    for arm_name, info in q_summary["scores"].items():
        log(f"Blind-4-Quadrant [{arm_name}]: {info['dti']:.6f} (95% CI [{info['ci95'][0]:.5f}, {info['ci95'][1]:.5f}]), withheld={info['withheld_positive_pixels']}")
    return {
        "withheld_positive_pixels": int(q_summary["scores"]["Quad_H77E_Phys96_global_bayes_submod"]["withheld_positive_pixels"]),
        "per_fold": q_per_fold,
        "pooled_summary": q_summary,
        "prior_repo_benchmark_H77_Step": 0.185834,
        "runtime_seconds": round(time.time() - t_q, 2),
    }


def main() -> int:
    t0 = time.time()
    grid = G.load_grid(ROOT / ".cache" / "data")
    X_fp, dirs, _ = FE.build_or_load_static_stack(ROOT, force=False)

    s20_rec, s20_terms = evaluate_seed(20, grid, X_fp, dirs)
    s21_rec, s21_terms = evaluate_seed(21, grid, X_fp, dirs)

    s30_npz = np.load(ROOT / ".cache" / "h77e_work" / "seed30_terms.npz")
    common_arms = [
        "Reference_H33_vis_excl3",
        "Arm0_B0_core_greedy",
        "Arm3b_phys_only_bayes_submod",
        "Arm6_H77E_bayes_submod_unconstrained",
        "Arm7_H77E_lagrangian_knapsack_submod",
    ]
    # Merge identical 20 km physical blocks across all 3 seeds so cluster bootstrap respects real geography!
    pooled_3seed_terms = {
        arm: s30_npz[arm] + s20_terms[arm] + s21_terms[arm]
        for arm in common_arms
    }
    pooled_3seed = EH.pooled_summary(
        pooled_3seed_terms, draws=2000, seed=520810, candidate="Arm7_H77E_lagrangian_knapsack_submod"
    )
    for arm_name, info in pooled_3seed["scores"].items():
        log(
            f"3-SEED POOLED ({info['withheld_positive_pixels']} withheld px) [{arm_name}]: "
            f"{info['dti']:.6f} (95% CI [{info['ci95'][0]:.5f}, {info['ci95'][1]:.5f}])"
        )

    quad_report = evaluate_blind_quadrants(grid, X_fp)

    out = {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "lane": "H77E",
        "experiment": "Exp2_ZeroTuning_MultiSeed_and_BlindProvince_Replication",
        "frozen_from_experiment_1": True,
        "seed20": s20_rec,
        "seed21": s21_rec,
        "three_seed_pooled_34235px": pooled_3seed,
        "blind_4_quadrant_transfer_57448px": quad_report,
        "head_to_head_prior_lanes_same_splits": {
            "Seed20_11321px": {
                "H77E_Arm7_ours": round(float(s20_rec["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["dti"]), 6),
                "H77_Step_PR9": 0.152718,
                "H77D_PR12": 0.140298,
                "ContactPhys_LiDAR5": 0.109300,
                "H78_GeothermalVents": 0.070800,
                "LIA_PR8": 0.070600,
            },
            "Seed30_11342px": {
                "H77E_Arm7_ours": round(float(json.loads((ROOT / "evidence" / "h77pd_exp1_ablation_holdout.json").read_text())["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["dti"]), 6),
                "H77D_PR12": 0.150110,
            },
            "Seed21_11572px": {
                "H77E_Arm7_ours": round(float(s21_rec["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["dti"]), 6),
                "H77_HSL_PR11_unique_emitted": 0.148000,
            },
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out_path = ROOT / "evidence" / "h77pd_exp2_multiseed_replication.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"Saved {out_path} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
