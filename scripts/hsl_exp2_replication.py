#!/usr/bin/env python3
"""GEMSDOE77 Lane H77-HSL — Experiment 2:
Independent Seed-21 whole-component holdout replication + Two-Draw Pooled Summary
(Seed 30 + Seed 21 = 22,914 withheld positive pixels, 8 quadrant folds).

Evaluator: vendor gems57.evaluate_holdout (gems57-pooled-hide-v2), alpha=0.2,
beta=0.8, 300 m triangular kernel, 1,000 paired 20 km physical-block bootstrap draws.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems57 import metric as M  # noqa: E402
from gems77hsl import features_hsl as FH, emit_hsl as EM  # noqa: E402
from hsl_exp1_holdout import (  # noqa: E402
    arm_channels,
    build_champion_vis_dots,
    log,
    train_predict_chw,
    train_predict_structural,
)

RAW = ROOT / "data" / "raw"
PROC = ROOT / ".cache" / "hsl_processed"
SEED_REP = 21
SEED_EXP1 = 30
SEL_B = 37_654
SEL_E = 2


def evaluate_draw(
    seed: int,
    stack_chw: np.ndarray,
    grid,
    u2_lid: np.ndarray,
    v2_lid: np.ndarray,
):
    foot = grid.footprint
    H, W = foot.shape
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    log(f"Seed {seed}: withheld={int(hidden.sum())} visible={int(visible.sum())}")

    rng = np.random.default_rng(1234 + seed)
    fold_samples = []
    for k, fold in enumerate(folds):
        train_region = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                train_region |= f2["region"]
        pos = np.flatnonzero(visible & train_region)
        if pos.size > 40_000:
            pos = rng.choice(pos, 40_000, replace=False)
        neg_mask = train_region & foot & ~visible & (dist_vis > 3)
        hard = np.flatnonzero(neg_mask & (dist_vis <= 15))
        easy = np.flatnonzero(neg_mask & (dist_vis > 15))
        n_neg = 2 * pos.size
        nh = min(hard.size, n_neg // 2)
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([
            rng.choice(hard, nh, replace=False) if nh else np.array([], int),
            rng.choice(easy, ne, replace=False) if ne else np.array([], int),
        ])
        idx = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        fold_samples.append((idx, y))

    arm_list = ["C2_standard", "C3_contactphys", "C4_contactphys_lidar5", "HSL_lidar12", "HSL_nolabel"]
    oof = {}
    for arm in arm_list:
        p_cache = PROC / f"hsl_oof_s{seed}_{arm}.npy"
        if p_cache.exists():
            oof[arm] = np.load(p_cache)
            log(f"s{seed}/{arm}: loaded cached OOF")
            continue
        cols = [FH.IDX[n] for n in arm_channels(arm)]
        surf = np.zeros((H, W), dtype=np.float32)
        for k, fold in enumerate(folds):
            idx, y = fold_samples[k]
            surf += train_predict_chw(stack_chw, cols, idx, y, fold["region"] & foot, f"s{seed}/{arm}/{fold['name']}")
        np.save(p_cache, surf)
        oof[arm] = surf

    p_struct_cache = PROC / f"hsl_oof_s{seed}_HSL_structural.npy"
    if p_struct_cache.exists():
        oof["HSL_structural"] = np.load(p_struct_cache)
        log(f"s{seed}/HSL_structural: loaded cached OOF")
    else:
        V_test = FH.compute_visible_fault_features(visible, foot, u2_lid, v2_lid)
        surf_st = train_predict_structural(
            stack_chw, oof["HSL_nolabel"], V_test, folds, visible, foot, u2_lid, v2_lid, seed_base=400 + seed
        )
        np.save(p_struct_cache, surf_st)
        oof["HSL_structural"] = surf_st
        del V_test
        gc.collect()

    def emit_pooled(p_surface: np.ndarray, submodular: bool = False) -> np.ndarray:
        allowed_folds = []
        tot = 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > SEL_E)
            allowed_folds.append(a)
            tot += int(a.sum())
        dots = np.zeros(foot.shape, dtype=bool)
        for a in allowed_folds:
            b = int(round(SEL_B * int(a.sum()) / max(tot, 1)))
            if b > 0:
                if submodular:
                    dots |= EM.emit_dots_submodular_bayes(p_surface, a, b)
                else:
                    dots |= EM.emit_dots_standard(p_surface, a, b)
        return dots

    def score_dots(dots: np.ndarray):
        terms = None
        per = {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": res["dti"], "n_truth": res["n_truth"], "emitted": res["n_emitted"]}
            terms = t if terms is None else terms + t
        return per, terms

    terms_by_arm = {}
    arms_out = {}

    rng2 = np.random.default_rng(999 + seed)
    rand_surf = rng2.random(foot.shape, dtype=np.float32) * foot
    dots_r = emit_pooled(rand_surf, submodular=False)
    per_r, terms_r = score_dots(dots_r)
    terms_by_arm["C0_random"] = terms_r
    arms_out["C0_random"] = {"dots": int(dots_r.sum()), "per_fold": per_r, "pooled_holdout_dti": float(EH.from_terms(terms_r.sum(axis=0)))}
    del rand_surf

    dots_c1 = build_champion_vis_dots(stack_chw, folds, visible, foot, SEL_B, SEL_E)
    per_c1, terms_c1 = score_dots(dots_c1)
    terms_by_arm["C1_champion_vis"] = terms_c1
    arms_out["C1_champion_vis"] = {"dots": int(dots_c1.sum()), "per_fold": per_c1, "pooled_holdout_dti": float(EH.from_terms(terms_c1.sum(axis=0)))}

    for arm in ["C2_standard", "C3_contactphys", "C4_contactphys_lidar5", "HSL_lidar12", "HSL_nolabel", "HSL_structural"]:
        dots = emit_pooled(oof[arm], submodular=False)
        per, terms = score_dots(dots)
        terms_by_arm[arm] = terms
        arms_out[arm] = {"dots": int(dots.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        log(f"s{seed}/{arm}: HOLDOUT-DTI={arms_out[arm]['pooled_holdout_dti']:.4f}")

    for base_arm in ("HSL_lidar12", "HSL_nolabel", "HSL_structural"):
        sub_name = f"{base_arm}_submod"
        dots = emit_pooled(oof[base_arm], submodular=True)
        per, terms = score_dots(dots)
        terms_by_arm[sub_name] = terms
        arms_out[sub_name] = {"dots": int(dots.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        log(f"s{seed}/{sub_name}: HOLDOUT-DTI={arms_out[sub_name]['pooled_holdout_dti']:.4f}")

    # Evaluate uniqueness-stratified two-pool emitter (62% structural + 38% off-corridor physical)
    allowed_folds = []
    tot_al = 0
    for fold in folds:
        a = fold["region"] & foot & ~visible & (dist_vis > SEL_E)
        allowed_folds.append(a)
        tot_al += int(a.sum())
    d1_all = np.zeros(foot.shape, dtype=bool)
    p_hyb = (0.70 * oof["HSL_nolabel"] + 0.30 * oof["HSL_structural"]).astype(np.float32)
    for a in allowed_folds:
        b = int(round(SEL_B * int(a.sum()) / max(tot_al, 1)))
        if b > 0:
            b1 = int(round(0.62 * b))
            d1_all |= EM.emit_dots_submodular_bayes(oof["HSL_structural"], a, b1)
    dist_d1 = ndimage.distance_transform_edt(~d1_all)
    d2_all = np.zeros(foot.shape, dtype=bool)
    for a in allowed_folds:
        b = int(round(SEL_B * int(a.sum()) / max(tot_al, 1)))
        if b > 0:
            b1 = int(round(0.62 * b))
            b2 = b - b1
            d2_all |= EM.emit_dots_submodular_bayes(p_hyb, a & (dist_d1 > 4.0), b2)
    dots_strat = d1_all | d2_all
    per_st, terms_st = score_dots(dots_strat)
    terms_by_arm["HSL_stratified_submod"] = terms_st
    arms_out["HSL_stratified_submod"] = {
        "dots": int(dots_strat.sum()),
        "per_fold": per_st,
        "pooled_holdout_dti": float(EH.from_terms(terms_st.sum(axis=0))),
    }
    log(f"s{seed}/HSL_stratified_submod: HOLDOUT-DTI={arms_out['HSL_stratified_submod']['pooled_holdout_dti']:.4f}")

    del oof
    gc.collect()
    return int(hidden.sum()), [f["receipt"] for f in folds], arms_out, terms_by_arm


def main():
    t0 = time.time()
    stack_chw = np.load(PROC / "hsl_stack.npy", mmap_mode="r")
    grid = G.load_grid(RAW)

    lid_path = RAW / "external" / "lidar_scarp_features_u8.tif"
    with rasterio.open(lid_path) as ds:
        step_u8 = ds.read(3).astype(np.float32) / np.float32(255.0)
        coh_u8 = ds.read(10).astype(np.float32) / np.float32(255.0)
        strike_u8 = ds.read(11).astype(np.float32)
        lvalid = (ds.read(12) > 0).astype(np.float32)
    theta_lid = (strike_u8 / np.float32(255.0)) * np.float32(np.pi)
    w_lid = coh_u8 * step_u8 * lvalid
    u2_lid = (w_lid * np.cos(np.float32(2.0) * theta_lid)).astype(np.float32)
    v2_lid = (w_lid * np.sin(np.float32(2.0) * theta_lid)).astype(np.float32)
    del step_u8, coh_u8, strike_u8, lvalid, theta_lid, w_lid
    gc.collect()

    # Evaluate Seed 30 (cached OOFs) and Seed 21 (independent replication)
    n30, rec30, arms30, terms30 = evaluate_draw(SEED_EXP1, stack_chw, grid, u2_lid, v2_lid)
    n21, rec21, arms21, terms21 = evaluate_draw(SEED_REP, stack_chw, grid, u2_lid, v2_lid)

    summary_s30_struct = EH.pooled_summary(terms30, draws=1000, seed=520810, candidate="HSL_structural_submod")
    summary_s30_strat = EH.pooled_summary(terms30, draws=1000, seed=520810, candidate="HSL_stratified_submod")
    summary_s30_nolabel = EH.pooled_summary(terms30, draws=1000, seed=520810, candidate="HSL_nolabel_submod")
    summary_s30_lidar12 = EH.pooled_summary(terms30, draws=1000, seed=520810, candidate="HSL_lidar12_submod")

    e1_path = ROOT / "evidence" / "hsl_holdout_exp1.json"
    if e1_path.exists():
        e1 = json.loads(e1_path.read_text())
        e1["arms"] = arms30
        e1["pooled_summary_hsl_structural_submod"] = summary_s30_struct
        e1["pooled_summary_hsl_stratified_submod"] = summary_s30_strat
        e1["pooled_summary_hsl_nolabel_submod"] = summary_s30_nolabel
        e1["pooled_summary_hsl_lidar12_submod"] = summary_s30_lidar12
        e1_path.write_text(json.dumps(e1, indent=2) + "\n")

    summary_s21_struct = EH.pooled_summary(terms21, draws=1000, seed=520810, candidate="HSL_structural_submod")
    summary_s21_strat = EH.pooled_summary(terms21, draws=1000, seed=520810, candidate="HSL_stratified_submod")
    summary_s21_nolabel = EH.pooled_summary(terms21, draws=1000, seed=520810, candidate="HSL_nolabel_submod")
    summary_s21_lidar12 = EH.pooled_summary(terms21, draws=1000, seed=520810, candidate="HSL_lidar12_submod")

    # Two-draw pooled block terms (22,914 withheld positives across 8 quadrant folds)
    terms_two_draw = {k: np.vstack([terms30[k], terms21[k]]) for k in terms30.keys()}
    summary_2draw_struct = EH.pooled_summary(terms_two_draw, draws=1000, seed=520810, candidate="HSL_structural_submod")
    summary_2draw_strat = EH.pooled_summary(terms_two_draw, draws=1000, seed=520810, candidate="HSL_stratified_submod")
    summary_2draw_nolabel = EH.pooled_summary(terms_two_draw, draws=1000, seed=520810, candidate="HSL_nolabel_submod")
    summary_2draw_lidar12 = EH.pooled_summary(terms_two_draw, draws=1000, seed=520810, candidate="HSL_lidar12_submod")

    out = {
        "evidence_class": "HOLDOUT-DTI",
        "lane": "H77-HSL",
        "preregistration_id": "GEMSDOE77-PREREG-HSL-1",
        "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "operating_point": {"budget": SEL_B, "exclusion_px": SEL_E},
        "seed21_replication": {
            "seed": SEED_REP,
            "withheld_positive_pixels": n21,
            "fold_receipts": rec21,
            "arms": arms21,
            "pooled_summary_hsl_structural_submod": summary_s21_struct,
            "pooled_summary_hsl_stratified_submod": summary_s21_strat,
            "pooled_summary_hsl_nolabel_submod": summary_s21_nolabel,
            "pooled_summary_hsl_lidar12_submod": summary_s21_lidar12,
        },
        "two_draw_pooled_s30_s21": {
            "seeds": [SEED_EXP1, SEED_REP],
            "withheld_positive_pixels": n30 + n21,
            "pooled_summary_hsl_structural_submod": summary_2draw_struct,
            "pooled_summary_hsl_stratified_submod": summary_2draw_strat,
            "pooled_summary_hsl_nolabel_submod": summary_2draw_nolabel,
            "pooled_summary_hsl_lidar12_submod": summary_2draw_lidar12,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "evidence" / "hsl_holdout_exp2_replication.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"EXP 2 DONE in {out['elapsed_s']}s -> {out_path}")
    log("=== TWO-DRAW POOLED HOLDOUT-DTI (22,914 withheld positives) ===")
    for k, v in summary_2draw_struct["scores"].items():
        log(f"  {k:26s}: DTI={v['dti']:.4f} CI95={v['ci95']}")
    log("=== TWO-DRAW PAIRED DIFFERENCES (HSL_structural_submod minus arm) ===")
    for k, v in summary_2draw_struct["paired_differences"].items():
        log(f"  vs {k:23s}: delta={v['delta']:+.4f} CI95=[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    log("=== TWO-DRAW PAIRED DIFFERENCES (HSL_nolabel_submod minus arm) ===")
    for k, v in summary_2draw_nolabel["paired_differences"].items():
        if k in ("C0_random", "C1_champion_vis", "C2_standard", "C3_contactphys", "C4_contactphys_lidar5", "HSL_nolabel"):
            log(f"  vs {k:23s}: delta={v['delta']:+.4f} CI95=[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
