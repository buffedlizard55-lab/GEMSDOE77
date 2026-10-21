#!/usr/bin/env python3
"""H77V Experiment 2 (`scripts/h77v_exp2_holdout.py`).

Dual-seed spatially blocked whole-component quadrant holdout evaluation on `gems57-pooled-hide-v2`
(Seed 30: 11,342 withheld positive pixels; Seed 20: 11,321 withheld positive pixels) using the
unmodified `vendor/gems57/evaluate_holdout.py` (`alpha=0.2`, `beta=0.8`, 300 m triangular kernel,
1,000-draw paired 20 km physical spatial-cluster bootstrap).

Evaluates 7 fold-rebuilt arms on identical spatial folds at `B = 37,654` dots (`excl_px = 3.0`):
  1. `A_h77v_gated_submod`: Primary H77V candidate — translation-invariant structural OOF HGB
     (`surf_base`) gated by the monotone physical Up-Gradient Advection-Backtracked Vent &
     Dilational Conduit surface (`S_h77v_advect`), emitted with 300 m submodular cover.
  2. `A_h77v_tree_submod`: Unconstrained 56-channel HGB splitting directly on the 16 regional
     hydrothermal/vent channels (`0.35*s_h77v_static + 0.45*s_h77v_mech + 0.20*s_vent_spec`),
     testing the "Basin-ID Tree-Split Trap" hypothesis.
  3. `A_base_no_h77v_submod`: Matched control dropping all 16 new H77V channels (36 base + 4 LOCO
     mechanics) with identical 300 m submodular cover.
  4. `A_h77v_phys_only_submod`: Pure label-free physical `S_h77v_advect` surface modulated by
     fold-visible LOCO mechanics (zero ML classifier), emitted with 300 m submodular cover.
  5. `A_h77v_greedy`: Identical `A_h77v_gated` surface emitted via standard greedy NMS (`emit77.emit_dots`).
  6. `A0_h33_foldfair`: Fold-fair H33 (`GEMSDOE32` 0.2778 live winner) construction masked by `fold['visible']`.
  7. `A0_random`: Uniform random allowed dots.
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
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH
from gems57 import grid as G
from gems57 import holdout as HO
from gems57 import metric as M
from gems77 import emit77
from src.gems77d.submodular import allocate_submodular_300m
from src.gems77v.features import (
    ALL_STATIC_NAMES,
    H77V_CHANNEL_NAMES,
    IDX_ALL_STATIC,
    IDX_BASE,
    IDX_H77V,
    LOCO_NAMES,
    STATIC_IDX,
    build_h77v_fold_mechanics,
    build_h77v_static_stack,
    load_h77v_domains,
)

DATA_DIR = ROOT / "data" / "raw"
WORK_DIR = ROOT / "data" / "processed" / "h77v_features"
EVIDENCE_PATH = ROOT / "evidence" / "h77v_exp2_holdout.json"
H33_PATH = ROOT / ".cache" / "registry" / "h33.tif"

TARGET_DOTS = 37_654
EXCL_PX = 3.0

IDX_SCARP_SUB = [
    STATIC_IDX[c]
    for c in (
        "tiger_road_dist_log",
        "lidar_ex_max",
        "lidar_step_deconf",
        "lidar_dipole_deconf",
        "lidar_strike_cont",
        "dem10_slope_p90_deconf",
        "dem10_curv_prof_deconf",
        "dem10_st_ridge_deconf",
    )
]
IDX_VENT_SPEC = IDX_H77V + IDX_SCARP_SUB

HGB_STATIC_KW = dict(
    max_iter=180,
    learning_rate=0.075,
    max_leaf_nodes=55,
    min_samples_leaf=50,
    l2_regularization=2.5,
    early_stopping=False,
    random_state=7721,
)
HGB_MECH_KW = dict(
    max_iter=180,
    learning_rate=0.075,
    max_leaf_nodes=47,
    min_samples_leaf=60,
    l2_regularization=3.0,
    early_stopping=False,
    random_state=7722,
)
HGB_VENT_KW = dict(
    max_iter=160,
    learning_rate=0.07,
    max_leaf_nodes=43,
    min_samples_leaf=55,
    l2_regularization=3.0,
    early_stopping=False,
    random_state=7723,
)


def log(msg: str) -> None:
    print(f"[h77v-exp2] {msg}", flush=True)


def fit_predict_fold_fp(
    X_fp: np.memmap,
    extra_fp: np.ndarray | None,
    cols: list[int],
    tr_fp_idx: np.ndarray,
    y_tr: np.ndarray,
    te_fp_idx: np.ndarray,
    kw: dict,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    x_tr = np.asarray(X_fp[tr_fp_idx][:, cols], dtype=np.float32)
    if extra_fp is not None:
        x_tr = np.hstack([x_tr, extra_fp[tr_fp_idx]])
    clf = HistGradientBoostingClassifier(**kw)
    clf.fit(x_tr, y_tr, sample_weight=sample_weight)
    del x_tr
    gc.collect()

    preds = np.empty(te_fp_idx.size, dtype=np.float32)
    chunk = 250_000
    for s in range(0, te_fp_idx.size, chunk):
        sl = te_fp_idx[s : s + chunk]
        xb = np.asarray(X_fp[sl][:, cols], dtype=np.float32)
        if extra_fp is not None:
            xb = np.hstack([xb, extra_fp[sl]])
        preds[s : s + chunk] = clf.predict_proba(xb)[:, 1].astype(np.float32)
        del xb
    del clf
    gc.collect()
    return preds


def run_seed(
    seed: int,
    grid,
    X_fp: np.memmap,
    doms: dict[str, np.ndarray],
    h33_dots: np.ndarray,
) -> dict:
    seed_cache = WORK_DIR / f"exp2_seed{seed}_eval.json"
    if seed_cache.exists():
        log(f"Loaded cached Seed {seed} evaluation from {seed_cache.name}")
        return json.loads(seed_cache.read_text())

    foot = grid.footprint
    cat = grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    receipts = [f["receipt"] for f in folds]
    log(f"=== SEED {seed}: withheld_positives={int(hidden.sum())}, visible={int(visible.sum())} ===")

    vis_mech_fp = build_h77v_fold_mechanics(
        visible, foot, doms["geod_rank"], cache_path=WORK_DIR / f"loco_vis_seed{seed}.npy"
    )

    rng = np.random.default_rng(1000 + seed)
    fold_io = []
    for k, fold in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        vis_tr = visible & tr_reg
        d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
        pos = fp_lut[np.flatnonzero(vis_tr & foot)]
        if pos.size > 48_000:
            pos = rng.choice(pos, 48_000, replace=False)
        neg_base = foot & tr_reg & ~vis_tr & (d_vis_tr > 3.0)
        hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 16.0))]
        easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 16.0))]
        n_neg = int(2.5 * pos.size)
        nh = min(hard.size, int(0.55 * n_neg))
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        te_idx = fp_lut[np.flatnonzero(fold["region"] & foot)]
        fold_io.append((fold["name"], tr_idx, y_tr, te_idx, fold["region"] & foot))

    oof_cache = WORK_DIR / f"oof_h77v_seed{seed}.npz"
    if oof_cache.exists():
        zc = np.load(oof_cache)
        s_h77v_static = zc["s_h77v_static"]
        s_h77v_mech = zc["s_h77v_mech"]
        s_vent_spec = zc["s_vent_spec"]
        s_base_static = zc["s_base_static"]
        s_base_mech = zc["s_base_mech"]
        log(f"[SEED {seed}] Loaded cached OOF probability surfaces from {oof_cache.name}")
    else:
        s_h77v_static = np.zeros((h, w), dtype=np.float32)
        s_h77v_mech = np.zeros((h, w), dtype=np.float32)
        s_vent_spec = np.zeros((h, w), dtype=np.float32)
        s_base_static = np.zeros((h, w), dtype=np.float32)
        s_base_mech = np.zeros((h, w), dtype=np.float32)

        d_hydro_fp = doms["d_hydro"]
        d_mech_fp = doms["d_mech"]

        for fname, tr_idx, y_tr, te_idx, te_mask in fold_io:
            f_cache = WORK_DIR / f"oof_h77v_seed{seed}_{fname}.npz"
            if f_cache.exists():
                zf = np.load(f_cache)
                s_h77v_static[te_mask] = zf["p_hs"]
                s_h77v_mech[te_mask] = zf["p_hm"]
                s_vent_spec[te_mask] = zf["p_vs"]
                s_base_static[te_mask] = zf["p_bs"]
                s_base_mech[te_mask] = zf["p_bm"]
                log(f"[SEED {seed}] Loaded cached fold {fname}")
                continue
            t_f = time.time()
            w_vent = (1.0 + 0.65 * d_hydro_fp[tr_idx] + 0.35 * d_mech_fp[tr_idx]).astype(np.float32)
            p_hs = fit_predict_fold_fp(X_fp, None, IDX_ALL_STATIC, tr_idx, y_tr, te_idx, HGB_STATIC_KW)
            p_hm = fit_predict_fold_fp(X_fp, vis_mech_fp, IDX_ALL_STATIC, tr_idx, y_tr, te_idx, HGB_MECH_KW)
            p_vs = fit_predict_fold_fp(
                X_fp, vis_mech_fp, IDX_VENT_SPEC, tr_idx, y_tr, te_idx, HGB_VENT_KW, sample_weight=w_vent
            )
            p_bs = fit_predict_fold_fp(X_fp, None, IDX_BASE, tr_idx, y_tr, te_idx, HGB_STATIC_KW)
            p_bm = fit_predict_fold_fp(X_fp, vis_mech_fp, IDX_BASE, tr_idx, y_tr, te_idx, HGB_MECH_KW)
            np.savez_compressed(f_cache, p_hs=p_hs, p_hm=p_hm, p_vs=p_vs, p_bs=p_bs, p_bm=p_bm)

            s_h77v_static[te_mask] = p_hs
            s_h77v_mech[te_mask] = p_hm
            s_vent_spec[te_mask] = p_vs
            s_base_static[te_mask] = p_bs
            s_base_mech[te_mask] = p_bm
            log(f"[SEED {seed}] Fold {fname} OOF models fitted in {time.time() - t_f:.1f}s")

        np.savez_compressed(
            oof_cache,
            s_h77v_static=s_h77v_static,
            s_h77v_mech=s_h77v_mech,
            s_vent_spec=s_vent_spec,
            s_base_static=s_base_static,
            s_base_mech=s_base_mech,
        )

    # Construct translation-invariant physical vent+dilational conduit prior modulated by fold-visible LOCO mechanics
    d_v_fp = vis_mech_fp[:, 0]
    lobe_fp = vis_mech_fp[:, 1]
    step_fp = vis_mech_fp[:, 2]
    surf_phys = np.zeros((h, w), dtype=np.float32)
    surf_phys[foot] = (
        doms["s_h77v_advect"] * (0.55 + 0.45 * np.exp(-np.maximum(d_v_fp - 3.0, 0.0) / 16.0))
        + 0.16 * lobe_fp
        + 0.12 * step_fp
    ).astype(np.float32)

    surf_base = (0.45 * s_base_static + 0.55 * s_base_mech).astype(np.float32)
    surf_tree = (0.35 * s_h77v_static + 0.45 * s_h77v_mech + 0.20 * s_vent_spec).astype(np.float32)
    surf_gated = (0.65 * surf_base + 0.35 * surf_phys).astype(np.float32)

    allowed_folds = []
    tot = 0
    for fold in folds:
        a = fold["region"] & foot & ~visible & (dist_vis > EXCL_PX)
        allowed_folds.append(a)
        tot += int(a.sum())

    def emit_submod_pooled(p_surf: np.ndarray, budget: int) -> np.ndarray:
        dots = np.zeros((h, w), dtype=bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= allocate_submodular_300m(
                    p_surf, a, target_n=b, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
                )
        return dots

    def emit_greedy_pooled(p_surf: np.ndarray, budget: int) -> np.ndarray:
        dots = np.zeros((h, w), dtype=bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= emit77.emit_dots(p_surf, a, b)
        return dots

    def score_dots(dots: np.ndarray):
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

    B = TARGET_DOTS
    terms_by_arm = {}
    arm_details = {}

    for aname, s_in in (
        ("A_h77v_gated_submod", surf_gated),
        ("A_h77v_tree_submod", surf_tree),
        ("A_base_no_h77v_submod", surf_base),
        ("A_h77v_phys_only_submod", surf_phys),
    ):
        d = emit_submod_pooled(s_in, B)
        per, t = score_dots(d)
        terms_by_arm[aname] = t
        arm_details[aname] = {
            "dots": int(d.sum()),
            "per_fold": per,
            "pooled_holdout_dti": round(float(EH.from_terms(t.sum(axis=0))), 6),
        }
        log(f"[SEED {seed} B={B}] {aname:26s}: HOLDOUT-DTI={arm_details[aname]['pooled_holdout_dti']:.5f} dots={int(d.sum())}")

    d_gr = emit_greedy_pooled(surf_gated, B)
    per_gr, t_gr = score_dots(d_gr)
    terms_by_arm["A_h77v_greedy"] = t_gr
    arm_details["A_h77v_greedy"] = {
        "dots": int(d_gr.sum()),
        "per_fold": per_gr,
        "pooled_holdout_dti": round(float(EH.from_terms(t_gr.sum(axis=0))), 6),
    }
    log(f"[SEED {seed} B={B}] {'A_h77v_greedy':26s}: HOLDOUT-DTI={arm_details['A_h77v_greedy']['pooled_holdout_dti']:.5f} dots={int(d_gr.sum())}")

    # A0_h33_foldfair (H33 binary dots masked by fold visible context)
    d_h33 = np.zeros((h, w), dtype=bool)
    for fold in folds:
        valid_f = fold["region"] & foot & ~fold["masked_known"] & (dist_vis > 2.0)
        d_h33 |= h33_dots & valid_f
    per_h33, t_h33 = score_dots(d_h33)
    terms_by_arm["A0_h33_foldfair"] = t_h33
    arm_details["A0_h33_foldfair"] = {
        "dots": int(d_h33.sum()),
        "per_fold": per_h33,
        "pooled_holdout_dti": round(float(EH.from_terms(t_h33.sum(axis=0))), 6),
    }
    log(f"[SEED {seed} B={B}] {'A0_h33_foldfair':26s}: HOLDOUT-DTI={arm_details['A0_h33_foldfair']['pooled_holdout_dti']:.5f} dots={int(d_h33.sum())}")

    # A0_random
    r_surf = np.random.default_rng(999 + seed).random((h, w), dtype=np.float32) * foot
    d_rnd = emit_greedy_pooled(r_surf, B)
    per_r, t_r = score_dots(d_rnd)
    terms_by_arm["A0_random"] = t_r
    arm_details["A0_random"] = {
        "dots": int(d_rnd.sum()),
        "per_fold": per_r,
        "pooled_holdout_dti": round(float(EH.from_terms(t_r.sum(axis=0))), 6),
    }
    log(f"[SEED {seed} B={B}] {'A0_random':26s}: HOLDOUT-DTI={arm_details['A0_random']['pooled_holdout_dti']:.5f} dots={int(d_rnd.sum())}")

    pooled = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_h77v_gated_submod")
    for ctrl, drec in pooled["paired_differences"].items():
        log(
            f"  -> [SEED {seed} B={B}] A_h77v_gated_submod vs {ctrl:26s}: "
            f"delta={drec['delta']:+.5f} CI95=[{drec['ci95'][0]:+.5f}, {drec['ci95'][1]:+.5f}]"
        )

    res_out = {
        "seed": seed,
        "withheld_positive_pixels": int(hidden.sum()),
        "fold_receipts": receipts,
        "target_dots": B,
        "excl_px": EXCL_PX,
        "arms": arm_details,
        "pooled_summary": pooled,
    }
    seed_cache.write_text(json.dumps(res_out, indent=2) + "\n")
    return res_out


def main() -> int:
    t0 = time.time()
    grid = G.load_grid(DATA_DIR)
    X_fp, meta = build_h77v_static_stack(DATA_DIR, WORK_DIR)
    doms = load_h77v_domains(WORK_DIR)

    with rasterio.open(H33_PATH) as sh:
        h33_arr = sh.read(1)
    h33_dots = np.isfinite(h33_arr) & (h33_arr > 0.5) & grid.footprint

    seed30 = run_seed(30, grid, X_fp, doms, h33_dots)
    seed20 = run_seed(20, grid, X_fp, doms, h33_dots)

    out = {
        "lane": "H77V-VENT-ADVECT-CONDUIT",
        "experiment": "exp2_dual_seed_holdout",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "implementation_sha256": EH.implementation_hashes(),
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "triangular_radius_m": M.R_M,
        "n_h77v_new_channels": len(H77V_CHANNEL_NAMES),
        "n_base_static_channels": len(IDX_BASE),
        "n_total_static_channels": len(ALL_STATIC_NAMES),
        "n_visible_loco_channels": len(LOCO_NAMES),
        "sgmc_channels_used": 0,
        "prior_submission_rasters_used_in_features": 0,
        "elapsed_seconds": round(time.time() - t0, 2),
        "seed30": seed30,
        "seed20": seed20,
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(out, indent=2) + "\n")
    log(f"Saved Experiment 2 holdout report -> {EVIDENCE_PATH} ({out['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
