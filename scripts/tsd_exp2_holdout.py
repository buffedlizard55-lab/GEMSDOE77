#!/usr/bin/env python3
"""Experiment 2 of 3 (H77-TSD): Dual-Seed Whole-Component Buffered Spatial Holdout & Ablation.

Evaluates the preregistered H77-TSD pipeline on two independent buffered whole-component
holdout draws (Seed 30: 11,342 withheld positives; Seed 20: 11,321 withheld positives)
using the unmodified shared evaluator vendor/gems57/evaluate_holdout.py (gems57-pooled-hide-v2).

Strictly obeys Parallel-Run Protocol Rules 2, 3, 4, 6:
- Zero withheld fault segments ever touch feature synthesis or classifier training.
- Zero SGMC layers and zero prior submission rasters used.
- Every number labeled HOLDOUT-DTI with evaluator version, withheld positive count, and 95% CI.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH
from gems57 import grid as G
from gems57 import holdout as HO
from gems57 import metric as M
from gems77 import emit77 as E77
from gems77 import features77 as F77
from gems77_tsd.features import (
    TSD_STATIC_IDX,
    build_tsd_static_features,
    compute_tsd_visible_mechanics,
)
from gems77d.features import clean_band, rank01_fp
from gems77d.submodular import allocate_submodular_300m

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
BUDGET = 37_654
EXCL = 3.0

# Zero-SGMC shared channels from data/processed/stack.npy
SHARED_KEEP_NAMES: list[str] = list(
    dict.fromkeys(
        F77.BLOCKS["raw"]
        + F77.BLOCKS["terrain"]
        + F77.BLOCKS["new"]
        + F77.BLOCKS["rad"]
        + F77.BLOCKS["lidar"]
    )
)
STD_BASELINE_NAMES: list[str] = list(
    dict.fromkeys(F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["rad"])
)

HGB_STATIC_KW = dict(
    max_iter=220,
    learning_rate=0.055,
    max_leaf_nodes=63,
    min_samples_leaf=80,
    l2_regularization=2.0,
    random_state=77,
)
HGB_MECH_KW = dict(
    max_iter=190,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=160,
    l2_regularization=4.0,
    random_state=177,
)
HGB_STD_KW = dict(
    max_iter=200,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=77,
)


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def assemble_fp_matrix(
    stack_mm: np.memmap,
    foot: np.ndarray,
    stack_names: list[str],
    extra_blocks: list[np.ndarray],
) -> np.ndarray:
    n_fp = int(foot.sum())
    n_extra = sum(b.shape[1] for b in extra_blocks)
    out = np.empty((n_fp, n_extra + len(stack_names)), dtype=np.float32)
    col = 0
    for b in extra_blocks:
        w = b.shape[1]
        out[:, col : col + w] = b
        col += w
    for nm in stack_names:
        out[:, col] = stack_mm[:, :, F77.IDX[nm]][foot]
        col += 1
    return out


def predict_chunked(clf: HistGradientBoostingClassifier, X_fp: np.ndarray, idx_sub: np.ndarray) -> np.ndarray:
    out = np.empty(idx_sub.size, dtype=np.float32)
    step = 300_000
    for s in range(0, idx_sub.size, step):
        sl = idx_sub[s : s + step]
        out[s : s + step] = clf.predict_proba(X_fp[sl])[:, 1].astype(np.float32)
    return out


def build_fold_io(folds: list[dict], foot: np.ndarray, visible: np.ndarray, seed: int) -> list[tuple]:
    """Construct strictly label-blind spatial-quadrant training splits with hard+easy negative sampling."""
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

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
        neg_base = foot & tr_reg & ~vis_tr & (d_vis_tr > EXCL)
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
    return fold_io


def fit_oof_surface(X_fp: np.ndarray, fold_io: list[tuple], shape: tuple[int, int], kw: dict) -> np.ndarray:
    surf = np.zeros(shape, dtype=np.float32)
    for fname, tr_idx, y_tr, te_idx, te_mask in fold_io:
        clf = HistGradientBoostingClassifier(**kw)
        clf.fit(X_fp[tr_idx], y_tr)
        surf[te_mask] = predict_chunked(clf, X_fp, te_idx)
        del clf
        gc.collect()
    return surf


def emit_submod_pooled(
    p_surf: np.ndarray,
    folds: list[dict],
    foot: np.ndarray,
    visible: np.ndarray,
    dist_vis: np.ndarray,
    budget: int,
    excl_px: float = 3.0,
) -> np.ndarray:
    h, w = foot.shape
    allowed_folds = []
    tot = 0
    for fold in folds:
        a = fold["region"] & foot & ~visible & (dist_vis > excl_px)
        allowed_folds.append(a)
        tot += int(a.sum())
    dots = np.zeros((h, w), dtype=bool)
    for a in allowed_folds:
        b = int(round(budget * int(a.sum()) / max(tot, 1)))
        if b > 0:
            dots |= allocate_submodular_300m(
                p_surf, a, target_n=b, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
            )
    return dots


def emit_greedy_pooled(
    p_surf: np.ndarray,
    folds: list[dict],
    foot: np.ndarray,
    visible: np.ndarray,
    dist_vis: np.ndarray,
    budget: int,
    excl_px: float = 3.0,
) -> np.ndarray:
    h, w = foot.shape
    allowed_folds = []
    tot = 0
    for fold in folds:
        a = fold["region"] & foot & ~visible & (dist_vis > excl_px)
        allowed_folds.append(a)
        tot += int(a.sum())
    dots = np.zeros((h, w), dtype=bool)
    for a in allowed_folds:
        b = int(round(budget * int(a.sum()) / max(tot, 1)))
        if b > 0:
            dots |= E77.emit_dots(p_surf, a, b)
    return dots


def score_dots_pooled(dots: np.ndarray, folds: list[dict], foot: np.ndarray) -> tuple[dict, np.ndarray]:
    terms = None
    per = {}
    for fold in folds:
        res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
        per[fold["name"]] = {
            "dti": float(res["dti"]),
            "n_truth": int(res["n_truth"]),
            "emitted": int(res["n_emitted"]),
        }
        terms = t if terms is None else terms + t
    return per, terms


def run_seed(
    seed: int,
    grid: G.Grid,
    tsd_mm: np.memmap,
    stack_mm: np.memmap,
    geod_rank_2d: np.ndarray,
) -> dict:
    foot = grid.footprint
    h, w = foot.shape
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    visible = folds[0]["visible"]
    hidden = folds[0]["hidden_all"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    log(f"=== Holdout Seed {seed}: withheld_positives={int(hidden.sum())} visible_positives={int(visible.sum())} ===")

    fold_io = build_fold_io(folds, foot, visible, seed)

    cache_npz = PROC / f"tsd_oof_v2_seed{seed}.npz"
    if cache_npz.exists():
        log(f"Seed {seed}: loading cached OOF surfaces from {cache_npz.name}...")
        z = np.load(cache_npz)
        s_tsd_static = z["s_tsd_static"]
        s_tsd_mech = z["s_tsd_mech"]
        s_std = z["s_std"]
    else:
        vis_mech_fp = compute_tsd_visible_mechanics(
            visible,
            foot,
            geod_rank_2d,
            tsd_mm[:, TSD_STATIC_IDX["tsd_transtens_index"]],
            tsd_mm[:, TSD_STATIC_IDX["tsd_swarm_excess"]],
        )

        # 1. Static H77-TSD + shared geophysical context (without LOCO relay mechanics)
        log(f"Seed {seed}: fitting s_tsd_static (10 H77-TSD + {len(SHARED_KEEP_NAMES)} shared channels)...")
        X_stat = assemble_fp_matrix(stack_mm, foot, SHARED_KEEP_NAMES, [tsd_mm])
        s_tsd_static = fit_oof_surface(X_stat, fold_io, (h, w), HGB_STATIC_KW)
        del X_stat
        gc.collect()

        # 2. Full H77-TSD + 5 fold-visible 12-bit LOCO relay channels + shared geophysical context
        log(f"Seed {seed}: fitting s_tsd_mech (10 H77-TSD + 5 LOCO relay + {len(SHARED_KEEP_NAMES)} shared channels)...")
        X_mech = assemble_fp_matrix(stack_mm, foot, SHARED_KEEP_NAMES, [tsd_mm, vis_mech_fp])
        s_tsd_mech = fit_oof_surface(X_mech, fold_io, (h, w), HGB_MECH_KW)
        del X_mech, vis_mech_fp
        gc.collect()

        # 3. Standard baseline HGB (RAW + TERRAIN + RAD only, no H77-TSD channels, no LOCO mechanics)
        log(f"Seed {seed}: fitting s_std ({len(STD_BASELINE_NAMES)} standard channels)...")
        X_std = assemble_fp_matrix(stack_mm, foot, STD_BASELINE_NAMES, [])
        s_std = fit_oof_surface(X_std, fold_io, (h, w), HGB_STD_KW)
        del X_std
        gc.collect()

        np.savez(
            cache_npz,
            s_tsd_static=s_tsd_static,
            s_tsd_mech=s_tsd_mech,
            s_std=s_std,
        )

    # Preregistered probability blend: 0.45 static H77-TSD view + 0.55 LOCO-relay H77-TSD view
    surf_tsd = (0.45 * s_tsd_static + 0.55 * s_tsd_mech).astype(np.float32)

    dots_by_arm = {
        "A_tsd_full": emit_submod_pooled(surf_tsd, folds, foot, visible, dist_vis, BUDGET, EXCL),
        "A_tsd_greedy": emit_greedy_pooled(surf_tsd, folds, foot, visible, dist_vis, BUDGET, EXCL),
        "A_tsd_static_only": emit_submod_pooled(s_tsd_static, folds, foot, visible, dist_vis, BUDGET, EXCL),
        "A_tsd_mech_only": emit_submod_pooled(s_tsd_mech, folds, foot, visible, dist_vis, BUDGET, EXCL),
        "A_std_greedy": emit_greedy_pooled(s_std, folds, foot, visible, dist_vis, BUDGET, EXCL),
        "A0_random": emit_greedy_pooled(
            (np.random.default_rng(700 + seed).random((h, w), dtype=np.float32) * foot).astype(np.float32),
            folds,
            foot,
            visible,
            dist_vis,
            BUDGET,
            EXCL,
        ),
    }

    terms_by = {}
    arms_out = {}
    for arm_name, dmask in dots_by_arm.items():
        per, t = score_dots_pooled(dmask, folds, foot)
        terms_by[arm_name] = t
        tot = t.sum(axis=0)
        pooled_dti = float(EH.from_terms(tot))
        arms_out[arm_name] = {
            "dots": int(dmask.sum()),
            "per_fold": per,
            "pooled_holdout_dti": pooled_dti,
            "tp_w": float(tot[0]),
            "fp_w": float(tot[1]),
            "fn_w": float(tot[2]),
            "n_truth": int(round(tot[3])),
        }
        log(f"Seed {seed} | {arm_name:20s}: HOLDOUT-DTI={pooled_dti:.5f} (dots={int(dmask.sum())})")

    pooled = EH.pooled_summary(terms_by, draws=1000, seed=520810 + seed, candidate="A_tsd_full")
    for ctrl, drec in pooled["paired_differences"].items():
        log(
            f"  -> [Seed {seed}] A_tsd_full vs {ctrl:20s}: "
            f"delta={drec['delta']:+.5f} CI95=[{drec['ci95'][0]:+.5f}, {drec['ci95'][1]:+.5f}]"
        )

    return {
        "seed": seed,
        "withheld_positive_pixels": int(hidden.sum()),
        "visible_positive_pixels": int(visible.sum()),
        "arms": arms_out,
        "pooled_summary": pooled,
    }


def main() -> int:
    t0 = time.time()
    grid = G.load_grid(RAW)
    foot = grid.footprint
    tsd_mm, _ = build_tsd_static_features(RAW, PROC)
    stack_mm = np.load(PROC / "stack.npy", mmap_mode="r")
    geod_rank_2d = rank01_fp(clean_band(stack_mm[:, :, F77.IDX["geod_2ndinv"]], foot), foot)

    res_s30 = run_seed(30, grid, tsd_mm, stack_mm, geod_rank_2d)
    res_s20 = run_seed(20, grid, tsd_mm, stack_mm, geod_rank_2d)

    out = {
        "experiment": "H77-TSD-E2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "implementation_sha256": EH.implementation_hashes(),
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "budget_dots": BUDGET,
        "exclusion_px": EXCL,
        "seed_30": res_s30,
        "seed_20": res_s20,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    for d in (ROOT / "evidence", ROOT / "docs" / "evidence"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "tsd_exp2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"Wrote evidence/tsd_exp2_holdout.json in {out['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
