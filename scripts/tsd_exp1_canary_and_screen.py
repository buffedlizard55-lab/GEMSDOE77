#!/usr/bin/env python3
"""Experiment 1 of 3 (H77-TSD): Feature Synthesis, Single-Feature Leakage Canaries & Standalone Screen.

Strictly obeys Parallel-Run Protocol Rules 2, 3, 4, 6:
- Reuses shared cached stack (data/processed/stack.npy), vendor/gems57/holdout.py,
  and vendor/gems57/evaluate_holdout.py (gems57-pooled-hide-v2).
- Tests every single H77-TSD feature alone on the holdout BEFORE trusting any model result;
  symmetric ROC-AUC > 0.90 triggers an immediate leakage stop.
- Labels every metric as HOLDOUT-DTI with evaluator version, withheld positive count, and 95% CI.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH
from gems57 import grid as G
from gems57 import holdout as HO
from gems57 import metric as M
from gems77 import features77 as F77
from gems77_tsd.features import (
    TSD_STATIC_IDX,
    TSD_STATIC_NAMES,
    TSD_VIS_NAMES,
    build_tsd_static_features,
    compute_tsd_visible_mechanics,
)
from gems77d.features import clean_band, rank01_fp
from gems77d.submodular import allocate_submodular_300m

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
BUDGET = 37_654
EXCL = 3


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    t0 = time.time()
    grid = G.load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.float32)

    log("Building / loading 10 static H77-TSD channels...")
    tsd_mm, meta = build_tsd_static_features(RAW, PROC)
    log(f"H77-TSD static stack ready: shape={meta['shape']}")

    folds, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    log(f"Seed 30 holdout: withheld_positives={int(hidden.sum())} visible_positives={int(visible.sum())}")

    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    geod_rank_2d = rank01_fp(clean_band(stack[:, :, F77.IDX["geod_2ndinv"]], foot), foot)
    vis_mech_fp = compute_tsd_visible_mechanics(
        visible,
        foot,
        geod_rank_2d,
        tsd_mm[:, TSD_STATIC_IDX["tsd_transtens_index"]],
        tsd_mm[:, TSD_STATIC_IDX["tsd_swarm_excess"]],
    )
    del geod_rank_2d

    # ---- 1. Single-feature symmetric ROC-AUC leakage canaries ----
    rng = np.random.default_rng(7730)
    pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
    neg_pool_fp = fp_lut[np.flatnonzero(foot & ~cat & (dist_cat > 3))]
    neg_fp = rng.choice(neg_pool_fp, size=min(200_000, neg_pool_fp.size), replace=False)
    can_fp = np.concatenate([pos_fp, neg_fp])
    y_can = np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])

    canaries = {}
    X_static_can = np.asarray(tsd_mm[can_fp], dtype=np.float32)
    for j, nm in enumerate(TSD_STATIC_NAMES):
        v = X_static_can[:, j]
        m = np.isfinite(v)
        auc = float(roc_auc_score(y_can[m], v[m])) if m.sum() > 100 else None
        sym = max(auc, 1.0 - auc) if auc is not None else None
        canaries[nm] = {"auc": auc, "symmetric_auc": sym, "leakage_flag": bool(sym and sym > 0.90)}
        log(f"canary {nm:28s}: AUC={auc:.4f} sym_AUC={sym:.4f}")

    X_vis_can = np.asarray(vis_mech_fp[can_fp], dtype=np.float32)
    for j, nm in enumerate(TSD_VIS_NAMES):
        v = X_vis_can[:, j]
        m = np.isfinite(v)
        auc = float(roc_auc_score(y_can[m], v[m])) if m.sum() > 100 else None
        sym = max(auc, 1.0 - auc) if auc is not None else None
        canaries[nm] = {"auc": auc, "symmetric_auc": sym, "leakage_flag": bool(sym and sym > 0.90)}
        log(f"canary {nm:28s}: AUC={auc:.4f} sym_AUC={sym:.4f}")

    leaked = [k for k, r in canaries.items() if r["leakage_flag"]]
    max_sym_auc = max(r["symmetric_auc"] for r in canaries.values() if r["symmetric_auc"] is not None)
    worst_canary_feature = max(
        (k for k in canaries if canaries[k]["symmetric_auc"] is not None),
        key=lambda k: canaries[k]["symmetric_auc"],
    )
    if leaked:
        raise RuntimeError(f"LEAKAGE CANARY FIRED (>0.90): {leaked}")

    # ---- 2. Standalone label-free surface screen on shared evaluator ----
    def to_2d(col_fp: np.ndarray) -> np.ndarray:
        a = np.zeros((h, w), dtype=np.float32)
        a[foot] = col_fp
        return a

    allowed = foot & ~visible & (dist_vis > EXCL)
    arms_surf = {
        "S_tsd_physics": to_2d(tsd_mm[:, TSD_STATIC_IDX["tsd_composite_surface"]]),
        "S_tsd_deficit": to_2d(tsd_mm[:, TSD_STATIC_IDX["tsd_strain_scarp_deficit"]]),
        "A0_random": (np.random.default_rng(999).random((h, w), dtype=np.float32) * foot).astype(np.float32),
    }

    terms_by = {}
    arms_out = {}
    for name, surf in arms_surf.items():
        dots = allocate_submodular_300m(
            surf, allowed, target_n=BUDGET, support_quantile=0.92, gamma=2.5, min_sep_px=2.20
        )
        terms = None
        per = {}
        for f in folds:
            res_f, t = EH.evaluate(dots.astype(np.float32), f, valid=foot, block_side=200)
            per[f["name"]] = {
                "dti": float(res_f["dti"]),
                "n_truth": int(res_f["n_truth"]),
                "emitted": int(res_f["n_emitted"]),
            }
            terms = t if terms is None else terms + t
        terms_by[name] = terms
        pooled_dti = float(EH.from_terms(terms.sum(axis=0)))
        arms_out[name] = {
            "dots": int(dots.sum()),
            "per_fold": per,
            "pooled_holdout_dti": pooled_dti,
        }
        log(f"Screen arm {name:18s}: HOLDOUT-DTI={pooled_dti:.5f} dots={int(dots.sum())}")

    summary = EH.pooled_summary(terms_by, draws=1000, seed=520810, candidate="S_tsd_physics")
    out = {
        "experiment": "H77-TSD-E1",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "seed": 30,
        "withheld_positive_pixels": int(hidden.sum()),
        "budget_dots": BUDGET,
        "exclusion_px": EXCL,
        "feature_meta": meta,
        "canaries": canaries,
        "max_single_feature_symmetric_auc": float(max_sym_auc),
        "worst_canary_feature": worst_canary_feature,
        "leakage_gt_0p90": leaked,
        "arms": arms_out,
        "pooled_summary": summary,
        "verdict": "PASS-CANARY-PROCEED-TO-E2",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    for d in (ROOT / "evidence", ROOT / "docs" / "evidence"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "tsd_exp1_canary_and_screen.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"Wrote evidence/tsd_exp1_canary_and_screen.json in {out['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
