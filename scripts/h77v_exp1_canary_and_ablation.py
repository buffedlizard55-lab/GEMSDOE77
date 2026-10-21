#!/usr/bin/env python3
"""H77V Experiment 1 (`scripts/h77v_exp1_canary_and_ablation.py`).

1. Protocol Rule 4 Leakage Canary:
   Tests each of the 16 new H77V channels (and all 36 base + 4 fold-visible LOCO channels)
   individually on the pinned `gems57-pooled-hide-v2` spatial holdout (Seed 30, 11,342 withheld
   positive pixels). Any channel with max(AUC, 1-AUC) > 0.90 triggers an immediate STOP-LEAKAGE halt.

2. Label-Free Physical Surface Ablation (Advection-Backtracked vs. Isotropic Well Smoothing):
   Evaluates the pure label-free H77V physical composite surface (`S_h77v_advect`) against the
   unshifted isotropic hydrothermal control (`S_hydro_isotropic`) and uniform random baseline (`A0_random`)
   under identical 300 m submodular dot placement (`B = 37,654` dots, `d_vis > 3 px`) using the
   unmodified `vendor/gems57/evaluate_holdout.py` (`gems57-pooled-hide-v2`).
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH
from gems57 import grid as G
from gems57 import holdout as HO
from src.gems77d.submodular import allocate_submodular_300m
from src.gems77v.features import (
    ALL_STATIC_NAMES,
    H77V_CHANNEL_NAMES,
    IDX_H77V,
    LOCO_NAMES,
    build_h77v_fold_mechanics,
    build_h77v_static_stack,
    load_h77v_domains,
)

DATA_DIR = ROOT / "data" / "raw"
WORK_DIR = ROOT / "data" / "processed" / "h77v_features"
EVIDENCE_PATH = ROOT / "evidence" / "h77v_exp1_canary_and_ablation.json"

TARGET_DOTS = 37_654
EXCL_PX = 3.0


def log(msg: str) -> None:
    print(f"[h77v-exp1] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    log("Loading H77V 52-channel static feature stack...")
    mm, meta = build_h77v_static_stack(DATA_DIR, WORK_DIR)
    doms = load_h77v_domains(WORK_DIR)
    log(f"Loaded static stack shape={mm.shape} in {time.time() - t0:.1f}s")

    grid = G.load_grid(DATA_DIR)
    foot = grid.footprint
    cat = grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    folds, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    withheld_total = int(hidden.sum())
    log(f"Built gems57-pooled-hide-v2 Seed 30 spatial folds: withheld_positives={withheld_total}")

    vis_mech_fp = build_h77v_fold_mechanics(
        visible, foot, doms["geod_rank"], cache_path=WORK_DIR / "loco_vis_seed30.npy"
    )

    # 1. Protocol Rule 4 single-feature ROC-AUC leakage canaries across all 56 features
    rng = np.random.default_rng(1030)
    pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    neg_pool_fp = fp_lut[np.flatnonzero(foot & ~cat & (dist_cat > 3.0))]
    neg_fp = rng.choice(neg_pool_fp, size=min(200_000, neg_pool_fp.size), replace=False)
    can_fp = np.concatenate([pos_fp, neg_fp])
    y_can = np.concatenate([np.ones(pos_fp.size, dtype=np.int8), np.zeros(neg_fp.size, dtype=np.int8)])

    log("Running Protocol Rule 4 single-feature ROC-AUC leakage canaries across all 56 features...")
    canary_rows = []
    max_sym_auc_h77v = 0.0
    max_sym_name_h77v = ""
    max_sym_auc_all = 0.0
    max_sym_name_all = ""

    Xc = np.asarray(mm[can_fp], dtype=np.float32)
    for ch_i, name in enumerate(ALL_STATIC_NAMES):
        x = Xc[:, ch_i]
        ok = np.isfinite(x)
        if ok.sum() < 100 or len(np.unique(y_can[ok])) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_can[ok], x[ok]))
        sym_auc = max(auc, 1.0 - auc)
        is_h77v = ch_i in IDX_H77V
        row = {
            "index": ch_i,
            "feature": name,
            "group": "h77v_new" if is_h77v else "base_static",
            "raw_auc": round(auc, 5),
            "symmetric_auc": round(sym_auc, 5),
            "valid_fraction": round(float(ok.mean()), 5),
            "passes_gate_le_0p90": bool(sym_auc <= 0.90),
        }
        canary_rows.append(row)
        if is_h77v and sym_auc > max_sym_auc_h77v:
            max_sym_auc_h77v = sym_auc
            max_sym_name_h77v = name
        if sym_auc > max_sym_auc_all:
            max_sym_auc_all = sym_auc
            max_sym_name_all = name
        assert sym_auc <= 0.90, f"STOP-LEAKAGE: {name} symmetric AUC={sym_auc:.4f} > 0.90"

    Vc = vis_mech_fp[can_fp]
    for l_i, name in enumerate(LOCO_NAMES):
        x = Vc[:, l_i]
        auc = float(roc_auc_score(y_can, x))
        sym_auc = max(auc, 1.0 - auc)
        canary_rows.append(
            {
                "index": len(ALL_STATIC_NAMES) + l_i,
                "feature": name,
                "group": "fold_visible_loco",
                "raw_auc": round(auc, 5),
                "symmetric_auc": round(sym_auc, 5),
                "valid_fraction": 1.0,
                "passes_gate_le_0p90": bool(sym_auc <= 0.90),
            }
        )
        if sym_auc > max_sym_auc_all:
            max_sym_auc_all = sym_auc
            max_sym_name_all = name
        assert sym_auc <= 0.90, f"STOP-LEAKAGE: {name} symmetric AUC={sym_auc:.4f} > 0.90"

    log(
        f"All 56 features passed Protocol Rule 4 Canary (<= 0.90): "
        f"max H77V={max_sym_auc_h77v:.4f} ({max_sym_name_h77v}), max all={max_sym_auc_all:.4f} ({max_sym_name_all})"
    )

    # 2. Direct Label-Free Physical Surface Ablation on gems57-pooled-hide-v2 (Seed 30)
    log("Evaluating label-free physical surface ablation on gems57-pooled-hide-v2 (Seed 30)...")
    d_v_fp = vis_mech_fp[:, 0]
    lobe_fp = vis_mech_fp[:, 1]
    step_fp = vis_mech_fp[:, 2]

    s_adv_2d = np.zeros((h, w), dtype=np.float32)
    s_adv_2d[foot] = (
        doms["s_h77v_advect"] * (0.55 + 0.45 * np.exp(-np.maximum(d_v_fp - 3.0, 0.0) / 16.0))
        + 0.16 * lobe_fp
        + 0.12 * step_fp
    ).astype(np.float32)

    s_iso_2d = np.zeros((h, w), dtype=np.float32)
    s_iso_2d[foot] = (
        doms["s_hydro_iso"] * (0.55 + 0.45 * np.exp(-np.maximum(d_v_fp - 3.0, 0.0) / 16.0))
        + 0.16 * lobe_fp
        + 0.12 * step_fp
    ).astype(np.float32)

    allowed_folds = []
    tot_allowed = 0
    for fold in folds:
        a = fold["region"] & foot & ~visible & (dist_vis > EXCL_PX)
        allowed_folds.append(a)
        tot_allowed += int(a.sum())

    def emit_submod_pooled(p_surf: np.ndarray, budget: int) -> np.ndarray:
        dots = np.zeros((h, w), dtype=bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot_allowed, 1)))
            if b > 0:
                dots |= allocate_submodular_300m(
                    p_surf, a, target_n=b, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
                )
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

    dots_adv = emit_submod_pooled(s_adv_2d, TARGET_DOTS)
    per_adv, t_adv = score_dots(dots_adv)

    dots_iso = emit_submod_pooled(s_iso_2d, TARGET_DOTS)
    per_iso, t_iso = score_dots(dots_iso)

    rng_r = np.random.default_rng(7710)
    dots_rnd = np.zeros((h, w), dtype=bool)
    for a in allowed_folds:
        b = int(round(TARGET_DOTS * int(a.sum()) / max(tot_allowed, 1)))
        cand = np.flatnonzero(a)
        pick = rng_r.choice(cand, size=b, replace=False)
        dots_rnd.ravel()[pick] = True
    per_rnd, t_rnd = score_dots(dots_rnd)

    pooled = EH.pooled_summary(
        {"S_h77v_advect": t_adv, "S_hydro_isotropic": t_iso, "A0_random": t_rnd},
        draws=1000,
        seed=520810,
        candidate="S_h77v_advect",
    )

    for arm_name, sc in pooled["scores"].items():
        log(f"  {arm_name:20s}: HOLDOUT-DTI = {sc['dti']:.5f} (95% CI [{sc['ci95'][0]:.5f}, {sc['ci95'][1]:.5f}])")
    for ctrl, drec in pooled["paired_differences"].items():
        log(f"  Paired Delta (S_h77v_advect - {ctrl}): {drec['delta']:+.5f} (95% CI [{drec['ci95'][0]:+.5f}, {drec['ci95'][1]:+.5f}])")

    out_payload = {
        "lane": "H77V-VENT-ADVECT-CONDUIT",
        "experiment": "exp1_canary_and_ablation",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "implementation_sha256": EH.implementation_hashes(),
        "seed": 30,
        "withheld_positives": withheld_total,
        "leakage_canary": {
            "auc_gate_limit": 0.90,
            "total_features_tested": len(canary_rows),
            "h77v_new_features_tested": len(H77V_CHANNEL_NAMES),
            "all_passed": True,
            "max_symmetric_auc_h77v": round(max_sym_auc_h77v, 5),
            "max_symmetric_auc_h77v_feature": max_sym_name_h77v,
            "max_symmetric_auc_all": round(max_sym_auc_all, 5),
            "max_symmetric_auc_all_feature": max_sym_name_all,
            "features": canary_rows,
        },
        "label_free_surface_ablation": {
            "target_dots": TARGET_DOTS,
            "excl_px": EXCL_PX,
            "per_fold": {
                "S_h77v_advect": per_adv,
                "S_hydro_isotropic": per_iso,
                "A0_random": per_rnd,
            },
            "pooled_summary": pooled,
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(out_payload, indent=2) + "\n")
    log(f"Saved Experiment 1 evidence -> {EVIDENCE_PATH} ({out_payload['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
