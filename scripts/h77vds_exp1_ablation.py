#!/usr/bin/env python3
"""Experiment 1 of 3 (`Lane H77E`):
1. Builds & verifies the 98-channel zero-SGMC, survey- & mining-deconfounded static feature memmap.
2. Runs single-feature symmetric ROC-AUC leakage canaries across ALL 106 channels (98 static + 8 fold-visible)
   on the official buffered whole-component Seed-30 holdout (`11,342` withheld positives; gate <= 0.90).
3. Evaluates strictly nested zero-SGMC physical arms on unmodified `vendor/gems57/evaluate_holdout.py`
   (`gems57-pooled-hide-v2`, alpha=0.2, beta=0.8, R=300 m, 2,000 paired 20 km block bootstrap draws).
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

from gems57 import evaluate_holdout as EH, grid as G, holdout as HO  # noqa: E402
from gems77vds import dinkelbach as DK, features as FE, models as MO  # noqa: E402
from gems77lia import allocate as LIA_AL  # noqa: E402
import rasterio  # noqa: E402

SEED = 30
BUDGET = 37_654
EXCL_PX = 3.0


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    t0 = time.time()
    log("Building / verifying 98-channel zero-SGMC static feature store...")
    bin_path, aux = FE.build_static_fp_memmap(ROOT, force=False)
    n_fp = int(aux["n_fp"])
    log(f"Static feature store ready in {time.time() - t0:.1f}s: n_fp={n_fp}, n_ch={len(FE.STATIC_CHANNEL_NAMES)}")

    grid = G.load_grid(ROOT / "data" / "raw")
    foot, cat = grid.footprint, grid.catalogue
    H, W = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(H * W, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    slope_2d = np.zeros((H, W), dtype=np.float32)
    slope_2d[foot] = np.nan_to_num(
        FE.read_static_rows(bin_path, n_fp, slice(None), [FE.STATIC_IDX["topo10_slope_max"]])[:, 0], nan=0.0
    )

    folds, _ = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    dist_vis = ndi.distance_transform_edt(~visible).astype(np.float32)
    log(f"Seed {SEED}: withheld_positive_pixels={int(hidden.sum())}, visible_pixels={int(visible.sum())}")

    # 1. Single-feature symmetric ROC-AUC leakage canaries across ALL 106 channels!
    log("Running single-feature symmetric ROC-AUC leakage canaries across all 106 channels...")
    vis_fp = FE.compute_visible_structural_fp(visible, foot, aux["u2_dir"], aux["v2_dir"], aux["geod_rank"])

    rng = np.random.default_rng(1000 + SEED)
    pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
    dist_cat = ndi.distance_transform_edt(~cat).astype(np.float32)
    neg_pool_fp = fp_lut[np.flatnonzero(foot & (~cat) & (dist_cat > 3.0))]
    neg_fp = rng.choice(neg_pool_fp, size=min(200_000, neg_pool_fp.size), replace=False)
    can_fp = np.concatenate([pos_fp, neg_fp])
    y_can = np.concatenate([np.ones(pos_fp.size, dtype=np.int8), np.zeros(neg_fp.size, dtype=np.int8)])

    Xc = FE.read_static_rows(bin_path, n_fp, can_fp)
    Vc = vis_fp[can_fp]
    canary_rows = []
    for j, nm in enumerate(FE.STATIC_CHANNEL_NAMES):
        v = Xc[:, j]
        m = np.isfinite(v)
        raw_auc = float(roc_auc_score(y_can[m], v[m])) if (m.sum() > 100 and len(np.unique(y_can[m])) == 2) else 0.5
        canary_rows.append({
            "channel": nm,
            "group": "static_phys98",
            "raw_auc": round(raw_auc, 6),
            "symmetric_auc": round(max(raw_auc, 1.0 - raw_auc), 6),
        })
    for j, nm in enumerate(FE.VISIBLE_CHANNEL_NAMES):
        v = Vc[:, j]
        raw_auc = float(roc_auc_score(y_can, v))
        canary_rows.append({
            "channel": nm,
            "group": "fold_visible8",
            "raw_auc": round(raw_auc, 6),
            "symmetric_auc": round(max(raw_auc, 1.0 - raw_auc), 6),
        })
    del Xc, Vc, vis_fp
    gc.collect()

    max_row = max(canary_rows, key=lambda r: r["symmetric_auc"])
    leak_flags = [r for r in canary_rows if r["symmetric_auc"] > 0.90]
    log(f"Canary gate check: max symmetric AUC = {max_row['symmetric_auc']:.4f} (`{max_row['channel']}`), flags={len(leak_flags)}")
    if leak_flags:
        raise RuntimeError(f"HARD STOP — LEAKAGE CANARY > 0.90: {leak_flags}")

    canary_payload = {
        "evidence_class": "LEAKAGE-CANARY",
        "evaluator_version": EH.VERSION,
        "seed": SEED,
        "withheld_positive_pixels": int(hidden.sum()),
        "channels_tested": len(canary_rows),
        "gate_threshold": 0.90,
        "max_symmetric_auc": max_row["symmetric_auc"],
        "max_channel": max_row["channel"],
        "passed": True,
        "sgmc_channels_used": 0,
        "top_10_informative_channels": sorted(canary_rows, key=lambda r: -r["symmetric_auc"])[:10],
        "rows": canary_rows,
    }
    (ROOT / "evidence" / "h77e_canaries.json").write_text(json.dumps(canary_payload, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "h77e_canaries.json").write_text(json.dumps(canary_payload, indent=2) + "\n")

    # 2. Strictly nested out-of-fold physical feature ablations on Seed 30 (B = 37,654 parity with GEMSDOE32)
    b1_cols = [FE.STATIC_IDX[n] for n in FE.BLOCKS["b1_raw_clean"]]
    b2_cols = b1_cols + [FE.STATIC_IDX[n] for n in FE.BLOCKS["b2_block_mine_deconf"]]
    b3_cols = b2_cols + [FE.STATIC_IDX[n] for n in FE.BLOCKS["b3_lidar12_chi_step"]]
    b4_cols = list(range(len(FE.STATIC_CHANNEL_NAMES)))

    def emit_dinkelbach_pooled(surf: np.ndarray, budget: int) -> np.ndarray:
        allowed_folds = [f["region"] & foot & (~visible) & (dist_vis > EXCL_PX) for f in folds]
        tot = sum(int(a.sum()) for a in allowed_folds)
        dots = np.zeros((H, W), dtype=bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                d_sub, _ = DK.allocate_dinkelbach_submodular_dti(surf, a, budget=b, min_sep_px=2.35)
                dots |= d_sub
        return dots

    def score_dots(dots: np.ndarray):
        terms = None
        per = {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": float(res["dti"]), "n_truth": int(res["n_truth"]), "emitted": int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        return per, terms

    terms_by_arm = {}
    arm_details = {}
    for aname, cols, use_lgb in (
        ("E1_B0_raw19_signfixed", b1_cols, False),
        ("E1_B1_survey_mine_deconf39", b2_cols, False),
        ("E1_B2_scarp_chi80", b3_cols, False),
        ("E1_B3_full_phys98_ens", b4_cols, True),
    ):
        log(f"Fitting {aname} ({len(cols)} channels, use_lgb={use_lgb})...")
        surf = MO.fit_predict_static_ensemble(
            bin_path, cols, folds, visible, foot, slope_2d, fp_lut, use_lgb=use_lgb, ht_weighted=True
        )
        if aname == "E1_B3_full_phys98_ens":
            np.save(ROOT / ".cache" / "h77e" / "oof_phys98_seed30.npy", surf)
        d = emit_dinkelbach_pooled(surf, BUDGET)
        del surf
        gc.collect()
        per, t = score_dots(d)
        terms_by_arm[aname] = t
        arm_details[aname] = {
            "dots": int(d.sum()),
            "per_fold": per,
            "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0))),
        }
        log(f"[E1 Seed {SEED} B={BUDGET}] {aname:28s}: HOLDOUT-DTI = {arm_details[aname]['pooled_holdout_dti']:.6f}  dots={int(d.sum())}")
        del d
        gc.collect()

    # Exact A0_h195_foldfair control (PR #6 / H33 line-set control) + Random control
    ext_h195 = ROOT / ".cache" / "data" / "external" / "champion_h195_line_distance_m.tif"
    if ext_h195.exists():
        with rasterio.open(ext_h195) as ds:
            dsup_2d = ds.read(1)
        rk = np.random.default_rng(7).random(H * W)
        d_a0 = np.zeros((H, W), dtype=bool)
        for fold in folds:
            valid_f = fold["region"] & foot & (~fold["masked_known"])
            ac = np.flatnonzero(((dsup_2d == 0) & valid_f & (dist_vis > 2.0)).ravel())
            sel_a0 = LIA_AL.greedy_select(ac[np.argsort(rk[ac])], 3.0)
            d_a0.ravel()[sel_a0] = True
        per_a0, t_a0 = score_dots(d_a0)
        terms_by_arm["A0_h195_foldfair"] = t_a0
        arm_details["A0_h195_foldfair"] = {
            "dots": int(d_a0.sum()),
            "per_fold": per_a0,
            "pooled_holdout_dti": float(EH.from_terms(t_a0.sum(axis=0))),
        }
        log(f"[E1 Seed {SEED}] {'A0_h195_foldfair':28s}: HOLDOUT-DTI = {arm_details['A0_h195_foldfair']['pooled_holdout_dti']:.6f}  dots={int(d_a0.sum())}")

    rng_r = np.random.default_rng(7700)
    d_rand = np.zeros((H, W), dtype=bool)
    allowed_folds = [f["region"] & foot & (~visible) & (dist_vis > EXCL_PX) for f in folds]
    tot = sum(int(a.sum()) for a in allowed_folds)
    for a in allowed_folds:
        b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
        d_rand.ravel()[rng_r.choice(np.flatnonzero(a), b, replace=False)] = True
    per_r, t_r = score_dots(d_rand)
    terms_by_arm["C0_random"] = t_r
    arm_details["C0_random"] = {
        "dots": int(d_rand.sum()),
        "per_fold": per_r,
        "pooled_holdout_dti": float(EH.from_terms(t_r.sum(axis=0))),
    }
    log(f"[E1 Seed {SEED}] {'C0_random':28s}: HOLDOUT-DTI = {arm_details['C0_random']['pooled_holdout_dti']:.6f}  dots={int(d_rand.sum())}")

    summary = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810, candidate="E1_B3_full_phys98_ens")
    payload = {
        "evidence_class": "HOLDOUT-DTI",
        "experiment": "H77E-E1-CANARIES-AND-PHYSICAL-ABLATION",
        "evaluator_version": EH.VERSION,
        "evaluator_hashes": EH.implementation_hashes(),
        "seed": SEED,
        "budget": BUDGET,
        "excl_px": EXCL_PX,
        "withheld_positive_pixels": int(hidden.sum()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "arms": arm_details,
        "pooled_summary": summary,
    }
    (ROOT / "evidence" / "h77e_exp1_ablation.json").write_text(json.dumps(payload, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "h77e_exp1_ablation.json").write_text(json.dumps(payload, indent=2) + "\n")
    log(f"E1 completed cleanly in {time.time() - t0:.1f}s!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
