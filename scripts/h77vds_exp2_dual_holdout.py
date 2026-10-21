#!/usr/bin/env python3
"""Experiment 2 of 3 (`Lane H77E`): Dual-Seed Spatial Quadrant Holdout (`Seed 30` + Lockbox `Seed 20`).

Two-phase memory-disjoint execution keeping peak RSS < 1.30 GiB on 3.8 GiB zero-swap containers:
  - Phase 1: Load scaled `X16` (`966 MB`), run all 106 single-feature leakage canaries on BOTH seeds,
    fit strictly leave-one-quadrant-out (LOQO) ensembles on Seed 30 and untouched lockbox Seed 20,
    save OOF surfaces to `.cache/h77e/`, and `del X16; gc.collect()` (`966 MB` freed!).
  - Phase 2: Open cached bit-packed `788`-sparse-witness 3-px halo matrix (`512 MB`) and run unified
    Dinkelbach submodular + `overlap_cap = 0.650` allocation & `vendor/gems57/evaluate_holdout.py`
    (`gems57-pooled-hide-v2`, alpha=0.2, beta=0.8, 300 m kernel, 2,000 paired block bootstrap draws).
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GDAL_CACHEMAX", "64")

import numpy as np
import rasterio
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH, grid as G, holdout as HO  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems77vds import dinkelbach as DK, features as FE, models as MO  # noqa: E402


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def collect_all_witness_paths(root: Path) -> list[str]:
    found: set[str] = set()
    for pat in (
        str(root / ".cache" / "registry" / "*.tif"),
        str(root / ".cache" / "siblings" / "*.tif"),
        str(root / ".cache" / "data" / "registry" / "rasters" / "*.tif"),
        str(root / "submissions" / "*.tif"),
    ):
        for p in glob.glob(pat):
            if "h77e-" not in Path(p).name.lower():
                found.add(str(Path(p).resolve()))
    return sorted(found)


def main() -> int:
    t0 = time.time()
    cache_dir = ROOT / ".cache" / "h77e"
    bin_path, aux = FE.build_static_fp_memmap(ROOT, force=False)
    n_fp = int(aux["n_fp"])

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

    # ------------------------------------------------------------------
    # PHASE 1: Fit LOQO models on Seed 30 & Lockbox Seed 20 (`X16` only in RAM!)
    # ------------------------------------------------------------------
    log("Phase 1: Loading scaled float16 98-channel feature matrix (`966 MB`)...")
    X16 = FE.load_scaled_f16_matrix(bin_path, n_fp)
    assert int(np.isinf(X16).sum()) == 0, "Unexpected Inf in X16"
    log(f"Loaded X16 in {time.time() - t0:.1f}s")

    canary_summary = {}
    for seed in (30, 20):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden = folds[0]["hidden_all"]
        visible = folds[0]["visible"]
        log(f"=== Phase 1 SEED {seed}: withheld={int(hidden.sum())}, visible={int(visible.sum())} ===")

        vis_fp = FE.compute_visible_structural_fp(visible, foot, aux["u2_dir"], aux["v2_dir"], aux["geod_rank"])

        rng = np.random.default_rng(1000 + seed)
        pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
        dist_cat = ndi.distance_transform_edt(~cat).astype(np.float32)
        neg_pool_fp = fp_lut[np.flatnonzero(foot & (~cat) & (dist_cat > 3.0))]
        neg_fp = rng.choice(neg_pool_fp, size=min(200_000, neg_pool_fp.size), replace=False)
        can_fp = np.concatenate([pos_fp, neg_fp])
        y_can = np.concatenate([np.ones(pos_fp.size, dtype=np.int8), np.zeros(neg_fp.size, dtype=np.int8)])

        Xc = X16[can_fp].astype(np.float32)
        Vc = vis_fp[can_fp]
        max_sym, max_nm = 0.5, ""
        for j, nm in enumerate(FE.STATIC_CHANNEL_NAMES):
            v = Xc[:, j]
            m = np.isfinite(v)
            if m.sum() > 100 and len(np.unique(y_can[m])) == 2:
                raw_a = float(roc_auc_score(y_can[m], v[m]))
                sa = max(raw_a, 1.0 - raw_a)
                if sa > max_sym:
                    max_sym, max_nm = sa, nm
        for j, nm in enumerate(FE.VISIBLE_CHANNEL_NAMES):
            raw_a = float(roc_auc_score(y_can, Vc[:, j]))
            sa = max(raw_a, 1.0 - raw_a)
            if sa > max_sym:
                max_sym, max_nm = sa, nm
        del Xc, Vc, dist_cat
        gc.collect()
        log(f"[SEED {seed}] Canary gate PASSED across all 106 channels: max symmetric AUC = {max_sym:.4f} (`{max_nm}`)")
        if max_sym > 0.90:
            raise RuntimeError(f"HARD STOP — LEAKAGE CANARY > 0.90 ON SEED {seed}: {max_nm}={max_sym}")
        canary_summary[seed] = (round(max_sym, 6), max_nm)

        surf_cache = cache_dir / f"surfs_seed{seed}.npz"
        if surf_cache.is_file():
            log(f"[SEED {seed}] Found existing cached OOF surfaces `{surf_cache.name}` — reusing!")
            del vis_fp
            MO.free_ram()
        else:
            t_fit = time.time()
            surfs = MO.fit_predict_h77e_full(X16, vis_fp, folds, visible, foot, slope_2d, aux, fp_lut, seed=7700 + seed)
            np.savez(surf_cache, **{k: v[foot].astype(np.float32) for k, v in surfs.items()})
            del surfs, vis_fp
            rss = MO.free_ram()
            log(f"[SEED {seed}] Saved 6 OOF surfaces in {time.time() - t_fit:.1f}s (VmRSS={rss:.0f} MiB)")

    # Free `X16` (`966 MB`) completely before opening `packed_halos_fp`!
    del X16, slope_2d, aux
    gc.collect()
    log("Phase 1 complete — freed `X16` (`966 MB`). Starting Phase 2 Dinkelbach allocation & scoring...")

    # ------------------------------------------------------------------
    # PHASE 2: Unified Dinkelbach + 788-witness `overlap_cap=0.650` allocation & bootstrap scoring
    # ------------------------------------------------------------------
    P = MO.FROZEN_H77E_PARAMS
    witness_paths = collect_all_witness_paths(ROOT)
    packed_halos_fp, sparse_names, dense_witnesses = DK.build_packed_sparse_halos_fp(
        witness_paths,
        foot,
        cache_dir=cache_dir,
        max_sparse_positives=P["max_sparse_positives"],
    )
    n_sparse = len(sparse_names)
    log(f"Opened cached packed sparse halos: n_sparse={n_sparse}, n_dense={len(dense_witnesses)}")

    gems32_path = ROOT / ".cache" / "registry" / "17a76895f68174cc93f3cb1597d25686f6d6bc67.tif"
    gems32_2d = None
    if gems32_path.is_file():
        with rasterio.open(gems32_path) as ds:
            gems32_2d = np.nan_to_num(ds.read(1).astype(np.float32), nan=0.0)

    out = {
        "evidence_class": "HOLDOUT-DTI",
        "experiment": "H77E-E2-DUAL-SEED-QUADRANT-HOLDOUT",
        "evaluator_version": EH.VERSION,
        "evaluator_hashes": EH.implementation_hashes(),
        "frozen_params": P,
        "sparse_witness_count": n_sparse,
        "dense_witnesses": dense_witnesses,
        "seeds": {},
    }

    for seed in (30, 20):
        t_s2 = time.time()
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden = folds[0]["hidden_all"]
        visible = folds[0]["visible"]
        dist_vis = ndi.distance_transform_edt(~visible).astype(np.float32)

        npz = np.load(cache_dir / f"surfs_seed{seed}.npz")
        surfs = {}
        for k in npz.files:
            arr = np.zeros((H, W), dtype=np.float32)
            arr[foot] = npz[k]
            surfs[k] = arr
        del npz

        excl = P["excl_px"]
        allowed_folds = [f["region"] & foot & (~visible) & (dist_vis > excl) for f in folds]
        allow_union = np.zeros((H, W), dtype=bool)
        for a in allowed_folds:
            allow_union |= a
        tot_allow = int(allow_union.sum())

        def emit_dinkelbach_for_budget(surf: np.ndarray, total_b: int, use_regcap: bool) -> tuple[np.ndarray, dict]:
            # Identical single-heap global allocation across the union domain (matching E3 production 1-to-1!)
            return DK.allocate_dinkelbach_submodular_dti(
                surf,
                allow_union,
                budget=total_b,
                min_sep_px=P["dinkelbach_min_sep_px"],
                pi_true=P["dinkelbach_pi_true"],
                temperature=P["dinkelbach_temperature"],
                fp_lambda=P["dinkelbach_fp_lambda"],
                packed_halos_fp=packed_halos_fp if use_regcap else None,
                fp_lut=fp_lut if use_regcap else None,
                n_sparse_rasters=n_sparse if use_regcap else 0,
                overlap_cap=P["sparse_overlap_cap"],
            )

        def score_dots(dots: np.ndarray):
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

        def stratified_recall(dots: np.ndarray) -> dict[str, float]:
            dist_pred = ndi.distance_transform_edt(~dots).astype(np.float32)
            cov = np.maximum(1.0 - dist_pred / 3.0, 0.0)
            near_m = hidden & foot & (dist_vis <= 12.0)
            far_m = hidden & foot & (dist_vis > 12.0)
            return {
                "n_near_le_1p2km": int(near_m.sum()),
                "n_detached_gt_1p2km": int(far_m.sum()),
                "weighted_recall_near_le_1p2km": round(float(cov[near_m].mean()), 6) if near_m.any() else 0.0,
                "weighted_recall_detached_gt_1p2km": round(float(cov[far_m].mean()), 6) if far_m.any() else 0.0,
                "weighted_recall_all": round(float(cov[hidden & foot].mean()), 6),
            }

        budgets_out = {}
        for B in (37_654, 44_090):
            terms_by_arm = {}
            arm_details = {}
            dinkel_opt_b = 0
            for arm_name, skey, rcap in (
                ("H77E_primary_regcap", "h77e_primary", True),
                ("H77E_unconstrained", "h77e_primary", False),
                ("H77E_arith_blend", "h77e_arith", False),
                ("H77E_only_hgb", "only_hgb", False),
                ("H77E_only_lgb", "only_lgb", False),
                ("H77E_phys98_only", "phys_ens", False),
            ):
                d, st = emit_dinkelbach_for_budget(surfs[skey], B, use_regcap=rcap)
                if arm_name == "H77E_primary_regcap":
                    dinkel_opt_b = int(st["dinkelbach_opt_budget"])
                per, t = score_dots(d)
                terms_by_arm[arm_name] = t
                rec = stratified_recall(d)
                arm_details[arm_name] = {
                    "dots": int(d.sum()),
                    "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0))),
                    "max_sparse_overlap": st["max_sparse_overlap_achieved"],
                    "sparse_cap_skips": st["sparse_overlap_cap_skips"],
                    "per_fold": per,
                    "stratified_recall": rec,
                }
                log(
                    f"[SEED {seed} B={B}] {arm_name:22s}: HOLDOUT-DTI = {arm_details[arm_name]['pooled_holdout_dti']:.6f} "
                    f"(near={rec['weighted_recall_near_le_1p2km']:.4f}, detached>1.2km={rec['weighted_recall_detached_gt_1p2km']:.4f}, "
                    f"max_ov={st['max_sparse_overlap_achieved']:.4f}, skips={st['sparse_overlap_cap_skips']})"
                )

            d_gr = np.zeros((H, W), dtype=bool)
            for a in allowed_folds:
                b = int(round(B * int(a.sum()) / max(tot_allow, 1)))
                if b > 0:
                    d_gr |= emit77.emit_dots(surfs["h77e_primary"], a, b)
            per_gr, t_gr = score_dots(d_gr)
            terms_by_arm["H77E_greedy_alloc"] = t_gr
            arm_details["H77E_greedy_alloc"] = {
                "dots": int(d_gr.sum()),
                "pooled_holdout_dti": float(EH.from_terms(t_gr.sum(axis=0))),
                "per_fold": per_gr,
                "stratified_recall": stratified_recall(d_gr),
            }
            log(f"[SEED {seed} B={B}] {'H77E_greedy_alloc':22s}: HOLDOUT-DTI = {arm_details['H77E_greedy_alloc']['pooled_holdout_dti']:.6f}")

            if gems32_2d is not None:
                per_g32, t_g32 = score_dots(gems32_2d > 0.5)
                terms_by_arm["GEMSDOE32_actual"] = t_g32
                rec_g32 = stratified_recall(gems32_2d > 0.5)
                arm_details["GEMSDOE32_actual"] = {
                    "dots": int(sum(v["emitted"] for v in per_g32.values())),
                    "pooled_holdout_dti": float(EH.from_terms(t_g32.sum(axis=0))),
                    "per_fold": per_g32,
                    "stratified_recall": rec_g32,
                }
                log(
                    f"[SEED {seed} B={B}] {'GEMSDOE32_actual':22s}: HOLDOUT-DTI = {arm_details['GEMSDOE32_actual']['pooled_holdout_dti']:.6f} "
                    f"(near={rec_g32['weighted_recall_near_le_1p2km']:.4f}, detached>1.2km={rec_g32['weighted_recall_detached_gt_1p2km']:.4f})"
                )

            rng_r = np.random.default_rng(8800 + seed + B)
            d_rand = np.zeros((H, W), dtype=bool)
            for a in allowed_folds:
                b = int(round(B * int(a.sum()) / max(tot_allow, 1)))
                d_rand.ravel()[rng_r.choice(np.flatnonzero(a), b, replace=False)] = True
            per_r, t_r = score_dots(d_rand)
            terms_by_arm["C0_random"] = t_r
            arm_details["C0_random"] = {
                "dots": int(d_rand.sum()),
                "pooled_holdout_dti": float(EH.from_terms(t_r.sum(axis=0))),
                "per_fold": per_r,
                "stratified_recall": stratified_recall(d_rand),
            }

            summary = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810, candidate="H77E_primary_regcap")
            budgets_out[str(B)] = {
                "budget": B,
                "dinkelbach_opt_budget_sum": dinkel_opt_b,
                "arms": arm_details,
                "pooled_summary": summary,
            }

        del surfs
        gc.collect()
        ms, mn = canary_summary[seed]
        out["seeds"][str(seed)] = {
            "seed": seed,
            "withheld_positive_pixels": int(hidden.sum()),
            "max_canary_symmetric_auc": ms,
            "max_canary_channel": mn,
            "phase2_seconds": round(time.time() - t_s2, 2),
            "receipts": [f["receipt"] for f in folds],
            "budgets": budgets_out,
        }

    out["total_withheld_positive_pixels"] = sum(v["withheld_positive_pixels"] for v in out["seeds"].values())
    out["elapsed_seconds"] = round(time.time() - t0, 2)
    (ROOT / "evidence" / "h77e_exp2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "h77e_exp2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"E2 completed cleanly in {time.time() - t0:.1f}s across {out['total_withheld_positive_pixels']} withheld positives!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
