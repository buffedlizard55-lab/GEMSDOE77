#!/usr/bin/env python3
"""Experiment 2 (`H77V-Exp2`): Spatially-Blocked Holdout Evaluation on `vendor/gems57/evaluate_holdout.py`.

Evaluates the 114-channel H77V Geothermal Vent & Breach Ensemble on the official shared
`gems57-pooled-hide-v2` holdout (`alpha=0.2`, `beta=0.8`, `R=300 m` triangular kernel) across
both Seed 20 (`11,321` withheld positive pixels) and Seed 30 (`11,342` withheld positive pixels),
comparing against `H33` fold-fair (`0.0961`), `H77D` (`0.1501` Seed 30 / `0.1403` Seed 20),
and `H77-Step` (`0.1527` Seed 20).

Strict Holdout Integrity:
  Every feature (`vis_*`) and every fold-fair reference halo is clipped strictly against
  `fold['visible']` (`d_vis > 2.0 px`), NEVER against full `existing_faults.tif` or any live
  submission raster whose collar mask knows where withheld `truth` segments sit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout as eval_mod  # noqa: E402
from gems57 import grid as grid_mod  # noqa: E402
from gems57 import holdout as holdout_mod  # noqa: E402
from gems77_step.emit import peak_nms_emit  # noqa: E402
from gems77v.emit import (  # noqa: E402
    build_hydrothermal_crest_2d,
    flank_zeroed_unique_submodular_sequence,
)
from gems77v.features import (  # noqa: E402
    build_h77v_static_features,
    compute_h77v_vlbco_mechanics_fp,
    pack_combined_ram_float16,
)
from gems77v.models import fit_h77v_vent_ensemble  # noqa: E402

DATA_DIR = ROOT / ".cache" / "data"
EXT_DIR = DATA_DIR / "data" / "external"
REG_CANONICAL_DIR = DATA_DIR / "registry" / "rasters"
BASE_MMAP = ROOT / ".cache" / "h77_features_fp.f32"
H77V_MMAP = ROOT / ".cache" / "h77v_static_12.f32"

H33_TIF = REG_CANONICAL_DIR / "GEMSDOE32__h33-2-b2-zeros.tif"
D28_TIF = REG_CANONICAL_DIR / "GEMSDOE25__dotted-h19-5-d2-8.tif"
H19_TIF = REG_CANONICAL_DIR / "19GEMSDOE__h19-5-powerlaw-budget-multiline.tif"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _make_quadrant_masks(cat: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fy, fx = np.nonzero(cat & fp)
    rmed = int(np.median(fy))
    cmed = int(np.median(fx))
    rr, cc = np.indices(fp.shape)
    quad = ((rr >= rmed).astype(np.int8) * 2 + (cc >= cmed).astype(np.int8))
    guard = (np.abs(rr - rmed) <= 35) | (np.abs(cc - cmed) <= 35)
    return quad, guard


def _sum_fold_terms(pred_2d: np.ndarray, folds: list[dict], fp: np.ndarray) -> np.ndarray:
    return np.sum([eval_mod.evaluate(pred_2d, f, fp, block_side=200)[1] for f in folds], axis=0)


def _build_foldfair_halos(d_vis_2d: np.ndarray, fp: np.ndarray) -> list[np.ndarray]:
    """Build fold-fair 3-px dilated halos using ONLY `d_vis_2d > 2.0` (zero `truth` collar leakage)."""
    yy, xx = np.mgrid[-3:4, -3:4]
    disk3 = (yy * yy + xx * xx) <= 9
    halos = []
    for p in (D28_TIF, H19_TIF):
        if p.is_file():
            with rasterio.open(p) as ds:
                arr = np.nan_to_num(ds.read(1), nan=0.0)
            dots = (arr > 0.5) & fp & (d_vis_2d > 2.0)
            if dots.any():
                halos.append(binary_dilation(dots, structure=disk3).ravel())
    return halos


def _run_single_seed(seed: int) -> dict:
    t0 = time.perf_counter()
    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    fp_idx = np.flatnonzero(fp)
    row_fp = (fp_idx // g.shape[1]).astype(np.int32)
    col_fp = (fp_idx % g.shape[1]).astype(np.int32)

    with rasterio.open(DATA_DIR / "existing_faults.tif") as ds:
        cat = (ds.read(1) > 0) & fp
    quad_2d, guard_2d = _make_quadrant_masks(cat, fp)
    quad_fp = quad_2d.ravel()[fp_idx]
    guard_fp = guard_2d.ravel()[fp_idx]
    del cat, quad_2d, guard_2d

    (
        X_base,
        base_names,
        X_v,
        v_names,
        has_lidar_fp,
        hydro_imp_fp,
        crest_fp,
        vent_pot_fp,
        _,
    ) = build_h77v_static_features(BASE_MMAP, H77V_MMAP, fp, g.transform, DATA_DIR, EXT_DIR)
    upflow_scarp_fp = np.asarray(X_v[v_names.index("h77v_upflow_scarp_coincidence"), :], dtype=np.float32)
    crest_v_2d = build_hydrothermal_crest_2d(crest_fp, vent_pot_fp, upflow_scarp_fp, fp)
    del crest_fp

    folds, _ = holdout_mod.buffered_component_draw(
        g, seed=seed, hide_frac=0.20, buffer_px=3, boundary_px=12
    )
    n_withheld = sum(int(f["truth"].sum()) for f in folds)

    vis_2d = np.ones(g.shape, dtype=bool)
    for f in folds:
        vis_2d &= f["visible"]
    vis_2d &= fp
    assert not np.any(vis_2d & np.logical_or.reduce([f["truth"] for f in folds]))

    d_vis_2d = distance_transform_edt(~vis_2d).astype(np.float32)
    d_vis_fp = d_vis_2d.ravel()[fp_idx]
    allow_hide = fp & (d_vis_2d > 2.0)

    oof_cache = ROOT / ".cache" / f"h77v_oof_seed{seed}.npy"
    if oof_cache.is_file():
        print(f"  [Seed {seed}] Reloading cached 114-channel H77V OOF surface...")
        p_oof_fp = np.load(oof_cache)
        del hydro_imp_fp, vent_pot_fp, upflow_scarp_fp, vis_2d, d_vis_fp, quad_fp, guard_fp, has_lidar_fp, row_fp, col_fp, X_base, X_v
        gc.collect()
    else:
        print(f"  [Seed {seed}] Computing 6 fold-safe V-LBCO mechanics channels...")
        Xm_train, Xm_infer, mech_names = compute_h77v_vlbco_mechanics_fp(
            vis_2d, fp, hydro_imp_fp, vent_pot_fp, upflow_scarp_fp, n_buckets=5, seed=7700 + seed
        )
        del hydro_imp_fp, vent_pot_fp, upflow_scarp_fp
        gc.collect()

        X16 = pack_combined_ram_float16(X_base, X_v)
        all_static_names = base_names + v_names
        del X_base, X_v
        gc.collect()
        all_names = all_static_names + mech_names

        print(f"  [Seed {seed}] Fitting 114-channel H77V OOF ensemble...")
        p_oof_fp = fit_h77v_vent_ensemble(
            X16,
            Xm_train,
            Xm_infer,
            all_names,
            vis_2d.ravel()[fp_idx],
            d_vis_fp,
            quad_fp,
            guard_fp,
            has_lidar_fp,
            row_fp,
            col_fp,
            allow_intra_province=True,
            seed_base=77200 + seed * 10,
        )
        np.save(oof_cache, p_oof_fp)
        del Xm_train, Xm_infer, X16, vis_2d, d_vis_fp, quad_fp, guard_fp, has_lidar_fp, row_fp, col_fp
        gc.collect()

    p_oof_2d = np.zeros(g.shape, dtype=np.float32)
    p_oof_2d.ravel()[fp_idx] = p_oof_fp
    del p_oof_fp

    print(f"  [Seed {seed}] Evaluating NMS and 300m Submodular allocations...")
    dots_nms = peak_nms_emit(p_oof_2d, allow_hide, n_dots=37654)
    terms_nms = _sum_fold_terms(dots_nms, folds, fp)
    del dots_nms

    candidate_budgets = [34625, 36250, 37654, 39000]
    seq_uncapped = flank_zeroed_unique_submodular_sequence(
        p_oof_2d, crest_v_2d, allow_hide, max_dots=max(candidate_budgets)
    )

    def _from_seq(seq: np.ndarray, k_dots: int) -> np.ndarray:
        arr = np.zeros(g.shape, dtype=np.float32)
        arr.ravel()[seq[:k_dots]] = 1.0
        return arr

    fold_terms_by_b = {}
    for b in candidate_budgets:
        arr_b = _from_seq(seq_uncapped, b)
        fold_terms_by_b[b] = [eval_mod.evaluate(arr_b, f, fp, block_side=200)[1] for f in folds]
        del arr_b

    oof_budgets = []
    oof_terms = []
    for k in range(4):
        best_b = max(
            candidate_budgets,
            key=lambda b: float(eval_mod.from_terms(sum(fold_terms_by_b[b][j].sum(axis=0) for j in range(4) if j != k))),
        )
        oof_budgets.append(best_b)
        oof_terms.append(fold_terms_by_b[best_b][k])

    lofo_budget = int(round(float(np.mean(oof_budgets))))
    dots_submod_37654 = _from_seq(seq_uncapped, 37654)
    terms_submod_37654 = _sum_fold_terms(dots_submod_37654, folds, fp)
    del dots_submod_37654

    dots_submod_36250 = _from_seq(seq_uncapped, 36250)
    terms_submod_36250 = _sum_fold_terms(dots_submod_36250, folds, fp)
    del dots_submod_36250

    dots_submod_lofo = _from_seq(seq_uncapped, lofo_budget)
    terms_submod_lofo = _sum_fold_terms(dots_submod_lofo, folds, fp)
    del dots_submod_lofo, seq_uncapped
    gc.collect()

    foldfair_halos = _build_foldfair_halos(d_vis_2d, fp)
    max_per_halo = int(np.floor(0.645 * 36250))
    seq_unique = flank_zeroed_unique_submodular_sequence(
        p_oof_2d,
        crest_v_2d,
        allow_hide,
        max_dots=37654,
        halo_flat_masks=foldfair_halos,
        max_per_halo=max_per_halo,
    )
    del p_oof_2d, crest_v_2d, foldfair_halos
    gc.collect()

    dots_unique_36250 = _from_seq(seq_unique, 36250)
    terms_unique_36250 = _sum_fold_terms(dots_unique_36250, folds, fp)
    del dots_unique_36250

    dots_unique_37654 = _from_seq(seq_unique, 37654)
    terms_unique_37654 = _sum_fold_terms(dots_unique_37654, folds, fp)
    del dots_unique_37654, seq_unique

    with rasterio.open(H33_TIF) as ds:
        h33_static = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)
    terms_h33_static = _sum_fold_terms(h33_static, folds, fp)
    del h33_static

    with rasterio.open(D28_TIF) as ds:
        d28_raw = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)
    h33_foldfair = np.where(d_vis_2d > 2.0, d28_raw, 0.0).astype(np.float32)
    del d28_raw, d_vis_2d
    terms_h33_foldfair = _sum_fold_terms(h33_foldfair, folds, fp)
    del h33_foldfair

    summary = eval_mod.pooled_summary(
        {
            "h77v_submod300_37654": terms_submod_37654,
            "h77v_submod300_36250": terms_submod_36250,
            "h77v_submod300_lofo": terms_submod_lofo,
            "h77v_submod300_lofo_strict_oof": np.sum(np.stack(oof_terms, axis=0), axis=0),
            "h77v_foldfair_unique_36250": terms_unique_36250,
            "h77v_foldfair_unique_37654": terms_unique_37654,
            "h77v_nms_37654": terms_nms,
            "h33_fold_fair_reclipped": terms_h33_foldfair,
            "h33_live_static": terms_h33_static,
        },
        draws=1000,
        seed=20261021 + seed,
        candidate="h77v_submod300_37654",
    )
    summary["seed"] = seed
    summary["withheld_positive_pixels"] = n_withheld
    summary["lofo_selected_budgets_per_fold"] = oof_budgets
    summary["lofo_mean_budget"] = lofo_budget
    summary["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    out_path = ROOT / ".cache" / f"h77v_exp2_seed{seed}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    sc = summary["scores"]
    print(
        f"  [Seed {seed} DONE in {summary['elapsed_seconds']}s] "
        f"H77V-37654={sc['h77v_submod300_37654']['dti']:.6f} (CI {sc['h77v_submod300_37654']['ci95']}) | "
        f"H77V-36250={sc['h77v_submod300_36250']['dti']:.6f} | "
        f"H77V-LOFO={sc['h77v_submod300_lofo']['dti']:.6f} | "
        f"H33-FoldFair={sc['h33_fold_fair_reclipped']['dti']:.6f}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.seed in (20, 30):
        _run_single_seed(args.seed)
        return 0

    t0 = time.perf_counter()
    for s in (20, 30):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--seed", str(s)], check=True)

    s20 = json.loads((ROOT / ".cache" / "h77v_exp2_seed20.json").read_text())
    s30 = json.loads((ROOT / ".cache" / "h77v_exp2_seed30.json").read_text())
    elapsed = round(time.perf_counter() - t0, 2)

    dti_s20 = s20["scores"]["h77v_submod300_37654"]["dti"]
    dti_s30 = s30["scores"]["h77v_submod300_37654"]["dti"]
    two_seed_mean = round(0.5 * (dti_s20 + dti_s30), 6)

    out = {
        "experiment": "H77V-Exp2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": eval_mod.VERSION,
        "implementation_sha256": {
            "evaluate_holdout.py": _sha256(ROOT / "vendor" / "gems57" / "evaluate_holdout.py"),
            "holdout.py": _sha256(ROOT / "vendor" / "gems57" / "holdout.py"),
            "metric.py": _sha256(ROOT / "vendor" / "gems57" / "metric.py"),
            "spatial.py": _sha256(ROOT / "vendor" / "gems57" / "spatial.py"),
        },
        "alpha": 0.2,
        "beta": 0.8,
        "kernel_radius_m": 300.0,
        "elapsed_seconds": elapsed,
        "two_seed_mean_dti_37654": two_seed_mean,
        "prior_repo_holdout_benchmarks": {
            "H77_Step_seed20_dti": 0.152718,
            "H77_Step_seed20_matched_37654_dti": 0.151581,
            "H77D_seed30_dti": 0.150108,
            "H77D_seed20_dti": 0.140298,
            "H77D_two_seed_mean_dti": 0.145203,
            "H33_fold_fair_seed20_dti": 0.096086,
            "H33_fold_fair_seed30_dti": 0.098898,
        },
        "seed20": s20,
        "seed30": s30,
    }
    for dst in (ROOT / "evidence" / "h77v_exp2_holdout.json", ROOT / "docs" / "evidence" / "h77v_exp2_holdout.json"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / ".cache" / "h77v_exp2_meta.json").write_text(
        json.dumps({"live_budget": 36250}, indent=2) + "\n"
    )
    print(f"[Exp 2/3] Completed both seeds in {elapsed:.1f}s | Two-Seed Mean HOLDOUT-DTI = {two_seed_mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
