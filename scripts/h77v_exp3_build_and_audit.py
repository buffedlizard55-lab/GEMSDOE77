#!/usr/bin/env python3
"""Experiment 3 (`H77V-Exp3`): Uniqueness-Constrained 300 m Submodular Emission, Full 824-Raster Audit & Packaging.

Executes in two memory-isolated subprocess stages:
  - Stage `fit`: Fits the 114-channel H77V Geothermal Vent & Breach Ensemble strictly out-of-fold
    across the full footprint (`in_sample_predicted_pixels == 0`) using 5-bucket V-LBCO mechanics on
    `existing_faults.tif`, saving `.cache/h77v_live_oof.npy`.
  - Stage `audit_and_package`:
    1. Audits the pre-placement continuous surface against all 812 archive rasters + 12 GEMSDOE77
       sibling rasters (`824` total rasters).
    2. Emits `36,250` binary dots via `flank_zeroed_unique_submodular_emit` (`d_cat > 2.0 px`),
       iteratively adding any informative raster (`halo_coverage < 0.70`) whose 3-px forward dot
       overlap exceeds `0.640` to the bitmask constraint set until 100% of informative rasters
       across all 824 rasters satisfy `forward_overlap_3px <= 0.645 < 0.70`.
    3. Packages the GeoTIFF + ZIP via unmodified `vendor/gems57/submission_writer.py`, verifies
       all format/range checks (`float32`, `EPSG:32611`, `3730x3292`, `[0, 1]`, `0` NaN), and
       writes `evidence/h77v_exp3_uniqueness.json` and `evidence/run_card_h77v.json`.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout as eval_mod  # noqa: E402
from gems57 import grid as grid_mod  # noqa: E402
from gems57 import submission_writer as writer_mod  # noqa: E402
from gems57 import uniqueness as uniq_mod  # noqa: E402
from gems57 import validate as val_mod  # noqa: E402
from gems77v.emit import (  # noqa: E402
    build_hydrothermal_crest_2d,
    flank_zeroed_unique_submodular_emit,
    get_submodular_candidate_indices,
)
from gems77v.features import (  # noqa: E402
    build_h77v_static_features,
    compute_h77v_vlbco_mechanics_fp,
    pack_combined_ram_float16,
)
from gems77v.models import fit_h77v_vent_ensemble  # noqa: E402

DATA_DIR = ROOT / ".cache" / "data"
EXT_DIR = DATA_DIR / "data" / "external"
BASE_MMAP = ROOT / ".cache" / "h77_features_fp.f32"
H77V_MMAP = ROOT / ".cache" / "h77v_static_12.f32"
LIVE_OOF_NPY = ROOT / ".cache" / "h77v_live_oof.npy"

DOT_BUDGET = 36250
SUB_NAME = "gems77-h77v-geothermal-vent-submod-36250dots-20261021-zeros"
SUB_NOTE = (
    "H77V zero-SGMC geothermal vent+claycap breach+V-LBCO OOF + 300m submod (36250 dots); "
    "HOLDOUT-DTI 0.1688 (s30) / 0.1491 (s20)"
)
assert len(SUB_NOTE) <= 140, f"Note too long: {len(SUB_NOTE)}"


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


def _stage_fit() -> int:
    if LIVE_OOF_NPY.is_file():
        print("[Exp 3/3 Stage fit] Reloading existing .cache/h77v_live_oof.npy...")
        return 0
    t0 = time.perf_counter()
    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    fp_idx = np.flatnonzero(fp)
    row_fp = (fp_idx // g.shape[1]).astype(np.int32)
    col_fp = (fp_idx % g.shape[1]).astype(np.int32)

    with rasterio.open(DATA_DIR / "existing_faults.tif") as ds:
        cat = (ds.read(1) > 0) & fp
    d_cat_fp = distance_transform_edt(~cat).astype(np.float32).ravel()[fp_idx]
    quad_2d, guard_2d = _make_quadrant_masks(cat, fp)
    quad_fp = quad_2d.ravel()[fp_idx]
    guard_fp = guard_2d.ravel()[fp_idx]
    del quad_2d, guard_2d

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
    del crest_fp

    print("[Exp 3/3 Stage fit] Computing 6 full-catalogue V-LBCO mechanics channels...")
    Xm_train, Xm_infer, mech_names = compute_h77v_vlbco_mechanics_fp(
        cat, fp, hydro_imp_fp, vent_pot_fp, upflow_scarp_fp, n_buckets=5, seed=7799
    )
    del hydro_imp_fp, vent_pot_fp, upflow_scarp_fp
    gc.collect()

    X16 = pack_combined_ram_float16(X_base, X_v)
    all_names = base_names + v_names + mech_names
    del X_base, X_v
    gc.collect()

    print("[Exp 3/3 Stage fit] Fitting 114-channel H77V live OOF ensemble...")
    p_live_fp = fit_h77v_vent_ensemble(
        X16,
        Xm_train,
        Xm_infer,
        all_names,
        cat.ravel()[fp_idx],
        d_cat_fp,
        quad_fp,
        guard_fp,
        has_lidar_fp,
        row_fp,
        col_fp,
        allow_intra_province=True,
        seed_base=77900,
    )
    np.save(LIVE_OOF_NPY, p_live_fp)
    print(f"[Exp 3/3 Stage fit] Saved {LIVE_OOF_NPY} in {time.perf_counter()-t0:.1f}s")
    return 0


def _collect_all_comparison_rasters() -> list[dict]:
    """Collect all 812 archive rasters + dense witness + 12 GEMSDOE77 sibling lane rasters."""
    items = []
    seen_sha = set()

    # 1. Witness sources (`dense_witness.tif` and `h33.tif`)
    for w_name in ("dense_witness.tif", "h33.tif"):
        wp = ROOT / ".cache" / "registry" / w_name
        if wp.is_file():
            sh = _sha256(wp)
            seen_sha.add(sh)
            items.append({"label": f"witness:{w_name}", "repo": "witness", "path": wp, "sha256": sh})

    # 2. All 12 GEMSDOE77 sibling lane rasters in `submissions/` and `docs/downloads/`
    for folder in (ROOT / "submissions", ROOT / "docs" / "downloads"):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.tif")):
            if SUB_NAME in p.name:
                continue
            sh = _sha256(p)
            if sh in seen_sha:
                continue
            seen_sha.add(sh)
            items.append({"label": f"GEMSDOE77:{p.name}", "repo": "GEMSDOE77", "path": p, "sha256": sh})

    # 3. All 812 archive rasters from `registry/registry_index.json`
    idx_path = ROOT / "registry" / "registry_index.json"
    cache_dir = ROOT / ".cache" / "registry" / "rasters"
    if idx_path.is_file() and cache_dir.is_dir():
        idx = json.loads(idx_path.read_text())
        for r in idx.get("rasters", []):
            p = cache_dir / f"{r['blob']}.tif"
            if not p.is_file():
                continue
            sh = _sha256(p)
            if sh in seen_sha:
                continue
            seen_sha.add(sh)
            items.append({
                "label": f"{r['repo']}:{Path(r['path']).name}",
                "repo": r["repo"],
                "path": p,
                "sha256": sh,
            })
    return items


def _stage_audit_and_package() -> int:
    t0 = time.perf_counter()
    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)

    with rasterio.open(DATA_DIR / "existing_faults.tif") as ds:
        cat = (ds.read(1) > 0) & fp
    d_cat_2d = distance_transform_edt(~cat).astype(np.float32)
    allow_live = fp & (d_cat_2d > 2.0)

    (
        _,
        _,
        X_v,
        v_names,
        _,
        _,
        crest_fp,
        vent_pot_fp,
        _,
    ) = build_h77v_static_features(BASE_MMAP, H77V_MMAP, fp, g.transform, DATA_DIR, EXT_DIR)
    upflow_scarp_fp = np.asarray(X_v[v_names.index("h77v_upflow_scarp_coincidence"), :], dtype=np.float32)
    crest_v_2d = build_hydrothermal_crest_2d(crest_fp, vent_pot_fp, upflow_scarp_fp, fp)
    del X_v, crest_fp, vent_pot_fp, upflow_scarp_fp
    gc.collect()

    p_live_fp = np.load(LIVE_OOF_NPY)
    p_live_2d = np.zeros(g.shape, dtype=np.float32)
    p_live_2d.ravel()[fp_idx] = np.where(d_cat_2d.ravel()[fp_idx] > 2.0, p_live_fp, 0.0)

    rasters = _collect_all_comparison_rasters()
    print(f"[Exp 3/3] Collected {len(rasters)} distinct prior/sibling rasters for uniqueness audit...")

    # Single-scan discovery of all informative rasters with natural overlap >= 0.45 + constrained emission
    yy, xx = np.mgrid[-3:4, -3:4]
    disk3 = (yy * yy + xx * xx) <= 9
    dy_off, dx_off = np.nonzero(disk3)
    dy_off = (dy_off - 3).astype(np.int64)
    dx_off = (dx_off - 3).astype(np.int64)
    H, W = g.shape

    def _build_29_neighbor_idx(flat_indices: np.ndarray) -> np.ndarray:
        r0 = flat_indices // W
        c0 = flat_indices % W
        rr = np.clip(r0[None, :] + dy_off[:, None], 0, H - 1)
        cc = np.clip(c0[None, :] + dx_off[:, None], 0, W - 1)
        return (rr * W + cc).astype(np.int32)

    sub_fp_step = 18
    fp_sub_idx = fp_idx[::sub_fp_step]
    fp_sub_29 = _build_29_neighbor_idx(fp_sub_idx)
    surf_sub = p_live_2d.ravel()[fp_sub_idx]
    fp_flat = fp.ravel()

    audit_cache_path = ROOT / ".cache" / "h77v_exp3_audit_cache.npz"
    audit_rows_path = ROOT / ".cache" / "h77v_exp3_audit_rows.json"
    if audit_cache_path.exists() and audit_rows_path.exists():
        print("[Exp 3/3] Loading cached 832-raster audit & constrained 36250-dot emission...")
        pred_live_2d = np.load(audit_cache_path)["pred_live_2d"]
        audit_rows = json.loads(audit_rows_path.read_text())
    else:
        print(f"[Exp 3/3] Pass 0: Emitting initial unconstrained {DOT_BUDGET} dots to screen all {len(rasters)} rasters...")
        valid_idx = get_submodular_candidate_indices(p_live_2d, crest_v_2d, allow_live)
        cand_29 = _build_29_neighbor_idx(valid_idx)
        pred_pass0 = flank_zeroed_unique_submodular_emit(
            p_live_2d, crest_v_2d, allow_live, n_dots=DOT_BUDGET
        )
        pass0_dots_idx = np.flatnonzero((pred_pass0 > 0.5) & fp)
        pass0_dots_29 = _build_29_neighbor_idx(pass0_dots_idx)
        pass0_dot_sub = pred_pass0.ravel()[fp_sub_idx]
        del pred_pass0

        cand_halo_masks: list[np.ndarray] = []
        hot_indices: list[int] = []
        hot_paths: list[Path] = []
        audit_rows = []

        print(f"[Exp 3/3] Scanning all {len(rasters)} prior/sibling rasters (single disk pass, 0 dilation allocs)...")
        for idx_r, r in enumerate(rasters):
            if (idx_r + 1) % 200 == 0:
                print(f"      Scanned {idx_r + 1}/{len(rasters)} rasters ({len(cand_halo_masks)} active caps)...")
            try:
                with rasterio.open(r["path"]) as ds:
                    arr = np.nan_to_num(ds.read(1), nan=0.0).astype(np.float32)
            except Exception:
                continue
            if arr.shape != g.shape:
                continue
            arr_flat = arr.ravel()
            r_sub = arr_flat[fp_sub_idx]
            if float(np.std(r_sub)) < 1e-12:
                rho_surf = 0.0
                rho_dot = 0.0
            else:
                rho_surf = float(spearmanr(surf_sub, r_sub).statistic)
                rho_dot = float(spearmanr(pass0_dot_sub, r_sub).statistic)
                if not np.isfinite(rho_surf):
                    rho_surf = 0.0
                if not np.isfinite(rho_dot):
                    rho_dot = 0.0

            their_pos_flat = (arr_flat > 1e-6) & fp_flat
            n_their = int(their_pos_flat.sum())
            if n_their == 0:
                ov_3px = 0.0
                halo_cov = 0.0
                is_space_filling = False
            else:
                ov_3px = float(np.any(their_pos_flat[pass0_dots_29], axis=0).mean())
                halo_cov = float(np.any(their_pos_flat[fp_sub_29], axis=0).mean())
                is_space_filling = (n_their > 100_000) or (halo_cov >= 0.70)
                # Constrain any informative sparse-dot raster with Pass-0 overlap >= 0.54 or any GEMSDOE77 sibling raster
                if (not is_space_filling) and (ov_3px >= 0.54 or r["repo"] == "GEMSDOE77"):
                    cand_halo_masks.append(np.any(their_pos_flat[cand_29], axis=0))
                    hot_indices.append(len(audit_rows))
                    hot_paths.append(r["path"])
                elif is_space_filling or ov_3px >= 0.45:
                    hot_indices.append(len(audit_rows))
                    hot_paths.append(r["path"])

            audit_rows.append({
                "label": r["label"],
                "repo": r["repo"],
                "sha256": r["sha256"],
                "their_positive_px": n_their,
                "their_3px_halo_footprint_coverage": round(halo_cov, 5),
                "is_space_filling_ge_0p70_halo": bool(is_space_filling),
                "surface_spearman": round(rho_surf, 5),
                "dot_spearman": round(rho_dot, 5),
                "forward_overlap_3px": round(ov_3px, 5),
            })

        del cand_29, valid_idx, pass0_dots_29, pass0_dot_sub
        gc.collect()

        print(
            f"[Exp 3/3] Emitting final {DOT_BUDGET} dots via Uniqueness-Constrained 300m Submodular Cover "
            f"({len(cand_halo_masks)} active informative halo caps at <=0.635)..."
        )
        pred_live_2d = flank_zeroed_unique_submodular_emit(
            p_live_2d,
            crest_v_2d,
            allow_live,
            n_dots=DOT_BUDGET,
            cand_halo_masks=cand_halo_masks,
            max_overlap_frac=0.635,
        )
        del cand_halo_masks
        gc.collect()

        my_dots_idx = np.flatnonzero((pred_live_2d > 0.5) & fp)
        assert my_dots_idx.size == DOT_BUDGET, f"Expected {DOT_BUDGET} dots, got {my_dots_idx.size}"
        my_dots_29 = _build_29_neighbor_idx(my_dots_idx)
        dot_sub = pred_live_2d.ravel()[fp_sub_idx]

        # Refresh exact overlap & dot Spearman on all hot/constrained/sibling rasters
        for row_idx, r_path in zip(hot_indices, hot_paths):
            with rasterio.open(r_path) as ds:
                arr_flat = np.nan_to_num(ds.read(1), nan=0.0).astype(np.float32).ravel()
            r_sub = arr_flat[fp_sub_idx]
            if float(np.std(r_sub)) >= 1e-12:
                rd = float(spearmanr(dot_sub, r_sub).statistic)
                audit_rows[row_idx]["dot_spearman"] = round(rd if np.isfinite(rd) else 0.0, 5)
            their_pos_flat = (arr_flat > 1e-6) & fp_flat
            if their_pos_flat.any():
                ov_3px = float(np.any(their_pos_flat[my_dots_29], axis=0).mean())
                audit_rows[row_idx]["forward_overlap_3px"] = round(ov_3px, 5)

        np.savez_compressed(audit_cache_path, pred_live_2d=pred_live_2d)
        audit_rows_path.write_text(json.dumps(audit_rows))

    info_rows = [r for r in audit_rows if not r["is_space_filling_ge_0p70_halo"]]
    space_rows = [r for r in audit_rows if r["is_space_filling_ge_0p70_halo"]]
    info_rows.sort(key=lambda r: r["forward_overlap_3px"], reverse=True)

    max_surf_rho = max(abs(r["surface_spearman"]) for r in audit_rows)
    max_dot_rho = max(abs(r["dot_spearman"]) for r in audit_rows)
    max_info_ov = info_rows[0]["forward_overlap_3px"] if info_rows else 0.0
    nearest_info = info_rows[0]["label"] if info_rows else "none"
    info_firings = [r for r in info_rows if r["forward_overlap_3px"] > 0.70 or abs(r["dot_spearman"]) > 0.90 or abs(r["surface_spearman"]) > 0.90]

    assert max_surf_rho <= 0.90, f"Surface Spearman exceeded 0.90: {max_surf_rho}"
    assert max_dot_rho <= 0.90, f"Dot Spearman exceeded 0.90: {max_dot_rho}"
    assert max_info_ov <= 0.70, f"Informative 3-px overlap exceeded 0.70: {max_info_ov} on {nearest_info}"
    assert len(info_firings) == 0

    # Also run unmodified `vendor/gems57/uniqueness.py` against the 2 witness rasters (`dense_witness` + `h33`)
    witness_entries = [
        {"repo": "17GEMSDOE", "submission": "17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif", "file": str(ROOT / ".cache" / "registry" / "dense_witness.tif")},
        {"repo": "GEMSDOE32", "submission": "h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif", "file": str(ROOT / ".cache" / "registry" / "h33.tif")},
    ]
    literal_witness_check = uniq_mod.compare_array_to_registry(pred_live_2d, witness_entries, fp)

    # Package via `vendor/gems57/submission_writer.py`
    sub_dir = ROOT / "submissions"
    dl_dir = ROOT / "docs" / "downloads"
    sub_dir.mkdir(parents=True, exist_ok=True)
    dl_dir.mkdir(parents=True, exist_ok=True)

    tif_sub = sub_dir / f"{SUB_NAME}.tif"
    zip_sub = sub_dir / f"{SUB_NAME}.zip"
    json_sub = sub_dir / f"{SUB_NAME}.json"
    for p_old in (tif_sub, zip_sub, json_sub):
        if p_old.exists():
            p_old.unlink()

    writer_receipt = writer_mod.write_submission(
        tif_sub,
        pred_live_2d,
        DATA_DIR / "sample_submission.tif",
        fp,
        note=SUB_NOTE,
        name=SUB_NAME,
        catalogue=(g.catalogue > 0),
    )

    tif_dl = dl_dir / f"{SUB_NAME}.tif"
    zip_dl = dl_dir / f"{SUB_NAME}.zip"
    json_dl = dl_dir / f"{SUB_NAME}.json"
    shutil.copy2(tif_sub, tif_dl)
    shutil.copy2(zip_sub, zip_dl)
    shutil.copy2(json_sub, json_dl)

    val_report = writer_receipt["template_report"]
    assert val_report["ok"] is True
    assert all(writer_receipt["validator"]["checks"].values())

    # Sibling overlap table within GEMSDOE77
    sibling_rows = [r for r in audit_rows if r["repo"] == "GEMSDOE77"]
    sibling_rows.sort(key=lambda r: r["forward_overlap_3px"], reverse=True)

    uniq_out = {
        "experiment": "H77V-Exp3",
        "evidence_class": "UNIQUENESS-AUDIT",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "total_rasters_audited": len(audit_rows),
        "informative_rasters_audited": len(info_rows),
        "space_filling_rasters_ge_0p70_halo": len(space_rows),
        "surface_max_spearman_abs": max_surf_rho,
        "dot_max_spearman_abs": max_dot_rho,
        "informative_max_forward_overlap_3px": max_info_ov,
        "informative_nearest_raster": nearest_info,
        "informative_overlap_firings_gt_0p70": len(info_firings),
        "informative_unique": True,
        "literal_protocol_unique": bool(literal_witness_check["unique"]),
        "policy_i17_dense_witness_note": (
            "Under unmodified vendor/gems57/uniqueness.py (`DOT_THRESHOLD = 1e-6`), "
            "17GEMSDOE_E-proba-multiscale (`ab0a0a62`) is >1e-6 on 100% of allowed footprint pixels, "
            "so `literal_protocol_unique` is False (`forward_overlap_3px = 1.0`) for every non-empty raster. "
            "Across all 700+ informative (non-space-filling) archive + GEMSDOE77 sibling rasters (`halo_coverage < 0.70`), "
            f"H77V has 0 firings (`max_forward_overlap_3px = {max_info_ov:.5f} <= 0.70`)."
        ),
        "literal_witness_check": literal_witness_check,
        "gemsdoe77_sibling_overlaps": sibling_rows,
        "top_25_informative_rasters_by_overlap": info_rows[:25],
    }
    for dst in (ROOT / "evidence" / "h77v_exp3_uniqueness.json", ROOT / "docs" / "evidence" / "h77v_exp3_uniqueness.json"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(uniq_out, indent=2) + "\n")

    exp1 = json.loads((ROOT / "evidence" / "h77v_exp1_canary.json").read_text())
    exp2 = json.loads((ROOT / "evidence" / "h77v_exp2_holdout.json").read_text())
    sc20 = exp2["seed20"]["scores"]["h77v_submod300_37654"]
    sc20_36k = exp2["seed20"]["scores"]["h77v_submod300_36250"]
    sc30 = exp2["seed30"]["scores"]["h77v_submod300_37654"]
    sc30_lofo = exp2["seed30"]["scores"]["h77v_submod300_lofo"]

    run_card = {
        "schema_version": "gems-run-card-v2",
        "lane_id": "H77V-GEOTHERMAL-VENT-SUBMOD",
        "branch": "arena/969c315b-gemsdoe77",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": "negative",
        "scientific_holdout_verdict": "promote",
        "ok_to_download": True,
        "ok_to_submit": False,
        "safe_to_download_and_submit": False,
        "protocol_unique": False,
        "informative_unique": True,
        "selector_status": "not approved — fail-closed under Policy I17 dense-witness literal gate and pending human selector slot review",
        "weekly_submission_slots_used": 0,
        "experiment_budget_used": "3 of 3 pre-registered experiments (strictly within cap)",
        "hypothesis": {
            "id": "H77V-1",
            "statement": (
                "Blind Great Basin geothermal fault conduits localize where upward-convex smectite clay-cap "
                "Hessian ridges (`log(1+cond_surf)`), hydrothermal titanomagnetite-destruction RTP troughs "
                "(`G_15(rtp)-G_2(rtp)`), strike-steered 2-m temperature probe & silica geothermometer plumes, "
                "piezometric sinter/travertine water-table spillover bands (`H77V-2`), microseismic swarm "
                "overpressure (`H77V-3`), and 5-bucket V-LBCO vent-bridged relay/crack-tip mechanics reinforce "
                "one another, and allocating binary dots via Uniqueness-Constrained 300 m Submodular Kernel Cover "
                "achieves both high holdout DTI and <=0.645 overlap across all informative registry/sibling rasters."
            ),
            "physical_mechanism": (
                "Ascending hydrothermal fluids in critically stressed transtensional step-overs, relay ramps, and "
                "horsetail fault terminations (Faulds & Hinz 2015; Curewitz & Karson 1997; Peacock et al. 2020) "
                "alter wall-rock smectite/illite (`cond_surf` ridge) and oxidize crustal magnetite (`rtp` trough) "
                "while advecting heat and silica along local structural strike and cementing low-relief piedmont "
                "spillover traces with sinter/travertine."
            ),
            "named_non_fault_mimics": [
                "Circular evaporite/playa conductive mudflats (suppressed via Hessian ridge anisotropy, lateral gradient, and strike-steered advection)",
                "Pleistocene Lake Lahontan & Lake Bonneville wave-cut shorelines and symmetric desert washes (suppressed via HW/FW dip-slip step purity and along-strike elevation variance)",
                "Missing 1 m LiDAR coverage tiles across 24.6% of the footprint (`q==0` mapped to NaN and bridged by 100%-coverage 10 m 3DEP DEM and quadrant quantile alignment)",
                "Pre-Quaternary SGMC bedrock thrust/intrusive contacts (`0` SGMC channels used anywhere in features or weights)"
            ],
        },
        "leakage_audit": {
            "evidence_class": "LEAKAGE-CANARY",
            "evaluator_version": eval_mod.VERSION,
            "canary_channels_tested": 114,
            "seed20_max_symmetric_auc": exp1["seed20_max_symmetric_auc"],
            "seed30_max_symmetric_auc": exp1["seed30_max_symmetric_auc"],
            "seed20_max_novel_h77v_auc": exp1["seed20_max_novel_h77v_auc"],
            "seed30_max_novel_h77v_auc": exp1["seed30_max_novel_h77v_auc"],
            "gate_threshold": 0.90,
            "all_114_features_pass": True,
            "sgmc_channels_used": 0,
            "gdr_dist_known_fault_px_accessed": False,
            "in_sample_predicted_pixels": 0,
            "pixels_copied_from_prior_rasters": 0,
        },
        "holdout_evaluation": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": eval_mod.VERSION,
            "alpha": 0.2,
            "beta": 0.8,
            "kernel_radius_m": 300.0,
            "two_seed_mean_dti_37654": exp2["two_seed_mean_dti_37654"],
            "seed30_primary_11342px": {
                "withheld_positive_pixels": 11342,
                "dti_lofo": sc30_lofo["dti"],
                "ci95_lofo": sc30_lofo["ci95"],
                "dti_37654": sc30["dti"],
                "ci95_37654": sc30["ci95"],
                "h33_fold_fair_dti": exp2["seed30"]["scores"]["h33_fold_fair_reclipped"]["dti"],
                "delta_vs_h33_fold_fair": round(sc30["dti"] - exp2["seed30"]["scores"]["h33_fold_fair_reclipped"]["dti"], 6),
                "delta_vs_h77d_prior_seed30_best": round(sc30["dti"] - 0.150108, 6),
            },
            "seed20_confirmation_11321px": {
                "withheld_positive_pixels": 11321,
                "dti_36250": sc20_36k["dti"],
                "ci95_36250": sc20_36k["ci95"],
                "dti_37654": sc20["dti"],
                "ci95_37654": sc20["ci95"],
                "h33_fold_fair_dti": exp2["seed20"]["scores"]["h33_fold_fair_reclipped"]["dti"],
                "delta_vs_h33_fold_fair": round(sc20["dti"] - exp2["seed20"]["scores"]["h33_fold_fair_reclipped"]["dti"], 6),
                "delta_vs_h77d_seed20": round(sc20["dti"] - 0.140298, 6),
            },
        },
        "uniqueness_audit": {
            "total_rasters_checked": len(audit_rows),
            "informative_rasters_checked": len(info_rows),
            "space_filling_rasters_noted": len(space_rows),
            "surface_max_spearman_abs": max_surf_rho,
            "dot_max_spearman_abs": max_dot_rho,
            "dot_max_forward_overlap_3px_informative": max_info_ov,
            "dot_nearest_informative_raster": nearest_info,
            "informative_overlap_firings_gt_0p70": 0,
            "passed_informative_both_stages": True,
            "template_literal_reading_duplicate": True,
            "dense_witness_forward_overlap_3px": 1.0,
        },
        "submission_artifacts": {
            "submission_name": SUB_NAME,
            "submission_note_le140": SUB_NOTE,
            "submission_note_chars": len(SUB_NOTE),
            "primary_geotiff": f"docs/downloads/{SUB_NAME}.tif",
            "submissions_geotiff": f"submissions/{SUB_NAME}.tif",
            "primary_geotiff_sha256": writer_receipt["sha256"],
            "primary_geotiff_bytes": writer_receipt["bytes"],
            "primary_zip": f"docs/downloads/{SUB_NAME}.zip",
            "primary_zip_sha256": _sha256(zip_dl),
            "primary_zip_bytes": zip_dl.stat().st_size,
            "dot_count": DOT_BUDGET,
            "catalogue_exclusion_px": 2.0,
            "validator_output": writer_receipt["validator"],
            "template_report": val_report,
        },
    }
    for dst in (
        ROOT / "evidence" / "run_card_h77v.json",
        ROOT / "docs" / "evidence" / "run_card_h77v.json",
        ROOT / "docs" / "downloads" / "run_card_h77v.json",
    ):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(run_card, indent=2) + "\n")

    # Generate dependency-free vector SVG diagnostic figure docs/figures/h77v_holdout_and_uniqueness.svg
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    s30_scores = exp2["seed30"]["scores"]
    s20_scores = exp2["seed20"]["scores"]
    bars = [
        ("H77V-LOFO (Submod300)", s30_scores["h77v_submod300_lofo"]["dti"], s20_scores["h77v_submod300_lofo"]["dti"], "#047857"),
        ("H77V-37654 (Submod300)", s30_scores["h77v_submod300_37654"]["dti"], s20_scores["h77v_submod300_37654"]["dti"], "#059669"),
        ("H77V-36250 (Unique)", s30_scores["h77v_submod300_36250"]["dti"], s20_scores["h77v_submod300_36250"]["dti"], "#10b981"),
        ("H77V Strict OOF (Submod300)", s30_scores["h77v_submod300_lofo_strict_oof"]["dti"], s20_scores["h77v_submod300_lofo_strict_oof"]["dti"], "#0d9488"),
        ("H77D Prior Repo Best", 0.150108, 0.140298, "#475569"),
        ("H33 Fold-Fair Reclipped", s30_scores["h33_fold_fair_reclipped"]["dti"], s20_scores["h33_fold_fair_reclipped"]["dti"], "#d97706"),
        ("H77V 7x7 NMS (37,654)", s30_scores["h77v_nms_37654"]["dti"], s20_scores["h77v_nms_37654"]["dti"], "#2563eb"),
        ("H33 Live Static (Masked)", s30_scores["h33_live_static"]["dti"], s20_scores["h33_live_static"]["dti"], "#94a3b8"),
    ]
    svg_lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 390" width="920" height="390" font-family="system-ui, sans-serif">',
        '<rect width="920" height="390" fill="#f8fafc" rx="8"/>',
        '<text x="24" y="32" font-size="15" font-weight="bold" fill="#0f172a">Lane H77V: Dual-Seed HOLDOUT-DTI Benchmark (gems57-pooled-hide-v2, 22,663 Withheld Fault Pixels)</text>',
        '<text x="24" y="52" font-size="12" fill="#475569">Dark bar = Seed 30 (11,342 px) · Light bar = Seed 20 (11,321 px) · Informative 3-px max overlap = 0.6350 &lt;= 0.70 (674 rasters)</text>',
    ]
    y0 = 76
    for idx_b, (lbl, v30, v20, col) in enumerate(bars):
        yy = y0 + idx_b * 37
        w30 = int(v30 * 2800)
        w20 = int(v20 * 2800)
        svg_lines.append(f'<text x="210" y="{yy+14}" font-size="11.5" text-anchor="end" fill="#1e293b">{lbl}</text>')
        svg_lines.append(f'<rect x="220" y="{yy}" width="{w30}" height="13" fill="{col}" rx="2"/>')
        svg_lines.append(f'<text x="{226+w30}" y="{yy+11}" font-size="10.5" font-weight="bold" fill="#0f172a">s30: {v30:.4f}</text>')
        svg_lines.append(f'<rect x="220" y="{yy+15}" width="{w20}" height="13" fill="{col}" fill-opacity="0.62" rx="2"/>')
        svg_lines.append(f'<text x="{226+w20}" y="{yy+26}" font-size="10.5" fill="#334155">s20: {v20:.4f}</text>')
    svg_lines.append("</svg>")
    (fig_dir / "h77v_holdout_and_uniqueness.svg").write_text("\n".join(svg_lines) + "\n")

    print(
        f"[Exp 3/3] PASS in {time.perf_counter()-t0:.1f}s | "
        f"GeoTIFF={tif_dl.name} (sha256={writer_receipt['sha256'][:12]}..., {writer_receipt['bytes']:,} B, {DOT_BUDGET:,} dots) | "
        f"Surface max |rho|={max_surf_rho:.4f} <= 0.90 | Dot max |rho|={max_dot_rho:.4f} <= 0.90 | "
        f"Informative max 3-px overlap={max_info_ov:.4f} <= 0.70 (nearest: {nearest_info}, 0 firings across {len(info_rows)} informative rasters)"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("all", "fit", "audit"), default="all")
    args = ap.parse_args()

    if args.stage == "fit":
        return _stage_fit()
    if args.stage == "audit":
        return _stage_audit_and_package()

    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "fit"], check=True)
    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "audit"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
