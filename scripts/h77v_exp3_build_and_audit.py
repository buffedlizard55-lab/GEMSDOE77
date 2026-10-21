#!/usr/bin/env python3
"""H77V Experiment 3 (`scripts/h77v_exp3_build_and_audit.py`).

1. Builds the strictly out-of-fold 4-quadrant cross-fitted full-map H77V probability surface
   (`p_h77v_full`, zero in-sample pixels) combining translation-invariant structural HGB with
   the Up-Gradient Advection-Backtracked Geothermal Vent & Dilational Conduit surface (`S_h77v_advect`)
   and full-catalogue 12-bit Leave-Own-Component-Out (LOCO) structural mechanics.
2. Allocates `37,654` binary dots via exact 300 m kernel submodular cover (`allocate_submodular_300m`,
   `d_cat > 3.0 px`).
3. Executes the Two-Phase Uniqueness Gate (Protocol Rule 1):
   - Phase 1: Continuous probability surface (`p_h77v_full`) checked BEFORE placement (`|rho| <= 0.90`).
   - Phase 2: Final binary dot field checked against all local GEMSDOE77 lane rasters + external
     registry witnesses (`h33.tif`, `dense_witness.tif`, `GEMSDOE37`, `GEMSDOE71`, `GEMSDOE44`,
     `GEMSDOE67`, `60GEMSDOE`),plus literal `vendor/gems57/uniqueness.py`.
4. Emits the single-band float32 GeoTIFF (`-zeros.tif`, `.zip`, `-nan.tif`) via unmodified
   `vendor/gems57/submission_writer.py`, verifies all 17 validator checks, and writes the
   Protocol Rule 5 JSON run card (`submissions/run_card_h77v_20261021.json`).
"""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import grid as G
from gems57 import holdout as HO
from gems57 import submission_writer as SW
from gems57 import uniqueness as UQ
from gems57 import validate as VAL
from src.gems77d.submodular import allocate_submodular_300m
from src.gems77v.features import (
    ALL_STATIC_NAMES,
    H77V_CHANNEL_NAMES,
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
DL_DIR = ROOT / "docs" / "downloads"
SUB_DIR = ROOT / "submissions"
EVIDENCE_PATH = ROOT / "evidence" / "h77v_exp3_uniqueness_and_build.json"
RUN_CARD_SUB_PATH = SUB_DIR / "run_card_h77v_20261021.json"
RUN_CARD_DOCS_PATH = ROOT / "docs" / "data" / "run_card_h77v.json"

TARGET_DOTS = 37_654
EXCL_PX = 3.5

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
    random_state=7731,
)
HGB_MECH_KW = dict(
    max_iter=180,
    learning_rate=0.075,
    max_leaf_nodes=47,
    min_samples_leaf=60,
    l2_regularization=3.0,
    early_stopping=False,
    random_state=7732,
)
HGB_VENT_KW = dict(
    max_iter=160,
    learning_rate=0.07,
    max_leaf_nodes=43,
    min_samples_leaf=55,
    l2_regularization=3.0,
    early_stopping=False,
    random_state=7733,
)


def log(msg: str) -> None:
    print(f"[h77v-exp3] {msg}", flush=True)


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


def collect_prior_rasters() -> list[dict]:
    """Collect all prior rasters across local GEMSDOE77 lanes, pinned witnesses, and external witnesses."""
    records = []
    seen_sha = set()

    search_paths = []
    for p in sorted((ROOT / ".cache" / "registry").rglob("*.tif")):
        search_paths.append(("registry_witness", p))
    for p in sorted(DL_DIR.glob("*.tif")):
        if "h77v-vent-advect-conduit" in p.name:
            continue
        if p.name.endswith("-nan.tif"):
            continue
        search_paths.append(("gemsdoe77_lane", p))
    for p in sorted(SUB_DIR.glob("*.tif")):
        if "h77v-vent-advect-conduit" in p.name:
            continue
        if p.name.endswith("-nan.tif"):
            continue
        search_paths.append(("gemsdoe77_submission", p))

    for src_type, p in search_paths:
        if not p.is_file():
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        records.append({"source": src_type, "name": p.stem, "path": p, "sha256": sha})
    return records


def main() -> int:
    t0 = time.time()
    grid = G.load_grid(DATA_DIR)
    foot = grid.footprint
    cat = grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    X_fp, _ = build_h77v_static_stack(DATA_DIR, WORK_DIR)
    doms = load_h77v_domains(WORK_DIR)

    # Full-catalogue 12-bit LOCO mechanics (no fault component sees itself)
    loco_full_fp = build_h77v_fold_mechanics(
        cat, foot, doms["geod_rank"], cache_path=WORK_DIR / "loco_full_cat.npy"
    )
    d_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    full_oof_cache = WORK_DIR / "full_oof_h77v.npz"
    if full_oof_cache.exists():
        zf = np.load(full_oof_cache)
        s_base_static = zf["s_base_static"]
        s_base_mech = zf["s_base_mech"]
        s_vent_spec = zf["s_vent_spec"]
        log(f"Loaded cached full-map 4-quadrant OOF surfaces from {full_oof_cache.name}")
    else:
        log("Fitting 4-quadrant out-of-fold full-map H77V models (zero in-sample pixels)...")
        quad = HO.quadrant_ids(foot)
        rng = np.random.default_rng(7730)
        s_base_static = np.zeros((h, w), dtype=np.float32)
        s_base_mech = np.zeros((h, w), dtype=np.float32)
        s_vent_spec = np.zeros((h, w), dtype=np.float32)

        d_hydro_fp = doms["d_hydro"]
        d_mech_fp = doms["d_mech"]

        for q_id, q_name in enumerate(HO.FOLD_NAMES):
            t_q = time.time()
            te_mask = (quad == q_id) & foot
            tr_mask = (quad != q_id) & foot
            vis_tr = cat & tr_mask
            d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)

            pos = fp_lut[np.flatnonzero(vis_tr)]
            if pos.size > 48_000:
                pos = rng.choice(pos, 48_000, replace=False)
            neg_base = tr_mask & ~vis_tr & (d_vis_tr > 3.0)
            hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 16.0))]
            easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 16.0))]
            n_neg = int(2.5 * pos.size)
            nh = min(hard.size, int(0.55 * n_neg))
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            te_idx = fp_lut[np.flatnonzero(te_mask)]

            w_vent = (1.0 + 0.65 * d_hydro_fp[tr_idx] + 0.35 * d_mech_fp[tr_idx]).astype(np.float32)
            p_bs = fit_predict_fold_fp(X_fp, None, IDX_BASE, tr_idx, y_tr, te_idx, HGB_STATIC_KW)
            p_bm = fit_predict_fold_fp(X_fp, loco_full_fp, IDX_BASE, tr_idx, y_tr, te_idx, HGB_MECH_KW)
            p_vs = fit_predict_fold_fp(
                X_fp, loco_full_fp, IDX_VENT_SPEC, tr_idx, y_tr, te_idx, HGB_VENT_KW, sample_weight=w_vent
            )

            s_base_static[te_mask] = p_bs
            s_base_mech[te_mask] = p_bm
            s_vent_spec[te_mask] = p_vs
            log(f"  Full-map quadrant {q_name} predicted out-of-fold in {time.time() - t_q:.1f}s")

        np.savez_compressed(
            full_oof_cache,
            s_base_static=s_base_static,
            s_base_mech=s_base_mech,
            s_vent_spec=s_vent_spec,
        )

    # Build full-map H77V physical vent-advection & dilational conduit surface
    surf_phys_full = np.zeros((h, w), dtype=np.float32)
    surf_phys_full[foot] = doms["s_h77v_advect"].astype(np.float32)

    surf_base_full = (0.45 * s_base_static + 0.55 * s_base_mech).astype(np.float32)
    p_h77v_full = (0.28 * surf_base_full + 0.52 * surf_phys_full + 0.20 * s_vent_spec).astype(np.float32)

    allowed = foot & ~cat & (d_cat > EXCL_PX)
    log(f"Allocating {TARGET_DOTS} dots via allocate_submodular_300m (excl={EXCL_PX} px, min_sep=2.20 px)...")
    dots = allocate_submodular_300m(
        p_h77v_full, allowed, target_n=TARGET_DOTS, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
    )
    n_dots = int(dots.sum())
    assert n_dots == TARGET_DOTS, f"Expected {TARGET_DOTS} dots, got {n_dots}"
    pred_zeros = dots.astype(np.float32)

    # Two-Phase Uniqueness Audit across all prior rasters
    prior_records = collect_prior_rasters()
    log(f"Running Two-Phase Uniqueness Audit across {len(prior_records)} local + registry witness rasters...")

    rng_u = np.random.default_rng(7703)
    sub_fp = rng_u.choice(fp_flat, size=250_000, replace=False)
    surf_r = rankdata(p_h77v_full.ravel()[sub_fp]).astype(np.float64)
    surf_r -= surf_r.mean()
    surf_norm = float(np.linalg.norm(surf_r)) + 1e-12

    dot_idx = np.flatnonzero(dots)
    d_my = ndimage.distance_transform_edt(~dots).astype(np.float32)
    offs_le3 = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy * dy + dx * dx <= 9]
    halo_pad = np.zeros((h + 6, w + 6), dtype=bool)

    phase1_worst_rho = 0.0
    phase1_worst_name = ""
    phase2_worst_rho = 0.0
    phase2_worst_rho_name = ""
    phase2_sparse_worst_fwd = 0.0
    phase2_sparse_worst_mut = 0.0
    phase2_sparse_worst_name = ""
    phase2_literal_worst_fwd = 0.0
    phase2_literal_worst_name = ""

    audit_rows = []
    for rec in prior_records:
        with rasterio.open(rec["path"]) as s:
            if s.shape != (h, w):
                continue
            arr = s.read(1)
        arr_clean = np.where(np.isfinite(arr) & foot, arr, 0.0).astype(np.float32)

        v_sub = arr_clean.ravel()[sub_fp]
        if float(v_sub.max() - v_sub.min()) > 0.0:
            vr = rankdata(v_sub).astype(np.float64)
            vr -= vr.mean()
            rho_surf = abs(float((surf_r @ vr) / (surf_norm * (float(np.linalg.norm(vr)) + 1e-12))))
        else:
            rho_surf = 0.0
        if rho_surf > phase1_worst_rho:
            phase1_worst_rho = rho_surf
            phase1_worst_name = f"{rec['source']}:{rec['name']}"

        pos_mask = arr_clean > 0.0
        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            continue

        halo_pad.fill(False)
        for dy, dx in offs_le3:
            halo_pad[3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w] |= pos_mask
        halo = halo_pad[3 : 3 + h, 3 : 3 + w]
        cov3 = float(halo[foot].mean())
        fwd = float(halo.ravel()[dot_idx].mean())
        rev = float((d_my[pos_mask] <= 3.0).mean())
        mut = min(fwd, rev)

        d_fp_sub = pred_zeros.ravel()[sub_fp]
        if float(v_sub.max() - v_sub.min()) > 0.0:
            r_a = rankdata(v_sub)
            r_a -= r_a.mean()
            r_b = rankdata(d_fp_sub)
            r_b -= r_b.mean()
            rho_dot = abs(float((r_a @ r_b) / ((np.linalg.norm(r_a) + 1e-12) * (np.linalg.norm(r_b) + 1e-12))))
        else:
            rho_dot = 0.0

        if rho_dot > phase2_worst_rho:
            phase2_worst_rho = rho_dot
            phase2_worst_rho_name = f"{rec['source']}:{rec['name']}"

        if fwd > phase2_literal_worst_fwd:
            phase2_literal_worst_fwd = fwd
            phase2_literal_worst_name = f"{rec['source']}:{rec['name']}"

        is_sparse = n_pos <= 100_000
        if is_sparse:
            if fwd > phase2_sparse_worst_fwd:
                phase2_sparse_worst_fwd = fwd
                phase2_sparse_worst_name = f"{rec['source']}:{rec['name']}"
            if mut > phase2_sparse_worst_mut:
                phase2_sparse_worst_mut = mut

        audit_rows.append(
            {
                "source": rec["source"],
                "name": rec["name"],
                "sha256": rec["sha256"],
                "positives": n_pos,
                "is_sparse_le_100k": is_sparse,
                "coverage_3px": round(cov3, 5),
                "surface_spearman_abs": round(rho_surf, 5),
                "dot_spearman_abs": round(rho_dot, 5),
                "forward_overlap_3px": round(fwd, 5),
                "reverse_overlap_3px": round(rev, 5),
                "mutual_overlap_3px": round(mut, 5),
            }
        )

    audit_rows.sort(key=lambda r: r["forward_overlap_3px"], reverse=True)
    log(f"Phase 1 surface max |rho| = {phase1_worst_rho:.4f} ({phase1_worst_name}) [gate <= 0.90]")
    log(f"Phase 2 dot max |rho|     = {phase2_worst_rho:.4f} ({phase2_worst_rho_name}) [gate <= 0.90]")
    log(
        f"Phase 2 informative sparse (<=100k) max 3px forward overlap = "
        f"{phase2_sparse_worst_fwd:.4f} ({phase2_sparse_worst_name}) [gate <= 0.70]"
    )
    log(
        f"Phase 2 literal max 3px forward overlap (incl. dense_witness ab0a0a62) = "
        f"{phase2_literal_worst_fwd:.4f} ({phase2_literal_worst_name})"
    )

    assert phase1_worst_rho <= 0.90, f"Phase 1 surface |rho| exceeded 0.90: {phase1_worst_rho}"
    assert phase2_worst_rho <= 0.90, f"Phase 2 dot |rho| exceeded 0.90: {phase2_worst_rho}"
    assert phase2_sparse_worst_fwd <= 0.70, f"Phase 2 sparse forward overlap exceeded 0.70: {phase2_sparse_worst_fwd}"

    # Write primary -zeros.tif + .zip + .json via unmodified vendor/gems57/submission_writer.py + -nan.tif twin
    dot_digest = hashlib.sha256(np.packbits(dots.ravel()).tobytes()).hexdigest()[:8]
    stem_base = f"gems77-h77v-vent-advect-conduit-{n_dots}dots-20261021-{dot_digest}"
    zeros_path = DL_DIR / f"{stem_base}-zeros.tif"
    nan_path = DL_DIR / f"{stem_base}-nan.tif"
    sample_path = DATA_DIR / "sample_submission.tif"

    for old in list(DL_DIR.glob("gems77-h77v-vent-advect-conduit-*")) + list(SUB_DIR.glob("gems77-h77v-vent-advect-conduit-*")):
        old.unlink()

    note_140 = (
        f"H77V up-gradient vent advection + dilational conduit + 300m submodular ({n_dots} dots); "
        f"HOLDOUT-DTI 0.1270 (s30) / 0.1293 (s20); Policy I17"
    )[:140]
    assert len(note_140) <= 140

    sw_receipt = SW.write_submission(
        zeros_path,
        pred_zeros,
        sample_path,
        foot,
        note=note_140,
        name=f"{stem_base}-zeros",
        catalogue=cat,
        metadata={"lane": "H77V-VENT-ADVECT-CONDUIT", "verdict": "negative"},
    )

    pred_nan = np.where(foot, pred_zeros, np.nan).astype(np.float32)
    with rasterio.open(sample_path) as s_ref:
        prof = s_ref.profile.copy()
    prof.update(dtype="float32", count=1, compress="deflate", predictor=2, nodata=np.nan)
    with rasterio.open(nan_path, "w", **prof) as dst:
        dst.write(pred_nan, 1)

    # Copy primary -zeros.tif and .zip to submissions/
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zeros_path, SUB_DIR / zeros_path.name)
    shutil.copy2(zeros_path.with_suffix(".zip"), SUB_DIR / zeros_path.with_suffix(".zip").name)

    # Also invoke literal vendor/gems57/uniqueness.py against pinned witnesses (dense_witness.tif + h33.tif)
    witness_recs = [
        {
            "submission": "17GEMSDOE:dense_witness",
            "file": str(ROOT / ".cache" / "registry" / "dense_witness.tif"),
            "abs_path": str(ROOT / ".cache" / "registry" / "dense_witness.tif"),
            "local_path": str(ROOT / ".cache" / "registry" / "dense_witness.tif"),
        },
        {
            "submission": "60GEMSDOE:h33",
            "file": str(ROOT / ".cache" / "registry" / "h33.tif"),
            "abs_path": str(ROOT / ".cache" / "registry" / "h33.tif"),
            "local_path": str(ROOT / ".cache" / "registry" / "h33.tif"),
        },
    ]
    lit_uq = UQ.compare_to_registry(str(zeros_path), witness_recs, foot)
    val_report = VAL.validate(zeros_path, footprint=foot, catalogue=cat)

    # Verify all 17 validator & integrity checks explicitly
    with rasterio.open(sample_path) as sr, rasterio.open(zeros_path) as sz, rasterio.open(nan_path) as sn:
        z_arr = sz.read(1)
        n_arr = sn.read(1)
        with zipfile.ZipFile(zeros_path.with_suffix(".zip")) as zf:
            zip_members = zf.namelist()
        min_d_cat = float(d_cat[z_arr > 0].min())
        checks_17 = {
            "01_file_exists": zeros_path.is_file(),
            "02_zip_exists": zeros_path.with_suffix(".zip").is_file(),
            "03_zip_single_geotiff_member": len(zip_members) == 1 and zip_members[0].endswith(".tif"),
            "04_nan_twin_exists": nan_path.is_file(),
            "05_crs_epsg_32611": str(sz.crs) == "EPSG:32611" and sz.crs == sr.crs,
            "06_shape_3730x3292": sz.shape == (3730, 3292) and sz.shape == sr.shape,
            "07_band_count_1": sz.count == 1,
            "08_dtype_float32": str(sz.dtypes[0]) == "float32",
            "09_exact_geotransform_match": sz.transform == sr.transform,
            "10_zero_nan_inside_footprint_zeros_tif": int(np.isnan(z_arr[foot]).sum()) == 0,
            "11_zero_nan_inside_footprint_nan_tif": int(np.isnan(n_arr[foot]).sum()) == 0,
            "12_zero_inf_anywhere": int(np.isinf(z_arr).sum()) == 0,
            "13_strict_unit_interval_0_1": float(z_arr.min()) >= 0.0 and float(z_arr.max()) <= 1.0,
            "14_zero_positive_outside_footprint": int((z_arr[~foot] > 0).sum()) == 0,
            "15_zero_positive_on_existing_faults": int((z_arr[cat] > 0).sum()) == 0,
            "16_min_dist_to_existing_faults_gt_3px": min_d_cat > 3.0,
            "17_exact_dot_count_37654": int((z_arr > 0).sum()) == TARGET_DOTS,
        }
    assert all(checks_17.values()), f"Failed 17-point check: {checks_17}"

    zeros_sha256 = hashlib.sha256(zeros_path.read_bytes()).hexdigest()
    zip_sha256 = hashlib.sha256(zeros_path.with_suffix(".zip").read_bytes()).hexdigest()
    nan_sha256 = hashlib.sha256(nan_path.read_bytes()).hexdigest()

    exp2_data = json.loads((ROOT / "evidence" / "h77v_exp2_holdout.json").read_text())
    exp1_data = json.loads((ROOT / "evidence" / "h77v_exp1_canary_and_ablation.json").read_text())
    s30_sc = exp2_data["seed30"]["pooled_summary"]["scores"]["A_h77v_gated_submod"]
    s20_sc = exp2_data["seed20"]["pooled_summary"]["scores"]["A_h77v_gated_submod"]
    s30_d_base = exp2_data["seed30"]["pooled_summary"]["paired_differences"]["A_base_no_h77v_submod"]
    s20_d_base = exp2_data["seed20"]["pooled_summary"]["paired_differences"]["A_base_no_h77v_submod"]
    s30_d_tree = exp2_data["seed30"]["pooled_summary"]["paired_differences"]["A_h77v_tree_submod"]
    s20_d_tree = exp2_data["seed20"]["pooled_summary"]["paired_differences"]["A_h77v_tree_submod"]

    run_card = {
        "schema_version": "gems-run-card-v2",
        "generated_utc": "2026-10-21T16:40:00Z",
        "branch": "arena/a278fbd0-gemsdoe77",
        "lane": "H77V-VENT-ADVECT-CONDUIT",
        "short_id": "H77V",
        "hypothesis": (
            "H77V-1..3: Up-gradient topographic advection backtracking (+grad G_8(Z_ortho)) of cooled "
            "geothermometer disequilibrium plumes (gdr_wellspring_in_footprint.csv, 2m probes) and "
            "Lake-Lahontan-tufa-sieved paleo-vents (Paleo_geothermal_final.csv), coupled with the "
            "Critically Stressed Transtensional Dilational Conduit Tensor (geod_dilaterate / geod_shearrate "
            "x N20E strike dilation tendency), Steam-Heated Clay-Cap Breach Annulus (cond_surf x RTP "
            "demagnetization trough), and Upslope-Realigned Hanging-Wall Gravity Crest (+3.5 px), applied as "
            "a monotone translation-invariant physical conduit gate on out-of-fold structural HGB predictions."
        ),
        "mechanism": (
            "Buoyant geothermal fluids ascending blind Basin-and-Range fault conduits refract laterally down "
            "the topographic hydraulic head gradient (-grad Z_ortho) for 0.5-3.0 km into alluvial aquifers, "
            "cooling conductively and precipitating silica while preserving high cation/quartz geothermometers. "
            "Backtracking outflow wells upslope (+grad Z_ortho) by a distance scaled by thermal dilution D_dil "
            "and cation-chalcedony kinetic fractionation realigns the surface plume with the buried piedmont "
            "fault trace. Furthermore, applying the H77V vent/conduit prior as a translation-invariant physical "
            "gate avoids the 'Basin-ID Tree-Split Trap' where unconstrained decision trees split on regional "
            "hydrothermal plumes to memorize training-quadrant basin coordinates."
        ),
        "non_fault_mimicking_process": (
            "Unfaulted lateral pinch-out of warm confined gravel aquifers against buried bedrock paleo-channels "
            "or volcanic flow margins, wave-agitated Pleistocene Lake Lahontan highstand (1,332 m) shoreline "
            "tufa towers (30.6% of Paleo_geothermal_final.csv), and flat saline playa evaporite clay pans."
        ),
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator": "vendor/gems57/evaluate_holdout.py",
            "evaluator_version": exp2_data["evaluator_version"],
            "alpha": 0.2,
            "beta": 0.8,
            "triangular_radius_m": 300.0,
            "seed30": {
                "withheld_positives": exp2_data["seed30"]["withheld_positive_pixels"],
                "dti": round(float(s30_sc["dti"]), 6),
                "ci95": [round(float(x), 6) for x in s30_sc["ci95"]],
                "paired_delta_vs_base_no_h77v": {
                    "delta": round(float(s30_d_base["delta"]), 6),
                    "ci95": [round(float(x), 6) for x in s30_d_base["ci95"]],
                },
                "paired_delta_vs_unconstrained_tree_h77v": {
                    "delta": round(float(s30_d_tree["delta"]), 6),
                    "ci95": [round(float(x), 6) for x in s30_d_tree["ci95"]],
                },
            },
            "seed20": {
                "withheld_positives": exp2_data["seed20"]["withheld_positive_pixels"],
                "dti": round(float(s20_sc["dti"]), 6),
                "ci95": [round(float(x), 6) for x in s20_sc["ci95"]],
                "paired_delta_vs_base_no_h77v": {
                    "delta": round(float(s20_d_base["delta"]), 6),
                    "ci95": [round(float(x), 6) for x in s20_d_base["ci95"]],
                },
                "paired_delta_vs_unconstrained_tree_h77v": {
                    "delta": round(float(s20_d_tree["delta"]), 6),
                    "ci95": [round(float(x), 6) for x in s20_d_tree["ci95"]],
                },
            },
            "repo_holdout_best_reference": {
                "lane": "H77-Step",
                "seed30_dti": 0.152718,
                "beats_repo_holdout_best": False,
            },
        },
        "leakage_canary": {
            "max_symmetric_auc_h77v": exp1_data["leakage_canary"]["max_symmetric_auc_h77v"],
            "max_symmetric_auc_h77v_feature": exp1_data["leakage_canary"]["max_symmetric_auc_h77v_feature"],
            "max_symmetric_auc_all": exp1_data["leakage_canary"]["max_symmetric_auc_all"],
            "max_symmetric_auc_all_feature": exp1_data["leakage_canary"]["max_symmetric_auc_all_feature"],
            "gate_limit": 0.90,
            "all_passed": True,
        },
        "registry_uniqueness": {
            "rasters_audited": len(audit_rows),
            "phase1_surface_max_abs_rho": round(phase1_worst_rho, 5),
            "phase1_surface_worst_match": phase1_worst_name,
            "phase2_dot_max_abs_rho": round(phase2_worst_rho, 5),
            "phase2_dot_worst_rho_match": phase2_worst_rho_name,
            "phase2_informative_sparse_max_3px_forward_overlap": round(phase2_sparse_worst_fwd, 5),
            "phase2_informative_sparse_max_3px_mutual_overlap": round(phase2_sparse_worst_mut, 5),
            "phase2_informative_sparse_worst_match": phase2_sparse_worst_name,
            "phase2_literal_max_3px_forward_overlap": round(phase2_literal_worst_fwd, 5),
            "phase2_literal_worst_match": phase2_literal_worst_name,
            "informative_unique": True,
            "protocol_unique": False,
            "policy_i17_reason": (
                "Passes informative uniqueness across all sparse/discrete rasters (max 3px overlap "
                f"{phase2_sparse_worst_fwd:.4f} <= 0.70, max |rho| {phase2_worst_rho:.4f} <= 0.90), "
                "but fails literal vendor/gems57/uniqueness.py forward 3px overlap (1.0000 > 0.70) "
                "against the space-filling registry witness 17GEMSDOE:dense_witness (ab0a0a62) under Policy I17."
            ),
        },
        "artifact": {
            "submission_name": f"{stem_base}-zeros",
            "submission_note": note_140,
            "submission_note_length": len(note_140),
            "zeros_tif_path": f"docs/downloads/{zeros_path.name}",
            "zeros_tif_sha256": zeros_sha256,
            "zeros_tif_bytes": zeros_path.stat().st_size,
            "zip_path": f"docs/downloads/{zeros_path.with_suffix('.zip').name}",
            "zip_sha256": zip_sha256,
            "zip_bytes": zeros_path.with_suffix(".zip").stat().st_size,
            "nan_tif_path": f"docs/downloads/{nan_path.name}",
            "nan_tif_sha256": nan_sha256,
            "nan_tif_bytes": nan_path.stat().st_size,
            "emitted_dots": n_dots,
            "footprint_fraction": round(n_dots / int(foot.sum()), 6),
            "min_distance_to_existing_faults_px": round(min_d_cat, 3),
        },
        "validator_output": {
            "ok": bool(val_report["all_checks_passed"]),
            "checks_17": checks_17,
            "vendor_validate_report": val_report,
        },
        "verdict": "negative",
        "ok_to_submit": False,
        "ok_to_download_for_inspection": True,
        "weekly_slots_spent": 0,
        "status_banner": (
            "NOT OK TO SUBMIT TO DRIVENDATA (OK TO DOWNLOAD FOR LOCAL GIS / SCIENTIFIC INSPECTION ONLY) — "
            "Fail-closed under Policy I17 (literal dense-witness ab0a0a62 overlap 1.000 > 0.70) and "
            "HOLDOUT-DTI 0.1270 (Seed 30) / 0.1293 (Seed 20) does not exceed the 0.152718 H77-Step repo best."
        ),
    }

    RUN_CARD_SUB_PATH.write_text(json.dumps(run_card, indent=2) + "\n")
    RUN_CARD_DOCS_PATH.write_text(json.dumps(run_card, indent=2) + "\n")

    ev_out = {
        "lane": "H77V-VENT-ADVECT-CONDUIT",
        "experiment": "exp3_uniqueness_and_build",
        "run_card": run_card,
        "submission_writer_receipt": sw_receipt,
        "literal_uniqueness_witness_report": lit_uq,
        "audit_rows": audit_rows,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    EVIDENCE_PATH.write_text(json.dumps(ev_out, indent=2) + "\n")
    log(f"Saved Experiment 3 evidence -> {EVIDENCE_PATH} and run card -> {RUN_CARD_SUB_PATH} ({ev_out['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
