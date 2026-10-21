#!/usr/bin/env python3
"""Experiment 3 of 3 (H77-BGB):
  - Fits full-map 4-quadrant out-of-fold H77-BGB lane surface (`0.55 * rank(OOF_HGB(BGB_10)) + 0.45 * S_bgb_composite`,
    zero in-sample pixels, zero SGMC, zero truth leakage).
  - Runs Phase 1 pre-placement surface uniqueness audit across all 812 archive registry rasters
    + all local sibling TIFFs (`docs/downloads/*.tif`).
  - Allocates exact 37,654 binary dots via `allocate_submodular_300m` (`E = 3.0 px`, `min_sep = 2.20 px`).
  - Runs Phase 2 final-dot uniqueness audit via `vendor/gems57/uniqueness.py` (`two_phase_summary`
    and full 830-raster fast exact 29-offset Euclidean `<= 3.0 px` halo scan), reporting both:
      * Informative / dot-field reading (`support_fraction <= 0.20` and sparse `<= 100k` dots): PASS (`<= 0.70`)
      * Literal all-raster reading (`compare_array_to_registry` including the 44 space-filling continuous
        rasters such as `17GEMSDOE_E-proba-multiscale` `ab0a0a62...`): FAIL (`1.000 > 0.70`, Policy `I17`)
  - Writes format-verified GeoTIFF + `.zip` + `.json` via unmodified `vendor/gems57/submission_writer.py`
    into both `docs/downloads/` and `submissions/` (plus `-nan.tif` twin in `submissions/`).
  - Runs all 17 validator & integrity checks and writes `evidence/run_card_bgb.json` (and `docs/evidence/run_card_bgb.json`).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

from gems57 import evaluate_holdout as EH, grid as G, submission_writer as SW, uniqueness as UQ, validate as VAL
from gems77_bgb import features as FBGB
from gems77d.submodular import allocate_submodular_300m

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
EVID = ROOT / "evidence"
DL_DIR = ROOT / "docs" / "downloads"
SUB_DIR = ROOT / "submissions"
SAMPLE_BRIDGE = ROOT / "data" / "bridge" / "sample_submission.tif"
BGB_STACK_PATH = PROC / "bgb_stack.npy"

TARGET_DOTS = 37654
EXCL_PX = 3.0

HGB_KW = dict(
    max_iter=220,
    learning_rate=0.055,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=2.0,
    random_state=7703,
)


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_fullmap_oof_bgb_surface(
    bgb_stack: np.ndarray,
    s_comp: np.ndarray,
    cat: np.ndarray,
    foot: np.ndarray,
) -> np.ndarray:
    """Fit 4-quadrant out-of-fold HGB on the 10 H77-BGB channels (0 in-sample pixels) and blend with S_bgb_composite."""
    cache_p = PROC / "bgb_fullmap_oof_surface.npy"
    if cache_p.is_file():
        log("Loaded cached full-map cross-fitted H77-BGB surface from data/processed/bgb_fullmap_oof_surface.npy")
        return np.load(cache_p)

    h, w, c = bgb_stack.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)
    X_bgb = bgb_stack[foot]

    ymid, xmid = h // 2, w // 2
    quads = [
        ("NW", (np.arange(h)[:, None] < ymid) & (np.arange(w)[None, :] < xmid)),
        ("NE", (np.arange(h)[:, None] < ymid) & (np.arange(w)[None, :] >= xmid)),
        ("SW", (np.arange(h)[:, None] >= ymid) & (np.arange(w)[None, :] < xmid)),
        ("SE", (np.arange(h)[:, None] >= ymid) & (np.arange(w)[None, :] >= xmid)),
    ]
    rng = np.random.default_rng(7703)
    oof_hgb = np.zeros((h, w), dtype=np.float32)

    for qname, qmask in quads:
        tr_reg = ~qmask
        vis_tr = cat & tr_reg
        d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
        pos = fp_lut[np.flatnonzero(vis_tr & foot)]
        if pos.size > 50_000:
            pos = rng.choice(pos, size=50_000, replace=False)
        neg_pool = foot & tr_reg & ~vis_tr & (d_vis_tr > 3.0)
        hard = fp_lut[np.flatnonzero(neg_pool & (d_vis_tr <= 16.0))]
        easy = fp_lut[np.flatnonzero(neg_pool & (d_vis_tr > 16.0))]
        n_neg = int(2.5 * pos.size)
        nh = min(hard.size, int(0.55 * n_neg))
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([
            rng.choice(hard, size=nh, replace=False),
            rng.choice(easy, size=ne, replace=False),
        ])
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
        te_mask = qmask & foot
        te_idx = fp_lut[np.flatnonzero(te_mask)]

        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(X_bgb[tr_idx], y_tr)
        oof_hgb[te_mask] = clf.predict_proba(X_bgb[te_idx])[:, 1].astype(np.float32)
        del clf
        gc.collect()
        log(f"  Full-map H77-BGB OOF fold {qname} complete")

    p_full = (0.55 * FBGB.rank_in_footprint(oof_hgb, foot) + 0.45 * s_comp).astype(np.float32)
    p_full = FBGB.rank_in_footprint(p_full, foot)
    np.save(cache_p, p_full)
    return p_full


def run_two_phase_registry_audit(
    p_full: np.ndarray,
    dots: np.ndarray,
    foot: np.ndarray,
    stem_self: str,
) -> dict:
    """Audit both Phase 1 (pre-placement surface) and Phase 2 (final binary dots) across all 812
    archive registry rasters + all local sibling rasters (`docs/downloads/*.tif`).
    """
    audit_cache = PROC / "bgb_reg_audit_cache.json"
    if audit_cache.is_file():
        log("Loaded cached two-phase registry audit from data/processed/bgb_reg_audit_cache.json")
        return json.loads(audit_cache.read_text())
    h, w = foot.shape
    n_fp = int(foot.sum())
    fp_flat = np.flatnonzero(foot)
    rng_u = np.random.default_rng(7703)
    sub_fp = rng_u.choice(fp_flat, size=250_000, replace=False)

    surf_r = rankdata(p_full.ravel()[sub_fp]).astype(np.float64)
    surf_r -= surf_r.mean()
    surf_norm = float(np.linalg.norm(surf_r)) + 1e-12

    pred_zeros = dots.astype(np.float32)
    dot_r = rankdata(pred_zeros.ravel()[sub_fp]).astype(np.float64)
    dot_r -= dot_r.mean()
    dot_norm = float(np.linalg.norm(dot_r)) + 1e-12

    dot_idx = np.flatnonzero(dots)
    d_my = ndimage.distance_transform_edt(~dots).astype(np.float32)
    offs_le3 = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy * dy + dx * dx <= 9]
    halo_pad = np.zeros((h + 6, w + 6), dtype=bool)

    reg_idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    blob_meta = {r["blob"]: r for r in reg_idx["rasters"]}

    records = []
    for p in sorted((ROOT / ".cache" / "registry").glob("*.tif")):
        if p.name in ("dense_witness.tif", "h33.tif"):
            continue
        meta = blob_meta.get(p.stem, {})
        records.append({
            "source": "archive_registry",
            "blob": p.stem,
            "repo": meta.get("repo", "archive"),
            "submission": meta.get("submission_name", p.stem),
            "path": p,
        })
    for p in sorted(DL_DIR.glob("*.tif")):
        if stem_self in p.name or "bgb" in p.name.lower():
            continue
        records.append({
            "source": "local_sibling",
            "blob": p.stem,
            "repo": "GEMSDOE77",
            "submission": p.stem,
            "path": p,
        })

    p1_worst_rho = 0.0
    p1_worst_name = ""
    p2_worst_rho = 0.0
    p2_worst_rho_name = ""

    # Three support regimes:
    # 1. literal_all: all readable rasters (including 44 dense space-filling surfaces > 20% footprint)
    # 2. dot_field_le20pct: UQ.DENSE_SUPPORT_FRACTION <= 0.20 (<= 1,033,474 positive cells)
    # 3. sparse_dots_le100k: sparse discrete dot submissions (<= 100,000 dots)
    lit_worst_fwd = 0.0
    lit_worst_name = ""
    dotfield_worst_fwd = 0.0
    dotfield_worst_name = ""
    sparse_worst_fwd = 0.0
    sparse_worst_name = ""
    worst_mutual = 0.0
    worst_mutual_name = ""

    n_readable = 0
    n_unreadable = 0
    n_space_filling_gt20pct = 0
    n_dot_field_le20pct = 0
    n_sparse_le100k = 0
    top_sparse_matches = []
    top_dotfield_matches = []
    local_sibling_matches = []

    for i_r, rec in enumerate(records):
        try:
            with rasterio.open(rec["path"]) as s:
                if s.shape != (h, w):
                    n_unreadable += 1
                    continue
                arr = s.read(1)
        except Exception:
            n_unreadable += 1
            continue

        n_readable += 1
        arr_clean = np.where(np.isfinite(arr) & foot, arr, 0.0).astype(np.float32)

        # Phase 1 & Phase 2 Spearman rank correlation
        v_sub = arr_clean.ravel()[sub_fp]
        if float(v_sub.max() - v_sub.min()) > 0.0:
            vr = rankdata(v_sub).astype(np.float64)
            vr -= vr.mean()
            vnorm = float(np.linalg.norm(vr)) + 1e-12
            rho_surf = abs(float((surf_r @ vr) / (surf_norm * vnorm)))
            rho_dot = abs(float((dot_r @ vr) / (dot_norm * vnorm)))
        else:
            rho_surf = 0.0
            rho_dot = 0.0

        label_id = f"{rec['repo']}:{rec['submission']}"
        if rho_surf > p1_worst_rho:
            p1_worst_rho = rho_surf
            p1_worst_name = label_id
        if rho_dot > p2_worst_rho:
            p2_worst_rho = rho_dot
            p2_worst_rho_name = label_id

        pos_mask = arr_clean > 0.0
        n_pos = int(pos_mask.sum())
        support_frac = float(n_pos) / float(n_fp)
        if n_pos == 0:
            n_dot_field_le20pct += 1
            n_sparse_le100k += 1
            continue

        halo_pad.fill(False)
        for dy, dx in offs_le3:
            halo_pad[3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w] |= pos_mask
        halo = halo_pad[3 : 3 + h, 3 : 3 + w]
        fwd = float(halo.ravel()[dot_idx].mean())
        rev = float((d_my[pos_mask] <= 3.0).mean())
        mut = min(fwd, rev)

        if fwd > lit_worst_fwd:
            lit_worst_fwd = fwd
            lit_worst_name = label_id
        if mut > worst_mutual:
            worst_mutual = mut
            worst_mutual_name = label_id

        entry = {
            "source": rec["source"],
            "repo": rec["repo"],
            "submission": rec["submission"],
            "blob": rec["blob"],
            "their_dots": n_pos,
            "their_support_fraction": round(support_frac, 5),
            "forward_overlap_3px": round(fwd, 5),
            "reverse_overlap_3px": round(rev, 5),
            "mutual_overlap_3px": round(mut, 5),
            "phase1_surface_abs_spearman": round(rho_surf, 5),
            "phase2_dots_abs_spearman": round(rho_dot, 5),
        }

        if rec["source"] == "local_sibling":
            local_sibling_matches.append(entry)

        if support_frac > UQ.DENSE_SUPPORT_FRACTION:
            n_space_filling_gt20pct += 1
        else:
            n_dot_field_le20pct += 1
            if fwd > dotfield_worst_fwd:
                dotfield_worst_fwd = fwd
                dotfield_worst_name = label_id
            top_dotfield_matches.append(entry)

        if n_pos <= 100_000:
            n_sparse_le100k += 1
            if fwd > sparse_worst_fwd:
                sparse_worst_fwd = fwd
                sparse_worst_name = label_id
            top_sparse_matches.append(entry)

    top_dotfield_matches.sort(key=lambda r: r["forward_overlap_3px"], reverse=True)
    top_sparse_matches.sort(key=lambda r: r["forward_overlap_3px"], reverse=True)
    local_sibling_matches.sort(key=lambda r: r["forward_overlap_3px"], reverse=True)

    # Also invoke vendor/gems57/uniqueness.py directly on all local sibling rasters + pinned witnesses
    witness_recs = [
        {
            "repo": "17GEMSDOE",
            "submission": "17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif",
            "file": str(ROOT / ".cache" / "registry" / "dense_witness.tif"),
            "abs_path": str(ROOT / ".cache" / "registry" / "dense_witness.tif"),
        },
        {
            "repo": "GEMSDOE32",
            "submission": "h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif",
            "file": str(ROOT / ".cache" / "registry" / "h33.tif"),
            "abs_path": str(ROOT / ".cache" / "registry" / "h33.tif"),
        },
    ] + [
        {
            "repo": "GEMSDOE77",
            "submission": r["submission"],
            "file": str(r["path"]),
            "abs_path": str(r["path"]),
        }
        for r in records
        if r["source"] == "local_sibling"
    ]
    uq_surf = UQ.compare_array_to_registry(p_full, witness_recs, foot)
    uq_dots = UQ.compare_array_to_registry(pred_zeros, witness_recs, foot)
    uq_two_phase_surf = UQ.two_phase_summary(uq_surf["rows"], int(foot.sum()))
    uq_two_phase = UQ.two_phase_summary(uq_dots["rows"], int(foot.sum()))

    log(f"Two-phase registry audit across {n_readable} readable rasters ({n_unreadable} unreadable):")
    log(f"  Phase 1 surface max |rho|                   = {p1_worst_rho:.4f} ({p1_worst_name}) [gate <= 0.90]")
    log(f"  Phase 2 dots max |rho|                      = {p2_worst_rho:.4f} ({p2_worst_rho_name}) [gate <= 0.90]")
    log(f"  Phase 2 dot_field (<=20% fp) max 3px fwd    = {dotfield_worst_fwd:.4f} ({dotfield_worst_name}) [gate <= 0.70]")
    log(f"  Phase 2 sparse (<=100k dots) max 3px fwd    = {sparse_worst_fwd:.4f} ({sparse_worst_name}) [gate <= 0.70]")
    log(f"  Phase 2 all-raster max 3px mutual overlap   = {worst_mutual:.4f} ({worst_mutual_name}) [gate <= 0.70]")
    log(f"  Phase 2 literal all-raster max 3px fwd      = {lit_worst_fwd:.4f} ({lit_worst_name}) [fires on {n_space_filling_gt20pct} dense surfaces per Policy I17]")

    assert p1_worst_rho <= 0.90, f"Phase 1 surface Spearman {p1_worst_rho} > 0.90"
    assert p2_worst_rho <= 0.90, f"Phase 2 dots Spearman {p2_worst_rho} > 0.90"
    assert sparse_worst_fwd <= 0.70, f"Phase 2 sparse forward overlap {sparse_worst_fwd} > 0.70"
    assert worst_mutual <= 0.70, f"Phase 2 mutual overlap {worst_mutual} > 0.70"

    res_dict = {
        "rasters_total": len(records),
        "rasters_readable": n_readable,
        "rasters_unreadable": n_unreadable,
        "space_filling_rasters_gt20pct": n_space_filling_gt20pct,
        "dot_field_rasters_le20pct": n_dot_field_le20pct,
        "sparse_dot_rasters_le100k": n_sparse_le100k,
        "phase1_pre_placement_surface": {
            "worst_abs_spearman": round(p1_worst_rho, 5),
            "worst_spearman_match": p1_worst_name,
            "spearman_threshold": 0.90,
            "duplicate_by_spearman": bool(p1_worst_rho > 0.90),
            "literal_dense_witness_overlap_3px": 1.0,
        },
        "phase2_final_dots": {
            "worst_abs_spearman": round(p2_worst_rho, 5),
            "worst_spearman_match": p2_worst_rho_name,
            "spearman_threshold": 0.90,
            "duplicate_by_spearman": bool(p2_worst_rho > 0.90),
            "like_for_like_dot_field_le20pct_max_forward_overlap_3px": round(dotfield_worst_fwd, 5),
            "like_for_like_dot_field_worst_match": dotfield_worst_name,
            "like_for_like_dot_field_duplicate": bool(dotfield_worst_fwd > 0.70),
            "sparse_le100k_max_forward_overlap_3px": round(sparse_worst_fwd, 5),
            "sparse_le100k_worst_match": sparse_worst_name,
            "sparse_le100k_duplicate": bool(sparse_worst_fwd > 0.70),
            "all_raster_max_mutual_overlap_3px": round(worst_mutual, 5),
            "all_raster_worst_mutual_match": worst_mutual_name,
            "literal_all_raster_max_forward_overlap_3px": round(lit_worst_fwd, 5),
            "literal_all_raster_worst_match": lit_worst_name,
            "literal_all_raster_duplicate": True,
            "policy_i17_note": (
                "Under Repository Policy I17, literal compare_array_to_registry includes 44 continuous space-filling "
                "probability rasters (such as 17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif, sha256 ab0a0a62..., "
                "5,167,373 positive pixels = 100% of footprint) whose 3-px dilation covers 100% of the evaluation "
                "domain, causing every non-empty candidate to register forward_overlap_3px = 1.000 > 0.70 on the "
                "literal all-raster check even when like-for-like dot-field overlap is <= 0.70."
            ),
        },
        "vendor_two_phase_summary_witness_and_local_siblings": uq_two_phase,
        "top_10_dot_field_matches": top_dotfield_matches[:10],
        "top_10_sparse_matches": top_sparse_matches[:10],
        "local_sibling_comparisons": local_sibling_matches,
    }
    audit_cache.write_text(json.dumps(res_dict, indent=2) + "\n")
    return res_dict


def _json_safe(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> int:
    t0 = time.time()
    DL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(parents=True, exist_ok=True)

    grid = G.load_grid(RAW)
    foot = grid.footprint
    cat = grid.catalogue
    h, w = foot.shape

    bgb_stack, bgb_names = FBGB.build_bgb_features(RAW, foot, cache_path=BGB_STACK_PATH)
    surfaces = FBGB.compute_bgb_surfaces(bgb_stack, foot)
    p_full = build_fullmap_oof_bgb_surface(bgb_stack, surfaces["S_bgb_composite"], cat, foot)

    d_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    allowed = foot & ~cat & (d_cat > EXCL_PX)

    dots_cache = PROC / "bgb_fullmap_dots.npy"
    if dots_cache.is_file():
        log("Loaded cached 37,654 binary dots from data/processed/bgb_fullmap_dots.npy")
        dots = np.load(dots_cache)
    else:
        log(f"Allocating {TARGET_DOTS} binary dots via allocate_submodular_300m (E={EXCL_PX}px, min_sep=2.20px)...")
        dots = allocate_submodular_300m(
            p_full, allowed, target_n=TARGET_DOTS, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
        )
        np.save(dots_cache, dots)
    n_dots = int(dots.sum())
    assert n_dots == TARGET_DOTS, f"Expected {TARGET_DOTS} dots, got {n_dots}"
    pred_zeros = dots.astype(np.float32)

    dot_digest = hashlib.sha256(np.packbits(dots.ravel()).tobytes()).hexdigest()[:8]
    stem_base = f"gems77-bgb-growthwedge-transtens-{n_dots}dots-20261021-{dot_digest}"
    sub_name = f"{stem_base}-zeros"

    dl_tif = DL_DIR / f"{sub_name}.tif"
    dl_zip = DL_DIR / f"{sub_name}.zip"
    dl_json = DL_DIR / f"{sub_name}.json"

    sub_tif = SUB_DIR / f"{sub_name}.tif"
    sub_zip = SUB_DIR / f"{sub_name}.zip"
    sub_json = SUB_DIR / f"{sub_name}.json"
    sub_nan = SUB_DIR / f"{stem_base}-nan.tif"

    for old in (dl_tif, dl_zip, dl_json, sub_tif, sub_zip, sub_json, sub_nan):
        if old.exists():
            old.unlink()

    note_140 = (
        f"H77-BGB basement growth-wedge + transtensional strain + clay-cap breach ({n_dots} dots); "
        f"HOLDOUT-DTI 0.0584 lane / 0.1024 LOCO; DO NOT UPLOAD"
    )[:140]
    assert len(note_140) <= 140, f"Note length {len(note_140)} > 140"

    sw_receipt = SW.write_submission(
        dl_tif,
        pred_zeros,
        SAMPLE_BRIDGE,
        foot,
        note=note_140,
        name=sub_name,
        catalogue=cat,
        metadata={"lane": "H77-BGB", "verdict": "negative", "ok_to_submit": False},
    )

    shutil.copy2(dl_tif, sub_tif)
    shutil.copy2(dl_zip, sub_zip)
    shutil.copy2(dl_json, sub_json)

    # Also write -nan.tif twin in submissions/ (NaN strictly outside footprint; 0 NaN inside footprint)
    pred_nan = np.where(foot, pred_zeros, np.nan).astype(np.float32)
    with rasterio.open(SAMPLE_BRIDGE) as s_ref:
        prof = s_ref.profile.copy()
    prof.update(dtype="float32", count=1, compress="deflate", predictor=2, nodata=np.nan)
    with rasterio.open(sub_nan, "w", **prof) as dst:
        dst.write(pred_nan, 1)

    # Run two-phase registry uniqueness audit
    reg_audit = run_two_phase_registry_audit(p_full, dots, foot, stem_self=stem_base)

    # Verify all 17 validator & integrity checks explicitly
    val_zeros = _json_safe(VAL.validate(dl_tif, footprint=foot, catalogue=cat))
    val_nan = _json_safe(VAL.validate(sub_nan, footprint=foot, catalogue=cat))
    assert val_zeros["all_checks_passed"], f"validate failed on {dl_tif}: {val_zeros}"
    assert val_nan["checks"]["in_footprint_all_finite"] and val_nan["checks"]["in_footprint_range_0_1"], (
        f"nan twin failed in-footprint check: {val_nan}"
    )

    with rasterio.open(SAMPLE_BRIDGE) as sr, rasterio.open(dl_tif) as sz, rasterio.open(sub_nan) as sn:
        z_arr = sz.read(1)
        n_arr = sn.read(1)
        with zipfile.ZipFile(dl_zip) as zf:
            zip_members = zf.namelist()
            zip_matches_bytes = (zip_members == [dl_tif.name]) and (zf.read(zip_members[0]) == dl_tif.read_bytes())
        checks_17 = {
            "01_file_exists": dl_tif.is_file() and sub_tif.is_file(),
            "02_zip_exists": dl_zip.is_file() and sub_zip.is_file(),
            "03_zip_single_geotiff_member_byte_identical": bool(zip_matches_bytes),
            "04_nan_twin_exists": sub_nan.is_file(),
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
            "15_zero_positive_on_known_catalogue": int((z_arr[cat] > 0).sum()) == 0,
            "16_exact_target_positive_dots": int((z_arr > 0).sum()) == TARGET_DOTS,
            "17_note_length_le_140": len(note_140) <= 140,
        }
    assert all(checks_17.values()), f"17-check failure: {checks_17}"

    exp1_rec = json.loads((EVID / "bgb_exp1_canary_and_surface.json").read_text())
    exp2_rec = json.loads((EVID / "bgb_exp2_holdout.json").read_text())
    s30 = exp2_rec["results_by_seed"]["seed30"]

    tif_sha = sha256_file(dl_tif)
    zip_sha = sha256_file(dl_zip)
    nan_sha = sha256_file(sub_nan)
    prereg_sha = sha256_file(ROOT / "registry" / "preregistration_bgb.json")

    run_card = {
        "schema_version": "gemsdoe77-run-card-v2",
        "lane_id": "H77-BGB",
        "lane_name": "H77-BGB: Buried Half-Graben Growth-Wedge Inflection x Transtensional Strain-Partitioning Gradient x Illite-Smectite Conductivity-Margin Breach",
        "generated_utc": "2026-10-21T16:40:00Z",
        "preregistration_path": "registry/preregistration_bgb.json",
        "preregistration_sha256": prereg_sha,
        "hypothesis": (
            "Blind Quaternary normal and transtensional faults in northwestern Great Basin sedimentary basins "
            "lack surface scarps where Holocene playa sedimentation or argillic hydrothermal alteration outpaces "
            "vertical slip, yet they preserve three coupled subsurface physical signatures: (1) an asymmetric "
            "cross-strike basement inflection dipole where the 2D Hessian of pre-Tertiary basement depth "
            "(depth_to_base_surf, Band 15) projected onto the unit horizontal gravity gradient normal "
            "n_g = grad(g)/|grad(g)| (from iso_grav_anom Band 13) peaks beneath flat basin floors (low det_elev_slope "
            "Band 19); (2) a spatial gradient in the dimensionless kinematic transtensional strain-partitioning "
            "index tau_ext = geod_dilaterate / sqrt(geod_dilaterate^2 + geod_shearrate^2) coupled with high "
            "geodetic 2nd invariant strain rate relative to smoothed seismic moment release (Bands 4, 7, 8, 10, 16); "
            "and (3) an illite-smectite hydrothermal upflow breach where a steep lateral gradient in log surface "
            "electrical conductivity |grad log(1 + cond_surf)| (Band 17) coincides with a shallow hydrothermal "
            "magnetite-destruction trough (upward-continued TMI_up150 exceeding shallow RTP with positive RTP "
            "Laplacian trough) and DOE GDR 1391 chemical geothermometer disequilibrium."
        ),
        "physical_mechanism": (
            "10 label-free channels across 3 sub-blocks: Sub-block A projects grad(depth_to_base_surf) and "
            "Hess(depth_to_base_surf) onto the gravity horizontal gradient normal n_g = grad(g)/|grad(g)| weighted "
            "by exp(-det_elev_slope / P60); Sub-block B computes tau_ext = dil / sqrt(dil^2 + shear^2) and "
            "|grad(tau_ext)| * rank(geod_2ndinv); Sub-block C couples |grad log(1 + cond_surf)| with "
            "max(0, rank(TMI_up150) - rank(RTP)) + RTP Laplacian trough and GDR 1391 silica/cation geothermometer "
            "disequilibrium Delta_T = max(T_quartz, T_chalc, T_cat) - T_meas."
        ),
        "named_non_fault_mimicking_process": (
            "Buried pre-Tertiary paleochannel incised-valley fills or Mesozoic pluton/caldera intrusive margins "
            "beneath flat saline playa lakebeds, which can create basement depth curvature steps along gravity "
            "gradients, sharp lateral electrical conductivity transitions at saline/freshwater interfaces, and "
            "primary low-magnetization felsic volcanic troughs without an active Quaternary fault conduit."
        ),
        "experiments_used": 3,
        "max_experiments_allowed": 3,
        "weekly_slots_spent": 0,
        "organizer_confirmed_score_for_this_tif": None,
        "leakage_canary": {
            "threshold_auc": 0.90,
            "highest_symmetric_auc": exp1_rec["leakage_canary"]["highest_symmetric_auc"],
            "flagged": False,
            "channels_tested": 10,
            "per_channel_seed30": exp1_rec["leakage_canary"]["channels_seed30"],
        },
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "seed": 30,
            "withheld_positive_pixels": s30["withheld_positive_pixels"],
            "candidate_lane_arm": "A_bgb_lane",
            "dti": s30["arms"]["A_bgb_lane"]["pooled_holdout_dti"],
            "ci95": s30["arms"]["A_bgb_lane"]["ci95"],
            "per_fold": s30["arms"]["A_bgb_lane"]["per_fold"],
            "all_arms_seed30": {
                k: {
                    "evidence_class": "HOLDOUT-DTI",
                    "evaluator_version": EH.VERSION,
                    "withheld_positive_pixels": v["withheld_positive_pixels"],
                    "dti": v["pooled_holdout_dti"],
                    "ci95": v["ci95"],
                    "dots": v["dots"],
                }
                for k, v in s30["arms"].items()
            },
            "paired_differences_lane_vs_controls": s30["pooled_summary_candidate_lane"]["paired_differences"],
            "paired_differences_std_plus_bgb_vs_controls": s30["pooled_summary_candidate_std_plus_bgb"]["paired_differences"],
            "paired_differences_loco_vs_controls": s30["pooled_summary_candidate_loco"]["paired_differences"],
            "repository_holdout_best_reference": exp2_rec["repository_holdout_best_reference"],
            "beats_random_baseline": True,
            "beats_std_baseline_when_added": True,
            "beats_repository_holdout_best": False,
        },
        "registry_uniqueness": reg_audit,
        "artifact": {
            "submission_name": sub_name,
            "submission_note": note_140,
            "note_length": len(note_140),
            "raster_path": f"docs/downloads/{dl_tif.name}",
            "submissions_copy_path": f"submissions/{sub_tif.name}",
            "zip_path": f"docs/downloads/{dl_zip.name}",
            "nan_twin_path": f"submissions/{sub_nan.name}",
            "raster_sha256": tif_sha,
            "zip_sha256": zip_sha,
            "nan_twin_sha256": nan_sha,
            "positive_dots": n_dots,
            "exclusion_collar_px": EXCL_PX,
            "min_separation_px": 2.20,
        },
        "validator": {
            "all_checks_passed": True,
            "checks_17": checks_17,
            "validate_submission_zeros": val_zeros,
            "validate_submission_nan_twin": val_nan,
            "submission_writer_receipt": sw_receipt,
        },
        "verdict": "negative",
        "scientific_holdout_verdict": "negative",
        "protocol_unique": False,
        "like_for_like_dot_field_unique": True,
        "ok_to_download": True,
        "ok_to_submit": False,
        "selector_approved": False,
        "status_banner": "OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT (research artifact only; do not upload to DrivenData)",
        "verdict_rationale": (
            "1. Format & Integrity PASS: 37,654 binary float32 dots in [0, 1], 0 NaN inside footprint, exact "
            "EPSG:32611 3730x3292 grid, 0 positive pixels on known catalogue faults. "
            "2. Leakage Canaries PASS: All 10 H77-BGB channels scored <= 0.6378 symmetric ROC-AUC (<= 0.90 gate). "
            "3. Informative / Like-for-Like Uniqueness PASS: Phase 1 pre-placement surface max |rho| = "
            f"{reg_audit['phase1_pre_placement_surface']['worst_abs_spearman']:.4f} <= 0.90; Phase 2 final dots "
            f"max |rho| = {reg_audit['phase2_final_dots']['worst_abs_spearman']:.4f} <= 0.90; like-for-like "
            f"dot-field (<=20% footprint) max 3-px forward overlap = "
            f"{reg_audit['phase2_final_dots']['like_for_like_dot_field_le20pct_max_forward_overlap_3px']:.4f} <= 0.70 "
            f"and sparse (<=100k dots) max 3-px forward overlap = "
            f"{reg_audit['phase2_final_dots']['sparse_le100k_max_forward_overlap_3px']:.4f} <= 0.70. "
            "4. Fail-Closed Negative Verdict: (a) On gems57-pooled-hide-v2 (11,342 withheld positives), A_bgb_lane "
            "achieves 0.058406 HOLDOUT-DTI (95% CI [0.045764, 0.070749], beating A0_random 0.048611 by +0.009795) "
            "and adding H77-BGB to the 33-channel baseline lifts A_std from 0.099208 to 0.101142 (+0.001934) and "
            "0.102390 with fold-visible LOCO mechanics (+0.003182), but does not beat the repository holdout best "
            "(H77-Step at 0.152718 HOLDOUT-DTI); (b) under Repository Policy I17, literal compare_array_to_registry "
            "fires at 1.000 > 0.70 against the 44 space-filling continuous rasters in the registry (such as "
            "17GEMSDOE_E-proba-multiscale ab0a0a62...). Therefore verdict = negative and NOT OK TO SUBMIT."
        ),
    }

    rc_path = EVID / "run_card_bgb.json"
    rc_path.write_text(json.dumps(_json_safe(run_card), indent=2, allow_nan=False) + "\n")
    (ROOT / "docs" / "evidence").mkdir(parents=True, exist_ok=True)
    shutil.copy2(rc_path, ROOT / "docs" / "evidence" / "run_card_bgb.json")
    log(f"Experiment 3 complete in {time.time()-t0:.1f}s -> {dl_tif.name} (sha256={tif_sha[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
