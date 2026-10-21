"""Unit and integration tests for the H77-BGB lane (Buried Growth-Wedge Basement Inflection
x Transtensional Strain-Partitioning Gradient x Illite-Smectite Conductivity-Margin Breach).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems77_bgb import features as FBGB


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_gravity_normal_basement_hessian_peaks_at_half_graben_step():
    """Cross-strike basement slope and Hessian inflection along gravity gradient normal peak at fault."""
    h, w = 80, 80
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    # Fault at x = 40: basement drops from 200 m (footwall x < 40) to 1800 m (hanging wall x > 40)
    depth_base = 200.0 + 1600.0 / (1.0 + np.exp(-(x - 40.0) / 1.5))
    # Isostatic gravity decreases into the hanging-wall basin
    iso_grav = 15.0 - 30.0 / (1.0 + np.exp(-(x - 40.0) / 2.0))

    step_n, infl_nn, grad_g, _ = FBGB.compute_gravity_normal_basement_hessian(
        depth_base, iso_grav, sigma_depth=1.5, sigma_grav=1.5
    )
    fault_band = step_n[:, 38:43].mean()
    far_band = step_n[:, 5:15].mean()
    assert fault_band > 25.0 * far_band

    hinge_infl = infl_nn[:, 36:45].max()
    far_infl = infl_nn[:, 5:15].max()
    assert hinge_infl > 15.0 * far_infl
    assert np.isfinite(grad_g).all()


def test_transtensional_partitioning_peaks_at_shear_to_dilation_boundary():
    """Gradient of transtensional mode index tau_ext peaks where shear transitions to dilatation."""
    h, w = 80, 80
    _, x = np.mgrid[0:h, 0:w].astype(np.float32)
    # Left domain: pure shear (dil = 0, shear = 50); right domain: strong extension (dil = 50, shear = 20)
    geod_dil = 50.0 / (1.0 + np.exp(-(x - 40.0) / 1.5))
    geod_shr = np.full((h, w), 35.0, dtype=np.float32)

    tau_ext, tau_grad = FBGB.compute_transtensional_partitioning(geod_dil, geod_shr, sigma_grad=1.5)
    assert np.all((tau_ext >= -1.0) & (tau_ext <= 1.0))
    boundary_grad = float(tau_grad[:, 38:43].mean())
    interior_grad = float(tau_grad[:, 5:15].mean())
    assert boundary_grad > 20.0 * interior_grad


def test_conductivity_demag_breach_peaks_at_clay_cap_margin_with_rtp_trough():
    """Illite-smectite conductivity margin x RTP demagnetization trough peaks at clay-cap edge."""
    h, w = 80, 80
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    foot = np.ones((h, w), dtype=bool)
    # Conductive smectite clay cap in x >= 40
    cond = 0.25 / (1.0 + np.exp(-(x - 40.0) / 1.2))
    # Shallow RTP has a hydrothermal demagnetization trough along the margin x = 40
    rtp = 100.0 - 80.0 * np.exp(-((x - 40.0) ** 2) / (2.0 * 2.0 ** 2))
    tmi_up150 = np.full((h, w), 100.0, dtype=np.float32)

    cond_margin, breach = FBGB.compute_conductivity_demag_breach(cond, rtp, tmi_up150, foot)
    assert float(cond_margin[:, 39:42].mean()) > 0.85
    assert float(breach[:, 39:42].mean()) > 0.78
    assert float(breach[:, 5:15].mean()) < 0.35


def test_zero_sgmc_and_zero_label_leakage_in_bgb_source():
    """Architectural firewall: src/gems77_bgb/features.py never touches labels, catalogue, SGMC, or dist_known_fault_px."""
    import ast
    tree = ast.parse((ROOT / "src" / "gems77_bgb" / "features.py").read_text())
    # Strip module/function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                node.body = node.body[1:]
    code_only = ast.unparse(tree)
    for forbidden in ("labels.tif", "existing_faults.tif", "derived_sgmc_faults", "dist_known_fault_px"):
        assert forbidden not in code_only, f"Forbidden token {forbidden} found in executable code"


def test_rank_in_footprint_edge_cases():
    """Verify rank_in_footprint handles NaNs, empty footprints, and shape mismatches safely."""
    import pytest
    foot = np.zeros((10, 10), dtype=bool)
    foot[2:8, 2:8] = True
    arr = np.full((10, 10), np.nan, dtype=np.float32)
    # All NaN inside footprint -> returns all 0.0 finite
    r_empty = FBGB.rank_in_footprint(arr, foot)
    assert np.isfinite(r_empty).all() and float(r_empty.sum()) == 0.0

    # Partial NaN inside footprint -> finite in [0, 1] inside footprint, 0.0 outside
    arr[3, 3] = 10.0
    arr[4, 4] = 20.0
    r_part = FBGB.rank_in_footprint(arr, foot)
    assert np.isfinite(r_part).all()
    assert float(r_part[~foot].max()) == 0.0
    assert float(r_part[4, 4]) > float(r_part[3, 3]) > 0.0

    with pytest.raises(ValueError):
        FBGB.rank_in_footprint(np.zeros((5, 5), dtype=np.float32), foot)


def test_bgb_run_card_and_leakage_canaries_and_uniqueness():
    """Verify evidence/run_card_bgb.json obeys every protocol and fail-closed rule."""
    rc_path = ROOT / "evidence" / "run_card_bgb.json"
    assert rc_path.is_file()
    rc = json.loads(rc_path.read_text())

    assert rc["lane_id"] == "H77-BGB"
    assert rc["verdict"] == "negative"
    assert rc["scientific_holdout_verdict"] == "negative"
    assert rc["ok_to_download"] is True
    assert rc["ok_to_submit"] is False
    assert rc["protocol_unique"] is False
    assert rc["like_for_like_dot_field_unique"] is True
    assert rc["weekly_slots_spent"] == 0
    assert rc["organizer_confirmed_score_for_this_tif"] is None
    assert rc["experiments_used"] == 3 <= rc["max_experiments_allowed"]
    assert rc["preregistration_sha256"] == _sha256(ROOT / rc["preregistration_path"])

    # Leakage canaries
    lc = rc["leakage_canary"]
    assert lc["flagged"] is False
    assert lc["highest_symmetric_auc"] <= lc["threshold_auc"] == 0.90
    assert len(lc["per_channel_seed30"]) == 10
    assert all(not v["flagged_leakage"] and v["max_symmetric_auc"] <= 0.90 for v in lc["per_channel_seed30"].values())

    # Holdout DTI labels
    hd = rc["holdout_dti"]
    assert hd["evidence_class"] == "HOLDOUT-DTI"
    assert hd["evaluator_version"] == "gems57-pooled-hide-v2"
    assert hd["withheld_positive_pixels"] == 11342
    assert len(hd["ci95"]) == 2 and hd["ci95"][0] <= hd["dti"] <= hd["ci95"][1]
    assert hd["all_arms_seed30"]["A_bgb_lane"]["dti"] > hd["all_arms_seed30"]["A0_random"]["dti"]
    assert hd["all_arms_seed30"]["A_std_plus_bgb"]["dti"] > hd["all_arms_seed30"]["A_std"]["dti"]
    assert hd["all_arms_seed30"]["A_bgb_loco"]["dti"] > hd["all_arms_seed30"]["A_std"]["dti"]
    assert hd["beats_repository_holdout_best"] is False

    # Two-phase registry uniqueness
    ru = rc["registry_uniqueness"]
    assert ru["phase1_pre_placement_surface"]["worst_abs_spearman"] <= 0.90
    assert ru["phase2_final_dots"]["worst_abs_spearman"] <= 0.90
    assert ru["phase2_final_dots"]["sparse_le100k_max_forward_overlap_3px"] <= 0.70
    assert ru["phase2_final_dots"]["all_raster_max_mutual_overlap_3px"] <= 0.70
    assert ru["phase2_final_dots"]["literal_all_raster_max_forward_overlap_3px"] > 0.70


def test_bgb_geotiff_and_zip_and_nan_twin():
    """Verify the generated H77-BGB GeoTIFF, single-TIFF ZIP, and CLI validator."""
    rc = json.loads((ROOT / "evidence" / "run_card_bgb.json").read_text())
    art = rc["artifact"]
    assert art["note_length"] == len(art["submission_note"]) <= 140

    tif_path = ROOT / art["raster_path"]
    sub_copy = ROOT / art["submissions_copy_path"]
    zip_path = ROOT / art["zip_path"]
    assert tif_path.is_file() and sub_copy.is_file() and zip_path.is_file()
    assert _sha256(tif_path) == _sha256(sub_copy) == art["raster_sha256"]
    assert _sha256(zip_path) == art["zip_sha256"]

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == [tif_path.name]
        assert zf.read(tif_path.name) == tif_path.read_bytes()

    with rasterio.open(ROOT / "data" / "bridge" / "sample_submission.tif") as s_ref:
        foot = np.isfinite(s_ref.read(1))
        ref_crs = s_ref.crs
        ref_shape = s_ref.shape
        ref_transform = s_ref.transform
    with rasterio.open(ROOT / "data" / "bridge" / "existing_faults.tif") as s_cat:
        cat = (s_cat.read(1) > 0) & foot

    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
        assert ds.crs == ref_crs and str(ds.crs) == "EPSG:32611"
        assert ds.shape == ref_shape == (3730, 3292)
        assert ds.transform == ref_transform
        assert ds.count == 1 and ds.dtypes[0] == "float32" and ds.nodata is None
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert int((arr > 0).sum()) == 37654
        assert int((arr[~foot] > 0).sum()) == 0
        assert int((arr[cat] > 0).sum()) == 0

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(tif_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    v_out = json.loads(proc.stdout)
    assert v_out["all_checks_passed"] is True
    assert v_out["ok_to_submit"] is False


def test_bgb_visible_on_home_and_subpages_unambiguous():
    """Verify H77-BGB is listed in lanes.json, index.html, bgb.html, and bgb-executive-summary.html."""
    rc = json.loads((ROOT / "evidence" / "run_card_bgb.json").read_text())
    tif_name = Path(rc["artifact"]["raster_path"]).name
    lanes = json.loads((ROOT / "docs" / "data" / "lanes.json").read_text())["lanes"]
    bgb_lane = next(x for x in lanes if x["lane"].startswith("H77-BGB:"))
    assert bgb_lane["verdict"] == "negative"
    assert bgb_lane["protocol_unique"] is False
    assert "Not OK to submit" in bgb_lane["upload"]
    assert bgb_lane["sha256"] == rc["artifact"]["raster_sha256"]

    for page_name in ("index.html", "bgb.html", "bgb-executive-summary.html"):
        txt = (ROOT / "docs" / page_name).read_text()
        assert tif_name in txt, page_name
        assert "Not OK to submit" in txt, page_name
        assert "not approved" in txt.lower(), page_name
