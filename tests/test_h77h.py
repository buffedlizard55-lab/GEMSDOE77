"""Verification suite for Lane H77H (Mesh-Deconfounded Hydrothermal-Structural Coupled Submodular Cover)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

from src.gems77h.features import EXCLUDED_MESH_CONTAMINATED_FEATURES, H77H_DELTA_NAMES, H77H_SLOCO_MECH_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_h77h_preregistration_and_run_card_provenance():
    prereg_path = ROOT / "registry" / "preregistration_h77h.json"
    rc_path = ROOT / "evidence" / "h77h" / "run_card.json"
    rc_copy = ROOT / "evidence" / "run_card_h77h.json"
    assert prereg_path.exists()
    assert rc_path.exists()
    assert rc_copy.exists()

    rc = json.loads(rc_path.read_text())
    assert rc == json.loads(rc_copy.read_text())
    assert rc["lane_id"] == "H77H"
    assert rc["preregistration_sha256"] == _sha256(prereg_path)
    assert rc["experiments_run"] <= 3
    assert rc["weekly_slots_spent"] == 0
    assert rc["ok_to_download"] is True
    assert rc["ok_to_submit"] is False
    assert rc["verdict"] == "negative"


def test_h77h_mesh_deconfounding_and_feature_provenance():
    rc = json.loads((ROOT / "evidence" / "h77h" / "run_card.json").read_text())
    prov = rc["feature_provenance"]
    assert prov["sgmc_in_features"] is False
    assert prov["sgmc_channels_used"] == 0
    assert prov["prior_submissions_in_features"] is False
    assert prov["prior_rasters_read"] == 0
    assert prov["min_regional_smoothing_sigma_px"] >= 35.0
    assert set(prov["excluded_mesh_contaminated_features"]) == set(EXCLUDED_MESH_CONTAMINATED_FEATURES)
    assert rc["novel_h77h_features"] == H77H_DELTA_NAMES
    assert rc["sloco_mechanics_features"] == H77H_SLOCO_MECH_NAMES
    for bad in EXCLUDED_MESH_CONTAMINATED_FEATURES:
        assert bad not in rc["novel_h77h_features"]


def test_h77h_leakage_canaries_pass_both_seeds():
    rc = json.loads((ROOT / "evidence" / "h77h" / "run_card.json").read_text())
    for seed_key in ("seed20", "seed30"):
        can = rc["leakage_canary_audit"][seed_key]
        assert can["n_features_checked"] == 137
        assert can["n_static_checked"] == 128
        assert can["n_mech_checked"] == 9
        assert can["passed"] is True
        assert can["violator_count"] == 0
        assert can["violators"] == []
        assert can["max_overall"]["symmetric_auc"] <= 0.90
        assert can["max_static_physics"]["symmetric_auc"] <= 0.90


def test_h77h_dual_seed_holdout_receipts():
    rc = json.loads((ROOT / "evidence" / "h77h" / "run_card.json").read_text())
    s20 = rc["holdout_evaluation"]["seed20"]["scores"]
    s30 = rc["holdout_evaluation"]["seed30"]["scores"]

    for scores, expected_pos in ((s20, 11321), (s30, 11342)):
        for arm_name, entry in scores.items():
            assert entry["evidence_class"] == "HOLDOUT-DTI", arm_name
            assert entry["evaluator_version"] == "gems57-pooled-hide-v2", arm_name
            assert entry["withheld_positive_pixels"] == expected_pos, arm_name
            assert len(entry["ci95"]) == 2
            assert 0.0 < entry["ci95"][0] <= entry["dti"] <= entry["ci95"][1] < 1.0

    # Empirical confirmation of IR-H77G-01 & IR-H77G-03:
    assert s20["ablation_base116_without_h77h_channels"]["dti"] > 0.15562
    assert s20["h77h_pure_submodular"]["dti"] > 0.153
    assert s30["h77h_pure_submodular"]["dti"] > 0.154
    assert s20["h77h_portfolio_unique_submodular"]["dti"] > 0.124
    assert s30["h77h_portfolio_unique_submodular"]["dti"] > 0.124
    assert s20["h77h_pure_physics_no_mechanics"]["dti"] > s20["h33_foldfair"]["dti"]
    assert s30["h77h_pure_physics_no_mechanics"]["dti"] > s30["h33_foldfair"]["dti"]
    assert s20["h77h_pure_submodular"]["dti"] > s20["h77h_peak_nms_control"]["dti"] + 0.03


def test_h77h_uniqueness_audit_exact_match_vs_vendor():
    rc = json.loads((ROOT / "evidence" / "h77h" / "run_card.json").read_text())
    uq = rc["uniqueness_audit"]
    p1 = uq["phase1_pre_placement_surface"]
    p2_inf = uq["phase2_post_placement_final_dots"]["informative_reading_lt_0p70_bg"]
    p2_lit = uq["phase2_post_placement_final_dots"]["literal_unfiltered_reading_policy_I17"]

    assert p1["pass_le_0p90"] is True
    assert p1["max_abs_spearman"] <= 0.90
    assert p1["firings_gt_0p90"] == 0

    assert p2_inf["informative_unique"] is True
    assert p2_inf["firings_gt_0p70"] == 0
    assert p2_inf["max_forward_3px_overlap"] <= 0.70
    assert p2_inf["max_abs_spearman"] <= 0.90
    assert p2_inf["identical_bytes_count"] == 0

    # Fail-closed literal Policy I17 reading:
    assert p2_lit["protocol_unique"] is False
    assert p2_lit["max_forward_3px_overlap"] == 1.0

    # Exact bit-for-bit verification against vendor/gems57/uniqueness.py:dot_overlap
    assert len(uq["top_15_vendor_verified_matches"]) == 15
    for item in uq["top_15_vendor_verified_matches"]:
        assert item["memmap_vs_vendor_abs_diff"] < 1e-7


def test_h77h_geotiff_and_zip_portal_compliance():
    rc = json.loads((ROOT / "evidence" / "h77h" / "run_card.json").read_text())
    art = rc["candidate_artifact"]

    assert 1 <= len(art["submission_name"]) <= 80
    assert 1 <= len(art["submission_note"]) <= 140
    assert art["submission_note_length"] == len(art["submission_note"])

    tif_docs = ROOT / art["tif_path"]
    zip_docs = ROOT / art["zip_path"]
    json_docs = tif_docs.with_suffix(".json")
    tif_sub = ROOT / "submissions" / tif_docs.name
    zip_sub = ROOT / "submissions" / zip_docs.name
    json_sub = ROOT / "submissions" / json_docs.name

    assert tif_docs.exists() and tif_sub.exists()
    assert zip_docs.exists() and zip_sub.exists()
    assert json_docs.exists() and json_sub.exists()
    assert _sha256(tif_docs) == art["tif_sha256"] == _sha256(tif_sub)
    assert _sha256(zip_docs) == art["zip_sha256"] == _sha256(zip_sub)
    assert _sha256(json_docs) == _sha256(json_sub)
    assert "119 informative rasters" in rc["verdict_reason"]

    with rasterio.open(tif_docs) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert (ds.height, ds.width) == (3730, 3292)
        assert str(ds.crs) == "EPSG:32611"
        assert ds.nodata is None
        arr = ds.read(1)
        assert arr.dtype == np.float32
        assert np.isnan(arr).sum() == 0
        assert np.isinf(arr).sum() == 0
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0
        assert int((arr > 0).sum()) == 36800
        assert set(np.unique(arr).tolist()) == {0.0, 1.0}

    with zipfile.ZipFile(zip_docs, "r") as zf:
        names = zf.namelist()
        assert names == [tif_docs.name]
        assert hashlib.sha256(zf.read(tif_docs.name)).hexdigest() == art["tif_sha256"]
