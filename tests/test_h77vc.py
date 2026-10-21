"""Tests for Lane H77V: Geothermal Vent Upflow & Smectite Clay-Cap Breach + 300 m Submodular Cover."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
SUB_NAME = "gems77-h77v-geothermal-vent-submod-36250dots-20261021-zeros"
EXPECTED_SHA256 = "83df91facbe17d8fd7afae6c851c514317f1d763f61781af954eef9834c2f7d2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_h77v_geotiff_and_zip_artifacts_valid():
    tif_sub = ROOT / "submissions" / f"{SUB_NAME}.tif"
    tif_dl = ROOT / "docs" / "downloads" / f"{SUB_NAME}.tif"
    zip_dl = ROOT / "docs" / "downloads" / f"{SUB_NAME}.zip"
    assert tif_sub.is_file() and tif_dl.is_file() and zip_dl.is_file()
    assert _sha256(tif_sub) == _sha256(tif_dl) == EXPECTED_SHA256

    with zipfile.ZipFile(zip_dl) as zf:
        assert zf.namelist() == [f"{SUB_NAME}.tif"]
        assert hashlib.sha256(zf.read(f"{SUB_NAME}.tif")).hexdigest() == EXPECTED_SHA256

    with rasterio.open(tif_dl) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert (ds.height, ds.width) == (3730, 3292)
        assert str(ds.crs) == "EPSG:32611"
        assert ds.nodata is None
        arr = ds.read(1)

    assert np.isfinite(arr).all()
    assert int(np.isnan(arr).sum()) == 0
    assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
    assert int((arr > 0.5).sum()) == 36250


def test_h77vc_run_card_and_evidence_receipts():
    rc = json.loads((ROOT / "evidence" / "run_card_h77vc.json").read_text())
    exp1 = json.loads((ROOT / "evidence" / "h77vc_exp1_canary.json").read_text())
    exp2 = json.loads((ROOT / "evidence" / "h77vc_exp2_holdout.json").read_text())
    exp3 = json.loads((ROOT / "evidence" / "h77vc_exp3_uniqueness.json").read_text())

    # Policy I17 fail-closed invariants
    assert rc["verdict"] == "negative"
    assert rc["scientific_holdout_verdict"] == "promote"
    assert rc["ok_to_download"] is True
    assert rc["ok_to_submit"] is False
    assert rc["safe_to_download_and_submit"] is False
    assert rc["protocol_unique"] is False
    assert rc["informative_unique"] is True
    assert rc["weekly_submission_slots_used"] == 0
    assert len(rc["submission_artifacts"]["submission_note_le140"]) <= 140
    assert rc["submission_artifacts"]["primary_geotiff_sha256"] == EXPECTED_SHA256

    # Exp 1: Rule 4 Leakage Canary (all 114 features <= 0.90 on both seeds)
    assert exp1["n_total_features"] == 114
    assert exp1["all_114_features_passed_both_seeds"] is True
    assert exp1["seed20_max_symmetric_auc"] <= 0.90
    assert exp1["seed30_max_symmetric_auc"] <= 0.90
    assert exp1["sgmc_channels_used"] == 0
    assert exp1["gdr_dist_known_fault_px_accessed"] is False

    # Exp 2: Rule 2 & Rule 3 Holdout DTI
    assert exp2["evaluator_version"] == "gems57-pooled-hide-v2"
    s30 = exp2["seed30"]["scores"]["h77v_submod300_lofo"]
    s20 = exp2["seed20"]["scores"]["h77v_submod300_36250"]
    assert s30["evidence_class"] == "HOLDOUT-DTI"
    assert s30["withheld_positive_pixels"] == 11342
    assert 0.168 < s30["dti"] < 0.170
    assert len(s30["ci95"]) == 2 and s30["ci95"][0] < s30["dti"] < s30["ci95"][1]
    assert s20["evidence_class"] == "HOLDOUT-DTI"
    assert s20["withheld_positive_pixels"] == 11321
    assert 0.148 < s20["dti"] < 0.150
    assert len(s20["ci95"]) == 2 and s20["ci95"][0] < s20["dti"] < s20["ci95"][1]

    # Exp 3: Rule 1 Uniqueness Audit
    assert exp3["total_rasters_audited"] >= 820
    assert exp3["informative_rasters_audited"] >= 670
    assert exp3["surface_max_spearman_abs"] <= 0.90
    assert exp3["dot_max_spearman_abs"] <= 0.90
    assert exp3["informative_max_forward_overlap_3px"] <= 0.70
    assert exp3["informative_overlap_firings_gt_0p70"] == 0


def test_h77vc_wired_into_lanes_and_site():
    lanes = json.loads((ROOT / "docs" / "data" / "lanes.json").read_text())["lanes"]
    h77vc = next(x for x in lanes if x["lane"].startswith("H77VC:"))
    assert h77vc["sha256"] == EXPECTED_SHA256
    assert h77vc["verdict"] == "negative"
    assert h77vc["protocol_unique"] is False
    assert h77vc["informative_unique"] is True
    assert "not ok to submit" in h77vc["upload"].lower()
    assert "inspection only" in h77vc["upload"].lower()

    page = (ROOT / "docs" / "h77vc.html").read_text()
    assert EXPECTED_SHA256 in page
    assert "Not OK to submit" in page
    assert "not approved" in page
