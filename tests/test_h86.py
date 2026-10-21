"""H86-LENHALO deliverable tests (negative lane, inspection-only).

These tests bind the *shipped bytes* to the claims on the site and in the run
card. They assert local measurements only: format/range preflight (not organizer
acceptance), the recorded HOLDOUT-DTI interval, and the fail-closed verdict.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = "gems77-h86-lenhalo-37654dots-20261021-zeros"
TIF = ROOT / "docs/downloads" / f"{NAME}.tif"
ZIP = ROOT / "docs/downloads" / f"{NAME}.zip"
RECEIPT = ROOT / "docs/downloads" / f"{NAME}.json"
CARD = ROOT / "evidence/run_card_h86.json"

SHA_TIF = "86f1394516d05aa5d545b8be7103472ca7bd6fe8ad651c7870a3fefb87e65345"
SHA_ZIP = "c81be31a96d66719017eb6985556138c4b2c515f43bc0fed212eaa26031bee66"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_artifacts_exist_and_hashes_match_run_card():
    card = json.loads(CARD.read_text())
    sub = card["submission_raster"]
    assert TIF.exists() and ZIP.exists() and RECEIPT.exists()
    assert _sha(TIF) == SHA_TIF == sub["sha256_tif"]
    assert _sha(ZIP) == SHA_ZIP == sub["sha256_zip"]
    assert TIF.stat().st_size == sub["bytes_tif"] == 826_436
    assert ZIP.stat().st_size == sub["bytes_zip"] == 175_941


def test_zip_contains_exactly_the_one_geotiff():
    with zipfile.ZipFile(ZIP) as zf:
        assert zf.namelist() == [TIF.name]
        assert zf.read(TIF.name) == TIF.read_bytes()


def test_grid_values_and_footprint_on_shipped_bytes():
    with rasterio.open(TIF) as src:
        arr = src.read(1)
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.height == 3730 and src.width == 3292
        assert str(src.crs) == "EPSG:32611"
        assert src.nodata is None
        assert [float(v) for v in tuple(src.transform)[:6]] == [
            100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0]
    assert np.isfinite(arr).all()
    assert not np.isinf(arr).any()
    assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
    values = np.unique(arr)
    assert values.tolist() == [0.0, 1.0]
    assert int((arr > 0).sum()) == 37_654


def test_zero_dots_on_catalogue_and_zeros_outside_footprint():
    with rasterio.open(TIF) as src:
        arr = src.read(1)
    with rasterio.open(ROOT / "data/bridge/sample_submission.tif") as src:
        footprint = np.isfinite(src.read(1))
    with rasterio.open(ROOT / "data/bridge/existing_faults.tif") as src:
        catalogue = src.read(1) > 0
    assert int((arr[~footprint] > 0).sum()) == 0
    assert int((arr[catalogue] > 0).sum()) == 0


def test_run_card_is_fail_closed_and_labels_are_honest():
    card = json.loads(CARD.read_text())
    assert card["verdict"] == "negative"
    assert card["ok_to_submit"] is False
    assert card["safe_to_download_and_submit"] is False
    assert card["safe_to_download_for_inspection"] is True
    assert card["selector_approval"] is False
    assert card["weekly_slots_used"] == 0
    assert card["holdout_dti"]["evidence_label"] == "HOLDOUT-DTI"
    assert card["holdout_dti"]["evaluator_version"] == "gems57-pooled-hide-v2"
    seeds = card["holdout_dti"]["seeds"]
    assert seeds["20"]["withheld_positive_pixels"] == 11_321
    assert seeds["30"]["withheld_positive_pixels"] == 11_342
    # recorded point estimates; a negative lane must stay below the incumbent
    incumbent = 0.152718
    for key in ("20", "30"):
        s = seeds[key]
        assert s["dti"] == pytest.approx(0.0208 if key == "20" else 0.0206, abs=5e-4)
        assert s["ci95"][0] < s["dti"] < s["ci95"][1]
        assert s["dti"] < incumbent
        assert s["canary_sym"] <= 0.90
    assert card["leakage_canary"]["pass"] is True
    assert len(card["submission_note"]) <= 140
    assert len(card["submission_name"]) <= 140


def test_validator_receipt_passed_on_the_shipped_bytes():
    rec = json.loads(RECEIPT.read_text())
    assert rec["sha256"] == SHA_TIF
    assert rec["all_checks_passed"] is True
    assert rec["n_nan"] == 0 and rec["n_infinite"] == 0
    assert rec["emitted_positive_pixels"] == 37_654
    assert rec["on_catalogue_positive_pixels"] == 0
    assert "not organizer acceptance" in rec["validation_class"]


def test_pages_and_readme_state_download_yes_submit_no():
    for page in ("h86.html", "h86-executive-summary.html", "index.html"):
        txt = (ROOT / "docs" / page).read_text()
        assert NAME in txt, page
        assert SHA_TIF in txt, page
        for href in re.findall(r'(?:href|src)="([^"]+)"', txt):
            if href.startswith(("http://", "https://", "#", "mailto:")):
                continue
            assert (ROOT / "docs" / href).exists(), (page, href)
    for page in ("h86.html", "h86-executive-summary.html"):
        txt = (ROOT / "docs" / page).read_text()
        assert "OK TO DOWNLOAD FOR INSPECTION" in txt
        assert "NOT OK TO SUBMIT" in txt
    head = (ROOT / "README.md").read_text()[:8000]
    assert "H86-LENHALO" in head
    assert SHA_TIF in head
    assert "OK TO DOWNLOAD FOR INSPECTION: YES" in head
    assert "OK TO SUBMIT: NO" in head


def test_download_link_targets_resolve_to_the_shipped_file():
    txt = (ROOT / "docs" / "index.html").read_text()
    assert f"downloads/{NAME}.tif" in txt
    assert f"downloads/{NAME}.zip" in txt
    assert (ROOT / "docs" / "downloads" / f"{NAME}.tif").is_file()
