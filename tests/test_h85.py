"""Regression tests for the H85 controlled research artifact."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_h85_research import build_surface


ROOT = Path(__file__).resolve().parents[1]


def test_h85_surface_is_deterministic_and_masked():
    cat = np.zeros((80, 90), dtype=bool)
    cat[39:42, 8:82] = True
    footprint = np.ones_like(cat, dtype=bool)
    a, meta_a = build_surface(cat, footprint)
    b, meta_b = build_surface(cat, footprint)
    assert np.array_equal(a, b)
    assert meta_a == meta_b
    assert np.isfinite(a).all()
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert not a[cat].any()
    assert meta_a["positive_surface_pixels"] == int((a > 0).sum())


def test_h85_run_card_is_explicitly_inspection_only():
    card = json.loads((ROOT / "evidence/h85_run_card.json").read_text())
    assert card["lane"] == "H85-FTMR: far-field fault-network medial-ridge surface"
    assert card["ok_to_download_for_inspection"] is True
    assert card["ok_to_submit"] is False
    assert card["verdict"] == "negative"
    assert card["holdout_dti"]["score"] is None
    assert card["correlation_overlap_vs_registry"]["duplicate_stop"] is True
    assert card["validator_output"]["all_checks_passed"] is True
    assert len(card["submission_note"]) <= 140


def test_h85_artifact_matches_run_card():
    import hashlib
    import rasterio

    card = json.loads((ROOT / "evidence/h85_run_card.json").read_text())
    tif = ROOT / card["raster"]["file"]
    assert tif.is_file()
    digest = hashlib.sha256(tif.read_bytes()).hexdigest()
    assert digest == card["raster_sha256"]
    with rasterio.open(tif) as ds:
        arr = ds.read(1)
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert ds.crs.to_epsg() == 32611
        assert arr.shape == (3730, 3292)
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert ds.nodata is None
