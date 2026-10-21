"""H77-XSEN lane: the generated TIF is format-valid, honestly labelled negative.

These tests pin the deliverable contract (portal range error cannot recur)
without ever asserting approval: verdict must stay negative / not OK to submit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = "gems77-xsen-coincide-hgb-37654dots-20261021-zeros"
TIF = ROOT / "submissions" / f"{NAME}.tif"


def _card():
    return json.loads((ROOT / "evidence" / "run_card_xsen.json").read_text())


def test_xsen_tif_format_and_hash():
    import rasterio
    assert TIF.exists()
    assert hashlib.sha256(TIF.read_bytes()).hexdigest() == _card()["submission_raster"]["sha256"]
    with rasterio.open(TIF) as s:
        assert s.count == 1 and s.dtypes[0] == "float32"
        assert (s.height, s.width) == (3730, 3292)
        assert s.crs.to_epsg() == 32611
        assert tuple(s.transform)[:6] == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
        assert s.nodata is None
        a = s.read(1)
    assert np.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert int((a > 0).sum()) == 37654
    with rasterio.open(ROOT / "data" / "bridge" / "existing_faults.tif") as s:
        cat = s.read(1) > 0
    with rasterio.open(ROOT / "data" / "bridge" / "sample_submission.tif") as s:
        foot = np.isfinite(s.read(1))
    assert int(((a > 0) & cat).sum()) == 0
    assert int(((a > 0) & ~foot).sum()) == 0
    assert (ROOT / "docs" / "downloads" / f"{NAME}.tif").read_bytes() == TIF.read_bytes()


def test_xsen_verdict_negative_and_note_within_limit():
    card = _card()
    assert card["verdict"] == "negative" and card["ok_to_submit"] is False
    assert card["submission_name"] == NAME
    assert len(card["submission_note"]) <= 140
    assert card["holdout_dti"]["promotion_bar_met"] is False
    h = card["holdout_dti"]
    assert h["evidence_class"] == "HOLDOUT-DTI" and h["withheld_positive_pixels"] > 0
    assert len(h["candidate_A_xsen"]["ci95"]) == 2
    assert card["uniqueness_vs_registry"]["phase2_final_dots"]["overlap_gate_pass"] is False
    lanes = json.loads((ROOT / "docs" / "data" / "lanes.json").read_text())["lanes"]
    xsen = [x for x in lanes if x.get("name") == NAME]
    assert len(xsen) == 1 and xsen[0]["protocol_unique"] is False
    assert "Not OK to submit" in xsen[0]["upload"]


def test_validate_submission_cli_uses_bridge_grid():
    src = (ROOT / "scripts" / "validate_submission.py").read_text()
    assert "data'/'bridge'" in src or 'data" / "bridge"' in src or "data/bridge" in src
