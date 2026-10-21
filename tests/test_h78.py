"""H78 lane: the shipped bytes, the run card and the site agree (no data cache needed)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = json.loads((ROOT / "evidence/h78_run_card.json").read_text())
SUB = json.loads((ROOT / "docs/data/h78_submission.json").read_text())


def test_run_card_has_every_protocol_field():
    for k in ("hypothesis", "mechanism", "non_fault_mimic", "holdout", "registry_comparison", "raster_sha256",
              "validator", "submission_name", "note", "verdict"):
        assert CARD[k] not in (None, "", {}), k
    h = CARD["holdout"]
    assert h["evidence_class"] == "HOLDOUT-DTI" and h["evaluator_version"] and h["withheld_positive_pixels"] > 0
    assert h["ci95"][0] <= h["dti"] <= h["ci95"][1]
    assert len(CARD["note"]) <= 140 and CARD["verdict"] in ("promote", "negative")
    assert not CARD["canary_any_auc_above_090"]
    assert CARD["weekly_slots_used"] == 0 and CARD["organizer_confirmed_score"] is None
    assert CARD["approved_for_weekly_slot"] is False


def test_verdict_follows_the_preregistered_rule():
    d = CARD["paired_difference_vs_champion_construction"]
    beats = d["ci95"][0] > 0
    assert CARD["beats_holdout_best"] == beats
    if not beats:
        assert CARD["verdict"] == "negative"


def test_uniqueness_informative_reading():
    rc = CARD["registry_comparison"]
    assert rc["phase1_worst_rho"] <= 0.90 and rc["phase2_worst_rho"] <= 0.90
    assert rc["phase2_worst_overlap_informative"] <= 0.70
    assert rc["phase2_overlap_firings_informative"] == 0 and rc["identical_decoded"] == 0
    assert rc["literal_firings_all_space_filling"] is True
    assert CARD["protocol_unique_informative_reading"] is True
    # repository policy I17: the literal reading binds, so space-filling firings make the file NOT protocol-unique
    assert CARD["protocol_unique"] is (CARD["literal_reading_firings_final_dots"] == 0)
    assert CARD["ok_to_submit"] is False


def test_shipped_tif_matches_card_and_is_portal_shaped():
    rasterio = pytest.importorskip("rasterio")
    p = ROOT / CARD["file"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == CARD["raster_sha256"] == SUB["primary"]["sha256"]
    with rasterio.open(p) as src:
        assert src.count == 1 and src.dtypes[0] == "float32" and src.nodata is None
        assert (src.height, src.width) == (3730, 3292) and src.crs.to_epsg() == 32611
        assert tuple(src.transform)[:6] == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
        a = src.read(1)
    assert np.isfinite(a).all() and np.isin(a, [0.0, 1.0]).all()
    assert int((a > 0).sum()) == SUB["dots"]


def test_zip_holds_exactly_the_tif():
    import zipfile
    z = ROOT / "docs/downloads" / SUB["zip"]["zip"]
    with zipfile.ZipFile(z) as f:
        assert f.namelist() == [SUB["primary"]["file"]]
        assert f.read(SUB["primary"]["file"]) == (ROOT / CARD["file"]).read_bytes()


def test_lanes_and_home_page_show_h78_status():
    lanes = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"]
    row = next(x for x in lanes if x["lane"].startswith("H78"))
    assert row["sha256"] == CARD["raster_sha256"] and row["verdict"] == CARD["verdict"]
    home = (ROOT / "docs/index.html").read_text()
    assert CARD["submission_name"] in home and CARD["raster_sha256"] in home
    readme = (ROOT / "README.md").read_text()
    # New lanes may precede H78; keep its historic file/hash documented without a brittle byte offset.
    assert CARD["submission_name"] in readme and CARD["raster_sha256"] in readme
