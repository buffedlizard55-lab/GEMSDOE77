import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_h79_negative_artifact_and_site_are_unambiguous():
    card = json.loads((ROOT / "evidence/h79_radhalo_run_card.json").read_text())
    tif = ROOT / "submissions" / f"{card['submission_name']}.tif"
    assert tif.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    assert card["ok_to_download"] is True
    assert card["ok_to_submit"] is False
    assert card["verdict"] == "negative"
    assert card["submission_slots_used"] == 0
    assert len(card["submission_note"]) <= 140
    assert card["holdout_dti"]["label"] == "HOLDOUT-DTI"
    assert card["holdout_dti"]["evaluator_version"] == "gems57-pooled-hide-v2"
    assert card["holdout_dti"]["beats_current_best"] is False
    assert card["correlation_overlap_vs_registry"]["duplicate_stop"] is True
    assert card["validator_output"]["all_checks_passed"] is True
    page = (ROOT / "docs/h79-radhalo.html").read_text()
    assert "OK to download" in page
    assert "NOT OK to submit" in page
    assert card["raster_sha256"] in page
