"""H82-RRD: field-operator unit tests + artifact/run-card consistency (fast, no data/raw needed)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_rrd_perfect_linear_pair_has_no_discordance():
    from gems82.fields import build_rrd
    rng = np.random.default_rng(0)
    foot = np.ones((60, 50), bool)
    tmi = rng.normal(size=(60, 50)).astype(np.float32)
    rtp = (2.0 * tmi + 1.0).astype(np.float32)
    mag = rng.normal(size=(60, 50)).astype(np.float32)
    ch = build_rrd(tmi, rtp, mag, foot)
    inner = np.s_[10:-10, 10:-10]
    assert float(np.median(ch["rrd_corr"][inner])) > 0.999
    assert float(np.median(ch["rrd_resid_rms"][inner])) < 1e-5
    assert abs(float(np.median(ch["rrd_slope"][inner])) - 2.0) < 1e-4
    for v in ch.values():
        assert np.isfinite(v).all()


def test_rrd_discordant_pair_has_residual():
    from gems82.fields import build_rrd
    rng = np.random.default_rng(1)
    foot = np.ones((60, 50), bool)
    tmi = rng.normal(size=(60, 50)).astype(np.float32)
    rtp = rng.normal(size=(60, 50)).astype(np.float32)  # independent -> discordant
    mag = rng.normal(size=(60, 50)).astype(np.float32)
    ch = build_rrd(tmi, rtp, mag, foot)
    assert float(np.median(ch["rrd_resid_rms"][10:-10, 10:-10])) > 0.1
    assert ch["S_rrd"].min() >= 0.0 and ch["S_rrd"].max() <= 1.0


@pytest.mark.skipif(not (ROOT / "evidence/h82_run_card.json").exists(), reason="run card not built")
def test_h82_run_card_consistency():
    card = json.loads((ROOT / "evidence/h82_run_card.json").read_text())
    assert card["verdict"] == "negative"
    assert card["ok_to_submit"] is False
    assert card["weekly_slots_used"] == 0
    assert card["organizer_confirmed_score"] is None
    assert len(card["note"]) <= 140 and len(card["submission_name"]) <= 140
    h = card["holdout_dti"]
    assert h["evidence_class"] == "HOLDOUT-DTI"
    assert h["evaluator_version"] == "gems57-pooled-hide-v2"
    assert h["withheld_positive_pixels"] == 11321
    assert h["winner_ci95"][0] <= h["winner_arm_A_std19_rrd"] <= h["winner_ci95"][1]
    tif = ROOT / card["file"]
    assert tif.exists()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    v = card["validator"]
    assert v["all_checks_passed"] and v["n_nan"] == 0 and v["min"] >= 0.0 and v["max"] <= 1.0
    assert card["leakage_canary"]["max_sym_auc"] <= 0.90
    # Negative must be for the recorded reasons: below incumbent AND overlap fired.
    assert h["winner_arm_A_std19_rrd"] < h["incumbent"]["dti"]
    assert card["correlation_overlap_vs_registry"]["phase2_max_overlap_informative_lt70_halo"] > 0.70
