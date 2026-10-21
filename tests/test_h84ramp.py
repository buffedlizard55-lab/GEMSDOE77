"""Tests for the H84-RAMP lane (B-LBCO deconfounded two-host relay-ramp & geodetic-geophysical transfer)."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems84.fields import (  # noqa: E402
    H84_NAMES,
    damage_collar_profile,
    extract_h84_from_frame,
    robust_z,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_h84_field_operators_and_self_collapse_tautology():
    """Verify IR-H84-01: d1==0 collapses 7/8 channels to 0, while d1>2 inside a facing corridor activates them."""
    assert len(H84_NAMES) == 8
    d_test = np.array([0.0, 1.0, 3.0, 9.0, 20.0], dtype=np.float32)
    col = damage_collar_profile(d_test)
    assert col[0] == 0.0 and col[1] == 0.0
    assert np.isclose(col[2], 1.0, atol=1e-5)
    assert 0.0 < col[3] < 1.0 and col[4] == 0.0

    # Synthetic 2-pixel test: pixel 0 is ON visible fault (d1=0), pixel 1 is in a facing stepover (d1=5, d2=6)
    rows = np.array([10, 10], dtype=np.int32)
    cols = np.array([10, 15], dtype=np.int32)
    d1 = np.array([0.0, 5.0], dtype=np.float32)
    ay1 = np.array([10, 10], dtype=np.int32)
    ax1 = np.array([10, 10], dtype=np.int32)
    c1 = np.array([1, 1], dtype=np.int32)
    d2 = np.array([11.0, 6.0], dtype=np.int32).astype(np.float32)
    ay2 = np.array([10, 10], dtype=np.int32)
    ax2 = np.array([21, 21], dtype=np.int32)
    c2 = np.array([2, 2], dtype=np.int32)
    comp_len = np.array([0.0, 120.0, 150.0], dtype=np.float32)
    strike_res = np.full((30, 30), 25.0, dtype=np.float32)
    bend_turn = np.full((30, 30), 0.6, dtype=np.float32)
    geod_pts = np.array([0.7, 0.7], dtype=np.float32)
    base_pts = np.array([0.65, 0.65], dtype=np.float32)

    mat = extract_h84_from_frame(
        rows, cols, d1, ay1, ax1, c1, d2, ay2, ax2, c2,
        comp_len, strike_res, bend_turn, geod_pts, base_pts
    )
    assert mat.shape == (2, 8)
    assert np.isfinite(mat).all()
    # Pixel 0 (d1 == 0): 7 of 8 channels collapse identically to 0.0 (only coplanar_sync at index 2 is non-zero)
    for j in (0, 1, 3, 4, 5, 6, 7):
        assert mat[0, j] == 0.0, f"Channel {H84_NAMES[j]} should collapse at d1=0"
    # Pixel 1 (d1=5, d2=6, opposite anchors along row 10 -> facing = +1): all 8 channels strictly positive
    for j in range(8):
        assert mat[1, j] > 0.0, f"Channel {H84_NAMES[j]} should be > 0 inside stepover corridor"

    z = robust_z(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32), np.array([1, 1, 1, 1, 0], dtype=bool))
    assert z[4] == 0.0 and np.isfinite(z).all()


def test_h84_preregistration_and_evidence_receipts():
    prereg = json.loads((ROOT / "registry" / "preregistration_h84ramp.json").read_text())
    assert prereg["lane_id"] == "H84-RAMP"
    assert len(prereg["ranked_hypotheses"]) == 5
    assert prereg["ranked_hypotheses"][0]["implemented_in_pass_1"] is True

    hyp_md = (ROOT / "research" / "h84ramp_hypotheses.md").read_text()
    assert "0.2778" in hyp_md and "0.4413" in hyp_md and "IR-H84-01" in hyp_md

    e1 = json.loads((ROOT / "evidence" / "h84ramp_e1_canary_surface.json").read_text())
    assert e1["leakage_pass"] is True
    assert e1["surface_spearman_gate_pass"] is True
    for seed in ("20", "30"):
        sc = e1["canaries_by_seed"][seed]
        assert sc["max_h84_symmetric_auc"] <= 0.90
        assert sc["max_std_symmetric_auc"] <= 0.90
        taut = e1["irregularity_IR_H84_01_self_collapse_tautology"][seed]
        assert taut["naive_visible_positive_means"]["h84_relay_medial_bridge"] == 0.0
        assert taut["blbco_deconfounded_visible_positive_means"]["h84_relay_medial_bridge"] > 0.05

    e2 = json.loads((ROOT / "evidence" / "h84ramp_e2_holdout.json").read_text())
    assert e2["evaluator_version"] == "gems57-pooled-hide-v2"
    for seed, exp_pos in (("20", 11321), ("30", 11342)):
        s = e2["seeds"][seed]["summary"]
        scores = s["scores"]
        diffs = s["paired_differences"]
        assert scores["A_std_plus_h84"]["withheld_positive_pixels"] == exp_pos
        assert scores["A_std_plus_h84"]["dti"] > 0.13
        assert scores["S_h84_physics"]["dti"] > 0.14
        # B-LBCO beats naive (IR-H84-01), std19 control, and random with 95% CIs excluding 0
        for ctrl in ("A_std_plus_h84_naive", "A_std_control", "A0_random"):
            assert diffs[ctrl]["delta"] > 0.04
            assert diffs[ctrl]["ci95"][0] > 0.0


def test_h84_run_card_geotiff_and_site_integration():
    rc = json.loads((ROOT / "evidence" / "h84ramp_run_card.json").read_text())
    assert rc["lane"] == "H84-RAMP"
    assert rc["verdict"] == "negative"
    assert rc["ok_to_download"] is True
    assert rc["ok_to_submit"] is False
    assert rc["selector_approved"] is False
    assert rc["weekly_slots_used"] == 0
    assert 1 <= len(rc["submission_note"]) <= 140
    assert 1 <= len(rc["submission_name"]) <= 140

    uq = rc["correlation_overlap_vs_registry"]
    assert uq["protocol_unique"] is False
    assert uq["informative_unique"] is True
    assert uq["informative_sparse_summary"]["max_forward_overlap_3px"] <= 0.70
    assert uq["informative_sparse_summary"]["max_abs_spearman"] <= 0.90
    assert uq["vs_h33_champion"]["my_dots_within_3px_of_theirs"] <= 0.70
    assert uq["vs_dense_witness"]["my_dots_within_3px_of_theirs"] == 1.0

    tif_dl = ROOT / rc["raster_path"]
    tif_sub = ROOT / "submissions" / f"{rc['submission_name']}.tif"
    zip_dl = ROOT / rc["zip_path"]
    assert tif_dl.is_file() and tif_sub.is_file() and zip_dl.is_file()
    assert _sha256(tif_dl) == rc["raster_sha256"] == _sha256(tif_sub)
    assert _sha256(zip_dl) == rc["zip_sha256"]

    with zipfile.ZipFile(zip_dl) as zf:
        assert zf.namelist() == [tif_dl.name]

    with rasterio.open(ROOT / "data" / "bridge" / "sample_submission.tif") as s_ref, rasterio.open(
        ROOT / "data" / "bridge" / "existing_faults.tif"
    ) as s_cat, rasterio.open(tif_dl) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.shape == (3730, 3292)
        assert src.crs.to_epsg() == 32611
        assert src.transform == s_ref.transform
        arr = src.read(1)
        foot = np.isfinite(s_ref.read(1))
        cat = s_cat.read(1) > 0
        assert np.isfinite(arr).all()
        assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
        assert np.all((arr == 0.0) | (arr == 1.0))
        assert int((arr > 0).sum()) == 37654
        assert int((arr[~foot] > 0).sum()) == 0
        assert int((arr[cat] > 0).sum()) == 0

    lanes = json.loads((ROOT / "docs" / "data" / "lanes.json").read_text())["lanes"]
    h84_lanes = [x for x in lanes if x["lane"].startswith("H84-RAMP:")]
    assert len(h84_lanes) == 1
    assert h84_lanes[0]["sha256"] == rc["raster_sha256"]
    assert h84_lanes[0]["verdict"] == "negative"
    assert h84_lanes[0]["protocol_unique"] is False

    for page_name in ("h84-ramp.html", "h84-ramp-executive-summary.html"):
        html_txt = (ROOT / "docs" / page_name).read_text()
        assert 'lang="en"' in html_txt
        assert "not approved" in html_txt.lower()
        assert rc["raster_sha256"] in html_txt
