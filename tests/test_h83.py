"""H83-NETSIB: feature geometry, leakage hygiene, run-card/byte binding, site wiring."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems83 import netsib as NS  # noqa: E402

CARD = json.loads((ROOT / "evidence/h83_run_card.json").read_text())


def _toy():
    """Two parallel E-W traces with a gap in one of them, on a 120x160 grid."""
    m = np.zeros((120, 160), bool)
    m[40, 10:60] = True
    m[40, 80:150] = True          # collinear continuation after a 20 px gap
    m[52, 10:150] = True          # sibling 12 px away
    fp = np.ones_like(m)
    return m, fp


def test_strike_field_recovers_ew_traces():
    m, _ = _toy()
    th = NS.strike_field(m)
    v = th[40, 20:50]
    # E-W trace => theta ~ 0 or ~ pi
    assert np.all(np.minimum(v, np.pi - v) < 0.2)


def test_kernels_are_normalised_and_oriented():
    kf, kb, ks = NS.kernels(0.0)
    for k in (kf, kb, ks):
        assert abs(float(k.sum()) - 1.0) < 1e-5 and (k >= 0).all()
    r = NS.R_KERNEL
    assert kf[r, r + 5] > 0 and kf[r, r - 5] == 0      # forward = +x
    assert kb[r, r - 5] > 0 and kb[r, r + 5] == 0
    assert ks[r, r] == 0 and ks[r + 10, r] > 0          # sibling band across strike


def test_gap_channel_peaks_in_collinear_gap():
    m, fp = _toy()
    f = NS.netsib_fields(m, fp)
    assert set(f) == set(NS.NAMES)
    gap_in = f["ns_gap"][40, 70]
    gap_off = f["ns_gap"][90, 70]                        # far from every trace
    assert gap_in > 10 * max(gap_off, 1e-9)
    assert f["ns_sib"][46, 30] > f["ns_sib"][100, 30]
    for v in f.values():
        assert v.dtype == np.float32 and np.isfinite(v).all()


def test_fields_zero_outside_footprint_and_reject_empty():
    m, fp = _toy()
    fp = fp.copy(); fp[:, :5] = False
    f = NS.netsib_fields(m, fp)
    assert all((v[:, :5] == 0).all() for v in f.values())
    with pytest.raises(ValueError):
        NS.netsib_fields(np.zeros_like(m), fp)


def test_fields_depend_only_on_the_state_mask():
    """Leakage hygiene: identical state => identical features, whatever was hidden."""
    m, fp = _toy()
    a = NS.netsib_fields(m, fp)
    b = NS.netsib_fields(m.copy(), fp.copy())
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_synthetic_withhold_whole_components_and_collar():
    rng = np.random.default_rng(0)
    m = np.zeros((200, 200), bool)
    for _ in range(60):
        r, c = rng.integers(5, 195, 2)
        m[r, max(c - 6, 0):c + 6] = True
    fp = np.ones_like(m)
    quad = (np.arange(200)[:, None] >= 100) * 2 + (np.arange(200)[None, :] >= 100)
    hidden, state = NS.synthetic_withhold(m, fp, quad, seed=3)
    assert hidden.any() and not (hidden & state).any()
    assert not (hidden & ~m).any()
    from scipy import ndimage as ndi
    comp, _ = ndi.label(m, np.ones((3, 3), bool))
    for lab in np.unique(comp[hidden]):
        assert hidden[comp == lab].all()                 # whole components only
    d = ndi.distance_transform_edt(~hidden)
    assert not (state & (d <= 3)).any()                  # 3 px collar removed
    h2, _ = NS.synthetic_withhold(m, fp, quad, seed=3)
    assert np.array_equal(hidden, h2)                    # deterministic


def test_deterministic_score_is_rank_in_unit_interval():
    m, fp = _toy()
    s = NS.deterministic_score(NS.netsib_fields(m, fp), fp)
    assert s.min() >= 0 and s.max() <= 1


# ---------------------------------------------------------------- run card / bytes

def test_run_card_binds_actual_bytes_and_note():
    tif = ROOT / CARD["download"]
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == CARD["raster_sha256"]
    sub = ROOT / "submissions" / f"{CARD['submission_name']}.tif"
    if sub.exists():
        assert sub.read_bytes() == tif.read_bytes()
    assert tif.stem == CARD["submission_name"]
    assert len(CARD["note"]) <= 140


def test_validator_passes_on_actual_bytes():
    from gems57.grid import load_grid
    from gems57.validate import validate
    g = load_grid(ROOT / "data" / "bridge")
    v = validate(ROOT / CARD["download"], g.footprint, g.catalogue)
    assert v["all_checks_passed"] and v["min"] >= 0 and v["max"] <= 1 and v["n_nan"] == 0


def test_zip_holds_exactly_one_identical_tiff():
    import zipfile
    with zipfile.ZipFile(ROOT / CARD["zip"]) as z:
        names = z.namelist()
        assert len(names) == 1 and names[0].endswith(".tif")
        assert hashlib.sha256(z.read(names[0])).hexdigest() == CARD["raster_sha256"]


def test_verdict_is_mechanical_and_negative_because_duplicate_fired():
    b = json.loads((ROOT / "evidence/h83_e3_build.json").read_text())
    fd = b["final_dot_uniqueness"]
    assert fd["duplicate"] is True and fd["identical_pixels_any"] is False
    assert fd["max_forward_overlap_informative"] > 0.70
    assert CARD["verdict"] == "negative" and CARD["ok_to_submit"] is False
    assert CARD["promotion_rule_results"]["d_uniqueness"] is False
    assert "DUPLICATE" in CARD["verdict_reason"]


def test_self_comparison_removed_from_registry_rows():
    """IR-83-02 regression: the candidate must never be compared with itself."""
    rows = json.loads((ROOT / "evidence/h83_e3_registry_rows.json").read_text())
    name = CARD["submission_name"]
    assert not any(name in r["raster"] for r in rows)
    assert not any(name in str(r.get("same_decoded_as", "")) for r in rows)
    assert not any(r.get("identical_pixels") for r in rows)
    src = (ROOT / "scripts/h83_build.py").read_text()
    assert "exclude=" in src and "IR-83-02" in src


def test_holdout_numbers_labelled_and_consistent():
    e2 = json.loads((ROOT / "evidence/h83_e2_holdout.json").read_text())
    assert e2["evaluator_version"] == "gems57-pooled-hide-v2"
    for s in ("20", "30"):
        c = CARD["holdout_dti"][s]
        assert c["evaluator_version"] == "gems57-pooled-hide-v2"
        assert c["withheld_positive_pixels"] > 10_000
        lo, hi = c["candidate_A_netsib_full"]["ci95"]
        assert lo < c["candidate_A_netsib_full"]["dti"] < hi
    e1 = json.loads((ROOT / "evidence/h83_e1_canary_surface.json").read_text())
    assert all(e1["canaries"][s]["max_auc"] <= 0.90 for s in ("20", "30"))


# ---------------------------------------------------------------- site wiring

def test_site_hero_and_guide_fail_closed():
    for f in ("index.html", "executive-summary.html", "h83.html"):
        s = (ROOT / "docs" / f).read_text()
        assert s.count("NEWEST (2026-10-21) · H83-NETSIB") == 1
        assert "NOT OK TO SUBMIT" in s
        assert "OK TO DOWNLOAD · ✅ OK TO SUBMIT" not in s
    g = (ROOT / "docs/h83-executive-summary.html").read_text()
    assert "Predicted values must be in range [0, 1]" in g and "single-band GeoTIFF" in g


def test_lanes_row_and_release_gate():
    rows = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"]
    r = next(x for x in rows if x["lane"].startswith("H83-NETSIB"))
    assert r["verdict"] == "negative" and r["protocol_unique"] is False
    assert "Not OK to submit" in r["upload"] and r["sha256"] == CARD["raster_sha256"]
    auth = json.loads((ROOT / "docs/data/submission-authorization.json").read_text())
    assert "h83" in json.dumps(auth)
    assert auth["repository_authorized_file_count"] == 0 and auth["repository_decision"] == "NOT OK TO SUBMIT"
