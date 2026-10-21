"""Lane H79: attribute rasters, fold-blindness of the context features, status policy, evidence consistency."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(tmp_path):
    """60 x 60 grid with two N-S traces (col 20 dips E; col 40 unspecified) on a westward-rising ramp."""
    from gems79 import attrs as AT
    h = w = 60
    X0, Y0, R = AT.X0, AT.Y0, AT.RES
    rows = []
    for rid, col in ((0, 20), (1, 40)):
        for r in range(5, 54):
            x, y0, y1 = X0 + col * R, Y0 - r * R, Y0 - (r + 1) * R
            rows.append(dict(record_id=rid, x0=x, y0=y0, x1=x, y1=y1, slip_mm_yr=0.1, slip_text="0.1", sense="N"))
    seg = tmp_path / "seg.csv"
    pd.DataFrame(rows).to_csv(seg, index=False)
    att = tmp_path / "att.csv"
    pd.DataFrame([
        dict(NAME="A", DIPDIRECT="E", RCODE2023=2, SCODE2023=4, FCODE2023=1, MAPSCALE=100, SLIPRTNUM=0.1),
        dict(NAME="B", DIPDIRECT="Unspecified", RCODE2023=5, SCODE2023=4, FCODE2023=2, MAPSCALE=250, SLIPRTNUM=0.001),
    ]).to_csv(att, index=False)
    cat = np.zeros((h, w), bool)
    cat[5:55, 20] = True
    cat[5:55, 40] = True
    foot = np.ones((h, w), bool)
    det = np.tile(np.linspace(100.0, 0.0, w, dtype=np.float32), (h, 1))  # higher in the west, lower in the east
    A = AT.build_attribute_rasters((h, w), cat, det, seg, att)
    return AT, A, cat, foot


def test_attribute_rasters_and_hanging_wall_sign(tmp_path):
    AT, A, cat, foot = _synthetic(tmp_path)
    rc = A["receipt"]
    assert rc["catalogue_pixels"] == int(cat.sum()) == rc["catalogue_pixels_with_record"]
    assert rc["pixels_with_dipdirect"] == 50  # only trace A carries a dip direction
    assert rc["topo_vs_attr_hangingwall_agreement"] == 1.0  # east is lower, trace A dips east
    rows = np.array([30 * 60 + 25, 30 * 60 + 15])  # east and west of trace A (col 20)
    X = AT.context_features(cat, A, foot, rows=rows)
    j_attr, j_topo, j_d = (AT.CTX_NAMES.index(n) for n in ("ctx_hw_attr", "ctx_hw_topo", "ctx_dvis"))
    assert X[0, j_attr] > 0 and X[1, j_attr] < 0          # + on the hanging wall (east), - on the footwall
    assert X[0, j_topo] > 0 and X[1, j_topo] < 0
    assert X[0, j_d] == pytest.approx(5.0) and X[1, j_d] == pytest.approx(5.0)
    assert X[0, AT.CTX_NAMES.index("ctx_rcode_near")] == 2
    assert X[0, AT.CTX_NAMES.index("ctx_mapscale_near")] == 100
    # a pixel nearest to trace B (unspecified dip) has NaN hw_attr but a defined hw_topo
    Xb = AT.context_features(cat, A, foot, rows=np.array([30 * 60 + 45]))
    assert np.isnan(Xb[0, j_attr]) and Xb[0, j_topo] > 0
    assert np.isfinite(np.delete(Xb[0], j_attr)).all()


def test_context_features_depend_only_on_visible_mask(tmp_path):
    AT, A, cat, foot = _synthetic(tmp_path)
    visible = cat.copy()
    visible[:, 40] = False  # hide trace B entirely
    rows = np.array([r * 60 + c for r in range(5, 55, 3) for c in range(0, 60, 7)])  # inside the traces' row span
    X1 = AT.context_features(visible, A, foot, rows=rows)
    # distance / attributes must now point to trace A everywhere, never to the hidden trace B
    j_d = AT.CTX_NAMES.index("ctx_dvis")
    cols = rows % 60
    assert np.allclose(X1[:, j_d], np.minimum(np.abs(cols - 20), AT.CLIP_PX), atol=1.01)
    assert (X1[:, AT.CTX_NAMES.index("ctx_rcode_near")] == 2).all()
    # and sub_hide never leaves a visible pixel inside the 3 px collar of a hidden component
    rng = np.random.default_rng(0)
    hid, vis_after = AT.sub_hide(cat, np.ones_like(cat), rng, frac=1.0, buffer_px=3)
    assert hid.sum() == cat.sum() and vis_after.sum() == 0
    hid, vis_after = AT.sub_hide(cat, np.ones_like(cat), np.random.default_rng(3), frac=0.5, buffer_px=3)
    from scipy import ndimage
    if hid.any() and vis_after.any():
        d = ndimage.distance_transform_edt(~hid)
        assert d[vis_after].min() > 3.0
    assert not (hid & vis_after).any()


def test_status_policy_is_mechanical_and_consistent():
    import build_home as BH
    assert BH.status_of({"quarantined": True, "format_valid": True, "informative_unique": True, "beats_holdout_best": True}) == "QUARANTINED"
    assert BH.status_of({"format_valid": True, "protocol_unique_like_for_like": True, "beats_holdout_best": True}) == "CLEARED_FOR_SELECTOR"
    assert BH.status_of({"format_valid": True, "protocol_unique_like_for_like": True, "beats_holdout_best": False}) == "DOWNLOAD_ONLY"
    assert BH.status_of({"format_valid": True, "protocol_unique_like_for_like": False, "beats_holdout_best": True}) == "DOWNLOAD_ONLY"
    lanes = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"]
    for x in lanes:
        st = BH.status_of(x)
        if st == "CLEARED_FOR_SELECTOR":
            assert x["upload"].startswith("OK to submit if the selector picks it"), x["file"]
        else:
            assert x["upload"].startswith("Not OK to submit"), x["file"]
        assert x.get("protocol_unique") is not True or x.get("literal_reading_measured_pass"), "literal PASS must be measured"


def test_h79_run_card_matches_evidence():
    card = json.loads((ROOT / "evidence/h79_run_card.json").read_text())
    assert card["weekly_slots_used"] == 0 and card["live_score"] is None and not card["organizer_confirmed"]
    assert card["ok_to_submit"] == (card["status"] == "CLEARED_FOR_SELECTOR")
    s20 = json.loads((ROOT / "evidence/h79_holdout_seed20.json").read_text())
    assert s20["leakage_flags"] == [] and s20["canary_max_symmetric_auc"] < 0.90
    c = s20["budgets"]["B37654"]["pooled_summary"]["scores"]["H79_full"]
    assert card["holdout"]["results"]["seed20"]["candidate"]["dti"] == pytest.approx(c["dti"])
    # the deliverable's audit is the one that decides its status (capped build if the amendment ran)
    tag = "h79c" if card.get("capped_emission_amendment_e3b") else "h79"
    u = json.loads((ROOT / f"evidence/{tag}_uniqueness.json").read_text())
    assert u["sha256"] == card["artifact"]["sha256"]
    lfl = (u["phase2_final_dots"]["max_rho"] <= 0.90 and u["phase1_surface_before_placement"]["max_rho"] <= 0.90
           and u["max_forward_overlap_discriminating"] <= 0.70)
    assert u["protocol_unique_like_for_like"] == lfl
    assert u["protocol_unique_literal"] is False  # non-discriminating rasters make the literal reading fail for any dot set
    if card["status"] == "CLEARED_FOR_SELECTOR":
        assert lfl and card["artifact"]["validator_all_checks_passed"]
    first = card.get("first_build_logged_duplicate")
    if first:  # the greedy build is recorded as a duplicate and is never the deliverable
        u0 = json.loads((ROOT / "evidence/h79_uniqueness.json").read_text())
        assert u0["max_forward_overlap_discriminating"] > 0.70 and u0["sha256"] == first["sha256"] != card["artifact"]["sha256"]
        cap = card["capped_emission_amendment_e3b"]
        for sd in ("seed20", "seed30"):
            assert cap["holdout"][sd]["H79_full@capped"] < cap["holdout"][sd]["H79_full@greedy"]  # the cost is visible
    lane = next(x for x in json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"] if x["page"] == "h79.html")
    assert lane["sha256"] == card["artifact"]["sha256"] and lane["status"] == card["status"] and lane["upload"] == card["upload"]
    halo = json.loads((ROOT / "registry/registry_halo_coverage.json").read_text())
    assert halo["n_universal_blockers"] >= 1 and halo["n_discriminating_at_0_70"] > 0
    site = json.loads((ROOT / "docs/data/h79_submission.json").read_text())
    assert site["primary"]["sha256"] == card["artifact"]["sha256"] and len(site["primary"]["note"]) <= 140


@pytest.mark.skipif(not (ROOT / "data/raw/sample_submission.tif").exists(), reason="restore data first")
def test_h79_tif_passes_shared_validator():
    from gems57 import validate as V
    from gems57.grid import load_grid
    card = json.loads((ROOT / "evidence/h79_run_card.json").read_text())
    g = load_grid(ROOT / "data/raw")
    rec = V.validate(ROOT / card["artifact"]["file"], g.footprint, g.catalogue)
    V.assert_submittable(rec)
    assert rec["sha256"] == card["artifact"]["sha256"]
