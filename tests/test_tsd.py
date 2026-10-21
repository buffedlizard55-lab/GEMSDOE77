"""End-to-end verification tests for Lane H77-TSD (Transtensional Strain-Partitioning
x Seismic Swarm Excess x MT Clay-Cap Margin + 12-Bit LOCO Relay Mechanics)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_tsd_preregistration_and_hypotheses_exist() -> None:
    prereg = json.loads((ROOT / "registry" / "preregistration_tsd.json").read_text())
    assert prereg["lane_id"] == "H77-TSD"
    assert len(prereg["static_tsd_channels"]) == 10
    assert len(prereg["fold_visible_loco_channels"]) == 5
    assert prereg["experiment_budget"]["max_experiments"] == 3

    hyp_md = (ROOT / "research" / "hypotheses-tsd-2026-10-21.md").read_text()
    for hid in ("H77-TSD", "H77-WORM", "H77-2MTP", "H77-3DEP-KNICK", "H77-MT3D"):
        assert hid in hyp_md
    assert "0.2778" in hyp_md
    assert "0.4413" in hyp_md


def test_tsd_exp1_canary_and_screen() -> None:
    e1 = json.loads((ROOT / "evidence" / "tsd_exp1_canary_and_screen.json").read_text())
    assert e1["evidence_class"] == "HOLDOUT-DTI"
    assert e1["evaluator_version"] == "gems57-pooled-hide-v2"
    assert e1["withheld_positive_pixels"] == 11342
    assert len(e1["canaries"]) == 15
    assert e1["max_single_feature_symmetric_auc"] <= 0.90
    assert e1["leakage_gt_0p90"] == []
    assert e1["arms"]["S_tsd_physics"]["pooled_holdout_dti"] > e1["arms"]["A0_random"]["pooled_holdout_dti"]


def test_tsd_exp2_dual_seed_holdout() -> None:
    e2 = json.loads((ROOT / "evidence" / "tsd_exp2_holdout.json").read_text())
    assert e2["evidence_class"] == "HOLDOUT-DTI"
    assert e2["evaluator_version"] == "gems57-pooled-hide-v2"
    for skey, exp_pos in (("seed_30", 11342), ("seed_20", 11321)):
        s = e2[skey]
        assert s["withheld_positive_pixels"] == exp_pos
        sc = s["pooled_summary"]["scores"]
        diffs = s["pooled_summary"]["paired_differences"]
        assert sc["A_tsd_full"]["dti"] > 0.13
        for ctrl in ("A_std_greedy", "A_tsd_static_only", "A_tsd_greedy", "A_tsd_mech_only", "A0_random"):
            assert diffs[ctrl]["delta"] > 0.0
            assert diffs[ctrl]["ci95"][0] > 0.0


def test_tsd_exp3_uniqueness_and_run_card() -> None:
    rc = json.loads((ROOT / "evidence" / "run_card_tsd.json").read_text())
    uq = json.loads((ROOT / "evidence" / "tsd_exp3_uniqueness.json").read_text())
    assert rc["lane_id"] == "H77-TSD"
    assert rc["verdict"] == "negative"
    assert rc["protocol_unique"] is False
    assert rc["safe_to_download"] is True
    assert rc["safe_to_submit"] is False
    assert uq["phase1_surface_max_spearman_abs"] <= 0.90
    assert uq["phase2_dot_max_spearman_abs"] <= 0.90
    assert uq["phase2_informative_max_forward_overlap_3px"] <= 0.70
    assert uq["informative_sparse_gates_pass"] is True
    assert uq["phase2_literal_max_forward_overlap_3px"] == 1.0

    sub = rc["submission"]
    assert 1 <= len(sub["note"]) <= 140
    tif_path = ROOT / sub["tif_path"]
    zip_path = ROOT / sub["zip_path"]
    assert tif_path.is_file()
    assert zip_path.is_file()
    assert _sha256(tif_path) == sub["tif_sha256"]
    assert _sha256(zip_path) == sub["zip_sha256"]

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert names == [tif_path.name]
        assert hashlib.sha256(zf.read(tif_path.name)).hexdigest() == sub["tif_sha256"]

    sample_path = ROOT / "data" / "bridge" / "sample_submission.tif"
    cat_path = ROOT / "data" / "bridge" / "existing_faults.tif"
    with rasterio.open(sample_path) as sr, rasterio.open(cat_path) as cr, rasterio.open(tif_path) as tr:
        foot = np.isfinite(sr.read(1))
        cat = cr.read(1) > 0
        arr = tr.read(1)
        assert tr.count == 1
        assert tr.dtypes[0] == "float32"
        assert tr.shape == (3730, 3292)
        assert str(tr.crs) == "EPSG:32611"
        assert tr.transform == sr.transform
        assert tr.nodata is None
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert int((arr > 0).sum()) == 37654
        assert int((arr[~foot] > 0).sum()) == 0
        assert int((arr[cat] > 0).sum()) == 0


def test_tsd_site_and_authorization_integration() -> None:
    tsd_html = (ROOT / "docs" / "tsd.html").read_text()
    idx_html = (ROOT / "docs" / "index.html").read_text()
    exec_html = (ROOT / "docs" / "executive-summary.html").read_text()
    for doc in (tsd_html, idx_html, exec_html):
        assert "gems77-tsd-transtens-swarm-submod-37654dots-20261021-zeros" in doc
        assert "NOT OK TO SUBMIT" in doc
        assert "4ebb1da423d26d9823ad15c4f8a0e722edc3e79272c8e720adcd28e6224af68c" in doc

    auth = json.loads((ROOT / "docs" / "data" / "submission-authorization.json").read_text())
    assert auth["repository_decision"] == "NOT OK TO SUBMIT"
    assert auth["repository_authorized_file_count"] == 0
    tsd_entries = [a for a in auth["artifacts"] if a["lane"].startswith("H77-TSD:")]
    assert len(tsd_entries) == 1
    assert tsd_entries[0]["authorized_to_submit"] is False
    assert tsd_entries[0]["decision"] == "NOT OK TO SUBMIT"
