from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

from gems57 import metric as M
from gems57.grid import HEIGHT, TRANSFORM, WIDTH
from gems77hsl import emit_hsl as EM, features_hsl as FH

ROOT = Path(__file__).resolve().parents[1]


def test_submodular_bayes_emitter_respects_mask_and_budget():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.0, 0.01, size=(64, 64)).astype(np.float32)
    # Create a strong ridge along row 32 (44 px wide, covered by ~10 dots of 3-px kernel)
    p[31:34, 10:54] = 0.85
    p[32, 30:35] = 0.99  # hotspot cluster inside ridge
    allowed = np.ones((64, 64), dtype=bool)
    allowed[:5, :] = False
    allowed[:, :5] = False

    dots_sub = EM.emit_dots_submodular_bayes(p, allowed, budget=10, min_sep=1.0)
    dots_std = EM.emit_dots_standard(p, allowed, budget=10, min_sep=1.0)

    assert dots_sub.dtype == bool and dots_sub.shape == (64, 64)
    assert int(dots_sub.sum()) == 10
    assert not (dots_sub & ~allowed).any()
    assert int(dots_std.sum()) == 10
    assert not (dots_std & ~allowed).any()

    # Submodular coverage spreads 10 dots across the 44-px ridge rather than clustering on the 5-px hotspot
    truth = np.zeros((64, 64), dtype=bool)
    truth[31:34, 10:54] = True
    dti_sub = M.dti_exact(dots_sub.astype(np.float32), truth)["dti"]
    dti_std = M.dti_exact(dots_std.astype(np.float32), truth)["dti"]
    assert dti_sub > dti_std


def test_visible_fault_features_strictly_depend_on_visible_mask():
    foot = np.ones((48, 48), dtype=bool)
    vis = np.zeros((48, 48), dtype=bool)
    vis[24, 12:36] = True
    u2 = np.ones((48, 48), dtype=np.float32) * 0.5
    v2 = np.zeros((48, 48), dtype=np.float32)

    feats = FH.compute_visible_fault_features(vis, foot, u2, v2)
    assert feats.shape == (len(FH.HSL4_VIS_STRUCT_NAMES), 48, 48)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    # Distance to visible fault must be 0 on visible fault pixels
    assert np.allclose(feats[0, vis], 0.0, atol=1e-6)
    # Tip cone / continuation on row 24, col 38 (2 px off East tip) should exceed far background at row 5, col 5
    assert feats[5, 24, 38] > feats[5, 5, 5]


def test_hsl_submission_bytes_validator_and_run_card():
    rc = json.loads((ROOT / "evidence" / "run_card_hsl.json").read_text())
    e1 = json.loads((ROOT / "evidence" / "hsl_holdout_exp1.json").read_text())
    e2 = json.loads((ROOT / "evidence" / "hsl_holdout_exp2_replication.json").read_text())
    e3 = json.loads((ROOT / "evidence" / "hsl_exp3_submission.json").read_text())

    tif_path = ROOT / rc["file"]
    zip_path = ROOT / rc["zip_file"]
    assert tif_path.exists() and zip_path.exists()
    assert hashlib.sha256(tif_path.read_bytes()).hexdigest() == rc["raster_sha256"]
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == rc["zip_sha256"]

    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
        assert ds.count == 1
        assert ds.shape == (HEIGHT, WIDTH)
        assert ds.crs.to_epsg() == 32611
        assert ds.transform == TRANSFORM
        assert ds.nodata is None
        assert ds.dtypes == ("float32",)
    assert np.isfinite(arr).all()
    assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
    assert int((arr > 0).sum()) == 37654 == rc["emitted_dots"]

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == [tif_path.name]
        assert zf.read(tif_path.name) == tif_path.read_bytes()

    # Check Run Card required fields (Protocol Rule 5)
    for req_key in (
        "hypothesis",
        "mechanism",
        "named_non_fault_mimic",
        "holdout_dti",
        "correlation_overlap_vs_registry",
        "raster_sha256",
        "validator",
        "submission_name",
        "note",
        "verdict",
    ):
        assert req_key in rc

    assert len(rc["note"]) <= 140
    assert 0 < len(rc["submission_name"]) <= 120
    assert rc["verdict"] == "promote"
    assert rc["weekly_slots_used"] == 0

    # Leakage canary check (Protocol Rule 4)
    assert e1["leakage_channels_gt_0p90_symmetric"] == []
    assert e1["canary_worst_overall_symmetric_auc"] <= 0.90
    assert e1["canary_worst_hsl_new_symmetric_auc"] <= 0.90

    # Holdout DTI CIs check (Protocol Rule 3)
    p2 = e2["two_draw_pooled_s30_s21"]
    assert p2["withheld_positive_pixels"] == 22914
    for summary_key in (
        "pooled_summary_hsl_structural_submod",
        "pooled_summary_hsl_stratified_submod",
        "pooled_summary_hsl_nolabel_submod",
        "pooled_summary_hsl_lidar12_submod",
    ):
        s = p2[summary_key]
        for arm_scores in s["scores"].values():
            assert 0.0 <= arm_scores["ci95"][0] <= arm_scores["dti"] <= arm_scores["ci95"][1] <= 1.0

    # Uniqueness audit check (Protocol Rule 1)
    uq = e3["uniqueness_audit"]
    assert uq["informative_registry_rasters"] == 765
    assert uq["phase1_max_informative_abs_spearman"] <= 0.90
    assert uq["phase2_max_informative_1way_overlap_3px"] <= 0.70
    assert uq["phase2_max_informative_abs_spearman_dots"] <= 0.90
    assert uq["informative_gate_passed"] is True
