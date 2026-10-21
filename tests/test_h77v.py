"""Unit and integration tests for Lane H77V (`H77V-VENT-ADVECT-CONDUIT`)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

from src.gems77v.features import (
    ALL_STATIC_NAMES,
    BASE_CHANNEL_NAMES,
    H77V_CHANNEL_NAMES,
    IDX_BASE,
    IDX_H77V,
    LOCO_NAMES,
    transport_upslope_2d,
)

ROOT = Path(__file__).resolve().parents[1]


def test_h77v_feature_definitions_and_anti_leakage() -> None:
    assert len(H77V_CHANNEL_NAMES) == 16
    assert len(BASE_CHANNEL_NAMES) == 36
    assert len(ALL_STATIC_NAMES) == 52
    assert len(LOCO_NAMES) == 4
    assert IDX_H77V == list(range(16))
    assert IDX_BASE == list(range(16, 52))

    import ast

    raw_code = (ROOT / "src" / "gems77v" / "features.py").read_text()
    tree = ast.parse(raw_code)
    # Collect all string constants that are NOT module/function docstrings
    doc_nodes = set()
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        doc_nodes.add(tree.body[0].value)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                doc_nodes.add(first.value)
    code_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in doc_nodes
    ]
    assert "dist_known_fault_px" not in code_strings
    assert "gdr_qfaults_traces.csv" not in code_strings
    assert "existing_faults.tif" not in code_strings
    assert "labels.tif" not in code_strings
    assert "src.read(14)" not in raw_code
    assert "src.read(18)" not in raw_code


def test_transport_upslope_2d_shifts_hanging_wall_crest_upslope() -> None:
    # Suppose elevation increases to the North (row 0 is high, row 19 is low -> uy_up = -1.0, ux_up = 0.0)
    # A hanging-wall gravity gradient peak at row 10 (down-slope of the fault at row 6)
    # should be shifted upslope toward row 6 when transported by shift_px = 4.0!
    field = np.zeros((20, 20), dtype=np.float32)
    field[10, 10] = 1.0
    uy_up = np.full((20, 20), -1.0, dtype=np.float32)
    ux_up = np.zeros((20, 20), dtype=np.float32)
    shifted = transport_upslope_2d(field, uy_up, ux_up, shift_px=4.0)
    assert shifted[6, 10] == np.float32(0.70)
    assert shifted[10, 10] == np.float32(0.30)


def test_h77v_preregistration_and_knowledge_docs_exist() -> None:
    prereg_json = ROOT / "registry" / "preregistration_h77v_20261021.json"
    hyp_md = ROOT / "research" / "hypotheses-h77v-2026-10-21.md"
    know_md = ROOT / "knowledge" / "03_advection_backtracking_and_vent_conduits_h77v.md"
    assert prereg_json.is_file()
    assert hyp_md.is_file()
    assert know_md.is_file()

    prereg = json.loads(prereg_json.read_text())
    assert prereg["lane"] == "H77V-VENT-ADVECT-CONDUIT"
    assert len(prereg["ranked_hypotheses"]) == 5
    assert prereg["protocol_rules"]["rule_6_budget"]["weekly_slots_to_spend"] == 0


def test_h77v_exp1_canary_and_ablation_evidence() -> None:
    exp1 = json.loads((ROOT / "evidence" / "h77v_exp1_canary_and_ablation.json").read_text())
    assert exp1["lane"] == "H77V-VENT-ADVECT-CONDUIT"
    assert exp1["evaluator_version"] == "gems57-pooled-hide-v2"
    assert exp1["withheld_positives"] == 11342

    can = exp1["leakage_canary"]
    assert can["all_passed"] is True
    assert can["total_features_tested"] == 56
    assert can["h77v_new_features_tested"] == 16
    assert can["max_symmetric_auc_h77v"] <= 0.90
    assert can["max_symmetric_auc_all"] <= 0.90
    assert all(f["symmetric_auc"] <= 0.90 for f in can["features"])

    scores = exp1["label_free_surface_ablation"]["pooled_summary"]["scores"]
    assert scores["S_h77v_advect"]["dti"] > scores["S_hydro_isotropic"]["dti"] > scores["A0_random"]["dti"]


def test_h77v_exp2_dual_seed_holdout_evidence() -> None:
    exp2 = json.loads((ROOT / "evidence" / "h77v_exp2_holdout.json").read_text())
    assert exp2["lane"] == "H77V-VENT-ADVECT-CONDUIT"
    assert exp2["evaluator_version"] == "gems57-pooled-hide-v2"
    assert exp2["sgmc_channels_used"] == 0
    assert exp2["prior_submission_rasters_used_in_features"] == 0

    for seed_key, expected_pos in (("seed30", 11342), ("seed20", 11321)):
        s = exp2[seed_key]
        assert s["withheld_positive_pixels"] == expected_pos
        ps = s["pooled_summary"]
        dti_gated = ps["scores"]["A_h77v_gated_submod"]["dti"]
        dti_base = ps["scores"]["A_base_no_h77v_submod"]["dti"]
        dti_tree = ps["scores"]["A_h77v_tree_submod"]["dti"]
        assert dti_gated > dti_base
        assert dti_gated > dti_tree
        # Paired 95% CIs vs base_no_h77v and vs unconstrained tree must strictly exclude zero on both seeds
        assert ps["paired_differences"]["A_base_no_h77v_submod"]["ci95"][0] > 0.0
        assert ps["paired_differences"]["A_h77v_tree_submod"]["ci95"][0] > 0.0


def test_h77v_exp3_artifact_runcard_and_site_integration() -> None:
    rc_path = ROOT / "submissions" / "run_card_h77v_20261021.json"
    assert rc_path.is_file()
    rc = json.loads(rc_path.read_text())

    assert rc["lane"] == "H77V-VENT-ADVECT-CONDUIT"
    assert rc["verdict"] == "negative"
    assert rc["ok_to_submit"] is False
    assert rc["weekly_slots_spent"] == 0
    assert rc["artifact"]["submission_note_length"] <= 140
    assert len(rc["artifact"]["submission_note"]) <= 140

    uq = rc["registry_uniqueness"]
    assert uq["phase1_surface_max_abs_rho"] <= 0.90
    assert uq["phase2_dot_max_abs_rho"] <= 0.90
    assert uq["phase2_informative_sparse_max_3px_forward_overlap"] <= 0.70
    assert uq["informative_unique"] is True
    assert uq["protocol_unique"] is False

    tif_path = ROOT / rc["artifact"]["zeros_tif_path"]
    zip_path = ROOT / rc["artifact"]["zip_path"]
    nan_path = ROOT / rc["artifact"]["nan_tif_path"]
    assert tif_path.is_file()
    assert zip_path.is_file()
    assert nan_path.is_file()

    assert hashlib.sha256(tif_path.read_bytes()).hexdigest() == rc["artifact"]["zeros_tif_sha256"]
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == rc["artifact"]["zip_sha256"]
    assert hashlib.sha256(nan_path.read_bytes()).hexdigest() == rc["artifact"]["nan_tif_sha256"]

    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
        assert str(ds.crs) == "EPSG:32611"
        assert ds.shape == (3730, 3292)
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert np.isfinite(arr).all()
        assert float(arr.min()) == 0.0
        assert float(arr.max()) == 1.0
        assert int((arr > 0).sum()) == 37654

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert len(names) == 1 and names[0].endswith(".tif")

    # Verify lanes.json and HTML page integration
    lanes = json.loads((ROOT / "docs" / "data" / "lanes.json").read_text())["lanes"]
    h77v_lanes = [x for x in lanes if x["lane"].startswith("H77V:")]
    assert len(h77v_lanes) == 1
    row = h77v_lanes[0]
    assert row["verdict"] == "negative"
    assert row["protocol_unique"] is False
    assert row["informative_unique"] is True
    assert "Not OK to submit" in row["upload"]
    assert (ROOT / "docs" / row["page"]).is_file()
