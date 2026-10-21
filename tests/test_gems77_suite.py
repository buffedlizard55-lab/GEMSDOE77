"""Comprehensive end-to-end verification & regression suite for H77-Step (`pytest`)."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout, grid, validate  # noqa: E402
from gems77_step import emit, features, models  # noqa: E402


EXPECTED_VENDOR_HASHES = {
    "evaluate_holdout.py": "ad86640b46f3c5fc6b1aceec4c4bfc82d03e35008b37ab2a8ad6076e6810ff3a",
    "holdout.py": "7a0a84176986190b527aa9c6a84216c2dab4b599f25b1143db5f420e85020ab1",
    "metric.py": "4afe10cbaa229948520c75194b1ebc32676e72544f05582f0725092b097b4d7f",
    "spatial.py": "fad7b09db9efc512247b9a176a7e7ce8631e287592b4e07902d0b9c38bed00f8",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_shared_evaluator_upstream_hashes() -> None:
    """Rule 2: Verify exact SHA-256 parity with upstream vendor/gems57 evaluator & metric."""
    assert evaluate_holdout.implementation_hashes() == EXPECTED_VENDOR_HASHES


def test_zero_sgmc_and_architectural_firewall() -> None:
    """Verify SGMC pre-Quaternary bedrock file is never referenced and model signatures have no truth."""
    feat_src = inspect.getsource(features)
    assert "sgmc_faults_100m.tif" not in feat_src.split('"""', 2)[-1]
    assert "truth" not in inspect.signature(models.fit_h77_deconfounded_ensemble).parameters
    assert "existing_faults" not in inspect.signature(models.fit_h77_deconfounded_ensemble).parameters


def test_flank_zeroed_submodular_eliminates_collar_ringing() -> None:
    """Verify H77-4 strictly zeroes q_pad outside allowed_mask so masked scarps pull 0 halo dots."""
    h, w = 60, 60
    allowed = np.ones((h, w), dtype=bool)
    allowed[28:33, :] = False
    prob = np.full((h, w), 0.10, dtype=np.float32)
    prob[28:33, :] = 0.9999
    prob[10, 10:50:4] = 0.85
    crest = np.full((h, w), 0.5, dtype=np.float32)
    dots = emit.flank_zeroed_crest_submodular_emit(prob, crest, allowed, n_dots=10, support_quantile=0.50)
    assert int((dots > 0.5).sum()) == 10
    assert int((dots[~allowed] > 0).sum()) == 0
    assert int((dots[27, :] > 0.5).sum()) == 0
    assert int((dots[33, :] > 0.5).sum()) == 0


def test_h77_step_leakage_canary_and_run_card_and_submission_geotiff() -> None:
    """Full CI-friendly verification using committed data/bridge reference grid & faults."""
    canary = json.loads((ROOT / "evidence" / "h77" / "leakage_canary.json").read_text())
    assert canary["all_pass_le_0p90"] is True
    assert canary["max_symmetric_auc"] <= 0.90
    assert canary["n_features_tested"] == 99

    rc = json.loads((ROOT / "evidence" / "h77" / "run_card.json").read_text())
    assert rc["budget_accounting"]["experiments_run"] <= 3
    assert rc["budget_accounting"]["runtime_seconds"] <= 7200
    assert 1 <= len(rc["submission_note"]) <= 140
    assert rc["holdout_dti"]["pooled_hide_v2"]["score"] > 0.15
    assert rc["holdout_dti"]["pooled_hide_v2"]["delta_vs_h33_fold_fair_ci_95"][0] > 0.0
    assert rc["uniqueness_vs_registry"]["pre_placement_surface_unique"] is True
    assert rc["uniqueness_vs_registry"]["post_placement_dots_unique"] is True

    tif_main = ROOT / "submissions" / "gems77-step-propensity-submod300.tif"
    tif_dl = ROOT / "docs" / "downloads" / "gems77-step-propensity-submod300.tif"
    zip_main = ROOT / "submissions" / "gems77-step-propensity-submod300.zip"
    assert tif_main.is_file() and tif_dl.is_file() and zip_main.is_file()
    assert _sha256(tif_main) == _sha256(tif_dl) == rc["submission_raster"]["sha256"]

    with zipfile.ZipFile(zip_main) as zf:
        assert zf.namelist() == [tif_main.name]
        assert zf.read(tif_main.name) == tif_main.read_bytes()

    bridge = ROOT / "data" / "bridge"
    g = grid.load_grid(bridge)
    with rasterio.open(bridge / "existing_faults.tif") as ds_cat:
        cat = (ds_cat.read(1) > 0) & g.footprint

    val_report = validate.validate(tif_main, g.footprint, cat)
    validate.assert_submittable(val_report)
    assert val_report["all_checks_passed"] is True
    assert val_report["n_nan"] == 0
    assert val_report["n_infinite"] == 0
    assert val_report["min"] == 0.0
    assert val_report["max"] == 1.0
