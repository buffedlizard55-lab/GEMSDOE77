"""Comprehensive end-to-end verification & regression suite for GEMSDOE77 (`pytest`)."""
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

from gems57 import evaluate_holdout, grid, uniqueness, validate  # noqa: E402
from gems77 import emit, features, models  # noqa: E402


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


def test_continuous_and_binary_uniqueness_fix(tmp_path: Path) -> None:
    """Rule 1 & Rule 2 (IRR-01): Continuous surfaces in [0, 1) with rho <= 0.90 pass uniqueness."""
    h, w = 40, 40
    fp = np.ones((h, w), dtype=bool)
    prior = np.zeros((h, w), dtype=np.float32)
    prior[5, 5] = 1.0
    prior[25, 30] = 1.0
    ref_tif = tmp_path / "prior_ref.tif"
    with rasterio.open(
        ref_tif, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs="EPSG:32611", transform=grid.TRANSFORM,
    ) as ds:
        ds.write(prior, 1)

    reg_idx = {
        "complete_accessible_scan": True,
        "errors": [],
        "rasters": [{
            "repo_first": "test-repo",
            "sources": ["prior_ref.tif"],
            "cache_file": str(ref_tif),
            "sha256": _sha256(ref_tif),
        }],
    }
    rng = np.random.default_rng(77)
    cont_surf = rng.uniform(0.01, 0.85, size=(h, w)).astype(np.float32)
    res_cont = uniqueness.compare_array_to_registry(cont_surf, reg_idx, fp, include_reverse=True)
    assert res_cont["is_continuous_surface"] is True
    assert res_cont["unique"] is True
    assert res_cont["stop_required"] is False

    # Zero-variance surface must STILL strictly fail closed!
    flat_surf = np.full((h, w), 0.25, dtype=np.float32)
    res_flat = uniqueness.compare_array_to_registry(flat_surf, reg_idx, fp, include_reverse=True)
    assert res_flat["unique"] is False
    assert res_flat["stop_required"] is True


def test_zero_sgmc_and_architectural_firewall() -> None:
    """Verify SGMC pre-Quaternary bedrock file is never referenced and model signatures have no truth."""
    feat_src = inspect.getsource(features)
    mod_src = inspect.getsource(models)
    assert "sgmc_faults_100m.tif" not in feat_src.split('"""', 2)[-1]
    assert "truth" not in inspect.signature(models.fit_h77_deconfounded_ensemble).parameters
    assert "existing_faults" not in inspect.signature(models.fit_h77_deconfounded_ensemble).parameters


def test_flank_zeroed_submodular_eliminates_collar_ringing() -> None:
    """Verify H77-4 strictly zeroes q_pad outside allowed_mask so masked scarps pull 0 halo dots."""
    h, w = 60, 60
    allowed = np.ones((h, w), dtype=bool)
    allowed[28:33, :] = False  # Masked catalogued fault strip
    prob = np.full((h, w), 0.10, dtype=np.float32)
    prob[28:33, :] = 0.9999   # Huge spike INSIDE masked collar (caused P9 bug!)
    prob[10, 10:50:4] = 0.85  # Real unmapped fault strand well outside collar
    crest = np.full((h, w), 0.5, dtype=np.float32)
    dots = emit.flank_zeroed_crest_submodular_emit(prob, crest, allowed, n_dots=10, support_quantile=0.50)
    assert int((dots > 0.5).sum()) == 10
    assert int((dots[~allowed] > 0).sum()) == 0
    # Zero dots wasted along rows 27 or 33 bordering the masked strip!
    assert int((dots[27, :] > 0.5).sum()) == 0
    assert int((dots[33, :] > 0.5).sum()) == 0


def test_leakage_canary_and_run_card_and_submission_geotiff() -> None:
    """End-to-end verification of canary <= 0.90, Run Card schema, and final .tif range [0, 1]."""
    canary = json.loads((ROOT / "evidence" / "h77" / "leakage_canary.json").read_text())
    assert canary["all_pass_le_0p90"] is True
    assert canary["max_symmetric_auc"] <= 0.90
    assert canary["n_features_tested"] == 99

    rc = json.loads((ROOT / "evidence" / "h77" / "run_card.json").read_text())
    assert rc["verdict"] == "promote"
    assert rc["safe_to_download_and_submit"] is True
    assert rc["budget_accounting"]["experiments_run"] <= 3
    assert rc["budget_accounting"]["runtime_seconds"] <= 7200
    assert 1 <= len(rc["submission_note"]) <= 140
    assert rc["holdout_dti"]["pooled_hide_v2"]["score"] > 0.15
    assert rc["holdout_dti"]["pooled_hide_v2"]["delta_vs_h33_fold_fair_ci_95"][0] > 0.0
    assert rc["uniqueness_vs_registry"]["pre_placement_surface_unique"] is True
    assert rc["uniqueness_vs_registry"]["post_placement_dots_unique"] is True

    tif_main = ROOT / "submissions" / "gems77-step-propensity-submod300.tif"
    tif_pages = ROOT / "docs" / "submissions" / "gems77-step-propensity-submod300.tif"
    zip_main = ROOT / "submissions" / "gems77-step-propensity-submod300.zip"
    assert tif_main.is_file() and tif_pages.is_file() and zip_main.is_file()
    assert _sha256(tif_main) == _sha256(tif_pages) == rc["submission_raster"]["sha256"]

    with zipfile.ZipFile(zip_main) as zf:
        assert zf.namelist() == [tif_main.name]
        assert zf.read(tif_main.name) == tif_main.read_bytes()

    g = grid.load_grid(ROOT / ".cache" / "data")
    with rasterio.open(ROOT / ".cache" / "data" / "existing_faults.tif") as ds_cat:
        cat = (ds_cat.read(1) > 0) & g.footprint

    val_report = validate.validate(tif_main, g.footprint, cat)
    validate.assert_submittable(val_report)
    assert val_report["all_checks_passed"] is True
    assert val_report["n_nan"] == 0
    assert val_report["n_infinite"] == 0
    assert val_report["min"] == 0.0
    assert val_report["max"] == 1.0
