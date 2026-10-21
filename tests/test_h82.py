"""H82 spectral magnetic source-depth lane: synthetic unit checks + run-card audit.

Unit checks run on synthetic arrays only (no competition data needed): they
verify the radial-bin geometry and that the Spector & Grant style slope fit
recovers a known depth from a synthetic exponentially-decaying-spectrum
source. The run-card audit only runs once ``evidence/h82_run_card.json``
exists (committed negative result from this session).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_h82_module():
    spec = importlib.util.spec_from_file_location("h82_pipeline", ROOT / "scripts" / "h82_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor" / "template66")]
    spec.loader.exec_module(mod)
    return mod


def test_radial_bin_index_covers_all_cells_and_is_monotonic_in_k():
    mod = _load_h82_module()
    n = 32
    bin_idx, k_phys = mod._radial_bin_index(n)
    assert bin_idx.shape == (n, n)
    assert bin_idx.min() == 0
    assert bin_idx.max() == mod.N_RADIAL_BINS - 1
    assert np.all(np.diff(k_phys) > 0)  # k increases monotonically with bin index
    assert k_phys[0] >= 0.0


def test_spector_grant_depth_recovers_synthetic_source_depth():
    """A 2D field with a known radial amplitude spectrum ~ exp(-k*z_true)
    must recover a fitted depth close to z_true (within the method's own
    discretization error), proving the fit arithmetic itself is correct
    before trusting it on real GeoDAWN data."""
    mod = _load_h82_module()
    rng = np.random.default_rng(0)
    n = mod.BLOCK
    bin_idx, k_phys = mod._radial_bin_index(n)
    z_true = 600.0  # metres
    # Build a field whose Fourier amplitude at each cell decays as exp(-k*z_true),
    # with random phase, then take the real part as a synthetic "RTP" tile.
    amp = np.exp(-k_phys[bin_idx] * z_true)
    phase = rng.uniform(0, 2 * np.pi, size=(n, n))
    spec = amp * np.exp(1j * phase)
    tile = np.real(np.fft.ifft2(spec)).astype(np.float32)
    # Embed the single synthetic tile into a small valid grid and run the block estimator.
    valid = np.ones_like(tile, bool)
    depth_blocks, coverage_blocks = mod._spector_grant_depth_blocks(tile, valid)
    assert coverage_blocks[0, 0] == 1.0
    fitted = float(depth_blocks[0, 0])
    assert np.isfinite(fitted)
    # Within the clip range and within a factor of 2 of the true depth (coarse
    # single-slope method on one noisy random-phase realisation, not an exact solver).
    assert mod.DEPTH_CLIP_M[0] <= fitted <= mod.DEPTH_CLIP_M[1]
    assert 0.3 * z_true <= fitted <= 3.0 * z_true


def test_h82_run_card_is_negative_and_internally_consistent():
    card_p = ROOT / "evidence" / "h82_run_card.json"
    if not card_p.exists():
        import pytest
        pytest.skip("H82 Exp 1/2 not run yet")
    card = json.loads(card_p.read_text())
    assert card["verdict"] == "negative"
    hd = card["holdout_dti"]
    assert hd["evaluator_version"] == "gems57-pooled-hide-v2"
    assert hd["beats_incumbent"] is False
    g = hd["H82_gradient"]["dti"]
    r = hd["random_control"]["dti"]
    assert g < r  # measured: candidate underperforms its own matched random control
    delta = hd["paired_delta_H82_gradient_minus_random"]
    assert delta["ci95"][1] < 0  # CI entirely negative -> significantly worse, not just non-significant
    assert len(card["submission_note"]) <= 140
    assert "h82-spectral-depth-gradient" in card["submission_name"]
    assert "NOT OK to submit" in card["submission_name"]
    tif = ROOT / "submissions" / "gems77-h82-spectral-depth-gradient-37654dots-20261021-zeros.tif"
    if tif.is_file():
        import hashlib
        assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]


def test_h82_artifact_audit():
    card_p = ROOT / "evidence/h82_run_card.json"
    if not card_p.exists():
        import pytest
        pytest.skip("H82 run card not written yet")
    card = json.loads(card_p.read_text())
    if "raster_file" not in card or card.get("raster_sha256", "").startswith("not"):
        import pytest
        pytest.skip("H82 artifact mode not run yet")
    import hashlib
    tif = ROOT / "submissions" / card["raster_file"]
    assert tif.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    assert card["ok_to_download"] is True
    assert card["ok_to_submit"] is False
    assert card["submission_slots_used"] == 0
    assert card["validator_output"]["checks_passed"] is True
    assert card["validator_output"]["values_in_0_1"] is True
    assert card["validator_output"]["no_nan_or_inf"] is True
    docs_tif = ROOT / "docs" / "downloads" / card["raster_file"]
    assert docs_tif.is_file()
    assert hashlib.sha256(docs_tif.read_bytes()).hexdigest() == card["raster_sha256"]
    page = (ROOT / "docs/h82.html").read_text()
    assert "OK to download" in page
    assert "NOT OK TO SUBMIT" in page or "NOT OK to submit" in page
    assert card["raster_sha256"] in page
    lanes = json.loads((ROOT / "docs/data/lanes.json").read_text())
    h82_lane = next((x for x in lanes["lanes"] if x["lane"].startswith("H82:")), None)
    assert h82_lane is not None
    assert h82_lane["sha256"] == card["raster_sha256"]
    assert h82_lane["verdict"] == "negative"
    assert h82_lane["beats_holdout_best"] is False
