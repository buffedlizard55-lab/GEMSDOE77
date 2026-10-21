"""Synthetic tests for the H77L lag operator (not evidence of fault discovery)."""
import numpy as np

from gems77lag.features import EROSION_PX, TENSOR_STENCIL_PX, build, normalize_surface


def _inputs(shift):
    h = w = 128
    x = np.arange(w, dtype=np.float32)[None, :]
    ridge = np.exp(-0.5 * ((x - 64.0) / 1.3)**2)
    step = np.broadcast_to((1.0 + 210.0 * ridge).astype(np.uint8), (h, w)).copy()
    coh = np.full((h, w), 230, dtype=np.uint8)
    valid = np.full((h, w), 255, dtype=np.uint8)
    # Clip a synthetic logistic's argument to avoid non-scientific overflow warnings.
    z = np.clip(-(x - (64.0 + shift)) * 3, -60, 60)
    thk = np.broadcast_to((1 + 220 / (1 + np.exp(z))).astype(np.uint8),
                          (h, w)).copy()
    return step, coh, valid, thk, np.ones((h, w), bool)


def test_filter_safe_eroded_support_includes_gradient_between_gaussians():
    assert TENSOR_STENCIL_PX == 16 and EROSION_PX >= TENSOR_STENCIL_PX


def test_displaced_boundary_stronger_than_same_pixel_at_lidar_crest():
    lagged = build(*_inputs(3))[0]
    coincident = build(*_inputs(0))[0]
    at_crest = np.s_[40:88, 64]
    assert float(np.median(lagged['candidate'][at_crest])) > 0
    assert float(np.median(lagged['candidate'][at_crest])) > float(
        np.median(coincident['candidate'][at_crest]))
    assert all(a.dtype == np.float32 for a in lagged.values())


def test_missing_data_cannot_create_visible_halo_inside_eroded_support():
    inputs = list(_inputs(3))
    inputs[3][:, 64:68] = 0  # unavailable ThK; never a chemical zero
    raw, fields, eligible, _ = build(*inputs)
    assert not eligible[:, 48:84].any()
    assert all(not (a[:, 48:84] > 0).any() for a in [*raw.values(), *fields.values()])
    assert all(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in fields.values())


def test_saturated_tail_not_mislabeled_as_sparse_prediction():
    eligible = np.ones((12, 12), bool)
    out, report = normalize_surface(np.ones((12, 12), np.float32), eligible)
    assert not out.any()
    assert report['positive_before'] == 144 and report['positive_after'] == 0


def test_bad_input_or_all_missing_fails_closed():
    inputs = list(_inputs(3))
    inputs[2][:] = 0
    import pytest
    with pytest.raises(ValueError, match='no filter-safe'):
        build(*inputs)
    inputs = list(_inputs(3)); inputs[0] = inputs[0].astype(np.int16)
    with pytest.raises(ValueError, match='uint8'):
        build(*inputs)


def test_fresh_checkout_validator_cli_uses_tracked_bridge_not_unrestored_raw():
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tif = root / 'docs/downloads/gems77-h77l-thk-flank-6b24305363fa-RESEARCH-ONLY.tif'
    got = subprocess.run([sys.executable, str(root / 'scripts/validate_submission.py'), str(tif)],
                         cwd=root, capture_output=True, text=True, check=True)
    report = json.loads(got.stdout)
    assert report['all_checks_passed'] and report['ok_to_submit'] is False
    assert report['meta']['crs'] == 'EPSG:32611' and report['meta']['shape'] == [3730, 3292]
