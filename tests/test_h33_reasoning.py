"""Algebra checks, not HOLDOUT-DTI measurements or leaderboard forecasts."""
from pathlib import Path

import numpy as np

from gems57.metric import ALPHA, dti_binary, kernel, marginal_accept

ROOT = Path(__file__).resolve().parents[1]
from scipy.ndimage import distance_transform_edt


def test_old_h33_inversion_note_is_explicitly_superseded():
    report = (ROOT / "reports/why_02778.md").read_text()
    script = (ROOT / "scripts/why_02778.py").read_text()
    assert report.startswith("> **HISTORICAL / SUPERSEDED:**")
    assert "No organizer receipt for this TIFF was verified" in report
    assert "not evidence of H33 score attribution" in script
    assert "ORGANIZER-CONFIRMED on the" not in script


def components(pred, truth):
    out = dti_binary(pred, truth)
    m = float(kernel(distance_transform_edt(~truth))[pred].sum()) if pred.any() else 0.0
    return out, m


def test_one_dot_can_credit_multiple_truth_pixels():
    truth = np.zeros((11, 11), bool)
    truth[5, 5:8] = True
    pred = np.zeros_like(truth)
    before, m = components(pred, truth)
    pred[5, 6] = True
    after, new_m = components(pred, truth)
    dt = after["tp"] - before["tp"]
    k = new_m - m
    assert dt > k  # disproves dT <= k and constant denominator increment
    assert not np.isclose(ALPHA * (dt + 1 - k), ALPHA)
    assert marginal_accept(dt, k, before["tp"], m, 0, int(truth.sum()))


def test_exact_marginal_inequality_matches_shared_metric():
    rng = np.random.default_rng(77)
    for _ in range(80):
        truth = rng.random((13, 13)) < 0.08
        truth[6, 6] = True
        pred = rng.random((13, 13)) < 0.1
        y, x = np.argwhere(~pred)[0]
        before, m = components(pred, truth)
        n = int(pred.sum())
        pred[y, x] = True
        after, new_m = components(pred, truth)
        dt = after["tp"] - before["tp"]
        k = new_m - m
        exact = after["dti"] > before["dti"]
        assert marginal_accept(dt, k, before["tp"], m, n, int(truth.sum())) == exact
        assert (dt > ALPHA * before["dti"] * (dt + 1 - k)) == exact
