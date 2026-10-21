"""Algebra regression checks, not holdout experiments or a replacement evaluator."""
import numpy as np
import pytest
from gems57.metric import dti_exact, max_cover


def check_marginal(pred, truth, position):
    before = dti_exact(pred, truth)
    after_pred = pred.copy()
    assert pred[position] == 0
    after_pred[position] = 1
    after = dti_exact(after_pred, truth)
    _, nearest, _ = max_cover(pred, truth)
    q = nearest[position]
    dt = after['tp'] - before['tp']
    denominator = before['tp'] + .2 * before['fp'] + .8 * before['fn']
    next_denominator = after['tp'] + .2 * after['fp'] + .8 * after['fn']
    assert next_denominator - denominator == pytest.approx(.2 * (dt + 1 - q))
    expected = (before['tp'] + dt) / (denominator + .2 * (dt + 1 - q))
    assert after['dti'] == pytest.approx(expected)
    margin = dt - .2 * before['dti'] * (dt + 1 - q)
    if abs(margin) > 1e-10:
        assert (after['dti'] > before['dti']) == (margin > 0)
    return dt, q, next_denominator - denominator


def test_one_dot_can_cover_multiple_truth_pixels():
    truth = np.zeros((9, 9), bool)
    truth[4, 3:6] = True
    dt, q, dd = check_marginal(np.zeros(truth.shape), truth, (4, 4))
    assert dt > q
    assert dd != pytest.approx(.2)


def test_redundant_dot_can_have_no_new_truth_credit():
    truth = np.zeros((9, 9), bool)
    truth[4, 4] = True
    pred = truth.astype(float)
    dt, q, dd = check_marginal(pred, truth, (4, 5))
    assert dt == 0
    assert q > 0
    assert dd != pytest.approx(.2)


def test_isolated_truth_special_case():
    truth = np.zeros((9, 9), bool)
    truth[4, 4] = True
    dt, q, dd = check_marginal(np.zeros(truth.shape), truth, (4, 5))
    assert dt == pytest.approx(q)
    assert dd == pytest.approx(.2)


def test_randomized_marginal_identity():
    rng = np.random.default_rng(904)
    for _ in range(100):
        truth = rng.random((9, 9)) < .15
        truth[0, 0] = True
        pred = (rng.random(truth.shape) < .1).astype(float)
        position = tuple(np.argwhere(pred == 0)[0])
        check_marginal(pred, truth, position)
