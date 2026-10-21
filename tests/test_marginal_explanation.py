"""Mathematical unit fixtures, NOT competition experiments or holdout scores.

Exercise the unchanged shared metric; prevent the prose assumption dT == k
from being mistaken for the general emission rule.
"""
import numpy as np
import pytest
from gems57.metric import ALPHA, BETA, dti_binary, kernel, marginal_accept


def components(pred, truth):
    result = dti_binary(pred, truth)
    t = result['tp']
    n = result['n_emitted']
    m = n - result['fp']
    k = result['n_truth']
    return result, t, m, n, k


def check_new_dot(pred, truth, position):
    before, t, m, n, count = components(pred, truth)
    new = pred.copy()
    assert not new[position]
    new[position] = True
    after, *_ = components(new, truth)
    dt = after['tp'] - before['tp']
    near = np.linalg.norm(np.argwhere(truth) - position, axis=1).min()
    self_credit = float(kernel(near))
    d0 = before['tp'] + ALPHA * before['fp'] + BETA * before['fn']
    d1 = after['tp'] + ALPHA * after['fp'] + BETA * after['fn']
    assert d1 - d0 == pytest.approx(ALPHA * (dt + 1 - self_credit))
    assert marginal_accept(dt, self_credit, t, m, n, count) == (after['dti'] > before['dti'])
    assert (dt > ALPHA * before['dti'] * (dt + 1 - self_credit)) == (after['dti'] > before['dti'])
    return dt, self_credit, d1 - d0


def test_one_dot_can_cover_multiple_truth_cells():
    truth = np.zeros((9, 9), bool)
    truth[4, 3:6] = True
    dt, credit, change = check_new_dot(np.zeros_like(truth), truth, (4, 4))
    assert dt > credit
    assert change != pytest.approx(ALPHA)


def test_redundant_dot_has_zero_marginal_truth_credit():
    truth = np.zeros((9, 9), bool)
    truth[4, 4] = True
    pred = truth.copy()
    dt, credit, change = check_new_dot(pred, truth, (4, 5))
    assert dt == 0 and credit > 0
    assert change != pytest.approx(ALPHA)


def test_general_marginal_identity_for_random_small_fields():
    rng = np.random.default_rng(77060)
    for _ in range(100):
        truth = rng.random((9, 9)) < 0.2
        truth[4, 4] = True
        pred = rng.random((9, 9)) < 0.1
        position = tuple(np.argwhere(~pred)[0])
        check_new_dot(pred, truth, position)
