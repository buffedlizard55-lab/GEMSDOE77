"""Metric identities/counterexamples, not a new geological experiment."""
import numpy as np
import pytest
from gems57.metric import dti_binary


@pytest.mark.parametrize('redundant', [False, True])
def test_self_credit_is_not_marginal_truth_credit(redundant):
    truth = np.zeros((15, 15), dtype=bool)
    truth[7, 6:9] = True
    pred = np.zeros_like(truth)
    if redundant:
        pred[7, 6:9] = True
    before = dti_binary(pred, truth)
    # Fresh center dot improves several truth pixels; adjacent extra dot
    # improves none if all truth pixels already have perfect coverage.
    pred[7, 9 if redundant else 7] = True
    after = dti_binary(pred, truth)
    q = 2/3 if redundant else 1.0
    delta_t = after['tp'] - before['tp']
    assert delta_t == pytest.approx(0 if redundant else 7/3)
    assert delta_t != pytest.approx(q)
    d0 = before['tp'] + .2*before['fp'] + .8*before['fn']
    d1 = after['tp'] + .2*after['fp'] + .8*after['fn']
    assert d1-d0 == pytest.approx(.2*(delta_t+1-q))
    improves = delta_t*(1-.2*before['dti']) > .2*before['dti']*(1-q)
    assert bool(improves) == (after['dti'] > before['dti'])
