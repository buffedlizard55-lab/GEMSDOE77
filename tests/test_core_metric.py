"""The fast metric in gems77lia.core must equal the shared template metric and the literal formula."""
import numpy as np

from gems.metric import tversky_terms, tversky_terms_bruteforce  # vendored template (byte-identical)
from gems77lia import core


def test_official_cross_example_tp_fn():
    # Problem page 967 worked example: 5-px vertical truth crossed by 5-px horizontal prediction.
    gt = np.zeros((9, 9)); gt[2:7, 4] = 1
    pr = np.zeros((9, 9), dtype=np.float32); pr[4, 2:7] = 1
    tp, fp, fn = core.dti_terms(pr, gt)
    assert abs(tp - 3.0) < 1e-6 and abs(fn - 2.0) < 1e-6
    # page states DTI(3.00, 1.89, 2.00) = 0.60 (rounded)
    assert round(core.dti(3.00, 1.89, 2.00), 2) == 0.60


def test_fast_metric_equals_template_and_bruteforce():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(30):
        h, w = int(rng.integers(5, 12)), int(rng.integers(5, 12))
        p = rng.random((h, w)).astype(np.float32)
        p[p < 0.7] = 0
        g = (rng.random((h, w)) < 0.15).astype(float)
        if g.sum() == 0:
            g[0, 0] = 1
        a = core.dti_terms(p, g)
        b = tversky_terms(p.astype(float), g)
        c = tversky_terms_bruteforce(p.astype(float), g)
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)), max(abs(x - y) for x, y in zip(a, c)))
    assert worst < 1e-4  # float32 accumulation vs float64


def test_identity_dti_closed_form():
    rng = np.random.default_rng(3)
    p = (rng.random((40, 40)) < 0.05).astype(np.float32)
    g = (rng.random((40, 40)) < 0.05)
    tp, fp, fn = core.dti_terms(p, g)
    assert abs(tp + fn - g.sum()) < 1e-4
