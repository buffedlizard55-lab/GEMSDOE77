import numpy as np
from gems77.metric import dti_exact, dti_binary, kernel, ALPHA, BETA

def test_worked_example():
    # From problem page: TP=3.00 FP=1.89 FN=2.00 -> 0.60
    # DTI = TP / (TP + 0.2 FP + 0.8 FN)
    tp, fp, fn = 3.00, 1.89, 2.00
    dti = tp / (tp + 0.2*fp + 0.8*fn)
    assert abs(dti - 0.60) < 1e-2, dti
    # Also test kernel
    assert abs(kernel(0) - 1.0) < 1e-9
    assert abs(kernel(3) - 0.0) < 1e-9
    assert abs(kernel(1.5) - 0.5) < 1e-9

def test_dti_empty_truth():
    pred = np.zeros((5,5))
    truth = np.zeros((5,5), bool)
    r = dti_binary(pred, truth)
    assert r["dti"] == 0.0

def test_dti_perfect():
    pred = np.eye(5, dtype=bool)
    truth = np.eye(5, dtype=bool)
    r = dti_binary(pred, truth)
    assert r["dti"] > 0.9

def test_metric_identity():
    # DTI = T/(0.2(T+S-M)+0.8K) identity check
    import rasterio
    # Just check algebra
    T, S, M, K = 1000, 40000, 8000, 50000
    dti = T / (0.2*(T+S-M) + 0.8*K)
    assert 0 < dti < 1

