"""Synthetic checks of the live-score inversion and the allocation primitives (no data needed)."""
import numpy as np

from gems77 import allocate as al, inversion as inv


def test_inversion_recovers_known_density():
    rng = np.random.default_rng(1)
    n_bins, n_rasters = 6, 80
    N = rng.integers(50_000, 500_000, n_bins).astype(float)
    h_true = np.array([0.08, 0.02, 0.0, 0.004, 0.0, 0.0])
    C = rng.uniform(0, 1, (n_rasters, n_bins)) * rng.uniform(1_000, 30_000, (n_rasters, 1))
    S = rng.uniform(20_000, 150_000, n_rasters)
    y = inv.predict(h_true, C, S, N, c=2.0)
    h = inv.fit_nnls(C, S, N, y, c=2.0)
    assert np.allclose(inv.predict(h, C, S, N, 2.0), y, atol=1e-6)
    assert np.allclose(h, h_true, atol=2e-3)


def test_loo_predictions_track_truth():
    rng = np.random.default_rng(2)
    N = np.array([1e5, 3e5, 9e5])
    h_true = np.array([0.05, 0.01, 0.0])
    C = rng.uniform(0, 1, (40, 3)) * rng.uniform(1_000, 20_000, (40, 1))
    S = rng.uniform(20_000, 90_000, 40)
    y = inv.predict(h_true, C, S, N) + rng.normal(0, 0.002, 40)
    lo = inv.loo(C, S, N, y)
    assert np.corrcoef(lo, y)[0, 1] > 0.95


def test_greedy_select_respects_min_separation():
    import gems77.core as core
    rng = np.random.default_rng(3)
    cand = rng.choice(core.H * core.W, 20_000, replace=False)
    sel = al.greedy_select(cand, 3.0)
    r, c = np.divmod(sel, core.W)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.c_[r, c]).query(np.c_[r, c], k=2)
    assert d[:, 1].min() >= 3.0


def test_overlap_fraction_euclidean_3px():
    import gems77.core as core
    theirs = np.zeros((core.H, core.W), bool)
    theirs[100, 100] = True
    mine = np.array([100 * core.W + 103, 100 * core.W + 104, 102 * core.W + 102])  # d = 3, 4, 2.83
    assert abs(al.overlap_fraction(mine, theirs) - 2 / 3) < 1e-9
