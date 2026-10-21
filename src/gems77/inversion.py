"""Live-score inversion: estimate where the hidden (new-fault) truth lives from organizer scores.

Model (derivation in docs/analysis.html#inversion)
--------------------------------------------------
Let h_j be the expected density of hidden-truth pixels per cell in bin j (bins = a
partition of the footprint by catalogue distance x support distance).  For raster i
with cover field M_i(y) = max_x p_i(x) k(|x - y|):

    E[TP_i] = sum_y h(y) M_i(y)  = sum_j h_j C_ij           (exact, linearity of E)
    FP_i    = S_i - Mk_i,  Mk_i = sum_x p_i(x) max_g k(d(x,g)) ~= TP_i / c
    K       = sum_j h_j N_j                                    (expected |G|)
    DTI_i   = TP_i / (0.2 (TP_i + FP_i) + 0.8 K)

If the public chunks are a spatially random subset, every term scales by the same
fraction, so the public DTI equals the full-map DTI in expectation.  For fixed c the
model is linear in h after multiplying through:

    sum_j h_j [C_ij (1 - 0.2 DTI_i (1 - 1/c)) - 0.8 DTI_i N_j] = 0.2 DTI_i S_i

which is solved by non-negative least squares, then refined in DTI space.
The result is an INSTRUMENT (a projection), never a score.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize


def design(cover_by_bin: np.ndarray, mass: np.ndarray, cells_per_bin: np.ndarray):
    return np.asarray(cover_by_bin, float), np.asarray(mass, float), np.asarray(cells_per_bin, float)


def predict(h: np.ndarray, C: np.ndarray, S: np.ndarray, N: np.ndarray, c: float = 2.0) -> np.ndarray:
    T = C @ h
    K = float(N @ h)
    return T / (0.2 * (T + S - T / c) + 0.8 * K + 1e-12)


def fit_nnls(C, S, N, y, c=2.0, ridge=0.0, w=None):
    """Linearised NNLS fit of h >= 0 (optionally ridge-regularised toward 0)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    A = C * (1.0 - 0.2 * y * (1.0 - 1.0 / c))[:, None] - 0.8 * y[:, None] * N[None, :]
    b = 0.2 * y * S
    # scale rows by 1/(0.2*S) so every equation is in DTI units (comparable residuals)
    sc = w / (0.2 * S + 0.8 * 1e4)
    A, b = A * sc[:, None], b * sc
    # column scaling for conditioning
    colsc = np.maximum(np.abs(A).max(axis=0), 1e-12)
    As = A / colsc
    if ridge > 0:
        As = np.vstack([As, np.sqrt(ridge) * np.eye(As.shape[1])])
        b = np.concatenate([b, np.zeros(As.shape[1])])
    hs, _ = optimize.nnls(As, b, maxiter=5000)
    return hs / colsc


def refine(h0, C, S, N, y, c=2.0, w=None, l2=0.0):
    """Refine h in DTI space (log-parameterised for positivity)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    eps = 1e-9
    x0 = np.log(np.maximum(h0, eps))

    def resid(x):
        h = np.exp(x)
        r = w * (predict(h, C, S, N, c) - y)
        if l2 > 0:
            r = np.concatenate([r, np.sqrt(l2) * (x - x0)])
        return r

    sol = optimize.least_squares(resid, x0, method="trf", max_nfev=4000)
    return np.exp(sol.x)


def loo(C, S, N, y, c=2.0, ridge=0.0, groups=None, do_refine=False):
    """Leave-one-out (or leave-one-group-out) predictions."""
    y = np.asarray(y, float)
    n = len(y)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    pred = np.full(n, np.nan)
    for gval in np.unique(groups):
        te = groups == gval
        tr = ~te
        h = fit_nnls(C[tr], S[tr], N, y[tr], c=c, ridge=ridge)
        if do_refine:
            h = refine(h, C[tr], S[tr], N, y[tr], c=c)
        pred[te] = predict(h, C[te], S[te], N, c)
    return pred
