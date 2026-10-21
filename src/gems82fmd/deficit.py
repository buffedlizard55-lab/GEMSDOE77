"""Censored catalogue-intensity model: the H82-FMD mapping deficit.

The mapped fault catalogue is treated as a spatially *censored* realisation of
an unknown fault network.  A Poisson-loss gradient-boosted regressor predicts
the mapped fault-pixel rate of a 2 km block from block-aggregated physics; the
positive residual

    D(b) = max(0, lambda_hat(b) - n_visible(b))

estimates how many fault pixels the setting of that block implies but the
catalogue does not contain.  The estimand of the competition metric is exactly
"faults absent from the catalogue", which is why the residual - and not the
raw intensity - is the quantity this lane allocates against.

Nothing in this module uses a per-pixel distance to the catalogue, so the
spatial PU self-distance tautology (IRR-03) cannot occur: the catalogue enters
only as an aggregate count per 2 km block.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

BLOCK_PX = 20          # 2 km at the 100 m competition grid
MIN_BLOCK_FP = 200     # >= 50 % of a block inside the footprint to be trainable
SMOOTH_PX = 10.0       # deficit smoothing (half a block) before use as a weight
FLOOR = 0.10           # multiplicative floor so the physics surface is never zeroed

HGB_POISSON = dict(loss="poisson", max_iter=200, learning_rate=0.06,
                   max_leaf_nodes=31, min_samples_leaf=20,
                   l2_regularization=1.0, early_stopping=False, random_state=8201)


def block_ids(shape: tuple[int, int], block_px: int = BLOCK_PX):
    """Per-pixel block index plus the block grid shape."""
    h, w = shape
    nby = (h + block_px - 1) // block_px
    nbx = (w + block_px - 1) // block_px
    yy = np.arange(h, dtype=np.int32)[:, None] // block_px
    xx = np.arange(w, dtype=np.int32)[None, :] // block_px
    return (yy * nbx + xx).astype(np.int32), (nby, nbx)


def aggregate(stack_fp: np.ndarray, blk_of_px: np.ndarray, n_blocks: int) -> np.ndarray:
    """Per-block mean and std of every channel (NaN-aware).

    ``stack_fp`` is the footprint-compacted channel matrix and ``blk_of_px``
    the block index of each of its rows.
    """
    n_ch = stack_fp.shape[1]
    out = np.zeros((n_blocks, 2 * n_ch), np.float32)
    for j in range(n_ch):
        col = np.asarray(stack_fp[:, j], np.float64)
        ok = np.isfinite(col)
        cnt = np.bincount(blk_of_px[ok], minlength=n_blocks)
        s1 = np.bincount(blk_of_px[ok], weights=col[ok], minlength=n_blocks)
        s2 = np.bincount(blk_of_px[ok], weights=col[ok] ** 2, minlength=n_blocks)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
            var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
        out[:, 2 * j] = mean
        out[:, 2 * j + 1] = np.sqrt(np.maximum(var, 0.0))
    return out


def block_counts(mask: np.ndarray, blk: np.ndarray, n_blocks: int) -> np.ndarray:
    """Number of True pixels of ``mask`` per block."""
    idx = blk[mask]
    return np.bincount(idx, minlength=n_blocks).astype(np.float64)


def deficit_field(block_feat: np.ndarray, n_fp_blk: np.ndarray, n_vis_blk: np.ndarray,
                  train_blocks: np.ndarray, predict_blocks: np.ndarray,
                  blk: np.ndarray, shape: tuple[int, int],
                  footprint: np.ndarray, *, smooth_px: float = SMOOTH_PX,
                  return_lambda: bool = False):
    """Fit the censored-intensity model and return the pixel-space weight in [0,1].

    ``train_blocks`` / ``predict_blocks`` are boolean masks over blocks; they
    must be disjoint in the spatially blocked design (train = other quadrants).
    """
    tr = train_blocks & (n_fp_blk >= MIN_BLOCK_FP)
    if tr.sum() < 50:
        raise ValueError("too few trainable blocks for the intensity model")
    rate = n_vis_blk[tr] / np.maximum(n_fp_blk[tr], 1.0)
    model = HistGradientBoostingRegressor(**HGB_POISSON)
    model.fit(block_feat[tr], rate, sample_weight=n_fp_blk[tr])

    lam = np.zeros(block_feat.shape[0], np.float64)
    pb = predict_blocks & (n_fp_blk > 0)
    if pb.any():
        lam[pb] = np.clip(model.predict(block_feat[pb]), 0.0, None) * n_fp_blk[pb]
    deficit = np.maximum(0.0, lam - n_vis_blk)
    deficit[~pb] = 0.0

    dmap = deficit[blk].astype(np.float32)
    dmap[~footprint] = 0.0
    if smooth_px > 0:
        dmap = ndimage.gaussian_filter(dmap, smooth_px, mode="nearest")
        dmap[~footprint] = 0.0
    inside = dmap[footprint & (dmap > 0)]
    scale = float(np.quantile(inside, 0.90)) if inside.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    weight = np.clip(dmap / scale, 0.0, 1.0).astype(np.float32)
    if return_lambda:
        return weight, lam, deficit
    return weight


def apply_weight(surface: np.ndarray, weight: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """``S_fmd = S_phys * (floor + (1 - floor) * W)`` -- never zeroes the physics."""
    return (np.asarray(surface, np.float32)
            * (np.float32(floor) + np.float32(1.0 - floor) * np.asarray(weight, np.float32)))
