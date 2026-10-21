"""H82-TCSTEP label-free fields from core band 6 (radiometric total count).

Physical signature (registry/preregistration_h82.json): a concealed normal
fault laterally offsets radiometrically distinct lithologies, producing a
sharp, strike-persistent step in airborne gamma total count where the
surface shows no scarp (low det_elev_slope, band 19).

Channels (all deterministic, label-free, footprint-ranked, tie-averaged):
  tc_step2 : rank01(|grad log TC| at sigma=2)
  tc_step5 : rank01(|grad log TC| at sigma=5)
  tc_coh   : structure-tensor coherence of log TC at sigma=3, in [0,1]
  tc_blind : sqrt(rank01(tc_step2_lin * (0.25 + 0.75*tc_coh))
                  * rank01(-det_elev_slope))
             where tc_step2_lin is the unranked sigma=2 step magnitude.

No catalogue-derived quantity enters any channel. NaN/sentinel handling:
values below half the float32 nodata sentinel or non-finite are set NaN,
then nearest-filled ONLY so spatial filters see a continuous field; ranks
are computed over footprint pixels of the filled field (filled cells inside
the footprint inherit a neighbour's value, which is the standard in-repo
convention, cf. src/gems77/features77.py fill_nearest).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

SENT = -3.4028234663852886e38

H82_NAMES = ["tc_step2", "tc_step5", "tc_coh", "tc_blind"]


def clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.where((a < np.float32(SENT * 0.5)) | ~np.isfinite(a),
                    np.float32(np.nan), a)


def fill_nearest(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a.astype(np.float32)
    if bad.all():
        raise ValueError("cannot nearest-fill an all-NaN field")
    idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                         return_indices=True)
    return a[tuple(idx)].astype(np.float32)


def rank01_tieavg(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tie-averaged percentile rank in [0,1] over mask; 0 outside mask.

    Tie-averaging matters (in-repo lesson IR-H80-04): scan-order tie
    spreading turns constant plateaus into spurious spatial gradients.
    """
    out = np.zeros(a.shape, np.float32)
    v = np.asarray(a, dtype=np.float64)[mask]
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return out
    r = np.zeros(v.size, np.float64)
    r[finite] = rankdata(v[finite], method="average") / float(finite.sum())
    out[mask] = r.astype(np.float32)
    return out


def _grad_mag(a: np.ndarray, sigma: float) -> np.ndarray:
    gx = ndimage.gaussian_filter(a, sigma, order=(0, 1), mode="nearest")
    gy = ndimage.gaussian_filter(a, sigma, order=(1, 0), mode="nearest")
    return np.hypot(gx, gy).astype(np.float32)


def _coherence(a: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    gx = ndimage.gaussian_filter(a, sigma, order=(0, 1), mode="nearest")
    gy = ndimage.gaussian_filter(a, sigma, order=(1, 0), mode="nearest")
    jxx = ndimage.gaussian_filter(gx * gx, sigma, mode="nearest")
    jyy = ndimage.gaussian_filter(gy * gy, sigma, mode="nearest")
    jxy = ndimage.gaussian_filter(gx * gy, sigma, mode="nearest")
    tr = jxx + jyy + 1e-12
    coh = 2.0 * np.sqrt(np.maximum(((jxx - jyy) * 0.5) ** 2 + jxy ** 2, 0.0)) / tr
    return np.clip(coh, 0.0, 1.0).astype(np.float32)


def build_h82_fields(tc_raw: np.ndarray, slope_raw: np.ndarray,
                     footprint: np.ndarray) -> dict[str, np.ndarray]:
    """Build the 4 H82 channels from raw band-6 TC and band-19 slope arrays."""
    foot = np.asarray(footprint, bool)
    tc = clean(tc_raw)
    if not np.isfinite(tc[foot]).any():
        raise ValueError("band 6 has no finite values inside the footprint")
    # log TC: TC is a positive count rate (in-footprint range measured
    # 2.95..88.57 in evidence/band_diagnostics.json); clip at a small
    # positive floor so any stray zero/negative cannot produce -inf.
    tc_pos = np.where(np.isfinite(tc), np.maximum(tc, np.float32(1e-3)), tc)
    logtc = fill_nearest(np.log(tc_pos).astype(np.float32))

    slope = fill_nearest(clean(slope_raw))
    slope_mag = np.abs(slope)

    step2 = _grad_mag(logtc, 2.0)
    step5 = _grad_mag(logtc, 5.0)
    coh = _coherence(logtc, 3.0)

    tc_step2 = rank01_tieavg(step2, foot)
    tc_step5 = rank01_tieavg(step5, foot)
    persistent = step2 * (0.25 + 0.75 * coh)
    low_slope = rank01_tieavg(-slope_mag, foot)
    tc_blind = np.sqrt(rank01_tieavg(persistent, foot) * low_slope).astype(np.float32)
    tc_blind = np.where(foot, tc_blind, 0.0).astype(np.float32)

    fields = {
        "tc_step2": tc_step2,
        "tc_step5": tc_step5,
        "tc_coh": np.where(foot, coh, 0.0).astype(np.float32),
        "tc_blind": tc_blind,
    }
    for name, f in fields.items():
        if not np.isfinite(f).all() or f.min() < 0.0 or f.max() > 1.0:
            raise ValueError(f"H82 field {name} violates finite [0,1] contract")
    return fields
