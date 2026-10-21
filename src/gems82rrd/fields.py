"""H82-RRD label-free fields from training_features bands 1/2/14.

Physics: RTP (band 2) is derived from TMI (band 14) assuming induced
magnetization.  Where remanent magnetization or vertical-axis block rotation
is present, the local linear relation RTP_hp ~ a*TMI_hp + b breaks: slope a(x),
correlation r(x) and residual energy change across the block boundary, i.e.
across the fault that bounds the block.  All operators here are Gaussian
filters / local moments -- no labels, no catalogue, no prior rasters.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

RRD_CHANNELS = [
    "rrd_resid_rms",   # log1p local RMS regression residual (RTP vs TMI)
    "rrd_corr",        # local correlation r(x) in [-1, 1]
    "rrd_corr_break",  # |grad r| smoothed
    "rrd_slope",       # local slope a(x), clipped
    "rrd_slope_break", # |grad a| smoothed
    "rrd_disc_abs",    # |z(hp RTP) - z(hp TMI)| smoothed
    "rrd_disc_grad",   # |grad (z(hp RTP) - z(hp TMI))|
    "rrd_mag_resid",   # log1p residual RMS, second pair (mag_anom vs RTP)
]

_EPS = 1e-12


def clean_band(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Sentinel/NaN -> footprint median (same policy as prior lanes)."""
    a = np.asarray(a, dtype=np.float32)
    bad = (~np.isfinite(a)) | (a < -1e30) | (a > 1e30)
    vals = a[footprint & ~bad]
    med = float(np.median(vals)) if vals.size else 0.0
    return np.where(bad, med, a).astype(np.float32)


def _g(a, s):
    return ndimage.gaussian_filter(a, sigma=s, mode="nearest")


def _gradmag(a, s=1.5):
    sm = _g(a, s)
    gy, gx = np.gradient(sm)
    return np.hypot(gy, gx).astype(np.float32)


def _local_regression(u: np.ndarray, v: np.ndarray, sw: float = 2.5):
    """Gaussian-weighted local moments of v ~ a*u + b. Returns a, r, resid_rms."""
    mu_u = _g(u, sw); mu_v = _g(v, sw)
    var_u = np.maximum(_g(u * u, sw) - mu_u * mu_u, 0.0)
    var_v = np.maximum(_g(v * v, sw) - mu_v * mu_v, 0.0)
    cov = _g(u * v, sw) - mu_u * mu_v
    a = cov / (var_u + _EPS)
    r = cov / np.sqrt(var_u * var_v + _EPS)
    resid = (v - mu_v) - a * (u - mu_u)
    resid_rms = np.sqrt(np.maximum(_g(resid * resid, sw), 0.0))
    return (a.astype(np.float32), np.clip(r, -1, 1).astype(np.float32),
            resid_rms.astype(np.float32))


def _robust_z(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    vals = a[footprint]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return ((a - med) / (mad + _EPS)).astype(np.float32)


def rank01(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Footprint percentile ranks in [0,1]; 0 outside footprint."""
    out = np.zeros(a.shape, np.float32)
    vals = a[footprint]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, np.float64)
    ranks[order] = np.arange(vals.size, dtype=np.float64)
    out[footprint] = (ranks / max(vals.size - 1, 1)).astype(np.float32)
    return out


def build_rrd(tmi: np.ndarray, rtp: np.ndarray, mag_anom: np.ndarray,
              footprint: np.ndarray) -> dict[str, np.ndarray]:
    """8 RRD channels + composite surface. Inputs must already be cleaned."""
    hp_tmi = (tmi - _g(tmi, 3.0)).astype(np.float32)
    hp_rtp = (rtp - _g(rtp, 3.0)).astype(np.float32)
    hp_mag = (mag_anom - _g(mag_anom, 3.0)).astype(np.float32)

    a, r, resid_rms = _local_regression(hp_tmi, hp_rtp)
    _, _, resid_mag = _local_regression(hp_rtp, hp_mag)

    z_diff = (_robust_z(hp_rtp, footprint) - _robust_z(hp_tmi, footprint)).astype(np.float32)
    ch = {
        "rrd_resid_rms": np.log1p(np.maximum(resid_rms, 0)).astype(np.float32),
        "rrd_corr": r,
        "rrd_corr_break": _gradmag(r, 1.5),
        "rrd_slope": np.clip(a, -3.0, 3.0).astype(np.float32),
        "rrd_slope_break": _gradmag(np.clip(a, -3.0, 3.0), 1.5),
        "rrd_disc_abs": _g(np.abs(z_diff), 1.5).astype(np.float32),
        "rrd_disc_grad": _gradmag(z_diff, 1.5),
        "rrd_mag_resid": np.log1p(np.maximum(resid_mag, 0)).astype(np.float32),
    }
    for v in ch.values():
        v[~footprint] = 0.0
    # Preregistered composite: geometric mean of the 4 break/energy ranks.
    parts = [rank01(ch[k], footprint) for k in
             ("rrd_resid_rms", "rrd_corr_break", "rrd_slope_break", "rrd_disc_grad")]
    comp = np.exp(np.mean([np.log(np.maximum(p, 1e-6)) for p in parts], axis=0))
    ch["S_rrd"] = np.where(footprint, comp, 0.0).astype(np.float32)
    return ch
