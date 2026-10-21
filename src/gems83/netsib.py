"""H83-NETSIB: visible fault-network sibling / gap / continuation geometry.

Every function here takes ONLY a boolean catalogue-state mask (the *visible*
catalogue in a holdout, a synthetically re-withheld catalogue for training rows,
or the full catalogue for the final build) plus the footprint. No truth, label
or hidden mask is ever read. See registry/preregistration_h83_netsib.json.

Geometry (fixed before any experiment):
* strike of visible traces from the structure tensor of the Gaussian-smoothed
  mask (sigma1 = 1 px, sigma2 = 3 px); 12 strike bins of 15 degrees;
* per bin, FFT convolution of the bin's trace mask with oriented kernels:
  forward / backward along-strike half kernels (|a| <= 25 px, lateral sigma 1.5
  px, exp(-|a|/12)) and a parallel-sibling band (4 <= |c| <= 20 px across
  strike, |a| <= 15 px, Gaussian along-strike sigma 8 px);
* isotropic Gaussian network density (sigma 4 / 12 / 32 px), log1p distance to
  the nearest trace, and local strike coherence (doubled-angle resultant at
  sigma 20 px).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import oaconvolve

N_BINS = 12
R_KERNEL = 25
NAMES = (
    "ns_logdist", "ns_den4", "ns_den12", "ns_den32",
    "ns_cont", "ns_gap", "ns_sib", "ns_cont_max", "ns_sib_max", "ns_coh",
)
STRUCT8 = np.ones((3, 3), bool)


def strike_field(mask: np.ndarray) -> np.ndarray:
    """Line direction theta in [0, pi) (x = column, y = row) at every pixel."""
    i = ndi.gaussian_filter(mask.astype(np.float32), 1.0)
    gy = ndi.sobel(i, axis=0)
    gx = ndi.sobel(i, axis=1)
    jxx = ndi.gaussian_filter(gx * gx, 3.0)
    jyy = ndi.gaussian_filter(gy * gy, 3.0)
    jxy = ndi.gaussian_filter(gx * gy, 3.0)
    phi = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)       # gradient (normal) direction
    theta = np.mod(phi + np.pi / 2.0, np.pi)           # trace direction
    return theta.astype(np.float32)


def kernels(theta: float, r: int = R_KERNEL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward half, backward half and sibling kernels for trace direction theta."""
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    ux, uy = np.cos(theta), np.sin(theta)
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    lat = np.exp(-0.5 * (across / 1.5) ** 2) * np.exp(-np.abs(along) / 12.0) * (np.abs(along) <= 25)
    kf = (lat * (along > 0.5)).astype(np.float32)
    kb = (lat * (along < -0.5)).astype(np.float32)
    ks = ((np.abs(across) >= 4) & (np.abs(across) <= 20) & (np.abs(along) <= 15)).astype(np.float32)
    ks *= np.exp(-0.5 * (along / 8.0) ** 2).astype(np.float32)
    kf /= kf.sum()
    kb /= kb.sum()
    ks /= ks.sum()
    return kf, kb, ks


def netsib_fields(mask: np.ndarray, footprint: np.ndarray) -> dict[str, np.ndarray]:
    """All NETSIB channels (float32 full-grid) from one catalogue-state mask."""
    m = np.asarray(mask, bool) & np.asarray(footprint, bool)
    if not m.any():
        raise ValueError("empty catalogue-state mask")
    out: dict[str, np.ndarray] = {}
    out["ns_logdist"] = np.log1p(ndi.distance_transform_edt(~m)).astype(np.float32)
    mf = m.astype(np.float32)
    for s in (4, 12, 32):
        out[f"ns_den{s}"] = ndi.gaussian_filter(mf, float(s)).astype(np.float32)
    theta = strike_field(m)
    bins = np.minimum((theta / np.pi * N_BINS).astype(np.int16), N_BINS - 1)
    cont = np.zeros(m.shape, np.float32)
    gap = np.zeros(m.shape, np.float32)
    sib = np.zeros(m.shape, np.float32)
    cont_max = np.zeros(m.shape, np.float32)
    sib_max = np.zeros(m.shape, np.float32)
    for k in range(N_BINS):
        mk = (m & (bins == k)).astype(np.float32)
        if not mk.any():
            continue
        kf, kb, ks = kernels((k + 0.5) * np.pi / N_BINS)
        f = np.maximum(oaconvolve(mk, kf, mode="same"), 0.0).astype(np.float32)
        b = np.maximum(oaconvolve(mk, kb, mode="same"), 0.0).astype(np.float32)
        s = np.maximum(oaconvolve(mk, ks, mode="same"), 0.0).astype(np.float32)
        c = f + b
        cont += c
        gap += np.sqrt(f * b)
        sib += s
        np.maximum(cont_max, c, out=cont_max)
        np.maximum(sib_max, s, out=sib_max)
        del f, b, s, c, mk
    out.update(ns_cont=cont, ns_gap=gap, ns_sib=sib, ns_cont_max=cont_max, ns_sib_max=sib_max)
    # local strike coherence of the network (doubled-angle resultant)
    w = mf
    c2 = ndi.gaussian_filter(w * np.cos(2 * theta), 20.0)
    s2 = ndi.gaussian_filter(w * np.sin(2 * theta), 20.0)
    den = ndi.gaussian_filter(w, 20.0)
    out["ns_coh"] = (np.hypot(c2, s2) / (den + 1e-6) * (den > 1e-4)).astype(np.float32)
    fp = np.asarray(footprint, bool)
    for k in out:
        out[k] = np.where(fp, out[k], 0.0).astype(np.float32)
    return out


def synthetic_withhold(mask: np.ndarray, footprint: np.ndarray, quad: np.ndarray,
                       seed: int, frac: float = 0.20, collar_px: int = 3):
    """Re-withhold whole 8-connected components of ``mask`` (same rule as the
    shared evaluator's buffered_component_draw): per quadrant, random components
    lying wholly in that quadrant until ``frac`` of its mask pixels; a
    ``collar_px`` context buffer is removed from the remaining state.

    Returns (hidden, state) where ``state`` is the catalogue state that the
    hidden pixels' features must be computed from.
    """
    m = np.asarray(mask, bool) & np.asarray(footprint, bool)
    comp, n = ndi.label(m, STRUCT8)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    hidden = np.zeros(m.shape, bool)
    for q in range(4):
        inq = quad == q
        cand = np.unique(comp[m & inq])
        clipped = np.unique(comp[m & ~inq])
        elig = np.setdiff1d(cand[cand > 0], clipped)
        if elig.size == 0:
            continue
        rng = np.random.default_rng(830_000 + 97 * seed + q)
        perm = rng.permutation(elig)
        cum = np.cumsum(sizes[perm])
        k = int(np.searchsorted(cum, frac * float((m & inq).sum()))) + 1
        hidden |= np.isin(comp, perm[:min(k, perm.size)]) & m
    collar = ndi.distance_transform_edt(~hidden) <= collar_px
    state = m & ~collar
    assert not (state & hidden).any()
    return hidden, state


def deterministic_score(fields: dict[str, np.ndarray], footprint: np.ndarray) -> np.ndarray:
    """Label-free NETSIB-only rank score used solely for the Exp-1 surface check."""
    fp = np.asarray(footprint, bool)
    acc = np.zeros(fp.shape, np.float64)
    for name, sign in (("ns_gap", 1), ("ns_sib", 1), ("ns_cont", 1), ("ns_logdist", -1)):
        v = sign * fields[name][fp].astype(np.float64)
        r = np.argsort(np.argsort(v, kind="stable"), kind="stable") / max(v.size - 1, 1)
        acc[fp] += r
    acc /= 4.0
    return np.where(fp, acc, 0.0).astype(np.float32)
