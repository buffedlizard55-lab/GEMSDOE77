"""Physical fields for GEMSDOE77, computed only from the official training features.

H77-1  straight-edge field ("SEF"): thin topographic edge lines that stay straight over
       km scales.  det_elev (band 12) is smoothed at sigma = 1 and 2 px; the gradient
       magnitude is non-maximum-suppressed across the edge normal (Canny-style thin
       lines); each surviving edge pixel is weighted by the orientation coherence of
       the structure tensor integrated over ~1.5 km (sigma = 7 px).  Faults are straight
       across ridges and valleys; drainage and lithologic edges are sinuous, so the
       coherence weight suppresses them.  Differs from GEMSDOE49 gate_ortho (gradient
       magnitude x orthogonality gate, no NMS) and from GEMSDOE32 scarp (profile
       curvature).
H77-2  strain-catalogue deficit ("SCD"): geodetic second invariant (band 4) and shear
       rate (band 7), rank-normalised, minus the rank of the local catalogue density
       within ~10 km.  Strain must be accommodated by faults; where strain is high but
       few faults are mapped, the catalogue is likely incomplete.  Used as a regional
       allocator, not a line detector.  No prior GEMSDOE repo used this ratio.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import core


def _filled(a: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Fill NaN by nearest valid value (avoids artificial edges at the footprint border)."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def straight_edge_field(sigmas=(1.0, 2.0), coh_sigma: float = 7.0) -> np.ndarray:
    g = core.grid()
    z = core.read_band(12)
    z = _filled(z, g.footprint).astype(np.float32)
    out = np.zeros_like(z)
    for s in sigmas:
        zs = ndimage.gaussian_filter(z, s)
        gy = ndimage.sobel(zs, axis=0) / 8.0
        gx = ndimage.sobel(zs, axis=1) / 8.0
        mag = np.hypot(gx, gy)
        # non-maximum suppression across the edge normal (8 directions)
        ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
        q = (np.round(ang / 45.0).astype(np.int8)) % 4   # 0:E-W normal, 1:45, 2:N-S, 3:135
        nms = np.zeros_like(mag, dtype=bool)
        offs = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
        for k, (dy, dx) in offs.items():
            a = np.roll(mag, (dy, dx), axis=(0, 1))
            b = np.roll(mag, (-dy, -dx), axis=(0, 1))
            nms |= (q == k) & (mag >= a) & (mag >= b)
        # structure-tensor orientation coherence at km scale
        jxx = ndimage.gaussian_filter(gx * gx, coh_sigma)
        jyy = ndimage.gaussian_filter(gy * gy, coh_sigma)
        jxy = ndimage.gaussian_filter(gx * gy, coh_sigma)
        tr = jxx + jyy
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-12)
        # rank-normalise magnitude inside the footprint so scales combine fairly
        m = mag.copy()
        v = m[g.footprint]
        r = np.empty_like(v)
        r[np.argsort(v)] = np.linspace(0, 1, v.size, dtype=np.float32)
        m[g.footprint] = r
        out = np.maximum(out, (m * coh * nms).astype(np.float32))
    out[~g.footprint] = 0
    return out


def strain_catalogue_deficit(radius_px: float = 50.0) -> np.ndarray:
    g = core.grid()

    def rank(a):
        a = _filled(a, g.footprint)
        v = a[g.footprint]
        r = np.empty_like(v)
        r[np.argsort(v)] = np.linspace(0, 1, v.size, dtype=np.float32)
        out = np.zeros_like(a, dtype=np.float32)
        out[g.footprint] = r
        return out

    strain = 0.5 * (rank(core.read_band(4)) + rank(core.read_band(7)))
    dens = ndimage.gaussian_filter(g.catalogue.astype(np.float32), radius_px / 2.0)
    deficit = strain - rank(dens)
    deficit[~g.footprint] = 0
    return deficit.astype(np.float32)
