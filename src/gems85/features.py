"""H85-EDTI label-free feature surface.

Every channel here is derived from the nineteen *official* ``training_features.tif``
bands only.  No channel reads the fault catalogue, so the same code path is used
per holdout fold and for the final full-catalogue build -- there is no
"distance to visible fault" tautology (the failure mode logged as ``IRR-03`` by
this repository) and nothing that cannot transfer to a genuinely unmapped fault.

Band numbering follows the official band descriptions printed by
``rasterio tags`` (verified in ``evidence/h85_band_audit.json``):

===  ==========================================================
1    Magnetic anomaly - deviation from Earth's expected field
2    Reduced to pole magnetic data (RTP)
3    Total magnetic intensity horizontal gradient
4    Geodetic second invariant (strain-rate magnitude)
5    Isostatic gravity anomaly slope
6    Tilt angle / total curvature (magnetic edge detector)
7    Geodetic shear rate
8    Geodetic dilatation rate
9    Total magnetic intensity vertical gradient
10   Distance to earthquake (n=100 km, a=15 deg)
11   Isostatic gravity anomaly vertical gradient
12   Detrended elevation
13   Isostatic gravity anomaly
14   Total magnetic intensity
15   Depth to basement surface
16   Earthquake intensity / density
17   Conductivity surface
18   Isostatic gravity anomaly horizontal gradient
19   Detrended elevation slope
===  ==========================================================
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

BAND_NAMES = {
    1: "mag_anomaly_deviation",
    2: "mag_reduced_to_pole",
    3: "tmi_horizontal_gradient",
    4: "geodetic_second_invariant",
    5: "iso_gravity_slope",
    6: "tilt_angle_total_curvature",
    7: "geodetic_shear_rate",
    8: "geodetic_dilatation_rate",
    9: "tmi_vertical_gradient",
    10: "distance_to_earthquake",
    11: "iso_gravity_vertical_gradient",
    12: "detrended_elevation",
    13: "iso_gravity_anomaly",
    14: "total_magnetic_intensity",
    15: "depth_to_basement",
    16: "earthquake_density",
    17: "conductivity_surface",
    18: "iso_gravity_horizontal_gradient",
    19: "detrended_elevation_slope",
}

NODATA = -3.4028234663852886e38


def _clean(band: np.ndarray) -> np.ndarray:
    """Replace the shipped nodata sentinel with NaN and return float32."""
    a = np.asarray(band, np.float32).copy()
    a[a <= -1e37] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


def _fill(a: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of NaN so derivative filters stay local."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _z(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = a[mask]
    v = v[np.isfinite(v)]
    if v.size < 32:
        return np.zeros_like(a, np.float32)
    med = float(np.median(v))
    scale = float(np.percentile(v, 75) - np.percentile(v, 25))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(v.std()) or 1.0
    return ((a - med) / (1.4826 * scale)).astype(np.float32)


def _grad(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy).astype(np.float32)


def _lap(a: np.ndarray) -> np.ndarray:
    return ndi.laplace(a).astype(np.float32)


def _gauss(a: np.ndarray, s: float) -> np.ndarray:
    return ndi.gaussian_filter(a, s, mode="nearest").astype(np.float32)


def _local_std(a: np.ndarray, s: float) -> np.ndarray:
    m = _gauss(a, s)
    return np.sqrt(np.maximum(_gauss(a * a, s) - m * m, 0.0)).astype(np.float32)


def _ridge(a: np.ndarray) -> np.ndarray:
    """Across-structure second-derivative magnitude (Canny/Blakely ridge operator).

    ``|lambda_1|`` of the Hessian: large on the crest/trough of a ridge or
    trough, zero on a uniform slope.  This is the *correct* across-strike
    operator; ``gems77g`` documented (``IR-H77G-01``) that a naive
    ``small*(big<0)`` variant collapses on straight lineaments.
    """
    dyy = ndi.correlate(a, np.array([[1.0, -2.0, 1.0]]), mode="nearest")
    dxx = ndi.correlate(a, np.array([[1.0], [-2.0], [1.0]]), mode="nearest")
    dxy = ndi.correlate(a, np.array([[1.0, 0.0, -1.0],
                                     [0.0, 0.0, 0.0],
                                     [-1.0, 0.0, 1.0]]) * 0.25, mode="nearest")
    tr = 0.5 * (dxx + dyy)
    det = dxx * dyy - dxy * dxy
    disc = np.sqrt(np.maximum(tr * tr - det, 0.0))
    lam1 = tr + disc
    lam2 = tr - disc
    return np.where(np.abs(lam1) >= np.abs(lam2), lam1, lam2).astype(np.float32)


def build_stack(bands: np.ndarray, mask: np.ndarray, log=print) -> tuple[list[str], np.ndarray]:
    """Return ``(names, stack)`` with ``stack`` shaped ``(n, H, W)`` float32.

    ``bands`` is the raw ``(19, H, W)`` array from ``training_features.tif``;
    ``mask`` is the boolean study-area footprint used for standardisation.
    """
    raw = {i: _clean(bands[i - 1]) for i in BAND_NAMES}
    filled = {i: _fill(raw[i]) for i in raw}

    feats: list[tuple[str, np.ndarray]] = []

    # 1. Every official band, robustly standardised.
    for i, name in BAND_NAMES.items():
        feats.append((f"z_{name}", _z(filled[i], mask)))

    # 2. First-derivative edge energy of the structural bands.
    for i in (2, 3, 6, 9, 12, 13, 14, 15, 17, 19):
        feats.append((f"grad_{BAND_NAMES[i]}", _z(_grad(filled[i]), mask)))

    # 3. Curvature / ridge operators: fault scarps and property contacts are
    #    lineaments, i.e. extrema of the second derivative across strike.
    for i in (2, 6, 12, 13, 14, 15, 17, 19):
        feats.append((f"lap_{BAND_NAMES[i]}", _z(_lap(filled[i]), mask)))
    for i in (2, 6, 12, 13, 14, 15):
        feats.append((f"ridge_{BAND_NAMES[i]}", _z(_ridge(filled[i]), mask)))

    # 4. Multi-scale texture: a 300 m fault is coherent at 2-10 px scales.
    for i in (6, 12, 14, 15, 17):
        for s in (2.0, 5.0):
            feats.append((f"std{s:g}_{BAND_NAMES[i]}", _z(_local_std(filled[i], s), mask)))

    # 5. Coherence between independent sensors: a real fault should perturb
    #    magnetics *and* gravity *and* topography together.
    z = {i: _z(filled[i], mask) for i in BAND_NAMES}
    feats.append(("x_couple_tmi_dem", (z[14] * z[12]).astype(np.float32)))
    feats.append(("x_couple_tmi_grav", (z[14] * z[13]).astype(np.float32)))
    for i in (6, 12, 14, 15, 17):
        feats.append((f"x_couple_ridge_{BAND_NAMES[i]}",
                      (_z(_ridge(filled[i]), mask) * z[19]).astype(np.float32)))

    names = [n for n, _ in feats]
    stack = np.stack([np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                      for _, a in feats], axis=0)
    log(f"[features] {len(names)} channels built")
    return names, stack
