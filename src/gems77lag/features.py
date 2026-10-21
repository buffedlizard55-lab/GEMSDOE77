"""Pre-registered H77L lagged Th/K rank-intensity versus lidar micro-scarp.

No catalogue or other submissions enter this transform. The uint8 channels
are separately quantized and are *not* calibrated radionuclide concentrations.
All numerical choices are frozen in registry/preregistration_h77l.json.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EROSION_PX = 16
# SciPy gaussian_filter's default radius is int(truncate*sigma + .5).
# Include the central-difference gradient between the two smoothing stages.
TENSOR_STENCIL_PX = int(4 * 1.3 + .5) + 1 + int(4 * 2.5 + .5)
if EROSION_PX < TENSOR_STENCIL_PX:
    raise ValueError("filter stencil reaches missing cells: increase erosion before testing")
FIELDS = ("lidar_step", "coh100", "ThK_rank", "tensor_anisotropy",
          "lag_response", "same_pixel_edge", "candidate")


def _sample_along_normal(image: np.ndarray, ny: np.ndarray, nx: np.ndarray,
                         distance: float) -> np.ndarray:
    """Linear resampling at a fixed offset; all stencil calls stay in eroded support."""
    h, w = image.shape
    coords = np.empty((2, h, w), np.float32)
    coords[0] = np.arange(h, dtype=np.float32)[:, None] + distance * ny
    coords[1] = np.arange(w, dtype=np.float32)[None, :] + distance * nx
    result = ndi.map_coordinates(image, coords, order=1, mode="nearest", prefilter=False)
    return np.asarray(result, np.float32)


def normalize_surface(response: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, dict]:
    """Label-free top-tail scaling; a degenerate saturated control emits nothing."""
    values = response[eligible & (response > 0)]
    out = np.zeros(response.shape, dtype=np.float32)
    if not values.size:
        return out, {"positive_before": 0, "p99": None, "p999": None,
                     "positive_after": 0, "reason": "no positive signal"}
    low, high = [float(x) for x in np.quantile(values, [0.99, 0.999])]
    if high > low:
        good = eligible & (response > low)
        out[good] = np.minimum((response[good] - low) / (high - low), 1.0)
        reason = "pre-registered >p99 of positive eligible values"
    else:
        # A tied/saturated top tail contains no ordering: don't pretend the
        # nearly ubiquitous maximum is a sparse, calibrated prediction.
        reason = "p99 and p999 tied: degenerate control, no mass emitted"
    return out, {"positive_before": int(values.size), "p99": low, "p999": high,
                 "positive_after": int((out > 0).sum()), "reason": reason}


def build(step_u8: np.ndarray, coh_u8: np.ndarray, lidar_valid_u8: np.ndarray,
          thk_u8: np.ndarray, footprint: np.ndarray) -> tuple[dict, dict, np.ndarray, dict]:
    """Return raw single-feature canaries and normalized candidate/control arms.

    Erosion by the 3x3 square 16 times excludes every missing-value boundary:
    SciPy's default Gaussian radius round(4*sigma) gives lidar 5px and tensor
    10px, plus 1px for gradient = 16px; ThK radius 5px + 3px bilinear offset
    + 1px interpolation is <16px. Data are zero-filled only for filtering;
    no output near the fill remains.
    """
    inputs = (step_u8, coh_u8, lidar_valid_u8, thk_u8)
    footprint = np.asarray(footprint, bool)
    if footprint.ndim != 2 or any(a.shape != footprint.shape or a.dtype != np.uint8 for a in inputs):
        raise ValueError("four grid-aligned uint8 bands and a 2-D footprint are required")
    support = footprint & (lidar_valid_u8 > 0) & (step_u8 > 0) & (thk_u8 > 0)
    eligible = ndi.binary_erosion(support, structure=np.ones((3, 3), bool),
                                  iterations=EROSION_PX, border_value=0)
    if not eligible.any():
        raise ValueError("no filter-safe joint lidar/ThK support")

    step = (step_u8.astype(np.float32) - 1.0) / 254.0
    coh = (coh_u8.astype(np.float32) - 1.0) / 254.0
    thk = (thk_u8.astype(np.float32) - 1.0) / 254.0
    step[~support] = 0.0
    coh[~support] = 0.0
    thk[~support] = 0.0
    np.maximum(step, 0.0, out=step)
    np.maximum(coh, 0.0, out=coh)
    np.maximum(thk, 0.0, out=thk)

    smoothed_step = ndi.gaussian_filter(step, 1.3, truncate=4, mode="constant")
    gy, gx = np.gradient(smoothed_step)
    del smoothed_step
    xx = ndi.gaussian_filter(gx * gx, 2.5, truncate=4, mode="constant")
    yy = ndi.gaussian_filter(gy * gy, 2.5, truncate=4, mode="constant")
    xy = ndi.gaussian_filter(gx * gy, 2.5, truncate=4, mode="constant")
    del gx, gy
    # The maximum-eigenvalue eigenvector is the normal to the dominant scarp
    # gradient; the ratio of eigenvalue difference to trace measures coherence.
    delta = np.sqrt((xx - yy)**2 + 4.0 * xy**2)
    anisotropy = delta / (xx + yy + 1e-8)
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    del delta, xx, yy, xy
    nx, ny = np.cos(angle), np.sin(angle)
    del angle
    thk_smoothed = ndi.gaussian_filter(thk, 1.2, truncate=4, mode="constant")
    ry, rx = np.gradient(thk_smoothed)
    near = 2.0 * np.abs(nx * rx + ny * ry)
    del ry, rx
    plus = _sample_along_normal(thk_smoothed, ny, nx, 3.0)
    plus = np.abs(plus - thk_smoothed)
    minus = _sample_along_normal(thk_smoothed, ny, nx, -3.0)
    np.subtract(minus, thk_smoothed, out=minus)
    np.abs(minus, out=minus)
    np.maximum(plus, minus, out=plus)
    del minus, nx, ny, thk_smoothed
    lag = plus - 1.25 * near
    np.maximum(lag, 0.0, out=lag)
    del plus
    same_pixel = step * coh * near
    del near
    candidate = step * coh * anisotropy * lag
    raw = dict(zip(FIELDS, (step, coh, thk, anisotropy, lag, same_pixel, candidate)))
    # All fields and normalization use covariates only. Values outside eligible
    # are zero before AUC/DTI, not after deriving the feature on label pixels.
    for v in raw.values():
        v[~eligible] = 0.0
        if not np.isfinite(v).all() or (v < 0).any():
            raise ValueError("non-finite/negative feature; do not silently repair")
    fields, scaling = {}, {}
    for name, response in raw.items():
        fields[name], scaling[name] = normalize_surface(response, eligible)
    return raw, fields, eligible, scaling
