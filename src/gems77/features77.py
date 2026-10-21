"""GEMSDOE77 feature stack: shipped bands + terrain derivatives + the new
expert-criteria contact-physics block (derived tilt / analytic signal /
monogenic phase congruency / pseudogravity-Poisson / front-base asymmetry).

Measured facts this module relies on (evidence/band6_resolution.json):
* band 6 of training_features.tif is radiometric total count (Spearman +1.0000
  vs the external GeoDAWN TC grid), NOT the tilt derivative its tag claims.
  Every tilt channel here is therefore derived from bands 9/3 (magnetics) and
  11/13 (gravity).
* band 18 (iso_grav_anom_hg) is signed, so it is not used as a gradient
  magnitude; gravity THDR is recomputed from band 13.

No channel is derived from the fault catalogue. Sentinel (-3.4e38) and
non-finite cells become NaN in stored channels; filter inputs are filled
(nearest-finite for FFT work, footprint-median elsewhere) and outputs are
masked back to NaN outside the footprint.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

SENT = -3.4028234663852886e38

RAW_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc_totalcount",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15", "iso_grav_anom_vg",
    "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf", "ieq_n100a15",
    "cond_surf", "iso_grav_anom_hg_signed", "det_elev_slope",
]  # band 6 renamed for what it measurably is; band 18 flagged signed

TERRAIN_NAMES = ["dem_curv1", "dem_curv25", "dem_ridge1", "dem_ridge2", "dem_ridge3p5",
                 "dem_relief5", "dem_tophat5", "st_coherence"]

NEW_NAMES = ["mag_tilt", "mag_tilt_abs", "mag_as_log", "grav_thdr_log", "grav_tilt",
             "grav_as_log", "contact_confluence", "base_step_along_maggrad",
             "base_grad_log", "pc_mag", "pc_grav", "pg_slope", "pg_corr",
             "pg_resid_grad_log", "frontness", "profcurv_signed"]

RAD_NAMES = ["rad_K", "rad_Th", "rad_U", "rad_TC", "rad_ThK", "rad_UK"]
LIDAR_NAMES = ["lidar_ex_max", "lidar_step_max", "lidar_downface_max", "lidar_coh100", "lidar_valid"]

BLOCKS = {
    "raw": RAW_NAMES,
    "terrain": TERRAIN_NAMES,
    "new": NEW_NAMES,
    "rad": RAD_NAMES,
    "lidar": LIDAR_NAMES,
}
CHANNELS = RAW_NAMES + TERRAIN_NAMES + NEW_NAMES + RAD_NAMES + LIDAR_NAMES
IDX = {n: i for i, n in enumerate(CHANNELS)}

BAND = {n: i + 1 for i, n in enumerate(RAW_NAMES)}


def clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.where((a < np.float32(SENT * 0.5)) | ~np.isfinite(a), np.float32(np.nan), a)


def fill_median(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(a[mask])) if mask.any() else 0.0
    return np.where(np.isfinite(a), a, np.float32(med)).astype(np.float32)


def fill_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaN by nearest finite neighbour (for FFT inputs)."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a.astype(np.float32)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float32)


def _sm(a, s):
    return ndimage.gaussian_filter(a, s, mode="nearest")


def hessian_ridge(a: np.ndarray, sigma: float) -> np.ndarray:
    axx = ndimage.gaussian_filter(a, sigma, order=(0, 2), mode="nearest")
    ayy = ndimage.gaussian_filter(a, sigma, order=(2, 0), mode="nearest")
    axy = ndimage.gaussian_filter(a, sigma, order=(1, 1), mode="nearest")
    tmp = np.sqrt(np.maximum(((axx - ayy) * 0.5) ** 2 + axy ** 2, 0.0))
    l1 = (axx + ayy) * 0.5 + tmp
    l2 = (axx + ayy) * 0.5 - tmp
    big = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
    small = np.where(np.abs(l1) >= np.abs(l2), l2, l1)
    return (-small * (big < 0)).astype(np.float32)


def structure_tensor(a: np.ndarray, sigma: float = 2.0):
    gx = ndimage.gaussian_filter(a, sigma, order=(0, 1), mode="nearest")
    gy = ndimage.gaussian_filter(a, sigma, order=(1, 0), mode="nearest")
    jxx = _sm(gx * gx, sigma); jyy = _sm(gy * gy, sigma); jxy = _sm(gx * gy, sigma)
    tr = jxx + jyy + 1e-12
    coh = (np.sqrt(np.maximum(((jxx - jyy) * 0.5) ** 2 + jxy ** 2, 0.0)) * 2.0 / tr)
    return np.clip(coh, 0, 1).astype(np.float32)


def profile_curvature(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    gx = ndimage.gaussian_filter(a, sigma, order=(0, 1), mode="nearest")
    gy = ndimage.gaussian_filter(a, sigma, order=(1, 0), mode="nearest")
    axx = ndimage.gaussian_filter(a, sigma, order=(0, 2), mode="nearest")
    ayy = ndimage.gaussian_filter(a, sigma, order=(2, 0), mode="nearest")
    axy = ndimage.gaussian_filter(a, sigma, order=(1, 1), mode="nearest")
    g2 = gx * gx + gy * gy
    num = axx * gx * gx + 2.0 * axy * gx * gy + ayy * gy * gy
    den = np.maximum(g2 * np.sqrt(1.0 + g2), 1e-12)
    return np.where(g2 > 1e-12, num / den, 0.0).astype(np.float32)


def rank01(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1] over mask (NaN-safe on masked values)."""
    out = np.zeros(a.shape, np.float32)
    v = a[mask].astype(np.float64)
    finite = np.isfinite(v)
    order = np.argsort(np.where(finite, v, np.inf), kind="stable")
    r = np.empty(v.size, np.float64)
    r[order] = np.linspace(0.0, 1.0, v.size)
    out[mask] = np.where(finite, r, 0.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Monogenic phase congruency (Felsberg/Kovesi, rfft implementation).
# PC = |sum_n (f_n, h1_n, h2_n)| / sum_n sqrt(f_n^2+h1_n^2+h2_n^2) in [0,1].
# ---------------------------------------------------------------------------
def phase_congruency(im: np.ndarray, nscale: int = 4, min_wavelength: float = 3.0,
                     mult: float = 2.1, sigma_onf: float = 0.55) -> np.ndarray:
    im = np.asarray(im, dtype=np.float64)
    rows, cols = im.shape
    IM = np.fft.rfft2(im)
    fx = np.fft.rfftfreq(cols)[None, :]          # cycles/px, >=0 half
    fy = np.fft.fftfreq(rows)[:, None]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    H1 = 1j * fx / radius                          # Riesz x (columns)
    H2 = 1j * fy / radius                          # Riesz y (rows)
    radius[0, 0] = 0.0
    sum_f = np.zeros((rows, cols)); sum_h1 = np.zeros((rows, cols)); sum_h2 = np.zeros((rows, cols))
    sum_an = np.zeros((rows, cols)); max_an = np.zeros((rows, cols))
    for s in range(nscale):
        wl = min_wavelength * (mult ** s)
        fo = 1.0 / wl
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.exp(-(np.log(np.maximum(radius, 1e-12) / fo) ** 2) /
                        (2.0 * np.log(sigma_onf) ** 2))
        lg[0, 0] = 0.0
        G = IM * lg
        f = np.fft.irfft2(G, s=(rows, cols))
        h1 = np.fft.irfft2(G * H1, s=(rows, cols))
        h2 = np.fft.irfft2(G * H2, s=(rows, cols))
        an = np.sqrt(f * f + h1 * h1 + h2 * h2)
        sum_f += f; sum_h1 += h1; sum_h2 += h2
        sum_an += an; max_an = np.maximum(max_an, an)
        del G, f, h1, h2, lg, an
    energy = np.sqrt(sum_f ** 2 + sum_h1 ** 2 + sum_h2 ** 2)
    pc = energy / (sum_an + 1e-12)
    # Kovesi frequency-spread weighting: a true broadband feature excites every
    # scale (sum_an/max_an -> nscale); single-filter ringing does not.
    if nscale > 1:
        width = (sum_an / (max_an + 1e-12) - 1.0) / (nscale - 1.0)
        weight = 1.0 / (1.0 + np.exp(10.0 * (0.5 - np.clip(width, 0.0, 1.0))))
        pc = pc * weight
    return np.clip(pc, 0.0, 1.0).astype(np.float32)


def pseudogravity(rtp_filled: np.ndarray) -> np.ndarray:
    """Vertical integration of RTP in the Fourier domain (arbitrary scale)."""
    rows, cols = rtp_filled.shape
    F = np.fft.rfft2(rtp_filled.astype(np.float64))
    fx = np.fft.rfftfreq(cols)[None, :]; fy = np.fft.fftfreq(rows)[:, None]
    k = 2.0 * np.pi * np.sqrt(fx * fx + fy * fy)
    k[0, 0] = 1.0
    PG = F / k
    PG[0, 0] = 0.0
    return np.fft.irfft2(PG, s=(rows, cols)).astype(np.float32)


def local_regression(pg_hp: np.ndarray, g_hp: np.ndarray, size: int = 25):
    """Windowed slope/correlation of g on pg via box-filter moments."""
    mx = ndimage.uniform_filter(pg_hp, size, mode="nearest")
    my = ndimage.uniform_filter(g_hp, size, mode="nearest")
    cov = ndimage.uniform_filter(pg_hp * g_hp, size, mode="nearest") - mx * my
    vx = np.maximum(ndimage.uniform_filter(pg_hp * pg_hp, size, mode="nearest") - mx * mx, 0.0)
    vy = np.maximum(ndimage.uniform_filter(g_hp * g_hp, size, mode="nearest") - my * my, 0.0)
    slope = np.where(vx > 1e-12, cov / np.maximum(vx, 1e-12), 0.0)
    corr = np.where((vx > 1e-12) & (vy > 1e-12), cov / np.sqrt(np.maximum(vx * vy, 1e-24)), 0.0)
    return slope.astype(np.float32), np.clip(corr, -1, 1).astype(np.float32)
