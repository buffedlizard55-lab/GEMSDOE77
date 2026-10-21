"""Channel stack for the H82-FMD lane.

19 shipped competition bands (band order verified from the embedded GDAL
descriptions of the SHA-pinned ``training_features.tif``) plus eight derived
channels.  Nothing here reads the fault catalogue, a historical submission, or
an external raster.

Derived channels and their published basis
------------------------------------------
``ridge_s1p5`` / ``ridge_s4``
    ``-lambda_min`` of the Hessian of ``det_elev`` at sigma = 1.5 and 4.0 px.
    The most-negative eigenvalue is the across-strike curvature of a crest, so
    the sign convention is the one that survives on a straight 1-D scarp.  (An
    earlier module in this repository used ``-small * (big < 0)``, which
    vanishes on straight ridges -- see IR-H77G-02; this file does not reuse it.)
``valley_s1p5``
    ``+lambda_max`` of the same Hessian: the conjugate (concave) response that
    marks fault-line valleys and sag ponds.
``relief5``
    5 px (500 m) local relief, ``max - min`` of ``det_elev``.
``tilt_mag``
    Magnetic tilt angle ``atan2(VDR, THDR)`` (Miller & Singh 1994) built from
    shipped band 9 (``tmi_vg``) and band 3 (``tmi_hg``); zero over a contact,
    and amplitude-independent, so weak concealed sources are not suppressed.
``tilt_grav``
    Gravity tilt ``atan2(iso_grav_anom_vg, |iso_grav_anom_hg|)``; band 18 is
    signed (measured, ``evidence/band_diagnostics.json``) so its absolute value
    is used as the horizontal-gradient magnitude.
``mag_edge_log``
    ``log1p`` of the magnetic horizontal gradient: edge strength on a scale
    where a 3-decade dynamic range is usable by a tree model.
``coh3``
    Structure-tensor coherence of ``det_elev`` at sigma = 3 px: how linear the
    local topographic fabric is, independent of its amplitude.
``base_grad_log``
    ``log1p`` of the gradient magnitude of ``depth_to_base_surf``: basement
    steps, which is where a range-bounding normal fault has to be.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

SENTINEL = -3.4028234663852886e38

RAW_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope",
    "tc_totalcount", "geod_shearrate", "geod_dilaterate", "tmi_vg",
    "deq_n100a15", "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi",
    "depth_to_base_surf", "ieq_n100a15", "cond_surf", "iso_grav_anom_hg_signed",
    "det_elev_slope",
]
DERIVED_NAMES = [
    "ridge_s1p5", "ridge_s4", "valley_s1p5", "relief5",
    "tilt_mag", "tilt_grav", "mag_edge_log", "coh3", "base_grad_log",
]
CHANNELS = RAW_NAMES + DERIVED_NAMES
BAND_INDEX = {n: i + 1 for i, n in enumerate(RAW_NAMES)}  # rasterio is 1-based


def read_raw(tif: Path, name: str) -> np.ndarray:
    """One shipped band as float32 with the -3.4e38 sentinel turned into NaN."""
    with rasterio.open(tif) as src:
        a = src.read(BAND_INDEX[name]).astype(np.float32)
    a[a <= np.float32(SENTINEL * 0.5)] = np.float32(np.nan)
    return a


def band_descriptions(tif: Path) -> list[str]:
    with rasterio.open(tif) as src:
        return [d or "" for d in src.descriptions]


def fill_nearest(a: np.ndarray) -> np.ndarray:
    """Replace NaN by the nearest finite value (filter inputs only)."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a.astype(np.float32)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float32)


def _hessian_eigs(z: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """(lambda_min, lambda_max) of the Gaussian-smoothed Hessian, per pixel."""
    zxx = ndimage.gaussian_filter(z, sigma, order=(0, 2), mode="nearest")
    zyy = ndimage.gaussian_filter(z, sigma, order=(2, 0), mode="nearest")
    zxy = ndimage.gaussian_filter(z, sigma, order=(1, 1), mode="nearest")
    tr = zxx + zyy
    diff = zxx - zyy
    root = np.sqrt(np.maximum(diff * diff + 4.0 * zxy * zxy, 0.0))
    lo = 0.5 * (tr - root)
    hi = 0.5 * (tr + root)
    return lo.astype(np.float32), hi.astype(np.float32)


def _coherence(z: np.ndarray, sigma: float) -> np.ndarray:
    gy, gx = np.gradient(z.astype(np.float32))
    jxx = ndimage.gaussian_filter(gx * gx, sigma, mode="nearest")
    jyy = ndimage.gaussian_filter(gy * gy, sigma, mode="nearest")
    jxy = ndimage.gaussian_filter(gx * gy, sigma, mode="nearest")
    tr = jxx + jyy
    root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(tr > 1e-12, root / tr, 0.0)
    return coh.astype(np.float32)


def derived_channels(tif: Path) -> dict[str, np.ndarray]:
    """All nine derived channels, each masked back to NaN where inputs were NaN."""
    out: dict[str, np.ndarray] = {}

    elev = read_raw(tif, "det_elev")
    valid = np.isfinite(elev)
    z = fill_nearest(elev)
    lo15, hi15 = _hessian_eigs(z, 1.5)
    out["ridge_s1p5"] = np.where(valid, -lo15, np.nan).astype(np.float32)
    out["valley_s1p5"] = np.where(valid, hi15, np.nan).astype(np.float32)
    del lo15, hi15
    lo4, _ = _hessian_eigs(z, 4.0)
    out["ridge_s4"] = np.where(valid, -lo4, np.nan).astype(np.float32)
    del lo4
    mx = ndimage.maximum_filter(z, size=5, mode="nearest")
    mn = ndimage.minimum_filter(z, size=5, mode="nearest")
    out["relief5"] = np.where(valid, mx - mn, np.nan).astype(np.float32)
    del mx, mn
    out["coh3"] = np.where(valid, _coherence(z, 3.0), np.nan).astype(np.float32)
    del z

    vg = read_raw(tif, "tmi_vg")
    hg = read_raw(tif, "tmi_hg")
    out["tilt_mag"] = np.arctan2(vg, np.abs(hg)).astype(np.float32)
    out["mag_edge_log"] = np.log1p(np.abs(hg)).astype(np.float32)
    del vg, hg

    gvg = read_raw(tif, "iso_grav_anom_vg")
    ghg = read_raw(tif, "iso_grav_anom_hg_signed")
    out["tilt_grav"] = np.arctan2(gvg, np.abs(ghg)).astype(np.float32)
    del gvg, ghg

    base = read_raw(tif, "depth_to_base_surf")
    bvalid = np.isfinite(base)
    b = fill_nearest(base)
    by, bx = np.gradient(b)
    out["base_grad_log"] = np.where(bvalid, np.log1p(np.hypot(by, bx)), np.nan).astype(np.float32)
    del base, b, by, bx

    missing = [n for n in DERIVED_NAMES if n not in out]
    if missing:
        raise RuntimeError(f"derived channels missing: {missing}")
    return out


def build_stack(tif: Path, footprint: np.ndarray, cache: Path) -> np.ndarray:
    """Footprint-compacted ``float32`` matrix of all channels, memory-mapped.

    Shape ``(n_footprint_pixels, len(CHANNELS))``.  Built once and reused.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        mm = np.load(cache, mmap_mode="r")
        if mm.shape == (int(footprint.sum()), len(CHANNELS)):
            return mm
    n = int(footprint.sum())
    mm = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32,
                                   shape=(n, len(CHANNELS)))
    for j, name in enumerate(RAW_NAMES):
        mm[:, j] = read_raw(tif, name)[footprint]
        mm.flush()
    der = derived_channels(tif)
    for k, name in enumerate(DERIVED_NAMES):
        mm[:, len(RAW_NAMES) + k] = der[name][footprint]
        mm.flush()
    del mm, der
    return np.load(cache, mmap_mode="r")


def crest_mask(surface: np.ndarray, valid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Across-ridge non-maximum suppression of a probability surface (Steger 1998).

    A per-pixel gradient-boosted fault probability is *blobby*: its super-level
    sets are 2-D patches a few pixels across.  The 300 m DTI kernel rewards a
    1-pixel-wide trace, because a dot placed on the flank of a patch earns the
    same ``alpha`` penalty as one on the crest but less kernel credit from a
    linear truth trace.  This routine keeps only pixels that are a local
    maximum of ``surface`` along the eigenvector of the most-negative Hessian
    eigenvalue (the across-ridge normal), sampled bilinearly at +/- 1 px.

    Returns a boolean crest mask inside ``valid``.
    """
    s = ndimage.gaussian_filter(np.asarray(surface, np.float32), sigma, mode="nearest")
    a = ndimage.gaussian_filter(s, sigma, order=(2, 0), mode="nearest")   # d2/dy2
    c = ndimage.gaussian_filter(s, sigma, order=(0, 2), mode="nearest")   # d2/dx2
    b = ndimage.gaussian_filter(s, sigma, order=(1, 1), mode="nearest")   # d2/dydx
    root = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b * b, 0.0))
    lam_min = 0.5 * (a + c - root)
    vy = -b
    vx = a - lam_min
    nrm = np.hypot(vy, vx)
    flat = nrm < 1e-20
    vy = np.where(flat, 1.0, vy / np.where(flat, 1.0, nrm))
    vx = np.where(flat, 0.0, vx / np.where(flat, 1.0, nrm))
    h, w = s.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.ones(s.shape, bool)
    for sign in (1.0, -1.0):
        samp = ndimage.map_coordinates(
            s, [np.clip(yy + sign * vy, 0, h - 1), np.clip(xx + sign * vx, 0, w - 1)],
            order=1, mode="nearest").astype(np.float32)
        out &= s >= samp
    return out & np.asarray(valid, bool) & (lam_min < 0)


# --- optional owner-mirrored external blocks (integrity-pinned, NOT organizer-authenticated) ---
EXT_SPECS = [
    ("external/lidar_scarp_features_u8.tif",
     ["lid_ex_max", "lid_ex_mean", "lid_step_max", "lid_lapneg_max", "lid_lappos_max",
      "lid_downface_max", "lid_upface_max", "lid_cross_max", "lid_relief", "lid_coh100",
      "lid_strike", "lid_valid"]),
    ("external/geodawn_rad_u8.tif", ["rad_K", "rad_Th", "rad_U", "rad_TC"]),
    ("external/geodawn_extensions_u8.tif", ["rad_ThK", "rad_UK", "rad_UTh", "mag_up150"]),
]
EXT_NAMES = [n for _, names in EXT_SPECS for n in names]


def build_ext_stack(raw_dir: Path, footprint: np.ndarray, cache: Path) -> np.ndarray:
    """Footprint-compacted matrix of the 20 mirrored external channels.

    Sources (derived by earlier sessions from USGS 3DEP lidar and the USGS
    GeoDAWN airborne survey, then mirrored on GitHub and SHA-256 pinned in
    ``registry/data_manifest.json``): integrity-pinned, *not*
    organizer-authenticated.  ``uint8`` quantisation is kept as-is; where the
    lidar ``valid`` band is 0 the eleven lidar morphology channels are set to
    NaN so the tree model treats the 3DEP coverage gap as missing rather than
    as a measured zero.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = int(footprint.sum())
    if cache.exists():
        mm = np.load(cache, mmap_mode="r")
        if mm.shape == (n, len(EXT_NAMES)):
            return mm
    mm = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32,
                                   shape=(n, len(EXT_NAMES)))
    col = 0
    for rel, names in EXT_SPECS:
        with rasterio.open(Path(raw_dir) / rel) as src:
            arr = src.read().astype(np.float32)        # (bands, H, W)
        if rel.startswith("external/lidar"):
            valid = arr[names.index("lid_valid")] > 0
            for b in range(len(names) - 1):            # every band except 'valid'
                arr[b] = np.where(valid, arr[b], np.nan)
        for b in range(len(names)):
            mm[:, col] = arr[b][footprint]
            col += 1
        mm.flush()
        del arr
    del mm
    return np.load(cache, mmap_mode="r")
