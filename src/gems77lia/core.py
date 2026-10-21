"""Core grid, catalogue and raster utilities for GEMSDOE77.

Everything here is derived from the restored, sha256-pinned competition inputs
(``.cache/gems_data``) - see ``scripts/restore_data.py``.

Official facts used (links in docs/sources.html):
  * grid: EPSG:32611, 100 m, 3730 rows x 3292 cols (problem page "Submission format";
    measured on labels.tif / sample_submission.tif);
  * scoring footprint = finite cells of sample_submission.tif = labels.tif != -1
    (5,167,373 cells, measured);
  * known USGS/INGENIOUS fault pixels (labels == 1, 60,988 cells) are masked from
    evaluation (DrivenData staff, forum thread 11516).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".cache" / "gems_data"
BRIDGE = ROOT / "data" / "bridge"
REG = ROOT / ".cache" / "registry"
WORK = ROOT / ".cache" / "work"
H, W = 3730, 3292
FEATURE_SENTINEL = -1e38


def resolve_core_file(name: str) -> Path:
    """Resolve a core grid/label raster from .cache/gems_data, data/raw, or data/bridge."""
    for base in (DATA, ROOT / "data" / "raw", BRIDGE):
        cand = base / name
        if cand.exists():
            return cand
    return DATA / name

BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15", "iso_grav_anom_vg",
    "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf", "ieq_n100a15", "cond_surf",
    "iso_grav_anom_hg", "det_elev_slope",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Grid:
    footprint: np.ndarray      # bool (H, W) - official scoring footprint
    catalogue: np.ndarray      # bool (H, W) - known USGS/INGENIOUS fault pixels
    transform: object
    crs: object


@lru_cache(maxsize=1)
def grid() -> Grid:
    import rasterio
    with rasterio.open(resolve_core_file("labels.tif")) as src:
        lab = src.read(1)
        transform, crs = src.transform, src.crs
    with rasterio.open(resolve_core_file("sample_submission.tif")) as src:
        ss = src.read(1)
    fp = lab != -1
    if not np.array_equal(fp, np.isfinite(ss)):
        raise RuntimeError("labels footprint != sample_submission finite mask")
    return Grid(footprint=fp, catalogue=(lab == 1), transform=transform, crs=crs)


@lru_cache(maxsize=1)
def catalogue_distance_px() -> np.ndarray:
    """Euclidean distance (pixels) from every cell to the nearest catalogue pixel."""
    g = grid()
    return ndimage.distance_transform_edt(~g.catalogue).astype(np.float32)


def read_band(i: int) -> np.ndarray:
    """Read one training-feature band (1-based) as float32 with NaN for nodata."""
    import rasterio
    with rasterio.open(DATA / "training_features.tif") as src:
        a = src.read(i).astype(np.float32)
    a[~np.isfinite(a) | (a < FEATURE_SENTINEL)] = np.nan
    return a


def read_prediction(path: str | Path) -> np.ndarray:
    """Read a submission-like raster as float32 mass in [0, 1]; NaN -> 0, outside footprint -> 0."""
    import rasterio
    with rasterio.open(path) as src:
        if (src.height, src.width) != (H, W):
            raise ValueError(f"{path}: shape {(src.height, src.width)} != {(H, W)}")
        a = src.read(1).astype(np.float32)
    a[~np.isfinite(a)] = 0.0
    a = np.clip(a, 0.0, 1.0)
    a[~grid().footprint] = 0.0
    return a


# ---------------------------------------------------------------- metric (fast, exact)
_OFFSETS = None


def _offsets(r_px: int = 3):
    global _OFFSETS
    if _OFFSETS is None:
        out = []
        for dy in range(-r_px, r_px + 1):
            for dx in range(-r_px, r_px + 1):
                k = 1.0 - np.hypot(dy, dx) / r_px
                if k > 0:
                    out.append((dy, dx, float(k)))
        _OFFSETS = out
    return _OFFSETS


def cover_field(p: np.ndarray) -> np.ndarray:
    """M(y) = max_x p(x) k(|x - y|): best kernel-weighted cover of each cell (R = 3 px)."""
    h, w = p.shape
    pad = np.pad(p, 3)
    m = np.zeros_like(p, dtype=np.float32)
    for dy, dx, k in _offsets():
        np.maximum(m, pad[3 + dy:3 + dy + h, 3 + dx:3 + dx + w] * np.float32(k), out=m)
    return m


def dti_terms(p: np.ndarray, truth: np.ndarray, score_mask: np.ndarray | None = None,
              kmax_truth: np.ndarray | None = None) -> tuple[float, float, float]:
    """Official TP_w, FP_w, FN_w with optional scoring mask (cells outside are ignored).

    ``score_mask`` restricts which predicted cells are charged FP (and which truth
    cells exist).  ``kmax_truth`` (= max_g k(d(x, g))) may be passed to avoid
    recomputing the truth EDT.  Exactly equals the template ``gems.metric`` when
    ``score_mask`` is None (tested in tests/test_core.py).
    """
    t = truth.astype(bool)
    if score_mask is not None:
        t = t & score_mask
        p = np.where(score_mask, p, 0.0).astype(np.float32)
    n_g = int(t.sum())
    m = cover_field(p)
    tp = float(m[t].sum())
    fn = float(n_g - tp)
    if kmax_truth is None:
        if n_g == 0:
            kmax_truth = np.zeros_like(p)
        else:
            d = ndimage.distance_transform_edt(~t)
            kmax_truth = np.clip(1.0 - d / 3.0, 0.0, None).astype(np.float32)
    fp = float(np.sum(p * (1.0 - kmax_truth)))
    return tp, fp, fn


def dti(tp: float, fp: float, fn: float, alpha: float = 0.2, beta: float = 0.8) -> float:
    return tp / (tp + alpha * fp + beta * fn + 1e-9)
