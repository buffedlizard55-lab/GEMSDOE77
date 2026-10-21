"""Shared H77E helpers: grid/stack loading and footprint-compacted caches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import grid as G  # noqa: E402

RAW = ROOT / "data" / "raw"
CACHE = ROOT / ".cache" / "h77e"
TF = RAW / "training_features.tif"

# Band order verified 2026-10-21 from embedded GDAL descriptions of the
# SHA-pinned training_features.tif (sha256 4371c82e...): 1 mag_anom, 2 rtp,
# 3 tmi_hg, 4 geod_2ndinv, 5 iso_grav_anom_slope, 6 tc (=radiometric total
# count, NOT tilt), 7 geod_shearrate, 8 geod_dilaterate, 9 tmi_vg,
# 10 deq_n100a15, 11 iso_grav_anom_vg, 12 det_elev, 13 iso_grav_anom,
# 14 tmi, 15 depth_to_base_surf, 16 ieq_n100a15, 17 cond_surf,
# 18 iso_grav_anom_hg (signed dg/dx), 19 det_elev_slope.
BAND_NAMES = ["mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope",
              "tc_totalcount", "geod_shearrate", "geod_dilaterate", "tmi_vg",
              "deq_n100a15", "iso_grav_anom_vg", "det_elev", "iso_grav_anom",
              "tmi", "depth_to_base_surf", "ieq_n100a15", "cond_surf",
              "iso_grav_anom_hg_signed", "det_elev_slope"]
BAND_INDEX = {n: i + 1 for i, n in enumerate(BAND_NAMES)}  # rasterio 1-based


def load_grid():
    return G.load_grid(ROOT / "data" / "bridge")


def read_band(name: str) -> np.ndarray:
    with rasterio.open(TF) as src:
        a = src.read(BAND_INDEX[name]).astype(np.float32)
    a[a <= -1e30] = np.nan
    return a


def verify_band_order() -> list[str]:
    with rasterio.open(TF) as src:
        desc = [src.descriptions[i] or "" for i in range(src.count)]
    got = [d.split(" - ")[0].strip() for d in desc]
    return got


def build_std_fp(footprint: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Footprint-compacted raw 19-band matrix (NaN = missing), cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "std_fp.npy"
    names_path = CACHE / "std_names.json"
    if path.exists() and names_path.exists():
        names = json.loads(names_path.read_text())
        if names == BAND_NAMES:
            return np.load(path, mmap_mode="r"), names
    n_fp = int(footprint.sum())
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                   shape=(n_fp, len(BAND_NAMES)))
    for j, name in enumerate(BAND_NAMES):
        mm[:, j] = read_band(name)[footprint]
        mm.flush()
    del mm
    names_path.write_text(json.dumps(BAND_NAMES) + "\n")
    return np.load(path, mmap_mode="r"), list(BAND_NAMES)
