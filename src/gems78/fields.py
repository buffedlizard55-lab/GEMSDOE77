"""H78 candidate off-support fields (GEMSDOE77, lane H78).

Every field is computed from pinned inputs restored by ``scripts/lia_restore_data.py``
(``.cache/gems_data``).  None uses the catalogue labels, so none can leak the
hide-and-recover holdout by construction (the leakage canary in E2 checks this anyway).

H78-1  lidar scarp-morphology consensus.  ``lidar_scarp_features_u8.tif`` holds 12 uint8
       bands derived by the owner's CI from USGS 3DEP 1 m DEM tiles (706/716 tiles):
       ex_max, ex_mean, step_max, lapneg_max, lappos_max, downface_max, upface_max,
       cross_max, relief, coh100, strike, valid.  The consensus is the mean footprint rank
       of the seven morphology operators that respond to a scarp (exposure step, step
       height, negative and positive Laplacian = scarp crest/base, down- and up-facing
       faces, cross-profile).  A real scarp lights several operators at once; a single
       artefact (road cut, canal bank, tile seam) usually lights one or two.  Cells
       without lidar coverage (valid == 0) get 0.
H78-2  geothermal manifestation proximity: Euclidean distance (km) to the nearest Hot or
       Warm well/spring of GDR submission 1391 ("Well and Spring Temperature and
       Chemistry", DOI 10.15121/1881483, CC BY 4.0).  The file's dist_known_fault_px
       column is label-derived and is NOT read.
H78-3  region blocks: a 3 x 3 partition of the grid (labels 0..8).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from gems77lia import core

LIDAR = core.DATA / "external" / "lidar_scarp_features_u8.tif"
WELLS = core.DATA / "external" / "gdr_wellspring_in_footprint.csv"
LIDAR_OPS = ("ex_max", "step_max", "lapneg_max", "lappos_max", "downface_max",
             "upface_max", "cross_max")


def _rank_in(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average-free dense rank in [0, 1] of ``a`` over ``mask`` (0 elsewhere)."""
    out = np.zeros(a.shape, np.float32)
    v = a[mask].astype(np.float64)
    order = np.argsort(v, kind="stable")
    r = np.empty(v.size, np.float32)
    r[order] = np.linspace(0.0, 1.0, v.size, dtype=np.float32)
    out[mask] = r
    return out


def lidar_bands() -> tuple[dict, np.ndarray]:
    import rasterio
    with rasterio.open(LIDAR) as src:
        names = list(src.descriptions)
        arr = src.read()
        if (src.height, src.width) != (core.H, core.W):
            raise ValueError("lidar grid mismatch")
    bands = {n: arr[i] for i, n in enumerate(names)}
    valid = bands["valid"] > 0
    return bands, valid


def lidar_consensus() -> np.ndarray:
    g = core.grid()
    bands, valid = lidar_bands()
    m = g.footprint & valid
    acc = np.zeros((core.H, core.W), np.float32)
    for n in LIDAR_OPS:
        acc += _rank_in(bands[n], m)
    acc /= len(LIDAR_OPS)
    acc[~m] = 0.0
    return acc


def lidar_single(name: str) -> np.ndarray:
    g = core.grid()
    bands, valid = lidar_bands()
    m = g.footprint & valid
    out = _rank_in(bands[name], m)
    out[~m] = 0.0
    return out


def thermal_distance_km(classes=("Hot", "Warm")) -> np.ndarray:
    import pandas as pd
    d = pd.read_csv(WELLS)
    cls = d["thermalclass"].astype(str).str.strip()
    d = d[cls.isin(classes)]
    m = np.zeros((core.H, core.W), bool)
    r = d["row"].to_numpy(int)
    c = d["col"].to_numpy(int)
    ok = (r >= 0) & (r < core.H) & (c >= 0) & (c < core.W)
    m[r[ok], c[ok]] = True
    if not m.any():
        raise RuntimeError("no thermal points")
    return (ndimage.distance_transform_edt(~m) * 0.1).astype(np.float32)


def region_blocks(n: int = 3) -> np.ndarray:
    rr = np.minimum((np.arange(core.H) * n) // core.H, n - 1)
    cc = np.minimum((np.arange(core.W) * n) // core.W, n - 1)
    return (rr[:, None] * n + cc[None, :]).astype(np.int16)
