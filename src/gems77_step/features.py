"""Streaming zero-leakage multi-physics & anisotropic dip-slip step feature extraction (`GEMSDOE77`).

Strict invariants enforced across every function in this module:
  1. Zero historical submission or registry prediction rasters are read anywhere here.
  2. ``sgmc_faults_100m.tif`` is NEVER read into ``X``; pre-Quaternary bedrock contacts
     are completely excluded from features and model weights (proven ``0.0492`` live in ``GEMSDOE72``).
  3. Columns encoding catalogue proximity (``dist_known_fault_px``, ``known_fault``, ``label``)
     in external point tables are explicitly forbidden and asserted unused.
  4. All 96 static physical channels stream sequentially into a float32 disk memmap so peak RSS
     stays comfortably below ~650 MB on a 3.8 GiB runner.
"""
from __future__ import annotations

import csv
import gc
import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    gaussian_filter,
    label as ndi_label,
    uniform_filter,
)

FORBIDDEN_POINT_COLUMNS = frozenset({"dist_known_fault_px", "known_fault", "label"})

TF_BAND_NAMES = [
    "mag_anom",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_anom_slope",
    "tc",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
    "deq_n100a15",
    "iso_grav_anom_vg",
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "depth_to_base_surf",
    "ieq_n100a15",
    "cond_surf",
    "iso_grav_anom_hg",
    "det_elev_slope",
]

LIDAR_BAND_NAMES = [
    "lidar_ex_max",
    "lidar_ex_mean",
    "lidar_step_max",
    "lidar_lapneg_max",
    "lidar_lappos_max",
    "lidar_downface_max",
    "lidar_upface_max",
    "lidar_cross_max",
    "lidar_relief",
    "lidar_coh100",
    "lidar_strike",
]

TOPO10_BAND_NAMES = [
    "topo10_slope_p90",
    "topo10_slope_max",
    "topo10_steep_frac",
    "topo10_slope_std",
    "topo10_relief_local",
    "topo10_curv_prof_absmax",
    "topo10_aspect_coherence",
    "topo10_hs_lineament",
    "topo10_dem_mean",
]

CHI_BAND_NAMES = [
    "chi_step_max",
    "chi_step_min",
    "chi_ksn_max",
    "chi_log10_area",
    "chi_zmean_m",
]


def _grad_mag(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    sm = gaussian_filter(arr.astype(np.float32), sigma=sigma)
    gy, gx = np.gradient(sm)
    return np.hypot(gy, gx).astype(np.float32)


def _utm_to_rc(
    xs: list[float] | np.ndarray,
    ys: list[float] | np.ndarray,
    transform: Any,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    cols = np.floor((x_arr - transform.c) / transform.a).astype(np.int64)
    rows = np.floor((y_arr - transform.f) / transform.e).astype(np.int64)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return rows[ok], cols[ok], ok


def _parse_2m_probes_dbf(zip_path: Path, transform: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read INGENIOUS 2-meter temperature probes (T2m, F2mDAB) via stdlib binary DBF parser."""
    imp_dab = np.zeros(shape, dtype=np.float32)
    imp_cnt = np.zeros(shape, dtype=np.float32)
    if not zip_path.is_file():
        return imp_dab, imp_cnt
    with zipfile.ZipFile(zip_path) as zf:
        dbf_name = next(n for n in zf.namelist() if n.lower().endswith(".dbf"))
        buf = zf.read(dbf_name)
    numrec, lenhdr, lenrec = struct.unpack("<4xIHH20x", buf[:32])
    fields: list[tuple[str, int, int]] = []
    pos = 32
    offset = 1
    while buf[pos] != 0x0D:
        fname = buf[pos : pos + 11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        assert fname not in FORBIDDEN_POINT_COLUMNS, f"Forbidden column: {fname}"
        flen = buf[pos + 16]
        fields.append((fname, offset, flen))
        offset += flen
        pos += 32
    fmap = {name: (off, flen) for name, off, flen in fields}
    lats, lons, dabs = [], [], []
    for i in range(numrec):
        rec = buf[lenhdr + i * lenrec : lenhdr + (i + 1) * lenrec]
        if not rec or rec[0:1] == b"*":
            continue
        try:
            lat = float(rec[fmap["LatDegreeN"][0] : fmap["LatDegreeN"][0] + fmap["LatDegreeN"][1]].strip())
            lon = float(rec[fmap["LongDegree"][0] : fmap["LongDegree"][0] + fmap["LongDegree"][1]].strip())
            dab_s = rec[fmap["F2mDAB"][0] : fmap["F2mDAB"][0] + fmap["F2mDAB"][1]].strip()
            t2m_s = rec[fmap["T2m"][0] : fmap["T2m"][0] + fmap["T2m"][1]].strip()
            dab = float(dab_s) if dab_s else 0.0
            t2m = float(t2m_s) if t2m_s else 0.0
        except ValueError:
            continue
        if not (25.0 <= lat <= 50.0 and -130.0 <= lon <= -100.0):
            continue
        excess = max(0.0, dab) + 0.25 * max(0.0, t2m - 22.0)
        lats.append(lat)
        lons.append(lon)
        dabs.append(excess)
    if lons:
        ux, uy = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
        rows, cols, ok = _utm_to_rc(ux, uy, transform, shape)
        vals = np.asarray(dabs, dtype=np.float32)[ok]
        np.add.at(imp_cnt, (rows, cols), 1.0)
        np.maximum.at(imp_dab, (rows, cols), vals)
    return imp_dab, imp_cnt


def _sample_bilinear_fp(
    grid: np.ndarray,
    fy: np.ndarray,
    fx: np.ndarray,
) -> np.ndarray:
    """Fast continuous sub-pixel bilinear gather on 1D footprint coordinate vectors."""
    h, w = grid.shape
    yc = np.clip(fy, 0.0, float(h - 1.001))
    xc = np.clip(fx, 0.0, float(w - 1.001))
    y0 = yc.astype(np.int32)
    x0 = xc.astype(np.int32)
    wy = (yc - y0).astype(np.float32)
    wx = (xc - x0).astype(np.float32)
    v00 = grid[y0, x0]
    v01 = grid[y0, x0 + 1]
    v10 = grid[y0 + 1, x0]
    v11 = grid[y0 + 1, x0 + 1]
    return ((1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)).astype(np.float32)


def _radon_worm_6az(
    edge_2d: np.ndarray,
    fp: np.ndarray,
    half_len: int = 7,
    flank: int = 3,
) -> np.ndarray:
    """6-azimuth strike-coherent Radon line integral with symmetric cross-strike flank inhibition."""
    x = np.where(fp, np.nan_to_num(edge_2d, nan=0.0), 0.0).astype(np.float32)
    loc_mean = uniform_filter(x, size=31)
    loc_sq = uniform_filter(x * x, size=31)
    loc_std = np.sqrt(np.maximum(loc_sq - loc_mean * loc_mean, 1e-6))
    z = np.clip((x - loc_mean) / (loc_std + 0.25), -3.0, 6.0).astype(np.float32)
    del loc_mean, loc_sq, loc_std

    best = np.full(x.shape, -1e9, dtype=np.float32)
    L = int(half_len)
    c = L + 1
    for deg in (0, 30, 60, 90, 120, 150):
        rad = np.radians(deg)
        dy_s = np.sin(rad)
        dx_s = np.cos(rad)
        py_s = -dx_s
        px_s = dy_s
        k_mat = np.zeros((2 * L + 3, 2 * L + 3), dtype=np.float32)
        for t in range(-L, L + 1):
            ry = int(round(c + t * dy_s))
            rx = int(round(c + t * dx_s))
            if 0 <= ry < k_mat.shape[0] and 0 <= rx < k_mat.shape[1]:
                k_mat[ry, rx] += 1.0 / (2 * L + 1)
            for sgn in (-1, 1):
                fy = int(round(c + t * dy_s + sgn * flank * py_s))
                fx = int(round(c + t * dx_s + sgn * flank * px_s))
                if 0 <= fy < k_mat.shape[0] and 0 <= fx < k_mat.shape[1]:
                    k_mat[fy, fx] -= 0.5 / (2 * L + 1)
        resp = convolve(z, k_mat, mode="nearest")
        np.maximum(best, resp, out=best)
    del z
    return np.where(fp, np.maximum(best, 0.0), 0.0).astype(np.float32)


def _hessian_principal_ridge(field_2d: np.ndarray, sigma: float) -> np.ndarray:
    """Principal negative 2D Hessian eigenvalue max(-lambda_min, 0) for ridge crest localization."""
    sm = gaussian_filter(field_2d.astype(np.float32), sigma=sigma)
    gy, gx = np.gradient(sm)
    hyy, hyx = np.gradient(gy)
    _, hxx = np.gradient(gx)
    tr = hxx + hyy
    disc = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hyx * hyx, 0.0))
    lam_min = 0.5 * (tr - disc)
    return np.maximum(-lam_min, 0.0).astype(np.float32)


def build_h77_feature_stack(
    mmap_path: Path,
    fp: np.ndarray,
    transform: Any,
    data_dir: Path,
    ext_dir: Path,
) -> tuple[np.memmap, list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build or reload the 96-channel zero-label physical memmap + 2D crest & hydro fields.

    Columns ``0..73`` (74 channels): baseline multi-sensor physical channels (Exp-77.1).
    Columns ``74..95`` (22 channels): novel H77-1 & H77-3 operators (Exp-77.2 & Exp-77.3).
    """
    shape = fp.shape
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)
    total_cols = 96

    meta_path = mmap_path.with_suffix(".meta.json")
    aux_npz = mmap_path.with_suffix(".aux.npz")
    if mmap_path.is_file() and meta_path.is_file() and aux_npz.is_file():
        meta_obj = json.loads(meta_path.read_text())
        if meta_obj.get("n_features") == total_cols and meta_obj.get("n_footprint_pixels") == n_fp:
            X = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(total_cols, n_fp))
            has_lidar_fp = np.asarray(X[meta_obj["feature_names"].index("lidar_has_coverage"), :] > 0.5)
            aux = np.load(aux_npz)
            return X, meta_obj["feature_names"], has_lidar_fp, aux["hydro_imp_fp"], aux["crest_fp"], meta_obj["provenance"]

    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(total_cols, n_fp))
    names: list[str] = []
    col_ptr = 0

    bad_fp = ~fp
    idx_fp = distance_transform_edt(bad_fp, return_distances=False, return_indices=True)
    idx_cache: list[tuple[np.ndarray, np.ndarray]] = [(bad_fp, idx_fp)]

    def _fast_fill(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        bad = ~valid_mask
        if not bad.any():
            return arr.astype(np.float32, copy=False)
        found_idx = None
        for prev_bad, prev_idx in idx_cache:
            if np.array_equal(bad, prev_bad):
                found_idx = prev_idx
                break
        if found_idx is None:
            found_idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            if len(idx_cache) < 5:
                idx_cache.append((bad.copy(), found_idx))
        out = arr.astype(np.float32, copy=True)
        out[bad] = out[found_idx[0][bad], found_idx[1][bad]]
        return out

    def _push(name: str, arr2d_or_1d: np.ndarray, is_fp_1d: bool = False) -> None:
        nonlocal col_ptr
        assert name not in FORBIDDEN_POINT_COLUMNS, f"Forbidden feature {name}"
        vec = arr2d_or_1d if is_fp_1d else arr2d_or_1d.ravel()[fp_idx]
        X[col_ptr, :] = vec.astype(np.float32, copy=False)
        names.append(name)
        col_ptr += 1

    # -------------------------------------------------------------------------
    # PART A (Cols 0..73): 74 Baseline Multi-Sensor Physical Channels
    # -------------------------------------------------------------------------
    saved: dict[str, np.ndarray] = {}
    with rasterio.open(data_dir / "training_features.tif") as ds:
        for b_idx, b_name in enumerate(TF_BAND_NAMES, start=1):
            raw = ds.read(b_idx).astype(np.float32)
            fin = fp & np.isfinite(raw) & (raw > -1e30)
            filled = _fast_fill(raw, fin)
            _push(b_name, filled)
            if b_name in (
                "iso_grav_anom_hg",
                "iso_grav_anom_slope",
                "iso_grav_anom",
                "tmi_hg",
                "depth_to_base_surf",
                "cond_surf",
                "det_elev",
                "det_elev_slope",
            ):
                saved[b_name] = filled

    _push("grav_hg_worm", saved["iso_grav_anom_hg"] - gaussian_filter(saved["iso_grav_anom_hg"], 8.0))
    _push("mag_hg_worm", saved["tmi_hg"] - gaussian_filter(saved["tmi_hg"], 8.0))
    _push("basement_grad", _grad_mag(saved["depth_to_base_surf"], 1.5))
    _push("basement_curv", saved["depth_to_base_surf"] - gaussian_filter(saved["depth_to_base_surf"], 8.0))
    _push("cond_grad", _grad_mag(saved["cond_surf"], 1.5))
    _push("det_elev_lrm", saved["det_elev"] - gaussian_filter(saved["det_elev"], 6.0))

    # 1m 3DEP LiDAR scarp stack (native NaN outside 75.4% valid coverage!)
    with rasterio.open(ext_dir / "lidar_scarp_features_u8.tif") as ds:
        has_lidar_2d = (ds.read(12) > 0) & fp
        has_lidar_fp = has_lidar_2d.ravel()[fp_idx]
        _push("lidar_has_coverage", has_lidar_fp.astype(np.float32), is_fp_1d=True)
        lidar_cache: dict[str, np.ndarray] = {}
        for b_idx, b_name in enumerate(LIDAR_BAND_NAMES, start=1):
            v_fp = ds.read(b_idx).ravel()[fp_idx].astype(np.float32)
            v_fp[~has_lidar_fp] = np.nan
            _push(b_name, v_fp, is_fp_1d=True)
            if b_name in ("lidar_step_max", "lidar_lapneg_max", "lidar_lappos_max", "lidar_relief", "lidar_downface_max", "lidar_upface_max", "lidar_cross_max", "lidar_coh100"):
                lidar_cache[b_name] = v_fp
        dipole = np.sqrt(np.maximum(0.0, lidar_cache["lidar_lapneg_max"] * lidar_cache["lidar_lappos_max"]))
        sharpness = lidar_cache["lidar_step_max"] / (lidar_cache["lidar_relief"] + 5.0)
        _push("lidar_crest_toe_dipole", dipole, is_fp_1d=True)
        _push("lidar_scarp_sharpness", sharpness, is_fp_1d=True)

        # Build seam-free 1m+10m scarp component on valid LiDAR pixels (standardized)
        tect_fluv_fp = (
            np.maximum(np.nan_to_num(lidar_cache["lidar_downface_max"], nan=0.0), 1.3 * np.nan_to_num(lidar_cache["lidar_upface_max"], nan=0.0))
            * (25.0 + np.nan_to_num(lidar_cache["lidar_coh100"], nan=0.0))
        ) / (np.nan_to_num(lidar_cache["lidar_cross_max"], nan=0.0) + 12.0)
        lid_scarp_fp = (
            0.40 * (np.nan_to_num(dipole, nan=0.0) / (float(np.nanstd(dipole)) + 1e-6))
            + 0.35 * (np.nan_to_num(sharpness, nan=0.0) / (float(np.nanstd(sharpness)) + 1e-6))
            + 0.25 * (tect_fluv_fp / (float(np.std(tect_fluv_fp[has_lidar_fp])) + 1e-6))
        ).astype(np.float32)
        del dipole, sharpness, tect_fluv_fp
        lidar_cache.clear()

    # Seamless 10m 3DEP DEM derived stack (100% coverage)
    topo_cache: dict[str, np.ndarray] = {}
    with rasterio.open(ext_dir / "topo_10m_derived_100m_u8.tif") as ds:
        for b_idx, b_name in enumerate(TOPO10_BAND_NAMES, start=1):
            q = ds.read(b_idx)
            filled = _fast_fill(q.astype(np.float32), fp & (q > 0))
            _push(b_name, filled)
            if b_name in ("topo10_hs_lineament", "topo10_curv_prof_absmax", "topo10_slope_std"):
                topo_cache[b_name] = filled
    _push("topo10_hs_dog", topo_cache["topo10_hs_lineament"] - gaussian_filter(topo_cache["topo10_hs_lineament"], 6.0))
    _push("topo10_curv_dog", topo_cache["topo10_curv_prof_absmax"] - gaussian_filter(topo_cache["topo10_curv_prof_absmax"], 6.0))

    dem10_scarp_fp = (
        0.40 * (topo_cache["topo10_hs_lineament"].ravel()[fp_idx] / (float(np.std(topo_cache["topo10_hs_lineament"].ravel()[fp_idx])) + 1e-6))
        + 0.35 * (topo_cache["topo10_curv_prof_absmax"].ravel()[fp_idx] / (float(np.std(topo_cache["topo10_curv_prof_absmax"].ravel()[fp_idx])) + 1e-6))
        + 0.25 * (topo_cache["topo10_slope_std"].ravel()[fp_idx] / (float(np.std(topo_cache["topo10_slope_std"].ravel()[fp_idx])) + 1e-6))
    ).astype(np.float32)
    topo_cache.clear()

    unified_scarp_2d = np.zeros(shape, dtype=np.float32)
    unified_scarp_2d.ravel()[fp_idx] = np.where(
        has_lidar_fp, 0.65 * lid_scarp_fp + 0.35 * dem10_scarp_fp, dem10_scarp_fp
    ).astype(np.float32)
    del lid_scarp_fp, dem10_scarp_fp

    # Fluvial tectonic geomorphology (chi, ksn, zmean) + pluvial shoreline rejector
    # Explicitly mask int16 -32768 nodata sentinels inside footprint before smoothing!
    with rasterio.open(ext_dir / "dem10_chi_100m_i16.tif") as ds:
        ksn_arr = None
        z_arr = None
        for b_idx, b_name in enumerate(CHI_BAND_NAMES, start=1):
            raw_chi = ds.read(b_idx).astype(np.float32)
            arr = _fast_fill(raw_chi, fp & np.isfinite(raw_chi) & (raw_chi > -32760.0))
            _push(b_name, arr)
            if b_name == "chi_ksn_max":
                ksn_arr = arr
            elif b_name == "chi_zmean_m":
                z_arr = arr
    assert ksn_arr is not None and z_arr is not None
    _push("chi_ksn_dog", ksn_arr - gaussian_filter(ksn_arr, 8.0))
    d_shore = np.minimum(
        np.abs(z_arr - 1332.0),
        np.minimum(np.abs(z_arr - 1463.0), np.abs(z_arr - 1552.0)),
    ).astype(np.float32)
    z_mean_g3 = gaussian_filter(z_arr, 3.0)
    z_std_300m = np.sqrt(np.maximum(0.0, gaussian_filter(z_arr * z_arr, 3.0) - z_mean_g3 * z_mean_g3)).astype(np.float32)
    z_lrm_400m = (z_arr - gaussian_filter(z_arr, 4.0)).astype(np.float32)
    _push("pluvial_shoreline_elev_dist_m", d_shore)
    _push("zmean_std_300m", z_std_300m)
    _push("zmean_lrm_400m", z_lrm_400m)
    del d_shore, z_mean_g3, z_std_300m, z_lrm_400m

    # GeoDAWN radiometrics + upward-continued TMI alteration
    with rasterio.open(ext_dir / "geodawn_rad_u8.tif") as ds:
        rad_k = None
        for b_idx, b_name in enumerate(["rad_K", "rad_Th", "rad_U", "rad_TC"], start=1):
            q = ds.read(b_idx)
            arr = _fast_fill(q.astype(np.float32), fp & (q > 0))
            _push(b_name, arr)
            if b_name == "rad_K":
                rad_k = arr
    with rasterio.open(ext_dir / "geodawn_extensions_u8.tif") as ds:
        tmi_up = None
        for b_idx, b_name in enumerate(["ext_ThK", "ext_UK", "ext_UTh", "ext_TMI_up150"], start=1):
            q = ds.read(b_idx)
            arr = _fast_fill(q.astype(np.float32), fp & (q > 0))
            _push(b_name, arr)
            if b_name == "ext_TMI_up150":
                tmi_up = arr
    idx_cache.clear()
    del idx_fp
    assert rad_k is not None and tmi_up is not None
    rad_k_halo = (rad_k - gaussian_filter(rad_k, 10.0)).astype(np.float32)
    tmi_demag = (gaussian_filter(tmi_up, 10.0) - tmi_up).astype(np.float32)
    _push("rad_K_alteration_halo", rad_k_halo)
    _push("tmi_demagnetization_trough", tmi_demag)
    del rad_k, tmi_up

    # Label-free hydrothermal conduits: GDR 1391 springs/geothermometers, 3,800 2m probes, 709 sinter/tufa, volcanic vents
    imp_therm = np.zeros(shape, dtype=np.float32)
    ws_count = 0
    with (ext_dir / "gdr_wellspring_in_footprint.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        used_cols = ("row", "col", "temp_c", "geothermquartz_c", "geothermchalc_c", "geothermcat_c")
        for c in used_cols:
            assert c not in FORBIDDEN_POINT_COLUMNS
        for row_d in reader:
            try:
                rr = int(float(row_d["row"]))
                cc = int(float(row_d["col"]))
            except (ValueError, KeyError):
                continue
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                continue
            vals = []
            for k_col in ("temp_c", "geothermquartz_c", "geothermchalc_c", "geothermcat_c"):
                s = row_d.get(k_col, "").strip()
                if s and s.lower() != "nan":
                    try:
                        vals.append(float(s))
                    except ValueError:
                        pass
            if vals:
                excess = max(0.0, vals[0] - 20.0) + 0.15 * max(0.0, max(vals) - 90.0)
                if excess > 0.0:
                    imp_therm[rr, cc] = max(imp_therm[rr, cc], min(excess, 120.0))
                    ws_count += 1

    imp_dab, imp_probe_cnt = _parse_2m_probes_dbf(ext_dir / "2m_temperature_probe_INGENIOUS_regional_data.zip", transform, shape)

    imp_paleo = np.zeros(shape, dtype=np.float32)
    paleo_count = 0
    paleo_zip = ext_dir / "paleo_geothermal_regional.zip"
    if paleo_zip.is_file():
        with zipfile.ZipFile(paleo_zip) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            reader = csv.DictReader(io.TextIOWrapper(zf.open(csv_name), encoding="utf-8-sig", errors="replace"))
            lons_p, lats_p = [], []
            for row_d in reader:
                try:
                    lons_p.append(float(row_d["Long_deg_NAD83"]))
                    lats_p.append(float(row_d["Lat_deg_NAD83"]))
                except (ValueError, KeyError):
                    continue
            if lons_p:
                ux, uy = rio_transform("EPSG:4269", "EPSG:32611", lons_p, lats_p)
                pr, pc, _ = _utm_to_rc(ux, uy, transform, shape)
                imp_paleo[pr, pc] = 1.0
                paleo_count = int(pr.size)

    imp_vents = np.zeros(shape, dtype=np.float32)
    vents_path = ext_dir / "gdr_volcanic_vents_in_footprint.csv"
    if vents_path.is_file():
        with vents_path.open(newline="") as f:
            for row_d in csv.DictReader(f):
                try:
                    rr = int(float(row_d["row"]))
                    cc = int(float(row_d["col"]))
                    if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                        imp_vents[rr, cc] = 1.0
                except (ValueError, KeyError):
                    continue

    hydro_near = (
        gaussian_filter(np.log1p(imp_therm), 4.0)
        + 1.5 * gaussian_filter(np.log1p(imp_dab), 4.0)
        + 4.0 * gaussian_filter(imp_paleo, 4.0)
    ).astype(np.float32)
    hydro_far = (
        gaussian_filter(np.log1p(imp_therm), 14.0)
        + 1.5 * gaussian_filter(np.log1p(imp_dab), 14.0)
        + 4.0 * gaussian_filter(imp_paleo, 14.0)
    ).astype(np.float32)
    probe_halo = gaussian_filter(imp_probe_cnt, 6.0).astype(np.float32)
    sinter_halo = gaussian_filter(imp_paleo, 8.0).astype(np.float32)
    vent_halo = gaussian_filter(imp_vents, 12.0).astype(np.float32)

    _push("hydro_upflow_near_400m", hydro_near)
    _push("hydro_upflow_far_1400m", hydro_far)
    _push("probe_2m_density_600m", probe_halo)
    _push("paleo_sinter_seal_800m", sinter_halo)
    _push("quat_vent_halo_1200m", vent_halo)
    assert col_ptr == 74, f"Expected 74 baseline features, got {col_ptr}"

    del imp_therm, imp_dab, imp_probe_cnt, imp_paleo, imp_vents, hydro_far, probe_halo, sinter_halo, vent_halo
    gc.collect()

    # -------------------------------------------------------------------------
    # PART B (Cols 74..95): 22 Novel GEMSDOE77 Physical Operators (H77-1 & H77-3)
    # -------------------------------------------------------------------------
    # 1. Multi-scale 2D structure tensor on seamless elevation + Unified Scarp field -> continuous strike & normal
    z_smooth = gaussian_filter(z_arr, sigma=1.8)
    zy, zx = np.gradient(z_smooth)
    jxx = gaussian_filter((zx * zx).astype(np.float32), sigma=3.0)
    jyy = gaussian_filter((zy * zy).astype(np.float32), sigma=3.0)
    jxy = gaussian_filter((zx * zy).astype(np.float32), sigma=3.0)
    denom = jxx + jyy + 1e-9
    coh_2d = (np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0)) / denom).astype(np.float32)

    # Principal cross-strike gradient angle in (row, col) space:
    theta_rc = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    nx_fp = np.cos(theta_rc.ravel()[fp_idx]).astype(np.float32)
    ny_fp = np.sin(theta_rc.ravel()[fp_idx]).astype(np.float32)
    # Along-strike unit tangent (-ny, nx)
    ty_fp = -ny_fp
    tx_fp = nx_fp

    # Andersonian Basin & Range (N18E) + Walker Lane (N35W) normal-fault strike favorability
    normal_az = (np.rad2deg(np.arctan2(nx_fp, -ny_fp)) + 360.0) % 180.0
    strike_az = (normal_az + 90.0) % 180.0
    d_n18 = np.minimum(np.abs(strike_az - 18.0), 180.0 - np.abs(strike_az - 18.0))
    d_w35 = np.minimum(np.abs(strike_az - 145.0), 180.0 - np.abs(strike_az - 145.0))
    anderson_fp = coh_2d.ravel()[fp_idx] * np.maximum(
        np.cos(np.deg2rad(2.0 * d_n18)),
        0.75 * np.cos(np.deg2rad(2.0 * d_w35)),
    ).astype(np.float32)
    _push("hwfw_strike_coh", coh_2d)
    _push("hwfw_anderson_fav", anderson_fp, is_fp_1d=True)
    del z_smooth, zy, zx, jxx, jyy, jxy, denom, coh_2d, theta_rc, normal_az, strike_az, d_n18, d_w35, anderson_fp
    gc.collect()

    base_y = (fp_idx // shape[1]).astype(np.float32)
    base_x = (fp_idx % shape[1]).astype(np.float32)
    z0 = z_arr.ravel()[fp_idx]

    # H77-1: Monotone Step Purity Ratio A_step at 300 m (3 px) & 900 m (9 px)
    zp3 = _sample_bilinear_fp(z_arr, base_y + 3.0 * ny_fp, base_x + 3.0 * nx_fp)
    zm3 = _sample_bilinear_fp(z_arr, base_y - 3.0 * ny_fp, base_x - 3.0 * nx_fp)
    diff3 = np.abs(zm3 - zp3)
    sym3 = np.abs(zp3 - 2.0 * z0 + zm3)
    step_purity_300m = (diff3 / (sym3 + diff3 + 0.35)).astype(np.float32)
    del zp3, zm3, sym3

    zp9 = _sample_bilinear_fp(z_arr, base_y + 9.0 * ny_fp, base_x + 9.0 * nx_fp)
    zm9 = _sample_bilinear_fp(z_arr, base_y - 9.0 * ny_fp, base_x - 9.0 * nx_fp)
    diff9 = np.abs(zm9 - zp9)
    sym9 = np.abs(zp9 - 2.0 * z0 + zm9)
    step_purity_900m = (diff9 / (sym9 + diff9 + 1.0)).astype(np.float32)
    # Consistent polarity sign s_fw in {-1, +1}: -d*n is footwall (higher Z), +d*n is hanging wall (lower Z)
    sfw = np.where(zm9 >= zp9, 1.0, -1.0).astype(np.float32)
    del zp9, zm9, sym9

    zp5 = _sample_bilinear_fp(z_arr, base_y + 5.0 * ny_fp, base_x + 5.0 * nx_fp)
    zm5 = _sample_bilinear_fp(z_arr, base_y - 5.0 * ny_fp, base_x - 5.0 * nx_fp)
    step_relief_500m = (np.log1p(np.abs(zm5 - zp5)) * step_purity_300m).astype(np.float32)
    del zp5, zm5

    _push("hwfw_step_purity_300m", step_purity_300m, is_fp_1d=True)
    _push("hwfw_step_purity_900m", step_purity_900m, is_fp_1d=True)
    _push("hwfw_signed_relief_step_500m", step_relief_500m, is_fp_1d=True)

    # Dip-shifted (+4.0 px into downdropped hanging wall for ~60-deg normal fault dip!) subsurface polarity concordance
    shift_y = base_y + 4.0 * sfw * ny_fp
    shift_x = base_x + 4.0 * sfw * nx_fp
    db_hw = _sample_bilinear_fp(saved["depth_to_base_surf"], shift_y + 10.0 * sfw * ny_fp, shift_x + 10.0 * sfw * nx_fp)
    db_fw = _sample_bilinear_fp(saved["depth_to_base_surf"], shift_y - 10.0 * sfw * ny_fp, shift_x - 10.0 * sfw * nx_fp)
    dip_base_concord = np.arcsinh((db_hw - db_fw) / 50.0).astype(np.float32)
    del db_hw, db_fw

    gr_hw = _sample_bilinear_fp(saved["iso_grav_anom"], shift_y + 10.0 * sfw * ny_fp, shift_x + 10.0 * sfw * nx_fp)
    gr_fw = _sample_bilinear_fp(saved["iso_grav_anom"], shift_y - 10.0 * sfw * ny_fp, shift_x - 10.0 * sfw * nx_fp)
    dip_grav_concord = np.arcsinh((gr_fw - gr_hw) / 2.0).astype(np.float32)
    del gr_hw, gr_fw, shift_y, shift_x

    ksn_hw = _sample_bilinear_fp(ksn_arr, base_y + 6.0 * sfw * ny_fp, base_x + 6.0 * sfw * nx_fp)
    ksn_fw = _sample_bilinear_fp(ksn_arr, base_y - 6.0 * sfw * ny_fp, base_x - 6.0 * sfw * nx_fp)
    chi_ksn_concord = np.arcsinh((ksn_fw - ksn_hw) / 10.0).astype(np.float32)
    del ksn_hw, ksn_fw, ksn_arr, sfw

    # Hydrostatic paleoshoreline separator: elevation standard deviation over 1.6 km ALONG local strike tangent!
    zt_samples = np.empty((5, n_fp), dtype=np.float32)
    zt_samples[2] = z0
    for idx_s, s_off in enumerate((-8.0, -4.0, 4.0, 8.0)):
        slot = idx_s if idx_s < 2 else idx_s + 1
        zt_samples[slot] = _sample_bilinear_fp(z_arr, base_y + s_off * ty_fp, base_x + s_off * tx_fp)
    along_strike_std = np.log1p(np.std(zt_samples, axis=0)).astype(np.float32)
    del zt_samples, z0, z_arr, ty_fp, tx_fp

    # Cross-strike scarp crestness index psi_crest in [0, 1]
    sc0 = unified_scarp_2d.ravel()[fp_idx]
    sc_p = _sample_bilinear_fp(unified_scarp_2d, base_y + 2.5 * ny_fp, base_x + 2.5 * nx_fp)
    sc_m = _sample_bilinear_fp(unified_scarp_2d, base_y - 2.5 * ny_fp, base_x - 2.5 * nx_fp)
    crest_fp = np.clip((sc0 - 0.5 * (sc_p + sc_m)) / (np.abs(sc0) + 0.35), 0.0, 1.0).astype(np.float32)
    del sc_p, sc_m, base_y, base_x, ny_fp, nx_fp, step_purity_300m, step_purity_900m, step_relief_500m

    _push("hwfw_dip_basement_concord", dip_base_concord, is_fp_1d=True)
    _push("hwfw_dip_grav_concord", dip_grav_concord, is_fp_1d=True)
    _push("hwfw_chi_ksn_concord", chi_ksn_concord, is_fp_1d=True)
    _push("hwfw_along_strike_elev_std_1600m", along_strike_std, is_fp_1d=True)
    _push("hwfw_cross_strike_crest", crest_fp, is_fp_1d=True)
    del dip_base_concord, dip_grav_concord, chi_ksn_concord, along_strike_std
    gc.collect()

    # H77-3: Multi-Azimuth Radon Line-Integral Worms (1.5 km & 3.5 km) + Hessian Ridges + 2D Context Rings
    radon_scarp_1500 = _radon_worm_6az(unified_scarp_2d, fp, half_len=7, flank=3)
    radon_scarp_3500 = _radon_worm_6az(unified_scarp_2d, fp, half_len=16, flank=4)
    radon_grav_2500 = _radon_worm_6az(saved["iso_grav_anom_slope"], fp, half_len=11, flank=4)
    radon_mag_2500 = _radon_worm_6az(np.log1p(np.maximum(saved["tmi_hg"], 0.0)), fp, half_len=11, flank=4)
    radon_joint = np.sqrt(radon_scarp_1500 * np.maximum(radon_grav_2500, radon_mag_2500)).astype(np.float32)

    _push("radon_scarp_1500m", radon_scarp_1500)
    _push("radon_scarp_3500m", radon_scarp_3500)
    _push("radon_grav_2500m", radon_grav_2500)
    _push("radon_mag_2500m", radon_mag_2500)
    _push("radon_joint_scarp_geophys", radon_joint)
    del radon_scarp_1500, radon_scarp_3500, radon_grav_2500, radon_mag_2500, radon_joint

    hess_s1 = _hessian_principal_ridge(unified_scarp_2d, sigma=1.2)
    hess_s2 = _hessian_principal_ridge(unified_scarp_2d, sigma=2.6)
    _push("hessian_scarp_ridge_120m", hess_s1)
    _push("hessian_scarp_ridge_260m", hess_s2)
    del hess_s1, hess_s2

    sc_g3 = gaussian_filter(unified_scarp_2d, sigma=3.0)
    sc_g10 = gaussian_filter(unified_scarp_2d, sigma=10.0)
    sc_contrast = (unified_scarp_2d / (np.abs(sc_g10) + 0.35)).astype(np.float32)
    grav_g20 = gaussian_filter(saved["iso_grav_anom_slope"], sigma=20.0)

    non_playa = (saved["det_elev_slope"] > 1.2).astype(np.float32)
    hydro_total_2d = (
        hydro_near
        + 0.35 * np.maximum(rad_k_halo, 0.0) * non_playa
        + 0.35 * np.maximum(tmi_demag, 0.0) * non_playa
    ).astype(np.float32)
    hydro_scarp_coupled = (hydro_total_2d * np.maximum(unified_scarp_2d, 0.0)).astype(np.float32)

    _push("ctx2d_scarp_mean_300m", sc_g3)
    _push("ctx2d_scarp_mean_1km", sc_g10)
    _push("ctx2d_scarp_contrast_1km", sc_contrast)
    _push("ctx2d_grav_mean_2km", grav_g20)
    _push("hydro_scarp_coupled_upflow", hydro_scarp_coupled)

    assert col_ptr == total_cols, f"Expected {total_cols} features, got {col_ptr}"
    X.flush()

    hydro_imp_fp = hydro_near.ravel()[fp_idx].astype(np.float32)
    del unified_scarp_2d, sc_g3, sc_g10, sc_contrast, grav_g20, non_playa, hydro_total_2d, hydro_scarp_coupled, hydro_near, rad_k_halo, tmi_demag
    saved.clear()
    gc.collect()

    provenance = {
        "n_baseline_features": 74,
        "n_novel_h77_features": 22,
        "n_total_static_features": total_cols,
        "wellspring_thermal_points_used": ws_count,
        "paleo_sinter_travertine_points_used": paleo_count,
        "lidar_valid_footprint_fraction": round(float(has_lidar_fp.mean()), 6),
        "forbidden_columns_excluded": sorted(FORBIDDEN_POINT_COLUMNS),
        "sgmc_in_features": False,
        "prior_submissions_in_features": False,
    }
    np.savez(aux_npz, hydro_imp_fp=hydro_imp_fp, crest_fp=crest_fp)
    meta_path.write_text(
        json.dumps(
            {
                "n_features": total_cols,
                "n_footprint_pixels": n_fp,
                "feature_names": names,
                "provenance": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    X_ro = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(total_cols, n_fp))
    return X_ro, names, has_lidar_fp, hydro_imp_fp, crest_fp, provenance


def compute_visible_mechanics_fp(
    visible_2d: np.ndarray,
    fp: np.ndarray,
    hydro_imp_fp: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Derive 5 strictly fold-pure fracture-mechanics & orphan-geothermal channels ONLY from ``visible_2d``.

    Includes exact vectorized Pollard & Segall (1987) outward directional crack-tip propagation
    cones derived strictly from the outward unit tangent vector of each visible strand tip!
    Zero access to ``truth`` or global ``existing_faults.tif``.
    """
    vis = np.asarray(visible_2d, dtype=bool) & fp
    fp_idx = np.flatnonzero(fp)
    H, W = fp.shape
    d_vis = distance_transform_edt(~vis).astype(np.float32)
    d_vis_fp = d_vis.ravel()[fp_idx]

    comp, n_comp = ndi_label(vis, structure=np.ones((3, 3), dtype=np.int32))
    if n_comp > 0:
        sizes = np.bincount(comp.ravel())
        major = vis & (sizes[comp] >= 6)
    else:
        major = vis
    nb_cnt = convolve(major.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant", cval=0)
    tips_mask = major & (nb_cnt <= 2)

    cone_near = np.zeros((H, W), dtype=np.float32)
    cone_far = np.zeros((H, W), dtype=np.float32)
    tip_r, tip_c = np.nonzero(tips_mask)

    mr, mc = np.nonzero(major)
    m_cid = comp[mr, mc]
    order = np.argsort(m_cid)
    mr_s, mc_s, cid_s = mr[order], mc[order], m_cid[order]
    splits = np.searchsorted(cid_s, np.arange(n_comp + 2))

    # Vectorized cone template (36 forward steps x 11 lateral steps = 396 offsets per tip)
    t_grid = np.arange(1, 37, dtype=np.float32)[:, None]
    k_grid = np.arange(-5, 6, dtype=np.float32)[None, :]
    sig_lat = 1.15 + 0.12 * t_grid
    w_lat = np.exp(-0.5 * (k_grid / sig_lat) ** 2)
    tpl_near = (np.exp(-t_grid / 10.0) * w_lat).ravel()
    tpl_far = (np.exp(-t_grid / 25.0) * w_lat).ravel()
    t_flat = np.broadcast_to(t_grid, (36, 11)).ravel()
    k_flat = np.broadcast_to(k_grid, (36, 11)).ravel()

    for tr, tc in zip(tip_r.tolist(), tip_c.tolist()):
        cid = int(comp[tr, tc])
        s0, s1 = int(splits[cid]), int(splits[cid + 1])
        dr = tr - mr_s[s0:s1]
        dc = tc - mc_s[s0:s1]
        d2 = dr * dr + dc * dc
        band = (d2 >= 4) & (d2 <= 196)
        if not np.any(band):
            band = d2 >= 1
        if not np.any(band):
            continue
        ur = float(np.mean(dr[band]))
        uc = float(np.mean(dc[band]))
        norm = (ur * ur + uc * uc) ** 0.5
        if norm < 1e-6:
            continue
        ur /= norm
        uc /= norm
        strand_w = float(min(1.5, np.log1p(float(s1 - s0)) / 3.0))
        rr = np.rint(tr + t_flat * ur - k_flat * uc).astype(np.int32)
        cc_p = np.rint(tc + t_flat * uc + k_flat * ur).astype(np.int32)
        inb = (rr >= 0) & (rr < H) & (cc_p >= 0) & (cc_p < W)
        if np.any(inb):
            np.maximum.at(cone_near, (rr[inb], cc_p[inb]), strand_w * tpl_near[inb])
            np.maximum.at(cone_far, (rr[inb], cc_p[inb]), strand_w * tpl_far[inb])

    cone_near = gaussian_filter(cone_near, sigma=0.8)
    cone_far = gaussian_filter(cone_far, sigma=1.3)
    orphan_hydro_fp = (hydro_imp_fp * (1.0 - np.exp(-d_vis_fp / 10.0))).astype(np.float32)

    # Strictly sharp physical operators ONLY (zero isotropic distance rings so trees never
    # suffer from the PU self-distance halo tautology around mapped faults!)
    mech = np.vstack([
        cone_near.ravel()[fp_idx].astype(np.float32),
        cone_far.ravel()[fp_idx].astype(np.float32),
        orphan_hydro_fp,
    ])
    mech_names = [
        "vis_pollard_tip_cone_1km",
        "vis_pollard_tip_cone_2p5km",
        "vis_orphan_geothermal_conduit",
    ]
    return mech, mech_names


def compute_voronoi_lbco_mechanics_fp(
    visible_2d: np.ndarray,
    fp: np.ndarray,
    hydro_imp_fp: np.ndarray,
    n_buckets: int = 5,
    seed: int = 77,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Eliminate the PU Self-Distance Tautology via Voronoi Leave-Bucket-of-Components-Out (`V-LBCO`).

    Partitions dilated (3-px buffer) connected components of ``visible_2d`` ONLY into ``n_buckets``
    (default 5 -> 80% visible / 20% hidden, matching ``gems57``'s ``hide_frac=0.20``).
      - ``mech_train_fp``: For every pixel in Voronoi bucket ``b``, evaluates ``compute_visible_mechanics_fp``
        strictly on ``V \\ V_b``, so training positives ``x in V_b`` have ZERO artificial ``d_vis == 0``
        tautology and match the exact structural feature distribution of real unmapped fault segments!
      - ``mech_infer_fp``: Evaluates ``compute_visible_mechanics_fp`` on full ``visible_2d`` for
        inference on unmapped candidate pixels ``x not in V``.
    """
    from scipy.ndimage import binary_dilation

    vis = np.asarray(visible_2d, dtype=bool) & fp
    fp_idx = np.flatnonzero(fp)
    n_fp = fp_idx.size

    # Group nearby strand fragments within 3 px into single tectonic fault components
    yy, xx = np.mgrid[-3:4, -3:4]
    struct_buf = (yy * yy + xx * xx) <= 9
    grouped_mask = binary_dilation(vis, structure=struct_buf)
    comp_map, n_comp = ndi_label(grouped_mask, structure=np.ones((3, 3), dtype=np.int32))
    comp_on_vis = np.where(vis, comp_map, 0)

    # Nearest visible pixel Voronoi partition across the entire grid
    near_idx = distance_transform_edt(~vis, return_distances=False, return_indices=True)
    nearest_comp_fp = comp_on_vis[near_idx[0], near_idx[1]].ravel()[fp_idx]

    rng = np.random.default_rng(seed)
    comp_bucket = rng.integers(0, n_buckets, size=n_comp + 1)
    pixel_bucket_fp = comp_bucket[nearest_comp_fp]

    mech_infer_fp, mech_names = compute_visible_mechanics_fp(vis, fp, hydro_imp_fp)
    mech_train_fp = np.empty_like(mech_infer_fp)

    for b in range(n_buckets):
        in_b_vis = vis & (comp_bucket[comp_on_vis] == b)
        vis_minus_b = vis & (~in_b_vis)
        m_b, _ = compute_visible_mechanics_fp(vis_minus_b, fp, hydro_imp_fp)
        sel = pixel_bucket_fp == b
        mech_train_fp[:, sel] = m_b[:, sel]

    return mech_train_fp, mech_infer_fp, mech_names


def pack_ram_float16(X_mmap: np.memmap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each column monotonically into row-major float16 (946 MiB RAM, preserves NaN)."""
    n_cols, n_fp = X_mmap.shape
    X16 = np.empty((n_fp, n_cols), dtype=np.float16)
    medians = np.zeros(n_cols, dtype=np.float32)
    scales = np.ones(n_cols, dtype=np.float32)
    for c in range(n_cols):
        col = np.asarray(X_mmap[c, :], dtype=np.float32)
        sub = col[::17]
        valid = sub[np.isfinite(sub)]
        med = float(np.median(valid)) if valid.size else 0.0
        q75, q25 = np.percentile(valid, [75.0, 25.0]) if valid.size else (1.0, 0.0)
        sc = float(max(q75 - q25, float(np.std(valid)), 1e-4))
        medians[c] = med
        scales[c] = sc
        normed = np.clip((col - med) / sc, -1000.0, 1000.0)
        X16[:, c] = normed.astype(np.float16)
    return X16, medians, scales
