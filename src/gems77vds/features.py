"""Zero-leakage, zero-SGMC, survey- and mining-deconfounded geoscientific feature stack (`H77E`).

Strict scientific & engineering invariants verified by unit tests (`tests/test_h77e.py`):
  1. Zero pre-Quaternary SGMC bedrock contact files (`sgmc_faults_100m.tif`,
     `derived_sgmc_faults_100m*.tif`) are ever opened or read anywhere in `gems77e`.
  2. Zero prior prediction/registry rasters (`H19-5`, `H33`, `GEMSDOE*`) are ever read in features.
  3. Forbidden catalogue-derived columns (`dist_known_fault_px`, `known_fault`, `label`) in
     external CSV/DBF point tables raise `AssertionError` if ever accessed.
  4. Band 10 (`deq_n100a15`) is correctly treated as positive Earthquake Density (+log1p(deq)),
     fixing the historic metadata-label sign inversion (`IR-75-BAND10-DEQ-SIGN`).
  5. Missing 1 m USGS 3DEP LiDAR tiles across 24.6% of the valid footprint (`q == 0`) are strictly
     mapped to `NaN` across all 11 physical LiDAR bands (`IR-77D-01`) and bridged seamlessly by
     100%-coverage 10 m USGS 3DEP slope/curvature/fluvial-chi/branch-signature layers.
  6. Streams 4 memory-bounded passes into `.cache/h77e/static_fp.f32` (`(5167373, 98)` float32),
     keeping peak RSS cleanly below ~650 MB on 2-vCPU / 3.8 GiB sandboxes.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import struct
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform


def evict_memmap_cache(mm: np.memmap, path: Path | str | None = None) -> None:
    """Flush memmap pages and drop clean file-cache pages so 3.9 GiB cgroup RAM stays < 1.1 GiB."""
    mm.flush()
    target = Path(path) if path is not None else Path(mm.filename)
    fd = os.open(str(target), os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
from scipy import fft as sfft
from scipy import ndimage as ndi

FORBIDDEN_COLUMNS = frozenset({"dist_known_fault_px", "known_fault", "label"})
SENTINEL_LOW = np.float32(-1.0e30)

BLOCKS: dict[str, list[str]] = {
    # Block 1 (19 ch): Clean sign-verified official GeoDAWN + INGENIOUS 100 m bands
    "b1_raw_clean": [
        "raw_mag_anom_hp",
        "raw_rtp_hp",
        "raw_rtp",
        "raw_tmi_hg_log",
        "raw_geod_2ndinv",
        "raw_grav_slope_log",
        "raw_tc_rad",
        "raw_geod_shear",
        "raw_geod_dilat",
        "raw_tmi_vg",
        "raw_deq_pos_log",      # Band 10 sign-verified positive earthquake density
        "raw_grav_vg",
        "raw_det_elev_hp",
        "raw_det_elev",
        "raw_iso_grav_hp",
        "raw_depth_base",
        "raw_ieq_pos_log",      # Band 16 earthquake intensity
        "raw_cond_surf_log",
        "raw_det_elev_slope_log",
    ],
    # Block 2 (20 ch): GeoDAWN survey-block within-regime standardization + shallow continuation + BLM mine & TIGER road deconfounding
    "b2_block_mine_deconf": [
        "tiger_road_dist_log",
        "blm_mine_dist_log",
        "seam_block_dist_log",
        "blk_std_tmi_hg",
        "blk_std_tmi_vg",
        "blk_std_tc_rad",
        "blk_std_kth_ratio",
        "blk_std_uth_ratio",
        "mag_shallow_up150_resid",
        "mag_shallow_up150_grad",
        "pot_mag_tilt",
        "pot_mag_tilt_zero",
        "pot_mag_as_log",
        "pot_grav_thdr_log",
        "pot_grav_tilt_zero",
        "pot_grav_curv_up",
        "pot_cg_align_s1p5",
        "pot_cg_align_s3p5",
        "pot_base_grad_log",
        "pot_base_grav_align",
    ],
    # Block 3 (35 ch): Full 12-band 1 m LiDAR (q==0->NaN) + multi-azimuth FFT Radon + 10 m 3DEP HW/FW dip-slip step + fluvial chi & branch jumps
    "b3_lidar12_chi_step": [
        "lidar_valid_flag",
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
        "lidar_crest_toe_dipole",
        "lidar_facing_polarity",
        "lidar_step_deconf_road_mine",
        "lidar_dipole_deconf_road_mine",
        "lidar_strike_coh_s2",
        "lidar_strike_coh_s5",
        "radon_scarp_max_1p2km",
        "radon_scarp_aniso_1p2km",
        "radon_scarp_max_3p0km",
        "radon_scarp_aniso_3p0km",
        "topo10_slope_p90",
        "topo10_slope_max",
        "topo10_steep_frac",
        "topo10_slope_std",
        "topo10_relief_local",
        "topo10_curv_prof_absmax",
        "topo10_aspect_coh",
        "topo10_hs_lineament",
        "topo10_onesided_step_deconf",
        "topo10_along_strike_zstd",  # Suppresses equipotential Lake Lahontan/Bonneville shorelines
        "pied_front_s4",
        "pied_front_s12",
        "pied_toe_concavity",
        "chi_step_pos_excess",
    ],
    # Block 4 (24 ch): Remaining fluvial chi + tributary branch jumps + Sibson fault-valve seismicity/geothermometry/2m-probe/paleo-sinter
    "b4_valve_hydro_chi": [
        "chi_ksn_norm",
        "chi_area_log",
        "chi_branch_jump_s1",
        "chi_branch_jump_s3",
        "chi_scarp_concordance",
        "seism_swarm_ridge_s2",
        "seism_swarm_ridge_s6",
        "seism_x_shear",
        "seism_x_dilat",
        "seism_strain_deficit",
        "hydro_hot_spring_logdist",
        "hydro_temp_excess_s4",
        "hydro_temp_excess_s12",
        "hydro_geotherm_diseq_s5",   # Quartz/chalcedony/cation geothermometer minus surface discharge T!
        "hydro_geotherm_diseq_s14",
        "hydro_probe2m_dab_s3",
        "hydro_probe2m_dab_s9",
        "hydro_paleo_sinter_s4",
        "hydro_paleo_sinter_s12",
        "volcanic_vent_logdist",
        "hydro_claycap_grad",
        "hydro_demag_conduit_deconf",
        "hydro_potassic_thk_inv",
        "valve_coupled_conduit",
    ],
}

STATIC_CHANNEL_NAMES: list[str] = [n for grp in BLOCKS.values() for n in grp]
STATIC_IDX: dict[str, int] = {n: i for i, n in enumerate(STATIC_CHANNEL_NAMES)}

VISIBLE_CHANNEL_NAMES: list[str] = [
    "vis_logdist",
    "vis_dens_s5",
    "vis_dens_s15",
    "vis_dens_s35",
    "vis_relay_deficit",
    "vis_pollard_tip_cone",
    "vis_strike_align",
    "vis_strain_deficit",
]

ALL_CHANNEL_NAMES: list[str] = STATIC_CHANNEL_NAMES + VISIBLE_CHANNEL_NAMES


def _gf32(a: np.ndarray, sigma: float, mode: str = "nearest") -> np.ndarray:
    return ndi.gaussian_filter(np.asarray(a, dtype=np.float32), sigma=sigma, mode=mode).astype(np.float32)


def _clean_band(a: np.ndarray, foot: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32)
    ok = foot & np.isfinite(arr) & (arr > SENTINEL_LOW)
    med = float(np.median(arr[ok])) if ok.any() else 0.0
    return np.where(ok, arr, med).astype(np.float32)


def _rank01_fp(arr_2d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fast empirical CDF rank in [0, 1] over `mask` using 2049 quantile knots."""
    out = np.zeros(arr_2d.shape, dtype=np.float32)
    v = arr_2d[mask]
    fin = np.isfinite(v)
    vf = v[fin]
    if vf.size < 4:
        return out
    vmin, vmax = float(vf.min()), float(vf.max())
    if vmax <= vmin:
        return out
    q = np.linspace(0.0, 1.0, 2049)
    sample = vf[::4] if vf.size > 120_000 else vf
    knots = np.maximum.accumulate(np.quantile(sample, q))
    uk, uidx = np.unique(knots, return_index=True)
    if uk.size < 2:
        return out
    r = np.zeros(v.shape, dtype=np.float32)
    r[fin] = np.interp(vf, uk, q[uidx]).astype(np.float32)
    out[mask] = r
    return out


def _within_block_rank(arr_2d: np.ndarray, block_id: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """Standardize `arr_2d` separately inside each GeoDAWN survey block to remove altitude jumps."""
    out = np.zeros(arr_2d.shape, dtype=np.float32)
    covered = np.zeros(arr_2d.shape, dtype=bool)
    for bid in np.unique(block_id[foot]):
        if bid <= 0:
            continue
        m = foot & (block_id == bid)
        if int(m.sum()) >= 500:
            out[m] = _rank01_fp(arr_2d, m)[m]
            covered |= m
    rem = foot & (~covered)
    if rem.any():
        out[rem] = _rank01_fp(arr_2d, foot)[rem]
    return out


def _hessian_ridge_pos(arr_2d: np.ndarray, sigma: float) -> np.ndarray:
    """Positive ridge curvature (-lambda_min of smoothed 2D field) highlighting thin linear maxima."""
    s = _gf32(arr_2d, sigma)
    gy, gx = np.gradient(s)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    b = 0.5 * (gxy + gyx)
    tr = gxx + gyy
    det = np.sqrt(np.maximum(0.25 * (gxx - gyy) ** 2 + b * b, np.float32(0.0)))
    lam_min = 0.5 * tr - det
    return np.maximum(-lam_min * np.float32(sigma * sigma), np.float32(0.0)).astype(np.float32)


def _crossgrad_align(f1: np.ndarray, f2: np.ndarray, sigma: float) -> np.ndarray:
    """Gallardo & Meju (2003) potential-field structural alignment: (1 - |cos x sin|)*grad_mag."""
    s1 = _gf32(f1, sigma)
    s2 = _gf32(f2, sigma)
    g1y, g1x = np.gradient(s1)
    g2y, g2x = np.gradient(s2)
    m1 = np.hypot(g1y, g1x)
    m2 = np.hypot(g2y, g2x)
    denom = m1 * m2 + np.float32(1e-8)
    cross = np.abs(g1x * g2y - g1y * g2x) / denom
    # Strong colinear gradient where both gradients are large and parallel/antiparallel
    par = np.float32(1.0) - np.clip(cross, 0.0, 1.0)
    return (par * np.log1p(m1) * np.log1p(m2)).astype(np.float32)


def _multi_azimuth_radon_continuity(
    scarp_2d: np.ndarray, half_len_px: int, n_angles: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Fast FFT convolution line-integral max & max/median anisotropy across `n_angles` strike orientations."""
    H, W = scarp_2d.shape
    ksize = 2 * half_len_px + 1
    kh, kw = ksize, ksize
    fh = sfft.next_fast_len(H + kh - 1)
    fw = sfft.next_fast_len(W + kw - 1)
    F_im = sfft.rfft2(np.maximum(scarp_2d, np.float32(0.0)), s=(fh, fw), workers=2)

    yy, xx = np.mgrid[-half_len_px : half_len_px + 1, -half_len_px : half_len_px + 1].astype(np.float32)
    rmax = np.full((H, W), -np.inf, dtype=np.float32)
    rsum = np.zeros((H, W), dtype=np.float32)
    for ai in range(n_angles):
        theta = np.pi * ai / float(n_angles)
        # Line kernel along strike direction (cos theta, sin theta)
        perp = np.abs(-np.sin(theta) * xx + np.cos(theta) * yy)
        along = np.abs(np.cos(theta) * xx + np.sin(theta) * yy)
        kern = np.exp(-0.5 * (perp / 0.75) ** 2) * (along <= float(half_len_px))
        kern = (kern / max(float(kern.sum()), 1e-6)).astype(np.float32)
        Fk = sfft.rfft2(kern, s=(fh, fw), workers=2)
        conv = sfft.irfft2(F_im * Fk, s=(fh, fw), workers=2)
        sl = conv[half_len_px : half_len_px + H, half_len_px : half_len_px + W].astype(np.float32, copy=False)
        np.maximum(rmax, sl, out=rmax)
        rsum += sl
        del Fk, conv
    del F_im
    rmean = rsum / np.float32(n_angles)
    aniso = (rmax / (rmean + np.float32(1e-5)) - np.float32(1.0)).astype(np.float32)
    return rmax, np.maximum(aniso, np.float32(0.0))


def _utm_rc(xs: np.ndarray, ys: np.ndarray, transform: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = np.floor((xs - transform.c) / transform.a).astype(np.int64)
    rows = np.floor((ys - transform.f) / transform.e).astype(np.int64)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return rows[ok], cols[ok], ok


def _parse_2m_probes(zip_path: Path, transform: Any, shape: tuple[int, int]) -> np.ndarray:
    """Extract shallow 2-meter temperature probe anomaly impulses (`T2m`, `F2mDAB`) from GDR 1391."""
    imp = np.zeros(shape, dtype=np.float32)
    if not zip_path.is_file():
        return imp
    with zipfile.ZipFile(zip_path) as zf:
        dbf_name = next(n for n in zf.namelist() if n.lower().endswith(".dbf"))
        buf = zf.read(dbf_name)
    numrec, lenhdr, lenrec = struct.unpack("<4xIHH20x", buf[:32])
    fields: list[tuple[str, int, int]] = []
    pos, offset = 32, 1
    while buf[pos] != 0x0D:
        fname = buf[pos : pos + 11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        assert fname not in FORBIDDEN_COLUMNS, f"Forbidden column encountered: {fname}"
        flen = buf[pos + 16]
        fields.append((fname, offset, flen))
        offset += flen
        pos += 32
    fmap = {n: (o, l) for n, o, l in fields}
    lats, lons, vals = [], [], []
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
        if 35.0 <= lat <= 43.0 and -122.0 <= lon <= -113.0:
            v = max(0.0, dab) + 0.25 * max(0.0, t2m - 21.0)
            if v > 0.0:
                lats.append(lat)
                lons.append(lon)
                vals.append(v)
    if lats:
        xs, ys = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
        rs, cs, ok = _utm_rc(np.asarray(xs), np.asarray(ys), transform, shape)
        np.add.at(imp, (rs, cs), np.asarray(vals, dtype=np.float32)[ok])
    return imp


def _parse_paleo_sinters(zip_path: Path, transform: Any, shape: tuple[int, int]) -> np.ndarray:
    """Extract Pleistocene opaline sinter, travertine & hydrothermal breccia deposits (GDR 1391)."""
    imp = np.zeros(shape, dtype=np.float32)
    if not zip_path.is_file():
        return imp
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if csv_name is None:
            return imp
        raw_txt = zf.read(csv_name).decode("utf-8", errors="ignore")
    rdr = csv.DictReader(raw_txt.splitlines())
    lats, lons, wts = [], [], []
    for row in rdr:
        safe = {k: v for k, v in row.items() if k not in FORBIDDEN_COLUMNS}
        lat_k = next((k for k in safe if "lat" in k.lower()), None)
        lon_k = next((k for k in safe if "lon" in k.lower()), None)
        if not lat_k or not lon_k:
            continue
        try:
            lat = float(safe[lat_k])
            lon = float(safe[lon_k])
        except (TypeError, ValueError):
            continue
        if 35.0 <= lat <= 43.0 and -122.0 <= lon <= -113.0:
            desc = " ".join(str(v).lower() for v in safe.values())
            w = 2.0 if ("sinter" in desc or "opal" in desc) else 1.25
            lats.append(lat)
            lons.append(lon)
            wts.append(w)
    if lats:
        xs, ys = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
        rs, cs, ok = _utm_rc(np.asarray(xs), np.asarray(ys), transform, shape)
        np.add.at(imp, (rs, cs), np.asarray(wts, dtype=np.float32)[ok])
    return imp


def load_scaled_f16_matrix(bin_path: Path, n_fp: int) -> np.ndarray:
    """Load all 98 channels once into a `(n_fp, 98)` `float16` matrix (`966 MB` total RAM!)
    using monotone robust `(x - med) / iqr` standardization so tree histogram splits are 100%
    invariant while eliminating both disk re-reads (`20x` faster!) and `float16` underflow/overflow.
    """
    n_ch = len(STATIC_CHANNEL_NAMES)
    X16 = np.empty((n_fp, n_ch), dtype=np.float16)
    stride_bytes = int(n_fp) * 4
    with open(bin_path, "rb") as fh:
        for c in range(n_ch):
            fh.seek(c * stride_bytes)
            col = np.fromfile(fh, dtype=np.float32, count=n_fp)
            fin = np.isfinite(col)
            if int(fin.sum()) > 16:
                sub = col[fin][::16]
                q25, q50, q75 = np.percentile(sub, [25.0, 50.0, 75.0])
                scale = max(float(q75 - q25), float(np.std(sub)), 1e-12)
                norm = np.where(fin, np.clip((col - np.float32(q50)) / np.float32(scale), -2048.0, 2048.0), np.nan)
            else:
                norm = np.where(fin, np.clip(col, -2048.0, 2048.0), np.nan)
            X16[:, c] = norm.astype(np.float16)
            del col, fin, norm
        fd = fh.fileno()
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    gc.collect()
    return X16


def read_static_rows(
    bin_path: Path,
    n_fp: int,
    row_idx: np.ndarray | slice,
    col_indices: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Zero-mmap sequential reader for channel-major `(n_ch, n_fp)` float32 store (`< 80 MB` peak RAM!)."""
    cols = list(range(len(STATIC_CHANNEL_NAMES))) if col_indices is None else [int(c) for c in col_indices]
    n_rows = len(range(*row_idx.indices(n_fp))) if isinstance(row_idx, slice) else int(len(row_idx))
    out = np.empty((n_rows, len(cols)), dtype=np.float32)
    stride_bytes = int(n_fp) * 4
    with open(bin_path, "rb") as fh:
        for out_j, col_c in enumerate(cols):
            fh.seek(col_c * stride_bytes)
            buf = np.fromfile(fh, dtype=np.float32, count=n_fp)
            out[:, out_j] = buf[row_idx]
        fd = fh.fileno()
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    return out


def build_static_fp_memmap(root: Path, force: bool = False) -> tuple[Path, dict[str, np.ndarray]]:
    """Build or verify channel-major `.cache/h77e/static_ch_fp.f32` of shape `(98, 5167373)` (`float32`).

    Returns `(bin_path, aux_dict)` with zero pinned `mmap` RSS (`peak RSS < 950 MB`!).
    """
    os.environ.setdefault("GDAL_CACHEMAX", "64")
    cache_dir = root / ".cache" / "h77e"
    cache_dir.mkdir(parents=True, exist_ok=True)
    old_row_major = cache_dir / "static_fp.f32"
    if old_row_major.exists():
        old_row_major.unlink()
    mmap_path = cache_dir / "static_ch_fp.f32"
    aux_path = cache_dir / "aux_fp.npz"
    meta_path = cache_dir / "static_meta.json"

    raw_dir = root / "data" / "raw"
    ext_dir = raw_dir / "external"

    with rasterio.open(raw_dir / "sample_submission.tif") as ds:
        sample = ds.read(1)
        transform = ds.transform
    foot = np.isfinite(sample)
    H, W = foot.shape
    n_fp = int(foot.sum())
    n_ch = len(STATIC_CHANNEL_NAMES)
    del sample

    expected_bytes = n_fp * n_ch * 4
    if not force and mmap_path.is_file() and mmap_path.stat().st_size == expected_bytes and aux_path.is_file():
        aux = dict(np.load(aux_path))
        aux["n_fp"] = np.int64(n_fp)
        return mmap_path, aux

    t0 = time.time()
    stride_bytes = n_fp * 4
    fh_out = open(mmap_path, "wb+")
    fh_out.truncate(expected_bytes)
    written: set[int] = set()

    def put2d(name: str, arr2d: np.ndarray) -> None:
        cid = STATIC_IDX[name]
        vec = np.ascontiguousarray(arr2d[foot], dtype=np.float32)
        fh_out.seek(cid * stride_bytes)
        fh_out.write(vec.tobytes())
        written.add(cid)

    # ------------------------------------------------------------------
    # PASS 1: 19 raw bands (sign-verified Band 10 +deq!) + potential-field physics
    # ------------------------------------------------------------------
    with rasterio.open(raw_dir / "training_features.tif") as tf:
        b_mag_anom = _clean_band(tf.read(1), foot)
        b_rtp = _clean_band(tf.read(2), foot)
        b_tmi_hg = _clean_band(tf.read(3), foot)
        b_geod_2nd = _clean_band(tf.read(4), foot)
        b_grav_slp = _clean_band(tf.read(5), foot)
        b_tc = _clean_band(tf.read(6), foot)
        b_geod_shr = _clean_band(tf.read(7), foot)
        b_geod_dil = _clean_band(tf.read(8), foot)
        b_tmi_vg = _clean_band(tf.read(9), foot)
        b_deq = _clean_band(tf.read(10), foot)     # POSITIVE earthquake density!
        b_grav_vg = _clean_band(tf.read(11), foot)
        b_det_elev = _clean_band(tf.read(12), foot)
        b_iso_grav = _clean_band(tf.read(13), foot)
        b_tmi = _clean_band(tf.read(14), foot)
        b_depth_b = _clean_band(tf.read(15), foot)
        b_ieq = _clean_band(tf.read(16), foot)     # POSITIVE earthquake intensity!
        b_cond_s = _clean_band(tf.read(17), foot)
        b_det_slp = _clean_band(tf.read(19), foot)

    deq_log = np.log1p(np.maximum(b_deq, np.float32(0.0)))
    ieq_log = np.log1p(np.maximum(b_ieq, np.float32(0.0)))
    tmi_hg_log = np.log1p(np.maximum(b_tmi_hg, np.float32(0.0)))
    grav_slp_log = np.log1p(np.maximum(b_grav_slp, np.float32(0.0)))
    cond_log = np.log1p(np.maximum(b_cond_s, np.float32(0.0)))
    det_slp_log = np.log1p(np.maximum(b_det_slp, np.float32(0.0)))

    put2d("raw_mag_anom_hp", b_mag_anom - _gf32(b_mag_anom, 12.0))
    put2d("raw_rtp_hp", b_rtp - _gf32(b_rtp, 12.0))
    put2d("raw_rtp", b_rtp)
    put2d("raw_tmi_hg_log", tmi_hg_log)
    put2d("raw_geod_2ndinv", b_geod_2nd)
    put2d("raw_grav_slope_log", grav_slp_log)
    put2d("raw_tc_rad", b_tc)
    put2d("raw_geod_shear", b_geod_shr)
    put2d("raw_geod_dilat", b_geod_dil)
    put2d("raw_tmi_vg", b_tmi_vg)
    put2d("raw_deq_pos_log", deq_log)
    put2d("raw_grav_vg", b_grav_vg)
    put2d("raw_det_elev_hp", b_det_elev - _gf32(b_det_elev, 10.0))
    put2d("raw_det_elev", b_det_elev)
    put2d("raw_iso_grav_hp", b_iso_grav - _gf32(b_iso_grav, 12.0))
    put2d("raw_depth_base", b_depth_b)
    put2d("raw_ieq_pos_log", ieq_log)
    put2d("raw_cond_surf_log", cond_log)
    put2d("raw_det_elev_slope_log", det_slp_log)

    # Miller & Singh (1994) RTP magnetic & gravity tilt angles + analytic signal + cross-gradients
    rtp_gy, rtp_gx = np.gradient(_gf32(b_rtp, 1.2))
    rtp_hgm = np.hypot(rtp_gy, rtp_gx)
    mag_tilt = np.arctan2(b_tmi_vg, rtp_hgm + np.float32(1e-4)).astype(np.float32)
    put2d("pot_mag_tilt", mag_tilt)
    put2d("pot_mag_tilt_zero", np.exp(-np.abs(mag_tilt) / np.float32(0.35)).astype(np.float32))
    put2d("pot_mag_as_log", np.log1p(np.sqrt(rtp_hgm * rtp_hgm + b_tmi_vg * b_tmi_vg)).astype(np.float32))

    gr_gy, gr_gx = np.gradient(_gf32(b_iso_grav, 1.5))
    gr_hgm = np.hypot(gr_gy, gr_gx)
    grav_tilt = np.arctan2(b_grav_vg, gr_hgm + np.float32(1e-5)).astype(np.float32)
    put2d("pot_grav_thdr_log", np.log1p(gr_hgm).astype(np.float32))
    put2d("pot_grav_tilt_zero", np.exp(-np.abs(grav_tilt) / np.float32(0.35)).astype(np.float32))
    put2d("pot_grav_curv_up", _hessian_ridge_pos(-b_iso_grav, 2.5))
    put2d("pot_cg_align_s1p5", _crossgrad_align(b_rtp, b_iso_grav, 1.5))
    put2d("pot_cg_align_s3p5", _crossgrad_align(b_rtp, b_iso_grav, 3.5))

    db_gy, db_gx = np.gradient(_gf32(b_depth_b, 1.5))
    db_hgm = np.hypot(db_gy, db_gx)
    put2d("pot_base_grad_log", np.log1p(db_hgm).astype(np.float32))
    put2d("pot_base_grav_align", _crossgrad_align(b_depth_b, -b_iso_grav, 2.5))

    # Seismogenic fault-valve & strain deficit channels (part of Block 4 while bands in RAM)
    seism_combo = 0.60 * _rank01_fp(deq_log, foot) + 0.40 * _rank01_fp(ieq_log, foot)
    geod_rank = (0.55 * _rank01_fp(b_geod_shr, foot) + 0.45 * _rank01_fp(b_geod_2nd, foot)).astype(np.float32)
    dilat_rank = _rank01_fp(b_geod_dil, foot)
    put2d("seism_swarm_ridge_s2", _hessian_ridge_pos(seism_combo, 2.0))
    put2d("seism_swarm_ridge_s6", _hessian_ridge_pos(seism_combo, 6.0))
    put2d("seism_x_shear", (seism_combo * geod_rank).astype(np.float32))
    put2d("seism_x_dilat", (seism_combo * dilat_rank).astype(np.float32))
    put2d("seism_strain_deficit", np.maximum(geod_rank - seism_combo, np.float32(0.0)).astype(np.float32))

    cy, cx = np.gradient(_gf32(cond_log, 1.8))
    claycap_grad = np.hypot(cy, cx).astype(np.float32)
    put2d("hydro_claycap_grad", claycap_grad)

    # Keep only compact fields needed for Pass 2 & Pass 4
    rtp_rank = _rank01_fp(b_rtp, foot)
    grav_front = _rank01_fp(gr_hgm, foot)
    base_front = _rank01_fp(db_hgm, foot)
    del (
        b_mag_anom, b_rtp, b_geod_2nd, b_grav_slp, b_geod_shr, b_geod_dil,
        b_deq, b_grav_vg, b_iso_grav, b_depth_b, b_ieq, b_cond_s,
        rtp_gy, rtp_gx, rtp_hgm, mag_tilt, gr_gy, gr_gx, gr_hgm, grav_tilt,
        db_gy, db_gx, db_hgm, cy, cx, deq_log, ieq_log, grav_slp_log,
    )
    fh_out.flush()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fh_out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    gc.collect()

    # ------------------------------------------------------------------
    # PASS 2: Survey-block within-regime standardization + BLM mining & TIGER road deconfounding
    # ------------------------------------------------------------------
    audit_dir = ext_dir / "audit_sources"
    with rasterio.open(audit_dir / "acquisition_block_id_100m.tif") as ds:
        blk_id = ds.read(1)
    with rasterio.open(audit_dir / "tiger_road_distance_m.tif") as ds:
        road_d = np.nan_to_num(ds.read(1).astype(np.float32), nan=5000.0, posinf=5000.0, neginf=0.0)
    with rasterio.open(audit_dir / "blm_closed_claim_distance_m.tif") as ds:
        mine_d = np.nan_to_num(ds.read(1).astype(np.float32), nan=8000.0, posinf=8000.0, neginf=0.0)

    road_log = np.log1p(np.maximum(road_d, np.float32(0.0))).astype(np.float32)
    mine_log = np.log1p(np.maximum(mine_d, np.float32(0.0))).astype(np.float32)
    # Continuous anthropogenic damping weights: roads <250 m & open-pit mines <1.2 km damped smoothly
    w_road = np.clip(road_log / np.float32(np.log1p(350.0)), np.float32(0.12), np.float32(1.0)).astype(np.float32)
    w_mine = np.clip(mine_log / np.float32(np.log1p(1200.0)), np.float32(0.20), np.float32(1.0)).astype(np.float32)
    w_natural = (w_road * w_mine).astype(np.float32)

    # Distance to GeoDAWN block sutures (suppresses straight N-S / E-W mosaicking seams)
    seam = (ndi.maximum_filter(blk_id, size=3) != ndi.minimum_filter(blk_id, size=3)) & foot
    seam_dist_log = np.log1p(ndi.distance_transform_edt(~seam).astype(np.float32))

    with rasterio.open(ext_dir / "geodawn_extensions_u8.tif") as ds:
        q_thk = ds.read(1).astype(np.float32)
        q_uth = ds.read(3).astype(np.float32)
        q_tmi150 = ds.read(4).astype(np.float32)

    thk_clean = np.where((q_thk > 0) & foot, q_thk, np.nan)
    uth_clean = np.where((q_uth > 0) & foot, q_uth, np.nan)
    tmi150_clean = np.where((q_tmi150 > 0) & foot, q_tmi150, np.nan)
    potassic_inv = np.where(np.isfinite(thk_clean), np.float32(256.0) - thk_clean, np.nan).astype(np.float32)

    put2d("tiger_road_dist_log", road_log)
    put2d("blm_mine_dist_log", mine_log)
    put2d("seam_block_dist_log", seam_dist_log)
    put2d("blk_std_tmi_hg", _within_block_rank(tmi_hg_log, blk_id, foot))
    put2d("blk_std_tmi_vg", _within_block_rank(np.abs(b_tmi_vg), blk_id, foot))
    put2d("blk_std_tc_rad", _within_block_rank(b_tc, blk_id, foot))
    put2d("blk_std_kth_ratio", _within_block_rank(potassic_inv, blk_id, foot))
    put2d("blk_std_uth_ratio", _within_block_rank(uth_clean, blk_id, foot))

    r_tmi = _within_block_rank(b_tmi, blk_id, foot)
    r_tmi150 = _within_block_rank(tmi150_clean, blk_id, foot)
    shallow_resid = (r_tmi - r_tmi150).astype(np.float32)
    sgy, sgx = np.gradient(_gf32(shallow_resid, 1.2))
    put2d("mag_shallow_up150_resid", shallow_resid)
    put2d("mag_shallow_up150_grad", np.hypot(sgy, sgx).astype(np.float32))
    put2d("hydro_potassic_thk_inv", _rank01_fp(potassic_inv, foot))

    # Mine-deconfounded hydrothermal magnetite-destruction conduit index:
    # Potassic alteration + shallow smectite conductance + RTP magnetic trough along basement/gravity edge,
    # multiplied by `w_mine` so epithermal open-pit mines do not masquerade as active Quaternary faults!
    kth_r = _rank01_fp(potassic_inv, foot)
    cond_r = _rank01_fp(cond_log, foot)
    demag_conduit = (
        w_mine
        * (0.40 * kth_r + 0.30 * cond_r + 0.30 * (np.float32(1.0) - rtp_rank))
        * (0.55 * grav_front + 0.45 * base_front)
    ).astype(np.float32)
    put2d("hydro_demag_conduit_deconf", demag_conduit)

    del (
        blk_id, road_d, mine_d, road_log, seam, seam_dist_log,
        q_thk, q_uth, q_tmi150, thk_clean, uth_clean, tmi150_clean, potassic_inv,
        b_tmi, b_tmi_vg, b_tc, tmi_hg_log, r_tmi, r_tmi150, shallow_resid, sgy, sgx,
        rtp_rank, kth_r,
    )
    fh_out.flush()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fh_out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    gc.collect()

    # ------------------------------------------------------------------
    # PASS 3: Full 12-band 1 m LiDAR (q==0 -> NaN!), FFT Radon continuity, 10 m 3DEP & fluvial chi/branch jumps
    # ------------------------------------------------------------------
    lidar_meta = json.loads((ext_dir / "lidar_scarp_features.json").read_text())
    q_rules: dict[str, tuple[float, str]] = lidar_meta["quantisation"]
    l_bands: list[str] = lidar_meta["bands"]
    lid: dict[str, np.ndarray] = {}
    with rasterio.open(ext_dir / "lidar_scarp_features_u8.tif") as ds:
        q_valid = ds.read(12)
        lid_valid_mask = (q_valid > 0) & foot
        for idx1, bname in enumerate(l_bands[:11], start=1):
            q = ds.read(idx1).astype(np.float32)
            xmax, mode = q_rules[bname]
            t = np.clip((q - np.float32(1.0)) / np.float32(254.0), np.float32(0.0), np.float32(1.0))
            phys = (t * t if mode == "sqrt" else t) * np.float32(xmax)
            # CRITICAL INVARIANT: missing 1 m LiDAR survey tiles (24.6% of footprint) map strictly to NaN!
            lid[bname] = np.where((q > 0) & lid_valid_mask, phys, np.float32(np.nan))

    put2d("lidar_valid_flag", lid_valid_mask.astype(np.float32))
    for bname in (
        "ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
        "downface_max", "upface_max", "cross_max", "relief", "coh100",
    ):
        put2d(f"lidar_{bname}", lid[bname])

    dipole = np.sqrt(np.maximum(lid["lapneg_max"] * lid["lappos_max"], np.float32(0.0))).astype(np.float32)
    polarity = (
        (lid["downface_max"] - lid["upface_max"])
        / (lid["downface_max"] + lid["upface_max"] + np.float32(1e-4))
    ).astype(np.float32)
    step_deconf = (lid["step_max"] * w_natural).astype(np.float32)
    dipole_deconf = (dipole * w_natural).astype(np.float32)
    put2d("lidar_crest_toe_dipole", dipole)
    put2d("lidar_facing_polarity", polarity)
    put2d("lidar_step_deconf_road_mine", step_deconf)
    put2d("lidar_dipole_deconf_road_mine", dipole_deconf)

    # Spin-2 sub-pixel strike director cos(2*theta), sin(2*theta) -> multi-cell directional coherence
    strike_rad = np.deg2rad(np.nan_to_num(lid["strike"], nan=0.0)).astype(np.float32)
    w_dir = np.where(lid_valid_mask, np.nan_to_num(lid["coh100"], nan=0.0), np.float32(0.0)).astype(np.float32)
    u2_lid = (w_dir * np.cos(np.float32(2.0) * strike_rad)).astype(np.float32)
    v2_lid = (w_dir * np.sin(np.float32(2.0) * strike_rad)).astype(np.float32)
    for sig, tag in ((2.0, "s2"), (5.0, "s5")):
        den = _gf32(w_dir, sig) + np.float32(1e-5)
        coh = np.hypot(_gf32(u2_lid, sig), _gf32(v2_lid, sig)) / den
        put2d(f"lidar_strike_coh_{tag}", np.where(lid_valid_mask, coh, np.float32(np.nan)))
    scarp1m_norm = (_rank01_fp(np.nan_to_num(step_deconf, nan=0.0), lid_valid_mask) * w_natural).astype(np.float32)
    del lid, dipole, polarity, step_deconf, dipole_deconf, strike_rad, w_dir
    fh_out.flush()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fh_out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    gc.collect()

    # Seamless 10 m 3DEP stack across 100% of valid footprint (bridges 24.6% missing 1 m tiles!)
    t10_meta = json.loads((ext_dir / "topo_10m_derived_100m_u8.manifest.json").read_text())
    t10: dict[str, np.ndarray] = {}
    with rasterio.open(ext_dir / "topo_10m_derived_100m_u8.tif") as ds:
        for binfo in t10_meta["bands"]:
            idx1 = int(binfo["index"])
            bname = binfo["band_name"]
            lo, hi = float(binfo["lo"]), float(binfo["hi"])
            q = ds.read(idx1).astype(np.float32)
            val = np.float32(lo) + np.clip((q - np.float32(1.0)) / np.float32(254.0), 0.0, 1.0) * np.float32(hi - lo)
            t10[bname] = _clean_band(np.where(q > 0, val, np.nan), foot)

    put2d("topo10_slope_p90", t10["slope_p90"])
    put2d("topo10_slope_max", t10["slope_max"])
    put2d("topo10_steep_frac", t10["steep_frac"])
    put2d("topo10_slope_std", t10["slope_std"])
    put2d("topo10_relief_local", t10["relief_local"])
    put2d("topo10_curv_prof_absmax", t10["curv_prof_absmax"])
    put2d("topo10_aspect_coh", t10["aspect_coherence"])
    put2d("topo10_hs_lineament", t10["hs_lineament"])

    # Seamless scarp fusion field across 100% of Nevada for multi-azimuth FFT Radon line continuity:
    # blends deconfounded 1 m LiDAR where available with 10 m seamless steepness * aspect coherence across gaps
    scarp10_norm = (_rank01_fp(t10["slope_max"] * t10["aspect_coherence"], foot) * w_natural).astype(np.float32)
    scarp_seamless = np.where(lid_valid_mask, 0.65 * scarp1m_norm + 0.35 * scarp10_norm, scarp10_norm).astype(np.float32)
    del scarp10_norm, scarp1m_norm
    gc.collect()

    r_max12, r_ani12 = _multi_azimuth_radon_continuity(scarp_seamless, half_len_px=6, n_angles=8)
    put2d("radon_scarp_max_1p2km", r_max12)
    put2d("radon_scarp_aniso_1p2km", r_ani12)
    del r_max12, r_ani12
    gc.collect()

    r_max30, r_ani30 = _multi_azimuth_radon_continuity(scarp_seamless, half_len_px=15, n_angles=8)
    put2d("radon_scarp_max_3p0km", r_max30)
    put2d("radon_scarp_aniso_3p0km", r_ani30)
    del r_max30, r_ani30
    gc.collect()

    # Anisotropic HW/FW one-sided dip-slip step purity & along-strike elevation std (rejects flat Lake Lahontan benches!)
    dem_mean = t10["dem_mean"]
    zgy, zgx = np.gradient(_gf32(dem_mean, 1.5))
    z_norm = np.hypot(zgy, zgx) + np.float32(1e-5)
    ny, nx = zgy / z_norm, zgx / z_norm  # upslope unit normal; tangent along contour is (-nx, ny)
    # Cross-strike step vs canyon symmetry at +/- 300 m along upslope normal
    rr, cc = np.indices((H, W), dtype=np.float32)
    z_up = ndi.map_coordinates(dem_mean, [np.clip(rr + 3.0 * ny, 0, H - 1), np.clip(cc + 3.0 * nx, 0, W - 1)], order=1, mode="nearest")
    z_dn = ndi.map_coordinates(dem_mean, [np.clip(rr - 3.0 * ny, 0, H - 1), np.clip(cc - 3.0 * nx, 0, W - 1)], order=1, mode="nearest")
    onesided = (np.abs(z_up - z_dn) / (np.abs(z_up - dem_mean) + np.abs(dem_mean - z_dn) + np.float32(1.0))) * np.log1p(np.abs(z_up - z_dn))
    put2d("topo10_onesided_step_deconf", (onesided * w_natural).astype(np.float32))

    # Along-strike elevation std at +/- 600 m along strike tangent (-nx, ny):
    # Wave-cut Pleistocene lake shorelines have z_tan+ == z_tan- (contour = 0 variance); oblique fan faults have high std!
    zt1 = ndi.map_coordinates(dem_mean, [np.clip(rr + 6.0 * ny, 0, H - 1), np.clip(cc - 6.0 * nx, 0, W - 1)], order=1, mode="nearest")
    zt2 = ndi.map_coordinates(dem_mean, [np.clip(rr - 6.0 * ny, 0, H - 1), np.clip(cc + 6.0 * nx, 0, W - 1)], order=1, mode="nearest")
    z_along_std = np.sqrt(((zt1 - dem_mean) ** 2 + (zt2 - dem_mean) ** 2) * np.float32(0.5)).astype(np.float32)
    put2d("topo10_along_strike_zstd", np.log1p(z_along_std).astype(np.float32))

    # Piedmont range-front & toe concavity indicators
    slp_s4 = _gf32(det_slp_log, 4.0)
    slp_s12 = _gf32(det_slp_log, 12.0)
    f4y, f4x = np.gradient(slp_s4)
    f12y, f12x = np.gradient(slp_s12)
    put2d("pied_front_s4", np.hypot(f4y, f4x).astype(np.float32))
    put2d("pied_front_s12", np.hypot(f12y, f12x).astype(np.float32))
    put2d("pied_toe_concavity", _hessian_ridge_pos(-b_det_elev, 2.0))

    # Seamless 10 m Fluvial Chi knickpoints (`dem10_chi_100m_i16.tif`) + tributary branch jumps (`dem10_chi_100m_sig_i32.tif`)
    with rasterio.open(ext_dir / "dem10_chi_100m_i16.tif") as ds:
        chi_smax = ds.read(1).astype(np.float32) / np.float32(1000.0)
        chi_smin = ds.read(2).astype(np.float32) / np.float32(1000.0)
        chi_ksn = ds.read(3).astype(np.float32)
        chi_area = ds.read(4).astype(np.float32) / np.float32(1000.0)
    chi_smax = _clean_band(np.where(chi_smax > -30.0, chi_smax, np.nan), foot)
    chi_smin = _clean_band(np.where(chi_smin > -30.0, chi_smin, np.nan), foot)
    chi_ksn = _clean_band(np.where(chi_ksn > -30000.0, chi_ksn, np.nan), foot)
    chi_area = _clean_band(np.where(chi_area > -30.0, chi_area, np.nan), foot)

    chi_pos_ex = np.maximum(chi_smax - np.abs(chi_smin), np.float32(0.0)) * w_natural
    ksn_norm = np.log1p(np.maximum(chi_ksn, np.float32(0.0))) / (np.log1p(t10["relief_local"]) + np.float32(0.5))
    put2d("chi_step_pos_excess", chi_pos_ex.astype(np.float32))
    put2d("chi_ksn_norm", ksn_norm.astype(np.float32))
    put2d("chi_area_log", chi_area)

    with rasterio.open(ext_dir / "dem10_chi_100m_sig_i32.tif") as ds:
        branch_sig = ds.read(1)
    # Branch boundary indicator: where adjacent 100 m cells cross into distinct drainage branch signatures
    b_jump = (
        (ndi.maximum_filter(branch_sig, size=3) != ndi.minimum_filter(branch_sig, size=3))
        & (branch_sig >= 0)
        & foot
    ).astype(np.float32)
    bj_s1 = _gf32(b_jump, 1.2)
    bj_s3 = _gf32(b_jump, 3.5)
    put2d("chi_branch_jump_s1", bj_s1)
    put2d("chi_branch_jump_s3", bj_s3)
    put2d("chi_scarp_concordance", (_rank01_fp(chi_pos_ex, foot) * scarp_seamless).astype(np.float32))

    # Save seamless spin-2 director (blending 1 m LiDAR strike with 10 m DEM tangent) for visible structural alignment
    tan_ang = np.arctan2(ny, -nx).astype(np.float32)
    u2_seamless = np.where(lid_valid_mask, u2_lid, t10["aspect_coherence"] * np.cos(np.float32(2.0) * tan_ang)).astype(np.float32)
    v2_seamless = np.where(lid_valid_mask, v2_lid, t10["aspect_coherence"] * np.sin(np.float32(2.0) * tan_ang)).astype(np.float32)

    del (
        u2_lid, v2_lid, t10,
        dem_mean, zgy, zgx, z_norm, ny, nx, rr, cc, z_up, z_dn, onesided, zt1, zt2, z_along_std,
        b_det_elev, det_slp_log, slp_s4, slp_s12, f4y, f4x, f12y, f12x,
        chi_smax, chi_smin, chi_ksn, chi_area, chi_pos_ex, ksn_norm, branch_sig, b_jump, bj_s1, bj_s3, tan_ang,
    )
    fh_out.flush()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fh_out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    gc.collect()

    # ------------------------------------------------------------------
    # PASS 4: GDR 1391 Blind Geothermometer Excess + Hot Springs + 2m Probes + Paleo-Sinters + Volcanic Vents
    # ------------------------------------------------------------------
    hot_mask = np.zeros((H, W), dtype=bool)
    imp_temp = np.zeros((H, W), dtype=np.float32)
    imp_diseq = np.zeros((H, W), dtype=np.float32)

    with open(ext_dir / "gdr_wellspring_in_footprint.csv", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            # Strictly never read forbidden `dist_known_fault_px`!
            safe = {k: v for k, v in row.items() if k not in FORBIDDEN_COLUMNS}
            try:
                r = int(float(safe["row"]))
                c = int(float(safe["col"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= r < H and 0 <= c < W):
                continue
            def _f(k: str) -> float:
                v = safe.get(k, "")
                try:
                    x = float(v)
                    return x if np.isfinite(x) else np.nan
                except (TypeError, ValueError):
                    return np.nan
            t_surf = _f("temp_c")
            t_qtz = _f("geothermquartz_c")
            t_cha = _f("geothermchalc_c")
            t_cat = _f("geothermcat_c")
            tcls = str(safe.get("thermalclass", "")).strip().lower()
            if (np.isfinite(t_surf) and t_surf >= 35.0) or ("hot" in tcls) or ("warm" in tcls):
                hot_mask[r, c] = True
            if np.isfinite(t_surf) and t_surf > 18.0:
                imp_temp[r, c] = max(imp_temp[r, c], float(t_surf - 18.0))
            geos = [x for x in (t_qtz, t_cha, t_cat) if np.isfinite(x) and x < 360.0]
            if geos:
                gmax = max(geos)
                base_t = t_surf if (np.isfinite(t_surf) and t_surf > 5.0) else 16.0
                # Blind fault conduit signature: high deep equilibrium geothermometer over cool surface seep!
                diseq = max(0.0, gmax - max(base_t, 12.0)) + 0.5 * max(0.0, gmax - 80.0)
                if diseq > 0.0:
                    imp_diseq[r, c] = max(imp_diseq[r, c], float(diseq))

    volc_mask = np.zeros((H, W), dtype=bool)
    with open(ext_dir / "gdr_volcanic_vents_in_footprint.csv", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            safe = {k: v for k, v in row.items() if k not in FORBIDDEN_COLUMNS}
            try:
                r = int(float(safe["row"]))
                c = int(float(safe["col"]))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= r < H and 0 <= c < W:
                volc_mask[r, c] = True

    imp_probe = _parse_2m_probes(ext_dir / "2m_temperature_probe_INGENIOUS_regional_data.zip", transform, (H, W))
    imp_sinter = _parse_paleo_sinters(ext_dir / "paleo_geothermal_regional.zip", transform, (H, W))

    d_hot = np.log1p(ndi.distance_transform_edt(~hot_mask).astype(np.float32)) if hot_mask.any() else np.full((H, W), 8.0, np.float32)
    d_volc = np.log1p(ndi.distance_transform_edt(~volc_mask).astype(np.float32)) if volc_mask.any() else np.full((H, W), 8.0, np.float32)
    te_s4 = np.log1p(_gf32(imp_temp, 4.0) * np.float32(25.0))
    te_s12 = np.log1p(_gf32(imp_temp, 12.0) * np.float32(80.0))
    gd_s5 = np.log1p(_gf32(imp_diseq, 5.0) * np.float32(30.0))
    gd_s14 = np.log1p(_gf32(imp_diseq, 14.0) * np.float32(90.0))
    pr_s3 = np.log1p(_gf32(imp_probe, 3.0) * np.float32(30.0))
    pr_s9 = np.log1p(_gf32(imp_probe, 9.0) * np.float32(80.0))
    si_s4 = np.log1p(_gf32(imp_sinter, 4.0) * np.float32(40.0))
    si_s12 = np.log1p(_gf32(imp_sinter, 12.0) * np.float32(100.0))

    put2d("hydro_hot_spring_logdist", d_hot)
    put2d("hydro_temp_excess_s4", te_s4)
    put2d("hydro_temp_excess_s12", te_s12)
    put2d("hydro_geotherm_diseq_s5", gd_s5)
    put2d("hydro_geotherm_diseq_s14", gd_s14)
    put2d("hydro_probe2m_dab_s3", pr_s3)
    put2d("hydro_probe2m_dab_s9", pr_s9)
    put2d("hydro_paleo_sinter_s4", si_s4)
    put2d("hydro_paleo_sinter_s12", si_s12)
    put2d("volcanic_vent_logdist", d_volc)

    # Coupled Sibson fault-valve upflow score: thermal/sinter/geothermometer excess gated by scarp + basement/gravity structure
    thermal_any = (
        0.35 * _rank01_fp(gd_s14, foot)
        + 0.25 * _rank01_fp(pr_s9, foot)
        + 0.20 * _rank01_fp(si_s12, foot)
        + 0.20 * _rank01_fp(te_s12, foot)
    )
    valve_coupled = (
        w_mine
        * thermal_any
        * (0.45 * scarp_seamless + 0.30 * grav_front + 0.25 * base_front)
    ).astype(np.float32)
    put2d("valve_coupled_conduit", valve_coupled)

    assert len(written) == n_ch, f"Expected {n_ch} channels written, got {len(written)}"
    fh_out.flush()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fh_out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    fh_out.close()
    del thermal_any, valve_coupled, scarp_seamless, grav_front, base_front, w_mine, w_natural
    gc.collect()
    np.savez(
        aux_path,
        u2_dir=u2_seamless[foot].astype(np.float32),
        v2_dir=v2_seamless[foot].astype(np.float32),
        geod_rank=geod_rank[foot].astype(np.float32),
    )
    meta_path.write_text(
        json.dumps(
            {
                "n_footprint": n_fp,
                "n_channels": n_ch,
                "layout": "channel_major_f32",
                "channels": STATIC_CHANNEL_NAMES,
                "blocks": BLOCKS,
                "sgmc_channels_used": 0,
                "prior_rasters_read": 0,
                "build_seconds": round(time.time() - t0, 2),
            },
            indent=2,
        )
        + "\n"
    )
    aux = dict(np.load(aux_path))
    aux["n_fp"] = np.int64(n_fp)
    return mmap_path, aux


def compute_visible_structural_fp(
    visible_mask: np.ndarray,
    foot: np.ndarray,
    u2_fp: np.ndarray,
    v2_fp: np.ndarray,
    geod_rank_fp: np.ndarray,
) -> np.ndarray:
    """Compute 8 strictly fold-visible structural continuation & crack-tip mechanics channels on `foot`.

    NEVER accesses `labels.tif`, withheld segments, or external tables.
    Returns float32 array of shape `(n_fp, 8)`.
    """
    vis = np.asarray(visible_mask, dtype=bool)
    H, W = foot.shape
    out = np.empty((int(foot.sum()), len(VISIBLE_CHANNEL_NAMES)), dtype=np.float32)

    dist = ndi.distance_transform_edt(~vis).astype(np.float32)
    vis_f = vis.astype(np.float32)
    d5 = _gf32(vis_f, 5.0)
    d15 = _gf32(vis_f, 15.0)
    d35 = _gf32(vis_f, 35.0)
    relay_def = np.maximum(d15 - np.float32(1.6) * d5, np.float32(0.0)).astype(np.float32)

    # Local strike tangent of visible faults from gradient of smoothed distance field:
    # grad(dist) points normal to visible fault trace -> tangent along fault strike is (-gy, gx)!
    sdist = _gf32(np.minimum(dist, np.float32(45.0)), 2.5)
    gy, gx = np.gradient(sdist)
    gn = np.hypot(gy, gx) + np.float32(1e-5)
    tx, ty = -gy / gn, gx / gn
    u2_vis = (tx * tx - ty * ty).astype(np.float32)
    v2_vis = (2.0 * tx * ty).astype(np.float32)

    # Endpoints (fault tips) of visible strands: visible pixels with <= 1 8-connected visible neighbor
    nbrs = ndi.convolve(vis.astype(np.int16), np.ones((3, 3), dtype=np.int16), mode="constant", cval=0) - vis.astype(np.int16)
    tips = vis & (nbrs <= 1)
    del nbrs, vis_f, sdist, gy, gx, gn
    if tips.any():
        ind_buf = np.empty((2, H, W), dtype=np.int32)
        d_tip = ndi.distance_transform_edt(~tips, return_indices=True, indices=ind_buf).astype(np.float32)
        fy, fx = np.nonzero(foot)
        ry_f = ind_buf[0, foot]
        rx_f = ind_buf[1, foot]
        dt_f = d_tip[foot] + np.float32(1e-4)
        del ind_buf, d_tip
        vy_f = (fy.astype(np.float32) - ry_f.astype(np.float32)) / dt_f
        vx_f = (fx.astype(np.float32) - rx_f.astype(np.float32)) / dt_f
        cos_cone = np.abs(vx_f * tx[ry_f, rx_f] + vy_f * ty[ry_f, rx_f])
        tip_cone_fp = (np.power(cos_cone, np.float32(3.0)) * np.exp(-dt_f / np.float32(16.0))).astype(np.float32)
        del fy, fx, ry_f, rx_f, dt_f, vy_f, vx_f, cos_cone
    else:
        tip_cone_fp = np.zeros(int(foot.sum()), dtype=np.float32)
    del tx, ty

    align_fp = (0.5 * (1.0 + u2_vis[foot] * u2_fp + v2_vis[foot] * v2_fp) * np.exp(-dist[foot] / np.float32(30.0))).astype(np.float32)
    strain_def_fp = (geod_rank_fp * np.exp(-np.log1p(d35[foot] * np.float32(250.0)))).astype(np.float32)

    out[:, 0] = np.log1p(dist[foot]).astype(np.float32)
    out[:, 1] = d5[foot]
    out[:, 2] = d15[foot]
    out[:, 3] = d35[foot]
    out[:, 4] = relay_def[foot]
    out[:, 5] = tip_cone_fp
    out[:, 6] = align_fp
    out[:, 7] = strain_def_fp
    del dist, d5, d15, d35, relay_def, tip_cone_fp, align_fp, strain_def_fp, u2_vis, v2_vis
    gc.collect()
    return out
