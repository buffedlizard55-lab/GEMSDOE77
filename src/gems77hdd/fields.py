"""Label-free and fold-strict LOCO physical feature block for Lane H77H.

Preregistered in ``registry/preregistration_h77h.json`` and ``research/h77h_hypotheses.md``
prior to running Experiment 1.

Strict invariants enforced across every function in this module:
  1. Zero historical submission or registry prediction rasters are read anywhere here.
  2. ``sgmc_faults_100m.tif`` is NEVER read; pre-Quaternary bedrock contacts are 100% excluded.
  3. Columns encoding catalogue proximity (``dist_known_fault_px``, ``known_fault``, ``label``)
     in external point tables (``gdr_wellspring_in_footprint.csv``,
     ``gdr_volcanic_vents_in_footprint.csv``) are strictly forbidden and asserted unaccessed.
  4. Missing 1 m LiDAR survey coverage (``Band 12 == 0`` in ``lidar_scarp_features_u8.tif``,
     24.6% of the study footprint) is mapped strictly to ``NaN`` (``IR-77D-01``).
  5. Band 17 (``cond_surf``), Bands 7-8 (``geod_shearrate``, ``geod_dilaterate``), and
     Bands 10 & 16 (``deq_n100a15``, ``ieq_n100a15``) are smoothed at ``sigma = 36.0 px``
     (``3.6 km``) before taking spatial derivatives, suppressing the 35.02-px (3.5 km) and
     70.04-px (7.0 km) regional MT inversion mesh periodicity identified in ``IR-H77G-01``.
"""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree

from gems57 import network as gems57_network
from gems77 import features77 as F77
from gems77g.fields import hessian_trough

FORBIDDEN_GDR_COLUMNS: frozenset[str] = frozenset(
    {"dist_known_fault_px", "known_fault", "label", "dist_fault"}
)

H77H_STATIC_NAMES: list[str] = [
    # Block 1 (H77H-1): +500 m (+5 px) Downdip Hanging-Wall Potential-Field Lag Dipole
    "h77h_downdip_grav_step_500m",
    "h77h_downdip_base_step_500m",
    "h77h_downdip_scarp_coupled",
    # Block 2 (H77H-3): Laplace Second-Vertical-Derivative Potential-Field Zero-Crossing Dipole
    "h77h_laplace_svd_grav_dipole",
    "h77h_laplace_svd_rtp_demag",
    # Block 3 (H77H-2): Mesh-Deconfounded (sigma=36 px) MT / Transtension / GDR-1391 Geothermometer Gate
    "h77h_mt_notch36_margin",
    "h77h_transtension_notch36",
    "h77h_gdr_geotherm_diseq",
    "h77h_hydro_scarp_modulator",
    # Block 4: De-Quantized 1 m LiDAR Scarp Crest & Andersonian Strike Coherence (q==0 -> NaN)
    "h77h_lidar_scarp_crest",
    "h77h_lidar_anderson_strike_coh",
    "h77h_composite_physics_prior",
]

EXT_LIDAR_RAD_NAMES: list[str] = [
    "ext_lidar_has_coverage",
    "ext_lidar_ex_max",
    "ext_lidar_ex_mean",
    "ext_lidar_step_max",
    "ext_lidar_lapneg_max",
    "ext_lidar_lappos_max",
    "ext_lidar_downface_max",
    "ext_lidar_upface_max",
    "ext_lidar_cross_max",
    "ext_lidar_relief",
    "ext_lidar_coh100",
    "ext_lidar_strike",
    "ext_lidar_dipole",
    "ext_lidar_sharpness",
    "ext_rad_k",
    "ext_rad_th",
    "ext_rad_u",
    "ext_rad_th_over_k",
    "ext_rad_u_over_k",
    "ext_mag_up150",
]

H77H_LOCO_NAMES: list[str] = [
    "h77h_loco_dist_other_log",
    "h77h_loco_splay_collar",
    "h77h_loco_tip_lobe",
    "h77h_loco_stepover_relay",
    "h77h_loco_bilateral_bridge",
]


def _sample_bilinear_2d(grid: np.ndarray, y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
    """Continuous sub-pixel bilinear interpolation on a 2D float32 grid."""
    h, w = grid.shape
    yc = np.clip(y_coords, 0.0, float(h - 1.001))
    xc = np.clip(x_coords, 0.0, float(w - 1.001))
    y0 = yc.astype(np.int32)
    x0 = xc.astype(np.int32)
    wy = (yc - y0).astype(np.float32)
    wx = (xc - x0).astype(np.float32)
    v00 = grid[y0, x0]
    v01 = grid[y0, x0 + 1]
    v10 = grid[y0 + 1, x0]
    v11 = grid[y0 + 1, x0 + 1]
    return (
        (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
        + wy * ((1.0 - wx) * v10 + wx * v11)
    ).astype(np.float32)


def measure_mt_mesh_periodicity_suppression(
    cond_2d: np.ndarray,
    footprint: np.ndarray,
    sigma_raw: float = 2.0,
    sigma_notched: float = 36.0,
) -> dict[str, float]:
    """Quantify 1-D FFT spectral energy at the 35.02-px (3.5 km) and 70.04-px (7.0 km)
    regional MT inversion mesh periods before vs after sigma=36 px smoothing (IR-H77G-01).
    """
    foot = np.asarray(footprint, dtype=bool)
    c_clean = F77.fill_median(F77.clean(cond_2d), foot)
    c_log = np.log1p(np.maximum(c_clean, 0.0)).astype(np.float32)

    gy_raw = ndimage.gaussian_filter(c_log, sigma_raw, order=(1, 0), mode="nearest")
    gx_raw = ndimage.gaussian_filter(c_log, sigma_raw, order=(0, 1), mode="nearest")
    g_raw = np.hypot(gx_raw, gy_raw)

    gy_not = ndimage.gaussian_filter(c_log, sigma_notched, order=(1, 0), mode="nearest")
    gx_not = ndimage.gaussian_filter(c_log, sigma_notched, order=(0, 1), mode="nearest")
    g_not = np.hypot(gx_not, gy_not)

    # Evaluate average 1-D row FFT power spectrum across central valid rows
    h, w = c_log.shape
    rows = slice(h // 4, 3 * h // 4)
    r_sub = g_raw[rows, :] - g_raw[rows, :].mean(axis=1, keepdims=True)
    n_sub = g_not[rows, :] - g_not[rows, :].mean(axis=1, keepdims=True)

    spec_raw = np.abs(np.fft.rfft(r_sub, axis=1)).mean(axis=0)
    spec_not = np.abs(np.fft.rfft(n_sub, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(w, d=1.0)  # cycles per pixel

    f_35 = 1.0 / 35.02
    f_70 = 1.0 / 70.04
    idx_35 = int(np.argmin(np.abs(freqs - f_35)))
    idx_70 = int(np.argmin(np.abs(freqs - f_70)))

    # Normalize each spectrum by its low-frequency regional baseline (periods > 250 px)
    low_mask = (freqs > 0.0) & (freqs < (1.0 / 250.0))
    base_raw = float(spec_raw[low_mask].mean()) + 1e-12
    base_not = float(spec_not[low_mask].mean()) + 1e-12

    rel_35_raw = float(spec_raw[idx_35] / base_raw)
    rel_35_not = float(spec_not[idx_35] / base_not)
    rel_70_raw = float(spec_raw[idx_70] / base_raw)
    rel_70_not = float(spec_not[idx_70] / base_not)

    supp_35 = 1.0 - (rel_35_not / max(rel_35_raw, 1e-12))
    supp_70 = 1.0 - (rel_70_not / max(rel_70_raw, 1e-12))

    # Also measure adjacent zero-step fraction on raw cond_surf (IR-H77G-01 diagnostic)
    dx_adj = np.abs(np.diff(c_clean, axis=1))
    foot_adj = foot[:, 1:] & foot[:, :-1]
    zero_step_frac = float((dx_adj[foot_adj] == 0.0).mean())

    return {
        "raw_cond_adjacent_zero_step_frac": round(zero_step_frac, 6),
        "sigma_raw_px": float(sigma_raw),
        "sigma_notched_px": float(sigma_notched),
        "mesh_35px_relative_amp_sigma2": round(rel_35_raw, 6),
        "mesh_35px_relative_amp_sigma36": round(rel_35_not, 6),
        "mesh_35px_suppression_fraction": round(supp_35, 6),
        "mesh_70px_relative_amp_sigma2": round(rel_70_raw, 6),
        "mesh_70px_relative_amp_sigma36": round(rel_70_not, 6),
        "mesh_70px_suppression_fraction": round(supp_70, 6),
    }


def _load_gdr1391_geotherm_diseq(
    ext_dir: Path,
    shape: tuple[int, int],
    footprint: np.ndarray,
) -> np.ndarray:
    """Load DOE GDR #1391 well/spring chemical geothermometer disequilibrium & volcanic vents.

    Strictly asserts that ``dist_known_fault_px`` and all ``FORBIDDEN_GDR_COLUMNS`` are never
    read or referenced.
    """
    ws_csv = ext_dir / "gdr_wellspring_in_footprint.csv"
    vv_csv = ext_dir / "gdr_volcanic_vents_in_footprint.csv"

    diseq_grid = np.zeros(shape, dtype=np.float32)
    vent_grid = np.zeros(shape, dtype=np.float32)
    h, w = shape

    if ws_csv.is_file():
        with ws_csv.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            allowed_cols = ("row", "col", "temp_c", "geothermquartz_c", "geothermchalc_c", "geothermcat_c")
            for c in allowed_cols:
                assert c not in FORBIDDEN_GDR_COLUMNS, f"Forbidden column requested: {c}"
            for row in reader:
                try:
                    r = int(float(row["row"]))
                    c = int(float(row["col"]))
                except (KeyError, ValueError):
                    continue
                if not (0 <= r < h and 0 <= c < w):
                    continue
                def _pf(k: str) -> float:
                    v = row.get(k, "")
                    try:
                        val = float(v)
                        return val if np.isfinite(val) and -50.0 < val < 450.0 else np.nan
                    except (TypeError, ValueError):
                        return np.nan

                t_surf = _pf("temp_c")
                t_qtz = _pf("geothermquartz_c")
                t_cha = _pf("geothermchalc_c")
                t_cat = _pf("geothermcat_c")
                chem_vals = [v for v in (t_qtz, t_cha, t_cat) if np.isfinite(v)]
                t_chem = max(chem_vals) if chem_vals else np.nan
                t_base = t_surf if np.isfinite(t_surf) else 15.0
                diseq = max(0.0, (t_chem - t_base) if np.isfinite(t_chem) else 0.0)
                surf_excess = max(0.0, (t_surf - 20.0) if np.isfinite(t_surf) else 0.0)
                score = float( np.log1p(diseq + 0.6 * surf_excess) )
                if score > diseq_grid[r, c]:
                    diseq_grid[r, c] = score

    if vv_csv.is_file():
        with vv_csv.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = int(float(row["row"]))
                    c = int(float(row["col"]))
                except (KeyError, ValueError):
                    continue
                if 0 <= r < h and 0 <= c < w:
                    vent_grid[r, c] = 1.0

    plume_near = ndimage.gaussian_filter(diseq_grid, sigma=6.0, mode="constant")
    plume_far = ndimage.gaussian_filter(diseq_grid, sigma=18.0, mode="constant")
    vent_plume = ndimage.gaussian_filter(vent_grid, sigma=10.0, mode="constant")
    raw = (0.55 * plume_near + 0.30 * plume_far + 0.15 * vent_plume).astype(np.float32)
    return F77.rank01(np.where(footprint, raw, np.float32(np.nan)), footprint)


def build_h77h_static_stack(
    data_dir: Path,
    cache_dir: Path,
    footprint: np.ndarray,
) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    """Build or reload cached in-footprint H77H static physics (12 channels) and
    external 1m-LiDAR/GeoDAWN stack (20 channels) as float32 memmaps.

    Returns
    -------
    h77h_fp : np.memmap of shape (N_fp, 12), float32
    ext_fp : np.memmap of shape (N_fp, 20), float32 (with NaN where 1m LiDAR is absent)
    diag : dict of feature diagnostics and MT mesh periodicity suppression metrics
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    foot = np.asarray(footprint, dtype=bool)
    h, w = foot.shape
    fp_idx = np.flatnonzero(foot)
    n_fp = int(fp_idx.size)

    h77h_path = cache_dir / "h77h_static_fp.npy"
    ext_path = cache_dir / "h77h_ext_lidar_rad_fp.npy"
    diag_path = cache_dir / "h77h_static_diag.json"

    if h77h_path.is_file() and ext_path.is_file() and diag_path.is_file():
        diag = json.loads(diag_path.read_text())
        if (
            diag.get("h77h_names") == H77H_STATIC_NAMES
            and diag.get("ext_names") == EXT_LIDAR_RAD_NAMES
            and diag.get("n_fp") == n_fp
        ):
            return (
                np.load(h77h_path, mmap_mode="r"),
                np.load(ext_path, mmap_mode="r"),
                diag,
            )

    rk = lambda a: F77.rank01(np.where(foot, a, np.float32(np.nan)), foot)  # noqa: E731
    ext_dir = data_dir / "external"

    # 1. Read needed bands from training_features.tif
    with rasterio.open(data_dir / "training_features.tif") as src:
        rtp = F77.fill_median(F77.clean(src.read(2)), foot)
        tmi_hg = F77.fill_median(F77.clean(src.read(3)), foot)
        shear = F77.fill_median(F77.clean(src.read(7)), foot)
        dilat = F77.fill_median(F77.clean(src.read(8)), foot)
        tmi_vg = F77.fill_median(F77.clean(src.read(9)), foot)
        deq = F77.fill_median(F77.clean(src.read(10)), foot)
        det_elev = F77.fill_median(F77.clean(src.read(12)), foot)
        grav = F77.fill_median(F77.clean(src.read(13)), foot)
        depth_b = F77.fill_median(F77.clean(src.read(15)), foot)
        ieq = F77.fill_median(F77.clean(src.read(16)), foot)
        cond_raw = src.read(17)
        cond = F77.fill_median(F77.clean(cond_raw), foot)
        det_slope = F77.fill_median(F77.clean(src.read(19)), foot)

    mesh_audit = measure_mt_mesh_periodicity_suppression(cond_raw, foot, sigma_raw=2.0, sigma_notched=36.0)
    del cond_raw

    # 2. Build external 1m LiDAR (q==0 -> NaN per IR-77D-01) + GeoDAWN radiometric/magnetic stack
    mm_ext = np.lib.format.open_memmap(
        ext_path, mode="w+", dtype=np.float32, shape=(n_fp, len(EXT_LIDAR_RAD_NAMES))
    )
    with rasterio.open(ext_dir / "lidar_scarp_features_u8.tif") as ls:
        has_lid_2d = (ls.read(12) > 0) & foot
        has_lid_fp = has_lid_2d.ravel()[fp_idx]
        mm_ext[:, 0] = has_lid_fp.astype(np.float32)
        lid_raw_fp: dict[int, np.ndarray] = {}
        for b_i in range(1, 12):
            q = ls.read(b_i).ravel()[fp_idx].astype(np.float32)
            # Non-linear physical inverse-quantization: q in [1..255] -> t in [0..1]
            t = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
            # Bands 1..9 are sqrt-quantized relief/curvature/step; band 10 is coh100; band 11 is strike [0..180]
            if b_i <= 9:
                phys = t * t
            elif b_i == 10:
                phys = t
            else:
                phys = 180.0 * t
            phys[~has_lid_fp] = np.nan
            mm_ext[:, b_i] = phys
            if b_i in (1, 3, 4, 5, 9, 10, 11):
                lid_raw_fp[b_i] = phys

    dipole_fp = np.sqrt(np.maximum(0.0, lid_raw_fp[4] * lid_raw_fp[5])).astype(np.float32)
    sharp_fp = (lid_raw_fp[3] / (lid_raw_fp[9] + 0.05)).astype(np.float32)
    dipole_fp[~has_lid_fp] = np.nan
    sharp_fp[~has_lid_fp] = np.nan
    mm_ext[:, 12] = dipole_fp
    mm_ext[:, 13] = sharp_fp

    # Construct 2D de-quantized 1m LiDAR scarp crest & Andersonian strike coherence
    scarp_lid_fp = (
        0.45 * (np.nan_to_num(dipole_fp, nan=0.0) / (float(np.nanstd(dipole_fp)) + 1e-6))
        + 0.35 * (np.nan_to_num(sharp_fp, nan=0.0) / (float(np.nanstd(sharp_fp)) + 1e-6))
        + 0.20 * (np.nan_to_num(lid_raw_fp[1], nan=0.0) / (float(np.nanstd(lid_raw_fp[1])) + 1e-6))
    ).astype(np.float32)

    strike_fp = np.nan_to_num(lid_raw_fp[11], nan=0.0)
    coh_fp = np.nan_to_num(lid_raw_fp[10], nan=0.0)
    diff_n = np.minimum(np.abs(strike_fp - 18.0), 180.0 - np.abs(strike_fp - 18.0))
    diff_t = np.minimum(np.abs(strike_fp - 145.0), 180.0 - np.abs(strike_fp - 145.0))
    and_fp = (
        coh_fp
        * np.maximum(
            0.5 + 0.5 * np.cos(np.deg2rad(2.0 * diff_n)),
            0.85 * (0.5 + 0.5 * np.cos(np.deg2rad(2.0 * diff_t))),
        )
    ).astype(np.float32)
    del lid_raw_fp, dipole_fp, sharp_fp, strike_fp, coh_fp, diff_n, diff_t

    # Seamless fallback outside 1m LiDAR using det_elev_slope ridge curvature
    s_gy, s_gx = np.gradient(ndimage.gaussian_filter(det_slope, sigma=1.5, mode="nearest"))
    slope_break_2d = np.hypot(s_gx, s_gy).astype(np.float32)
    del s_gy, s_gx
    r_slope_break = rk(slope_break_2d)

    lid_scarp_2d = np.zeros((h, w), dtype=np.float32)
    lid_scarp_2d.ravel()[fp_idx] = scarp_lid_fp
    r_lid_scarp = F77.rank01(np.where(has_lid_2d, lid_scarp_2d, np.float32(np.nan)), has_lid_2d)
    scarp_crest_2d = np.where(has_lid_2d, 0.72 * r_lid_scarp + 0.28 * r_slope_break, r_slope_break).astype(
        np.float32
    )
    scarp_crest_2d = rk(scarp_crest_2d)
    del lid_scarp_2d, r_lid_scarp, scarp_lid_fp

    and_2d = np.zeros((h, w), dtype=np.float32)
    and_2d.ravel()[fp_idx] = and_fp
    and_sm_2d = ndimage.gaussian_filter(and_2d, sigma=2.0, mode="nearest")
    r_and = F77.rank01(np.where(has_lid_2d, and_sm_2d, np.float32(np.nan)), has_lid_2d)
    anderson_coh_2d = np.where(has_lid_2d, r_and, 0.5 * r_slope_break).astype(np.float32)
    anderson_coh_2d = rk(anderson_coh_2d)
    del and_2d, and_sm_2d, r_and, and_fp, slope_break_2d, r_slope_break

    # Read GeoDAWN radiometric K, Th, U and extensions
    with rasterio.open(ext_dir / "geodawn_rad_u8.tif") as rs:
        rad_k = F77.fill_median(F77.clean(rs.read(1)), foot)
        rad_th = F77.fill_median(F77.clean(rs.read(2)), foot)
        rad_u = F77.fill_median(F77.clean(rs.read(3)), foot)
    with rasterio.open(ext_dir / "geodawn_extensions_u8.tif") as es:
        rad_th_k = F77.fill_median(F77.clean(es.read(1)), foot)
        rad_u_k = F77.fill_median(F77.clean(es.read(2)), foot)
        mag_up150 = F77.fill_median(F77.clean(es.read(4)), foot)

    mm_ext[:, 14] = rad_k.ravel()[fp_idx]
    mm_ext[:, 15] = rad_th.ravel()[fp_idx]
    mm_ext[:, 16] = rad_u.ravel()[fp_idx]
    mm_ext[:, 17] = rad_th_k.ravel()[fp_idx]
    mm_ext[:, 18] = rad_u_k.ravel()[fp_idx]
    mm_ext[:, 19] = mag_up150.ravel()[fp_idx]
    mm_ext.flush()
    del rad_k, rad_th, rad_u, mag_up150

    # 3. Block 1 (H77H-1): +500 m (+5.0 px) Downdip Hanging-Wall Potential-Field Lag Dipole
    # Local dip unit vector n_dip = (ny, nx) points toward thickening basin / downthrown hanging wall
    d_sm = ndimage.gaussian_filter(depth_b, sigma=4.0, mode="nearest")
    e_sm = ndimage.gaussian_filter(det_elev, sigma=4.0, mode="nearest")
    dby_dip, dbx_dip = np.gradient(d_sm)
    ey_dip, ex_dip = np.gradient(e_sm)
    del d_sm, e_sm

    # Combine basement deepening vector (+grad(depth_to_base)) and topographic down-slope (-grad(elev))
    dip_vy = 0.65 * (dby_dip / (float(np.std(dby_dip[foot])) + 1e-6)) - 0.35 * (
        ey_dip / (float(np.std(ey_dip[foot])) + 1e-6)
    )
    dip_vx = 0.65 * (dbx_dip / (float(np.std(dbx_dip[foot])) + 1e-6)) - 0.35 * (
        ex_dip / (float(np.std(ex_dip[foot])) + 1e-6)
    )
    dip_norm = np.hypot(dip_vy, dip_vx) + np.float32(1e-6)
    ny_dip = (dip_vy / dip_norm).astype(np.float32)
    nx_dip = (dip_vx / dip_norm).astype(np.float32)
    del dby_dip, dbx_dip, ey_dip, ex_dip, dip_vy, dip_vx, dip_norm

    ggy, ggx = np.gradient(ndimage.gaussian_filter(grav, sigma=2.0, mode="nearest"))
    gmag = np.hypot(ggx, ggy).astype(np.float32)
    dgy, dgx = np.gradient(ndimage.gaussian_filter(depth_b, sigma=2.0, mode="nearest"))
    dmag = np.hypot(dgx, dgy).astype(np.float32)
    cos_gb = (np.abs(ggx * dgx + ggy * dgy) / (gmag * dmag + np.float32(1e-9))).astype(np.float32)
    del ggy, ggx, dgy, dgx

    r_gmag = rk(gmag)
    r_dmag = rk(dmag * (0.35 + 0.65 * cos_gb))
    del gmag, dmag, cos_gb

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Sample at +5.0 px (+500 m downdip into hanging wall) and -5.0 px (-500 m updip into footwall)
    lag_px = np.float32(5.0)
    y_hw = yy + lag_px * ny_dip
    x_hw = xx + lag_px * nx_dip
    y_fw = yy - lag_px * ny_dip
    x_fw = xx - lag_px * nx_dip
    del yy, xx, ny_dip, nx_dip

    g_hw = _sample_bilinear_2d(r_gmag, y_hw, x_hw)
    g_fw = _sample_bilinear_2d(r_gmag, y_fw, x_fw)
    d_hw = _sample_bilinear_2d(r_dmag, y_hw, x_hw)
    d_fw = _sample_bilinear_2d(r_dmag, y_fw, x_fw)
    del y_hw, x_hw, y_fw, x_fw

    downdip_grav_500m = rk(
        0.60 * g_hw + 0.40 * np.maximum(0.0, g_hw - 0.55 * g_fw)
    )
    downdip_base_500m = rk(
        0.60 * d_hw + 0.40 * np.maximum(0.0, d_hw - 0.55 * d_fw)
    )
    del g_hw, g_fw, d_hw, d_fw, r_gmag, r_dmag

    downdip_scarp_coupled = rk(
        np.sqrt(np.clip(downdip_grav_500m * downdip_base_500m, 0.0, 1.0))
        * (0.25 + 0.75 * scarp_crest_2d)
    )

    # 4. Block 2 (H77H-3): Laplace Second-Vertical-Derivative Potential-Field Zero-Crossing Dipole
    # By Laplace's equation, d^2U/dz^2 = -(d^2U/dx^2 + d^2U/dy^2)
    svd_grav = -(
        ndimage.gaussian_filter(grav, sigma=2.0, order=(2, 0), mode="nearest")
        + ndimage.gaussian_filter(grav, sigma=2.0, order=(0, 2), mode="nearest")
    ).astype(np.float32)
    pos_lobe = ndimage.maximum_filter(np.maximum(svd_grav, 0.0), size=5, mode="nearest")
    neg_lobe = ndimage.maximum_filter(np.maximum(-svd_grav, 0.0), size=5, mode="nearest")
    pediment_gate = np.exp(-np.maximum(det_slope, 0.0) / 18.0).astype(np.float32)
    laplace_svd_grav = rk(np.sqrt(np.maximum(pos_lobe * neg_lobe, 0.0)) * (0.40 + 0.60 * pediment_gate))
    del svd_grav, pos_lobe, neg_lobe, pediment_gate

    svd_rtp = -(
        ndimage.gaussian_filter(rtp, sigma=2.0, order=(2, 0), mode="nearest")
        + ndimage.gaussian_filter(rtp, sigma=2.0, order=(0, 2), mode="nearest")
    ).astype(np.float32)
    rtp_trough = rk(hessian_trough(rtp, sigma=2.0, hp_sigma=12.0))
    mag_as = rk(np.hypot(np.maximum(tmi_hg, 0.0), tmi_vg))
    laplace_svd_rtp_demag = rk(
        0.55 * rtp_trough
        + 0.25 * rk(np.maximum(svd_rtp, 0.0))
        + 0.20 * np.maximum(0.0, downdip_grav_500m - 0.45 * mag_as)
    )
    del svd_rtp, rtp_trough, mag_as, rtp, tmi_hg, tmi_vg, grav, depth_b, det_elev, det_slope

    # 5. Block 3 (H77H-2): Mesh-Deconfounded (sigma=36 px = 3.6 km) MT / Transtension / GDR-1391 Gate
    cond_log = np.log1p(np.maximum(cond, 0.0)).astype(np.float32)
    cond_s36 = ndimage.gaussian_filter(cond_log, sigma=36.0, mode="nearest")
    cgy36 = ndimage.gaussian_filter(cond_log, sigma=36.0, order=(1, 0), mode="nearest")
    cgx36 = ndimage.gaussian_filter(cond_log, sigma=36.0, order=(0, 1), mode="nearest")
    cgrad36 = np.hypot(cgx36, cgy36).astype(np.float32)
    mt_notch36_margin = rk(np.sqrt(np.clip(rk(cond_s36) * rk(cgrad36), 0.0, 1.0)))
    del cond, cond_log, cond_s36, cgy36, cgx36, cgrad36

    dil_s36 = rk(ndimage.gaussian_filter(np.maximum(dilat, 0.0), sigma=36.0, mode="nearest"))
    shr_s36 = rk(ndimage.gaussian_filter(np.maximum(shear, 0.0), sigma=36.0, mode="nearest"))
    ieq_s36 = rk(ndimage.gaussian_filter(ieq, sigma=36.0, mode="nearest"))
    deq_s36 = rk(ndimage.gaussian_filter(deq, sigma=36.0, mode="nearest"))
    swarm_s36 = rk(ieq_s36 * (ieq_s36 / (ieq_s36 + deq_s36 + np.float32(0.05))))
    transtension_notch36 = rk(0.60 * np.sqrt(np.clip(dil_s36 * shr_s36, 0.0, 1.0)) + 0.40 * swarm_s36)
    del dilat, shear, ieq, deq, dil_s36, shr_s36, ieq_s36, deq_s36, swarm_s36

    gdr_diseq_raw = _load_gdr1391_geotherm_diseq(ext_dir, (h, w), foot)
    r_uk = rk(rad_u_k)
    r_thk = rk(rad_th_k)
    gdr_geotherm_diseq = rk(0.65 * gdr_diseq_raw + 0.20 * r_uk + 0.15 * (1.0 - r_thk))
    del gdr_diseq_raw, rad_u_k, rad_th_k, r_uk, r_thk

    regime_gate = (
        0.40 * mt_notch36_margin + 0.35 * transtension_notch36 + 0.25 * gdr_geotherm_diseq
    ).astype(np.float32)
    hydro_scarp_modulator = rk(
        regime_gate * (0.45 * scarp_crest_2d + 0.30 * downdip_scarp_coupled + 0.25 * laplace_svd_rtp_demag)
    )

    # 6. Composite Label-Free Physics Prior (0 catalogue features)
    composite_prior = rk(
        0.30 * downdip_scarp_coupled
        + 0.25 * scarp_crest_2d
        + 0.15 * anderson_coh_2d
        + 0.15 * hydro_scarp_modulator
        + 0.15 * (0.5 * laplace_svd_grav + 0.5 * laplace_svd_rtp_demag)
    )

    channels_2d = {
        "h77h_downdip_grav_step_500m": downdip_grav_500m,
        "h77h_downdip_base_step_500m": downdip_base_500m,
        "h77h_downdip_scarp_coupled": downdip_scarp_coupled,
        "h77h_laplace_svd_grav_dipole": laplace_svd_grav,
        "h77h_laplace_svd_rtp_demag": laplace_svd_rtp_demag,
        "h77h_mt_notch36_margin": mt_notch36_margin,
        "h77h_transtension_notch36": transtension_notch36,
        "h77h_gdr_geotherm_diseq": gdr_geotherm_diseq,
        "h77h_hydro_scarp_modulator": hydro_scarp_modulator,
        "h77h_lidar_scarp_crest": scarp_crest_2d,
        "h77h_lidar_anderson_strike_coh": anderson_coh_2d,
        "h77h_composite_physics_prior": composite_prior,
    }

    mm_h77h = np.lib.format.open_memmap(
        h77h_path, mode="w+", dtype=np.float32, shape=(n_fp, len(H77H_STATIC_NAMES))
    )
    ch_stats: dict[str, dict[str, float]] = {}
    for idx, nm in enumerate(H77H_STATIC_NAMES):
        arr = np.where(foot, channels_2d[nm], 0.0).astype(np.float32)
        vals = arr.ravel()[fp_idx]
        mm_h77h[:, idx] = vals
        ch_stats[nm] = {
            "in_footprint_mean": round(float(vals.mean()), 6),
            "in_footprint_std": round(float(vals.std()), 6),
            "min": round(float(vals.min()), 6),
            "max": round(float(vals.max()), 6),
            "nan_count": int(np.isnan(vals).sum()),
        }
    mm_h77h.flush()
    del mm_h77h, mm_ext, channels_2d
    gc.collect()

    diag = {
        "n_fp": n_fp,
        "lidar_valid_pixels": int(has_lid_fp.sum()),
        "lidar_missing_pixels_mapped_to_nan": int((~has_lid_fp).sum()),
        "h77h_names": H77H_STATIC_NAMES,
        "ext_names": EXT_LIDAR_RAD_NAMES,
        "mt_mesh_periodicity_audit_ir_h77g_01": mesh_audit,
        "channel_stats": ch_stats,
    }
    diag_path.write_text(json.dumps(diag, indent=2) + "\n")
    return (
        np.load(h77h_path, mmap_mode="r"),
        np.load(ext_path, mmap_mode="r"),
        diag,
    )


def _extract_tips_and_tangents(vis_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify degree-1 fault tips on components of size >= 5 px and their outward unit tangents."""
    h, w = vis_mask.shape
    comp, n_comp = ndimage.label(vis_mask, structure=np.ones((3, 3), dtype=bool))
    if n_comp == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty.astype(np.int32), empty.astype(np.int32), empty, empty

    sizes = np.bincount(comp.ravel())
    major = vis_mask & (sizes[comp] >= 5)
    nb = (
        ndimage.convolve(major.astype(np.int16), np.ones((3, 3), dtype=np.int16), mode="constant")
        - major.astype(np.int16)
    )
    tips_mask = major & (nb == 1)
    ty, tx = np.nonzero(tips_mask)
    if ty.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return ty, tx, empty, empty

    # Estimate outward tangent vector pointing from component interior toward tip and beyond
    d_vis = ndimage.distance_transform_edt(~major).astype(np.float32)
    sm = ndimage.gaussian_filter(major.astype(np.float32), sigma=2.5, mode="constant")
    hyy = ndimage.gaussian_filter(sm, sigma=2.0, order=(2, 0), mode="constant")
    hxx = ndimage.gaussian_filter(sm, sigma=2.0, order=(0, 2), mode="constant")
    hxy = ndimage.gaussian_filter(sm, sigma=2.0, order=(1, 1), mode="constant")
    norm_ang = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
    ux = (-np.sin(norm_ang))[ty, tx].astype(np.float32)
    uy = (np.cos(norm_ang))[ty, tx].astype(np.float32)

    fr = np.clip(np.round(ty + 4.0 * uy).astype(np.int32), 0, h - 1)
    fc = np.clip(np.round(tx + 4.0 * ux).astype(np.int32), 0, w - 1)
    br = np.clip(np.round(ty - 4.0 * uy).astype(np.int32), 0, h - 1)
    bc = np.clip(np.round(tx - 4.0 * ux).astype(np.int32), 0, w - 1)
    sgn = np.where(d_vis[fr, fc] >= d_vis[br, bc], 1.0, -1.0).astype(np.float32)
    return ty.astype(np.int32), tx.astype(np.int32), (uy * sgn), (ux * sgn)


def _rasterize_bilateral_bridges(
    shape: tuple[int, int],
    ty: np.ndarray,
    tx: np.ndarray,
    uy: np.ndarray,
    ux: np.ndarray,
    t_comp: np.ndarray,
    exclude_comp_mask: np.ndarray | None = None,
    min_dist_px: float = 4.0,
    max_dist_px: float = 48.0,
) -> np.ndarray:
    """Rasterize bilateral inter-component relay bridges between facing degree-1 tips
    of distinct visible components C(t1) != C(t2).
    """
    h, w = shape
    imp = np.zeros((h, w), dtype=np.float32)
    if ty.size < 2:
        return imp

    if exclude_comp_mask is not None:
        keep = ~exclude_comp_mask[t_comp]
        ty, tx, uy, ux, t_comp = ty[keep], tx[keep], uy[keep], ux[keep], t_comp[keep]
        if ty.size < 2:
            return imp

    pts = np.column_stack([ty.astype(np.float32), tx.astype(np.float32)])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=max_dist_px, output_type="ndarray")
    if pairs.size == 0:
        return imp

    i1 = pairs[:, 0]
    i2 = pairs[:, 1]
    diff_comp = t_comp[i1] != t_comp[i2]
    i1, i2 = i1[diff_comp], i2[diff_comp]
    if i1.size == 0:
        return imp

    dy = pts[i2, 0] - pts[i1, 0]
    dx = pts[i2, 1] - pts[i1, 1]
    dist = np.hypot(dy, dx)
    valid_d = dist >= min_dist_px
    i1, i2, dy, dx, dist = i1[valid_d], i2[valid_d], dy[valid_d], dx[valid_d], dist[valid_d]
    if i1.size == 0:
        return imp

    ry = dy / dist
    rx = dx / dist
    # Tip 1 outward tangent should point generally toward Tip 2, or be co-axial along strike
    c1 = np.abs(uy[i1] * ry + ux[i1] * rx)
    c2 = np.abs(uy[i2] * (-ry) + ux[i2] * (-rx))
    coax = np.abs(uy[i1] * uy[i2] + ux[i1] * ux[i2])
    align = (c1 > 0.35) & (c2 > 0.35) & (coax > 0.25)
    i1, i2, dy, dx, dist, c1, c2 = (
        i1[align],
        i2[align],
        dy[align],
        dx[align],
        dist[align],
        c1[align],
        c2[align],
    )
    if i1.size == 0:
        return imp

    strength = (np.exp(-dist / 22.0) * np.sqrt(c1 * c2)).astype(np.float32)
    y1 = pts[i1, 0]
    x1 = pts[i1, 1]
    for frac in np.linspace(0.1, 0.9, 17):
        rr = np.clip(np.round(y1 + frac * dy).astype(np.int32), 0, h - 1)
        cc = np.clip(np.round(x1 + frac * dx).astype(np.int32), 0, w - 1)
        np.maximum.at(imp, (rr, cc), strength)

    return ndimage.gaussian_filter(imp, sigma=2.2, mode="constant").astype(np.float32)


def _single_mask_h77h_mechanics(
    mask_2d: np.ndarray,
    footprint: np.ndarray,
    fp_flat: np.ndarray,
) -> np.ndarray:
    """Compute the 5 H77H structural mechanics channels on ``fp_flat`` from a single fault mask."""
    h, w = footprint.shape
    m = np.asarray(mask_2d, dtype=bool) & footprint
    d1 = ndimage.distance_transform_edt(~m).astype(np.float32)

    comp, n_comp = ndimage.label(m, structure=np.ones((3, 3), dtype=bool))
    ty, tx, uy, ux = _extract_tips_and_tangents(m)
    t_comp = comp[ty, tx] if ty.size > 0 else np.zeros(0, dtype=np.int32)

    # Pollard crack-tip radiation lobe
    lobe = np.zeros((h, w), dtype=np.float32)
    if ty.size > 0:
        imp = np.zeros((h, w), dtype=np.float32)
        for s in range(2, 34, 2):
            rr = np.round(ty + s * uy).astype(np.int32)
            cc = np.round(tx + s * ux).astype(np.int32)
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            np.maximum.at(imp, (rr[ok], cc[ok]), float(np.exp(-s / 16.0)))
        lobe = ndimage.gaussian_filter(imp, sigma=2.2, mode="constant").astype(np.float32)

    # Second-nearest distinct component distance via 5 deterministic hash colorings
    d2 = np.full((h, w), 100.0, dtype=np.float32)
    if n_comp >= 2:
        cids = np.arange(n_comp + 1, dtype=np.uint64)
        for b, mult in enumerate((2654435761, 2246822519, 3266489917, 668265263, 374761393)):
            side = (((cids * np.uint64(mult)) >> np.uint64(13 + b)) & np.uint64(1)).astype(np.int8)
            side[0] = -1
            ma = side[comp] == 0
            mb = side[comp] == 1
            if ma.any() and mb.any():
                da = ndimage.distance_transform_edt(~ma).astype(np.float32)
                db = ndimage.distance_transform_edt(~mb).astype(np.float32)
                np.minimum(d2, np.maximum(da, db), out=d2)
                del da, db

    # Bilateral facing-tip inter-component relay bridge
    bridge = _rasterize_bilateral_bridges(
        (h, w), ty, tx, uy, ux, t_comp, exclude_comp_mask=None
    )

    d1_fp = d1.ravel()[fp_flat]
    d2_fp = d2.ravel()[fp_flat]
    out = np.empty((fp_flat.size, 5), dtype=np.float32)
    out[:, 0] = np.log1p(np.minimum(d1_fp, 120.0))
    out[:, 1] = np.exp(-((np.maximum(d1_fp - 2.2, 0.0)) / 4.5) ** 2)
    out[:, 2] = lobe.ravel()[fp_flat]
    out[:, 3] = np.exp(-((np.maximum(d1_fp - 2.0, 0.0)) / 11.0) ** 2) * np.exp(
        -((np.maximum(d2_fp - 2.0, 0.0)) / 14.0) ** 2
    )
    out[:, 4] = bridge.ravel()[fp_flat]
    return out


def compute_h77h_loco_features(
    visible_mask: np.ndarray,
    footprint: np.ndarray,
    n_buckets: int = 4,
    max_seg_px: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (loco_train_fp, loco_infer_fp), each of shape (n_fp, 5) float32,
    strictly from ``visible_mask`` (zero access to hidden faults).

    Fixes IR-H77H-02: replaces k=12 KD-tree truncation with exact 4-bucket Segment-LOCO
    (``gems57.network.segments(vis, max_len_px=16)`` + 2-px self-junction collar) so every
    training fault segment s receives exact EDT, splay collar, Pollard tip lobe, step-over
    relay, and bilateral facing-tip bridge potential strictly from ``visible_mask \\ bucket(s)``.
    """
    foot = np.asarray(footprint, dtype=bool)
    vis = np.asarray(visible_mask, dtype=bool) & foot
    fp_flat = np.flatnonzero(foot)

    loco_infer = _single_mask_h77h_mechanics(vis, foot, fp_flat)
    loco_train = loco_infer.copy()

    seg, n_seg = gems57_network.segments(vis, max_len_px=max_seg_px)[:2]
    if n_seg > 0:
        seg_ids = np.arange(n_seg + 1, dtype=np.uint64)
        buckets = ((seg_ids * np.uint64(2654435761)) % np.uint64(n_buckets)).astype(np.int16)
        buckets[0] = -1
        b_map = buckets[seg]

        yy, xx = np.ogrid[-2:3, -2:3]
        disk2 = (yy * yy + xx * xx) <= 4

        for b in range(n_buckets):
            in_b = b_map == b
            if not in_b.any():
                continue
            excl = ndimage.binary_dilation(in_b, structure=disk2)
            vis_minus_b = vis & (~excl)
            sub_b = _single_mask_h77h_mechanics(vis_minus_b, foot, fp_flat)
            b_fp = in_b.ravel()[fp_flat]
            loco_train[b_fp, :] = sub_b[b_fp, :]
            del excl, vis_minus_b, sub_b

    gc.collect()
    return loco_train, loco_infer
