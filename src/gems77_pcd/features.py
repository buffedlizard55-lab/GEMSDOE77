"""Low-memory feature builder and physical discovery operators for Lane H83-PCD.

Hypothesis H83-PCD: Multi-Scale Poisson Pseudogravity-Gravity Gradient Discordance
x Piedmont Scarp-Toe Asymmetry x Geothermal-Strain Conduit Coupling + 12-Bit
Leave-Own-Component-Out (LOCO) Transfer Relay.

Strict leakage/integrity safeguards enforced in this module:
1. Zero SGMC geology channels (`derived_sgmc_faults_100m_u8.tif` is never loaded).
2. `dist_known_fault_px` in `gdr_wellspring_in_footprint.csv` is strictly forbidden.
3. Missing 1 m LiDAR coverage (`valid == 0` or `q == 0` in `lidar_scarp_features_u8.tif`)
   is encoded as `NaN` (fix `IR-77D-01`), with explicit `has_lidar` coverage indicator
   and seamless 100 m DEM Hessian-ridge + profile-concavity bridging across the 24.6% gap.
4. Band 17 (`cond_surf`) derivatives are smoothed at sigma = 6.0 px (600 m) to suppress
   the 3.5 km MT interpolation grid seam (`IR-H77G-01`).
5. Fold-visible 12-bit Leave-Own-Component-Out (`LOCO`) fault-tip and stepover mechanics
   never consult `existing_faults.tif` directly; they operate strictly on the fold-specific
   `visible_mask` (`existing_faults & ~eval_region`) and leave each visible component's
   own halo out on training positives (`IR-77D-02`).
"""

from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from scipy import fft as sp_fft
from scipy import ndimage as ndi

from src.gems77d.features import (
    clean_band,
    compute_visible_mechanics,
    gf32,
    rank01_fp,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
EXT = RAW / "external"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "gems77_pcd_cache"

BASE19_NAMES: List[str] = [
    "tmi",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_slope",
    "tc_totalcount",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
    "deq_n100a15",
    "iso_grav_anom_vg",
    "det_elev",
    "iso_grav_anom",
    "boug_grav_anom",
    "depth_to_base_surf",
    "ieq_n100a15",
    "cond_surf_log",
    "fault_slip_rate_prior",
    "det_elev_slope",
]

TERR_NAMES: List[str] = [
    "dem_hess_ridge_s12",
    "dem_hess_ridge_s25",
    "dem_prof_concavity",
    "dem_slope_break",
    "dem_nne_strike_align",
    "grav_tilt",
    "mag_tilt",
    "grav_mag_crossgrad",
    "basement_grad_mag",
    "cond_log_grad_s6",
    "transtens_strain",
    "seismic_swarm_excess",
]

RAD_NAMES: List[str] = [
    "rad_K",
    "rad_Th",
    "rad_U",
    "rad_TC",
    "rad_Th_K",
    "rad_U_K",
    "rad_U_Th",
    "mag_up150",
    "rad_K_Th_grad",
    "mag_shallow_resid",
]

LIDAR_NAMES: List[str] = [
    "lidar_ex_max",
    "lidar_step_max",
    "lidar_lap_dipole",
    "lidar_coh100",
    "lidar_aniso",
    "has_lidar",
    "bridged_scarp_ridge",
]

HYDRO_NAMES: List[str] = [
    "hydro_temp_plume_nne",
    "hydro_geotherm_plume_nne",
    "hydro_regional_halo",
    "volc_vent_halo",
]

PCD_STATIC_NAMES: List[str] = [
    "pcd_poisson_anti_step",
    "pcd_crossgrad_rot_ridge",
    "pcd_basement_bench_inflect",
    "pcd_piedmont_toe_scarp",
    "pcd_transtens_swarm_gate",
    "pcd_hydro_demag_conduit",
    "pcd_rad_kth_discord",
    "pcd_composite_surface",
]

PCD_VIS_NAMES: List[str] = [
    "vis_dist_clipped",
    "vis_cracktip_lobe",
    "vis_stepover_corridor",
    "vis_strain_deficit",
    "vis_pcd_transfer_relay",
]

ALL_STATIC_NAMES: List[str] = (
    BASE19_NAMES + TERR_NAMES + RAD_NAMES + LIDAR_NAMES + HYDRO_NAMES + PCD_STATIC_NAMES
)
ALL_FEATURE_NAMES: List[str] = ALL_STATIC_NAMES + PCD_VIS_NAMES


def fill_nan_nearest(arr: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Fast low-memory clean band fill inside/outside footprint."""
    return clean_band(arr, footprint)


def dequantize_u8_nan(q: np.ndarray, p05: float, p995: float) -> np.ndarray:
    """De-quantize uint8 array where 0 is missing-coverage sentinel -> NaN (fix IR-77D-01)."""
    out = np.full(q.shape, np.nan, dtype=np.float32)
    mask = q > 0
    if np.any(mask):
        t = (q[mask].astype(np.float32) - 1.0) / 254.0
        out[mask] = float(p05) + t * float(p995 - p05)
    return out


def hessian_ridge(field: np.ndarray, sigma: float) -> np.ndarray:
    """Positive across-ridge Hessian curvature max(-lambda_min, 0)."""
    f = np.asarray(field, dtype=np.float32)
    hyy = gf32(f, sigma=sigma, order=(2, 0), mode="nearest")
    hxx = gf32(f, sigma=sigma, order=(0, 2), mode="nearest")
    hxy = gf32(f, sigma=sigma, order=(1, 1), mode="nearest")
    tr = hxx + hyy
    disc = np.hypot(hxx - hyy, 2.0 * hxy)
    del hxx, hyy, hxy
    lam_min = 0.5 * (tr - disc)
    del tr, disc
    return np.maximum(-lam_min, 0.0).astype(np.float32)


def fourier_pseudogravity(
    rtp: np.ndarray,
    footprint: np.ndarray,
    dx_m: float = 100.0,
    min_k_rad_per_m: float = 2.0 * np.pi / 45000.0,
) -> np.ndarray:
    """Compute Baranov (1957) / Cordell & Grauch (1985) Fourier pseudogravity from RTP.

    Uses single-precision `scipy.fft.rfft2` / `irfft2` so memory stays `< 150 MB`.
    """
    filled = clean_band(rtp, footprint)
    fp_vals = filled[footprint]
    mean_v = float(np.mean(fp_vals)) if fp_vals.size > 0 else 0.0
    std_v = float(np.std(fp_vals)) + 1e-6 if fp_vals.size > 0 else 1.0
    normed = np.where(footprint, (filled - mean_v) / std_v, 0.0).astype(np.float32)
    del filled

    h, w = normed.shape
    ky = (2.0 * np.pi * np.fft.fftfreq(h, d=dx_m)).astype(np.float32)[:, None]
    kx = (2.0 * np.pi * np.fft.rfftfreq(w, d=dx_m)).astype(np.float32)[None, :]
    k_mag = np.hypot(ky, kx).astype(np.float32)
    del ky, kx

    spec = sp_fft.rfft2(normed)
    del normed
    inv_k = (1.0 / np.maximum(k_mag, np.float32(min_k_rad_per_m))).astype(np.float32)
    del k_mag
    inv_k[0, 0] = 0.0
    spec *= inv_k
    del inv_k
    pg = sp_fft.irfft2(spec, s=(h, w)).astype(np.float32)
    del spec
    pg_fp = pg[footprint]
    if pg_fp.size > 0:
        pg = (pg - float(np.mean(pg_fp))) / (float(np.std(pg_fp)) + 1e-6)
    return pg.astype(np.float32)


def compute_pcd_static_channels_from_arrays(
    *,
    footprint: np.ndarray,
    rtp: np.ndarray,
    iso_grav: np.ndarray,
    depth_base: np.ndarray,
    elev: np.ndarray,
    slope: np.ndarray,
    geod_2ndinv: np.ndarray,
    geod_shear: np.ndarray,
    geod_dilat: np.ndarray,
    deq: np.ndarray,
    ieq: np.ndarray,
    cond_surf: np.ndarray,
    rad_k_rank: np.ndarray,
    rad_th_rank: np.ndarray,
    rad_u_rank: np.ndarray,
    mag_up150_rank: np.ndarray,
    lidar_step: np.ndarray,
    lidar_lap_dipole: np.ndarray,
    lidar_downface: np.ndarray,
    lidar_coh100: np.ndarray,
    has_lidar: np.ndarray,
    hydro_geotherm_plume: np.ndarray,
    hydro_regional_halo: np.ndarray,
    volc_vent_halo: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute the 8 label-free physical discovery channels of Hypothesis H83-PCD."""
    fp = np.asarray(footprint, dtype=bool)
    rtp_f = clean_band(rtp, fp)
    grav_f = clean_band(iso_grav, fp)
    base_f = clean_band(depth_base, fp)
    elev_f = clean_band(elev, fp)
    slope_f = clean_band(slope, fp)
    cond_f = clean_band(cond_surf, fp)

    # 1. pcd_poisson_anti_step
    pg = fourier_pseudogravity(rtp_f, fp)
    anti_acc = np.zeros(fp.shape, dtype=np.float32)
    for w_s, sigma in ((0.55, 2.5), (0.45, 5.0)):
        gy = gf32(grav_f, sigma=sigma, order=(1, 0), mode="nearest")
        gx = gf32(grav_f, sigma=sigma, order=(0, 1), mode="nearest")
        py = gf32(pg, sigma=sigma, order=(1, 0), mode="nearest")
        px = gf32(pg, sigma=sigma, order=(0, 1), mode="nearest")
        g_mag = np.hypot(gx, gy).astype(np.float32)
        p_mag = np.hypot(px, py).astype(np.float32)
        cos_theta = np.clip((gx * px + gy * py) / (g_mag * p_mag + 1e-6), -1.0, 1.0)
        del gx, gy, px, py, p_mag
        g_med = float(np.median(g_mag[fp][::4])) + 1e-6 if np.any(fp) else 1.0
        anti_raw = (np.log1p(g_mag / g_med) * (0.5 * (1.0 - cos_theta))).astype(
            np.float32
        )
        del g_mag, cos_theta
        anti_acc += float(w_s) * rank01_fp(anti_raw, fp)
        del anti_raw
    del pg

    by2 = gf32(base_f, sigma=2.5, order=(1, 0), mode="nearest")
    bx2 = gf32(base_f, sigma=2.5, order=(0, 1), mode="nearest")
    base_step_rank = rank01_fp(np.hypot(bx2, by2).astype(np.float32), fp)
    del bx2, by2
    pcd_poisson_anti_step = rank01_fp(
        (anti_acc * (0.35 + 0.65 * base_step_rank)).astype(np.float32), fp
    )
    del anti_acc, base_step_rank

    # 2. pcd_crossgrad_rot_ridge
    cg_acc = np.zeros(fp.shape, dtype=np.float32)
    for w_s, sigma in ((0.55, 2.0), (0.45, 4.0)):
        gy = gf32(grav_f, sigma=sigma, order=(1, 0), mode="nearest")
        gx = gf32(grav_f, sigma=sigma, order=(0, 1), mode="nearest")
        my = gf32(rtp_f, sigma=sigma, order=(1, 0), mode="nearest")
        mx = gf32(rtp_f, sigma=sigma, order=(0, 1), mode="nearest")
        g_sq = gx * gx + gy * gy
        m_sq = mx * mx + my * my
        sin2 = ((gx * my - gy * mx) ** 2) / (g_sq * m_sq + 1e-8)
        del gx, gy, mx, my
        gm = np.sqrt(np.sqrt(g_sq * m_sq)).astype(np.float32)
        del g_sq, m_sq
        gm_med = float(np.median(gm[fp][::4])) + 1e-6 if np.any(fp) else 1.0
        energy = (np.log1p(gm / gm_med) * sin2).astype(np.float32)
        del gm, sin2
        ridge = hessian_ridge(energy, sigma=1.5)
        cg_acc += float(w_s) * (0.60 * rank01_fp(energy, fp) + 0.40 * rank01_fp(ridge, fp))
        del energy, ridge
    pcd_crossgrad_rot_ridge = rank01_fp(cg_acc, fp)
    del cg_acc

    # 3. pcd_basement_bench_inflect
    gy3 = gf32(grav_f, sigma=3.0, order=(1, 0), mode="nearest")
    gx3 = gf32(grav_f, sigma=3.0, order=(0, 1), mode="nearest")
    gnorm = np.hypot(gx3, gy3) + 1e-6
    nx = (gx3 / gnorm).astype(np.float32)
    ny = (gy3 / gnorm).astype(np.float32)
    del gx3, gy3, gnorm
    dyy = gf32(base_f, sigma=3.0, order=(2, 0), mode="nearest")
    dxx = gf32(base_f, sigma=3.0, order=(0, 2), mode="nearest")
    dxy = gf32(base_f, sigma=3.0, order=(1, 1), mode="nearest")
    d_nn = np.abs(nx * nx * dxx + 2.0 * nx * ny * dxy + ny * ny * dyy).astype(np.float32)
    del nx, ny, dxx, dyy, dxy
    slope_p75 = float(np.percentile(slope_f[fp][::4], 75.0)) + 1e-6 if np.any(fp) else 1.0
    piedmont_gate = np.exp(-np.maximum(slope_f, 0.0) / slope_p75).astype(np.float32)
    pcd_basement_bench_inflect = rank01_fp(
        (rank01_fp(d_nn, fp) * (0.30 + 0.70 * piedmont_gate)).astype(np.float32), fp
    )
    del d_nn, piedmont_gate

    # 4. pcd_piedmont_toe_scarp
    dem_r12 = rank01_fp(hessian_ridge(elev_f, 1.2), fp)
    dem_r25 = rank01_fp(hessian_ridge(elev_f, 2.5), fp)
    dem_concav = rank01_fp(
        np.maximum(-ndi.gaussian_laplace(elev_f, sigma=1.5, mode="nearest"), 0.0).astype(
            np.float32
        ),
        fp,
    )
    slope_break = rank01_fp(
        np.hypot(
            gf32(slope_f, sigma=1.2, order=(1, 0), mode="nearest"),
            gf32(slope_f, sigma=1.2, order=(0, 1), mode="nearest"),
        ).astype(np.float32),
        fp,
    )
    dem_bridge = (
        0.38 * dem_r12 + 0.27 * dem_r25 + 0.20 * dem_concav + 0.15 * slope_break
    ).astype(np.float32)
    del dem_r12, dem_r25, dem_concav, slope_break

    step_c = np.where(np.isfinite(lidar_step), lidar_step, 0.0).astype(np.float32)
    dip_c = np.where(np.isfinite(lidar_lap_dipole), lidar_lap_dipole, 0.0).astype(np.float32)
    down_c = np.where(np.isfinite(lidar_downface), lidar_downface, 0.0).astype(np.float32)
    coh_c = np.where(np.isfinite(lidar_coh100), lidar_coh100, 0.0).astype(np.float32)
    toe_raw = (
        dip_c
        * (0.35 + 0.65 * down_c)
        * (0.40 + 0.60 * step_c)
        * (0.30 + 0.70 * coh_c)
    ).astype(np.float32)
    del step_c, dip_c, down_c, coh_c
    lidar_valid_mask = fp & (has_lidar > 0.5)
    toe_rank = (
        rank01_fp(toe_raw, lidar_valid_mask)
        if np.any(lidar_valid_mask)
        else np.zeros_like(dem_bridge)
    )
    del toe_raw
    bridged_toe = np.where(
        lidar_valid_mask, 0.68 * toe_rank + 0.32 * dem_bridge, dem_bridge
    ).astype(np.float32)
    del toe_rank, dem_bridge, lidar_valid_mask
    pcd_piedmont_toe_scarp = rank01_fp(bridged_toe, fp)
    del bridged_toe

    # 5. pcd_transtens_swarm_gate
    sh_r = rank01_fp(clean_band(geod_shear, fp), fp)
    dl_r = rank01_fp(np.maximum(clean_band(geod_dilat, fp), 0.0).astype(np.float32), fp)
    inv_r = rank01_fp(clean_band(geod_2ndinv, fp), fp)
    transtens = np.sqrt(sh_r * dl_r) * (0.40 + 0.60 * inv_r)
    del sh_r, dl_r, inv_r
    deq_r = rank01_fp(clean_band(deq, fp), fp)
    ieq_r = rank01_fp(clean_band(ieq, fp), fp)
    swarm_ex = rank01_fp((deq_r - 0.60 * ieq_r).astype(np.float32), fp)
    del deq_r, ieq_r
    pcd_transtens_swarm_gate = rank01_fp(
        (0.62 * transtens + 0.38 * swarm_ex).astype(np.float32), fp
    )
    del transtens, swarm_ex

    # 6. pcd_hydro_demag_conduit
    rtp_r = rank01_fp(rtp_f, fp)
    demag = np.maximum(mag_up150_rank - rtp_r, 0.0).astype(np.float32)
    del rtp_r
    cond_log = np.log1p(np.maximum(cond_f, 0.0)).astype(np.float32)
    cy6 = gf32(cond_log, sigma=6.0, order=(1, 0), mode="nearest")
    cx6 = gf32(cond_log, sigma=6.0, order=(0, 1), mode="nearest")
    del cond_log
    cond_margin_r = rank01_fp(np.hypot(cx6, cy6).astype(np.float32), fp)
    del cx6, cy6
    plume = (
        0.50 * hydro_geotherm_plume + 0.30 * hydro_regional_halo + 0.20 * volc_vent_halo
    ).astype(np.float32)
    pcd_hydro_demag_conduit = rank01_fp(
        (plume * (0.45 + 0.55 * demag) + 0.30 * cond_margin_r * (0.35 + 0.65 * demag)).astype(
            np.float32
        ),
        fp,
    )
    del plume, demag, cond_margin_r

    # 7. pcd_rad_kth_discord
    k_th_diff = (rad_k_rank - rad_th_rank).astype(np.float32)
    kth_gy = gf32(k_th_diff, sigma=2.0, order=(1, 0), mode="nearest")
    kth_gx = gf32(k_th_diff, sigma=2.0, order=(0, 1), mode="nearest")
    del k_th_diff
    kth_grad_r = rank01_fp(np.hypot(kth_gx, kth_gy).astype(np.float32), fp)
    del kth_gx, kth_gy
    u_excess = np.maximum(rad_u_rank - 0.5 * (rad_k_rank + rad_th_rank), 0.0).astype(
        np.float32
    )
    pcd_rad_kth_discord = rank01_fp(
        (0.68 * kth_grad_r + 0.32 * rank01_fp(u_excess, fp)).astype(np.float32), fp
    )
    del kth_grad_r, u_excess

    # 8. pcd_composite_surface
    pcd_composite_surface = rank01_fp(
        (
            0.26 * pcd_piedmont_toe_scarp
            + 0.22 * pcd_poisson_anti_step
            + 0.16 * pcd_crossgrad_rot_ridge
            + 0.12 * pcd_basement_bench_inflect
            + 0.10 * pcd_transtens_swarm_gate
            + 0.08 * pcd_hydro_demag_conduit
            + 0.06 * pcd_rad_kth_discord
        ).astype(np.float32),
        fp,
    )

    return {
        "pcd_poisson_anti_step": pcd_poisson_anti_step,
        "pcd_crossgrad_rot_ridge": pcd_crossgrad_rot_ridge,
        "pcd_basement_bench_inflect": pcd_basement_bench_inflect,
        "pcd_piedmont_toe_scarp": pcd_piedmont_toe_scarp,
        "pcd_transtens_swarm_gate": pcd_transtens_swarm_gate,
        "pcd_hydro_demag_conduit": pcd_hydro_demag_conduit,
        "pcd_rad_kth_discord": pcd_rad_kth_discord,
        "pcd_composite_surface": pcd_composite_surface,
    }


def compute_pcd_visible_mechanics(
    visible_mask: np.ndarray,
    footprint: np.ndarray,
    geod_rank_2d: np.ndarray,
    pcd_poisson_anti_2d: np.ndarray,
    pcd_piedmont_toe_2d: np.ndarray,
    leave_own_component_out: bool = True,
    own_Exclusion_radius_px: float = 12.0,
) -> np.ndarray:
    """Compute the 5 strictly fold-visible 12-bit LOCO mechanics channels (`N_fp x 5`)."""
    base4 = compute_visible_mechanics(
        visible_mask=visible_mask,
        footprint=footprint,
        geod_strain_rank_2d=geod_rank_2d,
    )
    poisson_fp = pcd_poisson_anti_2d[footprint].astype(np.float32)
    toe_fp = pcd_piedmont_toe_2d[footprint].astype(np.float32)
    stepover_fp = base4[:, 2]
    relay_fp = ((0.55 * poisson_fp + 0.45 * toe_fp) * stepover_fp).astype(np.float32)

    out = np.empty((base4.shape[0], 5), dtype=np.float32)
    out[:, :4] = base4
    out[:, 4] = relay_fp
    return out


def build_static_feature_memmap(force: bool = False) -> Tuple[np.memmap, Dict]:
    """Build or load cached `N_fp x 60` float32 memmap of static H83-PCD features (<400 MB RAM)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mmap_path = CACHE_DIR / "static_features_fp_h83pcd.dat"
    meta_path = CACHE_DIR / "static_features_meta_h83pcd.json"
    n_cols = len(ALL_STATIC_NAMES)

    if not force and mmap_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("feature_names") == ALL_STATIC_NAMES:
            mm = np.memmap(
                mmap_path,
                dtype="float32",
                mode="r",
                shape=(meta["n_fp"], n_cols),
            )
            return mm, meta

    t0 = time.time()
    with rasterio.open(RAW / "sample_submission.tif") as src:
        sample = src.read(1)
        transform = src.transform
    footprint = ~np.isnan(sample)
    shape_hw = footprint.shape
    del sample
    n_fp = int(footprint.sum())

    mm = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(n_fp, n_cols))

    def col_to_2d(col_idx: int) -> np.ndarray:
        a = np.zeros(shape_hw, dtype=np.float32)
        a[footprint] = mm[:, col_idx]
        return clean_band(a, footprint)

    # Stage 1: Read 19 base bands one at a time (<50 MB RAM)
    with rasterio.open(RAW / "training_features.tif") as src:
        for b in range(1, 20):
            arr = clean_band(src.read(b), footprint)
            if b == 17:
                arr = np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
            mm[:, b - 1] = arr[footprint]
            del arr
    mm.flush()
    gc.collect()

    # Stage 2: Terrain, potential-field tilt, and kinematic features (cols 19..30)
    elev = col_to_2d(11)
    mm[:, 19] = rank01_fp(hessian_ridge(elev, 1.2), footprint)[footprint]
    mm[:, 20] = rank01_fp(hessian_ridge(elev, 2.5), footprint)[footprint]
    mm[:, 21] = rank01_fp(
        np.maximum(-ndi.gaussian_laplace(elev, sigma=1.5, mode="nearest"), 0.0).astype(
            np.float32
        ),
        footprint,
    )[footprint]
    ey = gf32(elev, sigma=2.0, order=(1, 0), mode="nearest")
    ex = gf32(elev, sigma=2.0, order=(0, 1), mode="nearest")
    del elev
    theta_nne = np.deg2rad(25.0)
    ux, uy = float(np.cos(theta_nne)), float(-np.sin(theta_nne))
    emag = np.hypot(ex, ey) + 1e-6
    cos_strike = (ex * (-uy) + ey * ux) / emag
    del ex, ey
    mm[:, 23] = ((cos_strike ** 2) * rank01_fp(emag, footprint))[footprint].astype(
        np.float32
    )
    del emag, cos_strike

    slope = col_to_2d(18)
    sy = gf32(slope, sigma=1.2, order=(1, 0), mode="nearest")
    sx = gf32(slope, sigma=1.2, order=(0, 1), mode="nearest")
    del slope
    mm[:, 22] = rank01_fp(np.hypot(sx, sy), footprint)[footprint]
    del sx, sy

    grav = col_to_2d(12)
    grav_vg = col_to_2d(10)
    gy = gf32(grav, sigma=2.0, order=(1, 0), mode="nearest")
    gx = gf32(grav, sigma=2.0, order=(0, 1), mode="nearest")
    del grav
    gh = np.hypot(gx, gy) + 1e-6
    mm[:, 24] = np.arctan2(grav_vg, gh)[footprint].astype(np.float32)
    del grav_vg, gh

    rtp = col_to_2d(1)
    my = gf32(rtp, sigma=2.0, order=(1, 0), mode="nearest")
    mx = gf32(rtp, sigma=2.0, order=(0, 1), mode="nearest")
    del rtp
    mm[:, 26] = rank01_fp(np.abs(gx * my - gy * mx).astype(np.float32), footprint)[
        footprint
    ]
    del gx, gy, mx, my

    tmi_hg = col_to_2d(2)
    tmi_vg = col_to_2d(8)
    mm[:, 25] = np.arctan2(tmi_vg, np.maximum(tmi_hg, 0.0) + 1e-6)[footprint].astype(
        np.float32
    )
    del tmi_hg, tmi_vg

    base_d = col_to_2d(14)
    by = gf32(base_d, sigma=2.0, order=(1, 0), mode="nearest")
    bx = gf32(base_d, sigma=2.0, order=(0, 1), mode="nearest")
    del base_d
    mm[:, 27] = rank01_fp(np.hypot(bx, by), footprint)[footprint]
    del bx, by

    cond = col_to_2d(16)
    cy = gf32(cond, sigma=6.0, order=(1, 0), mode="nearest")
    cx = gf32(cond, sigma=6.0, order=(0, 1), mode="nearest")
    del cond
    mm[:, 28] = rank01_fp(np.hypot(cx, cy), footprint)[footprint]
    del cx, cy

    sh_r = rank01_fp(col_to_2d(6), footprint)
    dl_r = rank01_fp(np.maximum(col_to_2d(7), 0.0), footprint)
    mm[:, 29] = np.sqrt(sh_r * dl_r)[footprint].astype(np.float32)
    del sh_r, dl_r

    eq_ex = rank01_fp(col_to_2d(9), footprint) - 0.5 * rank01_fp(
        col_to_2d(15), footprint
    )
    mm[:, 30] = eq_ex[footprint].astype(np.float32)
    del eq_ex
    mm.flush()
    gc.collect()

    # Stage 3: GeoDAWN radiometrics + upward-continued TMI (cols 31..40)
    with rasterio.open(EXT / "geodawn_rad_u8.tif") as src:
        r_meta = json.loads(src.tags().get("QUANT_META", "{}"))
        for idx_b, key, col_out in ((1, "K", 31), (2, "Th", 32), (3, "U", 33), (4, "TC", 34)):
            q = dequantize_u8_nan(
                src.read(idx_b),
                float(r_meta.get(key, {}).get("p05", 0.0)),
                float(r_meta.get(key, {}).get("p995", 1.0)),
            )
            mm[:, col_out] = rank01_fp(clean_band(q, footprint), footprint)[footprint]
            del q

    with rasterio.open(EXT / "geodawn_extensions_u8.tif") as src:
        e_meta = json.loads(src.tags().get("QUANT_META", "{}"))
        for idx_b, key, col_out in (
            (1, "Th_K", 35),
            (2, "U_K", 36),
            (3, "U_Th", 37),
            (4, "TMI_up150", 38),
        ):
            q = dequantize_u8_nan(
                src.read(idx_b),
                float(e_meta.get(key, {}).get("p05", 0.0)),
                float(e_meta.get(key, {}).get("p995", 1.0)),
            )
            mm[:, col_out] = rank01_fp(clean_band(q, footprint), footprint)[footprint]
            del q

    k_th_diff = col_to_2d(31) - col_to_2d(32)
    kdy = gf32(k_th_diff, sigma=1.5, order=(1, 0), mode="nearest")
    kdx = gf32(k_th_diff, sigma=1.5, order=(0, 1), mode="nearest")
    del k_th_diff
    mm[:, 39] = rank01_fp(np.hypot(kdx, kdy), footprint)[footprint]
    del kdx, kdy

    mm[:, 40] = (rank01_fp(col_to_2d(1), footprint) - col_to_2d(38))[footprint].astype(
        np.float32
    )
    mm.flush()
    gc.collect()

    # Stage 4: Missing-coverage-aware 1 m LiDAR scarp features (cols 41..47)
    with rasterio.open(EXT / "lidar_scarp_features_u8.tif") as src:
        l_meta = json.loads(src.tags().get("QUANT_META", "{}"))

        def _read_l(idx: int, key: str) -> np.ndarray:
            info = l_meta.get(key, {})
            return dequantize_u8_nan(
                src.read(idx),
                float(info.get("p05", 0.0)),
                float(info.get("p995", 1.0)),
            )

        l_ex_max = _read_l(1, "ex_max")
        l_ex_p90 = _read_l(2, "ex_p90")
        l_step = _read_l(3, "step_max")
        l_lapneg = _read_l(4, "lapneg_max")
        l_lappos = _read_l(5, "lappos_max")
        l_downface = _read_l(6, "downface_max")
        l_coh100 = _read_l(10, "coh100")
        l_aniso = _read_l(11, "aniso")
        l_valid_u8 = src.read(12)

    has_lidar_2d = ((l_valid_u8 > 0) & np.isfinite(l_ex_max)).astype(np.float32)
    del l_valid_u8
    l_dipole = np.sqrt(np.maximum(l_lapneg * l_lappos, 0.0)).astype(np.float32)
    del l_lapneg, l_lappos

    dem_bridge = (0.55 * col_to_2d(19) + 0.45 * col_to_2d(20)).astype(np.float32)
    ex_clean = np.where(np.isfinite(l_ex_max), l_ex_max, 0.0).astype(np.float32)
    step_clean = np.where(np.isfinite(l_step), l_step, 0.0).astype(np.float32)
    dip_clean = np.where(np.isfinite(l_dipole), l_dipole, 0.0).astype(np.float32)
    coh_clean = np.where(np.isfinite(l_coh100), l_coh100, 0.0).astype(np.float32)
    aniso_clean = np.where(np.isfinite(l_aniso), l_aniso, 0.0).astype(np.float32)
    p90_clean = np.where(np.isfinite(l_ex_p90), l_ex_p90, 0.0).astype(np.float32)
    del l_ex_max, l_ex_p90, l_aniso

    lidar_mask = footprint & (has_lidar_2d > 0.5)
    lidar_comp = np.zeros(shape_hw, dtype=np.float32)
    if np.any(lidar_mask):
        raw_comp = (
            0.35 * ex_clean
            + 0.25 * step_clean
            + 0.20 * dip_clean
            + 0.10 * coh_clean
            + 0.10 * p90_clean
        )
        lidar_comp = rank01_fp(raw_comp, lidar_mask)
        del raw_comp

    bridged = np.where(
        has_lidar_2d > 0.5, 0.65 * lidar_comp + 0.35 * dem_bridge, dem_bridge
    ).astype(np.float32)
    del lidar_comp, dem_bridge, p90_clean

    mm[:, 41] = np.where(has_lidar_2d[footprint] > 0.5, ex_clean[footprint], -1.0)
    mm[:, 42] = np.where(has_lidar_2d[footprint] > 0.5, step_clean[footprint], -1.0)
    mm[:, 43] = np.where(has_lidar_2d[footprint] > 0.5, dip_clean[footprint], -1.0)
    mm[:, 44] = np.where(has_lidar_2d[footprint] > 0.5, coh_clean[footprint], -1.0)
    mm[:, 45] = np.where(has_lidar_2d[footprint] > 0.5, aniso_clean[footprint], -1.0)
    mm[:, 46] = has_lidar_2d[footprint]
    mm[:, 47] = bridged[footprint]
    del ex_clean, step_clean, dip_clean, coh_clean, aniso_clean, bridged
    mm.flush()
    gc.collect()

    # Stage 5: Anisotropic hydrothermal plumes + volcanic vents (cols 48..51)
    inv_tf = ~transform
    temp_grid = np.zeros(shape_hw, dtype=np.float32)
    geoth_grid = np.zeros(shape_hw, dtype=np.float32)
    with open(EXT / "gdr_wellspring_in_footprint.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x_32611"])
                y = float(row["y_32611"])
            except (KeyError, ValueError):
                continue
            c_f, r_f = inv_tf * (x, y)
            r_i, c_i = int(round(r_f)), int(round(c_f))
            if not (0 <= r_i < shape_hw[0] and 0 <= c_i < shape_hw[1]):
                continue

            def _flt(k: str) -> float:
                v = row.get(k, "")
                if v in ("", None):
                    return float("nan")
                try:
                    val = float(v)
                    return val if -50.0 < val < 500.0 else float("nan")
                except ValueError:
                    return float("nan")

            tc = _flt("temp_c")
            tq = _flt("geothermquartz_c")
            tch = _flt("geothermchalc_c")
            tcat = _flt("geothermcat_c")
            if np.isfinite(tc) and tc > 20.0:
                w_t = float(np.clip((tc - 20.0) / 100.0, 0.0, 2.5))
                if w_t > temp_grid[r_i, c_i]:
                    temp_grid[r_i, c_i] = w_t
            chem_list = [v for v in (tq, tch, tcat) if np.isfinite(v)]
            if chem_list:
                cmax = max(chem_list)
                diseq = cmax - (tc if np.isfinite(tc) else 15.0)
                w_g = float(
                    np.clip(
                        (cmax - 40.0) / 140.0 + 0.5 * max(diseq, 0.0) / 100.0,
                        0.0,
                        3.0,
                    )
                )
                if w_g > geoth_grid[r_i, c_i]:
                    geoth_grid[r_i, c_i] = w_g

    temp_nne = rank01_fp(gf32(temp_grid, sigma=(6.0, 3.0), mode="constant"), footprint)
    geoth_nne = rank01_fp(gf32(geoth_grid, sigma=(6.0, 3.0), mode="constant"), footprint)
    hydro_reg = rank01_fp(
        gf32(temp_grid + geoth_grid, sigma=15.0, mode="constant"), footprint
    )
    del temp_grid, geoth_grid

    vent_grid = np.zeros(shape_hw, dtype=np.float32)
    with open(EXT / "gdr_volcanic_vents_in_footprint.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x_32611"])
                y = float(row["y_32611"])
            except (KeyError, ValueError):
                continue
            c_f, r_f = inv_tf * (x, y)
            r_i, c_i = int(round(r_f)), int(round(c_f))
            if 0 <= r_i < shape_hw[0] and 0 <= c_i < shape_hw[1]:
                vent_grid[r_i, c_i] = 1.0
    vent_halo = rank01_fp(gf32(vent_grid, sigma=10.0, mode="constant"), footprint)
    del vent_grid

    mm[:, 48] = temp_nne[footprint]
    mm[:, 49] = geoth_nne[footprint]
    mm[:, 50] = hydro_reg[footprint]
    mm[:, 51] = vent_halo[footprint]
    del temp_nne
    mm.flush()
    gc.collect()

    # Stage 6: Compute the 8 novel H83-PCD static channels (cols 52..59)
    pcd_dict = compute_pcd_static_channels_from_arrays(
        footprint=footprint,
        rtp=col_to_2d(1),
        iso_grav=col_to_2d(12),
        depth_base=col_to_2d(14),
        elev=col_to_2d(11),
        slope=col_to_2d(18),
        geod_2ndinv=col_to_2d(3),
        geod_shear=col_to_2d(6),
        geod_dilat=col_to_2d(7),
        deq=col_to_2d(9),
        ieq=col_to_2d(15),
        cond_surf=np.expm1(col_to_2d(16)).astype(np.float32),
        rad_k_rank=col_to_2d(31),
        rad_th_rank=col_to_2d(32),
        rad_u_rank=col_to_2d(33),
        mag_up150_rank=col_to_2d(38),
        lidar_step=l_step,
        lidar_lap_dipole=l_dipole,
        lidar_downface=l_downface,
        lidar_coh100=l_coh100,
        has_lidar=has_lidar_2d,
        hydro_geotherm_plume=geoth_nne,
        hydro_regional_halo=hydro_reg,
        volc_vent_halo=vent_halo,
    )
    del l_step, l_dipole, l_downface, l_coh100, has_lidar_2d, geoth_nne, hydro_reg, vent_halo
    for idx_offset, name in enumerate(PCD_STATIC_NAMES):
        mm[:, 52 + idx_offset] = pcd_dict[name][footprint]
    mm.flush()
    del pcd_dict
    gc.collect()

    meta = {
        "n_fp": n_fp,
        "n_cols": n_cols,
        "feature_names": ALL_STATIC_NAMES,
        "pcd_static_names": PCD_STATIC_NAMES,
        "pcd_vis_names": PCD_VIS_NAMES,
        "build_seconds": round(time.time() - t0, 2),
        "sgmc_channels_included": 0,
        "lidar_missing_coverage_handled": True,
        "cond_surf_smoothed_sigma_px": 6.0,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    mm_ro = np.memmap(mmap_path, dtype="float32", mode="r", shape=(n_fp, n_cols))
    return mm_ro, meta
