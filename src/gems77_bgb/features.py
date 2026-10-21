"""H77-BGB feature engineering: 10 label-free physical channels across 3 sub-blocks.

Sub-block A (Buried Half-Graben Growth-Wedge Inflection):
  0: bgb_01_basement_step_along_grav
  1: bgb_02_growth_wedge_inflection
  2: bgb_03_blind_basin_basement_dipole
  3: bgb_04_grav_basement_concordance

Sub-block B (Kinematic Transtensional Strain-Partitioning Gradient):
  4: bgb_05_transtensional_mode_tau
  5: bgb_06_strain_partition_grad
  6: bgb_07_aseismic_dilation_excess

Sub-block C (Illite-Smectite Conductivity-Margin Upflow Breach):
  7: bgb_08_conductivity_margin_grad
  8: bgb_09_upflow_demag_breach
  9: bgb_10_geotherm_diseq_coupled

Strictly zero catalogue (`existing_faults.tif`), zero truth (`labels.tif`), zero SGMC
(`derived_sgmc_faults_100m_u8.tif`), and zero `dist_known_fault_px` inputs.
"""
from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import rankdata

from gems77 import features77 as F77

BGB_NAMES: tuple[str, ...] = (
    "bgb_01_basement_step_along_grav",
    "bgb_02_growth_wedge_inflection",
    "bgb_03_blind_basin_basement_dipole",
    "bgb_04_grav_basement_concordance",
    "bgb_05_transtensional_mode_tau",
    "bgb_06_strain_partition_grad",
    "bgb_07_aseismic_dilation_excess",
    "bgb_08_conductivity_margin_grad",
    "bgb_09_upflow_demag_breach",
    "bgb_10_geotherm_diseq_coupled",
)


def rank_in_footprint(arr: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """Map finite values inside `foot` to uniform percentiles in (0, 1]; 0.0 outside `foot`."""
    if arr.shape != foot.shape:
        raise ValueError(f"Shape mismatch: arr {arr.shape} vs foot {foot.shape}")
    out = np.zeros(arr.shape, dtype=np.float32)
    vals = arr[foot]
    finite = np.isfinite(vals)
    if not np.any(finite):
        return out
    r = np.zeros(vals.shape, dtype=np.float32)
    r[finite] = (rankdata(vals[finite], method="average") / float(finite.sum())).astype(np.float32)
    if not np.all(finite):
        r[~finite] = 0.5
    out[foot] = np.clip(r, 0.0, 1.0)
    return out


def compute_gravity_normal_basement_hessian(
    depth_base: np.ndarray,
    iso_grav: np.ndarray,
    sigma_depth: float = 2.0,
    sigma_grav: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the gradient and 2D Hessian of `depth_base` onto the unit horizontal
    gravity-gradient vector `n_g = grad(g) / |grad(g)|`.

    Returns:
      step_along_n: |grad(D) . n_g| (m / px)
      inflection_nn: |n_g^T (Hess(D)) n_g| (m / px^2)
      grad_g_mag: |grad(g)|
      lap_d_abs: |laplacian(D)|
    """
    d_s = ndimage.gaussian_filter(depth_base.astype(np.float32), sigma=sigma_depth, mode="nearest")
    g_s = ndimage.gaussian_filter(iso_grav.astype(np.float32), sigma=sigma_grav, mode="nearest")

    gy, gx = np.gradient(g_s)
    grad_g_mag = np.hypot(gx, gy).astype(np.float32)
    denom = np.sqrt(gx * gx + gy * gy + 1e-8).astype(np.float32)
    nx = (gx / denom).astype(np.float32)
    ny = (gy / denom).astype(np.float32)

    dy, dx = np.gradient(d_s)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    hxy = (0.5 * (dyx + dxy)).astype(np.float32)

    step_along_n = np.abs(dx * nx + dy * ny).astype(np.float32)
    inflection_nn = np.abs(nx * nx * dxx + 2.0 * nx * ny * hxy + ny * ny * dyy).astype(np.float32)
    lap_d_abs = np.abs(dxx + dyy).astype(np.float32)
    return step_along_n, inflection_nn, grad_g_mag, lap_d_abs


def compute_transtensional_partitioning(
    geod_dil: np.ndarray,
    geod_shear: np.ndarray,
    sigma_grad: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dimensionless kinematic transtensional mode index tau_ext in [-1, 1]
    and its spatial gradient magnitude |grad(tau_ext)|.
    """
    dil = geod_dil.astype(np.float32)
    shr = np.maximum(0.0, geod_shear.astype(np.float32))
    tau_ext = (dil / np.sqrt(dil * dil + shr * shr + 1e-12)).astype(np.float32)
    tau_s = ndimage.gaussian_filter(tau_ext, sigma=sigma_grad, mode="nearest")
    ty, tx = np.gradient(tau_s)
    tau_grad_mag = np.hypot(tx, ty).astype(np.float32)
    return tau_ext, tau_grad_mag


def compute_conductivity_demag_breach(
    cond_surf: np.ndarray,
    rtp: np.ndarray,
    tmi_up150: np.ndarray,
    foot: np.ndarray,
    sigma_cond: float = 1.5,
    sigma_rtp: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the lateral gradient of log surface electrical conductivity (illite-smectite
    transition margin) and its coupling with shallow RTP hydrothermal demagnetization troughs.
    """
    c_pos = np.maximum(0.0, cond_surf.astype(np.float32))
    log_c = ndimage.gaussian_filter(np.log1p(c_pos), sigma=sigma_cond, mode="nearest")
    cy, cx = np.gradient(log_c)
    cond_margin = rank_in_footprint(np.hypot(cx, cy).astype(np.float32), foot)

    r_up = rank_in_footprint(tmi_up150.astype(np.float32), foot)
    r_rtp = rank_in_footprint(rtp.astype(np.float32), foot)
    demag_atten = np.maximum(0.0, r_up - r_rtp).astype(np.float32)

    rtp_s = ndimage.gaussian_filter(rtp.astype(np.float32), sigma=sigma_rtp, mode="nearest")
    lap_rtp = ndimage.laplace(rtp_s).astype(np.float32)
    # In reduced-to-pole magnetics, a local magnetic trough has positive Laplacian (-div grad < 0 => lap > 0)
    rtp_trough = rank_in_footprint(np.maximum(0.0, -lap_rtp), foot)

    breach_raw = cond_margin * (0.55 * demag_atten + 0.45 * rtp_trough)
    upflow_breach = rank_in_footprint(breach_raw, foot)
    return cond_margin, upflow_breach


def load_geothermometer_disequilibrium_field(
    csv_path: Path,
    shape: tuple[int, int],
    foot: np.ndarray,
) -> np.ndarray:
    """Build a label-free chemical geothermometer disequilibrium halo field from
    `gdr_wellspring_in_footprint.csv` (DOE GDR Submission #1391).

    Strictly ignores `dist_known_fault_px`. Uses only `row`, `col`, `temp_c`,
    `geothermquartz_c`, `geothermchalc_c`, and `geothermcat_c`.
    """
    h, w = shape
    imp_diseq = np.zeros((h, w), dtype=np.float32)
    imp_temp = np.zeros((h, w), dtype=np.float32)
    if not csv_path.is_file():
        return np.zeros((h, w), dtype=np.float32)

    def _flt(val: str | None) -> float:
        if val is None or val == "" or val.upper() == "NULL" or val.lower() == "nan":
            return float("nan")
        try:
            return float(val)
        except ValueError:
            return float("nan")

    with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Architectural guard: never read dist_known_fault_px
            r_i = _flt(row.get("row"))
            c_i = _flt(row.get("col"))
            if not (np.isfinite(r_i) and np.isfinite(c_i)):
                continue
            ri, ci = int(round(r_i)), int(round(c_i))
            if not (0 <= ri < h and 0 <= ci < w):
                continue
            t_meas = _flt(row.get("temp_c"))
            t_qtz = _flt(row.get("geothermquartz_c"))
            t_chl = _flt(row.get("geothermchalc_c"))
            t_cat = _flt(row.get("geothermcat_c"))
            chem_list = [t for t in (t_qtz, t_chl, t_cat) if np.isfinite(t) and 20.0 <= t <= 380.0]
            t_chem = max(chem_list) if chem_list else float("nan")
            t_m = t_meas if (np.isfinite(t_meas) and 0.0 <= t_meas <= 250.0) else 15.0
            diseq = max(0.0, (t_chem - t_m) if np.isfinite(t_chem) else 0.0)
            excess_t = max(0.0, t_m - 20.0)
            if diseq > imp_diseq[ri, ci]:
                imp_diseq[ri, ci] = float(np.log1p(diseq))
            if excess_t > imp_temp[ri, ci]:
                imp_temp[ri, ci] = float(np.log1p(excess_t))

    # Multi-scale thermal plume diffusion (3.0 km near-field + 7.5 km district halo)
    plume_near = ndimage.gaussian_filter(imp_diseq + 0.6 * imp_temp, sigma=30.0, mode="constant", cval=0.0)
    plume_far = ndimage.gaussian_filter(imp_diseq + 0.6 * imp_temp, sigma=75.0, mode="constant", cval=0.0)
    plume = (0.65 * rank_in_footprint(plume_near, foot) + 0.35 * rank_in_footprint(plume_far, foot)).astype(np.float32)
    return rank_in_footprint(plume, foot)


def _read_clean_band(src: rasterio.DatasetReader, band_idx: int, foot: np.ndarray) -> np.ndarray:
    arr = src.read(band_idx).astype(np.float32)
    bad = (~np.isfinite(arr)) | (arr <= -9000.0) | (np.abs(arr) > 1e10) | (~foot)
    arr[bad] = np.nan
    return F77.fill_nearest(arr)


def build_bgb_features(
    raw_dir: Path,
    foot: np.ndarray,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compute or load the 10-channel H77-BGB feature stack of shape `(H, W, 10)` (`float32`).

    All 10 channels are strictly in `[0, 1]` inside `foot` and `0.0` outside `foot`, with
    zero NaNs anywhere in the array.
    """
    h, w = foot.shape
    if cache_path is not None and cache_path.is_file():
        mm = np.load(cache_path, mmap_mode="r")
        if mm.shape == (h, w, len(BGB_NAMES)) and mm.dtype == np.float32:
            return np.asarray(mm, dtype=np.float32), BGB_NAMES

    tf_path = raw_dir / "training_features.tif"
    ext_path = raw_dir / "external" / "geodawn_extensions_u8.tif"
    csv_path = raw_dir / "external" / "gdr_wellspring_in_footprint.csv"

    with rasterio.open(tf_path) as src:
        rtp = _read_clean_band(src, 2, foot)
        geod_2nd = _read_clean_band(src, 4, foot)
        geod_shr = _read_clean_band(src, 7, foot)
        geod_dil = _read_clean_band(src, 8, foot)
        deq = _read_clean_band(src, 10, foot)
        grav_vg = _read_clean_band(src, 11, foot)
        iso_grav = _read_clean_band(src, 13, foot)
        depth_base = _read_clean_band(src, 15, foot)
        ieq = _read_clean_band(src, 16, foot)
        cond_surf = _read_clean_band(src, 17, foot)
        elev_slope = _read_clean_band(src, 19, foot)

    with rasterio.open(ext_path) as esrc:
        tmi_up150_raw = esrc.read(4).astype(np.float32)
        bad_u = (~np.isfinite(tmi_up150_raw)) | (~foot)
        tmi_up150_raw[bad_u] = np.nan
        tmi_up150 = F77.fill_nearest(tmi_up150_raw)
        del tmi_up150_raw, bad_u

    out = np.zeros((h, w, len(BGB_NAMES)), dtype=np.float32)

    # --- Sub-block A: Buried Half-Graben Growth-Wedge Inflection ---
    step_n, infl_nn, grad_g_mag, lap_d_abs = compute_gravity_normal_basement_hessian(
        depth_base, iso_grav, sigma_depth=2.0, sigma_grav=1.5
    )
    r_grad_g = rank_in_footprint(grad_g_mag, foot)
    c1 = rank_in_footprint(step_n * np.sqrt( np.maximum(0.0, r_grad_g) ), foot)
    c2 = rank_in_footprint(infl_nn * r_grad_g, foot)

    slope_pos = np.maximum(0.0, elev_slope)
    s60 = float(np.percentile(slope_pos[foot], 60.0)) + 1e-6
    flatness_wt = np.exp(-slope_pos / s60).astype(np.float32)
    c3 = rank_in_footprint(np.sqrt(c1 * c2) * flatness_wt, foot)

    gvgy, gvgx = np.gradient(ndimage.gaussian_filter(grav_vg, sigma=1.5, mode="nearest"))
    r_gvg_grad = rank_in_footprint(np.hypot(gvgx, gvgy).astype(np.float32), foot)
    r_lap_d = rank_in_footprint(lap_d_abs, foot)
    c4 = rank_in_footprint(r_gvg_grad * r_lap_d, foot)

    out[:, :, 0] = c1
    out[:, :, 1] = c2
    out[:, :, 2] = c3
    out[:, :, 3] = c4

    del step_n, infl_nn, grad_g_mag, lap_d_abs, gvgy, gvgx, r_gvg_grad, r_lap_d, flatness_wt, slope_pos
    gc.collect()

    # --- Sub-block B: Kinematic Transtensional Strain-Partitioning Gradient ---
    tau_ext, tau_grad_mag = compute_transtensional_partitioning(geod_dil, geod_shr, sigma_grad=2.0)
    r_geod2 = rank_in_footprint(geod_2nd, foot)
    e_sub = np.sqrt(c1 * r_grad_g).astype(np.float32)
    transtens_raw = rank_in_footprint(np.maximum(0.0, tau_ext) * np.maximum(0.0, geod_shr), foot)
    c5 = rank_in_footprint(transtens_raw * (0.25 + 0.75 * e_sub), foot)
    c6 = rank_in_footprint(tau_grad_mag * r_geod2 * (0.25 + 0.75 * e_sub), foot)

    deq_s = ndimage.gaussian_filter(deq, sigma=2.0, mode="nearest")
    deqy, deqx = np.gradient(deq_s)
    r_deq_grad = rank_in_footprint(np.hypot(deqx, deqy).astype(np.float32), foot)
    r_ieq = rank_in_footprint(ieq, foot)
    c7 = rank_in_footprint(r_geod2 * (1.0 - 0.70 * r_ieq) * r_deq_grad * (0.25 + 0.75 * r_grad_g), foot)

    out[:, :, 4] = c5
    out[:, :, 5] = c6
    out[:, :, 6] = c7

    del tau_ext, tau_grad_mag, transtens_raw, deq_s, deqy, deqx, r_deq_grad, r_ieq
    gc.collect()

    # --- Sub-block C: Illite-Smectite Conductivity-Margin Upflow Breach ---
    c8, c9 = compute_conductivity_demag_breach(cond_surf, rtp, tmi_up150, foot, sigma_cond=1.5, sigma_rtp=2.0)
    geotherm_halo = load_geothermometer_disequilibrium_field(csv_path, (h, w), foot)
    c10 = rank_in_footprint(geotherm_halo * np.sqrt(c1 * c9), foot)

    out[:, :, 7] = c8
    out[:, :, 8] = c9
    out[:, :, 9] = c10

    # Strict finite & footprint enforcement
    out[~foot, :] = 0.0
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(out, 0.0, 1.0, out=out)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)

    return out, BGB_NAMES


def compute_bgb_surfaces(bgb_stack: np.ndarray, foot: np.ndarray) -> dict[str, np.ndarray]:
    """Compute label-free physical sub-block surfaces and overall composite surface in `[0, 1]`."""
    s_growth = (
        0.35 * bgb_stack[:, :, 0]
        + 0.30 * bgb_stack[:, :, 1]
        + 0.20 * bgb_stack[:, :, 2]
        + 0.15 * bgb_stack[:, :, 3]
    ).astype(np.float32)
    s_strain = (
        0.40 * bgb_stack[:, :, 4]
        + 0.35 * bgb_stack[:, :, 5]
        + 0.25 * bgb_stack[:, :, 6]
    ).astype(np.float32)
    s_upflow = (
        0.35 * bgb_stack[:, :, 7]
        + 0.40 * bgb_stack[:, :, 8]
        + 0.25 * bgb_stack[:, :, 9]
    ).astype(np.float32)
    s_comp = (0.50 * s_growth + 0.25 * s_strain + 0.25 * s_upflow).astype(np.float32)
    return {
        "S_growth_wedge": rank_in_footprint(s_growth, foot),
        "S_strain_part": rank_in_footprint(s_strain, foot),
        "S_upflow_breach": rank_in_footprint(s_upflow, foot),
        "S_bgb_composite": rank_in_footprint(s_comp, foot),
    }
