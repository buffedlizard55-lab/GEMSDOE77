"""GEMSDOE77 HSL feature stack (77 label-free channels + 7 fold-visible structural channels).

Organizes channels into strictly nested blocks so C2_standard (33 ch), C3_contactphys
(49 ch), and C4_contactphys_lidar5 (54 ch) reproduce the prior GEMSDOE77 baselines on
identical spatial folds, while HSL adds:
  * hsl1_lidar12 (13 ch): full 12-band 1 m USGS 3DEP lidar scarp crest-toe Laplacian
    dipole, unilateral facing polarity, excess contrast, multi-cell spin-2 sub-pixel
    strike director continuity (sigma=2, 5 px), and wash-bridging scarp Hessian ridges.
  * hsl2_mag_hydro (5 ch): U/Th ratio (Band 3 of geodawn_extensions_u8.tif), 150 m
    upward-continued TMI (Band 4), shallow vertical magnetic continuation residual
    (rank01(tmi) - rank01(TMI_up150)) and its gradient, and hydrothermal magnetite-
    destruction / smectite clay-cap conduit index.
  * hsl3_gdr_vents (5 ch): INGENIOUS GDR 1391 thermal springs/wells (Hot/Warm or
    T >= 35 C log-distance, Gaussian thermal excess at sigma=4, 12 px, silica/cation
    geothermometer excess >= 70 C at sigma=8 px) and Quaternary volcanic vent
    log-distance. Column dist_known_fault_px is strictly excluded.
  * hsl4_vis_struct (7 ch, computed per fold strictly from fold['visible']):
    log-distance to visible fault, Gaussian visible-fault densities (sigma=5, 15, 35 px),
    stepover relay-deficit ratio, differential-geometric forward tip-propagation cone,
    and 1 m lidar-to-host spin-2 strike alignment.
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import fft as sfft
from scipy import ndimage

from gems77 import features77 as F0

HSL1_LIDAR12_NAMES = [
    "lidar_ex_mean",
    "lidar_lapneg_max",
    "lidar_lappos_max",
    "lidar_upface_max",
    "lidar_cross_max",
    "lidar_relief",
    "lidar_crest_toe_dipole",
    "lidar_facing_polarity",
    "lidar_excess_contrast",
    "lidar_strike_coh_s2",
    "lidar_strike_coh_s5",
    "lidar_scarp_ridge_s2",
    "lidar_scarp_ridge_s4",
]

HSL2_MAG_HYDRO_NAMES = [
    "rad_UTh",
    "tmi_up150",
    "mag_shallow_resid",
    "mag_shallow_grad",
    "hydro_demag_conduit",
]

HSL3_GDR_VENTS_NAMES = [
    "hydro_hot_spring_logdist",
    "hydro_temp_excess_s4",
    "hydro_temp_excess_s12",
    "hydro_geotherm_excess_s8",
    "volcanic_vent_logdist",
]

HSL4_VIS_STRUCT_NAMES = [
    "vis_logdist",
    "vis_dens_s5",
    "vis_dens_s15",
    "vis_dens_s35",
    "vis_relay_deficit",
    "vis_tip_cone",
    "vis_lidar_strike_align",
]

BLOCKS = {
    **F0.BLOCKS,
    "hsl1_lidar12": HSL1_LIDAR12_NAMES,
    "hsl2_mag_hydro": HSL2_MAG_HYDRO_NAMES,
    "hsl3_gdr_vents": HSL3_GDR_VENTS_NAMES,
}

CHANNELS = F0.CHANNELS + HSL1_LIDAR12_NAMES + HSL2_MAG_HYDRO_NAMES + HSL3_GDR_VENTS_NAMES
IDX = {n: i for i, n in enumerate(CHANNELS)}
PAD = 64


def _fast_phase_congruency(
    im: np.ndarray,
    nscale: int = 4,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
) -> np.ndarray:
    """Float32 multi-worker scipy.fft implementation of Felsberg/Kovesi phase congruency."""
    im = np.asarray(im, dtype=np.float32)
    rows, cols = im.shape
    IM = sfft.rfft2(im, workers=2)
    fx = sfft.rfftfreq(cols)[None, :].astype(np.float32)
    fy = sfft.fftfreq(rows)[:, None].astype(np.float32)
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    H1 = (1j * fx / radius).astype(np.complex64)
    H2 = (1j * fy / radius).astype(np.complex64)
    radius[0, 0] = 0.0
    sum_f = np.zeros((rows, cols), dtype=np.float32)
    sum_h1 = np.zeros((rows, cols), dtype=np.float32)
    sum_h2 = np.zeros((rows, cols), dtype=np.float32)
    sum_an = np.zeros((rows, cols), dtype=np.float32)
    max_an = np.zeros((rows, cols), dtype=np.float32)
    denom = np.float32(2.0 * (np.log(sigma_onf) ** 2))
    for s in range(nscale):
        wl = np.float32(min_wavelength * (mult ** s))
        fo = np.float32(1.0 / wl)
        lg = np.exp(-((np.log(np.maximum(radius, np.float32(1e-12)) / fo)) ** 2) / denom).astype(np.float32)
        lg[0, 0] = 0.0
        G = IM * lg
        f = sfft.irfft2(G, s=(rows, cols), workers=2)
        h1 = sfft.irfft2(G * H1, s=(rows, cols), workers=2)
        h2 = sfft.irfft2(G * H2, s=(rows, cols), workers=2)
        an = np.sqrt(f * f + h1 * h1 + h2 * h2)
        sum_f += f
        sum_h1 += h1
        sum_h2 += h2
        sum_an += an
        max_an = np.maximum(max_an, an)
    energy = np.sqrt(sum_f * sum_f + sum_h1 * sum_h1 + sum_h2 * sum_h2)
    pc = energy / (sum_an + np.float32(1e-12))
    if nscale > 1:
        width = (sum_an / (max_an + np.float32(1e-12)) - np.float32(1.0)) / np.float32(nscale - 1.0)
        weight = np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(10.0) * (np.float32(0.5) - np.clip(width, 0.0, 1.0))))
        pc = pc * weight
    return np.clip(pc, 0.0, 1.0).astype(np.float32)


def _fast_pseudogravity(rtp_filled: np.ndarray) -> np.ndarray:
    rows, cols = rtp_filled.shape
    F = sfft.rfft2(rtp_filled.astype(np.float32), workers=2)
    fx = sfft.rfftfreq(cols)[None, :].astype(np.float32)
    fy = sfft.fftfreq(rows)[:, None].astype(np.float32)
    k = np.float32(2.0 * np.pi) * np.sqrt(fx * fx + fy * fy)
    k[0, 0] = 1.0
    PG = F / k
    PG[0, 0] = 0.0
    return sfft.irfft2(PG, s=(rows, cols), workers=2).astype(np.float32)


def _padcrop(fn, a: np.ndarray) -> np.ndarray:
    p = np.pad(a, PAD, mode="reflect")
    r = fn(p)
    return r[PAD:-PAD, PAD:-PAD].astype(np.float32)


def _read(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(band)


def build_hsl_stack(root: Path, force: bool = False) -> Path:
    """Build and cache the 77-channel label-free HSL stack at .cache/hsl_processed/hsl_stack.npy."""
    raw_dir = root / "data" / "raw"
    proc_dir = root / ".cache" / "hsl_processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / "hsl_stack.npy"
    if out_path.exists() and not force:
        return out_path

    t0 = time.time()
    foot = np.isfinite(_read(raw_dir / "sample_submission.tif"))
    H, W = foot.shape
    stack = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(len(CHANNELS), H, W)
    )

    def put(name: str, arr: np.ndarray):
        out = np.where(foot, np.asarray(arr, dtype=np.float32), np.float32(np.nan))
        stack[IDX[name], :, :] = out

    # 1. Raw 19 bands
    raw = {}
    for i, name in enumerate(F0.RAW_NAMES, start=1):
        a = F0.clean(_read(raw_dir / "training_features.tif", i))
        stack[IDX[name], :, :] = a
        raw[name] = a
    print(f"[hsl_stack] raw 19 bands loaded ({time.time()-t0:.1f}s)", flush=True)

    dem_f = F0.fill_median(raw["det_elev"], foot)
    rtp_f = F0.fill_median(raw["rtp"], foot)
    rtp_n = F0.fill_nearest(raw["rtp"])
    grav_f = F0.fill_median(raw["iso_grav_anom"], foot)
    grav_n = F0.fill_nearest(raw["iso_grav_anom"])
    tmi_f = F0.fill_median(raw["tmi"], foot)
    depth_f = F0.fill_median(raw["depth_to_base_surf"], foot)
    cond_f = F0.fill_median(raw["cond_surf"], foot)

    # 2. Terrain block (8 ch) + profcurv_signed
    put("dem_curv1", ndimage.gaussian_laplace(dem_f, 1.0, mode="nearest"))
    put("dem_curv25", ndimage.gaussian_laplace(dem_f, 2.5, mode="nearest"))
    for nm, sg in (("dem_ridge1", 1.0), ("dem_ridge2", 2.0), ("dem_ridge3p5", 3.5)):
        put(nm, F0.hessian_ridge(dem_f, sg))
    put("dem_relief5", ndimage.maximum_filter(dem_f, 5) - ndimage.minimum_filter(dem_f, 5))
    put("dem_tophat5", dem_f - ndimage.grey_opening(dem_f, size=(5, 5)))
    coh = F0.structure_tensor(dem_f, 2.0)
    put("st_coherence", coh)
    put("profcurv_signed", F0.profile_curvature(dem_f, 1.0))

    # 3. Contact-physics block (16 ch)
    ggy, ggx = np.gradient(grav_f)
    hg = np.maximum(F0.fill_median(raw["tmi_hg"], foot), np.float32(0.0))
    vg = F0.fill_median(raw["tmi_vg"], foot)
    mag_tilt = np.arctan2(vg, np.maximum(hg, np.float32(1e-9))).astype(np.float32)
    mag_as = np.sqrt(hg * hg + vg * vg)
    grav_thdr = np.hypot(ggx, ggy).astype(np.float32)
    gvg = F0.fill_median(raw["iso_grav_anom_vg"], foot)
    grav_tilt = np.arctan2(gvg, np.maximum(grav_thdr, np.float32(1e-12))).astype(np.float32)
    grav_as = np.sqrt(grav_thdr * grav_thdr + gvg * gvg)
    mag_as_log = np.log1p(F0._sm(mag_as, 1.0)).astype(np.float32)

    put("mag_tilt", mag_tilt)
    put("mag_tilt_abs", np.abs(mag_tilt))
    put("mag_as_log", mag_as_log)
    put("grav_thdr_log", np.log1p(F0._sm(grav_thdr, 1.0)))
    put("grav_tilt", grav_tilt)
    put("grav_as_log", np.log1p(F0._sm(grav_as, 1.0)))

    r_mag = F0.rank01(np.log1p(mag_as), foot)
    r_grav = F0.rank01(np.log1p(grav_as), foot)
    wz_mag = np.float32(1.0) - np.abs(mag_tilt) / np.float32(np.pi / 2)
    wz_grav = np.float32(1.0) - np.abs(grav_tilt) / np.float32(np.pi / 2)
    put("contact_confluence", np.minimum(r_mag, r_grav) * np.sqrt(np.maximum(wz_mag * wz_grav, np.float32(0.0))))

    depth_sm = F0._sm(depth_f, 2.0)
    dpy, dpx = np.gradient(depth_sm)
    rpy, rpx = np.gradient(F0._sm(rtp_f, 1.0))
    nrm = np.hypot(rpx, rpy) + np.float32(1e-12)
    put("base_step_along_maggrad", (dpx * rpx + dpy * rpy) / nrm)
    put("base_grad_log", np.log1p(np.hypot(dpx, dpy)))

    slope_sm = F0._sm(F0.fill_median(raw["det_elev_slope"], foot), 1.0)
    front = (ndimage.maximum_filter(slope_sm, 5) - slope_sm) * (np.float32(0.25) + np.float32(0.75) * coh)
    put("frontness", front)

    put("pc_mag", _padcrop(_fast_phase_congruency, rtp_n))
    put("pc_grav", _padcrop(_fast_phase_congruency, grav_n))
    print(f"[hsl_stack] phase congruency done ({time.time()-t0:.1f}s)", flush=True)

    rng = np.random.default_rng(7)
    sample_idx = rng.choice(np.flatnonzero(foot), size=200_000, replace=False)
    pg = _padcrop(_fast_pseudogravity, rtp_n)
    pg_hp = (pg - F0._sm(pg, 15.0)).astype(np.float32)
    g_hp = (grav_n - F0._sm(grav_n, 15.0)).astype(np.float32)
    slope0 = float(
        np.cov(pg_hp.ravel()[sample_idx], g_hp.ravel()[sample_idx])[0, 1]
        / np.var(pg_hp.ravel()[sample_idx])
    )
    resid = (g_hp - np.float32(slope0) * pg_hp).astype(np.float32)
    lslope, lcorr = F0.local_regression(pg_hp, g_hp, size=25)
    put("pg_slope", lslope)
    put("pg_corr", lcorr)
    ry, rx = np.gradient(F0._sm(resid, 1.0))
    put("pg_resid_grad_log", np.log1p(np.hypot(rx, ry)))
    del pg, pg_hp, g_hp, resid, lslope, lcorr, ry, rx, rtp_n, grav_n, raw, dem_f, rtp_f, grav_f, depth_f
    gc.collect()
    print(f"[hsl_stack] contactphys done ({time.time()-t0:.1f}s)", flush=True)

    # 4. Base radiometrics (6 ch) & 5-band lidar (5 ch)
    for nm, relpath, band in (
        ("rad_K", "external/geodawn_rad_u8.tif", 1),
        ("rad_Th", "external/geodawn_rad_u8.tif", 2),
        ("rad_U", "external/geodawn_rad_u8.tif", 3),
        ("rad_TC", "external/geodawn_rad_u8.tif", 4),
        ("rad_ThK", "external/geodawn_extensions_u8.tif", 1),
        ("rad_UK", "external/geodawn_extensions_u8.tif", 2),
        ("lidar_ex_max", "external/lidar_scarp_features_u8.tif", 1),
        ("lidar_step_max", "external/lidar_scarp_features_u8.tif", 3),
        ("lidar_downface_max", "external/lidar_scarp_features_u8.tif", 6),
        ("lidar_coh100", "external/lidar_scarp_features_u8.tif", 10),
        ("lidar_valid", "external/lidar_scarp_features_u8.tif", 12),
    ):
        a = _read(raw_dir / relpath, band).astype(np.float32) / np.float32(255.0)
        put(nm, a)

    # 5. Hypothesis H77-HSL-1: Full 12-band 1 m Lidar Scarp Dipole, Facing Polarity & Strike Continuity (13 ch)
    lid_path = raw_dir / "external" / "lidar_scarp_features_u8.tif"
    ex_max = _read(lid_path, 1).astype(np.float32) / np.float32(255.0)
    ex_mean = _read(lid_path, 2).astype(np.float32) / np.float32(255.0)
    step_max = _read(lid_path, 3).astype(np.float32) / np.float32(255.0)
    lapneg = _read(lid_path, 4).astype(np.float32) / np.float32(255.0)
    lappos = _read(lid_path, 5).astype(np.float32) / np.float32(255.0)
    downface = _read(lid_path, 6).astype(np.float32) / np.float32(255.0)
    upface = _read(lid_path, 7).astype(np.float32) / np.float32(255.0)
    cross_max = _read(lid_path, 8).astype(np.float32) / np.float32(255.0)
    relief = _read(lid_path, 9).astype(np.float32) / np.float32(255.0)
    coh100 = _read(lid_path, 10).astype(np.float32) / np.float32(255.0)
    strike_u8 = _read(lid_path, 11).astype(np.float32)
    lvalid = (_read(lid_path, 12) > 0).astype(np.float32)

    put("lidar_ex_mean", ex_mean)
    put("lidar_lapneg_max", lapneg)
    put("lidar_lappos_max", lappos)
    put("lidar_upface_max", upface)
    put("lidar_cross_max", cross_max)
    put("lidar_relief", relief)

    # Crest-toe curvature dipole: paired convex crest (lapneg) + concave toe (lappos) across step
    dipole = np.sqrt(np.maximum(lapneg * lappos, np.float32(0.0))) * cross_max * lvalid
    put("lidar_crest_toe_dipole", dipole)

    # Unilateral facing polarity: asymmetric scarp drop vs symmetric ridge/gully
    polarity = (np.abs(downface - upface) / (downface + upface + np.float32(1e-4))) * step_max * lvalid
    put("lidar_facing_polarity", polarity)

    # Excess contrast inside 100 m cell
    put("lidar_excess_contrast", (ex_max - ex_mean) * lvalid)

    # Spin-2 axial director continuity of sub-pixel 1 m scarp strike across 200 m and 500 m
    theta_lid = (strike_u8 / np.float32(255.0)) * np.float32(np.pi)
    w_lid = coh100 * step_max * lvalid
    u2_lid = (w_lid * np.cos(np.float32(2.0) * theta_lid)).astype(np.float32)
    v2_lid = (w_lid * np.sin(np.float32(2.0) * theta_lid)).astype(np.float32)
    for nm, sg in (("lidar_strike_coh_s2", 2.0), ("lidar_strike_coh_s5", 5.0)):
        u_s = F0._sm(u2_lid, sg)
        v_s = F0._sm(v2_lid, sg)
        put(nm, np.hypot(u_s, v_s))

    scarp_energy = (step_max * coh100 * lvalid).astype(np.float32)
    put("lidar_scarp_ridge_s2", F0.hessian_ridge(scarp_energy, 2.0))
    put("lidar_scarp_ridge_s4", F0.hessian_ridge(scarp_energy, 4.0))

    # 6. Hypothesis H77-HSL-2: Shallow Vertical Magnetic Continuation Residual & Hydrothermal Demagnetization (5 ch)
    ext_path = raw_dir / "external" / "geodawn_extensions_u8.tif"
    rad_uth = _read(ext_path, 3).astype(np.float32) / np.float32(255.0)
    tmi_up150 = _read(ext_path, 4).astype(np.float32) / np.float32(255.0)
    put("rad_UTh", rad_uth)
    put("tmi_up150", tmi_up150)

    mag_shallow_resid = (F0.rank01(tmi_f, foot) - F0.rank01(tmi_up150, foot)).astype(np.float32)
    put("mag_shallow_resid", mag_shallow_resid)
    mry, mrx = np.gradient(F0._sm(mag_shallow_resid, 1.5))
    put("mag_shallow_grad", np.log1p(np.hypot(mrx, mry)))

    cy, cx = np.gradient(F0._sm(cond_f, 1.5))
    cond_grad = np.hypot(cx, cy).astype(np.float32)
    hydro_demag = (
        F0.rank01(cond_f, foot)
        * (np.float32(1.0) - F0.rank01(mag_as_log, foot))
        * F0.rank01(cond_grad, foot)
    ).astype(np.float32)
    put("hydro_demag_conduit", hydro_demag)

    # 7. Hypothesis H77-HSL-3: INGENIOUS GDR 1391 Geothermal Outflow & Quaternary Volcanic Vents (5 ch)
    # Strictly exclude dist_known_fault_px!
    ws_df = pd.read_csv(
        raw_dir / "external" / "gdr_wellspring_in_footprint.csv",
        usecols=["row", "col", "thermalclass", "temp_c", "geothermquartz_c", "geothermchalc_c", "geothermcat_c"],
    )
    rows_ws = np.clip(ws_df["row"].to_numpy(dtype=np.int32), 0, H - 1)
    cols_ws = np.clip(ws_df["col"].to_numpy(dtype=np.int32), 0, W - 1)
    tclass = ws_df["thermalclass"].fillna("").to_numpy()
    temp_c = ws_df["temp_c"].to_numpy(dtype=np.float32)

    is_hot_warm = np.isin(tclass, ["Hot", "Warm"]) | (np.isfinite(temp_c) & (temp_c >= 35.0))
    hot_mask = np.zeros((H, W), dtype=bool)
    hot_mask[rows_ws[is_hot_warm], cols_ws[is_hot_warm]] = True
    dist_hot = ndimage.distance_transform_edt(~hot_mask).astype(np.float32)
    put("hydro_hot_spring_logdist", -np.log1p(dist_hot))

    temp_grid = np.zeros((H, W), dtype=np.float32)
    valid_t = np.isfinite(temp_c) & (temp_c > 20.0)
    np.maximum.at(temp_grid, (rows_ws[valid_t], cols_ws[valid_t]), np.clip(temp_c[valid_t] - 20.0, 0.0, 280.0))
    put("hydro_temp_excess_s4", np.log1p(F0._sm(temp_grid, 4.0) * np.float32(16.0)))
    put("hydro_temp_excess_s12", np.log1p(F0._sm(temp_grid, 12.0) * np.float32(144.0)))

    q_c = ws_df["geothermquartz_c"].to_numpy(dtype=np.float32)
    ch_c = ws_df["geothermchalc_c"].to_numpy(dtype=np.float32)
    cat_c = ws_df["geothermcat_c"].to_numpy(dtype=np.float32)
    geotherm_max = np.nanmax(
        np.vstack([
            np.where(np.isfinite(q_c) & (q_c < 350.0), q_c, np.nan),
            np.where(np.isfinite(ch_c) & (ch_c < 350.0), ch_c, np.nan),
            np.where(np.isfinite(cat_c) & (cat_c < 350.0), cat_c, np.nan),
        ]),
        axis=0,
    )
    gt_grid = np.zeros((H, W), dtype=np.float32)
    valid_gt = np.isfinite(geotherm_max) & (geotherm_max > 70.0)
    np.maximum.at(
        gt_grid,
        (rows_ws[valid_gt], cols_ws[valid_gt]),
        np.clip(geotherm_max[valid_gt] - 70.0, 0.0, 250.0),
    )
    put("hydro_geotherm_excess_s8", np.log1p(F0._sm(gt_grid, 8.0) * np.float32(64.0)))

    vv_df = pd.read_csv(
        raw_dir / "external" / "gdr_volcanic_vents_in_footprint.csv",
        usecols=["row", "col"],
    )
    vent_mask = np.zeros((H, W), dtype=bool)
    vent_mask[
        np.clip(vv_df["row"].to_numpy(dtype=np.int32), 0, H - 1),
        np.clip(vv_df["col"].to_numpy(dtype=np.int32), 0, W - 1),
    ] = True
    dist_vent = ndimage.distance_transform_edt(~vent_mask).astype(np.float32)
    put("volcanic_vent_logdist", -np.log1p(dist_vent))

    stack.flush()
    del stack
    print(f"[hsl_stack] ALL 77 channels done ({time.time()-t0:.1f}s)", flush=True)
    manifest = {
        "path": str(out_path),
        "shape": [len(CHANNELS), H, W],
        "dtype": "float32",
        "channels": CHANNELS,
        "blocks": BLOCKS,
        "build_seconds": round(time.time() - t0, 2),
        "strictly_label_free": True,
        "excluded_columns": ["dist_known_fault_px"],
    }
    (proc_dir / "hsl_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_path


def compute_visible_fault_features(
    visible_mask: np.ndarray,
    foot: np.ndarray,
    u2_lid: np.ndarray,
    v2_lid: np.ndarray,
) -> np.ndarray:
    """Compute the 7 Hypothesis H77-HSL-4 continuation channels strictly from visible_mask.

    Returns float32 array of shape (7, H, W) ordered per HSL4_VIS_STRUCT_NAMES.
    """
    vis = np.asarray(visible_mask, dtype=bool)
    vis_f = vis.astype(np.float32)
    H, W = vis.shape
    out = np.zeros((len(HSL4_VIS_STRUCT_NAMES), H, W), dtype=np.float32)

    dist, (iy, ix) = ndimage.distance_transform_edt(~vis, return_indices=True)
    dist = dist.astype(np.float32)
    out[0] = np.log1p(dist)

    d5 = F0._sm(vis_f, 5.0)
    d15 = F0._sm(vis_f, 15.0)
    d35 = F0._sm(vis_f, 35.0)
    out[1] = d5
    out[2] = d15
    out[3] = d35
    out[4] = d15 / (d5 + np.float32(0.01))

    # Hessian ridge tangent spin-2 director on visible fault traces
    vis_s = F0._sm(vis_f, 2.0)
    gy, gx = np.gradient(vis_s)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    gxy = np.float32(0.5) * (gyx + gxy)
    D = np.hypot(gxx - gyy, np.float32(2.0) * gxy) + np.float32(1e-8)
    u2_vis = ((gxx - gyy) / D).astype(np.float32)
    v2_vis = ((np.float32(2.0) * gxy) / D).astype(np.float32)

    u2_near = u2_vis[iy, ix]
    v2_near = v2_vis[iy, ix]
    del u2_vis, v2_vis, gy, gx, gyy, gyx, gxy, gxx, D, vis_s

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - ix.astype(np.float32)
    dy = grid_y - iy.astype(np.float32)
    del grid_y, grid_x, iy, ix
    r2 = dx * dx + dy * dy + np.float32(1e-6)
    u2_r = (dx * dx - dy * dy) / r2
    v2_r = (np.float32(2.0) * dx * dy) / r2
    del dx, dy, r2

    cos2_cone = u2_r * u2_near + v2_r * v2_near
    del u2_r, v2_r
    out[5] = np.maximum(cos2_cone, np.float32(0.0)) * np.exp(-dist / np.float32(15.0))
    del cos2_cone
    out[6] = (u2_lid * u2_near + v2_lid * v2_near) * np.exp(-dist / np.float32(25.0))
    del u2_near, v2_near, dist
    gc.collect()

    out[:, ~foot] = np.nan
    return out
