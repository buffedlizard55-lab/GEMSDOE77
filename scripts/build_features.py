#!/usr/bin/env python3
"""Build the GEMSDOE77 feature stack (data/processed/stack.npy) + band diagnostics.

Channels/blocks are defined in src/gems77/features77.py. Nothing here reads
labels except the footprint (from sample_submission finite cells).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from gems77 import features77 as F  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "stack.npy"
PAD = 64


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read(path, i=1):
    with rasterio.open(path) as s:
        return s.read(i)


def padcrop(fn, a):
    p = np.pad(a, PAD, mode="reflect")
    r = fn(p)
    return r[PAD:-PAD, PAD:-PAD].astype(np.float32)


def main():
    t0 = time.time()
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    foot = np.isfinite(read(RAW / "sample_submission.tif"))
    H, W = foot.shape
    stack = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(H, W, len(F.CHANNELS)))

    raw = {}
    for i, name in enumerate(F.RAW_NAMES, start=1):
        a = F.clean(read(RAW / "training_features.tif", i))
        stack[:, :, F.IDX[name]] = a
        raw[name] = a
        log(f"raw band {i:2d} {name}: finite-in-foot={int(np.isfinite(a[foot]).sum())}")

    # ---- band identity diagnostics (subsampled Spearman) ----
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(np.flatnonzero(foot), size=200_000, replace=False)

    def spear(x, y):
        xv = np.asarray(x, dtype=np.float64).ravel()[sample_idx]
        yv = np.asarray(y, dtype=np.float64).ravel()[sample_idx]
        m = np.isfinite(xv) & np.isfinite(yv)
        return float(stats.spearmanr(xv[m], yv[m]).statistic)

    dem_f = F.fill_median(raw["det_elev"], foot)
    rtp_f = F.fill_median(raw["rtp"], foot)
    rtp_n = F.fill_nearest(raw["rtp"])
    grav_f = F.fill_median(raw["iso_grav_anom"], foot)
    grav_n = F.fill_nearest(raw["iso_grav_anom"])
    tmi_f = F.fill_median(raw["tmi"], foot)
    depth_f = F.fill_median(raw["depth_to_base_surf"], foot)

    ggy, ggx = np.gradient(grav_f)
    tgy, tgx = np.gradient(tmi_f)
    dgy, dgx = np.gradient(dem_f)
    # FFT vertical derivative of TMI (|k| multiplier) to identify band 9
    FT = np.fft.rfft2(np.pad(tmi_f.astype(np.float64), PAD, mode="reflect"))
    pr, pc_ = FT.shape[0], tmi_f.shape[1] + 2 * PAD
    fx = np.fft.rfftfreq(pc_)[None, :]; fy = np.fft.fftfreq(pr)[:, None]
    kk = 2 * np.pi * np.sqrt(fx * fx + fy * fy)
    tvd = np.fft.irfft2(FT * kk, s=(pr, pc_))[PAD:-PAD, PAD:-PAD].astype(np.float32)
    del FT, kk

    tc_ext = read(RAW / "external" / "geodawn_rad_u8.tif", 4).astype(np.float32)
    tilt_candidate = np.degrees(np.arctan2(np.nan_to_num(raw["tmi_vg"]),
                                           np.maximum(np.nan_to_num(raw["tmi_hg"]), 1e-12)))
    diag = {
        "band6_tag_claim": "tc - Tilt angle or total curvature - magnetic field derivative for edge detection",
        "band6_spearman_vs_external_TC_radiometric": spear(raw["tc_totalcount"], tc_ext),
        "band6_spearman_vs_derived_tilt_deg": spear(raw["tc_totalcount"], tilt_candidate),
        "band6_range_in_footprint": [float(np.nanmin(raw["tc_totalcount"][foot])),
                                     float(np.nanmax(raw["tc_totalcount"][foot]))],
        "band6_conclusion": "radiometric total count (tag is wrong); shipped stack has no tilt band",
        "band18_spearman_vs_grav_dx": spear(raw["iso_grav_anom_hg_signed"], ggx),
        "band18_spearman_vs_grav_dy": spear(raw["iso_grav_anom_hg_signed"], ggy),
        "band18_spearman_vs_grav_gradmag": spear(raw["iso_grav_anom_hg_signed"], np.hypot(ggx, ggy)),
        "band18_min_in_footprint": float(np.nanmin(raw["iso_grav_anom_hg_signed"][foot])),
        "band3_spearman_vs_tmi_gradmag": spear(raw["tmi_hg"], np.hypot(tgx, tgy)),
        "band9_spearman_vs_fft_vertical_derivative_tmi": spear(raw["tmi_vg"], tvd),
        "band19_spearman_vs_dem_gradmag": spear(raw["det_elev_slope"], np.hypot(dgx, dgy)),
        "n_sample": int(sample_idx.size),
    }
    log("diagnostics:", json.dumps(diag, indent=None)[:400], "...")

    # ---- terrain block ----
    def put(name, a):
        out = np.where(foot, np.asarray(a, np.float32), np.float32(np.nan)).astype(np.float32)
        stack[:, :, F.IDX[name]] = out
        log(f"channel {name}: done ({time.time()-t0:.0f}s)")

    put("dem_curv1", ndimage.gaussian_laplace(dem_f, 1.0, mode="nearest"))
    put("dem_curv25", ndimage.gaussian_laplace(dem_f, 2.5, mode="nearest"))
    for nm, sg in (("dem_ridge1", 1.0), ("dem_ridge2", 2.0), ("dem_ridge3p5", 3.5)):
        put(nm, F.hessian_ridge(dem_f, sg))
    put("dem_relief5", ndimage.maximum_filter(dem_f, 5) - ndimage.minimum_filter(dem_f, 5))
    put("dem_tophat5", dem_f - ndimage.grey_opening(dem_f, size=(5, 5)))
    coh = F.structure_tensor(dem_f, 2.0)
    put("st_coherence", coh)
    put("profcurv_signed", F.profile_curvature(dem_f, 1.0))

    # ---- potential-field basics ----
    hg = np.maximum(F.fill_median(raw["tmi_hg"], foot), 0.0)
    vg = F.fill_median(raw["tmi_vg"], foot)
    mag_tilt = np.arctan2(vg, np.maximum(hg, 1e-9)).astype(np.float32)
    mag_as = np.sqrt(hg * hg + vg * vg)
    grav_thdr = np.hypot(ggx, ggy).astype(np.float32)
    gvg = F.fill_median(raw["iso_grav_anom_vg"], foot)
    grav_tilt = np.arctan2(gvg, np.maximum(grav_thdr, 1e-12)).astype(np.float32)
    grav_as = np.sqrt(grav_thdr * grav_thdr + gvg * gvg)
    put("mag_tilt", mag_tilt)
    put("mag_tilt_abs", np.abs(mag_tilt))
    put("mag_as_log", np.log1p(F._sm(mag_as, 1.0)))
    put("grav_thdr_log", np.log1p(F._sm(grav_thdr, 1.0)))
    put("grav_tilt", grav_tilt)
    put("grav_as_log", np.log1p(F._sm(grav_as, 1.0)))

    r_mag = F.rank01(np.log1p(mag_as), foot)
    r_grav = F.rank01(np.log1p(grav_as), foot)
    wz_mag = 1.0 - np.abs(mag_tilt) / (np.pi / 2)
    wz_grav = 1.0 - np.abs(grav_tilt) / (np.pi / 2)
    put("contact_confluence", np.minimum(r_mag, r_grav) * np.sqrt(np.maximum(wz_mag * wz_grav, 0.0)))
    del r_mag, r_grav, wz_mag, wz_grav

    depth_sm = F._sm(depth_f, 2.0)
    dpy, dpx = np.gradient(depth_sm)
    rpy, rpx = np.gradient(F._sm(rtp_f, 1.0))
    nrm = np.hypot(rpx, rpy) + 1e-12
    put("base_step_along_maggrad", (dpx * rpx + dpy * rpy) / nrm)
    put("base_grad_log", np.log1p(np.hypot(dpx, dpy)))
    del depth_sm, dpy, dpx, rpy, rpx, nrm

    # ---- front-base asymmetry ----
    slope_sm = F._sm(F.fill_median(raw["det_elev_slope"], foot), 1.0)
    front = (ndimage.maximum_filter(slope_sm, 5) - slope_sm) * (0.25 + 0.75 * coh)
    put("frontness", front)
    del slope_sm, front, coh

    # ---- free everything the FFT stages do not need (4 GB sandbox) ----
    import gc
    for _nm in ("raw", "dem_f", "rtp_f", "grav_f", "tmi_f", "depth_f", "ggx", "ggy",
                "tgx", "tgy", "dgx", "dgy", "tvd", "tc_ext", "tilt_candidate", "hg",
                "vg", "gvg", "mag_tilt", "mag_as", "grav_thdr", "grav_tilt", "grav_as"):
        globals().pop(_nm, None); locals().pop(_nm, None)
    del raw, dem_f, rtp_f, grav_f, tmi_f, depth_f, ggx, ggy, tgx, tgy, dgx, dgy
    del tvd, tc_ext, tilt_candidate, hg, vg, gvg, mag_tilt, mag_as, grav_thdr, grav_tilt, grav_as
    gc.collect()
    log("freed intermediates before FFT stages")

    # ---- phase congruency (FFT, one field at a time) ----
    put("pc_mag", padcrop(lambda a: F.phase_congruency(a.astype(np.float64)), rtp_n))
    put("pc_grav", padcrop(lambda a: F.phase_congruency(a.astype(np.float64)), grav_n))

    # ---- pseudogravity / Poisson ----
    pg = padcrop(F.pseudogravity, rtp_n)
    pg_hp = (pg - F._sm(pg, 15.0)).astype(np.float32)
    g_hp = (grav_n - F._sm(grav_n, 15.0)).astype(np.float32)
    slope0 = float(np.cov(pg_hp.ravel()[sample_idx], g_hp.ravel()[sample_idx])[0, 1] /
                   np.var(pg_hp.ravel()[sample_idx]))
    resid = (g_hp - slope0 * pg_hp).astype(np.float32)
    lslope, lcorr = F.local_regression(pg_hp, g_hp, size=25)
    put("pg_slope", lslope)
    put("pg_corr", lcorr)
    ry, rx = np.gradient(F._sm(resid, 1.0))
    put("pg_resid_grad_log", np.log1p(np.hypot(rx, ry)))
    diag["pseudogravity_global_slope"] = slope0
    del pg, pg_hp, g_hp, resid, lslope, lcorr, ry, rx

    # ---- radiometrics / lidar (uint8 quantised; 0 outside survey -> NaN outside footprint) ----
    for nm, path, band in (("rad_K", "external/geodawn_rad_u8.tif", 1),
                           ("rad_Th", "external/geodawn_rad_u8.tif", 2),
                           ("rad_U", "external/geodawn_rad_u8.tif", 3),
                           ("rad_TC", "external/geodawn_rad_u8.tif", 4),
                           ("rad_ThK", "external/geodawn_extensions_u8.tif", 1),
                           ("rad_UK", "external/geodawn_extensions_u8.tif", 2),
                           ("lidar_ex_max", "external/lidar_scarp_features_u8.tif", 1),
                           ("lidar_step_max", "external/lidar_scarp_features_u8.tif", 3),
                           ("lidar_downface_max", "external/lidar_scarp_features_u8.tif", 6),
                           ("lidar_coh100", "external/lidar_scarp_features_u8.tif", 10),
                           ("lidar_valid", "external/lidar_scarp_features_u8.tif", 12)):
        a = read(RAW / path, band).astype(np.float32) / 255.0
        put(nm, a)

    stack.flush()
    del stack
    receipt = {
        "path": str(OUT), "shape": [H, W, len(F.CHANNELS)], "dtype": "float32",
        "channels": F.CHANNELS, "blocks": F.BLOCKS,
        "footprint_px": int(foot.sum()), "build_seconds": round(time.time() - t0, 1),
        "note": "NaN outside footprint and at sentinel cells; filters computed on filled copies.",
    }
    (ROOT / "registry" / "feature_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "band_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    # band6 resolution receipt (subset, referenced by preregistration)
    b6 = {k: v for k, v in diag.items() if k.startswith("band6")}
    (ROOT / "evidence" / "band6_resolution.json").write_text(json.dumps(b6, indent=2) + "\n")
    log(f"DONE in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
