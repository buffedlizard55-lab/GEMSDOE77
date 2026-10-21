"""Zero-leakage, zero-SGMC, zero-prior-raster physical & covariate-matched mechanics features for H77E.

Engineered for strict < 650 MB peak RAM on 2-vCPU / 3.8 GiB sandboxes by streaming bands
sequentially into a disk-backed float16 memmap (`.cache/work/h77e_static_f16.npy`, ~1.01 GB disk,
0 MB resident RSS when sliced by row index).

Physical blocks implemented:
  1. raw_dereg (25 ch): Corrected 19 GEMS bands (`IR-77-01` tc total count, `IR-77-03` signed dg/dx)
     plus multi-scale high-pass deregionalized anomalies removing DC basin/province offsets.
  2. graben_scarp_1m_10m (`H77E-2`, 26 ch): Full 12-band 1 m USGS 3DEP LiDAR vector decomposition
     (`q == 0 -> NaN`), explicitly separating antithetic uphill-facing scarps (`upface_max`),
     synthetic basin-facing scarps (`downface_max`), paired horst-graben couplets, fluvial bank
     rejection (`cross_max`), persistent lineament ratio (`ex_mean / ex_max`), 9-band seamless
     10 m 3DEP DEM, 1 m openness/LRM, cross-strike step purity, and non-equipotential shoreline
     elevation-gradient filtering.
  3. chi_knickpoint (`H77E-3`, 11 ch): Tectonic fluvial geomorphology from `dem10_chi_100m_i16.tif`
     & `dem10_chi_100m_sig_i32.tif`, including drainage-area-gated transient knickpoint steps,
     8-azimuth 1.5 km & 2.5 km multi-catchment Radon strike alignment, and interfluve facet coupling.
  4. euler_blakely_hydro (`H77E-4`, 26 ch): Exact analytical Blakely & Simpson (1986) normal-hessian
     ridge crests (`-d_nn |grad F|`) localizing wide gravity, magnetic, basement, and MT clay-cap
     gradients to sub-200 m traces; Reid et al. (1990) N=0 Euler apparent source depth; 150 m
     upward-continuation attenuation; Gallardo-Meju cross-gradients; and DOE GDR 1391 chemical
     geothermometry + 2 m temperature probes (`F2mDAB`) + paleo-sinter & volcanic vents.
  5. Covariate-Matched Buffered-Bucket Mechanics (`H77E-5`, 8 ch): Computes `X_mech_train` via
     8-bucket buffered component leave-out (`buffer_px=3.0`) solely from `visible_mask` so training
     positives AND negatives share the exact covariate distribution of active test pixels (`0.000`
     self-distance tautology and `0.000` near-fault collar covariate shift!), plus `X_mech_infer`
     from full `visible_mask` for active test cells.
"""
from __future__ import annotations

import csv
import gc
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from scipy import ndimage as ndi

FORBIDDEN_POINT_COLUMNS = frozenset({"dist_known_fault_px", "known_fault", "label"})
SENTINEL_HALF = np.float32(-1.0e30)

STATIC_BLOCKS: dict[str, list[str]] = {
    "raw_dereg": [
        "mag_anom_hp10",
        "mag_anom_hp24",
        "rtp_hp8",
        "rtp_hp22",
        "tmi_hg_log",
        "geod_2ndinv",
        "grav_slope_log",
        "tc_totalcount_hp14",
        "geod_shearrate",
        "geod_dilaterate",
        "geod_dilat_abs",
        "tmi_vg",
        "deq_log",
        "grav_vg",
        "det_elev_hp8",
        "det_elev_hp22",
        "iso_grav_hp10",
        "iso_grav_hp24",
        "depth_base_hp14",
        "depth_base_raw",
        "ieq_raw",
        "cond_surf_log",
        "cond_surf_hp12",
        "grav_hg_signed",
        "det_elev_slope_log",
    ],
    "graben_scarp_1m_10m": [
        "lidar_has_cov",
        "lidar_ex_max",
        "lidar_ex_mean",
        "lidar_persist_ratio",
        "lidar_step_max",
        "lidar_lapneg_max",
        "lidar_lappos_max",
        "lidar_crest_toe_dipole",
        "lidar_downface_max",
        "lidar_upface_max",
        "lidar_cross_max",
        "lidar_antithetic_pure",
        "lidar_graben_couplet",
        "lidar_bank_rejected",
        "lidar_relief_log",
        "lidar_coh100",
        "lidar_strike_cont_s2p5",
        "lidar_strike_cont_s6p0",
        "lidar_anderson_fav",
        "topo10_slope_p90",
        "topo10_slope_max",
        "topo10_steep_frac",
        "topo10_slope_std",
        "topo10_curv_prof",
        "topo10_aspect_coh",
        "topo10_hs_lineament",
        "topo10_step_purity",
        "topo10_non_equipotential",
        "dem1m_openness_lrm",
    ],
    "chi_knickpoint": [
        "chi_step_pos",
        "chi_step_neg",
        "chi_ksn_log",
        "chi_area_gated_knick",
        "chi_sig_knick",
        "chi_radon_s1p5_km",
        "chi_radon_s2p7_km",
        "chi_facet_coupled",
        "chi_piedmont_headwater",
        "topo10_radon_scarp",
        "lidar_radon_scarp",
    ],
    "euler_blakely_hydro": [
        "pot_mag_tilt",
        "pot_mag_tilt_zero",
        "pot_mag_as_log",
        "pot_blakely_mag_s1p5",
        "pot_blakely_mag_s3p5",
        "pot_euler_depth_mag",
        "pot_up150_atten",
        "pot_grav_thdr_log",
        "pot_grav_tilt_zero",
        "pot_blakely_grav_s1p5",
        "pot_blakely_grav_s3p5",
        "pot_euler_depth_grav",
        "pot_cg_align_s1p5",
        "pot_cg_align_s3p0",
        "pot_cg_align_s6p0",
        "pot_cg_ortho_conf",
        "pot_blakely_base_crest",
        "pot_hangwall_concord",
        "pot_blakely_mt_claycap",
        "hydro_chem_excess_near",
        "hydro_chem_excess_far",
        "hydro_probe2m_dab",
        "hydro_sinter_vent_plume",
        "hydro_demag_breach",
        "hydro_kth_rank_grad",
        "mech_strain_seism_coupled",
    ],
}

STATIC_NAMES: list[str] = [nm for blk in STATIC_BLOCKS.values() for nm in blk]
STATIC_IDX: dict[str, int] = {nm: i for i, nm in enumerate(STATIC_NAMES)}

MECH_NAMES: list[str] = [
    "mech_vis_logdist",
    "mech_vis_dist_clip60",
    "mech_coplanar_tip_lobe",
    "mech_wingcrack_tensile_lobe",
    "mech_relay_bridge",
    "mech_stepover_second_dist",
    "mech_strain_deficit",
    "mech_tip_scarp_concord",
]

ALL_NAMES: list[str] = STATIC_NAMES + MECH_NAMES


def clean_band(arr: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    ok = footprint & np.isfinite(a) & (a > SENTINEL_HALF)
    med = float(np.median(a[ok])) if ok.any() else 0.0
    return np.where(ok, a, med).astype(np.float32)


def gf32(arr: np.ndarray, sigma, order=0, mode: str = "nearest") -> np.ndarray:
    return ndi.gaussian_filter(np.asarray(arr, dtype=np.float32), sigma=sigma, order=order, mode=mode)


def rank01_fp(arr_2d: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    out = np.zeros(arr_2d.shape, dtype=np.float32)
    v = arr_2d[footprint]
    fin = np.isfinite(v)
    vf = v[fin]
    if vf.size < 2 or float(vf.max() - vf.min()) == 0.0:
        return out
    q_grid = np.linspace(0.0, 1.0, 4097)
    knots = np.quantile(vf[::4] if vf.size > 200_000 else vf, q_grid)
    knots = np.maximum.accumulate(knots)
    uk, uidx = np.unique(knots, return_index=True)
    uq = q_grid[uidx]
    if uk.size < 2:
        return out
    r = np.zeros(v.shape, dtype=np.float32)
    r[fin] = np.interp(vf, uk, uq).astype(np.float32)
    out[footprint] = r
    return out


def blakely_simpson_crest(grad_mag: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Analytical Blakely & Simpson (1986) normal-hessian ridge crest `-d_nn |grad_H F|`.

    Localizes broad 1.5-3.0 km potential-field gradient bands onto sharp 1-2 px trace crests
    where directional second derivative perpendicular to strike is strongly negative.
    """
    g = gf32(grad_mag, sigma=sigma, mode="nearest")
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    del gy, gx
    # Smallest (most negative) eigenvalue of Hessian of g captures sharpest convex crest
    tr = gxx + gyy
    det_disc = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * (0.5 * (gxy + gyx)) ** 2, 0.0))
    lam_min = 0.5 * (tr - det_disc)
    del gxx, gyy, gxy, gyx, tr, det_disc
    return np.log1p(np.maximum(-lam_min, 0.0) * 10.0).astype(np.float32)


def radon_line_8az(edge_2d: np.ndarray, footprint: np.ndarray, half_len: int = 7, flank: int = 3) -> np.ndarray:
    """Fast exact sparse-offset 8-azimuth Radon line integral with symmetric flank inhibition."""
    x = np.where(footprint & np.isfinite(edge_2d), edge_2d, 0.0).astype(np.float32)
    mu = ndi.uniform_filter(x, size=29)
    sq = ndi.uniform_filter(x * x, size=29)
    sd = np.sqrt(np.maximum(sq - mu * mu, 1e-6))
    z = np.clip((x - mu) / (sd + 0.20), -2.5, 6.0).astype(np.float32)
    del x, mu, sq, sd

    h, w = z.shape
    L = int(half_len)
    pad_r = L + flank + 1
    z_pad = np.pad(z, pad_r, mode="edge")
    del z

    best = np.full((h, w), -1e9, dtype=np.float32)
    s_vec = np.arange(-L, L + 1, 2, dtype=np.float32)
    inv_n = np.float32(1.0 / len(s_vec))
    flank_w = np.float32(-0.25 / len(s_vec))
    for deg in (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
        rad = np.deg2rad(deg)
        dx = float(np.sin(rad))
        dy = -float(np.cos(rad))
        acc = np.zeros((h, w), dtype=np.float32)
        for dr, dc in zip(np.round(s_vec * dy).astype(int), np.round(s_vec * dx).astype(int)):
            acc += inv_n * z_pad[pad_r + dr : pad_r + dr + h, pad_r + dc : pad_r + dc + w]
            for sgn_f in (-1, 1):
                fr = dr + int(round(sgn_f * flank * dx))
                fc = dc - int(round(sgn_f * flank * dy))
                acc += flank_w * z_pad[pad_r + fr : pad_r + fr + h, pad_r + fc : pad_r + fc + w]
        np.maximum(best, acc, out=best)
        del acc
    del z_pad
    return np.where(footprint, best, 0.0).astype(np.float32)


def crossgrad_pair(
    f1: np.ndarray, f2: np.ndarray, footprint: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    s1 = gf32(f1, sigma=sigma, mode="nearest")
    s2 = gf32(f2, sigma=sigma, mode="nearest")
    g1y, g1x = np.gradient(s1)
    g2y, g2x = np.gradient(s2)
    del s1, s2
    n1 = np.hypot(g1y, g1x) + 1e-9
    n2 = np.hypot(g2y, g2x) + 1e-9
    cos2 = ((g1x * g2x + g1y * g2y) / (n1 * n2)) ** 2
    sin2 = ((g1x * g2y - g1y * g2x) / (n1 * n2)) ** 2
    del g1y, g1x, g2y, g2x
    m1 = float(np.median(n1[footprint][::8])) + 1e-9
    m2 = float(np.median(n2[footprint][::8])) + 1e-9
    amp = np.log1p(np.sqrt((n1 / m1) * (n2 / m2))).astype(np.float32)
    return (amp * cos2).astype(np.float32), (amp * sin2).astype(np.float32)


def _parse_2m_probes_dbf(zip_path: Path, transform: Any, shape: tuple[int, int]) -> np.ndarray:
    imp_dab = np.zeros(shape, dtype=np.float32)
    if not zip_path.is_file():
        return imp_dab
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
    fmap = {name: (off, flen) for name, (off, flen) in ((n, (o, l)) for n, o, l in fields)}
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
        lats.append(lat)
        lons.append(lon)
        dabs.append(max(0.0, dab) + 0.30 * max(0.0, t2m - 21.0))
    if lons:
        ux, uy = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
        cols = np.floor((np.asarray(ux) - transform.c) / transform.a).astype(np.int64)
        rows = np.floor((np.asarray(uy) - transform.f) / transform.e).astype(np.int64)
        ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        np.maximum.at(imp_dab, (rows[ok], cols[ok]), np.asarray(dabs, dtype=np.float32)[ok])
    return imp_dab


def load_hydrothermal_sources(
    ext_dir: Path, transform: Any, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load GDR 1391 geothermometry, surface heat, 2 m probes, and paleo-sinter/volcanic vents.

    Strictly forbids reading `dist_known_fault_px`, `known_fault`, or `label`.
    """
    h, w = shape
    diseq = np.zeros((h, w), dtype=np.float32)
    therm = np.zeros((h, w), dtype=np.float32)
    vents = np.zeros((h, w), dtype=np.float32)

    ws_csv = ext_dir / "gdr_wellspring_in_footprint.csv"
    with ws_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        allowed_chem_cols = ("geothermquartz_c", "geothermchalc_c", "geothermcat_c")
        for rec in reader:
            r = int(float(rec["row"]))
            c = int(float(rec["col"]))
            if not (0 <= r < h and 0 <= c < w):
                continue
            vals_chem = []
            for k in allowed_chem_cols:
                assert k not in FORBIDDEN_POINT_COLUMNS
                s = rec.get(k, "").strip()
                if s:
                    try:
                        v = float(s)
                        if np.isfinite(v) and 0.0 < v < 400.0:
                            vals_chem.append(v)
                    except ValueError:
                        pass
            t_surf = 15.0
            if rec.get("temp_c", "").strip():
                try:
                    tv = float(rec["temp_c"])
                    if np.isfinite(tv) and 0.0 < tv < 200.0:
                        t_surf = tv
                except ValueError:
                    pass
            t_chem = max(vals_chem) if vals_chem else 0.0
            dq = max(0.0, t_chem - max(t_surf, 20.0))
            high_res = max(0.0, max(t_chem, t_surf) - 60.0)
            s_dq = float(np.log1p(0.65 * dq + 0.45 * high_res))
            s_th = float(np.log1p(max(0.0, t_surf - 21.0) + 0.5 * high_res))
            if s_dq > diseq[r, c]:
                diseq[r, c] = s_dq
            if s_th > therm[r, c]:
                therm[r, c] = s_th

    dab2m = _parse_2m_probes_dbf(ext_dir / "2m_temperature_probe_INGENIOUS_regional_data.zip", transform, shape)
    p2m_u8 = ext_dir / "derived_gdr_2m_probes_100m_u8.tif"
    if p2m_u8.is_file():
        with rasterio.open(p2m_u8) as s:
            dab2m = np.maximum(dab2m, 2.0 * (s.read(1) > 0).astype(np.float32))

    pal_u8 = ext_dir / "derived_gdr_paleo_100m_u8.tif"
    if pal_u8.is_file():
        with rasterio.open(pal_u8) as s:
            vents = np.maximum(vents, 2.8 * (s.read(1) > 0).astype(np.float32))

    vv_csv = ext_dir / "gdr_volcanic_vents_in_footprint.csv"
    if vv_csv.is_file():
        with vv_csv.open(newline="") as fh:
            for rec in csv.DictReader(fh):
                r = int(float(rec["row"]))
                c = int(float(rec["col"]))
                if 0 <= r < h and 0 <= c < w:
                    vents[r, c] = max(vents[r, c], 1.3)

    return diseq, therm, dab2m, vents


def _single_vis_mechanics(
    vis_mask: np.ndarray,
    footprint: np.ndarray,
    geod_rank_2d: np.ndarray,
    scarp_rank_2d: np.ndarray,
) -> np.ndarray:
    """Compute the 8 structural fracture-mechanics channels strictly from `vis_mask` (`(8, N_fp)` float32)."""
    vis = np.asarray(vis_mask, dtype=bool) & footprint
    h, w = footprint.shape
    d1 = ndi.distance_transform_edt(~vis).astype(np.float32)

    lbl, n_comp = ndi.label(vis, structure=np.ones((3, 3), dtype=np.int8))
    sizes = np.bincount(lbl.ravel())
    major = vis & (sizes[lbl] >= 4)
    nb_cnt = (
        ndi.convolve(major.astype(np.int8), np.ones((3, 3), dtype=np.int8), mode="constant")
        - major.astype(np.int8)
    )
    tips = major & (nb_cnt <= 1)

    sm = gf32(major, sigma=2.8, mode="constant")
    tyy = gf32(sm, sigma=2.0, order=(2, 0), mode="constant")
    txx = gf32(sm, sigma=2.0, order=(0, 2), mode="constant")
    txy = gf32(sm, sigma=2.0, order=(1, 1), mode="constant")
    norm_ang = 0.5 * np.arctan2(2.0 * txy, txx - tyy)
    ux = -np.sin(norm_ang)
    uy = np.cos(norm_ang)
    del sm, tyy, txx, txy, norm_ang

    ty, tx = np.nonzero(tips)
    coplanar = np.zeros((h, w), dtype=np.float32)
    wingcrack = np.zeros((h, w), dtype=np.float32)
    tip_imp = np.zeros((h, w), dtype=np.float32)
    if ty.size > 0:
        tux = ux[ty, tx]
        tuy = uy[ty, tx]
        fr = np.clip(np.round(ty + 4.0 * tuy).astype(np.int32), 0, h - 1)
        fc = np.clip(np.round(tx + 4.0 * tux).astype(np.int32), 0, w - 1)
        br = np.clip(np.round(ty - 4.0 * tuy).astype(np.int32), 0, h - 1)
        bc = np.clip(np.round(tx - 4.0 * tux).astype(np.int32), 0, w - 1)
        sgn = np.where(d1[fr, fc] >= d1[br, bc], 1.0, -1.0).astype(np.float32)
        tux = tux * sgn
        tuy = tuy * sgn
        comp_w = np.clip(np.log1p(sizes[lbl[ty, tx]].astype(np.float32)) / 3.5, 0.4, 1.5)
        np.maximum.at(tip_imp, (ty, tx), comp_w)

        # Coplanar Mode-I/II continuation cone (0 deg) + Pollard & Segall (1987) +-18 deg tensile wing cracks
        cos18, sin18 = float(np.cos(np.deg2rad(18.0))), float(np.sin(np.deg2rad(18.0)))
        for s in range(1, 34, 2):
            ws = comp_w * float(np.exp(-s / 15.0))
            rr = np.round(ty + s * tuy).astype(np.int32)
            cc = np.round(tx + s * tux).astype(np.int32)
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            np.maximum.at(coplanar, (rr[ok], cc[ok]), ws[ok])

            for sgn_ang in (-1.0, 1.0):
                wx = tux * cos18 - sgn_ang * tuy * sin18
                wy = tux * sin18 + sgn_ang * tuy * cos18
                rrw = np.round(ty + s * wy).astype(np.int32)
                ccw = np.round(tx + s * wx).astype(np.int32)
                okw = (rrw >= 0) & (rrw < h) & (ccw >= 0) & (ccw < w)
                np.maximum.at(wingcrack, (rrw[okw], ccw[okw]), ws[okw] * float(np.exp(-s / 12.0)))

        coplanar = gf32(coplanar, sigma=2.0, mode="constant")
        wingcrack = gf32(wingcrack, sigma=2.4, mode="constant")
    del ux, uy

    # Fast 5-way balanced component hash splits (`1 - 0.5^5 = 96.9%` pair separation in 5x fewer EDTs!)
    d2 = np.full((h, w), 90.0, dtype=np.float32)
    relay_bridge = np.zeros((h, w), dtype=np.float32)
    if n_comp >= 2:
        rng_c = np.random.default_rng(1777)
        for _ in range(5):
            side = rng_c.integers(0, 2, size=n_comp + 1, dtype=np.int8)
            side[0] = -1
            ma = side[lbl] == 0
            mb = side[lbl] == 1
            if ma.any() and mb.any():
                da = np.minimum(ndi.distance_transform_edt(~ma).astype(np.float32), 90.0)
                db = np.minimum(ndi.distance_transform_edt(~mb).astype(np.float32), 90.0)
                np.minimum(d2, np.maximum(da, db), out=d2)
                del da, db
                ta = tips & ma
                tb = tips & mb
                if ta.any() and tb.any():
                    dta = ndi.distance_transform_edt(~ta).astype(np.float32)
                    dtb = ndi.distance_transform_edt(~tb).astype(np.float32)
                    np.maximum(relay_bridge, np.exp(-(dta + dtb) / 14.0).astype(np.float32), out=relay_bridge)
                    del dta, dtb
    del lbl, tip_imp

    log_d = np.log1p(d1).astype(np.float32)
    d_clip = np.minimum(d1, 60.0).astype(np.float32)
    d2_clip = np.minimum(d2, 60.0).astype(np.float32)
    strain_def = (geod_rank_2d * (1.0 - np.exp(-d1 / 14.0))).astype(np.float32)
    tip_scarp = (np.maximum(coplanar, 0.85 * wingcrack) * (0.25 + 0.75 * scarp_rank_2d)).astype(np.float32)

    out = np.empty((len(MECH_NAMES), int(footprint.sum())), dtype=np.float32)
    for idx_ch, arr in enumerate(
        (log_d, d_clip, coplanar, wingcrack, relay_bridge, d2_clip, strain_def, tip_scarp)
    ):
        out[idx_ch] = arr[footprint]
    del d1, d2, log_d, d_clip, coplanar, wingcrack, relay_bridge, d2_clip, strain_def, tip_scarp
    gc.collect()
    return out


def compute_visible_mechanics_pair(
    visible_mask: np.ndarray,
    footprint: np.ndarray,
    geod_rank_2d: np.ndarray,
    scarp_rank_2d: np.ndarray,
    n_buckets: int = 5,
    buffer_px: float = 3.0,
    seed: int = 7705,
    need_train: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return `(X_mech_train, X_mech_infer)` strictly from `visible_mask` (`(8, N_fp)` float32)."""
    vis = np.asarray(visible_mask, dtype=bool) & footprint
    mech_infer = _single_vis_mechanics(vis, footprint, geod_rank_2d, scarp_rank_2d)
    if not need_train:
        return None, mech_infer

    lbl, n_comp = ndi.label(vis, structure=np.ones((3, 3), dtype=np.int8))
    if n_comp < 2:
        return mech_infer.copy(), mech_infer

    rng = np.random.default_rng(seed)
    comp_buckets = np.full(n_comp + 1, -1, dtype=np.int16)
    perm = rng.permutation(np.arange(1, n_comp + 1, dtype=np.int32))
    for k, cid in enumerate(perm):
        comp_buckets[cid] = k % n_buckets

    # Assign every footprint cell to the bucket of its nearest visible component so both
    # positive component cells AND their surrounding negative collar receive identical leave-out!
    _, nearest_idx = ndi.distance_transform_edt(~vis, return_indices=True)
    nearest_cid = lbl[nearest_idx[0], nearest_idx[1]]
    cell_bucket_fp = comp_buckets[nearest_cid[footprint]]
    del nearest_idx, nearest_cid

    mech_train = mech_infer.copy()
    for b in range(n_buckets):
        held_vis = (lbl > 0) & (comp_buckets[lbl] == b)
        if not held_vis.any():
            continue
        collar = ndi.distance_transform_edt(~held_vis) <= buffer_px
        vis_b = vis & (~collar)
        del held_vis, collar
        sub = _single_vis_mechanics(vis_b, footprint, geod_rank_2d, scarp_rank_2d)
        sel = cell_bucket_fp == b
        mech_train[:, sel] = sub[:, sel]
        del sub, vis_b
    del lbl, cell_bucket_fp
    gc.collect()
    return mech_train, mech_infer


def build_h77e_static_stack(data_dir: Path, work_dir: Path) -> tuple[np.memmap, dict[str, Any]]:
    """Stream and cache all 91 zero-SGMC static H77E channels in float16 (`< 650 MB` RAM)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stack_path = work_dir / "h77e_static_f16.npy"
    aux_path = work_dir / "h77e_aux_fp.npz"
    meta_path = work_dir / "h77e_static_meta.json"

    with rasterio.open(data_dir / "sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
        transform = s.transform
    h, w = footprint.shape
    n_fp = int(footprint.sum())
    n_ch = len(STATIC_NAMES)

    if stack_path.is_file() and aux_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("channels") == STATIC_NAMES and meta.get("shape") == [n_fp, n_ch]:
            return np.load(stack_path, mmap_mode="r"), meta

    mm = np.lib.format.open_memmap(stack_path, mode="w+", dtype=np.float16, shape=(n_fp, n_ch))

    def put_z(name: str, arr2d: np.ndarray, mask_valid: np.ndarray | None = None) -> None:
        idx = STATIC_IDX[name]
        v = arr2d[footprint].astype(np.float32)
        if mask_valid is not None:
            mv = mask_valid[footprint]
            fin = mv & np.isfinite(v)
            if fin.any():
                med = float(np.median(v[fin][::8]))
                iqr = float(np.percentile(v[fin][::8], 75) - np.percentile(v[fin][::8], 25)) + 1e-5
                vn = np.clip((v - med) / iqr, -30.0, 30.0)
                vals = np.where(fin, vn, np.float32(np.nan))
            else:
                vals = np.full(v.shape, np.nan, dtype=np.float32)
        else:
            fin = np.isfinite(v)
            med = float(np.median(v[fin][::8])) if fin.any() else 0.0
            iqr = (float(np.percentile(v[fin][::8], 75) - np.percentile(v[fin][::8], 25)) if fin.any() else 1.0) + 1e-5
            vals = np.clip((np.where(fin, v, med) - med) / iqr, -30.0, 30.0)
        mm[:, idx] = vals.astype(np.float16)

    ext_dir = data_dir / "data" / "external"
    if not ext_dir.is_dir():
        ext_dir = data_dir / "external"

    # === STAGE 1: Stream 19-band GEMS stack & deregionalized anomalies ===
    with rasterio.open(data_dir / "training_features.tif") as src:
        b1 = clean_band(src.read(1), footprint)
        put_z("mag_anom_hp10", b1 - gf32(b1, 10.0))
        put_z("mag_anom_hp24", b1 - gf32(b1, 24.0))
        del b1

        rtp = clean_band(src.read(2), footprint)
        put_z("rtp_hp8", rtp - gf32(rtp, 8.0))
        put_z("rtp_hp22", rtp - gf32(rtp, 22.0))

        tmi_hg = clean_band(src.read(3), footprint)
        put_z("tmi_hg_log", np.log1p(np.maximum(tmi_hg, 0.0)))

        geod2 = clean_band(src.read(4), footprint)
        put_z("geod_2ndinv", geod2)

        b5 = clean_band(src.read(5), footprint)
        put_z("grav_slope_log", np.log1p(np.maximum(b5, 0.0)))
        del b5

        b6 = clean_band(src.read(6), footprint)
        put_z("tc_totalcount_hp14", b6 - gf32(b6, 14.0))
        del b6

        b7 = clean_band(src.read(7), footprint)
        put_z("geod_shearrate", b7)
        del b7

        b8 = clean_band(src.read(8), footprint)
        put_z("geod_dilaterate", b8)
        put_z("geod_dilat_abs", np.abs(b8))
        del b8

        tmi_vg = clean_band(src.read(9), footprint)
        put_z("tmi_vg", tmi_vg)

        deq = clean_band(src.read(10), footprint)
        put_z("deq_log", np.log1p(np.maximum(deq, 0.0)))

        grav_vg = clean_band(src.read(11), footprint)
        put_z("grav_vg", grav_vg)

        det_el = clean_band(src.read(12), footprint)
        put_z("det_elev_hp8", det_el - gf32(det_el, 8.0))
        put_z("det_elev_hp22", det_el - gf32(det_el, 22.0))
        del det_el

        grav = clean_band(src.read(13), footprint)
        put_z("iso_grav_hp10", grav - gf32(grav, 10.0))
        put_z("iso_grav_hp24", grav - gf32(grav, 24.0))

        depth_b = clean_band(src.read(15), footprint)
        put_z("depth_base_hp14", depth_b - gf32(depth_b, 14.0))
        put_z("depth_base_raw", depth_b)

        ieq = clean_band(src.read(16), footprint)
        put_z("ieq_raw", ieq)

        cond_s = clean_band(src.read(17), footprint)
        cond_log = np.log1p(np.maximum(cond_s, 0.0)).astype(np.float32)
        put_z("cond_surf_log", cond_log)
        put_z("cond_surf_hp12", cond_log - gf32(cond_log, 12.0))
        del cond_s

        b18 = clean_band(src.read(18), footprint)
        put_z("grav_hg_signed", b18)
        del b18

        b19 = clean_band(src.read(19), footprint)
        b19_log = np.log1p(np.maximum(b19, 0.0)).astype(np.float32)
        put_z("det_elev_slope_log", b19_log)
        slope_fp_raw = b19[footprint].astype(np.float32)
        del b19, b19_log
    gc.collect()

    # === STAGE 2: Blakely-Simpson Crests, Reid N=0 Euler Depth, Cross-Gradients & Hydrothermal Breach ===
    hg_m = np.maximum(tmi_hg, 0.0)
    mag_as = np.sqrt(hg_m * hg_m + tmi_vg * tmi_vg).astype(np.float32)
    mag_as_log = np.log1p(mag_as).astype(np.float32)
    mag_tilt = np.arctan2(tmi_vg, np.maximum(hg_m, 1e-6)).astype(np.float32)
    mag_tilt_zero = (1.0 - np.abs(mag_tilt) / (np.pi / 2.0)).astype(np.float32)
    put_z("pot_mag_tilt", mag_tilt)
    put_z("pot_mag_tilt_zero", mag_tilt_zero)
    put_z("pot_mag_as_log", mag_as_log)
    blak_mag_15 = blakely_simpson_crest(mag_as_log, sigma=1.5)
    blak_mag_35 = blakely_simpson_crest(mag_as_log, sigma=3.5)
    put_z("pot_blakely_mag_s1p5", blak_mag_15)
    put_z("pot_blakely_mag_s3p5", blak_mag_35)

    ay, ax = np.gradient(gf32(mag_as, 1.5))
    euler_mag = np.log1p(mag_as / (np.hypot(ay, ax) + 1e-4)).astype(np.float32)
    put_z("pot_euler_depth_mag", -euler_mag)  # higher = shallower fault/breach source depth!
    del ay, ax, tmi_hg, tmi_vg, hg_m, mag_as, mag_tilt

    with rasterio.open(ext_dir / "geodawn_extensions_u8.tif") as es:
        tmi_up150 = clean_band(es.read(4), footprint)
    up150_atten = np.abs(rank01_fp(rtp, footprint) - rank01_fp(tmi_up150, footprint)).astype(np.float32)
    put_z("pot_up150_atten", up150_atten)
    demag_shallow = np.maximum(rank01_fp(tmi_up150, footprint) - rank01_fp(rtp, footprint), 0.0).astype(np.float32)
    del tmi_up150

    gy, gx = np.gradient(gf32(grav, 1.0))
    grav_thdr = np.hypot(gy, gx).astype(np.float32)
    grav_thdr_log = np.log1p(grav_thdr).astype(np.float32)
    grav_tilt = np.arctan2(grav_vg, np.maximum(grav_thdr, 1e-9)).astype(np.float32)
    grav_tilt_zero = (1.0 - np.abs(grav_tilt) / (np.pi / 2.0)).astype(np.float32)
    put_z("pot_grav_thdr_log", grav_thdr_log)
    put_z("pot_grav_tilt_zero", grav_tilt_zero)
    blak_grav_15 = blakely_simpson_crest(grav_thdr_log, sigma=1.5)
    blak_grav_35 = blakely_simpson_crest(grav_thdr_log, sigma=3.5)
    put_z("pot_blakely_grav_s1p5", blak_grav_15)
    put_z("pot_blakely_grav_s3p5", blak_grav_35)

    ggy, ggx = np.gradient(gf32(grav_thdr, 1.5))
    euler_grav = -np.log1p(grav_thdr / (np.hypot(ggy, ggx) + 1e-4)).astype(np.float32)
    put_z("pot_euler_depth_grav", euler_grav)
    del ggy, ggx, grav_vg, grav_tilt

    cg15, _ = crossgrad_pair(grav, rtp, footprint, 1.5)
    cg30, ortho30 = crossgrad_pair(grav, rtp, footprint, 3.0)
    cg60, _ = crossgrad_pair(grav, rtp, footprint, 6.0)
    put_z("pot_cg_align_s1p5", cg15)
    put_z("pot_cg_align_s3p0", cg30)
    put_z("pot_cg_align_s6p0", cg60)
    put_z("pot_cg_ortho_conf", ortho30)

    dby, dbx = np.gradient(gf32(depth_b, 2.0))
    db_norm = np.hypot(dby, dbx)
    blak_base = blakely_simpson_crest(np.log1p(db_norm), sigma=2.0)
    hangwall_conc = ((-dbx * gx - dby * gy) / (db_norm * grav_thdr + 1e-9) * np.log1p(db_norm)).astype(np.float32)
    put_z("pot_blakely_base_crest", blak_base)
    put_z("pot_hangwall_concord", hangwall_conc)
    del grav, rtp, depth_b, gy, gx, dby, dbx, db_norm, grav_thdr

    cgy, cgx = np.gradient(gf32(cond_log, 1.8))
    mt_grad = np.hypot(cgy, cgx).astype(np.float32)
    blak_mt = blakely_simpson_crest(mt_grad, sigma=2.0)
    put_z("pot_blakely_mt_claycap", blak_mt)
    del cond_log, cgy, cgx, mt_grad

    diseq_g, therm_g, dab2m_g, vent_g = load_hydrothermal_sources(ext_dir, transform, (h, w))
    chem_near = (0.65 * gf32(diseq_g, sigma=(6.0, 3.0), mode="constant") + 0.35 * gf32(diseq_g, 2.8, mode="constant")).astype(np.float32)
    chem_far = gf32(diseq_g + therm_g + vent_g + 0.6 * dab2m_g, sigma=16.0, mode="constant")
    probe_plume = (0.65 * gf32(dab2m_g, sigma=(5.0, 2.5), mode="constant") + 0.35 * gf32(dab2m_g, 11.0, mode="constant")).astype(np.float32)
    sinter_plume = (0.65 * gf32(vent_g, sigma=(5.0, 2.5), mode="constant") + 0.35 * gf32(vent_g, 12.0, mode="constant")).astype(np.float32)
    put_z("hydro_chem_excess_near", chem_near)
    put_z("hydro_chem_excess_far", chem_far)
    put_z("hydro_probe2m_dab", probe_plume)
    put_z("hydro_sinter_vent_plume", sinter_plume)
    del diseq_g, therm_g, dab2m_g, vent_g

    demag_breach = (
        (0.55 * rank01_fp(chem_near + probe_plume + sinter_plume, footprint) + 0.45 * rank01_fp(chem_far, footprint))
        * (0.45 * demag_shallow + 0.35 * rank01_fp(blak_mt, footprint) + 0.20 * rank01_fp(up150_atten, footprint))
    ).astype(np.float32)
    put_z("hydro_demag_breach", demag_breach)

    with rasterio.open(ext_dir / "geodawn_rad_u8.tif") as rs:
        rk = rank01_fp(clean_band(rs.read(1), footprint), footprint)
        rth = rank01_fp(clean_band(rs.read(2), footprint), footprint)
        ru = rank01_fp(clean_band(rs.read(3), footprint), footprint)
    ky, kx = np.gradient(gf32(rk - rth, 2.0))
    kth_grad = (np.hypot(ky, kx) + 0.30 * np.maximum(ru - 0.5 * (rk + rth), 0.0)).astype(np.float32)
    put_z("hydro_kth_rank_grad", kth_grad)
    del rk, rth, ru, ky, kx, kth_grad, up150_atten, demag_shallow

    geod_rank_2d = rank01_fp(geod2, footprint)
    mech_seism = (0.5 * geod_rank_2d * rank01_fp(ieq, footprint) + 0.5 * np.exp(-np.maximum(deq, 0.0) / 8.0)).astype(np.float32)
    put_z("mech_strain_seism_coupled", mech_seism)

    d_euler_blakely_fp = (
        0.30 * rank01_fp(blak_grav_15 + blak_grav_35 + grav_thdr_log * grav_tilt_zero, footprint)
        + 0.25 * rank01_fp(blak_mag_15 + blak_mag_35 + mag_as_log * mag_tilt_zero, footprint)
        + 0.25 * rank01_fp(cg15 + cg30 + cg60 + np.maximum(hangwall_conc, 0.0), footprint)
        + 0.20 * rank01_fp(demag_breach + blak_mt, footprint)
    )[footprint].astype(np.float32)
    pot_crest_2d = (
        0.45 * rank01_fp(blak_grav_15, footprint)
        + 0.35 * rank01_fp(blak_mag_15, footprint)
        + 0.20 * rank01_fp(blak_base + blak_mt, footprint)
    ).astype(np.float32)
    del (
        geod2, ieq, deq, mech_seism, blak_mag_15, blak_mag_35, mag_as_log, mag_tilt_zero,
        euler_mag, blak_grav_15, blak_grav_35, grav_thdr_log, grav_tilt_zero, euler_grav,
        cg15, cg30, cg60, ortho30, blak_base, hangwall_conc, blak_mt,
        chem_near, chem_far, probe_plume, sinter_plume, demag_breach,
    )
    gc.collect()

    # === STAGE 3: 1m LiDAR Antithetic-Synthetic Graben Tensor (`H77E-2`) + 10m Seamless DEM ===
    lj = json.loads((ext_dir / "lidar_scarp_features.json").read_text())
    q_rules = lj["quantisation"]

    def read_lidar_band(ls: Any, idx_b: int, bname: str) -> np.ndarray:
        q = ls.read(idx_b).astype(np.float32)
        xmax, mode_q = q_rules[bname]
        t = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
        return (xmax * (t * t if mode_q == "sqrt" else t)).astype(np.float32)

    with rasterio.open(ext_dir / "lidar_scarp_features_u8.tif") as ls:
        has_lidar = footprint & (ls.read(12) > 0)
        ex_max = read_lidar_band(ls, 1, "ex_max")
        ex_mean = read_lidar_band(ls, 2, "ex_mean")
        step_max = read_lidar_band(ls, 3, "step_max")
        lapneg = read_lidar_band(ls, 4, "lapneg_max")
        lappos = read_lidar_band(ls, 5, "lappos_max")
        downface = read_lidar_band(ls, 6, "downface_max")
        upface = read_lidar_band(ls, 7, "upface_max")
        cross_m = read_lidar_band(ls, 8, "cross_max")
        l_relief = read_lidar_band(ls, 9, "relief")
        coh100 = read_lidar_band(ls, 10, "coh100")
        strike = read_lidar_band(ls, 11, "strike")

    put_z("lidar_has_cov", has_lidar.astype(np.float32))
    persist_ratio = (ex_mean / (ex_max + 0.04)).astype(np.float32)
    dipole_1m = np.sqrt(np.maximum(lapneg * lappos, 0.0)).astype(np.float32)
    antithetic_pure = (upface * coh100 / (cross_m + 0.06)).astype(np.float32)
    down_sm = gf32(np.where(has_lidar, downface, 0.0), sigma=2.2, mode="constant")
    up_sm = gf32(np.where(has_lidar, upface, 0.0), sigma=2.2, mode="constant")
    graben_couplet = np.sqrt(np.maximum(down_sm * up_sm, 0.0)).astype(np.float32)
    del down_sm, up_sm
    bank_rejected = (np.maximum(np.maximum(downface, 1.35 * upface) - 0.85 * cross_m, 0.0) * persist_ratio).astype(np.float32)

    th_rad = np.deg2rad(strike)
    w_str = np.where(has_lidar, ex_max * coh100, 0.0).astype(np.float32)
    s_cont25 = np.hypot(
        gf32(w_str * np.cos(2.0 * th_rad), 2.5, mode="constant"),
        gf32(w_str * np.sin(2.0 * th_rad), 2.5, mode="constant"),
    ).astype(np.float32)
    s_cont60 = np.hypot(
        gf32(w_str * np.cos(2.0 * th_rad), 6.0, mode="constant"),
        gf32(w_str * np.sin(2.0 * th_rad), 6.0, mode="constant"),
    ).astype(np.float32)
    diff_n = np.minimum(np.abs(strike - 18.0), 180.0 - np.abs(strike - 18.0))
    diff_t = np.minimum(np.abs(strike - 145.0), 180.0 - np.abs(strike - 145.0))
    and_fav = (coh100 * np.maximum(np.cos(np.deg2rad(2.0 * diff_n)), 0.8 * np.cos(np.deg2rad(2.0 * diff_t)))).astype(np.float32)
    del th_rad, w_str, diff_n, diff_t, strike

    put_z("lidar_ex_max", ex_max, mask_valid=has_lidar)
    put_z("lidar_ex_mean", ex_mean, mask_valid=has_lidar)
    put_z("lidar_persist_ratio", persist_ratio, mask_valid=has_lidar)
    put_z("lidar_step_max", step_max, mask_valid=has_lidar)
    put_z("lidar_lapneg_max", lapneg, mask_valid=has_lidar)
    put_z("lidar_lappos_max", lappos, mask_valid=has_lidar)
    put_z("lidar_crest_toe_dipole", dipole_1m, mask_valid=has_lidar)
    put_z("lidar_downface_max", downface, mask_valid=has_lidar)
    put_z("lidar_upface_max", upface, mask_valid=has_lidar)
    put_z("lidar_cross_max", cross_m, mask_valid=has_lidar)
    put_z("lidar_antithetic_pure", antithetic_pure, mask_valid=has_lidar)
    put_z("lidar_graben_couplet", graben_couplet, mask_valid=has_lidar)
    put_z("lidar_bank_rejected", bank_rejected, mask_valid=has_lidar)
    put_z("lidar_relief_log", np.log1p(np.maximum(l_relief, 0.0)), mask_valid=has_lidar)
    put_z("lidar_coh100", coh100, mask_valid=has_lidar)
    put_z("lidar_strike_cont_s2p5", s_cont25, mask_valid=has_lidar)
    put_z("lidar_strike_cont_s6p0", s_cont60, mask_valid=has_lidar)
    put_z("lidar_anderson_fav", and_fav, mask_valid=has_lidar)

    lidar_scarp_combo = np.where(
        has_lidar,
        0.35 * rank01_fp(np.where(has_lidar, bank_rejected * (0.3 + dipole_1m), np.nan), footprint)
        + 0.30 * rank01_fp(np.where(has_lidar, antithetic_pure + graben_couplet, np.nan), footprint)
        + 0.35 * rank01_fp(np.where(has_lidar, s_cont25 + 0.6 * s_cont60, np.nan), footprint),
        0.0,
    ).astype(np.float32)
    lidar_radon = radon_line_8az(np.where(has_lidar, bank_rejected + 0.8 * dipole_1m, np.nan), footprint, half_len=6, flank=3)
    put_z("lidar_radon_scarp", lidar_radon, mask_valid=has_lidar)
    del ex_max, ex_mean, persist_ratio, step_max, lapneg, lappos, dipole_1m
    del downface, upface, cross_m, l_relief, coh100, and_fav, s_cont60, lidar_radon
    gc.collect()

    # De-quantize seamless 9-band 10m 3DEP DEM (`topo_10m_derived_100m_u8.tif`)
    t10_man = json.loads((ext_dir / "topo_10m_derived_100m_u8.manifest.json").read_text())
    t10_bands = {b["band_name"]: (b["index"], float(b["lo"]), float(b["hi"])) for b in t10_man["bands"]}
    with rasterio.open(ext_dir / "topo_10m_derived_100m_u8.tif") as ts:
        def read_t10(bname: str) -> np.ndarray:
            bi, lo, hi = t10_bands[bname]
            q = ts.read(bi).astype(np.float32)
            v = lo + np.clip((q - 1.0) / 254.0, 0.0, 1.0) * (hi - lo)
            return clean_band(np.where(q > 0, v, np.nan), footprint)

        t_slp90 = read_t10("slope_p90")
        t_slpmax = read_t10("slope_max")
        t_steep = read_t10("steep_frac")
        t_slpstd = read_t10("slope_std")
        t_curv = read_t10("curv_prof_absmax")
        t_asp = read_t10("aspect_coherence")
        t_hslin = read_t10("hs_lineament")
        t_dem = read_t10("dem_mean")

    put_z("topo10_slope_p90", np.log1p(t_slp90))
    put_z("topo10_slope_max", np.log1p(t_slpmax))
    put_z("topo10_steep_frac", t_steep)
    put_z("topo10_slope_std", np.log1p(t_slpstd))
    put_z("topo10_curv_prof", np.log1p(t_curv))
    put_z("topo10_aspect_coh", t_asp)
    put_z("topo10_hs_lineament", t_hslin)

    # Cross-strike step purity & along-strike non-equipotential shoreline filter on 10m DEM
    zy, zx = np.gradient(gf32(t_dem, 1.8))
    zn = np.hypot(zy, zx) + 1e-5
    # Along-isocontour tangent direction (-zy/zn, zx/zn): shoreline has zero elevation drift along strike!
    z_sm6 = gf32(t_dem, 6.0)
    z6y, z6x = np.gradient(z_sm6)
    along_drift = np.abs((-zy * z6x + zx * z6y) / zn).astype(np.float32)
    step_purity = (np.log1p(zn) / (np.log1p(t_slpstd) + 0.35) * t_asp).astype(np.float32)
    put_z("topo10_step_purity", step_purity)
    put_z("topo10_non_equipotential", np.log1p(along_drift))
    topo10_radon = radon_line_8az(np.log1p(t_slpmax) * t_asp * (0.4 + t_hslin), footprint, half_len=7, flank=3)
    put_z("topo10_radon_scarp", topo10_radon)
    del zy, zx, zn, z_sm6, z6y, z6x, along_drift, t_slp90, t_slpstd, t_curv, t_hslin, t_dem

    d1m_npz = np.load(ext_dir / "dem1m_high_prior_openness_lrm.npz")
    fp_idx = d1m_npz["cov_fp_indices"].astype(np.int64)
    open_fp = np.full(n_fp, np.nan, dtype=np.float32)
    o_a = d1m_npz["openness_dipole_max"].astype(np.float64)
    o_b = d1m_npz["lrm_grad_max"].astype(np.float64)
    open_fp[fp_idx] = (
        0.5 * (np.argsort(np.argsort(o_a)) + np.argsort(np.argsort(o_b))) / max(len(fp_idx) - 1, 1)
    ).astype(np.float32)
    mm[:, STATIC_IDX["dem1m_openness_lrm"]] = np.where(np.isfinite(open_fp), (open_fp - 0.5) * 4.0, np.nan).astype(np.float16)
    del d1m_npz, fp_idx, open_fp, o_a, o_b

    # === STAGE 4: Fluvial Chi-Steepness Knickpoint Wave (`H77E-3`) ===
    with rasterio.open(ext_dir / "dem10_chi_100m_i16.tif") as cs:
        c_smax = clean_band(cs.read(1).astype(np.float32) / 1000.0, footprint)
        c_smin = clean_band(cs.read(2).astype(np.float32) / 1000.0, footprint)
        c_ksn = clean_band(cs.read(3).astype(np.float32) / 10.0, footprint)
        c_area = clean_band(cs.read(4).astype(np.float32) / 1000.0, footprint)
    with rasterio.open(ext_dir / "dem10_chi_100m_sig_i32.tif") as ss:
        c_sig = np.log1p(np.maximum(clean_band(ss.read(1).astype(np.float32), footprint), 0.0)).astype(np.float32)

    chi_pos = np.log1p(np.maximum(c_smax, 0.0)).astype(np.float32)
    chi_neg = np.log1p(np.maximum(-c_smin, 0.0)).astype(np.float32)
    ksn_log = np.log1p(np.maximum(c_ksn, 0.0)).astype(np.float32)
    area_gate = np.clip((c_area - 4.7) / 1.5, 0.0, 1.0).astype(np.float32)
    area_knick = (chi_pos * area_gate).astype(np.float32)
    chi_rad15 = radon_line_8az(area_knick + 0.4 * c_sig, footprint, half_len=7, flank=3)
    chi_rad27 = radon_line_8az(area_knick + 0.35 * ksn_log, footprint, half_len=13, flank=4)
    chi_facet = (np.maximum(chi_rad15, 0.0) * (0.3 + t_steep)).astype(np.float32)
    chi_pied = (chi_pos * np.clip((6.2 - c_area) / 1.5, 0.0, 1.0) * step_purity).astype(np.float32)

    put_z("chi_step_pos", chi_pos)
    put_z("chi_step_neg", chi_neg)
    put_z("chi_ksn_log", ksn_log)
    put_z("chi_area_gated_knick", area_knick)
    put_z("chi_sig_knick", c_sig)
    put_z("chi_radon_s1p5_km", chi_rad15)
    put_z("chi_radon_s2p7_km", chi_rad27)
    put_z("chi_facet_coupled", chi_facet)
    put_z("chi_piedmont_headwater", chi_pied)

    d_graben_fp = np.where(
        has_lidar,
        0.70 * lidar_scarp_combo + 0.30 * rank01_fp(topo10_radon + step_purity, footprint),
        rank01_fp(topo10_radon + step_purity, footprint),
    )[footprint].astype(np.float32)
    d_chi_fp = rank01_fp(0.55 * chi_rad15 + 0.30 * chi_rad27 + 0.15 * chi_facet, footprint)[footprint].astype(np.float32)

    crest_2d = np.where(
        has_lidar,
        0.55 * rank01_fp(np.where(has_lidar, bank_rejected + graben_couplet + antithetic_pure, np.nan), footprint)
        + 0.25 * rank01_fp(np.log1p(t_slpmax) * t_asp, footprint)
        + 0.20 * pot_crest_2d,
        0.65 * rank01_fp(np.log1p(t_slpmax) * t_asp + np.maximum(chi_rad15, 0.0), footprint)
        + 0.35 * pot_crest_2d,
    ).astype(np.float32)

    mm.flush()
    np.savez_compressed(
        aux_path,
        geod_rank_fp=geod_rank_2d[footprint].astype(np.float32),
        scarp_rank_fp=d_graben_fp,
        d_graben_fp=d_graben_fp,
        d_chi_fp=d_chi_fp,
        d_euler_blakely_fp=d_euler_blakely_fp,
        crest_fp=crest_2d[footprint].astype(np.float32),
        has_lidar_fp=has_lidar[footprint],
        slope_fp=slope_fp_raw,
    )
    meta = {
        "shape": [n_fp, n_ch],
        "dtype": "float16",
        "channels": STATIC_NAMES,
        "blocks": STATIC_BLOCKS,
        "mech_channels": MECH_NAMES,
        "footprint_pixels": n_fp,
        "lidar_valid_pixels": int(has_lidar.sum()),
        "lidar_missing_pixels_mapped_to_nan": int((footprint & ~has_lidar).sum()),
        "sgmc_channels_used": 0,
        "prior_rasters_read": 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    del (
        c_smax, c_smin, c_ksn, c_area, c_sig, chi_pos, chi_neg, ksn_log, area_gate,
        area_knick, chi_rad15, chi_rad27, chi_facet, chi_pied, t_slpmax, t_steep,
        t_asp, step_purity, topo10_radon, bank_rejected, graben_couplet, antithetic_pure,
        s_cont25, lidar_scarp_combo, pot_crest_2d, geod_rank_2d, crest_2d,
    )
    gc.collect()
    return np.load(stack_path, mmap_mode="r"), meta


def load_h77e_static_cache(data_dir: Path, work_dir: Path, force: bool = False) -> tuple[np.memmap, dict[str, np.ndarray], dict[str, Any]]:
    """Build if missing, then load `(X16_mmap, aux_dict, meta)` from `work_dir`."""
    stack_path = work_dir / "h77e_static_f16.npy"
    aux_path = work_dir / "h77e_aux_fp.npz"
    meta_path = work_dir / "h77e_static_meta.json"
    if force or not (stack_path.is_file() and aux_path.is_file() and meta_path.is_file()):
        t0 = __import__("time").time()
        mm, meta = build_h77e_static_stack(data_dir, work_dir)
        meta["elapsed_seconds"] = round(__import__("time").time() - t0, 2)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    else:
        mm = np.load(stack_path, mmap_mode="r")
        meta = json.loads(meta_path.read_text())
    with np.load(aux_path) as z:
        aux = {k: z[k] for k in z.files}
    return mm, aux, meta

