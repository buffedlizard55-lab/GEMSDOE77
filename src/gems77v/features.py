"""H77V Geothermal Vent Advection-Backtracking & Dilational Conduit Feature Stack (`src/gems77v/features.py`).

Implements the 16 new preregistered H77V channels across 4 physical blocks:
  - Block A (H77V-1, 6 channels): Up-Gradient Topographic Advection Backtracking of Geothermometer
    Disequilibrium Plumes, Cation-Silica Kinetic Fractionation, and Lake Lahontan Tufa-Sieved Vents.
  - Block B (H77V-1, 4 channels): Critically Stressed Transtensional Dilational Conduit Tensor
    (dilatation-to-shear strain partitioning + Mohr-Coulomb dilation & slip tendency under N110E sigma_hmin).
  - Block C (H77V-2, 3 channels): Steam-Heated Clay-Cap Breach Electrical Conductivity Annulus
    coupled with Directional RTP Demagnetization Trough and Potassic (K/Th) Alteration Halo.
  - Block D (H77V-3, 3 channels): Upslope-Realigned Hanging-Wall Gravity/Basement Crest coupled with
    Alluvial-Fan Piedmont Toe Concavity and 1 m LiDAR Downface-Upface Scarp Polarity.

Plus 36 zero-SGMC multi-sensor base channels (16 raw non-SGMC bands, 10 potential-field coupling channels,
10 road-deconfounded 1 m LiDAR [q==0 -> NaN] and 10 m USGS 3DEP DEM scarp channels), and 4 strictly
fold-visible Leave-Own-Component-Out (LOCO) structural mechanics channels.

CRITICAL ANTI-LEAKAGE GUARANTEES:
  1. Never reads `dist_known_fault_px` from `gdr_wellspring_in_footprint.csv` or `gdr_volcanic_vents_in_footprint.csv`.
  2. Never reads `gdr_qfaults_traces.csv`, `existing_faults.tif`, or `labels.tif` in `build_h77v_static_stack`.
  3. Never reads SGMC Bands 0 or 1 (`geol_age`, `geol_type`) from `training_features.tif`.
  4. Maps 1 m LiDAR `q == 0` missing tiles (`24.6%` of footprint) to `np.nan` for native HGB missing-value splits.
"""
from __future__ import annotations

import csv
import gc
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from scipy import ndimage as ndi

from src.gems77d.features import compute_visible_mechanics

H77V_CHANNEL_NAMES: list[str] = [
    # Block A: H77V-1 Advection-Backtracking & Geothermometer Fractionation (6)
    "h77v_00_advect_backtrack_plume",
    "h77v_01_advect_vs_iso_shift",
    "h77v_02_cation_silica_kinetic_backtrack",
    "h77v_03_orifice_spring_sinter_pin",
    "h77v_04_probe2m_upgradient_plume",
    "h77v_05_backtrack_step_convergence",
    # Block B: H77V-1 Critically Stressed Transtensional Dilational Conduit Tensor (4)
    "h77v_06_transtensional_dilation_ratio",
    "h77v_07_crit_stress_dilation_tendency",
    "h77v_08_crit_stress_slip_tendency",
    "h77v_09_stepover_strain_curvature_gate",
    # Block C: H77V-2 Clay-Cap Breach & Hydrothermal Alteration Tri-Pole (3)
    "h77v_10_claycap_breach_annulus",
    "h77v_11_demag_trough_clay_coupling",
    "h77v_12_hydrothermal_tripole_index",
    # Block D: H77V-3 Upslope-Realigned Hanging-Wall Gravity Crest & Piedmont Toe Asymmetry (3)
    "h77v_13_upslope_realigned_grav_base_crest",
    "h77v_14_piedmont_toe_concavity_grav_lock",
    "h77v_15_lidar_scarp_polarity_vent_lock",
]

BASE_CHANNEL_NAMES: list[str] = [
    # 16 Zero-SGMC raw/filtered bands from training_features.tif
    "raw_mag_anom_hp",
    "raw_rtp",
    "raw_rtp_hp",
    "raw_tmi_hg_log",
    "raw_geod_2ndinv",
    "raw_grav_slope_log",
    "raw_tc_rad",
    "raw_geod_shear",
    "raw_geod_dilat",
    "raw_tmi_vg",
    "raw_deq",
    "raw_grav_vg",
    "raw_det_elev",
    "raw_det_elev_hp",
    "raw_iso_grav_hp",
    "raw_depth_base",
    "raw_ieq",
    "raw_cond_surf_log",
    "raw_det_elev_slope_log",
    # 9 Potential-field coupling & piedmont transition channels
    "pot_mag_tilt_zero",
    "pot_mag_as_log",
    "pot_grav_thdr_log",
    "pot_grav_tilt_zero",
    "pot_cg_align_s3p0",
    "pot_cg_ortho_conf",
    "pot_base_grad_log",
    "pot_base_grav_align",
    "pied_relief_asym_1k",
    # 8 Road-deconfounded 1 m LiDAR (q==0 -> NaN) & 10 m 3DEP DEM scarp channels
    "tiger_road_dist_log",
    "lidar_ex_max",
    "lidar_step_deconf",
    "lidar_dipole_deconf",
    "lidar_strike_cont",
    "dem10_slope_p90_deconf",
    "dem10_curv_prof_deconf",
    "dem10_st_ridge_deconf",
]

ALL_STATIC_NAMES: list[str] = H77V_CHANNEL_NAMES + BASE_CHANNEL_NAMES
STATIC_IDX: dict[str, int] = {n: i for i, n in enumerate(ALL_STATIC_NAMES)}

IDX_H77V: list[int] = list(range(len(H77V_CHANNEL_NAMES)))
IDX_BASE: list[int] = list(range(len(H77V_CHANNEL_NAMES), len(ALL_STATIC_NAMES)))
IDX_ALL_STATIC: list[int] = list(range(len(ALL_STATIC_NAMES)))

LOCO_NAMES: list[str] = [
    "loco_vis_dist_clipped",
    "loco_crack_tip_lobe",
    "loco_stepover_ramp",
    "loco_strain_deficit",
]


def gf32(
    arr: np.ndarray,
    sigma: float | tuple[float, float],
    order: int | tuple[int, int] = 0,
    mode: str = "nearest",
) -> np.ndarray:
    """Fast float32 Gaussian filter avoiding float64 intermediate expansion."""
    out = np.empty_like(arr, dtype=np.float32)
    ndi.gaussian_filter(arr.astype(np.float32, copy=False), sigma=sigma, order=order, output=out, mode=mode)
    return out


def clean_band(arr: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Replace non-finite, out-of-footprint, or float32 nodata sentinel (-3.4e38) values with in-footprint median."""
    out = arr.astype(np.float32, copy=True)
    ok = np.isfinite(out) & (np.abs(out) < 1e15) & footprint
    med = float(np.median(out[ok][::8])) if np.any(ok) else 0.0
    out[~ok] = med
    return out


def rank01_fp(arr: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Map finite in-footprint values to [0, 1] empirical rank quantiles."""
    out = np.zeros(arr.shape, dtype=np.float32)
    v = arr[footprint]
    ok = np.isfinite(v)
    if not np.any(ok):
        return out
    vals = v[ok]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(vals, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(vals), dtype=np.float32)
    tmp = np.full(v.shape, 0.5, dtype=np.float32)
    tmp[ok] = ranks
    out[footprint] = tmp
    return out


def crossgrad_pair(
    f1: np.ndarray, f2: np.ndarray, footprint: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gallardo & Meju (2004) parallel structural alignment vs orthogonal confounder."""
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


def unpack_topo10m_band(ts: rasterio.DatasetReader, band_idx: int, lo: float, hi: float, footprint: np.ndarray) -> np.ndarray:
    """Unpack uint8 quantized USGS 3DEP 10 m DEM derived band (`q==0` nodata)."""
    q = ts.read(band_idx).astype(np.float32)
    valid = (q > 0) & footprint
    v = lo + np.clip((q - 1.0) / 254.0, 0.0, 1.0) * (hi - lo)
    med = float(np.median(v[valid][::8])) if np.any(valid) else 0.5 * (lo + hi)
    v[~valid] = med
    return v.astype(np.float32)


def _parse_2m_probes(
    zip_path: Path, transform: rasterio.Affine, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read (row, col, excess_T2m) from 2m_temperature_probe_n83geo.dbf inside DOE GDR 1391 zip."""
    if not zip_path.exists():
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    with zipfile.ZipFile(zip_path) as zf:
        dbf_member = next(n for n in zf.namelist() if n.lower().endswith(".dbf"))
        raw = zf.read(dbf_member)
    nrec, hdr_len, rec_len = struct.unpack("<IHH", raw[4:12])
    fields = []
    pos = 32
    while raw[pos] != 0x0D:
        name = raw[pos : pos + 11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        ftype = chr(raw[pos + 11])
        flen = raw[pos + 16]
        fields.append((name, ftype, flen))
        pos += 32
    lons, lats, t2ms = [], [], []
    offset = hdr_len
    for _ in range(nrec):
        rec_bytes = raw[offset : offset + rec_len]
        offset += rec_len
        if not rec_bytes or rec_bytes[0] == 0x2A:
            continue
        p = 1
        row_d = {}
        for name, _ftype, flen in fields:
            row_d[name] = rec_bytes[p : p + flen].decode("ascii", errors="ignore").strip()
            p += flen
        try:
            lon = float(row_d.get("utm_e", "nan"))
            lat = float(row_d.get("utm_n", "nan"))
            t2m = float(row_d.get("T2m", "nan"))
        except ValueError:
            continue
        if np.isfinite(lon) and np.isfinite(lat) and np.isfinite(t2m) and -125.0 <= lon <= -110.0 and 35.0 <= lat <= 45.0:
            lons.append(lon)
            lats.append(lat)
            t2ms.append(t2m)
    if not lons:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    ux, uy = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
    cols = np.floor((np.asarray(ux) - transform.c) / transform.a).astype(np.int32)
    rows = np.floor((np.asarray(uy) - transform.f) / transform.e).astype(np.int32)
    t_arr = np.asarray(t2ms, dtype=np.float32)
    h, w = shape
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w) & (t_arr > 0.0) & (t_arr < 120.0)
    rows, cols, t_arr = rows[ok], cols[ok], t_arr[ok]
    med_t = float(np.median(t_arr)) if t_arr.size else 15.0
    excess = np.log1p(np.maximum(0.0, t_arr - med_t)).astype(np.float32)
    return rows, cols, excess


def _parse_paleo_vents_tufa_sieved(
    zip_path: Path, transform: rasterio.Affine, z_ortho: np.ndarray, footprint: np.ndarray
) -> np.ndarray:
    """Build tufa-sieved paleo-hydrothermal vent impulse grid from Paleo_geothermal_final.csv.

    Scientific Basis (IR-H77V-1):
      - `Sinter`, `Silicified fault breccia`, `Explosion craters`, `Silica vein`, `Calcite-qtz veins`:
        high-temperature (>175 C) fault-conduit vent deposits -> weight 2.2
      - `Travertine`, `Silica-cemented sand`, `Silica-cemented mud or ash`, `Sulfur`, `Alteration`,
        `Argillic alteration`, `Sulfate crust`: structural fault-vent discharge -> weight 1.5
      - `Tufa` (30.6% of in-footprint points): predominantly wave-agitated Pleistocene Lake Lahontan
        shoreline carbonate towers below the 1,332 m highstand (Z_ortho <= 1,340 m) -> downweighted to 0.15
        below 1,340 m (and 0.45 above 1,340 m).
    """
    h, w = footprint.shape
    grid = np.zeros((h, w), dtype=np.float32)
    if not zip_path.exists():
        return grid
    with zipfile.ZipFile(zip_path) as zf:
        csv_member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_member) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace"))
            lons, lats, dtypes = [], [], []
            for row in reader:
                try:
                    lon = float(row.get("Long_deg_NAD83", "nan"))
                    lat = float(row.get("Lat_deg_NAD83", "nan"))
                except ValueError:
                    continue
                if np.isfinite(lon) and np.isfinite(lat):
                    lons.append(lon)
                    lats.append(lat)
                    dtypes.append(row.get("Deposit_type", "").strip().lower())
    if not lons:
        return grid
    ux, uy = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
    cols = np.floor((np.asarray(ux) - transform.c) / transform.a).astype(np.int32)
    rows = np.floor((np.asarray(uy) - transform.f) / transform.e).astype(np.int32)
    for r, c, dt in zip(rows, cols, dtypes):
        if not (0 <= r < h and 0 <= c < w and footprint[r, c]):
            continue
        if any(k in dt for k in ("sinter", "breccia", "explosion", "vein")):
            w_dep = 2.2
        elif "tufa" in dt:
            w_dep = 0.15 if float(z_ortho[r, c]) <= 1340.0 else 0.45
        else:
            w_dep = 1.5
        if w_dep > grid[r, c]:
            grid[r, c] = w_dep
    return grid


def build_advection_backtracked_hydro_channels(
    data_dir: Path,
    transform: rasterio.Affine,
    footprint: np.ndarray,
    z_ortho: np.ndarray,
    uy_up: np.ndarray,
    ux_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Block A H77V-1 up-gradient advection-backtracked hydrothermal & kinetic fractionation grids.

    CRITICAL ANTI-LEAKAGE GUARANTEE:
      Explicitly ignores `dist_known_fault_px` in `gdr_wellspring_in_footprint.csv` and
      `gdr_volcanic_vents_in_footprint.csv`.

    Returns:
      (advect_plume, advect_vs_iso_shift, kinetic_backtrack, orifice_pin, probe2m_upgrad, iso_control_plume)
    """
    h, w = footprint.shape
    imp_advect_near = np.zeros((h, w), dtype=np.float32)
    imp_advect_far = np.zeros((h, w), dtype=np.float32)
    imp_iso_well = np.zeros((h, w), dtype=np.float32)
    imp_kinetic = np.zeros((h, w), dtype=np.float32)
    imp_orifice = np.zeros((h, w), dtype=np.float32)

    ws_csv = data_dir / "external" / "gdr_wellspring_in_footprint.csv"
    with ws_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        for rec in reader:
            # Explicit anti-leakage guard: never access rec["dist_known_fault_px"]
            r = int(float(rec["row"]))
            c = int(float(rec["col"]))
            if not (0 <= r < h and 0 <= c < w):
                continue
            layer_str = rec.get("layer", "").strip().lower()
            tclass_str = rec.get("thermalclass", "").strip().lower()  # handles 'Hot ' trailing space (IR-H77V-2)
            is_spring = layer_str.startswith("spring")

            def _parse_temp(col_name: str, hi_cap: float = 380.0) -> float:
                s = rec.get(col_name, "").strip()
                if not s:
                    return 0.0
                try:
                    v = float(s)
                    return v if (np.isfinite(v) and 0.0 < v < hi_cap) else 0.0
                except ValueError:
                    return 0.0

            t_meas = _parse_temp("temp_c", 200.0)
            t_qtz = _parse_temp("geothermquartz_c", 380.0)
            t_chalc = _parse_temp("geothermchalc_c", 380.0)
            t_cat = _parse_temp("geothermcat_c", 380.0)
            t_chem = max(t_qtz, t_chalc, t_cat)

            class_bonus = 10.0 if tclass_str == "hot" else (4.0 if tclass_str == "warm" else 0.0)
            t_eff_meas = max(t_meas, 15.0 + class_bonus)

            # Deep reservoir excess and conductive cooling / dilution ratio D_dil in [0, 1]
            dq_excess = max(0.0, t_chem - max(t_eff_meas, 20.0))
            res_excess = max(0.0, max(t_chem, t_eff_meas) - 55.0)
            plume_mag = float(np.log1p(0.65 * dq_excess + 0.45 * res_excess + 0.5 * max(0.0, t_eff_meas - 22.0)))
            if plume_mag <= 0.0:
                continue

            d_dil = float(np.clip((t_chem - max(t_eff_meas, 12.0)) / max(t_chem - 12.0, 20.0), 0.0, 1.0)) if t_chem > 25.0 else 0.35

            # Cation vs. Chalcedony kinetic fractionation gap (Fournier & Truesdell 1973; Giggenbach 1988)
            if t_cat > 0.0 and t_chalc > 0.0:
                kin_gap = float(np.log1p(max(0.0, t_cat - t_chalc) * (0.3 + 0.7 * d_dil)))
            elif t_qtz > 0.0 and t_chalc > 0.0:
                kin_gap = float(np.log1p(0.6 * max(0.0, t_qtz - t_chalc)))
            else:
                kin_gap = 0.0

            if is_spring:
                # Natural surface spring orifice: pin directly at (r, c) plus short upslope collar
                if plume_mag > imp_orifice[r, c]:
                    imp_orifice[r, c] = plume_mag
                if plume_mag > imp_iso_well[r, c]:
                    imp_iso_well[r, c] = plume_mag
                r2 = int(np.clip(round(r + 2.0 * float(uy_up[r, c])), 0, h - 1))
                c2 = int(np.clip(round(c + 2.0 * float(ux_up[r, c])), 0, w - 1))
                if plume_mag > imp_advect_near[r2, c2]:
                    imp_advect_near[r2, c2] = plume_mag
            else:
                # Drilled well sampling lateral outflow tongue: record unshifted control at (r, c),
                # and backtrack upslope along +u_up by distance scaled by thermal dilution D_dil
                if plume_mag > imp_iso_well[r, c]:
                    imp_iso_well[r, c] = plume_mag
                uy_i = float(uy_up[r, c])
                ux_i = float(ux_up[r, c])
                scale_d = 0.35 + 0.65 * d_dil
                for step_px, w_step, target_grid in (
                    (0.0, 0.35, imp_advect_near),
                    (4.5, 0.85, imp_advect_near),
                    (9.0, 1.00, imp_advect_near),
                    (14.0, 0.85, imp_advect_far),
                ):
                    rr = int(np.clip(round(r + step_px * scale_d * uy_i), 0, h - 1))
                    cc = int(np.clip(round(c + step_px * scale_d * ux_i), 0, w - 1))
                    val_s = w_step * plume_mag
                    if val_s > target_grid[rr, cc]:
                        target_grid[rr, cc] = val_s
                if kin_gap > 0.0:
                    for step_px, w_k in ((3.0, 0.7), (7.5, 1.0), (12.0, 0.8)):
                        rr = int(np.clip(round(r + step_px * scale_d * uy_i), 0, h - 1))
                        cc = int(np.clip(round(c + step_px * scale_d * ux_i), 0, w - 1))
                        vk = w_k * kin_gap
                        if vk > imp_kinetic[rr, cc]:
                            imp_kinetic[rr, cc] = vk

    # Tufa-sieved paleo-geothermal vents (Sinter, Travertine, Breccia, Explosion Craters vs Lahontan Tufa)
    paleo_zip = data_dir / "external" / "paleo_geothermal_regional.zip"
    paleo_grid = _parse_paleo_vents_tufa_sieved(paleo_zip, transform, z_ortho, footprint)
    imp_orifice = np.maximum(imp_orifice, paleo_grid)

    # Quaternary/Neogene volcanic vents (dike/conduit swarms)
    vv_csv = data_dir / "external" / "gdr_volcanic_vents_in_footprint.csv"
    if vv_csv.exists():
        with vv_csv.open(newline="") as fh:
            for rec in csv.DictReader(fh):
                r = int(float(rec["row"]))
                c = int(float(rec["col"]))
                if 0 <= r < h and 0 <= c < w and footprint[r, c]:
                    if 1.1 > imp_orifice[r, c]:
                        imp_orifice[r, c] = 1.1

    # 2-meter temperature probes backtracked upslope by 4 px (400 m) along +u_up
    p2m_zip = data_dir / "external" / "2m_temperature_probe_INGENIOUS_regional_data.zip"
    pr_r, pr_c, pr_ex = _parse_2m_probes(p2m_zip, transform, (h, w))
    imp_p2m = np.zeros((h, w), dtype=np.float32)
    for r, c, ex in zip(pr_r, pr_c, pr_ex):
        if ex <= 0.0:
            continue
        if ex > imp_p2m[r, c]:
            imp_p2m[r, c] = 0.6 * ex
        rr = int(np.clip(round(r + 4.0 * float(uy_up[r, c])), 0, h - 1))
        cc = int(np.clip(round(c + 4.0 * float(ux_up[r, c])), 0, w - 1))
        if ex > imp_p2m[rr, cc]:
            imp_p2m[rr, cc] = ex

    advect_plume = (
        0.60 * gf32(imp_advect_near, sigma=3.8, mode="constant")
        + 0.40 * gf32(imp_advect_near + imp_advect_far, sigma=9.0, mode="constant")
    ).astype(np.float32)

    iso_control_plume = (
        0.60 * gf32(imp_iso_well, sigma=3.8, mode="constant")
        + 0.40 * gf32(imp_iso_well, sigma=9.0, mode="constant")
    ).astype(np.float32)

    advect_vs_iso_shift = (advect_plume - iso_control_plume).astype(np.float32)
    kinetic_backtrack = (
        0.55 * gf32(imp_kinetic, sigma=4.0, mode="constant")
        + 0.45 * gf32(imp_kinetic, sigma=10.0, mode="constant")
    ).astype(np.float32)
    orifice_pin = (
        0.65 * gf32(imp_orifice, sigma=3.0, mode="constant")
        + 0.35 * gf32(imp_orifice, sigma=8.5, mode="constant")
    ).astype(np.float32)
    probe2m_upgrad = (
        0.60 * gf32(imp_p2m, sigma=3.5, mode="constant")
        + 0.40 * gf32(imp_p2m, sigma=9.0, mode="constant")
    ).astype(np.float32)

    del imp_advect_near, imp_advect_far, imp_iso_well, imp_kinetic, imp_orifice, imp_p2m, paleo_grid
    return advect_plume, advect_vs_iso_shift, kinetic_backtrack, orifice_pin, probe2m_upgrad, iso_control_plume


def transport_upslope_2d(field: np.ndarray, uy_up: np.ndarray, ux_up: np.ndarray, shift_px: float = 3.5) -> np.ndarray:
    """Advect a hanging-wall potential-field gradient upslope along +u_up by `shift_px` pixels.

    Because a 2D pixel grid (r, c) has value `field[r_src, c_src]` at the hanging-wall position
    `r_src = r - shift_px * uy_up[r, c]`, `c_src = c - shift_px * ux_up[r, c]` (down-slope from `(r, c)`),
    pulling from `(r - shift_px * uy_up, c - shift_px * ux_up)` moves the hanging-wall gravity/basement
    gradient crest UPSLOPE by `shift_px` pixels onto the surface piedmont fault trace!
    """
    h, w = field.shape
    rr, cc = np.indices((h, w), dtype=np.float32)
    r_src = np.clip( np.round(rr - shift_px * uy_up).astype(np.int32), 0, h - 1)
    c_src = np.clip(np.round(cc - shift_px * ux_up).astype(np.int32), 0, w - 1)
    shifted = field[r_src, c_src]
    return (0.70 * shifted + 0.30 * field).astype(np.float32)


def build_h77v_static_stack(data_dir: Path, work_dir: Path) -> tuple[np.memmap, dict]:
    """Build or open cached in-footprint `(N_fp, 52)` float32 static feature matrix for Lane H77V."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stack_path = work_dir / "h77v_fp_stack.npy"
    dom_path = work_dir / "h77v_domains_fp.npz"
    meta_path = work_dir / "h77v_stack_meta.json"

    with rasterio.open(data_dir / "sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
        transform = s.transform
    h, w = footprint.shape
    n_fp = int(footprint.sum())
    n_ch = len(ALL_STATIC_NAMES)

    if stack_path.exists() and dom_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("channels") == ALL_STATIC_NAMES and meta.get("shape") == [n_fp, n_ch]:
            return np.load(stack_path, mmap_mode="r"), meta

    mm = np.lib.format.open_memmap(stack_path, mode="w+", dtype=np.float32, shape=(n_fp, n_ch))

    def put(name: str, arr2d: np.ndarray, mask_valid: np.ndarray | None = None) -> None:
        idx = STATIC_IDX[name]
        vals = arr2d[footprint].astype(np.float32)
        if mask_valid is not None:
            mv = mask_valid[footprint]
            vals = np.where(mv & np.isfinite(vals), vals, np.float32(np.nan))
        mm[:, idx] = vals

    # === STAGE 1: Load 10m USGS 3DEP DEM orthometric elevation & scarp channels ===
    topo_path = data_dir / "external" / "topo_10m_derived_100m_u8.tif"
    topo_meta = json.loads((data_dir / "external" / "topo_10m_derived_100m_u8.manifest.json").read_text())
    bmap = {b["band_name"]: (b["index"], float(b["lo"]), float(b["hi"])) for b in topo_meta["bands"]}
    with rasterio.open(topo_path) as ts:
        slope_p90 = unpack_topo10m_band(ts, *bmap["slope_p90"], footprint)
        curv_prof = unpack_topo10m_band(ts, *bmap["curv_prof_absmax"], footprint)
        asp_coh = unpack_topo10m_band(ts, *bmap["aspect_coherence"], footprint)
        hs_lin = unpack_topo10m_band(ts, *bmap["hs_lineament"], footprint)
        z_ortho = unpack_topo10m_band(ts, *bmap["dem_mean"], footprint)

    # Regional hydraulic head gradient & unit upslope backtracking vector +u_up = +grad(G_8(Z_ortho))
    z_smooth = gf32(z_ortho, sigma=8.0, mode="nearest")
    zy_up, zx_up = np.gradient(z_smooth)
    z_grad_mag = np.hypot(zy_up, zx_up) + 0.5
    uy_up = (zy_up / z_grad_mag).astype(np.float32)
    ux_up = (zx_up / z_grad_mag).astype(np.float32)
    del z_smooth, zy_up, zx_up, z_grad_mag

    # === STAGE 2: Stream non-SGMC bands from training_features.tif ===
    with rasterio.open(data_dir / "training_features.tif") as src:
        b1 = clean_band(src.read(1), footprint)
        put("raw_mag_anom_hp", b1 - gf32(b1, 12.0))
        del b1

        rtp = clean_band(src.read(2), footprint)
        put("raw_rtp", rtp)
        put("raw_rtp_hp", rtp - gf32(rtp, 12.0))

        tmi_hg = clean_band(src.read(3), footprint)
        tmi_hg_log = np.log1p(np.maximum(tmi_hg, 0.0)).astype(np.float32)
        put("raw_tmi_hg_log", tmi_hg_log)
        del tmi_hg_log

        geod2 = clean_band(src.read(4), footprint)
        put("raw_geod_2ndinv", geod2)

        b5 = clean_band(src.read(5), footprint)
        put("raw_grav_slope_log", np.log1p(np.maximum(b5, 0.0)))
        del b5

        put("raw_tc_rad", clean_band(src.read(6), footprint))

        geod_shear = clean_band(src.read(7), footprint)
        put("raw_geod_shear", geod_shear)

        geod_dilat = clean_band(src.read(8), footprint)
        put("raw_geod_dilat", geod_dilat)

        tmi_vg = clean_band(src.read(9), footprint)
        put("raw_tmi_vg", tmi_vg)

        deq = clean_band(src.read(10), footprint)
        put("raw_deq", deq)

        grav_vg = clean_band(src.read(11), footprint)
        put("raw_grav_vg", grav_vg)

        det_elev = clean_band(src.read(12), footprint)
        put("raw_det_elev", det_elev)
        put("raw_det_elev_hp", det_elev - gf32(det_elev, 10.0))
        ry10, rx10 = np.gradient(gf32(det_elev, 10.0))
        put("pied_relief_asym_1k", np.hypot(ry10, rx10))
        del ry10, rx10
        lap_elev = gf32(det_elev, 4.0, order=(2, 0)) + gf32(det_elev, 4.0, order=(0, 2))

        grav = clean_band(src.read(13), footprint)
        put("raw_iso_grav_hp", grav - gf32(grav, 12.0))

        depth_b = clean_band(src.read(15), footprint)
        put("raw_depth_base", depth_b)

        ieq = clean_band(src.read(16), footprint)
        put("raw_ieq", ieq)

        cond_s = clean_band(src.read(17), footprint)
        cond_log = np.log1p(np.maximum(cond_s, 0.0)).astype(np.float32)
        put("raw_cond_surf_log", cond_log)
        del cond_s

        # Note: Band 18 is SGMC geol_age; Band 14 is SGMC geol_type -> explicitly skipped!
        det_slope = clean_band(src.read(19), footprint)
        put("raw_det_elev_slope_log", np.log1p(np.maximum(det_slope, 0.0)))

    # === STAGE 3: Potential-field coupling & Block D Upslope-Realigned Gravity/Basement Crest (H77V-3) ===
    hg_m = np.maximum(tmi_hg, 0.0)
    mag_tilt = np.arctan2(tmi_vg, np.maximum(hg_m, 1e-9)).astype(np.float32)
    mag_tilt_zero = (1.0 - np.abs(mag_tilt) / (np.pi / 2.0)).astype(np.float32)
    mag_as_log = np.log1p(np.sqrt(hg_m * hg_m + tmi_vg * tmi_vg)).astype(np.float32)
    del tmi_hg, tmi_vg, hg_m, mag_tilt

    gy, gx = np.gradient(gf32(grav, 1.2))
    grav_thdr = np.hypot(gy, gx).astype(np.float32)
    grav_thdr_log = np.log1p(grav_thdr).astype(np.float32)
    grav_tilt = np.arctan2(grav_vg, np.maximum(grav_thdr, 1e-12)).astype(np.float32)
    grav_tilt_zero = (1.0 - np.abs(grav_tilt) / (np.pi / 2.0)).astype(np.float32)
    del grav_vg, grav_tilt

    cg30, ortho30 = crossgrad_pair(grav, rtp, footprint, 3.0)

    dby, dbx = np.gradient(gf32(depth_b, 2.0))
    db_norm = np.hypot(dby, dbx)
    base_grad_log = np.log1p(db_norm).astype(np.float32)
    base_grav_align = ((-dbx * gx - dby * gy) / (db_norm * grav_thdr + 1e-9) * base_grad_log).astype(np.float32)
    del grav, depth_b, gy, gx, dby, dbx, db_norm, grav_thdr

    put("pot_mag_tilt_zero", mag_tilt_zero)
    put("pot_mag_as_log", mag_as_log)
    put("pot_grav_thdr_log", grav_thdr_log)
    put("pot_grav_tilt_zero", grav_tilt_zero)
    put("pot_cg_align_s3p0", cg30)
    put("pot_cg_ortho_conf", ortho30)
    put("pot_base_grad_log", base_grad_log)
    put("pot_base_grav_align", base_grav_align)

    # H77V-3 Channel 13: Upslope-realigned hanging-wall gravity & basement gradient crest (+3.5 px along +u_up)
    combined_subsurf_step = (
        0.45 * rank01_fp(grav_thdr_log * grav_tilt_zero, footprint)
        + 0.35 * rank01_fp(base_grad_log, footprint)
        + 0.20 * rank01_fp(cg30, footprint)
    ).astype(np.float32)
    h77v_13_realigned_crest = transport_upslope_2d(combined_subsurf_step, uy_up, ux_up, shift_px=3.5)
    put("h77v_13_upslope_realigned_grav_base_crest", h77v_13_realigned_crest)

    # H77V-3 Channel 14: Piedmont-toe concavity locked to upslope-realigned gravity/basement crest
    pied_concavity = (
        np.maximum(lap_elev, 0.0)
        * np.log1p(np.maximum(curv_prof, 0.0))
        * np.exp(-np.maximum(det_slope, 0.0) / 18.0)
    ).astype(np.float32)
    h77v_14_piedmont_lock = (
        rank01_fp(pied_concavity, footprint) * (0.4 + 0.6 * h77v_13_realigned_crest)
    ).astype(np.float32)
    put("h77v_14_piedmont_toe_concavity_grav_lock", h77v_14_piedmont_lock)

    d_pot_fp = (
        0.35 * rank01_fp(cg30, footprint)
        + 0.25 * rank01_fp(mag_as_log * mag_tilt_zero, footprint)
        + 0.25 * h77v_13_realigned_crest
        + 0.15 * rank01_fp(np.maximum(base_grav_align, 0.0), footprint)
    )[footprint].astype(np.float32)
    del mag_tilt_zero, mag_as_log, grav_thdr_log, grav_tilt_zero, cg30, ortho30
    del base_grad_log, base_grav_align, combined_subsurf_step, lap_elev, pied_concavity
    gc.collect()

    # === STAGE 4: Block A Up-Gradient Advection Backtracking & Block C Alteration Tri-Pole ===
    (
        h77v_00_advect,
        h77v_01_shift,
        h77v_02_kinetic,
        h77v_03_orifice,
        h77v_04_probe2m,
        iso_control_plume,
    ) = build_advection_backtracked_hydro_channels(data_dir, transform, footprint, z_ortho, uy_up, ux_up)
    del z_ortho, uy_up, ux_up

    h77v_05_step_conv = (
        (0.65 * rank01_fp(h77v_00_advect, footprint) + 0.35 * rank01_fp(h77v_03_orifice + h77v_04_probe2m, footprint))
        * (0.55 * h77v_13_realigned_crest + 0.45 * rank01_fp(h77v_14_piedmont_lock, footprint))
    ).astype(np.float32)

    put("h77v_00_advect_backtrack_plume", h77v_00_advect)
    put("h77v_01_advect_vs_iso_shift", h77v_01_shift)
    put("h77v_02_cation_silica_kinetic_backtrack", h77v_02_kinetic)
    put("h77v_03_orifice_spring_sinter_pin", h77v_03_orifice)
    put("h77v_04_probe2m_upgradient_plume", h77v_04_probe2m)
    put("h77v_05_backtrack_step_convergence", h77v_05_step_conv)

    # Block C: Clay-cap breach annulus, RTP demagnetization trough, and Hydrothermal Tri-Pole (H77V-2)
    cgy, cgx = np.gradient(gf32(cond_log, 2.0))
    cond_grad = np.hypot(cgy, cgx).astype(np.float32)
    del cgy, cgx
    cond_dome = np.maximum(0.0, gf32(cond_log, 8.0) - gf32(cond_log, 24.0)).astype(np.float32)
    # Suppress flat saline playa floors (det_elev < -15 m and det_slope < 0.8 deg)
    playa_supp = (1.0 - 0.75 * ((det_elev < -15.0) & (det_slope < 0.8)).astype(np.float32)).astype(np.float32)
    h77v_10_clay_breach = (cond_grad * (0.4 + 0.6 * rank01_fp(cond_dome, footprint)) * playa_supp).astype(np.float32)
    del cond_log, cond_grad, cond_dome, playa_supp, det_elev, det_slope

    with rasterio.open(data_dir / "external" / "geodawn_extensions_u8.tif") as es:
        tmi_up150 = clean_band(es.read(4), footprint)
    demag_diff = np.maximum(0.0, rank01_fp(tmi_up150, footprint) - rank01_fp(rtp, footprint)).astype(np.float32)
    rtp_trough = np.maximum(0.0, -(gf32(rtp, 3.0, order=(2, 0)) + gf32(rtp, 3.0, order=(0, 2)))).astype(np.float32)
    del tmi_up150, rtp
    h77v_11_demag_clay = (
        (0.55 * demag_diff + 0.45 * rank01_fp(rtp_trough, footprint))
        * (0.35 + 0.65 * rank01_fp(h77v_10_clay_breach, footprint))
    ).astype(np.float32)
    del demag_diff, rtp_trough

    with rasterio.open(data_dir / "external" / "geodawn_rad_u8.tif") as rs:
        rk = rank01_fp(clean_band(rs.read(1), footprint), footprint)
        rth = rank01_fp(clean_band(rs.read(2), footprint), footprint)
    k_th_excess = np.maximum(0.0, rk - rth).astype(np.float32)
    del rk, rth

    plume_gate = (
        0.50 * rank01_fp(h77v_00_advect, footprint)
        + 0.30 * rank01_fp(h77v_03_orifice, footprint)
        + 0.20 * rank01_fp(h77v_02_kinetic + h77v_04_probe2m, footprint)
    ).astype(np.float32)
    h77v_12_tripole = (
        plume_gate
        * (
            0.40 * rank01_fp(h77v_10_clay_breach, footprint)
            + 0.35 * rank01_fp(h77v_11_demag_clay, footprint)
            + 0.25 * k_th_excess
        )
    ).astype(np.float32)
    del k_th_excess

    put("h77v_10_claycap_breach_annulus", h77v_10_clay_breach)
    put("h77v_11_demag_trough_clay_coupling", h77v_11_demag_clay)
    put("h77v_12_hydrothermal_tripole_index", h77v_12_tripole)
    del h77v_10_clay_breach, h77v_11_demag_clay

    # === STAGE 5: Road-deconfounded 1m LiDAR + 10m 3DEP DEM + Block B Dilational Conduit Tensor ===
    with rasterio.open(data_dir / "external" / "tiger_road_distance_m.tif") as s:
        road_d = np.maximum(s.read(1).astype(np.float32), 0.0)
    w_road = (1.0 - 0.85 * np.exp(-((road_d / 90.0) ** 2))).astype(np.float32)
    put("tiger_road_dist_log", np.log1p(road_d))
    del road_d

    lj = json.loads((data_dir / "external" / "lidar_scarp_features.json").read_text())
    q_rules = lj["quantisation"]

    def read_lidar(ls: rasterio.DatasetReader, idx_b: int, bname: str) -> np.ndarray:
        q = ls.read(idx_b).astype(np.float32)
        xmax, mode_q = q_rules[bname]
        t = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
        return (xmax * (t * t if mode_q == "sqrt" else t)).astype(np.float32)

    with rasterio.open(data_dir / "external" / "lidar_scarp_features_u8.tif") as ls:
        has_lidar = footprint & (ls.read(12) > 0)
        ex_max = read_lidar(ls, 1, "ex_max")
        step_m = read_lidar(ls, 3, "step_max")
        lapn = read_lidar(ls, 4, "lapneg_max")
        lapp = read_lidar(ls, 5, "lappos_max")
        downface = read_lidar(ls, 6, "downface_max")
        upface = read_lidar(ls, 7, "upface_max")
        coh100 = read_lidar(ls, 10, "coh100")
        strike_lid = read_lidar(ls, 11, "strike")

    dipole_1m = (np.sqrt(np.maximum(lapn * lapp, 0.0)) * w_road).astype(np.float32)
    del lapn, lapp
    step_1m = (step_m * w_road).astype(np.float32)
    del step_m

    th_lid_rad = np.deg2rad(strike_lid)
    w_str = np.where(has_lidar, ex_max * coh100 * w_road, 0.0).astype(np.float32)
    c2 = gf32(w_str * np.cos(2.0 * th_lid_rad), sigma=2.5, mode="constant")
    s2 = gf32(w_str * np.sin(2.0 * th_lid_rad), sigma=2.5, mode="constant")
    strike_cont = np.hypot(c2, s2).astype(np.float32)
    del w_str, c2, s2

    put("lidar_ex_max", ex_max, mask_valid=has_lidar)
    put("lidar_step_deconf", step_1m, mask_valid=has_lidar)
    put("lidar_dipole_deconf", dipole_1m, mask_valid=has_lidar)
    put("lidar_strike_cont", strike_cont, mask_valid=has_lidar)
    del ex_max, step_1m

    # 10 m 3DEP DEM road-deconfounded channels & seamless strike angle theta(r, c)
    s90_dec = (slope_p90 * w_road).astype(np.float32)
    cprof_dec = (curv_prof * w_road).astype(np.float32)
    st_ridge10 = (s90_dec * asp_coh * (0.5 + 0.5 * rank01_fp(hs_lin, footprint))).astype(np.float32)
    put("dem10_slope_p90_deconf", s90_dec)
    put("dem10_curv_prof_deconf", cprof_dec)
    put("dem10_st_ridge_deconf", st_ridge10)

    # Seamless structural strike angle theta(r, c) in radians: use 1m LiDAR strike where valid, else 10m DEM structure tensor
    gy10, gx10 = np.gradient(gf32(s90_dec, 1.5))
    jxx = gf32(gx10 * gx10, 2.5)
    jyy = gf32(gy10 * gy10, 2.5)
    jxy = gf32(gx10 * gy10, 2.5)
    del gy10, gx10
    # Normal direction is 0.5 * arctan2(2*jxy, jxx - jyy); strike is normal + pi/2
    th_dem10_rad = (0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + 0.5 * np.pi) % np.pi
    del jxx, jyy, jxy
    theta_rad = np.where(has_lidar, th_lid_rad % np.pi, th_dem10_rad).astype(np.float32)
    del th_lid_rad, th_dem10_rad, strike_lid

    # Block B: Critically Stressed Transtensional Dilational Conduit Tensor (H77V-1 Channels 6..9)
    # Channel 6: Transtensional dilatation-to-total strain partitioning ratio * log(1 + |geod_2ndinv|)
    strain_tot = np.hypot(geod_shear, geod_dilat) + 1e-3
    dil_part = (geod_dilat / strain_tot).astype(np.float32)
    geod_amp = rank01_fp(np.abs(geod2), footprint)
    h77v_06_dil_ratio = (dil_part * (0.35 + 0.65 * geod_amp)).astype(np.float32)

    # Optimal normal/transtensional fault strike in NW Nevada: theta_opt = N20E (20 deg = 0.349066 rad)
    # perpendicular to sigma_hmin = N110E (Morris, Ferrill & Henderson 1996; Faulds & Hinz 2015)
    theta_opt = np.float32(np.deg2rad(20.0))
    d_theta = theta_rad - theta_opt
    t_dilation = (np.cos(d_theta) ** 2).astype(np.float32)
    t_slip = np.abs(np.sin(2.0 * d_theta)).astype(np.float32)

    r_lid_scarp = rank01_fp(np.where(has_lidar, strike_cont * (0.5 + dipole_1m), np.nan), footprint)
    r_d10_scarp = rank01_fp(st_ridge10 * (0.5 + rank01_fp(cprof_dec, footprint)), footprint)
    scarp_edge = np.where(has_lidar, 0.65 * r_lid_scarp + 0.35 * r_d10_scarp, r_d10_scarp).astype(np.float32)
    del r_lid_scarp, r_d10_scarp, strike_cont, dipole_1m, st_ridge10, cprof_dec

    pos_dil_rank = rank01_fp(np.maximum(0.0, geod_dilat), footprint)
    shear_rank = rank01_fp(np.abs(geod_shear), footprint)
    h77v_07_dil_tend = (t_dilation * scarp_edge * (0.40 + 0.60 * pos_dil_rank)).astype(np.float32)
    h77v_08_slip_tend = (t_slip * scarp_edge * (0.40 + 0.60 * shear_rank)).astype(np.float32)

    # Channel 9: Step-over strain-partitioning gradient * local strike curvature dispersion
    rdy, rdx = np.gradient(gf32(h77v_06_dil_ratio, 4.0))
    dil_grad = np.hypot(rdy, rdx).astype(np.float32)
    del rdy, rdx
    c2_s = gf32(np.cos(2.0 * theta_rad), 4.0)
    s2_s = gf32(np.sin(2.0 * theta_rad), 4.0)
    strike_disp = np.clip(1.0 - np.hypot(c2_s, s2_s), 0.0, 1.0).astype(np.float32)
    del c2_s, s2_s, theta_rad
    h77v_09_stepover_gate = (
        rank01_fp(dil_grad, footprint) * strike_disp * (0.4 + 0.6 * scarp_edge)
    ).astype(np.float32)
    del dil_grad, strike_disp, dil_part, strain_tot, t_dilation, t_slip, pos_dil_rank, shear_rank

    put("h77v_06_transtensional_dilation_ratio", h77v_06_dil_ratio)
    put("h77v_07_crit_stress_dilation_tendency", h77v_07_dil_tend)
    put("h77v_08_crit_stress_slip_tendency", h77v_08_slip_tend)
    put("h77v_09_stepover_strain_curvature_gate", h77v_09_stepover_gate)

    # Block D Channel 15: 1 m LiDAR one-sided face polarity (downface - upface) coupled with vent plume & dilation
    face_pol_1m = (np.maximum(0.0, downface - 0.65 * upface) * coh100 * w_road).astype(np.float32)
    face_pol_10m = (s90_dec * asp_coh * w_road).astype(np.float32)
    r_pol_1m = rank01_fp(np.where(has_lidar, face_pol_1m, np.nan), footprint)
    r_pol_10m = rank01_fp(face_pol_10m, footprint)
    scarp_pol = np.where(has_lidar, 0.70 * r_pol_1m + 0.30 * r_pol_10m, r_pol_10m).astype(np.float32)
    h77v_15_pol_vent = (
        scarp_pol * (0.40 + 0.35 * plume_gate + 0.25 * rank01_fp(h77v_07_dil_tend, footprint))
    ).astype(np.float32)
    put("h77v_15_lidar_scarp_polarity_vent_lock", h77v_15_pol_vent)

    # Save domain priors & Exp 1 physical surface comparison vectors on footprint
    s_h77v_advect_fp = (
        0.30 * rank01_fp(h77v_05_step_conv, footprint)
        + 0.20 * rank01_fp(h77v_12_tripole, footprint)
        + 0.25 * rank01_fp(h77v_07_dil_tend + 0.5 * h77v_09_stepover_gate, footprint)
        + 0.25 * rank01_fp(h77v_15_pol_vent + 0.5 * h77v_14_piedmont_lock, footprint)
    )[footprint].astype(np.float32)

    s_hydro_iso_fp = (
        0.55 * rank01_fp(iso_control_plume, footprint)
        + 0.45 * scarp_edge
    )[footprint].astype(np.float32)

    d_scarp_fp = scarp_edge[footprint].astype(np.float32)
    d_hydro_fp = rank01_fp(h77v_05_step_conv + h77v_12_tripole, footprint)[footprint].astype(np.float32)
    d_mech_fp = rank01_fp(
        0.5 * geod_amp * rank01_fp(ieq, footprint)
        + 0.3 * rank01_fp(h77v_07_dil_tend, footprint)
        + 0.2 * np.exp(-np.maximum(deq, 0.0) / 8.0),
        footprint,
    )[footprint].astype(np.float32)
    geod_rank_fp = geod_amp[footprint].astype(np.float32)

    mm.flush()
    np.savez_compressed(
        dom_path,
        d_scarp=d_scarp_fp,
        d_pot=d_pot_fp,
        d_hydro=d_hydro_fp,
        d_mech=d_mech_fp,
        geod_rank=geod_rank_fp,
        s_h77v_advect=s_h77v_advect_fp,
        s_hydro_iso=s_hydro_iso_fp,
    )
    meta = {
        "lane": "H77V-VENT-ADVECT-CONDUIT",
        "shape": [n_fp, n_ch],
        "dtype": "float32",
        "channels": ALL_STATIC_NAMES,
        "h77v_channels": H77V_CHANNEL_NAMES,
        "base_channels": BASE_CHANNEL_NAMES,
        "loco_channels": LOCO_NAMES,
        "footprint_pixels": n_fp,
        "lidar_valid_pixels": int(has_lidar.sum()),
        "lidar_missing_pixels_mapped_to_nan": int((footprint & ~has_lidar).sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    gc.collect()
    return np.load(stack_path, mmap_mode="r"), meta


def load_h77v_domains(work_dir: Path) -> dict[str, np.ndarray]:
    """Load cached domain composite vectors (`d_scarp`, `d_pot`, `d_hydro`, `d_mech`, `geod_rank`, etc.)."""
    z = np.load(work_dir / "h77v_domains_fp.npz")
    return {k: z[k].astype(np.float32) for k in z.files}


def build_h77v_fold_mechanics(
    visible_mask: np.ndarray,
    footprint: np.ndarray,
    geod_rank_fp: np.ndarray,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Compute or load 4 strictly fold-visible LOCO mechanics features `(N_fp, 4)`."""
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    geod_2d = np.zeros(footprint.shape, dtype=np.float32)
    geod_2d[footprint] = geod_rank_fp
    mech = compute_visible_mechanics(visible_mask, footprint, geod_2d)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mech.astype(np.float32))
    return mech
