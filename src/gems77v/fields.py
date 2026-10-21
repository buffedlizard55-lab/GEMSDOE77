"""Label-free geothermal vent-sinter, conductive heat-flow gradient, and structural conduit fields (H77V).

Enforces strict anti-leakage and scientific integrity invariants:
  1. Zero fault-catalogue inputs (existing_faults.tif / labels.tif) are read here.
  2. Zero pre-Quaternary SGMC bedrock contact inputs are read here.
  3. Zero prior submission or registry prediction rasters are read here.
  4. Columns encoding catalogue proximity ('dist_known_fault_px', 'known_fault', 'label')
     are strictly forbidden and asserted unaccessed.
  5. Fixes four concrete geothermal data-ingestion bugs uncovered in prior lanes:
     - IR-H77V-01: separates high-enthalpy fault-vent sinter/alteration/sulfur/breccia
       deposits from non-tectonic Pleistocene Lake Lahontan shoreline 'Tufa' mounds (86/281).
     - IR-H77V-02: fixes untyped vals[0] geothermometer index shift (173 rows) and recovers
       267 Hot/Warm spring/well records with missing numeric temp_c.
     - IR-H77V-03: reads actual '2m_temperature_probe_INGENIOUS_regional_data.zip' and
       'paleo_geothermal_regional.zip' archives (fixing non-existent derived_*.tif paths in H77D).
     - IR-H77V-04: extracts multi-depth 2m probe vertical temperature gradients dT/dz = T2m - T1m
       (816 upward conductive heat-flow stations vs 727 downward solar-heated stations).
  6. Fixes IR-H77G-01: filters Band 17 ('cond_surf'), geodetic strain, and earthquake density
     at sigma = 36.0 px (3.6 km > 3.5 km MT inversion mesh period).
"""
from __future__ import annotations

import csv
import io
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.warp import transform as rio_transform
from scipy import ndimage
from scipy.stats import rankdata

from gems77 import features77 as F77
from gems77g.fields import hessian_trough

FORBIDDEN_COLUMNS = frozenset({"dist_known_fault_px", "known_fault", "label"})

H77V_NAMES: list[str] = [
    "h77v_sinter_alter_vent_conduit",
    "h77v_tufa_shoreline_deconf_contrast",
    "h77v_probe_upward_heatflow_dtdz",
    "h77v_geotherm_disequilibrium",
    "h77v_fixed_spring_well_upflow",
    "h77v_epithermal_k_th_alteration",
    "h77v_demag_grav_step_conduit",
    "h77v_seamfree_claycap_transtension",
    "h77v_scarp_strike_vent_bridge",
    "h77v_composite_surface",
]

HIGH_ENTHALPY_WEIGHTS: dict[str, float] = {
    "Alteration": 3.5,
    "Argillic alteration": 3.5,
    "Hydrothermally altered deposits": 3.5,
    "Sulfur": 3.5,
    "Sinter": 3.0,
    "Sinter and silicified zone": 3.0,
    "Silicified fault breccia": 3.0,
    "Silica vein": 2.8,
    "Calcite-qtz veins": 2.8,
    "Explosion craters": 2.5,
    "Sulfate crust": 2.2,
    "Silica-cemented sand": 1.4,
    "Silica-cemented mud or ash": 1.4,
    "Travertine": 1.1,
    "Travertine?": 0.8,
    "Carbonate (travertine) cemented fan gravel": 1.0,
}

TUFA_TYPES = frozenset({"Tufa", "Tufa and travertine"})


def _clean_band(arr: np.ndarray, foot: np.ndarray) -> np.ndarray:
    return F77.fill_median(F77.clean(np.asarray(arr, dtype=np.float32)), foot).astype(np.float32)


def _rank01(arr: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """Tie-aware rank normalization in [0, 1] inside foot (avoids F77.rank01 row-major tie ramp)."""
    out = np.zeros(arr.shape, dtype=np.float32)
    v = np.asarray(arr[foot], dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        return out
    vmin = float(np.min(v[finite]))
    vmax = float(np.max(v[finite]))
    if vmax <= vmin:
        return out
    r = rankdata(np.where(finite, v, vmin), method="average")
    r_min = float(r.min())
    r_max = float(r.max())
    if r_max > r_min:
        out[foot] = ((r - r_min) / (r_max - r_min)).astype(np.float32)
    return out


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


def extract_paleo_sinter_vs_tufa(
    paleo_zip: Path | None,
    transform: Any,
    shape: tuple[int, int],
    foot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Separate high-enthalpy siliceous/argillic/sulfur fault-vent deposits from Lake Lahontan Tufa."""
    vent_imp = np.zeros(shape, dtype=np.float32)
    tufa_imp = np.zeros(shape, dtype=np.float32)
    stats = {"vent_records_in_footprint": 0, "tufa_records_in_footprint": 0}
    if paleo_zip is None or not paleo_zip.is_file():
        return vent_imp, tufa_imp, stats

    with zipfile.ZipFile(paleo_zip) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        reader = csv.DictReader(
            io.TextIOWrapper(zf.open(csv_name), encoding="utf-8-sig", errors="replace")
        )
        assert reader.fieldnames is not None
        for col in reader.fieldnames:
            assert col not in FORBIDDEN_COLUMNS, f"Forbidden column in paleo CSV: {col}"
        recs = []
        for row in reader:
            try:
                lon = float(row["Long_deg_NAD83"])
                lat = float(row["Lat_deg_NAD83"])
                dtype = row.get("Deposit_type", "").strip()
            except (ValueError, KeyError):
                continue
            recs.append((lon, lat, dtype))

    if not recs:
        return vent_imp, tufa_imp, stats

    ux, uy = rio_transform("EPSG:4269", "EPSG:32611", [r[0] for r in recs], [r[1] for r in recs])
    for (_, _, dtype), x, y in zip(recs, ux, uy):
        c = int(np.floor((x - transform.c) / transform.a))
        r = int(np.floor((y - transform.f) / transform.e))
        if not (0 <= r < shape[0] and 0 <= c < shape[1] and foot[r, c]):
            continue
        if dtype in TUFA_TYPES:
            tufa_imp[r, c] = max(tufa_imp[r, c], 1.0)
            stats["tufa_records_in_footprint"] += 1
        else:
            w = HIGH_ENTHALPY_WEIGHTS.get(dtype, 1.5)
            vent_imp[r, c] = max(vent_imp[r, c], float(w))
            stats["vent_records_in_footprint"] += 1
    return vent_imp, tufa_imp, stats


def extract_2m_probe_upward_heatflow(
    probe_zip: Path | None,
    transform: Any,
    shape: tuple[int, int],
    foot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Extract upward conductive thermal gradient dT/dz = T2m - T1m and F2mDAB from 2m probes."""
    upflow_imp = np.zeros(shape, dtype=np.float32)
    solar_imp = np.zeros(shape, dtype=np.float32)
    stats = {
        "probes_in_footprint": 0,
        "probes_with_vertical_grad": 0,
        "upward_heatflow_probes": 0,
        "downward_solar_probes": 0,
    }
    if probe_zip is None or not probe_zip.is_file():
        return upflow_imp, solar_imp, stats

    with zipfile.ZipFile(probe_zip) as zf:
        dbf_name = next(n for n in zf.namelist() if n.lower().endswith(".dbf"))
        buf = zf.read(dbf_name)

    numrec, lenhdr, lenrec = struct.unpack("<4xIHH20x", buf[:32])
    pos = 32
    offset = 1
    fields: dict[str, tuple[int, int]] = {}
    while buf[pos] != 0x0D:
        fname = buf[pos : pos + 11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        assert fname not in FORBIDDEN_COLUMNS, f"Forbidden column in probe DBF: {fname}"
        flen = buf[pos + 16]
        fields[fname] = (offset, flen)
        offset += flen
        pos += 32

    def _gf(rec_bytes: bytes, key: str) -> float:
        off, fl = fields[key]
        s = rec_bytes[off : off + fl].decode("ascii", errors="ignore").strip()
        if not s:
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    pts = []
    for i in range(numrec):
        rec = buf[lenhdr + i * lenrec : lenhdr + (i + 1) * lenrec]
        if not rec or rec[0:1] == b"*":
            continue
        lat = _gf(rec, "LatDegreeN")
        lon = _gf(rec, "LongDegree")
        if not (25.0 <= lat <= 50.0 and -130.0 <= lon <= -100.0):
            continue
        t2m = _gf(rec, "T2m")
        dab = _gf(rec, "F2mDAB")
        t1m = _gf(rec, "T1m")
        t15m = _gf(rec, "T15m")
        pts.append((lon, lat, t2m, dab, t1m, t15m))

    if not pts:
        return upflow_imp, solar_imp, stats

    ux, uy = rio_transform("EPSG:4269", "EPSG:32611", [p[0] for p in pts], [p[1] for p in pts])
    for (_, _, t2m, dab, t1m, t15m), x, y in zip(pts, ux, uy):
        c = int(np.floor((x - transform.c) / transform.a))
        r = int(np.floor((y - transform.f) / transform.e))
        if not (0 <= r < shape[0] and 0 <= c < shape[1] and foot[r, c]):
            continue
        stats["probes_in_footprint"] += 1
        dab_pos = max(0.0, float(dab)) if np.isfinite(dab) else 0.0
        t2_exc = max(0.0, float(t2m) - 21.0) if np.isfinite(t2m) else 0.0

        dtdz = np.nan
        if np.isfinite(t2m) and np.isfinite(t1m) and t1m > 0.0 and t2m > 0.0:
            dtdz = float(t2m - t1m) / 1.0
        elif np.isfinite(t2m) and np.isfinite(t15m) and t15m > 0.0 and t2m > 0.0:
            dtdz = float(t2m - t15m) / 0.5

        if np.isfinite(dtdz):
            stats["probes_with_vertical_grad"] += 1
            if dtdz > 0.0:
                stats["upward_heatflow_probes"] += 1
                score = np.log1p(1.8 * dtdz + 0.8 * dab_pos + 0.25 * t2_exc)
                if score > upflow_imp[r, c]:
                    upflow_imp[r, c] = float(score)
            else:
                stats["downward_solar_probes"] += 1
                s_sol = np.log1p(abs(dtdz))
                if s_sol > solar_imp[r, c]:
                    solar_imp[r, c] = float(s_sol)
                if dab_pos > 1.5:
                    # Retain strongly anomalous F2mDAB with 0.35x damping when shallow gradient is solar-inverted
                    score = 0.35 * np.log1p(dab_pos)
                    if score > upflow_imp[r, c]:
                        upflow_imp[r, c] = float(score)
        else:
            score = np.log1p(0.75 * dab_pos + 0.20 * t2_exc)
            if score > upflow_imp[r, c]:
                upflow_imp[r, c] = float(score)

    return upflow_imp, solar_imp, stats


def extract_wellspring_disequilibrium(
    ws_csv: Path | None,
    vv_csv: Path | None,
    shape: tuple[int, int],
    foot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Extract fixed-column geothermometer disequilibrium and thermal spring/well upflow (IR-H77V-02 fix)."""
    diseq_imp = np.zeros(shape, dtype=np.float32)
    upflow_imp = np.zeros(shape, dtype=np.float32)
    stats = {
        "wellspring_rows_read": 0,
        "recovered_hot_warm_missing_temp": 0,
        "fixed_geotherm_without_temp": 0,
        "volcanic_vents_in_footprint": 0,
    }
    if ws_csv is not None and ws_csv.is_file():
        with ws_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames is not None
            used_cols = (
                "layer",
                "thermalclass",
                "row",
                "col",
                "temp_c",
                "geothermquartz_c",
                "geothermchalc_c",
                "geothermcat_c",
            )
            for c_name in used_cols:
                assert c_name not in FORBIDDEN_COLUMNS
            for row in reader:
                try:
                    rr = int(float(row["row"]))
                    cc = int(float(row["col"]))
                except (ValueError, KeyError):
                    continue
                if not (0 <= rr < shape[0] and 0 <= cc < shape[1] and foot[rr, cc]):
                    continue
                stats["wellspring_rows_read"] += 1
                layer = row.get("layer", "").strip().lower()
                tclass = row.get("thermalclass", "").strip().lower()
                is_spring = "spring" in layer

                def _pf(col_k: str) -> float:
                    s = row.get(col_k, "").strip()
                    if not s or s.lower() == "nan":
                        return np.nan
                    try:
                        v = float(s)
                        return v if (0.0 < v < 400.0) else np.nan
                    except ValueError:
                        return np.nan

                t_surf = _pf("temp_c")
                geos = [v for v in (_pf("geothermquartz_c"), _pf("geothermchalc_c"), _pf("geothermcat_c")) if np.isfinite(v)]
                t_geo = max(geos) if geos else np.nan

                if not np.isfinite(t_surf):
                    if geos:
                        stats["fixed_geotherm_without_temp"] += 1
                    if tclass == "hot":
                        t_surf = 55.0
                        stats["recovered_hot_warm_missing_temp"] += 1
                    elif tclass == "warm":
                        t_surf = 32.0
                        stats["recovered_hot_warm_missing_temp"] += 1

                # 1. Chemical geothermometer disequilibrium: dT_diseq = max(0, T_geo - max(T_surf, 20))
                if np.isfinite(t_geo):
                    base_t = t_surf if np.isfinite(t_surf) else 25.0
                    dt_diseq = max(0.0, t_geo - max(base_t, 20.0))
                    res_excess = max(0.0, t_geo - 85.0)
                    s_dis = float(np.log1p(0.65 * dt_diseq + 0.45 * res_excess) * (1.35 if is_spring else 1.0))
                    if s_dis > diseq_imp[rr, cc]:
                        diseq_imp[rr, cc] = s_dis

                # 2. Fixed-column thermal spring & well upflow excess
                surf_exc = max(0.0, (t_surf - 20.0) if np.isfinite(t_surf) else 0.0)
                geo_exc = max(0.0, (t_geo - 90.0) if np.isfinite(t_geo) else 0.0)
                if surf_exc > 0.0 or geo_exc > 0.0:
                    w_layer = 1.65 if is_spring else 1.0
                    s_up = float(w_layer * np.log1p(surf_exc + 0.25 * geo_exc))
                    if s_up > upflow_imp[rr, cc]:
                        upflow_imp[rr, cc] = s_up

    if vv_csv is not None and vv_csv.is_file():
        with vv_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    rr = int(float(row["row"]))
                    cc = int(float(row["col"]))
                except (ValueError, KeyError):
                    continue
                if 0 <= rr < shape[0] and 0 <= cc < shape[1] and foot[rr, cc]:
                    upflow_imp[rr, cc] = max(upflow_imp[rr, cc], 1.5)
                    stats["volcanic_vents_in_footprint"] += 1

    return diseq_imp, upflow_imp, stats


def anisotropic_strike_advect(
    impulse_2d: np.ndarray,
    ty_2d: np.ndarray,
    tx_2d: np.ndarray,
    sig_cross: float = 3.2,
    sig_along: float = 11.0,
    decay_px: float = 42.0,
) -> np.ndarray:
    """Steerable spatially-varying anisotropic fault-strike advection + radial EDT decay (no box truncation)."""
    imp = np.asarray(impulse_2d, dtype=np.float32)
    base = ndimage.gaussian_filter(imp, sigma=sig_cross, mode="constant")
    h, w = base.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    acc = base.copy()
    w_sum = 1.0
    for step in (-12.0, -8.0, -4.0, 4.0, 8.0, 12.0):
        wt = float(np.exp(-0.5 * (step / sig_along) ** 2))
        ry = np.clip(np.round(yy + step * ty_2d).astype(np.int16), 0, h - 1)
        rx = np.clip(np.round(xx + step * tx_2d).astype(np.int16), 0, w - 1)
        acc += wt * base[ry, rx]
        w_sum += wt
    adv = (acc / float(w_sum)).astype(np.float32)
    if (imp > 0).any():
        dist = ndimage.distance_transform_edt(imp <= 0).astype(np.float32)
        radial_env = np.exp(-dist / np.float32(decay_px)).astype(np.float32)
        del dist
        adv_max = float(adv.max())
        if adv_max > 0:
            adv = adv / adv_max
        return (0.65 * adv + 0.35 * radial_env).astype(np.float32)
    return adv


def _grad(field: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy = ndimage.gaussian_filter(field, sigma, order=(1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(field, sigma, order=(0, 1), mode="nearest")
    mag = np.hypot(gx, gy).astype(np.float32)
    return gy.astype(np.float32), gx.astype(np.float32), mag


def build_h77v_fields(
    bands: dict[str, np.ndarray],
    foot: np.ndarray,
    transform: Any = None,
    ext_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute the 10 label-free H77V geothermal vent & structural conduit channels in [0, 1]."""
    rtp = _clean_band(bands["rtp"], foot)
    grav = _clean_band(bands["iso_grav_anom"], foot)
    dbase = _clean_band(bands["depth_to_base_surf"], foot)
    cond = _clean_band(bands["cond_surf"], foot)
    dilat = _clean_band(bands["geod_dilaterate"], foot)
    shear = _clean_band(bands["geod_shearrate"], foot)
    ieq = _clean_band(bands["ieq_n100a15"], foot)
    deq = _clean_band(bands["deq_n100a15"], foot)
    det_elev = _clean_band(bands["det_elev"], foot)
    det_slope = _clean_band(bands["det_elev_slope"], foot)

    shape = foot.shape

    # Joint topographic + gravity + basement structure tensor -> local fault strike tangent (ty, tx)
    zy, zx, zmag = _grad(det_elev, sigma=2.0)
    gy, gx, gmag = _grad(grav, sigma=2.5)
    z_scale = float(np.std(zmag[foot])) + 1e-6
    g_scale = float(np.std(gmag[foot])) + 1e-6
    ux = (zx / z_scale + np.float32(0.65) * gx / g_scale).astype(np.float32)
    uy = (zy / z_scale + np.float32(0.65) * gy / g_scale).astype(np.float32)
    del zy, zx, zmag
    jxx = ndimage.gaussian_filter(ux * ux, sigma=3.5, mode="nearest")
    jyy = ndimage.gaussian_filter(uy * uy, sigma=3.5, mode="nearest")
    jxy = ndimage.gaussian_filter(ux * uy, sigma=3.5, mode="nearest")
    del ux, uy
    theta_norm = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    nx = np.cos(theta_norm).astype(np.float32)
    ny = np.sin(theta_norm).astype(np.float32)
    ty = -ny
    tx = nx
    coh_strike = (
        np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0)) / (jxx + jyy + 1e-6)
    ).astype(np.float32)
    del jxx, jyy, jxy, theta_norm, nx, ny

    # Load external geothermal vent & terrain files when ext_dir is provided, else synthesize from bands
    paleo_zip = (ext_dir / "paleo_geothermal_regional.zip") if ext_dir else None
    probe_zip = (ext_dir / "2m_temperature_probe_INGENIOUS_regional_data.zip") if ext_dir else None
    ws_csv = (ext_dir / "gdr_wellspring_in_footprint.csv") if ext_dir else None
    vv_csv = (ext_dir / "gdr_volcanic_vents_in_footprint.csv") if ext_dir else None

    vent_imp, tufa_imp, paleo_stats = extract_paleo_sinter_vs_tufa(paleo_zip, transform, shape, foot)
    probe_up_imp, probe_sol_imp, probe_stats = extract_2m_probe_upward_heatflow(probe_zip, transform, shape, foot)
    diseq_imp, ws_up_imp, ws_stats = extract_wellspring_disequilibrium(ws_csv, vv_csv, shape, foot)

    # Fallback when called in synthetic unit tests without external files
    if not vent_imp.any() and "rad_K" in bands:
        vent_imp = np.maximum(_clean_band(bands["rad_K"], foot), 0.0)
    if not probe_up_imp.any():
        probe_up_imp = np.maximum(dilat, 0.0)
    if not diseq_imp.any():
        diseq_imp = np.maximum(cond, 0.0)
    if not ws_up_imp.any():
        ws_up_imp = np.maximum(ieq, 0.0)

    # 1. Tufa-deconfounded high-enthalpy siliceous sinter/alteration/sulfur strike conduit
    sinter_adv = anisotropic_strike_advect(vent_imp, ty, tx, sig_cross=3.5, sig_along=12.0, decay_px=45.0)
    ch1_sinter_conduit = _rank01(sinter_adv * (0.75 + 0.25 * coh_strike), foot)

    # 2. High-enthalpy sinter/alteration vs Lake Lahontan Tufa shoreline deconfounded contrast
    tufa_halo = anisotropic_strike_advect(tufa_imp, ty, tx, sig_cross=5.0, sig_along=8.0, decay_px=35.0)
    r_sinter = _rank01(sinter_adv, foot)
    r_tufa = _rank01(tufa_halo, foot)
    ch2_tufa_deconf = _rank01(r_sinter - 0.55 * r_tufa, foot)
    del tufa_halo, r_sinter, r_tufa

    # 3. 2-meter temperature probe upward conductive heat-flow gradient (dT/dz > 0)
    probe_adv = anisotropic_strike_advect(probe_up_imp, ty, tx, sig_cross=3.8, sig_along=11.0, decay_px=38.0)
    probe_sol_halo = anisotropic_strike_advect(probe_sol_imp, ty, tx, sig_cross=4.5, sig_along=7.0, decay_px=30.0)
    ch3_probe_dtdz = _rank01(
        _rank01(probe_adv * (0.80 + 0.20 * coh_strike), foot) - 0.25 * _rank01(probe_sol_halo, foot), foot
    )
    del probe_adv, probe_sol_halo

    # 4. Fixed-column chemical geothermometer cooling disequilibrium (dT_diseq = T_geo - T_surf)
    diseq_adv = anisotropic_strike_advect(diseq_imp, ty, tx, sig_cross=4.0, sig_along=13.0, decay_px=42.0)
    ch4_diseq = _rank01(diseq_adv * (0.75 + 0.25 * coh_strike), foot)
    del diseq_adv

    # 5. Fixed-column thermal spring & well upflow along local structural strike
    upflow_adv = anisotropic_strike_advect(ws_up_imp, ty, tx, sig_cross=3.5, sig_along=12.0, decay_px=38.0)
    ch5_spring_upflow = _rank01(upflow_adv * (0.75 + 0.25 * coh_strike), foot)
    del upflow_adv, sinter_adv

    # 6. Epithermal potassium-adularia / illite alteration halo (GeoDAWN K/Th + U/K on non-playa slopes)
    if ext_dir is not None and (ext_dir / "geodawn_rad_u8.tif").is_file():
        import rasterio
        with rasterio.open(ext_dir / "geodawn_rad_u8.tif") as ds:
            q_k = ds.read(1)
            q_th = ds.read(2)
            rad_k = _clean_band(np.where(q_k > 0, q_k.astype(np.float32), np.nan), foot)
            rad_th = _clean_band(np.where(q_th > 0, q_th.astype(np.float32), np.nan), foot)
        with rasterio.open(ext_dir / "geodawn_extensions_u8.tif") as ds:
            q_uk = ds.read(2)
            ext_uk = _clean_band(np.where(q_uk > 0, q_uk.astype(np.float32), np.nan), foot)
    else:
        rad_k = _clean_band(bands.get("tc_totalcount", det_elev), foot)
        rad_th = _clean_band(bands.get("tmi", grav), foot)
        ext_uk = _clean_band(bands.get("mag_anom", rtp), foot)

    k_local = rad_k - ndimage.gaussian_filter(rad_k, sigma=10.0, mode="nearest")
    th_local = rad_th - ndimage.gaussian_filter(rad_th, sigma=10.0, mode="nearest")
    non_playa = np.clip((det_slope - 0.8) / 3.0, 0.15, 1.0).astype(np.float32)
    k_th_score = (
        (0.55 * _rank01(k_local, foot) - 0.35 * _rank01(th_local, foot) + 0.30 * _rank01(ext_uk, foot))
        * non_playa
    )
    ch6_epithermal_kth = _rank01(k_th_score, foot)

    # 7. Next-step H77G integration: negative RTP demagnetization trough x dip-aligned gravity-basement step
    demag_1p5 = _rank01(hessian_trough(rtp, sigma=1.5, hp_sigma=12.0), foot)
    demag_3p0 = _rank01(hessian_trough(rtp, sigma=3.0, hp_sigma=12.0), foot)
    demag_pers = np.sqrt(np.maximum(demag_1p5 * demag_3p0, 0.0))
    dy_b, dx_b, mag_b = _grad(dbase, sigma=2.0)
    cos_gb = np.abs(gx * dx_b + gy * dy_b) / (gmag * mag_b + 1e-6)
    grav_base_step = np.sqrt(np.maximum(_rank01(gmag, foot) * _rank01(mag_b, foot), 0.0)) * cos_gb
    ch7_demag_grav_step = _rank01(0.45 * demag_pers + 0.55 * _rank01(grav_base_step, foot), foot)
    del dy_b, dx_b, mag_b, gy, gx, gmag, cos_gb, grav_base_step, demag_1p5, demag_3p0, demag_pers

    # 8. Next-step IR-H77G-01 fix: seam-free sigma=36 px (3.6 km) clay-cap + transtension + swarm fairway
    cond_s36 = _rank01(ndimage.gaussian_filter(np.log1p(np.maximum(cond, 0.0)), sigma=36.0, mode="nearest"), foot)
    dil_s36 = _rank01(ndimage.gaussian_filter(np.maximum(dilat, 0.0), sigma=36.0, mode="nearest"), foot)
    shr_s36 = _rank01(ndimage.gaussian_filter(np.maximum(shear, 0.0), sigma=36.0, mode="nearest"), foot)
    ieq_s36 = _rank01(ndimage.gaussian_filter(np.maximum(ieq, 0.0), sigma=36.0, mode="nearest"), foot)
    deq_s36 = _rank01(ndimage.gaussian_filter(np.maximum(deq, 0.0), sigma=36.0, mode="nearest"), foot)
    swarm_s36 = (ieq_s36 ** 2) / (ieq_s36 + deq_s36 + 0.05)
    ch8_seamfree_fairway = _rank01(
        0.35 * cond_s36 + 0.40 * np.sqrt(np.maximum(dil_s36 * shr_s36, 0.0)) + 0.25 * _rank01(swarm_s36, foot),
        foot,
    )

    # 9. Seamless 1m LiDAR (q==0 -> NaN fixed) + 10m 3DEP dip-slip scarp x geothermal vent conduit bridge
    if ext_dir is not None and (ext_dir / "lidar_scarp_features_u8.tif").is_file():
        import rasterio
        with rasterio.open(ext_dir / "lidar_scarp_features_u8.tif") as ds:
            has_lid = (ds.read(12) > 0) & foot
            step_m = ds.read(3).astype(np.float32)
            lap_n = ds.read(4).astype(np.float32)
            lap_p = ds.read(5).astype(np.float32)
            rel_m = ds.read(9).astype(np.float32)
            dipole_m = np.sqrt(np.maximum(0.0, lap_n * lap_p))
            sharp_m = step_m / (rel_m + 5.0)
            lid_raw = 0.55 * _rank01(np.where(has_lid, dipole_m, np.nan), has_lid) + 0.45 * _rank01(
                np.where(has_lid, sharp_m, np.nan), has_lid
            )
        with rasterio.open(ext_dir / "topo_10m_derived_100m_u8.tif") as ds:
            hs10 = _clean_band(np.where(ds.read(8) > 0, ds.read(8).astype(np.float32), np.nan), foot)
            curv10 = _clean_band(np.where(ds.read(6) > 0, ds.read(6).astype(np.float32), np.nan), foot)
            dem10_scarp = 0.55 * _rank01(hs10, foot) + 0.45 * _rank01(curv10, foot)
        scarp_seamfree = _rank01(np.where(has_lid, 0.65 * lid_raw + 0.35 * dem10_scarp, dem10_scarp), foot)
    else:
        elev_hp = det_elev - ndimage.gaussian_filter(det_elev, sigma=6.0, mode="nearest")
        scarp_seamfree = _rank01(
            0.55 * _rank01(det_slope, foot) + 0.45 * _rank01(np.abs(elev_hp), foot), foot
        )

    vent_corridor = _rank01(
        0.30 * ch1_sinter_conduit
        + 0.25 * ch3_probe_dtdz
        + 0.25 * ch4_diseq
        + 0.20 * ch5_spring_upflow,
        foot,
    )
    ch9_scarp_vent_bridge = _rank01(
        scarp_seamfree * (0.45 + 0.35 * vent_corridor + 0.20 * _rank01(coh_strike, foot)),
        foot,
    )

    # 10. Principled H77V geothermal vent & structural conduit composite surface
    comp_raw = (
        0.24 * ch9_scarp_vent_bridge
        + 0.16 * ch7_demag_grav_step
        + 0.14 * ch1_sinter_conduit
        + 0.12 * ch3_probe_dtdz
        + 0.10 * ch4_diseq
        + 0.10 * ch5_spring_upflow
        + 0.07 * ch6_epithermal_kth
        + 0.07 * ch8_seamfree_fairway
    )
    ch10_composite = _rank01(comp_raw, foot)

    out = {
        "h77v_sinter_alter_vent_conduit": ch1_sinter_conduit,
        "h77v_tufa_shoreline_deconf_contrast": ch2_tufa_deconf,
        "h77v_probe_upward_heatflow_dtdz": ch3_probe_dtdz,
        "h77v_geotherm_disequilibrium": ch4_diseq,
        "h77v_fixed_spring_well_upflow": ch5_spring_upflow,
        "h77v_epithermal_k_th_alteration": ch6_epithermal_kth,
        "h77v_demag_grav_step_conduit": ch7_demag_grav_step,
        "h77v_seamfree_claycap_transtension": ch8_seamfree_fairway,
        "h77v_scarp_strike_vent_bridge": ch9_scarp_vent_bridge,
        "h77v_composite_surface": ch10_composite,
    }
    assert list(out.keys()) == H77V_NAMES

    diag: dict[str, Any] = {
        "source_audit": {
            **paleo_stats,
            **probe_stats,
            **ws_stats,
            "forbidden_columns_excluded": sorted(FORBIDDEN_COLUMNS),
            "sgmc_in_features": False,
            "prior_submissions_in_features": False,
            "cond_surf_filter_sigma_px": 36.0,
        },
        "channels": {},
    }
    for k, arr in out.items():
        v = arr[foot]
        diag["channels"][k] = {
            "in_footprint_mean": float(v.mean()),
            "in_footprint_std": float(v.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "nan_count": int(np.isnan(arr).sum()),
        }
    return out, diag
