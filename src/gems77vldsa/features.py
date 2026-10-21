"""H77V 132-Channel Static Physical + 8-Channel Fold-Visible SLOCO Feature Builder.

Directly implements the three recommendations of docs/review-2026-10-21-h77g.md:
1. IR-H77G-01 Spectral Mesh De-Aliasing (sigma = 36 px = 3.6 km) of rectilinear
   3.5 km / 7.0 km MT (cond_surf), geodetic (geod_2ndinv, geod_shearrate,
   geod_dilaterate), and seismic (deq_n100a15, ieq_n100a15) inversion grids.
2. IR-H77G-02 Corrected Across-Valley Tubular Hessian Demagnetization Trough
   persistence across 150 m, 300 m, and 600 m scales + antiparallel gravity-vs-
   basement-depth step concordance + hyperbolic gravity relay-ramp saddle index.
3. IR-H77G-03 Multiplicative coupling of road- and mining-claim-deconfounded
   1 m LiDAR / 10 m 3DEP scarps and fluvial knickpoints with hydrothermal
   demagnetization troughs, gravity-basement steps, de-aliased MT clay-cap
   margins, and fold-visible SLOCO crack-tip / step-over relay corridors.

Strict Invariants:
- ZERO SGMC channels (derived_sgmc_faults_100m_u8.tif is never opened).
- ZERO catalogue leak columns (dist_known_fault_px / known_fault / label never accessed).
- ZERO prior submission rasters read during feature construction or model training.
"""
from __future__ import annotations

import csv
import gc
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from scipy.stats import rankdata

from gems57 import network as gems57_network
from gems77e.features import (
    SLOCO_MECH_NAMES as BASE_SLOCO_NAMES,
    _fast_nn_fill,
    _rank01_fp_2d,
    build_h77e_static_stack,
)

H77V_DELTA_NAMES: list[str] = [
    "h77v_mt_dealiased_3p6km",
    "h77v_mt_claycap_margin_3p6km",
    "h77v_transtensional_regime_3p6km",
    "h77v_demag_trough_s1p5",
    "h77v_demag_persistence_3scale",
    "h77v_grav_base_step_concord",
    "h77v_grav_relay_saddle",
    "h77v_scarp_x_demag_conduit",
    "h77v_knick_x_grav_base_step",
    "h77v_vent_x_dealiased_mt_margin",
    "h77v_scarp_x_transtensional_regime",
    "h77v_blm_mining_deconf_dist_log",
]

H77V_SLOCO_MECH_NAMES: list[str] = list(BASE_SLOCO_NAMES) + [
    "sloco_hydro_demag_tip_bridge",
]


def _tubular_demag_trough(rtp: np.ndarray, sigma: float) -> np.ndarray:
    """Corrected across-valley principal Hessian eigenvalue of -RTP (IR-H77G-02 fix).

    For a linear magnetic trough (local minimum across strike, near-zero curvature
    along strike), the Hessian H of +RTP has a large POSITIVE across-trough
    eigenvalue lambda_max > 0 and a small along-trough eigenvalue |lambda_min| << lambda_max.
    """
    sm = gaussian_filter(rtp, sigma=sigma, mode="nearest")
    hyy = gaussian_filter(sm, sigma=1.0, order=(2, 0), mode="nearest") * (sigma ** 2)
    hxx = gaussian_filter(sm, sigma=1.0, order=(0, 2), mode="nearest") * (sigma ** 2)
    hxy = gaussian_filter(sm, sigma=1.0, order=(1, 1), mode="nearest") * (sigma ** 2)
    tr = 0.5 * (hxx + hyy)
    disc = np.hypot(0.5 * (hxx - hyy), hxy)
    lam_across = np.maximum(tr + disc, 0.0)
    lam_along = np.abs(tr - disc)
    tubularity = 1.0 - np.minimum(lam_along, lam_across) / (lam_across + 1e-6)
    return (lam_across * tubularity).astype(np.float32)


def _hyperbolic_saddle(field: np.ndarray, sigma: float) -> np.ndarray:
    """Opposite-sign principal Hessian eigenvalue product sqrt(max(-lam1 * lam2, 0))."""
    sm = gaussian_filter(field, sigma=sigma, mode="nearest")
    hyy = gaussian_filter(sm, sigma=1.0, order=(2, 0), mode="nearest") * (sigma ** 2)
    hxx = gaussian_filter(sm, sigma=1.0, order=(0, 2), mode="nearest") * (sigma ** 2)
    hxy = gaussian_filter(sm, sigma=1.0, order=(1, 1), mode="nearest") * (sigma ** 2)
    det_h = hxx * hyy - hxy * hxy
    return np.sqrt(np.maximum(-det_h, 0.0)).astype(np.float32)


def _rasterize_2m_probes_and_sinter(
    ext_dir: Path,
    shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    """Rasterize INGENIOUS 2 m temperature probes (F2mDAB) via stdlib binary DBF parser."""
    from gems77_step.features import _parse_2m_probes_dbf

    imp_dab, imp_cnt = _parse_2m_probes_dbf(
        ext_dir / "2m_temperature_probe_INGENIOUS_regional_data.zip", transform, shape
    )
    probe_2d = np.log1p(np.maximum(imp_dab, 0.0) + 0.5 * imp_cnt).astype(np.float32)
    return gaussian_filter(probe_2d, sigma=8.0, mode="constant").astype(np.float32)


def build_h77v_static_stack(
    data_dir: Path,
    cache_dir: Path,
    fp: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build or load the 132-channel zero-SGMC H77V static stack as (n_fp, 132) float16."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext_dir = data_dir / "data" / "external"
    if not ext_dir.exists():
        ext_dir = data_dir / "external"
    with rasterio.open(data_dir / "training_features.tif") as ds_tr:
        transform = ds_tr.transform
    X120, names120, has_lidar_fp, crest_e_fp, geod_rank_2d, prov120 = build_h77e_static_stack(
        cache_dir, fp, transform, data_dir, ext_dir
    )

    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)
    n_delta = len(H77V_DELTA_NAMES)

    v_mmap_path = cache_dir / "X_h77v_delta12_fp.f32.mmap"
    v_meta_path = cache_dir / "X_h77v_delta12_meta.json"
    v_aux_path = cache_dir / "X_h77v_delta12_aux.npz"

    if v_mmap_path.exists() and v_meta_path.exists() and v_aux_path.exists():
        meta = json.loads(v_meta_path.read_text())
        if meta.get("n_fp") == n_fp and meta.get("names") == H77V_DELTA_NAMES:
            Xv = np.memmap(v_mmap_path, dtype=np.float32, mode="r", shape=(n_delta, n_fp))
            aux = np.load(v_aux_path)
            X132 = _append_float16(X120, Xv)
            return (
                X132,
                list(names120) + list(H77V_DELTA_NAMES),
                has_lidar_fp,
                aux["crest_v_fp"].astype(np.float32),
                aux["hydro_demag_conduit_2d"].astype(np.float32),
                {**prov120, **meta.get("provenance", {})},
            )

    shape = fp.shape
    ext_dir = data_dir / "data" / "external"
    if not ext_dir.exists():
        ext_dir = data_dir / "external"

    Xv = np.memmap(v_mmap_path, dtype=np.float32, mode="w+", shape=(n_delta, n_fp))
    v_ptr = 0

    def _put(name: str, arr2d_or_fp: np.ndarray, is_fp_1d: bool = False) -> None:
        nonlocal v_ptr
        assert H77V_DELTA_NAMES[v_ptr] == name, f"Channel mismatch: {H77V_DELTA_NAMES[v_ptr]} != {name}"
        vec = arr2d_or_fp.astype(np.float32) if is_fp_1d else arr2d_or_fp.ravel()[fp_idx].astype(np.float32)
        bad = ~np.isfinite(vec)
        if bad.any():
            med = float(np.nanmedian(vec[~bad])) if (~bad).any() else 0.0
            vec[bad] = med
        Xv[v_ptr, :] = vec
        v_ptr += 1

    with rasterio.open(data_dir / "training_features.tif") as ds:
        transform = ds.transform

        def _tf(b: int) -> np.ndarray:
            raw = ds.read(b).astype(np.float32)
            return _fast_nn_fill(raw, fp & np.isfinite(raw) & (raw > -1e30))

        rtp = _tf(2)
        geod_sh = _tf(7)
        geod_di = _tf(8)
        grav = _tf(13)
        z_base = _tf(15)
        ieq = _tf(16)
        cond = _tf(17)

    # 1. IR-H77G-01: Spectral Mesh De-Aliasing at sigma = 36 px (3.6 km > 35.02 px mesh period)
    log_cond = np.log1p(np.maximum(cond, 0.0)).astype(np.float32)
    del cond
    mt_dealiased = gaussian_filter(log_cond, sigma=36.0, mode="nearest").astype(np.float32)
    del log_cond
    mty, mtx = np.gradient(mt_dealiased)
    mt_margin = np.hypot(mty, mtx).astype(np.float32)
    del mty, mtx

    sh_36 = _rank01_fp_2d(gaussian_filter(geod_sh, sigma=36.0, mode="nearest"), fp)
    di_36 = _rank01_fp_2d(gaussian_filter(np.abs(geod_di), sigma=36.0, mode="nearest"), fp)
    ieq_36 = _rank01_fp_2d(gaussian_filter(ieq, sigma=36.0, mode="nearest"), fp)
    del geod_sh, geod_di, ieq
    transtens_36 = np.cbrt(np.maximum(sh_36 * di_36 * ieq_36, 0.0)).astype(np.float32)
    del sh_36, di_36, ieq_36

    _put("h77v_mt_dealiased_3p6km", mt_dealiased)
    _put("h77v_mt_claycap_margin_3p6km", mt_margin)
    _put("h77v_transtensional_regime_3p6km", transtens_36)
    del mt_dealiased

    # 2. IR-H77G-02: Corrected Across-Valley Tubular Hessian Demagnetization Troughs (150m, 300m, 600m)
    tau_1p5 = _tubular_demag_trough(rtp, sigma=1.5)
    tau_3p0 = _tubular_demag_trough(rtp, sigma=3.0)
    tau_6p0 = _tubular_demag_trough(rtp, sigma=6.0)
    del rtp
    r_t15 = _rank01_fp_2d(tau_1p5, fp)
    r_t30 = _rank01_fp_2d(tau_3p0, fp)
    r_t60 = _rank01_fp_2d(tau_6p0, fp)
    del tau_3p0, tau_6p0
    demag_persist = np.cbrt(np.maximum(r_t15 * r_t30 * r_t60, 0.0)).astype(np.float32)
    del r_t15, r_t30, r_t60

    _put("h77v_demag_trough_s1p5", tau_1p5)
    _put("h77v_demag_persistence_3scale", demag_persist)
    del tau_1p5

    # 3. Antiparallel Gravity-Basement Step Concordance & Hyperbolic Relay Saddle
    gy, gx = np.gradient(gaussian_filter(grav, sigma=2.0, mode="nearest"))
    by, bx = np.gradient(gaussian_filter(z_base, sigma=2.0, mode="nearest"))
    del z_base
    anti_dot = np.maximum(-(gy * by + gx * bx), 0.0)
    grav_base_concord = np.sqrt(anti_dot).astype(np.float32)
    del gy, gx, by, bx, anti_dot

    grav_saddle = _hyperbolic_saddle(grav, sigma=3.0)
    del grav

    _put("h77v_grav_base_step_concord", grav_base_concord)
    _put("h77v_grav_relay_saddle", grav_saddle)
    del grav_saddle

    # 4. IR-H77G-03: Coupling with BLM Mining De-confounder, 3DEP Knickpoints, Scarps & Hydrothermal Upflow
    blm_path = ext_dir / "blm_closed_claim_distance_m.tif"
    if blm_path.exists():
        with rasterio.open(blm_path) as ds:
            blm_raw = ds.read(1).astype(np.float32)
        blm_m = np.maximum(_fast_nn_fill(blm_raw, fp & np.isfinite(blm_raw) & (blm_raw >= 0.0)), 0.0)
        del blm_raw
    else:
        blm_m = np.full(shape, 2500.0, dtype=np.float32)
    blm_log = np.log1p(blm_m).astype(np.float32)
    w_blm_fp = (1.0 - 0.35 * np.exp(-blm_m.ravel()[fp_idx] / 250.0)).astype(np.float32)
    del blm_m

    # Knickpoint density from dem10_chi_100m_u8.tif
    with rasterio.open(ext_dir / "dem10_chi_100m_u8.tif") as ds:
        knick_n = ds.read(2).astype(np.float32)
        valid_n = np.maximum(ds.read(4).astype(np.float32), 1.0)
    knick_rate = gaussian_filter((knick_n / valid_n).astype(np.float32), sigma=1.5, mode="nearest")
    del knick_n, valid_n

    # Geothermometry + 2m probe + paleo-sinter upflow plume
    diseq_2d = np.zeros(shape, dtype=np.float32)
    with (ext_dir / "gdr_wellspring_in_footprint.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            try:
                rr = int(float(rec["row"]))
                cc = int(float(rec["col"]))
            except (ValueError, KeyError):
                continue
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                continue
            chems = []
            for kc in ("geothermquartz_c", "geothermchalc_c", "geothermcat_c"):
                s = rec.get(kc, "").strip()
                if s and s.lower() != "nan":
                    try:
                        v = float(s)
                        if np.isfinite(v) and 0.0 < v < 400.0:
                            chems.append(v)
                    except ValueError:
                        pass
            ts = 15.0
            st = rec.get("temp_c", "").strip()
            if st and st.lower() != "nan":
                try:
                    tv = float(st)
                    if np.isfinite(tv) and 0.0 < tv < 200.0:
                        ts = tv
                except ValueError:
                    pass
            tc_max = max(chems) if chems else 0.0
            dq = max(0.0, tc_max - max(ts, 20.0)) + 0.35 * max(0.0, max(tc_max, ts) - 70.0)
            if dq > 0.0:
                diseq_2d[rr, cc] = max(diseq_2d[rr, cc], float(np.log1p(dq)))

    probe_sinter_2d = _rasterize_2m_probes_and_sinter(ext_dir, shape, transform)
    vent_plume_2d = (
        gaussian_filter(diseq_2d, sigma=7.0, mode="constant") + 0.75 * probe_sinter_2d
    ).astype(np.float32)
    del diseq_2d, probe_sinter_2d

    r_gb_2d = _rank01_fp_2d(grav_base_concord, fp)
    r_mt_margin_2d = _rank01_fp_2d(mt_margin, fp)
    r_knick_2d = _rank01_fp_2d(knick_rate, fp)
    del grav_base_concord, mt_margin, knick_rate

    scarp_deconf_fp = (crest_e_fp * w_blm_fp).astype(np.float32)
    demag_fp = demag_persist.ravel()[fp_idx]
    transtens_fp = transtens_36.ravel()[fp_idx]

    scarp_x_demag_fp = (scarp_deconf_fp * (0.35 + 0.65 * demag_fp)).astype(np.float32)
    knick_x_gb_2d = (r_knick_2d * r_gb_2d).astype(np.float32)
    vent_x_mt_2d = ((0.25 + _rank01_fp_2d(vent_plume_2d, fp)) * r_mt_margin_2d).astype(np.float32)
    scarp_x_transtens_fp = (scarp_deconf_fp * (0.35 + 0.65 * transtens_fp)).astype(np.float32)

    _put("h77v_scarp_x_demag_conduit", scarp_x_demag_fp, is_fp_1d=True)
    _put("h77v_knick_x_grav_base_step", knick_x_gb_2d)
    _put("h77v_vent_x_dealiased_mt_margin", vent_x_mt_2d)
    _put("h77v_scarp_x_transtensional_regime", scarp_x_transtens_fp, is_fp_1d=True)
    _put("h77v_blm_mining_deconf_dist_log", blm_log)

    assert v_ptr == n_delta, f"Expected {n_delta}, got {v_ptr}"
    Xv.flush()

    hydro_demag_conduit_2d = (
        0.35 * demag_persist
        + 0.25 * r_gb_2d
        + 0.20 * r_mt_margin_2d
        + 0.20 * transtens_36
    ).astype(np.float32)

    crest_v_fp = np.clip(
        0.72 * scarp_deconf_fp
        + 0.16 * demag_fp * scarp_deconf_fp
        + 0.12 * knick_x_gb_2d.ravel()[fp_idx],
        0.0,
        1.0,
    ).astype(np.float32)

    np.savez_compressed(
        v_aux_path,
        crest_v_fp=crest_v_fp,
        hydro_demag_conduit_2d=hydro_demag_conduit_2d,
    )
    v_meta = {
        "n_fp": n_fp,
        "names": H77V_DELTA_NAMES,
        "provenance": {
            "spectral_dealiasing_sigma_px": 36.0,
            "blm_closed_claim_distance_m": "BLM MLRS closed mining claim distance de-confounder",
            "2m_temperature_probe_INGENIOUS": "Great Basin Center for Geothermal Energy 2 m probe database",
            "paleo_geothermal_regional": "USGS / GBCGE paleo-geothermal sinter, travertine, and tufa deposits",
            "sgmc_channels_used": 0,
            "forbidden_columns_accessed": [],
            "prior_rasters_read": 0,
        },
    }
    v_meta_path.write_text(json.dumps(v_meta, indent=2) + "\n")
    gc.collect()

    Xv_r = np.memmap(v_mmap_path, dtype=np.float32, mode="r", shape=(n_delta, n_fp))
    X132 = _append_float16(X120, Xv_r)
    return (
        X132,
        list(names120) + list(H77V_DELTA_NAMES),
        has_lidar_fp,
        crest_v_fp,
        hydro_demag_conduit_2d,
        {**prov120, **v_meta["provenance"]},
    )


def _append_float16(X_base: np.ndarray, X_delta: np.memmap) -> np.ndarray:
    n_fp, n_a = X_base.shape
    n_b = X_delta.shape[0]
    out = np.empty((n_fp, n_a + n_b), dtype=np.float16)
    out[:, :n_a] = X_base
    for j in range(n_b):
        v = np.asarray(X_delta[j], dtype=np.float32)
        out[:, n_a + j] = np.clip(v, -65000.0, 65000.0).astype(np.float16)
    return out


def _single_mask_mechanics_v(
    mask_2d: np.ndarray,
    fp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact EDT, second-component EDT approx, short (1.5km) & long (3.8km) Pollard tip cones."""
    h, w = fp.shape
    m = np.asarray(mask_2d, dtype=bool) & fp
    d1 = distance_transform_edt(~m).astype(np.float32)

    lbl, n_comp = gems57_network.ndi.label(m, structure=np.ones((3, 3), dtype=np.int8))
    sizes = np.bincount(lbl.ravel())
    major = m & (sizes[lbl] >= 4)
    nb = convolve(major.astype(np.int8), np.ones((3, 3), dtype=np.int8), mode="constant") - major.astype(np.int8)
    tips = major & (nb <= 1)

    sm = gaussian_filter(major.astype(np.float32), sigma=2.8, mode="constant")
    tyy = gaussian_filter(sm, sigma=1.8, order=(2, 0), mode="constant")
    txx = gaussian_filter(sm, sigma=1.8, order=(0, 2), mode="constant")
    txy = gaussian_filter(sm, sigma=1.8, order=(1, 1), mode="constant")
    ang = 0.5 * np.arctan2(2.0 * txy, txx - tyy)
    ux = -np.sin(ang)
    uy = np.cos(ang)
    del sm, tyy, txx, txy, ang

    ty, tx = np.nonzero(tips)
    lobe_short = np.zeros((h, w), dtype=np.float32)
    lobe_long = np.zeros((h, w), dtype=np.float32)
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
        imp_s = np.zeros((h, w), dtype=np.float32)
        imp_l = np.zeros((h, w), dtype=np.float32)
        for s in range(2, 40, 2):
            rr = np.round(ty + s * tuy).astype(np.int32)
            cc = np.round(tx + s * tux).astype(np.int32)
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            if s <= 22:
                np.maximum.at(imp_s, (rr[ok], cc[ok]), float(np.exp(-s / 11.0)))
            np.maximum.at(imp_l, (rr[ok], cc[ok]), float(np.exp(-s / 24.0)))
        lobe_short = gaussian_filter(imp_s, sigma=2.0, mode="constant").astype(np.float32)
        lobe_long = gaussian_filter(imp_l, sigma=3.5, mode="constant").astype(np.float32)
    del ux, uy

    d2 = np.full((h, w), 120.0, dtype=np.float32)
    if n_comp >= 2:
        cids = np.arange(n_comp + 1, dtype=np.uint64)
        for b, mult in enumerate((2654435761, 2246822519, 3266489917, 668265263, 374761393, 1597334677)):
            side = (((cids * np.uint64(mult)) >> np.uint64(13 + b)) & np.uint64(1)).astype(np.int8)
            side[0] = -1
            ma = side[lbl] == 0
            mb = side[lbl] == 1
            if ma.any() and mb.any():
                da = distance_transform_edt(~ma).astype(np.float32)
                db = distance_transform_edt(~mb).astype(np.float32)
                np.minimum(d2, np.maximum(da, db), out=d2)
                del da, db

    return d1, d2, lobe_short, lobe_long


def compute_h77v_sloco_mechanics_fp(
    visible_2d: np.ndarray,
    fp: np.ndarray,
    hydro_demag_conduit_2d: np.ndarray,
    n_buckets: int = 8,
    max_seg_px: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (X_mech_train, X_mech_infer), each (8, n_fp) float32, strictly from visible_2d."""
    vis = np.asarray(visible_2d, dtype=bool) & fp
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)
    conduit_fp = hydro_demag_conduit_2d.ravel()[fp_idx].astype(np.float32)

    def _pack8(d1: np.ndarray, d2: np.ndarray, ls: np.ndarray, ll: np.ndarray) -> np.ndarray:
        out = np.empty((8, n_fp), dtype=np.float32)
        d1_fp = d1.ravel()[fp_idx]
        d2_fp = d2.ravel()[fp_idx]
        ls_fp = ls.ravel()[fp_idx]
        ll_fp = ll.ravel()[fp_idx]
        relay = np.exp(-((np.maximum(d1_fp - 2.0, 0.0)) / 11.0) ** 2) * np.exp(
            -((np.maximum(d2_fp - 2.0, 0.0)) / 14.0) ** 2
        )
        out[0] = np.minimum(d1_fp, 80.0)
        out[1] = np.log1p(np.minimum(d1_fp, 150.0))
        out[2] = np.exp(-((np.maximum(d1_fp - 2.2, 0.0)) / 4.5) ** 2)
        out[3] = ls_fp
        out[4] = ll_fp
        out[5] = relay
        out[6] = conduit_fp * (1.0 - np.exp(-d1_fp / 14.0))
        out[7] = conduit_fp * (0.45 * np.maximum(ls_fp, ll_fp) + 0.55 * relay)
        return out

    d1_f, d2_f, ls_f, ll_f = _single_mask_mechanics_v(vis, fp)
    X_infer = _pack8(d1_f, d2_f, ls_f, ll_f)
    del d1_f, d2_f, ls_f, ll_f

    seg, n_seg = gems57_network.segments(vis, max_len_px=max_seg_px)[:2]
    seg_ids = np.arange(n_seg + 1, dtype=np.uint64)
    buckets = ((seg_ids * np.uint64(2654435761)) % np.uint64(n_buckets)).astype(np.int16)
    buckets[0] = -1
    b_map = buckets[seg]

    yy, xx = np.ogrid[-2:3, -2:3]
    disk2 = (yy * yy + xx * xx) <= 4

    X_train = X_infer.copy()
    for b in range(n_buckets):
        in_b = b_map == b
        if not in_b.any():
            continue
        excl = gems57_network.ndi.binary_dilation(in_b, structure=disk2)
        vis_minus_b = vis & (~excl)
        d1_b, d2_b, ls_b, ll_b = _single_mask_mechanics_v(vis_minus_b, fp)
        sub8 = _pack8(d1_b, d2_b, ls_b, ll_b)
        b_fp = in_b.ravel()[fp_idx]
        X_train[:, b_fp] = sub8[:, b_fp]
        del excl, vis_minus_b, d1_b, d2_b, ls_b, ll_b, sub8

    gc.collect()
    return X_train, X_infer


def run_single_feature_leakage_canaries(
    X_static: np.ndarray,
    static_names: list[str],
    X_mech_infer: np.ndarray,
    mech_names: list[str],
    pos_fp_idx: np.ndarray,
    neg_fp_idx: np.ndarray,
    ceiling: float = 0.90,
) -> dict[str, Any]:
    """Test every single feature (132 static + 8 SLOCO = 140) alone on holdout positives vs negatives."""
    rng = np.random.default_rng(77001)
    n_pos = min(int(pos_fp_idx.size), 25_000)
    n_neg = min(int(neg_fp_idx.size), 75_000)
    p_sub = rng.choice(pos_fp_idx, size=n_pos, replace=False) if pos_fp_idx.size > n_pos else pos_fp_idx
    n_sub = rng.choice(neg_fp_idx, size=n_neg, replace=False) if neg_fp_idx.size > n_neg else neg_fp_idx
    idx = np.concatenate([p_sub, n_sub])
    n1 = float(p_sub.size)
    n0 = float(n_sub.size)

    rows: list[dict[str, Any]] = []

    def _eval_vec(vec: np.ndarray, name: str, group: str) -> None:
        v = np.asarray(vec, dtype=np.float32)
        bad = ~np.isfinite(v)
        if bad.any():
            v = v.copy()
            v[bad] = 0.0
        rk = rankdata(v, method="average")
        auc = float((rk[: int(n1)].sum() - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))
        sym_auc = float(max(auc, 1.0 - auc))
        rows.append(
            {
                "feature": name,
                "group": group,
                "raw_auc": round(auc, 6),
                "symmetric_auc": round(sym_auc, 6),
                "pass_le_0p90": sym_auc <= ceiling,
            }
        )

    for j, nm in enumerate(static_names):
        grp = (
            "h77v_new_physics"
            if nm in H77V_DELTA_NAMES
            else ("h77e_delta_physics" if j >= 96 else "h77_base_physics")
        )
        _eval_vec(X_static[idx, j], nm, grp)
    for j, nm in enumerate(mech_names):
        _eval_vec(X_mech_infer[j, idx], nm, "sloco_fold_visible_mechanics")

    violators = [r for r in rows if not r["pass_le_0p90"]]
    static_rows = [r for r in rows if r["group"] != "sloco_fold_visible_mechanics"]
    h77v_rows = [r for r in rows if r["group"] == "h77v_new_physics"]
    worst_all = max(rows, key=lambda r: r["symmetric_auc"])
    worst_static = max(static_rows, key=lambda r: r["symmetric_auc"])
    worst_h77v = max(h77v_rows, key=lambda r: r["symmetric_auc"])
    return {
        "ceiling": ceiling,
        "n_features_checked": len(rows),
        "n_static_checked": len(static_rows),
        "n_h77v_new_checked": len(h77v_rows),
        "n_mechanics_checked": len(mech_names),
        "passed": len(violators) == 0,
        "violators": violators,
        "max_overall": worst_all,
        "max_static_physics": worst_static,
        "max_h77v_new_physics": worst_h77v,
        "rows": rows,
    }
