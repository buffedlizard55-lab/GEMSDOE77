"""Mesh-deconfounded (sigma>=35px) hydrothermal-structural coupled feature stack & SLOCO mechanics (`H77H`).

Directly implements the preregistered `H77H-1` architecture (`registry/preregistration_h77h.json`)
and resolves `IR-H77G-01`, `IR-H77G-02`, and `IR-H77G-03` from `docs/review-2026-10-21-h77g.md`:
  1. Purges short-wavelength (<35 px) MT conductance and seismicity derivative channels
     (`cond_grad` at sigma=1.5 px, `hydro_chem_claycap_coupled` at sigma=2.5 px, and
     `seism_updip_proj_35px` / `seism_updip_proj_50px` at sigma=3.0 px) that extract
     rectilinear 3.5 km / 7.0 km inversion-mesh boundaries (`IR-H77G-01`).
  2. Filters `cond_surf` (Band 17), geodetic strain rates (Bands 4, 7, 8), and earthquake
     densities (Bands 10, 16) strictly at sigma >= 35 px (3.5 km) as regional crustal-regime
     modulators.
  3. Couples the corrected across-valley Hessian RTP demagnetization trough (`hessian_trough`,
     `IR-H77G-02`), oriented gravity-basement step (`h77h_grav_base_step_align`), and basement
     relay saddle (`hessian_saddle`) directly with TIGER-deconfounded 1m/10m 3DEP scarp crests,
     10m fluvial knickpoint rates, and fold-pure SLOCO fracture-mechanics continuations (`IR-H77G-03`).
"""
from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gems57 import network as gems57_network  # noqa: E402
from gems77_step.features import build_h77_feature_stack  # noqa: E402
from gems77e.features import (  # noqa: E402
    FORBIDDEN_POINT_COLUMNS,
    H77E_DELTA_NAMES,
    _fast_nn_fill,
    _sample_bilinear_fp,
    _single_mask_mechanics,
    build_h77e_static_stack,
)
from gems77g.fields import hessian_saddle, hessian_trough  # noqa: E402

# Short-wavelength (<35 px) MT / seismicity mesh-derivative features excluded per IR-H77G-01
EXCLUDED_MESH_CONTAMINATED_FEATURES = frozenset({
    "cond_grad",
    "hydro_chem_claycap_coupled",
    "seism_updip_proj_35px",
    "seism_updip_proj_50px",
})

H77H_DELTA_NAMES = [
    "h77h_rtp_demag_trough_s1p5",
    "h77h_rtp_demag_trough_s3p0",
    "h77h_rtp_demag_persist",
    "h77h_grav_base_step_align",
    "h77h_grav_mag_decoupling",
    "h77h_basement_relay_saddle",
    "h77h_reg35_mt_conductance",
    "h77h_reg35_transtension_idx",
    "h77h_reg35_swarm_seismicity",
    "h77h_scarp_x_demag_coupled",
    "h77h_scarp_x_gravstep_coupled",
    "h77h_updip35_reg_seism_hydro",
]

H77H_SLOCO_MECH_NAMES = [
    "sloco_d1_trunc80",
    "sloco_d1_log1p",
    "sloco_collar_ring_450m",
    "sloco_pollard_tip_1500m",
    "sloco_pollard_tip_3800m",
    "sloco_relay_corridor_2comp",
    "sloco_orphan_reg35_strain",
    "sloco_tip_x_demag_grav",
    "sloco_relay_x_saddle_scarp",
]


def _rank01_fp_2d(arr: np.ndarray, fp: np.ndarray) -> np.ndarray:
    out = np.zeros(fp.shape, dtype=np.float32)
    v = np.nan_to_num(arr[fp].astype(np.float32), nan=0.0)
    out[fp] = (rankdata(v, method="average") / float(v.size)).astype(np.float32)
    return out


def _ensure_base_memmaps(
    cache_dir: Path,
    fp: np.ndarray,
    transform: Any,
    data_dir: Path,
    ext_dir: Path,
) -> tuple[np.memmap, list[str], np.memmap, list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Ensure `h77_features_fp.f32` (96 cols) and `h77e_delta_fp.f32` (24 cols) exist on disk without holding X120 in RAM."""
    e_mmap = cache_dir / "h77e_delta_fp.f32"
    e_meta = cache_dir / "h77e_delta.meta.json"
    e_aux = cache_dir / "h77e_delta.aux.npz"
    if not (e_mmap.is_file() and e_meta.is_file() and e_aux.is_file()):
        res = build_h77e_static_stack(cache_dir, fp, transform, data_dir, ext_dir)
        del res
        gc.collect()

    X96, names96, has_lidar_fp, _, _, base_prov = build_h77_feature_stack(
        cache_dir / "h77_features_fp.f32", fp, transform, data_dir, ext_dir
    )
    n_fp = int(fp.sum())
    X24 = np.memmap(e_mmap, dtype=np.float32, mode="r", shape=(len(H77E_DELTA_NAMES), n_fp))
    aux_e = np.load(e_aux)
    crest_deconf_fp = aux_e["crest_deconf_fp"].astype(np.float32)
    return X96, list(names96), X24, list(H77E_DELTA_NAMES), has_lidar_fp, crest_deconf_fp, base_prov


def build_h77h_static_stack(
    cache_dir: Path,
    fp: np.ndarray,
    transform: Any,
    data_dir: Path,
    ext_dir: Path,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Build or reload the 128-channel mesh-safe H77H static feature matrix (116 clean base + 12 H77H) in <1.4 GB RAM."""
    X96, names96, X24, names24, has_lidar_fp, crest_deconf_fp, base_prov = _ensure_base_memmaps(
        cache_dir, fp, transform, data_dir, ext_dir
    )
    keep96 = [i for i, nm in enumerate(names96) if nm not in EXCLUDED_MESH_CONTAMINATED_FEATURES]
    keep24 = [i for i, nm in enumerate(names24) if nm not in EXCLUDED_MESH_CONTAMINATED_FEATURES]
    clean_base_names = [names96[i] for i in keep96] + [names24[i] for i in keep24]
    assert len(clean_base_names) == 116, f"Expected 116 clean base features, got {len(clean_base_names)}"

    shape = fp.shape
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)
    n_delta = len(H77H_DELTA_NAMES)

    delta_mmap_path = cache_dir / "h77h_delta_fp.f32"
    delta_meta_path = cache_dir / "h77h_delta.meta.json"
    delta_aux_path = cache_dir / "h77h_delta.aux.npz"

    if delta_mmap_path.is_file() and delta_meta_path.is_file() and delta_aux_path.is_file():
        meta = json.loads(delta_meta_path.read_text())
        if meta.get("names") == H77H_DELTA_NAMES and meta.get("n_fp") == n_fp:
            Xd = np.memmap(delta_mmap_path, dtype=np.float32, mode="r", shape=(n_delta, n_fp))
            aux = np.load(delta_aux_path)
            X16 = _pack_from_memmaps(X96, keep96, X24, keep24, Xd)
            coupled_fields = {
                "reg35_regime_2d": aux["reg35_regime_2d"],
                "tip_coupler_2d": aux["tip_coupler_2d"],
                "relay_coupler_2d": aux["relay_coupler_2d"],
            }
            return (
                X16,
                clean_base_names + list(H77H_DELTA_NAMES),
                has_lidar_fp,
                crest_deconf_fp,
                coupled_fields,
                {**base_prov, **meta.get("provenance", {})},
            )

    Xd = np.memmap(delta_mmap_path, dtype=np.float32, mode="w+", shape=(n_delta, n_fp))
    d_ptr = 0

    def _put(name: str, arr: np.ndarray, is_fp_1d: bool = False) -> None:
        nonlocal d_ptr
        assert name == H77H_DELTA_NAMES[d_ptr], f"Order mismatch: {name} != {H77H_DELTA_NAMES[d_ptr]}"
        assert name not in FORBIDDEN_POINT_COLUMNS
        vec = arr if is_fp_1d else arr.ravel()[fp_idx]
        Xd[d_ptr, :] = vec.astype(np.float32, copy=False)
        d_ptr += 1

    with rasterio.open(data_dir / "training_features.tif") as ds:
        def _tf(b: int) -> np.ndarray:
            raw = ds.read(b).astype(np.float32)
            return _fast_nn_fill(raw, fp & np.isfinite(raw) & (raw > -1e30))

        # Channels 1-3: Multi-scale across-valley Hessian RTP demagnetization troughs (IR-H77G-02)
        rtp = _tf(2)
        t15 = _rank01_fp_2d(hessian_trough(rtp, sigma=1.5, hp_sigma=12.0), fp)
        t30 = _rank01_fp_2d(hessian_trough(rtp, sigma=3.0, hp_sigma=12.0), fp)
        del rtp
        demag_persist = _rank01_fp_2d(np.sqrt(np.clip(t15 * t30, 0.0, 1.0)), fp)

        # Channel 4: Oriented gravity-basement normal-fault step concordance
        grav = _tf(13)
        depth = _tf(15)
        ggy = gaussian_filter(grav, sigma=2.0, order=(1, 0), mode="nearest")
        ggx = gaussian_filter(grav, sigma=2.0, order=(0, 1), mode="nearest")
        gmag = np.hypot(ggx, ggy).astype(np.float32)
        del grav
        dgy = gaussian_filter(depth, sigma=2.0, order=(1, 0), mode="nearest")
        dgx = gaussian_filter(depth, sigma=2.0, order=(0, 1), mode="nearest")
        dmag = np.hypot(dgx, dgy).astype(np.float32)
        cos_align = np.abs(ggx * dgx + ggy * dgy) / (gmag * dmag + np.float32(1e-9))
        del ggy, ggx, dgy, dgx
        rg = _rank01_fp_2d(gmag, fp)
        rd = _rank01_fp_2d(dmag, fp)
        del gmag, dmag
        grav_base_step = _rank01_fp_2d(
            np.sqrt(np.clip(rg * rd, 0.0, 1.0)) * np.clip(cos_align, 0.0, 1.0), fp
        )
        del cos_align, rd

        # Channel 5: Gravity step over magnetic analytic signal decoupling (demagnetized fault step)
        tmi_hg = _tf(3)
        tmi_vg = _tf(9)
        mag_as = np.hypot(np.maximum(tmi_hg, 0.0), tmi_vg).astype(np.float32)
        del tmi_hg, tmi_vg
        r_as = _rank01_fp_2d(mag_as, fp)
        del mag_as
        grav_mag_decouple = _rank01_fp_2d(np.maximum(0.0, rg - np.float32(0.50) * r_as), fp)
        del r_as, rg

        # Channel 6: Subsurface basement-depth Hessian saddle (buried relay transfer zone)
        base_saddle = _rank01_fp_2d(hessian_saddle(depth, sigma=2.5), fp)
        del depth

        # Channels 7-9: Strictly regional (sigma = 35 px = 3.5 km and sigma = 50 px = 5.0 km)
        # low-pass filtered MT conductance, geodetic transtension, and seismicity (IR-H77G-01)
        cond = _tf(17)
        cond_log = np.log1p(np.maximum(cond, 0.0)).astype(np.float32)
        del cond
        reg35_mt = _rank01_fp_2d(
            0.60 * gaussian_filter(cond_log, sigma=35.0, mode="nearest")
            + 0.40 * gaussian_filter(cond_log, sigma=50.0, mode="nearest"),
            fp,
        )
        del cond_log

        geod_sh = _tf(7)
        geod_di = _tf(8)
        geod2 = _tf(4)
        sh_s35 = _rank01_fp_2d(gaussian_filter(np.maximum(geod_sh, 0.0), sigma=35.0, mode="nearest"), fp)
        di_s35 = _rank01_fp_2d(gaussian_filter(np.maximum(geod_di, 0.0), sigma=35.0, mode="nearest"), fp)
        g2_s35 = _rank01_fp_2d(gaussian_filter(np.maximum(geod2, 0.0), sigma=35.0, mode="nearest"), fp)
        del geod_sh, geod_di, geod2
        reg35_transtension = _rank01_fp_2d(
            0.45 * np.sqrt(sh_s35 * di_s35) + 0.35 * g2_s35 + 0.20 * sh_s35, fp
        )
        del sh_s35, di_s35, g2_s35

        ieq = _tf(16)
        deq = _tf(10)
        ieq_s35 = _rank01_fp_2d(gaussian_filter(np.maximum(ieq, 0.0), sigma=35.0, mode="nearest"), fp)
        deq_s35 = _rank01_fp_2d(gaussian_filter(np.maximum(deq, 0.0), sigma=35.0, mode="nearest"), fp)
        del ieq, deq
        reg35_swarm = _rank01_fp_2d(ieq_s35 * (ieq_s35 / (ieq_s35 + deq_s35 + np.float32(0.08))), fp)
        del ieq_s35, deq_s35

        det_elev = _tf(12)

    reg35_regime_2d = (
        0.40 * reg35_transtension + 0.35 * reg35_swarm + 0.25 * reg35_mt
    ).astype(np.float32)

    scarp_2d = np.zeros(shape, dtype=np.float32)
    scarp_2d.ravel()[fp_idx] = np.asarray(X24[names24.index("seamfree_scarp_deconf")], dtype=np.float32)
    radon_2d = np.zeros(shape, dtype=np.float32)
    radon_2d.ravel()[fp_idx] = np.asarray(X24[names24.index("radon_scarp_deconf_2100m")], dtype=np.float32)
    knick_2d = np.zeros(shape, dtype=np.float32)
    knick_2d.ravel()[fp_idx] = np.asarray(X24[names24.index("chi_knick_rate")], dtype=np.float32)

    r_scarp = _rank01_fp_2d(0.60 * scarp_2d + 0.40 * radon_2d, fp)
    r_knick = _rank01_fp_2d(knick_2d, fp)
    del scarp_2d, radon_2d, knick_2d

    # Channels 10 & 11: Direct Hydrothermal-Structural and Gravity-Structural Coupling (IR-H77G-03)
    scarp_x_demag = (
        r_scarp * demag_persist * (0.50 + 0.50 * reg35_regime_2d)
    ).astype(np.float32)
    scarp_x_gravstep = (
        (0.65 * r_scarp + 0.35 * r_knick) * grav_base_step * (0.50 + 0.50 * reg35_regime_2d)
    ).astype(np.float32)

    # Channel 12: Updip hanging-wall projection (+35 px) of sigma>=35 px regional seismicity & hydro
    z_sm = gaussian_filter(det_elev, sigma=2.0)
    zy, zx = np.gradient(z_sm)
    jxx = gaussian_filter((zx * zx).astype(np.float32), sigma=3.5)
    jyy = gaussian_filter((zy * zy).astype(np.float32), sigma=3.5)
    jxy = gaussian_filter((zx * zy).astype(np.float32), sigma=3.5)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    nx_fp = np.cos(theta.ravel()[fp_idx]).astype(np.float32)
    ny_fp = np.sin(theta.ravel()[fp_idx]).astype(np.float32)
    del z_sm, zy, zx, jxx, jyy, jxy, theta

    base_y = (fp_idx // shape[1]).astype(np.float32)
    base_x = (fp_idx % shape[1]).astype(np.float32)
    zp9 = _sample_bilinear_fp(det_elev, base_y + 9.0 * ny_fp, base_x + 9.0 * nx_fp)
    zm9 = _sample_bilinear_fp(det_elev, base_y - 9.0 * ny_fp, base_x - 9.0 * nx_fp)
    sfw = np.where(zm9 >= zp9, 1.0, -1.0).astype(np.float32)
    del det_elev, zp9, zm9

    diseq_2d = np.zeros(shape, dtype=np.float32)
    with (ext_dir / "gdr_wellspring_in_footprint.csv").open(newline="") as fh:
        for rec in csv.DictReader(fh):
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

    hydro_reg35_field = (
        0.55 * reg35_swarm
        + 0.25 * reg35_mt
        + 0.20 * _rank01_fp_2d(gaussian_filter(diseq_2d, sigma=8.0), fp)
    ).astype(np.float32)
    del diseq_2d

    hw35 = _sample_bilinear_fp(hydro_reg35_field, base_y + 35.0 * sfw * ny_fp, base_x + 35.0 * sfw * nx_fp)
    fw35 = _sample_bilinear_fp(hydro_reg35_field, base_y - 35.0 * sfw * ny_fp, base_x - 35.0 * sfw * nx_fp)
    updip35_reg_seism_hydro = (hw35 - 0.50 * fw35).astype(np.float32)
    del hydro_reg35_field, hw35, fw35, base_y, base_x, ny_fp, nx_fp, sfw

    _put("h77h_rtp_demag_trough_s1p5", t15)
    _put("h77h_rtp_demag_trough_s3p0", t30)
    _put("h77h_rtp_demag_persist", demag_persist)
    _put("h77h_grav_base_step_align", grav_base_step)
    _put("h77h_grav_mag_decoupling", grav_mag_decouple)
    _put("h77h_basement_relay_saddle", base_saddle)
    _put("h77h_reg35_mt_conductance", reg35_mt)
    _put("h77h_reg35_transtension_idx", reg35_transtension)
    _put("h77h_reg35_swarm_seismicity", reg35_swarm)
    _put("h77h_scarp_x_demag_coupled", scarp_x_demag)
    _put("h77h_scarp_x_gravstep_coupled", scarp_x_gravstep)
    _put("h77h_updip35_reg_seism_hydro", updip35_reg_seism_hydro, is_fp_1d=True)
    assert d_ptr == n_delta, f"Expected {n_delta}, got {d_ptr}"

    tip_coupler_2d = (
        0.35 * demag_persist + 0.35 * grav_base_step + 0.30 * r_scarp
    ).astype(np.float32)
    relay_coupler_2d = (
        0.40 * base_saddle + 0.35 * grav_base_step + 0.25 * np.maximum(r_scarp, r_knick)
    ).astype(np.float32)
    del t15, t30, demag_persist, grav_base_step, grav_mag_decouple, base_saddle
    del reg35_mt, reg35_transtension, reg35_swarm, scarp_x_demag, scarp_x_gravstep, r_scarp, r_knick

    Xd.flush()
    np.savez_compressed(
        delta_aux_path,
        reg35_regime_2d=reg35_regime_2d,
        tip_coupler_2d=tip_coupler_2d,
        relay_coupler_2d=relay_coupler_2d,
    )
    delta_meta = {
        "n_fp": n_fp,
        "names": H77H_DELTA_NAMES,
        "provenance": {
            "excluded_mesh_contaminated_features": sorted(EXCLUDED_MESH_CONTAMINATED_FEATURES),
            "min_regional_smoothing_sigma_px": 35.0,
            "min_regional_smoothing_km": 3.5,
            "hessian_trough_operator": "src/gems77g/fields.py:hessian_trough (across-valley lambda_1 > 0 weighted by tubularity)",
            "hydrothermal_structural_coupling": "Direct product of TIGER-deconfounded 1m/10m 3DEP scarp & 10m chi knickpoint rates with RTP demagnetization trough, gravity-basement step, and sigma>=35px crustal regime",
            "sgmc_channels_used": 0,
            "forbidden_columns_accessed": [],
            "prior_rasters_read": 0,
        },
    }
    delta_meta_path.write_text(json.dumps(delta_meta, indent=2) + "\n")
    gc.collect()

    Xd_r = np.memmap(delta_mmap_path, dtype=np.float32, mode="r", shape=(n_delta, n_fp))
    X16 = _pack_from_memmaps(X96, keep96, X24, keep24, Xd_r)
    coupled_fields = {
        "reg35_regime_2d": reg35_regime_2d,
        "tip_coupler_2d": tip_coupler_2d,
        "relay_coupler_2d": relay_coupler_2d,
    }
    return (
        X16,
        clean_base_names + list(H77H_DELTA_NAMES),
        has_lidar_fp,
        crest_deconf_fp,
        coupled_fields,
        {**base_prov, **delta_meta["provenance"]},
    )


def _pack_from_memmaps(
    X96: np.memmap,
    keep96: list[int],
    X24: np.memmap,
    keep24: list[int],
    Xd: np.memmap,
) -> np.ndarray:
    """Pack selected columns directly from disk memmaps into a single (n_fp, 128) float16 array."""
    n_fp = X96.shape[1]
    n_total = len(keep96) + len(keep24) + Xd.shape[0]
    out = np.empty((n_fp, n_total), dtype=np.float16)
    col = 0
    for j in keep96:
        v = np.asarray(X96[j], dtype=np.float32)
        out[:, col] = np.clip(v, -65000.0, 65000.0).astype(np.float16)
        col += 1
    for j in keep24:
        v = np.asarray(X24[j], dtype=np.float32)
        out[:, col] = np.clip(v, -65000.0, 65000.0).astype(np.float16)
        col += 1
    for j in range(Xd.shape[0]):
        v = np.asarray(Xd[j], dtype=np.float32)
        out[:, col] = np.clip(v, -65000.0, 65000.0).astype(np.float16)
        col += 1
    return out


def compute_h77h_sloco_mechanics_fp(
    visible_2d: np.ndarray,
    fp: np.ndarray,
    coupled_fields_2d: dict[str, np.ndarray],
    n_buckets: int = 8,
    max_seg_px: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (X_mech_train, X_mech_infer), each (9, n_fp) float32, strictly from ``visible_2d``."""
    vis = np.asarray(visible_2d, dtype=bool) & fp
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)

    reg35_fp = coupled_fields_2d["reg35_regime_2d"].ravel()[fp_idx]
    tip_coup_fp = coupled_fields_2d["tip_coupler_2d"].ravel()[fp_idx]
    rel_coup_fp = coupled_fields_2d["relay_coupler_2d"].ravel()[fp_idx]

    def _pack9(d1: np.ndarray, d2: np.ndarray, ls: np.ndarray, ll: np.ndarray) -> np.ndarray:
        out = np.empty((9, n_fp), dtype=np.float32)
        d1_fp = d1.ravel()[fp_idx]
        d2_fp = d2.ravel()[fp_idx]
        ls_fp = ls.ravel()[fp_idx]
        ll_fp = ll.ravel()[fp_idx]
        relay_fp = np.exp(-((np.maximum(d1_fp - 2.0, 0.0)) / 11.0) ** 2) * np.exp(
            -((np.maximum(d2_fp - 2.0, 0.0)) / 14.0) ** 2
        )
        out[0] = np.minimum(d1_fp, 80.0)
        out[1] = np.log1p(np.minimum(d1_fp, 150.0))
        out[2] = np.exp(-((np.maximum(d1_fp - 2.2, 0.0)) / 4.5) ** 2)
        out[3] = ls_fp
        out[4] = ll_fp
        out[5] = relay_fp
        out[6] = reg35_fp * (1.0 - np.exp(-d1_fp / 14.0))
        out[7] = (0.60 * ls_fp + 0.40 * ll_fp) * tip_coup_fp
        out[8] = relay_fp * rel_coup_fp
        return out

    d1_f, d2_f, ls_f, ll_f = _single_mask_mechanics(vis, fp, coupled_fields_2d["reg35_regime_2d"])
    X_infer = _pack9(d1_f, d2_f, ls_f, ll_f)
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
        d1_b, d2_b, ls_b, ll_b = _single_mask_mechanics(
            vis_minus_b, fp, coupled_fields_2d["reg35_regime_2d"]
        )
        sub9 = _pack9(d1_b, d2_b, ls_b, ll_b)
        b_fp = in_b.ravel()[fp_idx]
        X_train[:, b_fp] = sub9[:, b_fp]
        del excl, vis_minus_b, d1_b, d2_b, ls_b, ll_b, sub9

    gc.collect()
    return X_train, X_infer
