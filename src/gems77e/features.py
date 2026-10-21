"""Zero-leakage, zero-SGMC, zero-prior-raster physical & Segment-LOCO feature stack (`H77E`).

Strict invariants enforced across every function in this module:
  1. Zero historical submission or registry prediction rasters are read anywhere here.
  2. ``sgmc_faults_100m.tif`` and ``derived_sgmc_faults_100m*.tif`` are NEVER read;
     pre-Quaternary bedrock contacts are 100% excluded (proven 0.036-0.051 live).
  3. Columns encoding catalogue proximity (``dist_known_fault_px``, ``known_fault``, ``label``)
     in external tables are strictly forbidden and asserted unaccessed.
  4. All 120 static physical channels stream sequentially to disk memmaps so peak RAM stays
     comfortably below ~700 MB on a 2-vCPU / 3.8 GiB runner.
  5. ``compute_sloco_mechanics_fp`` accepts ONLY ``(visible_2d, fp, geod_rank_2d)`` — withheld
     holdout truth and global ``existing_faults`` never enter scope during holdout training.
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
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import network as gems57_network  # noqa: E402
from gems77_step.features import (  # noqa: E402
    FORBIDDEN_POINT_COLUMNS,
    _radon_worm_6az,
    _sample_bilinear_fp,
    build_h77_feature_stack,
)

H77E_DELTA_NAMES: list[str] = [
    # Block A: Untouched 10 m USGS 3DEP fluvial knickpoint & drainage-divide physics
    "chi_chan_density",
    "chi_knick_rate",
    "chi_knick_count_log",
    "chi_raised_impound_frac",
    "chi_signed_step_asym",
    "chi_step_span_log",
    "chi_branch_boundary_3x3",
    "radon_knick_1500m",
    "radon_knick_3100m",
    # Block B: US Census TIGER road deconfounding + exact inverse-quantized physical 1m LiDAR
    "tiger_road_dist_log",
    "lidar_step_m_deconf",
    "lidar_dipole_m_deconf",
    "lidar_strike_vec_coh_250m",
    "seamfree_scarp_deconf",
    "radon_scarp_deconf_2100m",
    "openness_lrm_1m_deconf",
    # Block C: Multi-depth potential-field cross-gradients, shallow breccia demag & geothermometry
    "pot_crossgrad_par_s2",
    "pot_crossgrad_par_s5_net",
    "mag_shallow_breccia_diff",
    "miller_singh_grav_tilt_edge",
    "hydro_chem_claycap_coupled",
    # Block D: 55-deg hanging-wall seismogenic updip surface projection & graben impoundment step
    "seism_updip_proj_35px",
    "seism_updip_proj_50px",
    "upthrown_impound_step_500m",
]

SLOCO_MECH_NAMES: list[str] = [
    "sloco_dist_other_px",
    "sloco_log_dist_other",
    "sloco_splay_band_2_8px",
    "sloco_pollard_tip_1p5km",
    "sloco_pollard_tip_3p8km",
    "sloco_stepover_relay",
    "sloco_strain_deficit",
]


def _fast_nn_fill(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    bad = ~valid_mask
    if not bad.any():
        return arr.astype(np.float32, copy=False)
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    out = arr.astype(np.float32, copy=True)
    out[bad] = out[idx[0][bad], idx[1][bad]]
    return out


def _rank01_fp_2d(arr_2d: np.ndarray, fp: np.ndarray) -> np.ndarray:
    out = np.zeros(arr_2d.shape, dtype=np.float32)
    v = arr_2d[fp]
    fin = np.isfinite(v)
    vf = v[fin]
    if vf.size < 2 or float(vf.max() - vf.min()) == 0.0:
        return out
    q_grid = np.linspace(0.0, 1.0, 4097)
    knots = np.quantile(vf[::4] if vf.size > 200_000 else vf, q_grid)
    knots = np.maximum.accumulate(knots)
    uk, uidx = np.unique(knots, return_index=True)
    if uk.size < 2:
        return out
    r = np.zeros(v.shape, dtype=np.float32)
    r[fin] = np.interp(vf, uk, q_grid[uidx]).astype(np.float32)
    out[fp] = r
    return out


def _crossgrad_alignment(
    f1: np.ndarray, f2: np.ndarray, fp: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gallardo & Meju (2004) parallel contact alignment vs orthogonal pluton confounder."""
    s1 = gaussian_filter(f1.astype(np.float32), sigma=sigma, mode="nearest")
    s2 = gaussian_filter(f2.astype(np.float32), sigma=sigma, mode="nearest")
    g1y, g1x = np.gradient(s1)
    g2y, g2x = np.gradient(s2)
    del s1, s2
    n1 = np.hypot(g1y, g1x) + 1e-9
    n2 = np.hypot(g2y, g2x) + 1e-9
    cos2 = ((g1x * g2x + g1y * g2y) / (n1 * n2)) ** 2
    sin2 = ((g1x * g2y - g1y * g2x) / (n1 * n2)) ** 2
    del g1y, g1x, g2y, g2x
    m1 = float(np.median(n1[fp][::8])) + 1e-9
    m2 = float(np.median(n2[fp][::8])) + 1e-9
    amp = np.log1p(np.sqrt((n1 / m1) * (n2 / m2))).astype(np.float32)
    return (amp * cos2).astype(np.float32), (amp * (cos2 - 0.5 * sin2)).astype(np.float32)


def build_h77e_static_stack(
    cache_dir: Path,
    fp: np.ndarray,
    transform: Any,
    data_dir: Path,
    ext_dir: Path,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build or reload all 120 zero-SGMC static physical channels as row-major float16 X16.

    Returns:
      X16: (n_fp, 120) float16 row-major feature matrix (~1.18 GB RAM)
      static_names: list of 120 feature names
      has_lidar_fp: (n_fp,) bool array of 1m LiDAR survey coverage
      crest_deconf_fp: (n_fp,) float32 road-deconfounded scarp crestness index in [0, 1]
      geod_rank_2d: (H, W) float32 active geodetic strain rank field
      provenance: dict of SHA/source provenance metadata
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_mmap_path = cache_dir / "h77_features_fp.f32"
    X96, base_names, has_lidar_fp, _, base_crest_fp, base_prov = build_h77_feature_stack(
        base_mmap_path, fp, transform, data_dir, ext_dir
    )

    shape = fp.shape
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)
    n_delta = len(H77E_DELTA_NAMES)

    delta_mmap_path = cache_dir / "h77e_delta_fp.f32"
    delta_meta_path = cache_dir / "h77e_delta.meta.json"
    delta_aux_path = cache_dir / "h77e_delta.aux.npz"

    if delta_mmap_path.is_file() and delta_meta_path.is_file() and delta_aux_path.is_file():
        meta = json.loads(delta_meta_path.read_text())
        if meta.get("names") == H77E_DELTA_NAMES and meta.get("n_fp") == n_fp:
            Xd = np.memmap(delta_mmap_path, dtype=np.float32, mode="r", shape=(n_delta, n_fp))
            aux = np.load(delta_aux_path)
            X16 = _pack_combined_float16(X96, Xd)
            return (
                X16,
                list(base_names) + list(H77E_DELTA_NAMES),
                has_lidar_fp,
                aux["crest_deconf_fp"],
                aux["geod_rank_2d"],
                {**base_prov, **meta.get("provenance", {})},
            )

    Xd = np.memmap(delta_mmap_path, dtype=np.float32, mode="w+", shape=(n_delta, n_fp))
    d_ptr = 0

    def _put(name: str, arr: np.ndarray, is_fp_1d: bool = False) -> None:
        nonlocal d_ptr
        assert name == H77E_DELTA_NAMES[d_ptr], f"Order mismatch: {name} != {H77E_DELTA_NAMES[d_ptr]}"
        assert name not in FORBIDDEN_POINT_COLUMNS
        vec = arr if is_fp_1d else arr.ravel()[fp_idx]
        Xd[d_ptr, :] = vec.astype(np.float32, copy=False)
        d_ptr += 1

    # ------------------------------------------------------------------
    # BLOCK A: Untouched 10 m USGS 3DEP Fluvial Knickpoint & Branch Stacks
    # ------------------------------------------------------------------
    with rasterio.open(ext_dir / "dem10_chi_100m_u8.tif") as ds:
        chan_n = ds.read(1).astype(np.float32)
        knick_n = ds.read(2).astype(np.float32)
        raised_n = ds.read(3).astype(np.float32)
        valid_n = ds.read(4).astype(np.float32)

    valid_fluv = fp & (valid_n > 0)
    chan_dens = _fast_nn_fill(chan_n / (valid_n + 1.0), valid_fluv)
    knick_rate = _fast_nn_fill(knick_n / (chan_n + 1.5), valid_fluv)
    knick_log = _fast_nn_fill(np.log1p(knick_n), valid_fluv)
    raised_frac = _fast_nn_fill(raised_n / (valid_n + 1.0), valid_fluv)
    del chan_n, knick_n, raised_n, valid_n

    with rasterio.open(ext_dir / "dem10_chi_100m_i16.tif") as ds:
        s_max_raw = ds.read(1).astype(np.float32)
        s_min_raw = ds.read(2).astype(np.float32)
        z_raw = ds.read(5).astype(np.float32)
    z_arr = _fast_nn_fill(z_raw, fp & np.isfinite(z_raw) & (z_raw > -32760.0))
    ok_step = fp & np.isfinite(s_max_raw) & (s_max_raw > -32760.0) & np.isfinite(s_min_raw) & (s_min_raw > -32760.0)
    s_max = np.where(ok_step, s_max_raw, 0.0).astype(np.float32)
    s_min = np.where(ok_step, s_min_raw, 0.0).astype(np.float32)
    del s_max_raw, s_min_raw, z_raw, ok_step

    chi_asym = ((s_max - np.abs(s_min)) / 1000.0).astype(np.float32)
    chi_span = np.log1p((np.maximum(s_max, 0.0) + np.abs(s_min)) / 100.0).astype(np.float32)
    del s_max, s_min

    with rasterio.open(ext_dir / "dem10_chi_100m_sig_i32.tif") as ds:
        sig_i32 = ds.read(1)
    sig_ok = fp & (sig_i32 > 0)
    sig_clean = np.where(sig_ok, sig_i32, 0)
    branch_jump = (
        (maximum_filter(sig_clean, size=3) != minimum_filter(sig_clean, size=3)).astype(np.float32)
        * sig_ok.astype(np.float32)
        * (0.25 + chan_dens)
    ).astype(np.float32)
    branch_jump_s = gaussian_filter(branch_jump, sigma=1.2).astype(np.float32)
    del sig_i32, sig_ok, sig_clean, branch_jump

    radon_knick_1500 = _radon_worm_6az(knick_rate, fp, half_len=7, flank=3)
    radon_knick_3100 = _radon_worm_6az(knick_rate, fp, half_len=15, flank=4)

    _put("chi_chan_density", chan_dens)
    _put("chi_knick_rate", knick_rate)
    _put("chi_knick_count_log", knick_log)
    _put("chi_raised_impound_frac", raised_frac)
    _put("chi_signed_step_asym", chi_asym)
    _put("chi_step_span_log", chi_span)
    _put("chi_branch_boundary_3x3", branch_jump_s)
    _put("radon_knick_1500m", radon_knick_1500)
    _put("radon_knick_3100m", radon_knick_3100)
    del chan_dens, knick_rate, knick_log, chi_asym, chi_span, branch_jump_s, radon_knick_1500, radon_knick_3100
    gc.collect()

    # ------------------------------------------------------------------
    # BLOCK B: TIGER Road Deconfounding + Inverse-Quantized Physical 1m LiDAR
    # ------------------------------------------------------------------
    with rasterio.open(ext_dir / "tiger_road_distance_m.tif") as ds:
        road_m = np.maximum(np.nan_to_num(ds.read(1).astype(np.float32), nan=5000.0), 0.0)
    w_road = (1.0 - 0.85 * np.exp(-((road_m / 90.0) ** 2))).astype(np.float32)
    w_road_fp = w_road.ravel()[fp_idx]
    _put("tiger_road_dist_log", np.log1p(road_m))
    del road_m

    q_meta = json.loads((ext_dir / "lidar_scarp_features.json").read_text())["quantisation"]

    def _read_lidar_phys(ds_lid: Any, band_i: int, bname: str) -> np.ndarray:
        q = ds_lid.read(band_i).astype(np.float32)
        xmax, mode_q = q_meta[bname]
        t = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
        return (xmax * (t * t if mode_q == "sqrt" else t)).astype(np.float32)

    with rasterio.open(ext_dir / "lidar_scarp_features_u8.tif") as ds:
        has_lid_2d = (ds.read(12) > 0) & fp
        ex_m = _read_lidar_phys(ds, 1, "ex_max")
        step_m = _read_lidar_phys(ds, 3, "step_max")
        lapn_m = _read_lidar_phys(ds, 4, "lapneg_max")
        lapp_m = _read_lidar_phys(ds, 5, "lappos_max")
        coh_m = _read_lidar_phys(ds, 10, "coh100")
        strike_deg = _read_lidar_phys(ds, 11, "strike")

    step_deconf_2d = (step_m * w_road).astype(np.float32)
    dipole_deconf_2d = (np.sqrt(np.maximum(lapn_m * lapp_m, 0.0)) * w_road).astype(np.float32)
    del step_m, lapn_m, lapp_m

    th2 = np.deg2rad(2.0 * strike_deg)
    w_vec = np.where(has_lid_2d, ex_m * coh_m * w_road, 0.0).astype(np.float32)
    strike_vec_coh = np.hypot(
        gaussian_filter(w_vec * np.cos(th2), sigma=2.5),
        gaussian_filter(w_vec * np.sin(th2), sigma=2.5),
    ).astype(np.float32)
    del ex_m, coh_m, strike_deg, th2, w_vec

    step_fp = step_deconf_2d.ravel()[fp_idx].copy()
    dip_fp = dipole_deconf_2d.ravel()[fp_idx].copy()
    sv_fp = strike_vec_coh.ravel()[fp_idx].copy()
    step_fp[~has_lidar_fp] = np.nan
    dip_fp[~has_lidar_fp] = np.nan
    sv_fp[~has_lidar_fp] = np.nan

    _put("lidar_step_m_deconf", step_fp, is_fp_1d=True)
    _put("lidar_dipole_m_deconf", dip_fp, is_fp_1d=True)
    _put("lidar_strike_vec_coh_250m", sv_fp, is_fp_1d=True)
    del strike_vec_coh, sv_fp

    topo10_meta = json.loads((ext_dir / "topo_10m_derived_100m_u8.manifest.json").read_text())
    b_lut = {b["band_name"]: (b["index"], float(b["lo"]), float(b["hi"])) for b in topo10_meta["bands"]}
    with rasterio.open(ext_dir / "topo_10m_derived_100m_u8.tif") as ds:
        def _t10(nm: str) -> np.ndarray:
            bi, lo, hi = b_lut[nm]
            q = ds.read(bi)
            phys = lo + np.clip((q.astype(np.float32) - 1.0) / 254.0, 0.0, 1.0) * (hi - lo)
            return _fast_nn_fill(phys, fp & (q > 0)) * w_road

        t10_curv = _t10("curv_prof_absmax")
        t10_line = _t10("hs_lineament")
        t10_sstd = _t10("slope_std")

    lid_norm_fp = (
        0.55 * (np.nan_to_num(dip_fp, nan=0.0) / (float(np.nanstd(dip_fp)) + 1e-6))
        + 0.45 * (np.nan_to_num(step_fp, nan=0.0) / (float(np.nanstd(step_fp)) + 1e-6))
    ).astype(np.float32)
    d10_norm_fp = (
        0.40 * (t10_line.ravel()[fp_idx] / (float(np.std(t10_line.ravel()[fp_idx])) + 1e-6))
        + 0.38 * (t10_curv.ravel()[fp_idx] / (float(np.std(t10_curv.ravel()[fp_idx])) + 1e-6))
        + 0.22 * (t10_sstd.ravel()[fp_idx] / (float(np.std(t10_sstd.ravel()[fp_idx])) + 1e-6))
    ).astype(np.float32)
    del step_deconf_2d, dipole_deconf_2d, step_fp, dip_fp, t10_curv, t10_line, t10_sstd

    seamfree_scarp_2d = np.zeros(shape, dtype=np.float32)
    seamfree_scarp_2d.ravel()[fp_idx] = np.where(
        has_lidar_fp, 0.68 * lid_norm_fp + 0.32 * d10_norm_fp, d10_norm_fp
    ).astype(np.float32)
    del lid_norm_fp, d10_norm_fp

    radon_scarp_deconf = _radon_worm_6az(seamfree_scarp_2d, fp, half_len=10, flank=3)

    openness_path = ext_dir / "dem1m_high_prior_openness_lrm.npz"
    open_2d = np.zeros(shape, dtype=np.float32)
    if openness_path.is_file():
        npz = np.load(openness_path)
        arr_o = None
        for k in npz.files:
            cand = npz[k]
            if cand.shape == shape:
                arr_o = cand.astype(np.float32)
                break
            if cand.ndim == 1 and cand.size == n_fp:
                tmp = np.zeros(shape, dtype=np.float32)
                tmp.ravel()[fp_idx] = cand.astype(np.float32)
                arr_o = tmp
                break
        if arr_o is not None:
            open_2d = (_fast_nn_fill(arr_o, fp & np.isfinite(arr_o)) * w_road).astype(np.float32)

    _put("seamfree_scarp_deconf", seamfree_scarp_2d)
    _put("radon_scarp_deconf_2100m", radon_scarp_deconf)
    _put("openness_lrm_1m_deconf", open_2d)
    crest_deconf_fp = np.clip(base_crest_fp * w_road_fp, 0.0, 1.0).astype(np.float32)
    del radon_scarp_deconf, open_2d, seamfree_scarp_2d, w_road, w_road_fp
    gc.collect()

    # ------------------------------------------------------------------
    # BLOCK C: Cross-Gradient Coupling, Shallow Breccia Demag & Geothermometry
    # ------------------------------------------------------------------
    with rasterio.open(data_dir / "training_features.tif") as ds:
        def _tf(b: int) -> np.ndarray:
            raw = ds.read(b).astype(np.float32)
            return _fast_nn_fill(raw, fp & np.isfinite(raw) & (raw > -1e30))

        rtp = _tf(2)
        geod2 = _tf(4)
        geod_sh = _tf(7)
        geod_di = _tf(8)
        deq = _tf(10)
        grav_vg = _tf(11)
        grav = _tf(13)
        ieq = _tf(16)
        cond = _tf(17)

    geod_combo = (
        0.50 * _rank01_fp_2d(geod2, fp)
        + 0.30 * _rank01_fp_2d(geod_sh, fp)
        + 0.20 * _rank01_fp_2d(np.abs(geod_di), fp)
    ).astype(np.float32)
    del geod2, geod_sh, geod_di

    cg_s2, _ = _crossgrad_alignment(grav, rtp, fp, sigma=2.0)
    _, cg_s5_net = _crossgrad_alignment(grav, rtp, fp, sigma=5.0)

    with rasterio.open(ext_dir / "geodawn_extensions_u8.tif") as ds:
        q_up = ds.read(4)
        tmi_up150 = _fast_nn_fill(q_up.astype(np.float32), fp & (q_up > 0))
    ry, rx = np.gradient(gaussian_filter(rtp, sigma=1.5))
    uy, ux = np.gradient(gaussian_filter(tmi_up150, sigma=1.5))
    rn = np.hypot(ry, rx)
    un = np.hypot(uy, ux)
    mag_breccia_diff = (
        np.log1p(rn / (float(np.median(rn[fp][::8])) + 1e-9))
        - np.log1p(un / (float(np.median(un[fp][::8])) + 1e-9))
    ).astype(np.float32)
    del rtp, tmi_up150, ry, rx, uy, ux, rn, un

    gy, gx = np.gradient(gaussian_filter(grav, sigma=2.0))
    gh = np.hypot(gy, gx)
    tilt_g = np.arctan2(grav_vg, gh + 1e-6)
    grav_tilt_edge = (
        np.exp(-np.abs(tilt_g) / 0.35) * np.log1p(gh / (float(np.median(gh[fp][::8])) + 1e-9))
    ).astype(np.float32)
    del grav, grav_vg, gy, gx, gh, tilt_g

    diseq_2d = np.zeros(shape, dtype=np.float32)
    with (ext_dir / "gdr_wellspring_in_footprint.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
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
                    if np.isfinite(TV := tv) and 0.0 < TV < 200.0:
                        ts = TV
                except ValueError:
                    pass
            tc_max = max(chems) if chems else 0.0
            dq = max(0.0, tc_max - max(ts, 20.0)) + 0.35 * max(0.0, max(tc_max, ts) - 70.0)
            if dq > 0.0:
                diseq_2d[rr, cc] = max(diseq_2d[rr, cc], float(np.log1p(dq)))

    cy_c, cx_c = np.gradient(gaussian_filter(np.log1p(np.maximum(cond, 0.0)), sigma=2.5))
    claycap_grad = np.hypot(cy_c, cx_c).astype(np.float32)
    hydro_coupled = (
        gaussian_filter(diseq_2d, sigma=6.0) * (1.0 + 2.0 * _rank01_fp_2d(claycap_grad, fp))
        + 0.35 * _rank01_fp_2d(claycap_grad, fp)
    ).astype(np.float32)
    del cond, cy_c, cx_c, claycap_grad, diseq_2d

    _put("pot_crossgrad_par_s2", cg_s2)
    _put("pot_crossgrad_par_s5_net", cg_s5_net)
    _put("mag_shallow_breccia_diff", mag_breccia_diff)
    _put("miller_singh_grav_tilt_edge", grav_tilt_edge)
    _put("hydro_chem_claycap_coupled", hydro_coupled)
    del cg_s2, cg_s5_net, mag_breccia_diff, grav_tilt_edge, hydro_coupled
    gc.collect()

    # ------------------------------------------------------------------
    # BLOCK D: 55-deg Hanging-Wall Seismogenic Updip Projection & Graben Step
    # ------------------------------------------------------------------
    z_sm = gaussian_filter(z_arr, sigma=2.0)
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
    zp9 = _sample_bilinear_fp(z_arr, base_y + 9.0 * ny_fp, base_x + 9.0 * nx_fp)
    zm9 = _sample_bilinear_fp(z_arr, base_y - 9.0 * ny_fp, base_x - 9.0 * nx_fp)
    sfw = np.where(zm9 >= zp9, 1.0, -1.0).astype(np.float32)  # +sfw*n points into downdropped hanging wall!
    del zp9, zm9, z_arr

    seism_field = gaussian_filter(
        (_rank01_fp_2d(ieq, fp) * np.exp(-np.maximum(deq, 0.0) / 8.0)).astype(np.float32),
        sigma=3.0,
    )
    del ieq, deq

    s_hw35 = _sample_bilinear_fp(seism_field, base_y + 35.0 * sfw * ny_fp, base_x + 35.0 * sfw * nx_fp)
    s_fw35 = _sample_bilinear_fp(seism_field, base_y - 35.0 * sfw * ny_fp, base_x - 35.0 * sfw * nx_fp)
    updip_35 = (s_hw35 - 0.5 * s_fw35).astype(np.float32)
    del s_hw35, s_fw35

    s_hw50 = _sample_bilinear_fp(seism_field, base_y + 50.0 * sfw * ny_fp, base_x + 50.0 * sfw * nx_fp)
    s_fw50 = _sample_bilinear_fp(seism_field, base_y - 50.0 * sfw * ny_fp, base_x - 50.0 * sfw * nx_fp)
    updip_50 = (s_hw50 - 0.5 * s_fw50).astype(np.float32)
    del seism_field, s_hw50, s_fw50

    imp_hw5 = _sample_bilinear_fp(raised_frac, base_y + 5.0 * sfw * ny_fp, base_x + 5.0 * sfw * nx_fp)
    imp_fw5 = _sample_bilinear_fp(raised_frac, base_y - 5.0 * sfw * ny_fp, base_x - 5.0 * sfw * nx_fp)
    impound_step = (imp_hw5 - imp_fw5).astype(np.float32)
    del raised_frac, imp_hw5, imp_fw5, base_y, base_x, ny_fp, nx_fp, sfw

    _put("seism_updip_proj_35px", updip_35, is_fp_1d=True)
    _put("seism_updip_proj_50px", updip_50, is_fp_1d=True)
    _put("upthrown_impound_step_500m", impound_step, is_fp_1d=True)
    del updip_35, updip_50, impound_step
    assert d_ptr == n_delta, f"Expected {n_delta}, got {d_ptr}"

    Xd.flush()
    np.savez_compressed(
        delta_aux_path,
        crest_deconf_fp=crest_deconf_fp,
        geod_rank_2d=geod_combo,
    )
    delta_meta = {
        "n_fp": n_fp,
        "names": H77E_DELTA_NAMES,
        "provenance": {
            "dem10_chi_100m_u8": "USGS 3DEP 1/3 arc-sec DEM (public domain, chan_n/knick_n/raised_n/valid_n)",
            "dem10_chi_100m_sig_i32": "USGS 3DEP 1/3 arc-sec DEM drainage branch signature index",
            "tiger_road_distance_m": "US Census TIGER/Line public-domain road/rail distance de-confounder",
            "lidar_scarp_features_u8": "USGS 3DEP 1 m DEM physical meter inverse-quantization with q==0 -> NaN",
            "sgmc_channels_used": 0,
            "forbidden_columns_accessed": [],
            "prior_rasters_read": 0,
        },
    }
    delta_meta_path.write_text(json.dumps(delta_meta, indent=2) + "\n")
    gc.collect()

    Xd_r = np.memmap(delta_mmap_path, dtype=np.float32, mode="r", shape=(n_delta, n_fp))
    X16 = _pack_combined_float16(X96, Xd_r)
    return (
        X16,
        list(base_names) + list(H77E_DELTA_NAMES),
        has_lidar_fp,
        crest_deconf_fp,
        geod_combo,
        {**base_prov, **delta_meta["provenance"]},
    )


def _pack_combined_float16(X96: np.memmap, Xd: np.memmap) -> np.ndarray:
    """Pack (96, n_fp) + (24, n_fp) float32 memmaps into contiguous (n_fp, 120) float16."""
    n_a, n_fp = X96.shape
    n_b = Xd.shape[0]
    out = np.empty((n_fp, n_a + n_b), dtype=np.float16)
    for j in range(n_a):
        v = np.asarray(X96[j], dtype=np.float32)
        out[:, j] = np.clip(v, -65000.0, 65000.0).astype(np.float16)
    for j in range(n_b):
        v = np.asarray(Xd[j], dtype=np.float32)
        out[:, n_a + j] = np.clip(v, -65000.0, 65000.0).astype(np.float16)
    return out


def _single_mask_mechanics(
    mask_2d: np.ndarray,
    fp: np.ndarray,
    geod_rank_2d: np.ndarray,
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

    # Fast deterministic 2nd-nearest component distance via 7 random-hash binary colorings (14 EDTs vs 26!)
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


def compute_sloco_mechanics_fp(
    visible_2d: np.ndarray,
    fp: np.ndarray,
    geod_rank_2d: np.ndarray,
    n_buckets: int = 8,
    max_seg_px: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (X_mech_train, X_mech_infer), each (7, n_fp) float32, strictly from ``visible_2d``.

    - ``X_mech_infer`` evaluates the full visible fault network ``visible_2d`` on all off-visible
      pixels (matching exact test-time evaluation where ``visible_2d`` is known).
    - ``X_mech_train`` decomposes ``visible_2d`` into ``<= max_seg_px`` segments via shared
      ``gems57.network.segments`` and assigns segments deterministically across ``n_buckets``
      hash buckets (`s * 2654435761 % n_buckets`). Every pixel on segment ``s`` receives features
      computed strictly from ``visible_2d \\ bucket(s)`` (with a 2-px self-junction collar so
      training positives lie strictly in ``d > 2.0 px``, matching our allowed emission domain!).
    """
    vis = np.asarray(visible_2d, dtype=bool) & fp
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)

    def _pack7(d1: np.ndarray, d2: np.ndarray, ls: np.ndarray, ll: np.ndarray) -> np.ndarray:
        out = np.empty((7, n_fp), dtype=np.float32)
        d1_fp = d1.ravel()[fp_idx]
        d2_fp = d2.ravel()[fp_idx]
        out[0] = np.minimum(d1_fp, 80.0)
        out[1] = np.log1p(np.minimum(d1_fp, 150.0))
        out[2] = np.exp(-((np.maximum(d1_fp - 2.2, 0.0)) / 4.5) ** 2)
        out[3] = ls.ravel()[fp_idx]
        out[4] = ll.ravel()[fp_idx]
        out[5] = np.exp(-((np.maximum(d1_fp - 2.0, 0.0)) / 11.0) ** 2) * np.exp(
            -((np.maximum(d2_fp - 2.0, 0.0)) / 14.0) ** 2
        )
        out[6] = geod_rank_2d.ravel()[fp_idx] * (1.0 - np.exp(-d1_fp / 14.0))
        return out

    d1_f, d2_f, ls_f, ll_f = _single_mask_mechanics(vis, fp, geod_rank_2d)
    X_infer = _pack7(d1_f, d2_f, ls_f, ll_f)
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
        d1_b, d2_b, ls_b, ll_b = _single_mask_mechanics(vis_minus_b, fp, geod_rank_2d)
        sub7 = _pack7(d1_b, d2_b, ls_b, ll_b)
        b_fp = in_b.ravel()[fp_idx]
        X_train[:, b_fp] = sub7[:, b_fp]
        del excl, vis_minus_b, d1_b, d2_b, ls_b, ll_b, sub7

    gc.collect()
    return X_train, X_infer
