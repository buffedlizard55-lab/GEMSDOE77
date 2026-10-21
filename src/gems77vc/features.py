"""Zero-SGMC Geothermal Vent Upflow, Smectite Clay-Cap Breach & V-LBCO Relay Mechanics (`Lane H77V`).

Implements the 15 novel preregistered H77V channels (12 static physical/geothermal channels +
3 fold-safe Voronoi Leave-Bucket-of-Components-Out vent-bridged step-over & crack-tip channels)
on top of the cached 96-channel physical feature stack (`111` total zero-SGMC channels).

Invariants enforced:
  1. Zero SGMC bedrock contact channels (`sgmc_channels_used == 0`).
  2. Zero label-derived columns (`dist_known_fault_px`, `known_fault`, `label`) read.
  3. Zero pixels copied from `H19-5`, `H33`, or any prior submission raster.
  4. All catalogue-dependent features (`vis_*`, 6 channels) are computed strictly from
     `visible_2d` at inference time and via 5-bucket Voronoi Leave-Bucket-of-Components-Out
     (`V-LBCO`) at training time so training positives never suffer from the `d_vis == 0` tautology.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label as ndi_label
from scipy.stats import rankdata

from gems77_step.features import build_h77_feature_stack, compute_visible_mechanics_fp

H77V_STATIC_NAMES = [
    "h77v_claycap_hessian_ridge",
    "h77v_claycap_lat_grad",
    "h77v_mag_destruction_trough",
    "h77v_clay_mag_destr_coupling",
    "h77v_strike_steered_2m_temp_plume",
    "h77v_strike_steered_geochem_plume",
    "h77v_piezo_sinter_spillover",
    "h77v_swarm_dilation_overpressure",
    "h77v_transtensional_crit_stress",
    "h77v_piedmont_blind_conduit_index",
    "h77v_upflow_scarp_coincidence",
    "h77v_master_vent_potential",
]

H77V_MECH_NAMES = [
    "vis_pollard_tip_cone_1km",
    "vis_pollard_tip_cone_2p5km",
    "vis_orphan_geothermal_conduit",
    "vis_vent_bridged_relay_corridor",
    "vis_vent_tip_propagation_2p5km",
    "vis_blind_stepover_vent_band",
]


def _robust_z_pos(arr_fp: np.ndarray, clip_max: float = 8.0) -> np.ndarray:
    """Robust positive z-score in [0, clip_max] using median and IQR on footprint pixels."""
    sub = arr_fp[::13]
    valid = sub[np.isfinite(sub)]
    if valid.size == 0:
        return np.zeros_like(arr_fp, dtype=np.float32)
    med = float(np.median(valid))
    q75, q25 = np.percentile(valid, [75.0, 25.0])
    scale = float(max(q75 - q25, float(np.std(valid)), 1e-5))
    return np.clip((arr_fp - med) / scale, 0.0, clip_max).astype(np.float32)


def _robust_z_signed(arr_fp: np.ndarray, clip_abs: float = 8.0) -> np.ndarray:
    sub = arr_fp[::13]
    valid = sub[np.isfinite(sub)]
    if valid.size == 0:
        return np.zeros_like(arr_fp, dtype=np.float32)
    med = float(np.median(valid))
    q75, q25 = np.percentile(valid, [75.0, 25.0])
    scale = float(max(q75 - q25, float(np.std(valid)), 1e-5))
    return np.clip((arr_fp - med) / scale, -clip_abs, clip_abs).astype(np.float32)


def _fp_to_2d(vec_fp: np.ndarray, fp: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.full(fp.shape, fill, dtype=np.float32)
    out[fp] = vec_fp.astype(np.float32, copy=False)
    return out


def _hessian_ridge_and_grad(arr_2d: np.ndarray, sigma: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Return upward-convex Hessian ridge max(-lambda_min, 0) and gradient magnitude."""
    sm = gaussian_filter(arr_2d.astype(np.float32), sigma=sigma)
    gy, gx = np.gradient(sm)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    tr = gxx + gyy
    det_disc = np.hypot(gxx - gyy, 2.0 * gxy)
    lam_min = 0.5 * (tr - det_disc)
    ridge = np.maximum(-lam_min, 0.0).astype(np.float32)
    grad = np.hypot(gy, gx).astype(np.float32)
    return ridge, grad


def _compute_structure_double_angle_fp(
    det_elev_2d: np.ndarray,
    grav_2d: np.ndarray,
    fp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local multi-physics structural strike double-angle (cos 2theta, sin 2theta)."""
    ey, ex = np.gradient(gaussian_filter(det_elev_2d, sigma=2.0))
    gy, gx = np.gradient(gaussian_filter(grav_2d, sigma=3.0))
    # Normal vector to fault strike is gradient (gx, gy); double-angle of normal:
    j_xx = gaussian_filter(ex * ex + 0.5 * gx * gx, sigma=3.5)[fp]
    j_yy = gaussian_filter(ey * ey + 0.5 * gy * gy, sigma=3.5)[fp]
    j_xy = gaussian_filter(ex * ey + 0.5 * gx * gy, sigma=3.5)[fp]
    denom = np.hypot(j_xx - j_yy, 2.0 * j_xy) + 1e-8
    cos2t_n = ((j_xx - j_yy) / denom).astype(np.float32)
    sin2t_n = ((2.0 * j_xy) / denom).astype(np.float32)
    return cos2t_n, sin2t_n


def _strike_steered_advection(
    field_2d: np.ndarray,
    cos2t_n_fp: np.ndarray,
    sin2t_n_fp: np.ndarray,
    fp: np.ndarray,
    sigma_along: float = 4.0,
) -> np.ndarray:
    """Compute strike-steered anisotropic plume enhancement along local structural strike."""
    sm = gaussian_filter(field_2d.astype(np.float32), sigma=1.8)
    gy, gx = np.gradient(sm)
    g_mag2 = gx * gx + gy * gy + 1e-8
    c2_fp = ((gx[fp] * gx[fp] - gy[fp] * gy[fp]) / g_mag2[fp]).astype(np.float32)
    s2_fp = ((2.0 * gx[fp] * gy[fp]) / g_mag2[fp]).astype(np.float32)
    alignment = np.clip(0.5 * (1.0 + c2_fp * cos2t_n_fp + s2_fp * sin2t_n_fp), 0.0, 1.0).astype(np.float32)
    plume_broad = gaussian_filter(field_2d.astype(np.float32), sigma=sigma_along)[fp]
    return (plume_broad * (0.5 + 0.5 * alignment)).astype(np.float32)


def build_h77v_static_features(
    base_mmap_path: Path,
    h77v_mmap_path: Path,
    fp: np.ndarray,
    transform: Any,
    data_dir: Path,
    ext_dir: Path,
    force: bool = False,
) -> tuple[np.memmap, list[str], np.memmap, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build or reload both the 96-ch base stack and the 12-ch H77V static geothermal vent block."""
    X_base, base_names, has_lidar_fp, hydro_imp_fp, crest_fp, base_prov = build_h77_feature_stack(
        base_mmap_path, fp, transform, data_dir, ext_dir
    )
    n_fp = int(X_base.shape[1])
    n_v = len(H77V_STATIC_NAMES)
    meta_path = h77v_mmap_path.with_suffix(".meta.json")

    if (not force) and h77v_mmap_path.is_file() and meta_path.is_file():
        meta_obj = json.loads(meta_path.read_text())
        if meta_obj.get("n_features") == n_v and meta_obj.get("n_footprint_pixels") == n_fp:
            X_v = np.memmap(h77v_mmap_path, dtype=np.float32, mode="r", shape=(n_v, n_fp))
            vent_pot_fp = np.asarray(X_v[H77V_STATIC_NAMES.index("h77v_master_vent_potential"), :], dtype=np.float32)
            return X_base, base_names, X_v, list(H77V_STATIC_NAMES), has_lidar_fp, hydro_imp_fp, crest_fp, vent_pot_fp, meta_obj

    h77v_mmap_path.parent.mkdir(parents=True, exist_ok=True)
    X_v = np.memmap(h77v_mmap_path, dtype=np.float32, mode="w+", shape=(n_v, n_fp))
    b_idx = {nm: i for i, nm in enumerate(base_names)}

    def _col(name: str) -> np.ndarray:
        return np.asarray(X_base[b_idx[name], :], dtype=np.float32)

    # 1-2. Smectite clay-cap Hessian ridge & lateral boundary gradient from log(1 + max(cond_surf, 0))
    cond_fp = np.log1p(np.maximum(_col("cond_surf"), 0.0)).astype(np.float32)
    cond_2d = _fp_to_2d(cond_fp, fp, fill=float(np.median(cond_fp[::17])))
    clay_ridge_2d, clay_grad_2d = _hessian_ridge_and_grad(cond_2d, sigma=2.5)
    f01_clay_ridge = clay_ridge_2d[fp].astype(np.float32)
    f02_clay_grad = clay_grad_2d[fp].astype(np.float32)
    del cond_2d, clay_ridge_2d, clay_grad_2d

    # 3. Hydrothermal titanomagnetite-destruction trough: max(G_15(rtp) - G_2(rtp), 0)
    rtp_fp = _col("rtp")
    rtp_2d = _fp_to_2d(rtp_fp, fp, fill=float(np.median(rtp_fp[::17])))
    rtp_reg = gaussian_filter(rtp_2d, sigma=15.0)
    rtp_loc = gaussian_filter(rtp_2d, sigma=2.0)
    f03_mag_destr = np.maximum(rtp_reg[fp] - rtp_loc[fp], 0.0).astype(np.float32)
    del rtp_fp, rtp_2d, rtp_reg, rtp_loc

    # 4. Multiplicative clay-cap + magnetite-destruction + potassic K/Th alteration coupling
    k_over_th_z = _robust_z_pos(_col("rad_K_alteration_halo") - 0.5 * _col("ext_ThK"))
    z_clay = _robust_z_pos(f01_clay_ridge + 0.5 * f02_clay_grad)
    z_magd = _robust_z_pos(f03_mag_destr)
    f04_coupling = (z_clay * z_magd * (1.0 + 0.35 * k_over_th_z)).astype(np.float32)

    # 5-6. Strike-steered 2m temperature probe & chemical geothermometer advection plumes
    det_elev_fp = _col("det_elev")
    det_elev_2d = _fp_to_2d(det_elev_fp, fp, fill=float(np.median(det_elev_fp[::17])))
    grav_fp = _col("iso_grav_anom")
    grav_2d = _fp_to_2d(grav_fp, fp, fill=float(np.median(grav_fp[::17])))
    cos2t_n_fp, sin2t_n_fp = _compute_structure_double_angle_fp(det_elev_2d, grav_2d, fp)
    del det_elev_2d, grav_fp, grav_2d

    probe_fp = _col("probe_2m_density_600m") + 0.5 * _col("quat_vent_halo_1200m")
    probe_2d = _fp_to_2d(probe_fp, fp, fill=0.0)
    f05_probe_steered = _strike_steered_advection(probe_2d, cos2t_n_fp, sin2t_n_fp, fp, sigma_along=3.5)
    del probe_fp, probe_2d

    geochem_fp = _col("hydro_upflow_near_400m") + 0.75 * _col("hydro_upflow_far_1400m")
    geochem_2d = _fp_to_2d(geochem_fp, fp, fill=0.0)
    f06_geochem_steered = _strike_steered_advection(geochem_2d, cos2t_n_fp, sin2t_n_fp, fp, sigma_along=4.0)
    del geochem_fp, geochem_2d, cos2t_n_fp, sin2t_n_fp

    # 7. Hypothesis H77V-2: Paleogeothermal sinter/travertine piezometric elevation-band spillover
    piezo_window = np.exp(-0.5 * ((det_elev_fp - 35.0) / 55.0) ** 2).astype(np.float32)
    z_grav_hg = _robust_z_pos(_col("iso_grav_anom_hg"))
    z_sinter = _robust_z_pos(_col("paleo_sinter_seal_800m"))
    z_probe = _robust_z_pos(f05_probe_steered)
    f07_piezo_sinter = (piezo_window * z_grav_hg * (0.4 + 0.6 * z_sinter + 0.4 * z_probe)).astype(np.float32)

    # 8. Hypothesis H77V-3: Fluid-driven microseismic swarm clustering ratio x positive crustal dilatation
    z_ieq = _robust_z_signed(_col("ieq_n100a15"))
    z_deq = _robust_z_signed(_col("deq_n100a15"))
    swarm_diff = np.maximum(z_ieq - 0.5 * z_deq, 0.0).astype(np.float32)
    z_dilate = _robust_z_pos(np.maximum(_col("geod_dilaterate"), 0.0))
    z_cond = _robust_z_pos(cond_fp)
    f08_swarm_overp = (swarm_diff * (0.35 + 0.65 * z_dilate) * (0.35 + 0.65 * z_cond)).astype(np.float32)
    del z_ieq, z_deq, swarm_diff, cond_fp

    # 9. Transtensional critical-stress index: shear rate x positive dilatation x basement curvature
    z_shear = _robust_z_pos(_col("geod_shearrate"))
    z_base_w = _robust_z_pos(np.abs(_col("basement_curv")) + 0.5 * _col("basement_grad"))
    f09_crit_stress = (z_shear * (0.3 + 0.7 * z_dilate) * (1.0 + 0.45 * z_base_w)).astype(np.float32)
    del z_shear, z_dilate, z_base_w

    # 10. Piedmont blind hydrothermal-scarp composite (10m 3DEP curvature + chi step + alteration in piedmont band)
    piedmont_band = np.exp(-0.5 * ((np.abs(det_elev_fp) - 55.0) / 75.0) ** 2).astype(np.float32)
    z_curv10 = _robust_z_pos(_col("topo10_curv_prof_absmax"))
    z_chi = _robust_z_pos(_col("chi_step_max"))
    f10_piedmont_conduit = (
        piedmont_band
        * (0.45 * z_curv10 + 0.35 * z_chi + 0.20 * z_grav_hg)
        * (0.5 + 0.5 * _robust_z_pos(f04_coupling + f07_piezo_sinter))
    ).astype(np.float32)
    del det_elev_fp, piedmont_band, z_curv10, z_chi, z_grav_hg

    # 11. Upflow-scarp coincidence: Dip-Slip Step Purity x Hydrothermal Upflow Halo
    z_step = _robust_z_pos(
        np.maximum(_col("hwfw_step_purity_300m"), 0.0)
        + 0.6 * np.maximum(_col("hwfw_step_purity_900m"), 0.0)
        + 0.5 * np.maximum(_col("hwfw_dip_grav_concord"), 0.0)
    )
    z_upflow = _robust_z_pos(
        0.30 * z_clay + 0.25 * z_magd + 0.25 * _robust_z_pos(f06_geochem_steered) + 0.20 * z_probe
    )
    f11_upflow_scarp = (z_step * (0.35 + 0.65 * z_upflow)).astype(np.float32)
    del z_step, z_upflow, z_clay, z_magd, z_probe, z_sinter

    # 12. Master Geothermal Vent & Breach Potential in [0, 1]
    raw_master = (
        0.22 * _robust_z_pos(f04_coupling)
        + 0.18 * _robust_z_pos(f05_probe_steered)
        + 0.18 * _robust_z_pos(f06_geochem_steered)
        + 0.14 * _robust_z_pos(f07_piezo_sinter)
        + 0.10 * _robust_z_pos(f08_swarm_overp)
        + 0.08 * _robust_z_pos(f09_crit_stress)
        + 0.10 * _robust_z_pos(f11_upflow_scarp)
    ).astype(np.float32)
    f12_master_vent = (rankdata(raw_master, method="average") / float(n_fp)).astype(np.float32)
    del raw_master

    cols = [
        f01_clay_ridge,
        f02_clay_grad,
        f03_mag_destr,
        f04_coupling,
        f05_probe_steered,
        f06_geochem_steered,
        f07_piezo_sinter,
        f08_swarm_overp,
        f09_crit_stress,
        f10_piedmont_conduit,
        f11_upflow_scarp,
        f12_master_vent,
    ]
    for i, arr in enumerate(cols):
        X_v[i, :] = arr
    X_v.flush()

    meta_obj = {
        "schema": "gems77v-static-v1",
        "n_features": n_v,
        "n_footprint_pixels": n_fp,
        "feature_names": list(H77V_STATIC_NAMES),
        "sgmc_channels_used": 0,
        "forbidden_columns_checked": ["dist_known_fault_px", "known_fault", "label"],
        "base_provenance": base_prov,
    }
    meta_path.write_text(json.dumps(meta_obj, indent=2) + "\n")
    gc.collect()
    return (
        X_base,
        base_names,
        X_v,
        list(H77V_STATIC_NAMES),
        has_lidar_fp,
        hydro_imp_fp,
        crest_fp,
        f12_master_vent,
        meta_obj,
    )


def compute_h77v_visible_mechanics_fp(
    visible_2d: np.ndarray,
    fp: np.ndarray,
    hydro_imp_fp: np.ndarray,
    vent_pot_fp: np.ndarray,
    upflow_scarp_fp: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute 6 fold-safe visible-fault mechanics channels (3 base + 3 H77V vent-bridged).

    Strictly derived from ``visible_2d`` ONLY (never touches withheld fault segments).
    """
    base_mech, _ = compute_visible_mechanics_fp(visible_2d, fp, hydro_imp_fp)
    vis = (np.asarray(visible_2d, dtype=bool) & fp).astype(np.float32)

    # En-echelon relay bridge between distinct visible strands (positive at 300m-2.5km step-over gaps)
    g_near = gaussian_filter(vis, sigma=3.0)
    g_mid = gaussian_filter(vis, sigma=8.5)
    g_far = gaussian_filter(vis, sigma=20.0)
    relay_bridge_fp = np.maximum(g_mid[fp] * g_far[fp] - 0.85 * (g_near[fp] ** 2), 0.0).astype(np.float32)
    del g_near, g_mid, g_far

    d_vis_fp = distance_transform_edt(~(visible_2d & fp)).astype(np.float32)[fp]

    # 4. Vent-bridged relay corridor: relay_bridge * master_vent_potential
    m4_vent_relay = (relay_bridge_fp * (0.25 + 0.75 * vent_pot_fp)).astype(np.float32)

    # 5. Vent-coupled Pollard crack-tip propagation (2.5 km cone x hydrothermal breach/scarp)
    z_scarp_v = _robust_z_pos(upflow_scarp_fp)
    m5_vent_tip = (
        base_mech[1, :] * (0.40 + 0.35 * vent_pot_fp + 0.25 * np.minimum(z_scarp_v / 3.0, 1.0))
    ).astype(np.float32)

    # 6. Blind step-over distance-band (300m to 2.5km) x hydrothermal vent potential
    stepover_band = (
        (1.0 - np.exp(-((d_vis_fp / 4.5) ** 2))) * np.exp(-((np.maximum(d_vis_fp - 11.0, 0.0) / 14.0) ** 2))
    ).astype(np.float32)
    m6_stepover_vent = (stepover_band * vent_pot_fp * (1.0 + 2.5 * relay_bridge_fp)).astype(np.float32)

    mech6 = np.vstack([
        base_mech[0, :],
        base_mech[1, :],
        base_mech[2, :],
        m4_vent_relay,
        m5_vent_tip,
        m6_stepover_vent,
    ]).astype(np.float32)
    return mech6, list(H77V_MECH_NAMES)


def compute_h77v_vlbco_mechanics_fp(
    visible_2d: np.ndarray,
    fp: np.ndarray,
    hydro_imp_fp: np.ndarray,
    vent_pot_fp: np.ndarray,
    upflow_scarp_fp: np.ndarray,
    n_buckets: int = 5,
    seed: int = 7705,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """5-bucket Voronoi Leave-Bucket-of-Components-Out (`V-LBCO`) for all 6 mechanics channels."""
    vis = np.asarray(visible_2d, dtype=bool) & fp
    fp_idx = np.flatnonzero(fp)

    yy, xx = np.mgrid[-3:4, -3:4]
    struct_buf = (yy * yy + xx * xx) <= 9
    grouped_mask = binary_dilation(vis, structure=struct_buf)
    comp_map, n_comp = ndi_label(grouped_mask, structure=np.ones((3, 3), dtype=np.int32))
    comp_on_vis = np.where(vis, comp_map, 0)

    near_idx = distance_transform_edt(~vis, return_distances=False, return_indices=True)
    nearest_comp_fp = comp_on_vis[near_idx[0], near_idx[1]].ravel()[fp_idx]

    rng = np.random.default_rng(seed)
    comp_bucket = rng.integers(0, n_buckets, size=n_comp + 1)
    pixel_bucket_fp = comp_bucket[nearest_comp_fp]

    mech_infer_fp, mech_names = compute_h77v_visible_mechanics_fp(
        vis, fp, hydro_imp_fp, vent_pot_fp, upflow_scarp_fp
    )
    mech_train_fp = np.empty_like(mech_infer_fp)

    for b in range(n_buckets):
        in_b_vis = vis & (comp_bucket[comp_on_vis] == b)
        vis_minus_b = vis & (~in_b_vis)
        m_b, _ = compute_h77v_visible_mechanics_fp(
            vis_minus_b, fp, hydro_imp_fp, vent_pot_fp, upflow_scarp_fp
        )
        sel = pixel_bucket_fp == b
        mech_train_fp[:, sel] = m_b[:, sel]

    return mech_train_fp, mech_infer_fp, mech_names


def pack_combined_ram_float16(
    X_base: np.memmap,
    X_v: np.memmap,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Pack 96 base + 12 H77V = 108 static channels into disk-backed row-major float16 memmap (<50 MB RSS)."""
    n_b, n_fp = X_base.shape
    n_v = X_v.shape[0]
    total = n_b + n_v
    if cache_path is None:
        cache_path = Path("/home/user/GEMSDOE77/.cache/h77v_X16.f16.mmap")
    if cache_path.is_file() and cache_path.stat().st_size == n_fp * total * 2:
        return np.fromfile(cache_path, dtype=np.float16).reshape(n_fp, total)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    X16_mm = np.memmap(cache_path, dtype=np.float16, mode="w+", shape=(n_fp, total))
    for c in range(total):
        col = np.asarray(X_base[c, :] if c < n_b else X_v[c - n_b, :], dtype=np.float32)
        sub = col[::17]
        valid = sub[np.isfinite(sub)]
        med = float(np.median(valid)) if valid.size else 0.0
        q75, q25 = np.percentile(valid, [75.0, 25.0]) if valid.size else (1.0, 0.0)
        sc = float(max(q75 - q25, float(np.std(valid)), 1e-4))
        normed = np.clip((col - med) / sc, -1000.0, 1000.0)
        X16_mm[:, c] = normed.astype(np.float16)
        del col, normed
    X16_mm.flush()
    del X16_mm
    gc.collect()
    return np.fromfile(cache_path, dtype=np.float16).reshape(n_fp, total)
