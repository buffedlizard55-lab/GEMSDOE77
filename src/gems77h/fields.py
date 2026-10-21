"""H77H feature block: Piedmont-Bajada Alluvial-Fan Toe Pinchout, Synthetic/Antithetic Scarp Polarity & En-Echelon Sinter-Relay Bridge.

Preregistered in ``registry/preregistration_h77h.json`` before Experiment 1.
Strict invariants:
  - Zero SGMC bedrock-contact inputs.
  - Zero prior submission or registry rasters read anywhere here.
  - ``dist_known_fault_px`` in external point CSVs is explicitly forbidden and asserted unused.
  - ``lidar_scarp_features_u8.tif`` uint8 quantization ``q == 0`` (outside 75.4% 1 m LiDAR coverage)
    is strictly treated as missing and bridged seamlessly with 10 m DEM sub-cell morphology.
  - All 12 static physical channels are 100% label-free; all 3 LOCO structural relay channels
    are derived strictly from ``visible_2d`` using 4-bucket Leave-Own-Component-Out (LOCO)
    so training positives never see their own fault component.
"""
from __future__ import annotations

import csv
import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from scipy import ndimage
from scipy.spatial import cKDTree

from gems77 import features77 as F77

FORBIDDEN_POINT_COLUMNS = frozenset({"dist_known_fault_px", "known_fault", "label"})

H77H_STATIC_NAMES = [
    "h77h_piedmont_tail_slope",
    "h77h_fan_scarp_polarity",
    "h77h_antithetic_graben_step",
    "h77h_rangefront_pinchout",
    "h77h_fanhead_ksn_drop",
    "h77h_buried_bench_grav_hg",
    "h77h_lahontan_deconf_break",
    "h77h_sinter_travertine_seal",
    "h77h_probe2m_geotherm_halo",
    "h77h_transtensional_release",
    "h77h_rad_provenance_dipole",
    "h77h_composite_surface",
]

H77H_LOCO_NAMES = [
    "h77h_loco_elliptical_relay",
    "h77h_loco_piedmont_bridge",
    "h77h_loco_damage_collar",
]


def piedmont_tail_slope_anomaly(
    slope_max: np.ndarray,
    slope_p90: np.ndarray,
    aspect_coherence: np.ndarray,
    det_elev_slope: np.ndarray,
    center_deg: float = 4.5,
    width_deg: float = 4.5,
) -> np.ndarray:
    """Compute sub-cell piedmont tail-slope excess ``(slope_max - slope_p90) * aspect_coherence``
    gated by the gentle alluvial-fan / bajada slope envelope.

    On a smooth coalescing alluvial fan (1-12 deg), ``slope_max`` and ``slope_p90`` within a 100 m
    cell are nearly equal. Where a narrow Quaternary fault scarp cuts the fan, ``slope_p90``
    remains low while ``slope_max`` spikes and ``aspect_coherence`` is high along the planar scarp.
    """
    s_max = np.asarray(slope_max, dtype=np.float32)
    s_p90 = np.asarray(slope_p90, dtype=np.float32)
    coh = np.clip(np.asarray(aspect_coherence, dtype=np.float32), 0.0, 1.0)
    env_slope = np.asarray(det_elev_slope, dtype=np.float32)
    tail_excess = np.maximum(0.0, s_max - s_p90)
    rel_tail = tail_excess / (s_p90 + np.float32(2.5))
    piedmont_env = np.exp(-0.5 * ((env_slope - np.float32(center_deg)) / np.float32(width_deg)) ** 2)
    return (tail_excess * (np.float32(0.5) + rel_tail) * (coh ** 1.5) * piedmont_env).astype(np.float32)


def lahontan_shoreline_suppressor(
    dem_mean: np.ndarray,
    highstands_m: tuple[float, ...] = (1332.0, 1463.0, 1552.0),
    elev_sigma_m: float = 18.0,
) -> np.ndarray:
    """Suppress horizontal Pleistocene Lake Lahontan (~1332 m) and Lake Bonneville (~1463, 1552 m)
    wave-cut shoreline berms that follow flat equipotential contours.

    Returns a multiplicative deconfounding factor in ``[0.15, 1.0]`` that is low on flat
    equipotential highstand contours and ``1.0`` on inclined tectonic scarps or away from
    pluvial highstand elevations.
    """
    z = np.asarray(dem_mean, dtype=np.float32)
    gy = ndimage.gaussian_filter(z, sigma=1.5, order=(1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(z, sigma=1.5, order=(0, 1), mode="nearest")
    z_smooth = ndimage.gaussian_filter(z, sigma=2.0, mode="nearest")
    z_var = np.maximum(0.0, ndimage.gaussian_filter((z - z_smooth) ** 2, sigma=2.0, mode="nearest"))
    z_relief = np.float32(2.0) * np.hypot(gx, gy) + np.sqrt(z_var)
    contour_flatness = np.exp(-z_relief / np.float32(6.0))
    band_hit = np.zeros_like(z, dtype=np.float32)
    for z0 in highstands_m:
        band_hit = np.maximum(
            band_hit,
            np.exp(-0.5 * ((z - np.float32(z0)) / np.float32(elev_sigma_m)) ** 2),
        )
    penalty = np.float32(0.85) * band_hit * contour_flatness
    return np.clip(1.0 - penalty, 0.15, 1.0).astype(np.float32)


def scarp_polarity_ratio(
    downface_max: np.ndarray,
    upface_max: np.ndarray,
    cross_max: np.ndarray,
    coh100: np.ndarray,
    lidar_valid: np.ndarray,
    curv_prof_absmax: np.ndarray,
    aspect_coherence: np.ndarray,
    slope_std: np.ndarray,
    footprint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (1) synthetic+antithetic scarp-to-wash polarity ratio and (2) antithetic graben step.

    Strictly respects ``lidar_valid > 0`` (treating uint8 ``0`` outside the 75.4% 1 m LiDAR footprint
    as missing) and bridges the remaining 24.6% of Nevada via rank-matched 10 m DEM sub-cell
    profile curvature and aspect coherence.
    """
    foot = np.asarray(footprint, dtype=bool)
    has_lidar = foot & (np.asarray(lidar_valid) > 0)
    df = np.asarray(downface_max, dtype=np.float32)
    uf = np.asarray(upface_max, dtype=np.float32)
    cr = np.asarray(cross_max, dtype=np.float32)
    c100 = np.asarray(coh100, dtype=np.float32) / np.float32(255.0)

    raw_pol_lidar = np.log1p((df + np.float32(1.25) * uf) / (cr + np.float32(8.0))) * (np.float32(0.25) + c100)
    raw_pol_dem10 = (
        np.asarray(curv_prof_absmax, dtype=np.float32)
        * (np.clip(np.asarray(aspect_coherence, dtype=np.float32), 0.0, 1.0) ** 1.5)
        / (np.asarray(slope_std, dtype=np.float32) + np.float32(1.0))
    )

    rk_dem10 = F77.rank01(np.where(foot, raw_pol_dem10, np.float32(np.nan)), foot)
    if has_lidar.any():
        rk_lidar = F77.rank01(np.where(has_lidar, raw_pol_lidar, np.float32(np.nan)), has_lidar)
        pol = np.where(has_lidar, np.float32(0.70) * rk_lidar + np.float32(0.30) * rk_dem10, rk_dem10)
    else:
        pol = rk_dem10
    pol = F77.rank01(np.where(foot, pol, np.float32(np.nan)), foot)

    # Antithetic hanging-wall graben step fraction (upface vs total scarp face)
    anti_raw = (uf / (df + uf + np.float32(5.0))) * np.log1p(uf + np.float32(0.35) * df)
    if has_lidar.any():
        rk_anti_l = F77.rank01(np.where(has_lidar, anti_raw, np.float32(np.nan)), has_lidar)
        anti = np.where(has_lidar, np.float32(0.75) * rk_anti_l + np.float32(0.25) * rk_dem10, rk_dem10)
    else:
        anti = rk_dem10
    anti = F77.rank01(np.where(foot, anti, np.float32(np.nan)), foot)
    return np.where(foot, pol, 0.0).astype(np.float32), np.where(foot, anti, 0.0).astype(np.float32)


def _extract_tips_and_components(mask_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify degree-1 fault tips and connected component labels on a binary fault mask."""
    m = np.asarray(mask_2d, dtype=bool)
    comp_lbl, _ = ndimage.label(m, structure=np.ones((3, 3), dtype=np.int8))
    if not m.any():
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.int32), comp_lbl
    nb_count = ndimage.convolve(m.astype(np.int8), np.ones((3, 3), dtype=np.int8), mode="constant", cval=0)
    # Degree-1 endpoints have 2 or 3 8-connected neighbors (including self)
    tips_mask = m & (nb_count >= 2) & (nb_count <= 3)
    rc = np.argwhere(tips_mask).astype(np.int32)
    if rc.shape[0] > 1:
        # Decimate tips closer than 2.5 px so a thick endpoint doesn't spawn duplicate tip points
        tree = cKDTree(rc.astype(np.float32))
        keep = np.ones(rc.shape[0], dtype=bool)
        for i in range(rc.shape[0]):
            if not keep[i]:
                continue
            nbrs = tree.query_ball_point(rc[i], r=2.5)
            for j in nbrs:
                if j > i:
                    keep[j] = False
        rc = rc[keep]
    tip_comp = comp_lbl[rc[:, 0], rc[:, 1]] if rc.shape[0] > 0 else np.zeros((0,), dtype=np.int32)
    return rc, tip_comp, comp_lbl


def elliptical_tip_relay_bridge(
    visible_2d: np.ndarray,
    footprint: np.ndarray,
    min_gap_px: float = 4.0,
    max_gap_px: float = 36.0,
    decay_px: float = 2.2,
    length_scale_px: float = 20.0,
) -> np.ndarray:
    """Compute en-echelon tip-to-tip elliptical relay bridge potential strictly from ``visible_2d``.

    For every pair of degree-1 tips ``(t_i, t_j)`` separated by ``L_ij in [min_gap_px, max_gap_px]``
    (excluding same-component short chord shortcuts), evaluates the confocal elliptical bridge:
        ``exp(-(d(p, t_i) + d(p, t_j) - L_ij) / decay_px) * exp(-L_ij / length_scale_px)``
    within the local bounding box of ``(t_i, t_j)``.
    """
    foot = np.asarray(footprint, dtype=bool)
    h, w = foot.shape
    out = np.zeros((h, w), dtype=np.float32)
    rc, tip_comp, _ = _extract_tips_and_components(visible_2d)
    if rc.shape[0] < 2:
        return out

    pts = rc.astype(np.float32)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=float(max_gap_px), output_type="ndarray")
    if pairs.size == 0:
        return out

    pad = int(np.ceil(3.5 * decay_px)) + 2
    for idx_i, idx_j in pairs:
        ri, ci = int(rc[idx_i, 0]), int(rc[idx_i, 1])
        rj, cj = int(rc[idx_j, 0]), int(rc[idx_j, 1])
        L = float(np.hypot(ri - rj, ci - cj))
        if L < min_gap_px:
            continue
        # If both tips belong to the same connected component, only allow long step-over gaps >= 10 px
        if tip_comp[idx_i] == tip_comp[idx_j] and L < 10.0:
            continue
        r0 = max(0, min(ri, rj) - pad)
        r1 = min(h, max(ri, rj) + pad + 1)
        c0 = max(0, min(ci, cj) - pad)
        c1 = min(w, max(ci, cj) + pad + 1)
        yy, xx = np.mgrid[r0:r1, c0:c1].astype(np.float32)
        d1 = np.hypot(yy - np.float32(ri), xx - np.float32(ci))
        d2 = np.hypot(yy - np.float32(rj), xx - np.float32(cj))
        slack = np.maximum(0.0, d1 + d2 - np.float32(L))
        val = (
            np.exp(-slack / np.float32(decay_px))
            * np.exp(-np.float32(L) / np.float32(length_scale_px))
        ).astype(np.float32)
        sub = out[r0:r1, c0:c1]
        np.maximum(sub, val, out=sub)

    out[~foot] = 0.0
    return out


def compute_h77h_loco_channels(
    visible_2d: np.ndarray,
    footprint: np.ndarray,
    lahontan_break: np.ndarray,
    fan_polarity: np.ndarray,
    n_buckets: int = 4,
) -> dict[str, np.ndarray]:
    """Compute the 3 fold-strict Leave-Own-Component-Out (LOCO) structural relay channels.

    Connected components of ``visible_2d`` are partitioned deterministically into ``n_buckets``
    groups (``comp_id % n_buckets``). Every pixel whose nearest visible fault component belongs
    to bucket ``b`` receives its relay bridge, piedmont continuation, and damage-collar distance
    potentials computed strictly from ``visible_other = visible_2d & (bucket != b)``, guaranteeing
    that no training positive pixel ever sees ``d == 0`` to its own fault component.
    """
    foot = np.asarray(footprint, dtype=bool)
    vis = np.asarray(visible_2d, dtype=bool) & foot
    h, w = foot.shape
    if not vis.any():
        z = np.zeros((h, w), dtype=np.float32)
        return {nm: z.copy() for nm in H77H_LOCO_NAMES}

    comp_lbl, n_comp = ndimage.label(vis, structure=np.ones((3, 3), dtype=np.int8))
    # Map every pixel in the grid to the component ID of its nearest visible fault pixel
    _, (nr, nc) = ndimage.distance_transform_edt(~vis, return_indices=True)
    nearest_comp = comp_lbl[nr, nc]
    nearest_bucket = nearest_comp % n_buckets

    relay_full = np.zeros((h, w), dtype=np.float32)
    d_tip_other = np.full((h, w), 250.0, dtype=np.float32)
    d_comp_other = np.full((h, w), 250.0, dtype=np.float32)

    for b in range(n_buckets):
        vis_other = vis & ((comp_lbl % n_buckets) != b)
        if not vis_other.any():
            continue
        assign_mask = foot & (nearest_bucket == b)
        if not assign_mask.any():
            continue
        rel_b = elliptical_tip_relay_bridge(vis_other, foot)
        relay_full[assign_mask] = rel_b[assign_mask]

        d_other = ndimage.distance_transform_edt(~vis_other).astype(np.float32)
        d_comp_other[assign_mask] = d_other[assign_mask]

        rc_tips, _, _ = _extract_tips_and_components(vis_other)
        if rc_tips.shape[0] > 0:
            tip_mask = np.zeros((h, w), dtype=bool)
            tip_mask[rc_tips[:, 0], rc_tips[:, 1]] = True
            d_t = ndimage.distance_transform_edt(~tip_mask).astype(np.float32)
            d_tip_other[assign_mask] = d_t[assign_mask]

    rk = lambda a: F77.rank01(np.where(foot, a, np.float32(np.nan)), foot)  # noqa: E731
    geom_gate = np.float32(0.50) * np.asarray(lahontan_break, dtype=np.float32) + np.float32(0.50) * np.asarray(
        fan_polarity, dtype=np.float32
    )
    piedmont_bridge_raw = (
        (np.float32(0.60) * np.exp(-d_tip_other / np.float32(14.0)) + np.float32(0.40) * relay_full)
        * (np.float32(0.20) + np.float32(0.80) * geom_gate)
    )
    collar_raw = np.exp(-0.5 * ((d_comp_other - np.float32(5.0)) / np.float32(3.5)) ** 2) * (
        d_comp_other > np.float32(2.0)
    )

    c_relay = np.where(foot, rk(relay_full), 0.0).astype(np.float32)
    c_bridge = np.where(foot, rk(piedmont_bridge_raw), 0.0).astype(np.float32)
    c_collar = np.where(foot, rk(collar_raw), 0.0).astype(np.float32)
    return {
        "h77h_loco_elliptical_relay": c_relay,
        "h77h_loco_piedmont_bridge": c_bridge,
        "h77h_loco_damage_collar": c_collar,
    }


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


def _build_sinter_travertine_seal(ext_dir: Path, transform: Any, shape: tuple[int, int], foot: np.ndarray) -> np.ndarray:
    """Kernel density of fossil silica-sinter (1.5x) and carbonate-travertine (1.0x) mounds."""
    imp = np.zeros(shape, dtype=np.float32)
    zpath = ext_dir / "paleo_geothermal_regional.zip"
    if zpath.is_file():
        lons, lats, wts = [], [], []
        with zipfile.ZipFile(zpath) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            reader = csv.DictReader(io.TextIOWrapper(zf.open(csv_name), encoding="latin1"))
            for col in reader.fieldnames or ():
                assert col not in FORBIDDEN_POINT_COLUMNS, f"Forbidden column: {col}"
            for row in reader:
                try:
                    lon = float(row["Long_deg_NAD83"])
                    lat = float(row["Lat_deg_NAD83"])
                except (KeyError, ValueError):
                    continue
                dtype = (row.get("Deposit_type") or "").strip().lower()
                w = 1.5 if "sinter" in dtype else 1.0
                lons.append(lon)
                lats.append(lat)
                wts.append(w)
        if lons:
            ux, uy = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
            rr, cc, ok = _utm_to_rc(ux, uy, transform, shape)
            np.add.at(imp, (rr, cc), np.asarray(wts, dtype=np.float32)[ok])
    raw = np.float32(0.65) * ndimage.gaussian_filter(imp, sigma=4.0, mode="constant") + np.float32(
        0.35
    ) * ndimage.gaussian_filter(imp, sigma=10.0, mode="constant")
    rk = F77.rank01(np.where(foot, raw, np.float32(np.nan)), foot)
    return np.where(foot, rk, 0.0).astype(np.float32)


def _build_probe2m_geotherm_halo(ext_dir: Path, transform: Any, shape: tuple[int, int], foot: np.ndarray) -> np.ndarray:
    """Fuse INGENIOUS 2-meter temperature probe F2mDAB/T2m anomalies with chemical geothermometer disequilibrium."""
    imp_probe = np.zeros(shape, dtype=np.float32)
    zpath = ext_dir / "2m_temperature_probe_INGENIOUS_regional_data.zip"
    if zpath.is_file():
        with zipfile.ZipFile(zpath) as zf:
            dbf_name = next(n for n in zf.namelist() if n.lower().endswith(".dbf"))
            buf = zf.read(dbf_name)
        numrec, lenhdr, lenrec = struct.unpack("<4xIHH20x", buf[:32])
        fields: list[tuple[str, int, int]] = []
        pos, offset = 32, 1
        while buf[pos] != 0x0D:
            fname = buf[pos : pos + 11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            assert fname not in FORBIDDEN_POINT_COLUMNS, f"Forbidden column: {fname}"
            flen = buf[pos + 16]
            fields.append((fname, offset, flen))
            offset += flen
            pos += 32
        fmap = {name: (off, flen) for name, off, flen in fields}
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
            if 25.0 <= lat <= 50.0 and -130.0 <= lon <= -100.0:
                val = max(0.0, dab) + 0.25 * max(0.0, t2m - 20.0)
                if val > 0:
                    lats.append(lat)
                    lons.append(lon)
                    dabs.append(val)
        if lons:
            ux, uy = rio_transform("EPSG:4269", "EPSG:32611", lons, lats)
            rr, cc, ok = _utm_to_rc(ux, uy, transform, shape)
            np.maximum.at(imp_probe, (rr, cc), np.asarray(dabs, dtype=np.float32)[ok])

    imp_well = np.zeros(shape, dtype=np.float32)
    wpath = ext_dir / "gdr_wellspring_in_footprint.csv"
    if wpath.is_file():
        allowed_cols = ("row", "col", "temp_c", "geothermquartz_c", "geothermchalc_c", "geothermcat_c")
        with wpath.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(allowed_cols).issubset(set(reader.fieldnames or ()))
            for row in reader:
                try:
                    r = int(float(row["row"]))
                    c = int(float(row["col"]))
                except (KeyError, ValueError):
                    continue
                if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                    continue
                ts = float(row["temp_c"]) if row.get("temp_c") else 15.0
                tq = float(row["geothermquartz_c"]) if row.get("geothermquartz_c") else 0.0
                tc = float(row["geothermchalc_c"]) if row.get("geothermchalc_c") else 0.0
                tcat = float(row["geothermcat_c"]) if row.get("geothermcat_c") else 0.0
                diseq = max(0.0, max(tq, tc, tcat) - max(ts, 15.0)) + max(0.0, ts - 25.0)
                if diseq > imp_well[r, c]:
                    imp_well[r, c] = np.float32(np.log1p(diseq))

    p_sm = np.float32(0.60) * ndimage.gaussian_filter(imp_probe, sigma=4.0, mode="constant") + np.float32(
        0.40
    ) * ndimage.gaussian_filter(imp_probe, sigma=10.0, mode="constant")
    w_sm = np.float32(0.60) * ndimage.gaussian_filter(imp_well, sigma=4.0, mode="constant") + np.float32(
        0.40
    ) * ndimage.gaussian_filter(imp_well, sigma=10.0, mode="constant")
    rk_p = F77.rank01(np.where(foot, p_sm, np.float32(np.nan)), foot)
    rk_w = F77.rank01(np.where(foot, w_sm, np.float32(np.nan)), foot)
    comb = F77.rank01(np.where(foot, np.float32(0.55) * rk_p + np.float32(0.45) * rk_w, np.float32(np.nan)), foot)
    return np.where(foot, comb, 0.0).astype(np.float32)


def build_h77h_static_fields(
    raw_dir: Path,
    ext_dir: Path,
    footprint: np.ndarray,
    transform: Any,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Compute all 12 preregistered zero-catalogue H77H static channels and summary diagnostics."""
    foot = np.asarray(footprint, dtype=bool)
    shape = foot.shape
    rk = lambda a: np.where(foot, F77.rank01(np.where(foot, a, np.float32(np.nan)), foot), 0.0).astype(np.float32)  # noqa: E731

    # Read needed bands from training_features.tif
    with rasterio.open(raw_dir / "training_features.tif") as ds:
        geod_2ndinv = F77.fill_median(F77.clean(ds.read(4).astype(np.float32)), foot)
        geod_shear = F77.fill_median(F77.clean(ds.read(7).astype(np.float32)), foot)
        geod_dilat = F77.fill_median(F77.clean(ds.read(8).astype(np.float32)), foot)
        depth_base = F77.fill_median(F77.clean(ds.read(15).astype(np.float32)), foot)
        grav_hg_signed = F77.fill_median(F77.clean(ds.read(18).astype(np.float32)), foot)
        det_elev_slope = F77.fill_median(F77.clean(ds.read(19).astype(np.float32)), foot)

    # Read 10 m DEM sub-cell statistics from topo_10m_derived_100m_u8.tif
    m_topo = json.loads((ext_dir / "topo_10m_derived_100m_u8.manifest.json").read_text())
    topo_bands = {}
    with rasterio.open(ext_dir / "topo_10m_derived_100m_u8.tif") as ds:
        for b in m_topo["bands"]:
            q = ds.read(int(b["index"])).astype(np.float32)
            lo, hi = float(b["lo"]), float(b["hi"])
            val = lo + np.clip(q - 1.0, 0.0, 254.0) / 254.0 * (hi - lo)
            val[q == 0] = np.nan
            topo_bands[b["band_name"]] = F77.fill_median(val, foot)

    # Read 1 m LiDAR scarp features (respecting valid > 0)
    with rasterio.open(ext_dir / "lidar_scarp_features_u8.tif") as ds:
        lidar_downface = ds.read(6).astype(np.float32)
        lidar_upface = ds.read(7).astype(np.float32)
        lidar_cross = ds.read(8).astype(np.float32)
        lidar_coh100 = ds.read(10).astype(np.float32)
        lidar_valid = ds.read(12)

    # Read 10 m DEM chi/ksn features
    with rasterio.open(ext_dir / "dem10_chi_100m_i16.tif") as ds:
        ksn_raw = ds.read(3).astype(np.float32)
        ksn_raw[ksn_raw <= -32760] = np.nan
        chi_ksn = F77.fill_median(ksn_raw / 100.0, foot)
        area_raw = ds.read(4).astype(np.float32)
        area_raw[area_raw <= -32760] = np.nan
        chi_log10_area = F77.fill_median(area_raw / 1000.0, foot)

    # Read Th/K ratio from geodawn_extensions_u8.tif (Band 1)
    with rasterio.open(ext_dir / "geodawn_extensions_u8.tif") as ds:
        thk_q = ds.read(1).astype(np.float32)
        thk_q[thk_q == 0] = np.nan
        th_over_k = F77.fill_median(thk_q, foot)

    piedmont_env = np.exp(-0.5 * ((det_elev_slope - np.float32(4.5)) / np.float32(4.5)) ** 2)

    # 1. h77h_piedmont_tail_slope
    raw_tail = piedmont_tail_slope_anomaly(
        topo_bands["slope_max"],
        topo_bands["slope_p90"],
        topo_bands["aspect_coherence"],
        det_elev_slope,
    )
    c1_tail = rk(raw_tail)

    # 2 & 3. h77h_fan_scarp_polarity and h77h_antithetic_graben_step
    c2_pol, c3_anti = scarp_polarity_ratio(
        lidar_downface,
        lidar_upface,
        lidar_cross,
        lidar_coh100,
        lidar_valid,
        topo_bands["curv_prof_absmax"],
        topo_bands["aspect_coherence"],
        topo_bands["slope_std"],
        foot,
    )

    # 4. h77h_rangefront_pinchout
    gy_sf = ndimage.gaussian_filter(topo_bands["steep_frac"], sigma=2.0, order=(1, 0), mode="nearest")
    gx_sf = ndimage.gaussian_filter(topo_bands["steep_frac"], sigma=2.0, order=(0, 1), mode="nearest")
    grad_steep = np.hypot(gx_sf, gy_sf)
    c4_pinch = rk(grad_steep * topo_bands["aspect_coherence"] * np.log1p(np.maximum(topo_bands["relief_local"], 0.0)))

    # 5. h77h_fanhead_ksn_drop
    ksn_drop = np.maximum(
        0.0,
        ndimage.gaussian_filter(chi_ksn, sigma=1.5, mode="nearest")
        - ndimage.gaussian_filter(chi_ksn, sigma=5.0, mode="nearest"),
    )
    c5_ksn = rk(ksn_drop * np.log1p(np.maximum(chi_log10_area, 0.0)) * (np.float32(0.25) + piedmont_env))

    # 6. h77h_buried_bench_grav_hg
    gy_db = ndimage.gaussian_filter(depth_base, sigma=2.5, order=(1, 0), mode="nearest")
    gx_db = ndimage.gaussian_filter(depth_base, sigma=2.5, order=(0, 1), mode="nearest")
    grad_db = np.hypot(gx_db, gy_db)
    c6_bench = rk(rk(np.abs(grav_hg_signed)) * rk(grad_db) * (np.float32(0.30) + piedmont_env))

    # 7. h77h_lahontan_deconf_break
    deconf = lahontan_shoreline_suppressor(topo_bands["dem_mean"])
    c7_deconf = rk(raw_tail * deconf)

    # 8. h77h_sinter_travertine_seal
    c8_sinter = _build_sinter_travertine_seal(ext_dir, transform, shape, foot)

    # 9. h77h_probe2m_geotherm_halo
    c9_probe = _build_probe2m_geotherm_halo(ext_dir, transform, shape, foot)

    # 10. h77h_transtensional_release
    r_dil = rk(np.maximum(geod_dilat, 0.0))
    r_shr = rk( np.maximum(geod_shear, 0.0))
    r_inv = rk(np.maximum(geod_2ndinv, 0.0))
    c10_trans = rk(np.cbrt(np.clip(r_dil * r_shr * r_inv, 0.0, 1.0)))

    # 11. h77h_rad_provenance_dipole
    gy_thk = ndimage.gaussian_filter(th_over_k, sigma=2.0, order=(1, 0), mode="nearest")
    gx_thk = ndimage.gaussian_filter(th_over_k, sigma=2.0, order=(0, 1), mode="nearest")
    c11_rad = rk(np.hypot(gx_thk, gy_thk) * (np.float32(0.25) + piedmont_env))

    # 12. h77h_composite_surface (label-free physical composite)
    hydro_term = np.float32(0.35) + np.float32(0.65) * (
        np.float32(0.40) * c8_sinter + np.float32(0.35) * c9_probe + np.float32(0.25) * c10_trans
    )
    geom_term = (
        np.float32(0.32) * c7_deconf
        + np.float32(0.28) * c2_pol
        + np.float32(0.18) * c4_pinch
        + np.float32(0.12) * c3_anti
        + np.float32(0.10) * c5_ksn
    )
    sub_term = np.float32(0.35) + np.float32(0.65) * (np.float32(0.65) * c6_bench + np.float32(0.35) * c11_rad)
    c12_comp = rk(geom_term * np.sqrt(np.clip(sub_term * hydro_term, 1e-6, 1.0)))

    out = {
        "h77h_piedmont_tail_slope": c1_tail,
        "h77h_fan_scarp_polarity": c2_pol,
        "h77h_antithetic_graben_step": c3_anti,
        "h77h_rangefront_pinchout": c4_pinch,
        "h77h_fanhead_ksn_drop": c5_ksn,
        "h77h_buried_bench_grav_hg": c6_bench,
        "h77h_lahontan_deconf_break": c7_deconf,
        "h77h_sinter_travertine_seal": c8_sinter,
        "h77h_probe2m_geotherm_halo": c9_probe,
        "h77h_transtensional_release": c10_trans,
        "h77h_rad_provenance_dipole": c11_rad,
        "h77h_composite_surface": c12_comp,
    }
    diag = {
        nm: {
            "in_footprint_mean": float(arr[foot].mean()),
            "in_footprint_std": float(arr[foot].std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "nan_count": int(np.isnan(arr).sum()),
        }
        for nm, arr in out.items()
    }
    return out, diag
