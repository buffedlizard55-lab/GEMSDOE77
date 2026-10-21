"""H84-RAMP: Deconfounded Two-Host En-Echelon Relay-Ramp & Geodetic-Geophysical Stepover Transfer.

Reuses the pinned shared template ``vendor/gems57/relay_bend_anatomy.py``
(``two_host_frame`` exact bitplane second-host Euclidean distance transform and
``host_bend_fields`` multi-scale spin-2 structure tensor turning) and couples
them with transtensional geodetic strain (``geod_shearrate``, ``geod_dilaterate``)
and potential-field basement-step transfer (``depth_to_base_surf``, ``iso_grav_anom_hg``,
``tmi_hg``, ``det_elev_slope``).

Eliminates the Relay-Stepover Self-Collapse Tautology (``IR-H84-01``: on visible
training positives ``x in V``, ``d1(x, V) == 0`` forces ``relay_ratio == 0``,
``relay_facing == 0``, and ``damage_collar == 0``) via 5-bucket Buffered-Component
Leave-Bucket-Out (``B-LBCO``) evaluated on ``V \\ V_b``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from gems57.network import STRUCT3
from gems57.relay_bend_anatomy import host_bend_fields, two_host_frame

H84_NAMES: tuple[str, ...] = (
    "h84_relay_facing_corridor",
    "h84_relay_medial_bridge",
    "h84_two_host_coplanar_sync",
    "h84_multiscale_bend_splay",
    "h84_damage_collar_kernel",
    "h84_ramp_geod_transtension",
    "h84_ramp_basement_step_bridge",
    "h84_composite_physics",
)


def robust_z(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median / IQR robust z-score clipped to [-5, 5], zero outside valid."""
    a = np.asarray(arr, dtype=np.float32)
    m = np.asarray(valid, dtype=bool) & np.isfinite(a)
    if not m.any():
        return np.zeros_like(a, dtype=np.float32)
    vals = a[m]
    med = float(np.median(vals))
    q75, q25 = np.percentile(vals, [75.0, 25.0])
    scale = max(float(q75 - q25) / 1.349, 1e-6)
    z = np.clip((a - med) / scale, -5.0, 5.0).astype(np.float32)
    return np.where(m, z, 0.0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -12.0, 12.0)))).astype(np.float32)


def damage_collar_profile(d1: np.ndarray) -> np.ndarray:
    """Kernel-scale damage collar: 0 on trace (d<=1), peak at 3 px (300 m), 0 by 15 px (1.5 km)."""
    d = np.asarray(d1, dtype=np.float32)
    out = np.zeros_like(d, dtype=np.float32)
    m1 = (d > 1.0) & (d <= 3.0)
    out[m1] = np.sin(0.5 * np.pi * (d[m1] - 1.0) / 2.0) ** 2
    m2 = (d > 3.0) & (d <= 15.0)
    out[m2] = np.cos(0.5 * np.pi * (d[m2] - 3.0) / 12.0) ** 2
    return out.astype(np.float32)


def prepare_static_geophys_context(
    geod_shear: np.ndarray,
    geod_dilate: np.ndarray,
    depth_base: np.ndarray,
    grav_hg: np.ndarray,
    tmi_hg: np.ndarray,
    elev_slope: np.ndarray,
    footprint: np.ndarray,
) -> dict[str, np.ndarray]:
    """Precompute catalogue-independent geodetic and basement-step transfer fields in [0, 1]."""
    fp = np.asarray(footprint, dtype=bool)
    z_shear = robust_z(geod_shear, fp)
    z_dil = robust_z(np.abs(geod_dilate), fp)
    geod_trans = _sigmoid(0.70 * z_shear + 0.50 * z_dil) * fp

    db_smooth = ndi.gaussian_filter(np.where(fp, depth_base, 0.0).astype(np.float32), sigma=1.5)
    gy, gx = np.gradient(db_smooth)
    grad_db = np.hypot(gy, gx).astype(np.float32)
    z_db = robust_z( np.log1p(grad_db), fp)
    z_grav = robust_z(np.log1p(np.maximum(grav_hg, 0.0)), fp)
    z_mag = robust_z(np.log1p(np.maximum(tmi_hg, 0.0)), fp)
    z_slope = robust_z(elev_slope, fp)
    # Piedmont alluvial cover gate: favors low-to-moderate slope where scarps are buried
    piedmont_gate = _sigmoid(0.8 - 0.5 * z_slope)
    base_bridge = _sigmoid(0.55 * z_db + 0.45 * z_grav + 0.25 * z_mag) * piedmont_gate * fp
    return {
        "geod_trans": geod_trans.astype(np.float32),
        "base_bridge": base_bridge.astype(np.float32),
    }


def extract_h84_from_frame(
    rows: np.ndarray,
    cols: np.ndarray,
    d1: np.ndarray,
    ay1: np.ndarray,
    ax1: np.ndarray,
    c1: np.ndarray,
    d2: np.ndarray,
    ay2: np.ndarray,
    ax2: np.ndarray,
    c2: np.ndarray,
    comp_len: np.ndarray,
    strike_resolved: np.ndarray,
    bend_turn: np.ndarray,
    geod_trans_at_pts: np.ndarray,
    base_bridge_at_pts: np.ndarray,
) -> np.ndarray:
    """Evaluate the 8 H84-RAMP channels at 1-D pixel coordinates (rows, cols)."""
    d1 = np.asarray(d1, dtype=np.float32)
    d2 = np.asarray(d2, dtype=np.float32)
    d_sum = np.maximum(d1 + d2, 1e-6)

    dy1 = (rows - ay1).astype(np.float32)
    dx1 = (cols - ax1).astype(np.float32)
    dy2 = (rows - ay2).astype(np.float32)
    dx2 = (cols - ax2).astype(np.float32)

    # When d1 == 0 (naive evaluation on a visible pixel), dy1 == dx1 == 0 -> facing = 0
    denom = np.maximum(d1 * d2, 1e-6)
    raw_facing = np.where(d1 > 1e-5, np.clip(-(dy1 * dy2 + dx1 * dx2) / denom, -1.0, 1.0), 0.0)
    corridor_decay = np.exp(-d_sum / 30.0).astype(np.float32)
    facing_corridor = (raw_facing * corridor_decay).astype(np.float32)
    pos_facing = np.maximum(raw_facing, 0.0)

    relay_ratio = np.where(d1 > 1e-5, np.clip(d1 / d_sum, 0.0, 0.5), 0.0).astype(np.float32)
    medial_sym = (4.0 * relay_ratio * (1.0 - relay_ratio)).astype(np.float32)
    medial_bridge = (medial_sym * pos_facing * np.exp(-d_sum / 25.0)).astype(np.float32)

    s1 = strike_resolved[ay1, ax1]
    s2 = strike_resolved[ay2, ax2]
    cos2_diff = np.cos(np.radians(2.0 * (s1 - s2))).astype(np.float32)
    l1 = np.log1p(comp_len[np.clip(c1, 0, len(comp_len) - 1)]).astype(np.float32)
    l2 = np.log1p(comp_len[np.clip(c2, 0, len(comp_len) - 1)]).astype(np.float32)
    len_weight = np.clip(np.sqrt(l1 * l2) / 8.0, 0.0, 1.0)
    coplanar_sync = (cos2_diff * len_weight * np.exp(-d_sum / 35.0)).astype(np.float32)

    turn1 = bend_turn[ay1, ax1].astype(np.float32)
    bend_splay = np.where(d1 > 1e-5, turn1 * np.exp(-d1 / 12.0), 0.0).astype(np.float32)

    collar = damage_collar_profile(d1)

    active_step = np.clip(0.65 * pos_facing * np.exp(-d_sum / 32.0) + 0.35 * collar, 0.0, 1.0)
    geod_ramp = (geod_trans_at_pts * active_step).astype(np.float32)
    base_ramp = (base_bridge_at_pts * active_step).astype(np.float32)

    # Zero-training physical composite in [0, 1], gated off the <=2 px catalogue flank
    off_flank = (d1 > 2.0).astype(np.float32)
    raw_comp = (
        0.30 * collar
        + 0.22 * medial_bridge
        + 0.16 * np.clip(bend_splay / 1.5, 0.0, 1.0)
        + 0.16 * geod_ramp
        + 0.16 * base_ramp
    ) * off_flank
    comp_phys = np.clip(raw_comp, 0.0, 1.0).astype(np.float32)

    return np.column_stack(
        [
            facing_corridor,
            medial_bridge,
            coplanar_sync,
            bend_splay,
            collar,
            geod_ramp,
            base_ramp,
            comp_phys,
        ]
    ).astype(np.float32)


def build_h84_surface_fields(
    visible: np.ndarray,
    footprint: np.ndarray,
    static_ctx: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build full-footprint H84 feature matrix (N_fp, 8) and 2D composite surface from visible faults."""
    vis = np.asarray(visible, dtype=bool) & np.asarray(footprint, dtype=bool)
    fp = np.asarray(footprint, dtype=bool)
    th = two_host_frame(vis)
    hb = host_bend_fields(vis, fine_px=3.0, coarse_px=9.0)
    s_fine = hb["strike_fine"]
    s_coarse = hb["strike_coarse"]
    strike_res = np.where(np.isfinite(s_fine), s_fine, np.where(np.isfinite(s_coarse), s_coarse, 0.0)).astype(
        np.float32
    )
    bend_turn = np.asarray(hb["turn"], dtype=np.float32)

    fy, fx = np.nonzero(fp)
    mat_fp = extract_h84_from_frame(
        rows=fy,
        cols=fx,
        d1=th["d1"][fy, fx],
        ay1=th["iy1"][fy, fx],
        ax1=th["ix1"][fy, fx],
        c1=th["c1"][fy, fx],
        d2=th["d2"][fy, fx],
        ay2=th["iy2"][fy, fx],
        ax2=th["ix2"][fy, fx],
        c2=th["c2"][fy, fx],
        comp_len=th["comp_len"],
        strike_resolved=strike_res,
        bend_turn=bend_turn,
        geod_trans_at_pts=static_ctx["geod_trans"][fy, fx],
        base_bridge_at_pts=static_ctx["base_bridge"][fy, fx],
    )
    comp_2d = np.zeros(fp.shape, dtype=np.float32)
    comp_2d[fy, fx] = mat_fp[:, -1]
    aux = {
        "strike_resolved": strike_res,
        "bend_turn": bend_turn,
        "d1": th["d1"],
        "d2": th["d2"],
        "iy1": th["iy1"],
        "ix1": th["ix1"],
    }
    return mat_fp, comp_2d, aux


def _query_bitplane_two_host_kdtree(
    sub_vis: np.ndarray,
    qy: np.ndarray,
    qx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact bitplane cKDTree two-host query on subset mask ``sub_vis`` at points ``(qy, qx)``."""
    comp, ncomp = ndi.label(sub_vis, structure=STRUCT3)
    if ncomp < 2:
        raise ValueError("sub_vis requires at least 2 connected components")
    comp_len = np.bincount(comp.ravel(), minlength=ncomp + 1).astype(np.float32)
    vy, vx = np.nonzero(sub_vis)
    vc = comp[vy, vx]
    pts = np.column_stack([vy.astype(np.float64), vx.astype(np.float64)])
    qpts = np.column_stack([qy.astype(np.float64), qx.astype(np.float64)])

    t1 = cKDTree(pts)
    d1, i1 = t1.query(qpts, k=1)
    d1 = d1.astype(np.float32)
    ay1 = vy[i1].astype(np.int32)
    ax1 = vx[i1].astype(np.int32)
    c1 = vc[i1].astype(np.int32)

    nbits = int(np.ceil(np.log2(ncomp + 1)))
    d2 = np.full(qy.size, np.inf, dtype=np.float32)
    ay2 = np.zeros(qy.size, dtype=np.int32)
    ax2 = np.zeros(qy.size, dtype=np.int32)
    c2 = np.zeros(qy.size, dtype=np.int32)
    for b in range(nbits):
        bit_c1 = (c1 >> b) & 1
        for val in (0, 1):
            m = ((vc >> b) & 1) == val
            if not m.any():
                continue
            tb = cKDTree(pts[m])
            vym, vxm, vcm = vy[m], vx[m], vc[m]
            use_q = np.flatnonzero(bit_c1 != val)
            if not use_q.size:
                continue
            db, ib = tb.query(qpts[use_q], k=1)
            better = db < d2[use_q]
            u = use_q[better]
            ib_b = ib[better]
            d2[u] = db[better].astype(np.float32)
            ay2[u] = vym[ib_b].astype(np.int32)
            ax2[u] = vxm[ib_b].astype(np.int32)
            c2[u] = vcm[ib_b].astype(np.int32)
    return d1, ay1, ax1, c1, d2, ay2, ax2, c2, comp_len


def compute_blbco_h84_rows(
    visible: np.ndarray,
    footprint: np.ndarray,
    fy_tr: np.ndarray,
    fx_tr: np.ndarray,
    aux_full: dict[str, np.ndarray],
    static_ctx: dict[str, np.ndarray],
    n_buckets: int = 5,
    seed: int = 8400,
) -> np.ndarray:
    """Evaluate 8 H84-RAMP channels at training pixels ``(fy_tr, fx_tr)`` via 5-bucket B-LBCO.

    Groups ``visible`` into 3-px-buffered fault components (matching ``buffered_component_draw``)
    and assigns each component to one of ``n_buckets`` buckets. For training pixels whose nearest
    visible component belongs to bucket ``b``, evaluates the exact bitplane two-host frame on
    ``V \\ V_b``, eliminating the ``d1 == 0`` self-collapse tautology on training positives.
    """
    vis = np.asarray(visible, dtype=bool) & np.asarray(footprint, dtype=bool)
    comp, n_comp = ndi.label(vis, structure=STRUCT3)
    rng = np.random.default_rng(seed)
    bucket_of_comp = rng.integers(0, n_buckets, size=n_comp + 1)

    iy1_full = aux_full["iy1"][fy_tr, fx_tr]
    ix1_full = aux_full["ix1"][fy_tr, fx_tr]
    anchor_comp = comp[iy1_full, ix1_full]
    row_bucket = bucket_of_comp[anchor_comp]

    out = np.zeros((fy_tr.size, len(H84_NAMES)), dtype=np.float32)
    strike_res = aux_full["strike_resolved"]
    bend_turn = aux_full["bend_turn"]

    for b in range(n_buckets):
        sel = np.flatnonzero(row_bucket == b)
        if not sel.size:
            continue
        in_b_vis = vis & (bucket_of_comp[comp] == b)
        collar_b = ndi.distance_transform_edt(~in_b_vis) <= 3.0
        sub_vis = vis & (~collar_b)
        if not sub_vis.any():
            sub_vis = vis & (~in_b_vis)
        qy, qx = fy_tr[sel], fx_tr[sel]
        d1, ay1, ax1, c1, d2, ay2, ax2, c2, comp_len = _query_bitplane_two_host_kdtree(
            sub_vis, qy, qx
        )
        out[sel] = extract_h84_from_frame(
            rows=qy,
            cols=qx,
            d1=d1,
            ay1=ay1,
            ax1=ax1,
            c1=c1,
            d2=d2,
            ay2=ay2,
            ax2=ax2,
            c2=c2,
            comp_len=comp_len,
            strike_resolved=strike_res,
            bend_turn=bend_turn,
            geod_trans_at_pts=static_ctx["geod_trans"][qy, qx],
            base_bridge_at_pts=static_ctx["base_bridge"][qy, qx],
        )
    return out
