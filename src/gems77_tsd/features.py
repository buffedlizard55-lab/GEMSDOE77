"""Zero-leakage, zero-SGMC, zero-prior-raster feature synthesis for Lane H77-TSD.

Computes 10 static in-footprint channels (stored at data/processed/tsd_features.npy)
and 5 strictly fold-visible 12-bit Leave-Own-Component-Out (LOCO) structural relay
channels. Strictly forbids reading any prior submission raster, any SGMC layer, or
the 'dist_known_fault_px' column of DOE GDR 1391 CSVs.
"""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

from gems77 import features77 as F77
from gems77d.features import clean_band, compute_visible_mechanics, gf32, rank01_fp

TSD_STATIC_NAMES: list[str] = [
    "tsd_transtens_index",
    "tsd_swarm_excess",
    "tsd_mt_cap_margin",
    "tsd_basement_pullapart",
    "tsd_hydro_upflow_aniso",
    "tsd_rad_kth_alter_edge",
    "tsd_lidar_deconf_dipole",
    "tsd_pot_tilt_bridge",
    "tsd_strain_scarp_deficit",
    "tsd_composite_surface",
]
TSD_STATIC_IDX: dict[str, int] = {n: i for i, n in enumerate(TSD_STATIC_NAMES)}

TSD_VIS_NAMES: list[str] = [
    "vis_dist_clipped",
    "vis_cracktip_lobe",
    "vis_stepover_corridor",
    "vis_strain_deficit",
    "vis_transtens_relay",
]


def _load_gdr1391_grids(raw_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load DOE GDR 1391 chemical geothermometer disequilibrium & volcanic vents.

    CRITICAL ANTI-LEAKAGE GUARANTEE:
    Never accesses 'dist_known_fault_px'.
    """
    h, w = shape
    chem_excess = np.zeros((h, w), dtype=np.float32)
    vents = np.zeros((h, w), dtype=np.float32)

    ws_csv = raw_dir / "external" / "gdr_wellspring_in_footprint.csv"
    with ws_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        for rec in reader:
            r = int(float(rec["row"]))
            c = int(float(rec["col"]))
            if not (0 <= r < h and 0 <= c < w):
                continue
            vals_chem = []
            for k in ("geothermquartz_c", "geothermchalc_c", "geothermcat_c"):
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
            score = float(np.log1p(0.65 * dq + 0.35 * high_res + 0.25 * max(0.0, t_surf - 22.0)))
            if score > chem_excess[r, c]:
                chem_excess[r, c] = score

    vv_csv = raw_dir / "external" / "gdr_volcanic_vents_in_footprint.csv"
    if vv_csv.exists():
        with vv_csv.open(newline="") as fh:
            for rec in csv.DictReader(fh):
                r = int(float(rec["row"]))
                c = int(float(rec["col"]))
                if 0 <= r < h and 0 <= c < w:
                    vents[r, c] = max(vents[r, c], 1.25)

    return chem_excess, vents


def build_tsd_static_features(
    raw_dir: Path,
    proc_dir: Path,
    force: bool = False,
) -> tuple[np.memmap, dict]:
    """Build or open cached in-footprint (N_fp, 10) float32 H77-TSD feature matrix."""
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / "tsd_features.npy"
    meta_path = proc_dir / "tsd_features_meta.json"

    with rasterio.open(raw_dir / "sample_submission.tif") as s:
        foot = np.isfinite(s.read(1))
    h, w = foot.shape
    n_fp = int(foot.sum())
    n_ch = len(TSD_STATIC_NAMES)

    if not force and out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("channels") == TSD_STATIC_NAMES and meta.get("shape") == [n_fp, n_ch]:
            return np.load(out_path, mmap_mode="r"), meta

    stack = np.load(proc_dir / "stack.npy", mmap_mode="r")
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_fp, n_ch))

    def sband(name: str) -> np.ndarray:
        return clean_band(stack[:, :, F77.IDX[name]], foot)

    # 1. Transtensional strain-partitioning mode index (Band 7 shear x Band 8 positive dilatation x Band 4 I2)
    shear = sband("geod_shearrate")
    dilat = sband("geod_dilaterate")
    inv2 = sband("geod_2ndinv")
    r_shear = rank01_fp(shear, foot)
    r_dilat_pos = rank01_fp(np.maximum(dilat, 0.0), foot)
    r_inv2 = rank01_fp(inv2, foot)
    del shear, dilat, inv2
    transtens_2d = rank01_fp(
        np.sqrt(np.maximum(r_shear * r_dilat_pos, 0.0)) * (0.5 + 0.5 * r_inv2),
        foot,
    )
    del r_shear, r_dilat_pos, r_inv2
    mm[:, TSD_STATIC_IDX["tsd_transtens_index"]] = transtens_2d[foot]

    # 2. Fluid-driven seismic swarm excess (Band 10 dependent EQ vs Band 16 independent EQ)
    deq = sband("deq_n100a15")
    ieq = sband("ieq_n100a15")
    r_deq = rank01_fp(deq, foot)
    r_ieq = rank01_fp(ieq, foot)
    del deq, ieq
    swarm_2d = rank01_fp(r_deq - 0.65 * r_ieq, foot)
    del r_deq, r_ieq
    mm[:, TSD_STATIC_IDX["tsd_swarm_excess"]] = swarm_2d[foot]

    # 3. MT clay-cap lateral conductance boundary perched over shallow conductive basement
    cond = sband("cond_surf")
    depth_b = sband("depth_to_base_surf")
    c_log = gf32(np.log1p(np.maximum(cond, 0.0)), sigma=2.0)
    cy, cx = np.gradient(c_log)
    cond_grad = np.hypot(cy, cx).astype(np.float32)
    del cond, c_log, cy, cx
    base_upwarp = -(gf32(depth_b, sigma=4.0, order=(2, 0)) + gf32(depth_b, sigma=4.0, order=(0, 2)))
    del depth_b
    mt_cap_2d = rank01_fp(
        rank01_fp(cond_grad, foot) * (0.35 + 0.65 * rank01_fp(base_upwarp, foot)),
        foot,
    )
    del cond_grad, base_upwarp
    mm[:, TSD_STATIC_IDX["tsd_mt_cap_margin"]] = mt_cap_2d[foot]

    # 4. Pull-apart / releasing-step gravity-basement saddle
    grav = sband("iso_grav_anom")
    base_grad = sband("base_grad_log")
    grav_curv_pos = np.maximum(
        gf32(grav, sigma=3.0, order=(2, 0)) + gf32(grav, sigma=3.0, order=(0, 2)),
        0.0,
    )
    del grav
    pullapart_2d = rank01_fp(
        rank01_fp(grav_curv_pos, foot) * rank01_fp(base_grad, foot) * (0.4 + 0.6 * transtens_2d),
        foot,
    )
    del grav_curv_pos, base_grad
    mm[:, TSD_STATIC_IDX["tsd_basement_pullapart"]] = pullapart_2d[foot]

    # 5. Anisotropic NNE/NW hydrothermal upflow plume coupled with magnetic demagnetization
    chem_g, vent_g = _load_gdr1391_grids(raw_dir, (h, w))
    src_hydro = chem_g + 0.85 * vent_g
    del chem_g, vent_g
    plume_near = 0.65 * gf32(src_hydro, sigma=(6.5, 3.0), mode="constant") + 0.35 * gf32(
        src_hydro, sigma=3.5, mode="constant"
    )
    plume_far = gf32(src_hydro, sigma=16.0, mode="constant")
    del src_hydro
    rtp = sband("rtp")
    with rasterio.open(raw_dir / "external" / "geodawn_extensions_u8.tif") as es:
        tmi_up150 = clean_band(es.read(4), foot)
    demag = np.maximum(rank01_fp(tmi_up150, foot) - rank01_fp(rtp, foot), 0.0).astype(np.float32)
    del tmi_up150, rtp
    hydro_2d = rank01_fp(
        (0.60 * rank01_fp(plume_near, foot) + 0.40 * rank01_fp(plume_far, foot))
        * (0.45 + 0.55 * rank01_fp(demag + 0.5 * mt_cap_2d, foot)),
        foot,
    )
    del plume_near, plume_far, demag
    mm[:, TSD_STATIC_IDX["tsd_hydro_upflow_aniso"]] = hydro_2d[foot]

    # 6. Potassium-thorium alteration-boundary ridge + selective uranium halo
    rk = rank01_fp(sband("rad_K"), foot)
    rth = rank01_fp(sband("rad_Th"), foot)
    ru = rank01_fp(sband("rad_U"), foot)
    ky, kx = np.gradient(gf32(rk - rth, sigma=2.0))
    u_sel = np.maximum(ru - 0.5 * (rk + rth), 0.0)
    del rk, rth, ru
    rad_alter_2d = rank01_fp(np.hypot(ky, kx) + 0.35 * u_sel, foot)
    del ky, kx, u_sel
    mm[:, TSD_STATIC_IDX["tsd_rad_kth_alter_edge"]] = rad_alter_2d[foot]

    # 7. Missing-coverage-aware 1 m LiDAR scarp crest-toe Laplacian dipole bridged with 100 m ridge/frontness
    with rasterio.open(raw_dir / "external" / "lidar_scarp_features_u8.tif") as ls:
        has_lidar = foot & (ls.read(12) > 0)
        q_step = ls.read(3).astype(np.float32)
        q_lapn = ls.read(4).astype(np.float32)
        q_lapp = ls.read(5).astype(np.float32)
        q_coh = ls.read(10).astype(np.float32)
    step_u = np.clip((q_step - 1.0) / 254.0, 0.0, 1.0)
    lapn_u = np.clip((q_lapn - 1.0) / 254.0, 0.0, 1.0)
    lapp_u = np.clip((q_lapp - 1.0) / 254.0, 0.0, 1.0)
    coh_u = np.clip((q_coh - 1.0) / 254.0, 0.0, 1.0)
    del q_step, q_lapn, q_lapp, q_coh
    raw_dip = np.sqrt(np.maximum(lapn_u * lapp_u, 0.0)) * (0.4 + 0.6 * step_u) * (0.3 + 0.7 * coh_u)
    del step_u, lapn_u, lapp_u, coh_u
    r_dip_valid = rank01_fp(np.where(has_lidar, raw_dip, np.nan), foot)
    del raw_dip
    r_bridge_100m = 0.55 * rank01_fp(sband("dem_ridge1"), foot) + 0.45 * rank01_fp(sband("frontness"), foot)
    lidar_dip_2d = rank01_fp(
        np.where(has_lidar, 0.70 * r_dip_valid + 0.30 * r_bridge_100m, r_bridge_100m),
        foot,
    )
    del r_dip_valid, r_bridge_100m
    mm[:, TSD_STATIC_IDX["tsd_lidar_deconf_dipole"]] = lidar_dip_2d[foot]

    # 8. Multi-sensor potential-field zero-tilt contact & phase-congruency bridge
    cc = sband("contact_confluence")
    pcm = sband("pc_mag")
    pcg = sband("pc_grav")
    pot_bridge_2d = rank01_fp(
        rank01_fp(cc, foot) * (0.35 + 0.65 * rank01_fp(np.sqrt(np.maximum(pcm * pcg, 0.0)), foot)),
        foot,
    )
    del cc, pcm, pcg
    mm[:, TSD_STATIC_IDX["tsd_pot_tilt_bridge"]] = pot_bridge_2d[foot]

    # 9. Unmapped-scarp transtensional strain + swarm deficit
    deficit_2d = rank01_fp(
        (0.60 * transtens_2d + 0.40 * swarm_2d) * (0.55 * lidar_dip_2d + 0.45 * pot_bridge_2d),
        foot,
    )
    mm[:, TSD_STATIC_IDX["tsd_strain_scarp_deficit"]] = deficit_2d[foot]

    # 10. Standalone physical discovery composite surface
    comp_2d = rank01_fp(
        0.30 * deficit_2d
        + 0.25 * lidar_dip_2d
        + 0.15 * pot_bridge_2d
        + 0.15 * hydro_2d
        + 0.10 * mt_cap_2d
        + 0.05 * pullapart_2d,
        foot,
    )
    mm[:, TSD_STATIC_IDX["tsd_composite_surface"]] = comp_2d[foot]
    mm.flush()
    del mm, transtens_2d, swarm_2d, mt_cap_2d, pullapart_2d, hydro_2d, rad_alter_2d, lidar_dip_2d, pot_bridge_2d, deficit_2d, comp_2d
    gc.collect()

    meta = {
        "shape": [n_fp, n_ch],
        "dtype": "float32",
        "channels": TSD_STATIC_NAMES,
        "footprint_pixels": n_fp,
        "lidar_valid_pixels": int(has_lidar.sum()),
        "lidar_missing_bridged_pixels": int((foot & ~has_lidar).sum()),
        "sgmc_channels_used": 0,
        "prior_submission_rasters_used": 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return np.load(out_path, mmap_mode="r"), meta


def compute_tsd_visible_mechanics(
    visible_mask: np.ndarray,
    footprint: np.ndarray,
    geod_strain_rank_2d: np.ndarray,
    tsd_transtens_fp: np.ndarray,
    tsd_swarm_fp: np.ndarray,
) -> np.ndarray:
    """Compute 5 strictly fold-visible 12-bit LOCO structural & transtensional-relay channels."""
    base_4 = compute_visible_mechanics(visible_mask, footprint, geod_strain_rank_2d)
    stepover_fp = base_4[:, 2]
    transtens_relay_fp = (
        (0.60 * np.asarray(tsd_transtens_fp, dtype=np.float32) + 0.40 * np.asarray(tsd_swarm_fp, dtype=np.float32))
        * stepover_fp
    ).astype(np.float32)
    return np.column_stack([base_4, transtens_relay_fp]).astype(np.float32)
