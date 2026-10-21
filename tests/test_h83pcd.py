"""Unit and artifact verification tests for Lane H83-PCD."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from scipy import ndimage as ndi

from src.gems77_pcd.allocator import (
    allocate_submodular_300m_pcd,
    build_sparse_halo_bitmask,
)
from src.gems77_pcd.features import (
    PCD_STATIC_NAMES,
    PCD_VIS_NAMES,
    compute_pcd_static_channels_from_arrays,
    compute_pcd_visible_mechanics,
    fourier_pseudogravity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pcd_static_channels_synthetic_poisson_anticoncordance() -> None:
    """Verify all 8 H83-PCD static channels are finite in [0, 1] and detect Poisson anti-concordance."""
    rng = np.random.default_rng(83)
    h, w = 96, 96
    fp = np.ones((h, w), dtype=bool)
    fp[:4, :] = False
    fp[:, :4] = False

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Left half (x around 28): concordant step where both gravity and RTP increase together
    step_concordant = np.tanh((xx - 28.0) / 3.0).astype(np.float32)
    # Right half (x around 68): fault with hydrothermal demagnetization / basement juxtaposition
    # where gravity increases across step while RTP decreases (Poisson anti-concordant)
    step_discordant = np.tanh((xx - 68.0) / 3.0).astype(np.float32)

    iso_grav = (
        step_concordant + step_discordant + 0.03 * rng.standard_normal((h, w))
    ).astype(np.float32)
    rtp = (
        step_concordant - step_discordant + 0.03 * rng.standard_normal((h, w))
    ).astype(np.float32)
    depth_base = (
        2.0 * step_discordant + 0.5 * step_concordant + 0.02 * rng.standard_normal((h, w))
    ).astype(np.float32)

    pg = fourier_pseudogravity(rtp, fp)
    assert pg.shape == (h, w)
    assert np.all(np.isfinite(pg))

    # Partial LiDAR coverage (right half has missing coverage NaN)
    has_lidar = (xx < 56).astype(np.float32)
    lidar_step = np.where(has_lidar > 0.5, 0.7, np.nan).astype(np.float32)
    lidar_dip = np.where(has_lidar > 0.5, 0.6, np.nan).astype(np.float32)
    lidar_down = np.where(has_lidar > 0.5, 0.8, np.nan).astype(np.float32)
    lidar_coh = np.where(has_lidar > 0.5, 0.75, np.nan).astype(np.float32)

    uni = lambda: rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)
    out = compute_pcd_static_channels_from_arrays(
        footprint=fp,
        rtp=rtp,
        iso_grav=iso_grav,
        depth_base=depth_base,
        elev=uni() * 200.0,
        slope=uni() * 8.0,
        geod_2ndinv=uni(),
        geod_shear=uni(),
        geod_dilat=uni(),
        deq=uni(),
        ieq=uni(),
        cond_surf=uni() * 10.0,
        rad_k_rank=uni(),
        rad_th_rank=uni(),
        rad_u_rank=uni(),
        mag_up150_rank=uni(),
        lidar_step=lidar_step,
        lidar_lap_dipole=lidar_dip,
        lidar_downface=lidar_down,
        lidar_coh100=lidar_coh,
        has_lidar=has_lidar,
        hydro_geotherm_plume=uni(),
        hydro_regional_halo=uni(),
        volc_vent_halo=uni(),
    )

    assert list(out.keys()) == PCD_STATIC_NAMES
    for name, arr in out.items():
        assert arr.shape == (h, w), name
        assert arr.dtype == np.float32, name
        assert np.all(np.isfinite(arr)), f"Non-finite values in {name}"
        assert float(arr[fp].min()) >= 0.0 - 1e-6, name
        assert float(arr[fp].max()) <= 1.0 + 1e-6, name
        assert float(np.abs(arr[~fp]).max()) == 0.0, name

    # Verify Poisson anti-concordance is stronger on the discordant fault (x ~ 68)
    # than on the concordant contact (x ~ 28)
    anti = out["pcd_poisson_anti_step"]
    mean_discordant = float(anti[20:76, 65:72].mean())
    mean_concordant = float(anti[20:76, 25:32].mean())
    assert mean_discordant > mean_concordant + 0.20


def test_pcd_visible_mechanics_loco_shape_and_finiteness() -> None:
    """Verify 5-channel fold-visible 12-bit LOCO mechanics including vis_pcd_transfer_relay."""
    h, w = 64, 64
    fp = np.ones((h, w), dtype=bool)
    vis = np.zeros((h, w), dtype=bool)
    vis[12:28, 20] = True
    vis[34:50, 23] = True  # Relay stepover geometry

    geod = np.full((h, w), 0.6, dtype=np.float32)
    poisson = np.full((h, w), 0.8, dtype=np.float32)
    toe = np.full((h, w), 0.7, dtype=np.float32)

    mech = compute_pcd_visible_mechanics(
        visible_mask=vis,
        footprint=fp,
        geod_rank_2d=geod,
        pcd_poisson_anti_2d=poisson,
        pcd_piedmont_toe_2d=toe,
        leave_own_component_out=True,
    )
    assert mech.shape == (int(fp.sum()), len(PCD_VIS_NAMES))
    assert mech.dtype == np.float32
    assert np.all(np.isfinite(mech))
    assert float(mech[:, 4].max()) > 0.0


def test_submodular_allocator_enforces_sparse_overlap_cap() -> None:
    """Verify bitwise 3-px sparse halo cap strictly enforces <= max_overlap_fraction."""
    rng = np.random.default_rng(77)
    h, w = 80, 80
    allowed = np.ones((h, w), dtype=bool)
    prob = rng.uniform(0.05, 1.0, size=(h, w)).astype(np.float32)

    # Create a reference sparse halo that covers the top 85% of the grid
    halo1 = np.zeros((h, w), dtype=bool)
    halo1[:68, :] = True
    halo2 = np.zeros((h, w), dtype=bool)
    halo2[:, :68] = True

    bits, crowd = build_sparse_halo_bitmask(allowed, [halo1, halo2])
    dots, meta = allocate_submodular_300m_pcd(
        prob_surface=prob,
        allowed_mask=allowed,
        target_n=120,
        gamma=2.0,
        min_spacing_px=0.0,
        max_candidates=6400,
        sparse_halo_bits=bits,
        sparse_crowd_fraction=crowd,
        num_sparse_rasters=2,
        max_overlap_fraction=0.60,
    )
    n_emitted = int((dots > 0.5).sum())
    assert n_emitted == 120
    ov1 = float((dots[halo1] > 0.5).sum()) / float(n_emitted)
    ov2 = float((dots[halo2] > 0.5).sum()) / float(n_emitted)
    assert ov1 <= 0.6001
    assert ov2 <= 0.6001
    assert meta["max_realized_sparse_overlap_fraction"] <= 0.6001


def test_h83pcd_preregistration_and_artifacts() -> None:
    """Verify preregistration JSON and (when generated) run card and GeoTIFF contract."""
    prereg_path = REPO_ROOT / "registry" / "preregistration_h83pcd.json"
    assert prereg_path.exists()
    prereg = json.loads(prereg_path.read_text())
    assert prereg["lane_id"] == "H83-PCD"
    assert prereg["static_pcd_channels"] == PCD_STATIC_NAMES
    assert prereg["fold_visible_loco_channels"] == PCD_VIS_NAMES

    run_card_path = REPO_ROOT / "evidence" / "run_card_h83pcd.json"
    if not run_card_path.exists():
        pytest.skip("H83-PCD pipeline artifacts not yet generated")

    rc = json.loads(run_card_path.read_text())
    assert rc["lane_id"] == "H83-PCD"
    assert rc["verdict"] == "negative"
    assert len(rc["submission_note"]) <= 140
    assert rc["leakage_canary"]["max_symmetric_auc"] <= 0.90
    assert rc["uniqueness_vs_registry"]["informative_sparse_scope"]["passes_sparse_gate"] is True
    assert (
        rc["uniqueness_vs_registry"]["informative_sparse_scope"]["max_forward_overlap_3px"]
        <= 0.70
    )

    tif_path = REPO_ROOT / rc["artifact_paths"]["submission_tif"]
    assert tif_path.exists()
    with rasterio.open(tif_path) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert (src.height, src.width) == (3730, 3292)
        assert str(src.crs) == "EPSG:32611"
        arr = src.read(1)
        assert np.all(np.isfinite(arr))
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0
        assert int((arr > 0.5).sum()) == 37654
