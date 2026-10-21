"""Unit and artifact verification tests for H77H (Piedmont-Bajada Alluvial-Fan Toe Pinchout x Scarp Polarity x Sinter-Relay Bridge)."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

from gems77h.fields import (
    H77H_LOCO_NAMES,
    H77H_STATIC_NAMES,
    compute_h77h_loco_channels,
    elliptical_tip_relay_bridge,
    lahontan_shoreline_suppressor,
    piedmont_tail_slope_anomaly,
    scarp_polarity_ratio,
)

ROOT = Path(__file__).resolve().parents[1]


def test_piedmont_tail_slope_anomaly_isolates_subcell_fan_scarp():
    # Smooth bajada background: slope_max ~= slope_p90 = 4.0 deg
    s_p90 = np.full((32, 32), 4.0, dtype=np.float32)
    s_max = np.full((32, 32), 4.2, dtype=np.float32)
    coh = np.full((32, 32), 0.35, dtype=np.float32)
    env = np.full((32, 32), 4.5, dtype=np.float32)
    # Narrow scarp along row 16: slope_p90 stays 4.5 deg, slope_max spikes to 22 deg, high aspect coherence
    s_max[16, 6:26] = 22.0
    coh[16, 6:26] = 0.92
    resp = piedmont_tail_slope_anomaly(s_max, s_p90, coh, env)
    assert resp.shape == (32, 32)
    assert np.isfinite(resp).all()
    assert float(resp[16, 16]) > 20.0 * float(resp[8, 16])


def test_lahontan_shoreline_suppressor_damps_flat_highstand_contour():
    # Flat shoreline terrace at 1332 m vs inclined tectonic scarp at 1332 m vs non-highstand 1800 m
    dem_flat = np.full((32, 32), 1332.0, dtype=np.float32)
    yy, _ = np.mgrid[0:32, 0:32].astype(np.float32)
    dem_steep = 1332.0 + 12.0 * (yy - 16.0)
    dem_high = np.full((32, 32), 1800.0, dtype=np.float32)

    f_flat = lahontan_shoreline_suppressor(dem_flat)
    f_steep = lahontan_shoreline_suppressor(dem_steep)
    f_high = lahontan_shoreline_suppressor(dem_high)

    assert float(f_flat[16, 16]) < 0.25
    assert float(f_steep[16, 16]) > 0.80
    assert float(f_high[16, 16]) > 0.99


def test_scarp_polarity_ratio_respects_lidar_valid_zero_sentinel():
    rng = np.random.default_rng(7708)
    foot = np.ones((40, 40), dtype=bool)
    foot[0:4, :] = False
    valid = np.zeros((40, 40), dtype=np.uint8)
    valid[4:24, :] = 1  # top half has LiDAR, bottom half (rows 24..39) has valid == 0

    df = rng.uniform(5, 120, size=(40, 40)).astype(np.float32)
    uf = rng.uniform(5, 120, size=(40, 40)).astype(np.float32)
    cr = rng.uniform(5, 80, size=(40, 40)).astype(np.float32)
    coh100 = rng.uniform(50, 240, size=(40, 40)).astype(np.float32)
    # Put bogus sentinel zeros where valid == 0
    df[valid == 0] = 0.0
    uf[valid == 0] = 0.0
    cr[valid == 0] = 0.0

    curv = rng.uniform(1.0, 20.0, size=(40, 40)).astype(np.float32)
    acoh = rng.uniform(0.2, 0.95, size=(40, 40)).astype(np.float32)
    sstd = rng.uniform(0.5, 5.0, size=(40, 40)).astype(np.float32)

    pol, anti = scarp_polarity_ratio(df, uf, cr, coh100, valid, curv, acoh, sstd, foot)
    for arr in (pol, anti):
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert np.all(arr[~foot] == 0.0)
        # Non-LiDAR footprint cells are bridged by 10m DEM, not clamped to zero
        assert float(arr[28:38, 8:32].mean()) > 0.20


def test_elliptical_tip_relay_bridge_and_loco_channels():
    foot = np.ones((48, 48), dtype=bool)
    vis = np.zeros((48, 48), dtype=bool)
    # Two en-echelon fault segments separated by an 8-px relay step-over around (24, 24)
    vis[22, 8:21] = True
    vis[26, 27:40] = True
    # Add two more components so all 4 LOCO buckets have support
    vis[8, 8:18] = True
    vis[40, 30:42] = True

    bridge = elliptical_tip_relay_bridge(vis, foot)
    assert np.isfinite(bridge).all()
    assert float(bridge[24, 24]) > 0.35
    assert float(bridge[5, 40]) < 0.05

    lah = np.full((48, 48), 0.5, dtype=np.float32)
    pol = np.full((48, 48), 0.5, dtype=np.float32)
    loco = compute_h77h_loco_channels(vis, foot, lah, pol)
    assert list(loco.keys()) == H77H_LOCO_NAMES
    for nm in H77H_LOCO_NAMES:
        arr = loco[nm]
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def test_h77h_artifact_matches_run_card_and_fails_closed():
    card = json.loads((ROOT / "evidence" / "run_card_h77h.json").read_text())
    tif = ROOT / card["raster_path"]
    zp = ROOT / card["zip_path"]
    assert tif.is_file() and zp.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    assert hashlib.sha256(zp.read_bytes()).hexdigest() == card["zip_sha256"]
    with zipfile.ZipFile(zp) as zf:
        assert zf.namelist() == [tif.name]
    assert card["verdict"] == "negative"
    assert card["ok_to_download"] is True
    assert card["ok_to_submit"] is False
    assert card["safe_to_download_and_submit"] is False
    assert card["protocol_unique"] is False
    assert card["informative_unique"] is True
    assert card["weekly_slots_used"] == 0
    assert card["organizer_confirmed_score_for_this_tif"] is None
    assert len(card["submission_note"]) <= 140
    assert len(H77H_STATIC_NAMES) == 12
    assert card["leakage_canary"]["pass"] is True
    assert card["leakage_canary"]["max_symmetric_auc"] <= 0.90
    assert card["holdout_dti"]["seed_20"]["A_h77h_guarded"]["dti"] > 0.152718
    assert card["holdout_dti"]["seed_30"]["A_h77h_guarded"]["dti"] > 0.152718
    with rasterio.open(tif) as src:
        a = src.read(1)
        assert src.count == 1 and src.dtypes == ("float32",)
        assert src.crs.to_epsg() == 32611 and src.shape == (3730, 3292)
        assert src.nodata is None
    assert np.isfinite(a).all() and float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert int((a > 0).sum()) == 37654
