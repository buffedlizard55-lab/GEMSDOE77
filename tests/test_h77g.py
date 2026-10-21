"""Unit and artifact verification tests for H77G (Hydrothermal Demagnetization Trough x Gravity-Basement Step)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from gems77g.fields import H77G_NAMES, build_h77g_fields, hessian_saddle, hessian_trough

ROOT = Path(__file__).resolve().parents[1]


def test_hessian_trough_detects_synthetic_linear_magnetic_low():
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    # Linear negative magnetic trough along x = 20 and positive magnetic ridge along x = 44
    trough = -np.exp(-0.5 * ((xx - 20.0) / 1.8) ** 2)
    ridge = +np.exp(-0.5 * ((xx - 44.0) / 1.8) ** 2)
    field = (trough + ridge).astype(np.float32)
    resp = hessian_trough(field, sigma=1.5, hp_sigma=12.0)
    assert resp.shape == (64, 64)
    assert np.isfinite(resp).all()
    assert float(resp[32, 20]) > 0.10
    assert float(resp[32, 44]) == 0.0


def test_hessian_saddle_detects_hyperbolic_pass():
    yy, xx = np.mgrid[-16:17, -16:17].astype(np.float32)
    saddle_field = (0.05 * (xx ** 2 - yy ** 2)).astype(np.float32)
    bowl_field = (0.05 * (xx ** 2 + yy ** 2)).astype(np.float32)
    s_resp = hessian_saddle(saddle_field, sigma=2.0)
    b_resp = hessian_saddle(bowl_field, sigma=2.0)
    assert float(s_resp[16, 16]) > 0.05
    assert float(b_resp[16, 16]) == 0.0


def test_build_h77g_fields_all_finite_in_0_1():
    rng = np.random.default_rng(77)
    foot = np.zeros((48, 48), dtype=bool)
    foot[6:42, 6:42] = True
    needed = [
        "rtp", "tmi_hg", "tmi_vg", "iso_grav_anom", "depth_to_base_surf",
        "cond_surf", "geod_dilaterate", "geod_shearrate", "ieq_n100a15", "deq_n100a15",
    ]
    bands = {nm: rng.normal(size=(48, 48)).astype(np.float32) for nm in needed}
    bands["rtp"][0, 0] = -3.4e38  # sentinel outside footprint
    out, diag = build_h77g_fields(bands, foot)
    assert list(out.keys()) == H77G_NAMES
    for nm in H77G_NAMES:
        arr = out[nm]
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert np.all(arr[~foot] == 0.0)
        assert diag[nm]["nan_count"] == 0


def test_h77g_artifact_matches_run_card_and_is_negative():
    card = json.loads((ROOT / "evidence" / "run_card_h77g.json").read_text())
    tif = ROOT / card["raster_path"]
    assert tif.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    assert card["verdict"] == "negative"
    assert card["ok_to_download"] is True
    assert card["ok_to_submit"] is False
    assert card["safe_to_download_and_submit"] is False
    assert card["protocol_unique"] is False
    assert card["informative_unique"] is True
    assert card["weekly_slots_used"] == 0
    assert card["organizer_confirmed_score_for_this_tif"] is None
    assert len(card["submission_note"]) <= 140
    with rasterio.open(tif) as src:
        a = src.read(1)
        assert src.count == 1 and src.dtypes == ("float32",)
        assert src.crs.to_epsg() == 32611 and src.shape == (3730, 3292)
    assert np.isfinite(a).all() and float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert int((a > 0).sum()) == 37654
