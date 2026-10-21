"""H77E unit tests: upward continuation physics + Euler solver recovery."""
import hashlib
import json
from pathlib import Path

import numpy as np

from gems77e.fields import euler_solve_windows, upward_derivatives

ROOT = Path(__file__).resolve().parents[1]


def test_upward_continuation_attenuates_short_wavelengths():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:128, 0:128]
    short = np.sin(2 * np.pi * x / 4.0)          # 4 px wavelength
    long = np.sin(2 * np.pi * x / 64.0)          # 64 px wavelength
    f = (short + long).astype(np.float32)
    outs = upward_derivatives(f, heights_m=(0.0, 800.0))
    amp = []
    for tx, ty, tz, t in outs:
        # amplitude of the continued field itself
        amp.append(float(t.std()))
    assert outs[1][3].std() < outs[0][3].std()
    # short wavelength is attenuated far more than the long one:
    ratio_short = float(np.abs(np.fft.rfft2(outs[1][3])[0, 32]) /
                        np.abs(np.fft.rfft2(outs[0][3])[0, 32]))
    assert ratio_short < 0.05


def test_euler_solver_recovers_synthetic_source():
    # Homogeneous field of degree -1 about (x0, y0, z0): T = 1/|r - r0|,
    # Euler: (r-r0).grad(T) = -1 * T  => structural index N = 1 here.
    h = w = 61
    x0t, y0t, z0t = 33.0, 27.0, -12.0  # z up: source below plane => z0 < 0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (xx - x0t) ** 2 + (yy - y0t) ** 2 + z0t ** 2
    T = 1.0 / np.sqrt(r2)
    dT = -T ** 3
    Tx = dT * (xx - x0t)
    Ty = dT * (yy - y0t)
    Tz = dT * (0.0 - z0t)  # dT/dz at z=0 along +z(up): (z - z0) with z=0
    cy = np.array([30, 28, 35])
    cx = np.array([30, 36, 25])
    sol, rel, ok = euler_solve_windows(
        Tx.astype(np.float32), Ty.astype(np.float32), Tz.astype(np.float32),
        T.astype(np.float32), cy, cx, half=3, si=1.0)
    assert ok.all() and (rel < 0.05).all()
    assert np.allclose(sol[:, 0], x0t, atol=1.5)
    assert np.allclose(sol[:, 1], y0t, atol=1.5)
    assert np.allclose(sol[:, 2], z0t, atol=1.5)


def test_euler_solver_si0_contact_recovers_source():
    # Degree-0 homogeneous field T = (z-z0)/|r-r0| (a contact-like source):
    # (r-r0).grad(T) = 0, i.e. structural index 0, background unidentifiable.
    h = w = 61
    x0t, y0t, z0t = 34.0, 26.0, -10.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - x0t, yy - y0t
    r = np.sqrt(dx ** 2 + dy ** 2 + z0t ** 2)
    u = 0.0 - z0t
    T = u / r
    Tx = -u * dx / r ** 3
    Ty = -u * dy / r ** 3
    Tz = (dx ** 2 + dy ** 2) / r ** 3
    cy = np.array([30, 25, 36])
    cx = np.array([30, 38, 24])
    sol, rel, ok = euler_solve_windows(
        Tx.astype(np.float32), Ty.astype(np.float32), Tz.astype(np.float32),
        T.astype(np.float32), cy, cx, half=3, si=0.0)
    assert sol.shape == (3, 3)
    assert ok.all() and (rel < 0.05).all()
    assert np.allclose(sol[:, 0], x0t, atol=1.5)
    assert np.allclose(sol[:, 1], y0t, atol=1.5)
    assert np.allclose(sol[:, 2], z0t, atol=1.5)


def test_h77e_artifact_matches_run_card_and_is_negative():
    card = json.loads((ROOT / "evidence" / "run_card_h77e.json").read_text())
    tif = ROOT / "docs" / "downloads" / f"{card['submission_name']}.tif"
    assert tif.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    assert card["verdict"] == "negative" and card["ok_to_submit"] is False
    assert card["safe_to_download"] is True and card["protocol_unique"] is False
    assert card["weekly_slots_used"] == 0 and card["organizer_confirmed_score"] is None
    assert len(card["note"]) <= 140
    import rasterio
    with rasterio.open(tif) as src:
        a = src.read(1)
        assert src.count == 1 and src.dtypes == ("float32",)
        assert src.crs.to_epsg() == 32611 and src.shape == (3730, 3292)
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert int((a > 0).sum()) == 37654
