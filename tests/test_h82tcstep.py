"""H82-TCSTEP field unit tests (synthetic, no data restore required)."""
import numpy as np
import pytest

from gems77h82.fields import H82_NAMES, build_h82_fields, rank01_tieavg


def _synthetic():
    rng = np.random.default_rng(82)
    tc = np.full((120, 140), 10.0, np.float32)
    tc[:, 70:] = 40.0  # sharp vertical lithologic step at col 70
    tc += rng.normal(0, 0.5, tc.shape).astype(np.float32)
    slope = np.full((120, 140), 0.01, np.float32)
    slope[:40, :] = 5.0  # steep (scarp-like) top strip
    foot = np.ones((120, 140), bool)
    foot[:3, :] = False
    return tc, slope, foot


def test_fields_contract():
    tc, slope, foot = _synthetic()
    f = build_h82_fields(tc, slope, foot)
    assert set(f) == set(H82_NAMES)
    for v in f.values():
        assert np.isfinite(v).all() and v.min() >= 0.0 and v.max() <= 1.0
        assert (v[~foot] == 0).all()


def test_step_fires_on_step_not_on_flat():
    tc, slope, foot = _synthetic()
    f = build_h82_fields(tc, slope, foot)
    step_zone = f["tc_step2"][60:110, 66:75].mean()
    flat_zone = f["tc_step2"][60:110, 10:40].mean()
    assert step_zone > 0.9 and flat_zone < step_zone - 0.25


def test_blind_gate_suppresses_steep_strip():
    tc, slope, foot = _synthetic()
    f = build_h82_fields(tc, slope, foot)
    steep = f["tc_blind"][5:35, 66:75].mean()
    gentle = f["tc_blind"][60:110, 66:75].mean()
    assert gentle > steep + 0.2


def test_rank_ties_are_averaged_not_scan_ordered():
    a = np.zeros((10, 10), np.float32)
    a[0, 0] = 1.0
    mask = np.ones((10, 10), bool)
    r = rank01_tieavg(a, mask)
    zeros = r[a == 0]
    assert np.allclose(zeros, zeros[0])  # plateau -> one shared rank
    assert r[0, 0] == pytest.approx(1.0)


def test_sentinel_and_nan_inputs_do_not_propagate():
    tc, slope, foot = _synthetic()
    tc[50:60, 50:60] = np.float32(-3.4028234663852886e38)
    slope[20, 20] = np.nan
    f = build_h82_fields(tc, slope, foot)
    for v in f.values():
        assert np.isfinite(v).all()
