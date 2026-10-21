"""H85-XWC regression tests: field math on synthetic data + artifact contract."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = "gems77-h85xwc-magxgrav-coherence-37654dots-20261021-zeros"
TIF = ROOT / "docs" / "downloads" / f"{NAME}.tif"


def test_coherence_recovers_shared_edge_not_single_field_edge():
    from gems77h85.fields import coherence_surface

    rng = np.random.default_rng(85)
    fp = np.ones((128, 128), bool)
    yy, xx = np.mgrid[:128, :128]
    shared = np.tanh((xx - 64) / 2.0)              # edge present in BOTH fields
    solo = np.tanh((yy - 32) / 2.0)                # edge in RTP only
    rtp = 3 * shared + 3 * solo + 0.05 * rng.standard_normal(fp.shape)
    grav = 3 * shared + 0.05 * rng.standard_normal(fp.shape)
    out = coherence_surface(rtp, grav, fp)
    s = out["xwc"]
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert np.isfinite(s).all()
    near_shared = s[:, 62:67].mean()
    near_solo = s[30:35, 8:40].mean()
    assert near_shared > near_solo


def test_bandpass_removes_dc_and_keeps_target_wavelength():
    from gems77h85.fields import bandpass

    yy, xx = np.mgrid[:128, :128]
    wave = np.sin(2 * np.pi * xx / 8.0) + 5.0      # 8 px wavelength + DC offset
    out = bandpass(wave, 8.0)
    assert abs(out.mean()) < 0.05
    assert np.corrcoef(out.ravel(), np.sin(2 * np.pi * xx.ravel() / 8.0))[0, 1] > 0.99


def test_artifact_matches_run_card_and_validator():
    rasterio = pytest.importorskip("rasterio")
    assert TIF.is_file()
    rc = json.loads((ROOT / "evidence" / "h85xwc_run_card.json").read_text())
    assert hashlib.sha256(TIF.read_bytes()).hexdigest() == rc["raster_sha256"]
    assert rc["verdict"] == "negative" and rc["ok_to_submit"] is False
    assert rc["ok_to_download"] is True
    assert len(rc["note"]) <= 140
    with rasterio.open(TIF) as ds:
        a = ds.read(1)
        assert ds.count == 1 and ds.dtypes == ("float32",)
        assert (ds.height, ds.width) == (3730, 3292)
        assert str(ds.crs) == "EPSG:32611"
        assert tuple(ds.transform)[:6] == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
        assert ds.nodata is None
    assert np.isfinite(a).all()                    # 0 NaN / Inf anywhere
    assert a.min() >= 0.0 and a.max() <= 1.0        # portal [0,1] contract
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert int((a > 0).sum()) == 37_654
    zpath = TIF.with_suffix(".zip")
    with zipfile.ZipFile(zpath) as z:
        assert z.namelist() == [TIF.name]
        assert z.read(TIF.name) == TIF.read_bytes()
