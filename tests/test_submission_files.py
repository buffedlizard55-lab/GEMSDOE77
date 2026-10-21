"""The shipped files pass the portal-equivalence validator; known failure modes are caught.

Skipped when the restored inputs (.cache/gems_data) are absent, e.g. on a fresh clone.
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache" / "gems_data"
pytestmark = pytest.mark.skipif(not (DATA / "sample_submission.tif").exists(), reason="restore data first")


def _sub():
    return json.loads((ROOT / "docs" / "data" / "submission.json").read_text())


def test_shipped_primary_and_twin_pass():
    from gems77 import validate as v
    s = _sub()
    p = v.validate(ROOT / "docs/downloads" / s["primary"]["file"], mode="zeros", name=s["name"], note=s["note"])
    t = v.validate(ROOT / "docs/downloads" / s["twin_nan"]["file"], mode="nan", name=s["name"], note=s["note"])
    assert p["passed"] and t["passed"]
    assert p["sha256"] == s["primary"]["sha256"]
    assert len(s["note"]) <= 140


def _write(tmp_path, arr):
    import rasterio
    with rasterio.open(DATA / "sample_submission.tif") as ref:
        prof = ref.profile
    prof.update(nodata=None, compress="deflate")
    out = tmp_path / "x.tif"
    with rasterio.open(out, "w", **prof) as d:
        d.write(arr.astype(np.float32), 1)
    return out


def test_nan_inside_footprint_is_rejected(tmp_path):
    from gems77 import core, validate as v
    a = np.zeros((core.H, core.W), np.float32)
    fp = core.grid().footprint
    r, c = np.argwhere(fp)[0]
    a[r, c] = np.nan
    assert not v.validate(_write(tmp_path, a), mode="zeros", name="t", note="")["passed"]


def test_feature_sentinel_is_rejected(tmp_path):
    from gems77 import core, validate as v
    a = np.zeros((core.H, core.W), np.float32)
    r, c = np.argwhere(core.grid().footprint)[0]
    a[r, c] = -3.4028235e38
    assert not v.validate(_write(tmp_path, a), mode="zeros", name="t", note="")["passed"]
