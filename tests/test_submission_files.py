"""The shipped files pass the portal-equivalence validator; known failure modes are caught.

Skipped when the restored inputs (.cache/gems_data) are absent, e.g. on a fresh clone.
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT / ".cache" / "gems_data"
    if (ROOT / ".cache" / "gems_data" / "sample_submission.tif").exists()
    else ROOT / "data" / "bridge"
)
pytestmark = pytest.mark.skipif(not (DATA / "sample_submission.tif").exists(), reason="restore data first")


def _sub():
    return json.loads((ROOT / "docs" / "data" / "submission.json").read_text())


def test_primary_passes_but_nan_outside_twin_is_not_submittable():
    from gems77lia import validate as v
    s = _sub()
    p = v.validate(ROOT / "docs/downloads" / s["primary"]["file"], mode="zeros", name=s["name"], note=s["note"])
    t = v.validate(ROOT / "docs/downloads" / s["twin_nan"]["file"], mode="nan", name=s["name"], note=s["note"])
    assert p["passed"]
    assert p["sha256"] == s["primary"]["sha256"]
    assert not t["passed"]
    statuses = {c["check"]: c["status"] for c in t["checks"]}
    assert statuses["all raster cells finite"] == "FAIL"
    assert statuses["zeros outside official footprint"] == "FAIL"
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


def test_legacy_audit_marks_nan_outside_as_diagnostic_not_valid():
    from gems77.submission import audit_geotiff
    import rasterio
    with rasterio.open(DATA / "sample_submission.tif") as src:
        footprint = np.isfinite(src.read(1))
    with rasterio.open(DATA / "labels.tif") as src:
        labels = src.read(1) > 0
    stem = "gems77-h77-lia-38323dots-20261021-aa5c9331"
    primary = audit_geotiff(ROOT / "docs/downloads" / f"{stem}-zeros.tif", footprint, labels, "zeros")
    diagnostic = audit_geotiff(ROOT / "docs/downloads" / f"{stem}-nan.tif", footprint, labels, "nan")
    assert primary["all_checks_passed"]
    assert not diagnostic["all_checks_passed"]
    assert not diagnostic["checks"]["validator_range_0_1_guaranteed"]
    assert not diagnostic["checks"]["outside_footprint_zero"]
    assert not diagnostic["checks"]["nan_mode_not_submittable"]


def test_nan_inside_footprint_is_rejected(tmp_path):
    from gems77lia import core, validate as v
    a = np.zeros((core.H, core.W), np.float32)
    fp = core.grid().footprint
    r, c = np.argwhere(fp)[0]
    a[r, c] = np.nan
    assert not v.validate(_write(tmp_path, a), mode="zeros", name="t", note="")["passed"]


def test_nan_outside_footprint_is_rejected(tmp_path):
    from gems77lia import core, validate as v
    a = np.zeros((core.H, core.W), np.float32)
    outside = ~core.grid().footprint
    r, c = np.argwhere(outside)[0]
    a[r, c] = np.nan
    result = v.validate(_write(tmp_path, a), mode="zeros", name="t", note="")
    statuses = {c["check"]: c["status"] for c in result["checks"]}
    assert not result["passed"]
    assert statuses["all raster cells finite"] == "FAIL"
    assert statuses["zeros outside official footprint"] == "FAIL"


def test_feature_sentinel_is_rejected(tmp_path):
    from gems77lia import core, validate as v
    a = np.zeros((core.H, core.W), np.float32)
    r, c = np.argwhere(core.grid().footprint)[0]
    a[r, c] = -3.4028235e38
    assert not v.validate(_write(tmp_path, a), mode="zeros", name="t", note="")["passed"]
