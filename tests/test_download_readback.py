import hashlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from audit_downloads import audit, validate


def write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:32611",
                       transform=from_origin(243350, 4508550, 100, 100)) as src:
        src.write(arr.astype("float32"), 1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.001, 1.001])
def test_nonfinite_and_out_of_range_fail(tmp_path, bad):
    sample, tif = tmp_path / "sample.tif", tmp_path / "candidate.tif"
    write(sample, np.zeros((5, 5)))
    arr = np.zeros((5, 5)); arr[2, 2] = bad
    write(tif, arr)
    result = validate(tif, sample, hashlib.sha256(tif.read_bytes()).hexdigest())
    assert not result["format_pass"]
    assert not result["checks"]["values_in_0_1"]
    assert result["authorized_to_submit"] is False


def test_valid_file_is_not_authorized_and_hash_matters(tmp_path):
    sample, tif = tmp_path / "sample.tif", tmp_path / "candidate.tif"
    write(sample, np.zeros((5, 5)))
    write(tif, np.eye(5))
    result = validate(tif, sample, hashlib.sha256(tif.read_bytes()).hexdigest())
    assert result["format_pass"] and not result["authorized_to_submit"]
    assert not validate(tif, sample, "wrong")["format_pass"]


def test_shape_mismatch_returns_failed_check_not_index_error(tmp_path):
    sample, tif = tmp_path / "sample.tif", tmp_path / "candidate.tif"
    write(sample, np.zeros((5, 5)))
    write(tif, np.zeros((4, 5)))
    result = validate(tif, sample, hashlib.sha256(tif.read_bytes()).hexdigest())
    assert not result["format_pass"]
    assert not result["checks"]["shape_matches"]


def test_every_current_listed_download_really_passes():
    result = audit()
    assert result["all_format_pass"]
    assert result["listed_count"] == len(result["artifacts"]) > 0
    assert all(not x["authorized_to_submit"] for x in result["artifacts"])
