#!/usr/bin/env python3
"""Independent readback of every *listed* download, not scientific clearance.

Reads no feature stack and runs no experiment. Checks actual TIFF bytes against
the sample grid and manifest hash, not only the manifest's format_valid flag.
Archive/NaN twins not listed in lanes.json are explicitly outside this scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def validate(path, sample, expected_hash):
    with rasterio.open(sample) as src:
        footprint = np.isfinite(src.read(1))
        grid = (src.crs, src.shape, src.transform)
    with rasterio.open(path) as src:
        arr = src.read(1)
        same_shape = src.shape == footprint.shape
        checks = {
            "single_band": src.count == 1,
            "crs_matches": src.crs == grid[0] and src.crs is not None,
            "shape_matches": same_shape,
            "transform_matches": src.transform == grid[2],
            "finite_entire_raster": bool(np.isfinite(arr).all()),
            "no_nan_inside_footprint": bool(same_shape and np.isfinite(arr[footprint]).all()),
            "values_in_0_1": bool(np.isfinite(arr).all() and (arr >= 0).all() and (arr <= 1).all()),
            "nodata_none": src.nodata is None,
            "sha256_matches_manifest": digest(path) == expected_hash,
        }
    return {"sha256": digest(path), "checks": checks,
            "format_pass": all(checks.values()), "authorized_to_submit": False}


def audit(root=ROOT):
    sample = root / "data/bridge/sample_submission.tif"
    manifest = json.loads((root / "registry/data_manifest.json").read_text())
    pin = next(x["sha256"] for x in manifest["files"] if x["id"] == "sample_submission")
    if digest(sample) != pin:
        raise ValueError("sample hash mismatch")
    rows = json.loads((root / "docs/data/lanes.json").read_text())["lanes"]
    results = []
    for row in rows:
        try:
            path = (root / "docs" / row["file"]).resolve()
            if not path.is_relative_to((root / "docs/downloads").resolve()):
                raise ValueError("download outside docs/downloads")
            result = validate(path, sample, row["sha256"])
        except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as exc:
            result = {"format_pass": False, "error": str(exc), "authorized_to_submit": False}
        results.append({"lane": row["lane"], "file": row["file"], **result})
    return {"schema": "gems-listed-download-readback-v1",
            "evidence_class": "FORMAT-INTEGRITY-MEASUREMENT; NOT SCORE OR PORTAL ACCEPTANCE",
            "scope": "Every primary TIFF in docs/data/lanes.json; no archive or NaN twins, no uniqueness certification",
            "manifest_sha256": digest(root / "docs/data/lanes.json"),
            "reference_sha256": pin, "listed_count": len(results),
            "all_format_pass": bool(results) and all(x["format_pass"] for x in results),
            "authorized_to_submit": False, "artifacts": results}


if __name__ == "__main__":
    output = audit()
    dest = ROOT / "evidence/listed_download_audit.json"
    dest.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "artifacts"}, indent=2))
    raise SystemExit(0 if output["all_format_pass"] else 1)
