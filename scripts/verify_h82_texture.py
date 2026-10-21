#!/usr/bin/env python3
"""Independent H82 artifact audit: no training, generation, scoring or repair."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def verify():
    card = json.loads((ROOT / "evidence/h82_texture/run_card.json").read_text())
    tif = ROOT / card["raster_path"]
    checks = {}
    with rasterio.open(tif) as ds, rasterio.open(ROOT / "data/bridge/sample_submission.tif") as sample:
        a = ds.read(1)
        fp = np.isfinite(sample.read(1))
        checks["single_band_float32"] = ds.count == 1 and ds.dtypes == ("float32",)
        checks["crs_shape_transform_equal_sample"] = (ds.crs, ds.shape, ds.transform) == (sample.crs, sample.shape, sample.transform)
        checks["nodata_none"] = ds.nodata is None
    with rasterio.open(ROOT / "data/bridge/existing_faults.tif") as ds:
        catalogue = (ds.read(1) > 0) & fp
    checks["all_cells_finite"] = bool(np.isfinite(a).all())
    checks["no_nan_inside_footprint"] = bool(np.isfinite(a[fp]).all())
    checks["all_values_0_1"] = bool(a.min() >= 0 and a.max() <= 1)
    checks["outside_zero"] = bool((a[~fp] == 0).all())
    checks["catalogue_zero"] = bool((a[catalogue] == 0).all())
    checks["sha256_matches_card"] = hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    checks["note_at_most_140"] = 0 < len(card["submission_note"]) <= 140
    checks["preregistration_pin"] = hashlib.sha256((ROOT / "registry/preregistration_h82_texture.json").read_bytes()).hexdigest() == card["preregistration_sha256"]
    with zipfile.ZipFile(tif.with_suffix(".zip")) as archive:
        checks["zip_exactly_one_identical_tiff"] = archive.namelist() == [tif.name] and archive.read(tif.name) == tif.read_bytes()
    checks["negative_not_approved_no_slot"] = (card["verdict"] == "negative" and not card["ok_to_submit"]
        and not card["selector_approved"] and card["weekly_slots_used"] == 0)
    checks["duplicate_stop_before_dots"] = (card["registry"]["surface_max_forward_overlap_3px"] > .70
        and card["registry"]["final_dots"].startswith("NOT PLACED"))
    checks["no_canary_leakage_flag"] = not card["leakage_canary"]["flagged"]
    comparisons = []
    for other in sorted((ROOT / "docs/downloads").glob("*.tif")):
        if other == tif:
            continue
        with rasterio.open(other) as ds:
            same_grid = (ds.shape, ds.crs, ds.transform) == (a.shape, rasterio.crs.CRS.from_epsg(32611),
                rasterio.Affine(100, 0, 243350, 0, -100, 4508550))
            equal = bool(np.array_equal(ds.read(1)[fp], a[fp])) if same_grid and ds.count == 1 else None
        comparisons.append(dict(file=other.name, same_grid=same_grid, identical_footprint_values=equal))
    checks["pixel_distinct_current_local_inventory"] = bool(comparisons) and all(
        row["identical_footprint_values"] is False for row in comparisons)
    record = dict(evidence_class="INDEPENDENT ARTIFACT READBACK, not a score or organizer acceptance",
        current_local_inventory=comparisons, current_local_inventory_count=len(comparisons),
        file=card["raster_path"], sha256=card["raster_sha256"], checks=checks,
        passed=all(checks.values()), min=float(a.min()), max=float(a.max()),
        nan_count=int(np.isnan(a).sum()), shape=list(a.shape), positive_support_pixels=int((a > 0).sum()))
    path = ROOT / "evidence/h82_texture/independent_audit.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
