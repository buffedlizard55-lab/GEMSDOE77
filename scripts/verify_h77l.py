#!/usr/bin/env python3
"""Pass-2 independent audit of H77L bytes, grid, inventory scope and source pins.

Does NOT rerun the experiment or choose a submission. The dense witness is
restored only for independent verification and is not copied into the model.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "evidence/h77l_run_card.json"


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit():
    card = json.loads(CARD.read_text())
    pins = json.loads((ROOT / "registry/template_pin.json").read_text())
    man = {entry["id"]: entry for entry in json.loads(
        (ROOT / "registry/data_manifest.json").read_text())["files"]}
    tif = ROOT / card["raster_path"]
    witness = ROOT / ".cache/registry/dense_witness.tif"
    h33 = ROOT / ".cache/registry/h33.tif"
    if not witness.is_file() or not h33.is_file():
        raise FileNotFoundError("restore pinned witnesses via scripts/restore_registry.py")
    with rasterio.open(ROOT / "data/bridge/sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
        ref = (s.count, s.crs, s.shape, s.transform)
    with rasterio.open(ROOT / "data/bridge/existing_faults.tif") as s:
        catalogue = (s.read(1) > 0) & fp
    with rasterio.open(tif) as s:
        arr = s.read(1)
        meta = (s.count, s.crs, s.shape, s.transform)
        dtype, nodata = s.dtypes[0], s.nodata
    with rasterio.open(witness) as s:
        dense = (s.read(1) > 0) & fp
        same_grid = (s.count, s.crs, s.shape, s.transform) == ref
    with rasterio.open(h33) as s:
        previous = (s.read(1) > 0) & fp
        h33_grid = (s.count, s.crs, s.shape, s.transform) == ref
    checks = {
        "readback_sha256_matches_card": digest(tif) == card["raster_sha256"],
        "source_sample_sha256_pinned": digest(ROOT / "data/bridge/sample_submission.tif") == man["sample_submission"]["sha256"],
        "source_catalogue_sha256_pinned": digest(ROOT / "data/bridge/existing_faults.tif") == man["existing_faults"]["sha256"],
        "witness_sha256_pinned": digest(witness) == "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be",
        "h33_sha256_pinned": digest(h33) == "c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9",
        "reference_grid_identical": meta == ref and same_grid and h33_grid,
        "float32_single_band_no_nodata": dtype == "float32" and meta[0] == 1 and nodata is None,
        "finite_entire_file": bool(np.isfinite(arr).all()),
        "range_0_1_entire_file": bool((arr >= 0).all() and (arr <= 1).all()),
        "no_nan_inside_footprint": bool(np.isfinite(arr[fp]).all()),
        "outside_sample_footprint_zero": bool((arr[~fp] == 0).all()),
        "known_fault_pixels_zero": bool((arr[catalogue] == 0).all()),
        "distinct_pixel_field_vs_h33": bool(not np.array_equal(arr > 0, previous)),
        "no_full_registry_claim": card["registry"]["full_registry_certified"] is False,
        "no_slot_approval": card["ok_to_submit"] is False and card["verdict"] == "negative"
                            and card["weekly_slots_spent"] == 0,
        "preregistration_hash_matches_frozen_card": digest(ROOT / "registry/preregistration_h77l.json") == card["preregistration_sha256"],
        "shared_template_unmodified": all(digest(ROOT / "vendor/gems57" / f) == sha
                                         for f, sha in pins["files"].items()),
    }
    zip_path = ROOT / card["zip_path"]
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        checks["one_tiff_zip_equals_primary_bytes"] = (
            members == [tif.name] and z.read(members[0]) == tif.read_bytes())
    # Independent of the shared uniqueness implementation: the allowed domain
    # is a subset of the pixel-exact positive witness support, hence no distance
    # transform or dilation approximation can change the overlap conclusion.
    allowed = fp & ~catalogue
    idx = (arr > 0) & allowed
    overlap = float(dense[idx].mean()) if idx.any() else None
    checks["universal_overlap_reproduced"] = bool(dense[allowed].all() and overlap == 1.0)
    local_prior_tifs = [other for other in (ROOT / "docs/downloads").glob("*.tif") if other != tif]
    checks["current_raster_not_an_old_local_tif_byte_copy"] = all(
        digest(other) != card["raster_sha256"] for other in local_prior_tifs)
    def _different_pixels(other):
        with rasterio.open(other) as s:
            if s.shape != arr.shape or s.count != 1:
                return True
            old = s.read(1)
        return not np.array_equal(np.nan_to_num(old, nan=0.0), arr)
    checks["pixel_distinct_from_all_local_prior_tifs"] = all(
        _different_pixels(other) for other in local_prior_tifs)
    stats = {
        "reference_shape_rows_cols": list(ref[2]), "catalogue_positive_pixels": int(catalogue.sum()),
        "eligible_pixels_reported_by_fold_regions": sum(f["evaluation_px"] for f in
            json.loads((ROOT / "evidence/h77l_holdout.json").read_text())["split_receipts"]),
        "withheld_positive_pixels": card["holdout_dti"]["withheld_positive_pixels"],
        "withheld_share_of_known_catalogue": card["holdout_dti"]["withheld_positive_pixels"] / int(catalogue.sum()),
        "local_prior_tifs_checked": len(local_prior_tifs),
        "postwrite_positive_pixels": int(idx.sum()),
        "dense_witness_positive_pixels": int(dense.sum()),
        "h33_positive_pixels": int(previous.sum()),
        "forward_overlap_dense_witness": overlap,
        "h33_min_catalogue_distance_px": float(ndi.distance_transform_edt(~catalogue)[previous].min())
    }
    checks["positive_research_surface"] = bool(idx.any())
    checks["withheld_subset_of_known_catalogue"] = 0 < stats["withheld_positive_pixels"] <= int(catalogue.sum())
    result = {"evidence_class": "INDEPENDENT-READBACK-AND-SCOPE", "file": card["raster_path"],
              "checks": checks, "stats": stats, "all_checks_passed": bool(all(checks.values())),
              "notes": ["Verified the two fetched pinned witnesses and this repository's TIFF byte hashes, not a complete registry scan.",
                        "A local format PASS and an independent dense-raster overlap of 1.0 do not authorize competition upload.",
                        "H33 raster measurements are not an organizer submission-page receipt."]}
    (ROOT / "evidence/h77l_independent_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(audit())
