"""DrivenData GeoTIFF Submission Builder & Range-[0, 1] Validator Auditor.

Permanently fixes the DrivenData error `"Predicted values must be in range [0, 1]"` by:
1. Sanitizing all float32 `-3.4028235e+38` sentinels and NaNs inside the 5,167,373-pixel footprint.
2. Enforcing `np.clip(pred, 0.0, 1.0)` across all 5,167,373 in-footprint pixels and `pred[labels] = 0.0`.
3. Emitting paired GeoTIFFs for each candidate:
   - `-zeros.tif`: `0.0` outside footprint (`nodata=None`), so all 12,279,160 / 12,279,160 raster cells
     satisfy `(arr >= 0.0) & (arr <= 1.0)` with zero NaNs anywhere.
   - `-nan.tif`: `NaN` outside footprint (`nodata=np.nan`), matching the historical `*-nan.tif` convention
     while guaranteeing zero NaNs or out-of-range values inside the 5,167,373-pixel footprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

EXPECTED_HEIGHT = 3730
EXPECTED_WIDTH = 3292
EXPECTED_TOTAL_PIXELS = EXPECTED_HEIGHT * EXPECTED_WIDTH
EXPECTED_FOOTPRINT_PIXELS = 5_167_373


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_geotiff(
    tif_path: Path,
    foot: np.ndarray,
    labels: np.ndarray,
    mode: str = "zeros",
) -> dict:
    """Strict 12-point DrivenData submission validator audit."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        crs_str = str(src.crs)
        transform_tuple = tuple(round(float(x), 4) for x in src.transform)[:6]
        dtypes = src.dtypes
        count = src.count
        height, width = src.height, src.width
        nodata_val = src.nodata

    in_foot = arr[foot]
    out_foot = arr[~foot]

    in_foot_finite = int(np.isfinite(in_foot).sum())
    in_foot_nan = int(np.isnan(in_foot).sum())
    in_foot_inf = int(np.isinf(in_foot).sum())
    in_foot_sentinel = int((in_foot < -1e30).sum())
    in_foot_min = float(np.nanmin(in_foot))
    in_foot_max = float(np.nanmax(in_foot))
    in_foot_in_01 = bool(
        (in_foot_finite == EXPECTED_FOOTPRINT_PIXELS)
        and np.all((in_foot >= 0.0) & (in_foot <= 1.0))
    )

    if mode == "zeros":
        full_grid_in_01 = bool(
            (int(np.isfinite(arr).sum()) == EXPECTED_TOTAL_PIXELS)
            and np.all((arr >= 0.0) & (arr <= 1.0))
        )
        out_foot_ok = bool(np.all(out_foot == 0.0))
    else:
        full_grid_in_01 = in_foot_in_01
        out_foot_ok = bool(np.all(np.isnan(out_foot)))

    emitted_dots = int((in_foot > 0.5).sum())
    on_catalogue_dots = int(((arr > 0.5) & labels).sum())

    checks = {
        "single_band": count == 1,
        "dtype_float32": dtypes[0] == "float32",
        "dimensions_3730x3292": (height == EXPECTED_HEIGHT and width == EXPECTED_WIDTH),
        "crs_epsg_32611": "32611" in crs_str,
        "in_footprint_all_finite": in_foot_finite == EXPECTED_FOOTPRINT_PIXELS,
        "in_footprint_zero_nan": in_foot_nan == 0,
        "in_footprint_zero_inf": in_foot_inf == 0,
        "in_footprint_zero_sentinel": in_foot_sentinel == 0,
        "in_footprint_range_0_1": in_foot_in_01,
        "outside_footprint_compliant": out_foot_ok,
        "zero_on_catalogue_leakage": on_catalogue_dots == 0,
        "validator_range_0_1_guaranteed": full_grid_in_01,
    }
    all_passed = all(checks.values())
    if not all_passed:
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Submission audit failed for {tif_path.name}: {failed}")

    return {
        "filename": tif_path.name,
        "size_bytes": tif_path.stat().st_size,
        "sha256": sha256_file(tif_path),
        "mode": mode,
        "crs": crs_str,
        "transform": list(transform_tuple),
        "shape": [height, width],
        "dtype": dtypes[0],
        "nodata": None if nodata_val is None or (isinstance(nodata_val, float) and np.isnan(nodata_val)) else float(nodata_val),
        "nodata_repr": str(nodata_val),
        "emitted_positive_pixels": emitted_dots,
        "footprint_fraction": round(emitted_dots / float(EXPECTED_FOOTPRINT_PIXELS), 6),
        "on_catalogue_positive_pixels": on_catalogue_dots,
        "in_footprint_finite_pixels": in_foot_finite,
        "in_footprint_min": in_foot_min,
        "in_footprint_max": in_foot_max,
        "full_grid_finite_pixels": int(np.isfinite(arr).sum()),
        "checks": checks,
        "all_checks_passed": all_passed,
    }


def write_submission_pair(
    mask: np.ndarray,
    foot: np.ndarray,
    labels: np.ndarray,
    template_tif: Path,
    out_dir: Path,
    slug: str,
    timestamp_tag: str,
    candidate_meta: dict,
) -> dict:
    """Write and audit both `-zeros.tif` and `-nan.tif` variants for a candidate mask."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_mask = (mask & foot & ~labels).astype(np.float32)
    # Digest of pixel coordinates for deterministic content hash tag
    digest8 = hashlib.sha256(np.packbits(clean_mask > 0.5)).hexdigest()[:8]

    zeros_name = f"gemsdoe32-{slug}-{timestamp_tag}-{digest8}-zeros.tif"
    nan_name = f"gemsdoe32-{slug}-{timestamp_tag}-{digest8}-nan.tif"
    zeros_path = out_dir / zeros_name
    nan_path = out_dir / nan_name

    with rasterio.open(template_tif) as tpl:
        profile_base = tpl.profile.copy()

    profile_zeros = profile_base.copy()
    profile_zeros.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=3,
        zlevel=9,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=None,
    )
    arr_zeros = np.where(foot, np.clip(clean_mask, 0.0, 1.0), 0.0).astype(np.float32)
    with rasterio.open(zeros_path, "w", **profile_zeros) as dst:
        dst.write(arr_zeros, 1)

    profile_nan = profile_zeros.copy()
    profile_nan.update(nodata=np.nan)
    arr_nan = np.where(foot, np.clip(clean_mask, 0.0, 1.0), np.nan).astype(np.float32)
    with rasterio.open(nan_path, "w", **profile_nan) as dst:
        dst.write(arr_nan, 1)

    audit_zeros = audit_geotiff(zeros_path, foot, labels, mode="zeros")
    audit_nan = audit_geotiff(nan_path, foot, labels, mode="nan")

    fmt_kwargs = dict(
        dots=audit_zeros["emitted_positive_pixels"],
        cat_hid=candidate_meta["catalogue_hidden_mean"],
        sgmc_cal=candidate_meta["sgmc_prevalence_calibrated_dti"],
        drift_cal=candidate_meta["drift_corrected_holdout_mean"],
        slot=candidate_meta["slot_decision"],
    )
    bundle = {
        "candidate_id": candidate_meta["candidate_id"],
        "slug": slug,
        "content_digest8": digest8,
        "description": candidate_meta["description"],
        "submission_note_zeros": candidate_meta["submission_note_zeros"].format(
            filename=zeros_name, sha8=audit_zeros["sha256"][:8], **fmt_kwargs
        ),
        "submission_note_nan": candidate_meta["submission_note_nan"].format(
            filename=nan_name, sha8=audit_nan["sha256"][:8], **fmt_kwargs
        ),
        "zeros_tif": audit_zeros,
        "nan_tif": audit_nan,
        "holdout_metrics": {
            "catalogue_hidden_mean": candidate_meta["catalogue_hidden_mean"],
            "catalogue_hidden_per_quadrant": candidate_meta["catalogue_hidden_per_quadrant"],
            "sgmc_prevalence_calibrated_dti": candidate_meta["sgmc_prevalence_calibrated_dti"],
            "drift_corrected_holdout_mean": candidate_meta["drift_corrected_holdout_mean"],
            "drift_corrected_per_quadrant": candidate_meta["drift_corrected_per_quadrant"],
            "predicted_leaderboard_dti": candidate_meta["predicted_leaderboard_dti"],
            "ei_drift_corrected": candidate_meta["ei_drift_corrected"],
            "slot_decision": candidate_meta["slot_decision"],
        },
    }

    sidecar_path = out_dir / f"gemsdoe32-{slug}-{timestamp_tag}-{digest8}-audit.json"
    sidecar_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle
