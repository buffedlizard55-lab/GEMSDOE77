#!/usr/bin/env python3
"""Build and audit the H85 controlled research raster.

H85 is a deliberately narrow, catalogue-only feasibility lane.  It is *not* a
replacement for the missing competition feature stack and it is not an
authorization to upload.  The surface is a multi-scale medial-ridge score of
the visible mapped-fault mask:

* Euclidean distance transform of the mapped-fault layer;
* a 0.8--3.5 km far-field annulus, chosen before writing the file;
* local maxima of the distance transform (a relay/medial-zone signature); and
* an absolute Gaussian-Laplacian term at two scales.

The shared writer is used unchanged.  The pinned registry preflight is run
before the artifact is retained.  If the hash-pinned dense witness blocks the
surface, this script records that duplicate and continues only to write a
research-only, format-valid negative artifact so the requested download path
is concrete.  It never places final dots, never calls a selector, and never
claims a holdout score.  A generated artifact is therefore explicitly
``OK TO DOWNLOAD FOR INSPECTION`` and ``NOT OK TO SUBMIT``.

This is intentionally different from the prior dot allocators: it is a
continuous probability surface from a distance-transform medial ridge, not a
copied raster, an H19/H33 subset, a DEM scarp, a radiometric stack, or a
learned model.  The no-dense-witness policy must be resolved before a future
candidate can be promoted.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57.grid import load_grid  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems57.uniqueness import dot_overlap  # noqa: E402
from gems57.validate import validate  # noqa: E402
from scripts.preflight_registry import audit as audit_preflight  # noqa: E402

STAMP = "20261021"
LANE = "H85-FTMR: far-field fault-network medial-ridge surface"
NOTE = (
    "H85 far-field medial-ridge research surface; format PASS; Policy I17 dense-witness duplicate; DO NOT SUBMIT."
)


def _robust_unit(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map finite values under ``mask`` to [0, 1] without percentile leakage."""
    out = np.zeros(values.shape, dtype=np.float32)
    x = np.asarray(values, dtype=np.float64)[mask]
    if x.size == 0:
        return out
    lo, hi = np.percentile(x, [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[mask] = 1.0
        return out
    out[mask] = np.clip((values[mask] - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out


def build_surface(catalogue: np.ndarray, footprint: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the preregistered H85 surface and measurement metadata.

    ``catalogue`` and ``footprint`` are masks only; no hidden labels or
    catalogue-derived training features are introduced.  The function accepts
    small arrays for unit tests, while the CLI uses the pinned 3730 x 3292 grid.
    """
    cat = np.asarray(catalogue, dtype=bool)
    fp = np.asarray(footprint, dtype=bool)
    if cat.ndim != 2 or cat.shape != fp.shape or not fp.any():
        raise ValueError("catalogue and footprint must be matching non-empty 2-D masks")
    if (cat & ~fp).any():
        raise ValueError("catalogue mask must be contained in footprint")
    if not cat.any():
        raise ValueError("H85 needs at least one mapped-fault pixel")

    # Distance is measured to the nearest mapped-fault cell.  A Gaussian
    # smoothing of the distance field is used only for the curvature signature;
    # it is not a fitted or hidden-label-derived probability.
    distance = ndi.distance_transform_edt(~cat).astype(np.float32)
    allowed = fp & ~cat
    annulus = allowed & (distance >= 8.0) & (distance <= 35.0)

    # A medial/relay candidate is a local maximum of distance from the mapped
    # network.  The two Laplacian scales suppress a single-pixel raster edge
    # and retain only multi-scale concavity/convexity transitions.
    medial = distance >= ndi.maximum_filter(distance, size=7, mode="nearest")
    lap2 = np.abs(ndi.laplace(ndi.gaussian_filter(distance, sigma=2.0)))
    lap4 = np.abs(ndi.laplace(ndi.gaussian_filter(distance, sigma=4.0)))
    lap = 0.65 * _robust_unit(lap2, annulus) + 0.35 * _robust_unit(lap4, annulus)
    radial = np.exp(-0.5 * ((distance - 18.0) / 9.0) ** 2).astype(np.float32)
    score = np.where(annulus & medial, 0.55 * radial + 0.45 * lap, 0.0).astype(np.float32)
    score[~allowed] = 0.0
    score[~np.isfinite(score)] = 0.0
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    # Do not emit positive mapped-fault or out-of-footprint cells even if an
    # upstream array implementation changes its boundary convention.
    score[cat | ~fp] = 0.0
    return score, {
        "operator": "distance_transform + 7px medial maximum + abs Gaussian Laplacian sigma 2/4px",
        "distance_annulus_px": [8.0, 35.0],
        "distance_peak_px": 18.0,
        "positive_surface_pixels": int((score > 0).sum()),
        "surface_min": float(score.min()),
        "surface_max": float(score.max()),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _surface_spearman(a: np.ndarray, b: np.ndarray, footprint: np.ndarray) -> float | None:
    """Compute rank correlation without importing a second registry policy."""
    from scipy.stats import spearmanr

    m = np.asarray(footprint, bool) & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return None
    return float(spearmanr(a[m], b[m]).statistic)


def build_run_card(*, output: Path, zip_path: Path, receipt: dict, validator: dict,
                   surface_meta: dict, preflight: dict, surface_overlap: float,
                   h33_rho: float | None) -> dict:
    dense_blocker = bool(preflight.get("proof", {}).get("universal_overlap_blocker"))
    return {
        "schema": "gems-h85-run-card-v1",
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/eedbff3b-gemsdoe77",
        "lane": LANE,
        "hypothesis": (
            "Unmapped blind faults may occupy far-field relay corridors where the mapped-fault distance field "
            "forms a multi-scale medial ridge; a distance-transform ridge can be tested without reusing a prior "
            "H19/H33 dot set."
        ),
        "mechanism": (
            "The H85 surface combines an 0.8--3.5 km distance annulus, 7-pixel local maxima of Euclidean "
            "distance to the mapped-fault network, and absolute Gaussian-Laplacian curvature at sigma 2 and 4 px."
        ),
        "named_non_fault_process_that_could_mimic_it": (
            "Pluvial shoreline benches, alluvial-fan toes, resistant lithologic boundaries, and the edge of a "
            "masked raster footprint can all create distance-transform medial ridges without a fault."
        ),
        "holdout_dti": {
            "evidence_class": "NO HOLDOUT-DTI RESULT — stopped before experiment by the universal duplicate preflight",
            "evaluator_version": "gems57-pooled-hide-v2 (not run)",
            "protocol_if_authorized": {
                "hidden_unit": "whole fault segments",
                "buffer_px": 3,
                "catalogue_features_from_visible_faults_only": True,
                "visible_faults_masked_exactly": True,
                "metric": "pooled DTI",
                "alpha": 0.2,
                "beta": 0.8,
                "triangular_kernel_radius_m": 300,
            },
            "withheld_positive_pixels": None,
            "score": None,
            "ci95": None,
        },
        "leakage_canary": {
            "evidence_class": "NOT RUN — no holdout experiment was authorized",
            "features_tested": [],
            "auc_max": None,
            "threshold": 0.90,
        },
        "correlation_overlap_vs_registry": {
            "evidence_class": "REGISTRY-MEASUREMENT",
            "surface_preplacement_spearman_vs_h33": h33_rho,
            "surface_preplacement_forward_3px_overlap_vs_dense_witness": surface_overlap,
            "rho_limit": 0.90,
            "overlap_limit": 0.70,
            "dense_witness_sha256": preflight.get("input_sha256", {}).get("witness"),
            "duplicate_stop": bool(dense_blocker or surface_overlap > 0.70 or (h33_rho is not None and abs(h33_rho) > 0.90)),
            "preplacement_uniqueness_check": {
                "status": "MEASURED; FAILS PROMOTION GATE",
                "surface_forward_overlap": surface_overlap,
                "surface_rank_correlation_vs_h33": h33_rho,
                "evidence_class": "REGISTRY-MEASUREMENT; not a score",
            },
            "postplacement_final_dot_check": {
                "status": "NOT RUN — no final dots were placed after the mandatory preplacement stop",
                "required_before_weekly_slot": True,
                "evidence_class": "NO FINAL-DOT MEASUREMENT",
            },
            "registry_scope": "two restored hash-pinned witnesses only; not a complete organizer registry",
            "final_dots": "not placed after surface duplicate stop",
        },
        "raster": {
            "file": str(output.relative_to(ROOT)),
            "zip_file": str(zip_path.relative_to(ROOT)),
            "sha256": receipt["sha256"],
            "zip_sha256": receipt["zip_sha256"],
            "bytes": receipt["bytes"],
            "positive_surface_pixels": surface_meta["positive_surface_pixels"],
        },
        "raster_sha256": receipt["sha256"],
        "validator_output": {
            "validator": "gems57.validate.validate + shared submission_writer readback",
            "all_checks_passed": bool(validator["all_checks_passed"]),
            "no_nan_inside_footprint": bool(validator["checks"].get("in_footprint_all_finite")),
            "values_in_0_1": bool(validator["checks"].get("in_footprint_range_0_1")),
            "crs_shape_transform_match": bool(
                validator["checks"].get("crs_epsg_32611")
                and validator["checks"].get("dimensions_3730x3292")
                and validator["checks"].get("transform_matches_sample_submission")
            ),
            "checks": validator["checks"],
        },
        "surface_operator": surface_meta,
        "submission_name": receipt["submission_name"],
        "submission_note": receipt["note"],
        "note_chars": receipt["note_chars"],
        "ok_to_download_for_inspection": True,
        "ok_to_submit": False,
        "selector_approved": False,
        "selector_decision": "NOT REQUESTED — preflight stopped the lane; selector approval is separate from the weekly cap",
        "weekly_slots_used": 0,
        "weekly_slot_check": "NOT APPLICABLE — no selector-approved upload was requested",
        "experiments_run": 0,
        "verdict": "negative",
        "verdict_reason": (
            "The artifact is byte-distinct and format-valid, but the mandatory literal 3-pixel registry rule "
            "fires on the restored dense witness before final-dot placement. No holdout score exists and no "
            "selector approved a slot. Do not upload."
        ),
        "preflight": preflight,
    }


def main() -> int:
    grid = load_grid()
    preflight = audit_preflight(ROOT)
    if preflight["proof"]["universal_overlap_blocker"] is not True:
        raise SystemExit("H85 expects the pinned universal blocker; refusing to invent clearance")

    # The preflight is the surface stop.  We still create one deterministic
    # research-only file so the owner can inspect the exact format path.  No
    # final dots are selected, and this branch never calls the holdout evaluator.
    surface, surface_meta = build_surface(grid.catalogue, grid.footprint)
    dense_path = ROOT / ".cache" / "registry" / "dense_witness.tif"
    h33_path = ROOT / ".cache" / "registry" / "h33.tif"
    import rasterio

    with rasterio.open(dense_path) as src:
        dense = src.read(1).astype(np.float32)
    with rasterio.open(h33_path) as src:
        h33 = src.read(1).astype(np.float32)
    surface_overlap = dot_overlap(surface > 0, dense > 0)
    h33_rho = _surface_spearman(surface, h33, grid.footprint)

    pred_digest = hashlib.sha256(np.ascontiguousarray(surface).tobytes()).hexdigest()
    name = f"gemsdoe77-h85-medial-ridge-{pred_digest[:8]}-RESEARCH-ONLY"
    out = ROOT / "docs" / "downloads" / f"{name}.tif"
    siblings = (out, out.with_suffix(".zip"), out.with_suffix(".json"))
    existing_artifact = any(p.exists() for p in siblings)
    if existing_artifact:
        # A tracked artifact is safe to re-audit, but never silently overwrite it.
        if not all(p.is_file() for p in siblings):
            raise FileExistsError(f"partial H85 output set; refusing to repair automatically: {siblings}")
        with rasterio.open(out) as src:
            existing = src.read(1)
        if not np.array_equal(existing, surface):
            raise ValueError("existing H85 TIFF differs from the deterministic surface; refusing to overwrite")
        with zipfile.ZipFile(out.with_suffix(".zip")) as archive:
            if archive.namelist() != [out.name] or archive.read(out.name) != out.read_bytes():
                raise ValueError("existing H85 ZIP is not a single-TIFF byte-identical package")
        receipt = json.loads(out.with_suffix(".json").read_text())
        if receipt.get("sha256") != _sha256(out):
            raise ValueError("existing H85 receipt does not match the TIFF; refusing to overwrite")
    else:
        receipt = write_submission(
            out,
            surface,
            ROOT / "data" / "bridge" / "sample_submission.tif",
            grid.footprint,
            catalogue=grid.catalogue,
            name=name,
            note=NOTE,
            metadata={
                "lane": LANE,
                "research_only": True,
                "surface_preplacement_duplicate": True,
                "preflight_decision": preflight["decision"],
                "final_dots": "not placed",
            },
        )
    validator = validate(out, grid.footprint, grid.catalogue)
    if not validator["all_checks_passed"]:
        raise ValueError("H85 artifact failed its shared readback validation")
    card = build_run_card(
        output=out,
        zip_path=out.with_suffix(".zip"),
        receipt=receipt,
        validator=validator,
        surface_meta=surface_meta,
        preflight=preflight,
        surface_overlap=surface_overlap,
        h33_rho=h33_rho,
    )
    evidence = ROOT / "evidence" / "h85_run_card.json"
    evidence.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # The Pages tree mirrors machine-readable evidence; it is not a second card.
    docs_evidence = ROOT / "docs" / "evidence" / "h85_run_card.json"
    docs_evidence.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(evidence, docs_evidence)
    print(json.dumps({
        "artifact": str(out.relative_to(ROOT)),
        "zip": str(out.with_suffix(".zip").relative_to(ROOT)),
        "sha256": receipt["sha256"],
        "positive_surface_pixels": surface_meta["positive_surface_pixels"],
        "surface_overlap_vs_dense_witness": surface_overlap,
        "h33_surface_spearman": h33_rho,
        "validator_all_checks_passed": validator["all_checks_passed"],
        "ok_to_download_for_inspection": True,
        "ok_to_submit": False,
        "verdict": "negative",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
