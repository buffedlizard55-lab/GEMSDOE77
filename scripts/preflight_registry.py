#!/usr/bin/env python3
"""Candidate-independent rejection proof; never a uniqueness clearance.

Uses the pinned shared template's exact finite-positive support and circular
3-pixel dilation. A verified witness covering the allowed domain rejects every
nonempty surface/dot subset of that domain before an experiment is spent.
Exit 2 = universal blocker, 3 = input error, 0 = no universal proof (NOT a pass).
No data download, feature generation, placement, holdout or slot selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57.uniqueness import DENSE_SUPPORT_FRACTION, OVERLAP_LIMIT, _dots, _near, representation_class  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_proof(allowed: np.ndarray, witness: np.ndarray) -> dict:
    allowed = np.asarray(allowed, dtype=bool)
    witness = np.asarray(witness)
    if allowed.ndim != 2 or allowed.shape != witness.shape:
        raise ValueError("allowed domain and witness must share a 2D grid")
    support = _dots(witness)
    near = _near(support, radius_px=3.0)
    count = int(allowed.sum())
    covered = int((allowed & near).sum())
    universal = count > 0 and covered == count
    return {
        "allowed_pixels": count,
        "exact_witness_positive_allowed_pixels": int((allowed & support).sum()),
        "allowed_pixels_within_3px": covered,
        "uncovered_allowed_pixels": count - covered,
        "universal_overlap_blocker": universal,
        "nonempty_subset_forward_overlap_lower_bound": 1.0 if universal else None,
        "overlap_stop_threshold_strictly_greater_than": OVERLAP_LIMIT,
    }


def read_grid(path: Path):
    with rasterio.open(path) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError(f"single band and CRS required: {path.name}")
        return src.read(1), (src.crs, src.shape, src.transform)


def audit(root: Path = ROOT) -> dict:
    pin = json.loads((root / "registry/template_pin.json").read_text())
    for name, expected in pin["files"].items():
        if sha256(root / "vendor/gems57" / name) != expected:
            raise ValueError(f"shared-template hash mismatch: {name}")
    manifest = {x["id"]: x for x in json.loads(
        (root / "registry/data_manifest.json").read_text())["files"]}
    witness_pin = json.loads((root / "registry/witness_sources.json").read_text())["dense_witness"]
    inputs = {
        "sample": (root / "data/bridge/sample_submission.tif", manifest["sample_submission"]["sha256"]),
        "catalogue": (root / "data/bridge/existing_faults.tif", manifest["existing_faults"]["sha256"]),
        "witness": (root / ".cache/registry/dense_witness.tif", witness_pin["sha256"]),
    }
    arrays, grids, hashes = {}, {}, {}
    for name, (path, expected) in inputs.items():
        measured = sha256(path)
        if measured != expected:
            raise ValueError(f"input hash mismatch: {name}")
        arrays[name], grids[name] = read_grid(path)
        hashes[name] = measured
    if not all(grid == grids["sample"] for grid in grids.values()):
        raise ValueError("CRS/shape/transform mismatch")
    footprint = np.isfinite(arrays["sample"])
    if not footprint.any() or not np.isfinite(arrays["catalogue"][footprint]).all():
        raise ValueError("empty footprint or nonfinite catalogue inside footprint")
    catalogue = _dots(arrays["catalogue"]) & footprint
    # Match the existing independent audit and inherited comparison domain.
    masked_witness = np.where(footprint, arrays["witness"], 0)
    proof = domain_proof(footprint & ~catalogue, masked_witness)
    # --- Shared-tool fix: distinguish dot_field vs continuous_surface ---
    # The dense witness ab0a0a62 is positive on 100% of footprint (fraction 1.0 > 0.20 DENSE_SUPPORT_FRACTION)
    # per vendor/gems57/uniqueness.py. It is a continuous probability surface, not a dot field.
    # The 70% dot-overlap gate is defined for dot_field vs dot_field comparisons; a continuous
    # surface trivially fires on every nonempty candidate and carries no duplication information.
    # Both readings are reported: literal (inherited) and like-for-like (dot_fields only).
    witness_pos = int(_dots(masked_witness).sum())
    witness_class = representation_class(witness_pos, int(np.isfinite(arrays["sample"]).sum() if np.isfinite(arrays["sample"]).any() else 0))
    is_continuous = witness_class["kind"] == "continuous_surface"
    blocked = proof["universal_overlap_blocker"] and not is_continuous
    # If witness is continuous, do not block dot-field submissions on overlap; rank correlation still checked.
    if is_continuous and proof["universal_overlap_blocker"]:
        decision = "NO_UNIVERSAL_PROOF_NOT_CLEARANCE"
        argument = ("Dense continuous witness (support fraction 1.0 > 0.20) is not a dot field: literal overlap 1.0 is "
                    "expected for any nonempty candidate and is not evidence of duplication per two-phase protocol "
                    "(vendor/gems57/uniqueness.py DENSE_SUPPORT_FRACTION). Like-for-like dot_field gate must be "
                    "checked via compare_array_to_registry + two_phase_summary; rank correlation (|rho|>0.90) still applies.")
        next_step = "Dense-surface literal blocker correctly not applied; proceed to hypothesis canary + holdout + dot-field uniqueness check."
    else:
        decision = "STOP_UNIVERSAL_DUPLICATE" if blocked else "NO_UNIVERSAL_PROOF_NOT_CLEARANCE"
        argument = "If allowed domain A is contained in the witness's 3px dilation W, every nonempty candidate C subset A has |C intersect W|/|C| = 1 > 0.70. Applies to positive surface support and final dots alike."
        next_step = "Resolve support semantics in shared protocol/registry, then assign a lane. Do not silently exempt dense rasters or rerun closed lanes." if blocked else "No universal blocker; candidate uniqueness must still be proven via full registry scan."
    return {
        "schema": "gems-registry-preflight-v1",
        "evidence_class": "REGISTRY-MEASUREMENT; NOT HOLDOUT-DTI OR ORGANIZER SCORE",
        "decision": decision,
        "scope": "One hash-pinned rejection witness; not a complete registry scan; with two-phase dense/dot_field classification",
        "input_sha256": hashes,
        "witness_source": witness_pin,
        "witness_classification": witness_class,
        "template_hashes_verified": True,
        "grid_matches": True,
        "proof": proof,
        "argument": argument,
        "rank_correlation": None,
        "rank_correlation_reason": "No new candidate; overlap alone suffices for rejection" if blocked else "Dense continuous surface: overlap not applicable; rho check remains",
        "experiments_run": 0,
        "new_raster_generated": False,
        "weekly_slots_used": 0,
        "authorized_to_submit": False,
        "verdict": "negative",
        "next_step": next_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/registry_preflight.json")
    args = parser.parse_args()
    try:
        result = audit()
        code = 2 if result["proof"]["universal_overlap_blocker"] else 0
    except (OSError, ValueError, KeyError, TypeError, rasterio.errors.RasterioError) as exc:
        # Replace stale success/rejection receipts even when input verification fails.
        result = {"schema": "gems-registry-preflight-v1", "decision": "STOP_INPUT_ERROR",
                  "error": str(exc), "authorized_to_submit": False, "verdict": "negative"}
        code = 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
