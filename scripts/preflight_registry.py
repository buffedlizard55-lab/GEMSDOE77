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
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57.uniqueness import OVERLAP_LIMIT, _dots, _near  # noqa: E402


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


def verify_template_pin(root: Path, pin: dict | None = None) -> None:
    """Verify upstream pins and any explicitly declared additive local hashes.

    A declared amendment is accepted only when it names the exact upstream hash
    already pinned in ``files`` and supplies an exact local SHA-256. This does
    not permit evaluator/threshold changes: the science test separately checks
    the amendment declaration, added symbols, unchanged constants and audit
    provenance. Undeclared local edits remain fail-closed.
    """
    if pin is None:
        pin = json.loads((root / "registry/template_pin.json").read_text())
    files = pin["files"]
    amendments = pin.get("local_amendments", {})
    if not isinstance(files, dict):
        raise ValueError("shared-template pin schema invalid")

    # Historical pins use a mapping keyed by filename. The H85 shared-template
    # pin records an audited amendment as a list so it can retain both upstream
    # and local hashes without pretending that the latter is unmodified. Accept
    # only these two explicit shapes; every other shape remains fail-closed.
    amendment_style = "mapping"
    if isinstance(amendments, list):
        amendment_style = "list"
        normalized = {}
        for declared in amendments:
            if not isinstance(declared, dict):
                raise ValueError("shared-template pin schema invalid")
            name = declared.get("file")
            upstream_sha = declared.get("upstream_sha256")
            local_sha = declared.get("local_sha256")
            if (not isinstance(name, str) or name in normalized or name not in files
                    or not isinstance(upstream_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", upstream_sha)
                    or not isinstance(local_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", local_sha)
                    or files[name] != local_sha
                    or not isinstance(declared.get("irregularity"), str)
                    or not isinstance(declared.get("change"), str)
                    or not isinstance(declared.get("tests"), str)):
                raise ValueError(f"invalid declared shared-template amendment: {name}")
            normalized[name] = declared
        amendments = normalized
    elif not isinstance(amendments, dict):
        raise ValueError("shared-template pin schema invalid")

    if set(amendments) - set(files):
        raise ValueError("local amendment has no upstream file pin")
    for name, upstream_sha in files.items():
        declared = amendments.get(name)
        expected = upstream_sha
        if declared is not None:
            if amendment_style == "mapping":
                if (not isinstance(declared, dict)
                        or declared.get("upstream_sha256") != upstream_sha
                        or declared.get("measurement_semantics_unchanged") is not True
                        or not isinstance(declared.get("local_sha256"), str)):
                    raise ValueError(f"invalid declared shared-template amendment: {name}")
            expected = declared["local_sha256"]
        if sha256(root / "vendor/gems57" / name) != expected:
            raise ValueError(f"shared-template hash mismatch: {name}")


def audit(root: Path = ROOT) -> dict:
    pin = json.loads((root / "registry/template_pin.json").read_text())
    verify_template_pin(root, pin)
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
    blocked = proof["universal_overlap_blocker"]
    return {
        "schema": "gems-registry-preflight-v1",
        "evidence_class": "REGISTRY-MEASUREMENT; NOT HOLDOUT-DTI OR ORGANIZER SCORE",
        "decision": "STOP_UNIVERSAL_DUPLICATE" if blocked else "NO_UNIVERSAL_PROOF_NOT_CLEARANCE",
        "scope": "One hash-pinned rejection witness; not a complete registry scan",
        "input_sha256": hashes,
        "witness_source": witness_pin,
        "template_hashes_verified": True,
        "grid_matches": True,
        "proof": proof,
        "argument": "If allowed domain A is contained in the witness's 3px dilation W, every nonempty candidate C subset A has |C intersect W|/|C| = 1 > 0.70. Applies to positive surface support and final dots alike.",
        "rank_correlation": None,
        "rank_correlation_reason": "No new candidate; overlap alone suffices for rejection",
        "experiments_run": 0,
        "new_raster_generated": False,
        "weekly_slots_used": 0,
        "authorized_to_submit": False,
        "verdict": "negative",
        "next_step": "Resolve support semantics in shared protocol/registry, then assign a lane. Do not silently exempt dense rasters or rerun closed lanes.",
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
