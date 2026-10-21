#!/usr/bin/env python3
"""Measure how the registry overlap gate depends on what counts as a 'dot'.

REGISTRY MEASUREMENT only: not HOLDOUT-DTI, not an organizer score, not a
uniqueness clearance and not a candidate test. It reports the fraction of the
allowed footprint lying within 3 px of each pinned witness under (a) the shared
definition (every finite positive pixel is a dot) and (b) a binary top-N
convention (the N highest-valued pixels are dots). It exists so the owner can
decide the support semantics with numbers in hand. Requires the pinned witnesses
from `scripts/restore_registry.py` (.cache/registry/).
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
from gems57.uniqueness import OVERLAP_LIMIT, _dots, _near  # noqa: E402

TOP_N = 37654  # dot budget used across the GEMSDOE77 lanes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "evidence/support_semantics_sensitivity.json")
    args = ap.parse_args()
    pins = json.loads((ROOT / "registry/witness_sources.json").read_text())
    with rasterio.open(ROOT / "data/bridge/sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
    with rasterio.open(ROOT / "data/bridge/existing_faults.tif") as s:
        catalogue = _dots(s.read(1)) & footprint
    allowed = footprint & ~catalogue
    n_allowed = int(allowed.sum())
    result = {
        "schema": "gems-support-semantics-sensitivity-v1",
        "evidence_class": "REGISTRY-MEASUREMENT; NOT HOLDOUT-DTI OR ORGANIZER SCORE",
        "allowed_pixels": n_allowed,
        "overlap_stop_threshold_strictly_greater_than": OVERLAP_LIMIT,
        "top_n_binary_convention": TOP_N,
        "witnesses": {},
    }
    for key, rel in (("dense_witness", ".cache/registry/dense_witness.tif"),
                     ("h33", ".cache/registry/h33.tif")):
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing {rel}; run scripts/restore_registry.py first")
        digest = sha256(path)
        if digest != pins[key]["sha256"]:
            raise SystemExit(f"hash mismatch for {rel}")
        with rasterio.open(path) as s:
            raw = s.read(1)
        values = np.where(footprint, np.where(np.isfinite(raw), raw, 0), 0)
        shared = _dots(values)
        frac_shared = float((allowed & _near(shared)).sum() / n_allowed)
        vals = values[allowed]
        thr = np.partition(vals, -TOP_N)[-TOP_N]
        topn = allowed & (values >= thr)
        frac_topn = float((allowed & _near(topn)).sum() / n_allowed)
        result["witnesses"][key] = {
            "sha256": digest,
            "shared_definition_positive_pixels": int(shared.sum()),
            "shared_definition_within_3px_fraction": frac_shared,
            "top_n_within_3px_fraction": frac_topn,
            "top_n_dot_count": int(topn.sum()),
            "exceeds_0_70_under_shared_definition": frac_shared > OVERLAP_LIMIT,
            "exceeds_0_70_under_top_n_convention": frac_topn > OVERLAP_LIMIT,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
