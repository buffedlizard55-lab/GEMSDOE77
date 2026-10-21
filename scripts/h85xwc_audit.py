#!/usr/bin/env python3
"""H85-XWC independent audit: re-derive Exp3 receipt from the bytes on disk.

Re-reads the published GeoTIFF (never rewrites it), re-runs the shared
validator and the full local uniqueness scan, and writes
``evidence/h85xwc_exp3_build.json`` as strict JSON (NaN -> null).
This is the automated, no-human-inspector gate for the lane.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import uniqueness, validate  # noqa: E402
from gems57.grid import load_grid  # noqa: E402

NAME = "gems77-h85xwc-magxgrav-coherence-37654dots-20261021-zeros"
TIF = ROOT / "docs" / "downloads" / f"{NAME}.tif"


def clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main() -> None:
    grid = load_grid()
    surf = np.load(ROOT / ".cache" / "h85" / "surfaces.npz")["xwc"]
    with rasterio.open(TIF) as src:
        dots = src.read(1) > 0
    sha = hashlib.sha256(TIF.read_bytes()).hexdigest()
    val = validate.validate(TIF, grid.footprint, grid.catalogue)

    rows, errors = [], []
    paths = sorted((ROOT / "docs" / "downloads").glob("*.tif"))
    extra = [ROOT / "data" / "raw" / "scored" /
             "gems24-h25-1-dotted-h19-5-d2-8-20261002-e56ea318af89-nan.tif",
             ROOT / "data" / "raw" / "scored" /
             "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif"]
    fp_n = int(grid.footprint.sum())
    for path in paths + [p for p in extra if p.exists()]:
        if path == TIF:
            continue
        try:
            with rasterio.open(path) as src:
                if (src.height, src.width) != grid.footprint.shape or src.count != 1:
                    errors.append(f"{path.name}: grid/band mismatch, skipped")
                    continue
                theirs = src.read(1)
            pos = int((np.isfinite(theirs) & (theirs > 0) & grid.footprint).sum())
            frac = pos / fp_n
            rho = uniqueness.surface_rho(surf, theirs, grid.footprint)
            rows.append({
                "file": path.name, "positive_in_footprint": pos,
                "positive_fraction": frac,
                "surface_spearman_full_footprint": rho["spearman_full_footprint"],
                "forward_dot_overlap_3px": uniqueness.dot_overlap(dots, theirs),
                "class": "sparse_dot_raster" if frac <= 0.10 else "dense_surface",
            })
            print(f"  {path.name[:60]:60s} frac={frac:.4f}", flush=True)
        except Exception as exc:
            errors.append(f"{path.name}: {exc!r}")

    sparse = [r for r in rows if r["class"] == "sparse_dot_raster"]
    dense = [r for r in rows if r["class"] == "dense_surface"]

    def _max(rs, k):
        vals = [abs(r[k]) for r in rs
                if isinstance(r[k], float) and math.isfinite(r[k])]
        return float(max(vals)) if vals else None

    receipt = clean({
        "experiment": "H85-XWC Exp3 final build + uniqueness + validation (audit re-derivation)",
        "evidence_class": "FORMAT/REGISTRY MEASUREMENT; NOT A SCORE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file": TIF.name, "sha256": sha,
        "dots": int(dots.sum()),
        "dots_on_catalogue": int((dots & grid.catalogue).sum()),
        "min_catalogue_distance_px": float(
            ndi.distance_transform_edt(~grid.catalogue)[dots].min()),
        "validator": val,
        "uniqueness": {
            "rasters_compared": len(rows), "errors": errors,
            "sparse_dot_rasters": len(sparse), "dense_surfaces": len(dense),
            "max_abs_surface_rho_sparse": _max(sparse, "surface_spearman_full_footprint"),
            "max_abs_surface_rho_dense": _max(dense, "surface_spearman_full_footprint"),
            "max_forward_dot_overlap_3px_sparse": _max(sparse, "forward_dot_overlap_3px"),
            "max_forward_dot_overlap_3px_dense_literal": _max(dense, "forward_dot_overlap_3px"),
            "worst_sparse_by_overlap": max(
                sparse, key=lambda r: r["forward_dot_overlap_3px"], default=None),
            "rows": rows,
            "irregularity_IR_H85_01": (
                "Literal reading: dense continuous registry surfaces positive on ~100% of "
                "allowed cells force forward dot overlap near 1.0 for ANY non-empty candidate "
                "(incl. H33 itself); that gate is reported verbatim and remains the protocol "
                "owner's to resolve. Informative reading applies the protocol's two tests as "
                "typed: Spearman rho to surfaces, 3px dot overlap to dot rasters."),
        },
    })
    out = ROOT / "evidence" / "h85xwc_exp3_build.json"
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("wrote", out, "sha256", sha)


if __name__ == "__main__":
    main()
