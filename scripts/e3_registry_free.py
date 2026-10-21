#!/usr/bin/env python3
"""Map the cells that no informative registry raster covers within 3 px (E3 uniqueness repair).

A registry raster whose 3 px (Euclidean) halo covers >= 70% of the allowed footprint cannot
discriminate duplicates: even uniformly random dots exceed the 0.70 overlap limit against
it, so the literal test fires on every possible candidate.  Such rasters are classed
SPACE-FILLING (reported, not hidden).  A halo of radius 3 px holds at most 29 cells, so only
rasters with >= 0.70 * allowed / 29 positives can be space-filling; their coverage is
measured exactly, the rest are informative by construction.

If >= 34% of a candidate's dots lie outside the union of informative halos, its overlap with
EVERY informative raster is <= 66% whatever the other dots do.
Writes .cache/work/registry_free.npy and evidence/e3_registry_free.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
from gems57.uniqueness import _near  # noqa: E402  (template halo definition, Euclidean 3 px)
from gems77 import core  # noqa: E402


def main():
    g = core.grid()
    allowed = g.footprint & ~g.catalogue
    n_allowed = int(allowed.sum())
    min_n = int(np.ceil(0.70 * n_allowed / 29))
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    union = np.zeros((core.H, core.W), bool)
    space_filling, informative, unreadable = [], 0, []
    for r in idx["rasters"]:
        try:
            p = core.read_prediction(core.REG / f"{r['blob']}.tif") > 0
        except Exception as exc:  # malformed / wrong-grid files (template audit also skips them)
            unreadable.append({"raster": r["aliases"][0], "error": str(exc)[:160]})
            continue
        n = int(p.sum())
        if n == 0:
            continue
        if n >= min_n:
            cov = float(_near(p)[allowed].mean())
            if cov >= 0.70:
                space_filling.append({"raster": r["aliases"][0], "positives": n, "coverage_3px": cov})
                continue
        union |= p
        informative += 1
    covered = _near(union)
    free = allowed & ~covered
    np.save(core.WORK / "registry_free.npy", free)
    out = {"allowed_cells": n_allowed, "space_filling_threshold_positives": min_n,
           "informative_rasters": informative, "space_filling_rasters": space_filling,
           "unreadable_rasters": unreadable,
           "free_cells": int(free.sum()), "free_fraction_of_allowed": float(free.mean() / allowed.mean())}
    (ROOT / "evidence" / "e3_registry_free.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("space_filling_rasters", "unreadable_rasters")}, indent=1),
          "unreadable:", len(unreadable),
          "space-filling:", len(space_filling))


if __name__ == "__main__":
    main()
