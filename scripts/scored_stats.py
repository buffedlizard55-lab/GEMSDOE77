#!/usr/bin/env python3
"""Per-raster structural statistics for every organizer-scored registry raster.

For each raster behind a score in registry/live_scores.csv (matched in
registry/registry_index.json) this records, from the bytes:
  * sha256, emitted cells (p > 0 in footprint), mass S, binary or continuous,
  * mass on catalogue pixels (masked by the organizer => pure waste of effort),
  * mass by distance band to the catalogue (pixels),
  * median nearest-neighbour spacing of emitted cells (dot pattern),
  * the cover field M_i(y) = max_x p(x) k(|x-y|) summed per distance band
    (the exact linear coefficient of a truth pixel in band b on TP_w).
Output: evidence/scored_stats.json (+ .npz with per-band design matrices).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, spatial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems77lia import core  # noqa: E402

BANDS = [0, 1, 2, 3, 4, 6, 8, 12, 20, 35, 60, 100, 1e9]  # catalogue-distance band edges (px)


def band_index(dcat: np.ndarray) -> np.ndarray:
    """0 = on catalogue (masked); 1.. = (edge[i-1], edge[i]] bands."""
    b = np.digitize(dcat, BANDS[1:], right=True) + 1
    b[dcat == 0] = 0
    return b.astype(np.int16)


def main() -> None:
    g = core.grid()
    dcat = core.catalogue_distance_px()
    bidx = band_index(dcat)
    bidx[~g.footprint] = -1
    nb = len(BANDS)
    cells_per_band = np.array([int((bidx == b).sum()) for b in range(nb)])
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    seen = {}
    rows = []
    A_cover, A_mass = [], []
    for e in idx["scored"]:
        path = core.REG / f"{e['blob']}.tif"
        if e["blob"] in seen:
            r = dict(seen[e["blob"]]); r.update({"submission_name": e["submission_name"],
                                                "site_repo": e["site_repo"], "public_score": e["public_score"],
                                                "duplicate_of_blob": True})
            rows.append(r)
            continue
        p = core.read_prediction(path)
        on_cat = float(p[g.catalogue].sum())
        p_eval = p.copy()
        p_eval[g.catalogue] = 0.0          # organizer masks known-fault pixels
        pos = p_eval > 0
        n = int(pos.sum())
        vals = p_eval[pos]
        binary = bool(np.all((vals == 1.0))) if n else True
        mass_band = np.bincount(bidx[bidx >= 0], weights=p_eval[bidx >= 0], minlength=nb)
        m = core.cover_field(p_eval)
        cover_band = np.bincount(bidx[bidx >= 0], weights=m[bidx >= 0], minlength=nb)
        cover_band[0] = 0.0                 # truth cannot sit on masked catalogue pixels
        rr, cc = np.nonzero(pos)
        if n > 1:
            sub = np.random.default_rng(0).choice(n, size=min(n, 20000), replace=False)
            tree = spatial.cKDTree(np.c_[rr, cc])
            d, _ = tree.query(np.c_[rr[sub], cc[sub]], k=2)
            nn_med = float(np.median(d[:, 1]))
        else:
            nn_med = float("nan")
        r = {"submission_name": e["submission_name"], "site_repo": e["site_repo"],
             "public_score": e["public_score"], "blob": e["blob"], "path": f"{e['repo']}:{e['path']}",
             "sha256": core.sha256_file(path), "emitted_cells": n, "mass": float(vals.sum()),
             "binary": binary, "max_value": float(vals.max()) if n else 0.0,
             "mass_on_catalogue_masked": on_cat,
             "mass_by_band": mass_band.tolist(), "cover_by_band": cover_band.tolist(),
             "median_nn_spacing_px": nn_med,
             "cover_total": float(m[g.footprint & ~g.catalogue].sum())}
        seen[e["blob"]] = r
        rows.append(r)
        print(f"{e['public_score']:.4f} n={n:>8} S={r['mass']:>10.0f} bin={binary!s:5} nn={nn_med:5.2f} "
              f"oncat={on_cat:8.0f} {e['submission_name'][:60]}", flush=True)
    out = {"band_edges_px": BANDS, "cells_per_band": cells_per_band.tolist(),
           "note": "band 0 = catalogue pixels (masked by organizer, forum 11516); mass there is excluded from mass_by_band[1:]",
           "rows": rows}
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "scored_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
