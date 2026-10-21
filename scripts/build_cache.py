#!/usr/bin/env python3
"""Build the .cache/work intermediates used by E2/E3 (run after restore_data.py + fetch_registry.py).

  scored_sparse.npz  sparse (index, value) copies of the 95 distinct organizer-scored rasters,
                     catalogue pixels zeroed (organizer masks them, forum 11516)
  d_h195.npy         Euclidean distance (px) to the H19-5 line set (19GEMSDOE, scored 0.1922)
  f_sef.npy          H77-1 straight-edge field  (gems77lia.fields.straight_edge_field)
  f_scd.npy          H77-2 strain-catalogue deficit (gems77lia.fields.strain_catalogue_deficit)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems77lia import core, fields  # noqa: E402

H19_5 = "h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan"


def main():
    g = core.grid()
    core.WORK.mkdir(parents=True, exist_ok=True)
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    store, seen = {}, set()
    for e in idx["scored"]:
        if e["blob"] in seen:
            continue
        seen.add(e["blob"])
        p = core.read_prediction(core.REG / f"{e['blob']}.tif")
        p[g.catalogue] = 0
        fi = np.flatnonzero(p > 0).astype(np.int32)
        store["i_" + e["blob"]] = fi
        store["v_" + e["blob"]] = p.ravel()[fi].astype(np.float32)
    np.savez_compressed(core.WORK / "scored_sparse.npz", **store)
    blob = [e["blob"] for e in idx["scored"] if e["submission_name"] == H19_5][0]
    m = np.zeros(core.H * core.W, bool)
    m[store["i_" + blob]] = True
    np.save(core.WORK / "d_h195.npy", ndimage.distance_transform_edt(~m.reshape(core.H, core.W)).astype(np.float32))
    np.save(core.WORK / "f_sef.npy", fields.straight_edge_field())
    np.save(core.WORK / "f_scd.npy", fields.strain_catalogue_deficit())
    print("cache built:", len(seen), "scored rasters")


if __name__ == "__main__":
    main()
