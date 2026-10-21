#!/usr/bin/env python3
"""Build label-blind holdout halo index `.cache/audit_halos_holdout_fp.npy` across all `5,167,373`
footprint pixels (`grid.footprint`), bridging `existing_faults.tif` zero-perforations in historical
submissions via nearest non-catalogue neighbor before computing 3-px halos (`IR-H82-03` safeguard).

Guarantees ZERO indirect `existing_faults` perforation leakage during holdout uniqueness evaluation,
while `.cache/audit_halos_packed.npy` retains raw unmodified halos for live submission auditing!
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as grid_mod  # noqa: E402

CACHE = ROOT / ".cache"
DATA_DIR = CACHE / "data"


def main() -> int:
    t0 = time.perf_counter()
    meta = json.loads((CACHE / "audit_meta.json").read_text())
    entries = meta["entries"]
    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    cat = g.catalogue & fp
    fp_idx = np.flatnonzero(fp)
    n_fp = int(fp_idx.size)
    n_bytes = (n_fp + 7) // 8
    h_grid, w_grid = fp.shape

    # Nearest non-catalogue pixel index so historical submissions that zeroed `existing_faults.tif`
    # cannot leak catalogue perforations into holdout evaluation (`IR-H82-03`)!
    _, near_noncat = distance_transform_edt(cat, return_distances=True, return_indices=True)
    disk_offsets = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy * dy + dx * dx <= 9]

    def pack_one(entry: dict) -> np.ndarray:
        p = ROOT / entry["path"]
        with rasterio.open(p) as ds:
            arr = ds.read(1).copy()
        # Bridge zero-masked catalogue perforations before computing 3-px halo
        arr[cat] = arr[near_noncat[0][cat], near_noncat[1][cat]]
        pos_mask = np.isfinite(arr) & (arr > 0)
        n_on = int(pos_mask.sum())
        if n_on == 0:
            halo_fp = np.zeros(n_fp, dtype=bool)
        elif n_on <= 350_000:
            rr, cc = np.nonzero(pos_mask)
            out = np.zeros(h_grid * w_grid, dtype=bool)
            for dy, dx in disk_offsets:
                ry = rr + dy
                cx = cc + dx
                ok = (ry >= 0) & (ry < h_grid) & (cx >= 0) & (cx < w_grid)
                out[ry[ok] * w_grid + cx[ok]] = True
            halo_fp = out[fp_idx]
        else:
            padded = np.pad(pos_mask, 3, mode="constant", constant_values=False)
            out2d = np.zeros_like(pos_mask, dtype=bool)
            for dy, dx in disk_offsets:
                out2d |= padded[3 + dy : 3 + dy + h_grid, 3 + dx : 3 + dx + w_grid]
            halo_fp = out2d.ravel()[fp_idx]
        return np.packbits(halo_fp, bitorder="big")

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(pack_one, entries))

    mm = np.lib.format.open_memmap(
        CACHE / "audit_halos_holdout_fp.npy", mode="w+", dtype=np.uint8, shape=(len(entries), n_bytes)
    )
    for i, r in enumerate(rows):
        mm[i] = r
    mm.flush()
    del mm
    (CACHE / "audit_meta_holdout_fp.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_valid": len(entries),
                "n_allow": n_fp,
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built holdout perforation-free index across {len(entries)} rasters in {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
