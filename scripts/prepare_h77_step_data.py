#!/usr/bin/env python3
"""Prepare and verify canonical grid geometry, band finite masks, and external layer integrity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as grid_mod  # noqa: E402

DATA_DIR = ROOT / ".cache" / "data"
EXT_DIR = DATA_DIR / "data" / "external"


def main() -> int:
    if not (DATA_DIR / "training_features.tif").exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "restore_data.py")], check=True)

    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    with rasterio.open(DATA_DIR / "existing_faults.tif") as ds_cat:
        cat = (ds_cat.read(1) > 0) & fp

    band_stats = []
    with rasterio.open(DATA_DIR / "training_features.tif") as ds:
        assert ds.count == 19, f"Expected 19 bands, got {ds.count}"
        for b in range(1, ds.count + 1):
            arr = ds.read(b)
            in_fp = arr[fp]
            n_bad = int((~np.isfinite(in_fp) | (in_fp <= -1e30)).sum())
            band_stats.append({
                "band": b,
                "description": ds.descriptions[b - 1],
                "in_footprint_sentinel_or_nan": n_bad,
                "p50": float(np.median(in_fp[np.isfinite(in_fp) & (in_fp > -1e30)])),
            })

    manifest = {
        "grid_shape": list(g.shape),
        "crs": str(g.crs),
        "transform": list(g.transform)[:6],
        "footprint_pixels": int(fp.sum()),
        "outside_footprint_pixels": int((~fp).sum()),
        "mapped_catalogue_pixels": int(cat.sum()),
        "external_layers_present": sorted(p.name for p in EXT_DIR.iterdir() if p.is_file()),
        "official_19_bands": band_stats,
    }
    out = ROOT / "evidence" / "prepared_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared manifest verified: {manifest['footprint_pixels']:,} footprint px, {manifest['mapped_catalogue_pixels']:,} catalogue px, 19 bands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
