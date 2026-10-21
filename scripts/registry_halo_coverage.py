#!/usr/bin/env python3
"""Registry-wide halo-coverage table (shared gems57.uniqueness.saturation_certificate per raster).

For every cached registry raster, measure the fraction of ALLOWED cells (footprint minus the
mapped catalogue) that lie within the literal 3 px halo of that raster's positive support.
Uniformly random dots have forward overlap equal to this fraction, so the brief's 0.70 test can
only discriminate against rasters whose coverage is below 0.70 ("discriminating"); against the
others the test is vacuous (docs/policy_uniqueness.md, IR-79-03/IR-79-10).

Output: registry/registry_halo_coverage.json  {blob: {covered_allowed_fraction, positive_pixels, ...}}
Rasters that are not on the competition grid are recorded with an error and skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import uniqueness as U  # noqa: E402
from gems57.grid import TRANSFORM, load_grid  # noqa: E402

REG = ROOT / ".cache" / "registry"
OUT = ROOT / "registry" / "registry_halo_coverage.json"


def main() -> None:
    grid = load_grid(ROOT / "data" / "raw")
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    table = dict(prev.get("rasters", {}))
    files = sorted(REG.glob("*.tif"))
    t0 = time.time()
    for i, p in enumerate(files, 1):
        blob = p.stem
        if blob in table and "error" not in table[blob]:
            continue
        try:
            cert = U.saturation_certificate(p, grid.footprint, grid.catalogue,
                                            expected_grid=(grid.footprint.shape, grid.crs, TRANSFORM))
            table[blob] = {k: cert[k] for k in ("covered_allowed_fraction", "positive_pixels", "allowed_pixels",
                                                 "uncovered_allowed_pixels", "universal_overlap_blocker", "witness_sha256")}
            table[blob]["discriminating_at_0_70"] = cert["covered_allowed_fraction"] < U.OVERLAP_LIMIT
        except Exception as exc:  # noqa: BLE001 - recorded per raster
            table[blob] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        if i % 100 == 0 or i == len(files):
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    ok = [v for v in table.values() if "error" not in v]
    summary = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "gems57.uniqueness.saturation_certificate per raster; allowed = footprint & ~catalogue; halo = 3 px Euclidean",
        "n_rasters": len(table), "n_ok": len(ok), "n_errors": len(table) - len(ok),
        "n_discriminating_at_0_70": sum(1 for v in ok if v["discriminating_at_0_70"]),
        "n_non_discriminating": sum(1 for v in ok if not v["discriminating_at_0_70"]),
        "n_universal_blockers": sum(1 for v in ok if v["universal_overlap_blocker"]),
        "overlap_limit": U.OVERLAP_LIMIT,
        "rasters": table,
    }
    OUT.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "rasters"}, indent=1))


if __name__ == "__main__":
    main()
