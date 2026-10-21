#!/usr/bin/env python3
"""H77E uniqueness audit: surface (pre-placement artifact) + final dots.

Uses the unmodified vendor/gems57/uniqueness.py `compare_array_to_registry`
(literal thresholds: Spearman rho > 0.90 or forward 3-px dot overlap > 0.70).
Registry scope is STATED and incomplete: GitHub REST was rate-limited
(5000/5000, measured 2026-10-21), so the checked set is (a) every prior lane
TIFF published in this repo's docs/downloads, and (b) 32 OWNER-REPORT scored
rasters (score >= 0.14) fetched by git sparse clone, including the H33 0.2778
file. `complete_accessible_scan` is therefore False and the vendor tool fails
closed (protocol_unique = False) -- the same bottom line as every lane in
this repo under Policy I17 (dense witness ab0a0a62 covers 100% of the allowed
footprint; evidence/registry_saturation.json). The informative reading
(sparse, non-space-filling rasters only) is reported as a diagnostic.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

import h77e_common as C  # noqa: E402
from gems57 import uniqueness as UQ  # noqa: E402

BUILD = json.loads((ROOT / "evidence" / "h77e_e3_build.json").read_text())
CAND = ROOT / BUILD["tif"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def collect():
    recs, seen = [], set()

    def add(repo, name, path):
        path = Path(path)
        if not path.is_file():
            return
        s = sha(path)
        if s in seen:
            return
        seen.add(s)
        recs.append({"repo_first": repo, "sources": [name], "cache_file": str(path), "sha256": s})

    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")):
        if "h77e" in p.name or p.name.endswith("-nan.tif"):
            continue
        add("GEMSDOE77-local", p.name, p)
    for p in sorted((ROOT / ".cache" / "h77e_registry").glob("*.tif")):
        add(p.name.split("__")[0], p.name, p)
    add("GEMSDOE24-local", "h19-5 scored (data/raw)", C.RAW / "scored" / "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif")
    return recs


def main() -> int:
    t0 = time.time()
    grid = C.load_grid()
    foot, cat = grid.footprint, grid.catalogue
    recs = collect()
    idx = {"rasters": recs, "complete_accessible_scan": False,
           "errors": ["registry incomplete: REST rate-limited; git-sparse scored subset + local lanes only"]}
    print(f"registry records: {len(recs)}", flush=True)

    # space-filling classification: 3px-halo coverage of allowed cells
    allowed = foot & ~cat
    space_ids = set()
    coverage = {}
    for r in recs:
        with rasterio.open(r["cache_file"]) as src:
            a = src.read(1)
        sup = np.isfinite(a) & (a > 0) & foot
        cov = float(UQ._near(sup)[allowed].mean()) if sup.any() else 0.0
        coverage[r["sha256"]] = cov
        if cov >= 0.70:
            space_ids.add(r["sha256"])

    with rasterio.open(CAND) as src:
        cand = src.read(1)
    surf = np.load(C.CACHE / "full_surface.npy")
    surf = np.where(foot, surf, 0.0).astype(np.float32)

    print("phase 1: surface ...", flush=True)
    p1 = UQ.compare_array_to_registry(surf, idx, foot, file_sha256=None)
    print("phase 2: final dots ...", flush=True)
    p2 = UQ.compare_array_to_registry(cand, idx, foot, file_sha256=sha(CAND))

    def informative(rows):
        rows2 = [r for r in rows if "error" not in r and r["sha256"] not in space_ids]
        worst_rho = max(rows2, key=lambda r: abs(r["spearman_full_footprint"] or 0), default=None)
        worst_ov = max(rows2, key=lambda r: r["my_dots_within_3px_of_theirs"], default=None)
        return {
            "n_informative": len(rows2),
            "worst_abs_spearman": (abs(worst_rho["spearman_full_footprint"]) if worst_rho else None),
            "worst_rho_submission": (worst_rho["submission"] if worst_rho else None),
            "worst_forward_overlap_3px": (worst_ov["my_dots_within_3px_of_theirs"] if worst_ov else None),
            "worst_overlap_submission": (worst_ov["submission"] if worst_ov else None),
            "duplicate_by_rho_or_overlap": bool(worst_rho and abs(worst_rho["spearman_full_footprint"] or 0) > 0.90)
                                           or bool(worst_ov and worst_ov["my_dots_within_3px_of_theirs"] > 0.70),
            "top5_by_overlap": sorted(
                ({"submission": r["submission"], "overlap": r["my_dots_within_3px_of_theirs"],
                  "abs_spearman": abs(r["spearman_full_footprint"] or 0)} for r in rows2),
                key=lambda d: -d["overlap"])[:5],
        }

    out = {
        "candidate": BUILD["submission_name"],
        "registry_scope": "local GEMSDOE77 lane TIFFs + 32 git-fetched scored rasters (score>=0.14, incl H33); NOT the full 812-raster registry (REST rate-limited)",
        "registry_records": len(recs),
        "space_filling_rasters_in_checked_set": sorted(space_ids),
        "coverage_3px_by_sha256": coverage,
        "phase1_surface": {
            "worst_spearman_full_footprint": p1["worst_spearman_full_footprint"],
            "worst_rho_submission": p1["worst_rho_submission"],
            "literal_worst_dot_overlap": p1["worst_dot_overlap"],
            "note": "continuous surface support is >0 on the whole footprint, so literal overlap is trivially ~1; Spearman rho is the meaningful surface criterion (as in prior lanes)",
            "informative": informative(p1["rows"]),
            "vendor_unique": p1["unique"], "vendor_duplicate_count": p1["duplicate_count"],
        },
        "phase2_final_dots": {
            "worst_spearman_full_footprint": p2["worst_spearman_full_footprint"],
            "worst_rho_submission": p2["worst_rho_submission"],
            "worst_dot_overlap": p2["worst_dot_overlap"],
            "worst_overlap_submission": p2["worst_overlap_submission"],
            "informative": informative(p2["rows"]),
            "vendor_unique": p2["unique"], "vendor_duplicate_count": p2["duplicate_count"],
            "byte_unique_among_checked": p2["byte_unique_among_checked"],
            "pixel_unique_among_checked": p2["pixel_unique_among_checked"],
        },
        "protocol_unique": False,
        "protocol_unique_reason": "vendor fail-closed: registry incomplete in this session (REST rate limit) AND Policy I17 dense witness ab0a0a62 covers 100% of allowed footprint (evidence/registry_saturation.json, repo-measured)",
        "rows_phase2": p2["rows"],
        "elapsed_s": time.time() - t0,
    }
    (ROOT / "evidence" / "h77e_uniqueness.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("phase1_surface", "phase2_final_dots")}, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
