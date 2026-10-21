#!/usr/bin/env python3
"""Lane H79 uniqueness audit with the shared gems57.uniqueness module (unmodified).

Phase 1: the placement surface (allowed cells only, scaled to [0,1]) BEFORE dot placement.
Phase 2: the final dot raster as written to docs/downloads.
Each phase is one process (`--phase 1|2`) so both can run on the two sandbox CPUs; `--merge`
combines the two JSONs into evidence/h79_uniqueness.json with the literal reading, the
like-for-like (dot-field) reading, and the explicit list of space-filling rasters against
which the literal 3 px test is vacuous (IR-79-03).
"""
from __future__ import annotations

import argparse
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
from gems57 import uniqueness as U  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from h79_build import registry_for_template  # noqa: E402

RAW = ROOT / "data" / "raw"
WORK = ROOT / ".cache" / "work" / "h79"
EVID = ROOT / "evidence"


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run_phase(phase: int, tif: Path, tag: str = "h79") -> dict:
    grid = load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    reg, n_expected = registry_for_template()
    if phase == 1:
        surf = np.load(WORK / "surface_full.npy")
        dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
        allowed = foot & ~cat & (dist_cat > 3.0)
        arr = np.where(allowed, surf, 0.0).astype(np.float32)
        arr = (arr / max(float(arr.max()), 1e-9)).astype(np.float32)
        include_reverse = False
    else:
        with rasterio.open(tif) as src:
            arr = src.read(1).astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        include_reverse = True
    t0 = time.time()
    done = {"n": 0}

    def prog(i, n, row):
        done["n"] = i
        if i % 100 == 0 or i == n:
            log(f"phase {phase}: {i}/{n} ({time.time()-t0:.0f}s)")

    r = U.compare_array_to_registry(arr, reg, foot, include_reverse=include_reverse, progress=prog)
    rows = r.get("rows") or r.get("comparisons") or []
    ok_rows = [x for x in rows if "error" not in x]
    summ = U.two_phase_summary(rows, int(foot.sum()), expected_rasters=n_expected) if ok_rows else None
    n_foot = int(foot.sum())
    for x in ok_rows:
        x["support_fraction"] = x["their_dots"] / n_foot
        x["blob"] = Path(x["file"]).stem
    compact = [{k: x.get(k) for k in ("blob", "submission", "repo", "their_dots", "support_fraction", "representation",
                                      "my_dots_within_3px_of_theirs", "their_dots_within_3px_of_mine", "spearman_full_footprint", "jaccard_dot_sets")}
               for x in ok_rows]
    (EVID / f"{tag}_uniqueness_phase{phase}_rows.json").write_text(json.dumps(compact, sort_keys=True))
    out = {
        "tif": str(tif.relative_to(ROOT)) if tif.is_relative_to(ROOT) else str(tif), "sha256": hashlib.sha256(tif.read_bytes()).hexdigest() if phase == 2 else None,
        "my_dots": int((arr > 0).sum()) if phase == 2 else None,
        "phase": phase, "rasters_checked": len(ok_rows), "rasters_expected_index": n_expected,
        "rasters_in_cache": len(reg["rasters"]), "errors": [x for x in rows if "error" in x],
        "summary": summ,
        "max_rho": max((x["spearman_full_footprint"] for x in ok_rows if x.get("spearman_full_footprint") is not None), default=None),
        "max_forward_overlap_all": max((x["my_dots_within_3px_of_theirs"] for x in ok_rows), default=None),
        "max_forward_overlap_dot_fields": max((x["my_dots_within_3px_of_theirs"] for x in ok_rows if x.get("representation") == "dot_field"), default=None),
        "worst_overlap_rows": sorted(ok_rows, key=lambda x: -(x.get("my_dots_within_3px_of_theirs") or 0))[:15],
        "worst_rho_rows": sorted([x for x in ok_rows if x.get("spearman_full_footprint") is not None], key=lambda x: -x["spearman_full_footprint"])[:10],
        "space_filling_rasters": sorted([{"submission": x["submission"], "repo": x["repo"], "support_fraction": x["support_fraction"],
                                          "forward_overlap": x["my_dots_within_3px_of_theirs"]}
                                         for x in ok_rows if x.get("representation") == "continuous_surface"],
                                        key=lambda d: -d["forward_overlap"]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (EVID / f"{tag}_uniqueness_phase{phase}.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    log(f"phase {phase} done: checked={len(ok_rows)} max_rho={out['max_rho']} fwd_all={out['max_forward_overlap_all']} fwd_dot={out['max_forward_overlap_dot_fields']}")
    return out


def merge(tag: str = "h79", build_file: str = "h79_build.json", tif_arg: str | None = None) -> None:
    # phase 1 (surface BEFORE placement) is shared by every emission of the same surface
    p1_path = EVID / ("h79_uniqueness_phase1.json" if tag in ("h79", "h79c") else f"{tag}_uniqueness_phase1.json")
    p1 = json.loads(p1_path.read_text()) if p1_path.exists() else {"max_rho": None, "rasters_checked": 0, "max_forward_overlap_all": None, "max_forward_overlap_dot_fields": None, "worst_rho_rows": [], "note": "surface before placement not available for this file"}
    p2 = json.loads((EVID / f"{tag}_uniqueness_phase2.json").read_text())
    rows2 = json.loads((EVID / f"{tag}_uniqueness_phase2_rows.json").read_text())
    if build_file == "-":  # incumbent audit: no build receipt; describe the TIF itself
        tif = Path(tif_arg or p2["tif"])
        tif = tif if tif.is_absolute() else ROOT / tif
        with rasterio.open(tif) as src:
            a = src.read(1)
        build = {"tif": str(tif.relative_to(ROOT)), "sha256": hashlib.sha256(tif.read_bytes()).hexdigest(),
                 "dots": int((np.isfinite(a) & (a > 0)).sum()), "emission": "incumbent lane file (not built here)"}
    else:
        build = json.loads((EVID / build_file).read_text())
    halo = json.loads((ROOT / "registry" / "registry_halo_coverage.json").read_text())["rasters"]
    for r in rows2:
        hc = halo.get(r["blob"], {})
        r["halo_coverage_allowed"] = hc.get("covered_allowed_fraction")
        r["discriminating"] = bool(hc.get("discriminating_at_0_70", False))
    disc = [r for r in rows2 if r["discriminating"]]
    nond = [r for r in rows2 if not r["discriminating"]]
    max_fwd_disc = max((r["my_dots_within_3px_of_theirs"] for r in disc), default=0.0)
    worst_disc = sorted(disc, key=lambda r: -r["my_dots_within_3px_of_theirs"])[:10]
    rho_fail = (p2["max_rho"] is not None and p2["max_rho"] > U.RHO_LIMIT) or (p1["max_rho"] is not None and p1["max_rho"] > U.RHO_LIMIT)
    lit_dup = rho_fail or (p2["max_forward_overlap_all"] or 0) > U.OVERLAP_LIMIT
    lfl_dup = rho_fail or max_fwd_disc > U.OVERLAP_LIMIT
    out = {
        "candidate": build["tif"], "sha256": build["sha256"], "dots": build["dots"],
        "rule": {"rho_limit": U.RHO_LIMIT, "overlap_limit": U.OVERLAP_LIMIT, "radius_px": 3,
                 "literal": "every registry raster, every positive finite pixel is a dot",
                 "like_for_like": ("rho test on all rasters; 3 px forward-overlap test on DISCRIMINATING rasters only, i.e. rasters whose "
                                   "3 px halo covers < 0.70 of the allowed cells (random dots would already exceed 0.70 against the others); "
                                   "measured per raster with gems57.uniqueness.saturation_certificate (registry/registry_halo_coverage.json)")},
        "discriminating_rasters": len(disc), "non_discriminating_rasters": len(nond),
        "max_forward_overlap_discriminating": max_fwd_disc,
        "worst_discriminating_rows": worst_disc,
        "non_discriminating_list": sorted([{"submission": r["submission"], "repo": r["repo"], "halo_coverage_allowed": r["halo_coverage_allowed"],
                                            "forward_overlap": r["my_dots_within_3px_of_theirs"], "support_fraction": r["support_fraction"]} for r in nond],
                                          key=lambda d: -(d["halo_coverage_allowed"] or 0)),
        "phase1_surface_before_placement": {k: p1[k] for k in ("rasters_checked", "max_rho", "max_forward_overlap_all", "max_forward_overlap_dot_fields", "worst_rho_rows")},
        "phase2_final_dots": {k: p2[k] for k in ("rasters_checked", "max_rho", "max_forward_overlap_all", "max_forward_overlap_dot_fields", "worst_overlap_rows", "worst_rho_rows", "space_filling_rasters", "errors")},
        "literal_reading_duplicate": bool(lit_dup),
        "like_for_like_duplicate": bool(lfl_dup),
        "protocol_unique_literal": not lit_dup,
        "protocol_unique_like_for_like": not lfl_dup,
        "n_space_filling_rasters": len(nond),
        "scope": "905 cached rasters: 809/812 indexed (3 blobs no longer at HEAD) + 96 published after the index (git snapshot 2026-10-21 14:22 UTC); not an organizer registry",
    }
    out["emission"] = build.get("emission", "greedy (pre-registered)")
    (EVID / f"{tag}_uniqueness.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    log(json.dumps({k: out[k] for k in ("literal_reading_duplicate", "like_for_like_duplicate", "n_space_filling_rasters", "discriminating_rasters", "max_forward_overlap_discriminating")}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=(1, 2))
    ap.add_argument("--tif", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="h79", help="evidence file prefix (h79 = first build, h79c = capped build)")
    ap.add_argument("--build-file", default="h79_build.json")
    args = ap.parse_args()
    if args.merge:
        merge(args.tag, args.build_file, args.tif)
        return
    tif = Path(args.tif) if args.tif else ROOT / json.loads((EVID / args.build_file).read_text())["tif"]
    run_phase(args.phase, tif, args.tag)


if __name__ == "__main__":
    main()
