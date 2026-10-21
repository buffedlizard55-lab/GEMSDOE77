#!/usr/bin/env python3
"""H82 parallel-run uniqueness audit (Rule 1), both phases.

Phase 1 -- the ranking surface *before* dot placement: Spearman rank
correlation against every registry raster (limit 0.90).
Phase 2 -- the final written dot field: fraction of my dots within 3 px of
theirs (limit 0.70), plus reverse overlap and Jaccard as diagnostics.

Both phases use ``vendor/gems57/uniqueness.py`` unchanged for the measurement.
The only addition is the *reporting* split introduced upstream in
``gems57.uniqueness.space_filling_flags``: a registry raster whose 3 px halo
already covers >= 70 % of the scored footprint cannot discriminate anything,
because a uniformly random dot field also "fires" against it. Those rasters are
listed separately instead of being silently dropped: the literal reading and
the informative reading are both reported.

Usage::

    python scripts/h82fmd_uniqueness.py surface <surface.npy>
    python scripts/h82fmd_uniqueness.py dots <submission.tif>
    python scripts/h82fmd_uniqueness.py merge
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import grid as G            # noqa: E402
from gems57 import uniqueness as U      # noqa: E402

REG = ROOT / ".cache" / "registry"
WORK = ROOT / ".cache" / "h82fmd"


def registry_rows() -> list[dict]:
    """Every hash-pinned registry raster that is present in the local cache,
    plus every other GeoTIFF published in this repository's downloads."""
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    rec = json.loads((ROOT / "registry" / "registry_receipt.json").read_text())["sha256_by_blob"]
    rows, seen = [], set()
    for r in idx["rasters"]:
        if r["blob"] in rec and (REG / f"{r['blob']}.tif").exists():
            rows.append({"repo_first": r["repo"], "sources": r["aliases"],
                         "cache_file": str(REG / f"{r['blob']}.tif"), "sha256": rec[r["blob"]]})
            seen.add(rec[r["blob"]])
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")):
        if p.name.startswith("gems77-h82"):
            continue
        s = U._sha(p)
        if s in seen:
            continue
        seen.add(s)
        rows.append({"repo_first": "GEMSDOE77", "sources": [f"GEMSDOE77:docs/downloads/{p.name}"],
                     "cache_file": str(p), "sha256": s})
    return rows


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "merge"
    grid = G.load_grid(ROOT / "data" / "bridge")
    fp = grid.footprint
    WORK.mkdir(parents=True, exist_ok=True)
    rows = registry_rows()
    reg = {"rasters": rows, "complete_accessible_scan": True, "errors": []}
    t0 = time.time()

    if which == "merge":
        out = {"registry_rasters_compared": len(rows),
               "scope": ("All GeoTIFFs indexed by registry/registry_index.json (GitHub tree API over every "
                         "buffedlizard55-lab GEMSDOE* repository, candidate-submission paths, de-duplicated by "
                         "blob sha) that are present in the local SHA-256-verified cache, plus every other "
                         "GeoTIFF published in this repository's docs/downloads. This is NOT a complete "
                         "organizer registry: private, unlinked or later-published rasters cannot be seen."),
               "limits": {"spearman_rho": U.RHO_LIMIT, "forward_dot_overlap_3px": U.OVERLAP_LIMIT,
                          "jaccard": U.JACCARD_LIMIT}}
        for phase in ("phase1_surface", "phase2_dots"):
            f = WORK / f"uniq_{phase}.json"
            if f.exists():
                d = json.loads(f.read_text())
                if d.get("rows"):           # keep the evidence file readable; full table stays in .cache
                    d["rows_top50"] = d.pop("rows")[:50]
                    d["rows_full_table"] = str(f.relative_to(ROOT))
                out[phase] = d
        lit = out.get("phase2_dots", {}).get("literal_max_overlap")
        inf = out.get("phase2_dots", {}).get("informative_max_overlap")
        rho = out.get("phase1_surface", {}).get("max_abs_rho")
        rho2 = out.get("phase2_dots", {}).get("max_abs_rho")
        out["verdict"] = {
            "informative_unique": bool(
                inf is not None and inf <= U.OVERLAP_LIMIT
                and rho is not None and rho <= U.RHO_LIMIT
                and rho2 is not None and rho2 <= U.RHO_LIMIT),
            "literal_unique": bool(lit is not None and lit <= U.OVERLAP_LIMIT
                                   and rho is not None and rho <= U.RHO_LIMIT),
            "surface_rank_unique": bool(rho is not None and rho <= U.RHO_LIMIT
                                        and rho2 is not None and rho2 <= U.RHO_LIMIT),
            "density_matched_unique": bool(
                (out.get("phase2_dots", {}).get("density_strata", {})
                    .get("sparse_peer_le_2x_budget", {}).get("firings_over_0p70")) == 0),
            "max_abs_rho_surface": rho, "max_abs_rho_dots": rho2,
            "informative_max_overlap": inf, "literal_max_overlap": lit,
            "explanation": ("informative_unique excludes registry rasters whose own 3 px halo already covers "
                            ">= 70 % of the scored footprint, against which even a uniformly random dot field "
                            "'fires'; literal_unique applies the limit to every raster including those."),
        }
        (ROOT / "evidence" / "h82fmd_uniqueness.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out.get("verdict", {}), indent=2))
        return 0

    if which == "surface":
        arr = np.load(sys.argv[2]).astype(np.float32)
        arr[~fp] = 0.0
        mx = float(arr.max())
        if mx > 0:
            arr = arr / mx
        phase, sha = "phase1_surface", None
    else:
        with rasterio.open(sys.argv[2]) as src:
            arr = src.read(1).astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr[~fp] = 0.0
        phase, sha = "phase2_dots", U._sha(Path(sys.argv[2]))

    def prog(i, n, row):
        if i % 100 == 0 or i == n:
            print(f"  {phase} {i}/{n} ({time.time()-t0:.0f}s)", flush=True)

    res = U.compare_array_to_registry(arr, reg, fp, include_reverse=(phase == "phase2_dots"),
                                      file_sha256=sha, progress=prog)
    all_rows = [r for r in (res.get("rows") or res.get("comparisons") or []) if "error" not in r]
    errors = [r for r in (res.get("rows") or res.get("comparisons") or []) if "error" in r]
    flags = U.space_filling_flags(all_rows, fp, radius_px=3.0)
    informative = [r for r in all_rows if not flags[r["sha256"]]["space_filling"]]

    def worst(rs, key):
        vals = [(abs(r[key]), r) for r in rs if r.get(key) is not None]
        return max(vals, key=lambda t: t[0]) if vals else (None, None)

    ov_lit, row_lit = worst(all_rows, "my_dots_within_3px_of_theirs")
    ov_inf, row_inf = worst(informative, "my_dots_within_3px_of_theirs")
    rho_all, row_rho = worst(all_rows, "spearman_full_footprint")
    out = {
        "phase": phase,
        "candidate": sys.argv[2],
        "candidate_sha256": sha,
        "registry_rasters_compared": len(all_rows),
        "registry_read_errors": len(errors),
        "space_filling_count": int(sum(1 for v in flags.values() if v["space_filling"])),
        "space_filling": sorted(
            [{"submission": r["submission"], "their_dots": r["their_dots"],
              "halo_coverage": flags[r["sha256"]]["halo_coverage"],
              "my_overlap": r["my_dots_within_3px_of_theirs"]}
             for r in all_rows if flags[r["sha256"]]["space_filling"]],
            key=lambda d: -d["halo_coverage"]),
        "literal_max_overlap": ov_lit,
        "literal_max_overlap_vs": (row_lit or {}).get("submission"),
        "informative_rasters": len(informative),
        "informative_max_overlap": ov_inf,
        "informative_max_overlap_vs": (row_inf or {}).get("submission"),
        "max_abs_rho": rho_all,
        "max_abs_rho_vs": (row_rho or {}).get("submission"),
        "any_identical_bytes": any(r.get("identical_bytes") for r in all_rows),
        "any_identical_decoded": any(r.get("identical_decoded_predictions") for r in all_rows),
        "top10_informative_overlap": sorted(
            [{"submission": r["submission"], "overlap": r["my_dots_within_3px_of_theirs"],
              "their_dots": r["their_dots"], "rho": r["spearman_full_footprint"]}
             for r in informative], key=lambda d: -(d["overlap"] or 0))[:10],
        "top10_abs_rho": sorted(
            [{"submission": r["submission"], "rho": r["spearman_full_footprint"]}
             for r in all_rows if r.get("spearman_full_footprint") is not None],
            key=lambda d: -abs(d["rho"]))[:10],
        "shared_tool_summary": res.get("summary", res.get("verdict")),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if phase == "phase2_dots":
        n_fp = int(fp.sum())
        # The shared tool's own two-reading summary (literal + like-for-like at
        # DENSE_SUPPORT_FRACTION), computed here rather than reimplemented.
        out["shared_two_phase_summary"] = U.two_phase_summary(all_rows, n_fp,
                                                              expected_rasters=len(all_rows) + len(errors))
        # Density strata: a 3 px dot-overlap test only carries information when the
        # comparator's own support is comparable to the candidate's. Reported as a
        # diagnostic; it does NOT replace the preregistered verdict above.
        budget = int(np.count_nonzero(U._dots(arr) & fp))
        strata = {"sparse_peer_le_2x_budget": [], "dot_field_le_20pct": [],
                  "continuous_surface_gt_20pct": [], "space_filling_halo_ge_70pct": []}
        for r in all_rows:
            nd = int(r.get("their_dots", 0) or 0)
            if flags[r["sha256"]]["space_filling"]:
                k = "space_filling_halo_ge_70pct"
            elif nd > 0.20 * n_fp:
                k = "continuous_surface_gt_20pct"
            elif nd <= 2 * budget:
                k = "sparse_peer_le_2x_budget"
            else:
                k = "dot_field_le_20pct"
            strata[k].append(r)
        def _stratum(rs):
            if not rs:
                return {"rasters": 0}
            w = max(rs, key=lambda r: r.get("my_dots_within_3px_of_theirs") or 0.0)
            return {"rasters": len(rs),
                    "worst_forward_overlap": w["my_dots_within_3px_of_theirs"],
                    "worst_forward_overlap_vs": w["submission"],
                    "worst_forward_overlap_their_dots": w["their_dots"],
                    "firings_over_0p70": sum(1 for r in rs
                                             if (r.get("my_dots_within_3px_of_theirs") or 0) > U.OVERLAP_LIMIT)}
        out["my_dots"] = budget
        out["density_strata"] = {k: _stratum(v) for k, v in strata.items()}
        out["worst_reverse_overlap"] = max((r.get("their_dots_within_3px_of_mine") or 0.0) for r in all_rows)
        out["worst_jaccard"] = max((r.get("jaccard_dot_sets") or 0.0) for r in all_rows)
        out["rows"] = [{"submission": r["submission"], "repo": r.get("repo"), "sha256": r["sha256"],
                        "their_dots": r["their_dots"],
                        "forward": r["my_dots_within_3px_of_theirs"],
                        "reverse": r["their_dots_within_3px_of_mine"],
                        "jaccard": r["jaccard_dot_sets"], "rho": r["spearman_full_footprint"],
                        "halo_coverage": flags[r["sha256"]]["halo_coverage"],
                        "halo_coverage_exact": flags[r["sha256"]]["exact"]}
                       for r in sorted(all_rows, key=lambda r: -(r["my_dots_within_3px_of_theirs"] or 0))]
    (WORK / f"uniq_{phase}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("phase", "registry_rasters_compared", "space_filling_count",
                                          "literal_max_overlap", "informative_max_overlap",
                                          "max_abs_rho")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
