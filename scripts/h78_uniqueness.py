#!/usr/bin/env python3
"""H78 parallel-run uniqueness audit with the shared template tool (gems57.uniqueness, unmodified).

Registry = the 812 indexed rasters + every TIFF published in GEMSDOE73-76 at clone time + the
GEMSDOE77 lane TIFFs (scripts/h78_e3_build_and_holdout.registry_files, sha256 pinned).
Phase 1 (before placement): the placement surface (.cache/work/h78_surface.npy).
Phase 2 (final dots): the written primary TIFF.
Both the literal reading and the like-for-like reading (gems57.uniqueness.two_phase_summary)
are reported; space-filling rasters (3 px halo >= 70 % of allowed cells, listed in
evidence/h78_e3_registry.json) are reported separately because uniformly random dots exceed the
0.70 overlap limit against them.
Usage: h78_uniqueness.py phase1 | phase2 | merge   (the two phases run in parallel processes;
`merge` combines .cache/work/h78_uniq_phase*.json into evidence/h78_uniqueness.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import uniqueness as U  # noqa: E402
from gems77lia import core  # noqa: E402
from h78_e3_build_and_holdout import registry_files  # noqa: E402


def main():
    t0 = time.time()
    g = core.grid()
    files = registry_files()
    reg = {"rasters": [{"repo_first": r["repo"], "sources": [r["id"]], "cache_file": r["file"], "sha256": r["sha256"]}
                       for r in files], "complete_accessible_scan": True, "errors": []}
    sub = json.loads((ROOT / "docs" / "data" / "h78_submission.json").read_text())
    import rasterio
    with rasterio.open(ROOT / "docs" / "downloads" / sub["primary"]["file"]) as src:
        final = src.read(1).astype(np.float32)
    surf = np.load(core.WORK / "h78_surface.npy").astype(np.float32)
    surf[~g.footprint] = 0
    sf = {x["id"] for x in json.loads((ROOT / "evidence" / "h78_e3_registry.json").read_text())["space_filling"]}
    out = {"registry_files": len(files), "space_filling_ids": sorted(sf)}
    which = sys.argv[1] if len(sys.argv) > 1 else "merge"
    if which == "merge":
        for ph in ("phase1_surface_before_placement", "phase2_final_dots"):
            out[ph] = json.loads((core.WORK / f"h78_uniq_{ph}.json").read_text())
        return finish(out, sub, t0)

    def prog(i, n, row):
        if i % 50 == 0 or i == n:
            print(f"  {which} {i}/{n} ({time.time() - t0:.0f}s)", flush=True)

    todo = {"phase1": [("phase1_surface_before_placement", surf)], "phase2": [("phase2_final_dots", final)]}[which]
    for phase, arr in todo:
        r = U.compare_array_to_registry(arr, reg, g.footprint, include_reverse=phase.startswith("phase2"),
                                        file_sha256=sub["primary"]["sha256"] if phase.startswith("phase2") else None,
                                        progress=prog)
        rows = r.get("rows") or r.get("comparisons") or []
        ok_rows = [x for x in rows if "error" not in x]
        summ = U.two_phase_summary(rows, int(g.footprint.sum()), expected_rasters=len(files)) if ok_rows else None
        informative = [x for x in ok_rows if x["submission"] not in sf]
        worst_inf_ov = max(informative, key=lambda x: x["my_dots_within_3px_of_theirs"]) if informative else None
        rhos = [x for x in ok_rows if x.get("spearman_full_footprint") is not None]
        worst_rho = max(rhos, key=lambda x: x["spearman_full_footprint"]) if rhos else None
        out[phase] = {
            "summary": summ,
            "informative_reading": {
                "rasters": len(informative),
                "worst_forward_overlap": worst_inf_ov["my_dots_within_3px_of_theirs"] if worst_inf_ov else None,
                "worst_overlap_submission": worst_inf_ov["submission"] if worst_inf_ov else None,
                "overlap_firings": sum(x["my_dots_within_3px_of_theirs"] > U.OVERLAP_LIMIT for x in informative),
                "jaccard_firings": sum(x["jaccard_dot_sets"] > U.JACCARD_LIMIT for x in informative),
                "identical_decoded": sum(bool(x["identical_decoded_predictions"]) for x in ok_rows)},
            "worst_rho": {"rho": worst_rho["spearman_full_footprint"], "submission": worst_rho["submission"]} if worst_rho else None,
            "rho_firings": sum(x["spearman_full_footprint"] > U.RHO_LIMIT for x in rhos),
            "space_filling_rows": [{k: x.get(k) for k in ("submission", "my_dots_within_3px_of_theirs", "their_dots",
                                                          "spearman_full_footprint")} for x in ok_rows if x["submission"] in sf],
            "worst_overlap_rows": sorted([{k: x.get(k) for k in ("submission", "my_dots_within_3px_of_theirs", "their_dots",
                                                                 "spearman_full_footprint", "jaccard_dot_sets")}
                                          for x in ok_rows], key=lambda x: -x["my_dots_within_3px_of_theirs"])[:15],
            "errors": [x for x in rows if "error" in x]}
        print(phase, json.dumps(out[phase]["informative_reading"]), "worst rho", out[phase]["worst_rho"],
              f"({time.time() - t0:.0f}s)", flush=True)
        (core.WORK / f"h78_uniq_{phase}.json").write_text(json.dumps(out[phase], indent=1, default=str))


def finish(out, sub, t0):
    p2 = out["phase2_final_dots"]
    p1 = out["phase1_surface_before_placement"]
    out["candidate"] = {"name": sub["name"], "primary_sha256": sub["primary"]["sha256"], "dots": sub["dots"]}
    out["protocol_unique_informative"] = bool(
        p2["informative_reading"]["overlap_firings"] == 0 and p2["rho_firings"] == 0
        and p1["rho_firings"] == 0 and p2["informative_reading"]["identical_decoded"] == 0)
    out["runtime_s"] = round(time.time() - t0, 1)
    (ROOT / "evidence" / "h78_uniqueness.json").write_text(json.dumps(out, indent=1, default=str))
    print("protocol_unique_informative:", out["protocol_unique_informative"])


if __name__ == "__main__":
    main()
