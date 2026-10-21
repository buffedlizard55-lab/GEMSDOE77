#!/usr/bin/env python3
"""Parallel-run uniqueness audit with the shared template tool (gems57.uniqueness), unmodified.

Phase 1 (before placement): the continuous placement surface (S1 expected credit, scaled to
[0, 1]) - rank correlation vs every registry raster.
Phase 2 (final dots): the submitted dot set - rank correlation AND 3 px Euclidean dot overlap
vs every registry raster.  Both the literal reading and the like-for-like reading (dot
fields only; support > 20% of the footprint = continuous surface) are reported by
gems57.uniqueness.two_phase_summary.
Writes evidence/e3_uniqueness.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import uniqueness as U  # noqa: E402
from gems77lia import allocate as al, core  # noqa: E402
from e3_build_candidate import bins_like_e2  # noqa: E402


def registry_for_template():
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    rec = json.loads((ROOT / "registry" / "registry_receipt.json").read_text())["sha256_by_blob"]
    rasters = [{"repo_first": r["repo"], "sources": r["aliases"], "cache_file": str(core.REG / f"{r['blob']}.tif"),
                "sha256": rec[r["blob"]]} for r in idx["rasters"] if r["blob"] in rec]
    return {"rasters": rasters, "complete_accessible_scan": True, "errors": []}, len(idx["rasters"])


def main():
    g = core.grid()
    reg, n_expected = registry_for_template()
    sub = json.loads((ROOT / "docs" / "data" / "submission.json").read_text())
    dots = np.load(core.WORK / "candidate_dots.npy")
    final = np.zeros((core.H, core.W), np.float32)
    final[dots] = 1.0
    # placement surface (phase 1)
    ev = json.loads((ROOT / "evidence" / "e2_hypothesis_test.json").read_text())
    t80, t95 = ev["sef_thresholds"]["t80"], ev["sef_thresholds"]["t95"]
    res = {r["model"][:2]: r for r in ev["results"] if r["ridge"] == 1e-3}
    sef = np.load(core.WORK / "f_sef.npy")
    dsup = np.load(core.WORK / "d_h195.npy")
    dcat = core.catalogue_distance_px()
    valid = g.footprint & ~g.catalogue
    eb = np.where(sef <= 0, 0, np.where(sef <= t80, 1, np.where(sef <= t95, 2, 3)))
    hm1 = al.hmap_from_bins(bins_like_e2(sef, dcat, dsup, valid, t80, t95, eb), res["S1"]["h"])
    surf = al.credit_map(hm1)
    surf = surf + 1e-3 * sef           # within-bin tie-break used by the placement
    surf[~valid] = 0
    surf = (surf / surf.max()).astype(np.float32)
    prev = ROOT / "evidence" / "e3_uniqueness.json"
    out = {}
    phases = (("phase1_surface_before_placement", surf), ("phase2_final_dots", final))
    if "--phase2-only" in sys.argv and prev.exists():
        old = json.loads(prev.read_text())
        out["phase1_surface_before_placement"] = old["phase1_surface_before_placement"]
        out["first_build_logged_duplicate"] = {"phase2_final_dots": old["phase2_final_dots"],
                                               "candidate": old.get("candidate")}
        phases = phases[1:]
    for phase, arr in phases:
        r = U.compare_array_to_registry(arr, reg, g.footprint, include_reverse=(phase.startswith("phase2")))
        rows = r.get("rows") or r.get("comparisons") or []
        summ = U.two_phase_summary(rows, int(g.footprint.sum()), expected_rasters=n_expected) if rows else None
        out[phase] = {"template_result_keys": sorted(r.keys()), "summary": summ,
                      "worst_rows": sorted([x for x in rows if "error" not in x],
                                           key=lambda x: -(x.get("my_dots_within_3px_of_theirs") or 0))[:12],
                      "worst_rho_rows": sorted([x for x in rows if "error" not in x and x.get("spearman_full_footprint") is not None],
                                               key=lambda x: -x["spearman_full_footprint"])[:8],
                      "errors": [x for x in rows if "error" in x]}
        print(phase, json.dumps({k: summ[k] for k in ("rasters_checked",)} if summ else {}),
              "literal", json.dumps(summ["literal_reading"] if summ else {}, default=str)[:400])
        if summ:
            print("  like-for-like", json.dumps(summ["like_for_like_reading"], default=str)[:600])
    out["candidate"] = {"name": sub["name"], "primary_sha256": sub["primary"]["sha256"], "dots": int(dots.sum())}
    (ROOT / "evidence" / "e3_uniqueness.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
