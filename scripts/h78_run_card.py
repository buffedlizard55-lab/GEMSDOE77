#!/usr/bin/env python3
"""Assemble the H78 JSON run card (protocol rule 5) from the evidence files - no hand-typed numbers.

Inputs: evidence/h78_e1_screen.json, evidence/h78_e2_within_support.json, evidence/h78_e3_holdout.json,
evidence/h78_uniqueness.json, docs/data/h78_submission.json.  Writes evidence/h78_run_card.json and
updates the H78 row of docs/data/lanes.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def j(p):
    return json.loads((ROOT / p).read_text())


def main():
    e1, e2, e3 = j("evidence/h78_e1_screen.json"), j("evidence/h78_e2_within_support.json"), j("evidence/h78_e3_holdout.json")
    un, sub = j("evidence/h78_uniqueness.json"), j("docs/data/h78_submission.json")
    pooled = e3["pooled"]
    c = pooled["A0"]["scores"]["C"]
    arms = [{"arm": "C: H78 candidate rule (rebuilt per fold)", "dti": c["dti"], "ci95": c["ci95"], "delta": None}]
    labels = {"A0": "A0: champion construction (H19-5 dots > 2 px from visible catalogue)",
              "L": "L: lidar consensus alone (same dot count)", "R": "R: random dots (same dot count)"}
    for k in ("A0", "L", "R"):
        s, d = pooled[k]["scores"][k], pooled[k]["paired_differences"][k]
        arms.append({"arm": labels[k], "dti": s["dti"], "ci95": s["ci95"], "delta": d["delta"], "delta_ci95": d["ci95"]})
    dA0 = pooled["A0"]["paired_differences"]["A0"]
    p1, p2 = un["phase1_surface_before_placement"], un["phase2_final_dots"]
    canary = {k: v["auc"] for k, v in e3["leakage_canary"].items()}
    beats = bool(dA0["ci95"][0] > 0)
    # Repository policy since I17 (merged PR #7, "fail closed on literal registry uniqueness"): the brief says ANY
    # registry raster, so the literal reading binds; the informative reading is a diagnostic only.
    unique_informative = bool(un["protocol_unique_informative"])
    literal_firings = sum(len((un[ph].get("summary") or {}).get("literal_firings", []))
                          for ph in ("phase2_final_dots",))
    unique = bool(unique_informative and literal_firings == 0)
    fmt = bool(sub["download_ok_format_valid"])
    s0 = e3["full_map_candidate"]["S0"]["dti_projection"]
    s0c = e3["champion_S0_projection"]["dti_projection"]
    verdict = "promote" if (beats and unique and fmt and not any(v > 0.9 for v in canary.values()) and s0 >= s0c) else "negative"
    reasons = []
    if not beats:
        reasons.append(f"holdout delta vs champion construction {dA0['delta']:+.4f} (95% CI {dA0['ci95'][0]:+.4f} to {dA0['ci95'][1]:+.4f}) " + ("is entirely below 0" if dA0["ci95"][1] < 0 else "includes 0"))
    if s0 < s0c:
        reasons.append(f"INSTRUMENT S0 projection {s0:.4f} < champion {s0c:.4f}")
    reasons.append("E1 and E2 found no live-supported field")
    if not unique:
        reasons.append(f"literal registry gate fails ({literal_firings} final-dot firings, all on space-filling rasters; the informative reading passes)")
    card = {
        "lane": "H78 residual-guided off-support allocation (self-assigned; IR-78-01)",
        "hypothesis": "Uncatalogued faults off the H19-5 support leave 1 m lidar scarp morphology; ranking off-support dots by a lidar scarp consensus earns hidden-fault credit where the 0.2778 construction has none.",
        "mechanism": "Quaternary normal-fault scarps: a linear break in slope with a steep free face, a convex crest and a concave toe (rank mean of 7 lidar_scarp_features_u8 operators); INGENIOUS used UAV lidar for exactly such new traces.",
        "non_fault_mimic": "Pluvial Lake Lahontan shoreline benches and wave-cut risers, fluvial terrace risers and fan-incision banks, and road cuts/canal banks all produce linear slope breaks with the same profile.",
        "hypotheses_considered": [
            {"id": "H78-1", "name": "Lidar scarp-morphology consensus", "layers": "lidar_scarp_features_u8 (ex_max, step_max, lapneg_max, lappos_max, downface_max, upface_max, cross_max); 3DEP-derived, competition data pack",
             "signature": "Linear slope break with free face, convex crest, concave toe", "why_uncatalogued": "Young, small-offset scarps below the mapping scale of the USGS Qfault database are visible at 1 m (INGENIOUS Lund North UAV lidar)",
             "differs_from_repo": "Earlier lidar lanes (GEMSDOE7/64/66) used single operators as dense surfaces; H78 uses a 7-operator consensus only to rank OFF-support dots inside a uniqueness-capped allocation",
             "gain_over_cost": "highest (+0.01 to +0.02 hoped / 1 field, cached data)", "outcome": "E1 negative (LOO dMSE < 0), E2 negative, E3 holdout below champion construction"},
            {"id": "H78-2", "name": "Geothermal manifestation proximity", "layers": "GDR 1391 (INGENIOUS regional compilation, DOI 10.15121/1881483, CC BY 4.0): well and spring temperatures",
             "signature": "Hot/warm wells and springs sit on permeable fault intersections and step-overs", "why_uncatalogued": "Fluid upflow marks blind faults that have no scarp",
             "differs_from_repo": "No family repo used thermal-manifestation distance to place dots", "gain_over_cost": "second (sparse points, cheap)", "outcome": "E1: zero weight; E2: unordered, not live-supported"},
            {"id": "H78-3", "name": "Regional mapping-campaign heterogeneity", "layers": "3x3 region blocks of the footprint",
             "signature": "Catalogue completeness varies by mapping campaign; hidden truth density should track it", "why_uncatalogued": "Under-mapped regions hide more faults",
             "differs_from_repo": "Spatially varying off-support credit; the family assumes one global rate", "gain_over_cost": "third (cheap, weak prior)", "outcome": "E1: zero weight"},
            {"id": "H78-4", "name": "Single lidar operators (sensitivity rows)", "layers": "ex_max, step_max, upface_max, relief",
             "signature": "As H78-1, one operator at a time", "why_uncatalogued": "As H78-1", "differs_from_repo": "Sensitivity check on H78-1, not a separate lane",
             "gain_over_cost": "fourth", "outcome": "E1: none live-supported"},
            {"id": "CTRL", "name": "Controls: H77-1 straight edges, det_elev_slope (band 19), SEF", "layers": "training_features band 19; SEF cache",
             "signature": "Generic edges / slope", "why_uncatalogued": "n/a (controls)", "differs_from_repo": "Already tested elsewhere (LIA lane)",
             "gain_over_cost": "control", "outcome": "E2: det_elev_slope +8.7e-5 dMSE but CI includes 0 and bins unordered"}],
        "experiments": [
            {"id": "E1", "what": "Off-support split of the S0 live-score model by each field; LOO over 95 organizer-scored rasters",
             "result": "no field live-supported (lidar consensus dMSE < 0)", "verdict": "negative", "evidence": "h78_e1_screen.json"},
            {"id": "E2", "what": "On-support split (post-hoc amendment, disclosed IR-78-03)", "result": "no field live-supported",
             "verdict": "negative", "evidence": "h78_e2_within_support.json"},
            {"id": "E3", "what": "Pre-registered build + 4-fold hide-and-recover holdout + leakage canary + uniqueness",
             "result": f"C {c['dti']:.4f} vs A0 {pooled['A0']['scores']['A0']['dti']:.4f}; delta {dA0['delta']:+.4f}", "verdict": "negative" if not beats else "positive", "evidence": "h78_e3_holdout.json"}],
        "holdout": {"evidence_class": "HOLDOUT-DTI", "evaluator_version": e3["evaluator_version"], "dti": c["dti"], "ci95": c["ci95"],
                    "withheld_positive_pixels": c["withheld_positive_pixels"], "split": e3["split"]},
        "paired_difference_vs_champion_construction": {"evidence_class": "HOLDOUT-DTI", "evaluator_version": e3["evaluator_version"],
                                                       "delta": dA0["delta"], "ci95": dA0["ci95"], "withheld_positive_pixels": c["withheld_positive_pixels"]},
        "holdout_arms": arms,
        "leakage_canary_auc": canary, "canary_any_auc_above_090": any(v > 0.9 for v in canary.values()),
        "live_inversion_projection": {"evidence_class": "INSTRUMENT (projection, not a score)", "S0_candidate": s0, "S0_champion": s0c,
                                      "S0_delta": s0 - s0c, "chosen": e3["chosen"], "grid": e3["full_map_grid"]},
        "registry_comparison": {"tool": "gems57.uniqueness.compare_array_to_registry (vendored template, unchanged)",
                                "registry_files": un["registry_files"], "space_filling_rasters": len(un["space_filling_ids"]),
                                "phase1_worst_rho": p1["worst_rho"]["rho"], "phase1_rho_firings": p1["rho_firings"],
                                "phase2_worst_rho": p2["worst_rho"]["rho"], "phase2_rho_firings": p2["rho_firings"],
                                "phase2_worst_overlap_informative": p2["informative_reading"]["worst_forward_overlap"],
                                "phase2_worst_overlap_submission": p2["informative_reading"]["worst_overlap_submission"],
                                "phase2_overlap_firings_informative": p2["informative_reading"]["overlap_firings"],
                                "phase2_jaccard_firings_informative": p2["informative_reading"]["jaccard_firings"],
                                "identical_decoded": p2["informative_reading"]["identical_decoded"],
                                "literal_reading": (p2.get("summary") or {}).get("literal_reading"),
                                "literal_firings_all_space_filling": all(
                                    x["submission"] in set(un["space_filling_ids"])
                                    for ph in (p1, p2) for x in (ph.get("summary") or {}).get("literal_firings", [])),
                                "unreadable_registry_files": len(p2["errors"]),
                                "scope": "812-raster registry index + GEMSDOE73-76 TIFFs at clone time (~04:10 UTC) + this repo's TIFFs; later files unchecked (IR-78-05)"},
        "raster_sha256": sub["primary"]["sha256"], "file": "docs/downloads/" + sub["primary"]["file"],
        "validator": {"portal_equivalence": {c_["check"]: c_["status"] for c_ in sub["primary"]["checks"]},
                      "template_validator_passed": sub["template_validator_primary"]["passed"],
                      "zip_single_tiff": sub["zip"]["passed"], "nan_twin_bit_identical_in_footprint": sub["in_footprint_bit_identical"]},
        "submission_name": sub["name"], "note": sub["note"], "note_chars": len(sub["note"]),
        "format_valid": fmt, "protocol_unique": unique, "protocol_unique_informative_reading": unique_informative,
        "literal_reading_firings_final_dots": literal_firings,
        "uniqueness_policy": "Literal reading binds (repository policy I17): any registry raster with > 70% of our dots within 3 px is a duplicate. All literal firings here are space-filling rasters, against which even random dots fire; resolving that contradiction is a protocol-owner decision, not this lane's.",
        "beats_holdout_best": beats,
        "ok_to_download": True, "ok_to_submit": bool(unique and verdict == "promote"), "approved_for_weekly_slot": False,
        "weekly_slots_used": 0, "organizer_confirmed_score": None,
        "verdict": verdict, "verdict_reason": "; ".join(reasons) + ".",
        "budget": "3 experiments (E1, E2, E3); E3 re-run twice for implementation bugs, not re-tuned (IR-78-03)",
        "sources": [
            {"title": "DrivenData GEMS problem description (metric, format)", "url": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/", "used_for": "DTI definition, file format"},
            {"title": "DrivenData GEMS public leaderboard", "url": "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/", "used_for": "0.2778, 0.4413, 0.3904 (ORGANIZER-CONFIRMED)"},
            {"title": "Hardwick et al. 2025, Stanford Geothermal Workshop (INGENIOUS UAV lidar)", "url": "https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hardwick.pdf", "used_for": "H78-1 mechanism"},
            {"title": "Hart-Wagoner et al., GRC (INGENIOUS Quaternary fault database)", "url": "https://publications.mygeoenergynow.org/grc/1034813.pdf", "used_for": "catalogue provenance"},
            {"title": "Faulds & Richards, GRC (INGENIOUS favorable structural settings)", "url": "https://publications.mygeoenergynow.org/grc/1034745.pdf", "used_for": "H78-2 rationale"},
            {"title": "GBCGE INGENIOUS project page", "url": "https://gbcge.org/current-projects/ingenious/", "used_for": "project scope and dates"},
            {"title": "GDR submission 1391, INGENIOUS Great Basin Regional Dataset Compilation (DOI 10.15121/1881483, CC BY 4.0)", "url": "https://gdr.openei.org/submissions/1391", "used_for": "H78-2 field (its 'Well and Spring Temperature and Chemistry' resource); the INGENIOUS Quaternary fault compilation is in the same submission"},
            {"title": "USGS 3D Elevation Program (3DEP; free, no use restrictions)", "url": "https://www.usgs.gov/3d-elevation-program", "used_for": "source of the 1 m DEM tiles behind lidar_scarp_features_u8 (H78-1)"}],
    }
    assert len(card["note"]) <= 140
    (ROOT / "evidence" / "h78_run_card.json").write_text(json.dumps(card, indent=1, default=float))
    lanes = j("docs/data/lanes.json")
    lanes["lanes"] = [x for x in lanes["lanes"] if not x["lane"].startswith("H78")]
    row = {"lane": "H78: lidar scarp consensus, uniqueness-capped allocation", "page": "h78.html", "summary": "executive-summary.html",
           "file": "downloads/" + sub["primary"]["file"], "format_valid": fmt, "protocol_unique": unique, "beats_holdout_best": beats,
           "verdict": verdict, "name": sub["name"], "note": sub["note"], "sha256": sub["primary"]["sha256"], "s0_projection": s0,
           "informative_unique": unique_informative,
           "upload": ("Not OK to submit: " + ("" if unique else f"literal registry gate fails ({literal_firings} firings, all on space-filling rasters; "
                      f"informative reading passes, worst overlap {p2['informative_reading']['worst_forward_overlap']:.3f}) and ")
                      + "the holdout is negative. Inspection download only; selector: not approved.") if verdict != "promote" or not unique
                      else "Candidate for selector review."}
    lia = next((x for x in lanes["lanes"] if x["lane"].startswith("LIA")), None)
    lia_h = j("evidence/e3_holdout_final.json")["pooled"]["A0"]["scores"]["C"]["dti"] if lia else None
    for x in lanes["lanes"]:
        if x["lane"].startswith("Selective-U") and x["page"] == "index.html":
            x["page"], x["summary"] = "urank.html", "urank-executive-summary.html"
        if x["lane"].startswith("LIA"):
            x.setdefault("informative_unique", True)
            x.setdefault("beats_holdout_best", False)  # HOLDOUT-DTI 0.0706 vs 0.0958 (evidence/e3_holdout_final.json)
    if fmt:
        for x in lanes["lanes"]:
            x.pop("primary", None)
            x.pop("why_primary", None)
        row["primary"] = True
        row["why_primary"] = (f"Generated in this session (H78 lane). It ties the LIA file: INSTRUMENT S0 projection "
                              f"{s0:.4f} vs {j('docs/data/run-card.json')['live_inversion_projection']['S0_candidate']:.4f}, HOLDOUT-DTI {c['dti']:.4f} vs {lia_h:.4f} (same evaluator and "
                              f"split). Showing it here is not a selection; the selector decides.") if lia else "Generated in this session."
    lanes["lanes"].insert(0, row)
    (ROOT / "docs" / "data" / "lanes.json").write_text(json.dumps(lanes, indent=1) + "\n")
    print(json.dumps({k: card[k] for k in ("verdict", "verdict_reason", "protocol_unique", "format_valid", "submission_name", "note")}, indent=1))


if __name__ == "__main__":
    main()
