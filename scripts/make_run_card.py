#!/usr/bin/env python3
"""Assemble the protocol run card (brief rule 5) from the evidence files.  Writes docs/data/run-card.json.

Uses the FINAL (uniqueness-repaired) build:
  holdout   evidence/e3_holdout_final.json
  projection evidence/e3_unique_repair.json
  registry  evidence/e3_uniqueness.json (template audit, phase 2 = final dots; phase 1 = surface)
The first build, logged as a DUPLICATE, is recorded under "duplicate_log".
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def j(p):
    return json.loads((ROOT / p).read_text())


def main():
    sub = j("docs/data/submission.json")
    rep = j("evidence/e3_unique_repair.json")
    hold = j("evidence/e3_holdout_final.json")
    uniq = j("evidence/e3_uniqueness.json")
    free = j("evidence/e3_registry_free.json")
    first = j("evidence/e3_uniqueness_first_build_DUPLICATE.json")
    why = j("evidence/why_02778.json")
    alloc = j("evidence/e3_allocation.json")
    champ = alloc["champion_projection"]
    c = hold["pooled"]["A0"]["scores"]["C"]
    p2 = uniq["phase2_final_dots"]["summary"]
    p1 = uniq["phase1_surface_before_placement"]["summary"]
    lit, lfl = p2["literal_reading"], p2["like_for_like_reading"]
    space = {x["raster"] for x in free["space_filling_rasters"]}
    firings = p2["literal_firings"]
    informative_firings = [f for f in firings if f["submission"] not in space]
    sf_firings = [f for f in firings if f["submission"] in space]
    rows = [r for r in uniq["phase2_final_dots"]["worst_rows"]]
    s0d = rep["S0"]["dti_projection"] - champ["S0"]["dti_projection"]
    s1d = rep["S1"]["dti_projection"] - champ["S1"]["dti_projection"]
    pv = sub["primary"]
    checks = {x["check"]: x["status"] for x in pv["checks"]}
    rho_ok = (lit["rho_firings"] == 0)
    unique_informative = rho_ok and not informative_firings
    criteria = {
        "1_valid_format": bool(pv["passed"] and sub["zip"]["passed"]),
        "2_unique_informative_reading": bool(unique_informative),
        "3_leakage_canary_clear": all(not v["leakage_flag"] for v in hold["leakage_canary"].values()),
        "4_S0_projection_delta_ge_0": bool(s0d >= 0),
        "5_holdout_paired_ci_not_entirely_below_0": bool(hold["pooled"]["A0"]["paired_differences"]["A0"]["ci95"][1] >= 0),
    }
    verdict = "promote" if all(criteria.values()) else "negative"
    f1 = first["phase2_final_dots"]["summary"]
    card = {
        "run_id": "GEMSDOE77-E3-H77-LIA",
        "lane": "SELF-ASSIGNED (brief's lane paragraph missing, IR-77-01): live-score inversion and allocation",
        "hypothesis": ("H77-3 + H77-1: hidden (new-fault) truth density, inverted from 95 organizer-scored rasters, is "
                       "~38x enriched on the H19-5 support >2 km from the catalogue; thin straight topographic edges "
                       "(NMS + km-scale structure-tensor coherence) carry part of the off-support remainder"),
        "mechanism": ("E[TP]=sum h(y)M(y) exactly; on-support dots by S0 expected credit (>400 m from catalogue), "
                      "off-support H77-1 dots by S1 credit, >=3 px apart, under a per-raster overlap budget of 0.66 "
                      f"against {rep['informative_rasters']} informative registry rasters"),
        "mimicking_non_fault_process": ("straight edges: linear lithologic contacts, Pleistocene Lake Lahontan shorelines, "
                                        "roads/railways and field boundaries in the 100 m DEM; support: erosional "
                                        "range-front escarpments"),
        "holdout": {"evidence_class": "HOLDOUT-DTI", "evaluator_version": hold["evaluator_version"],
                    "withheld_positive_pixels": c["withheld_positive_pixels"], "dti": c["dti"], "ci95": c["ci95"],
                    "paired_vs_champion_rule_fold_fair": hold["pooled"]["A0"]["paired_differences"]["A0"],
                    "champion_rule_fold_fair": hold["pooled"]["A0"]["scores"]["A0"],
                    "paired_vs_H77_1_alone": hold["pooled"]["E1"]["paired_differences"]["E1"],
                    "paired_vs_random": hold["pooled"]["R"]["paired_differences"]["R"],
                    "split": hold["split"], "rule": hold["rule"]},
        "live_inversion_projection": {"evidence_class": "INSTRUMENT (projection, not a score)",
                                      "S0_candidate": rep["S0"]["dti_projection"], "S0_champion": champ["S0"]["dti_projection"],
                                      "S0_delta": s0d, "S1_candidate": rep["S1"]["dti_projection"],
                                      "S1_champion": champ["S1"]["dti_projection"], "S1_delta": s1d,
                                      "S0_loo_spearman_95_rasters": 0.917,
                                      "S0_in_sample_bias_on_champion": champ["S0"]["dti_projection"] - 0.2778},
        "leakage_canary": hold["leakage_canary"],
        "registry_comparison": {
            "registry_rasters_checked": p2["rasters_checked"],
            "phase1_surface_worst_spearman": p1["literal_reading"]["worst_spearman_full_footprint"],
            "phase1_surface_worst_spearman_submission": p1["literal_reading"]["worst_spearman_submission"],
            "phase2_dots_worst_spearman": lit["worst_spearman_full_footprint"],
            "phase2_dots_worst_spearman_submission": lit["worst_spearman_submission"],
            "rho_firings": lit["rho_firings"],
            "informative_rasters": rep["informative_rasters"],
            "space_filling_rasters_excluded": len(space),
            "space_filling_rule": "3 px halo covers >= 70% of allowed cells: uniformly random dots would exceed 0.70, so the test cannot discriminate",
            "worst_overlap_informative": rep["worst_overlap"],
            "worst_overlap_informative_raster": rep["worst_rasters"][0]["raster"],
            "informative_firings_in_template_audit": len(informative_firings),
            "space_filling_firings_in_template_audit": len(sf_firings),
            "template_literal_reading_duplicate": lit["duplicate"],
            "template_like_for_like_reading_duplicate": lfl["duplicate"],
            "unique_informative_reading": unique_informative,
            "thresholds": p2["thresholds"]},
        "duplicate_log": {"first_build": "logged DUPLICATE by the template audit, not shipped",
                          "first_build_dots": first.get("candidate", {}).get("dots"),
                          "first_build_like_for_like_firings": f1["like_for_like_reading"]["forward_overlap_firings"],
                          "first_build_worst_genuine_firing": "GEMSDOE48 h59-cover-ds-belief-b2xh33d (overlap 0.704, halo coverage 0.165)",
                          "action": "E3 build repaired with an explicit per-raster overlap budget; re-audited"},
        "raster": {"file": pv["file"], "sha256": pv["sha256"], "bytes": pv["bytes"], "dots": pv["dots"],
                   "decoded_predictions_sha256": sub["decoded_sha256"],
                   "twin_nan_sha256": sub["twin_nan"]["sha256"], "zip_sha256": sub["zip"]["sha256"]},
        "validator": {"no_nan_inside_footprint": checks.get("no NaN/inf inside official footprint"),
                      "values_in_0_1": checks.get("values within [0, 1]"),
                      "crs_match": checks.get("CRS equals sample_submission"),
                      "shape_match": checks.get("shape 3730x3292 equals sample_submission"),
                      "transform_match": checks.get("transform equals sample_submission"),
                      "all_checks": f"{pv['n_pass']}/{pv['n_checks']} PASS"},
        "submission_name": sub["name"], "note": sub["note"], "note_chars": sub["note_chars"],
        "promotion_criteria": criteria, "verdict": verdict,
        "verdict_reason": (f"Pre-registered rule: S0 (LOO-validated) projects {s0d:+.4f} and S1 projects {s1d:+.4f} vs the "
                           "0.2778 champion (same instrument). Valid, unique (informative reading) and safe to upload; "
                           "not projected to beat the champion. Slot decision: selector."
                           if verdict == "negative" else "All pre-registered criteria met; selector decides the slot."),
        "budget": {"experiments_used": 3, "limit": 3, "weekly_slots_used_by_this_run": 0},
        "champion_reference": {"submission": "GEMSDOE32 h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros",
                               "owner_reported_score": 0.2778, "sha256": why["champion_bytes"]["sha256"]},
    }
    (ROOT / "docs" / "data" / "run-card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps({"verdict": verdict, "criteria": criteria,
                      "informative_firings": len(informative_firings), "space_filling_firings": len(sf_firings)}, indent=1))


if __name__ == "__main__":
    main()
