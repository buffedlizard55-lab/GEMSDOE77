#!/usr/bin/env python3
"""Assemble the protocol run card (brief rule 5) from the evidence files.  Writes docs/data/run-card.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def j(p):
    return json.loads((ROOT / p).read_text())


def main():
    sub = j("docs/data/submission.json")
    alloc = j("evidence/e3_allocation.json")
    hold = j("evidence/e3_holdout.json")
    uniq = j("evidence/e3_uniqueness.json")
    why = j("evidence/why_02778.json")
    c = hold["pooled"]["A0"]["scores"]["C"]
    p2 = uniq["phase2_final_dots"]["summary"]
    p1 = uniq["phase1_surface_before_placement"]["summary"]
    sel = alloc["selected"]
    champ = alloc["champion_projection"]
    s0d = sel["S0"]["dti_projection"] - champ["S0"]["dti_projection"]
    s1d = sel["S1"]["dti_projection"] - champ["S1"]["dti_projection"]
    pv = sub["primary"]
    checks = {x["check"]: x["status"] for x in pv["checks"]}
    lfl = p2["like_for_like_reading"]
    lit = p2["literal_reading"]
    unique_like = (not lfl["duplicate"])
    criteria = {
        "1_valid_format": bool(pv["passed"] and sub["zip"]["passed"]),
        "2_unique_like_for_like": bool(unique_like),
        "3_leakage_canary_clear": all(not v["leakage_flag"] for v in hold["leakage_canary"].values()),
        "4_S0_projection_delta_ge_0": bool(s0d >= 0),
        "5_holdout_paired_ci_not_entirely_below_0": bool(hold["pooled"]["A0"]["paired_differences"]["A0"]["ci95"][1] >= 0),
    }
    verdict = "promote" if all(criteria.values()) else "negative"
    card = {
        "run_id": "GEMSDOE77-E3-H77-LIA",
        "lane": "SELF-ASSIGNED (brief's lane paragraph missing, IR-77-01): live-score inversion and allocation",
        "hypothesis": ("H77-3 + H77-1: hidden (new-fault) truth density, inverted from 95 organizer-scored rasters, "
                       "is ~38x enriched on the H19-5 support >2 km from the catalogue. Thin straight topographic edges "
                       "(NMS + km-scale structure-tensor coherence) carry the off-support remainder"),
        "mechanism": ("E[TP]=sum h(y)M(y) exactly; dots placed greedily by expected credit, >=3 px apart, pruned below "
                      "alpha*DTI; 65% on the live-validated support (B=4 px), 35% on H77-1 straight edges >3 px from it"),
        "mimicking_non_fault_process": ("Straight edges: linear lithologic contacts, shorelines of Pleistocene Lake Lahontan, "
                                        "roads/railways and agricultural field boundaries in the 100 m DEM; support: "
                                        "range-front erosional escarpments"),
        "holdout": {"evidence_class": "HOLDOUT-DTI", "evaluator_version": hold["evaluator_version"],
                    "withheld_positive_pixels": c["withheld_positive_pixels"], "dti": c["dti"], "ci95": c["ci95"],
                    "paired_vs_champion_rule_fold_fair": hold["pooled"]["A0"]["paired_differences"]["A0"],
                    "paired_vs_H77_1_alone": hold["pooled"]["E1"]["paired_differences"]["E1"],
                    "paired_vs_random": hold["pooled"]["R"]["paired_differences"]["R"],
                    "split": hold["split"]},
        "live_inversion_projection": {"evidence_class": "INSTRUMENT (projection, not a score)",
                                      "S0_candidate": sel["S0"]["dti_projection"], "S0_champion": champ["S0"]["dti_projection"],
                                      "S0_delta": s0d, "S1_candidate": sel["S1"]["dti_projection"],
                                      "S1_champion": champ["S1"]["dti_projection"], "S1_delta": s1d,
                                      "S0_loo_spearman_95_rasters": 0.917},
        "leakage_canary": hold["leakage_canary"],
        "registry_comparison": {
            "registry_rasters_checked": p2["rasters_checked"],
            "phase1_surface_worst_spearman": p1["literal_reading"]["worst_spearman_full_footprint"],
            "phase1_surface_worst_spearman_submission": p1["literal_reading"]["worst_spearman_submission"],
            "phase2_dots_worst_spearman": lit["worst_spearman_full_footprint"],
            "phase2_dots_worst_spearman_submission": lit["worst_spearman_submission"],
            "phase2_worst_overlap_dot_fields": lfl["worst_forward_overlap_dot_fields"],
            "phase2_worst_overlap_dot_field_submission": lfl["worst_forward_overlap_dot_field_submission"],
            "literal_worst_overlap_any_raster": lit["worst_forward_overlap"],
            "literal_worst_overlap_submission": lit["worst_overlap_submission"],
            "literal_forward_overlap_firings": lit["forward_overlap_firings"],
            "like_for_like_duplicate": lfl["duplicate"], "literal_duplicate": lit["duplicate"],
            "thresholds": p2["thresholds"]},
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
        "verdict_reason": ("Pre-registered rule: S0 (LOO-validated) projects the candidate below the champion "
                           f"(delta {s0d:+.4f}); S1 (in-sample) projects {s1d:+.4f}. Valid, unique and safe to upload; "
                           "a live score would measure off-support straight-edge credit. Slot decision: selector."
                           if verdict == "negative" else "All pre-registered criteria met; selector decides the slot."),
        "budget": {"experiments_used": 3, "limit": 3, "weekly_slots_used_by_this_run": 0},
        "champion_reference": {"submission": "GEMSDOE32 h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros",
                               "owner_reported_score": 0.2778, "sha256": why["champion_bytes"]["sha256"]},
    }
    (ROOT / "docs" / "data" / "run-card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps({"verdict": verdict, "criteria": criteria}, indent=1))


if __name__ == "__main__":
    main()
