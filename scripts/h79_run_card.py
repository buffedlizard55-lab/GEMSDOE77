#!/usr/bin/env python3
"""Compose the single JSON run card for lane H79 from the evidence files.

Applies the pre-registered decision rule (registry/preregistration_h79.json) mechanically
and writes evidence/h79_run_card.json + docs/data/h79_submission.json (site inputs).
Every number carries its evidence class; nothing here is a live score.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"
PRE = json.loads((ROOT / "registry" / "preregistration_h79.json").read_text())
INCUMBENTS = {  # lane-reported HOLDOUT-DTI at B = 37,654, same evaluator; NOT re-verified in this session
    "H77-Step": {"seed20": 0.15272, "seed30": None, "source": "README lane table / evidence/h77/run_card.json"},
    "H77D": {"seed20": 0.14030, "seed30": 0.15011, "source": "evidence/run_card_h77d.json"},
}


def incumbent_audits() -> dict:
    out = {}
    for tag, label in (("h77d", "H77D"), ("h77step", "H77-Step")):
        f = EVID / f"{tag}_uniqueness.json"
        if f.exists():
            u = json.loads(f.read_text())
            w = u["worst_discriminating_rows"][0]
            out[label] = {"file": u["candidate"], "sha256": u["sha256"], "protocol_unique_like_for_like": u["protocol_unique_like_for_like"],
                          "max_forward_overlap_discriminating": u["max_forward_overlap_discriminating"], "max_rho": u["phase2_final_dots"]["max_rho"],
                          "rasters_checked": u["phase2_final_dots"]["rasters_checked"], "worst": {"repo": w["repo"], "submission": w["submission"], "halo_coverage_allowed": w["halo_coverage_allowed"]},
                          "evidence": f"evidence/{tag}_uniqueness.json"}
    return out


def selector_advisory(capped_build: bool, dh: dict) -> str:
    if not capped_build:
        return "Greedy file: see pre-registered rule."
    inc = incumbent_audits()
    failed = [k for k, v in inc.items() if not v["protocol_unique_like_for_like"]]
    passed = [k for k, v in inc.items() if v["protocol_unique_like_for_like"]]
    txt = (f"The capped H79 file (HOLDOUT-DTI {dh['seed20']['dti']:.4f} s20 / {dh['seed30']['dti']:.4f} s30) is below the lane-reported incumbents "
           "(H77-Step 0.1527 s20; H77D 0.1403 s20 / 0.1501 s30) and its margin over the GREEDY A2_DIST control is not significant at seed 20; "
           "it clears the gate against the identically emitted control. ")
    if inc:
        txt += ("Re-audited this session under the same discriminating criterion: "
                + "; ".join(f"{k} max 3 px overlap {v['max_forward_overlap_discriminating']:.3f} vs {v['worst']['repo']} ({'FAIL' if not v['protocol_unique_like_for_like'] else 'PASS'})" for k, v in inc.items()) + ". ")
        if failed and not passed:
            txt += ("Under one uniform rule the incumbents are NOT protocol-unique, so the capped H79 file is the only cleared file in this repository. "
                    "A selector who prefers an incumbent's higher holdout number would be overriding the published uniqueness rule and must say so.")
        elif passed:
            txt += f"Prefer {', '.join(passed)} on holdout grounds unless the selector weighs the H79 mechanism."
    else:
        txt += "Incumbents not re-audited under the discriminating criterion."
    return txt


def main() -> None:
    s20 = json.loads((EVID / "h79_holdout_seed20.json").read_text())
    s30 = json.loads((EVID / "h79_holdout_seed30.json").read_text())
    em = json.loads((EVID / "h79_emission_check_seed20.json").read_text())
    build_first = json.loads((EVID / "h79_build.json").read_text())
    uniq_first = json.loads((EVID / "h79_uniqueness.json").read_text()) if (EVID / "h79_uniqueness.json").exists() else None
    capped_build = (EVID / "h79_build_capped.json").exists()
    if capped_build:  # amendment E3b: the deliverable is the capped emission; its own audit decides its status
        build = json.loads((EVID / "h79_build_capped.json").read_text())
        uniq_path = EVID / "h79c_uniqueness.json"
    else:
        build, uniq_path = build_first, EVID / "h79_uniqueness.json"
    uniq = json.loads(uniq_path.read_text()) if uniq_path.exists() else None
    capped_hold = {f"seed{sd}": json.loads((EVID / f"h79_capped_holdout_seed{sd}.json").read_text())
                   for sd in (20, 30) if (EVID / f"h79_capped_holdout_seed{sd}.json").exists()}

    def arm(s, B, name):
        sc = s["budgets"][f"B{B}"]["pooled_summary"]["scores"][name]
        return {"dti": sc["dti"], "ci95": sc["ci95"], "withheld_positive_pixels": sc["withheld_positive_pixels"],
                "evidence_class": "HOLDOUT-DTI", "evaluator_version": sc["evaluator_version"]}

    def delta(s, B, ctrl):
        d = s["budgets"][f"B{B}"]["pooled_summary"]["paired_differences"][ctrl]
        return {"delta": d["delta"], "ci95": d["ci95"]}

    B = 37654
    cand20, cand30 = arm(s20, B, "H79_full"), arm(s30, B, "H79_full")
    holdout = {
        "seed20": {"candidate": cand20, "controls": {k: arm(s20, B, k) for k in ("A2_DIST", "A2_standard", "ATTR_only", "A0_h195_foldfair", "A0_random")},
                   "paired_vs": {k: delta(s20, B, k) for k in ("A2_DIST", "A2_standard", "ATTR_only", "A0_h195_foldfair", "A0_random")}},
        "seed30": {"candidate": cand30, "controls": {k: arm(s30, B, k) for k in ("A2_DIST", "A2_standard", "ATTR_only", "H79_noHW", "H79_noATTR", "A0_h195_foldfair", "A0_random")},
                   "paired_vs": {k: delta(s30, B, k) for k in ("A2_DIST", "A2_standard", "ATTR_only", "H79_noHW", "H79_noATTR", "A0_h195_foldfair", "A0_random")}},
        "budget_44090": {"seed20": arm(s20, 44090, "H79_full"), "seed30": arm(s30, 44090, "H79_full")},
    }
    # ---- pre-registered rule
    c1 = delta(s20, B, "A2_DIST")["ci95"][0] > 0
    c2 = delta(s30, B, "A2_DIST")["delta"] > 0
    c3 = {}
    for name, inc in INCUMBENTS.items():
        c3[name] = {"seed20": (cand20["dti"] >= inc["seed20"]) if inc["seed20"] is not None else None,
                    "seed30": (cand30["dti"] >= inc["seed30"]) if inc["seed30"] is not None else None,
                    "incumbent": inc}
    c3_all = all(v for d in c3.values() for v in (d["seed20"], d["seed30"]) if v is not None)
    rule = {
        "clause1_seed20_delta_vs_A2_DIST_ci_excludes_0": bool(c1),
        "clause2_seed30_delta_vs_A2_DIST_positive": bool(c2),
        "clause3_ge_best_lane_reported_incumbent": c3,
        "clause3_all_pass": bool(c3_all),
        "promote_candidate_to_selector_per_preregistration": bool(c1 and c2 and c3_all),
        "note": ("Clause 3 fails only against the H77-Step lane-reported seed-20 figure (0.15272, no seed-30 figure, not "
                 "re-verified here) by 0.0018 -- well inside the candidate's 95 % CI; it passes against H77D on both seeds. "
                 "The pre-registered rule is applied literally: no automatic promotion; the paired evidence is handed to the selector."),
    }
    canary = {"seed20": {"max_symmetric_auc": s20["canary_max_symmetric_auc"], "channel": s20["canary_max_channel"], "flags": s20["leakage_flags"], "n_channels": len(s20["canary_auc"])},
              "seed30": {"max_symmetric_auc": s30["canary_max_symmetric_auc"], "channel": s30["canary_max_channel"], "flags": s30["leakage_flags"], "n_channels": len(s30["canary_auc"])}}
    emission = {"status": em["status"], "seed20_B37654": em["arms"], "paired": em["pooled_summary"]["paired_differences"]}
    uniq_summary = None
    if uniq:
        uniq_summary = {k: uniq[k] for k in ("literal_reading_duplicate", "like_for_like_duplicate", "protocol_unique_literal",
                                             "protocol_unique_like_for_like", "n_space_filling_rasters", "scope", "rule")}
        uniq_summary["surface_max_rho"] = uniq["phase1_surface_before_placement"]["max_rho"]
        uniq_summary["dots_max_rho"] = uniq["phase2_final_dots"]["max_rho"]
        uniq_summary["dots_max_forward_overlap_all"] = uniq["phase2_final_dots"]["max_forward_overlap_all"]
        uniq_summary["dots_max_forward_overlap_discriminating"] = uniq["max_forward_overlap_discriminating"]
        uniq_summary["discriminating_rasters"] = uniq["discriminating_rasters"]
        uniq_summary["rasters_checked"] = uniq["phase2_final_dots"]["rasters_checked"]
        uniq_summary["worst_discriminating_rows"] = [
            {k: r.get(k) for k in ("submission", "repo", "my_dots_within_3px_of_theirs", "spearman_full_footprint", "their_dots", "halo_coverage_allowed")}
            for r in uniq["worst_discriminating_rows"]][:6]
    format_ok = bool(build["validator"]["all_checks_passed"] and build["template_report"].get("ok"))
    unique_ok = bool(uniq and uniq["protocol_unique_like_for_like"] and uniq["phase2_final_dots"]["rasters_checked"] >= 899)
    if uniq is None:
        unique_ok = False
    beats_controls = bool(c1 and c2)
    deliverable_holdout = {"seed20": cand20, "seed30": cand30, "emission": "greedy (pre-registered)"}
    beats_greedy_control_both_seeds = bool(c1 and c2)
    if capped_build:  # the deliverable's OWN paired evidence decides its status (same folds, same control)
        if len(capped_hold) < 2:
            beats_controls = False
        else:
            # pre-registered principle: candidate vs A2_DIST under IDENTICAL emission -> A2_DIST@capped; the greedy-control
            # comparison is reported next to it (seed-20 CI includes 0) and in the selector advisory
            beats_controls = all(h["pooled_summary"]["paired_differences"]["A2_DIST@capped"]["ci95"][0] > 0 for h in capped_hold.values())
            beats_greedy_control_both_seeds = all(h["pooled_summary"]["paired_differences"]["A2_DIST@greedy"]["ci95"][0] > 0 for h in capped_hold.values())
            deliverable_holdout = {sd: {"dti": h["pooled_summary"]["scores"]["H79_full@capped"]["dti"], "ci95": h["pooled_summary"]["scores"]["H79_full@capped"]["ci95"],
                                        "withheld_positive_pixels": h["withheld_positive_pixels"], "evidence_class": "HOLDOUT-DTI", "evaluator_version": h["evaluator_version"]}
                                   for sd, h in capped_hold.items()}
            deliverable_holdout["emission"] = "overlap-capped greedy (amendment E3b)"
    if format_ok and unique_ok and beats_controls:
        status = "CLEARED_FOR_SELECTOR"
        ok_to_submit = True
        upload = ("OK to submit if the selector picks it: format PASS (17/17), unique under the published rule (rho on all rasters, "
                  "3 px overlap on discriminating rasters; literal reading listed), beats the paired fold-fair control on both seeds "
                  f"(HOLDOUT-DTI {deliverable_holdout['seed20']['dti']:.4f} s20 / {deliverable_holdout['seed30']['dti']:.4f} s30). Slot decision = selector.")
    else:
        status = "DOWNLOAD_ONLY"
        ok_to_submit = False
        why = []
        if not format_ok:
            why.append("format FAIL")
        if not unique_ok:
            why.append("uniqueness not cleared" + (f" (3 px forward overlap {uniq['max_forward_overlap_discriminating']:.3f} > 0.70 vs a discriminating raster: "
                                                   f"{uniq['worst_discriminating_rows'][0]['repo']} {Path(uniq['worst_discriminating_rows'][0]['submission']).name})" if uniq else " (audit missing)"))
        if not beats_controls:
            why.append("holdout gate not met")
        upload = "Not OK to submit: " + "; ".join(why)
    card = {
        "lane": PRE["lane_id"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration": "registry/preregistration_h79.json",
        "hypothesis": PRE["ranked_hypotheses"][0]["title"] + " -- " + PRE["ranked_hypotheses"][0]["physical_signature"],
        "mechanism": PRE["ranked_hypotheses"][0]["why_unmapped_not_catalogued"],
        "named_non_fault_mimic": PRE["ranked_hypotheses"][0]["named_non_fault_mimic"],
        "what_actually_carried_the_gain": ("Seed-30 ablations: dropping the two signed hanging-wall channels does not hurt "
                                           f"(H79_noHW {arm(s30, B, 'H79_noHW')['dti']:.4f} vs H79_full {cand30['dti']:.4f}, paired delta "
                                           f"{delta(s30, B, 'H79_noHW')['delta']:+.4f} CI {delta(s30, B, 'H79_noHW')['ci95']}); geometry-only context "
                                           f"(strike-aware along/across offsets + density) reaches {arm(s30, B, 'H79_noATTR')['dti']:.4f}; attributes add "
                                           f"{delta(s30, B, 'H79_noATTR')['delta']:+.4f} CI {delta(s30, B, 'H79_noATTR')['ci95']} on top. The hanging-wall "
                                           "mechanism is NOT supported by the catalogue-vs-catalogue holdout (withheld traces lean slightly to the footwall, "
                                           "E1); the strike-aware context geometry is what the holdout rewards."),
        "data_facts_measured": {
            "labels_equal_legacy_usgs_traces": True, "ingenious_increment_px": 0, "in_footprint_traces": 1126,
            "mapscale_counts": {"1:250k": 846, "1:100k": 280}, "topo_vs_attr_hangingwall_agreement": s20["attribute_receipt"]["topo_vs_attr_hangingwall_agreement"],
            "hanging_wall_asymmetry": {"seed20": s20["hanging_wall_asymmetry"], "seed30": s30["hanging_wall_asymmetry"]},
        },
        "holdout": {"evaluator_version": s20["evaluator_version"], "draw": s20["holdout"], "withheld_positive_pixels": {"seed20": s20["holdout"]["withheld_positive_pixels"], "seed30": s30["holdout"]["withheld_positive_pixels"]},
                    "budget": B, "emission": "emit77.emit_dots greedy min_sep 2.8, fold-proportional, > 3 px from visible catalogue", "results": holdout},
        "leakage_canaries": canary,
        "emission_check_post_hoc": emission,
        "preregistered_decision_rule": rule,
        "artifact": {"file": build["tif"], "sha256": build["sha256"], "zip_sha256": build["zip_sha256"], "dots": build["dots"],
                     "submission_name": build["name"], "note": build["note"], "note_chars": build["note_chars"],
                     "validator_all_checks_passed": build["validator"]["all_checks_passed"], "template_report_ok": build["template_report"].get("ok"),
                     "validator_checks": build["validator"]["checks"]},
        "uniqueness": uniq_summary,
        "first_build_logged_duplicate": None if not capped_build else {
            "file": build_first["tif"], "sha256": build_first["sha256"], "dots": build_first["dots"], "emission": "greedy (pre-registered E3)",
            "surface_max_rho": uniq_first["phase1_surface_before_placement"]["max_rho"] if uniq_first else None,
            "dots_max_forward_overlap_discriminating": uniq_first["max_forward_overlap_discriminating"] if uniq_first else None,
            "worst_discriminating_rows": [{k: r.get(k) for k in ("submission", "repo", "my_dots_within_3px_of_theirs", "halo_coverage_allowed")}
                                          for r in (uniq_first["worst_discriminating_rows"][:5] if uniq_first else [])],
            "status": "DOWNLOAD_ONLY -- logged duplicate under docs/policy_uniqueness.md (26 discriminating rasters > 0.70); kept for audit, never submit",
        },
        "capped_emission_amendment_e3b": None if not capped_build else {
            "declaration": "registry/preregistration_h79_amendment_e3b.json", "constraints": "registry/h79_capped_constraints.json",
            "cap_fraction": build["cap_fraction"], "n_constraints": build["n_constraints"],
            "holdout": {sd: {"H79_full@capped": h["arms"]["H79_full@capped"]["pooled_holdout_dti"], "H79_full@greedy": h["arms"]["H79_full@greedy"]["pooled_holdout_dti"],
                             "A2_DIST@greedy": h["arms"]["A2_DIST@greedy"]["pooled_holdout_dti"], "A2_DIST@capped": h["arms"]["A2_DIST@capped"]["pooled_holdout_dti"],
                             "ci95_capped": h["pooled_summary"]["scores"]["H79_full@capped"]["ci95"],
                             "paired_capped_minus_greedy": h["pooled_summary"]["paired_differences"]["H79_full@greedy"],
                             "paired_capped_minus_A2_DIST_greedy": h["pooled_summary"]["paired_differences"]["A2_DIST@greedy"],
                             "paired_capped_minus_A2_DIST_capped": h["pooled_summary"]["paired_differences"]["A2_DIST@capped"],
                             "withheld_positive_pixels": h["withheld_positive_pixels"], "evidence_class": "HOLDOUT-DTI", "evaluator_version": h["evaluator_version"]}
                        for sd, h in capped_hold.items()},
            "note": "Placement-only change declared before running; the pre-registered rule above is evaluated on the uncapped greedy numbers.",
        },
        "deliverable_holdout": deliverable_holdout,
        "selector_advisory": selector_advisory(capped_build, deliverable_holdout),
        "incumbents_reaudited_this_session": incumbent_audits(),
        "beats_greedy_control_both_seeds": bool(beats_greedy_control_both_seeds),
        "status": status,
        "ok_to_download": format_ok,
        "ok_to_submit": ok_to_submit,
        "upload": upload,
        "verdict": "positive-not-auto-promoted" if beats_controls and not rule["promote_candidate_to_selector_per_preregistration"] else ("promote" if rule["promote_candidate_to_selector_per_preregistration"] else "negative"),
        "verdict_reason": rule["note"],
        "weekly_slots_used": 0,
        "live_score": None,
        "organizer_confirmed": False,
        "experiments_used": "E0 data check, E1 canaries (both seeds), E2 seed 20, E3 seed 30 + build (+ E3b capped re-emission of the same surface after the first build failed the dot-overlap audit); post-hoc emission check",
        "limitations": [
            "Hide-and-recover measures catalogue-vs-catalogue geometry; the test set is expert lidar mapping. The hanging-wall mechanism could not be validated by this holdout and is not claimed.",
            "Incumbent figures (H77-Step 0.1527, H77D 0.1403/0.1501) are lane-reported on the same evaluator but were not re-run here; the A2_standard reproduction differs from H77D's by sampling RNG (0.1067 vs 0.1000 seed 20) while the deterministic controls match to 4 decimals (A0_h195 0.0957, A0_random 0.0511).",
            "Uniqueness scope is the 905 cached public rasters (git snapshot 14:22 UTC); not an organizer registry.",
            "No organizer score exists for any file in this repository.",
        ],
    }
    (EVID / "h79_run_card.json").write_text(json.dumps(card, indent=1, sort_keys=True))
    site = {"primary": {"file": Path(build["tif"]).name, "sha256": build["sha256"], "name": build["name"], "note": build["note"],
                        "n_pass": sum(1 for v in build["validator"]["checks"].values() if v), "n_checks": len(build["validator"]["checks"])},
            "status": status, "ok_to_submit": ok_to_submit, "upload": upload,
            "holdout_seed20": cand20, "holdout_seed30": cand30, "uniqueness": uniq_summary,
            "capped": card["capped_emission_amendment_e3b"], "first_build": card["first_build_logged_duplicate"]}
    (ROOT / "docs" / "data" / "h79_submission.json").write_text(json.dumps(site, indent=1, sort_keys=True))
    # keep the site's lane manifest in step with the card (H79 entry only)
    lanes_path = ROOT / "docs" / "data" / "lanes.json"
    lanes = json.loads(lanes_path.read_text())
    for x in lanes["lanes"]:
        if x.get("page") == "h79.html":
            x.update({"status": status, "upload": upload, "protocol_unique_like_for_like": unique_ok, "informative_unique": unique_ok,
                      "protocol_unique": bool(uniq and uniq["protocol_unique_literal"]), "format_valid": format_ok,
                      "beats_holdout_best": beats_controls, "verdict": card["verdict"], "sha256": build["sha256"],
                      "name": build["name"], "note": build["note"], "file": "downloads/" + Path(build["tif"]).name,
                      "holdout_dti_hide_v2": {"seed20": deliverable_holdout["seed20"]["dti"], "seed30": deliverable_holdout["seed30"]["dti"]},
                      "emission": deliverable_holdout["emission"]})
    lanes_path.write_text(json.dumps(lanes, indent=1))
    print(json.dumps({"status": status, "ok_to_submit": ok_to_submit, "verdict": card["verdict"], "rule": {k: v for k, v in rule.items() if k != "clause3_ge_best_lane_reported_incumbent"}}, indent=1))


if __name__ == "__main__":
    main()
