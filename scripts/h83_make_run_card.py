#!/usr/bin/env python3
"""Compile the H83-NETSIB run card from recorded evidence (no hand-typed numbers).

The verdict is computed mechanically from registry/preregistration_h83_netsib.json
promotion_rule; every number is copied from evidence/h83_*.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
INCUMBENT = 0.152718


def j(p):
    return json.loads((ROOT / p).read_text())


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pre = j("registry/preregistration_h83_netsib.json")
    e1 = j("evidence/h83_e1_canary_surface.json")
    e2 = j("evidence/h83_e2_holdout.json")
    e3 = j("evidence/h83_e3_surface.json")
    b = j("evidence/h83_e3_build.json")
    tif = ROOT / b["file"]
    assert sha(tif) == b["file_sha256"] == b["validator"]["sha256"], "hash binding failed"
    seeds = {}
    for s, v in e2["seeds"].items():
        sc = v["summary"]["scores"]
        pdiff = v["summary"]["paired_differences"]
        seeds[s] = {
            "label": "HOLDOUT-DTI", "evaluator_version": e2["evaluator_version"],
            "withheld_positive_pixels": sc["A_netsib_full"]["withheld_positive_pixels"],
            "candidate_A_netsib_full": {"dti": round(sc["A_netsib_full"]["dti"], 6),
                                        "ci95": [round(x, 6) for x in sc["A_netsib_full"]["ci95"]]},
            "control_A_base": {"dti": round(sc["A_base"]["dti"], 6), "ci95": [round(x, 6) for x in sc["A_base"]["ci95"]]},
            "arm_S_netsib_only": {"dti": round(sc["S_netsib_only"]["dti"], 6), "ci95": [round(x, 6) for x in sc["S_netsib_only"]["ci95"]]},
            "random_A0": {"dti": round(sc["A0_random"]["dti"], 6), "ci95": [round(x, 6) for x in sc["A0_random"]["ci95"]]},
            "paired_delta_vs_A_base": {"delta": round(pdiff["A_base"]["delta"], 6), "ci95": [round(x, 6) for x in pdiff["A_base"]["ci95"]]},
            "paired_delta_vs_S_netsib_only": {"delta": round(pdiff["S_netsib_only"]["delta"], 6), "ci95": [round(x, 6) for x in pdiff["S_netsib_only"]["ci95"]]},
            "bootstrap": v["summary"]["bootstrap"]["method"] + f", {v['summary']['bootstrap']['clusters']} clusters, {v['summary']['bootstrap']['draws']} draws",
        }
    c = {s: seeds[s]["candidate_A_netsib_full"]["dti"] for s in seeds}
    rule_a = all(v > INCUMBENT for v in c.values())
    rule_b = all(seeds[s]["paired_delta_vs_A_base"]["ci95"][0] > 0 for s in seeds)
    rule_c = bool(e1["leakage_pass"])
    u = b["final_dot_uniqueness"]
    surf_ok = e1["deterministic_netsib_surface_scan"]["pass"] and e3["surface_scan_before_placement"]["pass"]
    rule_d = bool(surf_ok and not u["duplicate"])
    val_ok = bool(b["validator"]["all_checks_passed"])
    promote = rule_a and rule_b and rule_c and rule_d and val_ok
    rows = json.loads((EV / "h83_e3_registry_rows.json").read_text())
    firings = sorted(({"raster": r["raster"], "forward_overlap_3px": round(r["forward_overlap_3px"], 4),
                       "their_dots": r["their_dots"], "their_3px_cover_of_allowed": round(r["their_3px_cover_of_allowed"], 4),
                       "chance_corrected": round((r["forward_overlap_3px"] - r["their_3px_cover_of_allowed"]) / (1 - r["their_3px_cover_of_allowed"]), 4),
                       "spearman": round(r["spearman_full_footprint"] or 0.0, 4)}
                      for r in rows if r.get("informative") and r["forward_overlap_3px"] > 0.70),
                     key=lambda r: -r["forward_overlap_3px"])
    comparable = [r for r in rows if "forward_overlap_3px" in r and r["their_dots"] <= 2 * 37654]
    max_comparable = max((r["forward_overlap_3px"] for r in comparable), default=None)
    bits = []
    if not rule_a: bits.append("did not beat the 0.152718 incumbent on both seeds")
    if not rule_b: bits.append("paired lift over the base control not significant")
    if not rule_c: bits.append("leakage canary fired")
    if not rule_d: bits.append(f"DUPLICATE logged: final dots exceed the 0.70 forward 3-px overlap limit against {len(firings)} informative registry rasters (worst {firings[0]['forward_overlap_3px'] if firings else 'n/a'}; all with |rho| <= {max((abs(f['spearman']) for f in firings), default=0):.3f})")
    if not val_ok: bits.append("format validator failed")
    reason = "All promotion conditions met." if promote else "Stopped as preregistered: " + "; ".join(bits) + "."
    ci_lb_beats = all(seeds[s]["candidate_A_netsib_full"]["ci95"][0] > INCUMBENT for s in seeds)
    card = {
        "schema": "gems-parallel-run-card-v1",
        "lane": "H83-NETSIB",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branch": "arena/e8283608-gemsdoe77",
        "preregistration": "registry/preregistration_h83_netsib.json",
        "hypothesis": pre["hypothesis"],
        "mechanism": pre["mechanism"],
        "named_non_fault_mimic": pre["named_non_fault_mimics"][0],
        "other_mimics": pre["named_non_fault_mimics"][1:],
        "experiments_used": 3,
        "leakage_canary": {"max_symmetric_auc_by_seed": {s: round(v["max_auc"], 4) for s, v in e1["canaries"].items()},
                           "max_feature_by_seed": {s: v["max_feature"] for s, v in e1["canaries"].items()},
                           "ceiling": 0.90, "pass": rule_c},
        "holdout_dti": seeds,
        "incumbent": {"label": "HOLDOUT-DTI", "score": INCUMBENT, "evaluator_version": "gems57-pooled-hide-v2",
                      "withheld_positive_pixels": 11321, "ci95": [0.131548, 0.175185], "source": "evidence/h77/run_card.json"},
        "beats_incumbent_point_estimate_both_seeds": rule_a,
        "beats_incumbent_ci_lower_bound_both_seeds": ci_lb_beats,
        "seed20_delta_vs_incumbent_point": round(c["20"] - INCUMBENT, 6),
        "registry": {
            "pre_placement_deterministic_surface_max_abs_spearman": e1["deterministic_netsib_surface_scan"]["max_abs_rho"],
            "pre_placement_deterministic_surface_worst": e1["deterministic_netsib_surface_scan"]["worst_raster"],
            "pre_placement_candidate_surface_max_abs_spearman": e3["surface_scan_before_placement"]["max_abs_rho"],
            "pre_placement_candidate_surface_worst": e3["surface_scan_before_placement"]["worst_raster"],
            "surface_rasters_compared": e3["surface_scan_before_placement"]["rasters_compared_unique_decoded"],
            "final_dots": {k: v for k, v in u.items() if k != "errors"},
            "scan_errors": u["errors"],
            "interpretation": "IR-83-01: dot-overlap applied where informative (their 3-px cover < 70% of allowed domain) plus matched-budget top-37,654 for dense rasters; literal all-raster value reported alongside.",
            "scope": "Every raster in registry/registry_index.json (812 entries, built 2026-10-21T02:30Z) fetched and SHA-verified this session, plus both pinned witnesses and every local TIFF; deduplicated by decoded pixels. Not a complete organizer registry.",
        },
        "raster_sha256": b["file_sha256"],
        "raster_bytes": b["bytes"],
        "zip_sha256": b["zip_sha256"],
        "download": b["download"], "zip": b["zip"],
        "validator_output": {"all_checks_passed": val_ok, "checks": b["validator"]["checks"],
                             "min": b["validator"]["min"], "max": b["validator"]["max"],
                             "n_nan": b["validator"]["n_nan"], "emitted_positive_pixels": b["validator"]["emitted_positive_pixels"],
                             "on_catalogue_positive_pixels": b["validator"].get("on_catalogue_positive_pixels"),
                             "meta": b["validator"]["meta"]},
        "submission_name": b["submission_name"], "note": b["note"], "note_chars": b["note_chars"],
        "promotion_rule_results": {"a_beats_incumbent_both_seeds": rule_a, "b_paired_delta_vs_base_ci_excludes_0": rule_b,
                                   "c_no_leakage": rule_c, "d_uniqueness": rule_d, "format_validator": val_ok},
        "verdict": "promote" if promote else "negative",
        "verdict_reason": reason,
        "duplicate_logged": bool(u["duplicate"]),
        "informational_overlap_firings": firings,
        "ok_to_download": True,
        "ok_to_submit": promote,
        "weekly_slots_used": 0,
        "organizer_score": None,
        "caveats": [
            "HOLDOUT-DTI is a local hide-and-recover measurement, not a leaderboard score; no leaderboard projection is made.",
            "Seed-20 improvement over the incumbent is a point-estimate improvement only; the 95% CIs overlap.",
            f"Informational only (NOT the preregistered rule): against registry rasters with <= 2x our dot budget (75,308 dots), the max forward overlap is {max_comparable}.",
            "S_netsib_only (network geometry alone) scored marginally higher than the preregistered candidate on both seeds; paired CI includes 0. Not swapped post hoc.",
            "Inputs are owner mirrors, SHA-pinned, not organizer-authenticated.",
        ],
    }
    (EV / "h83_run_card.json").write_text(json.dumps(card, indent=1) + "\n")
    print(json.dumps({k: card[k] for k in ("verdict", "ok_to_submit", "promotion_rule_results", "raster_sha256", "submission_name")}, indent=1))


if __name__ == "__main__":
    main()
