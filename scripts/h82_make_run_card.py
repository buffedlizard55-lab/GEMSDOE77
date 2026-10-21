#!/usr/bin/env python3
"""Assemble evidence/h82_run_card.json from Exp1-3 receipts (no new experiments)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
e1 = json.loads((EV / "h82_e1_canary_surface.json").read_text())
e2 = json.loads((EV / "h82_e2_holdout.json").read_text())
e3 = json.loads((EV / "h82_e3_build.json").read_text())
s = e2["pooled_summary"]["scores"]["A_std19_rrd"]
paired = e2["pooled_summary"]["paired_differences"]["A_std19"]
card = {
    "schema": "gems-parallel-run-card-v1",
    "lane_id": "H82-RRD",
    "lane_repo": "buffedlizard55-lab/GEMSDOE77",
    "branch": "arena/7ad85a23-gemsdoe77",
    "session_date_utc": "2026-10-21",
    "hypothesis": "RTP-TMI local regression discordance (remanence / vertical-axis block rotation) marks block-bounding faults missed by amplitude magnetic and DEM scarp detectors.",
    "mechanism": "8 label-free channels from bands 1/2/14 (local slope/correlation/residual breaks at sigma_w=2.5px on sigma=3px high-pass fields); winner arm A_std19_rrd = HGB(19 raw + 8 RRD), 37,654 binary dots, min-sep 2.8px, >2px off catalogue.",
    "named_non_fault_process_that_could_mimic_it": "Unfaulted margins of reversely-magnetized volcanic flow units; GeoDAWN survey-block levelling seams.",
    "holdout_dti": {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": e2["evaluator_version"],
        "seed": 20,
        "withheld_positive_pixels": s["withheld_positive_pixels"],
        "winner_arm_A_std19_rrd": s["dti"],
        "winner_ci95": s["ci95"],
        "arms": {k: {"dti": v["dti"], "ci95": v["ci95"]}
                 for k, v in e2["pooled_summary"]["scores"].items()},
        "paired_delta_vs_A_std19_control": paired,
        "incumbent": e2["incumbent"],
        "beats_incumbent": False,
    },
    "leakage_canary": {
        "features_tested": len(e1["canaries"]),
        "max_sym_auc": e1["canary_worst_sym_auc"],
        "ceiling": 0.90,
        "pass": e1["canary_pass_lt_090"],
    },
    "correlation_overlap_vs_registry": {
        "scope": "Phase-1 surface: sampled Spearman (n=200k, seed 82) vs 37 local-scope rasters. Phase-2 dots: exact 3px forward overlap vs 37 local-scope rasters incl. hash-verified H33 champion. NOT the complete family registry.",
        "phase1_surface_max_abs_rho_sampled": e1["surface_uniqueness"]["max_abs_rho"],
        "phase1_pass_le_090": bool(e1["surface_uniqueness"]["max_abs_rho"] <= 0.90),
        "phase2_max_overlap_literal": e3["uniqueness_final_dots"]["max_overlap_literal"],
        "phase2_max_overlap_informative_lt70_halo": e3["uniqueness_final_dots"]["max_overlap_informative"],
        "phase2_informative_firings_gt070": e3["uniqueness_final_dots"]["informative_firings_gt070"],
        "phase2_vs_h33_champion_overlap": e3["uniqueness_final_dots"]["vs_h33"]["forward_overlap_3px"],
        "phase2_pass_le_070": False,
        "byte_novel_sha_not_in_local_scope": True,
    },
    "raster_sha256": e3["sha256"],
    "raster_bytes": e3["bytes"],
    "zip_sha256": e3["zip_sha256"],
    "validator": e3["validator"],
    "submission_name": e3["submission_name"],
    "note": e3["note"],
    "note_chars": e3["note_chars"],
    "file": e3["file"],
    "ok_to_download_for_inspection": True,
    "ok_to_submit": False,
    "approved_for_weekly_slot": False,
    "weekly_slots_used": 0,
    "organizer_confirmed_score": None,
    "experiments_run": 3,
    "verdict": "negative",
    "reason": ("Winner 0.0686 HOLDOUT-DTI (gems57-pooled-hide-v2, seed 20, 11,321 withheld, "
               "95% CI in run card) is far below the 0.152718 incumbent; the paired RRD-block "
               "lift over the std19 control is +0.0068 with CI including 0 (not significant); "
               "pure S_rrd physics scored below the random control; final HGB dots fire the "
               "3px-overlap duplicate gate vs 3 informative local rasters (max 0.786). "
               "The file is format-valid and pixel-distinct from H33 (overlap 0.321), but it "
               "must not spend a weekly slot."),
    "irregularities": [
        "IR-82-01: family-best 0.2778 has no organizer receipt; GEMSDOE32 site labels h33-2-b2 UNSCORED (projected 0.2747 model).",
        "IR-82-02: shipped band tags unreliable (band6 = TC not tilt; band18 signed dx; bands 10/16 hedged).",
        "IR-82-03: first Exp3 uniqueness run self-matched the submissions/ mirror (spurious 1.0); fixed by hash exclusion and rerun; corrected values in this card.",
    ],
}
(EV / "h82_run_card.json").write_text(json.dumps(card, indent=2) + "\n")
print(json.dumps({"verdict": card["verdict"], "sha256": card["raster_sha256"],
                  "holdout": card["holdout_dti"]["winner_arm_A_std19_rrd"]}, indent=2))
