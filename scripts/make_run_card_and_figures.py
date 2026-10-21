#!/usr/bin/env python3
"""Assemble the protocol run card (docs/downloads/run_card.json + evidence/)
and simple overview figures from measured receipts only."""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"; PROC = ROOT / "data" / "processed"

e1 = json.loads((ROOT / "evidence" / "holdout_exp1.json").read_text())
e3 = json.loads((ROOT / "evidence" / "holdout_exp3_replication.json").read_text())
fb = json.loads((ROOT / "evidence" / "final_bundle.json").read_text())
v2 = json.loads((ROOT / "evidence" / "verification_pass2.json").read_text())
bd = json.loads((ROOT / "evidence" / "band_diagnostics.json").read_text())
s1 = e1["pooled_summary"]["scores"]["A4_primary"]
d_a2 = e1["pooled_summary"]["paired_differences"]["A2_standard"]
fr = e3["frozen_rule"]
su, du = fb["surface_uniqueness"], fb["dots_uniqueness"]

card = {
    "run_card": "GEMSDOE77-2026-10-21",
    "hypothesis": "Expert-mapped hidden faults include (a) buried basement contacts where magnetic tilt zero-crossings/analytic-signal ridges coincide with gravity-tilt edges and a signed basement-depth step, and (b) low-slip range fronts whose scarp is buried but whose front-base slope asymmetry remains linear. A supervised ranker using these correctly derived contact-physics channels (the shipped 'tilt' band is measurably radiometric total count) should recover withheld fault segments better than the standard band/terrain block.",
    "mechanism": "HGB (LOQO whole-component holdout for validation; full-catalogue fit for the artifact) over 19 shipped bands + terrain derivatives + 16 new channels (derived mag/gravity tilt, analytic signals, tilt-confluence, signed basement step along magnetic gradient, monogenic phase congruency on RTP/gravity, pseudogravity-Poisson slope/correlation/residual-gradient, front-base asymmetry, signed profile curvature). Dots: template expected-credit ranking, min-separation 2.8 px, budget/exclusion selected on Exp1 scan (44,090 dots, >3 px from catalogue). No catalogue-derived features, no SGMC training population, no lidar in the primary arm.",
    "named_non_fault_mimic": "Intrusive/lithologic contacts and caldera margins (potential-field edges without throw); pluvial Lake Lahontan shoreline benches (front-base mimic); GeoDAWN survey-block/flight-line seams (phase-congruent straight edges).",
    "holdout_dti": {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": e1["evaluator_version"],
        "primary_arm_A4": {"dti": s1["dti"], "ci95": s1["ci95"], "withheld_positive_pixels": s1["withheld_positive_pixels"], "draw": "seed30"},
        "control_A2_standard": e1["pooled_summary"]["scores"]["A2_standard"],
        "paired_delta_A4_minus_A2_seed30": {"delta": d_a2["delta"], "ci95": d_a2["ci95"], "gate": "FAIL (CI includes 0)"},
        "replication_seed21": {"delta": fr["seed21_delta"], "ci95": fr["seed21_ci95"], "withheld_positive_pixels": e3["withheld_positive_pixels"]},
        "two_draw_pooled": {"delta": fr["two_draw_delta"], "ci95": fr["two_draw_ci95"], "withheld_positive_pixels_total": s1["withheld_positive_pixels"] + e3["withheld_positive_pixels"]},
        "random_control_seed30": e1["pooled_summary"]["scores"]["A0_random"],
        "ablation_A5_with_lidar_seed30_point_only": e1["pooled_summary"]["scores"]["A5_primary_lidar"],
        "newphysics_only_A3_seed30": e1["pooled_summary"]["scores"]["A3_newphysics"],
        "sgmc_offcatalogue_proxy_not_holdout": e1["sgmc_offcatalogue_proxy"],
        "leakage_canary_worst_new_symmetric_auc": e1["canary_worst_new_symmetric_auc"],
    },
    "correlation_overlap_vs_registry": {
        "registry_rasters_checked_locally": fb["registry_rasters_checked"],
        "scope_caveat": fb["registry_scope_caveat"],
        "surface": {"worst_spearman": su["worst_spearman_full_footprint"], "worst_submission": su["worst_rho_submission"], "duplicate_count": su["duplicate_count"]},
        "final_dots": {"worst_spearman": du["worst_spearman_full_footprint"], "worst_mutual_3px_overlap": du["worst_dot_overlap"],
                       "worst_forward_3px_overlap": du["worst_forward_overlap"], "worst_forward_submission": du["worst_forward_submission"],
                       "duplicate_count": du["duplicate_count"]},
        "thresholds": {"rho_limit": 0.90, "overlap_limit": 0.70},
        "template_unique_bool_note": "Template 'unique' is False only because the local subset lacks a complete-inventory certificate; zero duplicates fired among the 43 checked rasters (two_phase readings in evidence/final_bundle.json). Full-inventory certification remains a selector step.",
    },
    "raster_sha256": fb["writer_receipt"]["sha256"],
    "raster_file": fb["submission_file"],
    "validator_output": {
        "writer_checks_all_passed": fb["writer_receipt"]["validator"]["all_checks_passed"],
        "independent_pass2_all_passed": v2["all_passed"],
        "no_nan_inside_footprint": True, "values_in_0_1": True,
        "n_nan_anywhere": v2["checks"]["n_nan"], "min": v2["checks"]["min"], "max": v2["checks"]["max"],
        "crs_shape_transform_match_sample": bool(v2["checks"]["crs_matches_sample"] and v2["checks"]["transform_matches_sample"]),
        "dots": v2["checks"]["dots_gt_0"], "dots_on_catalogue": v2["checks"]["dots_on_catalogue"],
        "min_dot_distance_to_catalogue_px": v2["checks"]["min_dot_distance_to_catalogue_px"],
        "metric_exact_vs_bruteforce_max_err": v2["checks"]["metric_exact_vs_bruteforce_max_err"],
    },
    "band_discoveries_measured": {
        "band6_is_radiometric_total_count_spearman": bd["band6_spearman_vs_external_TC_radiometric"],
        "band6_vs_derived_tilt_spearman": bd["band6_spearman_vs_derived_tilt_deg"],
        "band18_vs_gravity_dx_spearman": bd["band18_spearman_vs_grav_dx"],
        "band18_vs_gravity_gradmag_spearman": bd["band18_spearman_vs_grav_gradmag"],
    },
    "submission_name": fb["submission_name"],
    "submission_note_le_140": fb["note"],
    "submission_note_chars": fb["note_chars"],
    "download_ok_format_valid": True,
    "submit_ok_selector_promoted": False,
    "verdict": "negative",
    "verdict_reason": "Preregistered primary gate failed on seed-30 draw and under the amended two-draw rule: the new-physics block does not significantly improve pooled HOLDOUT-DTI over the standard block (two-draw paired delta +0.0026, 95% CI includes 0). Format-validity and uniqueness gates passed. Do not spend a weekly slot on this artifact; it is published as an auditable negative deliverable with full receipts.",
    "organizer_confirmed_scores_copied": {
        "public_leaderboard_1_2026-10-21": "xiaofanhu 0.4413 (DrivenData leaderboard page fetched 2026-10-21)",
        "this_artifact": "none (never submitted by this session)",
        "family_h33_0p2778": "OWNER-REPORT (user brief; no organizer receipt in inspected repos)",
    },
    "experiments_used": "3 of 3 (Exp1 holdout, Exp2 final build, Exp3 replication)",
    "slots_spent": 0,
}
(ROOT / "evidence" / "run_card.json").write_text(json.dumps(card, indent=2) + "\n")
(ROOT / "docs" / "downloads" / "run_card.json").write_text(json.dumps(card, indent=2) + "\n")
for src, dst in (("evidence/holdout_exp1.json", "docs/downloads/holdout_exp1.json"),
                 ("evidence/holdout_exp3_replication.json", "docs/downloads/holdout_exp3_replication.json"),
                 ("evidence/verification_pass2.json", "docs/downloads/verification_pass2.json"),
                 ("evidence/band_diagnostics.json", "docs/downloads/band_diagnostics.json")):
    (ROOT / dst).write_text((ROOT / src).read_text())
print("run card written; verdict:", card["verdict"])

# ---- figures (downsample x4) ----
with rasterio.open(RAW / "sample_submission.tif") as s:
    foot = np.isfinite(s.read(1))
with rasterio.open(RAW / "labels.tif") as s:
    cat = (s.read(1) > 0) & foot
surf = np.load(PROC / "final_surface_A4_primary.npy")
dots = np.load(PROC / "final_dots.npy")
f = 4
h, w = foot.shape[0] // f * f, foot.shape[1] // f * f

def ds_max(a):
    return a[:h, :w].reshape(h // f, f, w // f, f).max(axis=(1, 3))

def ds_mean(a):
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))

footd, catd, dotsd = ds_max(foot), ds_max(cat), ds_max(dots)
surfd = ds_mean(np.nan_to_num(surf))
surfd = surfd / (np.percentile(surfd[footd > 0], 99.5) + 1e-12)
Hh, Ww = footd.shape
img = np.zeros((Hh, Ww * 2 + 8, 3), np.uint8)
left = (np.clip(surfd, 0, 1) * 255).astype(np.uint8)
img[:, :Ww, 0] = left; img[:, :Ww, 1] = (left * 0.55).astype(np.uint8); img[:, :Ww, 2] = (60 + left * 0.3).astype(np.uint8)
img[:, :Ww][footd == 0] = (24, 24, 30)
right = np.zeros((Hh, Ww, 3), np.uint8); right[:] = (18, 20, 28)
right[footd > 0] = (52, 56, 66)
right[catd > 0] = (255, 84, 84)
right[dotsd > 0] = (255, 214, 64)
img[:, Ww + 8:] = right
Image.fromarray(img).save(ROOT / "docs" / "assets" / "fig_surface_and_dots.png") if (ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True) is None else None
print("figure written docs/assets/fig_surface_and_dots.png (left: final surface; right: catalogue red, emitted dots yellow, x4 downsample max/mean)")

# holdout bar figure (simple PIL bars)
arms = [("A5 +lidar\n(ablation)", e1["pooled_summary"]["scores"]["A5_primary_lidar"]),
        ("A4 primary", e1["pooled_summary"]["scores"]["A4_primary"]),
        ("A2 standard", e1["pooled_summary"]["scores"]["A2_standard"]),
        ("A3 new-phys\nonly", e1["pooled_summary"]["scores"]["A3_newphysics"]),
        ("frontness\nalone", e1["pooled_summary"]["scores"]["S_frontness"]),
        ("random", e1["pooled_summary"]["scores"]["A0_random"])]
W2, H2 = 1100, 560
img2 = Image.new("RGB", (W2, H2), (250, 250, 252))
from PIL import ImageDraw
dr = ImageDraw.Draw(img2)
dr.text((30, 18), "HOLDOUT-DTI (gems57-pooled-hide-v2, seed30, 11,342 withheld px) - whiskers: 95% CI", fill=(20, 20, 30))
base_y, top_y, x0 = 460, 70, 110
maxv = 0.14
for i in range(8):
    v = maxv * i / 7; y = base_y - (v / maxv) * (base_y - top_y)
    dr.line([(x0, y), (W2 - 30, y)], fill=(220, 222, 230)); dr.text((55, y - 6), f"{v:.2f}", fill=(90, 90, 110))
bw = (W2 - x0 - 60) // len(arms)
colors = [(150, 150, 165), (38, 110, 200), (120, 120, 135), (44, 160, 120), (230, 150, 40), (190, 190, 200)]
for i, ((nm, s), col) in enumerate(zip(arms, colors)):
    x = x0 + 20 + i * bw
    y = base_y - (s["dti"] / maxv) * (base_y - top_y)
    lo = base_y - (s["ci95"][0] / maxv) * (base_y - top_y); hi = base_y - (s["ci95"][1] / maxv) * (base_y - top_y)
    dr.rectangle([x, y, x + bw - 36, base_y], fill=col)
    dr.line([(x + (bw - 36) / 2, hi), (x + (bw - 36) / 2, lo)], fill=(30, 30, 40), width=3)
    dr.text((x, hi - 22), f"{s['dti']:.4f}", fill=(20, 20, 30))
    for j, ln in enumerate(nm.split("\n")):
        dr.text((x, base_y + 10 + j * 16), ln, fill=(40, 40, 55))
dr.text((30, H2 - 30), "Catalogue hide-and-recover ONLY - not leaderboard scores. A4-A2 paired delta CI includes 0 (see run card).", fill=(120, 60, 60))
img2.save(ROOT / "docs" / "assets" / "fig_holdout_arms.png")
print("figure written docs/assets/fig_holdout_arms.png")
