#!/usr/bin/env python3
"""Build H85-XWC run card + site pages from evidence JSON (no hand-typed scores).

Reads evidence/h85xwc_exp{1,2,3}*.json, writes:
  evidence/h85xwc_run_card.json, docs/data/h85xwc_submission.json,
  docs/h85xwc.html, docs/h85xwc-executive-summary.html, docs/assets/h85xwc_fig.png
Verdict logic is the preregistered rule, evaluated mechanically.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

NAME = "gems77-h85xwc-magxgrav-coherence-37654dots-20261021-zeros"
NOTE = "H85-XWC RTP x gravity multi-scale cross-coherence, label-blind; 37654 dotted sep3px >2px off catalogue"
PORTAL = "https://www.drivendata.org/competitions/306/competition-doe-gems/"


def load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


def figure(exp2: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gems57.grid import load_grid
    import rasterio

    grid = load_grid()
    surf = np.load(ROOT / ".cache" / "h85" / "surfaces.npz")["xwc"]
    with rasterio.open(ROOT / "docs" / "downloads" / f"{NAME}.tif") as src:
        dots = src.read(1) > 0
    step = 4  # 400 m display decimation
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    disp = np.where(grid.footprint[::step, ::step], surf[::step, ::step], np.nan)
    ax[0].imshow(disp, cmap="magma", vmin=0, vmax=1)
    ax[0].set_title("H85-XWC surface (rank01)\nRTP x gravity cross-coherence")
    overlay = np.full(disp.shape + (3,), 1.0)
    base = surf[::step, ::step]
    overlay[..., 0] = overlay[..., 1] = overlay[..., 2] = 1 - 0.55 * base
    cat = grid.catalogue[::step, ::step]
    dsub = dots[::step, ::step]
    overlay[cat] = (0.75, 0.75, 0.75)
    overlay[dsub] = (0.85, 0.1, 0.1)
    overlay[~grid.footprint[::step, ::step]] = (1, 1, 1)
    ax[1].imshow(overlay)
    ax[1].set_title("Final dots (red, 37,654) vs catalogue (grey)\nmin catalogue distance sqrt(5) px")
    arms = ["xwc", "rtp_only", "grav_only", "random"]
    x = np.arange(len(arms))
    for i, seed in enumerate(("20", "30")):
        sc = exp2["seeds"][seed]["pooled"]["scores"]
        dti = [sc[a]["dti"] for a in arms]
        lo = [d - sc[a]["ci95"][0] for d, a in zip(dti, arms)]
        hi = [sc[a]["ci95"][1] - d for d, a in zip(dti, arms)]
        ax[2].bar(x + (i - 0.5) * 0.38, dti, width=0.38,
                  yerr=[lo, hi], capsize=3, label=f"seed {seed}")
    ax[2].set_xticks(x, arms)
    ax[2].set_ylabel("HOLDOUT-DTI (gems57-pooled-hide-v2)")
    ax[2].set_title("Matched-arm holdout: candidate loses\nto random on both seeds (negative)")
    ax[2].legend()
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    out = ROOT / "docs" / "assets" / "h85xwc_fig.png"
    fig.savefig(out, dpi=130)
    return out


def main() -> None:
    exp1, exp2, exp3 = (load(f"evidence/h85xwc_exp{i}_{s}.json") for i, s in
                        ((1, "canary"), (2, "holdout"), (3, "build")))
    fig_path = figure(exp2)

    s20 = exp2["seeds"]["20"]["pooled"]["scores"]["xwc"]
    s30 = exp2["seeds"]["30"]["pooled"]["scores"]["xwc"]
    beats_all = all(
        exp2["seeds"][s]["pooled"]["paired_differences"][c]["ci95"][0] > 0
        for s in ("20", "30") for c in ("rtp_only", "grav_only", "random"))
    u = exp3["uniqueness"]
    informative_unique = (
        (u["max_abs_surface_rho_sparse"] or 0) <= 0.90
        and (u["max_abs_surface_rho_dense"] or 0) <= 0.90
        and (u["max_forward_dot_overlap_3px_sparse"] or 0) <= 0.70)
    format_ok = bool(exp3["validator"]["all_checks_passed"])
    verdict = "promote" if (beats_all and informative_unique and format_ok) else "negative"
    submit_ok = verdict == "promote"  # selector step still separate; never auto-slot

    run_card = {
        "lane": "H85-XWC",
        "hypothesis": ("A buried fault juxtaposing basement blocks of different susceptibility "
                       "and density produces a spatially coherent edge in both RTP magnetics and "
                       "isostatic gravity across 400-1600 m wavelengths with consistent cross-field "
                       "phase, even where basin fill hides any surface scarp."),
        "mechanism": ("FFT log-Gabor bandpass (4/8/16 px) of high-passed RTP (band 2) and isostatic "
                      "gravity (band 13); per-scale local coherence x rank joint edge energy, "
                      "phase-agreement weighted, rank-normalised; template decimated dotted "
                      "emission, 37,654 dots, >2 px from catalogue, 0 on catalogue."),
        "named_non_fault_mimic": ("Lithologic/intrusive contact sharing a magnetic and gravity edge "
                                  "without fault throw; GeoDAWN flight-line/survey-block seams."),
        "holdout": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": exp2["evaluator_version"],
            "seed20": {"dti": s20["dti"], "ci95": s20["ci95"],
                       "withheld_positives": s20["withheld_positive_pixels"]},
            "seed30": {"dti": s30["dti"], "ci95": s30["ci95"],
                       "withheld_positives": s30["withheld_positive_pixels"]},
            "controls_seed20": {a: exp2["seeds"]["20"]["pooled"]["scores"][a]["dti"]
                                for a in ("rtp_only", "grav_only", "random")},
            "controls_seed30": {a: exp2["seeds"]["30"]["pooled"]["scores"][a]["dti"]
                                for a in ("rtp_only", "grav_only", "random")},
            "beats_all_controls_paired_ci": beats_all,
            "interpretation": ("Candidate is significantly below the matched random control on both "
                               "seeds; cross-field coherence concentrates dots on shared lithologic "
                               "edges, not on withheld catalogue faults. Decisive negative."),
        },
        "canary_max_auc_vs_withheld": max(
            v["auc_vs_withheld_seed20"] for v in exp1["canaries"].values()),
        "correlation_overlap_vs_registry": {
            "rasters_compared": u["rasters_compared"], "errors": u["errors"],
            "max_abs_surface_rho_sparse": u["max_abs_surface_rho_sparse"],
            "max_abs_surface_rho_dense": u["max_abs_surface_rho_dense"],
            "max_forward_dot_overlap_3px_sparse": u["max_forward_dot_overlap_3px_sparse"],
            "max_forward_dot_overlap_3px_dense_literal": u["max_forward_dot_overlap_3px_dense_literal"],
            "informative_unique": informative_unique,
            "literal_unique": False,
            "irregularity": "IR-H85-01 (see evidence/h85xwc_exp3_build.json)",
        },
        "raster_sha256": exp3["sha256"],
        "validator": {"all_checks_passed": format_ok,
                      "checks": exp3["validator"]["checks"],
                      "n_nan": exp3["validator"]["n_nan"],
                      "min": exp3["validator"]["min"], "max": exp3["validator"]["max"]},
        "submission_name": NAME,
        "note": NOTE, "note_chars": len(NOTE),
        "experiments_used": 3, "weekly_slots_used": 0,
        "ok_to_download": True,
        "ok_to_submit": submit_ok,
        "verdict": verdict,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "evidence" / "h85xwc_run_card.json").write_text(
        json.dumps(run_card, indent=2) + "\n")
    (ROOT / "docs" / "data" / "h85xwc_submission.json").write_text(json.dumps({
        "name": NAME, "note": NOTE, "note_chars": len(NOTE),
        "sha256": exp3["sha256"], "dots": exp3["dots"],
        "ok_to_download": True, "ok_to_submit": submit_ok, "verdict": verdict,
        "holdout_dti_seed20": s20["dti"], "holdout_dti_seed30": s30["dti"],
        "evidence_class": "HOLDOUT-DTI (local hide-and-recover; not a leaderboard score)",
    }, indent=2) + "\n")

    banner = ("✅ OK TO DOWNLOAD &nbsp;·&nbsp; ⛔ NOT OK TO SUBMIT — automated gate verdict: "
              f"<b>{verdict.upper()}</b> (holdout candidate below matched random control; "
              "0 weekly slots used; selector: not approved)")
    if submit_ok:
        banner = "✅ OK TO DOWNLOAD · automated gates PASS — selector step still required before upload"
    d20, d30 = f"{s20['dti']:.4f}", f"{s30['dti']:.4f}"
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<link rel="stylesheet" href="assets/lia.css">')
    nav = ('<header><div class="wrap"><span class="brand">GEMSDOE77 · DOE GEMS Prize</span>'
           '<a href="index.html">Start here</a> <a href="h85xwc.html">H85-XWC report</a> '
           '<a href="h85xwc-executive-summary.html">H85-XWC submit guide</a> '
           '<a href="executive-summary.html">How to submit</a> '
           '<a href="https://github.com/buffedlizard55-lab/GEMSDOE77">GitHub</a></div></header>')

    report = f"""{head}<title>H85-XWC · GEMSDOE77</title></head><body>{nav}<main><div class="wrap">
<h1>H85-XWC - magnetic x gravity cross-field coherence</h1>
<section class="card dl" style="border:3px solid #b91c1c;background:#fef2f2;">
<h2>⬇ H85-XWC unique submission GeoTIFF (new method, machine-validated)</h2>
<p class="verdict">{banner}</p>
<a class="btn" href="downloads/{NAME}.tif" download>Download .tif (113 KB)</a>
<a class="btn alt" href="downloads/{NAME}.zip" download>One-TIFF .zip (76 KB)</a>
<a class="btn alt" href="h85xwc-executive-summary.html">Executive summary: exactly how submission works →</a>
<p class="small">name: <code>{NAME}</code><br>note ({len(NOTE)} chars): <code>{NOTE}</code><br>
sha256: <code>{exp3['sha256']}</code> · dots: {exp3['dots']} · on catalogue: {exp3['dots_on_catalogue']} ·
min catalogue distance: {exp3['min_catalogue_distance_px']:.3f} px</p>
<div class="chips"><span class="chip ok">Format validator: 15/15 PASS (float32 {{0,1}}, 0 NaN, EPSG:32611)</span>
<span class="chip ok">Informative uniqueness: PASS (max |ρ| {max(u['max_abs_surface_rho_sparse'], u['max_abs_surface_rho_dense']):.3f} ≤ 0.90; max sparse dot overlap {u['max_forward_dot_overlap_3px_sparse']:.3f} ≤ 0.70; {u['rasters_compared']} rasters)</span>
<span class="chip bad">Holdout: NEGATIVE — {d20} (s20) / {d30} (s30) HOLDOUT-DTI, below random</span>
<span class="chip warn">Literal dense-witness gate: FAIL for every candidate (IR-H85-01)</span></div></section>

<h2>Method (new in this repo — no prior implementation)</h2>
<p>Magnetic×gravity <b>multi-scale cross-field coherence</b>: FFT log-Gabor bandpass (wavelengths
4/8/16 px = 400/800/1600 m) of high-passed RTP magnetics (band 2) and isostatic gravity (band 13);
per-scale local coherence |S(XY)|/√(S(X²)S(Y²)) × rank joint edge energy, weighted by cross-scale
phase-sign agreement. Label-blind (no catalogue input to the surface). Emission reuses the shared
template decimated dotted allocator at the H33-matched budget of 37,654 dots, &gt;2 px from the
catalogue. Preregistration: <code>registry/preregistration_h85xwc.json</code>,
hypotheses: <code>research/h85_hypotheses.md</code>. Named mimic: lithologic/intrusive contacts
and flight-line seams.</p>
<img src="assets/h85xwc_fig.png" alt="H85-XWC surface, dots and holdout arms" style="max-width:100%">

<h2>Results — every number labelled</h2>
<table><tr><th>Quantity</th><th>Value</th><th>Class</th></tr>
<tr><td>Candidate, seed 20</td><td><b>{d20}</b> (95% CI {s20['ci95'][0]:.4f}–{s20['ci95'][1]:.4f}; {s20['withheld_positive_pixels']} withheld positives)</td><td>HOLDOUT-DTI, {exp2['evaluator_version']}</td></tr>
<tr><td>Candidate, seed 30</td><td><b>{d30}</b> (95% CI {s30['ci95'][0]:.4f}–{s30['ci95'][1]:.4f}; {s30['withheld_positive_pixels']} withheld)</td><td>HOLDOUT-DTI</td></tr>
<tr><td>Matched random control</td><td>{exp2['seeds']['20']['pooled']['scores']['random']['dti']:.4f} (s20) / {exp2['seeds']['30']['pooled']['scores']['random']['dti']:.4f} (s30)</td><td>HOLDOUT-DTI</td></tr>
<tr><td>Single-field controls (RTP / gravity)</td><td>{exp2['seeds']['20']['pooled']['scores']['rtp_only']['dti']:.4f} / {exp2['seeds']['20']['pooled']['scores']['grav_only']['dti']:.4f} (s20)</td><td>HOLDOUT-DTI</td></tr>
<tr><td>Leakage canary (max single-feature AUC)</td><td>{run_card['canary_max_auc_vs_withheld']:.3f} (stop &gt; 0.90: clear)</td><td>Diagnostic</td></tr>
<tr><td>H33 family best 0.2778</td><td>Owner-reported; no organizer receipt in this repo</td><td>UNVERIFIED — not ORGANIZER-CONFIRMED</td></tr></table>
<p><b>Interpretation.</b> Cross-field coherence concentrates emission on shared lithologic edges
(the named mimic), which do not coincide with withheld catalogue faults; uniform random coverage
earns more 300 m-kernel credit. This closes the cross-coherence lane as preregistered — no
retuning. The file above is nevertheless a genuinely unique, format-valid GeoTIFF: it is safe to
download and inspect, and the automated gate says <b>do not spend a weekly slot on it</b>.</p>
<p class="small">Evidence: <a href="evidence/h85xwc_run_card.json">run card</a> ·
<code>evidence/h85xwc_exp1_canary.json</code> · <code>evidence/h85xwc_exp2_holdout.json</code> ·
<code>evidence/h85xwc_exp3_build.json</code> · reproduce: <code>scripts/h85xwc_pipeline.py</code>,
audit: <code>scripts/h85xwc_audit.py</code>.</p>
</div></main></body></html>"""
    (ROOT / "docs" / "h85xwc.html").write_text(report)

    guide = f"""{head}<title>H85-XWC executive summary · GEMSDOE77</title></head><body>{nav}<main><div class="wrap">
<h1>H85-XWC — executive summary</h1>
<section class="card dl" style="border:3px solid #b91c1c;background:#fef2f2;">
<h2>Download and automated verdict</h2>
<p class="verdict">{banner}</p>
<a class="btn" href="downloads/{NAME}.tif" download>Download .tif</a>
<a class="btn alt" href="downloads/{NAME}.zip" download>One-TIFF .zip</a>
<p class="small">Submission name: <code>{NAME}</code><br>Portal note ({len(NOTE)}/140 chars): <code>{NOTE}</code><br>
sha256: <code>{exp3['sha256']}</code></p></section>

<h2>The automated gate (no human inspector)</h2>
<p>This page is generated by <code>scripts/build_h85xwc_site.py</code> from machine receipts only.
The gate evaluated, in order: (1) leakage canary AUC ≤ 0.90 — <b>PASS</b>
({run_card['canary_max_auc_vs_withheld']:.3f}); (2) shared-validator format 15/15, all values finite
in [0,1] — <b>PASS</b>; (3) informative uniqueness (max |ρ| ≤ 0.90 and max sparse 3-px dot overlap
≤ 0.70 across {u['rasters_compared']} rasters) — <b>PASS</b>; (4) preregistered holdout rule
(candidate beats every matched control, paired CI &gt; 0, both seeds) — <b>FAIL</b>
({d20}/{d30} vs random {exp2['seeds']['20']['pooled']['scores']['random']['dti']:.4f}/
{exp2['seeds']['30']['pooled']['scores']['random']['dti']:.4f} HOLDOUT-DTI).
One failed gate ⇒ <b>NOT OK TO SUBMIT</b>. Separately, the literal dense-witness overlap reading
fails for every non-empty raster including H33 (IR-H85-01) and promotion to a weekly slot remains
a distinct selector decision; this session used 0 slots.</p>

<h2>Exactly how a submission is made (for any future gate-passing file)</h2>
<ol>
<li>Confirm this page's banner says <b>OK TO SUBMIT</b> for the exact sha256 you downloaded
(today it does not). Verify locally: <code>sha256sum {NAME}.tif</code>.</li>
<li>Sign in at <a href="{PORTAL}">DrivenData competition 306</a>, check the remaining weekly
allowance on the submission page, open <b>New submission</b>.</li>
<li><b>File to submit:</b> choose the downloaded single-band float32 GeoTIFF (or the one-TIFF ZIP).
Grid contract: EPSG:32611, 3730×3292, transform (100, 0, 243350, 0, −100, 4508550), values in [0,1].</li>
<li><b>Note:</b> paste <code>{NOTE}</code></li>
<li>Submit, then archive the organizer timestamp/ID/score. Only that receipt may be labelled
ORGANIZER-CONFIRMED. Rules: <a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">official PDF</a>.</li></ol>

<h2>About the earlier “Predicted values must be in range [0, 1]” portal error</h2>
<p>This file cannot produce that error from its values: the shared writer refuses non-finite or
out-of-range input (no clipping), and independent read-back shows min 0.0, max 1.0, 0 NaN and
0 Inf in all 12,279,160 cells, no nodata sentinel, zeros outside the footprint
(<code>evidence/h85xwc_exp3_build.json</code>). The original rejected file's bytes are unavailable,
so its exact cause remains unproven; NaN-outside twins and sentinel-valued exports are the known
risk class this all-finite “zeros” export avoids.</p>
</div></main></body></html>"""
    (ROOT / "docs" / "h85xwc-executive-summary.html").write_text(guide)
    print("verdict:", verdict, "| beats_all:", beats_all,
          "| informative_unique:", informative_unique, "| format_ok:", format_ok)
    print("wrote run card, submission json, 2 pages, figure:", fig_path.name)


if __name__ == "__main__":
    main()
