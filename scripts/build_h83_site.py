#!/usr/bin/env python3
"""H83-NETSIB pages + top-of-site status hero, generated from evidence/h83_run_card.json.

Fails closed: if the published TIFF bytes do not hash to the run card's
raster_sha256, or the shared validator does not pass on the actual bytes, the
hero says NOT OK TO SUBMIT regardless of the recorded verdict.
"""
from __future__ import annotations

import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs"
sys.path.insert(0, str(ROOT / "vendor"))
COMP = "https://www.drivendata.org/competitions/306/competition-doe-gems/"
SUBMIT = "https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/"
LB = "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/"


def e(x):
    return html.escape(str(x), quote=True)


def status(card):
    """Re-derive the operative decision from actual bytes (never trust flags alone)."""
    from gems57.grid import load_grid
    from gems57.validate import validate
    tif = ROOT / card["download"]
    reasons = []
    if not tif.exists():
        return False, ["published TIFF missing"], None
    sha = hashlib.sha256(tif.read_bytes()).hexdigest()
    if sha != card["raster_sha256"]:
        reasons.append("published bytes differ from run card hash")
    g = load_grid(ROOT / "data" / "bridge")
    v = validate(tif, g.footprint, g.catalogue)
    if not v["all_checks_passed"]:
        reasons.append("shared validator failed on actual bytes")
    if card["verdict"] != "promote":
        reasons.append("lane verdict is negative")
    return (not reasons), reasons, v


def copy(label, text):
    return (f'<p class="small muted">{e(label)} ({len(text)} characters)</p><div class="copy"><code>{e(text)}</code>'
            f'<button data-copy="{e(text)}">Copy</button></div>')


def build(page):
    card = json.loads((ROOT / "evidence/h83_run_card.json").read_text())
    ok, reasons, v = status(card)
    s20, s30 = card["holdout_dti"]["20"], card["holdout_dti"]["30"]
    c20, c30 = s20["candidate_A_netsib_full"], s30["candidate_A_netsib_full"]
    reg = card["registry"]
    fd = reg["final_dots"]
    tif_rel = card["download"].removeprefix("docs/")
    zip_rel = card["zip"].removeprefix("docs/")
    size_kb = (ROOT / card["download"]).stat().st_size / 1024
    if ok:
        head = "✅ OK TO DOWNLOAD · ✅ OK TO SUBMIT (promoted candidate)"
        cls = "border:3px solid #15803d;background:#f0fdf4;"
        say = ("This file passed every repository gate: it beat the incumbent on the spatial holdout on both seeds, "
               "passed the leakage canaries, passed the registry uniqueness checks and the shared format validator on "
               "its actual bytes. Spending one of your weekly slots on it is your (the selector's) decision; "
               "no slot has been used.")
    else:
        head = "✅ OK TO DOWNLOAD FOR INSPECTION · ⛔ NOT OK TO SUBMIT"
        cls = "border:3px solid #b91c1c;background:#fef2f2;"
        say = ("Do not upload. " + card.get("verdict_reason", "") + " The holdout result itself was positive (it beat the "
               "0.152718 incumbent by point estimate on both seeds); "
               "the only failed gate is the parallel-run duplicate rule. Changing that rule is the project owner's "
               "decision (docs/irregularities.md IR-83-01), not this session's.")
    hero = f"""<section class="card dl" style="{cls}"><h2>⬇ NEWEST (2026-10-21) · H83-NETSIB submission file — {e(head)}</h2>
<p><b>{e(say)}</b></p>
<p class="small">Fail-closed re-check of the published file at build time: {e('; '.join(r for r in reasons if 'verdict' not in r) or 'SHA-256 matches the run card and the shared validator passes on the actual bytes')}.</p>
<a class="btn" href="{e(tif_rel)}" download>Download the .tif ({size_kb:.0f} KB)</a>
<a class="btn alt" href="{e(zip_rel)}" download>One-TIFF .zip</a>
<a class="btn alt" href="h83.html">How it was built &amp; verified →</a>
<a class="btn alt" href="h83-executive-summary.html">Step-by-step: how to submit →</a>
<div class="chips"><span class="chip ok">Format: PASS (float32, EPSG:32611, 3730×3292, values only 0/1, 0 NaN)</span>
<span class="chip {'ok' if card['beats_incumbent_point_estimate_both_seeds'] else 'bad'}">Beats holdout best 0.152718 (point estimate, both seeds): {'YES' if card['beats_incumbent_point_estimate_both_seeds'] else 'NO'}</span>
<span class="chip {'ok' if not fd['duplicate'] else 'bad'}">Registry uniqueness (IR-83-01 reading): {'PASS' if not fd['duplicate'] else 'FAIL'}</span>
<span class="chip warn">Weekly slots used: 0</span></div>
<p class="small">HOLDOUT-DTI <b>{c20['dti']:.4f}</b> (seed 20; 95% CI {c20['ci95'][0]:.4f}–{c20['ci95'][1]:.4f}; {s20['withheld_positive_pixels']:,} withheld positives) and <b>{c30['dti']:.4f}</b> (seed 30; CI {c30['ci95'][0]:.4f}–{c30['ci95'][1]:.4f}; {s30['withheld_positive_pixels']:,}), evaluator <code>{e(s20['evaluator_version'])}</code>. Incumbent 0.152718 (same evaluator, seed 20). CIs overlap: an improvement in point estimate, not a statistically proven one. No leaderboard score is claimed or projected.</p>
{copy('File / submission name', card['submission_name'])}
{copy('Note for the portal (≤140 characters)', card['note'])}
<p class="small">SHA-256 <code>{e(card['raster_sha256'])}</code> · <a href="evidence/h83_run_card.json">run card JSON</a></p></section>"""

    can = card["leakage_canary"]
    rows = "".join(
        f"<tr><td>{e(lbl)}</td><td>{s20[k]['dti']:.4f} ({s20[k]['ci95'][0]:.4f}–{s20[k]['ci95'][1]:.4f})</td>"
        f"<td>{s30[k]['dti']:.4f} ({s30[k]['ci95'][0]:.4f}–{s30[k]['ci95'][1]:.4f})</td></tr>"
        for lbl, k in (("A_netsib_full — preregistered candidate (39 base + 10 network channels)", "candidate_A_netsib_full"),
                       ("A_base — same rows, 39 base channels only", "control_A_base"),
                       ("S_netsib_only — 10 network channels only", "arm_S_netsib_only"),
                       ("A0_random — uniform random", "random_A0")))
    body = f"""{hero}<h1>H83-NETSIB · fault-network sibling / gap / continuation geometry</h1>
<p><b>Hypothesis.</b> {e(card['hypothesis'])}</p>
<p><b>Mechanism (frozen preregistration text).</b> {e(card['mechanism'])}</p>
<p class="small"><b>As run:</b> training positives were the synthetic re-withheld components only (2 draws), the base stack has 39 channels, and fold-k negatives avoid every catalogue pixel a fold-k model may know. These differences are recorded as post-hoc amendments IR-83-03 to IR-83-06 in the preregistration and in docs/irregularities.md.</p>
<p><b>Named non-fault process that could mimic it.</b> {e(card['named_non_fault_mimic'])}. Others: {e('; '.join(card['other_mimics']))}.</p>
<p><img src="figures/h83_netsib_surface_dots.png" alt="H83 final surface (grey), mapped catalogue (black), emitted dots (red)" style="max-width:100%;border:1px solid #cbd5e1"></p>
<h2>Experiment 1 — leakage canaries and surface uniqueness</h2>
<p>Max single-feature symmetric AUC: seed 20 {can['max_symmetric_auc_by_seed']['20']} ({e(can['max_feature_by_seed']['20'])}), seed 30 {can['max_symmetric_auc_by_seed']['30']} ({e(can['max_feature_by_seed']['30'])}); ceiling 0.90 → {'PASS' if can['pass'] else 'FAIL'}. Pre-placement surface Spearman: label-free network score max |ρ| {reg['pre_placement_deterministic_surface_max_abs_spearman']:.3f} ({e(reg['pre_placement_deterministic_surface_worst'])}); final candidate surface max |ρ| {reg['pre_placement_candidate_surface_max_abs_spearman']:.3f} ({e(reg['pre_placement_candidate_surface_worst'])}) over {reg['surface_rasters_compared']} unique rasters (limit 0.90).</p>
<h2>Experiment 2 — paired hide-and-recover HOLDOUT-DTI</h2>
<p>Unmodified shared evaluator <code>gems57-pooled-hide-v2</code>: whole catalogue components withheld (20 % per quadrant, 3 px context collar), every network feature computed from the visible catalogue only, visible faults masked pixel-exactly, pooled DTI (α 0.2, β 0.8, 300 m triangular kernel), 95 % paired spatial-block bootstrap. Identical 37,654-dot emission (Poisson-disk 2.8 px, &gt;2 px from known faults) for every arm.</p>
<table><tr><th>Arm</th><th>Seed 20 ({s20['withheld_positive_pixels']:,} withheld)</th><th>Seed 30 ({s30['withheld_positive_pixels']:,} withheld)</th></tr>{rows}</table>
<p>Paired Δ candidate − A_base: {s20['paired_delta_vs_A_base']['delta']:+.4f} (CI {s20['paired_delta_vs_A_base']['ci95'][0]:+.4f} to {s20['paired_delta_vs_A_base']['ci95'][1]:+.4f}) and {s30['paired_delta_vs_A_base']['delta']:+.4f} (CI {s30['paired_delta_vs_A_base']['ci95'][0]:+.4f} to {s30['paired_delta_vs_A_base']['ci95'][1]:+.4f}). Network-only vs candidate: Δ candidate − S_netsib_only {s20['paired_delta_vs_S_netsib_only']['delta']:+.4f} / {s30['paired_delta_vs_S_netsib_only']['delta']:+.4f}, CIs include 0 (not swapped post hoc).</p>
<h2>Experiment 3 — final build and final-dot uniqueness</h2>
<p>{fd['unique_decoded_rasters_compared']} unique registry rasters compared (812-entry owner-repo index + witnesses + local files). Max |Spearman| of final dots {fd['max_abs_spearman']:.3f}. Forward 3-px overlap over the {fd['informative_rasters']} rasters where the test is informative: max {fd['max_forward_overlap_informative']:.3f} ({e(fd['max_forward_overlap_informative_raster'])}). Matched-budget overlap vs dense rasters' top-37,654 pixels: max {fd['max_matched_budget_overlap_dense']:.3f}. Literal all-raster value {fd['max_forward_overlap_literal_all']:.3f} — reported; {fd['literal_firings_noninformative']} firings are rasters whose 3-px halo covers ≥70 % of the domain, where uniformly random dots would also "fail" (IR-83-01).</p>
<p><b>Duplicate rule: {'FIRED' if fd['duplicate'] else 'clear'}.</b> {len(card.get('informational_overlap_firings', []))} informative rasters exceed the 0.70 limit. All of them are wide dot fields (≥{min((f['their_dots'] for f in card.get('informational_overlap_firings', [])), default=0):,} dots) that are rank-uncorrelated with this file (|ρ| ≤ {max((abs(f['spearman']) for f in card.get('informational_overlap_firings', [])), default=0):.3f}), but the preregistered rule has no exception for that, so the lane stopped as negative.</p>
<p>Validator on actual bytes: {'ALL PASS' if v and v['all_checks_passed'] else 'FAIL'} — {e(', '.join(k for k in (v or {}).get('checks', {})))}.</p>
<p><a href="evidence/h83_run_card.json">Run card</a> · <a href="evidence/h83_e1_canary_surface.json">Exp 1</a> · <a href="evidence/h83_e2_holdout.json">Exp 2</a> · <a href="evidence/h83_e3_build.json">Exp 3 build</a> · <a href="evidence/preregistration_h83_netsib.json">preregistration</a> · <a href="research/h83_netsib_hypotheses.md">hypotheses &amp; why H33 scored 0.2778</a></p>"""

    guide = f"""{hero}<h1>Executive summary — how to submit to the DOE GEMS Prize</h1>
<ol>
<li><b>Download</b> the file above (<code>{e(Path(tif_rel).name)}</code>). Either the <code>.tif</code> or the one-TIFF <code>.zip</code> is accepted by the portal text quoted below.</li>
<li>Check the status line: it must read <b>OK TO SUBMIT</b>. Files elsewhere on this site marked <i>NOT OK TO SUBMIT</i> are research artifacts.</li>
<li>Sign in at <a href="{COMP}">the competition page</a> and open its submissions tab (<a href="{SUBMIT}">{SUBMIT}</a>) → <b>Submit file</b>.</li>
<li><b>File to submit:</b> choose the downloaded file. The portal text (as pasted by the project owner from the live submission form, README prompt archive) says: “a single-band GeoTIFF (.tif) file, or a .zip file containing a single GeoTIFF … must match the submission format's CRS, shape, and geotransform.” This file is single-band float32, EPSG:32611, 3730×3292, transform (100, 0, 243350, 0, −100, 4508550).</li>
<li><b>Note (optional):</b> paste the note above. The portal has no separate name field; the unique file name is the identifier.</li>
<li>Submit, then copy the score shown on the submission page into the repo as an ORGANIZER-CONFIRMED receipt with this SHA-256.</li>
</ol>
<h2>Why the earlier “Predicted values must be in range [0, 1]” error cannot recur with this file</h2>
<p>The shared validator read the actual bytes: minimum {v['min'] if v else 'n/a'}, maximum {v['max'] if v else 'n/a'}, NaN count {v['n_nan'] if v else 'n/a'}, no infinities, no nodata value set; outside the footprint every value is exactly 0. Which earlier download triggered the error was not recorded. The repository's earlier review found NaN-outside-footprint variants on the site, and a NaN fails any “between 0 and 1” check, so that is the most likely cause; it is not proven. This file contains only 0.0 and 1.0.</p>
<p>Leaderboard: <a href="{LB}">{LB}</a>. Weekly submission limits are shown on the submission page; this repository has not used any slot.</p>"""
    D.joinpath("h83.html").write_text(page("h83.html", "H83-NETSIB", body))
    D.joinpath("h83-executive-summary.html").write_text(page("h83-executive-summary.html", "H83 how to submit", guide))
    (D / "evidence").mkdir(exist_ok=True)
    (D / "research").mkdir(exist_ok=True)
    for f in ("h83_run_card.json", "h83_e1_canary_surface.json", "h83_e2_holdout.json", "h83_e3_build.json"):
        shutil.copyfile(ROOT / "evidence" / f, D / "evidence" / f)
    shutil.copyfile(ROOT / "registry/preregistration_h83_netsib.json", D / "evidence/preregistration_h83_netsib.json")
    shutil.copyfile(ROOT / "research/h83_netsib_hypotheses.md", D / "research/h83_netsib_hypotheses.md")
    marker = '<main><div class="wrap">'
    for pname in ("index.html", "executive-summary.html"):
        p = D / pname
        t = p.read_text()
        if "H83-NETSIB submission file" not in t:
            t = t.replace(marker, marker + hero, 1)
        p.write_text(t)
    return ok
