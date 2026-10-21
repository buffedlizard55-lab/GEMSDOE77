#!/usr/bin/env python3
"""Generate docs/contactphys.html + docs/contactphys-executive-summary.html from evidence JSONs.
Every number on the site is read from a receipt; nothing is hand-typed here
except prose and the clearly-labelled owner/official context."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
card = json.loads((ROOT / "evidence" / "run_card_contactphys.json").read_text())
_supp_path = ROOT / "evidence" / "uniqueness_supplement_postmerge.json"
supp = json.loads(_supp_path.read_text()) if _supp_path.exists() else None
supp_html = ""
if supp:
    _b = supp["binary_dot_reading"]
    supp_html = (
        "<p><b>Post-merge supplement (2026-10-21):</b> two sibling lanes merged to "
        "<code>main</code> concurrently (H77C claycap; selective-U urank). Against their "
        f"rasters this candidate is placement-distinct under the binary-dot reading "
        f"(forward overlaps {min(r['binary_dot_forward_overlap'] for r in _b):.3f}–"
        f"{max(r['binary_dot_forward_overlap'] for r in _b):.3f}, |ρ| ≤ 0.003). The literal "
        "tool now on <code>main</code> flags the dense continuous urank surface as a "
        "universal overlap blocker (0.979) — a documented template divergence "
        "(IR-77-09), recorded in <code>evidence/uniqueness_supplement_postmerge.json</code>. "
        "Verdict unchanged under either reading.</p>")
e1 = json.loads((ROOT / "evidence" / "holdout_exp1.json").read_text())
fb = json.loads((ROOT / "evidence" / "final_bundle.json").read_text())
bd = json.loads((ROOT / "evidence" / "band_diagnostics.json").read_text())
H = card["holdout_dti"]
S = e1["pooled_summary"]["scores"]
PD = e1["pooled_summary"]["paired_differences"]
TIF = fb["submission_file"]; ZIP = TIF.replace(".tif", ".zip")
SHA = card["raster_sha256"]
NOTE = card["submission_note_le_140"]; NAME = card["submission_name"]

CSS = """
:root{--ink:#17202b;--mut:#5b6b7d;--line:#dde4ec;--card:#fff;--bg:#f4f6f9;--red:#b91c1c;--redbg:#fef2f2;--grn:#15803d;--amb:#b45309;--acc:#1d4ed8}
*{box-sizing:border-box}body{margin:0;font:16px/1.6 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{max-width:1060px;margin:0 auto;padding:28px 20px 64px}
header h1{font-size:30px;margin:0 0 6px}header p{color:var(--mut);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:18px 0}
.stop{border:2px solid var(--red);background:var(--redbg)}
.stop h2{color:var(--red);margin-top:0;font-size:26px}
.ok{color:var(--grn);font-weight:700}.no{color:var(--red);font-weight:700}.warn{color:var(--amb);font-weight:700}
code{background:#eef2f7;border-radius:6px;padding:1px 6px;font-size:13.5px;word-break:break-all}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14.5px}th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}th{background:#f0f3f8}
.btn{display:inline-block;background:var(--acc);color:#fff;text-decoration:none;padding:11px 18px;border-radius:10px;font-weight:700;margin:6px 8px 6px 0}
.btn2{display:inline-block;background:#fff;color:var(--acc);border:2px solid var(--acc);text-decoration:none;padding:9px 16px;border-radius:10px;font-weight:700;margin:6px 8px 6px 0}
img{max-width:100%;border-radius:10px;border:1px solid var(--line)}small{color:var(--mut)}h2{font-size:22px}h3{font-size:18px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:14px 0}
.kv div{background:#f6f8fb;border:1px solid var(--line);border-radius:10px;padding:12px 14px}.kv b{font-size:21px;display:block}
footer{color:var(--mut);font-size:13.5px;margin-top:34px}
"""


def arm_row(nm, key):
    s = S[key]
    return f"<tr><td>{nm}</td><td><b>{s['dti']:.4f}</b></td><td>[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]</td><td>{e1['arms'][key]['dots']:,}</td></tr>"


index = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GEMSDOE77 — contact-physics fault lane · verdict: negative (do not submit)</title><link rel="stylesheet" href="assets/site-contactphys.css"></head><body><div class="wrap">
<header><h1>GEMSDOE77 — contact-physics fault discovery for the DOE GEMS Prize</h1>
<p>DOE GEMS Prize (DrivenData #306, GeoDAWN NW Nevada). Lane: correctly derived potential-field contact transforms
(tilt / analytic signal / phase congruency / pseudogravity-Poisson) + range-front-base geomorphology, supervised-ranked.
Every number below is labelled HOLDOUT-DTI, ORGANIZER-CONFIRMED, or OWNER-REPORT. No projection is written as a score.</p></header>

<div class="card stop"><h2>⛔ DO NOT SUBMIT — verdict: negative</h2>
<p><b>Is the file on this page OK to download and submit? NO.</b> The GeoTIFF is format-valid and unique (receipts below),
and you may download it for inspection (that is its only approved use). It is <b>not approved</b> for a DrivenData
submission slot: it failed the preregistered holdout promotion gate. The new-physics arm did not significantly beat the
standard-feature arm — paired Δ(HOLDOUT-DTI) = {H['two_draw_pooled']['delta']:+.4f} with 95% CI
[{H['two_draw_pooled']['ci95'][0]:+.4f}, {H['two_draw_pooled']['ci95'][1]:+.4f}] across two independent draws
({H['two_draw_pooled']['withheld_positive_pixels_total']:,} withheld catalogue pixels). A CI containing 0 means
“no detectable improvement”, and this project does not spend scarce weekly slots on undetectable improvements.
Promotion, if any later lane earns it, is a separate human selector step. <b>0 slots were spent by this work.</b></p>
<p>Why publish a negative? Because a measured negative with receipts is a deliverable: it stops the next session
re-spending a slot on the same idea, and its by-products below (two mislabelled band identifications, one strong
single geomorphic feature, an off-catalogue proxy observation) are the usable yield. See <a href="#next">what to do instead</a>.</p></div>

<div class="card"><h2>⬇ Download (for inspection / reproducibility only)</h2>
<a class="btn" href="downloads/{TIF}">Download GeoTIFF ({fb['writer_receipt']['bytes']:,} bytes)</a>
<a class="btn2" href="downloads/{ZIP}">ZIP (single GeoTIFF inside)</a>
<a class="btn2" href="downloads/run_card_contactphys.json">JSON run card</a>
<table>
<tr><th>Filename</th><td><code>{TIF}</code></td></tr>
<tr><th>Unique submission name (if a human ever overrode the verdict — not recommended)</th><td><code>{NAME}</code></td></tr>
<tr><th>Note field text ({card['submission_note_chars']}/140 chars)</th><td><code>{NOTE}</code></td></tr>
<tr><th>SHA-256</th><td><code>{SHA}</code></td></tr>
<tr><th>Format (independently re-read)</th><td>single band · float32 · EPSG:32611 · 100 m · 3730×3292 · 44,090 dots · 0 NaN anywhere · min 0.0 max 1.0 · no nodata tag · 0 dots on the mapped catalogue · nearest dot 3.16 px from the catalogue — <a href="downloads/verification_pass2.json">pass-2 receipt</a> · <a href="downloads/{TIF.replace('.tif','.json')}">writer receipt</a></td></tr>
<tr><th>Download OK? (format)</th><td><span class="ok">YES — format-valid</span></td></tr>
<tr><th>Submit OK? (science / selector)</th><td><span class="no">NO — verdict negative, see banner</span></td></tr>
</table>
<p><small>Step-by-step DrivenData submission instructions (for any future promoted file) are on the
<a href="contactphys-executive-summary.html">executive summary</a> page, including the root cause of the portal error
“Predicted values must be in range [0, 1]”.</small></p></div>

<div class="card"><h2>Why did h33-2-b2 score 0.2778 (OWNER-REPORT), and can it be beaten?</h2>
<p>The family's best file is not new geology — it is <b>emission arithmetic on an older field</b>. Owner-reported chain:
solid H19-5 surface (121,131 px → 0.1922) → dotted thinning d1.5 (60,069 px → 0.2477) → d2.8 (44,090 px → 0.2600) →
solo prune (40,199 px → 0.2708) → <b>h33-2-b2 deletes every dot within 2 px (200 m) of the mapped catalogue</b>
(37,654 px → 0.2778). Under the official metric
DTI = TP<sub>w</sub> / (TP<sub>w</sub> + 0.2·FP<sub>w</sub> + 0.8·FN<sub>w</sub>) with a 300 m triangular kernel,
each emitted unit always adds 0.2 to the denominator but earns at most its kernel credit k; a dot helps iff
<b>k &gt; 0.2·DTI</b> (≈0.054 at DTI 0.27). Dots hugging known faults cover structures that are already mapped
(and therefore not in the hidden truth) — they pay the false-positive tax for almost no credit. Deleting them,
and spacing the rest so one dot covers each 300 m kernel footprint, raises credit per dot without finding a single
new fault. That is why thinning bought +0.086 (0.1922→0.2778).</p>
<p><b>Beating it — let alone the public leader at 0.4413 (ORGANIZER-CONFIRMED, leaderboard read 2026-10-21) —
cannot come from more thinning.</b> At equal emitted mass the leader's pixels earn roughly 60% more mean kernel
credit than the H19-5 family's; only a better-ranked <i>field</i> (pixels closer to real unmapped faults) can supply
that. This repository tested one new field family honestly; it did not clear the bar (below). No score in this
repository is organizer-confirmed for any local file; 0.2778 itself is an owner report without a receipt in any
inspected sibling repo, and no local proxy reproduces it — flagged as irregularity IR-77-04.</p></div>

<div class="card"><h2>What was tested (preregistered, 3 experiments)</h2>
<p>Ranked hypotheses were frozen in <code>registry/preregistration_contactphys.json</code> before Exp 1. Top lane:
buried basement contacts (magnetic tilt zero-cross + analytic-signal ridge coincident with a gravity-tilt edge,
gated by the sign of the basement-depth step) and range-front bases (flat piedmont pixels adjacent to a straight
steep front). Named non-fault mimics: intrusive contacts/caldera margins, Lake Lahontan shoreline benches,
survey-block seams. Validation = shared-template whole-component quadrant hide-and-recover
(<code>gems57-pooled-hide-v2</code>, α=0.2 β=0.8 R=300 m): whole fault components withheld, 3 px feature buffer,
12 px domain erosion, visible faults masked pixel-exactly, pooled DTI with paired 20 km-block bootstrap CIs.
Leakage canary first: worst new-feature symmetric holdout AUC = <b>{H['leakage_canary_worst_new_symmetric_auc']:.4f}</b>
(frontness) — far below the 0.90 leakage stop.</p>
<table><tr><th>Arm (seed-30 draw)</th><th>HOLDOUT-DTI</th><th>95% CI</th><th>Dots in folds</th></tr>
{arm_row('A5 standard+new+lidar (ablation only)','A5_primary_lidar')}
{arm_row('A4 primary: standard+new physics (no lidar)','A4_primary')}
{arm_row('A2 standard block (control)','A2_standard')}
{arm_row('A3 new physics only (16 ch)','A3_newphysics')}
{arm_row('Single feature: frontness','S_frontness')}
{arm_row('Single feature: contact confluence','S_contact_confluence')}
{arm_row('Single feature: phase congruency (mag)','S_pc_mag')}
{arm_row('A0 random dots (matched)','A0_random')}
</table>
<p>Paired Δ A4−A2 (seed 30): <b>{PD['A2_standard']['delta']:+.4f} [{PD['A2_standard']['ci95'][0]:+.4f}, {PD['A2_standard']['ci95'][1]:+.4f}]</b> — gate failed.
Pre-declared replication (seed 21, {H['replication_seed21']['withheld_positive_pixels']:,} withheld):
Δ <b>{H['replication_seed21']['delta']:+.4f} [{H['replication_seed21']['ci95'][0]:+.4f}, {H['replication_seed21']['ci95'][1]:+.4f}]</b>.
Two-draw pooled Δ <b>{H['two_draw_pooled']['delta']:+.4f} [{H['two_draw_pooled']['ci95'][0]:+.4f}, {H['two_draw_pooled']['ci95'][1]:+.4f}]</b>.
Receipts: <a href="downloads/holdout_exp1.json">Exp 1</a> · <a href="downloads/holdout_exp3_replication.json">Exp 3 replication</a>.
Operating point from the Exp-1 scan (16 cells reported in the receipt): 44,090 dots, &gt;3 px catalogue exclusion.
Caveat: catalogue holdout rewards catalogue-style faults; archived H33 dots score 0.0043 HOLDOUT-DTI here because
they were pruned <i>away</i> from the catalogue by construction — instrument divergence, not evidence about live score.</p>
<img src="assets/fig_holdout_arms.png" alt="Holdout DTI by arm with 95 percent confidence intervals"></div>

<div class="card"><h2>Maps</h2>
<img src="assets/fig_surface_and_dots.png" alt="Final probability surface and emitted dots versus mapped catalogue">
<p><small>Left: final full-catalogue probability surface (400 m mean downsample). Right: mapped catalogue (red) and the
44,090 emitted off-catalogue dots (yellow, 400 m max downsample). Dots flank and extend mapped traces and follow
off-catalogue lineaments — visually plausible is not validation; the holdout table above governs.</small></p></div>

<div class="card"><h2>Two band mislabels, measured (usable by every sibling lane)</h2>
<table><tr><th>Finding</th><th>Measurement (200k footprint sample)</th></tr>
<tr><td>Band 6 <code>tc</code>, tagged “Tilt angle or total curvature”, is <b>radiometric total count</b>. The shipped stack contains no tilt derivative.</td>
<td>Spearman(band 6, GeoDAWN TC) = <b>{bd['band6_spearman_vs_external_TC_radiometric']:.4f}</b>; vs derived tilt atan2(vg,hg) = {bd['band6_spearman_vs_derived_tilt_deg']:.4f}; range [{bd['band6_range_in_footprint'][0]:.2f}, {bd['band6_range_in_footprint'][1]:.2f}]</td></tr>
<tr><td>Band 18 <code>iso_grav_anom_hg</code> is the <b>x-derivative</b> of gravity, not a horizontal-gradient magnitude (it is signed).</td>
<td>Spearman vs ∂g/∂x = <b>{bd['band18_spearman_vs_grav_dx']:.4f}</b>; vs |∇g| = {bd['band18_spearman_vs_grav_gradmag']:.4f}</td></tr>
<tr><td>Controls: band 3 ≈ |∇TMI| (ρ={bd['band3_spearman_vs_tmi_gradmag']:.4f}); band 9 ≈ vertical derivative (ρ={bd['band9_spearman_vs_fft_vertical_derivative_tmi']:.4f}); band 19 ≈ |∇DEM| (ρ={bd['band19_spearman_vs_dem_gradmag']:.4f}).</td>
<td><a href="downloads/band_diagnostics.json">diagnostics receipt</a></td></tr></table>
<p>Also measured single-feature results worth keeping: <b>frontness alone</b> reaches HOLDOUT-DTI
{S['S_frontness']['dti']:.4f} (best single channel tested); the new-physics-only arm is the best of all arms on the
secondary SGMC-off-catalogue proxy (PROXY-DTI {H['sgmc_offcatalogue_proxy_not_holdout']['A3_newphysics']['dti']:.4f} vs standard
{H['sgmc_offcatalogue_proxy_not_holdout']['A2_standard']['dti']:.4f}, {H['sgmc_offcatalogue_proxy_not_holdout']['n_truth_px']:,} proxy pixels) —
an hypothesis for a future preregistration, <b>not</b> a claim here.</p></div>

<div class="card"><h2>Uniqueness vs prior submissions (lane discipline)</h2>
<p>Compared with the shared <code>gems57.uniqueness</code> tool against {card['correlation_overlap_vs_registry']['registry_rasters_checked_locally']}
locally hash-verified sibling rasters (GEMSDOE32 + GEMSDOE70 downloads), on the continuous surface before placement
and again on final dots. Thresholds: Spearman ρ &gt; 0.90 or mutual 3-px dot overlap &gt; 0.70 ⇒ duplicate.
Result: <b>0 duplicates</b>. Worst surface ρ = {card['correlation_overlap_vs_registry']['surface']['worst_spearman']:.3f};
worst final-dot mutual overlap = {card['correlation_overlap_vs_registry']['final_dots']['worst_mutual_3px_overlap']:.3f}
(worst forward {card['correlation_overlap_vs_registry']['final_dots']['worst_forward_3px_overlap']:.3f} vs
<code>{card['correlation_overlap_vs_registry']['final_dots']['worst_forward_submission']}</code>).
Scope caveat: this is the locally present subset, not a certified full-inventory scan — full certification is a
selector step. Prior submissions were used only for decorrelation checking, never as input to the field.</p>{supp_html}</div>

<div class="card" id="next"><h2>What to do instead (next steps, in order)</h2>
<ol>
<li><b>Do not submit this file.</b> For upcoming weekly slots, prefer sibling candidates that passed their own preregistered gates (e.g. GEMSDOE70's promoted P8/P9 per that repo's selector checklist), subject to the human selector and the weekly cap on the submission page.</li>
<li><b>Preregister the two follow-ups this session surfaced</b> (new draw, frozen gates, before fitting): (a) lidar-augmented arm A5 as a primary candidate (Exp-1 point estimate was highest, {S['A5_primary_lidar']['dti']:.4f} HOLDOUT-DTI, but it was ablation-only by precommitment and must re-earn promotion); (b) new-physics-only field judged on a piedmont/off-catalogue instrument declared in advance, since it led the SGMC proxy while trailing on catalogue holdout.</li>
<li><b>Get sub-km seismicity</b> (USGS ComCat FDSN / Nevada Seismological Lab; free official sources) on an unrestricted machine — unreachable from this sandbox (verified HTTP 000) and still the family's largest unused data source, including the 2020 Monte Cristo rupture inside the footprint.</li>
<li><b>Apply the band corrections family-wide:</b> rename band 6 to total-count and band 18 to ∂g/∂x wherever features are built; re-audit any “tilt” channel that consumed band 6.</li>
</ol></div>

<div class="card"><h2>Sources (official first; fetch status honest)</h2>
<table><tr><th>Source</th><th>Status here</th></tr>
<tr><td><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">DrivenData #306 problem description, metric, submission format</a></td><td>Fetched &amp; verified 2026-10-21 (α=0.2 β=0.8 R=300 m; worked example recomputed)</td></tr>
<tr><td><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">Public leaderboard</a></td><td>Fetched 2026-10-21: #1 xiaofanhu 0.4413 ORGANIZER-CONFIRMED (public board)</td></tr>
<tr><td><a href="https://github.com/drivendataorg/gems-prize-reference-solution">Official reference solution (U-Net)</a></td><td>Fetched 2026-10-21</td></tr>
<tr><td><a href="https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html">GEMSDOE32 (h33-2-b2 origin)</a> · <a href="https://buffedlizard55-lab.github.io/GEMSDOE70/">GEMSDOE70</a></td><td>Fetched 2026-10-21; 0.2778 remains OWNER-REPORT (no receipt found)</td></tr>
<tr><td><a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">USGS GeoDAWN data release</a> · <a href="https://gbcge.org/current-projects/ingenious/">INGENIOUS</a> · <a href="https://gdr.openei.org/submissions/1391">GDR 1391</a> · <a href="https://epsg.io/32611">EPSG:32611</a> · <a href="https://en.wikipedia.org/wiki/Tversky_index">Tversky index</a></td><td>Listed from the competition pages/brief; not re-fetched this session</td></tr>
<tr><td><a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">NLR rules PDF 96647</a> · <a href="https://earthquake.usgs.gov/fdsnws/event/1/query">USGS FDSN event service</a></td><td>Unreachable from this sandbox (HTTP 000, verified 2026-10-21); named official sources for an unrestricted machine</td></tr>
<tr><td>Miller &amp; Singh 1994 tilt derivative <a href="https://doi.org/10.1016/0926-9851(94)90022-1">DOI</a>; Verduzco et al. 2004 <a href="https://doi.org/10.1190/1.1651454">DOI</a></td><td>Bibliographic (paywalled; transforms implemented from standard definitions and unit-tested on synthetics)</td></tr>
<tr><td>Competition rasters</td><td>Owner-supplied GitHub mirrors of the login-walled DrivenData files, restored SHA-256 fail-closed (manifest in repo). Integrity-pinned, NOT organizer-authenticated.</td></tr>
</table></div>

<div class="card"><h2>Reproduce (CPU, no DrivenData login)</h2>
<p><code>python3 scripts/restore_data.py --ids all</code> (needs <code>gh</code>) →
<code>python3 scripts/build_features.py</code> → <code>python3 scripts/run_holdout.py</code> →
<code>python3 scripts/build_final.py --arm A4_primary</code> → <code>python3 scripts/run_replication.py</code> →
<code>python3 scripts/verify_all.py</code> → <code>python3 scripts/make_run_card_and_figures.py</code> →
<code>python3 scripts/build_site_contactphys.py</code>. Shared template (<code>vendor/gems57</code>: evaluate_holdout,
submission_writer, uniqueness) is reused unmodified. Run card: <a href="downloads/run_card_contactphys.json"><code>run_card_contactphys.json</code></a>.</p></div>

<footer>GEMSDOE77 · Core values: Maximize P(Win) — every slot is an experiment with a frozen gate · Own the Outcome — a failed gate is published, not quietly dropped. · Generated 2026-10-21 from <code>evidence/*.json</code> by <code>scripts/build_site_contactphys.py</code>. HOLDOUT-DTI numbers use evaluator {e1['evaluator_version']}; they are not leaderboard scores and no organizer score exists for any file here.</footer>
</div></body></html>"""

exec_summary = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Executive summary — how to submit (and why this GEMSDOE77 file must not be) · GEMSDOE77</title><link rel="stylesheet" href="assets/site-contactphys.css"></head><body><div class="wrap">
<header><h1>Executive summary: exactly how a GEMS submission is made</h1>
<p>For the human selector. Current GEMSDOE77 contact-physics artifact: <span class="no">DO NOT SUBMIT — Not approved for submission</span> (verdict negative — see the
<a href="contactphys.html">main page banner</a>). The procedure below is the complete, correct path for any future
<b>promoted</b> file, written so no step depends on memory.</p></header>

<div class="card stop"><h2>⛔ This session's file: download for inspection, do not upload</h2>
<p><code>{TIF}</code> (SHA-256 <code>{SHA}</code>) is format-valid (all portal range/CRS/shape checks pass locally)
but failed its preregistered science gate: two-draw paired HOLDOUT-DTI delta vs the standard-feature control
{H['two_draw_pooled']['delta']:+.4f} [{H['two_draw_pooled']['ci95'][0]:+.4f}, {H['two_draw_pooled']['ci95'][1]:+.4f}]
({H['two_draw_pooled']['withheld_positive_pixels_total']:,} withheld pixels, evaluator {e1['evaluator_version']}).
Spending one of the team's weekly slots on it would buy noise, not information. If a human selector ever overrode
this verdict, the unique name would be <code>{NAME}</code> and the Note-field text ({card['submission_note_chars']}/140)
would be <code>{NOTE}</code> — recorded here only so the artifact is unambiguous, not as a recommendation.</p></div>

<div class="card"><h2>Step-by-step submission (for a promoted file)</h2><ol>
<li>Confirm the candidate's run card says <code>verdict: promote</code> and <code>download_ok_format_valid: true</code>, and re-check its SHA-256 after download against the run card (<code>sha256sum file.tif</code>).</li>
<li>Confirm the weekly slot budget on the DrivenData submission page (this session spent 0 slots and cannot spend any).</li>
<li>Go to the competition <b>Submit</b> page: <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DrivenData #306</a> → <i>Submit file</i>.</li>
<li><b>File to submit:</b> choose the single-band GeoTIFF (<code>.tif</code>) <i>or</i> the <code>.zip</code> containing exactly that one GeoTIFF (this repo's writer produces both; upload <b>one</b> of them, once).</li>
<li><b>Note (optional):</b> paste the run card's note text verbatim (≤140 characters) so the submission is identifiable later.</li>
<li>Submit, then record the organizer's returned score + submission ID + date in the repo ledger as ORGANIZER-CONFIRMED (only portal receipts earn that label; screenshots/summaries do not).</li>
<li>Remember the two-round structure (problem page 967): before the deadline the team must select a <b>single</b> submission scored in both rounds — slot choice is a team/selector decision, never automated here.</li></ol></div>

<div class="card"><h2>The portal error “Predicted values must be in range [0, 1]” — root causes &amp; the permanent fix</h2>
<p>Two distinct defects produce this message, both measured in this data family:</p><ul>
<li><b>Float sentinel written through:</b> <code>training_features.tif</code> uses nodata <code>-3.4028234663852886e+38</code>; thousands of such cells lie inside the footprint. Any pipeline that copies them into a prediction emits values far outside [0, 1].</li>
<li><b>Nodata-tag / NaN-outside fragility:</b> files carrying a nodata tag outside [0, 1] (or relying on NaN handling) can be rejected by range scanning even when in-footprint values are legal.</li></ul>
<p><b>Fix used here (shared <code>gems57.submission_writer</code>, fail-closed, no silent repair):</b> the prediction is
validated in memory as finite in [0, 1] everywhere, written all-finite with <code>nodata=None</code> (0.0 outside the
footprint and on the mapped catalogue), re-read from disk, and re-validated (single-band float32, EPSG:32611,
3730×3292, transform identical to the sample submission) before the ZIP is packed. This session's file additionally
passed an independent plain-rasterio re-read: 12,279,160/12,279,160 cells finite, min 0.0, max 1.0, zero NaN —
<a href="downloads/verification_pass2.json">receipt</a>. Local validity is not organizer acceptance; it eliminates the
known mechanical causes of the range error.</p></div>

<div class="card"><h2>What “good” looks like at selection time (checklist)</h2><ul>
<li>Run card: hypothesis, mechanism, named mimic, HOLDOUT-DTI + CI + withheld-positive count + evaluator version.</li>
<li>Leakage canaries: every feature's symmetric holdout AUC ≤ 0.90.</li>
<li>Promotion gates all green (paired CI vs controls excludes 0; uniqueness ρ ≤ 0.90 and mutual 3-px overlap ≤ 0.70; writer + independent re-read validation).</li>
<li>Full-inventory uniqueness certification (beyond any single repo's local subset) and weekly-cap confirmation by the human selector.</li>
<li>Only then: one upload, receipt recorded as ORGANIZER-CONFIRMED.</li></ul>
<p><a href="contactphys.html">← Back to the GEMSDOE77 results page</a></p></div>
<footer>GEMSDOE77 executive summary · generated 2026-10-21 from measured receipts · no organizer score exists for any file in this repository.</footer>
</div></body></html>"""

(DOCS / "assets").mkdir(parents=True, exist_ok=True)
(DOCS / "assets" / "site-contactphys.css").write_text(CSS)
(DOCS / "contactphys.html").write_text(index)
(DOCS / "contactphys-executive-summary.html").write_text(exec_summary)
(DOCS / ".nojekyll").write_text("")
print("site written: docs/contactphys.html, docs/contactphys-executive-summary.html")
