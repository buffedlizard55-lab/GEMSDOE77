#!/usr/bin/env python3
"""Build docs/ago.html — the AGO hybrid lane technical page."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs"
REPO = "https://github.com/buffedlizard55-lab/GEMSDOE77"

info = json.loads(Path("/tmp/ago_final_info.json").read_text())
card = json.loads(Path("evidence/ago_run_card.json").read_text())

# Build the page HTML
with open(D / "assets/lia.css") as f:
    pass  # check exists

NAV = [("index.html", "Start here"), ("executive-summary.html", "How to submit"),
       ("ago.html", "AGO · NEW primary"),
       ("h79.html", "H79 · research-only"), ("h79-radhalo.html", "H79-RH · negative"),
       ("h77l.html", "H77L · research-only"), ("h77d.html", "H77D (Zero-SGMC)"),
       ("h77d-executive-summary.html", "H77D guide"), ("h77-step.html", "H77-Step"),
       ("h77e.html", "H77E · negative"), ("h78.html", "H78 lane"), ("lia.html", "LIA lane"),
       ("contactphys.html", "Contact-physics lane"), ("urank.html", "Selective-U lane"),
       ("lia-analysis.html", "Why 0.2778"), ("lia-sources.html", "Sources")]

nav_html = " ".join(f'<a href="{p}"{" class=here" if p == "ago.html" else ""}>{t}</a>' for p, t in NAV)

# Primary file info
p = info["primary"]
h = info["high"]
s = info["safe"]

body = f"""
<h1>AGO: ML Backbone × Geophysical Displacement — Hybrid Fault Prediction</h1>
<p class="verdict"><b>Verdict: NEGATIVE (research) — NOT OK TO SUBMIT.</b>
Local range checks do not guarantee portal acceptance; the cause of the earlier range error is unverified.
No organizer score exists, no weekly slot was spent, and the selector has not approved any AGO variant.
Download for inspection only; do not upload.</p>

<div class="card dl" style="border: 2px solid #2a7ae2; background: #f0f7ff;">
<h2>⬇ AGO-Hybrid35 research TIFF — inspection download only; NOT OK TO SUBMIT</h2>
<p><b>AGO-Hybrid35</b>: 35% of 37,654 dots are far-field geophysics maxima displaced &gt;5px from the ML backbone;
65% retained as backbone dots on strong grav-mag contacts. Overlap <b>0.65 &lt; 0.70 PASSED</b> (backbone-specific).</p>
<p>
<a class="btn" href="downloads/{p['tag']}-zeros.tif" download>Download .tif — zeros outside (inspection only)</a>
<a class="btn alt" href="downloads/{p['tag']}-zeros.zip" download>Same bytes as one-TIFF ZIP (inspection only)</a>
<span class="warning"><b>NaN-outside variant:</b> diagnostic only; it is non-finite outside the footprint and fails the strict all-pixel preflight. Do not download/use as a submission file.</span>
</p>
<div class="chips">
<span class="chip ok">Local all-pixel range check: PASS (not portal acceptance)</span>
<span class="chip ok">Backbone overlap: 0.65 &lt; 0.70 PASS</span>
<span class="chip bad">Literal dense witness: 1.0 &gt; 0.70 FAIL (Policy I17 — needs owner fix)</span>
<span class="chip warn">Selector: not approved</span>
<span class="chip warn">No organizer score</span>
</div>
<p class="small">File: <code>{p['tag']}-zeros.tif</code> · {Path(D / f"downloads/{p['tag']}-zeros.tif").stat().st_size // 1024} KB · SHA-256 <code>{p['sha256_zeros']}</code></p>
<p class="small">Submission name (copy for DrivenData form — 67 chars):</p>
<div class="copy"><code>{p['tag']}</code><button data-copy="{p['tag']}">Copy</button></div>
<p class="small">Note for DrivenData form (copy — 99 chars):</p>
<div class="copy"><code>AGO hybrid35: ML backbone x geophys selective displacement (35% far-field infill, 37654 dots, 0 on catalogue)</code><button data-copy="AGO hybrid35: ML backbone x geophys selective displacement (35% far-field infill, 37654 dots, 0 on catalogue)">Copy</button></div>
<p class="small"><a href="downloads/{p['tag']}.json">Format/validator receipt (JSON)</a> · <a href="evidence/ago_run_card.json">Full run card (JSON)</a> · <a href="executive-summary.html">How to submit →</a></p>
<p class="small"><b>Can I submit this file?</b> <b>No.</b> All AGO variants fail the repository release gates; no selector approved them. The local all-pixel range check is not proof of portal acceptance and the previous rejection cause remains unknown. Use the repository-wide authorization receipt before any future upload.</p>
</div>

<section class="card">
<h2>Two additional AGO research variants — not approved for submission</h2>
<table>
<tr><th>Variant</th><th>Download (.tif / .zip)</th><th>Overlap vs D2.8</th><th>Holdout (quick)</th><th>Portal</th></tr>
<tr><td><b>Hybrid50</b> — 50% displaced (most unique)</td>
<td><a download href="downloads/{s['tag']}-zeros.tif">{s['tag']}-zeros.tif</a><br><a download href="downloads/{s['tag']}-zeros.zip">.zip</a><br><small>SHA <code>{s['sha256_zeros'][:16]}…</code></small></td>
<td>0.50 <span class="chip ok">PASS</span></td><td>0.0894</td><td><span class="chip ok">PASS</span></td></tr>
<tr><td><b>Select</b> — 0% displaced (pure subset, highest holdout)</td>
<td><a download href="downloads/{h['tag']}-zeros.tif">{h['tag']}-zeros.tif</a><br><a download href="downloads/{h['tag']}-zeros.zip">.zip</a><br><small>SHA <code>{h['sha256_zeros'][:16]}…</code></small></td>
<td>1.00 <span class="chip bad">FAIL</span></td><td>0.1444</td><td><span class="chip ok">PASS</span></td></tr>
</table>
<p class="small muted">All three: 37,654 binary dots, 0 on catalogue, single-band float32, EPSG:32611, 3730×3292, transform identical to sample_submission.tif. NaN-outside twins are diagnostic only and fail the strict all-pixel preflight; do not use them.</p>
</section>

<h2>Why this lane is genuinely new</h2>
<p>The H33 family improved leaderboard by <em>blindly</em> removing near-catalogue dots (all 6,436 within 200 m). This lane does <em>selectively</em>:</p>
<table>
<tr><th>Step</th><th>What</th><th>Why it matters</th></tr>
<tr><td>1. Score backbone</td><td>Combined grav-mag gradient strength: <code>sqrt(grav_gmag × rtp_gmag) × (0.5 + 0.5 × dilatation)</code></td><td>Strong contacts = likely real structure; dilatation weights active (geothermal-relevant) extension</td></tr>
<tr><td>2. Keep top 65%</td><td>24,476 dots on strongest contacts stay</td><td>Preserves ML backbone where geophysics corroborates it (likely blind continuations)</td></tr>
<tr><td>3. Displace bottom 35%</td><td>13,178 new dots at highest geophysics &gt;5px from any backbone dot (Poisson min_dist=2.8)</td><td>Fills inter-backbone gaps with independent geophysics — creates spatial novelty (overlap 0.65 &lt; 0.70)</td></tr>
</table>
<p><b>Different from every prior lane:</b></p>
<ul>
<li><b>vs H33</b>: H33 blindly prunes by catalogue distance; AGO scores by geophysics and <em>adds</em> far-field dots</li>
<li><b>vs H77-Step</b>: Step uses V-LBCO crack-tip mechanics + Radon + submodular packing from scratch; AGO reuses H19-5 backbone with geophysics reranking</li>
<li><b>vs AGO-v1 (polar asymmetry)</b>: Pure polarity field scored 0.067 (= random); the hybrid uses the ML backbone as anchor</li>
<li><b>vs H77D</b>: Hydrothermal + LOCO + SCARPS; not a backbone-reuse strategy</li>
<li><b>vs H78</b>: LiDAR consensus on off-support dots; no backbone reuse or displacement</li>
</ul>

<h2>Five hypotheses ranked (expected DTI gain ÷ cost)</h2>
<table>
<tr><th>Rank</th><th>Hypothesis</th><th>Layers &amp; signature</th><th>Why off-catalogue?</th><th>Difference from repo</th><th>Cost</th><th>Status</th></tr>
<tr><td>1 ★</td><td><b>ML backbone × geophysical displacement (THIS LANE)</b></td><td>H19-5 D2.8 backbone + grav-mag gradient × dilatation; 35% far-field infill</td><td>ML backbone misses some blind fronts; geophysics fills gaps where training was sparse</td><td>No prior lane reused backbone with scored displacement</td><td>Low (reuse + Poisson)</td><td>TESTED: 0.103 holdout, 0.65 overlap PASS</td></tr>
<tr><td>2</td><td>Anisotropic dip-slip polarity asymmetry</td><td>iso_grav_anom + RTP: directional sampling ±300 m along gradient (hangingwall low, footwall high)</td><td>Blind fronts under alluvium have no scarp but retain basement grav step</td><td>H77-Step used step tensor magnitude; this uses <em>directional polarity</em></td><td>Low</td><td>TESTED: 0.067 (= random) — REJECTED</td></tr>
<tr><td>3</td><td>Conductivity curvature × gravity slope (hydrothermal halo)</td><td>cond_surf + iso_grav_anom_slope curvature</td><td>Alteration changes conductivity near blind geothermal conduits</td><td>No lane has used cond_surf productively</td><td>Low-medium</td><td>NOT TESTED — needs next session</td></tr>
<tr><td>4</td><td>Multi-scale strain divergence × openness</td><td>geod_2ndinv + geod_dilat / det_elev openness</td><td>Relay ramps visible at specific scales</td><td>Multi-scale decomposition not in cached stack</td><td>Medium-high</td><td>NOT TESTED — needs GPU + new features</td></tr>
<tr><td>5</td><td>Tilt-derivative zero-contour persistence</td><td>Derived tilt from TMI bands filtered by depth_to_base coherence</td><td>Under-cover contacts have persistent zero-contours</td><td>Band 6 is NOT tilt (it's radiometric TC per forensics)</td><td>Low</td><td>NOT TESTED — genuine tilt derivation needed</td></tr>
</table>

<h2>Holdout evaluation (quick 4-quadrant spatial)</h2>
<p>Quick holdout (4 quadrants split at midlines ±30px buffer; visible faults masked; same evaluation grid as prior quick checks).
Not gems57-pooled-hide-v2 — see caveat.</p>
<table>
<tr><th>File (quick evaluator)</th><th>Dots</th><th>Holdout-DTI</th><th>vs D2.8 raw (0.1667)</th><th>vs Random (0.0688)</th><th>Overlap vs D2.8</th></tr>
<tr><td>D2.8 raw (44,090 dots)</td><td>44,090</td><td>0.1667</td><td>—</td><td>+0.0979</td><td>—</td></tr>
<tr><td>H19-5 solid (121K dots)</td><td>121,106</td><td>0.1716</td><td>+0.0049</td><td>+0.1028</td><td>—</td></tr>
<tr><td><b>AGO-Hybrid35 (PRIMARY)</b></td><td>37,654</td><td><b>0.1034</b></td><td>−0.0633</td><td><b>+0.0346</b></td><td><b>0.650 PASS</b></td></tr>
<tr><td>AGO-Hybrid50 (safe uniqueness)</td><td>37,654</td><td>0.0894</td><td>−0.0773</td><td>+0.0206</td><td>0.500 PASS</td></tr>
<tr><td>AGO-Select (no displacement)</td><td>37,654</td><td>0.1444</td><td>−0.0223</td><td>+0.0756</td><td>1.000 FAIL</td></tr>
<tr><td>AGO-polar (pure geophysics)</td><td>37,654</td><td>0.0667</td><td>−0.1000</td><td>−0.0021</td><td>—</td></tr>
<tr><td>Random (37654 dots)</td><td>37,654</td><td>0.0688</td><td>—</td><td>—</td><td>—</td></tr>
</table>
<p class="small muted">Leakage canary: individual grav/RTM gradient magnitudes score 0.056–0.065 on this holdout (all below random); the combined field alone scores 0.067 (random-level). The ML backbone dominates holdout performance; displacement trades ~0.04 holdout for uniqueness. These are <em>quick-evaluator</em> numbers (30px quadrant buffer, not gems57-pooled-hide-v2 with 8-connected-component buffered hide-and-recover), so not comparable to H77-Step 0.1527.</p>

<h2>Local format measurements — not organizer acceptance</h2>
<p>Raster readback found all cells finite and in [0,1] for the tested zeros variants. This is a local measurement only; it cannot guarantee portal acceptance. The prior error's cause is unverified, and no AGO file is approved to submit.</p>
<table>
<tr><th>Check</th><th>Result (all 3 AGO files)</th></tr>
<tr><td>Single band, float32, 3730×3292</td><td>PASS</td></tr>
<tr><td>CRS EPSG:32611, transform == sample_submission.tif</td><td>PASS</td></tr>
<tr><td>All 5,165,852 footprint pixels finite in [0,1]</td><td>PASS (min 0.0, max 1.0, 0 NaN)</td></tr>
<tr><td>All 12,279,160 pixels finite in [0,1] (zeros outside)</td><td>PASS</td></tr>
<tr><td>0 dots on mapped catalogue (60988 fault pixels)</td><td>PASS</td></tr>
<tr><td>No nodata sentinel, no Inf, no NaN anywhere (zeros variant)</td><td>PASS</td></tr>
</table>
<p class="small"><b>NaN-outside twins are diagnostics only and fail the strict all-pixel preflight.</b> Do not use them. Local checks are not portal acceptance.</p>

<h2>Submission status</h2>
<p class="verdict"><b>Do not submit an AGO variant.</b> Every variant fails repository release gates and no selector approved a slot. This page is an inspection record only. See <a href="data/submission-authorization.json">the authorization receipt</a>; local format checks do not imply eligibility or portal acceptance.</p>

<h2>Non-fault mimic &amp; caveats</h2>
<p><b>Named non-fault process:</b> Paleozoic thrust contacts, intrusive aureoles, and pluvial Lake Lahontan shoreline benches can mimic grav-mag steps — discriminated here by requiring positive geodetic dilatation and by the ML backbone's topographic context (which pure geophysics lacks). Pluvial benches are horizontal, not dipping, so polarity disambiguation helps.</p>
<p class="small muted">The quick holdout is NOT the organizer-evaluated hide-and-recover (gems57-pooled-hide-v2); it is a spatial-quadrant proxy for relative ranking only. No organizer score exists for any AGO file. The H33→0.2778 attribution is OWNER-REPORT (no authenticated receipt) and mechanism-only. The backbone's H19-5 training used the same training_features.tif bands, so holdout vs H19-5 is in-sample for the backbone but out-of-sample for the geophysical reranking.</p>

<h2>Technical supplements</h2>
<ul>
<li><a href="evidence/ago_run_card.json">Full JSON run card (hypothesis, mechanism, holdout, registry, validator, SHA, verdict)</a> — its historical <code>ok_to_submit_competition</code> text is superseded: local format checks are not portal acceptance and this lane is NOT OK TO SUBMIT.</li>
<li><a href="downloads/{p['tag']}.json">Primary file receipt (SHA, bytes, CRS, validator checks)</a></li>
<li><a href="evidence/prior_forensics.json">Prior H33 forensics (why thinning helped the metric)</a></li>
</ul>
"""

NAV_STR = " ".join(f'<a href="{p}"{" class=here" if p == "ago.html" else ""}>{t}</a>' for p, t in NAV)
hdr = f'<header><div class="wrap"><span class="brand">GEMSDOE77 · DOE GEMS Prize</span>{NAV_STR} <a href="{REPO}">GitHub</a></div></header>'
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AGO · NEW hybrid submission · GEMSDOE77</title>
<meta name="description" content="GEMSDOE77 AGO lane: ML backbone × geophysical displacement — NEW hybrid submission GeoTIFFs">
<link rel="stylesheet" href="assets/lia.css"></head><body>
{hdr}<main><div class="wrap">{body}</div></main>
<footer><div class="wrap">All files are research artifacts; no selector has approved a weekly slot. Generated from <code>evidence/ago_run_card.json</code> and <code>docs/downloads/*.json</code> receipts. Number labels: <b>ORGANIZER-CONFIRMED</b> (public leaderboard), <b>HOLDOUT-DTI</b> (local spatial hide-and-recover).</div></footer>
<script>document.querySelectorAll('button[data-copy]').forEach(b=>b.onclick=()=>{{navigator.clipboard.writeText(b.dataset.copy);b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1200)}})</script>
</body></html>"""

(D / "ago.html").write_text(html)
print(f"Wrote docs/ago.html ({len(html)//1024} KB)")

# Also ensure it's reachable from index
print("Done")
