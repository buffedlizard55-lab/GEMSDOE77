"""Static site from audited evidence. No browser-side training or slot selection."""
import html
import json
import shutil
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs'
E = ROOT/'evidence'

def load(name): return json.loads((E/name).read_text())
def esc(x): return html.escape(str(x), quote=True)
def md(name):
    body = markdown.markdown((DOC/name).read_text(),extensions=['tables','fenced_code'])
    return body.replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')

def shell(title,active,body):
    nav=''.join(f'<a href="{file}"'+(' aria-current="page"' if key==active else '')+f'>{label}</a>'
       for file,key,label in [('index.html','home','Start here'),('h79-orient.html','h79','H79-ORIENT'),('h79-orient-executive-summary.html','h79-submit','H79 guide'),
                              ('h77d.html','h77d','H77D (Zero-SGMC)'),('h77d-executive-summary.html','h77d-submit','H77D guide'),
                              ('h77-step.html','h77-step','H77-Step'),('urank.html','overview','Selective-U lane'),('urank-executive-summary.html','submit','Selective-U guide'),
                              ('research.html','research','Research'),('sources.html','sources','Source feed'),('audit.html','audit','Audit trail')])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · GEMS / 77</title><meta name="description" content="Auditable fault-detection research. Download the locally validated research GeoTIFF. Not approved for competition submission.">
<link rel="stylesheet" href="assets/site.css"><script defer src="assets/site.js"></script></head><body>
<header><div class="header"><a class="brand" href="index.html">GEMS / 77 <span>Structural discovery lab</span></a><nav aria-label="Main">{nav}</nav></div></header>
<main>{body}</main><footer><span>Evidence before optimism. Maximize P(Win) · Own the Outcome.</span><span>Research, not a resource assessment. <a href="PROJECT_PROMPT.md">Standing request</a> · <a href="https://github.com/buffedlizard55-lab/GEMSDOE77">Repository</a></span></footer></body></html>'''

def main():
    card=load('run_card.json'); hold=load('holdout.json'); receipt=load('artifact_receipt.json')
    sources=load('source_health.json'); canaries=load('canaries.json')
    for p in E.glob('*.json'):
        target=DOC/'evidence'/p.name; target.parent.mkdir(exist_ok=True); shutil.copyfile(p,target)
    file=Path(card['file']).name
    h=card['holdout']; best=hold['scores'][card['best_comparable_control']]
    score=f'{h["dti"]:.5f}'; ci=f'{h["ci95"][0]:.5f}–{h["ci95"][1]:.5f}'
    maximum_auc=max(r['auc'] for r in canaries['rows'])
    manifest={'file':file,'sha256':card['raster_sha256'],'format_valid':True,'ok_to_download':True,
              'ok_to_submit':False,'protocol_unique':False,'verdict':'negative','submission_name':card['submission_name'],
              'note':card['note'],'organizer_confirmed_score':None,'weekly_cap':None,'weekly_slots_used':0,
              'holdout_valid_for_promotion':False,'holdout_caveat':card['holdout_caveat']}
    (DOC/'downloads/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    download=f'''<div class="panel download-panel"><span class="badge negative">Not OK to submit</span>
<h2>A new TIFF. A clear stop.</h2><p>Generated from cached radiometric features—not a copied submission. Local format checks pass, but the literal uniqueness gate fails.</p>
<div class="buttons"><a class="button" download href="downloads/{file}">↓ Download research TIFF</a><a class="button secondary" download href="downloads/{Path(file).with_suffix('.zip')}">Single-TIFF ZIP</a></div>
<p class="fine">{receipt['bytes']/1e6:.1f} MB · float32 · single band · values in [0, 1]<br>Safe to download for inspection. <strong>Do not upload to the competition.</strong></p>
<div class="file-label">SHA-256 / actual TIFF bytes</div><div class="hash">{esc(card['raster_sha256'])}</div>
<p class="fine"><a href="downloads/{Path(file).with_suffix('.json')}">Writer receipt</a> · <a href="evidence/run_card.json">JSON run card</a> · <a href="urank-executive-summary.html">Submission instructions</a></p></div>'''
    h77_step_banner = '''<div class="panel" style="border:2px solid #b91c1c;background:#fef2f2;margin-bottom:1.25rem;">
<span class="badge negative">Not OK to submit · Inspection download only</span>
<h2>Archived H77-Step Lane: <code>gems77-step-propensity-submod300.tif</code> (0.1527 HOLDOUT-DTI; Not Approved)</h2>
<p><strong>Recorded local spatial holdout:</strong> <code>0.1527</code> HOLDOUT-DTI <code>[0.1315, 0.1752]</code> on <code>gems57-pooled-hide-v2</code> (11,321 withheld positive pixels) and <code>0.1858</code> <code>[0.1619, 0.2093]</code> on Blind 4-Quadrant Transfer. Strictly binary <code>{0.0, 1.0}</code> in <code>[0, 1]</code> with <code>0 NaN</code> across all <code>12,279,160</code> cells. <strong>Not OK to submit:</strong> fails the binding literal registry-uniqueness gate (100%-coverage dense witness <code>ab0a0a62...</code> plus <code>post_placement_dots_max_forward_3px_overlap_real_models = 0.7350 &gt; 0.70</code> in <code>evidence/h77/run_card.json</code>); overall repository verdict is <strong>negative</strong> and <strong>Selector status: Not approved</strong>.</p>
<div class="buttons"><a class="button" download href="downloads/gems77-step-propensity-submod300.tif">↓ Download H77-Step .tif for Inspection (134.9 KB)</a><a class="button secondary" href="h77-step.html">Full H77-Step Evidence &amp; Figures →</a><a class="button secondary" href="h77-step-executive-summary.html">H77-Step Inspection Brief →</a></div>
<p class="fine"><strong>Filename:</strong> <code>gems77-step-propensity-submod300.tif</code> · <strong>SHA-256:</strong> <code>f55105a091281be14957fff2f5aad9a85fa5b1acc31558e9fcd2aae9fdd588b0</code><br><strong>Note (122/140):</strong> <code>GEMSDOE77: HW/FW dip-slip step tensor + V-LBCO crack-tip mechanics + Radon multi-scale ensemble + flank-zeroed 300m submod</code></p></div>'''
    overview=h77_step_banner+f'''<div class="eyebrow">DOE GEMS research · H77 · 21 October 2026</div><h1>Find structure.<br>Keep the evidence honest.</h1>
<p class="intro">A reproducible, independently generated fault-detection pilot. Download its GeoTIFF below—and see exactly why it is <strong>not approved for submission</strong>.</p>
<div class="hero-grid">{download}<div class="panel"><span class="eyebrow">Experiment 01 / stopped</span><h2>Selective-U curvature</h2><p>Subtract the K/Th common mode from U rank-intensity, then emphasize narrow, directional curvature. A test of concealed structure—not a claimed geothermal-vent discovery.</p>
<img class="preview" src="assets/surface.png" alt="Downsampled H77 radiometric curvature surface over the irregular study-area footprint">
<p class="fine">Research surface, not calibrated probabilities. Brighter = higher response. Image uses pixel coordinates, not a basemap.</p></div></div>
<div class="metrics"><div class="metric"><span>HOLDOUT-DTI</span><strong>{score}</strong><small>95% CI {ci}</small></div>
<div class="metric"><span>Hidden positive pixels</span><strong>{h['withheld_positive_pixels']:,}</strong><small>{esc(h['evaluator_version'])} · pooled four-fold</small></div>
<div class="metric"><span>Real submission slots used</span><strong>0</strong><small>Weekly cap not observed. Selector approval required.</small></div></div>
<div class="notice"><strong>Negative result, useful evidence.</strong> The paired gain over the strongest ablation is not statistically clear. A post-run audit also found potential diagonal nodata-stencil contact; the historical holdout result is confounded and not valid for promotion. A dense registry surface also forces 100% forward overlap under the literal shared rule. We stopped before dot placement; a new filename is not uniqueness clearance.</div>
<div class="two"><section><h2>What was verified</h2><ul><li>Shared evaluator and writer copied byte-for-byte from a pinned template.</li><li>Whole-component hide/recover, buffered spatial folds and per-feature canaries.</li><li>Actual TIFF readback: no NaN/Inf, bounded values, matching CRS, shape and transform.</li><li>ZIP contains exactly the same single TIFF.</li></ul></section>
<section><h2>What remains unverified</h2><ul><li>No authenticated organizer receipt or current leaderboard.</li><li>Mirror integrity is not proof of official input origin.</li><li>Full-family novelty and the current weekly cap are not established.</li><li>Official portal, PDF and science-host access is restricted here.</li></ul></section></div>
<p class="fine"><strong>Concurrent artifact notice:</strong> inherited H77C files are preserved but also NOT approved. Their earlier registry PASS was not measured clearance; the shared recheck rejects the artifact. <a href="evidence/inherited_h77c_review.json">Inherited-artifact review</a>.</p><div class="buttons"><a class="button secondary" href="research.html">Read the scientific assessment →</a><a class="button secondary" href="sources.html">Explore the source feed →</a></div>'''
    guide=h77_step_banner+f'''<div class="eyebrow">Executive summary / submission guide</div><h1>Download is easy.<br>Approval is separate.</h1>
<div class="notice"><strong>Current verdict: NEGATIVE. Do not submit this file.</strong> The research TIFF is format-valid but not protocol-cleared. No real submission has been made.</div>{download}
<section class="prose"><h2>How to submit a future approved candidate</h2><p>The supplied form accepts a single-band GeoTIFF or a ZIP containing one TIFF. The current portal and rules could not be independently re-read here; check the official page before using a real slot.</p>
<ol class="steps"><li><strong>Wait for explicit selector approval.</strong> Require a complete registry clearance, clean canaries, a defensible paired holdout improvement against a fold-rebuilt incumbent, authenticated inputs/rules and a known remaining weekly allowance. This candidate does not qualify.</li>
<li><strong>Download the approved TIFF or its single-TIFF ZIP.</strong> Do not submit a webpage, receipt JSON, Markdown, preview image or the entire repository.</li>
<li><strong>Check exact bytes and validator.</strong> Match the SHA-256 in its receipt. Require one band, EPSG:32611, shape 3730 × 3292, the sample's geotransform and finite values in [0,1]. These dimensions are measured against the owner mirror, not independently organizer-authenticated.</li>
<li><strong>Open the official competition portal.</strong> Use <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DrivenData GEMS</a> → the competition's submission form after signing in and accepting current rules. The precise current menu may differ. Never share credentials in chat.</li>
<li><strong>Choose the approved file and paste its note.</strong> Check the current weekly allowance before pressing the portal's submit control. Do not use the research-only file on this page.</li>
<li><strong>Keep the organizer receipt.</strong> Record exact filename/hash, note, UTC timestamp and organizer-reported result. Only then label a result ORGANIZER-CONFIRMED. Local validation is not acceptance.</li></ol>
<h2>Current research file identity</h2><p class="file-label">Name <button class="copy" data-copy="submission-name">Copy</button></p><div id="submission-name" class="hash">{esc(card['submission_name'])}</div>
<p class="file-label">Note · {len(card['note'])} characters <button class="copy" data-copy="submission-note">Copy</button></p><div id="submission-note" class="hash">{esc(card['note'])}</div>
<h2>Preventing the previous range error</h2><p>The original failure's root cause is not known. Our conservative export policy rejects non-finite and out-of-range predictions before writing, uses explicit zero outside the footprint, removes no-data sentinels, and reads the actual TIFF back. The writer does not silently clip or fill invalid model output. Local PASS does not guarantee portal acceptance.</p>
<h2>Reproduce locally</h2><pre>python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_competition_data.sh
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/audit_artifact.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_site.py</pre><p>The experiment runner refuses to re-run a finalized lane. Its stop is intentional. GitHub Pages hosts precomputed evidence and files; it cannot run a model in your browser or submit on your behalf.</p></section>'''
    rows=''.join(f'<tr><td>{esc(n)}</td><td>{r["dti"]:.6f}</td><td>{r["ci95"][0]:.6f}–{r["ci95"][1]:.6f}</td><td>{r["withheld_positive_pixels"]:,}</td></tr>' for n,r in hold['scores'].items())
    research=f'''<div class="eyebrow">Research / one hypothesis tested</div><h1>No claimed breakthrough.<br>A measured next step.</h1><div class="notice">All results below are <strong>HOLDOUT-DTI</strong>, evaluator {esc(h['evaluator_version'])}, {h['withheld_positive_pixels']:,} withheld positives, paired physical-block conditional 95% intervals. Not organizer scores. <strong>Post-run support audit found a boundary confound; these historical results are not valid promotion evidence.</strong> <a href="evidence/support_audit.json">Audit</a>.</div>
<div class="table-wrap"><table><thead><tr><th>Feature / arm</th><th>HOLDOUT-DTI</th><th>95% CI</th><th>Hidden positives</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="fine">Maximum per-feature/quadrant AUC = {maximum_auc:.5f} (LEAKAGE-CANARY diagnostic, not a score); no AUC exceeded the preregistered 0.90 threshold. Low AUC is not proof of zero leakage.</p>
<div class="prose">{md('knowledge.md')}<div class="line"></div>{md('hypotheses.md')}</div>'''
    checks=''.join(f'<tr><td>{esc(k)}</td><td><span class="badge good">PASS</span></td></tr>' for k,v in card['validator'].items() if v)
    audithtml=f'''<div class="eyebrow">Legacy Selective-U audit / immutable evidence</div><h1>Inspect the result.<br>Not just the claim.</h1><div class="notice"><strong>Format PASS ≠ submission approval.</strong> This page records the older Selective-U lane; its final dot placement/check was not run. The current H79-ORIENT artifact has a separate <a href="h79-orient.html">technical report</a> with both pre-placement and final-dot registry checks.</div>
<div class="two"><section><h2>TIFF preflight</h2><div class="table-wrap"><table><tbody>{checks}</tbody></table></div></section>
<section><h2>Receipts & reproducibility</h2><ul>'''+''.join(f'<li><a href="evidence/{p.name}">{esc(p.stem.replace("_"," "))}</a></li>' for p in sorted(E.glob('*.json')))+f'''</ul><p><a href="evidence/experiment.log">Original experiment log</a> · <a href="evidence/tests-pass2.txt">Pass 2 test output</a> · <a href="evidence/tests-pass3.txt">Pass 3 test output</a> · <a href="review.md">Three-pass review</a></p></section></div><div class="prose">{md('irregularities.md')}</div>'''
    repo_rows=''.join(f'<tr><td><a href="{esc(r.get("review_page") or r["url"])}">{esc(r["repo"])}</a></td><td>{esc(r["status"])}</td><td>{r.get("download_tif_count","—")}</td><td><code>{esc(r.get("commit", "")[:12])}</code></td><td>Not verified</td></tr>' for r in sources['repos'])
    official_rows=''.join(f'<tr><td><a href="{esc(r["url"])}">{esc(r["name"])}</a></td><td>{esc(r["status"])}</td><td>{esc(r.get("scope") or r.get("verification") or r.get("error"))}</td></tr>' for r in sources['official_sources'])
    sourcehtml=f'''<div class="eyebrow">Source feed / auditable freshness</div><h1>Sources, not certainty.</h1><p class="intro">A timestamped inventory of the supplied family repositories and page bytes, plus explicit official-source access status. No scraped mention is treated as an organizer score receipt.</p>
<p><span class="badge good">{sum(r['status']=='retrieved' for r in sources['repos'])} repository pages retrieved</span> <span class="fine">Checked {esc(sources['checked_utc'])}</span></p><div class="notice">This feed tracks GitHub source snapshots, not live competition standings. Scheduled refreshes publish a new timestamp when successful. Retrieval errors remain visible. Nothing here auto-selects a weekly slot.</div>
<h2>Latest focused source check</h2><p>The current branch review separately hash-checked the GEMSDOE32 owner page and inspected the manifest-pinned GDR well/spring CSV schema. The result is not an organizer receipt or a complete source inventory. Official DrivenData/USGS/GDR/Copernicus hosts were not fetched in that review. See the <a href="evidence/source_check_f7b973cd.json">machine-readable focused source check</a> and <a href="review-latest.html">current branch review</a>.</p>
<h2>Official sources</h2><div class="table-wrap"><table><thead><tr><th>Source</th><th>Access status</th><th>Verification boundary</th></tr></thead><tbody>{official_rows}</tbody></table></div>
<h2>Family site inventory</h2><label for="source-search">Filter by repository or status</label><br><input id="source-search" class="search" type="search" placeholder="e.g. GEMSDOE32" autocomplete="off">
<div class="table-wrap"><table><thead><tr><th>Repository / pinned review page</th><th>Status</th><th>Download TIFF paths</th><th>Commit</th><th>Organizer receipt</th></tr></thead><tbody id="repo-rows">{repo_rows}</tbody></table></div>
<p class="fine">Inventory counts are measurements, not score claims. Download paths may include old, quarantined or duplicate variants. This is not a full raster uniqueness scan. <a href="source-health.json">Full machine-readable feed, page hashes and paths</a>.</p>
<h2>Verification policy</h2><p>Only the official reference code was obtainable through an official GitHub repository. Data packs and the GeoDAWN report extraction come from owner mirrors. Source existence and matching hashes prove consistency, not scientific interpretation, current licensing, data eligibility or organizer origin. Restricted hosts are not described as nonexistent or down.</p>'''
    # Additive (LIA lane, PR #6): optional all-lanes box from docs/data/lanes.json at the top of the two entry pages,
    # so every lane's download and status is visible first. Absent file -> no box; nothing else changes.
    lanes_box=''
    lj=DOC/'data/lanes.json'
    if lj.exists():
        L=json.loads(lj.read_text())
        rows=''.join(f'<tr><td><a href="{esc(x["page"])}">{esc(x["lane"])}</a></td><td><a download href="{esc(x["file"])}">{esc(Path(x["file"]).name)}</a></td><td>{esc(x["verdict"])}</td><td>{esc(x["upload"])}</td></tr>' for x in L['lanes'])
        lanes_box=f'<div class="panel"><span class="eyebrow">All lanes in this repository</span><div class="table-wrap"><table><tr><th>Lane</th><th>File</th><th>Verdict</th><th>Upload status</th></tr>{rows}</table></div><p class="fine">{esc(L["note"])}</p></div>'
    h77d_banner=''
    if (E/'run_card_h77d.json').exists():
        from build_h77d_site import build_h77d_pages
        h77d_banner=build_h77d_pages()
    overview=h77d_banner+lanes_box+overview; guide=h77d_banner+lanes_box+guide
    # Pages restructure (H78 lane): index.html / executive-summary.html are now the repository-wide
    # "Start here" pages written by scripts/build_home.py; this lane's pages moved to urank*.html.
    for filename,title,active,body in [('urank.html','Selective-U lane','overview',overview),('urank-executive-summary.html','Selective-U submission guide','submit',guide),
           ('research.html','Scientific assessment','research',research),('audit.html','Audit trail','audit',audithtml),('sources.html','Source feed','sources',sourcehtml)]:
        (DOC/filename).write_text(shell(title,active,body))
    shutil.copyfile(E/'experiment.log',DOC/'evidence/experiment.log')
    (DOC/'.nojekyll').write_text('')
    print('Built five static pages with explicit download/submission status.')

if __name__ == '__main__': main()
