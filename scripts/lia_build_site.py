#!/usr/bin/env python3
"""Build the archived LIA pages with fail-closed status and corrected H33 provenance.

The old submission manifest is stale and does not match the frozen LIA run card. This
builder deliberately reads the run card for historical lane evidence, never generates a
new candidate, and keeps the repository-wide authorization separate from local format.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs"
REPO = "https://github.com/buffedlizard55-lab/GEMSDOE77"
PAGES = [
    ("lia.html", "LIA lane overview"),
    ("lia-executive-summary.html", "How to submit"),
    ("lia-analysis.html", "H33 score provenance"),
    ("lia-sources.html", "Sources & limits"),
    ("index.html", "All GEMSDOE77 lanes"),
]


def e(value) -> str:
    return html.escape(str(value), quote=True)


def page(filename: str, title: str, body: str) -> str:
    nav = " ".join(
        f'<a href="{path}"{" class=here" if path == filename else ""}>{label}</a>'
        for path, label in PAGES
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)} · GEMSDOE77</title>
<link rel="stylesheet" href="assets/lia.css"></head><body>
<header><div class="wrap"><span class="brand">GEMSDOE77 · DOE GEMS Prize</span>{nav}
<a href="{REPO}">GitHub</a></div></header><main><div class="wrap">{body}</div></main>
<footer><div class="wrap">Historical research only. <b>NOT OK TO SUBMIT.</b> Selector status: <b>Not approved</b>. OWNER-REPORT, HOLDOUT-DTI, INSTRUMENT, and organizer row values are distinct evidence classes. A format receipt is not portal acceptance or selector approval.</div></footer>
</body></html>'''


def main() -> None:
    card = json.loads((DOC / "data/run-card.json").read_text())
    hold = card["holdout"]
    control = hold["champion_rule_fold_fair"]
    paired = hold["paired_vs_champion_rule_fold_fair"]
    registry = card["registry_comparison"]
    raster = card["raster"]
    file = Path(raster["file"]).name
    ci = hold["ci95"]
    control_ci = control["ci95"]
    paired_ci = paired["ci95"]

    overview = f'''<div class="notice"><strong>Archived LIA lane — NEGATIVE; NOT OK TO SUBMIT.</strong>
The raster is available for inspection only. The literal registry audit records {registry['space_filling_firings_in_template_audit']} overlap firings on space-filling rasters; its narrower informative-subset reading is diagnostic, not clearance. Selector approval: no. Weekly slots used by this lane: {card['budget']['weekly_slots_used_by_this_run']}.</div>
<p><b>Can I upload it?</b> <b>No.</b> Literal protocol uniqueness: FAIL; the informative subset is a diagnostic only.</p>
<h1>LIA: historical live-score inversion and allocation experiment</h1>
<p>This lane is closed. Its score inversion was fitted to a family ledger of reported values; any fitted density or expected-credit surface is <strong>INSTRUMENT</strong>, conditional on those inputs and model assumptions. It is not hidden truth, a leaderboard estimate, or a verified explanation of the H33 score.</p>
<section class="card dl"><h2>Inspection download — not a submission</h2>
<p><a class="btn" download href="downloads/{e(file)}">Download archived LIA TIFF</a></p>
<p>SHA-256 <code>{e(raster['sha256'])}</code> · {raster['dots']:,} dots · local holdout and registry evidence in the run card.</p>
<p><b>OK to download for inspection:</b> YES. <b>OK to submit:</b> NO. Local format checks do not prove portal acceptance.</p></section>
<h2>Recorded blocked holdout result</h2>
<p><strong>Candidate {hold['dti']:.5f} HOLDOUT-DTI</strong> (95% CI [{ci[0]:.5f}, {ci[1]:.5f}]; evaluator <code>{e(hold['evaluator_version'])}</code>; {hold['withheld_positive_pixels']:,} withheld positives) versus fold-rebuilt H19-5/H33-derived control <strong>{control['dti']:.5f} HOLDOUT-DTI</strong> (95% CI [{control_ci[0]:.5f}, {control_ci[1]:.5f}], same evaluator/count). Paired Δ was <strong>{paired['delta']:+.5f} HOLDOUT-DTI</strong> (95% CI [{paired_ci[0]:+.5f}, {paired_ci[1]:+.5f}], same evaluator/count). This local hide-and-recover result is below its comparator and is not a live score.</p>
<p><a href="lia-analysis.html">Read the corrected H33 provenance analysis</a> · <a href="lia-executive-summary.html">Submission guide for a future separately approved file</a>.</p>'''

    guide = '''<h1>Submission guide — this lane is not eligible</h1>
<div class="notice"><strong>Do not submit the LIA TIFF.</strong> Its verdict is negative, literal uniqueness is not cleared, and no selector has approved it. This page cannot make the artifact eligible.<br><b>Can I upload it?</b> <b>No.</b> <b>Literal protocol uniqueness: FAIL.</b> The informative-subset diagnostic does not override the literal rule. do not upload this file.</div>
<h2>For a future, separately approved candidate</h2>
<ol><li>Require a current authorization receipt and explicit selector approval.</li>
<li>Confirm current official rules, upload limit, file contract, and allowed data sources.</li>
<li>Verify the approved TIFF's exact SHA-256 and local grid/range validation; local PASS is not portal acceptance.</li>
<li>Upload only the approved file through the official competition portal and retain its organizer receipt.</li>
<li>Label a file-specific score ORGANIZER-CONFIRMED only when an organizer receipt or unambiguous organizer record links that score to those bytes.</li></ol>
<p>See the repository-wide <a href="executive-summary.html">current submission guide</a>. No credential or portal interaction was performed by this review.</p>'''

    analysis = '''<h1>H33 0.2778: reported attribution, conditional mechanism</h1>
<div class="notice"><strong>Evidence boundary.</strong> An archived 2026-10-21 public leaderboard read records 0.2778 for participant <code>extradr19</code> at row #48. It does not identify a submission TIFF or hash. The family attribution of 0.2778 to the GEMSDOE32 H33 artifact remains <strong>OWNER-REPORT</strong>; no inspected organizer receipt links the row to those bytes. Official pages were not freshly fetched in this review.</div>
<p>Archived owner-mirror forensics describe a sparse H19-5-derived raster with progressive thinning and mapped-catalogue-distance pruning. The sequence is compatible with reducing redundant prediction mass under a distance-tolerant metric, but it is not a controlled experiment. It does not show that those operations caused the reported score, that removed dots were false positives, that hidden faults lie on the H19-5 support, or that an optimum was reached.</p>
<h2>Exact marginal condition (including redundancy)</h2>
<p>For <code>DTI = T / (α·(T + S − M) + β·K)</code>, let <code>k</code> be a candidate dot's nearest-truth kernel weight and <code>dT</code> its marginal increase in weighted true-positive credit. With <code>α+β=1</code>, the exact improvement condition is <code>dT &gt; α·DTI·(dT + 1 − k)</code>. Only for a nonredundant dot (<code>dT=k</code>) does it reduce to <code>k &gt; α·DTI</code>. This algebra says nothing about unknown labels at H33's coordinates.</p>
<p>The archived H33-static reference measured <strong>0.004313 HOLDOUT-DTI</strong> on one local catalogue hide-and-recover draw (evaluator <code>gems57-pooled-hide-v2</code>; 11,342 withheld positives; 95% CI <code>[0.003508, 0.005188]</code>). This is a proxy-label measurement, not a public-score estimate or authentication of the file-to-score link.</p>
<p>Read the <a href="../research/why-h33-scored-02778.md">corrected H33 analysis</a>, <a href="../reports/why_02778.md">superseded-report correction</a>, and <a href="review-2026-10-21-current-session.md">current-session review</a>.</p>'''

    sources = '''<h1>Source and verification limits</h1>
<p>The official-source links below are retained for manual review. They were not fetched during the current repository review, so current rule text, access, availability, and provenance are not re-certified here.</p>
<ul><li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">DrivenData problem and metric page</a></li>
<li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">DrivenData public leaderboard</a> — prior snapshot only; participant rows do not identify a TIFF/hash.</li>
<li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/data/">DrivenData data page</a> — authenticated access/provenance not checked here.</li>
<li><a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">USGS GeoDAWN</a> · <a href="https://gdr.openei.org/submissions/1391">DOE GDR 1391</a> · <a href="https://www.usgs.gov/3d-elevation-program">USGS 3DEP</a></li></ul>
<p>Local bridge rasters and owner-mirror hashes establish what is present in this checkout, not organizer origin, license, completeness, or current availability.</p>'''

    pages = [
        ("lia.html", "LIA lane overview", overview),
        ("lia-executive-summary.html", "LIA submission guide", guide),
        ("lia-analysis.html", "H33 score provenance", analysis),
        ("lia-sources.html", "LIA sources and limits", sources),
    ]
    for name, title, body in pages:
        (DOC / name).write_text(page(name, title, body))
    (DOC / ".nojekyll").write_text("")
    print("Built four archived LIA pages with fail-closed submission status and corrected H33 provenance.")


if __name__ == "__main__":
    main()
