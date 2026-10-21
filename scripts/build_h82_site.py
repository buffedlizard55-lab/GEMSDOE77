"""Publish H82 evidence/status; never run an experiment during a Pages build."""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs"


def build(render):
    card = json.loads((ROOT / "evidence/h82_texture/run_card.json").read_text())
    hold = json.loads((ROOT / "evidence/h82_texture/holdout.json").read_text())["summary"]
    path = ROOT / card["raster_path"]
    if card["ok_to_submit"] or card["verdict"] != "negative" or card["protocol_unique"]:
        raise ValueError("H82 is closed and must remain inspection-only")
    if hashlib.sha256(path.read_bytes()).hexdigest() != card["raster_sha256"]:
        raise ValueError("H82 artifact differs from frozen run card")
    esc = html.escape
    tif = card["raster_path"].removeprefix("docs/")
    score = card["holdout_dti"]
    lo, hi = score["ci95"]
    box = f'''<section class="card dl research" id="h82-status"><h2>H82-TEXTURE: new research-only TIFF — negative result</h2>
<p class="verdict"><b>OK TO DOWNLOAD: YES, for inspection. NOT OK TO SUBMIT.</b>
Newly computed K/Th texture-boundary surface, not a copied prior prediction. Format PASS does not mean competition approval.
Literal registry overlap fails; stopped before dot placement. Selector not approved; no slot used.</p>
<a class="btn" href="{esc(tif)}" download>Download H82 research .tif ({path.stat().st_size / 1e6:.1f} MB)</a>
<a class="btn alt" href="{esc(tif.replace('.tif', '.zip'))}" download>One-TIFF ZIP</a>
<a class="btn alt" href="h82-texture.html">Results, evidence and submission status</a>
<p><b>{score['dti']:.6f} HOLDOUT-DTI</b> (95% CI [{lo:.6f}, {hi:.6f}];
{score['withheld_positive_pixels']:,} withheld positives; <code>{score['evaluator_version']}</code>).
Below matched constant-output control; not comparable to historical sparse-dot incumbents.</p>
<p class="small">SHA-256 <code>{card['raster_sha256']}</code></p></section>'''
    rows = ''.join(f"<tr><td>{esc(name)}</td><td>{s['dti']:.6f}</td><td>[{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}]</td></tr>"
                   for name, s in hold['scores'].items())
    body = f'''<h1>H82-TEXTURE: local radiometric second-moment boundary</h1>{box}
<h2>Scientific finding</h2>
<p>Concordant K/Th local-standard-deviation boundaries did not outperform a constant surface at the preregistered
continuous-output operating point. Every table entry is HOLDOUT-DTI from <code>{score['evaluator_version']}</code>,
{score['withheld_positive_pixels']:,} withheld positives; 95% intervals resample aligned physical 20 km blocks (1000 draws).
These are not organizer-confirmed results or a forecast. This single test does not disprove every texture-based detector.</p>
<table><tr><th>Single-feature arm</th><th>HOLDOUT-DTI</th><th>95% CI</th></tr>{rows}</table>
<p>Paired candidate-minus-constant: {card['paired_vs_constant']['delta']:.6f} HOLDOUT-DTI,
95% CI [{card['paired_vs_constant']['ci95'][0]:.6f}, {card['paired_vs_constant']['ci95'][1]:.6f}].
Leakage-canary diagnostic (not a score): {card['leakage_canary']['tests']} feature-fold tests,
maximum symmetric AUC {card['leakage_canary']['max_symmetric_auc']:.6f}; none exceeds 0.90.</p>
<h2>Mechanism and alternative explanation</h2>
<p>{esc(card['mechanism'])}. The inputs are independently quantized rank intensities, not elemental concentrations.
A covered fault could separate smooth sediment from heterogeneous bedrock, but a depositional fan-to-lake sediment boundary
or acquisition noise can do the same without faulting. No vent, heat source or productive reservoir is identified.</p>
<h2>What is unique, and what failed?</h2>
<p>The decoded surface differs from all {card['local_inventory_compared']} prior local TIFF downloads checked.
This is distinction only, not a full-registry uniqueness certificate. The two SHA-pinned rejection witnesses give
maximum surface Spearman rho {card['registry']['surface_max_spearman']:.6f}, but forward 3-pixel support overlap 1.0
against the full-support witness exceeds 0.70. The unchanged template therefore rejects it. No density exemption,
mutual-overlap substitution or post-failure dot placement was used. Readback reproduces the checked surface exactly.</p>
<h2>Format readback</h2>
<p>Single-band float32; EPSG:32611; 3730 × 3292; transform (100, 0, 243350, 0, -100, 4508550).
All cells finite; zero NaN/Inf; min 0, max {card['validator']['max']:.9f} within [0,1]; outside-footprint and catalogue values zero;
no nodata tag. All 15 shared-validator checks pass. This avoids out-of-range values locally; it does not establish portal acceptance
or the cause of the user's earlier range error.</p>
<h2>Name and note — reference only, DO NOT UPLOAD</h2>
<p><code>{esc(card['submission_name'])}</code></p><p><code>{esc(card['submission_note'])}</code></p>
<h2>Evidence and sources</h2>
<ul><li><a href="evidence/h82_texture/run_card.json">Frozen run card</a> · <a href="evidence/h82_texture/holdout.json">Every arm, canary, split and CI</a></li>
<li><a href="evidence/h82_texture/surface_registry.json">Surface rejection</a> · <a href="evidence/h82_texture/readback_registry.json">Postwrite same-surface audit</a> · <a href="evidence/h82_texture/independent_audit.json">Independent readback audit</a></li>
<li><a href="research/h82_texture_hypotheses.md">Three hypotheses ranked before implementation; provenance limitations</a></li>
<li><a href="https://github.com/drivendataorg/gems-prize-reference-solution">Official reference code</a> (README freshly retrieved via GitHub API).</li>
<li><a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">Official USGS GeoDAWN source</a> (manual review only; host unavailable in this sandbox).</li></ul>
<h2>Why H33's reported result is plausible, but not proof</h2>
<p>The retained <a href="evidence/prior_forensics.json">byte forensics</a> show H33 is the earlier 44,090-dot D2.8 field
with 6,436 near-catalogue dots removed, leaving 37,654. Sparse placement can reduce redundant false-positive mass under the
300 m triangular metric, and pruning can remove unhelpful near-catalogue predictions. Without an exact-file organizer receipt,
0.2778 remains OWNER-REPORT. Neither that lineage nor the conflicting requested leaderboard targets prove a causal effect,
an optimal 200 m buffer, or a future gain. H82 supplies no evidence of beating it.</p>'''
    (D / "h82-texture.html").write_text(render("h82-texture.html", "H82 · negative research", body))
    # Put the current session status before historical banners on both entry pages.
    for name in ("index.html", "executive-summary.html"):
        p = D / name
        text = p.read_text()
        text = re.sub(r'<section class="card dl research" id="h82-status">.*?</section>', "", text, flags=re.S)
        marker = '<main><div class="wrap">'
        text = text.replace(marker, marker + box, 1)
        p.write_text(text)
    target = D / "evidence/h82_texture"
    target.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "evidence/h82_texture").glob("*.json"):
        shutil.copyfile(source, target / source.name)
    shutil.copyfile(ROOT / "research/h82_texture_hypotheses.md", D / "research/h82_texture_hypotheses.md")


if __name__ == "__main__":
    from build_home import page
    build(page)
