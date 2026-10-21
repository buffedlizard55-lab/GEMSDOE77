#!/usr/bin/env python3
"""Render the branch-local H82-SE report after the shared site builder.

H82-SE (strike entropy) is distinct from upstream H82 (cool-outlet chemistry),
H82-TEXTURE, and H82R. This builder consumes only frozen evidence; it does not
rerun experiments, adjust uniqueness policy, select submissions, or spend slots.
"""
from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def e(value) -> str:
    return html.escape(str(value), quote=True)


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def build() -> None:
    card = load("evidence/h82se_run_card.json")
    e1 = load("evidence/h82se_e1_surface.json")
    p = Path(card["writer_receipt"]["file"])
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        p = ROOT / "submissions" / Path(card["writer_receipt"]["file"]).name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != card["raster_sha256"]:
        raise ValueError("H82-SE TIFF missing or SHA-256 differs from frozen run card")
    if card.get("ok_to_submit") is not False or card.get("verdict") != "negative":
        raise ValueError("H82-SE site builder fails closed unless lane is negative and not approved")
    h20 = card["holdout"]["seed20"]
    h30 = card["holdout"]["seed30"]
    u = card["correlation_overlap_vs_registry"]
    tif = p.name
    zipname = p.with_suffix(".zip").name
    note = card["submission_note"]
    if len(note) > 140:
        raise ValueError("submission note exceeds 140 characters")

    report_body = f"""<h1>H82-SE · local fault-strike entropy</h1>
<section class="card dl research"><h2>OK TO DOWNLOAD FOR INSPECTION — NOT OK TO SUBMIT</h2>
<p class="verdict"><b>Do not upload this TIFF.</b> H82-SE is a negative local research result: both holdout seeds are below the retained local incumbent and final-dot overlap exceeds the registered threshold. The indexed scan was incomplete, and later parallel-lane TIFFs merged after the frozen audit were not compared. The known duplicate-stop is sufficient to reject promotion. No selector approval or weekly slot was used.</p>
<a class="btn" href="downloads/{e(tif)}" download>Download H82-SE .tif</a>
<a class="btn alt" href="downloads/{e(zipname)}" download>One-TIFF .zip</a>
<p><b>Reference name only:</b> <code>{e(card['submission_name'])}</code><br><b>Reference note only — do not submit:</b> <code>{e(note)}</code><br><b>SHA-256:</b> <code>{e(card['raster_sha256'])}</code></p>
<div class="chips"><span class="chip ok">Local format: PASS</span><span class="chip bad">Holdout gate: FAIL</span><span class="chip bad">Final-dot uniqueness: FAIL</span><span class="chip warn">Selector: not approved</span></div></section>
<h2>Preregistered hypothesis</h2>
<p>Visible mapped-fault strands with locally mixed axial orientations may mark transfer/intersection zones from which additional unrecorded strands continue. The candidate uses a spin-2 Gaussian structure-tensor entropy multiplied by visible-trace density. The named mimic is digitization bends, dense mapped swarms, and non-geothermal intersections. Every catalogue-derived holdout feature uses only the fold-visible fault mask.</p>
<p>Ranked hypotheses, external-data availability checks, and official-source links: <a href="research/h82se_hypotheses.md">H82-SE research record</a> · <a href="evidence/preregistration_h82se.json">preregistration JSON</a>.</p>
<h2>Results — HOLDOUT-DTI, not leaderboard scores</h2>
<table><tr><th>Seed</th><th>HOLDOUT-DTI</th><th>Withheld positives</th><th>95% CI</th><th>Evaluator</th></tr>
<tr><td>20 (primary)</td><td>{h20['dti']:.6f}</td><td>{h20['withheld_positive_pixels']:,}</td><td>{h20['ci95'][0]:.6f}–{h20['ci95'][1]:.6f}</td><td>{e(h20['evaluator_version'])}</td></tr>
<tr><td>30 (confirmation)</td><td>{h30['dti']:.6f}</td><td>{h30['withheld_positive_pixels']:,}</td><td>{h30['ci95'][0]:.6f}–{h30['ci95'][1]:.6f}</td><td>{e(h30['evaluator_version'])}</td></tr></table>
<p>The retained local reference is <b>0.152718 HOLDOUT-DTI</b> (<code>{e(h20['evaluator_version'])}</code>, Seed 20, 11,321 withheld positives, 95% CI [0.131548, 0.175185]). Both H82-SE values are lower. The feature-alone canary symmetric AUC was {e1['canary']['symmetric_auc']:.6f} (11,321 withheld positives; threshold 0.90, PASS).</p>
<h2>Registry gate and file validation</h2>
<p>Pre-placement surface maximum absolute Spearman was {e1['surface_max_abs_spearman']:.6f} across {e1['preplacement_uniqueness']['registry_rasters_checked']} readable comparisons (below 0.90). The frozen final-dot audit found maximum forward 3-px overlap {u['final_worst_forward_3px_overlap']:.6f} (limit 0.70), {u['duplicate_firings']} duplicate firings, and maximum absolute Spearman {u['final_max_abs_spearman']:.6f}. It read {u['final_registry_rasters_checked']} of 863 entries and was incomplete. This is a historical pre-merge registry snapshot; later parallel lanes were not re-audited against H82-SE. No complete uniqueness certificate is claimed.</p>
<p>Local writer validation: one float32 band; EPSG:32611; 3730×3292; exact sample transform; finite values in [0,1]; no NaN/Inf; 37,654 positive pixels and zero on the mapped catalogue. Format PASS is not portal acceptance.</p>
<h2>Frozen evidence and official sources</h2>
<ul><li><a href="evidence/h82se_run_card.json">H82-SE run card</a></li><li><a href="evidence/h82se_e1_surface.json">E1 canary and surface audit</a></li><li><a href="evidence/h82se_e2_holdout.json">E2 dual-seed HOLDOUT-DTI</a></li><li><a href="evidence/h82se_e3_uniqueness.json">E3 final-dot audit</a></li><li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">Official task and metric</a></li><li><a href="https://gbcge.org/current-projects/ingenious/">GBCGE INGENIOUS project</a></li><li><a href="https://www.usgs.gov/publications/structural-discontinuities-and-their-control-on-hydrothermal-systems-great-basin-usa">USGS structural-discontinuities publication</a></li></ul>
<p><a href="index.html">Return to current start page</a> · <a href="executive-summary.html">How to submit only after approval</a></p>"""
    nav = '<header><div class="wrap"><span class="brand">GEMSDOE77 · DOE GEMS Prize</span><a href="index.html">Start here</a> <a href="h82-strike-entropy.html" class="here">H82-SE · negative</a> <a href="executive-summary.html">How to submit</a></div></header>'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H82-SE strike entropy · negative</title><meta name="description" content="H82-SE negative research lane; download for inspection only, do not submit."><link rel="stylesheet" href="assets/lia.css"></head><body>{nav}<main><div class="wrap">{report_body}</div></main><footer><div class="wrap"><b>Not approved to submit.</b> No selector approval or weekly slot was used.</div></footer></body></html>'''
    (DOCS / "h82-strike-entropy.html").write_text(page)

    for src, dst in (
        ("evidence/h82se_run_card.json", "docs/evidence/h82se_run_card.json"),
        ("evidence/h82se_e1_surface.json", "docs/evidence/h82se_e1_surface.json"),
        ("evidence/h82se_e2_holdout.json", "docs/evidence/h82se_e2_holdout.json"),
        ("evidence/h82se_e3_uniqueness.json", "docs/evidence/h82se_e3_uniqueness.json"),
        ("registry/preregistration_h82se.json", "docs/evidence/preregistration_h82se.json"),
        ("research/h82se_hypotheses.md", "docs/research/h82se_hypotheses.md"),
    ):
        out = ROOT / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / src, out)

    hero = f'''<section class="card dl research" id="h82se-status"><h2>H82-SE strike entropy: new research TIFF, inspection only</h2><p class="verdict"><b>OK TO DOWNLOAD FOR INSPECTION: YES. NOT OK TO SUBMIT: NO.</b> Seed-20 HOLDOUT-DTI {h20['dti']:.4f} (11,321 withheld positives; 95% CI [{h20['ci95'][0]:.4f}, {h20['ci95'][1]:.4f}]); final-dot 3-px overlap {u['final_worst_forward_3px_overlap']:.3f} exceeds 0.70. This is distinct from the parallel H82 cool-outlet and H82-TEXTURE lanes.</p><a class="btn" href="downloads/{e(tif)}" download>Download H82-SE .tif</a> <a class="btn alt" href="downloads/{e(zipname)}" download>One-TIFF .zip</a> <a class="btn alt" href="h82-strike-entropy.html">H82-SE report</a><p class="small">SHA-256 <code>{e(card['raster_sha256'])}</code>. The scan is incomplete and predates later parallel-lane TIFFs; format validation is not portal approval.</p></section>'''
    brand = '<span class="brand">GEMSDOE77 · DOE GEMS Prize</span>'
    for filename_page in ("index.html", "executive-summary.html"):
        path = DOCS / filename_page
        content = path.read_text()
        anchor = "<main><div class=\"wrap\">"
        if anchor not in content:
            raise ValueError(f"cannot find insertion point in {filename_page}")
        if 'id="h82se-status"' not in content:
            content = content.replace(anchor, anchor + hero, 1)
        header_start, header_end = content.find("<header"), content.find("</header>")
        if header_start < 0 or header_end < 0:
            raise ValueError(f"cannot find navigation header in {filename_page}")
        header = content[header_start:header_end]
        if 'href="h82-strike-entropy.html"' not in header:
            content = content.replace(brand, brand + '<a href="h82-strike-entropy.html">H82-SE · negative</a>', 1)
        path.write_text(content)
    print("published H82-SE report and inspection-only status card")


if __name__ == "__main__":
    build()
