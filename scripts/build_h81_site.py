#!/usr/bin/env python3
"""H81 lane page + lanes.json entry + regenerated entry pages (generated, rerunnable).

Reads frozen evidence (evidence/h81_run_card.json, h81_e1/e2/e3, build) and
writes docs/h81.html, appends the H81 row to docs/data/lanes.json (idempotent),
regenerates docs/data/submission-authorization.json via release_gate, and
rebuilds the entry pages via build_home. Never hand-edit generated pages.
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def e(x):
    return html.escape(str(x), quote=True)


NAV = [("index.html", "Start here"), ("executive-summary.html", "How to submit"),
       ("h81.html", "H81 · slip-cone"),
       ("h79.html", "H79 · research-only"),
       ("h79-radhalo.html", "H79-RH · negative"),
       ("h77l.html", "H77L · research-only"), ("h77d.html", "H77D (Zero-SGMC)"),
       ("h77-step.html", "H77-Step"), ("h77e.html", "H77E · negative"),
       ("h78.html", "H78 lane"), ("lia.html", "LIA lane"),
       ("contactphys.html", "Contact-physics lane"),
       ("urank.html", "Selective-U lane"),
       ("lia-analysis.html", "Why 0.2778"), ("lia-sources.html", "Sources")]


def main() -> None:
    card = json.loads((ROOT / "evidence/h81_run_card.json").read_text())
    e1 = json.loads((ROOT / "evidence/h81_e1_surface.json").read_text())
    e2 = json.loads((ROOT / "evidence/h81_e2_holdout.json").read_text())
    build = json.loads((ROOT / "evidence/h81_e3_build.json").read_text())
    h = card["holdout_dti"]
    cov = card["correlation_overlap_vs_registry"]
    name = card["submission_name"]
    tif = f"downloads/{name}.tif"
    can_max = max(v["symmetric_auc"] for v in e1["leakage_canary"].values())
    scores = e2["summary"]["scores"]
    order = ["H81_slip", "tip_uniform", "normal_only", "scarp_gated", "random"]
    rows = "\n".join(
        f"<tr><td>{a}</td><td>{scores[a]['dti']:.5f}</td>"
        f"<td>[{scores[a]['ci95'][0]:.5f}, {scores[a]['ci95'][1]:.5f}]</td></tr>"
        for a in order if a in scores)
    nav = " ".join(f'<a href="{p}"{" class=here" if p == "h81.html" else ""}>{t}</a>'
                   for p, t in NAV)
    body = f"""<h1>H81: slip-budget fault-tip continuation cones</h1>
<p class="verdict"><b>OK to download for inspection: YES. NOT OK TO SUBMIT.</b>
This is a newly generated, byte-unique research artifact. Under the binding literal
registry rule it {(("fails (duplicate stop)" if cov["duplicate_stop"] else "passes"))};
no selector has approved a weekly slot. Selector status for every file on this site:
<b>not approved</b>.</p>
<section class="card dl"><h2>Download (inspection only — DO NOT SUBMIT)</h2>
<a class="btn" href="{tif}" download>Download .tif</a>
<a class="btn alt" href="{tif[:-4]}.zip" download>Same file as one-TIFF .zip</a>
<p class="small">Reference NaN-outside twin (not portal-safe, diagnostic reference only; no download link): <code>{e(build['nan_twin'])}</code></p>
<div class="chips"><span class="chip ok">OK to download: YES</span>
<span class="chip bad">OK to submit: NO</span>
<span class="chip {'bad' if cov['duplicate_stop'] else 'ok'}">Literal uniqueness: {'FAIL' if cov['duplicate_stop'] else 'PASS'}</span>
<span class="chip warn">Selector: not approved</span></div>
<p><b>Submission name (reference only):</b> <code>{e(name)}</code></p>
<p><b>Note (reference only, {len(card['submission_note'])}/140):</b> <code>{e(card['submission_note'])}</code></p>
<p class="small">sha256: <code>{card['raster_sha256']}</code></p></section>
<h2>Result</h2>
<p><b>{h['value']:.5f} HOLDOUT-DTI</b> (evaluator <code>{h['evaluator_version']}</code>,
{h['withheld_positive_pixels']:,} withheld positives, 95% CI
[{h['ci95'][0]:.5f}, {h['ci95'][1]:.5f}]). Incumbent {e2['current_holdout_best']['score']};
beats-current-best: <b>{'YES' if h['beats_current_best'] else 'NO'}</b> ({e(h['beats_rule'])});
verdict: <b>{e(card['verdict'])}</b>. Zero weekly slots used.</p>
<table><tr><th>Arm</th><th>HOLDOUT-DTI</th><th>95% CI</th></tr>{rows}</table>
<h2>Method (frozen before Exp 1)</h2>
<p>Strand-level fault tips (degree-1 nodes of each connected trace component of
public Qfault-derived segmentation) emit deterministic outward propagation cones
(2.0 km, half-angle 30&deg;, linear decay) weighted by component slip-rate rank.
No ML, no potential fields, no radiometrics, no lidar in the primary surface.
Component-LOCO: a strand touching hidden truth within 5 px is dropped from every
fold. Full specification: <code>registry/preregistration_h81.json</code> +
amendments <code>IR-81-03/04</code>; hypotheses: <code>research/h81_hypotheses.md</code>.</p>
<h2>Gates</h2>
<ul>
<li>Leakage canary: max symmetric AUC {can_max:.4f} (limit 0.90) — {'<b>FIRED</b>' if can_max > 0.90 else 'clean'}.</li>
<li>Surface uniqueness: max |Spearman &rho;| {e1['surface_max_abs_spearman']:.4f} (limit 0.90).</li>
<li>Final dots: max |&rho;| {cov['final_max_abs_spearman'] if cov['final_max_abs_spearman'] is not None else 'n/a'},
worst forward 3-px overlap {cov['final_worst_forward_3px_overlap']:.4f} (limit 0.70)
vs <code>{e(str(cov['final_worst_overlap_submission']))}</code>;
{cov['registry_rasters_checked']} rasters checked, complete scan: {cov['complete_accessible_scan']}.</li>
<li>Validator: all local checks {'PASS' if card['validator_output']['all_checks_passed'] else 'FAIL'}
(single-band float32, EPSG:32611, 3730&times;3292, finite [0,1], 0 on catalogue).</li>
</ul>
<h2>Mimic and limits</h2>
<p>{e(card['named_non_fault_process_that_could_mimic_it'])} Mapping-gap strand ends
from erosion/cover, digitisation fragments, and already-linked overlaps can all
mimic tips. The holdout is catalogue hide-and-recover — a local estimator, not a
leaderboard score. No organizer receipt exists for this file.</p>
<p class="small">Evidence: <code>evidence/h81_run_card.json</code>,
<code>evidence/h81_e1_surface.json</code>, <code>evidence/h81_e2_holdout.json</code>,
<code>evidence/h81_e3_build.json</code>, <code>evidence/h81_e3_uniqueness.json</code>.
Run <code>scripts/h81_pipeline.py surface|holdout|artifact|final-uniqueness|runcard</code>
after restoring data to reproduce.</p>"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>H81 slip-cone · GEMSDOE77</title>
<link rel="stylesheet" href="assets/lia.css"></head><body>
<header><div class="wrap"><span class="brand">GEMSDOE77 · DOE GEMS Prize</span>{nav}
<a href="https://github.com/buffedlizard55-lab/GEMSDOE77">GitHub</a></div></header>
<main><div class="wrap">{body}</div></main>
<footer><div class="wrap">Selector status for every file on this site: <b>not approved</b>.
Generated by <code>scripts/build_h81_site.py</code> from frozen evidence.</div></footer>
</body></html>"""
    (ROOT / "docs/h81.html").write_text(page)
    # lanes.json entry (idempotent)
    lp = ROOT / "docs/data/lanes.json"
    lanes = json.loads(lp.read_text())
    lanes["lanes"] = [l for l in lanes["lanes"] if not l["lane"].startswith("H81:")]
    up = ("Not OK to submit. OK to download for inspection only. "
          f"HOLDOUT-DTI {h['value']:.5f} ({h['evaluator_version']}, "
          f"{h['withheld_positive_pixels']:,} withheld positives, 95% CI "
          f"{h['ci95'][0]:.5f}-{h['ci95'][1]:.5f}); literal gate "
          f"{'FAIL' if cov['duplicate_stop'] else 'PASS'}; selector not approved; zero weekly slots.")
    lanes["lanes"].append({
        "lane": "H81: slip-budget fault-tip continuation cones (deterministic, catalogue-only)",
        "page": "h81.html", "summary": "executive-summary.html",
        "file": tif, "run_card": "evidence/h81_run_card.json",
        "format_valid": True,
        "protocol_unique": bool(not cov["duplicate_stop"]),
        "beats_holdout_best": bool(h["beats_current_best"]),
        "verdict": card["verdict"], "name": name, "note": card["submission_note"],
        "sha256": card["raster_sha256"], "holdout_dti": h["value"],
        "holdout_ci95": h["ci95"], "upload": up, "primary": False,
        "why_primary": "H81 strand-tip slip-cone lane; clearance per release gate (not approved)."})
    if "H81" not in lanes.get("note", ""):
        lanes["note"] = (lanes.get("note", "")
                         + " H81 (slip-cone, negative, research-only) is listed for inspection only.")
    lp.write_text(json.dumps(lanes, indent=1) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/release_gate.py"),
                    "--verify-fail-closed"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_home.py")], check=True)
    print("h81 site built")


if __name__ == "__main__":
    main()
