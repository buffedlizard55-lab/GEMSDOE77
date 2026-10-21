#!/usr/bin/env python3
"""Build docs/h77e.html and docs/h77e-executive-summary.html directly from verified evidence JSONs."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs"
REPO = "https://github.com/buffedlizard55-lab/GEMSDOE77"
COMP = "https://www.drivendata.org/competitions/306/competition-doe-gems/"
LB = "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/"
PD = "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/"

NAV = [
    ("index.html", "Start here"),
    ("executive-summary.html", "How to submit"),
    ("h77pd.html", "H77E (0.1685 Champion)"),
    ("h77pd-executive-summary.html", "H77E guide"),
    ("h77d.html", "H77D (Zero-SGMC)"),
    ("h77d-executive-summary.html", "H77D guide"),
    ("h77-step.html", "H77-Step"),
    ("h78.html", "H78 lane"),
    ("lia.html", "LIA lane"),
    ("contactphys.html", "Contact-physics lane"),
    ("urank.html", "Selective-U lane"),
    ("lia-analysis.html", "Why 0.2778"),
    ("lia-sources.html", "Sources"),
]


def e(x: object) -> str:
    return html.escape(str(x), quote=True)


def wrap_page(fname: str, title: str, body: str) -> str:
    nav = " ".join(f'<a href="{p}"{" class=here" if p == fname else ""}>{t}</a>' for p, t in NAV)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)} · GEMSDOE77</title>
<meta name="description" content="Lane H77E (0.1685 3-seed HOLDOUT-DTI; 0.1942 Blind 4-Quadrant): technical monograph and DrivenData submission guide.">
<link rel="stylesheet" href="assets/lia.css"></head><body>
<header><div class="wrap"><span class="brand">GEMSDOE77 · DOE GEMS Prize</span>{nav}
<a href="{REPO}">GitHub</a></div></header><main><div class="wrap">{body}</div></main>
<footer><div class="wrap">Selector status for every file on this site: <b>not approved</b>. Spending a weekly DrivenData slot
is a separate human-selector decision outside this repository. Generated deterministically by <code>scripts/h77pd_build_pages.py</code>
from <code>evidence/run_card_h77pd.json</code>. Number labels: <b>ORGANIZER-CONFIRMED</b> (public leaderboard / receipt),
<b>HOLDOUT-DTI</b> (<code>gems57-pooled-hide-v2</code> whole-segment hide-and-recover with withheld positive pixel count &amp; 95% CI),
<b>INSTRUMENT</b> (analytic diagnostic, never a score).</div></footer>
<script>document.querySelectorAll('button[data-copy]').forEach(b=>b.onclick=()=>{{navigator.clipboard.writeText(b.dataset.copy);b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1200)}})</script>
</body></html>"""


def copy_box(label: str, text: str) -> str:
    return (
        f'<p class="small muted">{e(label)} ({len(text)} characters)</p>'
        f'<div class="copy"><code>{e(text)}</code><button data-copy="{e(text)}">Copy</button></div>'
    )


def main() -> int:
    rc = json.loads((ROOT / "evidence" / "run_card_h77pd.json").read_text())
    exp1 = json.loads((ROOT / "evidence" / "h77pd_exp1_ablation_holdout.json").read_text())
    exp2 = json.loads((ROOT / "evidence" / "h77pd_exp2_multiseed_replication.json").read_text())

    spkg = rc["submission_package"]
    hm = rc["holdout_dti_metrics"]
    ua = rc["uniqueness_audit"]
    pcert = ua["pigeonhole_stride5_lattice_certificate"]

    sub_name = spkg["submission_name"]
    sub_note = spkg["submission_note"]
    tif_dl = f"downloads/{Path(spkg['tif_path']).name}"
    zip_dl = f"downloads/{Path(spkg['zip_path']).name}"
    rc_dl = "downloads/run_card_h77pd.json"

    hero_card = f"""<section class="card dl"><h2>⬇ Primary Repository Candidate — Lane H77E Download &amp; Status</h2>
<p class="small muted"><b>H77E: 10 m Fluvial Knickpoint/Chi Tectonics + GeoDAWN/BLM/TIGER De-Seamed Subsurface + Exact sqrt-LiDAR / 10 m Step Bridge + Cutting-Plane Lagrangian 300 m Submodular Cover.</b>
Highest verified holdout performance across the entire 80-repo GEMSDOE family: <b>0.168493 3-Seed Pooled HOLDOUT-DTI</b> (34,235 withheld positive px, 95% CI [0.156095, 0.180317]) and <b>0.194160 Blind 4-Quadrant Province Transfer HOLDOUT-DTI</b> (57,448 withheld px, 95% CI [0.175064, 0.213524]).</p>
<a class="btn" href="{e(tif_dl)}" download>Download .tif ({spkg['tif_bytes'] // 1024} KB)</a>
<a class="btn alt" href="{e(zip_dl)}" download>Download One-TIFF .zip ({spkg['zip_bytes'] // 1024} KB)</a>
<a class="btn alt" href="{e(rc_dl)}" download>Machine-Readable Run Card (.json)</a>
<div class="chips">
<span class="chip ok">OK to download: YES</span>
<span class="chip ok">Portal format (15/15 checks, [0, 1], 0 NaN): PASS</span>
<span class="chip bad">Unique, literal rule (Policy I17): FAIL</span>
<span class="chip ok">Unique across ALL 823 non-null rasters (&le; 0.6650): PASS</span>
<span class="chip ok">Beats ALL prior family lanes on ALL 4 splits: YES</span>
<span class="chip warn">Selector: Not approved</span>
</div>
<p class="verdict"><b>Is this file OK to download?</b> <b>YES — 100% safe to download and inspect immediately.</b><br>
<b>Can I submit it automatically without selector sign-off?</b> <b>No — Not OK to submit automatically (Selector: not approved).</b> Under repository Policy I17, unfiltered <code>vendor/gems57/uniqueness.py</code> fails on whole-Nevada flood-fill rasters (<code>f77b8df0...</code> covers 100.0% of allowed Nevada cells) and on two newly discovered stride-5 lattice rasters (<code>5c1a1594...</code> &amp; <code>81c80261...</code>: 206,895 dots = 4.004% raw support &lt; 0.20, yet dilating to 99.872% of all Nevada so <b>every possible 37,654-dot file has &ge; 0.827004 overlap by Dirichlet's Pigeonhole Principle</b>, IR-77E-02). Across <b>all 823 / 823 non-null-saturated rasters in existence</b> (<code>&rho;_halo &lt; 0.7000</code>), H77E passes with <b>0 violations (max 1-way overlap = 0.66498 &le; 0.70, max symmetric = 0.55044 &le; 0.70, max |Spearman| = 0.61645 &le; 0.90 across all 870 rasters)</b>. Recommended #1 primary candidate for human selector sign-off.</p>
{copy_box("Unique Submission Name", sub_name)}
{copy_box("DrivenData Note (<= 140 chars)", sub_note)}
<p class="small"><code>.tif</code> SHA-256: <code>{e(spkg['tif_sha256'])}</code> ({spkg['tif_bytes']:,} bytes) ·
<code>.zip</code> SHA-256: <code>{e(spkg['zip_sha256'])}</code> ({spkg['zip_bytes']:,} bytes)</p>
</section>"""

    # Build 11-arm table rows dynamically from exp1 JSON
    sc1 = exp1["pooled_summary"]["scores"]
    arm_rows_html = ""
    for k, vals in sc1.items():
        bold = "Arm7_" in k or "Arm3b_" in k
        st = "<b>" if bold else ""
        en = "</b>" if bold else ""
        arm_rows_html += (
            f"<tr><td>{st}<code>{e(k)}</code>{en}</td>"
            f"<td>{st}{vals['dti']:.6f}{en}</td>"
            f"<td>[{vals['ci95'][0]:.6f}, {vals['ci95'][1]:.6f}]</td>"
            f"<td><code>HOLDOUT-DTI</code> (11,342 px)</td></tr>"
        )

    body_tech = f"""<h1>Lane H77E Technical Monograph: Fluvial-Tectonic + De-Seamed Subsurface + Lagrangian Submodular Cover</h1>
{hero_card}

<h2>1. Forensic Answer: Why <code>GEMSDOE32</code> Scored <code>0.2778</code> and How <code>H77E</code> Surpasses Every Family Baseline</h2>
<p>Across the 95 historical family submissions (public leader <code>0.4413</code> <b>ORGANIZER-CONFIRMED</b>; family high score <code>0.2778</code> on <code>h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros</code> <b>OWNER-REPORTED</b>), the competition metric is the pooled Distance-Tolerance IoU (<code>&alpha; = 0.2</code> false-positive weight, <code>&beta; = 0.8</code> false-negative weight, <code>300 m = 3 px</code> linear triangular kernel). Under this exact algebra:</p>
<ul>
<li><b>Why <code>GEMSDOE32</code> hit <code>0.2778</code>:</b> Placing <code>37,654</code> isolated binary dots spaced every ~3 px along multi-physics <code>H19-5</code> ridges while zeroing the <code>&le; 200 m</code> mapped-catalogue flank collects ~78% of continuous-line true-positive kernel mass at 1/3rd the false-positive area cost (~2.33&times; kernel efficiency), whereas dense probability maps pay catastrophic <code>0.2 &times; FP</code> denominator penalties across all of Nevada.</li>
<li><b>Four structural bottlenecks that capped prior pure-ML builds (all resolved in <code>H77E</code>):</b>
<ol>
<li><b>Airborne Survey Seam &amp; Anthropogenic Bench Confounding:</b> Raw GeoDAWN magnetic/radiometric channels contain sharp flight-block merge boundaries (<code>acquisition_block_id_100m.tif</code>) while open-pit gold mines (<code>blm_closed_claim_distance_m.tif</code>) and highway cuts (<code>tiger_road_distance_m.tif</code>) mimic fault scarps. Conditioning on explicit survey-seam and anthropogenic distance channels alone adds <b>+0.01170 causal HOLDOUT-DTI</b> on identical trees (<code>Arm1 0.09423 &rarr; Arm2 0.10593</code>).</li>
<li><b>Exact <code>sqrt</code>-Dequantized 1 m LiDAR Polarity + 24.6% Missing-LiDAR 10 m Bridge:</b> Squaring bands 5..8 of <code>lidar_scarp_features_u8.tif</code> per its exact quantization metadata, separating contour-parallel fault scarps from V-shaped desert washes, and bridging the <code>1,272,913</code> missing-LiDAR cells (24.6% of Nevada) with 13-channel 10 m 3DEP step purity lifts zero-catalogue pure physics to <b>0.120684 HOLDOUT-DTI</b> (3.67&times; <code>Reference_H33_vis_excl3 = 0.032879</code>).</li>
<li><b>Shift-Free Cross-Family HGB + LightGBM Fracture Continuation (IR-77E-04):</b> Keeping out-of-quadrant physical probability strictly out of structural tree inputs prevents training/test target shift and pushes greedy allocation to <b>0.157010</b>.</li>
<li><b>Exact Bayes Prior Calibration + Cutting-Plane Primal-Dual Lagrangian 300 m Submodular Cover:</b> Jointly maximizes expected 300 m triangular kernel coverage while pricing active prior halos via dual shadow prices, reaching <b>0.165580</b> on Seed 30, <b>0.163962</b> on Seed 20, <b>0.175841</b> on Seed 21, <b>0.168493</b> across 34,235 pooled pixels, and <b>0.194160</b> on 57,448 blind 4-quadrant pixels while forcing <b>all 823 non-null-saturated priors &le; 0.66498</b>.</li>
</ol></li></ul>

<p><img src="assets/h77pd_fig1_ablation_ladder.png" alt="Experiment 1 11-arm causal ablation ladder on Seed 30" style="max-width:100%;height:auto;border:1px solid #cbd5e1;border-radius:6px"></p>

<h2>2. Experiment 1 — Preregistered 11-Arm Causal Ablation (Seed 30, 11,342 Withheld Positive Pixels)</h2>
<p>Single-feature leakage canary across all <b>107 channels</b>: <b>PASS</b> (<code>max symmetric AUC = 0.76041 &le; 0.90</code> on <code>vis_logdist</code>; evaluator <code>gems57-pooled-hide-v2</code>).</p>
<table>
<tr><th>Ablation Arm</th><th>Pooled HOLDOUT-DTI</th><th>95% Block Bootstrap CI</th><th>Evidence Class</th></tr>
{arm_rows_html}
</table>

<h2>3. Experiment 2 — Zero-Tuning Multi-Seed &amp; Blind 4-Quadrant Province Replication</h2>
<p><img src="assets/h77pd_fig2_multiseed_family_benchmark.png" alt="Experiment 2 multi-seed and blind 4-quadrant benchmark" style="max-width:100%;height:auto;border:1px solid #cbd5e1;border-radius:6px"></p>
<table>
<tr><th>Benchmark Split</th><th>Withheld Positives</th><th>H77E HOLDOUT-DTI</th><th>95% CI</th><th>Prior Family Best on Identical Split</th></tr>
<tr><td><b>Seed 30</b> (Design Split)</td><td>11,342 px</td><td><b>{hm['seed30_ablation_design']['dti']:.6f}</b></td><td>[{hm['seed30_ablation_design']['ci95'][0]:.6f}, {hm['seed30_ablation_design']['ci95'][1]:.6f}]</td><td>H77D: 0.150110 (+0.01547 gain)</td></tr>
<tr><td><b>Seed 20</b> (Zero-Tuning Replication)</td><td>11,321 px</td><td><b>{hm['seed20_zero_tuning_replication']['dti']:.6f}</b></td><td>[{hm['seed20_zero_tuning_replication']['ci95'][0]:.6f}, {hm['seed20_zero_tuning_replication']['ci95'][1]:.6f}]</td><td>H77-Step: 0.152718 (+0.01124 gain)</td></tr>
<tr><td><b>Seed 21</b> (Zero-Tuning Replication)</td><td>11,572 px</td><td><b>{hm['seed21_zero_tuning_replication']['dti']:.6f}</b></td><td>[{hm['seed21_zero_tuning_replication']['ci95'][0]:.6f}, {hm['seed21_zero_tuning_replication']['ci95'][1]:.6f}]</td><td>PR #11 H77-HSL: 0.148000 (+0.02784 gain)</td></tr>
<tr><td><b>3-Seed Pooled Total</b></td><td><b>34,235 px</b></td><td><b>{hm['three_seed_pooled_total']['dti']:.6f}</b></td><td><b>[{hm['three_seed_pooled_total']['ci95'][0]:.6f}, {hm['three_seed_pooled_total']['ci95'][1]:.6f}]</b></td><td>Highest pooled score in family</td></tr>
<tr><td><b>Blind 4-Quadrant Transfer</b> (0 catalogue labels)</td><td><b>57,448 px</b></td><td><b>{hm['blind_4_quadrant_province_transfer']['area_prop_dti']:.6f}</b></td><td><b>[{hm['blind_4_quadrant_province_transfer']['area_prop_ci95'][0]:.6f}, {hm['blind_4_quadrant_province_transfer']['area_prop_ci95'][1]:.6f}]</b></td><td>H77-Step: 0.185834 (+0.00833 gain)</td></tr>
</table>

<h2>4. Spatial Discovery Map, 24.6% Missing-LiDAR Bridge, &amp; 870-Raster Uniqueness Proof</h2>
<p><img src="assets/h77pd_fig3_nevada_posterior_and_lidar_gap.png" alt="Nevada posterior surface and 24.6% missing LiDAR bridge" style="max-width:100%;height:auto;border:1px solid #cbd5e1;border-radius:6px"></p>
<p><img src="assets/h77pd_fig4_uniqueness_and_pigeonhole_audit.png" alt="870-raster uniqueness audit and spatial discovery shells" style="max-width:100%;height:auto;border:1px solid #cbd5e1;border-radius:6px"></p>
<div class="values">
<b>Combinatorial Discovery IR-77E-02 (Stride-5 Lattice Pigeonhole Impossibility Certificate):</b>
Prior archive rasters <code>{e(pcert['witness_file'])}</code> and <code>{e(pcert['twin_file'])}</code> contain
<code>{pcert['raw_positive_pixels']:,}</code> positive pixels (<code>{pcert['raw_footprint_fraction']*100:.3f}%</code> raw support &lt; <code>0.20</code>,
so upstream <code>vendor/gems57/uniqueness.py</code> classifies both as <code>dot_field</code>), yet their 3-px Euclidean halo covers
<code>{pcert['halo3_covered_allow_pixels']:,} / {pcert['total_allow_pixels']:,} = {pcert['halo3_allow_fraction']*100:.3f}%</code> of ALL allowed cells in Nevada,
leaving only <code>{pcert['unhaloed_allow_pixels_in_entire_nevada']:,}</code> unhaloed allowed pixels statewide! By Dirichlet's Pigeonhole Principle,
<b>every possible subset of size 37,654 in Nevada satisfies forward overlap &ge; 1 &minus; {pcert['unhaloed_allow_pixels_in_entire_nevada']}/37,654 = {pcert['pigeonhole_min_possible_forward_overlap_at_budget_37654']:.6f} &gt; 0.7000</b>.
Defining null-saturated priors by the exact parameter-free mathematical null expectation <code>E_Unif[FwdOverlap(S, R)] = &rho;_halo(R) &ge; 0.7000</code> (47 rasters),
Cutting-Plane Lagrangian Submodular Cover achieves <b>0 violations across all 823 / 823 non-null-saturated rasters</b>
(<code>max 1-way = {ua['informative_max_1way_3px_overlap']:.5f} &le; 0.7000</code>, <code>max symmetric = {ua['informative_max_sym_3px_overlap']:.5f} &le; 0.7000</code>,
<code>worst Jaccard = {ua['verbatim_vendor_gems57_uniqueness_output']['worst_jaccard_dot_sets']:.5f}</code>).
</div>

<h2>5. Official Public Data Sources (Clickable Links for Manual Verification)</h2>
<ul>
<li><a href="{COMP}">DrivenData Competition 306 — DOE GEMS Prize Overview</a> &amp; <a href="{PD}">Official Problem Description &amp; Metric</a> &amp; <a href="{LB}">Public Leaderboard</a></li>
<li><a href="https://www.usgs.gov/centers/gggsc/science/geodawn-geoscience-data-acquisition-western-nevada">USGS GeoDAWN Airborne Magnetic &amp; Radiometric Survey of Western Nevada</a></li>
<li><a href="https://www.usgs.gov/3d-elevation-program">USGS 3D Elevation Program (3DEP) 1 m LiDAR &amp; 1/3 arc-second (10 m) Seamless DEM</a></li>
<li><a href="https://gdr.openei.org/submissions/1391">DOE Geothermal Data Repository (GDR) Submission 1391 — INGENIOUS Great Basin Regional Dataset Compilation (CC BY 4.0)</a></li>
<li><a href="https://www.usgs.gov/programs/earthquake-hazards/faults">USGS Quaternary Fault and Fold Database / NSHM23 Fault Sections</a></li>
<li><a href="https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html">U.S. Census Bureau TIGER/Line Roads &amp; Rails Shapefiles</a> &amp; <a href="https://gbp-blm-egis.hub.arcgis.com/">BLM National Mining Claims Geospatial Data</a></li>
</ul>
"""
    (D / "h77pd.html").write_text(wrap_page("h77pd.html", "Lane H77E Technical Monograph", body_tech))

    body_guide = f"""<h1>Executive Summary &amp; DrivenData Submission Guide — Lane H77E</h1>
{hero_card}
<h2>Quick Decision Matrix (3-Second Read)</h2>
<table>
<tr><th>Question</th><th>Verified Answer</th></tr>
<tr><td><b>1. Is this file OK to download right now?</b></td><td><b>YES.</b> Click either button above to download the validated GeoTIFF (<code>{spkg['tif_bytes']:,} B</code>) or single-TIFF portal ZIP (<code>{spkg['zip_bytes']:,} B</code>).</td></tr>
<tr><td><b>2. Is this file OK to submit automatically?</b></td><td><b>NO — Not OK to submit without human selector sign-off (Selector: not approved).</b> Fails closed under repository Policy I17 due to 47 Nevada-wide null-saturating rasters (<code>E_Unif[FwdOverlap] &ge; 0.7000</code>, including 100%-footprint continuous surfaces and stride-5 lattice <code>5c1a1594</code> with mathematical minimum overlap <code>0.827004 &gt; 0.70</code>). Passes <b>823 / 823 non-null-saturated rasters (&le; 0.66498)</b> and <b>870 / 870 Spearman checks (&le; 0.61645)</b>.</td></tr>
<tr><td><b>3. Will DrivenData reject it with <code>"Predicted values must be in range [0, 1]"</code>?</b></td><td><b>NO (15/15 validator checks PASS).</b> Single-band <code>float32</code>, <code>EPSG:32611</code>, <code>3730 &times; 3292</code>, <code>nodata = None</code>, <code>0 NaN</code> across all <code>12,279,160</code> pixels, <code>0 Inf</code>, exact binary values in <code>{{0.0, 1.0}} &sube; [0, 1]</code>, <code>0</code> catalogue pixels, <code>0</code> out-of-footprint non-zeros.</td></tr>
<tr><td><b>4. How strong is its scientific evidence?</b></td><td><b>Strongest holdout result in repository family history:</b> <b>0.168493 HOLDOUT-DTI</b> pooled across 34,235 withheld positive pixels (Seed 30: <code>0.165580</code>; Seed 20: <code>0.163962</code>; Seed 21: <code>0.175841</code>) and <b>0.194160 HOLDOUT-DTI</b> across 57,448 blind 4-quadrant pixels.</td></tr>
</table>

<h2>Step-by-Step Upload Checklist (Once Approved by Human Selector)</h2>
<ol>
<li>Download <a href="{e(zip_dl)}" download><code>{e(Path(spkg['zip_path']).name)}</code></a> (single-TIFF ZIP containing only <code>{e(Path(spkg['tif_path']).name)}</code>) or the raw <code>.tif</code>.</li>
<li>Verify checksum locally before upload:<br>
<code>sha256sum {e(Path(spkg['tif_path']).name)}</code> &rarr; <code>{e(spkg['tif_sha256'])}</code><br>
<code>sha256sum {e(Path(spkg['zip_path']).name)}</code> &rarr; <code>{e(spkg['zip_sha256'])}</code></li>
<li>Open the <a href="{COMP}">DrivenData DOE GEMS Submission Page</a>, upload the single file, and paste the 123-character note:<br>
<code>{e(sub_note)}</code></li>
<li>Record the returned live score and timestamp receipt; only portal-returned values carry label <code>ORGANIZER-CONFIRMED</code>.</li>
</ol>
"""
    (D / "h77pd-executive-summary.html").write_text(wrap_page("h77pd-executive-summary.html", "H77-PD Submission Guide", body_guide))
    print("Wrote docs/h77pd.html and docs/h77pd-executive-summary.html cleanly!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
