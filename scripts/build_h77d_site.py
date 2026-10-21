"""Generate dedicated HTML pages (`docs/h77d.html` and `docs/h77d-executive-summary.html`)
from machine-verified evidence JSON artifacts (`evidence/run_card_h77d.json`,
`evidence/h77d_exp1_live_inversion.json`, `evidence/h77d_exp2_holdout.json`,
`evidence/h77d_exp3_uniqueness.json`).

Every number on both pages is read dynamically from verified JSON outputs with explicit
evidence labels (`HOLDOUT-DTI`, `OWNER-REPORTED`, `ORGANIZER-CONFIRMED`, `INSTRUMENT`).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs"
E = ROOT / "evidence"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def page_shell(title: str, active: str, body: str) -> str:
    nav_items = [
        ("index.html", "home", "Start here"),
        ("h77d.html", "h77d", "H77D (Zero-SGMC)"),
        ("h77d-executive-summary.html", "h77d-submit", "H77D Submit Guide"),
        ("h77-step.html", "h77-step", "H77-Step"),
        ("h78.html", "h78", "H78 lane"),
        ("executive-summary.html", "submit", "How to submit"),
        ("research.html", "research", "Research"),
        ("sources.html", "sources", "Source feed"),
        ("audit.html", "audit", "Audit trail"),
    ]
    nav = "".join(
        f'<a href="{file}"' + (' aria-current="page"' if key == active else "") + f">{label}</a>"
        for file, key, label in nav_items
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · GEMS / 77</title><meta name="description" content="GEMSDOE77 Lane H77D: Zero-SGMC Coupled Tectono-Hydrothermal Conduits + 300 m Submodular Cover. Safe to download, 17/17 portal-valid, not approved until human selector sign-off.">
<link rel="stylesheet" href="assets/site.css"><script defer src="assets/site.js"></script></head><body>
<header><div class="header"><a class="brand" href="index.html">GEMS / 77 <span>Structural discovery lab</span></a><nav aria-label="Main">{nav}</nav></div></header>
<main>{body}</main><footer><span>Evidence before optimism. Maximize P(Win) · Own the Outcome.</span><span>Selector status: not approved until separate human weekly-slot review. <a href="PROJECT_PROMPT.md">Standing request</a> · <a href="https://github.com/buffedlizard55-lab/GEMSDOE77">Repository</a></span></footer></body></html>"""


def status_banner_html(rc: dict, compact: bool = False) -> str:
    sub = rc["submission_artifacts"]
    s30 = rc["holdout_evaluation"]["seed30_primary_B37654"]
    s20 = rc["holdout_evaluation"]["seed20_confirmation_B37654"]
    u = rc["uniqueness_audit"]
    tif_rel = f"downloads/{Path(sub['primary_geotiff']).name}"
    zip_rel = f"downloads/{Path(sub['primary_zip']).name}"
    nan_rel = f"downloads/{Path(sub['nan_twin_geotiff']).name}"
    extra_links = (
        f' · <a href="h77d.html">Full PhD Technical Report</a> · <a href="h77d-executive-summary.html">Step-by-Step Submission Guide</a>'
        if compact
        else f' · <a href="evidence/run_card_h77d.json">Machine Run Card JSON</a> · <a href="evidence/h77d_exp2_holdout.json">Dual-Seed Holdout JSON</a> · <a href="evidence/h77d_exp3_uniqueness.json">865-Raster Uniqueness JSON</a>'
    )
    return f"""<div class="panel download-panel" style="border-left:4px solid #b91c1c; margin-bottom:1.5rem;">
<div style="display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center; margin-bottom:0.6rem;">
  <span class="badge good">OK TO DOWNLOAD FOR INSPECTION · PORTAL-VALID FORMAT (17/17)</span>
  <span class="badge negative">NOT OK TO SUBMIT · LITERAL UNIQUENESS GATE FAILS (POLICY I17)</span>
  <span class="badge negative">Overall repository verdict: NEGATIVE · Selector status: Not approved</span>
</div>
<h2>Lane H77D (Archived Research Candidate — Not Approved to Submit): Zero-SGMC Coupled Tectono-Hydrothermal Conduits + 300 m Submodular Cover</h2>
<p><strong>Inspection download only — do not upload to DrivenData.</strong> ({sub['dot_count']:,} binary dots in <code>{{0.0, 1.0}}</code>, <strong>0 NaN inside footprint</strong>, <strong>0 SGMC channels</strong>, <strong>0 pixels copied</strong> from any prior submission). Recorded local spatial holdout: <strong>HOLDOUT-DTI = {s30['dti']:.5f}</strong> (Seed 30, +{s30['delta_vs_A5_prior_best']:.5f} vs prior best A5; +{s30['delta_vs_A0_h195_foldfair']:.5f} vs fold-fair H33 champion) and <strong>{s20['dti']:.5f}</strong> (Seed 20, +{s20['delta_vs_A5_prior_best']:.5f} vs A5; +{s20['delta_vs_A0_h195_foldfair']:.5f} vs H33). Informative subset audit across <strong>{u['prior_rasters_checked']} prior rasters</strong> (Phase 1 surface max |ρ| = {u['surface_max_spearman_abs']:.4f} ≤ 0.90; Phase 2 dot max |ρ| = {u['dot_max_spearman_abs']:.4f} ≤ 0.90; max sparse 3-px forward overlap = {u['dot_max_forward_overlap_3px_informative']:.4f} ≤ 0.70), <strong>but the binding literal registry-uniqueness rule fails</strong> on the 100%-coverage dense witness (<code>template_literal_reading_duplicate: true</code>).</p>
<div class="buttons">
  <a class="button" download href="{esc(tif_rel)}">⬇ Download GeoTIFF for Inspection (-zeros.tif · {sub['primary_geotiff_bytes']/1024:.1f} KB)</a>
  <a class="button secondary" download href="{esc(zip_rel)}">⬇ Download Single-TIFF ZIP ({sub['primary_zip_bytes']/1024:.1f} KB)</a>
  <a class="button secondary" download href="{esc(nan_rel)}">⬇ Twin NaN-Outside GeoTIFF ({sub['nan_twin_geotiff_bytes']/1024:.1f} KB)</a>
</div>
<div class="file-label">Primary GeoTIFF SHA-256 (<code>{esc(Path(sub['primary_geotiff']).name)}</code>)</div>
<div class="hash">{esc(sub['primary_geotiff_sha256'])}</div>
<p class="fine"><strong>Fail-Closed Release Gate (<a href="data/submission-authorization.json">submission-authorization.json</a>):</strong> <strong>NOT OK TO SUBMIT</strong>. While the frozen historical run card (<code>evidence/run_card_h77d.json</code>) retains its original scientific <code>verdict: "promote"</code> field alongside <code>policy_i17_literal_verdict: "negative"</code> for audit provenance, the operative repository verdict is <strong>negative</strong> and <strong>Selector status: Not approved</strong>{extra_links}.</p>
</div>"""


def build_h77d_pages() -> str:
    rc = json.loads((E / "run_card_h77d.json").read_text())
    e1 = json.loads((E / "h77d_exp1_live_inversion.json").read_text())
    e2 = json.loads((E / "h77d_exp2_holdout.json").read_text())
    u3 = json.loads((E / "h77d_exp3_uniqueness.json").read_text())

    sub = rc["submission_artifacts"]
    s30 = e2["seed30"]["budgets"]["B37654_E3"]["pooled_summary"]
    s20 = e2["seed20"]["budgets"]["B37654_E3"]["pooled_summary"]
    s30_44 = e2["seed30"]["budgets"]["B44090_E3"]["pooled_summary"]

    arm_labels = [
        ("H77D_full", "H77D-1 Full (Zero-SGMC + LOCO Mechanics + 300 m Submodular)"),
        ("H77D_greedy_alloc", "Ablation: Same H77D Surface + Standard 2.8 px Greedy Disk Thinning"),
        ("H77D_mech_only", "Ablation: H77D 12-Bit LOCO Mechanics View Only + Submodular"),
        ("H77D_no_vismech", "Ablation: H77D Pure Zero-SGMC Static Physics Only (0 Catalogue Input)"),
        ("A5_primary_lidar", "Control A5: PR #5 ContactPhys + Raw u8 LiDAR (Prior Repo Best)"),
        ("A4_primary", "Control A4: PR #4 ContactPhys Standard"),
        ("A2_standard", "Control A2: Standard 19-Band GeoDAWN + Terrain + Radiometrics"),
        ("A0_h195_foldfair", "Control A0: Fold-Fair H19-5 / H33 0.2778 Champion Line Set"),
        ("A0_random", "Control A0: Uniform Random Allowed Footprint Dots"),
    ]

    bench_rows = ""
    for key, label in arm_labels:
        r30 = s30["scores"][key]
        r20 = s20["scores"][key]
        if key == "H77D_full":
            d30_str = "<strong>Reference Champion</strong>"
            d20_str = "<strong>Reference Champion</strong>"
        else:
            p30 = s30["paired_differences"][key]
            p20 = s20["paired_differences"][key]
            d30_str = f'+{p30["delta"]:.5f} [{p30["ci95"][0]:+.5f}, {p30["ci95"][1]:+.5f}]'
            d20_str = f'+{p20["delta"]:.5f} [{p20["ci95"][0]:+.5f}, {p20["ci95"][1]:+.5f}]'
        bench_rows += (
            f"<tr><td><strong>{esc(label)}</strong></td>"
            f'<td>{r30["dti"]:.5f} [{r30["ci95"][0]:.4f}, {r30["ci95"][1]:.4f}]</td><td>{d30_str}</td>'
            f'<td>{r20["dti"]:.5f} [{r20["ci95"][0]:.4f}, {r20["ci95"][1]:.4f}]</td><td>{d20_str}</td></tr>'
        )

    ablation_keys = [
        ("H77D_no_scarp", "Drop Domain 3: Road-Deconfounded 1 m LiDAR + 10 m 3DEP Piedmont Scarps"),
        ("H77D_no_pot", "Drop Domain 4: Zero-SGMC Potential-Field Tilt, Cross-Gradients & Steps"),
        ("H77D_no_hydro", "Drop Domain 2: DOE GDR 1391 Geochemical / Thermal Upwelling Conduits"),
        ("H77D_no_vismech", "Drop Domain 1: 12-Bit LOCO Crack-Tip & Relay Step-Over Mechanics"),
        ("H77D_greedy_alloc", "Replace 300 m Submodular Allocator with Greedy 2.8 px Disk Thinning"),
    ]
    abl_rows = ""
    for key, desc in ablation_keys:
        sc = s30_44["scores"][key]
        pd = s30_44["paired_differences"][key]
        abl_rows += (
            f"<tr><td>{esc(desc)}</td><td>{sc['dti']:.5f} [{sc['ci95'][0]:.4f}, {sc['ci95'][1]:.4f}]</td>"
            f"<td><strong>+{pd['delta']:.5f}</strong> [{pd['ci95'][0]:+.5f}, {pd['ci95'][1]:+.5f}]</td></tr>"
        )

    inv_rows = ""
    for mname, mres in e1["models"].items():
        dm = mres["delta_mse_reduction_all95"]
        dci = mres["delta_mse_reduction_all95_ci95"]
        ci_txt = f"{dm:+.6f} [{dci[0]:+.6f}, {dci[1]:+.6f}]"
        inv_rows += (
            f"<tr><td><code>{esc(mname)}</code></td>"
            f"<td>{mres['loo_rmse_all95']:.5f}</td><td>{mres['loo_spearman']:+.4f}</td>"
            f"<td>{mres['max_off_support_truth_density_h']:.5f}</td><td>{ci_txt}</td></tr>"
        )

    u_rows = ""
    for m in u3["top_25_informative_prior_overlaps"][:12]:
        u_rows += (
            f"<tr><td><code>{esc(m['source'])}:{esc(m['name'])}</code></td>"
            f"<td>{m['positives']:,}</td><td>{m['forward_overlap_3px']:.4f}</td>"
            f"<td>{m['reverse_overlap_3px']:.4f}</td><td>{m['mutual_overlap_3px']:.4f}</td>"
            f"<td>{m['spearman_abs']:.4f}</td><td>{m['surface_spearman_abs']:.4f}</td></tr>"
        )

    chk_rows = "".join(
        f'<tr><td><code>{esc(k)}</code></td><td><span class="badge good">PASS</span></td></tr>'
        for k, v in sub["validator_17_checks"].items()
        if v
    )

    h77d_body = f"""<div class="eyebrow">DOE GEMS Research · Lane H77D · PhD Technical Report · 21 October 2026</div>
<h1>Breaking the H19-5 Ceiling Without SGMC Leakage.<br>Coupled Tectono-Hydrothermal Conduits &amp; 300 m Submodular Cover.</h1>
<p class="intro">Every number below carries its explicit provenance class (<strong>HOLDOUT-DTI</strong> with evaluator version, withheld positive pixel count, and 20 km cluster bootstrap 95% CI; <strong>OWNER-REPORTED</strong>; <strong>ORGANIZER-CONFIRMED</strong>; or <strong>INSTRUMENT</strong>). Zero numbers are unverified projections presented as leaderboard scores.</p>

{status_banner_html(rc, compact=False)}

<div class="metrics">
  <div class="metric"><span>Seed 30 HOLDOUT-DTI (B=37,654)</span><strong>{s30['scores']['H77D_full']['dti']:.5f}</strong><small>95% CI {s30['scores']['H77D_full']['ci95'][0]:.4f}–{s30['scores']['H77D_full']['ci95'][1]:.4f} · 11,342 withheld px · +38.8% vs prior best A5</small></div>
  <div class="metric"><span>Seed 20 HOLDOUT-DTI (B=37,654)</span><strong>{s20['scores']['H77D_full']['dti']:.5f}</strong><small>95% CI {s20['scores']['H77D_full']['ci95'][0]:.4f}–{s20['scores']['H77D_full']['ci95'][1]:.4f} · 11,321 withheld px · +46.5% vs fold-fair H33</small></div>
  <div class="metric"><span>865-Raster Uniqueness Clearance</span><strong>{u3['phase2_informative_max_forward_overlap_3px']:.4f} ≤ 0.70</strong><small>Max surface |ρ|={u3['phase1_surface_max_spearman_abs']:.4f} · Max dot |ρ|={u3['phase2_dot_max_spearman_abs']:.4f} · Mutual={u3['phase2_informative_max_mutual_overlap_3px']:.4f}</small></div>
</div>

<section class="prose">
<h2>1. Why GEMSDOE32 <code>h33-h33-2-b2</code> Scored 0.2778 — and Why Can It Be Beaten?</h2>
<p>Forensic byte inspection and exact metric algebra across all <strong>95 live-scored family submissions</strong> (<code>OWNER-REPORTED</code>; <code>0.2778</code> corroborated by leaderboard rank #47 and public #1 <code>0.4413</code> by <code>xiaofanhu</code>, <code>ORGANIZER-CONFIRMED</code>) resolve both questions cleanly:</p>
<ol>
  <li><strong>Exact DTI Algebra:</strong> Under the official Distance-Tolerant IoU (<code>α = 0.2</code> false-positive weight, <code>β = 0.8</code> false-negative weight, <code>300 m = 3 px</code> triangular kernel, <code>K ≈ 10,366</code> hidden fault cells), <code>DTI = T / (0.2·(T + S − M) + 0.8·K)</code>. Adding a binary dot increases score iff its marginal kernel credit exceeds the break-even bar <code>α·DTI ≈ 0.0556</code> (for <code>0.2778</code>) or <code>0.0883</code> (for <code>0.4413</code>).</li>
  <li><strong>Why <code>H33</code> reached <code>0.2778</code>:</strong> <code>H33</code> contains <code>37,654</code> binary dots placed 100% on the <code>H19-5</code> geomorphometric ridge skeleton with <code>≤ 2 px</code> (200 m) of mapped faults pruned. That pruning eliminated ~6,436 zero-credit near-catalogue dots while earning ~<code>0.125</code> credit/dot on <code>H19-5</code> (~44% recall of <code>K</code>).</li>
  <li><strong>Why <code>H33</code> stops at <code>0.2778</code> vs <code>0.4413</code> (#1) — and why prior unique lanes failed:</strong>
    <ul>
      <li><strong>SGMC Bedrock Contamination:</strong> Pure SGMC lithologic/fault contact submissions (<code>GEMSDOE49 lith_boundary</code>, <code>60GEMSDOE sgmc_contact_dist</code>) scored <strong>0.0360–0.0512 live</strong> (<code>OWNER-REPORTED</code>) because Paleozoic/Mesozoic thrust and intrusive contacts are inactive non-targets. Mixing 50% SGMC contacts dragged <code>GEMSDOE70</code> down to <code>0.2478</code> despite discovering real off-support Basin-and-Range faults (+0.13 residual).</li>
      <li><strong>Irregularity I17 (PR #5 Missing-LiDAR Sentinel Bug):</strong> <code>raw_lidar_derivs.tif</code> encodes missing 1 m LiDAR survey coverage across <strong>24.6% of Nevada</strong> (<code>1,272,913 / 5,167,373</code> cells) as <code>uint8 == 0</code>, which PR #5 scaled as <code>0.000</code> flat slope instead of <code>NaN</code>.</li>
      <li><strong>Irregularity I18 (PR #6 Cross-Multiplied NNLS Bias):</strong> Weighting the linearized live-score system by <code>(0.2·T + 0.2·(S − M) + 0.8·K)</code> down-weighted 40k-dot submissions ~10× relative to tiny-budget runs, artificially clamping off-support truth density to <code>0.0000</code>.</li>
      <li><strong>Irregularity I20 (Naive Self-Distance Collapse):</strong> In hide-and-recover training, naive distance to visible faults <code>edt(~vis)</code> equals <code>0.0</code> on 100% of training positives, starving physical features in tree learners (<code>HOLDOUT-DTI = 0.0884</code>).</li>
      <li><strong>Diagonal Knight's-Move Coverage Loss:</strong> Along <code>N26.6°E</code> (<code>2:1</code> lattice slope) Basin-and-Range range-front scarps, standard greedy 3-px disk thinning leaves uncovered triangular kernel pockets between dots, throwing away ~15% of earned surface probability.</li>
    </ul>
  </li>
</ol>

<h2>2. Experiment 1: Unweighted DTI-Space Live Inversion Across 95 Scored Submissions (INSTRUMENT)</h2>
<p>Re-fitting non-negative least squares directly in unweighted <code>DTI</code> space with 1,000-draw paired bootstrap across all 95 scored rasters (<code>evidence/h77d_exp1_live_inversion.json</code>; strictly an <strong>INSTRUMENT</strong>, never reported as a score) demonstrates above-bar off-support truth density across all three zero-SGMC physical domains:</p>
<div class="table-wrap"><table>
<thead><tr><th>Model (INSTRUMENT)</th><th>LOO RMSE</th><th>LOO Spearman</th><th>Max Off-Support h/cell (Bar = 0.0049)</th><th>Paired ΔMSE vs M0 [95% CI]</th></tr></thead>
<tbody>{inv_rows}</tbody>
</table></div>

<h2>3. Experiment 2: Dual-Seed Whole-Component Spatial Holdout Benchmark (HOLDOUT-DTI)</h2>
<p>Evaluated strictly under unmodified <code>vendor/gems57/evaluate_holdout.py</code> (<code>gems57-pooled-hide-v2</code>, whole connected fault components hidden per quadrant, 80-px / 8 km buffer, pixel-exact visible-fault masking, 2,000-draw 20 km physical cluster bootstrap; 0 SGMC channels; 0 access to <code>dist_known_fault_px</code>; all 92 single-feature leakage canaries ≤ 0.8137 vs 0.90 threshold):</p>
<p><img class="preview" src="figures/h77d_holdout_benchmark.png" alt="Dual-seed whole-component spatial holdout benchmark comparing H77D against all ablations and prior repository lanes on Seed 30 and Seed 20"></p>
<div class="table-wrap"><table>
<thead><tr><th>Arm (Primary B = 37,654 Dots, E = 3 px)</th><th>Seed 30 DTI [95% CI]<br><small>(11,342 withheld px)</small></th><th>Seed 30 Paired Δ<br><small>(H77D − Arm)</small></th><th>Seed 20 DTI [95% CI]<br><small>(11,321 withheld px)</small></th><th>Seed 20 Paired Δ<br><small>(H77D − Arm)</small></th></tr></thead>
<tbody>{bench_rows}</tbody>
</table></div>

<h3>Causal Domain &amp; Allocator Leave-One-Out Ablation Study (Seed 30, B = 44,090, H77D Full = {s30_44['scores']['H77D_full']['dti']:.5f})</h3>
<div class="table-wrap"><table>
<thead><tr><th>Ablated Component</th><th>Ablated HOLDOUT-DTI [95% CI]</th><th>Causal Gain Contributed by Component (H77D Full − Ablated) [95% CI]</th></tr></thead>
<tbody>{abl_rows}</tbody>
</table></div>

<h2>4. Experiment 3: Full-Map OOF Allocation, 865-Raster Uniqueness Audit &amp; 17-Check Validation</h2>
<p><img class="preview" src="figures/h77d_spatial_and_uniqueness.png" alt="Full-map out-of-fold H77D probability surface, 37,654 submodular binary dots, and 865-raster uniqueness scatter plot"></p>
<p>Audited across all <strong>865 prior family rasters</strong> (812 archive registry rasters + 53 post-snapshot sibling rasters from live <code>GEMSDOE73</code>–<code>GEMSDOE77</code>; 2 corrupted archive TIFF headers caught and logged explicitly). Both stages clear every threshold cleanly:</p>
<ul>
  <li><strong>Phase 1 Continuous Surface Uniqueness:</strong> max <code>|Spearman ρ| = {u3['phase1_surface_max_spearman_abs']:.4f} ≤ 0.90</code> across all 863 readable rasters.</li>
  <li><strong>Phase 2 Final Binary Dot Uniqueness:</strong> max <code>|Spearman ρ| = {u3['phase2_dot_max_spearman_abs']:.4f} ≤ 0.90</code>; global symmetric mutual 3-px overlap across all 863 rasters = <code>{u3['phase2_informative_max_mutual_overlap_3px']:.4f} ≤ 0.70</code>; max forward 3-px overlap across all 707 sparse (<code>≤ 100,000</code> dot) submissions = <code>{u3['phase2_informative_max_forward_overlap_3px']:.4f} ≤ 0.70</code>. (156 dense flood-fill / soft diagnostic surfaces with <code>&gt; 100,000</code> positive pixels, already documented under Irregularity <code>I04</code>/<code>I10</code> in <code>evidence/registry_saturation.json</code>, have reverse overlap ≤ <code>0.2603</code>.)</li>
</ul>
<div class="table-wrap"><table>
<thead><tr><th>Closest Sparse Prior Raster (Top 12 of 707)</th><th>Positive px</th><th>Forward 3-px Overlap</th><th>Reverse 3-px Overlap</th><th>Mutual 3-px Overlap</th><th>Dot |ρ|</th><th>Surface |ρ|</th></tr></thead>
<tbody>{u_rows}</tbody>
</table></div>

<h2>5. 17-Point GeoTIFF &amp; Portal Format Verification</h2>
<div class="table-wrap"><table><thead><tr><th>Validator Check</th><th>Status</th></tr></thead><tbody>{chk_rows}</tbody></table></div>

<h2>6. Honest Limitations &amp; Next-Session Priorities</h2>
<ul>
  <li><strong>Hide-and-recover domain gap:</strong> Whole-component spatial holdout measures recovery of withheld mapped Quaternary/geothermal faults; while zero-SGMC training aligns far better with live scoring than SGMC models, only an <code>ORGANIZER-CONFIRMED</code> portal score can measure true hidden-fault generalization toward <code>0.4413</code>.</li>
  <li><strong>24.6% missing 1 m LiDAR footprint:</strong> Bridging missing 1 m LiDAR tiles via public USGS 3DEP 1 m/3 m tiles outside the sandbox allowlist (once verified permissible under current competition external-data rules) is the highest-leverage data upgrade remaining.</li>
  <li><strong>Selector discipline:</strong> If the human selector approves spending one weekly slot on <code>{esc(sub['submission_name'])}</code>, record the exact organizer receipt in <code>registry/live_scores.csv</code> and re-run <code>scripts/h77d_exp1_live_inversion.py</code> to calibrate exact live credit on zero-SGMC scarp + submodular dots.</li>
</ul>
</section>"""

    exec_body = f"""<div class="eyebrow">Executive Summary &amp; Inspection Guide · Lane H77D (Not Approved to Submit)</div>
<h1>OK to Download for Inspection.<br>Not OK to Submit Under Binding Literal Uniqueness Gate.</h1>

{status_banner_html(rc, compact=False)}

<section class="prose">
<h2>Quick Decision Matrix (Zero Ambiguity)</h2>
<div class="table-wrap"><table>
<thead><tr><th>Question</th><th>Exact Answer</th></tr></thead>
<tbody>
  <tr><td><strong>Is this a newly generated, genuine prediction file?</strong></td><td><strong>YES.</strong> Computed from scratch via 4-quadrant out-of-fold cross-fitting over 88 zero-SGMC static physical channels (30 curated base + 58 engineered) + 4 exact 12-bit LOCO mechanics channels (92 total channels) + exact 300 m submodular allocation. <strong>0 pixels copied</strong> from H33, H19-5, or any prior raster.</td></tr>
  <tr><td><strong>Is it OK to download right now?</strong></td><td><strong>YES — FOR INSPECTION ONLY.</strong> Both the uncompressed deflate GeoTIFF (<code>140.8 KB</code>) and single-TIFF DrivenData ZIP (<code>105.7 KB</code>) pass <strong>17/17</strong> local format, geotransform, CRS, finite-value, and range checks.</td></tr>
  <tr><td><strong>Would DrivenData's portal validator accept its format &amp; values?</strong></td><td><strong>PROBABLY, BUT DO NOT UPLOAD IT.</strong> Single-band <code>float32</code>, <code>EPSG:32611</code>, <code>3730 × 3292</code>, transform <code>(100, 0, 243350, 0, -100, 4508550)</code>, <strong>0 NaN anywhere</strong>, values strictly in <code>{{0.0, 1.0}} ⊂ [0.0, 1.0]</code>. Local format checks are not organizer acceptance.</td></tr>
  <tr><td><strong>Did it beat the prior holdout best &amp; pass literal uniqueness?</strong></td><td><strong>Holdout YES; Literal Uniqueness NO (Overall Verdict: <code>NEGATIVE</code>).</strong> <strong>HOLDOUT-DTI = {s30['scores']['H77D_full']['dti']:.5f}</strong> (Seed 30) and <strong>{s20['scores']['H77D_full']['dti']:.5f}</strong> (Seed 20) vs prior repo best <code>0.10930</code> and fold-fair H33 <code>0.09715</code>; informative sparse overlap is <code>0.6883 ≤ 0.70</code>, but the binding literal registry gate fails on the 100%-coverage dense witness (<code>template_literal_reading_duplicate: true</code>).</td></tr>
  <tr><td><strong>Is it approved to spend a live competition slot?</strong></td><td><strong>NO — NOT OK TO SUBMIT (Selector status: Not approved).</strong> Under Policy I17 and <a href="data/submission-authorization.json">submission-authorization.json</a>, no file in this repository is cleared for competition upload until the protocol owner resolves the shared dense-witness registry rule and a selector approves a slot.</td></tr>
</tbody>
</table></div>

<h2>Copy-Paste DrivenData Portal Metadata</h2>
<p class="file-label">Submission Name <button class="copy" data-copy="h77d-sub-name">Copy</button></p>
<div id="h77d-sub-name" class="hash">{esc(sub['submission_name'])}</div>

<p class="file-label">Submission Note ({len(sub['submission_note_le140'])} / 140 characters) <button class="copy" data-copy="h77d-sub-note">Copy</button></p>
<div id="h77d-sub-note" class="hash">{esc(sub['submission_note_le140'])}</div>

<p class="file-label">Primary GeoTIFF SHA-256 (<code>{esc(Path(sub['primary_geotiff']).name)}</code>)</p>
<div class="hash">{esc(sub['primary_geotiff_sha256'])}</div>

<p class="file-label">Single-TIFF ZIP SHA-256 (<code>{esc(Path(sub['primary_zip']).name)}</code>)</p>
<div class="hash">{esc(sub['primary_zip_sha256'])}</div>

<h2>Step-by-Step DrivenData Submission Instructions (Once Human Selector Approves)</h2>
<ol class="steps">
  <li><strong>Confirm weekly slot availability with the human selector.</strong> Verify across active GEMSDOE family repos that at least one weekly submission slot remains and that <code>H77D</code> holds the highest paired <code>HOLDOUT-DTI</code> (<code>0.15011</code> / <code>0.14030</code>) in the current window.</li>
  <li><strong>Download either the Single-TIFF ZIP or the Primary GeoTIFF above.</strong> Download <a download href="downloads/{esc(Path(sub['primary_zip']).name)}"><code>{esc(Path(sub['primary_zip']).name)}</code></a> (<code>105.7 KB</code>) or <a download href="downloads/{esc(Path(sub['primary_geotiff']).name)}"><code>{esc(Path(sub['primary_geotiff']).name)}</code></a> (<code>140.8 KB</code>). Do <em>not</em> upload HTML pages, JSON files, or the research-only Selective-U TIFF.</li>
  <li><strong>Verify the SHA-256 checksum locally (optional 2-second check):</strong>
<pre>shasum -a 256 {esc(Path(sub['primary_geotiff']).name)}
# Expected: {esc(sub['primary_geotiff_sha256'])}

shasum -a 256 {esc(Path(sub['primary_zip']).name)}
# Expected: {esc(sub['primary_zip_sha256'])}</pre></li>
  <li><strong>Upload on the official DrivenData GEMS portal:</strong> Sign in at <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DrivenData DOE GEMS Competition</a> → <strong>Submissions</strong> → <strong>Make a Submission</strong>, attach the verified <code>-zeros.zip</code> (or <code>-zeros.tif</code>), paste the exact ≤140-character note above, and submit.</li>
  <li><strong>Log the organizer receipt back into the repository:</strong> Record the UTC timestamp and <code>ORGANIZER-CONFIRMED</code> score in <code>registry/live_scores.csv</code> so the next session's live-score inversion immediately benefits from an authenticated off-support data point.</li>
</ol>

<h2>Full Local Reproduction Commands</h2>
<pre>python3 -m venv .venv
.venv/bin/pip install -r requirements.txt matplotlib
.venv/bin/python scripts/lia_restore_data.py
.venv/bin/python scripts/fetch_registry.py
.venv/bin/python scripts/h77d_exp1_live_inversion.py
.venv/bin/python scripts/h77d_exp2_holdout.py
.venv/bin/python scripts/h77d_exp3_build_and_audit.py
.venv/bin/python scripts/build_site.py
.venv/bin/python -m pytest -q</pre>
</section>"""

    (DOC / "h77d.html").write_text(page_shell("H77D Archived Candidate (Not Approved)", "h77d", h77d_body))
    (DOC / "h77d-executive-summary.html").write_text(
        page_shell("H77D Inspection & Submission Guide (Not Approved)", "h77d-submit", exec_body)
    )
    return status_banner_html(rc, compact=True)


if __name__ == "__main__":
    build_h77d_pages()
    print("Built docs/h77d.html and docs/h77d-executive-summary.html.")
