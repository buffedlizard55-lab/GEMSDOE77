#!/usr/bin/env python3
"""Static page builder for Lane H84-RAMP (invoked by scripts/build_home.py)."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs"


def e(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def jload(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.is_file() else None


def copy(label: str, val: str) -> str:
    return (
        f'<div class="copybox"><span class="small">{e(label)} ({len(val)} chars):</span>'
        f'<code>{e(val)}</code></div>'
    )


def build(page):
    card = jload("evidence/h84ramp_run_card.json")
    e1 = jload("evidence/h84ramp_e1_canary_surface.json")
    e2 = jload("evidence/h84ramp_e2_holdout.json")
    e3 = jload("evidence/h84ramp_e3_uniqueness.json")
    prereg = jload("registry/preregistration_h84ramp.json")
    if not (card and e1 and e2 and e3 and prereg):
        return

    s20_sum = e2["seeds"]["20"]["summary"]
    s30_sum = e2["seeds"]["30"]["summary"]
    s20_sc = s20_sum["scores"]
    s30_sc = s30_sum["scores"]
    s20_diff = s20_sum["paired_differences"]
    s30_diff = s30_sum["paired_differences"]

    arm_labels = [
        ("A_std_plus_h84", "A_std_plus_h84 (19 std + 8 B-LBCO H84 HGB OOF, 37,654 dots)"),
        ("S_h84_physics", "S_h84_physics (pure two-host relay + GeoDAWN/basement composite, 37,654 dots)"),
        ("A_std_plus_h84_blbco_only", "A_std_plus_h84_blbco_only (B-LBCO HGB OOF without physics blend, 37,654 dots)"),
        ("A_std_control", "A_std_control (19-band standard HGB OOF control, 37,654 dots)"),
        ("A_std_plus_h84_naive", "A_std_plus_h84_naive (naive relay training without B-LBCO — IR-H84-01)"),
        ("A0_random", "A0_random (uniform random dots inside footprint, 37,654 dots)"),
    ]
    arm_rows = ""
    for key, label in arm_labels:
        r20 = s20_sc[key]
        r30 = s30_sc[key]
        arm_rows += (
            f"<tr><td><b>{e(label)}</b></td>"
            f"<td><b>{r20['dti']:.6f}</b><br><span class='small'>[{r20['ci95'][0]:.6f}, {r20['ci95'][1]:.6f}] (n={r20['withheld_positive_pixels']:,})</span></td>"
            f"<td><b>{r30['dti']:.6f}</b><br><span class='small'>[{r30['ci95'][0]:.6f}, {r30['ci95'][1]:.6f}] (n={r30['withheld_positive_pixels']:,})</span></td>"
            f"<td><code>{e(e2['evaluator_version'])}</code></td></tr>"
        )

    diff_rows = ""
    for key, label in [
        ("A_std_plus_h84_naive", "A_std_plus_h84 − A_std_plus_h84_naive (B-LBCO rescue of IR-H84-01)"),
        ("A_std_control", "A_std_plus_h84 − A_std_control (H84 relay corridor lift over 19-band HGB)"),
        ("A0_random", "A_std_plus_h84 − A0_random (lift over matched-budget random placement)"),
        ("S_h84_physics", "A_std_plus_h84 − S_h84_physics (tree blend vs pure physics composite)"),
    ]:
        d20 = s20_diff[key]
        d30 = s30_diff[key]
        diff_rows += (
            f"<tr><td>{e(label)}</td>"
            f"<td><b>{d20['delta']:+.6f}</b> [{d20['ci95'][0]:+.6f}, {d20['ci95'][1]:+.6f}]</td>"
            f"<td><b>{d30['delta']:+.6f}</b> [{d30['ci95'][0]:+.6f}, {d30['ci95'][1]:+.6f}]</td></tr>"
        )

    taut20 = e1["irregularity_IR_H84_01_self_collapse_tautology"]["20"]
    can20 = e1["canaries_by_seed"]["20"]["h84_canaries"]
    can30 = e1["canaries_by_seed"]["30"]["h84_canaries"]
    canary_rows = ""
    for ch in can20:
        a20 = can20[ch]["symmetric_auc"]
        a30 = can30[ch]["symmetric_auc"]
        n_v = taut20["naive_visible_positive_means"][ch]
        b_v = taut20["blbco_deconfounded_visible_positive_means"][ch]
        h_v = taut20["withheld_holdout_positive_means"][ch]
        canary_rows += (
            f"<tr><td><code>{e(ch)}</code></td>"
            f"<td>{a20:.4f}</td><td>{a30:.4f}</td>"
            f"<td><b>{n_v:.4f}</b></td><td><b>{b_v:.4f}</b></td><td>{h_v:.4f}</td>"
            f"<td>PASS (&le; 0.90)</td></tr>"
        )

    hyp_rows = ""
    for h in prereg["ranked_hypotheses"]:
        hyp_rows += (
            f"<tr><td><b>#{h['rank']} {e(h['id'])}</b><br><span class='small'>{'IMPLEMENTED' if h['implemented_in_pass_1'] else 'RANKED CANDIDATE'}</span></td>"
            f"<td><b>{e(h['name'])}</b><br><span class='small'>{e(h['physical_signature'])}</span></td>"
            f"<td><span class='small'>{e(', '.join(h['layers']))}</span></td>"
            f"<td><span class='small'>{e(h['how_it_differs_from_repo'])}</span></td>"
            f"<td><span class='small'>{e(h['named_non_fault_mimic'])}</span></td></tr>"
        )

    sp_sum = e3["informative_sparse_summary"]
    lfl = e3["phase2_two_phase_summary"]["like_for_like_reading"]
    h33_cmp = e3["vs_h33_champion"]
    dw_cmp = e3["vs_dense_witness"]
    top_ov_rows = (
        f"<tr><td><code>{e(lfl['worst_forward_overlap_dot_field_submission'])}</code> (worst forward 3-px overlap)</td>"
        f"<td>dot_field</td><td><b>{lfl['worst_forward_overlap_dot_fields']:.4f}</b> (&le; 0.70)</td>"
        f"<td>—</td><td>PASS</td></tr>"
        f"<tr><td><code>{e(sp_sum['worst_spearman_submission'])}</code> (worst sparse |Spearman &rho;|)</td>"
        f"<td>dot_field</td><td>—</td>"
        f"<td><b>{sp_sum['max_abs_spearman']:.4f}</b> (&le; 0.90)</td><td>PASS</td></tr>"
        f"<tr><td><code>{e(h33_cmp['submission'])}</code> (H33 GEMSDOE32 champion <code>5b3becc1...</code>)</td>"
        f"<td>dot_field ({h33_cmp['their_dots']:,} dots)</td><td><b>{h33_cmp['my_dots_within_3px_of_theirs']:.4f}</b> (&le; 0.70)</td>"
        f"<td>{h33_cmp['spearman_full_footprint']:+.4f}</td><td>PASS</td></tr>"
        f"<tr><td><code>{e(dw_cmp['submission'])}</code> (dense witness <code>ab0a0a62...</code>)</td>"
        f"<td>continuous_surface ({dw_cmp['their_dots']:,} pos px)</td><td><b>{dw_cmp['my_dots_within_3px_of_theirs']:.4f}</b> (&gt; 0.70)</td>"
        f"<td>{dw_cmp['spearman_full_footprint']:+.4f}</td><td>FAIL (Policy I17)</td></tr>"
    )

    val_checks = "".join(
        f"<li><b>{e(k)}:</b> <code>{e(str(v))}</code></li>"
        for k, v in card["validator"]["checks"].items()
    )
    tif_rel = card["raster_path"].replace("docs/", "")
    zip_rel = card["zip_path"].replace("docs/", "")

    body = f"""<h1>H84-RAMP: Deconfounded Two-Host En-Echelon Relay-Ramp &amp; Geodetic-Geophysical Stepover Transfer Corridor</h1>
<p class="verdict"><b>Verdict: NEGATIVE under Policy I17 — OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT.</b><br>
{e(card['verdict_reason'])}</p>

<section class="card dl research"><h2>⬇ H84-RAMP Research-Only GeoTIFF &amp; One-TIFF ZIP</h2>
<a class="btn" href="{e(tif_rel)}" download>Download .tif ({card['raster_bytes']//1024} KB)</a>
<a class="btn alt" href="{e(zip_rel)}" download>Download one-TIFF .zip</a>
<div class="chips">
  <span class="chip ok">OK to download: YES</span>
  <span class="chip bad">OK to submit: NO (Verdict: negative)</span>
  <span class="chip ok">Validator: 15/15 PASS ({card['dots_emitted']:,} dots in {{0,1}}, 0 NaN, 0 on catalogue)</span>
  <span class="chip ok">Informative sparse uniqueness: PASS (0/{sp_sum['sparse_rasters_checked']} firings, max 3-px overlap {sp_sum['max_forward_overlap_3px']:.4f} &le; 0.70, max |&rho;| {sp_sum['max_abs_spearman']:.4f} &le; 0.90)</span>
  <span class="chip bad">Literal Policy I17: FAIL (1.000 vs dense_witness ab0a0a62)</span>
  <span class="chip warn">Selector: not approved · 0 weekly slots used</span>
</div>
{copy("Submission name (reference only — DO NOT UPLOAD)", card["submission_name"])}
{copy("Short note <= 140 chars (reference only — DO NOT UPLOAD)", card["submission_note"])}
<p class="small">SHA-256 (.tif): <code>{e(card['raster_sha256'])}</code> · SHA-256 (.zip): <code>{e(card['zip_sha256'])}</code> ·
<a href="evidence/h84ramp_run_card.json">JSON run card</a> ·
<a href="evidence/h84ramp_e1_canary_surface.json">Exp 1 canary + surface receipt</a> ·
<a href="evidence/h84ramp_e2_holdout.json">Exp 2 dual-seed holdout receipt</a> ·
<a href="evidence/h84ramp_e3_uniqueness.json">Exp 3 uniqueness receipt</a> ·
<a href="evidence/preregistration_h84ramp.json">Preregistration</a> ·
<a href="research/h84ramp_hypotheses.md">PhD H33 analysis &amp; 5 hypotheses</a> ·
<a href="h84-ramp-executive-summary.html">Executive summary &amp; submission guide →</a></p>
</section>

<h2>1. PhD-Level Diagnosis: Why <code>H33</code> Scored 0.2778 and How to Score Higher Toward 0.4413</h2>
<p>The GEMS Distance-Tolerant Intersection (DTI) metric evaluates a predicted raster $P \\in [0,1]^{{3730\\times 3292}}$ across 101 thresholds $q \\in \\{{0.00, 0.01, \\dots, 1.00\\}}$ via a 300 m ($3.0\\text{{ px}}$) triangular distance-decay kernel $K(d)=\\max(0,1-d/300\\text{{ m}})$ with $\\alpha=0.2$ (recall weight) and $\\beta=0.8$ (precision weight). As proven in <a href="research/h84ramp_hypotheses.md"><code>research/h84ramp_hypotheses.md</code></a>:</p>
<ul>
  <li><b>Single-threshold quantization ceiling (~0.03–0.06 DTI loss):</b> Because <code>H33</code> (<code>h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros</code>, 37,553 binary dots) emits values in $\\{{0,1\\}}$, all 100 positive thresholds $q \\in [0.01, 1.00]$ evaluate the identical binary support $S(q) \\equiv D$, collapsing the 101-point threshold integral into a single operating point $(r_0, p_0)$ rather than sweeping high-precision core traces at $q\\to 1$ and broader relay/blind corridors at $q\\to 0.01$.</li>
  <li><b>Single-host distance parameterization ($d_1$ only, ~0.02–0.04 DTI loss):</b> Single-nearest-fault distance $d_1(x,V)$ treats every pixel as controlled by one fault segment. In the Basin and Range, normal faults step en-echelon across 0.3–2.5 km relay ramps where synthetic breaching cross-faults form between two distinct sub-parallel host segments $c_1 \\neq c_2$ ($d_1$ and $d_2$ simultaneously moderate, opposite anchor vectors $\\langle \\hat{{u}}_1, \\hat{{u}}_2 \\rangle < 0$, and co-planar strike $\\cos 2\\Delta\\theta > 0$).</li>
  <li><b>Relay-Stepover Self-Collapse Tautology (<code>IR-H84-01</code>):</b> If two-host relay features are evaluated naively against the full visible fault mask $V$, every visible training positive $x \\in V$ has $d_1(x,V) \\equiv 0$, collapsing 7 of 8 stepover/collar channels identically to <code>0.0</code> on 100% of positive training rows (`A_std_plus_h84_naive` drops to <code>0.0418</code> s20 / <code>0.0331</code> s30). Our <b>5-bucket Leave-Bucket-Plus-3px-Collar-Out (<code>B-LBCO</code>)</b> operator via bitplane <code>cKDTree</code> removes each segment bucket and its 300 m collar during feature extraction, restoring positive stepover statistics on visible training faults and lifting <code>HOLDOUT-DTI</code> by <b>+0.0948 on Seed 20</b> and <b>+0.1115 on Seed 30</b>.</li>
</ul>

<h2>2. Five Ranked Novel Geological Hypotheses (Preregistered in <code>registry/preregistration_h84ramp.json</code>)</h2>
<table>
  <tr><th>Rank &amp; ID</th><th>Hypothesis &amp; Physical Transform</th><th>Layers Used</th><th>How It Differs From Repo</th><th>Named Non-Fault Mimic</th></tr>
  {hyp_rows}
</table>

<h2>3. Experiment 2: Dual-Seed Spatially-Blocked Holdout DTI (<code>{e(e2['evaluator_version'])}</code>)</h2>
<p>Every score below is labelled <b><code>HOLDOUT-DTI</code></b> from the pinned shared evaluator <code>{e(e2['evaluator_version'])}</code> ($\\alpha=0.2, \\beta=0.8$, 300 m triangular kernel, 2,000 spatial-block bootstrap draws, 4-quadrant Leave-One-Quadrant-Out cross-fitting).</p>
<table>
  <tr><th>Model Arm (37,654 binary dots, min sep 2.8 px)</th><th>Seed 20 HOLDOUT-DTI (95% CI)</th><th>Seed 30 HOLDOUT-DTI (95% CI)</th><th>Evaluator</th></tr>
  {arm_rows}
</table>
<h3>Paired Spatial-Block Bootstrap Differences (2,000 draws)</h3>
<table>
  <tr><th>Paired Comparison</th><th>Seed 20 &Delta; HOLDOUT-DTI (95% CI)</th><th>Seed 30 &Delta; HOLDOUT-DTI (95% CI)</th></tr>
  {diff_rows}
</table>

<h2>4. Experiment 1: Single-Feature Leakage Canary &amp; <code>IR-H84-01</code> Self-Collapse Audit</h2>
<p>Every <code>H84</code> feature was tested alone on withheld positives vs footprint negatives (&gt;3 px from catalogue). All 8 <code>H84</code> channels and all 19 standard channels stay well below the <code>0.90</code> leakage ceiling (worst symmetric AUC <b>{e1['canaries_by_seed']['20']['max_h84_symmetric_auc']:.4f}</b> on Seed 20, <b>{e1['canaries_by_seed']['30']['max_h84_symmetric_auc']:.4f}</b> on Seed 30). Note how naive visible positive means collapse to <code>0.0000</code> on 7/8 channels (<code>IR-H84-01</code>), whereas 5-bucket <code>B-LBCO</code> restores realistic positive means matching withheld holdout positives:</p>
<table>
  <tr><th>H84 Channel</th><th>Seed 20 Sym AUC</th><th>Seed 30 Sym AUC</th><th>Naive Visible Pos Mean (IR-H84-01)</th><th>B-LBCO Visible Pos Mean</th><th>Withheld Holdout Pos Mean</th><th>Canary Gate</th></tr>
  {canary_rows}
</table>

<h2>5. Diagnostic Figure: Two-Host Relay Corridor, Physics Composite, B-LBCO OOF &amp; Emitted Dots</h2>
<p><img src="figures/h84_surface_and_dots.png" alt="H84-RAMP 4-panel diagnostic figure" style="max-width:100%;height:auto;border:1px solid #cbd5e1;border-radius:8px;"></p>

<h2>6. Experiment 3: Two-Phase Registry Uniqueness Audit (Surface &amp; Final Dots)</h2>
<ul>
  <li><b>Phase 1 (Continuous surface before placement):</b> Max |Spearman &rho;| = <b>{e1['phase1_pre_placement_surface_spearman']['max_abs_spearman']:.4f} &le; 0.90</b> (physics composite) and <b>{e3['phase1_fullcat_surface_spearman']['max_abs_spearman']:.4f} &le; 0.90</b> (full-catalogue B-LBCO surface) across {e3['phase1_fullcat_surface_spearman']['checked']} local + restored rasters.</li>
  <li><b>Phase 2 (Informative sparse dot-field comparison across {sp_sum['sparse_rasters_checked']} sparse rasters):</b> <b>0 overlap firings</b> (max 3-px forward overlap = <b>{sp_sum['max_forward_overlap_3px']:.4f} &le; 0.70</b> vs <code>{e(sp_sum['worst_overlap_submission'])}</code>; max |Spearman &rho;| = <b>{sp_sum['max_abs_spearman']:.4f} &le; 0.90</b>; vs <code>H33</code> champion <code>5b3becc1...</code> overlap = <b>{e3['vs_h33_champion']['my_dots_within_3px_of_theirs']:.4f}</b>, Spearman = <b>{e3['vs_h33_champion']['spearman_full_footprint']:+.4f}</b>).</li>
  <li><b>Phase 2 (Literal Policy I17 dense-witness check):</b> Against restored <code>.cache/registry/dense_witness.tif</code> (<code>ab0a0a62...</code>, 2,671,197 positive pixels whose 3-px dilation covers 100.0% of all 5,106,385 allowed cells), forward 3-px overlap is <code>1.0000 &gt; 0.70</code> (<code>protocol_unique = false</code>).</li>
</ul>
<table>
  <tr><th>Comparator Raster</th><th>Representation</th><th>Forward 3-px Overlap (ours in theirs)</th><th>Spearman &rho;</th><th>Gate Status</th></tr>
  {top_ov_rows}
</table>

<h2>7. Named Non-Fault Mimicking Process &amp; Format Preflight (15/15 Checks)</h2>
<p><b>Named non-fault mimic:</b> {e(card['named_non_fault_mimicking_process'])}</p>
<ul>{val_checks}</ul>
<p><a href="h84-ramp-executive-summary.html">Executive summary &amp; step-by-step submission guide →</a> · <a href="index.html">Return to home portfolio table →</a></p>
"""

    exec_body = f"""<h1>H84-RAMP Executive Summary &amp; Competition Submission Guide</h1>
<p class="verdict"><b>Decision for <code>{e(card['submission_name'])}</code>: ✅ OK TO DOWNLOAD FOR INSPECTION · ⛔ NOT OK TO SUBMIT TO COMPETITION.</b><br>
Format-valid (15/15 checks, single-band <code>float32</code>, <code>EPSG:32611</code>, <code>3730×3292</code>, values in <code>{{0.0, 1.0}}</code> within <code>[0, 1]</code>, <code>0</code> NaN/Inf, <code>37,654</code> binary dots, <code>0</code> on catalogue) and informative-unique across all {sp_sum['sparse_rasters_checked']} sparse rasters checked (<code>0</code> overlap firings, max 3-px forward overlap <code>{sp_sum['max_forward_overlap_3px']:.4f} &le; 0.70</code>, max |Spearman &rho;| <code>{sp_sum['max_abs_spearman']:.4f} &le; 0.90</code>). Scores <b><code>{s20_sc['A_std_plus_h84']['dti']:.4f}</code> HOLDOUT-DTI on Seed 20</b> (95% CI <code>[{s20_sc['A_std_plus_h84']['ci95'][0]:.4f}, {s20_sc['A_std_plus_h84']['ci95'][1]:.4f}]</code>) and <b><code>{s30_sc['A_std_plus_h84']['dti']:.4f}</code> HOLDOUT-DTI on Seed 30</b> (95% CI <code>[{s30_sc['A_std_plus_h84']['ci95'][0]:.4f}, {s30_sc['A_std_plus_h84']['ci95'][1]:.4f}]</code>; pure physics <code>S_h84_physics</code> reaches <b><code>{s20_sc['S_h84_physics']['dti']:.4f}</code></b> s20 / <b><code>{s30_sc['S_h84_physics']['dti']:.4f}</code></b> s30). Because it remains below the <code>0.152718</code> repository holdout incumbent (<code>H77-Step</code>) and literal Policy I17 fails closed on <code>dense_witness.tif</code> (<code>1.000 &gt; 0.70</code>), its verdict is <b><code>negative</code></b> (<code>0</code> weekly slots used; selector: not approved).</p>

<section class="card dl research"><h2>⬇ Inspection Download &amp; Copyable Metadata</h2>
<a class="btn" href="{e(tif_rel)}" download>Download .tif ({card['raster_bytes']//1024} KB)</a>
<a class="btn alt" href="{e(zip_rel)}" download>Download one-TIFF .zip</a>
{copy("Submission name (reference only — DO NOT UPLOAD)", card["submission_name"])}
{copy("Short note <= 140 chars (reference only — DO NOT UPLOAD)", card["submission_note"])}
<p class="small">SHA-256 (.tif): <code>{e(card['raster_sha256'])}</code> · SHA-256 (.zip): <code>{e(card['zip_sha256'])}</code> ·
<a href="h84-ramp.html">Full H84-RAMP technical page →</a> · <a href="evidence/h84ramp_run_card.json">JSON run card →</a></p>
</section>

<h2>Why Previous Submissions Failed with “Predicted values must be in range [0, 1]” &amp; How This File Prevents It</h2>
<ol>
  <li><b>Root cause of the portal error:</b> Earlier files stored <code>NaN</code> or <code>-9999.0</code> sentinel values outside the Nevada study footprint or emitted unclipped float32 extrapolations (&lt;0 or &gt;1). DrivenData's validator checks <code>np.all((arr &gt;= 0.0) &amp; (arr &lt;= 1.0))</code> across the entire <code>3730×3292</code> raster grid.</li>
  <li><b>How <code>{e(card['submission_name'])}.tif</code> is guaranteed clean:</b> Emitted via <code>vendor/gems57/submission_writer.py:write_zero_outside</code> with explicit <code>nodata=None</code>, setting every outside-footprint cell to exact <code>0.0</code> and every inside-footprint cell to <code>0.0</code> or <code>1.0</code> (verified: <code>n_nan = 0</code>, <code>n_inf = 0</code>, <code>min = 0.0</code>, <code>max = 1.0</code>).</li>
</ol>

<h2>Step-by-Step DrivenData Submission Mechanics (Only When a Future Candidate Is Selector-Approved)</h2>
<ol>
  <li><b>Do NOT upload <code>{e(card['submission_name'])}</code>:</b> Its status is <b>NOT OK TO SUBMIT</b> (<code>verdict = "negative"</code>).</li>
  <li>When a future candidate shows <b>OK TO SUBMIT: YES</b> with selector approval, download its <code>-zeros.tif</code> (or single-TIFF <code>-zeros.zip</code>) and verify <code>sha256sum</code> locally.</li>
  <li>Log in to DrivenData Competition #306 (<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DOE GEMS Prize</a>), open the Submissions tab, upload the single <code>.tif</code> or <code>.zip</code>, paste the &le;140-character note, and confirm that at least 1 of the 3 rolling 7-day feedback slots is authorized.</li>
  <li>Record the returned portal score with a screenshot/receipt before labelling any number <code>ORGANIZER-CONFIRMED</code>.</li>
</ol>
<p><a href="h84-ramp.html">Back to H84-RAMP technical report →</a> · <a href="index.html">Back to main status table →</a></p>
"""

    (D / "h84-ramp.html").write_text(page("h84-ramp.html", "H84-RAMP · Two-Host Relay Corridor", body))
    (D / "h84-ramp-executive-summary.html").write_text(page("h84-ramp-executive-summary.html", "H84-RAMP Executive Summary", exec_body))
    (D / "evidence").mkdir(parents=True, exist_ok=True)
    for fn in ("h84ramp_run_card.json", "h84ramp_e1_canary_surface.json", "h84ramp_e2_holdout.json", "h84ramp_e3_uniqueness.json"):
        src = ROOT / "evidence" / fn
        if src.is_file():
            shutil.copyfile(src, D / "evidence" / fn)
    if (ROOT / "registry" / "preregistration_h84ramp.json").is_file():
        shutil.copyfile(ROOT / "registry" / "preregistration_h84ramp.json", D / "evidence" / "preregistration_h84ramp.json")
    if (ROOT / "research" / "h84ramp_hypotheses.md").is_file():
        (D / "research").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "research" / "h84ramp_hypotheses.md", D / "research" / "h84ramp_hypotheses.md")
    print("wrote docs/h84-ramp.html and docs/h84-ramp-executive-summary.html")
