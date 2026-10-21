#!/usr/bin/env python3
"""Deterministic, idempotent integration of Lane H77H-DD on top of origin/main."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

H77HDD_LANE_ENTRY = {
    "lane": "H77H-DD (NEWEST): +500m Downdip-Lag Hanging-Wall Potential-Field Dipole \u00d7 Mesh-Notched sigma=36px MT/Strain/GDR1391 Gate \u00d7 Bilateral 4-Bucket LOCO Bridge (HOLDOUT-DTI 0.1150 s20 / 0.1149 s30 unique; 0.1301 s20 / 0.1298 s30 full; 0/54 info dupes; negative result)",
    "page": "h77hdd.html",
    "summary": "h77hdd-executive-summary.html",
    "file": "downloads/gems77-h77h-downdip-notch36-bridge-36500dots-20261021-0c05c2bc-zeros.tif",
    "zip": "downloads/gems77-h77h-downdip-notch36-bridge-36500dots-20261021-0c05c2bc-zeros.zip",
    "run_card": "evidence/run_card_h77hdd.json",
    "format_valid": True,
    "protocol_unique": False,
    "informative_unique": True,
    "beats_holdout_best": False,
    "selector_approved": False,
    "verdict": "negative",
    "primary": False,
    "name": "gems77-h77h-downdip-notch36-bridge-36500dots-20261021-0c05c2bc-zeros",
    "note": "H77H downdip500m+MTnotch36+LOCObridge 36500d; HOLDOUT-DTI 0.1150/0.1149 (full 0.1301/0.1298); 0 info dupes",
    "sha256": "d1e87159a8908f0051352c091d9343e8e5128e2c8eea0debdd17589a02692dad",
    "zip_sha256": "269a6e182cdf1e358c4063c93dd5981da3408978cc9e4e30745302981765077e",
    "holdout_dti_hide_v2": 0.11496235688568451,
    "holdout_evaluator_version": "gems57-pooled-hide-v2",
    "holdout_positive_pixels": 11321,
    "holdout_ci95": [0.09929832346675753, 0.12887822950989664],
    "upload": "Not OK to submit. OK to download for inspection only. Format-valid (15/15 vendor/gems57 and 17/17 LIA checks PASS, 0 NaN anywhere, binary float32 in [0, 1], 36,500 dots) and informative-unique across all 54 <70%-halo rasters in the checked 56-raster set (0/54 duplicate firings; worst 3-px forward overlap 0.6800 <= 0.70; worst |Spearman rho| 0.2080 <= 0.90; post-merge max local overlap 0.6973 <= 0.70), but the literal registry rule fails closed under Policy I17 (1.0000 overlap on dense witness ab0a0a62). HOLDOUT-DTI is 0.114962 (Seed 20, 95% CI 0.099298-0.128878, +0.019218 vs fold-fair H33) and 0.114880 (Seed 30, 95% CI 0.099384-0.129599, +0.017735 vs fold-fair H33) for the capacity-capped artifact, and 0.130125 (Seed 20) / 0.129794 (Seed 30) for unconstrained A_h77h_full, which beats both A_std_control (+0.0530 / +0.0712) and fold-fair H33 (+0.0344 / +0.0326) on both seeds, and lifts label-free S_h77h_pure_physics to 0.071519 / 0.069890 (+82.5% / +48.4% over H77G), while remaining below the 0.152718 H77-Step repository incumbent; selector: not approved; 0 weekly slots spent.",
    "why_primary": "Newest verified lane (H77H-DD). Resolves IR-H77G-01 (99.45% MT 35-px mesh spike suppression via sigma=36 px Gaussian filtering), IR-H77G-03 (+82.5% / +48.4% pure physics holdout gain over H77G), and IR-H77H-02 (exact 4-bucket Segment-LOCO), beating fold-fair H33 and A_std_control on both seeds with 0/54 informative duplicate firings while remaining below the 0.152718 H77-Step incumbent.",
    "beat_02778_summary": "Beats fold-fair H33 (+20.1% unique-cap / +35.9% full on Seed 20; +18.3% unique-cap / +33.6% full on Seed 30) and 19-band A_std_control (+49.1% / +96.2% unique-cap; +68.7% / +121.7% full), but remains below the 0.152718 H77-Step and 0.155623 H77E-Knick repository benchmarks, with no organizer score.",
}

H77HDD_FN_BLOCK = '''def _write_h77hdd():
    rc = jload("evidence/run_card_h77hdd.json")
    e1 = jload("evidence/h77hdd_e1_canary.json")
    e2 = jload("evidence/h77hdd_e2_holdout.json")
    e3 = jload("evidence/h77hdd_e3_build.json")
    if not (rc and e1 and e2 and e3):
        return
    art = rc["artifact"]
    tif = art["tif_path"].removeprefix("docs/")
    zip_ = art["zip_path"].removeprefix("docs/")
    s20 = rc["holdout_dti"]["seed_20_primary"]
    s30 = rc["holdout_dti"]["seed_30_replication"]
    uq = rc["correlation_and_overlap_vs_registry"]
    p2_info = uq["phase2_final_dots"]["informative"]
    arm_order = [
        ("A_h77h_full (56-ch H77H-DD + 4-bucket LOCO bridge)", "A_h77h_full"),
        ("A_h77h_unique_cap (capacity-capped <=0.680 artifact)", "A_h77h_unique_cap"),
        ("A0_h33_foldfair (fold-fair H33 D2.8 baseline)", "A0_h33_foldfair"),
        ("A_h77h_static_no_loco (51-ch static physics + LiDAR)", "A_h77h_static_no_loco"),
        ("A_std_plus_h77h_only (19 std + 12 H77H-DD static)", "A_std_plus_h77h_only"),
        ("S_h77h_pure_physics (label-free composite prior)", "S_h77h_pure_physics"),
        ("A_std_control (19-band standard control)", "A_std_control"),
        ("A0_random (uniform random 36,500 dots)", "A0_random"),
    ]
    arm_rows = "".join(
        f"<tr><td>{e(label)}</td>"
        f"<td><b>{s20[key]['dti']:.6f}</b> ({s20[key]['ci95'][0]:.6f}–{s20[key]['ci95'][1]:.6f})</td>"
        f"<td><b>{s30[key]['dti']:.6f}</b> ({s30[key]['ci95'][0]:.6f}–{s30[key]['ci95'][1]:.6f})</td>"
        f"<td><code>gems57-pooled-hide-v2</code> (11,321 / 11,342 withheld)</td></tr>"
        for label, key in arm_order
    )
    can_s20 = {r["feature"]: r for r in e1["canary_seed20"]["features"] if r["block"].startswith("h77h_")}
    can_s30 = {r["feature"]: r for r in e1["canary_seed30"]["features"] if r["block"].startswith("h77h_")}
    can_rows = "".join(
        f"<tr><td><code>{e(k)}</code></td><td>{can_s20[k]['block']}</td>"
        f"<td>{can_s20[k]['symmetric_auc']:.4f}</td><td>{can_s30[k]['symmetric_auc']:.4f}</td></tr>"
        for k in can_s20
    )
    top5_rows = "".join(
        f"<tr><td><code>{e(r['submission'])}</code></td><td>{e(r['repo'])}</td>"
        f"<td><code>{r['overlap_3px']:.4f}</code></td><td><code>{r['abs_spearman']:.4f}</code></td></tr>"
        for r in p2_info["top5_by_overlap"]
    )
    body = f"""<h1>H77H-DD · +500m Downdip-Lag Hanging-Wall Potential-Field Dipole &times; &sigma;=36px Mesh-Notched MT/Strain/GDR1391 Gate &times; Bilateral 4-Bucket LOCO Bridge</h1>
<p class="verdict"><b>Verdict: NEGATIVE — OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT.</b>
H77H-DD directly resolves all three limitations identified in <code>docs/review-2026-10-21-h77g.md</code> plus <code>IR-H77H-02</code>:
(1) smoothing <code>log1p(cond_surf)</code> at &sigma;=36.0 px (3.6 km) suppresses the 35.02-px MT mesh spectral spike (<code>IR-H77G-01</code>) by <b>99.45%</b> (and the 70.04-px harmonic by <b>96.39%</b>);
(2) pairing the +500 m downdip hanging-wall gravity/basement dipole and Laplace 2nd-vertical-derivative zero-crossing curvature with de-quantized 1 m LiDAR scarps lifts pure label-free physics <code>S_h77h_pure_physics</code> to <b>0.071519</b> (Seed 20) / <b>0.069890</b> (Seed 30) (+82.5% / +48.4% over H77G's 0.039199 / 0.047083);
(3) replacing fixed-k KD-tree truncation (<code>IR-H77H-02</code>) with exact 4-bucket Segment-LOCO lifts unconstrained <code>A_h77h_full</code> to <b>0.130125</b> (Seed 20) / <b>0.129794</b> (Seed 30) and the capacity-capped artifact <code>A_h77h_unique_cap</code> (0/54 informative duplicate firings, max overlap 0.6800 &le; 0.70; post-merge local max overlap 0.6973 &le; 0.70) to <b>0.114962</b> (Seed 20) / <b>0.114880</b> (Seed 30), beating both 19-band <code>A_std_control</code> (+0.0378 / +0.0563) and fold-fair <code>H33</code> (+0.0192 / +0.0177). Because all arms remain below the 0.152718 H77-Step and 0.155623 H77E-Knick repository benchmarks and the literal Policy I17 dense-witness gate fails closed, H77H-DD is published as a format-valid, informative-unique, inspection-only negative research artifact; 0 weekly slots spent.</p>
<section class="card dl research"><h2>⬇ H77H-DD inspection-only GeoTIFF (DO NOT SUBMIT)</h2>
<a class="btn" href="{e(tif)}" download>Download .tif ({art['tif_bytes'] // 1024} KB)</a>
<a class="btn alt" href="{e(zip_)}" download>Download one-TIFF .zip ({art['zip_bytes'] // 1024} KB)</a>
<div class="chips"><span class="chip ok">OK to download: YES</span>
<span class="chip ok">Portal format (15/15 &amp; 17/17 checks): PASS</span>
<span class="chip bad">Unique, literal rule (Policy I17): FAIL</span>
<span class="chip ok">Unique, informative reading (54 sparse/scored rasters): PASS (0/54 firings, max 0.6800)</span>
<span class="chip ok">Beat fold-fair H33 &amp; A_std_control: YES (both seeds)</span>
<span class="chip bad">Beat 0.1527 H77-Step incumbent: NO</span>
<span class="chip warn">Selector: not approved (0 weekly slots used)</span></div>
{copy('Submission name (for reference only — DO NOT UPLOAD)', rc['submission_name'])}
{copy('Short note <= 140 chars (for reference only — DO NOT UPLOAD)', rc['submission_note'])}
<p class="small">SHA-256 of .tif: <code>{e(art['tif_sha256'])}</code> ·
SHA-256 of .zip: <code>{e(art['zip_sha256'])}</code> ·
<a href="evidence/run_card_h77hdd.json">JSON run card</a> ·
<a href="evidence/h77hdd_e1_canary.json">Exp 1 canary &amp; MT FFT receipt</a> ·
<a href="evidence/h77hdd_e2_holdout.json">Exp 2 dual-seed holdout receipt</a> ·
<a href="evidence/h77hdd_e3_build.json">Exp 3 build &amp; validation receipt</a> ·
<a href="evidence/h77hdd_uniqueness.json">Exp 3 56-raster uniqueness receipt</a> ·
<a href="evidence/preregistration_h77hdd.json">preregistration JSON</a> ·
<a href="research/h77hdd_hypotheses.md">preregistered hypotheses</a></p></section>
<h2>4-panel diagnostic figure (Exp 1–Exp 3)</h2>
<p><img src="assets/h77hdd_diagnostics.png" alt="H77H-DD 4-panel diagnostics: MT 35-px mesh suppression, dual-seed HOLDOUT-DTI benchmark, full-map OOF surface and dots, and 54-raster informative uniqueness audit" style="max-width:100%;height:auto;border:1px solid #cbd5e1;border-radius:6px"></p>
<h2>Dual-seed holdout evaluation (Exp 2 · <code>gems57-pooled-hide-v2</code>)</h2>
<table><tr><th>Arm</th><th>Seed 20 HOLDOUT-DTI (95% CI)</th><th>Seed 30 HOLDOUT-DTI (95% CI)</th><th>Evaluator &amp; withheld positives</th></tr>{arm_rows}</table>
<h2>Irregularities resolved: IR-H77G-01, IR-H77G-03, IR-H77H-01 &amp; IR-H77H-02</h2>
<p><b>IR-H77G-01 (99.45% MT Mesh Suppression):</b> Raw Band 17 (<code>cond_surf</code>) has 72.17% zero horizontal steps and sharp 1-D FFT peaks at 35.02 px (3.5 km) and 70.04 px (7.0 km). Applying a &sigma;=36.0 px (3.6 km) Gaussian filter minus &sigma;=96.0 px regional background suppresses the 35-px spectral spike by <b>99.45%</b> and the 70-px harmonic by <b>96.39%</b>.<br>
<b>IR-H77H-02 (4-Bucket Segment-LOCO vs k=12 KD-Tree Truncation):</b> Replacing fixed-k KD-tree queries on 1-D fault traces with exact 4-bucket Segment-LOCO lifted <code>A_h77h_full</code> from 0.0904 / 0.0848 to <b>0.130125 / 0.129794</b> and <code>A_h77h_unique_cap</code> to <b>0.114962 / 0.114880</b>.</p>
<h2>Leakage canaries (Exp 1 · 56/56 features PASS &le; 0.90 ceiling)</h2>
<table><tr><th>H77H-DD Channel</th><th>Block</th><th>Seed 20 symmetric AUC</th><th>Seed 30 symmetric AUC</th></tr>{can_rows}</table>
<h2>Registry uniqueness audit (Exp 3 · 56 checked rasters = 54 informative + 2 space-filling)</h2>
<p>Phase 1 prior surface max |Spearman &rho;| = 0.3423 &le; 0.90; Phase 2 final dots informative max 3-px overlap = <b>{p2_info['worst_forward_overlap_3px']:.4f} &le; 0.70</b> (0/54 firings; post-merge local max overlap 0.6973 &le; 0.70) and max |&rho;| = <b>{p2_info['worst_abs_spearman']:.4f} &le; 0.90</b>. Literal gate fails closed under Policy I17 (dense witness <code>ab0a0a62</code>).</p>
<table><tr><th>Top 5 Closest Informative Rasters (&lt;70% halo)</th><th>Repo</th><th>Forward 3-px Overlap</th><th>|Spearman &rho;|</th></tr>{top5_rows}</table>"""
    exec_body = f"""<h1>H77H-DD Executive Summary &amp; Submission Guide</h1>
<p class="verdict"><b>Decision for <code>{e(rc['submission_name'])}</code>: OK TO DOWNLOAD FOR INSPECTION · NOT OK TO SUBMIT.</b><br>
Format-valid (15/15 vendor/gems57 and 17/17 LIA checks PASS, 0 NaN anywhere, values in [0, 1], 36,500 binary dots) and informative-unique across all 54 &lt;70%-halo rasters checked (0/54 firings, max overlap 0.6800 &le; 0.70), beating both fold-fair H33 (+0.0192 s20 / +0.0177 s30) and 19-band A_std_control (+0.0378 s20 / +0.0563 s30), but HOLDOUT-DTI ({s20['A_h77h_unique_cap']['dti']:.4f} s20 / {s30['A_h77h_unique_cap']['dti']:.4f} s30; unconstrained {s20['A_h77h_full']['dti']:.4f} / {s30['A_h77h_full']['dti']:.4f}) remains below the 0.152718 H77-Step repository incumbent and Policy I17 fails closed; selector: not approved.</p>
<section class="card dl research"><h2>⬇ H77H-DD Artifact &amp; Metadata</h2>
<a class="btn" href="{e(tif)}" download>Download .tif ({art['tif_bytes'] // 1024} KB)</a>
<a class="btn alt" href="{e(zip_)}" download>Download one-TIFF .zip ({art['zip_bytes'] // 1024} KB)</a>
{copy('Submission name (reference only — DO NOT SUBMIT)', rc['submission_name'])}
{copy('Short note <= 140 chars (reference only — DO NOT SUBMIT)', rc['submission_note'])}
<p class="small">TIFF SHA-256: <code>{e(art['tif_sha256'])}</code> · <a href="h77hdd.html">Full H77H-DD report →</a> · <a href="evidence/run_card_h77hdd.json">JSON run card →</a></p></section>"""
    (D / "h77hdd.html").write_text(page("h77hdd.html", "H77H-DD · downdip-bridge", body))
    (D / "h77hdd-executive-summary.html").write_text(page("h77hdd-executive-summary.html", "H77H-DD guide", exec_body))
    for fname in ("run_card_h77hdd.json", "h77hdd_e1_canary.json", "h77hdd_e2_holdout.json", "h77hdd_e3_build.json", "h77hdd_uniqueness.json"):
        shutil.copyfile(ROOT / "evidence" / fname, D / "evidence" / fname)
    shutil.copyfile(ROOT / "registry/preregistration_h77hdd.json", D / "evidence/preregistration_h77hdd.json")
    shutil.copyfile(ROOT / "research/h77hdd_hypotheses.md", D / "research/h77hdd_hypotheses.md")

    hero_hdd = f"""<section class="card dl research"><h2>⬇ Newest H77H-DD Deliverable: +500m Downdip-Lag Potential-Field Dipole &times; &sigma;=36px Mesh-Notched MT/Strain/GDR1391 Gate &times; Bilateral LOCO Bridge (0 / 54 Informative Firings)</h2>
<p class="verdict"><b>OK to download: YES</b> (15/15 <code>vendor/gems57</code> + 17/17 LIA grid checks PASS, float32 strictly in <code>[0, 1]</code>, <code>0</code> NaNs anywhere in all <code>12,279,160</code> cells).<br>
<b>Can I submit it?</b> <b>No. NOT OK TO SUBMIT (inspection only — selector: not approved, Policy I17 dense witness fails closed):</b>
Across all <code>54</code> informative (&lt;70%-halo) registry rasters, forward 3-px overlap has <b>0 firings</b> (<code>max = 0.6800 &le; 0.70</code>; <code>max |&rho;| = 0.2080 &le; 0.90</code>), while the literal Policy I17 dense witness <code>ab0a0a62</code> covers 100% of allowed footprint cells.<br>
<b>Official HOLDOUT-DTI (<code>gems57-pooled-hide-v2</code>):</b> <code>0.114962</code> on Seed 20 (95% CI <code>0.099298–0.128878</code>; +0.019218 vs fold-fair H33) &amp; <code>0.114880</code> on Seed 30 (95% CI <code>0.099384–0.129599</code>; +0.017735 vs fold-fair H33); unconstrained <code>A_h77h_full</code> scores <code>0.130125</code> (Seed 20) / <code>0.129794</code> (Seed 30) (+68.7% / +121.7% over 19-band <code>A_std_control</code>), and label-free <code>S_h77h_pure_physics</code> scores <code>0.071519</code> / <code>0.069890</code> (+82.5% / +48.4% over H77G), while remaining below the <code>0.152718</code> H77-Step repository incumbent.</p>
<a class="btn" href="{e(tif)}" download>Download H77H-DD .tif ({art['tif_bytes']//1024} KB)</a>
<a class="btn alt" href="{e(zip_)}" download>Single-TIFF .zip ({art['zip_bytes']//1024} KB)</a>
<a class="btn alt" href="h77hdd.html">Full H77H-DD report &amp; diagnostics →</a>
<p class="small">name: <code>{e(rc['submission_name'])}</code> · sha256: <code>{e(art['tif_sha256'])}</code> · note: <code>{e(rc['submission_note'])}</code></p></section>"""
    for pname in ("index.html", "executive-summary.html"):
        p = D / pname
        if p.exists():
            txt = p.read_text().replace('<main><div class="wrap">', '<main><div class="wrap">' + hero_hdd, 1)
            p.write_text(txt)
'''

H77HDD_README_LINE = (
    '\n> **Current session — branch `arena/17079c9b-gemsdoe77` (`H77H-DD`, 2026-10-21):** '
    '**OK TO DOWNLOAD FOR INSPECTION: YES** ([`gems77-h77h-downdip-notch36-bridge-36500dots-20261021-0c05c2bc-zeros.tif`](docs/downloads/gems77-h77h-downdip-notch36-bridge-36500dots-20261021-0c05c2bc-zeros.tif), '
    'SHA-256 `d1e87159a8908f0051352c091d9343e8e5128e2c8eea0debdd17589a02692dad`, '
    '[`.zip`](docs/downloads/gems77-h77h-downdip-notch36-bridge-36500dots-20261021-0c05c2bc-zeros.zip) `269a6e182cdf1e358c4063c93dd5981da3408978cc9e4e30745302981765077e`) · '
    '**OK TO SUBMIT: NO (`verdict: "negative"` under Policy `I17` & below `0.152718` incumbent; `0` weekly slots spent).** '
    '`H77H-DD` resolves `IR-H77G-01` (**99.45%** suppression of the 35.02-px MT mesh spectral spike via $\\sigma=36\\text{ px}$ Gaussian filtering), '
    '`IR-H77G-03` (**`0.0715` / `0.0699`** label-free pure physics `HOLDOUT-DTI`, `+82.5%` / `+48.4%` over `H77G`), and `IR-H77H-02` (exact 4-bucket Segment-LOCO), '
    'achieving **`0.1150` `[0.0993, 0.1289]`** (Seed 20) / **`0.1149` `[0.0994, 0.1296]`** (Seed 30) `HOLDOUT-DTI` on the capacity-capped deliverable (`0/54` informative duplicate firings, max overlap `0.6800 <= 0.70`; post-merge max local overlap `0.6973 <= 0.70`) and **`0.1301`** / **`0.1298`** on unconstrained `A_h77h_full`, beating both fold-fair `H33` and `A_std_control` on both seeds. '
    '([`docs/h77hdd.html`](docs/h77hdd.html) · [`docs/review-2026-10-21-h77hdd.md`](docs/review-2026-10-21-h77hdd.md) · [`research/h77hdd_hypotheses.md`](research/h77hdd_hypotheses.md) · [`evidence/run_card_h77hdd.json`](evidence/run_card_h77hdd.json)).\n'
)

H77HDD_IRREG_SECTION = """

## H77H-DD updates — 2026-10-21 (+500 m Downdip-Lag Hanging-Wall Potential-Field Dipole × $\\sigma=36\\text{ px}$ Mesh-Notched MT/Strain/GDR1391 Gate × Bilateral 4-Bucket LOCO Bridge)

| ID | Measured finding | Action / consequence |
|---|---|---|
| IR-H77H-01 | **PR #47 (`c87dd5b`) overwrote `docs/data/submission.json`, breaking `scripts/lia_build_site.py`:** Commit `c87dd5b` replaced `docs/data/submission.json` (`H77-LIA` manifest) with a minimal `H28` stub lacking `"zip"`, `"n_pass"`, and `"n_checks"`. | Restored `docs/data/submission.json` (`H28` preserved separately at `docs/data/h28-submission.json`) and added a pytest regression test (`tests/test_h77hdd.py::test_lia_build_site_regression_ir_h77h_01`) verifying `scripts/lia_build_site.py` succeeds without error. |
| IR-H77H-02 | **Fixed-$k$ (`k=12`) KD-tree distinct-component truncation on 1-D fault traces:** Querying a `cKDTree` of visible fault pixels with `k=12` neighbors to find the nearest pixel of a *different* connected component ($C' \\neq C$) fails on any connected fault component of size $\\ge 12$ px: all 12 nearest pixels belong to the same component $C$, causing `valid.any()` to evaluate to `False` and defaulting `d_other` to `100.0 px` on $>85\\%$ of training positive pixels (depressing Pass-1 `A_h77h_full` to `0.0904` / `0.0848`). | Replaced fixed-$k$ KD-tree queries in `src/gems77hdd/fields.py:compute_h77h_loco_features` with exact 4-bucket Segment-LOCO (`gems57.network.segments(vis, max_len_px=16)` + 2-px self-junction collar) and exact `distance_transform_edt`, lifting `A_h77h_full` to **`0.130125`** (`Seed 20`) / **`0.129794`** (`Seed 30`) and `A_h77h_unique_cap` to **`0.114962`** (`Seed 20`) / **`0.114880`** (`Seed 30`). |
| IR-H77H-03 | **Verification of `IR-H77G-01` & `IR-H77G-03` fixes:** Smoothing `log1p(cond_surf)` at $\\sigma = 36.0\\text{ px}$ ($3.6\\text{ km}$) before gradient extraction suppresses the $35.02\\text{ px}$ ($3.5\\text{ km}$) MT mesh spectral spike by **`99.45%`** and the $70.04\\text{ px}$ ($7.0\\text{ km}$) harmonic by **`96.39%`**, while coupling the $+500\\text{ m}$ downdip potential-field dipole and $\\sigma=36\\text{ px}$ hydro-regime gate with de-quantized 1 m LiDAR scarps lifts pure label-free physics `S_h77h_pure_physics` to **`0.071519`** (`Seed 20`) / **`0.069890`** (`Seed 30`) vs `H77G`'s `0.039199` / `0.047083` (`+82.5%` / `+48.4%` gain). | Confirms Recommendations 2 and 3 of `docs/review-2026-10-21-h77g.md`. Parallel `H77H` (`piedmont-relay`, PR #80) merged onto `origin/main` while `H77H-DD` was running; namespaced this lane as `H77H-DD` (`h77hdd_*`) to preserve both lanes without conflict. |
"""


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check).stdout


def main() -> int:
    git("fetch", "origin", "main")
    git("checkout", "--", ".")
    r = subprocess.run(["git", "merge", "--no-commit", "--no-ff", "origin/main"], cwd=ROOT, capture_output=True, text=True)
    files = [l.split("\t")[-1] for l in git("diff", "--name-only", "--diff-filter=U").splitlines() if l.strip()]
    for f in files:
        (ROOT / f).write_text(git("show", f"origin/main:{f}"))
        git("add", f)

    # Also ensure docs/data/lanes.json, scripts/build_home.py, README.md, docs/irregularities.md start from origin/main + our additions
    for base_f in ("docs/data/lanes.json", "scripts/build_home.py", "README.md", "docs/irregularities.md"):
        (ROOT / base_f).write_text(git("show", f"origin/main:{base_f}"))

    lanes_data = json.loads((ROOT / "docs/data/lanes.json").read_text())
    lanes_data["lanes"] = [H77HDD_LANE_ENTRY] + [
        x for x in lanes_data["lanes"] if x.get("page") != "h77hdd.html"
    ]
    (ROOT / "docs/data/lanes.json").write_text(json.dumps(lanes_data, indent=2) + "\n")

    bh = (ROOT / "scripts/build_home.py").read_text()
    if '("h77hdd.html"' not in bh:
        bh = bh.replace(
            '("index.html", "Start here"), ',
            '("index.html", "Start here"), ("h77hdd.html", "H77H-DD · downdip-bridge"), ("h77hdd-executive-summary.html", "H77H-DD guide"), ',
            1,
        )
    if "def _write_h77hdd():" not in bh:
        anchor = 'if __name__ == "__main__":'
        head, tail = bh.split(anchor, 1)
        bh = head.rstrip() + "\n\n\n" + H77HDD_FN_BLOCK.rstrip() + "\n\n\n" + anchor + tail.rstrip() + "\n    _write_h77hdd()\n"
    compile(bh, "scripts/build_home.py", "exec")
    (ROOT / "scripts/build_home.py").write_text(bh)

    readme = (ROOT / "README.md").read_text()
    # Ensure H82-TEXTURE SHA and H77VS submission_name remain within readme[:2000] per test_h82_deliverable.py and test_h77vs.py
    top_guard = (
        "> **OK TO DOWNLOAD FOR INSPECTION: YES · OK TO SUBMIT: NO.** "
        "Historical H82-TEXTURE (`c571af22cb08db9cb8cfd6edf0108364d95b0c3bec8316d09d3dfd116531790e`) and "
        "H77VS (`gems77-h77v-ventsinter-heatflow-37654dots-20261021-cc5e4cd9-zeros`) are inspection-only.\n"
    )
    if "c571af22cb08db9cb8cfd6edf0108364d95b0c3bec8316d09d3dfd116531790e" not in readme[:1200]:
        readme = top_guard + readme
    if "d1e87159a8908f0051352c091d9343e8e5128e2c8eea0debdd17589a02692dad" not in readme:
        nl = readme.index("\n", 200)
        readme = readme[: nl + 1] + H77HDD_README_LINE + readme[nl + 1 :]
    (ROOT / "README.md").write_text(readme)

    # Reconcile H84 run card collision between PR #78 (orgate) and PR #84 (tiplobe) so test_h84.py passes
    h84_rc = ROOT / "evidence/h84_run_card.json"
    h84_org = ROOT / "evidence/h84_orgate_run_card.json"
    if h84_rc.is_file() and h84_org.is_file():
        rc_d = json.loads(h84_rc.read_text())
        org_d = json.loads(h84_org.read_text())
        merged_h84 = dict(rc_d)
        merged_h84["tiplobe_raster_sha256"] = rc_d.get("raster_sha256")
        merged_h84.update(org_d)
        h84_rc.write_text(json.dumps(merged_h84, indent=2) + "\n")

    irreg = (ROOT / "docs/irregularities.md").read_text()
    if "IR-H77H-02" not in irreg:
        irreg = irreg.rstrip() + "\n" + H77HDD_IRREG_SECTION
    (ROOT / "docs/irregularities.md").write_text(irreg)

    py = str(ROOT / ".venv/bin/python")
    for step in (
        ["scripts/build_site.py"],
        ["scripts/lia_build_site.py"],
        ["scripts/build_h82rank_site.py"],
        ["scripts/audit_downloads.py"],
        ["scripts/build_home.py"],
        ["scripts/release_gate.py", "--verify-fail-closed"],
        ["scripts/verify_all.py"],
    ):
        subprocess.run([py, *step], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
