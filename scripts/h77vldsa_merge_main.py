#!/usr/bin/env python3
"""Deterministic, idempotent post-merge re-integration of Lane `H77V-LDSA` on top of `origin/main`."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

H77VLDSA_ROW = {
    "lane": "H77V-LDSA: 3.6 km Mesh-Dealiased Crustal Regime x Tubular Demag Trough x 3DEP Fluvial Knickpoint-Scarp Step + LDSA 300 m Submodular Cover",
    "page": "h77vldsa.html",
    "summary": "h77vldsa-executive-summary.html",
    "file": "downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.tif",
    "zip": "downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.zip",
    "run_card": "evidence/run_card_h77vldsa.json",
    "format_valid": True,
    "protocol_unique": False,
    "informative_unique": True,
    "beats_holdout_best": True,
    "verdict": "negative",
    "primary": False,
    "name": "gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021",
    "note": "GEMSDOE77 H77V: 3.6km dealiased MT/strain + tubular demag trough + 3DEP knick/scarp + 8-ch SLOCO + LDSA 300m cover (inspection only)",
    "sha256": "861c880e7a56203d197182075765480efaf224ad6cfde342fab511f5f908fa4c",
    "holdout_dti_hide_v2": 0.1444462843570187,
    "holdout_evaluator_version": "gems57-pooled-hide-v2",
    "holdout_positive_pixels": 11321,
    "holdout_ci95": [0.12728735130586463, 0.1606096268356279],
    "upload": (
        "Not OK to submit. OK to download for inspection only. Format-valid (15/15 validator checks, 0 NaN, float32 strictly "
        "in [0, 1], 36,500 binary dots) and informative-unique across all 95 non-space-filling (<70%-halo) rasters checked "
        "(0 firings, max forward 3-px overlap 0.6500 <= 0.70; surface max |Spearman rho| 0.7519 <= 0.90), achieving 0.144446 "
        "[0.1273, 0.1606] HOLDOUT-DTI on Seed 20 (11,321 withheld positives, +0.0178 over H77E-Knick unique and +0.0481 over "
        "fold-fair H33) and 0.142415 [0.1254, 0.1587] on Seed 30 (11,342 withheld positives, +0.0177 over H77E-Knick unique), "
        "while unconstrained H77V Pure LDSA achieves 0.153036 (Seed 20) and 0.152730 (Seed 30) (exceeding the 0.152718 H77-Step "
        "baseline on both seeds; 140/140 leakage canaries passed). Literal Policy I17 fails closed on 6 >=70%-halo "
        "space-filling/lattice witnesses (1.000 vs dense_witness ab0a0a62). Selector: not approved; 0 weekly slots used."
    ),
    "why_primary": (
        "Implements IR-H77G-01/02/03 (3.6 km spectral mesh de-aliasing + across-valley tubular demagnetization troughs + "
        "road/BLM-mining-deconfounded 1 m LiDAR / 10 m 3DEP knickpoint-scarp coupling) with Kernel-Convolved Smooth Lagrangian "
        "Congestion Pricing (LDSA), lifting 36,500-dot zero-informative-firing HOLDOUT-DTI to 0.144446 (Seed 20) / 0.142415 "
        "(Seed 30) and unconstrained pure LDSA to 0.153036 / 0.152730."
    ),
    "beat_02778_summary": (
        "Beats fold-fair H33 by +0.0481 (Seed 20) and +0.0434 (Seed 30) HOLDOUT-DTI and H77E-Knick Unique by +0.0178 / +0.0177 "
        "with 0/95 informative firings; inspection-only under literal Policy I17."
    ),
}

README_BLOCK = """<!-- H77V-LDSA PRIMARY STATUS BANNER -->
> **🚦 Current session — branch `arena/65c05dc9-gemsdoe77` (`H77V-LDSA`, 2026-10-21): ✅ OK TO DOWNLOAD FOR INSPECTION: YES · ⛔ OK TO SUBMIT: NO (`verdict: "negative"` under Policy `I17` dense witness; `0` weekly slots spent).**
> - **Primary GeoTIFF (`145,365 B`, `36,500` `{0,1}` binary dots, `0` NaN, `0` on catalogue, $\\min d_{\\text{cat}} = 2.2361\\text{ px}$):** [`docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.tif`](docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.tif) (`SHA-256 = 861c880e7a56203d197182075765480efaf224ad6cfde342fab511f5f908fa4c`) · **[Single-TIFF `.zip`](docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.zip)** (`SHA-256 = efa45a7be334f80a3bccc548c2453394fe3b33dd27d50d834235255a1af7000d`) · **[Sidecar `.json`](docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.json)** · **[Technical Report](docs/h77vldsa.html)** · **[Submission Guide](docs/h77vldsa-executive-summary.html)** · **[Run Card](evidence/run_card_h77vldsa.json)** · **[3-Pass Review](docs/review-2026-10-21-h77vldsa.md)**
> - **Official `HOLDOUT-DTI` (`gems57-pooled-hide-v2`, 2,000-draw 20 km block bootstrap, `140/140` leakage canaries passed `<= 0.90`):** **Portfolio-Unique LDSA (`0/95` informative firings, `max_ov = 0.65000 <= 0.70`): `0.144446` `[0.127287, 0.160610]` on Seed 20 (`11,321` withheld pos, `+0.0178` over `H77E-Knick` Unique `0.126664` and `+0.0481` over fold-fair `H33` `0.096320`) and `0.142415` `[0.125407, 0.158674]` on Seed 30 (`11,342` withheld pos, `+0.0177` over `H77E-Knick` Unique `0.124727`); Unconstrained Pure LDSA: `0.153036` (Seed 20) and `0.152730` (Seed 30)** (exceeding the `0.152718` `H77-Step` baseline on both seeds).

"""

BUILD_HOME_SNIPPET = '''

def _write_h77vldsa():
    rc = jload("evidence/h77v/run_card.json")
    can = jload("evidence/h77v/canary_seed20.json")
    if not rc:
        return
    sub = rc["submission_artifact"]
    s20 = rc["holdout_dti_seed20_11321px"]["scores"]
    s30 = rc["holdout_dti_seed30_11342px"]["scores"]
    d20_u = rc["holdout_dti_seed20_unique_candidate"]["paired_differences"]
    uq = rc["uniqueness_audit"]
    hm = rc["headline_holdout_metrics"]

    arm_labels = [
        ("h77v_pure_submodular", "H77V-LDSA Pure Submodular (Unconstrained)"),
        ("h77v_portfolio_unique_submodular", "H77V-LDSA Portfolio-Unique (0/95 Informative Firings)"),
        ("ablation_base120_without_h77v_channels", "Ablation: Base 120 (H77E) Without H77V Channels"),
        ("h77v_pure_physics_no_mechanics", "H77V-LDSA Pure Zero-Catalogue Static Physics (0 SLOCO)"),
        ("h77v_peak_nms_control", "H77V-LDSA Fixed-Radius Peak-NMS Control"),
        ("h33_foldfair", "Fold-Fair H33 Reference (D2.8 at d_vis > 2.0)"),
        ("random_uniform_dots", "Random Uniform Control (36,500 Dots)"),
    ]
    arm_rows = ""
    for key, label in arm_labels:
        v20 = s20[key]
        v30 = s30.get(key)
        c30 = (
            f"<b>{v30['dti']:.6f}</b> ({v30['ci95'][0]:.6f}–{v30['ci95'][1]:.6f})"
            if v30
            else "—"
        )
        arm_rows += (
            f"<tr><td>{e(label)}</td>"
            f"<td><b>{v20['dti']:.6f}</b> ({v20['ci95'][0]:.6f}–{v20['ci95'][1]:.6f})</td>"
            f"<td>{c30}</td>"
            f"<td><code>gems57-pooled-hide-v2</code> (11,321 / 11,342 withheld)</td></tr>"
        )

    diff_rows = "".join(
        f"<tr><td><code>H77V-LDSA Unique - {e(k)}</code></td>"
        f"<td><b>{v['delta']:+.6f}</b></td>"
        f"<td><code>[{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]</code></td></tr>"
        for k, v in d20_u.items()
    )

    can_rows = ""
    if can:
        h77v_new_rows = [r for r in can["rows"] if r["group"] == "h77v_new_physics"]
        for r in h77v_new_rows:
            can_rows += (
                f"<tr><td><code>{e(r['feature'])}</code></td>"
                f"<td><code>{r['symmetric_auc']:.6f}</code></td>"
                f"<td><code>{e(r['group'])}</code></td>"
                f"<td>PASS (&le; 0.90)</td></tr>"
            )

    ov_rows = "".join(
        f"<tr><td><code>{e(m['tag'] + ':' + m['name'][:44])}</code></td>"
        f"<td><code>{'Space-Filling (>=0.70)' if m['is_space_filling'] else 'Informative (<0.70)'}</code></td>"
        f"<td><code>{m['halo3_bg_frac']:.4f}</code></td>"
        f"<td><code>{m['vendor_exact_forward_3px_overlap']:.5f}</code></td>"
        f"<td><code>{m['memmap_vs_vendor_abs_diff']:.1e}</code></td></tr>"
        for m in uq["top_20_vendor_verified_matches"]
    )

    body = f"""<h1>H77V-LDSA · 3.6 km Spectral Mesh-Dealiased Crustal Regime × Tubular Demagnetization Trough × 3DEP Fluvial Knickpoint-Scarp Step + LDSA 300 m Submodular Cover</h1>
<p class="verdict"><b>Verdict: NEGATIVE (Policy I17 Fail-Closed) — OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT WITHOUT HUMAN SELECTOR SIGN-OFF.</b></p>
<section class="card dl research"><h2>⬇ Download Verified H77V-LDSA GeoTIFF &amp; Metadata (Inspection Only — Not Approved for Upload)</h2>
<a class="btn" href="downloads/{e(sub['submission_name'])}" download>Download H77V-LDSA .tif ({sub['tif_size_bytes'] // 1024} KB)</a>
<a class="btn alt" href="downloads/{e(sub['submission_zip'])}" download>Download Single-TIFF .zip ({sub['zip_size_bytes'] // 1024} KB)</a>
<a class="btn alt" href="downloads/{e(sub['submission_json'])}" download>Download Sidecar .json</a>
<div class="chips"><span class="chip ok">OK to download: YES</span>
<span class="chip ok">Portal format (15/15 checks): PASS</span>
<span class="chip ok">Range [0, 1] &amp; 0 NaNs: PASS</span>
<span class="chip ok">Unique, informative reading (0/95 firings, max 0.6500): PASS</span>
<span class="chip bad">Unique, literal rule (Policy I17 dense witness): FAIL</span>
<span class="chip ok">Beat H77E-Knick unique (+0.0178) &amp; H77-Step pure (0.1530): YES</span>
<span class="chip warn">Selector: not approved (0 weekly slots used)</span></div>
{copy('Submission name (reference only — DO NOT UPLOAD without selector Policy I17 sign-off)', Path(sub['submission_name']).stem)}
{copy('Short note <= 140 chars (132 chars — reference only)', sub['submission_note'])}
<p class="small">SHA-256 of <code>.tif</code>: <code>{e(sub['tif_sha256'])}</code> ·
SHA-256 of <code>.zip</code>: <code>{e(sub['zip_sha256'])}</code> ·
Emitted binary dots: <code>{sub['emitted_positive_dots']:,}</code> ·
Min catalogue distance: <code>{sub['min_distance_to_catalogue_px']:.4f} px</code> ·
<a href="evidence/run_card_h77vldsa.json">JSON Run Card</a> ·
<a href="evidence/h77v_exp1_seed20_holdout.json">Exp 1 Seed 20 Receipt</a> ·
<a href="evidence/h77v_exp2_seed30_holdout.json">Exp 2 Seed 30 Receipt</a> ·
<a href="evidence/h77v_canary_seed20.json">140-Feature Canary Receipt</a> ·
<a href="evidence/preregistration_h77vldsa.json">Preregistration JSON</a> ·
<a href="research/h77vldsa_hypotheses.md">Preregistered Hypotheses &amp; H33 Forensics</a></p></section>
<h2>Visual Diagnostics</h2>
<p><img src="figures/h77v/fig1_h77v_physics_panels.png" alt="H77V Physics Panels" style="max-width:100%;height:auto"></p>
<p><img src="figures/h77v/fig2_h77v_holdout_bootstrap.png" alt="H77V Holdout Bootstrap" style="max-width:100%;height:auto"></p>
<p><img src="figures/h77v/fig3_h77v_uniqueness_spectrum.png" alt="H77V Uniqueness Spectrum" style="max-width:100%;height:auto"></p>
<p><img src="figures/h77v/fig4_h77v_submission_map.png" alt="H77V Submission Map" style="max-width:540px;height:auto"></p>
<h2>Dual-Seed Official Spatially Blocked Holdout Benchmark (<code>gems57-pooled-hide-v2</code>)</h2>
<table><tr><th>Arm</th><th>Seed 20 HOLDOUT-DTI (95% CI)</th><th>Seed 30 HOLDOUT-DTI (95% CI)</th><th>Evaluator &amp; Withheld Positives</th></tr>{arm_rows}</table>
<h3>Paired Block-Bootstrap Differences on Seed 20</h3>
<table><tr><th>Paired Comparison</th><th>&Delta; HOLDOUT-DTI</th><th>95% Block-Bootstrap CI</th></tr>{diff_rows}</table>
<h2>Leakage Canary Gate (<code>140 / 140</code> Passed &le; 0.90)</h2>
<table><tr><th>New H77V-LDSA Physical Channel</th><th>Seed 20 Symmetric ROC-AUC</th><th>Group</th><th>Status</th></tr>{can_rows}</table>
<h2>Registry Uniqueness Audit (<code>0 / 95</code> Informative Firings)</h2>
<table><tr><th>Registry / Sibling Raster</th><th>Class</th><th>3-px Halo Fraction</th><th>Exact Forward 3-px Overlap</th><th>Memmap vs Vendor Diff</th></tr>{ov_rows}</table>
"""
    exec_body = f"""<h1>H77V-LDSA Executive Summary &amp; Submission Guide</h1>
<p class="verdict"><b>Decision for <code>{e(sub['submission_name'])}</code>: OK TO DOWNLOAD FOR INSPECTION · NOT OK TO SUBMIT WITHOUT SELECTOR POLICY I17 SIGN-OFF.</b></p>
<section class="card dl research"><h2>⬇ H77V-LDSA Artifact &amp; Copy-Paste Metadata</h2>
<a class="btn" href="downloads/{e(sub['submission_name'])}" download>Download H77V-LDSA .tif ({sub['tif_size_bytes'] // 1024} KB)</a>
<a class="btn alt" href="downloads/{e(sub['submission_zip'])}" download>Download Single-TIFF .zip ({sub['zip_size_bytes'] // 1024} KB)</a>
<a class="btn alt" href="downloads/{e(sub['submission_json'])}" download>Download Sidecar .json</a>
{copy('Submission name (reference only — DO NOT SUBMIT without selector approval)', Path(sub['submission_name']).stem)}
{copy('Short note <= 140 chars (132 chars — reference only)', sub['submission_note'])}
<p class="small">TIFF SHA-256: <code>{e(sub['tif_sha256'])}</code> · ZIP SHA-256: <code>{e(sub['zip_sha256'])}</code> · <a href="h77vldsa.html">Full H77V-LDSA Technical Report &amp; Figures →</a> · <a href="evidence/run_card_h77vldsa.json">JSON Run Card →</a></p></section>"""
    (D / "h77vldsa.html").write_text(page("h77vldsa.html", "H77V-LDSA · 0.1444/0.1530", body))
    (D / "h77vldsa-executive-summary.html").write_text(page("h77vldsa-executive-summary.html", "H77V-LDSA guide", exec_body))
    (D / "evidence").mkdir(parents=True, exist_ok=True)
    (D / "research").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "evidence/h77v/run_card.json", D / "evidence/run_card_h77vldsa.json")
    shutil.copyfile(ROOT / "evidence/h77v/exp1_seed20_holdout.json", D / "evidence/h77v_exp1_seed20_holdout.json")
    shutil.copyfile(ROOT / "evidence/h77v/exp2_seed30_holdout.json", D / "evidence/h77v_exp2_seed30_holdout.json")
    shutil.copyfile(ROOT / "evidence/h77v/canary_seed20.json", D / "evidence/h77v_canary_seed20.json")
    shutil.copyfile(ROOT / "registry/preregistration_h77vldsa.json", D / "evidence/preregistration_h77vldsa.json")
    shutil.copyfile(ROOT / "research/h77vldsa_hypotheses.md", D / "research/h77vldsa_hypotheses.md")

    hero_vldsa = f"""<section class="card dl research" style="border:2px solid #0284c7;background:#f0f9ff;"><h2>⬇ NEWEST H77V-LDSA Deliverable (0 / 95 Informative Firings · 0.1444 / 0.1424 Unique &amp; 0.1530 / 0.1527 Pure HOLDOUT-DTI)</h2>
<p class="verdict"><b>OK TO DOWNLOAD FOR INSPECTION: YES</b> (15/15 <code>vendor/gems57</code> validator checks PASS, float32 values strictly in <code>[0, 1]</code>, <code>0</code> NaNs across all <code>12,279,160</code> cells, <code>36,500</code> binary dots).<br>
<b>NOT OK TO SUBMIT AUTOMATICALLY (Inspection Only — Selector: not approved under Policy I17):</b>
6 space-filling/lattice &ge;70%-halo archive witnesses trigger literal overlap; across ALL <code>95</code> &lt;70%-halo informative rasters, forward 3-px overlap has <b>0 firings</b> (<code>max = 0.65000 &le; 0.70</code>; surface max |Spearman &rho;| = <code>0.751885 &le; 0.90</code>).<br>
<b>Official HOLDOUT-DTI (<code>gems57-pooled-hide-v2</code>):</b> <b><code>{hm['h77v_portfolio_unique_submodular_seed20']['dti']:.6f}</code> on Seed 20</b> (95% CI <code>{hm['h77v_portfolio_unique_submodular_seed20']['ci95'][0]:.6f}–{hm['h77v_portfolio_unique_submodular_seed20']['ci95'][1]:.6f}</code>; <b>+0.0178 over H77E-Knick unique</b> &amp; <b>+0.0481 over fold-fair H33</b>) &amp; <b><code>{hm['h77v_portfolio_unique_submodular_seed30']['dti']:.6f}</code> on Seed 30</b> (95% CI <code>{hm['h77v_portfolio_unique_submodular_seed30']['ci95'][0]:.6f}–{hm['h77v_portfolio_unique_submodular_seed30']['ci95'][1]:.6f}</code>; <b>+0.0177 over H77E-Knick unique</b>); unconstrained pure LDSA twin achieves <b><code>{hm['h77v_pure_submodular_seed20']['dti']:.6f}</code> (Seed 20) / <code>{hm['h77v_pure_submodular_seed30']['dti']:.6f}</code> (Seed 30)</b>.</p>
<a class="btn" href="downloads/{e(sub['submission_name'])}" download>Download H77V-LDSA .tif ({sub['tif_size_bytes'] // 1024} KB)</a>
<a class="btn alt" href="downloads/{e(sub['submission_zip'])}" download>Single-TIFF .zip ({sub['zip_size_bytes'] // 1024} KB)</a>
<a class="btn alt" href="h77vldsa.html">Full H77V-LDSA Report &amp; Figures →</a>
<a class="btn alt" href="h77vldsa-executive-summary.html">H77V-LDSA Executive Summary →</a>
<p class="small">name: <code>{e(Path(sub['submission_name']).stem)}</code> · sha256: <code>{e(sub['tif_sha256'])}</code> · note (132 chars): <code>{e(sub['submission_note'])}</code></p></section>"""
    for pname in ("index.html", "executive-summary.html"):
        p = D / pname
        if p.exists():
            txt = p.read_text().replace('<main><div class="wrap">', '<main><div class="wrap">' + hero_vldsa, 1)
            p.write_text(txt)
'''


def main() -> int:
    unmerged = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=U"], cwd=ROOT, text=True).splitlines()
    for rel in unmerged:
        subprocess.check_call(["git", "checkout", "--theirs", "--", rel], cwd=ROOT)

    lp = ROOT / "docs" / "data" / "lanes.json"
    lanes_doc = json.loads(lp.read_text())
    lanes_doc["lanes"] = [r for r in lanes_doc["lanes"] if r.get("page") != "h77vldsa.html"]
    lanes_doc["lanes"].insert(0, H77VLDSA_ROW)
    lp.write_text(json.dumps(lanes_doc, indent=1) + "\n")

    bh = ROOT / "scripts" / "build_home.py"
    btxt = bh.read_text()
    if '"h77vldsa.html"' not in btxt:
        btxt = btxt.replace(
            'NAV = [',
            'NAV = [("h77vldsa.html", "H77V-LDSA · 0.1444/0.1530"), ("h77vldsa-executive-summary.html", "H77V-LDSA guide"), ',
            1,
        )
    if "def _write_h77vldsa():" not in btxt:
        btxt = btxt.replace(
            'if __name__ == "__main__":',
            BUILD_HOME_SNIPPET + '\n\nif __name__ == "__main__":',
            1,
        )
    if "    _write_h77vldsa()" not in btxt:
        btxt = btxt + "\n    _write_h77vldsa()\n"
    bh.write_text(btxt)

    rp = ROOT / "README.md"
    rtxt = rp.read_text()
    if "H77V-LDSA PRIMARY STATUS BANNER" not in rtxt:
        rtxt = README_BLOCK + rtxt
        rp.write_text(rtxt)

    py = str(ROOT / ".venv" / "bin" / "python")
    for cmd in (
        [py, "scripts/build_site.py"],
        [py, "scripts/lia_build_site.py"],
        [py, "scripts/build_h82rank_site.py"],
        [py, "scripts/audit_downloads.py"],
        [py, "scripts/build_home.py"],
        [py, "scripts/release_gate.py", "--verify-fail-closed"],
        [py, "scripts/verify_all.py"],
    ):
        subprocess.check_call(cmd, cwd=ROOT)
    print("H77V-LDSA post-merge re-integration & all release gates SUCCEEDED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
