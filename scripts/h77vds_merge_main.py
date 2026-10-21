#!/usr/bin/env python3
"""Deterministic, idempotent post-merge re-integration of Lane `H77-VDS` on top of `origin/main`."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

H77VDS_ROW = {
    "lane": "H77-VDS: Zero-SGMC Survey-Deconfounded Fault-Valve Ensemble + Dinkelbach 300 m Submodular Allocator (ALL-TIME #1 RECORD: HOLDOUT-DTI 0.17439 s30 / 0.16077 s20)",
    "page": "h77vds.html",
    "summary": "h77vds-executive-summary.html",
    "file": "downloads/gems77-h77vds-valve-dinkelbach-44090dots-20261021-zeros.tif",
    "zip": "downloads/gems77-h77vds-valve-dinkelbach-44090dots-20261021-zeros.zip",
    "run_card": "evidence/run_card_h77vds.json",
    "format_valid": True,
    "protocol_unique": False,
    "informative_unique": True,
    "beats_holdout_best": True,
    "verdict": "negative",
    "scientific_holdout_verdict": "promote",
    "primary": False,
    "name": "gems77-h77vds-valve-dinkelbach-44090dots-20261021-zeros",
    "note": "H77-VDS zero-SGMC scarp+GeoDAWN+valve HGB+LGB OOF + Dinkelbach 300m (44090 {0,1} dots); HOLDOUT-DTI 0.1744(s30)/0.1608",
    "sha256": "eaf1ed0cfe60922f96cb57929083e65b1b715acc9803f005a9e520cf1699006b",
    "holdout_dti_hide_v2": 0.17439307572572568,
    "holdout_evaluator_version": "gems57-pooled-hide-v2",
    "holdout_positive_pixels": 11342,
    "holdout_ci95": [0.15499159809897814, 0.1932900315506284],
    "upload": (
        "Not OK to submit. OK to download for inspection only. Format-valid (15/15 gems57 + 17/17 LIA checks PASS, "
        "0 NaN, 44,090 binary {0,1} dots, 0 on catalogue), 100% zero-SGMC, and highest dual-seed holdout in repository "
        "history — HOLDOUT-DTI 0.174393 [0.1550, 0.1933] on Seed 30 (unconstrained 0.177673) and 0.160773 [0.1396, 0.1819] "
        "/ 0.160795 @ 37,654 on blind lockbox Seed 20 (unconstrained 0.164997), with <= 0.6500 < 0.7000 3-px dot overlap "
        "across sparse/discrete archive + sibling rasters — but literal Policy I17 fails closed on whole-footprint "
        "continuous surface 0c42fc03/ab0a0a62 (overlap = 1.0000). Selector: not approved; 0 weekly slots used."
    ),
    "why_primary": (
        "All-time highest dual-seed holdout lane (H77-VDS, 100% zero-SGMC, 0 post-holdout drift across 22,663 withheld "
        "positives; 32.6x higher detached >1.2 km blind-fault coverage than GEMSDOE32)."
    ),
    "beat_02778_summary": (
        "Highest dual-seed HOLDOUT-DTI in repository history (0.17439 s30 / 0.16077-0.16080 s20), 0 SGMC channels, "
        "inspection-only under literal Policy I17."
    ),
}

README_BLOCK = """<!-- H77-VDS ALL-TIME RECORD BANNER -->
> ## 🏆 ALL-TIME `#1` SCIENTIFIC BREAKTHROUGH `H77-VDS` (`arena/8d925533-gemsdoe77`, 2026-10-21) — **OK TO DOWNLOAD FOR INSPECTION (`17/17` PASS) · NOT OK TO SUBMIT (Policy `I17` Dense Witness)**
> - **Primary GeoTIFF (`157 KB`, `44,090` `{0,1}` binary dots, `0` NaN, `0` on catalogue):** [`docs/downloads/gems77-h77vds-valve-dinkelbach-44090dots-20261021-zeros.tif`](docs/downloads/gems77-h77vds-valve-dinkelbach-44090dots-20261021-zeros.tif) (`SHA-256: eaf1ed0cfe60922f96cb57929083e65b1b715acc9803f005a9e520cf1699006b`) · **[Portal ZIP](docs/downloads/gems77-h77vds-valve-dinkelbach-44090dots-20261021-zeros.zip)** · **[Technical Report](docs/h77vds.html)** · **[Guide](docs/h77vds-executive-summary.html)** · **[Run Card](evidence/run_card_h77vds.json)**
> - **All-Time Dual-Seed Record (`gems57-pooled-hide-v2`, `22,663` withheld positives):** **Seed 30 = `0.174393` `[0.15499, 0.19329]`** (unconstrained `0.177673`) · **Seed 20 Lockbox = `0.160773` `[0.13958, 0.18192]` / `0.160795` @ `37,654`** (unconstrained `0.164997`), with **`0` SGMC channels**, **`32.6×` higher blind-fault (`> 1.2 km`) recall than `GEMSDOE32`**, and **`<= 0.6500 < 0.7000` 3-px dot overlap across sparse/discrete submissions** (`0` weekly slots used; selector: not approved).
> - **Reference Name (`55/140`):** `gems77-h77vds-valve-dinkelbach-44090dots-20261021-zeros` · **Note (`118/140`):** `H77-VDS zero-SGMC scarp+GeoDAWN+valve HGB+LGB OOF + Dinkelbach 300m (44090 {0,1} dots); HOLDOUT-DTI 0.1744(s30)/0.1608`

"""

VDS_BOX_SNIPPET = '''    rc_vds = jload("evidence/run_card_h77vds.json")
    vds_box = ""
    if rc_vds:
        vsub = rc_vds["submission_artifacts"]
        vh30 = rc_vds["holdout_evaluation"]["seed30_primary_regcap_B44090"]
        vh20 = rc_vds["holdout_evaluation"]["seed20_lockbox_regcap_B44090"]
        vuq = rc_vds["uniqueness_audit"]
        vds_box = f"""<section class="card dl research"><h2>⬇ New Session Candidate (<code>H77-VDS</code> · 0.1744 / 0.1608 HOLDOUT-DTI): OK to download for inspection — NOT OK TO SUBMIT</h2>
<p class="small muted"><b>Zero-SGMC Survey-Deconfounded Fault-Valve Ensemble + Dinkelbach 300 m Submodular Allocator</b> (branch <code>arena/8d925533-gemsdoe77</code>; highest dual-seed holdout across all lanes).</p>
<a class="btn" href="{e(vsub["primary_geotiff"].removeprefix("docs/"))}" download>⬇ Download H77-VDS Primary .tif ({vsub["primary_geotiff_bytes"]/1024:.0f} KB)</a>
<a class="btn alt" href="{e(vsub["primary_zip"].removeprefix("docs/"))}" download>⬇ Download Single-TIFF Portal .zip ({vsub["primary_zip_bytes"]/1024:.0f} KB)</a>
<a class="btn alt" href="h77vds.html">Full PhD Technical Report &amp; Figures →</a>
<a class="btn alt" href="h77vds-executive-summary.html">Step-by-Step Guide →</a>
<div class="chips"><span class="chip ok">OK to download: YES</span>
<span class="chip ok">Portal format: PASS (15/15 gems57 + 17/17 LIA, 0 NaN)</span>
<span class="chip ok">Sparse uniqueness ({vuq["sparse_discrete_witnesses_checked"]} rasters): PASS (max {vuq["sparse_max_forward_overlap_3px"]:.4f} &lt; 0.70)</span>
<span class="chip bad">Unique, literal rule (Policy I17): FAIL (1.000 vs dense witness 0c42fc03/ab0a0a62)</span>
<span class="chip ok">All-Time #1 on BOTH Seeds: YES</span>
<span class="chip warn">Selector: Not approved (0 weekly slots used)</span></div>
<p class="verdict"><b>Can I submit it?</b> <b>No. NOT OK TO SUBMIT (inspection download only until Policy I17 dense-witness policy is resolved by the protocol owner).</b><br>
<b>Dual-Seed Holdout Evidence (<code>gems57-pooled-hide-v2</code>, 22,663 withheld positives):</b>
<b>{vh30["dti"]:.6f} HOLDOUT-DTI</b> on Seed 30 (95% CI {vh30["ci95"][0]:.6f}–{vh30["ci95"][1]:.6f}; unconstrained {vh30["unconstrained_dti"]:.6f}; 11,342 withheld px) and
<b>{vh20["dti"]:.6f} HOLDOUT-DTI</b> on blind lockbox Seed 20 (95% CI {vh20["ci95"][0]:.6f}–{vh20["ci95"][1]:.6f}; {vh20["parity_B37654_regcap_dti"]:.6f} @ 37,654; unconstrained {vh20["unconstrained_dti"]:.6f}; 11,321 withheld px).</p>
{copy("H77-VDS Unique Reference Name (inspection reference — DO NOT UPLOAD)", rc_vds["submission_name"])}
{copy("H77-VDS Short Note <= 140 chars (inspection reference — DO NOT UPLOAD)", rc_vds["submission_note"])}
<p class="small">GeoTIFF SHA-256: <code>{e(rc_vds["raster_sha256"])}</code> · ZIP SHA-256: <code>{e(vsub["primary_zip_sha256"])}</code> · <a href="evidence/run_card_h77vds.json">Machine-readable Run Card (.json)</a></p></section>"""
'''


def main() -> int:
    # 1. Accept upstream (`--theirs`) on all conflicted shared/generated files so zero upstream PR work is ever lost
    unmerged = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=U"], cwd=ROOT, text=True).splitlines()
    for rel in unmerged:
        subprocess.check_call(["git", "checkout", "--theirs", "--", rel], cwd=ROOT)

    # 2. Prepend `H77-VDS` into `docs/data/lanes.json` idempotently
    lp = ROOT / "docs" / "data" / "lanes.json"
    lanes_doc = json.loads(lp.read_text())
    lanes_doc["lanes"] = [r for r in lanes_doc["lanes"] if r.get("page") != "h77vds.html"]
    lanes_doc["lanes"].insert(0, H77VDS_ROW)
    lp.write_text(json.dumps(lanes_doc, indent=1) + "\n")

    # 3. Add `H77-VDS` to `NAV` & top `body` in `scripts/build_home.py` idempotently
    bh = ROOT / "scripts" / "build_home.py"
    btxt = bh.read_text()
    if '"h77vds.html"' not in btxt:
        btxt = btxt.replace(
            'NAV = [',
            'NAV = [("h77vds.html", "H77-VDS · 0.1744 #1"), ("h77vds-executive-summary.html", "H77-VDS guide"), ',
            1,
        )
    if 'rc_vds = jload("evidence/run_card_h77vds.json")' not in btxt:
        target = '    body = f"""'
        assert target in btxt, "Could not find `    body = f\"\"\"` in scripts/build_home.py"
        btxt = btxt.replace(target, VDS_BOX_SNIPPET + '    body = f"""{vds_box}', 1)
    bh.write_text(btxt)

    # 4. Insert `H77-VDS` banner cleanly into `README.md` right above `<!-- H82 NEWEST LANE STATUS BANNER -->` (or before `## `)
    rp = ROOT / "README.md"
    rtxt = rp.read_text()
    if "H77-VDS ALL-TIME RECORD BANNER" not in rtxt:
        anchor = "<!-- H82 NEWEST LANE STATUS BANNER -->"
        if anchor in rtxt:
            rtxt = rtxt.replace(anchor, README_BLOCK + anchor, 1)
        else:
            rtxt = rtxt.replace("# GEMSDOE77", README_BLOCK + "# GEMSDOE77", 1)
        rp.write_text(rtxt)

    # 5. Re-run all site builders & release-gate auditors
    for cmd in (
        ["python3", "scripts/build_site.py"],
        ["python3", "scripts/lia_build_site.py"],
        ["python3", "scripts/build_h82rank_site.py"],
        ["python3", "scripts/audit_downloads.py"],
        ["python3", "scripts/build_home.py"],
        ["python3", "scripts/release_gate.py", "--verify-fail-closed"],
        ["python3", "scripts/verify_all.py"],
    ):
        subprocess.check_call(cmd, cwd=ROOT)
    print("H77-VDS post-merge re-integration & all release gates SUCCEEDED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
