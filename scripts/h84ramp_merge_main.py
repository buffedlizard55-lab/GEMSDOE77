#!/usr/bin/env python3
"""Deterministic re-integration of origin/main into Lane H84-RAMP."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

H84RAMP_LANE = {
    "lane": "H84-RAMP: B-LBCO deconfounded two-host en-echelon relay-ramp + GeoDAWN/basement stepover corridor (0.1366 s20 / 0.1446 s30 HOLDOUT-DTI; negative, inspection only)",
    "page": "h84-ramp.html",
    "exec_page": "h84-ramp-executive-summary.html",
    "summary": "h84-ramp-executive-summary.html",
    "file": "downloads/gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros.tif",
    "zip": "downloads/gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros.zip",
    "evidence": "evidence/h84ramp_run_card.json",
    "run_card": "evidence/h84ramp_run_card.json",
    "prereg": "registry/preregistration_h84ramp.json",
    "format_valid": True,
    "protocol_unique": False,
    "informative_unique": True,
    "beats_holdout_best": False,
    "selector_approved": False,
    "verdict": "negative",
    "primary": False,
    "name": "gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros",
    "note": "H84-RAMP B-LBCO two-host relay+geod+base LOQO 37654 dots; HOLDOUT-DTI 0.1366 s20/0.1446 s30; DO NOT UPLOAD",
    "sha256": "f06ada5aa834ceaf65e8700e36bbd46eddb96929b42c9041c72c8369c1022838",
    "zip_sha256": "8499c3427d4f48e6a2be8929693236a9e9d7555b52fc559df7260512bdd12de1",
    "holdout_dti_hide_v2": 0.136572,
    "holdout_evaluator_version": "gems57-pooled-hide-v2",
    "holdout_positive_pixels": 11321,
    "holdout_ci95": [0.123687, 0.149983],
    "upload": "Not OK to submit. OK to download for inspection only. Format-valid (15/15 checks, 0 NaN, {0,1} values, 37,654 binary dots, 0 on catalogue, min distance 5.00 px) and informative-unique across all sparse dot-field rasters checked (0 overlap firings, max 3-px forward overlap 0.5299 <= 0.70, max |Spearman rho| 0.0601 <= 0.90; vs H33 overlap 0.2265). B-LBCO deconfounding resolves IR-H84-01 (+0.0948 s20 / +0.1115 s30 over naive relay training; +0.0608 s20 / +0.0774 s30 over 19-band control; +0.0921 s20 / +0.0977 s30 over random), reaching 0.1366 [0.1237, 0.1500] HOLDOUT-DTI on Seed 20 and 0.1446 [0.1277, 0.1619] on Seed 30 (pure physics S_h84_physics = 0.1423 s20 / 0.1476 s30). Below 0.152718 H77-Step / 0.1744 H77-VDS repository best and literal Policy I17 fails closed vs dense_witness (ab0a0a62 -> 1.000). Selector: not approved; 0 weekly slots used.",
    "weekly_slots_used": 0,
    "why_primary": "H84-RAMP lane (2026-10-21); negative under Policy I17 and below repository holdout best; clearance per release gate (not approved).",
    "beat_02778_summary": "Not known. HOLDOUT-DTI 0.1366 (s20) / 0.1446 (s30) is below the repository holdout best; no organizer score exists for this file and none may be inferred.",
}

H84RAMP_README_BLOCK = """
<!-- H84-RAMP STATUS BANNER (2026-10-21; branch arena/693062aa-gemsdoe77) -->
> ## 🚦 H84-RAMP DOWNLOAD & SUBMISSION STATUS (2026-10-21 — Deconfounded B-LBCO Two-Host En-Echelon Relay-Ramp & Geodetic-Geophysical Stepover Transfer Corridor)
>
> - **✅ OK TO DOWNLOAD FOR INSPECTION: YES** — [`docs/downloads/gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros.tif`](docs/downloads/gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros.tif) (`144,529 B`, SHA-256 `f06ada5aa834ceaf65e8700e36bbd46eddb96929b42c9041c72c8369c1022838`; one-TIFF [`.zip`](docs/downloads/gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros.zip) SHA-256 `8499c3427d4f48e6a2be8929693236a9e9d7555b52fc559df7260512bdd12de1`). Local format preflight PASS (`15/15` checks: single-band `float32`, `EPSG:32611`, `3730×3292`, values in `{0.0, 1.0}`, `0` NaN/Inf, `nodata=None`, `37,654` binary dots, `0` on catalogue, `min_dot_distance_to_catalogue_px = 5.00 px`; not organizer acceptance).
> - **⛔ OK TO SUBMIT: NO — NOT OK TO SUBMIT (Verdict: `negative` under Policy I17; Selector: not approved; 0 weekly slots used).** On dual-seed spatial holdout (`gems57-pooled-hide-v2`, $\\alpha=0.2$, $\\beta=0.8$, 300 m kernel, 2,000 bootstrap draws), `A_std_plus_h84` scores **`0.1446` `HOLDOUT-DTI` on Seed 30** (`95% CI [0.1277, 0.1619]`, `11,342` withheld positives; pure physics `S_h84_physics` = **`0.1476`**, `95% CI [0.1288, 0.1666]`; paired **`+0.1115`** over `A_std_plus_h84_naive` without `B-LBCO`, **`+0.0774`** over `A_std_control` 19-band control, and **`+0.0977`** over `A0_random`, all 95% CIs excluding 0) and **`0.1366` `HOLDOUT-DTI` on Seed 20** (`95% CI [0.1237, 0.1500]`, `11,321` withheld positives; pure physics `S_h84_physics` = **`0.1423`**, `95% CI [0.1279, 0.1564]`; paired **`+0.0948`** over naive, **`+0.0608`** over 19-band control, and **`+0.0921`** over random). Passes informative sparse uniqueness across all sparse dot-field rasters checked (**`0` overlap firings**, `max 3-px forward overlap = 0.5299 <= 0.70`, `max |Spearman rho| = 0.0601 <= 0.90`; vs `H33` overlap `0.2265`), but fails literal Policy I17 (`1.000 > 0.70` vs `dense_witness.tif` `ab0a0a62`) and remains below the `0.152718` `H77-Step` / `0.174393` `H77-VDS` repository holdout best.
> - **📖** [`docs/h84-ramp.html`](docs/h84-ramp.html) · [`docs/h84-ramp-executive-summary.html`](docs/h84-ramp-executive-summary.html) · [`evidence/h84ramp_run_card.json`](evidence/h84ramp_run_card.json) · [`research/h84ramp_hypotheses.md`](research/h84ramp_hypotheses.md) · [`registry/preregistration_h84ramp.json`](registry/preregistration_h84ramp.json) · [`evidence/h84ramp_e1_canary_surface.json`](evidence/h84ramp_e1_canary_surface.json) · [`evidence/h84ramp_e2_holdout.json`](evidence/h84ramp_e2_holdout.json) · [`evidence/h84ramp_e3_uniqueness.json`](evidence/h84ramp_e3_uniqueness.json) · [`docs/review-2026-10-21-arena-693062aa.md`](docs/review-2026-10-21-arena-693062aa.md).
> - Name (reference only — DO NOT UPLOAD): `gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros` · Note (106/140): `H84-RAMP B-LBCO two-host relay+geod+base LOQO 37654 dots; HOLDOUT-DTI 0.1366 s20/0.1446 s30; DO NOT UPLOAD`
"""

H84RAMP_IR_BLOCK = """
## H84-RAMP updates — 2026-10-21 (Deconfounded B-LBCO Two-Host En-Echelon Relay-Ramp & Geodetic-Geophysical Stepover Transfer Corridor; branch `arena/693062aa-gemsdoe77`)

| ID | Measured finding | Action / consequence |
|---|---|---|
| IR-H84RAMP-01 (IR-H84-01) | **The Relay-Stepover Self-Collapse Tautology under Naive Visible-Fault Training:** In `vendor/gems57/relay_bend_anatomy.py:two_host_frame(visible, ...)`, every pixel $x \\in V$ on a visible training fault has $d_1(x, V) \\equiv 0$. Consequently, any stepover or damage-collar operator gated by $d_1 > 0$ collapses identically to `0.0000` on 100% of visible training positive pixels (`7 of 8` `H84` channels have naive visible positive mean `0.000000` in `evidence/h84ramp_e1_canary_surface.json`). Training `HistGradientBoostingClassifier` on naive `H84` features (`A_std_plus_h84_naive`) teaches the trees that `stepover > 0` implies `y = 0`, collapsing holdout performance to `0.041762` on Seed 20 and `0.033091` on Seed 30 (below uniform random `0.044499` / `0.046917`). | Implemented 5-bucket **Leave-Bucket-Plus-3px-Collar-Out (`B-LBCO`)** via bitplane `cKDTree` (`src/gems84/fields.py:build_blbco_two_host_frame`), removing each segment bucket $B_b$ and its $3.0\\text{ px}$ ($300\\text{ m}$) collar when computing $(d_1, d_2, \\theta_1, \\theta_2)$ for pixels in $B_b \\cup \\text{collar}(B_b, 3\\text{ px})$. This restores realistic non-zero stepover means on visible training positives (`0.1555` vs `0.2072` on withheld holdout positives) and raises `A_std_plus_h84` `HOLDOUT-DTI` to **`0.136572`** on Seed 20 (paired **`+0.094809`** over naive, 95% CI `[+0.073557, +0.116740]`) and **`0.144613`** on Seed 30 (paired **`+0.111521`** over naive, 95% CI `[+0.087498, +0.137060]`). |
| IR-H84RAMP-02 (IR-H84-02) | **Near-Catalogue Collar Overlap vs Stepover Transfer Corridor Separation:** Prior collar lanes (`H83`, `H77E-pure-submod`, `H77E-knick-deconf-submod`) place dots along the immediate $3.0\\text{–}3.5\\text{ px}$ single-fault collar ($d_1 \\approx 3\\text{ px}$), causing any new candidate weighted toward the $3\\text{ px}$ collar peak to exceed $0.70$ 3-px forward overlap against those sibling rasters. | Weighting the `H84-RAMP` full-catalogue placement surface by the two-host stepover transfer ramp (`step_collar` peaking at $d_1 \\approx 6.5\\text{ px}$ + `h84_relay_medial_bridge`, `min_dot_distance_to_catalogue_px = 5.00 px`) places all `37,654` dots in the inter-tip stepover and outer damage transfer zone ($5.0\\text{–}15.0\\text{ px}$), achieving **`0` sparse overlap firings** across all sparse rasters checked (`max 3-px forward overlap = 0.5299 <= 0.70`, `max |Spearman rho| = 0.0601 <= 0.90`; vs `H33` overlap `0.2265`). Literal Policy I17 still fails closed (`1.000 > 0.70` vs `dense_witness.tif` `ab0a0a62`) and holdout DTI remains below the `0.152718` `H77-Step` / `0.174393` `H77-VDS` repository best, so the lane verdict is `negative` (`NOT OK TO SUBMIT`, `0` weekly slots used). |
"""


def apply():
    # 1. docs/data/lanes.json
    lanes_path = ROOT / "docs" / "data" / "lanes.json"
    data = json.loads(lanes_path.read_text())
    data["lanes"] = [x for x in data["lanes"] if not x["lane"].startswith("H84-RAMP:")]
    data["lanes"].append(H84RAMP_LANE)
    lanes_path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")

    # 2. docs/irregularities.md
    ir_path = ROOT / "docs" / "irregularities.md"
    ir_txt = ir_path.read_text()
    if "IR-H84RAMP-01" not in ir_txt:
        ir_path.write_text(ir_txt.rstrip() + "\n" + H84RAMP_IR_BLOCK)

    # 3. README.md
    rm_path = ROOT / "README.md"
    rm_txt = rm_path.read_text()
    if "f06ada5aa834ceaf65e8700e36bbd46eddb96929b42c9041c72c8369c1022838" not in rm_txt:
        anchor = "# GEMSDOE77 — DOE GEMS Prize: audited fault-detection research portfolio"
        if anchor in rm_txt:
            rm_txt = rm_txt.replace(anchor, H84RAMP_README_BLOCK.strip() + "\n\n" + anchor, 1)
            rm_path.write_text(rm_txt)

    # 4. scripts/build_home.py
    bh_path = ROOT / "scripts" / "build_home.py"
    bh_txt = bh_path.read_text()
    if "build_h84ramp_site" not in bh_txt:
        bh_txt = bh_txt.rstrip() + "\n\n    from build_h84ramp_site import build as build_h84ramp\n    build_h84ramp(page)\n"
        bh_path.write_text(bh_txt)


if __name__ == "__main__":
    apply()
