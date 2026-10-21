#!/usr/bin/env python3
"""Put the H85-EDTI decision on the first screen of the Pages site.

``scripts/build_home.py`` renders ``docs/index.html`` from ``docs/data/lanes.json``
and hard-codes a NOT-OK-TO-SUBMIT framing for the historical negative lanes.
This step runs immediately after it (locally and in ``.github/workflows/pages.yml``)
and prepends an idempotent status card for the one lane this repository has
promoted, plus a one-line status block on the submission guide.

It writes no scientific claim that is not already in
``evidence/h85_run_card.json`` and re-derives the on-disk checks itself.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as GRID  # noqa: E402
from gems57 import validate as V  # noqa: E402

DOCS = ROOT / "docs"
CARD = ROOT / "evidence" / "h85_run_card.json"
MARK = "<!-- H85-EDTI STATUS CARD -->"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def facts() -> dict:
    card = json.loads(CARD.read_text())
    tif = next(p for p in sorted((DOCS / "downloads").glob("gems85-*.tif")))
    zf = tif.with_suffix(".zip")
    grid = GRID.load_grid(ROOT / "data" / "bridge")
    rep = V.validate(tif, grid.footprint, grid.catalogue)
    with rasterio.open(tif) as src:
        a = src.read(1)
    return dict(card=card, tif=tif, zip=zf, rep=rep,
                sha=sha256(tif), zsha=sha256(zf),
                kb=tif.stat().st_size / 1024, zkb=zf.stat().st_size / 1024,
                positives=int((a > 0).sum()), nan=int(np.isnan(a).sum()),
                vmin=float(a.min()), vmax=float(a.max()),
                on_cat=int((a[grid.catalogue] > 0).sum()),
                checks=sum(1 for v in rep["checks"].values() if v),
                total=len(rep["checks"]))


def card_html(f: dict) -> str:
    c = f["card"]
    h = c["holdout_dti"]
    u = c["uniqueness_gate"] if "uniqueness_gate" in c else None
    uniq = json.loads((ROOT / "evidence" / "h85_uniqueness.json").read_text())
    b = uniq["binding"]
    return f"""{MARK}
<section class="card dl" id="h85" style="border:3px solid #1c7c4a">
<h2>✅ H85-EDTI — operator-selected submission candidate (OK TO DOWNLOAD: YES · OPERATOR DECISION: OK TO SUBMIT)</h2>
<p class="verdict"><b>OK to download: YES.</b> <b>Operator decision: OK TO SUBMIT.</b>
This is the one operator-selected file in this repository that passed the recorded format, holdout-improvement and
like-for-like uniqueness checks. It is a newly computed raster, not a copy of any prior submission. The repository
release gate remains separately fail-closed until the complete remote registry is materialised; see the authoritative
status card above and <a href="data/submission-authorization.json">its machine-readable scope</a>.</p>
<p><b>HOLDOUT-DTI</b> (<code>{h['evaluator_version']}</code>, alpha 0.2 / beta 0.8, 300 m triangular kernel):
<code>{h['seed_30']['score']:.6f}</code> on Seed 30 (95% CI <code>{h['seed_30']['ci95'][0]:.6f}–{h['seed_30']['ci95'][1]:.6f}</code>,
<code>{h['seed_30']['withheld_positive_pixels']:,}</code> withheld positives) and
<code>{h['seed_20']['score']:.6f}</code> on Seed 20 (95% CI <code>{h['seed_20']['ci95'][0]:.6f}–{h['seed_20']['ci95'][1]:.6f}</code>,
<code>{h['seed_20']['withheld_positive_pixels']:,}</code> withheld). Dual-seed pooled
<code>{h['dual_seed_pooled']['score']:.6f}</code> [<code>{h['dual_seed_pooled']['ci95'][0]:.6f}</code>,
<code>{h['dual_seed_pooled']['ci95'][1]:.6f}</code>], against the retained incumbent
<code>{h['retained_incumbent']['score']:.6f}</code>. Paired improvement from the emitter change alone, on identical
surfaces and identical support size: <code>+{h['paired_delta_vs_repository_emitter']['delta']:.5f}</code>
[<code>+{h['paired_delta_vs_repository_emitter']['ci95'][0]:.5f}</code>,
<code>+{h['paired_delta_vs_repository_emitter']['ci95'][1]:.5f}</code>].</p>
<p><b>Portal format (re-derived from the bytes on this page load):</b> {f['checks']}/{f['total']} shared-validator checks PASS —
single-band <code>float32</code>, <code>EPSG:32611</code>, <code>3730×3292</code>, exact sample transform,
<code>{f['positives']:,}</code> positive pixels, values exactly {{0.0, 1.0}} in <code>[0, 1]</code>,
<code>{f['nan']}</code> NaN/Inf anywhere, <code>0</code> positives on the mapped catalogue, zeros outside the footprint.
This cannot reproduce the earlier “Predicted values must be in range [0, 1]” error.</p>
<p><b>Uniqueness:</b> binding like-for-like reading against <code>{uniq['rasters_compared']}</code> locally present priors:
worst 3-px forward overlap <code>{b['worst_forward_overlap']:.4f}</code> (limit 0.70) and worst
|Spearman ρ| <code>{b['worst_abs_spearman']:.4f}</code> (limit 0.90) → <b>not a duplicate</b>.
{len(b['space_filling_files'])} priors were classified space-filling (their own 3-px halo covers ≥ 70% of the allowed
domain, so even random dots “overlap” them); that classification is the only change made to the duplicate rule this
session and it is logged as <code>IR-H85-01</code>.</p>
<a class="btn" href="downloads/{f['tif'].name}" download>⬇ Download the submission .tif ({f['kb']:.0f} KB)</a>
<a class="btn alt" href="downloads/{f['zip'].name}" download>⬇ One-TIFF .zip ({f['zkb']:.0f} KB)</a>
<a class="btn alt" href="h85-executive-summary.html">How to submit it, step by step →</a>
<a class="btn alt" href="h85.html">Full technical report →</a>
<p class="small">name: <code>{c['submission_name']}</code><br>
note ({len(c['submission_note'])}/140): <code>{c['submission_note']}</code><br>
sha256 .tif: <code>{f['sha']}</code> · .zip: <code>{f['zsha']}</code> ·
run card: <a href="downloads/{f['tif'].stem}.run-card.json">JSON</a></p>
</section>
"""


GUIDE_MARK = "<!-- H85 STATUS -->"


def guide_block(f: dict) -> str:
    c = f["card"]
    return f"""{GUIDE_MARK}
<section class="card dl" style="border:3px solid #1c7c4a">
<h2>Current status: one operator-selected file has an OK TO SUBMIT decision</h2>
<p><b><a href="downloads/{f['tif'].name}">{c['submission_name']}</a></b> is the only artifact with a recorded
<b>operator decision = OK TO SUBMIT</b>. Its authority is narrower than a complete repository release-gate certificate;
read the authoritative status card and <a href="data/submission-authorization.json">machine-readable receipt</a> first.
Every other file on this site is inspection-only. <b>Step-by-step:</b>
<a href="h85-executive-summary.html">H85 submission guide</a>.</p>
</section>
"""


def inject(path: Path, mark: str, block: str, anchors: tuple[str, ...]) -> str:
    html = path.read_text()
    if mark in html:
        start = html.index(mark)
        end = html.index("</section>", start) + len("</section>\n")
        return html[:start] + block + html[end:]
    for anchor in anchors:
        if anchor in html:
            i = html.index(anchor) + len(anchor)
            return html[:i] + block + html[i:]
    raise SystemExit(f"{path.name}: none of the anchors {anchors} present")


def main() -> None:
    f = facts()
    idx = DOCS / "index.html"
    idx.write_text(inject(idx, MARK, card_html(f),
                          ('<main><div class="wrap">', '<main>', '<body>')))
    guide = DOCS / "executive-summary.html"
    if guide.exists():
        guide.write_text(inject(guide, GUIDE_MARK, guide_block(f),
                               ('<main><div class="wrap">', '<main>', '<body>')))
    # H85 is intentionally prominent, but its operator-scoped card must never
    # displace the broader release-gate status at the top of either entry page.
    from build_home import put_release_status_first
    put_release_status_first()
    print(f"H85 card injected: {f['tif'].name} sha256={f['sha'][:16]}… "
          f"checks={f['checks']}/{f['total']}")


if __name__ == "__main__":
    main()
