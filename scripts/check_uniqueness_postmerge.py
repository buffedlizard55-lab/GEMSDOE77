"""Post-merge uniqueness supplement: contact-physics candidate vs the two
concurrently merged GEMSDOE77 lane rasters (PRs #1-#4), which did not exist in
this lane's local 43-raster registry at Exp-2 time.

Uses the vendor tool AS MERGED ON main (literal >0 support, forward gate) and
additionally reports the binary-dot (==1.0) reading used at Exp 2, so both
template versions' verdicts are on record. Writes
evidence/uniqueness_supplement_postmerge.json. This lane's verdict is already
negative / DO NOT SUBMIT; the supplement is a completeness record for the
selector, not a promotion attempt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src")]
from gems57 import uniqueness  # noqa: E402

MINE = ROOT / "docs/downloads/gems77-contactphys-44090dots-20261021T030258Z-ca70a971-zeros.tif"
OTHERS = [
    ROOT / "docs/downloads/gems77-h77-urank-curvature-0d868dc271bb-research.tif",
    ROOT / "docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.tif",
]
SAMPLE = ROOT / "data/bridge/sample_submission.tif"


def main() -> None:
    with rasterio.open(SAMPLE) as s:
        footprint = np.isfinite(s.read(1))
    with rasterio.open(MINE) as s:
        mine = s.read(1)
    records = [{"submission": p.stem, "file": str(p), "abs_path": str(p),
                "local_path": str(p)} for p in OTHERS]
    literal = uniqueness.compare_to_registry(str(MINE), records, footprint)
    rows = []
    for p in OTHERS:
        with rasterio.open(p) as s:
            theirs = s.read(1)
        my_dots = np.isfinite(mine) & (np.abs(mine - 1.0) < 1e-6)
        their_dots = np.isfinite(theirs) & (np.abs(theirs - 1.0) < 1e-6)
        rows.append({
            "submission": p.stem,
            "their_positive_px_literal": int((np.isfinite(theirs) & (theirs > 0)).sum()),
            "their_binary_dots": int(their_dots.sum()),
            "binary_dot_forward_overlap": uniqueness.dot_overlap(
                my_dots.astype(np.float32), their_dots.astype(np.float32)),
            "binary_dot_reverse_overlap": uniqueness.dot_overlap(
                their_dots.astype(np.float32), my_dots.astype(np.float32)),
        })
    out = {
        "kind": "uniqueness_supplement_postmerge",
        "date": "2026-10-21",
        "candidate": MINE.name,
        "why": "Two sibling lanes merged to main concurrently (PRs #1-#4); their rasters "
               "were absent from this lane's Exp-2 local registry (43 rasters).",
        "literal_tool_on_main": {k: literal.get(k) for k in
                                 ("registry_rasters_checked", "registry_rasters_expected",
                                  "worst_spearman_abs", "worst_dot_overlap",
                                  "duplicate_count", "unique", "rows")},
        "binary_dot_reading": rows,
        "note": "Vendor divergence: main carries the literal >0-support uniqueness tool; "
                "this lane's Exp 2 used GEMSDOE70's 2026-10-20 binary-dot/mutual version. "
                "Recorded as IR-77-09. Lane verdict (negative, DO NOT SUBMIT) is "
                "unchanged under either reading.",
    }
    (ROOT / "evidence" / "uniqueness_supplement_postmerge.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("candidate", "binary_dot_reading")}, indent=1)[:1200])
    print("literal worst_overlap:", literal.get("worst_dot_overlap"),
          "worst |rho|:", literal.get("worst_spearman_abs"),
          "duplicates:", literal.get("duplicate_count"))


if __name__ == "__main__":
    main()
