#!/usr/bin/env python3
"""Post-merge uniqueness addendum for H82-RRD: exact 3px forward overlap of the
frozen H82-RRD dots vs rasters merged by parallel sessions after Exp 3 was run.
Read-only audit; does not modify the candidate or its run card."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import uniqueness as UQ  # noqa: E402

MINE = ROOT / "docs/downloads/gems77-h82-rrd-discord-37654dots-20261021-c069b8d8-zeros.tif"


def main():
    with rasterio.open(MINE) as ds:
        mine = ds.read(1)
    seen, rows = set(), []
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")) + sorted((ROOT / "submissions").glob("*.tif")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in seen or p.name == MINE.name:
            continue
        seen.add(h)
        with rasterio.open(p) as ds:
            a = ds.read(1)
        a = np.where(np.isfinite(a), a, 0).astype(np.float32)
        rows.append({"file": p.name, "sha256": h, "their_dots": int((a > 0).sum()),
                     "forward_overlap_3px": UQ.dot_overlap(mine, a)})
    rows.sort(key=lambda r: -r["forward_overlap_3px"])
    out = {"candidate": MINE.name,
           "scope": "all sha-deduped local docs/downloads + submissions TIFFs present after merging origin/main (parallel H82/BGB/H77V lanes included); NOT complete family registry",
           "n_compared": len(rows),
           "max_forward_overlap_3px": rows[0]["forward_overlap_3px"] if rows else 0.0,
           "firings_gt070": [r for r in rows if r["forward_overlap_3px"] > 0.70],
           "top10": rows[:10], "rows": rows}
    (ROOT / "evidence" / "h82rrd_uniqueness_postmerge.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "h82rrd_uniqueness_postmerge.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"compared {len(rows)}; max overlap {out['max_forward_overlap_3px']:.4f}; firings>0.70: {len(out['firings_gt070'])}")


if __name__ == "__main__":
    main()
