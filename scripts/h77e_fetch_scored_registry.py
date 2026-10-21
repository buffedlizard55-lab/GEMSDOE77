#!/usr/bin/env python3
"""Fetch top scored registry rasters via git sparse clone (REST API is rate-limited).

Scope (stated, not hidden): every registry_index scored raster with
OWNER-REPORT score >= 0.14, plus the GEMSDOE32 H33 file. This is an
informative uniqueness subset, NOT the full 812-raster registry; the lane
therefore cannot claim protocol_unique (fail-closed, Policy I17 also applies).
Outputs .cache/h77e_registry/<repo>__<name>.tif and a receipt JSON.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache" / "h77e_registry"
CLONES = ROOT / ".cache" / "h77e_repos"
MIN_SCORE = 0.14


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600, **kw)


def main():
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    want: dict[str, list[dict]] = {}
    for e in idx["scored"]:
        if e["public_score"] >= MIN_SCORE:
            want.setdefault(e["repo"], []).append(e)
    OUT.mkdir(parents=True, exist_ok=True)
    CLONES.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for repo, entries in sorted(want.items()):
        dest = CLONES / repo
        try:
            if not dest.exists():
                run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                     f"https://github.com/buffedlizard55-lab/{repo}.git", str(dest)])
            paths = sorted({str(Path(e["path"]).parent) for e in entries})
            run(["git", "-C", str(dest), "sparse-checkout", "set", *paths])
            run(["git", "-C", str(dest), "checkout"])
            for e in entries:
                src = dest / e["path"]
                if src.is_file():
                    dst = OUT / f"{repo}__{src.name}"
                    shutil.copyfile(src, dst)
                    receipt[e["submission_name"]] = {"file": str(dst), "repo": repo,
                                                     "path": e["path"], "score": e["public_score"]}
                else:
                    receipt[e["submission_name"]] = {"error": "not found after sparse checkout",
                                                     "repo": repo, "path": e["path"]}
        except Exception as ex:  # keep going; scope is documented either way
            receipt[f"{repo}__clone"] = {"error": str(ex)[:300]}
            print(f"FAILED {repo}: {ex}", flush=True)
        print(f"done {repo}", flush=True)
    (ROOT / "evidence" / "h77e_registry_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"fetched {sum(1 for v in receipt.values() if 'file' in v)} scored rasters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
