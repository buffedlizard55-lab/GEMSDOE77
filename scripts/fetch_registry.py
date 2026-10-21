#!/usr/bin/env python3
"""Fetch prior-submission rasters (the "registry") from the owner's sibling repos.

Two uses:
  1. ``--scored``: the rasters behind every organizer score listed in
     ``registry/live_scores.csv`` (the user's transcription of the submission
     page).  These are the live-calibration set.
  2. ``--all``: every candidate-submission raster (``.tif``) published under a
     downloads/submissions/artifacts path of any GEMSDOE repo, de-duplicated by
     git blob sha.  This is the uniqueness registry for the lane check.

Input: ``registry/registry_index.json`` (built by ``scripts/index_registry.py``
from the GitHub tree API).  Output: ``.cache/registry/<blobsha>.tif`` plus
``registry/registry_receipt.json`` with sha256 of every fetched file.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "registry"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, blob: str, size: int) -> tuple[str, str]:
    out = CACHE / f"{blob}.tif"
    if out.exists() and out.stat().st_size == size:
        return blob, sha256_file(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with open(tmp, "wb") as fh:
        subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw",
                        f"repos/buffedlizard55-lab/{repo}/git/blobs/{blob}"], check=True, stdout=fh)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"size mismatch {repo}:{blob}")
    tmp.rename(out)
    return blob, sha256_file(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "registry" / "registry_index.json"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    idx = json.loads(Path(args.index).read_text())
    todo = {}
    for e in idx["rasters"]:
        if e["path"].lower().endswith((".tif", ".tiff")):
            todo[e["blob"]] = (e["repo"], e["blob"], e["size"])
    receipts = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for blob, sha in ex.map(lambda t: fetch(*t), todo.values()):
            receipts[blob] = sha
    (ROOT / "registry" / "registry_receipt.json").write_text(json.dumps(
        {"n_files": len(receipts), "sha256_by_blob": receipts}, indent=1, sort_keys=True))
    print(f"fetched/verified {len(receipts)} rasters")


if __name__ == "__main__":
    main()
