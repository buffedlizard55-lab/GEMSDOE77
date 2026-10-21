#!/usr/bin/env python3
"""Restore the competition inputs from the owner's integrity-pinned GitHub mirrors.

Why this exists
---------------
The organizer copies of ``training_features.tif``, ``labels.tif`` and
``sample_submission.tif`` are login-walled at
https://www.drivendata.org/competitions/306/competition-doe-gems/data/ and this
sandbox cannot reach drivendata.org, dropbox.com or raw.githubusercontent.com.
It *can* reach api.github.com, so every file is fetched through the Git blob API
(``GET /repos/{owner}/{repo}/git/blobs/{sha}`` with the raw media type, which
serves blobs up to 100 MB) and its sha256 is checked against
``registry/data_manifest.json``.

A pin proves the bytes are the same bytes the owner's earlier sessions used;
it does NOT prove they are the organizer's bytes.  That limitation is stated on
the site and in the README.

Usage::

    python scripts/restore_data.py                # core + external
    python scripts/restore_data.py --group core   # only the four core files

Files land in ``.cache/gems_data/`` (git-ignored; never committed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "registry" / "data_manifest.json"
DEST = ROOT / ".cache" / "gems_data"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_index(repo: str, ref: str, cache: dict) -> dict:
    key = (repo, ref)
    if key not in cache:
        out = subprocess.run(
            ["gh", "api", f"repos/{repo}/git/trees/{ref}?recursive=1"],
            check=True, capture_output=True, text=True).stdout
        cache[key] = {t["path"]: t for t in json.loads(out)["tree"] if t["type"] == "blob"}
    return cache[key]


def fetch_blob(repo: str, blob_sha: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        subprocess.run(
            ["gh", "api", "-H", "Accept: application/vnd.github.raw",
             f"repos/{repo}/git/blobs/{blob_sha}"],
            check=True, stdout=fh)


def restore(entry: dict, trees: dict) -> dict:
    dest = DEST / entry["dest"]
    if dest.exists() and dest.stat().st_size == entry["bytes"] and sha256_file(dest) == entry["sha256"]:
        return {"id": entry["id"], "status": "present", "sha256_ok": True}
    idx = tree_index(entry["repo"], entry["ref"], trees)
    paths = entry.get("parts") or [entry["path"]]
    tmp_parts = []
    for p in paths:
        meta = idx[p]
        tmp = DEST / ".parts" / (Path(p).name)
        fetch_blob(entry["repo"], meta["sha"], tmp)
        if tmp.stat().st_size != meta["size"]:
            raise RuntimeError(f"size mismatch for {p}: {tmp.stat().st_size} != {meta['size']}")
        tmp_parts.append(tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        for tmp in tmp_parts:
            out.write(tmp.read_bytes())
            tmp.unlink()
    got = sha256_file(dest)
    ok = got == entry["sha256"] and dest.stat().st_size == entry["bytes"]
    if not ok:
        raise RuntimeError(f"sha256 mismatch for {entry['id']}: {got} != {entry['sha256']}")
    return {"id": entry["id"], "status": "restored", "sha256_ok": True, "sha256": got,
            "bytes": dest.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["core", "external", "all"], default="all")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    trees: dict = {}
    receipts = []
    for e in manifest["files"]:
        if args.group != "all" and e.get("group") != args.group:
            continue
        r = restore(e, trees)
        receipts.append(r)
        print(json.dumps(r))
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "restore_receipt.json").write_text(json.dumps(
        {"storage_root": str(DEST.relative_to(ROOT)), "files": receipts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
