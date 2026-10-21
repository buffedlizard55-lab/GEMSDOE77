#!/usr/bin/env python3
"""Restore SHA-256-pinned competition mirrors listed in registry/data_manifest.json.

Mirrors are owner-supplied public GitHub blobs (DrivenData files are login-walled).
Integrity-pinned, NOT organizer-authenticated. Fails closed on any hash mismatch.
Adapted from GEMSDOE32 scripts/restore_data.py (same manifest schema).
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(repo: str, ref: str, path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with tmp.open("wb") as out:
            subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}",
                            "-H", "Accept: application/vnd.github.raw"],
                           stdout=out, check=True)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

def restore_entry(entry: dict, root: Path) -> dict:
    dest = root / entry["dest"]
    if dest.exists() and sha256_file(dest) == entry["sha256"]:
        return {"id": entry["id"], "dest": entry["dest"], "status": "present",
                "bytes": dest.stat().st_size, "sha256": entry["sha256"]}
    if "parts" in entry:
        tmp = dest.with_suffix(".assembling")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as out:
            for part in entry["parts"]:
                p = root / "raw_parts" / Path(part).name
                fetch(entry["repo"], entry["ref"], part, p)
                with p.open("rb") as src:
                    shutil.copyfileobj(src, out, 1 << 20)
                p.unlink()
        got = sha256_file(tmp)
        if got != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"HASH MISMATCH {entry['id']}: {got} != {entry['sha256']}")
        tmp.replace(dest)
    else:
        fetch(entry["repo"], entry["ref"], entry["path"], dest)
        got = sha256_file(dest)
        if got != entry["sha256"]:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"HASH MISMATCH {entry['id']}: {got} != {entry['sha256']}")
    return {"id": entry["id"], "dest": entry["dest"], "status": "restored",
            "bytes": dest.stat().st_size, "sha256": entry["sha256"]}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", default="all", help="comma-separated manifest ids or 'all'")
    ap.add_argument("--manifest", default=str(ROOT / "registry" / "data_manifest.json"))
    ap.add_argument("--target-dir", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    root = Path(args.target_dir); root.mkdir(parents=True, exist_ok=True)
    want = None if args.ids == "all" else set(args.ids.split(","))
    results = []
    for e in man["files"]:
        if want is None or e["id"] in want:
            r = restore_entry(e, root); results.append(r)
            print(f"{r['status']:>8}  {e['dest']:60s} sha256={r['sha256'][:12]}…", flush=True)
    receipt = {"schema_version": 1, "verified_files_count": len(results),
               "storage_root": str(root), "files": results}
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "restore_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(results)} files -> data/restore_receipt.json")

if __name__ == "__main__":
    main()
