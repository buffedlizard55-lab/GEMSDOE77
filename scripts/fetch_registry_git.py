#!/usr/bin/env python3
"""Fetch the uniqueness registry rasters with git (sparse, blobless clones) instead of the REST API.

Why: ``scripts/fetch_registry.py`` pulls every blob through ``gh api`` (one REST call
per raster, 812 calls). Parallel sessions share the same 5,000/h core quota, and on
2026-10-21 the quota fell from 3,221 to 1,091 in fifteen minutes (IR-79-02). Git
smart-HTTP transfers are not metered by that quota, so this script clones each
sibling repository with ``--filter=blob:none --depth 1``, sparse-checks-out only the
directories that hold candidate rasters, and copies every ``*.tif`` into
``.cache/registry/<git-blob-sha>.tif`` -- the exact layout the shared tools expect.

Rasters whose blob sha is listed in ``registry/registry_index.json`` are recorded as
``indexed``; rasters published after the index was built are kept too and recorded
as ``post_index`` (they widen the uniqueness check rather than narrowing it).

Output: ``registry/registry_receipt_git.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "registry"
OWNER = "buffedlizard55-lab"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path | None = None, timeout: int = 600) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout


def fetch_repo(repo: str, dirs: list[str], wanted_blobs: set[str], log) -> dict:
    out = {"repo": repo, "dirs": dirs, "indexed": 0, "post_index": 0, "skipped_large": 0, "error": None}
    tmp = Path(tempfile.mkdtemp(prefix=f"reg_{repo}_", dir=str(ROOT / ".cache")))
    try:
        url = f"https://github.com/{OWNER}/{repo}.git"
        git("clone", "--quiet", "--depth", "1", "--filter=blob:none", "--no-checkout", "--sparse", url, str(tmp / "r"))
        wd = tmp / "r"
        git("sparse-checkout", "set", "--no-cone", *dirs, cwd=wd)
        git("checkout", "--quiet", cwd=wd)
        head = git("rev-parse", "HEAD", cwd=wd).strip()
        out["head"] = head
        ls = git("ls-files", "-s", cwd=wd)
        for line in ls.splitlines():
            meta, path = line.split("\t", 1)
            blob = meta.split()[1]
            if not path.lower().endswith((".tif", ".tiff")):
                continue
            src = wd / path
            if not src.exists():
                continue
            if src.stat().st_size > 5_000_000 and blob not in wanted_blobs:
                out["skipped_large"] += 1
                continue
            dest = CACHE / f"{blob}.tif"
            if not dest.exists():
                shutil.copyfile(src, dest)
            if blob in wanted_blobs:
                out["indexed"] += 1
            else:
                out["post_index"] += 1
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"{repo:12s} indexed={out['indexed']:3d} post_index={out['post_index']:3d} err={out['error']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(ROOT / "registry" / "registry_index.json"))
    ap.add_argument("--extra-repos", default="GEMSDOE73,GEMSDOE74,GEMSDOE75,GEMSDOE76,GEMSDOE78,GEMSDOE79,GEMSDOE80,GEMSDOE81,GEMSDOE82",
                    help="siblings created after the index; scanned at docs/downloads + submissions")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    idx = json.loads(Path(args.index).read_text())
    by_repo: dict[str, set[str]] = defaultdict(set)
    wanted: set[str] = set()
    for e in idx["rasters"]:
        if e["path"].lower().endswith((".tif", ".tiff")):
            by_repo[e["repo"]].add(e["path"].rsplit("/", 1)[0] + "/")
            wanted.add(e["blob"])
    for r in [x for x in args.extra_repos.split(",") if x]:
        by_repo.setdefault(r, set()).update({"docs/downloads/", "submissions/", "docs/submissions/"})

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    results = [fetch_repo(repo, sorted(dirs), wanted, log) for repo, dirs in sorted(by_repo.items())]
    have = {p.stem for p in CACHE.glob("*.tif")}
    missing = sorted(wanted - have)
    receipt = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "git clone --depth 1 --filter=blob:none --sparse per sibling repo; *.tif copied to .cache/registry/<blob>.tif",
        "index_rasters": len(wanted),
        "index_rasters_present": len(wanted & have),
        "index_rasters_missing": missing,
        "cache_rasters_total": len(have),
        "elapsed_s": round(time.time() - t0, 1),
        "repos": results,
        "sha256_by_blob": {p.stem: sha256_file(p) for p in sorted(CACHE.glob("*.tif"))},
    }
    (ROOT / "registry" / "registry_receipt_git.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    log(f"done: {len(wanted & have)}/{len(wanted)} indexed rasters present, {len(have)} total, "
        f"{len(missing)} missing, {receipt['elapsed_s']}s")


if __name__ == "__main__":
    main()
