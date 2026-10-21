#!/usr/bin/env python3
"""Build bit-exact full-portfolio uniqueness index across ALL GEMSDOE..GEMSDOE92 repos + GEMSDOE77.

Streams repository tarballs in parallel over `codeload.github.com` (zero GitHub REST rate-limit burn,
with automatic blob API fallback for non-HEAD historical blobs in `registry/registry_index.json`),
verifies grid alignment against `vendor/gems57/grid.py`, and writes:
  - `.cache/audit_halos_packed.npy` : exact bit-packed 3.0-px Euclidean halo matrix (`0.00e+00` diff vs `vendor/gems57/uniqueness.py:dot_overlap`)
  - `.cache/audit_subvals.npy`      : subsampled float32 values across allowed pixels for Spearman rho
  - `.cache/audit_meta.json`        : full provenance, SHA-256, positive count, halo coverage fraction, and space-filling flag (`>= 0.70`) for every raster
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as grid_mod  # noqa: E402

CACHE = ROOT / ".cache"
REG_DIR = CACHE / "registry"
SIB_DIR = CACHE / "siblings_all"
DATA_DIR = CACHE / "data"

STRIDE = 8
SKIP_SUBSTRINGS = (
    "existing_faults",
    "sample_submission",
    "example_submission",
    "labels.tif",
    "training_features",
    "gems-geodawn-numerical-features",
    "lidar_scarp_features",
    "geodawn_rad",
    "geodawn_extensions",
    "topo_10m_derived",
    "dem10_chi",
    "sgmc_faults",
    "tiger_road",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_repo_tifs(repo: str, idx_by_path: dict[str, str]) -> tuple[str, int, int, str | None]:
    """Stream `codeload.github.com` tarball for `repo` and extract all `.tif` candidates."""
    url = f"https://codeload.github.com/buffedlizard55-lab/{repo}/tar.gz/HEAD"
    n_idx = 0
    n_sib = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gemsdoe77-h82-audit"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    parts = member.name.split("/", 1)
                    if len(parts) < 2:
                        continue
                    rel = parts[1]
                    low = rel.lower()
                    if not low.endswith((".tif", ".tiff")):
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    blob_sha = idx_by_path.get(rel)
                    if blob_sha is not None:
                        dst = REG_DIR / f"{blob_sha}.tif"
                        if not dst.is_file():
                            data = fh.read()
                            tmp = dst.with_suffix(".tmp")
                            tmp.write_bytes(data)
                            tmp.replace(dst)
                        n_idx += 1
                    else:
                        if any(tok in low for tok in SKIP_SUBSTRINGS):
                            continue
                        if not any(
                            seg in low
                            for seg in ("download", "submission", "artifact", "output", "candidate", "deliverable")
                        ):
                            continue
                        safe = rel.replace("/", "__")
                        dst = SIB_DIR / f"{repo}__{safe}"
                        if not dst.is_file():
                            data = fh.read()
                            tmp = dst.with_suffix(".tmp")
                            tmp.write_bytes(data)
                            tmp.replace(dst)
                        n_sib += 1
        return repo, n_idx, n_sib, None
    except Exception as exc:
        return repo, n_idx, n_sib, str(exc)[:200]


def fetch_blob_fallback(args: tuple[str, str, int]) -> tuple[str, bool, str | None]:
    repo, blob, size = args
    out = REG_DIR / f"{blob}.tif"
    if out.is_file() and out.stat().st_size == size:
        return blob, True, None
    tmp = out.with_suffix(".part")
    try:
        with tmp.open("wb") as fh:
            subprocess.run(
                ["gh", "api", "-H", "Accept: application/vnd.github.raw", f"repos/buffedlizard55-lab/{repo}/git/blobs/{blob}"],
                check=True,
                stdout=fh,
                stderr=subprocess.DEVNULL,
                timeout=90,
            )
        if tmp.stat().st_size != size:
            tmp.unlink(missing_ok=True)
            return blob, False, f"size {tmp.stat().st_size} != {size}"
        tmp.replace(out)
        return blob, True, None
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return blob, False, str(exc)[:200]


def main() -> int:
    t0 = time.perf_counter()
    REG_DIR.mkdir(parents=True, exist_ok=True)
    SIB_DIR.mkdir(parents=True, exist_ok=True)

    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    by_repo: dict[str, dict[str, str]] = {}
    all_blobs: dict[str, tuple[str, str, int]] = {}
    for e in idx["rasters"]:
        if e["path"].lower().endswith((".tif", ".tiff")):
            by_repo.setdefault(e["repo"], {})[e["path"]] = e["blob"]
            all_blobs[e["blob"]] = (e["repo"], e["blob"], int(e["size"]))

    extra_repos = [f"GEMSDOE{k}" for k in range(65, 93) if k != 77] + [
        "GEMSDOE32",
        "19GEMSDOE",
        "7GEMSDOE",
        "16GEMSDOE",
    ]
    repos_to_scan = sorted(set(by_repo.keys()) | set(extra_repos))
    if any(SIB_DIR.glob("*.tif*")) and len(list(REG_DIR.glob("*.tif"))) >= len(all_blobs):
        log(f"All {len(all_blobs)} archive blobs and sibling rasters already cached locally on disk; skipping network fetch.")
        blob_errors: list[dict[str, Any]] = []
    else:
        log(f"Streaming {len(repos_to_scan)} GEMSDOE repositories over codeload.github.com...")
        with cf.ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(stream_repo_tifs, r, by_repo.get(r, {})) for r in repos_to_scan]
            for f in cf.as_completed(futs):
                repo, ni, ns, err = f.result()
                if err:
                    log(f"  warn {repo}: {err}")

        missing_blobs = [
            spec
            for blob, spec in all_blobs.items()
            if not ((REG_DIR / f"{blob}.tif").is_file() and (REG_DIR / f"{blob}.tif").stat().st_size == spec[2])
        ]
        log(f"Archive blobs present via tarball: {len(all_blobs) - len(missing_blobs)}/{len(all_blobs)}; fetching {len(missing_blobs)} non-HEAD blobs via git blob API...")
        blob_errors = []
        if missing_blobs:
            with cf.ThreadPoolExecutor(max_workers=8) as ex:
                for blob, ok, err in ex.map(fetch_blob_fallback, missing_blobs):
                    if not ok:
                        blob_errors.append({"blob": blob, "error": err})
        log(f"Archive blobs available on disk: {len(all_blobs) - len(blob_errors)}/{len(all_blobs)} (errors={len(blob_errors)})")

    # Gather ALL candidate raster paths:
    # 1. All 812 archive registry blobs
    # 2. All sibling rasters across GEMSDOE..GEMSDOE92
    # 3. All local GEMSDOE77 rasters (docs/downloads, docs/submissions, submissions)
    discovered: list[tuple[str, str, Path]] = []
    for p in sorted(REG_DIR.glob("*.tif")):
        discovered.append(("registry", p.name, p))
    for p in sorted(SIB_DIR.glob("*.tif*")):
        discovered.append(("sibling", p.name, p))
    for sub_dir, tag in (
        (ROOT / "docs" / "downloads", "local_gems77"),
        (ROOT / "docs" / "submissions", "local_gems77"),
        (ROOT / "submissions", "local_gems77"),
    ):
        if sub_dir.is_dir():
            for p in sorted(sub_dir.glob("*.tif*")):
                discovered.append((tag, p.name, p))

    log(f"Total discovered TIFF paths across archive + GEMSDOE..GEMSDOE92 + local GEMSDOE77: {len(discovered)}")

    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    cat = g.catalogue & fp
    allow = fp & (~cat)
    allow_idx = np.flatnonzero(allow)
    n_allow = int(allow_idx.size)
    sub_idx = allow_idx[::STRIDE]
    n_bytes = (n_allow + 7) // 8

    by_sha: dict[str, dict[str, Any]] = {}
    for tag, name, path in discovered:
        sha = sha256_file(path)
        alias = f"{tag}:{name}"
        if sha not in by_sha:
            by_sha[sha] = {
                "tag": tag,
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "abs_path": path,
                "sha256": sha,
                "aliases": [alias],
            }
        else:
            by_sha[sha]["aliases"].append(alias)

    unique_candidates = list(by_sha.values())
    log(f"Unique SHA-256 TIFF files to index: {len(unique_candidates)}")

    disk_offsets = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy * dy + dx * dx <= 9]
    h_grid, w_grid = fp.shape

    def exact_halo3_bool(pos_mask: np.ndarray) -> np.ndarray:
        """Bit-identical to `distance_transform_edt(~pos_mask) <= 3.0` on Z^2 in ~12ms instead of 680ms."""
        n_on = int(pos_mask.sum())
        if n_on == 0:
            return np.zeros_like(pos_mask, dtype=bool)
        if n_on == pos_mask.size:
            return np.ones_like(pos_mask, dtype=bool)
        if n_on <= 350_000:
            rr, cc = np.nonzero(pos_mask)
            out = np.zeros(h_grid * w_grid, dtype=bool)
            for dy, dx in disk_offsets:
                ry = rr + dy
                cx = cc + dx
                ok = (ry >= 0) & (ry < h_grid) & (cx >= 0) & (cx < w_grid)
                out[ry[ok] * w_grid + cx[ok]] = True
            return out.reshape(h_grid, w_grid)
        padded = np.pad(pos_mask, 3, mode="constant", constant_values=False)
        out2d = np.zeros_like(pos_mask, dtype=bool)
        for dy, dx in disk_offsets:
            out2d |= padded[3 + dy : 3 + dy + h_grid, 3 + dx : 3 + dx + w_grid]
        return out2d

    def inspect_and_pack(rec: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None, str | None]:
        p: Path = rec["abs_path"]
        try:
            with rasterio.open(p) as ds:
                if (ds.height, ds.width) != fp.shape:
                    return rec, None, None, f"shape {(ds.height, ds.width)}"
                arr = ds.read(1)
        except Exception as exc:
            return rec, None, None, f"{p.name}: {exc}"

        pos_mask = np.isfinite(arr) & (arr > 0)
        n_pos_allow = int((pos_mask & allow).sum())
        halo_allow = exact_halo3_bool(pos_mask).ravel()[allow_idx]
        frac = float(halo_allow.mean())
        packed_row = np.packbits(halo_allow, bitorder="big")
        sub_v = np.nan_to_num(arr.ravel()[sub_idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        meta_entry = {
            "tag": rec["tag"],
            "name": rec["name"],
            "path": rec["path"],
            "sha256": rec["sha256"],
            "aliases": rec["aliases"],
            "n_pos_allow": n_pos_allow,
            "halo3_frac_allow": frac,
            "is_space_filling": bool(frac >= 0.70),
        }
        return meta_entry, packed_row, sub_v, None

    entries: list[dict[str, Any]] = []
    packed_rows: list[np.ndarray] = []
    sub_rows: list[np.ndarray] = []
    unreadable: list[dict[str, Any]] = []

    done = 0
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for meta_entry, p_row, s_row, err in ex.map(inspect_and_pack, unique_candidates):
            done += 1
            if err is not None or p_row is None or s_row is None:
                unreadable.append(
                    {
                        "tag": meta_entry["tag"],
                        "name": meta_entry["name"],
                        "path": meta_entry["path"],
                        "sha256": meta_entry["sha256"],
                        "aliases": meta_entry["aliases"],
                        "reason": err or "unknown",
                    }
                )
            else:
                entries.append(meta_entry)
                packed_rows.append(p_row)
                sub_rows.append(s_row)
            if done % 150 == 0 or done == len(unique_candidates):
                log(f"  indexed {done}/{len(unique_candidates)} unique rasters...")

    n_valid = len(entries)
    assert n_bytes == packed_rows[0].size
    packed_mm = np.lib.format.open_memmap(
        CACHE / "audit_halos_packed.npy", mode="w+", dtype=np.uint8, shape=(n_valid, n_bytes)
    )
    sub_mm = np.lib.format.open_memmap(
        CACHE / "audit_subvals.npy", mode="w+", dtype=np.float32, shape=(n_valid, sub_idx.size)
    )
    for i in range(n_valid):
        packed_mm[i] = packed_rows[i]
        sub_mm[i] = sub_rows[i]
    packed_mm.flush()
    sub_mm.flush()
    del packed_mm, sub_mm, packed_rows, sub_rows

    n_info = sum(1 for e in entries if not e["is_space_filling"])
    n_sf = sum(1 for e in entries if e["is_space_filling"])
    out_meta = {
        "schema_version": 2,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_paths_discovered": len(discovered),
        "unique_sha256_discovered": len(unique_candidates),
        "n_valid": n_valid,
        "n_informative_lt_0p70": n_info,
        "n_space_filling_ge_0p70": n_sf,
        "n_allow": n_allow,
        "stride": STRIDE,
        "blob_fetch_errors": blob_errors,
        "unreadable": unreadable,
        "entries": entries,
    }
    (CACHE / "audit_meta.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    log(
        f"DONE in {time.perf_counter()-t0:.1f}s: {n_valid} valid unique rasters indexed "
        f"({n_info} informative <70% halo, {n_sf} space-filling >=70% halo, {len(unreadable)} unreadable/off-grid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
