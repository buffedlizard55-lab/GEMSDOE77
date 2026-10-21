"""Build and verify the bit-packed 300 m halo & subsampled value index for `PortfolioCapacityIndex`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import metric as metric_mod  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_DISK3_OFFSETS = [
    (dy, dx)
    for dy in range(-3, 4)
    for dx in range(-3, 4)
    if dy * dy + dx * dx <= 9
]


def _fast_near_3px(support: np.ndarray) -> np.ndarray:
    """Bit-identical to `gems57.uniqueness._near(support, 3.0)` via 29 vectorized slice shifts."""
    h, w = support.shape
    pad = 3
    p = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    p[pad : pad + h, pad : pad + w] = support
    out = np.zeros((h, w), dtype=bool)
    for dy, dx in _DISK3_OFFSETS:
        out |= p[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
    return out


def ensure_h77h_audit_index(
    cache_dir: Path,
    foot: np.ndarray,
    cat: np.ndarray,
    exclude_substr: str = "h77h",
    stride: int = 29,
) -> dict[str, Any]:
    """Ensure `audit_halos_packed.npy`, `audit_subvals.npy`, and `audit_meta.json` exist in `cache_dir`."""
    packed_path = cache_dir / "audit_halos_packed.npy"
    subvals_path = cache_dir / "audit_subvals.npy"
    meta_path = cache_dir / "audit_meta.json"

    allow_idx = np.flatnonzero(foot & (~cat))
    n_allow = int(allow_idx.size)
    n_packed_bytes = (n_allow + 7) // 8
    sub_fp_idx = allow_idx[::stride]
    n_sub = int(sub_fp_idx.size)

    if packed_path.is_file() and subvals_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if int(meta.get("n_allow", 0)) == n_allow and int(meta.get("stride", 0)) == stride:
            return meta

    candidate_sources: list[tuple[str, Path]] = []
    for p in sorted((ROOT / ".cache" / "registry").glob("*.tif")):
        candidate_sources.append(("witness", p))
    for p in sorted((ROOT / ".cache" / "data" / "registry" / "rasters").glob("*.tif")):
        candidate_sources.append(("canonical_registry", p))
    for p in sorted((ROOT / ".cache" / "h77e_registry").glob("*.tif")):
        candidate_sources.append(("archive_registry", p))
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")):
        if exclude_substr not in p.name and not p.name.endswith("-nan.tif"):
            candidate_sources.append(("local_gems77", p))
    for p in sorted((ROOT / "submissions").glob("*.tif")):
        if exclude_substr not in p.name and not p.name.endswith("-nan.tif"):
            candidate_sources.append(("local_submissions", p))

    by_sha: dict[str, dict[str, Any]] = {}
    for tag, p in candidate_sources:
        if not p.is_file() or exclude_substr in p.name:
            continue
        sha = _sha256_file(p)
        alias = f"{tag}:{p.name}"
        if sha in by_sha:
            if alias not in by_sha[sha]["aliases"]:
                by_sha[sha]["aliases"].append(alias)
            continue
        by_sha[sha] = {
            "tag": tag,
            "name": p.name,
            "path": str(p),
            "sha256": sha,
            "aliases": [alias],
        }

    entries: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    packed_rows: list[np.ndarray] = []
    subval_rows: list[np.ndarray] = []

    for sha, rec in by_sha.items():
        p = Path(rec["path"])
        try:
            with rasterio.open(p) as ds:
                if ds.shape != foot.shape:
                    unreadable.append({**rec, "reason": f"shape {ds.shape}"})
                    continue
                arr = ds.read(1).astype(np.float32)
        except Exception as exc:
            unreadable.append({**rec, "reason": str(exc)[:160]})
            continue

        t_mask = np.isfinite(arr) & (arr > 0)
        n_pos_allow = int((t_mask & foot & (~cat)).sum())
        halo3 = _fast_near_3px(t_mask) if t_mask.any() else np.zeros(foot.shape, dtype=bool)
        halo_allow = halo3.ravel()[allow_idx]
        halo_frac = float(halo_allow.mean())
        is_sf = bool(halo_frac >= 0.70)

        packed_rows.append(np.packbits(halo_allow.astype(np.uint8)))
        sub_v = np.where(np.isfinite(arr.ravel()[sub_fp_idx]), arr.ravel()[sub_fp_idx], 0.0).astype(np.float32)
        subval_rows.append(sub_v)

        entries.append({
            **rec,
            "n_pos_allow": n_pos_allow,
            "halo3_frac_allow": round(halo_frac, 6),
            "is_space_filling": is_sf,
        })

    packed_mat = np.vstack(packed_rows) if packed_rows else np.zeros((0, n_packed_bytes), dtype=np.uint8)
    subvals_mat = np.vstack(subval_rows) if subval_rows else np.zeros((0, n_sub), dtype=np.float32)
    np.save(packed_path, packed_mat)
    np.save(subvals_path, subvals_mat)

    meta = {
        "n_valid": len(entries),
        "n_allow": n_allow,
        "stride": stride,
        "entries": entries,
        "unreadable": unreadable,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta
