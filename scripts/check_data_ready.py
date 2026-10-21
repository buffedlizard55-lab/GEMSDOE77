#!/usr/bin/env python3
"""Read-only check that locally restored data matches its manifest and receipt.

The tracked restore receipt records a previous restore operation; it does not
prove that ignored ``data/raw`` files exist in the current checkout. This tool
checks the bytes currently on disk and the corresponding receipt rows without
modifying either.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing file"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable JSON: {exc}"


def check_data_ready(
    manifest_path: Path,
    data_root: Path,
    receipt_path: Path,
    requested_ids: str = "all",
) -> dict[str, Any]:
    """Return a fail-closed audit of current bytes against manifest + receipt."""
    manifest, manifest_error = _load_json(manifest_path)
    if manifest_error:
        return {"ready": False, "error": f"manifest {manifest_error}", "files": []}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return {"ready": False, "error": "manifest must be a JSON object with a files list", "files": []}

    entries = manifest["files"]
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            return {"ready": False, "error": "manifest contains an entry without a string id", "files": []}
        if entry["id"] in by_id:
            return {"ready": False, "error": f"duplicate manifest id: {entry['id']}", "files": []}
        by_id[entry["id"]] = entry

    if requested_ids == "all":
        selected = list(by_id)
    else:
        selected = [item.strip() for item in requested_ids.split(",") if item.strip()]
        if not selected:
            return {"ready": False, "error": "--ids must be 'all' or a comma-separated non-empty id list", "files": []}
        unknown = sorted(set(selected) - set(by_id))
        if unknown:
            return {"ready": False, "error": f"unknown manifest id(s): {', '.join(unknown)}", "files": []}
        # Preserve caller order while avoiding duplicate work.
        selected = list(dict.fromkeys(selected))

    receipt, receipt_error = _load_json(receipt_path)
    receipt_by_id: dict[str, dict[str, Any]] = {}
    if receipt_error is None:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("files"), list):
            receipt_error = "receipt must be a JSON object with a files list"
        else:
            for row in receipt["files"]:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    receipt_error = "receipt contains an entry without a string id"
                    break
                if row["id"] in receipt_by_id:
                    receipt_error = f"receipt contains duplicate id: {row['id']}"
                    break
                receipt_by_id[row["id"]] = row

    data_root = data_root.resolve()
    results: list[dict[str, Any]] = []
    for file_id in selected:
        entry = by_id[file_id]
        dest = entry.get("dest")
        expected_sha = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        if not isinstance(dest, str) or not isinstance(expected_sha, str) or not isinstance(expected_bytes, int):
            return {"ready": False, "error": f"manifest entry {file_id} lacks dest/sha256/bytes", "files": results}

        path = (data_root / dest).resolve()
        if path != data_root and data_root not in path.parents:
            return {"ready": False, "error": f"manifest destination escapes data root: {file_id}", "files": results}

        issues: list[str] = []
        actual_bytes: int | None = None
        actual_sha: str | None = None
        if not path.is_file():
            issues.append("current file is missing")
        else:
            actual_bytes = path.stat().st_size
            actual_sha = _sha256(path)
            if actual_bytes != expected_bytes:
                issues.append(f"byte count mismatch ({actual_bytes} != {expected_bytes})")
            if actual_sha != expected_sha:
                issues.append("sha256 mismatch")

        if receipt_error:
            issues.append(f"restore receipt {receipt_error}")
        else:
            row = receipt_by_id.get(file_id)
            if row is None:
                issues.append("no matching restore receipt row")
            else:
                if row.get("dest") != dest:
                    issues.append("receipt destination mismatch")
                if row.get("sha256") != expected_sha:
                    issues.append("receipt sha256 mismatch")
                if row.get("bytes") != expected_bytes:
                    issues.append("receipt byte count mismatch")
                if row.get("status") not in {"restored", "present"}:
                    issues.append("receipt status is not restored/present")

        results.append({
            "id": file_id,
            "dest": dest,
            "present": path.is_file(),
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "ready": not issues,
            "issues": issues,
        })

    return {
        "ready": all(row["ready"] for row in results),
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "receipt": str(receipt_path),
        "selected_count": len(results),
        "files": results,
        "note": "A SHA-pinned owner mirror proves byte integrity against that manifest, not organizer authentication.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="registry/data_manifest.json")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--receipt", default="data/restore_receipt.json")
    parser.add_argument("--ids", default="all", help="'all' or comma-separated manifest IDs")
    args = parser.parse_args()

    result = check_data_ready(
        _repo_path(args.manifest),
        _repo_path(args.data_root),
        _repo_path(args.receipt),
        args.ids,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
