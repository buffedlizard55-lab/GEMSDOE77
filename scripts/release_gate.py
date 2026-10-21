#!/usr/bin/env python3
"""Fail-closed competition release gate for every published GeoTIFF.

This does not validate scientific claims or grant selector approval. It combines
already-recorded lane decisions so a format PASS can never be mistaken for
permission to upload. The generated receipt is suitable for the Pages site.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANES = ROOT / "docs" / "data" / "lanes.json"
DEFAULT_OUT = ROOT / "docs" / "data" / "submission-authorization.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assess(row: dict, docs: Path) -> dict:
    path = docs / row["file"]
    checks = {
        "artifact_exists": path.is_file(),
        "format_valid": row.get("format_valid") is True,
        "literal_protocol_unique": row.get("protocol_unique") is True,
        "beats_accepted_holdout_best": row.get("beats_holdout_best") is True,
        "scientific_verdict_promote": row.get("verdict") == "promote",
        "selector_approved": row.get("selector_approved") is True,
    }
    expected = row.get("sha256")
    measured = sha256(path) if path.is_file() else None
    checks["sha256_matches"] = bool(expected and measured == expected)
    authorized = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "lane": row["lane"],
        "file": row["file"],
        "sha256": measured,
        "checks": checks,
        "authorized_to_submit": authorized,
        "decision": "OK TO SUBMIT" if authorized else "NOT OK TO SUBMIT",
        "failed_gates": failed,
    }


def build(lanes_path: Path = DEFAULT_LANES, out: Path = DEFAULT_OUT) -> dict:
    source = json.loads(lanes_path.read_text())
    rows = [assess(row, ROOT / "docs") for row in source["lanes"]]
    receipt = {
        "schema": "gems-submission-authorization-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": (
            "Authorization requires artifact existence, matching SHA-256, local format PASS, "
            "literal protocol uniqueness, improvement over the accepted holdout best, a promote "
            "verdict, and explicit selector approval. Missing fields fail closed."
        ),
        "repository_authorized_file_count": sum(r["authorized_to_submit"] for r in rows),
        "repository_decision": "OK TO SUBMIT" if any(r["authorized_to_submit"] for r in rows) else "NOT OK TO SUBMIT",
        "artifacts": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=Path, default=DEFAULT_LANES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    receipt = build(args.lanes, args.out)
    print(json.dumps({
        "decision": receipt["repository_decision"],
        "authorized_files": receipt["repository_authorized_file_count"],
        "receipt": str(args.out),
    }, indent=2))
    return 0 if receipt["repository_authorized_file_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
