#!/usr/bin/env python3
"""Verify present sibling ZIP downloads against independently validated TIFFs.

No extraction, training, selection or organizer submission. Missing sibling ZIPs
are reported, not treated as advertised downloads. This is not a site link crawl.
"""
from __future__ import annotations

import hashlib
import json
import stat
import zipfile
import zlib
from pathlib import Path

from audit_downloads import ROOT, audit, digest


def validate_archive(archive: Path, primary: Path, expected_hash: str) -> dict:
    """Require exactly one root TIFF identical to the primary, with bounded reads."""
    checks = {}
    try:
        size = primary.stat().st_size
        checks["primary_sha256_matches"] = digest(primary) == expected_hash
        with zipfile.ZipFile(archive) as src:
            members = src.infolist()
            checks["exactly_one_member"] = len(members) == 1
            if len(members) != 1:
                raise ValueError("ZIP must contain exactly one member")
            member = members[0]
            mode = member.external_attr >> 16
            checks.update(
                member_name_matches=member.filename == primary.name,
                member_is_tiff=Path(member.filename).suffix.lower() == ".tif",
                member_not_directory=not member.is_dir(),
                member_not_symlink=not stat.S_ISLNK(mode),
                member_not_encrypted=not bool(member.flag_bits & 1),
                member_size_matches=member.file_size == size,
            )
            if not all(checks.values()):
                raise ValueError("ZIP structure or primary integrity mismatch")
            h = hashlib.sha256()
            read = 0
            with src.open(member) as stream:
                while chunk := stream.read(min(1 << 20, size - read + 1)):
                    read += len(chunk)
                    if read > size:
                        raise ValueError("ZIP exceeds primary size")
                    h.update(chunk)
            checks["read_size_matches"] = read == size
            checks["member_sha256_matches"] = h.hexdigest() == expected_hash
        return {"checks": checks, "archive_pass": all(checks.values()),
                "archive_sha256": digest(archive), "authorized_to_submit": False}
    except (OSError, ValueError, RuntimeError, NotImplementedError, zipfile.BadZipFile,
            EOFError, zlib.error) as exc:
        return {"checks": checks, "archive_pass": False, "error": str(exc),
                "authorized_to_submit": False}


def audit_archives(root: Path = ROOT) -> dict:
    # Independently reopen the TIFFs; never trust a stale PASS receipt.
    tiffs = audit(root)
    rows, absent = [], []
    for item in tiffs["artifacts"]:
        primary = root / "docs" / item["file"]
        archive = primary.with_suffix(".zip")
        if not archive.exists():
            absent.append(item["file"])
            continue
        try:
            if not archive.resolve().is_relative_to((root / "docs/downloads").resolve()):
                raise ValueError("ZIP outside docs/downloads")
            result = validate_archive(archive, primary, item.get("sha256", ""))
        except (OSError, ValueError) as exc:
            result = {"archive_pass": False, "error": str(exc)}
        rows.append({"file": str(archive.relative_to(root / "docs")),
                     "primary_format_pass": item["format_pass"], **result})
    return {
        "schema": "gems-sibling-archive-readback-v1",
        "evidence_class": "FORMAT-INTEGRITY-MEASUREMENT; NOT SCORE OR PORTAL ACCEPTANCE",
        "scope": "Present sibling ZIPs of primary TIFFs in lanes.json; not all site ZIP links",
        "manifest_sha256": tiffs["manifest_sha256"],
        "reference_sha256": tiffs["reference_sha256"],
        "checked_archive_count": len(rows), "primaries_without_sibling_zip": absent,
        "all_primary_format_pass": tiffs["all_format_pass"],
        "all_archive_pass": bool(rows) and tiffs["all_format_pass"] and all(
            x["archive_pass"] and x["primary_format_pass"] for x in rows),
        "authorized_to_submit": False, "artifacts": rows,
    }


if __name__ == "__main__":
    output = audit_archives()
    (ROOT / "evidence/submission_archive_audit.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "artifacts"}, indent=2))
    raise SystemExit(0 if output["all_archive_pass"] else 1)
