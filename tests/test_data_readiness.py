from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_data_ready.py"


def _fixture(tmp_path: Path, *, receipt_status: str = "restored"):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    dest = "core/example.bin"
    content = b"verified test fixture\n"
    target = data_root / dest
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    manifest = {
        "files": [{
            "id": "example",
            "dest": dest,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }]
    }
    receipt = {
        "verified_files_count": 1,
        "storage_root": "/previous/machine/path",
        "files": [{
            "id": "example",
            "dest": dest,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "status": receipt_status,
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return data_root, target, manifest_path, receipt_path


def _run(tmp_path: Path, data_root: Path, manifest: Path, receipt: Path, ids="all"):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest", str(manifest),
            "--data-root", str(data_root),
            "--receipt", str(receipt),
            "--ids", ids,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_data_readiness_accepts_matching_bytes_and_receipt_in_a_new_checkout(tmp_path):
    data_root, _, manifest, receipt = _fixture(tmp_path)
    result = _run(tmp_path, data_root, manifest, receipt)
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ready"] is True
    assert payload["files"][0]["issues"] == []
    # Absolute storage_root in a previous receipt is informational, not a stale-path blocker.
    assert payload["data_root"] == str(data_root.resolve())


def test_data_readiness_fails_when_receipt_claims_exist_but_file_is_missing(tmp_path):
    data_root, target, manifest, receipt = _fixture(tmp_path)
    target.unlink()
    result = _run(tmp_path, data_root, manifest, receipt)
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ready"] is False
    assert "current file is missing" in payload["files"][0]["issues"]


def test_data_readiness_fails_on_hash_or_size_mismatch(tmp_path):
    data_root, target, manifest, receipt = _fixture(tmp_path)
    target.write_bytes(b"edited")
    result = _run(tmp_path, data_root, manifest, receipt)
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    issues = payload["files"][0]["issues"]
    assert "sha256 mismatch" in issues
    assert any("byte count mismatch" in issue for issue in issues)


def test_data_readiness_fails_when_receipt_does_not_match_manifest(tmp_path):
    data_root, _, manifest, receipt = _fixture(tmp_path)
    data = json.loads(receipt.read_text())
    data["files"][0]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(data))
    result = _run(tmp_path, data_root, manifest, receipt)
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert "receipt sha256 mismatch" in payload["files"][0]["issues"]


def test_data_readiness_rejects_unknown_ids_and_path_traversal(tmp_path):
    data_root, _, manifest, receipt = _fixture(tmp_path)
    unknown = _run(tmp_path, data_root, manifest, receipt, "missing")
    assert unknown.returncode == 1
    assert "unknown manifest id" in json.loads(unknown.stdout)["error"]

    data = json.loads(manifest.read_text())
    data["files"][0]["dest"] = "../escape.bin"
    manifest.write_text(json.dumps(data))
    traversal = _run(tmp_path, data_root, manifest, receipt)
    assert traversal.returncode == 1
    assert "escapes data root" in json.loads(traversal.stdout)["error"]
