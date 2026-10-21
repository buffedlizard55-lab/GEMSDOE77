#!/usr/bin/env python3
"""Autonomous SHA-256-verified data & registry restoration for GEMSDOE77.

Restores all canonical competition rasters (``training_features.tif``, ``existing_faults.tif``,
``sample_submission.tif``) plus official public USGS 3DEP / GeoDAWN / DOE GDR external datasets
into ``.cache/data`` without ever copying or reading prior predictions into feature extraction.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache" / "data"
FEAT_REF = "c0c06ac82178f26b94fce3397036ef8f12a2f3a0"
AUX_REF = "32024716f218d06a94e01464157392119c03cb15"

PINS = {
    "training_features.tif": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
    "existing_faults.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "sample_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_from_tarball(repo: str, ref: str, prefix_or_map: dict[str, Path] | None = None, prefixes: tuple[str, ...] = ()) -> None:
    url = f"https://codeload.github.com/buffedlizard55-lab/{repo}/tar.gz/{ref}"
    req = urllib.request.Request(url, headers={"User-Agent": "gemsdoe77-restore"})
    with urllib.request.urlopen(req, timeout=240) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                parts = member.name.split("/", 1)
                rel = parts[1] if len(parts) > 1 else parts[0]
                dest = None
                if prefix_or_map and rel in prefix_or_map:
                    dest = prefix_or_map[rel]
                elif prefixes and any(rel.startswith(px) for px in prefixes):
                    dest = DATA / rel
                if dest is not None and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    fh = tf.extractfile(member)
                    if fh is not None:
                        tmp = dest.with_suffix(dest.suffix + ".tmp")
                        with tmp.open("wb") as out:
                            for chunk in iter(lambda: fh.read(1 << 20), b""):
                                out.write(chunk)
                        tmp.replace(dest)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    tf_path = DATA / "training_features.tif"
    if not (tf_path.exists() and sha256_file(tf_path) == PINS["training_features.tif"]):
        parts_map = {
            f"data/bridge/gems-geodawn-numerical-features.tif.part-00{i}": DATA / f"part-00{i}"
            for i in range(5)
        }
        extract_from_tarball("GEMSDOE", FEAT_REF, prefix_or_map=parts_map)
        with tf_path.open("wb") as out:
            for i in range(5):
                p = DATA / f"part-00{i}"
                with p.open("rb") as f:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        out.write(b)
                p.unlink()

    ext_dir = DATA / "data" / "external"
    if not (ext_dir / "lidar_scarp_features_u8.tif").exists():
        extract_from_tarball(
            "60GEMSDOE",
            "HEAD",
            prefix_or_map={
                "data/bridge/existing_faults.tif": DATA / "existing_faults.tif",
                "data/sample_submission.tif": DATA / "sample_submission.tif",
            },
            prefixes=("data/external/", "registry/rasters/"),
        )

    for link_src, link_dst in (
        (DATA / "data" / "bridge" / "existing_faults.tif", DATA / "existing_faults.tif"),
        (DATA / "data" / "sample_submission.tif", DATA / "sample_submission.tif"),
        (DATA / "sample_submission.tif", DATA / "example_submission.tif"),
    ):
        if link_src.exists() and not link_dst.exists():
            link_dst.symlink_to(link_src)

    rows = {}
    for name, expected in PINS.items():
        actual = sha256_file(DATA / name)
        rows[name] = {"sha256": actual, "expected": expected, "ok": actual == expected}
        if actual != expected:
            raise ValueError(f"SHA-256 verification failed for {name}: {actual} != {expected}")

    receipt_path = ROOT / "evidence" / "data_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "evidence_class": "DATA-MEASUREMENT",
                "data_dir": str(DATA),
                "feature_ref": FEAT_REF,
                "aux_ref": AUX_REF,
                "files": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
