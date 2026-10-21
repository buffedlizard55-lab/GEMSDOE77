#!/usr/bin/env python3
"""Write the GEMSDOE77 submission files from the E3 dot set and validate them (fails closed).

Outputs (docs/downloads/):
  <name>-zeros.tif   PRIMARY. All 12,279,160 cells finite: 1.0 on dots, 0.0 elsewhere, no
                     nodata tag, 256x256 deflate tiles - the same encoding as the
                     GEMSDOE32 0.2778 file that the portal accepted and scored.
  <name>-nan.tif     Twin with bit-identical in-footprint values and NaN outside the official
                     footprint, written by the shared template writer (gems.submission_writer).
  <name>-zeros.zip   One-member zip of the primary TIF (form accepts a .zip with one GeoTIFF).
Validation: gems77.validate (official sample_submission footprint) + the template validator.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
from gems77 import core, validate as v77  # noqa: E402
from gems import submission_writer as tw, validate_submission as tv  # noqa: E402  (template)

STAMP = "20261021"


def main():
    import rasterio
    g = core.grid()
    dots = np.load(core.WORK / "candidate_dots.npy")
    assert dots.shape == (core.H, core.W) and not (dots & ~g.footprint).any() and not (dots & g.catalogue).any()
    pred = np.zeros((core.H, core.W), np.float32)
    pred[dots] = 1.0
    digest = hashlib.sha256(pred.tobytes()).hexdigest()
    n = int(dots.sum())
    base = f"gems77-h77-lia-{n}dots-{STAMP}-{digest[:8]}"
    dsup = np.load(core.WORK / "d_h195.npy")
    on_pct = round(100 * float((dots & (dsup == 0)).sum()) / n)
    note = (f"GEMSDOE77 H77 live-inversion allocation: {on_pct}% H19-5 support + {100 - on_pct}% straight-edge dots, "
            f"{n} binary, 0 on catalogue")
    assert len(note) <= 140, len(note)
    out = ROOT / "docs" / "downloads"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("gems77-*"):
        old.unlink()
    zeros = out / f"{base}-zeros.tif"
    with rasterio.open(core.DATA / "sample_submission.tif") as ref:
        prof = dict(driver="GTiff", height=ref.height, width=ref.width, count=1, dtype="float32",
                    crs=ref.crs, transform=ref.transform, nodata=None, tiled=True,
                    blockxsize=256, blockysize=256, compress="deflate")
    with rasterio.open(zeros, "w", **prof) as dst:
        dst.write(pred, 1)
    nan = out / f"{base}-nan.tif"
    tw.write_submission(nan, pred.astype(np.float64), g.footprint, core.DATA / "sample_submission.tif",
                        overwrite=True, name=f"{base}-nan", note=note)
    receipt_tmp = Path(str(nan) + ".receipt.json")
    zpath = out / f"{base}-zeros.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(zeros, arcname=zeros.name)
    rep = {"name": f"{base}-zeros", "note": note, "note_chars": len(note), "decoded_sha256": digest,
           "dots": n,
           "primary": v77.validate(zeros, mode="zeros", name=f"{base}-zeros", note=note),
           "twin_nan": v77.validate(nan, mode="nan", name=f"{base}-nan", note=note),
           "zip": v77.validate_zip(zpath, zeros),
           "template_validator_primary": tv.validate(zeros, core.DATA / "labels.tif", core.DATA / "training_features.tif"),
           "template_validator_twin": tv.validate(nan, core.DATA / "labels.tif", core.DATA / "training_features.tif"),
           "template_writer_receipt": json.loads(receipt_tmp.read_text())}
    receipt_tmp.unlink()
    # bit-identity of in-footprint values between the two encodings
    with rasterio.open(zeros) as a, rasterio.open(nan) as b:
        za, nb = a.read(1), b.read(1)
    rep["in_footprint_bit_identical"] = bool(np.array_equal(za[g.footprint], nb[g.footprint]))
    ok = rep["primary"]["passed"] and rep["twin_nan"]["passed"] and rep["zip"]["passed"] and rep["in_footprint_bit_identical"]
    rep["download_ok"] = bool(ok)
    (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "data" / "submission.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("name", "note_chars", "dots", "download_ok")}, indent=1))
    print("primary", rep["primary"]["n_pass"], "/", rep["primary"]["n_checks"], rep["primary"]["sha256"])
    print("twin   ", rep["twin_nan"]["n_pass"], "/", rep["twin_nan"]["n_checks"], rep["twin_nan"]["sha256"])
    for lab in ("template_validator_primary", "template_validator_twin"):
        fails = [c["check"] + ": " + c["detail"] for c in rep[lab]["checks"] if c["status"] == "FAIL"]
        print(lab, "passed" if rep[lab]["passed"] else f"FAILS: {fails}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
