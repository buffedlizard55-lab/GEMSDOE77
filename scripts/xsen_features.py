#!/usr/bin/env python3
"""H77-XSEN new channels: magnetic analytic-signal ridge x radiometric boundary.

Preregistered in registry/preregistration_xsen.json. Reads ONLY:
  training_features.tif band 3 (tmi_hg), band 9 (tmi_vg)
  external/geodawn_extensions_u8.tif bands 1 (ThK), 2 (UK)
No catalogue, no prior submission raster is read. Output:
  data/processed/xsen.npy  (H, W, 8) float32, 0 outside footprint.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems77 import features77 as F  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "xsen.npy"
NAMES = ["xsen_as_ridge_s1p5", "xsen_as_ridge_s3", "xsen_as_persist",
         "xsen_radgrad_thk", "xsen_radgrad_uk", "xsen_radgrad",
         "xsen_coincide", "xsen_coincide_min"]


def gradmag(a, sigma=1.5):
    gy = ndimage.gaussian_filter(a, sigma, order=(1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(a, sigma, order=(0, 1), mode="nearest")
    return np.hypot(gx, gy).astype(np.float32)


def main():
    t0 = time.time()
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    with rasterio.open(RAW / "sample_submission.tif") as s:
        foot = np.isfinite(s.read(1))
    H, W = foot.shape
    with rasterio.open(RAW / "training_features.tif") as s:
        hg = F.clean(s.read(3)); vg = F.clean(s.read(9))
    hg_f = F.fill_median(hg, foot); vg_f = F.fill_median(vg, foot)
    log_as = np.log1p(np.hypot(hg_f, vg_f)).astype(np.float32)
    del hg, vg, hg_f, vg_f
    r1 = F.hessian_ridge(log_as, 1.5); r3 = F.hessian_ridge(log_as, 3.0)
    del log_as
    with rasterio.open(RAW / "external" / "geodawn_extensions_u8.tif") as s:
        thk = s.read(1).astype(np.float32); uk = s.read(2).astype(np.float32)
    g_thk = gradmag(thk); g_uk = gradmag(uk); del thk, uk
    out = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(H, W, len(NAMES)))
    rk = lambda a: F.rank01(np.where(foot, a, np.float32(np.nan)), foot)  # noqa: E731
    rr1, rr3 = rk(r1), rk(r3); del r1, r3
    persist = np.minimum(rr1, rr3).astype(np.float32)
    rp = rk(persist)
    gt, gu = rk(g_thk), rk(g_uk); del g_thk, g_uk
    radgrad = np.maximum(gt, gu).astype(np.float32)
    coinc = np.sqrt(np.clip(rp, 0, 1) * np.clip(radgrad, 0, 1)).astype(np.float32)
    coinc_min = np.minimum(rp, radgrad).astype(np.float32)
    for i, a in enumerate([rr1, rr3, rp, gt, gu, radgrad, coinc, coinc_min]):
        out[:, :, i] = np.where(foot, a, 0.0).astype(np.float32)
        print(f"{NAMES[i]}: in-foot mean={float(out[foot, i].mean()):.4f}", flush=True)
    out.flush(); del out
    receipt = {"file": str(OUT), "channels": NAMES, "shape": [H, W, len(NAMES)],
               "footprint_px": int(foot.sum()), "seconds": round(time.time() - t0, 1),
               "inputs": ["training_features.tif bands 3,9", "geodawn_extensions_u8.tif bands 1,2"]}
    (ROOT / "evidence" / "xsen_features_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
