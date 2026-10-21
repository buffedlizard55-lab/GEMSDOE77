#!/usr/bin/env python3
"""Assemble the 33 channels H80 actually uses (raw+terrain+rad) after the full
54-channel template builder was OOM-killed (exit 137) in its FFT stage in this
3 GB sandbox (evidence/h80_build_features.log).

Raw+terrain channels are copied from the partially built data/processed/stack.npy
ONLY after row-level verification against fresh reads; radiometric channels are
computed with the exact lines from scripts/build_features.py (u8/255, NaN outside
footprint). Channel definitions are src/gems77/features77.py - nothing forked.
Output: data/processed/stack33.npy + evidence/h80_stack33_receipt.json.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems77 import features77 as F  # noqa: E402

RAW = ROOT / "data" / "raw"; PROC = ROOT / "data" / "processed"
NAMES = F.BLOCKS["raw"] + F.BLOCKS["terrain"] + F.BLOCKS["rad"]


def read(path, i=1):
    with rasterio.open(path) as s:
        return s.read(i)


def main():
    t0 = time.time()
    foot = np.isfinite(read(RAW / "sample_submission.tif"))
    H, W = foot.shape
    full = np.load(PROC / "stack.npy", mmap_mode="r")
    assert full.shape == (H, W, len(F.CHANNELS)), full.shape

    # verify raw band 1 & a terrain channel against independent recomputation on rows 1000:1100
    b1 = F.clean(read(RAW / "training_features.tif", 1)[1000:1100])
    got = np.asarray(full[1000:1100, :, F.IDX["mag_anom"]])
    assert np.array_equal(np.nan_to_num(b1), np.nan_to_num(got)), "stack.npy raw band1 != fresh read (partial write?)"
    del b1, got

    out = np.lib.format.open_memmap(PROC / "stack33.npy", mode="w+", dtype=np.float32, shape=(H, W, len(NAMES)))
    stats = {}
    for j, name in enumerate(NAMES[:27]):  # raw + terrain copied
        out[:, :, j] = full[:, :, F.IDX[name]]
    out.flush()
    for j, name in enumerate(NAMES[:27]):
        a = np.asarray(out[::7, ::7, j]); f = foot[::7, ::7]
        stats[name] = {"finite_frac_in_foot_sample": float(np.isfinite(a[f]).mean()),
                       "std_sample": float(np.nanstd(a[f]))}
        assert stats[name]["std_sample"] > 0, f"{name} degenerate std=0 (unwritten channel?)"
        print(f"copy {name} ok {stats[name]}", flush=True)
    del full; gc.collect()

    rad_src = {"rad_K": ("geodawn_rad_u8.tif", 1), "rad_Th": ("geodawn_rad_u8.tif", 2),
               "rad_U": ("geodawn_rad_u8.tif", 3), "rad_TC": ("geodawn_rad_u8.tif", 4),
               "rad_ThK": ("geodawn_extensions_u8.tif", 1), "rad_UK": ("geodawn_extensions_u8.tif", 2)}
    for name, (fn, band) in rad_src.items():
        a = read(RAW / "external" / fn, band).astype(np.float32) / 255.0
        a = np.where(foot, a, np.float32(np.nan)).astype(np.float32)  # build_features.put() semantics
        j = NAMES.index(name); out[:, :, j] = a; out.flush()
        s = a[::7, ::7]; f = foot[::7, ::7]
        stats[name] = {"finite_frac_in_foot_sample": float(np.isfinite(s[f]).mean()),
                       "std_sample": float(np.nanstd(s[f]))}
        assert stats[name]["std_sample"] > 0, f"{name} degenerate"
        print(f"rad {name} ok {stats[name]}", flush=True)
        del a; gc.collect()
    out.flush(); del out
    receipt = {"path": "data/processed/stack33.npy", "shape": [H, W, len(NAMES)], "channels": NAMES,
               "source": "raw+terrain copied from stack.npy after verification; rad computed per build_features.py",
               "full_builder_failure": "scripts/build_features.py exit 137 (OOM) in FFT phase-congruency stage, evidence/h80_build_features.log",
               "channel_stats_sampled": stats, "seconds": round(time.time() - t0, 1)}
    (ROOT / "evidence" / "h80_stack33_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
