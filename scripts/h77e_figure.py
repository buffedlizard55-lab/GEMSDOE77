#!/usr/bin/env python3
"""Diagnostic figure for lane H77E: fields, surface, dots (downsampled)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import h77e_common as C  # noqa: E402

BUILD = json.loads((ROOT / "evidence" / "h77e_e3_build.json").read_text())


def main():
    grid = C.load_grid()
    foot, cat = grid.footprint, grid.catalogue
    new_fp = np.load(C.CACHE / "new_fp.npy", mmap_mode="r")
    names = json.loads((C.CACHE / "new_names.json").read_text())
    surf = np.load(C.CACHE / "full_surface.npy")
    with rasterio.open(ROOT / BUILD["tif"]) as src:
        dots = src.read(1) > 0

    def scatter(name):
        a = np.zeros(foot.shape, np.float32)
        a[foot] = new_fp[:, names.index(name)]
        return a

    panels = [
        (scatter("ucp_mag_persist"), "UCP magnetic persistence"),
        (scatter("ucp_cross_min"), "UCP cross-field min"),
        (scatter("euler_density"), "Euler density (SI=0)"),
        (surf, "H77E OOF surface (comb+ctx)"),
        (np.where(cat, 1.0, np.where(dots, 0.45, 0.0)), "dots (light) + catalogue (dark)"),
        (np.where(cat, 1.0, np.where(dots, 0.45, 0.0)), "zoom: dots + catalogue"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (a, title) in zip(axes.ravel(), panels):
        ax.imshow(np.where(foot, a, np.nan)[::4, ::4], cmap="viridis", vmin=0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    # zoom last panel to a dot-dense quadrant
    axes.ravel()[-1].imshow(np.where(foot, panels[-1][0], np.nan)[1000:2200:2, 1200:2400:2],
                            cmap="viridis", vmin=0)
    fig.suptitle("GEMSDOE77 H77E: UCP-Euler fields, OOF surface and emitted dots (downsampled)\n"
                 "HOLDOUT-DTI 0.1305 (s20) / 0.1342 (s30) vs repo best 0.1527 -- NEGATIVE, not OK to submit",
                 fontsize=10)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "h77e_surface_and_dots.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
