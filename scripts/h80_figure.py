#!/usr/bin/env python3
"""H80 figure from on-disk artifacts only (no recomputation of scores)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from gems57 import grid as G  # noqa: E402

PROC = ROOT / "data" / "processed"
g = G.load_grid()
surf = np.load(PROC / "h80_fullmap_surface.npy", mmap_mode="r")
ota = np.load(PROC / "ota_full.npy", mmap_mode="r")
dots = np.load(PROC / "h80_final_dots.npy")
e2 = json.loads((ROOT / "evidence" / "h80_exp2.json").read_text())["pooled_summary"]["scores"]
D = 4  # decimation for display only

fig, ax = plt.subplots(2, 2, figsize=(11, 10))
a = np.array(ota[::D, ::D, 0], dtype=np.float32); a[~g.footprint[::D, ::D]] = np.nan
ax[0, 0].imshow(a, cmap="inferno", vmin=0.9, vmax=1.0)
ax[0, 0].set_title("A  OTA orphan-thermal alignment field (rank, display >0.90)\n630 orphan hot points -> collinearity corridors")
a = np.array(surf[::D, ::D], dtype=np.float32); a[~g.footprint[::D, ::D]] = np.nan
ax[0, 1].imshow(a, cmap="viridis", vmin=0, vmax=0.9)
ax[0, 1].set_title("B  Full-map HGB surface (33 std + 4 OTA channels)")
show = np.full(dots[::D, ::D].shape, np.nan)
base = g.footprint[::D, ::D]
img = np.full((*show.shape, 3), 0.93, np.float32)
img[~base] = (0.65, 0.65, 0.65)
img[g.catalogue[::D, ::D]] = (0.75, 0.1, 0.1)
img[dots[::D, ::D]] = (0.05, 0.2, 0.9)
ax[1, 0].imshow(img)
ax[1, 0].set_title("C  Emitted dots (blue, 37,654) vs mapped catalogue (red)\ndots >200 m from catalogue, 0 on catalogue")

order = ["A0_random", "S_ota_xridge", "A_std", "A_ota"]
vals = [e2[k]["dti"] for k in order]
lo = [e2[k]["dti"] - e2[k]["ci95"][0] for k in order]
hi = [e2[k]["ci95"][1] - e2[k]["dti"] for k in order]
ax[1, 1].bar(order, vals, yerr=[lo, hi], capsize=4, color=["#bbbbbb", "#dd8855", "#4477aa", "#228833"])
ax[1, 1].axhline(0.152718, ls="--", c="r", lw=1)
ax[1, 1].text(3.4, 0.155, "repo incumbent 0.152718\n(H77-Step, cross-draw)", color="r", ha="right", fontsize=8)
ax[1, 1].set_ylabel("HOLDOUT-DTI (gems57-pooled-hide-v2,\n11,342 withheld px, seed 30)")
ax[1, 1].set_title("D  Paired arms: A_ota - A_std = +0.0068 [-0.0007, 0.0149] (n.s.)\nbelow incumbent -> verdict negative", fontsize=10)
for a_ in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("GEMSDOE77 H80-OTA orphan thermal alignment - negative result (HOLDOUT-DTI is not a leaderboard score)", fontsize=12)
fig.tight_layout()
out = ROOT / "docs" / "figures" / "h80_surface_dots_arms.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140)
print(out, out.stat().st_size)
