#!/usr/bin/env python3
"""Generate diagnostic figures for the H77-BGB lane:
  1. docs/figures/bgb_overview.png: 4-panel spatial map of S_growth_wedge, S_strain_part,
     S_upflow_breach, and the final 37,654 submodular dots (with known catalogue faults).
  2. docs/figures/bgb_holdout_ablation.png: Holdout DTI comparison bar chart with 95% block
     bootstrap CIs + single-feature ROC-AUC leakage canary profile vs 0.90 threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import grid as G
from gems77_bgb import features as FBGB

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
EVID = ROOT / "evidence"
FIG_DIR = ROOT / "docs" / "figures"


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grid = G.load_grid(RAW)
    foot = grid.footprint
    cat = grid.catalogue

    bgb_stack = np.load(PROC / "bgb_stack.npy", mmap_mode="r")
    surfaces = FBGB.compute_bgb_surfaces(bgb_stack, foot)
    p_full = np.load(PROC / "bgb_fullmap_oof_surface.npy")
    dots = np.load(PROC / "bgb_fullmap_dots.npy")

    # Downsample by 4x for crisp rendering
    step = 4
    f_ds = foot[::step, ::step]

    def masked_ds(arr: np.ndarray) -> np.ma.MaskedArray:
        sub = arr[::step, ::step]
        return np.ma.masked_where(~f_ds, sub)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), dpi=130)
    panels = [
        ("Sub-block A: Basement Growth-Wedge\nInflection (S_growth_wedge)", surfaces["S_growth_wedge"], "viridis"),
        ("Sub-block B: Transtensional Strain-\nPartitioning Gradient (S_strain_part)", surfaces["S_strain_part"], "magma"),
        ("Sub-block C: Conductivity-Margin\nDemag Breach (S_upflow_breach)", surfaces["S_upflow_breach"], "plasma"),
        ("Full-Map Cross-Fitted H77-BGB\nSurface + 37,654 Submodular Dots", p_full, "inferno"),
    ]
    for ax, (title, arr, cmap) in zip(axes, panels):
        im = ax.imshow(masked_ds(arr), cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Overlay subsampled emitted dots in cyan on panel 4
    dy, dx = np.nonzero(dots[::step, ::step])
    axes[3].scatter(dx, dy, s=0.18, c="#00ffff", alpha=0.65, label="37,654 BGB dots")
    axes[3].legend(loc="lower right", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "H77-BGB: Buried Growth-Wedge Inflection × Transtensional Strain-Partitioning × Clay-Cap Breach (Research Only — DO NOT UPLOAD)",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bgb_overview.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Holdout DTI + Leakage Canaries
    rc = json.loads((EVID / "run_card_bgb.json").read_text())
    arms = rc["holdout_dti"]["all_arms_seed30"]
    canaries = rc["leakage_canary"]["per_channel_seed30"]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=130)

    arm_order = [
        ("A0_random", "A0_random\n(Uniform Control)", "#7f8c8d"),
        ("A_bgb_surface", "A_bgb_surface\n(Label-Free BGB)", "#2980b9"),
        ("A_bgb_hgb_only", "A_bgb_hgb_only\n(10-ch BGB HGB)", "#16a085"),
        ("A_bgb_lane", "A_bgb_lane\n(Candidate TIF Arm)", "#27ae60"),
        ("A_std", "A_std\n(33-ch Shared Base)", "#8e44ad"),
        ("A_std_plus_bgb", "A_std + BGB\n(43-ch Static)", "#d35400"),
        ("A_bgb_loco", "A_bgb_loco\n(43-ch + 4 LOCO)", "#c0392b"),
    ]
    y_pos = np.arange(len(arm_order))
    vals = [arms[k]["dti"] for k, _, _ in arm_order]
    lo_err = [arms[k]["dti"] - arms[k]["ci95"][0] for k, _, _ in arm_order]
    hi_err = [arms[k]["ci95"][1] - arms[k]["dti"] for k, _, _ in arm_order]
    colors = [c for _, _, c in arm_order]
    labels = [lbl for _, lbl, _ in arm_order]

    ax1.barh(y_pos, vals, xerr=[lo_err, hi_err], color=colors, alpha=0.88, capsize=3.5, height=0.62)
    ax1.axvline(0.152718, color="#111111", linestyle="--", linewidth=1.4, label="Repo Holdout Best (H77-Step: 0.1527)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("HOLDOUT-DTI (gems57-pooled-hide-v2, 11,342 withheld positives)", fontsize=9)
    ax1.set_title("Experiment 2: Spatially Blocked Holdout DTI & 95% Block-Bootstrap CI", fontsize=9.5, fontweight="bold")
    ax1.set_xlim(0.0, 0.175)
    for i, v in enumerate(vals):
        ax1.text(v + hi_err[i] + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="x", alpha=0.3)

    ch_names = list(canaries.keys())
    sym_aucs = [canaries[k]["max_symmetric_auc"] for k in ch_names]
    short_names = [k.replace("bgb_", "") for k in ch_names]
    yc = np.arange(len(ch_names))
    ax2.barh(yc, sym_aucs, color="#34495e", alpha=0.85, height=0.6)
    ax2.axvline(0.90, color="#e74c3c", linestyle="--", linewidth=1.5, label="Leakage Canary Gate (AUC <= 0.90)")
    ax2.axvline(0.50, color="#95a5a6", linestyle=":", linewidth=1.0, label="Random (0.50)")
    ax2.set_yticks(yc)
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.set_xlabel("Max Symmetric ROC-AUC on Withheld Faults", fontsize=9)
    ax2.set_title("Experiment 1: Single-Feature Leakage Canaries (All 10 PASS <= 0.90)", fontsize=9.5, fontweight="bold")
    ax2.set_xlim(0.45, 0.96)
    for i, v in enumerate(sym_aucs):
        ax2.text(v + 0.006, i, f"{v:.4f}", va="center", fontsize=7.8)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "bgb_holdout_ablation.png", bbox_inches="tight")
    plt.close(fig2)
    print("Generated docs/figures/bgb_overview.png and docs/figures/bgb_holdout_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
