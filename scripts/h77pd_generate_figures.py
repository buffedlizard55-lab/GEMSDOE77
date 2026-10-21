#!/usr/bin/env python3
"""Generate 4 publication-grade diagnostic figures for Lane H77E directly from verified artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    exp1 = json.loads((ROOT / "evidence" / "h77pd_exp1_ablation_holdout.json").read_text())
    exp2 = json.loads((ROOT / "evidence" / "h77pd_exp2_multiseed_replication.json").read_text())
    rc = json.loads((ROOT / "evidence" / "run_card_h77pd.json").read_text())

    out_dir = ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "grid.color": "#cbd5e1",
        "grid.alpha": 0.45,
    })

    # ── Figure 1: 11-Arm Seed-30 Causal Ablation Ladder (11,342 Withheld Positives) ──
    arms_order = [
        ("Reference_H33_vis_excl3", "Ref: H33 Vis-Excl3\n(0.2778 Champion Rule)"),
        ("Arm0_B0_core_greedy", "Arm 0: B0 Core\n(Greedy Disk Alloc)"),
        ("Arm1_plus_H77E1_fluvial_greedy", "Arm 1: + H77E-1 Fluvial\nKnickpoint & Chi"),
        ("Arm2_plus_H77E2_deseam_greedy", "Arm 2: + H77E-2 GeoDAWN/\nBLM/TIGER Deseamed MT"),
        ("Arm3_plus_H77E3_step_gap_greedy", "Arm 3: + H77E-3 sqrt-LiDAR\n& 10m Step Bridge"),
        ("Arm3b_phys_only_bayes_submod", "Arm 3b: Pure Zero-Label\nPhys96 + Submod"),
        ("Arm4_plus_H77E4_ensemble_greedy", "Arm 4: + Cross-Family\nHGB+LGBM Continuation"),
        ("Arm5_H77E_rank4_submod", "Arm 5: + Rank^4 Power\n300m Submodular"),
        ("Arm6_H77E_bayes_submod_unconstrained", "Arm 6: + Exact Bayes\nPrior Shift Submod"),
        ("Arm8_H77E_logodds_lagrangian_submod", "Arm 8: Log-Odds Fusion\nSensitivity Check"),
        ("Arm7_H77E_lagrangian_knapsack_submod", "Arm 7: H77E Full\n(Lagrangian Submodular)"),
    ]
    sc1 = exp1["pooled_summary"]["scores"]
    labels1 = [lbl for _, lbl in arms_order]
    vals1 = [sc1[k]["dti"] for k, _ in arms_order]
    lo1 = [sc1[k]["dti"] - sc1[k]["ci95"][0] for k, _ in arms_order]
    hi1 = [sc1[k]["ci95"][1] - sc1[k]["dti"] for k, _ in arms_order]
    colors1 = [
        "#64748b", "#94a3b8", "#38bdf8", "#0284c7", "#0369a1",
        "#0f766e", "#6366f1", "#4f46e5", "#7c3aed", "#9333ea", "#ea580c"
    ]

    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=220)
    y_pos = np.arange(len(labels1))
    bars = ax.barh(y_pos, vals1, xerr=[lo1, hi1], height=0.68, color=colors1,
                   edgecolor="#0f172a", linewidth=0.8, error_kw=dict(ecolor="#0f172a", lw=1.2, capsize=3.5))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels1, fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Pooled Whole-Segment HOLDOUT-DTI (gems57-pooled-hide-v2, B = 37,654 dots, 95% Block CI)", fontsize=10.5, fontweight="bold")
    ax.set_title("Experiment 1: Preregistered 11-Arm Causal Ablation Ladder (Seed 30 — 11,342 Withheld Positive Pixels)", fontsize=12.0, fontweight="bold", pad=12)
    ax.set_xlim(0.0, 0.205)
    ax.grid(axis="x", linestyle="--")
    ax.axvline(0.15011, color="#16a34a", linestyle=":", linewidth=1.5, label="Prior Repo Zero-SGMC Best H77D (0.15011)")
    for b, v, (k, _) in zip(bars, vals1, arms_order):
        ci_l, ci_h = sc1[k]["ci95"]
        ax.text(ci_h + 0.0022, b.get_y() + b.get_height() / 2,
                f"{v:.5f}  [{ci_l:.3f}, {ci_h:.3f}]",
                va="center", fontsize=8.6, fontweight="bold" if "Arm7" in k else "normal", color="#0f172a")
    ax.legend(loc="lower right", frameon=True, fontsize=9.0)
    fig.tight_layout()
    fig.savefig(out_dir / "h77pd_fig1_ablation_ladder.png")
    plt.close(fig)

    # ── Figure 2: Multi-Seed Zero-Tuning Replication + Blind 4-Quadrant Benchmark ──
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 5.6), dpi=220, gridspec_kw={"width_ratios": [1.35, 1.0]})

    bench_names = [
        "Seed 30\n(11,342 px)",
        "Seed 20\n(11,321 px)",
        "Seed 21\n(11,572 px)",
        "3-Seed Pooled\n(34,235 px)",
    ]
    h77e_means = [
        exp1["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["dti"],
        exp2["seed20"]["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["dti"],
        exp2["seed21"]["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["dti"],
        exp2["three_seed_pooled_34235px"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["dti"],
    ]
    h77e_cis = [
        exp1["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["ci95"],
        exp2["seed20"]["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["ci95"],
        exp2["seed21"]["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["ci95"],
        exp2["three_seed_pooled_34235px"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]["ci95"],
    ]
    prior_best = [0.15011, 0.152718, 0.14800, 0.15028]
    prior_lbls = ["H77D (0.1501)", "H77-Step (0.1527)", "PR#11 HSL (0.1480)", "Best Prior Avg"]

    x = np.arange(len(bench_names))
    w = 0.36
    axL.bar(x - w / 2, prior_best, width=w, color="#64748b", edgecolor="#0f172a", label="Prior Best Lane on Same Split")
    err_lo = [m - c[0] for m, c in zip(h77e_means, h77e_cis)]
    err_hi = [c[1] - m for m, c in zip(h77e_means, h77e_cis)]
    axL.bar(x + w / 2, h77e_means, width=w, yerr=[err_lo, err_hi], color="#ea580c", edgecolor="#0f172a",
            error_kw=dict(ecolor="#0f172a", lw=1.3, capsize=4), label="Lane H77E (Zero-Tuning Replication)")
    for i in range(len(x)):
        axL.text(x[i] - w / 2, prior_best[i] + 0.003, f"{prior_lbls[i]}", ha="center", va="bottom", fontsize=7.6, rotation=0)
        axL.text(x[i] + w / 2, h77e_cis[i][1] + 0.003, f"{h77e_means[i]:.4f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#9a3412")
    axL.set_xticks(x)
    axL.set_xticklabels(bench_names, fontsize=9.5, fontweight="bold")
    axL.set_ylim(0.0, 0.218)
    axL.set_ylabel("HOLDOUT-DTI (gems57-pooled-hide-v2)", fontsize=10.2, fontweight="bold")
    axL.set_title("A. Multi-Seed Hide-and-Recover Replication (34,235 px Total)", fontsize=11.0, fontweight="bold")
    axL.grid(axis="y", linestyle="--")
    axL.legend(loc="upper left", fontsize=8.8)

    q_sc = exp2["blind_4_quadrant_transfer_57448px"]["pooled_summary"]["scores"]
    q_items = [
        ("H19_5_Ref", "H19-5\nLineaments\n(PR #9 Ref)", "#94a3b8"),
        ("Quad_B0_core_greedy", "B0 Core\nGreedy\n(Baseline)", "#64748b"),
        ("H77_Step_Ref", "Prior Best\nH77-Step\n(PR #9 Ref)", "#0284c7"),
        ("Quad_H77E_Phys96_global_bayes_submod", "H77E Phys96\nGlobal\nSubmodular", "#7c3aed"),
        ("Quad_H77E_Phys96_area_prop_submod", "H77E Phys96\nArea-Prop\nSubmodular", "#ea580c"),
    ]
    q_vals, q_lo, q_hi, q_lbls, q_cols = [], [], [], [], []
    for k, lbl, col in q_items:
        if k == "H19_5_Ref":
            m, c = 0.163942, [0.141, 0.187]
        elif k == "H77_Step_Ref":
            m, c = 0.185834, [0.161898, 0.209307]
        else:
            m, c = q_sc[k]["dti"], q_sc[k]["ci95"]
        q_vals.append(m)
        q_lo.append(m - c[0])
        q_hi.append(c[1] - m)
        q_lbls.append(lbl)
        q_cols.append(col)
    xq = np.arange(len(q_vals))
    axR.bar(xq, q_vals, yerr=[q_lo, q_hi], width=0.56, color=q_cols, edgecolor="#0f172a",
            error_kw=dict(ecolor="#0f172a", lw=1.3, capsize=4))
    for i, (m, hi_e) in enumerate(zip(q_vals, q_hi)):
        axR.text(i, m + hi_e + 0.003, f"{m:.4f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    axR.set_xticks(xq)
    axR.set_xticklabels(q_lbls, fontsize=8.2, fontweight="bold")
    axR.set_ylim(0.0, 0.238)
    axR.set_ylabel("Blind Province Transfer HOLDOUT-DTI (57,448 px)", fontsize=10.0, fontweight="bold")
    axR.set_title("B. Zero-Catalogue Blind 4-Quadrant Transfer", fontsize=11.0, fontweight="bold")
    axR.grid(axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "h77pd_fig2_multiseed_family_benchmark.png")
    plt.close(fig)

    # ── Figure 3: Nevada Posterior Surface, 24.6% Missing-LiDAR Bridge, & Final Dots ──
    surf = np.load(ROOT / ".cache" / "h77e_work" / "live_fullcat_surfaces.npz")
    dirs = np.load(ROOT / ".cache" / "h77e_work" / "directors_h77e.npz")
    with rasterio.open(ROOT / rc["submission_package"]["tif_path"]) as ds:
        dots = ds.read(1) > 0
    with rasterio.open(ROOT / ".cache" / "data" / "existing_faults.tif") as ds:
        cat = ds.read(1) > 0
    with rasterio.open(ROOT / ".cache" / "data" / "sample_submission.tif") as ds:
        foot = np.isfinite(ds.read(1))

    p_phys_grid = np.where(foot, surf["p_phys"], np.nan).astype(np.float32)
    lvalid = dirs["lvalid"].astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.2), dpi=155)
    im0 = axes[0].imshow(np.log10(np.clip(p_phys_grid, 1e-4, 1.0))[::3, ::3], cmap="magma", vmin=-3.0, vmax=-0.5)
    axes[0].set_title("A. Zero-Catalogue 96-Ch Physical Posterior\nlog10 P_phys(x) (Out-of-Quadrant OOF)", fontsize=10.2, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, label="log10 P_phys")

    gap_map = np.zeros(dots.shape, dtype=np.uint8)
    gap_map[foot & lvalid] = 1
    gap_map[foot & (~lvalid)] = 2
    from matplotlib.colors import ListedColormap
    cmap_gap = ListedColormap(["#f8fafc", "#e2e8f0", "#fde68a"])
    axes[1].imshow(gap_map[::3, ::3], cmap=cmap_gap, vmin=0, vmax=2)
    cy, cx = np.nonzero(cat[::3, ::3])
    axes[1].scatter(cx, cy, s=0.12, c="#475569", alpha=0.45, label="Mapped Catalogue (0 dots emitted)")
    dy_in, dx_in = np.nonzero((dots & lvalid)[::2, ::2])
    dy_gap, dx_gap = np.nonzero((dots & (~lvalid))[::2, ::2])
    axes[1].scatter(dx_in * (2 / 3), dy_in * (2 / 3), s=0.35, c="#0284c7", alpha=0.75, label=f"Dots in 1m LiDAR ({37654 - 6133:,})")
    axes[1].scatter(dx_gap * (2 / 3), dy_gap * (2 / 3), s=0.48, c="#dc2626", alpha=0.85, label="Dots in 24.6% Missing-LiDAR Gap (6,133)")
    axes[1].set_title("B. 37,654 Emitted Dots Across Nevada &\nSeamless 24.6% Missing-1m-LiDAR Bridge", fontsize=10.2, fontweight="bold")
    axes[1].axis("off")
    axes[1].legend(loc="lower left", markerscale=7.0, fontsize=7.8, framealpha=0.92)

    # Panel C: High-resolution zoom (65 km x 65 km central Nevada geothermal corridor)
    r0, r1, c0, c1 = 1350, 2000, 1250, 1900
    p_ens_zoom = surf["p_ens"][r0:r1, c0:c1]
    axes[2].imshow(np.log10(np.clip(p_ens_zoom, 1e-4, 1.0)), cmap="viridis", vmin=-3.1, vmax=-0.3)
    czy, czx = np.nonzero(cat[r0:r1, c0:c1])
    dzy, dzx = np.nonzero(dots[r0:r1, c0:c1])
    axes[2].scatter(czx, czy, s=0.5, c="#f43f5e", alpha=0.8, label="Mapped Faults (Hard Collar Zeroed)")
    axes[2].scatter(dzx, dzy, s=2.4, c="#facc15", edgecolors="none", alpha=0.95, label="H77E 300 m Submodular Dots")
    axes[2].set_title("C. 65 km × 65 km Basin-Range Detail:\nCrest-Centered 300 m Submodular Packing", fontsize=10.2, fontweight="bold")
    axes[2].axis("off")
    axes[2].legend(loc="lower right", markerscale=3.0, fontsize=8.0, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_dir / "h77pd_fig3_nevada_posterior_and_lidar_gap.png")
    plt.close(fig)

    # ── Figure 4: 870-Raster Uniqueness Spectrum, Sibling Overlaps & Spatial Shells ──
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.0, 5.6), dpi=200)

    def _clean_lbl(raw_id: str) -> str:
        s = raw_id.split(":", 1)[-1].replace(".tif", "")
        if "__" in s:
            parts = s.split("__")
            return f"{parts[0]}: {parts[-1][:24]}"
        return f"reg:{s[:14]}" if raw_id.startswith("registry:") else s[:30]

    top_inf = rc["uniqueness_audit"]["top_15_informative_rasters_by_1way_overlap"]
    seen_lbl = set()
    comb_rows = []
    for r in top_inf[:8] + [x for x in rc["uniqueness_audit"]["sibling_and_champion_comparison"] if x["rho_halo_allow"] < 0.70]:
        lb = _clean_lbl(r["id"])
        if lb not in seen_lbl and len(comb_rows) < 10:
            seen_lbl.add(lb)
            comb_rows.append(r)

    names_u = [_clean_lbl(r["id"]) for r in comb_rows]
    fwd_u = [r["fwd_1way_overlap"] for r in comb_rows]
    sym_u = [r["sym_overlap"] for r in comb_rows]
    y_u = np.arange(len(comb_rows))
    axA.barh(y_u - 0.18, fwd_u, height=0.34, color="#ea580c", label="1-Way Forward 3-px Overlap")
    axA.barh(y_u + 0.18, sym_u, height=0.34, color="#0284c7", label="Symmetric 3-px Overlap")
    axA.axvline(0.70, color="#dc2626", linestyle="--", linewidth=1.8, label="Protocol 0.70 Uniqueness Ceiling")
    axA.set_yticks(y_u)
    axA.set_yticklabels(names_u, fontsize=8.2, fontweight="bold")
    axA.set_xlim(0.0, 0.86)
    axA.set_xlabel("3-Pixel (300 m) Halo Overlap Fraction", fontsize=9.8, fontweight="bold")
    axA.set_title("A. 823/823 Non-Null Priors <= 0.6650", fontsize=10.5, fontweight="bold")
    axA.grid(axis="x", linestyle="--")
    axA.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    sp = rc["submission_package"]["spatial_distribution"]
    shell_lbls = [
        "0–300 m\nTip\nCollar",
        "300 m–1 km\nStep-Over\nZone",
        "1–2.5 km\nPiedmont\nSplay",
        "> 2.5 km\nBlind Basin\nInterior",
        "Inside 24.6%\nMissing-LiDAR\nGap",
    ]
    shell_pcts = [
        sp["shell_0_to_3px_300m_frac"] * 100.0,
        sp["shell_3_to_10px_1km_frac"] * 100.0,
        sp["shell_10_to_25px_2p5km_frac"] * 100.0,
        sp["shell_gt_25px_blind_basin_frac"] * 100.0,
        sp["dots_in_lidar_gap_fraction"] * 100.0,
    ]
    shell_cols = ["#38bdf8", "#0284c7", "#4f46e5", "#7c3aed", "#dc2626"]
    xs = np.arange(len(shell_lbls))
    axB.bar(xs, shell_pcts, width=0.52, color=shell_cols, edgecolor="#0f172a")
    for i, pct in enumerate(shell_pcts):
        n_px = int(round(pct / 100.0 * sp["emitted_dots"]))
        axB.text(i, pct + 0.8, f"{pct:.1f}%\n({n_px:,})", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    axB.set_xticks(xs)
    axB.set_xticklabels(shell_lbls, fontsize=8.3, fontweight="bold")
    axB.set_ylim(0.0, 46.0)
    axB.set_ylabel("Share of 37,654 Emitted Dots (%)", fontsize=9.8, fontweight="bold")
    axB.set_title("B. Discovery Shells & Missing-LiDAR Bridge", fontsize=10.5, fontweight="bold")
    axB.grid(axis="y", linestyle="--")

    fig.tight_layout(w_pad=3.2)
    fig.savefig(out_dir / "h77pd_fig4_uniqueness_and_pigeonhole_audit.png")
    plt.close(fig)

    print("Generated all 4 H77E publication figures cleanly in docs/assets/!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
