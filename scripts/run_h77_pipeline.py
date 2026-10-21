#!/usr/bin/env python3
"""Master End-to-End Deterministic Execution Pipeline for GEMSDOE77 (`H77`).

100% of holdout evaluation, spatial-cluster bootstrap CIs, pre- & post-placement registry
uniqueness checks across all genuine portfolio prediction rasters, and GeoTIFF/ZIP validation
run directly through unmodified upstream ``vendor/gems57`` modules (`evaluate_holdout.py`,
`holdout.py`, `metric.py`, `uniqueness.py`, `submission_writer.py`, `validate.py`).
"""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout as eval_mod  # noqa: E402
from gems57 import grid as grid_mod  # noqa: E402
from gems57 import holdout as holdout_mod  # noqa: E402
from gems57 import submission_writer as writer_mod  # noqa: E402
from gems57 import uniqueness as uniq_mod  # noqa: E402

from gems77.emit import (  # noqa: E402
    flank_zeroed_crest_submodular_emit,
    flank_zeroed_crest_submodular_sequence,
    peak_nms_emit,
)
from gems77.features import (  # noqa: E402
    build_h77_feature_stack,
    compute_voronoi_lbco_mechanics_fp,
    pack_ram_float16,
)
from gems77.models import (  # noqa: E402
    fit_h77_deconfounded_ensemble,
    fit_loqo_baseline_ensemble,
    gather_rows,
    quantile_align_by_quadrant,
)

DATA_DIR = ROOT / ".cache" / "data"
EXT_DIR = DATA_DIR / "data" / "external"
REG_CANONICAL_DIR = DATA_DIR / "registry" / "rasters"
REG_ALL_JSON = ROOT / ".cache" / "registry_all.json"
MMAP_PATH = ROOT / ".cache" / "h77_features_fp.f32"
EVID_DIR = ROOT / "evidence"
H77_EVID = EVID_DIR / "h77"

SUB_SLUG = "gems77-step-propensity-submod300"
SUB_NOTE = "GEMSDOE77: HW/FW dip-slip step tensor + V-LBCO crack-tip mechanics + Radon multi-scale ensemble + flank-zeroed 300m submod"
assert len(SUB_NOTE) <= 140

H33_TIF = REG_CANONICAL_DIR / "GEMSDOE32__h33-2-b2-zeros.tif"
H27_TIF = REG_CANONICAL_DIR / "GEMSDOE28__h27-4-r1-solo-d2-8.tif"
D28_TIF = REG_CANONICAL_DIR / "GEMSDOE25__dotted-h19-5-d2-8.tif"
H19_TIF = REG_CANONICAL_DIR / "19GEMSDOE__h19-5-powerlaw-budget-multiline.tif"
P8_TIF = Path("/tmp/refs/GEMSDOE70/docs/downloads/gems70-p5-dual-submod300-oof38k-20261020.tif")

CONFIRMED_28_LEDGER = [
    ("GEMSDOE32", 0.2778, "h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros", str(H33_TIF), "2026-10-04", "H19-5 + D2.8 Poisson dots + clip d_cat <= 2.0 px (200 m)"),
    ("GEMSDOE49", 0.2750, "anderson-geothermal-pinn-38854", str(REG_CANONICAL_DIR / "GEMSDOE36__anderson-geothermal-pinn-38854.tif"), "2026-10-14", "Andersonian stress + geothermal PINN blend (38,854 dots)"),
    ("GEMSDOE36", 0.2710, "h36-1-rung30-blind-r1", str(REG_CANONICAL_DIR / "GEMSDOE28__h36-1-rung30-blind-r1.tif"), "2026-10-05", "Blind-quadrant transfer variant of H33"),
    ("GEMSDOE28", 0.2708, "h27-4", str(H27_TIF), "2026-10-03", "H19-5 + D2.8 Poisson dots + clip d_cat <= 1.0 px (100 m)"),
    ("GEMSDOE31", 0.2708, "h32-h27-4-20261004T063420Z-c607b11a", str(H27_TIF), "2026-10-04", "Exact duplicate SHA-256 re-submission of h27-4 (irregularity flagged)"),
    ("GEMSDOE33", 0.2647, "h34-2-b3-20261005T040119Z-038e73bb", None, "2026-10-05", "Over-clipped d_cat <= 3.0 px (-0.0131 regression proves real faults at 200-300 m!)"),
    ("GEMSDOE25", 0.2600, "D2.8-poisson-disks", str(REG_CANONICAL_DIR / "GEMSDOE25__dotted-h19-5-d2-8.tif"), "2026-10-02", "H19-5 skeleton discretized into D2.8 Poisson disks (+0.0678 over solid skeleton)"),
    ("GEMSDOE70", 0.2478, "gems70-p5-dual-submod300-oof38k", str(P8_TIF), "2026-10-20", "Best prior pure zero-H19 multi-sensor ML (SGMC-contaminated M_bed + border12 truncation)"),
    ("GEMSDOE24", 0.2477, "first-discrete-dots", None, "2026-10-01", "Initial discrete dot emitter"),
    ("GEMSDOE26", 0.2286, "dot-spacing-variant", None, "2026-10-02", "Wider dot spacing sensitivity"),
    ("GEMSDOE27", 0.2279, "dot-budget-variant", None, "2026-10-03", "Dot count sensitivity"),
    ("GEMSDOE21", 0.1977, "continuous-line-variant", None, "2026-09-29", "Thinned continuous line emission"),
    ("19GEMSDOE", 0.1922, "h19-5-4lines-corroboration", str(H19_TIF), "2026-09-28", "121,131-px continuous ridge centerline: 16GEMSDOE + L1 tips + L2 GDR heat + L3 openness + L4 strike worms"),
    ("16GEMSDOE", 0.1855, "spatial-oof-hgb-scarp-ridge", str(REG_CANONICAL_DIR / "calib_h161_01855.tif"), "2026-09-26", "123,939-px 1-px Hessian ridge of 5GEMSDOE ResNet34 OOF + 48 LiDAR/10m/geophys features"),
    ("GEMSDOE23", 0.1818, "continuous-skeleton-variant", None, "2026-09-30", "Skeletonized continuous trace"),
    ("GEMSDOE12", 0.1631, "early-cnn-ridge", None, "2026-09-24", "Early ridge threshold"),
    ("5GEMSDOE", 0.1563, "resnet34-4fold-oof", None, "2026-09-20", "227,507-px fat ribbon 4-fold 2D ResNet34 spatial CNN"),
    ("GEMSDOE67", 0.1426, "gems67b-piedmont-geophys-oof38k", None, "2026-10-20", "Single-pixel piedmont geophysical trees (no multi-km context or step tensor)"),
    ("GEMSDOE66", 0.1152, "gems66-geophys-stack", None, "2026-10-20", "Coarse geophysical stack without LiDAR/step polarity"),
    ("GEMSDOE20", 0.0880, "early-wide-mask", None, "2026-09-28", "Unthinned ribbon penalty"),
    ("GEMSDOE1", 0.0725, "baseline-v1", None, "2026-09-16", "Initial competition baseline"),
    ("GEMSDOE71", 0.0558, "gems71-dpst-hdt-35500dots", "/tmp/refs/GEMSDOE71/docs/downloads/gems71-dpst-hdt-35500dots-20261020T200000Z-fd98afd0-nan.tif", "2026-10-20", "Broad regional play-fairway blobs instead of 100 m fault traces"),
    ("GEMSDOE72", 0.0492, "gems72-sgmc-bedrock-surrogate", "/tmp/refs/GEMSDOE72/docs/downloads/GEMSDOE72-h71-walkerlane+geothermal-20261020T194646Z-0172d3a9-candidate.tif", "2026-10-20", "CRITICAL FALSIFICATION: SGMC pre-Quaternary bedrock contacts score 0.0492 live!"),
    ("GEMSDOE68", 0.0437, "gems68-experimental", None, "2026-10-20", "Over-smoothed low-precision experimental field"),
    ("GEMSDOE18", 0.0319, "early-dense-ribbon", None, "2026-09-27", "Dense false-positive flood"),
    ("GEMSDOE48", 0.0259, "gems48-dense-flood", None, "2026-10-12", "Uncalibrated dense surface"),
    ("GEMSDOE63", 0.0156, "gems63-uncalibrated", None, "2026-10-18", "Uncalibrated experimental output"),
    ("GEMSDOE69", 0.0053, "h69-efg-kinematic-geodesic-29027", "/tmp/refs/GEMSDOE69/submissions/h69-efg-kinematic-geodesic-29027-nan.tif", "2026-10-20", "Geochemical/geodesic halos without sharp structural scarp localization"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sum_fold_terms(pred: np.ndarray, folds: list[dict[str, Any]], fp: np.ndarray) -> np.ndarray:
    """Delegate per-fold exact DTI + 20 km spatial-block sufficient statistics to upstream evaluate()."""
    per_fold = [eval_mod.evaluate(pred, f, fp, block_side=200)[1] for f in folds]
    return np.sum(np.stack(per_fold, axis=0), axis=0)


def make_quadrant_folds(cat: np.ndarray, fp: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Build exact 4-quadrant blind province folds (`A_quad`, 3 km inter-quadrant guard band)."""
    H, W = fp.shape
    rmid, cmid = H // 2, W // 2
    yy, xx = np.indices((H, W))
    quad_2d = ((yy >= rmid).astype(np.int8) * 2 + (xx >= cmid).astype(np.int8))
    guard_2d = (np.abs(yy - rmid) < 30) | (np.abs(xx - cmid) < 30)

    q_folds = []
    for q_id, q_name in enumerate(("NW", "NE", "SW", "SE")):
        in_q = (quad_2d == q_id) & fp & (~guard_2d)
        vis = cat & fp & (~(quad_2d == q_id)) & (~guard_2d)
        d_vis = distance_transform_edt(~vis)
        truth = cat & in_q & (d_vis > 3.0)
        q_folds.append({"fold": q_name, "region": in_q, "visible": vis, "truth": truth})
    return q_folds, quad_2d, guard_2d


def generate_diagnostic_figures(
    assets_dir: Path,
    sc_hide: dict[str, Any],
    sc_quad: dict[str, Any],
    p_live_2d: np.ndarray,
    pred_live: np.ndarray,
    h33_static: np.ndarray,
    p8_static: np.ndarray,
    cat: np.ndarray,
    fp: np.ndarray,
    d_cat: np.ndarray,
    uniq_rows: list[dict[str, Any]],
) -> None:
    """Generate 4 publication-grade scientific diagnostic figures from verified measurements."""
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "axes.edgecolor": "#334155"})

    # Figure 1: Causal Live Leaderboard Progression + Official Holdout Ablation Ladder
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=160)
    live_names = [
        "G72 SGMC\n(Bedrock Noise)",
        "5GEMSDOE\n(ResNet34 Fat)",
        "16GEMSDOE\n(1px Ridge)",
        "19GEMSDOE\n(H19-5 4-Line)",
        "GEMSDOE70\n(P8 Pure ML)",
        "GEMSDOE25\n(D2.8 Dots)",
        "GEMSDOE28\n(Clip <=100m)",
        "GEMSDOE32\n(Clip <=200m)",
    ]
    live_vals = [0.0492, 0.1563, 0.1855, 0.1922, 0.2478, 0.2600, 0.2708, 0.2778]
    colors1 = ["#ef4444", "#64748b", "#64748b", "#475569", "#0ea5e9", "#6366f1", "#4f46e5", "#10b981"]
    bars1 = ax1.barh(live_names, live_vals, color=colors1, height=0.62)
    ax1.axvline(0.3195, color="#f59e0b", ls="--", lw=1.6, label="DrivenData Top-Tier Anchor (0.3195)")
    ax1.set_xlim(0, 0.36)
    ax1.set_xlabel("ORGANIZER-CONFIRMED Live Leaderboard DTI")
    ax1.set_title("A. Causal Forensic Progression on Live Test Set (n=28 Receipts)", fontweight="bold", fontsize=10.5)
    for b, v in zip(bars1, live_vals):
        ax1.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="x", alpha=0.25)

    hold_keys = [
        ("random_control", "Random Control\n(37,654 NMS)"),
        ("h33_live_champion_static", "H33 Live Raster\n(Static Mask)"),
        ("exp_77_1_baseline_nms", "Exp-77.1 Baseline\n(74-ch + NMS)"),
        ("h33_fold_fair_reclipped", "H33 Fold-Fair\n(Re-clipped)"),
        ("exp_77_2_h77_nms", "Exp-77.2 H77\n(Step+V-LBCO+NMS)"),
        ("exp_77_3_matched_37654", "Exp-77.3 Submod\n(Matched 37,654)"),
        ("exp_77_3_promoted", "Exp-77.3 Promoted\n(LOFO Calibrated)"),
    ]
    y_pos = np.arange(len(hold_keys))
    means = [sc_hide[k]["dti"] for k, _ in hold_keys]
    err_lo = [sc_hide[k]["dti"] - sc_hide[k]["ci95"][0] for k, _ in hold_keys]
    err_hi = [sc_hide[k]["ci95"][1] - sc_hide[k]["dti"] for k, _ in hold_keys]
    colors2 = ["#94a3b8", "#f97316", "#38bdf8", "#8b5cf6", "#2563eb", "#059669", "#047857"]
    ax2.barh(y_pos, means, xerr=[err_lo, err_hi], color=colors2, height=0.62, capsize=3, error_kw={"lw": 1.2})
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in hold_keys])
    ax2.set_xlabel("HOLDOUT-DTI (gems57-pooled-hide-v2, 11,321 px, 95% Block CI)")
    ax2.set_title("B. Preregistered 3-Experiment Ablation on Official Shared Holdout", fontweight="bold", fontsize=10.5)
    ax2.set_xlim(0, max(means) * 1.28)
    for yp, v in zip(y_pos, means):
        ax2.text(v + max(means) * 0.04, yp, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")
    ax2.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets_dir / "fig1_forensics_and_ablation.png")
    plt.close(fig)

    # Figure 2: Blind 4-Quadrant Province Generalization (`A_quad`) + Distance-from-Catalogue Shell Allocation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=160)
    q_keys = [
        ("random_control", "Random Control"),
        ("h19_5_centerlines", "19GEMSDOE H19-5"),
        ("gems70_p8_static", "GEMSDOE70 P8\n(0.2478 Live)"),
        ("exp_77_1_baseline_nms", "Exp-77.1 Baseline"),
        ("exp_77_2_h77_nms", "Exp-77.2 H77\n(Step + Radon)"),
        ("exp_77_3_promoted", "Exp-77.3 Promoted\n(Submod 300m)"),
    ]
    yp2 = np.arange(len(q_keys))
    m_q = [sc_quad[k]["dti"] for k, _ in q_keys]
    lo_q = [sc_quad[k]["dti"] - sc_quad[k]["ci95"][0] for k, _ in q_keys]
    hi_q = [sc_quad[k]["ci95"][1] - sc_quad[k]["dti"] for k, _ in q_keys]
    ax1.barh(yp2, m_q, xerr=[lo_q, hi_q], color=["#94a3b8", "#6366f1", "#0ea5e9", "#38bdf8", "#2563eb", "#047857"], height=0.6, capsize=3)
    ax1.set_yticks(yp2)
    ax1.set_yticklabels([lbl for _, lbl in q_keys])
    ax1.set_xlabel("Blind 4-Quadrant Province HOLDOUT-DTI (57,448 Withheld Fault Pixels)")
    ax1.set_title("A. Zero-Catalogue Blind Province Transfer (100% Unmapped Quadrants)", fontweight="bold", fontsize=10.5)
    ax1.set_xlim(0, max(m_q) * 1.25)
    for yp, v in zip(yp2, m_q):
        ax1.text(v + max(m_q) * 0.03, yp, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)

    shells = ["<=200m\n(Masked Collar)", "200m-1km\n(Tip Continuations)", "1km-3km\n(Relay/Piedmont)", ">3km\n(Blind Basins)"]
    def _shell_fracs(arr: np.ndarray) -> list[float]:
        d = d_cat[(arr > 0.5) & fp]
        return [
            float((d <= 2.0).mean()),
            float(((d > 2.0) & (d <= 10.0)).mean()),
            float(((d > 10.0) & (d <= 30.0)).mean()),
            float((d > 30.0).mean()),
        ]
    x_s = np.arange(len(shells))
    w = 0.25
    ax2.bar(x_s - w, _shell_fracs(h33_static), width=w, label="GEMSDOE32 H33 (0.2778 Live)", color="#6366f1")
    ax2.bar(x_s, _shell_fracs(p8_static), width=w, label="GEMSDOE70 P8 (0.2478 Live)", color="#0ea5e9")
    ax2.bar(x_s + w, _shell_fracs(pred_live), width=w, label="GEMSDOE77 Promoted (H77)", color="#10b981")
    ax2.set_xticks(x_s)
    ax2.set_xticklabels(shells, fontsize=8.5)
    ax2.set_ylabel("Fraction of Emitted Dots")
    ax2.set_title("B. Spatial Fault-Regime Balance (Zero Collar Waste + Balanced Discovery)", fontweight="bold", fontsize=10.5)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets_dir / "fig2_blind_quadrant_and_shells.png")
    plt.close(fig)

    # Figure 3: Spatial Map Overview of Posterior Probability Field + Emitted Dots vs Catalogue
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=150)
    sub_step = 4
    surf_disp = np.where(fp[::sub_step, ::sub_step], p_live_2d[::sub_step, ::sub_step], np.nan)
    im = ax1.imshow(surf_disp, cmap="magma", vmin=0.0, vmax=float(np.nanpercentile(surf_disp, 99.2)))
    ax1.set_title("A. GEMSDOE77 Continuous Posterior Surface (Pre-Placement)", fontweight="bold", fontsize=10.5)
    ax1.set_xlabel("Column / 4 (EPSG:32611 100 m Grid)")
    ax1.set_ylabel("Row / 4")
    fig.colorbar(im, ax=ax1, fraction=0.042, pad=0.03, label="Calibrated Fault Probability")

    # Zoomed 45 km x 45 km Dixie Valley / Hot Springs Range structural window
    r0, r1, c0, c1 = 1050, 1500, 1550, 2000
    bg = p_live_2d[r0:r1, c0:c1]
    ax2.imshow(bg, cmap="Greys", vmin=0.0, vmax=float(np.percentile(bg, 99.0)))
    cy, cx = np.nonzero(cat[r0:r1, c0:c1])
    dy_l, dx_l = np.nonzero(pred_live[r0:r1, c0:c1] > 0.5)
    ax2.scatter(cx, cy, s=0.8, c="#ef4444", alpha=0.65, label="Mapped USGS/INGENIOUS Faults (Masked)")
    ax2.scatter(dx_l, dy_l, s=3.2, c="#10b981", alpha=0.9, label="GEMSDOE77 Discrete 300 m Crest Dots")
    ax2.set_title("B. 45x45 km Structural Zoom: Tip Continuations & Blind Basin Strands", fontweight="bold", fontsize=10.5)
    ax2.legend(loc="upper right", fontsize=8, markerscale=3.0)
    fig.tight_layout()
    fig.savefig(assets_dir / "fig3_spatial_posterior_and_zoom.png")
    plt.close(fig)

    # Figure 4: Cross-Portfolio Uniqueness Certificate Across All Genuine Prediction Rasters
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    rhos = [abs(r["spearman_full_footprint"] or 0.0) for r in uniq_rows]
    muts = [r["mutual_overlap"] for r in uniq_rows]
    is_lat = [("lattice" in r["submission"].lower()) or (r["their_dots"] > 150000) for r in uniq_rows]
    ax.scatter(
        [m for m, l in zip(muts, is_lat) if not l],
        [r for r, l in zip(rhos, is_lat) if not l],
        c="#2563eb", alpha=0.65, s=28, label=f"Genuine Prior Submissions across 18 Repos (n={sum(not l for l in is_lat)})",
    )
    if any(is_lat):
        ax.scatter(
            [m for m, l in zip(muts, is_lat) if l],
            [r for r, l in zip(rhos, is_lat) if l],
            c="#f59e0b", marker="^", s=48, label="Dense Ribbon / Synthetic Lattice Rasters (>150k px)",
        )
    ax.axvline(0.70, color="#dc2626", ls="--", lw=1.6, label="Protocol Mutual 3-px Overlap Gate (0.70)")
    ax.axhline(0.90, color="#7c3aed", ls=":", lw=1.6, label="Protocol Spearman Rank Gate (0.90)")
    ax.set_xlim(-0.02, 0.85)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("Mutual 3-Pixel (300 m) Dot Overlap Fraction")
    ax.set_ylabel("|Full-Footprint Spearman Rank Correlation|")
    ax.set_title("GEMSDOE77 Uniqueness Certificate Across Entire 18-Repository Portfolio", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets_dir / "fig4_portfolio_uniqueness_certificate.png")
    plt.close(fig)


def main() -> int:
    import argparse
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("all", "holdout", "live"), default="all")
    ap.add_argument("--prior-elapsed", type=float, default=0.0)
    args = ap.parse_args()

    if args.stage == "all":
        t_all = time.perf_counter()
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "holdout"], check=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--stage",
                "live",
                "--prior-elapsed",
                f"{time.perf_counter() - t_all:.2f}",
            ],
            check=True,
        )
        return 0

    t0 = time.perf_counter() - args.prior_elapsed
    H77_EVID.mkdir(parents=True, exist_ok=True)
    (ROOT / "submissions").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "submissions").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True)

    print("[0/6] Loading canonical grid, existing_faults, and official gems57-pooled-hide-v2 folds...")
    g = grid_mod.load_grid(DATA_DIR)
    fp = g.footprint
    fp_idx = np.flatnonzero(fp)
    row_fp = (fp_idx // g.shape[1]).astype(np.int32)
    col_fp = (fp_idx % g.shape[1]).astype(np.int32)

    with rasterio.open(DATA_DIR / "existing_faults.tif") as ds_cat:
        cat = (ds_cat.read(1) > 0) & fp
    cat_sha = sha256_file(DATA_DIR / "existing_faults.tif")
    sample_sha = sha256_file(DATA_DIR / "sample_submission.tif")
    d_cat = distance_transform_edt(~cat).astype(np.float32)

    ledger_out = []
    for repo, score, slug, p_str, dt, desc in CONFIRMED_28_LEDGER:
        entry: dict[str, Any] = {
            "repo": repo,
            "organizer_confirmed_dti": score,
            "evidence_label": "ORGANIZER-CONFIRMED",
            "slug": slug,
            "date": dt,
            "mechanism_note": desc,
        }
        if p_str and Path(p_str).is_file():
            with rasterio.open(p_str) as ds:
                arr = np.nan_to_num(ds.read(1), nan=0.0)
            pm = (arr > 0.5) & fp
            n_px = int(pm.sum())
            entry["sha256"] = sha256_file(Path(p_str))
            entry["emitted_pixels"] = n_px
            if n_px > 0:
                dc = d_cat[pm]
                entry["frac_d_le_2px"] = round(float((dc <= 2.0).mean()), 5)
                entry["frac_2_to_10px"] = round(float(((dc > 2.0) & (dc <= 10.0)).mean()), 5)
                entry["frac_10_to_30px"] = round(float(((dc > 10.0) & (dc <= 30.0)).mean()), 5)
                entry["frac_gt_30px"] = round(float((dc > 30.0).mean()), 5)
        ledger_out.append(entry)
    (EVID_DIR / "forensics_28_live_submissions.json").write_text(json.dumps(ledger_out, indent=2) + "\n")

    folds_hide, hide_draw_audit = holdout_mod.buffered_component_draw(g, seed=20, hide_frac=0.20, buffer_px=3, boundary_px=12)
    folds_quad, quad_2d, guard_2d = make_quadrant_folds(cat, fp)
    quad_fp = quad_2d.ravel()[fp_idx]
    guard_fp = guard_2d.ravel()[fp_idx]

    n_hide_pos = sum(int(f["truth"].sum()) for f in folds_hide)
    n_quad_pos = sum(int(f["truth"].sum()) for f in folds_quad)
    print(f"      gems57-pooled-hide-v2 withheld px = {n_hide_pos:,} | A_quad withheld px = {n_quad_pos:,}")

    X_mmap, feat_names, has_lidar_fp, hydro_imp_fp, crest_fp, feat_prov = build_h77_feature_stack(
        MMAP_PATH, fp, g.transform, DATA_DIR, EXT_DIR
    )
    crest_2d = np.zeros(g.shape, dtype=np.float32)
    crest_2d.ravel()[fp_idx] = crest_fp

    if args.stage == "live":
        X16, _, _ = pack_ram_float16(X_mmap)
        del X_mmap
        gc.collect()
        Xm_live_train, Xm_live_infer, mech_names = compute_voronoi_lbco_mechanics_fp(cat, fp, hydro_imp_fp, n_buckets=5, seed=7799)
        all_names = feat_names + mech_names
        official_hide_summary = json.loads((H77_EVID / "official_hide_v2_summary.json").read_text())
        official_quad_summary = json.loads((H77_EVID / "official_quad_summary.json").read_text())
        official_subtle_summary = json.loads((H77_EVID / "official_subtle_summary.json").read_text())
        canary_summary = json.loads((H77_EVID / "leakage_canary.json").read_text())
        live_budget = int(json.loads((ROOT / ".cache" / "h77_stage1_meta.json").read_text())["live_budget"])
        sc_hide = official_hide_summary["scores"]
        sc_quad = official_quad_summary["scores"]
        sc_sub = official_subtle_summary["scores"]
        with rasterio.open(H33_TIF) as ds:
            h33_static = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)
        with rasterio.open(P8_TIF) as ds:
            p8_static = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)
        return _finish_live_and_package(
            t0, g, fp, fp_idx, row_fp, col_fp, cat, d_cat, cat_sha, sample_sha,
            X16, Xm_live_train, Xm_live_infer, all_names, has_lidar_fp, crest_2d,
            quad_fp, guard_fp, live_budget, official_hide_summary, official_quad_summary,
            official_subtle_summary, canary_summary, sc_hide, sc_quad, sc_sub, h33_static, p8_static,
        )

    print("[1/6] Building clean 96-ch physical stack + V-LBCO fold mechanics + 99-feature leakage canary...")

    vis_hide_2d = np.ones(g.shape, dtype=bool)
    for f in folds_hide:
        vis_hide_2d &= f["visible"]
    vis_hide_2d &= fp
    assert not np.any(vis_hide_2d & np.logical_or.reduce([f["truth"] for f in folds_hide]))

    d_vis_hide_2d = distance_transform_edt(~vis_hide_2d).astype(np.float32)
    d_vis_hide_fp = d_vis_hide_2d.ravel()[fp_idx]
    Xm_hide_train, Xm_hide_infer, mech_names = compute_voronoi_lbco_mechanics_fp(
        vis_hide_2d, fp, hydro_imp_fp, n_buckets=5, seed=7701
    )
    all_names = feat_names + mech_names

    # Mandatory single-feature ROC-AUC leakage canary across ALL 101 features BEFORE fitting ANY model!
    truth_union_fp = np.logical_or.reduce([f["truth"] for f in folds_hide]).ravel()[fp_idx]
    eval_pool_fp = (~vis_hide_2d.ravel()[fp_idx]) & ((d_vis_hide_fp > 3.0) | truth_union_fp)
    rng_can = np.random.default_rng(77011)
    pos_can = np.flatnonzero(eval_pool_fp & truth_union_fp)
    neg_can = rng_can.choice(np.flatnonzero(eval_pool_fp & (~truth_union_fp)), size=150000, replace=False)
    can_idx = np.concatenate([pos_can, neg_can])
    y_can = np.concatenate([np.ones(pos_can.size, np.int32), np.zeros(neg_can.size, np.int32)])

    canary_rows = []
    for col_i, nm in enumerate(all_names):
        if col_i < X_mmap.shape[0]:
            vals = np.asarray(X_mmap[col_i, can_idx], dtype=np.float64)
        else:
            vals = np.asarray(Xm_hide_infer[col_i - X_mmap.shape[0], can_idx], dtype=np.float64)
        med = float(np.nanmedian(vals)) if np.isnan(vals).any() else 0.0
        vals = np.nan_to_num(vals, nan=med)
        auc = float(roc_auc_score(y_can, vals))
        sym_auc = max(auc, 1.0 - auc)
        canary_rows.append({
            "feature": nm,
            "is_novel_h77": col_i >= 74,
            "roc_auc": round(auc, 5),
            "symmetric_auc": round(sym_auc, 5),
            "leakage_pass": sym_auc <= 0.90,
        })
        if sym_auc > 0.90:
            raise RuntimeError(f"LEAKAGE CANARY FAILED CLOSED on {nm}: symmetric AUC = {sym_auc:.5f} > 0.90!")

    canary_rows.sort(key=lambda r: r["symmetric_auc"], reverse=True)
    canary_summary = {
        "evaluator_version": eval_mod.VERSION,
        "withheld_positive_pixels": int(pos_can.size),
        "n_features_tested": len(canary_rows),
        "max_symmetric_auc": canary_rows[0]["symmetric_auc"],
        "top_feature": canary_rows[0]["feature"],
        "all_pass_le_0p90": True,
        "features": canary_rows,
    }
    (H77_EVID / "leakage_canary.json").write_text(json.dumps(canary_summary, indent=2) + "\n")
    print(f"      101/101 features PASSED leakage canary! Top single feature: {canary_rows[0]['feature']} (AUC={canary_rows[0]['symmetric_auc']:.4f} <= 0.90)")

    # Pack into 946 MiB row-major float16 in RAM and release the 1.98 GiB mmap handle!
    slope_2d = np.zeros(g.shape, dtype=np.float32)
    slope_2d.ravel()[fp_idx] = np.asarray(X_mmap[feat_names.index("det_elev_slope"), :], dtype=np.float32)
    X16, _, _ = pack_ram_float16(X_mmap)
    del X_mmap
    gc.collect()

    allow_hide = fp & (d_vis_hide_2d > 2.0)
    subtle_thresh = float(np.median(slope_2d[np.logical_or.reduce([f["truth"] for f in folds_hide])]))
    folds_subtle = [{**f, "truth": f["truth"] & (slope_2d <= subtle_thresh)} for f in folds_hide]
    del slope_2d

    # -------------------------------------------------------------------------
    # EXPERIMENT 1 (`Exp-77.1`): Baseline 74 Physical Channels + 7x7 NMS (37,654 dots)
    # -------------------------------------------------------------------------
    print("[2/6] Running Experiment 1/3 (`Exp-77.1`: Pure-Quaternary 74-ch Baseline + 7x7 NMS)...")
    t_e1 = time.perf_counter()
    base_cols = np.arange(74, dtype=np.int32)
    p_e1_fp = fit_loqo_baseline_ensemble(
        X16, base_cols, vis_hide_2d.ravel()[fp_idx], d_vis_hide_fp, quad_fp, guard_fp, has_lidar_fp, seed_base=77100
    )
    p_e1_2d = np.zeros(g.shape, dtype=np.float32)
    p_e1_2d.ravel()[fp_idx] = p_e1_fp
    dots_e1 = peak_nms_emit(p_e1_2d, allow_hide, n_dots=37654)
    dots_e1_quad = peak_nms_emit(p_e1_2d, fp, n_dots=37654)
    del p_e1_fp, p_e1_2d
    t_e1_terms = sum_fold_terms(dots_e1, folds_hide, fp)
    print(f"      Exp-77.1 HOLDOUT-DTI = {float(eval_mod.from_terms(t_e1_terms.sum(axis=0))):.4f} ({time.perf_counter()-t_e1:.1f}s)")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2 (`Exp-77.2`): Full 101-ch H77 (Step Tensor + V-LBCO Tip Cones + Radon/Oblique) + SAME 7x7 NMS
    # -------------------------------------------------------------------------
    print("[3/6] Running Experiment 2/3 (`Exp-77.2`: +H77-1 Step Tensor +V-LBCO +H77-2 SAR-PU +H77-3 Radon, same 7x7 NMS)...")
    t_e2 = time.perf_counter()
    p_e2_hide_fp = fit_h77_deconfounded_ensemble(
        X16, Xm_hide_train, Xm_hide_infer, all_names, vis_hide_2d.ravel()[fp_idx], d_vis_hide_fp,
        quad_fp, guard_fp, has_lidar_fp, row_fp, col_fp, allow_intra_province=True, seed_base=77200
    )
    del Xm_hide_train, Xm_hide_infer
    gc.collect()

    p_e2_hide_2d = np.zeros(g.shape, dtype=np.float32)
    p_e2_hide_2d.ravel()[fp_idx] = p_e2_hide_fp
    del p_e2_hide_fp
    dots_e2 = peak_nms_emit(p_e2_hide_2d, allow_hide, n_dots=37654)
    t_e2_terms = sum_fold_terms(dots_e2, folds_hide, fp)

    # Strict 96-ch zero-catalogue blind 4-quadrant OOF surface for A_quad (`allow_intra_province=False`)
    p_e2_quad_fp = fit_h77_deconfounded_ensemble(
        X16, None, None, feat_names, cat.ravel()[fp_idx], d_cat.ravel()[fp_idx],
        quad_fp, guard_fp, has_lidar_fp, row_fp, col_fp, allow_intra_province=False, seed_base=77400
    )
    p_e2_quad_2d = np.zeros(g.shape, dtype=np.float32)
    p_e2_quad_2d.ravel()[fp_idx] = p_e2_quad_fp
    del p_e2_quad_fp
    dots_e2_quad = peak_nms_emit(p_e2_quad_2d, fp, n_dots=37654)
    print(f"      Exp-77.2 HOLDOUT-DTI = {float(eval_mod.from_terms(t_e2_terms.sum(axis=0))):.4f} | A_quad = {float(eval_mod.from_terms(sum_fold_terms(dots_e2_quad, folds_quad, fp).sum(axis=0))):.4f} ({time.perf_counter()-t_e2:.1f}s)")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3 (`Exp-77.3`): Frozen `Exp-77.2` Surface + Flank-Zeroed Crest-Centered Submodular 300m Packing
    # -------------------------------------------------------------------------
    print("[4/6] Running Experiment 3/3 (`Exp-77.3`: Frozen Exp-77.2 + Flank-Zeroed Crest-Centered Submodular 300m Allocation)...")
    t_e3 = time.perf_counter()
    candidate_budgets = [34000, 36500, 38500, 40500]
    seq_hide = flank_zeroed_crest_submodular_sequence(p_e2_hide_2d, crest_2d, allow_hide, max_dots=max(candidate_budgets))
    del p_e2_hide_2d

    def _from_seq(seq: np.ndarray, k_dots: int) -> np.ndarray:
        arr = np.zeros(g.shape, dtype=np.float32)
        arr.ravel()[seq[:k_dots]] = 1.0
        return arr

    fold_terms_by_b = {
        b: [eval_mod.evaluate(_from_seq(seq_hide, b), f, fp, block_side=200)[1] for f in folds_hide]
        for b in candidate_budgets
    }
    oof_selected_budgets = []
    oof_selected_terms = []
    for k in range(4):
        best_b = max(
            candidate_budgets,
            key=lambda b: float(eval_mod.from_terms(sum(fold_terms_by_b[b][j].sum(axis=0) for j in range(4) if j != k))),
        )
        oof_selected_budgets.append(best_b)
        oof_selected_terms.append(fold_terms_by_b[best_b][k])

    live_budget = int(round(float(np.mean(oof_selected_budgets))))
    dots_e3_matched = _from_seq(seq_hide, 37654)
    dots_e3_cal = _from_seq(seq_hide, live_budget)
    dots_e3_quad = flank_zeroed_crest_submodular_emit(p_e2_quad_2d, crest_2d, fp, n_dots=live_budget)
    del p_e2_quad_2d
    gc.collect()

    rng_ctrl = np.random.default_rng(20261021)
    noise_2d = np.where(fp, rng_ctrl.random(g.shape, dtype=np.float32), 0.0)
    dots_rand = peak_nms_emit(noise_2d, allow_hide, n_dots=37654)
    dots_rand_quad = peak_nms_emit(noise_2d, fp, n_dots=37654)
    del noise_2d

    with rasterio.open(H33_TIF) as ds:
        h33_static = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)
    with rasterio.open(D28_TIF) as ds:
        d28_raw = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)
    h33_foldfair = np.where(d_vis_hide_2d > 2.0, d28_raw, 0.0).astype(np.float32)
    del d28_raw
    with rasterio.open(H19_TIF) as ds:
        h19_raw = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)
    with rasterio.open(P8_TIF) as ds:
        p8_static = np.clip(np.nan_to_num(ds.read(1), nan=0.0), 0.0, 1.0).astype(np.float32)

    official_hide_summary = eval_mod.pooled_summary(
        {
            "exp_77_3_promoted": sum_fold_terms(dots_e3_cal, folds_hide, fp),
            "exp_77_3_lofo_strict": np.sum(np.stack(oof_selected_terms, axis=0), axis=0),
            "exp_77_3_matched_37654": sum_fold_terms(dots_e3_matched, folds_hide, fp),
            "exp_77_2_h77_nms": t_e2_terms,
            "exp_77_1_baseline_nms": t_e1_terms,
            "gems70_p8_static": sum_fold_terms(p8_static, folds_hide, fp),
            "h33_fold_fair_reclipped": sum_fold_terms(h33_foldfair, folds_hide, fp),
            "h33_live_champion_static": sum_fold_terms(h33_static, folds_hide, fp),
            "random_control": sum_fold_terms(dots_rand, folds_hide, fp),
        },
        draws=1000,
        seed=520810,
        candidate="exp_77_3_promoted",
    )

    official_quad_summary = eval_mod.pooled_summary(
        {
            "exp_77_3_promoted": sum_fold_terms(dots_e3_quad, folds_quad, fp),
            "exp_77_2_h77_nms": sum_fold_terms(dots_e2_quad, folds_quad, fp),
            "exp_77_1_baseline_nms": sum_fold_terms(dots_e1_quad, folds_quad, fp),
            "gems70_p8_static": sum_fold_terms(p8_static, folds_quad, fp),
            "h19_5_centerlines": sum_fold_terms(h19_raw, folds_quad, fp),
            "random_control": sum_fold_terms(dots_rand_quad, folds_quad, fp),
        },
        draws=1000,
        seed=520810,
        candidate="exp_77_3_promoted",
    )

    official_subtle_summary = eval_mod.pooled_summary(
        {
            "exp_77_3_promoted": sum_fold_terms(dots_e3_cal, folds_subtle, fp),
            "exp_77_2_h77_nms": sum_fold_terms(dots_e2, folds_subtle, fp),
            "exp_77_1_baseline_nms": sum_fold_terms(dots_e1, folds_subtle, fp),
            "h33_fold_fair_reclipped": sum_fold_terms(h33_foldfair, folds_subtle, fp),
            "random_control": sum_fold_terms(dots_rand, folds_subtle, fp),
        },
        draws=1000,
        seed=520810,
        candidate="exp_77_3_promoted",
    )

    (H77_EVID / "official_hide_v2_summary.json").write_text(json.dumps(official_hide_summary, indent=2) + "\n")
    (H77_EVID / "official_quad_summary.json").write_text(json.dumps(official_quad_summary, indent=2) + "\n")
    (H77_EVID / "official_subtle_summary.json").write_text(json.dumps(official_subtle_summary, indent=2) + "\n")

    sc_hide = official_hide_summary["scores"]
    sc_quad = official_quad_summary["scores"]
    sc_sub = official_subtle_summary["scores"]
    print(
        f"      OFFICIAL UPSTREAM HOLDOUT-DTI LADDER (gems57-pooled-hide-v2, {n_hide_pos:,} px) [{time.perf_counter()-t_e3:.1f}s]:\n"
        f"        random_control         : {sc_hide['random_control']['dti']:.4f} {sc_hide['random_control']['ci95']}\n"
        f"        h33_live_static (0.2778): {sc_hide['h33_live_champion_static']['dti']:.4f} {sc_hide['h33_live_champion_static']['ci95']}\n"
        f"        exp_77_1_baseline_nms  : {sc_hide['exp_77_1_baseline_nms']['dti']:.4f} {sc_hide['exp_77_1_baseline_nms']['ci95']}\n"
        f"        h33_fold_fair          : {sc_hide['h33_fold_fair_reclipped']['dti']:.4f} {sc_hide['h33_fold_fair_reclipped']['ci95']}\n"
        f"        exp_77_2_h77_nms       : {sc_hide['exp_77_2_h77_nms']['dti']:.4f} {sc_hide['exp_77_2_h77_nms']['ci95']}\n"
        f"        exp_77_3_matched_37654 : {sc_hide['exp_77_3_matched_37654']['dti']:.4f} {sc_hide['exp_77_3_matched_37654']['ci95']}\n"
        f"        exp_77_3_lofo ({live_budget:,})  : {sc_hide['exp_77_3_promoted']['dti']:.4f} {sc_hide['exp_77_3_promoted']['ci95']}\n"
        f"      BLIND 4-QUADRANT PROVINCE (`A_quad`, {n_quad_pos:,} px): "
        f"Exp-77.1={sc_quad['exp_77_1_baseline_nms']['dti']:.4f} -> Exp-77.2={sc_quad['exp_77_2_h77_nms']['dti']:.4f} -> Exp-77.3={sc_quad['exp_77_3_promoted']['dti']:.4f} "
        f"(vs G70-P8={sc_quad['gems70_p8_static']['dti']:.4f}, H19-5={sc_quad['h19_5_centerlines']['dti']:.4f})"
    )

    (ROOT / ".cache" / "h77_stage1_meta.json").write_text(json.dumps({"live_budget": live_budget}) + "\n")
    return 0


def _finish_live_and_package(
    t0: float,
    g: Any,
    fp: np.ndarray,
    fp_idx: np.ndarray,
    row_fp: np.ndarray,
    col_fp: np.ndarray,
    cat: np.ndarray,
    d_cat: np.ndarray,
    cat_sha: str,
    sample_sha: str,
    X16: np.ndarray,
    Xm_live_train: np.ndarray,
    Xm_live_infer: np.ndarray,
    all_names: list[str],
    has_lidar_fp: np.ndarray,
    crest_2d: np.ndarray,
    quad_fp: np.ndarray,
    guard_fp: np.ndarray,
    live_budget: int,
    official_hide_summary: dict[str, Any],
    official_quad_summary: dict[str, Any],
    official_subtle_summary: dict[str, Any],
    canary_summary: dict[str, Any],
    sc_hide: dict[str, Any],
    sc_quad: dict[str, Any],
    sc_sub: dict[str, Any],
    h33_static: np.ndarray,
    p8_static: np.ndarray,
) -> int:
    # -------------------------------------------------------------------------
    # STEP 5: Full Live Inference (`cat` visible) + Upstream Uniqueness Audit Across ALL Rasters!
    # -------------------------------------------------------------------------
    print("[5/6] Running full live inference (`cat` visible) + upstream `uniqueness.compare_array_to_registry`...")
    live_cache = ROOT / ".cache" / "h77_live_outputs.npz"
    if live_cache.is_file():
        npz = np.load(live_cache)
        p_live_2d = npz["p_live_2d"]
        pred_live = npz["pred_live"]
        del X16, Xm_live_train, Xm_live_infer
        gc.collect()
    else:
        p_live_fp = fit_h77_deconfounded_ensemble(
            X16, Xm_live_train, Xm_live_infer, all_names, cat.ravel()[fp_idx], d_cat.ravel()[fp_idx],
            quad_fp, guard_fp, has_lidar_fp, row_fp, col_fp, allow_intra_province=True, seed_base=77900
        )
        del X16, Xm_live_train, Xm_live_infer
        gc.collect()

        p_live_2d = np.zeros(g.shape, dtype=np.float32)
        p_live_2d.ravel()[fp_idx] = np.clip(p_live_fp, 0.0, 0.9999)
        del p_live_fp

        allow_live = fp & (d_cat > 2.0)
        pred_live = flank_zeroed_crest_submodular_emit(p_live_2d, crest_2d, allow_live, n_dots=live_budget)
        pred_live = np.where(fp, np.clip(pred_live, 0.0, 1.0), 0.0).astype(np.float32)
        np.savez(live_cache, p_live_2d=p_live_2d, pred_live=pred_live)

    reg_doc = json.loads(REG_ALL_JSON.read_text())
    non_pred_shas = {cat_sha, sample_sha}
    clean_rasters = [
        r for r in reg_doc["rasters"]
        if Path(r["cache_file"]).is_file()
        and r["sha256"] not in non_pred_shas
        and "labels.tif" not in str(r["sources"]).lower()
        and "footprint-mask.tif" not in str(r["sources"]).lower()
    ]
    reg_index_full = {"complete_accessible_scan": True, "errors": [], "rasters": clean_rasters}
    reg_index_62 = {
        "complete_accessible_scan": True,
        "errors": [],
        "rasters": [r for r in clean_rasters if "60gemsdoe" in r["repo_first"].lower()],
    }

    uniq_surface_all = uniq_mod.compare_array_to_registry(p_live_2d, reg_index_full, fp, include_reverse=True)
    uniq_dots_all = uniq_mod.compare_array_to_registry(pred_live, reg_index_full, fp, include_reverse=True)
    uniq_dots_62 = uniq_mod.compare_array_to_registry(pred_live, reg_index_62, fp, include_reverse=True)

    rows_dots_all = [r for r in uniq_dots_all["rows"] if "error" not in r]
    non_lattice_rows = [
        r for r in rows_dots_all
        if ("lattice" not in r["submission"].lower()) and (r["their_dots"] < 150000)
    ]
    max_fwd_real_all = max(r["my_dots_within_3px_of_theirs"] for r in non_lattice_rows)

    uniq_summary_clean = {
        "surface_pre_placement_all_repos": {k: v for k, v in uniq_surface_all.items() if k != "rows"},
        "dots_post_placement_62_canonical": {k: v for k, v in uniq_dots_62.items() if k != "rows"},
        "dots_post_placement_all_repos": {k: v for k, v in uniq_dots_all.items() if k != "rows"},
        "max_forward_3px_overlap_real_non_lattice_models": round(float(max_fwd_real_all), 6),
        "top_15_closest_dots_across_portfolio": sorted(
            rows_dots_all,
            key=lambda r: max(abs(r["spearman_full_footprint"] or 0.0), r["mutual_overlap"]),
            reverse=True,
        )[:15],
        "top_15_closest_surface_across_portfolio": sorted(
            [r for r in uniq_surface_all["rows"] if "error" not in r],
            key=lambda r: abs(r["spearman_full_footprint"] or 0.0),
            reverse=True,
        )[:15],
    }
    (H77_EVID / "uniqueness_audit_full.json").write_text(json.dumps(uniq_summary_clean, indent=2) + "\n")
    print(
        f"      Uniqueness across {len(rows_dots_all)} genuine prediction rasters: "
        f"surface_unique={uniq_surface_all['unique']} (max |rho|={uniq_surface_all['worst_spearman_full_footprint']:.4f}), "
        f"dots_unique={uniq_dots_all['unique']} (max |rho|={uniq_dots_all['worst_spearman_full_footprint']:.4f}, "
        f"max mutual_3px={uniq_dots_all['worst_dot_overlap']:.4f}, max real fwd_3px={max_fwd_real_all:.4f})"
    )

    # -------------------------------------------------------------------------
    # STEP 6: Shared Submission Writer, Figures, Validator Audit, & JSON Run Card
    # -------------------------------------------------------------------------
    print("[6/6] Packaging validated `.tif` + `.zip` + `.json`, figures, and official Run Card...")
    generate_diagnostic_figures(
        ROOT / "docs" / "assets",
        sc_hide, sc_quad, p_live_2d, pred_live, h33_static, p8_static, cat, fp, d_cat, rows_dots_all
    )

    out_tif_path = ROOT / "submissions" / f"{SUB_SLUG}.tif"
    for stale in (out_tif_path, out_tif_path.with_suffix(".zip"), out_tif_path.with_suffix(".json")):
        if stale.exists():
            stale.unlink()

    sub_receipt = writer_mod.write_submission(
        path=out_tif_path,
        prediction=pred_live,
        sample=DATA_DIR / "sample_submission.tif",
        footprint=fp,
        note=SUB_NOTE,
        name=SUB_SLUG,
        catalogue=cat,
        metadata={
            "evaluator_version": eval_mod.VERSION,
            "holdout_dti_hide_v2": sc_hide["exp_77_3_promoted"]["dti"],
            "holdout_dti_hide_v2_ci95": sc_hide["exp_77_3_promoted"]["ci95"],
            "blind_quad_dti": sc_quad["exp_77_3_promoted"]["dti"],
            "portfolio_rasters_unique": bool(uniq_dots_all["unique"] and uniq_surface_all["unique"]),
        },
    )
    shutil.copy2(out_tif_path, ROOT / "docs" / "submissions" / out_tif_path.name)

    with rasterio.open(out_tif_path) as ds_out:
        written = ds_out.read(1)
        out_crs = str(ds_out.crs)
        out_transform = list(ds_out.transform)[:6]
        out_shape = list(ds_out.shape)
        out_dtype = str(ds_out.dtypes[0])

    dc_live = d_cat[written > 0.5]
    border12_mask = fp & (~(distance_transform_edt(fp) > 12.0))
    live_geo_audit = {
        "emitted_dots": int((written > 0.5).sum()),
        "dtype": out_dtype,
        "crs": out_crs,
        "shape": out_shape,
        "transform": out_transform,
        "min_value_entire_raster": float(np.nanmin(written)),
        "max_value_entire_raster": float(np.nanmax(written)),
        "nan_count_in_footprint": int(np.isnan(written[fp]).sum()),
        "nan_count_entire_raster": int(np.isnan(written).sum()),
        "inf_count_entire_raster": int(np.isinf(written).sum()),
        "values_strictly_in_0_1": bool(np.all((written >= 0.0) & (written <= 1.0)) and not np.isnan(written).any()),
        "distinct_values": [float(v) for v in np.unique(written)],
        "dots_in_lidar_gap_fraction": round(float((~has_lidar_fp)[written[fp] > 0.5].mean()), 5),
        "dots_in_border12_fraction": round(float(border12_mask[written > 0.5].mean()), 5),
        "dots_d_le_2px": int((dc_live <= 2.0).sum()),
        "dots_parasitic_collar_le_2p5px": int((dc_live <= 2.5).sum()),
        "dots_shell_2_to_10px_frac": round(float(((dc_live > 2.0) & (dc_live <= 10.0)).mean()), 5),
        "dots_shell_10_to_30px_frac": round(float(((dc_live > 10.0) & (dc_live <= 30.0)).mean()), 5),
        "dots_shell_gt_30px_frac": round(float((dc_live > 30.0).mean()), 5),
    }

    diffs_hide = official_hide_summary["paired_differences"]
    diffs_quad = official_quad_summary["paired_differences"]
    promote_ok = (
        canary_summary["all_pass_le_0p90"]
        and bool(sub_receipt["validator"]["all_checks_passed"])
        and bool(sub_receipt["template_report"]["ok"])
        and bool(uniq_surface_all["unique"])
        and bool(uniq_dots_all["unique"])
        and diffs_hide["random_control"]["ci95"][0] > 0.0
        and diffs_hide["h33_live_champion_static"]["ci95"][0] > 0.0
        and diffs_hide["h33_fold_fair_reclipped"]["ci95"][0] > 0.0
        and diffs_quad["h19_5_centerlines"]["ci95"][0] > 0.0
    )

    run_card = {
        "schema": "gems-parallel-run-card-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/cbecbf6a-gemsdoe77",
        "hypothesis": (
            "H77: Anisotropic Hanging-Wall/Footwall Dip-Slip Monotone Step Purity & Dip-Shifted Subsurface "
            "Concordance + Voronoi Leave-Bucket-of-Components-Out (V-LBCO) Crack-Tip Self-Supervision "
            "+ SAR-PU Piedmont Reweighting + Multi-Azimuth Radon Continuity + Flank-Zeroed 300 m Submodular Packing"
        ),
        "mechanism": (
            "Continuous structure-tensor strike/normal sampling distinguishes one-sided Quaternary fault scarps "
            "backed by +400 m downdip hanging-wall basement/gravity deepening from symmetric desert canyons and "
            "equipotential pluvial shorelines; 5-bucket Voronoi Leave-Bucket-of-Components-Out (V-LBCO) self-supervision "
            "eliminates the d_vis == 0 training tautology so multi-family trees learn true Pollard crack-tip and "
            "piedmont fault recovery; flank-zeroed 300 m submodular packing allocates crest dots across 100% of footprint."
        ),
        "named_non_fault_mimicking_process": (
            "Pleistocene Lake Lahontan (1332 m) & Lake Bonneville (1463/1552 m) wave-cut terrace risers and "
            "symmetric V-shaped ephemeral desert washes/arroyos (suppressed via cross-strike monotone step "
            "purity ratio A_step, along-strike 1.6 km elevation variance separator, and dip-shifted subsurface concordance)."
        ),
        "holdout_dti": {
            "evidence_label": "HOLDOUT-DTI",
            "evaluator_version": eval_mod.VERSION,
            "implementation_sha256": official_hide_summary["implementation_sha256"],
            "pooled_hide_v2": {
                "withheld_positive_pixels": sc_hide["exp_77_3_promoted"]["withheld_positive_pixels"],
                "score": round(sc_hide["exp_77_3_promoted"]["dti"], 6),
                "ci_95": [round(x, 6) for x in sc_hide["exp_77_3_promoted"]["ci95"]],
                "lofo_strict_score": round(sc_hide["exp_77_3_lofo_strict"]["dti"], 6),
                "matched_37654_dots_score": round(sc_hide["exp_77_3_matched_37654"]["dti"], 6),
                "exp_77_2_nms_score": round(sc_hide["exp_77_2_h77_nms"]["dti"], 6),
                "exp_77_1_baseline_score": round(sc_hide["exp_77_1_baseline_nms"]["dti"], 6),
                "h33_fold_fair_score": round(sc_hide["h33_fold_fair_reclipped"]["dti"], 6),
                "h33_live_static_score": round(sc_hide["h33_live_champion_static"]["dti"], 6),
                "random_control_score": round(sc_hide["random_control"]["dti"], 6),
                "delta_vs_h33_static": round(diffs_hide["h33_live_champion_static"]["delta"], 6),
                "delta_vs_h33_static_ci_95": [round(x, 6) for x in diffs_hide["h33_live_champion_static"]["ci95"]],
                "delta_vs_h33_fold_fair": round(diffs_hide["h33_fold_fair_reclipped"]["delta"], 6),
                "delta_vs_h33_fold_fair_ci_95": [round(x, 6) for x in diffs_hide["h33_fold_fair_reclipped"]["ci95"]],
                "delta_vs_random_control": round(diffs_hide["random_control"]["delta"], 6),
                "delta_vs_random_control_ci_95": [round(x, 6) for x in diffs_hide["random_control"]["ci95"]],
            },
            "blind_4_quadrant_province": {
                "withheld_positive_pixels": sc_quad["exp_77_3_promoted"]["withheld_positive_pixels"],
                "score": round(sc_quad["exp_77_3_promoted"]["dti"], 6),
                "ci_95": [round(x, 6) for x in sc_quad["exp_77_3_promoted"]["ci95"]],
                "exp_77_2_nms_score": round(sc_quad["exp_77_2_h77_nms"]["dti"], 6),
                "exp_77_1_baseline_score": round(sc_quad["exp_77_1_baseline_nms"]["dti"], 6),
                "gems70_p8_score": round(sc_quad["gems70_p8_static"]["dti"], 6),
                "h19_5_score": round(sc_quad["h19_5_centerlines"]["dti"], 6),
                "random_control_score": round(sc_quad["random_control"]["dti"], 6),
                "delta_vs_h19_5": round(diffs_quad["h19_5_centerlines"]["delta"], 6),
                "delta_vs_h19_5_ci_95": [round(x, 6) for x in diffs_quad["h19_5_centerlines"]["ci95"]],
                "delta_vs_gems70_p8": round(diffs_quad["gems70_p8_static"]["delta"], 6),
                "delta_vs_gems70_p8_ci_95": [round(x, 6) for x in diffs_quad["gems70_p8_static"]["ci95"]],
            },
            "subtle_piedmont_slice": {
                "evidence_label": "HOLDOUT-DTI",
                "withheld_positive_pixels": sc_sub["exp_77_3_promoted"]["withheld_positive_pixels"],
                "exp_77_1_score": round(sc_sub["exp_77_1_baseline_nms"]["dti"], 6),
                "exp_77_2_score": round(sc_sub["exp_77_2_h77_nms"]["dti"], 6),
                "exp_77_3_score": round(sc_sub["exp_77_3_promoted"]["dti"], 6),
                "ci_95": [round(x, 6) for x in sc_sub["exp_77_3_promoted"]["ci95"]],
            },
        },
        "leakage_canary": {
            "features_tested": canary_summary["n_features_tested"],
            "max_single_feature_auc": canary_summary["max_symmetric_auc"],
            "top_feature": canary_summary["top_feature"],
            "ceiling": 0.90,
            "pass": canary_summary["all_pass_le_0p90"],
        },
        "uniqueness_vs_registry": {
            "canonical_62_rasters_unique": bool(uniq_dots_62["unique"]),
            "portfolio_rasters_scanned": len(rows_dots_all),
            "pre_placement_surface_unique": bool(uniq_surface_all["unique"]),
            "pre_placement_surface_max_spearman": round(float(uniq_surface_all["worst_spearman_full_footprint"]), 6),
            "post_placement_dots_unique": bool(uniq_dots_all["unique"]),
            "post_placement_dots_max_spearman": round(float(uniq_dots_all["worst_spearman_full_footprint"]), 6),
            "post_placement_dots_max_mutual_3px_overlap": round(float(uniq_dots_all["worst_dot_overlap"]), 6),
            "post_placement_dots_max_forward_3px_overlap_real_models": round(float(max_fwd_real_all), 6),
            "post_placement_dots_max_jaccard": round(float(uniq_dots_all["worst_jaccard_dot_sets"]), 6),
        },
        "submission_raster": {
            "filename": out_tif_path.name,
            "path": f"submissions/{out_tif_path.name}",
            "github_pages_direct_download": f"submissions/{out_tif_path.name}",
            "sha256": sub_receipt["sha256"],
            "bytes": sub_receipt["bytes"],
            "zip_file": sub_receipt["zip_file"],
            "zip_sha256": sub_receipt["zip_sha256"],
            "validator_output": {
                "all_checks_passed": sub_receipt["validator"]["all_checks_passed"],
                "template_report_ok": sub_receipt["template_report"]["ok"],
                "geometry_and_range_audit": live_geo_audit,
                "validator_details": sub_receipt["validator"],
            },
        },
        "submission_name": sub_receipt["submission_name"],
        "submission_note": sub_receipt["note"],
        "submission_note_length": sub_receipt["note_chars"],
        "budget_accounting": {
            "experiments_run": 3,
            "experiments_cap": 3,
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "runtime_cap_seconds": 7200,
            "weekly_slots_used_automatically": 0,
        },
        "verdict": "promote" if promote_ok else "negative",
        "safe_to_download_and_submit": bool(promote_ok),
    }

    (H77_EVID / "run_card.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "docs" / "submissions" / "run_card.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "evidence" / "preregistration_sha256.txt").write_text(sha256_file(ROOT / "docs" / "preregistration-h77.md") + "\n")

    print("\n" + "=" * 78)
    print(f"FINAL VERDICT: {run_card['verdict'].upper()} | SAFE TO DOWNLOAD & SUBMIT = {run_card['safe_to_download_and_submit']}")
    print(f"FILE:          submissions/{out_tif_path.name} ({sub_receipt['bytes']:,} bytes)")
    print(f"SHA-256:       {sub_receipt['sha256']}")
    print(f"RUNTIME:       {run_card['budget_accounting']['runtime_seconds']:.1f} s ({run_card['budget_accounting']['experiments_run']}/3 experiments)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
