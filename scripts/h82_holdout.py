#!/usr/bin/env python3
"""GEMSDOE77 H82 Experiment 1: hanging-wall fault-damage collar, hide-and-recover holdout.

Preregistered at registry/preregistration_h82.json (Rank-1 hypothesis from research/h82_hypotheses.md).
Everything here is geometry-only: the SHA-pinned bridge grid (data/bridge), the shared gems57 holdout
instrument (buffered whole-component hide-and-recover), and the shared DTI evaluator
(gems57-pooled-hide-v2, alpha=0.2, beta=0.8, 300 m triangular kernel). No training, no external data.

Outputs: evidence/h82_holdout.json (canary + holdout arms + pooled CI) and evidence/h82_dots_ranked.npy
(deterministic re-derivable emission mask used by scripts/h82_build_submission.py).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems57 import metric as M  # noqa: E402
from gems77 import emit77  # noqa: E402

BRIDGE = ROOT / "data" / "bridge"
SEED = 30
BUDGET = 37_654
MIN_DOT_SEP_PX = 2.8
COLLAR_PEAK_PX = 3.0      # one 300 m kernel radius
COLLAR_CUTOFF_PX = 12.0   # ~1.2 km


def collar_profile(d: np.ndarray) -> np.ndarray:
    """Zero on the trace, peaked at one kernel radius, zero beyond ~1.2 km."""
    d = np.asarray(d, np.float32)
    p = np.zeros_like(d)
    m1 = (d > 1.0) & (d <= COLLAR_PEAK_PX)
    p[m1] = np.sin(np.pi / 2.0 * (d[m1] - 1.0) / (COLLAR_PEAK_PX - 1.0)) ** 2
    m2 = (d > COLLAR_PEAK_PX) & (d <= COLLAR_CUTOFF_PX)
    p[m2] = np.cos(np.pi / 2.0 * (d[m2] - COLLAR_PEAK_PX) / (COLLAR_CUTOFF_PX - COLLAR_PEAK_PX)) ** 2
    return p


def symmetric_auc(values: np.ndarray, label: np.ndarray) -> float | None:
    """Rank AUC; symmetric so either polarity counts (canary convention)."""
    m = np.isfinite(values) & (label >= 0)
    if m.sum() < 50 or len(np.unique(label[m])) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(label[m].astype(int), values[m]))
    return max(auc, 1.0 - auc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args()
    t0 = time.time()

    grid = G.load_grid(BRIDGE)
    foot, cat = grid.footprint, grid.catalogue
    log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
    log(f"footprint {int(foot.sum()):,} px  catalogue {int(cat.sum()):,} px")

    folds, quad = HO.buffered_component_draw(grid, seed=args.seed, hide_frac=0.20, buffer_px=3)
    hidden_all = folds[0]["hidden_all"]
    log(f"folds: {[f['name'] for f in folds]}  hidden positives {int(hidden_all.sum()):,}  "
        f"visible {int(folds[0]['visible'].sum()):,}")

    # ---- shared per-fold quantities ----
    dist_to_visible = ndi.distance_transform_edt(~folds[0]["visible"]).astype(np.float32)
    # negative pixels for canary: footprint, off catalogue, >3 px from any catalogue
    dist_cat = ndi.distance_transform_edt(~cat).astype(np.float32)
    neg_pool = np.flatnonzero(foot & ~cat & (dist_cat > 3))
    rng = np.random.default_rng(1234)

    results = {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "holdout_seed": args.seed,
        "fold_split": "buffered-whole-components-loqo-v2",
        "budget": args.budget,
        "withheld_positive_pixels_total_draw": int(hidden_all.sum()),
        "canary_auc": {},
        "arms": {},
        "runtime_note": "CPU sandbox, deterministic; geometry-only, no training, no external data",
    }

    # ---- leakage canary: surface alone, per fold ----
    canary = {}
    collar_flat = collar_profile(dist_to_visible).ravel()
    collar_full_flat = collar_profile(ndi.distance_transform_edt(~cat)).ravel()
    pos_idx = np.flatnonzero(hidden_all)
    neg_idx = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])
    canary["collar_visible_only"] = symmetric_auc(collar_flat[idx], y)
    # leaky reference: distance to FULL catalogue (uses hidden truth as feature - not for submission)
    canary["collar_full_catalogue_LEAKY_CONTROL"] = symmetric_auc(collar_full_flat[idx], y)
    worst = max((v for v in canary.values() if v is not None), default=0.0)
    results["canary_auc"] = canary
    results["canary_worst_symmetric_auc"] = worst
    results["leakage_flag_gt_0p90"] = worst > 0.90
    log(f"canary worst symmetric AUC {worst:.4f}  leakage_flag {worst > 0.90}")
    del collar_flat, collar_full_flat, idx, y; gc.collect()

    # ---- emission: per fold, budget split by scored-domain area ----
    region_areas = {f["name"]: int(f["region"].sum()) for f in folds}
    total_area = float(sum(region_areas.values()))
    dot_mask = np.zeros(foot.shape, bool)
    per_budget = {}
    for f in folds:
        b = max(1, int(round(args.budget * region_areas[f["name"]] / total_area)))
        per_budget[f["name"]] = b
        p = collar_profile(dist_to_visible)
        p = np.where(foot & f["region"], p, 0.0).astype(np.float32)
        allowed = f["region"] & foot & ~cat & (dist_to_visible > 0.5)
        dots = emit77.emit_dots(p, allowed, b, min_sep=MIN_DOT_SEP_PX)
        dot_mask |= dots
        log(f"fold {f['name']}: budget {b:,} emitted {int(dots.sum()):,}")

    dot_mask &= foot
    results["dots_emitted"] = int(dot_mask.sum())
    results["dots_on_catalogue"] = int((dot_mask & cat).sum())
    results["min_dot_distance_to_catalogue_px"] = float(
        ndi.distance_transform_edt(~cat)[dot_mask].min()) if dot_mask.any() else None
    np.save(ROOT / "evidence" / "h82_dots_ranked.npy", dot_mask)
    log(f"total dots {int(dot_mask.sum()):,}")

    # ---- exact per-fold DTI + pooled terms ----
    terms_by_arm = {"H82": None, "A0_random": None}
    per_fold = {}
    for arm in ("H82", "A0_random"):
        dots = dot_mask if arm == "H82" else None
        if arm == "A0_random":
            r2 = np.random.default_rng(999)
            surf = np.zeros(foot.shape, np.float32)
            surf[r2.random(foot.shape) < 0.05] = 1.0  # candidate pool for emitter
            dots = np.zeros(foot.shape, bool)
            for f in folds:
                b = per_budget[f["name"]]
                allowed = f["region"] & foot & ~cat & (dist_to_visible > 0.5)
                sub = emit77.emit_dots(surf, allowed, b, min_sep=MIN_DOT_SEP_PX)
                dots |= sub
            dots &= foot
        terms = None
        pf = {}
        for f in folds:
            res, t = EH.evaluate(dots.astype(np.float32), f, valid=foot, block_side=200)
            pf[f["name"]] = {"dti": res["dti"], "n_truth": int(res["n_truth"]),
                             "emitted": int(res["n_emitted"]), "coverage": res["coverage"]}
            terms = t if terms is None else terms + t
        terms_by_arm[arm] = terms
        results["arms"][arm] = {"dots": int(dots.sum()), "per_fold": pf,
                                "pooled_holdout_dti": float(M.dti_from_components(
                                    terms[:, 0].sum(), terms[:, 1].sum(), terms[:, 2].sum()))}
        log(f"{arm}: pooled HOLDOUT-DTI={results['arms'][arm]['pooled_holdout_dti']:.6f} "
            f"dots={int(dots.sum()):,}")

    summary = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="H82")
    results["pooled_summary"] = summary
    s = summary["scores"]["H82"]
    log(f"H82 HOLDOUT-DTI {s['dti']:.6f} 95% CI [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}] "
        f"withheld {s['withheld_positive_pixels']}")
    ctrl = summary["best_comparable_control"]
    log(f"best control {ctrl}: {summary['scores'][ctrl]['dti']:.6f}; "
        f"delta vs random {summary['paired_differences']['A0_random']['delta']:+.6f} "
        f"CI [{summary['paired_differences']['A0_random']['ci95'][0]:+.6f}, "
        f"{summary['paired_differences']['A0_random']['ci95'][1]:+.6f}]")

    # ---- registry uniqueness (final dots), locally accessible subset only ----
    idx_path = ROOT / "registry" / "registry_index.json"
    raw_index = json.loads(idx_path.read_text())
    # The pinned index stores one record per raster; convert to the gate's record schema.
    # Missing completeness certificate => diagnostic-only scan (fail-closed: never a clearance).
    registry_records = [
        dict(
            repo=r.get("repo"),
            submission=(r.get("aliases") or [r.get("path", "").rsplit("/", 1)[-1]])[0],
            file=r.get("path"),
            sha256=r.get("blob"),
            sources=list(r.get("aliases") or []),
        )
        for r in raw_index.get("rasters", [])
    ]
    registry_index = registry_records  # list form -> records_from_index treats it as diagnostic-only
    sys.path.insert(0, str(ROOT / "vendor"))
    from gems57 import uniqueness as U  # noqa: E402

    def prog(i, n, row):
        if i % 100 == 0 or i == n:
            log(f"registry gate dots {i}/{n}")

    uniq_report = U.compare_array_to_registry(
        dot_mask.astype(np.float32), registry_index, foot,
        file_sha256=hashlib.sha256(dot_mask.astype(np.float32).tobytes()).hexdigest(),
        progress=prog,
        expected_grid=(foot.shape, grid.crs, G.TRANSFORM),
    )
    results["uniqueness_vs_registry"] = {
        "phase": "dots",
        "registry_rasters_expected": uniq_report["registry_rasters_expected"],
        "registry_rasters_checked": uniq_report["registry_rasters_checked"],
        "complete_accessible_scan": uniq_report["complete_accessible_scan"],
        "missing_or_invalid": uniq_report["missing_or_invalid"],
        "worst_spearman_full_footprint": uniq_report["worst_spearman_full_footprint"],
        "worst_forward_overlap_3px": uniq_report["worst_dot_overlap"],
        "literal_firings": int(uniq_report["duplicate_count"]),
        "protocol_unique": bool(uniq_report["unique"]),
        "scope": uniq_report["scope"],
    }
    log(f"registry: checked {uniq_report['registry_rasters_checked']}/"
        f"{uniq_report['registry_rasters_expected']} complete={uniq_report['complete_accessible_scan']} "
        f"worst_rho={uniq_report['worst_spearman_full_footprint']} worst_ov3px={uniq_report['worst_dot_overlap']} "
        f"duplicate_count={uniq_report['duplicate_count']} unique={uniq_report['unique']}")

    results["elapsed_s"] = round(time.time() - t0, 1)
    (ROOT / "evidence").mkdir(exist_ok=True)
    out = ROOT / "evidence" / "h82_holdout.json"
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    log(f"DONE {results['elapsed_s']}s -> {out}")


if __name__ == "__main__":
    main()
