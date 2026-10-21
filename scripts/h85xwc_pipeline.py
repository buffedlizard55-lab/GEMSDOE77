#!/usr/bin/env python3
"""H85-XWC pipeline: Exp1 canaries -> Exp2 holdout -> Exp3 build+audit.

Uses ONLY the shared template for scoring/emission/writing/validation
(``vendor/gems57``); no private metric.  Preregistration:
``registry/preregistration_h85xwc.json`` / ``research/h85_hypotheses.md``.

Stages (idempotent; surfaces cached in ``.cache/h85/``):
    .venv/bin/python scripts/h85xwc_pipeline.py exp1
    .venv/bin/python scripts/h85xwc_pipeline.py exp2
    .venv/bin/python scripts/h85xwc_pipeline.py exp3
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import emit as emit_mod  # noqa: E402
from gems57 import evaluate_holdout, holdout, submission_writer, uniqueness, validate  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems77h85.fields import coherence_surface  # noqa: E402

CACHE = ROOT / ".cache" / "h85"
CACHE.mkdir(parents=True, exist_ok=True)
FEATURES = ROOT / "data" / "raw" / "training_features.tif"
EVID = ROOT / "evidence"
BUDGET = 37_654
NAME = "gems77-h85xwc-magxgrav-coherence-37654dots-20261021-zeros"
NOTE = "H85-XWC RTP x gravity multi-scale cross-coherence, label-blind; 37654 dotted sep3px >2px off catalogue"
assert len(NOTE) <= 140, len(NOTE)
SEEDS = (20, 30)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def surfaces(grid) -> dict[str, np.ndarray]:
    cache = CACHE / "surfaces.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    with rasterio.open(FEATURES) as src:
        if (src.height, src.width) != grid.footprint.shape:
            raise ValueError("training_features grid mismatch")
        rtp = src.read(2)    # band 2 = rtp (verified band map, vendor/gems57/geo.py)
        grav = src.read(13)  # band 13 = iso_grav_anom
    out = coherence_surface(rtp, grav, grid.footprint)
    np.savez_compressed(cache, **out)
    return out


def allowed_mask(grid, visible_catalogue: np.ndarray) -> np.ndarray:
    """Footprint cells >2 px from the *visible* catalogue (H33 hygiene rule)."""
    dist = ndi.distance_transform_edt(~visible_catalogue)
    return grid.footprint & (dist > 2.0)


def dotted(p: np.ndarray, allowed: np.ndarray, budget: int = BUDGET) -> np.ndarray:
    ranked, _ = emit_mod.emit_decimated(np.asarray(p, np.float32), allowed, tile_px=3)
    return emit_mod.emit_prefix(ranked, budget)


def random_dots(allowed: np.ndarray, budget: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(allowed)
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(budget, idx.size), replace=False)
    out = np.zeros(allowed.shape, bool)
    out.ravel()[pick] = True
    return out


def auc_canary(score: np.ndarray, truth: np.ndarray, domain: np.ndarray,
               seed: int = 85, neg_cap: int = 500_000) -> float:
    """Mann-Whitney AUC on all positives + seeded negative subsample (diagnostic)."""
    pos = score[truth & domain].astype(np.float64)
    neg_idx = np.flatnonzero(domain & ~truth)
    rng = np.random.default_rng(seed)
    if neg_idx.size > neg_cap:
        neg_idx = rng.choice(neg_idx, size=neg_cap, replace=False)
    neg = score.ravel()[neg_idx].astype(np.float64)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(allv.size)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    srt = allv[order]
    i = 0
    r_sorted = ranks[order]
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            r_sorted[i:j + 1] = r_sorted[i:j + 1].mean()
        i = j + 1
    ranks[order] = r_sorted
    rp = ranks[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def exp1() -> dict:
    grid = load_grid()
    surf = surfaces(grid)
    folds, _ = holdout.buffered_component_draw(grid, seed=20, hide_frac=0.20,
                                               buffer_px=3)
    hidden = folds[0]["hidden_all"]
    domain = grid.footprint & ~folds[0]["masked_known"]
    canaries = {}
    for key in ("xwc", "rtp_only", "grav_only", "coherence_mean", "phase_agreement"):
        canaries[key] = {
            "auc_vs_withheld_seed20": auc_canary(surf[key], hidden, domain),
            "auc_vs_full_catalogue_diagnostic": auc_canary(
                surf[key], grid.catalogue, grid.footprint),
        }
    leak = [k for k, v in canaries.items() if v["auc_vs_withheld_seed20"] > 0.90]
    receipt = {
        "experiment": "H85-XWC Exp1 single-feature leakage canaries",
        "evidence_class": "CANARY-DIAGNOSTIC; NOT HOLDOUT-DTI OR ORGANIZER SCORE",
        "generated_utc": now(),
        "features_sha256": sha256(FEATURES),
        "canaries": canaries,
        "leakage_stop_features_over_090": leak,
        "withheld_positives_seed20": int(hidden.sum()),
        "surface_stats": {k: {"min": float(v.min()), "max": float(v.max()),
                               "positive_in_footprint": int(((v > 0) & grid.footprint).sum())}
                          for k, v in surf.items()},
    }
    (EVID / "h85xwc_exp1_canary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if leak:
        raise SystemExit(f"LEAKAGE STOP: {leak} above AUC 0.90")
    return receipt


def exp2() -> dict:
    grid = load_grid()
    surf = surfaces(grid)
    result: dict = {
        "experiment": "H85-XWC Exp2 buffered whole-component holdout",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": evaluate_holdout.VERSION,
        "generated_utc": now(),
        "budget_dots_per_arm_global": BUDGET,
        "seeds": {},
    }
    for seed in SEEDS:
        folds, _ = holdout.buffered_component_draw(grid, seed=seed, hide_frac=0.20,
                                                   buffer_px=3)
        visible = folds[0]["visible"]
        allowed = allowed_mask(grid, visible)
        arms = {
            "xwc": dotted(surf["xwc"], allowed),
            "rtp_only": dotted(surf["rtp_only"], allowed),
            "grav_only": dotted(surf["grav_only"], allowed),
            "random": random_dots(allowed, BUDGET, seed=10_000 + seed),
        }
        terms_by_arm: dict[str, np.ndarray] = {}
        per_fold = {}
        for arm, dots in arms.items():
            pred = dots.astype(np.float32)
            total = None
            per_fold[arm] = {}
            for fold in folds:
                res, terms = evaluate_holdout.evaluate(pred, fold, grid.footprint)
                total = terms if total is None else total + terms
                per_fold[arm][fold["name"]] = {
                    "dti": float(res["dti"]), "n_truth": int(res["n_truth"]),
                    "n_emitted_in_region": int(res["n_emitted"])}
            terms_by_arm[arm] = total
            np.save(CACHE / f"dots_seed{seed}_{arm}.npy", dots)
        pooled = evaluate_holdout.pooled_summary(terms_by_arm, draws=1000,
                                                 candidate="xwc")
        result["seeds"][str(seed)] = {
            "pooled": pooled,
            "per_fold": per_fold,
            "fold_receipts": [f["receipt"] for f in folds],
            "dots_per_arm": {a: int(d.sum()) for a, d in arms.items()},
            "dots_on_full_catalogue": {a: int((d & grid.catalogue).sum())
                                        for a, d in arms.items()},
        }
        print(f"seed {seed}: " + ", ".join(
            f"{a}={pooled['scores'][a]['dti']:.4f}" for a in pooled["scores"]))
    (EVID / "h85xwc_exp2_holdout.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def exp3() -> dict:
    grid = load_grid()
    surf = surfaces(grid)
    allowed = allowed_mask(grid, grid.catalogue)  # full catalogue for the live build
    dots = dotted(surf["xwc"], allowed)
    pred = np.where(dots, 1.0, 0.0).astype(np.float32)
    out_path = ROOT / "docs" / "downloads" / f"{NAME}.tif"
    sample = ROOT / "data" / "bridge" / "sample_submission.tif"
    writer_receipt = submission_writer.write_submission(
        out_path, pred, sample, grid.footprint, note=NOTE, name=NAME,
        catalogue=grid.catalogue,
        metadata={"lane": "H85-XWC", "budget_dots": BUDGET,
                  "surface": "RTP x isostatic-gravity multi-scale cross-coherence"})
    val = validate.validate(out_path, grid.footprint, grid.catalogue)

    # --- uniqueness: surface rho (all) + 3px dot overlap (sparse dot rasters) ---
    mine_surf = surf["xwc"]
    rows, errors = [], []
    candidates = sorted((ROOT / "docs" / "downloads").glob("*.tif"))
    extra = [ROOT / "data" / "raw" / "scored" /
             "gems24-h25-1-dotted-h19-5-d2-8-20261002-e56ea318af89-nan.tif",
             ROOT / "data" / "raw" / "scored" /
             "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif"]
    fp_n = int(grid.footprint.sum())
    for path in candidates + [p for p in extra if p.exists()]:
        if path == out_path:
            continue
        try:
            with rasterio.open(path) as src:
                if (src.height, src.width) != grid.footprint.shape or src.count != 1:
                    errors.append(f"{path.name}: grid/band mismatch, skipped")
                    continue
                theirs = src.read(1)
            pos = int((np.isfinite(theirs) & (theirs > 0) & grid.footprint).sum())
            frac = pos / fp_n
            rho = uniqueness.surface_rho(mine_surf, theirs, grid.footprint)
            row = {"file": path.name, "positive_in_footprint": pos,
                   "positive_fraction": frac,
                   "surface_spearman_full_footprint": rho["spearman_full_footprint"]}
            # dot overlap is the protocol's dot test; meaningful for dot rasters.
            # Computed for every raster and classified, never silently dropped.
            row["forward_dot_overlap_3px"] = uniqueness.dot_overlap(dots, theirs)
            row["class"] = "sparse_dot_raster" if frac <= 0.10 else "dense_surface"
            rows.append(row)
            print(f"  uniq {path.name[:60]:60s} frac={frac:.4f} "
                  f"rho={row['surface_spearman_full_footprint']:+.4f} "
                  f"ovl={row['forward_dot_overlap_3px']:.4f}", flush=True)
        except Exception as exc:  # recorded, never hidden
            errors.append(f"{path.name}: {exc!r}")
    sparse = [r for r in rows if r["class"] == "sparse_dot_raster"]
    dense = [r for r in rows if r["class"] == "dense_surface"]
    def _max(rs, k):
        vals = [abs(r[k]) for r in rs if r[k] is not None and np.isfinite(r[k])]
        return float(max(vals)) if vals else None
    uniq = {
        "rasters_compared": len(rows), "errors": errors,
        "sparse_dot_rasters": len(sparse), "dense_surfaces": len(dense),
        "max_abs_surface_rho_sparse": _max(sparse, "surface_spearman_full_footprint"),
        "max_abs_surface_rho_dense": _max(dense, "surface_spearman_full_footprint"),
        "max_forward_dot_overlap_3px_sparse": _max(sparse, "forward_dot_overlap_3px"),
        "max_forward_dot_overlap_3px_dense_literal": _max(dense, "forward_dot_overlap_3px"),
        "worst_sparse_by_overlap": max(sparse, key=lambda r: r["forward_dot_overlap_3px"],
                                       default=None),
        "rows": rows,
        "irregularity_IR_H85_01": (
            "Literal reading: dense continuous registry surfaces positive on ~100% of "
            "allowed cells force forward dot overlap = 1.0 for ANY non-empty candidate "
            "(incl. H33 itself); that gate is reported here verbatim and remains the "
            "protocol owner's to resolve. Informative reading applies the protocol's "
            "two tests as typed: Spearman rho to surfaces, 3px dot overlap to dot rasters."),
    }
    receipt = {
        "experiment": "H85-XWC Exp3 final build + uniqueness + validation",
        "evidence_class": "FORMAT/REGISTRY MEASUREMENT; NOT A SCORE",
        "generated_utc": now(),
        "file": out_path.name, "sha256": sha256(out_path),
        "dots": int(dots.sum()), "dots_on_catalogue": int((dots & grid.catalogue).sum()),
        "min_catalogue_distance_px": float(
            ndi.distance_transform_edt(~grid.catalogue)[dots].min()) if dots.any() else None,
        "writer_receipt_file": f"{NAME}.json",
        "validator": val,
        "uniqueness": uniq,
    }
    (EVID / "h85xwc_exp3_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "uniqueness"}, indent=2))
    print({k: v for k, v in uniq.items() if k != "rows"})
    return receipt


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"exp1": exp1, "exp2": exp2, "exp3": exp3}[stage]()
