#!/usr/bin/env python3
"""H79: fixed lagged-radiometric halo experiment, holdout, uniqueness and artifact.

Modes are deliberately separate so the pre-placement surface gate runs before
placement: ``surface``, ``holdout``, ``artifact``, ``final-uniqueness``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor" / "template66")]
from gems57 import evaluate_holdout as eh  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems57.holdout import buffered_component_draw  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems57.uniqueness import compare_array_to_registry  # noqa: E402
from gems77lia.allocate import greedy_select  # noqa: E402

DATA = ROOT / ".cache" / "gems_data"
WORK = ROOT / ".cache" / "work"
REGINDEX = ROOT / "registry" / "registry_index.json"
N_DOTS = 37_654
LAG = 3
SIGMA = 1.5
SEED = 20
NAME = "gems77-h79-radhalo-lag3-37654-20261021-zeros"
NOTE = "H79 lag-3px U-residual halo paired to K-Th edge; 37654 spaced dots; HOLDOUT negative—research only, do not submit"


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _rank01(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, np.float32)
    v = a[valid]
    r = rankdata(v, method="average").astype(np.float32)
    out[valid] = (r - 1.0) / max(len(r) - 1, 1)
    return out


def features() -> dict[str, np.ndarray]:
    WORK.mkdir(parents=True, exist_ok=True)
    cache = WORK / "h79_features.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import rasterio
    with rasterio.open(DATA / "external" / "geodawn_rad_u8.tif") as src:
        q = src.read([1, 2, 3]).astype(np.float32)
    grid = load_grid(DATA)
    rad_valid = grid.footprint & np.all(q > 0, axis=0)
    # Quantized channels are independent rank-intensities, not concentrations.
    k, th, u = (q[i] / np.float32(254.0) for i in range(3))
    common = (k + th) * np.float32(0.5)
    residual = u - common
    c = ndimage.gaussian_filter(np.where(rad_valid, common, 0), SIGMA)
    r = ndimage.gaussian_filter(np.where(rad_valid, residual, 0), SIGMA)
    cgy, cgx = np.gradient(c); rgy, rgx = np.gradient(r)
    common_edge = np.hypot(cgy, cgx).astype(np.float32)
    residual_edge = np.hypot(rgy, rgx).astype(np.float32)
    shifted = np.zeros_like(residual_edge)
    # Four fixed normals, both flanks. np.roll wrap is subsequently removed by an eroded mask.
    for dy, dx in ((0, LAG), (LAG, 0), (LAG, LAG), (LAG, -LAG)):
        for s in (-1, 1):
            shifted = np.maximum(shifted, np.roll(residual_edge, (s * dy, s * dx), axis=(0, 1)))
    valid = ndimage.binary_erosion(rad_valid, iterations=LAG + 2)
    cr = _rank01(common_edge, valid)
    rr = _rank01(residual_edge, valid)
    sr = _rank01(shifted, valid)
    # Spatial phase-offset response: joint edge evidence, downweight coincident selective-U edges.
    surface = np.sqrt(cr * sr) * (sr / (sr + rr + np.float32(1e-6)))
    surface[~valid] = 0
    if surface.max() > 0:
        surface /= surface.max()
    out = {"common_edge_rank": cr, "residual_edge_rank": rr,
           "shifted_residual_edge_rank": sr, "surface": surface.astype(np.float32),
           "radiometric_valid": valid.astype(np.uint8)}
    np.savez_compressed(cache, **out)
    return out


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.concatenate([pos, neg]); ranks = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def canaries(f: dict[str, np.ndarray]) -> dict:
    grid = load_grid(DATA); folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=.20, buffer_px=3)
    rng = np.random.default_rng(79)
    vals = {k: {"p": [], "n": []} for k in ("common_edge_rank", "residual_edge_rank", "shifted_residual_edge_rank", "surface")}
    valid_rad = f["radiometric_valid"].astype(bool)
    for fold in folds:
        region = fold["region"] & grid.footprint & ~fold["masked_known"] & valid_rad
        truth = fold["truth"] & region
        neg = region & ~truth
        ni = np.flatnonzero(neg); ni = rng.choice(ni, min(300_000, len(ni)), replace=False)
        pi = np.flatnonzero(truth)
        for k in vals:
            vals[k]["p"].append(f[k].ravel()[pi]); vals[k]["n"].append(f[k].ravel()[ni])
    out = {}
    for k, v in vals.items():
        a = _auc(np.concatenate(v["p"]), np.concatenate(v["n"]))
        symmetric = max(a, 1-a)
        out[k] = {"auc": a, "symmetric_auc": symmetric, "leakage_flag": symmetric > .90}
    return out


def registry_index() -> dict:
    idx = json.loads(REGINDEX.read_text())
    receipt = json.loads((ROOT / "registry" / "registry_receipt.json").read_text())["sha256_by_blob"]
    rasters = []
    for r in idx["rasters"]:
        blob = r["blob"]
        if blob not in receipt:
            continue
        rasters.append({"repo_first": r["repo"], "sources": r["aliases"],
                        "cache_file": str(ROOT / ".cache" / "registry" / f"{blob}.tif"),
                        "sha256": receipt[blob]})
    return {"rasters": rasters, "complete_accessible_scan": len(rasters) == len(receipt), "errors": []}


def uniqueness(arr: np.ndarray, phase: str) -> dict:
    grid = load_grid(DATA)
    def progress(i, n, row):
        if i % 100 == 0 or i == n:
            print(f"{phase}: {i}/{n}", flush=True)
    report = compare_array_to_registry(arr, registry_index(), grid.footprint,
                                       include_reverse=phase != "surface", progress=progress,
                                       expected_grid=(grid.shape, grid.crs, grid.transform))
    report["phase"] = phase
    return report


def run_surface() -> None:
    f = features(); can = canaries(f); uq = uniqueness(f["surface"], "surface")
    max_abs_rho = max((abs(r["spearman_full_footprint"]) for r in uq["rows"]
                       if r.get("spearman_full_footprint") is not None), default=0.0)
    out = {"experiment": "H79-E1", "feature_definition": json.loads((ROOT / "registry/preregistration_h79_radhalo.json").read_text())["fixed_transform"],
           "leakage_canary": can, "surface_max_abs_spearman": max_abs_rho,
           "surface_duplicate": max_abs_rho > .90, "registry": uq}
    _json(ROOT / "evidence/h79_radhalo_e1_surface.json", out)
    print(json.dumps({"canary_max": max(v["symmetric_auc"] for v in can.values()),
                      "surface_max_abs_spearman": max_abs_rho,
                      "duplicate": out["surface_duplicate"]}, indent=2))
    if out["surface_duplicate"]:
        raise SystemExit("STOP: H79 surface duplicates a registry raster")


def emit(surface: np.ndarray, allowed: np.ndarray, limit=N_DOTS) -> np.ndarray:
    idx = np.flatnonzero(allowed & (surface > 0))
    order = idx[np.argsort(-surface.ravel()[idx], kind="stable")]
    sel = greedy_select(order, 3.0, limit=limit)
    p = np.zeros(surface.shape, np.float32); p.ravel()[sel] = 1
    if len(sel) != limit:
        raise RuntimeError(f"only {len(sel)} dots emitted, need {limit}")
    return p


def run_holdout() -> None:
    e1 = json.loads((ROOT / "evidence/h79_radhalo_e1_surface.json").read_text())
    if e1["surface_duplicate"] or any(v["leakage_flag"] for v in e1["leakage_canary"].values()):
        raise SystemExit("E1 gate failed; no holdout")
    f = features(); grid = load_grid(DATA); folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=.20, buffer_px=3)
    rng = np.random.default_rng(7901); random_key = rng.random(grid.shape).astype(np.float32)
    terms = {k: [] for k in ("H79", "common_edge", "random")}; receipts = []
    for fold in folds:
        allowed = grid.footprint & ~fold["masked_known"]
        arms = {"H79": emit(f["surface"], allowed),
                "common_edge": emit(f["common_edge_rank"], allowed),
                "random": emit(random_key, allowed)}
        for k, p in arms.items():
            _, t = eh.evaluate(p, fold, grid.footprint); terms[k].append(t)
        receipts.append({**fold["receipt"], "dots_per_arm": N_DOTS})
    pooled = {k: np.sum(v, axis=0) for k, v in terms.items()}
    summary = eh.pooled_summary(pooled, draws=2000, candidate="H79")
    out = {"experiment": "H79-E2", "evidence_class": "HOLDOUT-DTI",
           "evaluator_version": eh.VERSION, "fold_receipts": receipts,
           "summary": summary, "implementation_sha256": eh.implementation_hashes(),
           "current_holdout_best": {"score": .152718, "source": "evidence/h77/run_card.json"}}
    _json(ROOT / "evidence/h79_radhalo_e2_holdout.json", out)
    print(json.dumps(summary, indent=2))


def run_artifact() -> None:
    # E3 always preserves a unique negative-result deliverable if E1 passed; it is not selector approval.
    e1 = json.loads((ROOT / "evidence/h79_radhalo_e1_surface.json").read_text())
    if e1["surface_duplicate"]:
        raise SystemExit("surface duplicate: protocol requires stop")
    f = features(); grid = load_grid(DATA)
    pred = emit(f["surface"], grid.footprint & ~grid.catalogue)
    out = ROOT / "submissions" / f"{NAME}.tif"
    for p in (out, out.with_suffix(".zip"), out.with_suffix(".json")):
        p.unlink(missing_ok=True)
    receipt = write_submission(out, pred, DATA / "sample_submission.tif", grid.footprint,
                               catalogue=grid.catalogue, name=NAME, note=NOTE,
                               metadata={"lane": "H79-RH", "selector_approved": False})
    for p in (out, out.with_suffix(".zip"), out.with_suffix(".json")):
        shutil.copy2(p, ROOT / "docs" / "downloads" / p.name)
    print(json.dumps(receipt, indent=2))


def run_final_uniqueness() -> None:
    import rasterio
    p = ROOT / "submissions" / f"{NAME}.tif"
    with rasterio.open(p) as src: arr = src.read(1)
    uq = uniqueness(arr, "final-dots")
    _json(ROOT / "evidence/h79_radhalo_e3_uniqueness.json", uq)
    print(json.dumps({k: uq[k] for k in ("registry_rasters_checked", "worst_spearman_full_footprint", "worst_dot_overlap", "duplicate_count", "unique", "stop_required")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("surface", "holdout", "artifact", "final-uniqueness")); a = ap.parse_args()
    {"surface": run_surface, "holdout": run_holdout, "artifact": run_artifact, "final-uniqueness": run_final_uniqueness}[a.mode]()

if __name__ == "__main__": main()
