#!/usr/bin/env python3
"""H82: windowed spectral magnetic source-depth gradient (Spector & Grant 1970 style).

Modes mirror the repository's established per-lane pattern (see
``scripts/h79_radhalo_pipeline.py``): ``surface`` (feature + leakage canary,
must pass before any holdout run), ``holdout`` (pooled HOLDOUT-DTI vs a
matched random control, using the unchanged shared evaluator).

Reuses ``vendor/gems57`` (holdout, metric, evaluate_holdout) unchanged.
Reads only the cached, SHA-256-pinned RTP band (band 2 of
``data/raw/training_features.tif``) plus the grid/footprint/catalogue bridge
files. No catalogue-derived feature, no lidar, no prior submission raster is
read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from gems77lia.allocate import greedy_select  # noqa: E402

BRIDGE = ROOT / "data" / "bridge"
TRAIN_TIF = ROOT / "data" / "raw" / "training_features.tif"
WORK = ROOT / ".cache" / "work"
N_DOTS = 37_654
MIN_SEP = 2.8
SEED = 20
NAME = "gems77-h82-spectral-depth-gradient-37654dots-20261021-zeros"
NOTE = "H82 spectral depth-gradient (Spector-Grant); HOLDOUT-DTI 0.0258 < random 0.0466 and incumbent 0.1527; negative, do not submit"
BLOCK = 32
N_RADIAL_BINS = 16
FIT_BINS = (1, 8)  # inclusive band index range used for the slope fit
DEPTH_CLIP_M = (100.0, 4000.0)
PIXEL_M = 100.0
RTP_BAND_INDEX = 2  # 1-based rasterio band index; "rtp" per training_features.tif tags
EXPECTED_RTP_SHA256_NOTE = "band read fresh each run from the SHA-pinned training_features.tif (registry/data_manifest.json)"


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _rank01(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, np.float32)
    v = a[valid]
    r = rankdata(v, method="average").astype(np.float32)
    out[valid] = (r - 1.0) / max(len(r) - 1, 1)
    return out


def _radial_bin_index(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed (n, n) radial-bin index array and the physical k (rad/m) per bin."""
    fy = np.fft.fftfreq(n, d=1.0)
    fx = np.fft.fftfreq(n, d=1.0)
    kr_cyc = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # cycles/pixel
    max_k = kr_cyc.max()
    edges = np.linspace(0.0, max_k * 1.0001, N_RADIAL_BINS + 1)
    bin_idx = np.clip(np.digitize(kr_cyc, edges) - 1, 0, N_RADIAL_BINS - 1)
    bin_centers_cyc = 0.5 * (edges[:-1] + edges[1:])
    k_phys = 2.0 * np.pi * bin_centers_cyc / PIXEL_M  # rad/m
    return bin_idx, k_phys


def _spector_grant_depth_blocks(rtp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Depth-to-source-ensemble per BLOCK x BLOCK tile (NaN where under-covered)."""
    h, w = rtp.shape
    pad_h = (-h) % BLOCK
    pad_w = (-w) % BLOCK
    rtp_p = np.pad(rtp, ((0, pad_h), (0, pad_w)), mode="reflect")
    valid_p = np.pad(valid, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    ny, nx = rtp_p.shape[0] // BLOCK, rtp_p.shape[1] // BLOCK
    hann1 = np.hanning(BLOCK)
    taper2d = np.outer(hann1, hann1).astype(np.float32)
    taper_norm = float((taper2d ** 2).sum())
    bin_idx, k_phys = _radial_bin_index(BLOCK)
    lo, hi = FIT_BINS
    fit_bins = np.arange(lo, hi + 1)
    kx = k_phys[fit_bins]
    sx = kx.sum()
    sxx = (kx ** 2).sum()
    nfit = len(kx)
    depth_blocks = np.full((ny, nx), np.nan, np.float32)
    coverage_blocks = np.zeros((ny, nx), np.float32)
    bin_masks = [bin_idx == b for b in fit_bins]
    for by in range(ny):
        row = rtp_p[by * BLOCK:(by + 1) * BLOCK]
        vrow = valid_p[by * BLOCK:(by + 1) * BLOCK]
        for bx in range(nx):
            tile = row[:, bx * BLOCK:(bx + 1) * BLOCK]
            vtile = vrow[:, bx * BLOCK:(bx + 1) * BLOCK]
            cov = float(vtile.mean())
            coverage_blocks[by, bx] = cov
            if cov < 0.5:
                continue
            fill = float(tile[vtile].mean()) if vtile.any() else 0.0
            clean = np.where(vtile, tile, fill).astype(np.float32)
            clean = clean - clean.mean()
            spec = np.fft.fft2(clean * taper2d)
            power = (np.abs(spec) ** 2) / taper_norm
            y = np.empty(nfit, np.float64)
            for j, m in enumerate(bin_masks):
                y[j] = np.log(np.sqrt(max(power[m].mean(), 1e-12)))
            sy = y.sum()
            sxy = float((kx * y).sum())
            denom = nfit * sxx - sx * sx
            if abs(denom) < 1e-12:
                continue
            slope = (nfit * sxy - sx * sy) / denom
            depth = -slope
            if np.isfinite(depth) and DEPTH_CLIP_M[0] <= depth <= DEPTH_CLIP_M[1]:
                depth_blocks[by, bx] = depth
    return depth_blocks, coverage_blocks


def features() -> dict[str, np.ndarray]:
    WORK.mkdir(parents=True, exist_ok=True)
    cache = WORK / "h82_features.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import rasterio
    with rasterio.open(TRAIN_TIF) as src:
        if src.descriptions[RTP_BAND_INDEX - 1] is None or "rtp" not in src.descriptions[RTP_BAND_INDEX - 1].lower():
            raise AssertionError(f"band {RTP_BAND_INDEX} is not tagged rtp: {src.descriptions[RTP_BAND_INDEX - 1]!r}")
        rtp = src.read(RTP_BAND_INDEX).astype(np.float32)
        nodata = src.nodatavals[RTP_BAND_INDEX - 1]
    grid = load_grid(BRIDGE)
    valid = grid.footprint & np.isfinite(rtp)
    if nodata is not None and np.isfinite(nodata):
        valid &= rtp != np.float32(nodata)

    depth_blocks, coverage_blocks = _spector_grant_depth_blocks(rtp, valid)
    # Fill empty blocks via nearest-neighbour on the block grid, then resample.
    if np.isnan(depth_blocks).all():
        raise RuntimeError("no block passed the 50% coverage gate; cannot build a depth surface")
    nan_mask = np.isnan(depth_blocks)
    if nan_mask.any():
        idx = ndimage.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        depth_blocks_filled = depth_blocks[tuple(idx)]
    else:
        depth_blocks_filled = depth_blocks
    zoom_y = rtp.shape[0] / depth_blocks_filled.shape[0]
    zoom_x = rtp.shape[1] / depth_blocks_filled.shape[1]
    # Crop the padded block grid back exactly onto (zoom then crop to original h,w).
    depth_full_padded = ndimage.zoom(depth_blocks_filled, (zoom_y, zoom_x), order=1)
    depth_full = depth_full_padded[: rtp.shape[0], : rtp.shape[1]].astype(np.float32)
    depth_full = ndimage.gaussian_filter(depth_full, sigma=4.0)

    gy, gx = np.gradient(depth_full)
    grad_mag = np.hypot(gy, gx).astype(np.float32)

    valid_eroded = ndimage.binary_erosion(valid, iterations=BLOCK)
    depth_raw_rank = _rank01(depth_full, valid_eroded)
    depth_gradient_rank = _rank01(grad_mag, valid_eroded)

    out = {
        "depth_full": depth_full,
        "grad_mag": grad_mag,
        "depth_raw_rank": depth_raw_rank,
        "depth_gradient_rank": depth_gradient_rank,
        "valid": valid_eroded.astype(np.uint8),
        "block_coverage_mean": np.array([float(coverage_blocks.mean())], np.float32),
        "block_finite_frac": np.array([float((~nan_mask).mean())], np.float32),
    }
    np.savez_compressed(cache, **out)
    return out


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.concatenate([pos, neg])
    ranks = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def canaries(f: dict[str, np.ndarray]) -> dict:
    grid = load_grid(BRIDGE)
    folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    rng = np.random.default_rng(82)
    valid = f["valid"].astype(bool)
    names = ("depth_raw_rank", "depth_gradient_rank")
    vals = {k: {"p": [], "n": []} for k in names}
    for fold in folds:
        region = fold["region"] & grid.footprint & ~fold["masked_known"] & valid
        truth = fold["truth"] & region
        neg = region & ~truth
        ni = np.flatnonzero(neg)
        ni = rng.choice(ni, min(300_000, len(ni)), replace=False)
        pi = np.flatnonzero(truth)
        for k in names:
            vals[k]["p"].append(f[k].ravel()[pi])
            vals[k]["n"].append(f[k].ravel()[ni])
    out = {}
    for k, v in vals.items():
        a = _auc(np.concatenate(v["p"]), np.concatenate(v["n"]))
        symmetric = max(a, 1 - a)
        out[k] = {"auc": a, "symmetric_auc": symmetric, "leakage_flag": symmetric > 0.90}
    return out


def run_surface() -> None:
    f = features()
    can = canaries(f)
    out = {
        "experiment": "H82-E1",
        "evidence_class": "leakage-canary + feature diagnostic (not HOLDOUT-DTI)",
        "preregistration": "registry/preregistration_h82.json",
        "block_coverage_mean": float(f["block_coverage_mean"][0]),
        "block_finite_frac_before_nn_fill": float(f["block_finite_frac"][0]),
        "depth_full_stats_m": {
            "min": float(np.nanmin(f["depth_full"])),
            "max": float(np.nanmax(f["depth_full"])),
            "mean": float(np.nanmean(f["depth_full"])),
            "median": float(np.nanmedian(f["depth_full"])),
        },
        "leakage_canary": can,
        "any_leakage_flag": any(v["leakage_flag"] for v in can.values()),
    }
    _json(ROOT / "evidence/h82_e1_surface.json", out)
    print(json.dumps(out, indent=2))
    if out["any_leakage_flag"]:
        raise SystemExit("STOP: H82 canary exceeds 0.90 symmetric AUC (leakage)")


def emit(surface: np.ndarray, allowed: np.ndarray, limit=N_DOTS) -> np.ndarray:
    idx = np.flatnonzero(allowed & (surface > 0))
    order = idx[np.argsort(-surface.ravel()[idx], kind="stable")]
    sel = greedy_select(order, MIN_SEP, limit=limit)
    p = np.zeros(surface.shape, np.float32)
    p.ravel()[sel] = 1
    actual = len(sel)
    return p, actual


def run_holdout() -> None:
    e1 = json.loads((ROOT / "evidence/h82_e1_surface.json").read_text())
    if e1["any_leakage_flag"]:
        raise SystemExit("E1 leakage gate failed; no holdout")
    f = features()
    grid = load_grid(BRIDGE)
    folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    rng = np.random.default_rng(8201)
    random_key = rng.random(grid.shape).astype(np.float32)

    arm_terms: dict[str, list] = {"H82_gradient": [], "H82_raw_depth": [], "random": []}
    fold_receipts = []
    budget_notes = []
    for fold in folds:
        allowed = grid.footprint & ~fold["masked_known"]
        p_grad, n_grad = emit(f["depth_gradient_rank"], allowed)
        p_raw, n_raw = emit(f["depth_raw_rank"], allowed)
        p_rand, n_rand = emit(random_key, allowed)
        budget_notes.append({"fold": fold["name"], "n_grad": n_grad, "n_raw": n_raw, "n_rand": n_rand})
        for key, p in (("H82_gradient", p_grad), ("H82_raw_depth", p_raw), ("random", p_rand)):
            _, terms = eh.evaluate(p.astype(np.float64), fold, grid.footprint)
            arm_terms[key].append(terms)
        fold_receipts.append(fold["receipt"])

    pooled_terms = {k: np.sum(v, axis=0) for k, v in arm_terms.items()}
    summary = eh.pooled_summary(pooled_terms, draws=1000, candidate="H82_gradient")

    out = {
        "experiment": "H82-E2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": eh.VERSION,
        "preregistration": "registry/preregistration_h82.json",
        "emitted_dot_budgets_per_fold": budget_notes,
        "fold_receipts": fold_receipts,
        "summary": summary,
        "implementation_sha256": eh.implementation_hashes(),
        "retained_repo_incumbent": {
            "label": "HOLDOUT-DTI H77-Step, gems57-pooled-hide-v2, seed 20, 11321 withheld px",
            "value": 0.152718,
            "source": "evidence/run_card.json (H77-Step lane); cross-draw reference, NOT a paired comparison",
        },
        "beats_incumbent": bool(summary["scores"]["H82_gradient"]["dti"] > 0.152718
                                 and summary["scores"]["H82_gradient"]["ci95"][0] > 0.152718),
    }
    _json(ROOT / "evidence/h82_e2_holdout.json", out)
    print(json.dumps({
        "H82_gradient_dti": summary["scores"]["H82_gradient"]["dti"],
        "H82_gradient_ci95": summary["scores"]["H82_gradient"]["ci95"],
        "H82_raw_depth_dti": summary["scores"]["H82_raw_depth"]["dti"],
        "random_dti": summary["scores"]["random"]["dti"],
        "paired_delta_vs_random": summary["paired_differences"]["random"],
        "beats_incumbent_0.152718": out["beats_incumbent"],
    }, indent=2))


def run_artifact() -> None:
    """Build a research-only, NOT-OK-TO-SUBMIT inspection artifact.

    H82's own preregistered promotion gate (registry/preregistration_h82.json)
    was not met (evidence/h82_run_card.json: negative, below the matched
    random control). This artifact exists only so the negative result is an
    inspectable, byte-verifiable deliverable, matching every other negative
    lane in this repository (H77G, H79, H80, H81, XSEN). It is never treated
    as a submission candidate and no weekly slot is used.
    """
    import shutil
    e2 = json.loads((ROOT / "evidence/h82_e2_holdout.json").read_text())
    if e2["beats_incumbent"]:
        raise SystemExit("refusing to auto-build an artifact for a lane that claims to beat the incumbent; requires manual review")
    f = features()
    grid = load_grid(BRIDGE)
    pred, n = emit(f["depth_gradient_rank"], grid.footprint & ~grid.catalogue)
    out = ROOT / "submissions" / f"{NAME}.tif"
    for p in (out, out.with_suffix(".zip"), out.with_suffix(".json")):
        p.unlink(missing_ok=True)
    receipt = write_submission(out, pred.astype(np.float32), BRIDGE / "sample_submission.tif",
                               grid.footprint, catalogue=grid.catalogue, name=NAME, note=NOTE,
                               metadata={"lane": "H82", "selector_approved": False,
                                         "holdout_dti": 0.025822451132462867,
                                         "beats_incumbent": False, "verdict": "negative"})
    for p in (out, out.with_suffix(".zip"), out.with_suffix(".json")):
        shutil.copy2(p, ROOT / "docs" / "downloads" / p.name)
    print(json.dumps(receipt, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("surface", "holdout", "artifact"))
    a = ap.parse_args()
    {"surface": run_surface, "holdout": run_holdout, "artifact": run_artifact}[a.mode]()


if __name__ == "__main__":
    main()
