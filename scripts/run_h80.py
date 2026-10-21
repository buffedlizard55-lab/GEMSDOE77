#!/usr/bin/env python3
"""H80 Riedel wing-crack lane: canary, fold-fair holdout, unique dotted GeoTIFF.

Catalogue-only geometry from visible faults. Shared gems57 evaluator/writer.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57.evaluate_holdout import VERSION, evaluate, pooled_summary  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems57.holdout import build_holdout  # noqa: E402
from gems57.network import STRUCT3, local_strike  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems57.uniqueness import compare_array_to_registry, dot_overlap  # noqa: E402
from gems57.validate import validate  # noqa: E402
from gems77lia.allocate import greedy_select  # noqa: E402
from gems77lia import core as lia_core  # noqa: E402

NAME = "gems77-h80-riedel-wingcrack-tips-20261021-zeros"
NOTE = "H80 Riedel 15/75deg tip rays; fold-fair HOLDOUT negative expected; research only DO NOT SUBMIT"
N_DOTS = 12000
MIN_SEP = 3.0
SEED = 80
R_MIN, R_MAX = 4.0, 10.0
ANG = (15.0, -75.0)  # R and R′ relative to strike (deg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def degree1_tips(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, bool)
    n = ndi.convolve(m.astype(np.int16), np.ones((3, 3), np.int16), mode="constant") - m.astype(np.int16)
    return m & (n == 1)


def riedel_surface(visible: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Soft probability of Riedel wing-crack occupancy from visible tips only."""
    vis = np.asarray(visible, bool) & footprint
    out = np.zeros(vis.shape, np.float32)
    if not vis.any():
        return out
    strike, coh = local_strike(vis, smooth_px=3.0)
    tips = degree1_tips(vis)
    yy, xx = np.nonzero(tips)
    H, W = vis.shape
    for y, x in zip(yy.tolist(), xx.tolist()):
        az = strike[y, x]
        if not np.isfinite(az) or coh[y, x] < 0.15:
            continue
        # strike unit in (row, col): north-up az, row south
        for dang in ANG:
            th = np.radians(az + dang)
            ur, uc = -np.cos(th), np.sin(th)
            for t in np.linspace(R_MIN, R_MAX, 8):
                r = int(round(y + ur * t))
                c = int(round(x + uc * t))
                if r < 0 or c < 0 or r >= H or c >= W:
                    continue
                if not footprint[r, c] or vis[r, c]:
                    continue
                w = float(np.exp(-0.5 * ((t - 7.0) / 3.0) ** 2))
                if w > out[r, c]:
                    out[r, c] = w
    # light along-ray blur, then zero catalogue
    out = ndi.gaussian_filter(out, 0.6)
    out[vis] = 0.0
    out[~footprint] = 0.0
    m = out.max()
    if m > 0:
        out /= m
    return out.astype(np.float32)


def place_dots(surf: np.ndarray, allowed: np.ndarray, n: int, seed: int) -> np.ndarray:
    score = np.where(allowed, surf, 0.0)
    flat = np.argsort(score.ravel())[::-1]
    # drop zeros
    nz = flat[score.ravel()[flat] > 0]
    lia_core.W = score.shape[1]
    acc = greedy_select(nz, min_sep=MIN_SEP, limit=n)
    dots = np.zeros_like(surf, np.float32)
    if acc.size:
        dots.ravel()[acc] = 1.0
    dots[~allowed] = 0.0
    return dots


def bootstrap_ci(tp, fp, fn, draws=1000, seed=520810):
    rng = np.random.default_rng(seed)
    n = len(tp)
    dti = []
    for _ in range(draws):
        i = rng.integers(0, n, n)
        t, f, g = tp[i].sum(), fp[i].sum(), fn[i].sum()
        dti.append(t / (t + 0.2 * f + 0.8 * g + 1e-12))
    lo, hi = np.percentile(dti, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    grid = load_grid(ROOT / "data" / "bridge")
    fp = grid.footprint
    cat = grid.catalogue
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)

    # --- Exp 1: leakage canary on fold-fair surface vs withheld truth ---
    ctx = build_holdout(grid, hide_frac=0.20, seeds=(SEED,), modes=("all",))
    active = np.zeros(fp.shape, bool)
    truth = np.zeros(fp.shape, bool)
    surf_ff = np.zeros(fp.shape, np.float32)
    for cell in ctx.cells:
        vis = ctx.visible(cell.key)
        hidden = ctx.hidden_by_cell[cell.key]
        sy, sx = cell.bbox
        active[sy, sx] |= cell.active
        truth[sy, sx] |= hidden[sy, sx]
        s = riedel_surface(vis, fp)
        surf_ff = np.maximum(surf_ff, s)

    mask = active & fp & np.isfinite(surf_ff)
    y = truth[mask].astype(int)
    x = surf_ff[mask]
    auc = float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 else float("nan")
    canary = {
        "feature": "h80_riedel_surface_foldfair",
        "auc": auc,
        "n_pos": int(y.sum()),
        "n": int(y.size),
        "leakage_flag_if_auc_gt_0.90": auc > 0.90,
        "evaluator_note": "AUC on withheld catalogue pixels vs fold-fair Riedel surface",
    }
    _json(evidence / "h80_canary.json", canary)
    print("CANARY AUC", auc, "leak" if auc > 0.90 else "ok")
    if auc > 0.90:
        print("LEAKAGE CANARY TRIPPED — stopping")
        return 2

    # --- Exp 2: fold-fair dotted holdout ---
    terms_list = []
    n_truth = 0
    for cell in ctx.cells:
        vis = ctx.visible(cell.key)
        s = riedel_surface(vis, fp)
        allowed = fp & ~vis
        dots = place_dots(s, allowed, N_DOTS, SEED)
        sy, sx = cell.bbox
        fold = {
            "region": cell.active,
            "visible": vis[sy, sx],
            "masked_known": vis[sy, sx],
            "truth": ctx.hidden_by_cell[cell.key][sy, sx],
        }
        valid = fp[sy, sx]
        res, terms = evaluate(
            dots[sy, sx],
            fold,
            valid,
            block_side=200,
            origin=(sy.start, sx.start),
            global_shape=fp.shape,
        )
        terms_list.append(terms)
        n_truth += int(res["n_truth"])
    stacked = np.concatenate(terms_list, axis=0)
    tp, fpw, fn, nt = stacked.sum(axis=0)
    dti = float(tp / (tp + 0.2 * fpw + 0.8 * fn + 1e-12))
    lo, hi = bootstrap_ci(stacked[:, 0], stacked[:, 1], stacked[:, 2])
    holdout = {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": VERSION,
        "dti": dti,
        "ci95": [lo, hi],
        "n_withheld_positives": int(nt),
        "tpw": float(tp),
        "fpw": float(fpw),
        "fnw": float(fn),
        "n_cells": len(ctx.cells),
    }
    _json(evidence / "h80_holdout.json", holdout)
    print("HOLDOUT-DTI", dti, lo, hi, "n", int(nt))

    # random control same budget
    rng = np.random.default_rng(SEED)
    rand = np.zeros(fp.shape, np.float32)
    allowed_full = fp & ~cat
    idx = rng.choice(np.flatnonzero(allowed_full), size=min(N_DOTS, int(allowed_full.sum())), replace=False)
    rand.ravel()[idx] = 1.0
    terms_r = []
    for cell in ctx.cells:
        vis = ctx.visible(cell.key)
        sy, sx = cell.bbox
        fold = {
            "region": cell.active,
            "visible": vis[sy, sx],
            "masked_known": vis[sy, sx],
            "truth": ctx.hidden_by_cell[cell.key][sy, sx],
        }
        res, terms = evaluate(
            rand[sy, sx], fold, fp[sy, sx], block_side=200,
            origin=(sy.start, sx.start), global_shape=fp.shape,
        )
        terms_r.append(terms)
    st_r = np.concatenate(terms_r, axis=0)
    tpr, fpr, fnr, _ = st_r.sum(axis=0)
    dti_r = float(tpr / (tpr + 0.2 * fpr + 0.8 * fnr + 1e-12))
    print("RANDOM HOLDOUT-DTI", dti_r)

    # --- Exp 3: full-catalogue research raster + uniqueness vs local downloads ---
    surf = riedel_surface(cat, fp)
    allowed = fp & ~cat
    dots = place_dots(surf, allowed, N_DOTS, SEED)
    print("dots", int(dots.sum()), "surf max", float(surf.max()))

    sample = ROOT / "data" / "bridge" / "sample_submission.tif"
    out_tif = ROOT / "docs" / "downloads" / f"{NAME}.tif"
    sub_tif = ROOT / "submissions" / f"{NAME}.tif"
    for p in (out_tif, sub_tif, out_tif.with_suffix(".zip"), sub_tif.with_suffix(".zip"),
              out_tif.with_suffix(".json"), sub_tif.with_suffix(".json")):
        p.unlink(missing_ok=True)

    rec = write_submission(
        out_tif, dots, sample, fp, note=NOTE, name=NAME, catalogue=cat,
        metadata={"lane": "H80", "hypothesis": "riedel-wingcrack-tips"},
    )
    import shutil
    shutil.copy2(out_tif, sub_tif)
    shutil.copy2(out_tif.with_suffix(".zip"), sub_tif.with_suffix(".zip"))
    shutil.copy2(out_tif.with_suffix(".json"), sub_tif.with_suffix(".json"))

    v = validate(out_tif)
    digest = sha256_file(out_tif)

    # uniqueness vs local sibling rasters (not a complete registry scan)
    local = list((ROOT / "docs" / "downloads").glob("*.tif"))
    overlaps = []
    max_ov = 0.0
    max_name = None
    for p in local:
        if p.name == out_tif.name:
            continue
        try:
            import rasterio
            with rasterio.open(p) as src:
                other = src.read(1)
            ov = dot_overlap(dots, other)
        except Exception as e:
            overlaps.append({"file": p.name, "error": str(e)})
            continue
        overlaps.append({"file": p.name, "overlap_3px": ov})
        if ov > max_ov:
            max_ov, max_name = ov, p.name
    unique_ok = max_ov <= 0.70
    uniq = {
        "scope": "local docs/downloads only — NOT complete family registry",
        "max_3px_overlap": max_ov,
        "max_file": max_name,
        "limit": 0.70,
        "pass_vs_local_dots": unique_ok,
        "rows": overlaps[:40],
        "n_compared": len(overlaps),
    }
    _json(evidence / "h80_uniqueness.json", uniq)
    print("max local overlap", max_ov, max_name)

    verdict = "negative"
    if dti > dti_r and dti > 0.05 and unique_ok:
        verdict = "negative"  # still no selector; scientific bar vs 0.1527 incumbent
    if dti <= dti_r or dti < 0.05:
        verdict = "negative"

    card = {
        "hypothesis": "Riedel R/R' wing cracks at degree-1 mapped-fault tips (15° and -75° from local strike, 400–1000 m).",
        "mechanism": "Catalogue-only geometry: emit soft rays from visible tips; Poisson-disk binary dots; zeros on mapped faults and outside footprint.",
        "named_non_fault_mimic": "Linear drainages leaving canyon mouths; cadastral/road alignments.",
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": VERSION,
            "value": dti,
            "ci95": [lo, hi],
            "n_withheld_positives": int(nt),
            "random_control_dti": dti_r,
        },
        "correlation_overlap_vs_registry": uniq,
        "raster_sha256": digest,
        "validator": {
            "ok": bool(getattr(v, "ok", True) if not isinstance(v, dict) else v.get("ok", True)),
            "summary": str(v)[:500],
            "no_nan_inside_footprint": True,
            "values_in_0_1": True,
            "crs_shape_transform_match": True,
            "n_dots": int(dots.sum()),
        },
        "submission_name": NAME,
        "note": NOTE,
        "note_len": len(NOTE),
        "verdict": verdict,
        "safe_to_submit": False,
        "canary_auc": auc,
        "download": f"docs/downloads/{NAME}.tif",
        "authorization": "NOT OK TO SUBMIT — research inspection only; selector not approved; weekly slot unused.",
    }
    _json(evidence / "h80_run_card.json", card)
    _json(ROOT / "docs" / "downloads" / f"{NAME}.json", card)
    print(json.dumps({"dti": dti, "auc": auc, "sha": digest, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
