#!/usr/bin/env python3
"""Lane H79 E3 -- full-catalogue fit, dot emission, GeoTIFF packaging and uniqueness audit.

Steps
  1. Fit the H79_full HistGradientBoosting model on the whole footprint with the SAME
     sub-hide sampling as the holdout (positives = sub-hidden catalogue components whose
     context features come from the remaining visible traces), `--subdraws` draws.
  2. Predict the full footprint -> placement surface (saved to .cache/work/h79/surface_full.npy).
  3. Emit B dots (emit77.emit_dots, min_sep 2.8) on foot & ~catalogue & dist_cat > 3 px.
  4. Write TIF + ZIP + receipt with vendor/gems57/submission_writer.py (fail-closed; values
     {0,1}, zeros outside the footprint, no positive mass on the catalogue).
  5. Uniqueness (shared gems57.uniqueness, unmodified): surface BEFORE placement and final
     dots vs every cached registry raster (indexed + post-index); literal and like-for-like.
Writes evidence/h79_build.json. Nothing here is a score; the HOLDOUT-DTI lives in
evidence/h79_holdout_seed*.json and the verdict in evidence/h79_run_card.json.
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
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import submission_writer as SW  # noqa: E402
from gems57 import uniqueness as U  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems77 import features77 as F77  # noqa: E402
from gems79 import attrs as AT  # noqa: E402
from h79_holdout import HGB_LEGACY_KW, STATIC_CHANNELS, load_attr_rasters, static_rows  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
WORK = ROOT / ".cache" / "work" / "h79"
REG = ROOT / ".cache" / "registry"


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def registry_for_template() -> tuple[dict, int]:
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    rec = json.loads((ROOT / "registry" / "registry_receipt_git.json").read_text())["sha256_by_blob"]
    rasters, seen = [], set()
    for r in idx["rasters"]:
        if r["blob"] in rec:
            rasters.append({"repo_first": r["repo"], "sources": r["aliases"], "cache_file": str(REG / f"{r['blob']}.tif"),
                            "sha256": rec[r["blob"]]})
            seen.add(r["blob"])
    for blob, sha in rec.items():  # rasters published after the index was built
        if blob not in seen:
            rasters.append({"repo_first": "post-index", "sources": [f"post-index:{blob}"],
                            "cache_file": str(REG / f"{blob}.tif"), "sha256": sha})
    return {"rasters": rasters, "complete_accessible_scan": True, "errors": []}, len(idx["rasters"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=37654)
    ap.add_argument("--subdraws", type=int, default=6)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-uniqueness", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    grid = load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    A = load_attr_rasters(grid)
    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    flat = stack.reshape(-1, stack.shape[2])
    static_cols = [F77.IDX[c] for c in STATIC_CHANNELS]
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    surf_path = WORK / "surface_full.npy"
    if surf_path.exists():
        surf = np.load(surf_path)
        log("loaded cached full surface")
    else:
        rng = np.random.default_rng(79_000)
        Xs_parts, Xc_parts, y_parts = [], [], []
        allr = np.ones_like(foot)
        for s_i in range(args.subdraws):
            hid, vis_s = AT.sub_hide(cat, allr, rng, frac=0.20, buffer_px=3)
            d_s = ndimage.distance_transform_edt(~vis_s).astype(np.float32)
            pos = np.flatnonzero((hid & foot).ravel())
            base = foot & ~vis_s & ~hid & (d_s > 3.0)
            hard = np.flatnonzero((base & (d_s <= 16.0)).ravel())
            easy = np.flatnonzero((base & (d_s > 16.0)).ravel())
            n_neg = int(2.5 * pos.size)
            nh = min(hard.size, int(0.55 * n_neg))
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
            rows = np.concatenate([pos, neg])
            y_parts.append(np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)]))
            Xc_parts.append(AT.context_features(vis_s, A, foot, rows=rows))
            Xs_parts.append(static_rows(flat, rows, static_cols))
            log(f"sub-draw {s_i}: pos={pos.size} neg={neg.size}")
            del d_s, base, hid, vis_s
        y = np.concatenate(y_parts)
        X = np.hstack([np.vstack(Xs_parts), np.vstack(Xc_parts)])
        del Xs_parts, Xc_parts
        gc.collect()
        clf = HistGradientBoostingClassifier(**HGB_LEGACY_KW)
        clf.fit(X, y)
        log(f"fit on {y.size} rows x {X.shape[1]} features ({time.time()-t0:.0f}s)")
        del X
        gc.collect()
        rows = np.flatnonzero(foot.ravel())
        Xc = AT.context_features(cat, A, foot, rows=rows)
        pred = np.empty(rows.size, np.float32)
        for s in range(0, rows.size, 250_000):
            sl = slice(s, s + 250_000)
            pred[sl] = clf.predict_proba(np.hstack([static_rows(flat, rows[sl], static_cols), Xc[sl]]))[:, 1]
        del Xc
        surf = np.zeros((h, w), np.float32)
        surf.ravel()[rows] = pred
        np.save(surf_path, surf)
        log(f"full surface predicted ({time.time()-t0:.0f}s)")

    allowed = foot & ~cat & (dist_cat > 3.0)
    dots = emit77.emit_dots(surf, allowed, args.budget)
    final = np.zeros((h, w), np.float32)
    final[dots] = 1.0
    np.save(WORK / "final_dots_flat.npy", np.flatnonzero(dots.ravel()).astype(np.int32))
    log(f"emitted {int(dots.sum())} dots (budget {args.budget}); min dist to catalogue = {float(dist_cat[dots].min()):.2f} px")

    out_tif = Path(args.out) if args.out else ROOT / "docs" / "downloads" / f"{args.name}.tif"
    receipt = SW.write_submission(out_tif, final, RAW / "sample_submission.tif", foot, note=args.note, name=args.name,
                                  catalogue=cat, metadata={"lane": "H79-ATTR-HANGINGWALL", "budget": args.budget,
                                                           "emission": "emit77.emit_dots greedy min_sep 2.8, excl dist_cat > 3 px"})
    log(f"wrote {out_tif} sha256={receipt['sha256']} bytes={receipt['bytes']}")

    result = {"tif": str(out_tif.relative_to(ROOT)), "sha256": receipt["sha256"], "zip_sha256": receipt["zip_sha256"],
              "dots": int(dots.sum()), "budget": args.budget, "name": args.name, "note": args.note, "note_chars": len(args.note),
              "validator": receipt["validator"], "template_report": receipt["template_report"],
              "surface_sha256": hashlib.sha256(np.ascontiguousarray(surf).tobytes()).hexdigest()}
    if not args.skip_uniqueness:
        reg, n_expected = registry_for_template()
        surf01 = np.where(allowed, surf, 0.0).astype(np.float32)
        surf01 = (surf01 / max(float(surf01.max()), 1e-9)).astype(np.float32)
        for phase, arr in (("phase1_surface_before_placement", surf01), ("phase2_final_dots", final)):
            r = U.compare_array_to_registry(arr, reg, foot, include_reverse=phase.startswith("phase2"))
            rows = r.get("rows") or r.get("comparisons") or []
            summ = U.two_phase_summary(rows, int(foot.sum()), expected_rasters=n_expected) if rows else None
            ok_rows = [x for x in rows if "error" not in x]
            result[phase] = {
                "rasters_checked": len(ok_rows), "errors": [x for x in rows if "error" in x][:20],
                "summary": summ,
                "worst_overlap_rows": sorted(ok_rows, key=lambda x: -(x.get("my_dots_within_3px_of_theirs") or 0))[:12],
                "worst_rho_rows": sorted([x for x in ok_rows if x.get("spearman_full_footprint") is not None],
                                         key=lambda x: -x["spearman_full_footprint"])[:8],
                "max_rho": max((x["spearman_full_footprint"] for x in ok_rows if x.get("spearman_full_footprint") is not None), default=None),
                "max_forward_overlap_dot_fields": max((x["my_dots_within_3px_of_theirs"] for x in ok_rows
                                                       if x.get("representation") == "dot_field"), default=None),
                "max_forward_overlap_all": max((x["my_dots_within_3px_of_theirs"] for x in ok_rows), default=None),
            }
            log(phase, json.dumps({k: result[phase][k] for k in ("rasters_checked", "max_rho", "max_forward_overlap_dot_fields", "max_forward_overlap_all")}))
    result["elapsed_seconds"] = round(time.time() - t0, 1)
    (ROOT / "evidence" / "h79_build.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    log("wrote evidence/h79_build.json")


if __name__ == "__main__":
    main()
