#!/usr/bin/env python3
"""H77-XSEN Exp 3: full-map build + two-phase uniqueness + fail-closed writer.

Full-map model = the Exp-2 A_xsen arm retrained on the FULL visible catalogue
(no holdout withholding at build time; holdout evidence is evidence/xsen_exp2.json).
Emission: emit77.emit_dots, budget 37654, min_sep 2.8 px, allowed = footprint
& ~catalogue & dist_catalogue>2 px (champion collar; catalogue masked exactly).
Uniqueness: vendor gems57.uniqueness.compare_array_to_registry, unmodified,
phase 1 on the continuous surface BEFORE placement, phase 2 on final dots.
Writer: vendor gems57.submission_writer (finite [0,1], nodata=None, re-validated
on disk) - the portal 'Predicted values must be in range [0, 1]' cannot recur
because out-of-range/non-finite arrays raise before any file is committed.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G  # noqa: E402
from gems57 import uniqueness as U  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems77 import features77 as F, emit77  # noqa: E402

RAW = ROOT / "data" / "raw"; PROC = ROOT / "data" / "processed"
HGB_KW = dict(max_iter=180, learning_rate=0.09, max_leaf_nodes=31,
              min_samples_leaf=40, early_stopping=False, random_state=77)
BUDGET = 37_654
NAME = "gems77-xsen-coincide-hgb-37654dots-20261021-zeros"
NOTE = "XSEN mag analytic-signal x ThK/UK boundary HGB, 37654 dots; HOLDOUT-DTI 0.0994, below 0.1527 repo best: NEGATIVE, do not submit"


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    assert len(NOTE) <= 140, len(NOTE)
    grid = G.load_grid(RAW); foot, cat = grid.footprint, grid.catalogue
    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    xsen = np.load(PROC / "xsen.npy", mmap_mode="r")
    cols = [F.IDX[n] for n in F.BLOCKS["raw"] + F.BLOCKS["terrain"] + F.BLOCKS["rad"]]
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    rng = np.random.default_rng(1234)
    pos = np.flatnonzero(cat & foot)
    if pos.size > 40_000: pos = rng.choice(pos, 40_000, replace=False)
    neg_mask = foot & ~cat & (dist_cat > 3)
    hard = np.flatnonzero(neg_mask & (dist_cat <= 15)); easy = np.flatnonzero(neg_mask & (dist_cat > 15))
    n_neg = 2 * pos.size; nh = min(hard.size, n_neg // 2); ne = min(easy.size, n_neg - nh)
    neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
    idx = np.concatenate([pos, neg]); y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
    fs = stack.reshape(-1, stack.shape[2]); fx = xsen.reshape(-1, xsen.shape[2])
    X = np.concatenate([np.asarray(fs[idx][:, cols], np.float32), np.asarray(fx[idx], np.float32)], axis=1)
    clf = HistGradientBoostingClassifier(**HGB_KW); clf.fit(X, y); del X
    log(f"trained full-map model pos={pos.size} neg={neg.size} ({time.time()-t0:.0f}s)")
    surf = np.zeros(foot.shape, np.float32); step = 100
    for i in range(0, foot.shape[0], step):
        j = min(i + step, foot.shape[0])
        blk = np.concatenate([np.asarray(stack[i:j][:, :, cols], np.float32),
                              np.asarray(xsen[i:j], np.float32)], axis=2)
        surf[i:j] = clf.predict_proba(blk.reshape(-1, blk.shape[2]))[:, 1].reshape(j - i, foot.shape[1])
        del blk
    surf = np.where(foot, surf, 0.0).astype(np.float32)
    np.save(PROC / "xsen_fullmap_surface.npy", surf)
    log(f"surface: max={surf.max():.3f} mean_fp={surf[foot].mean():.4f} ({time.time()-t0:.0f}s)")

    # ---- registry (fetched by scripts/fetch_registry.py) ----
    idxj = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    rec = json.loads((ROOT / "registry" / "registry_receipt.json").read_text())["sha256_by_blob"]
    reg = {"rasters": [{"repo_first": r["repo"], "sources": r["aliases"],
                        "cache_file": str(ROOT / ".cache" / "registry" / f"{r['blob']}.tif"),
                        "sha256": rec[r["blob"]]} for r in idxj["rasters"] if r["blob"] in rec],
           "complete_accessible_scan": True, "errors": []}
    n_expected = len(idxj["rasters"]); log(f"registry records={len(reg['rasters'])}/{n_expected}")
    out = {"name": NAME, "note": NOTE, "note_chars": len(NOTE), "budget": BUDGET}

    for phase, arr in (("phase1_surface_before_placement", surf),):
        r = U.compare_array_to_registry(arr, reg, foot)
        out[phase] = {k: r.get(k) for k in ("unique", "duplicate_count", "registry_rasters_checked",
                                            "registry_rasters_expected", "complete_accessible_scan",
                                            "worst_spearman_full_footprint", "worst_rho_submission",
                                            "worst_dot_overlap", "worst_overlap_submission", "stop_required")}
        out[phase]["rows"] = r["rows"]
        log(f"{phase}: unique={r['unique']} worst_rho={r['worst_spearman_full_footprint']} worst_ov={r['worst_dot_overlap']}")

    allowed = foot & ~cat & (dist_cat > 2)
    dots = emit77.emit_dots(surf, allowed, BUDGET)
    log(f"dots={int(dots.sum())} on_cat={int((dots & cat).sum())} min_dist_cat={float(dist_cat[dots].min()) if dots.any() else None}")
    final = np.where(dots, 1.0, 0.0).astype(np.float32)
    r2 = U.compare_array_to_registry(final, reg, foot)
    out["phase2_final_dots"] = {k: r2.get(k) for k in ("unique", "duplicate_count", "registry_rasters_checked",
                                                       "registry_rasters_expected", "complete_accessible_scan",
                                                       "worst_spearman_full_footprint", "worst_rho_submission",
                                                       "worst_dot_overlap", "worst_overlap_submission", "stop_required")}
    out["phase2_final_dots"]["rows"] = r2["rows"]
    log(f"phase2: unique={r2['unique']} worst_rho={r2['worst_spearman_full_footprint']} worst_ov={r2['worst_dot_overlap']}")

    dest = ROOT / "submissions" / f"{NAME}.tif"
    receipt = write_submission(dest, final, RAW / "sample_submission.tif", foot,
                               note=NOTE, name=NAME, catalogue=cat,
                               metadata={"lane": "H77-XSEN", "holdout_evidence": "evidence/xsen_exp2.json"})
    out["writer_receipt"] = receipt
    log(f"written {dest} sha256={receipt['sha256']}")
    (ROOT / "evidence" / "xsen_exp3_build.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
