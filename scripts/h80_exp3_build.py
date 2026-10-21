#!/usr/bin/env python3
"""H80-OTA Experiment 3 of 3: full-map build, uniqueness, fail-closed writer.

Phases (protocol rule 1: check surface BEFORE placement AND final dots):
  build   -> OTA vs full catalogue, train, predict, phase-1 surface check vs
             local lane TIFFs, emit dots, write TIF+ZIP via vendor writer
  compare -> phase-2 final-dots vs full fetched registry (812) + local lanes,
             written to evidence/h80_exp3_compare.json
Scientific verdict comes from Exp 2 vs the preregistered bar; this script never
grants upload authorization.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, uniqueness as U  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems77h80 import fields as OTA  # noqa: E402

RAW = ROOT / "data" / "raw"; PROC = ROOT / "data" / "processed"
HGB_KW = dict(max_iter=180, learning_rate=0.09, max_leaf_nodes=31,
              min_samples_leaf=40, early_stopping=False, random_state=77)
BUDGET = 37_654
NAME = "gems77-h80-orphan-thermal-align-37654dots-20261021-zeros"
NOTE = "H80-OTA orphan hot-well collinearity HGB 37654 dots; HOLDOUT-DTI 0.1039 seed30; negative do not submit"
SENT = -3.4028234663852886e38


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_band(i):
    with rasterio.open(RAW / "training_features.tif") as ds:
        a = ds.read(i).astype(np.float32)
    return np.where((a < SENT * 0.5) | ~np.isfinite(a), np.float32(np.nan), a)


def local_records():
    recs = []
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")) + sorted((ROOT / "submissions").glob("*.tif")):
        if NAME in p.name:
            continue
        recs.append(dict(repo="GEMSDOE77-local", submission=f"{p.parent.name}/{p.name}", file=str(p)))
    return recs


def summarize(rows, label):
    ok = [r for r in rows if "error" not in r]
    sparse = [r for r in ok if r.get("their_dots", 10**9) <= 150_000]
    def worst(rs, k):
        return max(rs, key=lambda r: (r.get(k) if r.get(k) is not None else -2), default=None)
    out = {"label": label, "checked": len(ok), "errors": len(rows) - len(ok)}
    for tag, rs in (("all", ok), ("sparse_le150k", sparse)):
        wr = worst([r for r in rs if r.get("spearman_full_footprint") is not None], "spearman_full_footprint")
        wo = worst(rs, "my_dots_within_3px_of_theirs")
        out[tag] = {
            "n": len(rs),
            "worst_spearman": (wr["spearman_full_footprint"] if wr else None),
            "worst_spearman_submission": (wr["submission"] if wr else None),
            "worst_forward_overlap_3px": (wo["my_dots_within_3px_of_theirs"] if wo else None),
            "worst_overlap_submission": (wo["submission"] if wo else None),
            "duplicate_firings_rho_or_overlap": int(sum(1 for r in rs if r.get("duplicate_by_rho") or r.get("duplicate_by_overlap"))),
            "identical_decoded": int(sum(1 for r in rs if r.get("identical_decoded_predictions"))),
        }
    return out


def build():
    t0 = time.time()
    assert len(NOTE) <= 140 and len(NAME) <= 140
    grid = G.load_grid(); foot, cat = grid.footprint, grid.catalogue
    band3 = read_band(3); band5 = read_band(5)
    ota_full, diag = OTA.build_ota_fields(RAW / "external" / "gdr_wellspring_in_footprint.csv",
                                          foot, cat, band3, band5)
    del band3, band5; gc.collect()
    np.save(PROC / "ota_full.npy", ota_full)
    log("OTA full diag:", json.dumps(diag))

    stack = np.load(PROC / "stack33.npy", mmap_mode="r")
    ota = np.load(PROC / "ota_full.npy", mmap_mode="r")
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    rng = np.random.default_rng(1234)
    pos = np.flatnonzero(cat & foot)
    if pos.size > 40_000: pos = rng.choice(pos, 40_000, replace=False)
    neg_mask = foot & ~cat & (dist_cat > 3)
    hard = np.flatnonzero(neg_mask & (dist_cat <= 15)); easy = np.flatnonzero(neg_mask & (dist_cat > 15))
    n_neg = 2 * pos.size; nh = min(hard.size, n_neg // 2); ne = min(easy.size, n_neg - nh)
    neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
    idx = np.concatenate([pos, neg]); y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
    fs = stack.reshape(-1, stack.shape[2]); fo = ota.reshape(-1, ota.shape[2])
    X = np.concatenate([np.asarray(fs[idx], np.float32), np.asarray(fo[idx], np.float32)], axis=1)
    clf = HistGradientBoostingClassifier(**HGB_KW); clf.fit(X, y); del X; gc.collect()
    log(f"trained full-map pos={pos.size} neg={neg.size} ({time.time()-t0:.0f}s)")

    surf = np.zeros(foot.shape, np.float32)
    for i in range(0, foot.shape[0], 100):
        j = min(i + 100, foot.shape[0])
        blk = np.concatenate([np.asarray(stack[i:j], np.float32), np.asarray(ota[i:j], np.float32)], axis=2)
        surf[i:j] = clf.predict_proba(blk.reshape(-1, blk.shape[2]))[:, 1].reshape(j - i, foot.shape[1])
        del blk
    surf = np.where(foot, surf, 0.0).astype(np.float32)
    assert np.isfinite(surf).all() and surf.max() > 0.40, f"surface degenerate max={surf.max()}"
    np.save(PROC / "h80_fullmap_surface.npy", surf)
    log(f"surface max={surf.max():.3f} mean_fp={surf[foot].mean():.4f} ({time.time()-t0:.0f}s)")

    # phase 1: surface vs local lane TIFFs, BEFORE placement
    r1 = U.compare_array_to_registry(surf, local_records(), foot)
    s1 = summarize(r1["rows"], "phase1_surface_vs_local_lanes")
    log("phase1:", json.dumps(s1["all"]), "sparse:", json.dumps(s1["sparse_le150k"]))

    allowed = foot & ~cat & (dist_cat > 2)
    dots = emit77.emit_dots(surf, allowed, BUDGET)
    np.save(PROC / "h80_final_dots.npy", dots)
    log(f"dots={int(dots.sum())} on_cat={int((dots & cat).sum())} "
        f"min_dist_cat={float(dist_cat[dots].min()) if dots.any() else None}")
    final = np.where(dots, 1.0, 0.0).astype(np.float32)

    dest = ROOT / "submissions" / f"{NAME}.tif"
    receipt = write_submission(dest, final, ROOT / "data" / "bridge" / "sample_submission.tif", foot,
                               note=NOTE, name=NAME, catalogue=cat,
                               metadata={"lane": "H80-OTA", "holdout_evidence": "evidence/h80_exp2.json",
                                         "verdict": "negative", "ok_to_submit": False})
    log(f"written {dest} sha256={receipt['sha256']}")
    out = {"name": NAME, "note": NOTE, "note_chars": len(NOTE), "budget": BUDGET,
           "ota_full_diagnostics": diag,
           "train": {"pos": int(pos.size), "neg": int(neg.size)},
           "surface": {"max": float(surf.max()), "mean_footprint": float(surf[foot].mean())},
           "phase1_surface_before_placement": s1, "phase1_rows": r1["rows"],
           "dots": int(dots.sum()), "dots_on_catalogue": int((dots & cat).sum()),
           "min_catalogue_distance_px": float(dist_cat[dots].min()) if dots.any() else None,
           "writer_receipt": receipt, "seconds": round(time.time() - t0, 1)}
    (ROOT / "evidence" / "h80_exp3_build.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"done build {time.time()-t0:.0f}s")


def compare():
    t0 = time.time()
    grid = G.load_grid(); foot = grid.footprint
    tif = ROOT / "submissions" / f"{NAME}.tif"
    with rasterio.open(tif) as ds:
        final = ds.read(1).astype(np.float32)
    file_sha = hashlib.sha256(tif.read_bytes()).hexdigest()

    idxj = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    rec = json.loads((ROOT / "registry" / "registry_receipt.json").read_text())["sha256_by_blob"]
    reg = {"rasters": [{"repo_first": r["repo"], "sources": r["aliases"],
                        "cache_file": str(ROOT / ".cache" / "registry" / f"{r['blob']}.tif"),
                        "sha256": rec[r["blob"]]} for r in idxj["rasters"] if r["blob"] in rec],
           "complete_accessible_scan": True, "errors": []}
    log(f"registry records={len(reg['rasters'])}/{len(idxj['rasters'])}")
    r2 = U.compare_array_to_registry(final, reg, foot, file_sha256=file_sha,
                                     progress=lambda i, n, row: log(f"reg {i}/{n}") if i % 200 == 0 else None)
    s2 = summarize(r2["rows"], "phase2_final_dots_vs_fetched_registry812")
    log("phase2 registry:", json.dumps(s2))
    champ = [r for r in r2["rows"] if "e5eb6e7e" in str(r.get("submission", ""))]
    log(f"champion rows={len(champ)}")

    r3 = U.compare_array_to_registry(final, local_records(), foot, file_sha256=file_sha)
    s3 = summarize(r3["rows"], "phase2_final_dots_vs_local_lanes")
    log("phase2 local:", json.dumps(s3))

    out = {"name": NAME, "file_sha256": file_sha,
           "registry_index_built_utc": idxj.get("built_utc"),
           "phase2_registry": {k: r2.get(k) for k in ("unique", "duplicate_count", "registry_rasters_checked",
                                                      "registry_rasters_expected", "complete_accessible_scan",
                                                      "missing_or_invalid", "stop_required", "verdict")},
           "phase2_registry_summary": s2, "champion_h33_rows": champ,
           "phase2_local_summary": s3, "phase2_local_rows": r3["rows"],
           "seconds": round(time.time() - t0, 1)}
    (ROOT / "evidence" / "h80_exp3_compare.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"done compare {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["build", "compare"])
    a = ap.parse_args()
    build() if a.phase == "build" else compare()
