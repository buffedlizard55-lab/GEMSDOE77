#!/usr/bin/env python3
"""GEMSDOE77 Exp 2: final full-catalogue model -> surface -> dots -> submission.

Uses the Exp-1 selected operating point (evidence/holdout_exp1.json). Writes via
the shared fail-closed gems57.submission_writer. Computes uniqueness vs the
locally hash-verified registry subset on BOTH the continuous surface and the
final dots. Verdict logic is in scripts/make_run_card.py; this script never
spends a submission slot.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G  # noqa: E402
from gems57 import uniqueness as UQ, submission_writer as SW  # noqa: E402
from gems77 import features77 as F, emit77  # noqa: E402
from run_holdout import arm_channels, HGB_KW  # noqa: E402  (scripts/ on sys.path when run directly)

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def registry_index():
    recs = []
    for p in sorted((ROOT / "data" / "registry_rasters").glob("*.tif")):
        recs.append({"repo": "local-sibling-clone", "submission": p.name,
                     "file": str(p.resolve()), "sha256": sha256_file(p)})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A4_primary")
    args = ap.parse_args()
    t0 = time.time()
    exp1 = json.loads((ROOT / "evidence" / "holdout_exp1.json").read_text())
    sel = exp1["selected_operating_point"]
    budget, excl = int(sel["budget"]), int(sel["exclusion_px"])
    log(f"operating point from Exp1: budget={budget} exclusion={excl}px arm={args.arm}")

    grid = G.load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    flat = stack.reshape(-1, stack.shape[2])
    cols = [F.IDX[n] for n in arm_channels(args.arm)]

    surf_path = PROC / f"final_surface_{args.arm}.npy"
    if surf_path.exists():
        surf = np.load(surf_path); log("loaded cached final surface")
    else:
        rng = np.random.default_rng(20261021)
        pos = np.flatnonzero(cat)
        neg_mask = foot & ~cat & (dist_cat > 3)
        hard = np.flatnonzero(neg_mask & (dist_cat <= 15))
        easy = np.flatnonzero(neg_mask & (dist_cat > 15))
        n_neg = 3 * pos.size
        neg = np.concatenate([rng.choice(hard, min(hard.size, n_neg // 2), replace=False),
                              rng.choice(easy, min(easy.size, n_neg - min(hard.size, n_neg // 2)), replace=False)])
        idx = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        X = np.asarray(flat[idx][:, cols], dtype=np.float32)
        log(f"training final HGB on {X.shape[0]} rows x {X.shape[1]} ch (pos={pos.size})")
        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(X, y); del X; gc.collect()
        surf = np.zeros(foot.shape, np.float32)
        ys, xs = np.nonzero(foot)
        r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        for i in range(r0, r1, 160):
            blk = np.asarray(stack[i:min(i + 160, r1), c0:c1][:, :, cols], dtype=np.float32)
            pr = clf.predict_proba(blk.reshape(-1, blk.shape[2]))[:, 1].reshape(blk.shape[:2])
            m = foot[i:min(i + 160, r1), c0:c1]
            surf[i:min(i + 160, r1), c0:c1] = np.where(m, pr, 0.0)
            del blk, pr
        del clf; gc.collect()
        np.save(surf_path, surf)
        log(f"final surface done ({time.time()-t0:.0f}s)")

    surf = np.clip(np.where(foot, surf, 0.0), 0, 1).astype(np.float32)
    reg = registry_index()
    log(f"uniqueness (surface) vs {len(reg)} registry rasters ...")
    surf_uq = UQ.compare_array_to_registry(surf, reg, foot)
    surf_two = UQ.two_phase_summary(surf_uq.get("rows", []), int(foot.sum())) if "rows" in surf_uq else {}

    allowed = foot & ~cat & (dist_cat > excl)
    dots = emit77.emit_dots(surf, allowed, budget)
    log(f"emitted {int(dots.sum())} dots (budget {budget})")
    dot_field = dots.astype(np.float32)
    log("uniqueness (final dots) ...")
    dot_uq = UQ.compare_array_to_registry(dot_field, reg, foot)
    dot_two = UQ.two_phase_summary(dot_uq.get("rows", []), int(foot.sum())) if "rows" in dot_uq else {}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest8 = hashlib.sha256(np.packbits(dots)).hexdigest()[:8]
    stem = f"gems77-contactphys-{int(dots.sum())}dots-{ts}-{digest8}-zeros"
    out_path = ROOT / "docs" / "downloads" / f"{stem}.tif"
    name = stem[:140]
    note = (f"GEMSDOE77 contact-physics: derived tilt/AS/phase+front-base HGB, "
            f"{int(dots.sum())} off-cat dots >{excl}px, min-sep 2.8px")[:140]
    assert len(note) <= 140 and len(name) <= 140
    receipt = SW.write_submission(out_path, dot_field, RAW / "sample_submission.tif", foot,
                                  note=note, name=name, catalogue=cat,
                                  metadata={"arm": args.arm, "budget": budget, "exclusion_px": excl,
                                            "exp1_evidence": "evidence/holdout_exp1.json"})
    bundle = {
        "submission_file": out_path.name, "submission_name": name, "note": note,
        "note_chars": len(note), "writer_receipt": receipt,
        "operating_point": sel, "arm": args.arm,
        "channels_used": arm_channels(args.arm),
        "surface_uniqueness": surf_uq, "surface_two_phase": surf_two,
        "dots_uniqueness": dot_uq, "dots_two_phase": dot_two,
        "registry_rasters_checked": len(reg),
        "registry_scope_caveat": "Locally hash-verified subset from GEMSDOE32+GEMSDOE70 clones; full-inventory certification is a selector step.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (ROOT / "evidence" / "final_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    np.save(PROC / "final_dots.npy", dots)
    log(f"WROTE {out_path} sha256={receipt['sha256'][:16]}... ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
