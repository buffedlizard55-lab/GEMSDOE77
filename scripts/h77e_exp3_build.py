#!/usr/bin/env python3
"""H77E Experiment 3 of 3: full-map cross-fitted build + packaging.

Trains the Exp-2 candidate arm (A_comb_ctx: raw 19 + H77E 11 + 2 LOCO
catalogue-context channels) as a 4-quadrant out-of-fold ensemble on the FULL
catalogue (0 in-sample pixels), emits 37,654 binary dots >3 px from the
catalogue via the shared 300 m submodular allocator, and packages the result
with the unmodified vendor/gems57 submission writer (fail-closed, all-finite
[0,1], nodata=None). Uniqueness is a separate step
(scripts/h77e_uniqueness.py) run on the surface (pre-placement ranks cached
here as .cache/h77e/full_surface.npy) and on the final dots.

Holdout verdict from Exp 2 governs the label: candidate 0.1305 (seed 20) /
0.1342 (seed 30) HOLDOUT-DTI did NOT beat the repo best 0.15272, so this file
is a download-only research candidate, NOT OK TO SUBMIT, selector not approved.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

import h77e_common as C  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import submission_writer as SW  # noqa: E402
from gems77d.submodular import allocate_submodular_300m  # noqa: E402
from h77e_exp2_holdout import tips_of  # noqa: E402

HGB_KW = dict(max_iter=180, learning_rate=0.075, max_leaf_nodes=55,
              min_samples_leaf=50, l2_regularization=2.5,
              early_stopping=False, random_state=7701)
BUDGET = 37654


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    t0 = time.time()
    grid = C.load_grid()
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    std_fp, _ = C.build_std_fp(foot)
    new_fp = np.load(C.CACHE / "new_fp.npy", mmap_mode="r")
    X = np.hstack([np.asarray(std_fp, dtype=np.float32), np.asarray(new_fp, dtype=np.float32)])
    log(f"static matrix {X.shape}")

    comp, _ = ndimage.label(cat, np.ones((3, 3), bool))
    tips = tips_of(cat)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    dist_tip = ndimage.distance_transform_edt(~tips).astype(np.float32)

    surf_cache = C.CACHE / "full_surface.npy"
    if surf_cache.exists():
        surf = np.load(surf_cache)
        log("loaded cached full-map surface")
    else:
        surf = np.zeros((h, w), dtype=np.float32)
        ymid, xmid = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        quads = [("NW", (yy < ymid) & (xx < xmid)), ("NE", (yy < ymid) & (xx >= xmid)),
                 ("SW", (yy >= ymid) & (xx < xmid)), ("SE", (yy >= ymid) & (xx >= xmid))]
        rng = np.random.default_rng(770)
        for name, qmask in quads:
            tr_mask = foot & ~qmask
            cat_tr = cat & tr_mask
            d_tr = ndimage.distance_transform_edt(~cat_tr).astype(np.float32)
            pos = fp_lut[np.flatnonzero(cat_tr)]
            if pos.size > 48_000:
                pos = rng.choice(pos, 48_000, replace=False)
            base = foot & tr_mask & ~cat_tr & (d_tr > 3.0)
            hard = fp_lut[np.flatnonzero(base & (d_tr <= 16.0))]
            easy = fp_lut[np.flatnonzero(base & (d_tr > 16.0))]
            n_neg = int(2.5 * pos.size)
            nh = min(hard.size, int(0.55 * n_neg))
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])

            def loco(rows_fp, pts_mask):
                pts_yx = np.argwhere(pts_mask)
                tree = cKDTree(pts_yx.astype(np.float32))
                rows_yx = np.stack([fp_flat[rows_fp] // w, fp_flat[rows_fp] % w], 1).astype(np.float32)
                own = comp[rows_yx[:, 0].astype(int), rows_yx[:, 1].astype(int)]
                d, idx = tree.query(rows_yx, k=8)
                pt_comp = comp[pts_yx[:, 0], pts_yx[:, 1]]
                valid = pt_comp[idx] != own[:, None]
                first = np.argmax(valid, axis=1)
                has = valid.any(axis=1)
                out = np.where(has, d[np.arange(len(rows_fp)), first], 100.0)
                return np.log1p(np.clip(out, 0, 100)).astype(np.float32)

            tr_ctx = np.stack([loco(tr_idx, cat_tr), loco(tr_idx, tips & tr_mask)], 1)
            te_idx = fp_lut[np.flatnonzero(qmask & foot)]
            te_yx = fp_flat[te_idx]
            te_ctx = np.stack([np.log1p(np.clip(dist_cat.ravel()[te_yx], 0, 100)),
                               np.log1p(np.clip(dist_tip.ravel()[te_yx], 0, 100))], 1).astype(np.float32)
            x_tr = np.hstack([X[tr_idx], tr_ctx])
            clf = HistGradientBoostingClassifier(**HGB_KW)
            clf.fit(x_tr, y_tr)
            del x_tr
            x_te = np.hstack([X[te_idx], te_ctx])
            preds = np.empty(te_idx.size, dtype=np.float32)
            for s in range(0, te_idx.size, 250_000):
                sl = slice(s, min(s + 250_000, te_idx.size))
                preds[sl] = clf.predict_proba(x_te[sl])[:, 1]
            surf.ravel()[fp_flat[te_idx]] = preds
            del clf, x_te, preds
            gc.collect()
            log(f"quadrant {name} done ({time.time()-t0:.0f}s)")
        np.save(surf_cache, surf)

    # ---- emission ----
    allowed = foot & ~cat & (dist_cat > 3.0)
    dots = allocate_submodular_300m(surf, allowed, target_n=BUDGET, support_quantile=0.968,
                                    gamma=3.5, min_sep_px=2.20)
    log(f"dots: {int(dots.sum())}")
    dots_sha = hashlib.sha256(np.ascontiguousarray(dots).tobytes()).hexdigest()
    name = f"gems77-h77e-ucpeuler-ctx-{int(dots.sum())}dots-20261021-{dots_sha[:8]}-zeros"
    note = ("H77E UCP-Euler+LOCO-ctx submod 37654; HOLDOUT-DTI 0.1305/0.1342 < best 0.1527; "
            "NEGATIVE not selector-approved")
    assert len(note) <= 140, len(note)
    pred = np.where(foot, dots.astype(np.float32), 0.0).astype(np.float32)

    out_tif = ROOT / "docs" / "downloads" / f"{name}.tif"
    sample = ROOT / "data" / "bridge" / "sample_submission.tif"
    receipt = SW.write_submission(out_tif, pred, sample, foot, note=note, name=name, catalogue=cat)
    log(f"writer receipt: {json.dumps(receipt)[:400]}")

    # -nan diagnostic twin (outside-footprint NaN; NOT the portal file)
    nan_pred = np.where(foot, dots.astype(np.float32), np.nan).astype(np.float32)
    nan_path = ROOT / "docs" / "downloads" / f"{name.replace('-zeros', '-nan')}.tif"
    G.write_submission(nan_path, nan_pred, mode="nan")

    res = {"experiment": "H77E-E3-BUILD", "submission_name": name, "note": note,
           "note_chars": len(note), "dots": int(dots.sum()),
           "dots_decoded_sha256": dots_sha, "tif": str(out_tif.relative_to(ROOT)),
           "tif_sha256": hashlib.sha256(out_tif.read_bytes()).hexdigest(),
           "tif_bytes": out_tif.stat().st_size,
           "zip_sha256": hashlib.sha256(out_tif.with_suffix('.zip').read_bytes()).hexdigest(),
           "nan_twin": str(nan_path.relative_to(ROOT)),
           "writer_receipt": receipt, "elapsed_s": time.time() - t0}
    (ROOT / "evidence" / "h77e_e3_build.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    log("wrote evidence/h77e_e3_build.json")
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
