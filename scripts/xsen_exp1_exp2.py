#!/usr/bin/env python3
"""H77-XSEN Exp 1 (canaries + surface-alone screen) and Exp 2 (HGB arms).

Evaluator: vendor gems57.evaluate_holdout (gems57-pooled-hide-v2), unchanged.
Holdout: HO.buffered_component_draw(seed=30, hide_frac=0.20, buffer_px=3).
Emitter: gems77.emit77.emit_dots, budget 37654, min_sep 2.8, dist_visible>2.
Every catalogue-derived quantity inside the holdout uses fold['visible'] only.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems57 import metric as M  # noqa: E402
from gems77 import features77 as F, emit77  # noqa: E402

RAW = ROOT / "data" / "raw"; PROC = ROOT / "data" / "processed"
HGB_KW = dict(max_iter=180, learning_rate=0.09, max_leaf_nodes=31,
              min_samples_leaf=40, early_stopping=False, random_state=77)
BUDGET, EXCL = 37_654, 2
XSEN_NAMES = ["xsen_as_ridge_s1p5", "xsen_as_ridge_s3", "xsen_as_persist",
              "xsen_radgrad_thk", "xsen_radgrad_uk", "xsen_radgrad",
              "xsen_coincide", "xsen_coincide_min"]


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, required=True)
    args = ap.parse_args(); t0 = time.time()
    grid = G.load_grid(RAW); foot, cat = grid.footprint, grid.catalogue
    folds, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]; visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    log(f"holdout seed30: withheld={int(hidden.sum())} visible={int(visible.sum())}")

    def emit_pooled(p):
        tots = [int((f["region"] & foot & ~visible & (dist_vis > EXCL)).sum()) for f in folds]
        tot = max(sum(tots), 1); dots = np.zeros(foot.shape, bool)
        for f, a_n in zip(folds, tots):
            b = int(round(BUDGET * a_n / tot))
            if b > 0:
                dots |= emit77.emit_dots(p, f["region"] & foot & ~visible & (dist_vis > EXCL), b)
        return dots

    def score(dots):
        terms = None; per = {}
        for f in folds:
            res, t = EH.evaluate(dots.astype(np.float32), f, valid=foot, block_side=200)
            per[f["name"]] = {"dti": res["dti"], "n_truth": res["n_truth"], "emitted": res["n_emitted"]}
            terms = t if terms is None else terms + t
        return per, terms

    xsen = np.load(PROC / "xsen.npy", mmap_mode="r")

    if args.exp == 1:
        rng = np.random.default_rng(1234)
        pos_idx = np.flatnonzero(hidden)
        neg_pool = np.flatnonzero(foot & ~cat & (dist_cat > 3))
        neg_idx = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        y = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])
        can = {}
        flat = xsen.reshape(-1, xsen.shape[2])
        Xc = np.asarray(flat[idx], dtype=np.float32)
        for j, n in enumerate(XSEN_NAMES):
            v = Xc[:, j]; m = np.isfinite(v)
            auc = float(roc_auc_score(y[m], v[m])) if m.sum() > 100 else None
            can[n] = {"auc": auc, "symmetric_auc": max(auc, 1 - auc) if auc is not None else None}
            log(f"canary {n}: AUC={auc}")
        del Xc, flat
        leak = [n for n, r in can.items() if r["symmetric_auc"] and r["symmetric_auc"] > 0.90]
        arms_surf = {
            "S_xsen_coincide": np.asarray(xsen[:, :, 6], np.float32),
            "S_as_only": np.asarray(xsen[:, :, 2], np.float32),
            "S_rad_only": np.asarray(xsen[:, :, 5], np.float32),
            "A0_random": np.random.default_rng(999).random(foot.shape, dtype=np.float32) * foot,
        }
        terms_by = {}; res = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
                             "alpha": M.ALPHA, "beta": M.BETA, "kernel_radius_m": M.R_M,
                             "seed": 30, "withheld_positive_pixels": int(hidden.sum()),
                             "budget": BUDGET, "exclusion_px": EXCL, "canaries": can,
                             "leakage_gt_0p90": leak, "arms": {}}
        for name, surf in arms_surf.items():
            dots = emit_pooled(np.where(foot, surf, 0).astype(np.float32))
            per, terms = score(dots); terms_by[name] = terms
            res["arms"][name] = {"dots": int(dots.sum()), "per_fold": per,
                                 "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
            log(f"{name}: HOLDOUT-DTI={res['arms'][name]['pooled_holdout_dti']:.4f} dots={int(dots.sum())}")
            np.save(PROC / f"xsen_dots_{name}.npy", dots)
        res["pooled_summary"] = EH.pooled_summary(terms_by, draws=1000, seed=520810, candidate="S_xsen_coincide")
        res["verdict"] = "STOP-LEAKAGE" if leak else "screen-complete"
        res["seconds"] = round(time.time() - t0, 1)
        (ROOT / "evidence" / "xsen_exp1.json").write_text(json.dumps(res, indent=2) + "\n")
        log("wrote evidence/xsen_exp1.json"); return

    # ---------------- Exp 2: HGB arms ----------------
    # NOTE (bug fix, first attempt invalidated): a materialised combined
    # memmap was OOM-killed mid-write and a later run mistook the all-zero
    # file for a built one (evidence/xsen_exp2_INVALID_combined_memmap.json).
    # Arms now gather from the two source memmaps directly; no combined file.
    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    cols_std = [F.IDX[n] for n in F.BLOCKS["raw"] + F.BLOCKS["terrain"] + F.BLOCKS["rad"]]
    cols_xsen = list(range(len(XSEN_NAMES)))  # indices into xsen, not the stack
    rng = np.random.default_rng(1234)
    fold_samples = []
    for k, fold in enumerate(folds):
        tr = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k: tr |= f2["region"]
        pos = np.flatnonzero(visible & tr)
        if pos.size > 40_000: pos = rng.choice(pos, 40_000, replace=False)
        neg_mask = tr & ~cat & (dist_cat > 3)
        hard = np.flatnonzero(neg_mask & (dist_cat <= 15)); easy = np.flatnonzero(neg_mask & (dist_cat > 15))
        n_neg = 2 * pos.size; nh = min(hard.size, n_neg // 2); ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False) if nh else np.array([], int),
                              rng.choice(easy, ne, replace=False) if ne else np.array([], int)])
        fold_samples.append((np.concatenate([pos, neg]),
                             np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])))
        log(f"fold {fold['name']}: pos={pos.size} neg={neg.size}")

    flat_stack = stack.reshape(-1, stack.shape[2])
    flat_xsen = xsen.reshape(-1, xsen.shape[2])

    def train_predict(cols, use_xsen, idx, y, region):
        X = np.asarray(flat_stack[idx][:, cols], dtype=np.float32)
        if use_xsen:
            X = np.concatenate([X, np.asarray(flat_xsen[idx], dtype=np.float32)], axis=1)
        clf = HistGradientBoostingClassifier(**HGB_KW); clf.fit(X, y); del X; gc.collect()
        out = np.zeros(foot.shape, np.float32)
        ys, xs = np.nonzero(region); r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pred = np.zeros((r1 - r0, c1 - c0), np.float32); step = 100
        for i in range(0, pred.shape[0], step):
            j = min(i + step, pred.shape[0])
            blk_s = np.asarray(stack[r0 + i:r0 + j, c0:c1][:, :, cols], dtype=np.float32)
            if use_xsen:
                blk_x = np.asarray(xsen[r0 + i:r0 + j, c0:c1], dtype=np.float32)
                blk = np.concatenate([blk_s, blk_x], axis=2).reshape(-1, len(cols) + len(XSEN_NAMES))
                del blk_s, blk_x
            else:
                blk = blk_s.reshape(-1, len(cols)); del blk_s
            pred[i:j] = clf.predict_proba(blk)[:, 1].reshape(j - i, c1 - c0)
            del blk
        out[r0:r1, c0:c1] = np.where(region[r0:r1, c0:c1], pred, 0.0)
        del pred, clf; gc.collect(); return out

    terms_by = {}; res = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
                          "seed": 30, "withheld_positive_pixels": int(hidden.sum()),
                          "budget": BUDGET, "exclusion_px": EXCL, "arms": {}}
    # reuse Exp-1 surface/random terms for the pooled comparison
    for name in ("S_xsen_coincide", "A0_random"):
        p = PROC / f"xsen_dots_{name}.npy"
        if p.exists():
            _, terms_by[name] = score(np.load(p))
    for arm, cols, use_xsen in (("A_std", cols_std, False), ("A_xsen", cols_std, True)):
        surf = np.zeros(foot.shape, np.float32)
        for k, fold in enumerate(folds):
            idx, y = fold_samples[k]
            surf += train_predict(cols, use_xsen, idx, y, fold["region"])
            log(f"{arm}/{fold['name']} predicted ({time.time()-t0:.0f}s)")
        np.save(PROC / f"xsen_oof_{arm}.npy", surf)
        assert np.isfinite(surf).all() and surf.max() > 0.40, f"{arm} OOF surface degenerate (max={surf.max()})"
        dots = emit_pooled(surf); np.save(PROC / f"xsen_dots_{arm}.npy", dots)
        per, terms = score(dots); terms_by[arm] = terms
        res["arms"][arm] = {"dots": int(dots.sum()), "per_fold": per,
                            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
                            "n_channels": len(cols) + (len(XSEN_NAMES) if use_xsen else 0),
                            "oof_max": float(surf.max()), "oof_mean_footprint": float(surf[foot].mean())}
        log(f"{arm}: HOLDOUT-DTI={res['arms'][arm]['pooled_holdout_dti']:.4f}")
    res["pooled_summary"] = EH.pooled_summary(terms_by, draws=1000, seed=520810, candidate="A_xsen")
    res["seconds"] = round(time.time() - t0, 1)
    (ROOT / "evidence" / "xsen_exp2.json").write_text(json.dumps(res, indent=2) + "\n")
    log("wrote evidence/xsen_exp2.json")


if __name__ == "__main__":
    main()
