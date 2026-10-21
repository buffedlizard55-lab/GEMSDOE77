#!/usr/bin/env python3
"""H77E Experiment 2 of 3: dual-seed hide-and-recover holdout.

Arms (all B=37,654 dots, >3 px from visible catalogue, shared 300 m
submodular allocator, vendor gems57-pooled-hide-v2 evaluator):
  A_std       raw 19-band HGB (control)
  A_ucp       H77E new fields only (11)
  A_comb      std + new (30)
  A_comb_ctx  comb + LOCO catalogue-context (dist to a *different* visible
              component + dist to visible tip of a different component;
              train rows via KD-tree distinct-component queries, inference
              rows via EDT to visible faults -- non-catalogue pixels belong
              to no component, so the two agree where it matters)
  A0_champion fold-fair H19-5 construction (random-order dots on the H19-5
              support >2 px from visible catalogue, >=3 px apart)
  A0_random   random surface, same allocator
Seed 20 = primary (all arms); seed 30 = replication (A_std, A_comb_ctx only).
Every catalogue-based feature is derived from the draw's *visible* faults
only; visible faults are masked pixel-exactly by the evaluator.
"""
from __future__ import annotations

import gc
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
from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems77d.submodular import allocate_submodular_300m  # noqa: E402
from gems77lia import allocate as LIA_AL  # noqa: E402

HGB_KW = dict(max_iter=180, learning_rate=0.075, max_leaf_nodes=55,
              min_samples_leaf=50, l2_regularization=2.5,
              early_stopping=False, random_state=7701)
BUDGET = 37654
EXCL_PX = 3.0
H19 = C.RAW / "scored" / "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def tips_of(mask: np.ndarray) -> np.ndarray:
    n = ndimage.convolve(mask.astype(np.int16), np.ones((3, 3), np.int16), mode="constant") - mask.astype(np.int16)
    return mask & (n == 1)


def run_seed(seed, grid, std_fp, std_names, new_fp, new_names, arms_wanted):
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    comp, _ = ndimage.label(visible, np.ones((3, 3), bool))
    tips_vis = tips_of(visible)
    dist_tip = ndimage.distance_transform_edt(~tips_vis).astype(np.float32)
    log(f"=== SEED {seed}: withheld={int(hidden.sum())} visible={int(visible.sum())} ===")

    rng = np.random.default_rng(1000 + seed)
    # ---- per-fold training row sets + catalogue-context features ----
    fold_io = []
    for k, fold in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        vis_tr = visible & tr_reg
        d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
        pos = fp_lut[np.flatnonzero(vis_tr & foot)]
        if pos.size > 48_000:
            pos = rng.choice(pos, 48_000, replace=False)
        neg_base = foot & tr_reg & ~vis_tr & (d_vis_tr > 3.0)
        hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 16.0))]
        easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 16.0))]
        n_neg = int(2.5 * pos.size)
        nh = min(hard.size, int(0.55 * n_neg))
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        te_idx = fp_lut[np.flatnonzero(fold["region"] & foot)]

        # LOCO-style context for TRAIN rows: distance to nearest visible fault
        # pixel of a *different* component (KD-tree, k=8 neighbours).
        def loco(rows_fp, pts_mask, own_from):
            pts_yx = np.argwhere(pts_mask)  # (n,2) row,col
            tree = cKDTree(pts_yx.astype(np.float32))
            rows_yx = np.stack([fp_flat[rows_fp] // w, fp_flat[rows_fp] % w], 1).astype(np.float32)
            own = own_from(rows_yx)
            d, idx = tree.query(rows_yx, k=8)
            pt_comp = comp[pts_yx[:, 0], pts_yx[:, 1]]
            nb_comp = pt_comp[idx]
            valid = nb_comp != own[:, None]
            first = np.argmax(valid, axis=1)
            has = valid.any(axis=1)
            out = np.where(has, d[np.arange(len(rows_fp)), first], 100.0)
            return np.log1p(np.clip(out, 0, 100)).astype(np.float32)

        tr_ctx = np.stack([
            loco(tr_idx, vis_tr, lambda yx: comp[yx[:, 0].astype(int), yx[:, 1].astype(int)]),
            loco(tr_idx, tips_vis & tr_reg, lambda yx: comp[yx[:, 0].astype(int), yx[:, 1].astype(int)]),
        ], 1)
        # INFERENCE rows: EDT to visible (rows are mostly non-catalogue, so the
        # nearest visible fault is by definition another component).
        te_yx = fp_flat[te_idx]
        te_ctx = np.stack([np.log1p(np.clip(dist_vis.ravel()[te_yx], 0, 100)),
                           np.log1p(np.clip(dist_tip.ravel()[te_yx], 0, 100))], 1).astype(np.float32)
        fold_io.append((fold, tr_idx, y_tr, tr_ctx, te_idx, te_ctx))
        log(f" fold {fold['name']}: train={tr_idx.size} test={te_idx.size}")

    # ---- feature assembly per arm ----
    Xall_cache = {}

    def arm_matrix(arm):
        if arm in Xall_cache:
            return Xall_cache[arm]
        if arm == "A_std":
            m = np.asarray(std_fp, dtype=np.float32)
        elif arm == "A_ucp":
            m = np.asarray(new_fp, dtype=np.float32)
        else:  # A_comb / A_comb_ctx share the static part
            m = np.hstack([np.asarray(std_fp, dtype=np.float32),
                           np.asarray(new_fp, dtype=np.float32)])
        Xall_cache[arm] = m
        return m

    def fit_predict(arm, tr_idx, y_tr, tr_ctx, te_idx, te_ctx):
        X = arm_matrix(arm)
        x_tr = X[tr_idx]
        x_te = X[te_idx]
        if arm == "A_comb_ctx":
            x_tr = np.hstack([x_tr, tr_ctx])
            x_te = np.hstack([x_te, te_ctx])
        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(x_tr, y_tr)
        preds = np.empty(te_idx.size, dtype=np.float32)
        for s in range(0, te_idx.size, 250_000):
            sl = slice(s, min(s + 250_000, te_idx.size))
            preds[sl] = clf.predict_proba(x_te[sl])[:, 1]
        del clf, x_tr, x_te
        gc.collect()
        return preds

    surfs = {}
    for arm in arms_wanted:
        if arm.startswith("A0"):
            continue
        cache = C.CACHE / f"oof_{arm}_seed{seed}.npy"
        if cache.exists():
            surfs[arm] = np.load(cache)
            log(f" {arm}: loaded cached OOF surface")
            continue
        surf = np.zeros((h, w), dtype=np.float32)
        for fold, tr_idx, y_tr, tr_ctx, te_idx, te_ctx in fold_io:
            surf.ravel()[fp_flat[te_idx]] = fit_predict(arm, tr_idx, y_tr, tr_ctx, te_idx, te_ctx)
            log(f" {arm} fold {fold['name']} done")
        np.save(cache, surf)
        surfs[arm] = surf

    # ---- emission + scoring ----
    def emit_submod(surf):
        allowed_folds, tot = [], 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > EXCL_PX)
            allowed_folds.append(a)
            tot += int(a.sum())
        dots = np.zeros((h, w), bool)
        for a in allowed_folds:
            b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= allocate_submodular_300m(surf, a, target_n=b, support_quantile=0.968,
                                                 gamma=3.5, min_sep_px=2.20)
        return dots

    def score_dots(dots):
        terms, per = None, {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": float(res["dti"]), "n_truth": int(res["n_truth"]),
                                 "emitted": int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        return per, terms

    terms_by_arm, details = {}, {}
    for arm in arms_wanted:
        if arm == "A0_random":
            surf = np.random.default_rng(999 + seed).random((h, w), dtype=np.float32) * foot
            d = emit_submod(surf)
        elif arm == "A0_champion":
            import rasterio
            with rasterio.open(H19) as src:
                sup = src.read(1)
            sup = np.isfinite(sup) & (sup > 0)
            rk = np.random.default_rng(7).random(h * w)
            d = np.zeros((h, w), bool)
            for fold in folds:
                valid_f = fold["region"] & foot & ~fold["masked_known"] & (dist_vis > 2.0)
                cand = np.flatnonzero((sup & valid_f).ravel())
                sel = LIA_AL.greedy_select(cand[np.argsort(rk[cand])], 3.0)
                d.ravel()[sel] = True
        else:
            d = emit_submod(surfs[arm])
        per, t = score_dots(d)
        terms_by_arm[arm] = t
        details[arm] = {"dots": int(d.sum()), "per_fold": per,
                        "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0)))}
        log(f" [seed {seed}] {arm}: HOLDOUT-DTI={details[arm]['pooled_holdout_dti']:.5f} dots={int(d.sum())}")
        np.save(C.CACHE / f"dots_{arm}_seed{seed}.npy", d)

    cand = "A_comb_ctx" if "A_comb_ctx" in terms_by_arm else "A_comb"
    pooled = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate=cand)
    return {"seed": seed, "withheld_positive_pixels": int(hidden.sum()),
            "fold_receipts": [f["receipt"] for f in folds],
            "arms": details, "pooled_summary": pooled}


def main() -> int:
    t0 = time.time()
    grid = C.load_grid()
    foot = grid.footprint
    std_fp, std_names = C.build_std_fp(foot)
    new_fp = np.load(C.CACHE / "new_fp.npy", mmap_mode="r")
    new_names = json.loads((C.CACHE / "new_names.json").read_text())

    out = {"experiment": "H77E-E2", "evidence_class": "HOLDOUT-DTI",
           "evaluator_version": EH.VERSION, "alpha": 0.2, "beta": 0.8,
           "kernel_radius_m": 300.0, "budget_dots": BUDGET, "exclusion_px": EXCL_PX,
           "arms_seed20": ["A_std", "A_ucp", "A_comb", "A_comb_ctx", "A0_champion", "A0_random"],
           "arms_seed30": ["A_std", "A_comb_ctx"], "seeds": {}}
    r20 = run_seed(20, grid, std_fp, std_names, new_fp, new_names, out["arms_seed20"])
    out["seeds"]["20"] = r20
    (ROOT / "evidence" / "h77e_e2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    log("seed20 written")
    r30 = run_seed(30, grid, std_fp, std_names, new_fp, new_names, out["arms_seed30"])
    out["seeds"]["30"] = r30
    out["elapsed_s"] = time.time() - t0
    (ROOT / "evidence" / "h77e_e2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"DONE in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
