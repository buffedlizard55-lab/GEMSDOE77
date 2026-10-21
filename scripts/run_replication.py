#!/usr/bin/env python3
"""GEMSDOE77 Exp 3 (amended preregistration): seed-21 replication, A4 vs A2 only.

Frozen rule (registry/preregistration_amendment_exp3.json): promote only if
seed-21 paired CI AND two-draw pooled paired CI both exclude 0 positively.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor")); sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems77 import features77 as F, emit77  # noqa: E402
from run_holdout import arm_channels, train_predict  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SEED = 21
BUDGET, EXCL = 44_090, 3


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    grid = G.load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    folds, _ = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]; visible = folds[0]["visible"]
    log(f"seed21 holdout: withheld={int(hidden.sum())} visible={int(visible.sum())}")
    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    rng = np.random.default_rng(4321)

    fold_samples = []
    for k, fold in enumerate(folds):
        train_region = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                train_region |= f2["region"]
        pos = np.flatnonzero(visible & train_region)
        if pos.size > 40_000:
            pos = rng.choice(pos, 40_000, replace=False)
        neg_mask = train_region & ~cat & (dist_cat > 3)
        hard = np.flatnonzero(neg_mask & (dist_cat <= 15))
        easy = np.flatnonzero(neg_mask & (dist_cat > 15))
        n_neg = 2 * pos.size
        nh = min(hard.size, n_neg // 2); ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
        idx = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        fold_samples.append((idx, y))

    def emit_pooled(p):
        allowed_folds, tot = [], 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > EXCL)
            allowed_folds.append(a); tot += int(a.sum())
        dots = np.zeros(foot.shape, bool)
        for a in allowed_folds:
            b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= emit77.emit_dots(p, a, b)
        return dots

    def score_dots(dots):
        terms, per = None, {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": res["dti"], "n_truth": res["n_truth"]}
            terms = t if terms is None else terms + t
        return per, terms

    terms_by_arm, out = {}, {"evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
                             "seed": SEED, "withheld_positive_pixels": int(hidden.sum()),
                             "budget": BUDGET, "exclusion_px": EXCL, "arms": {}}
    for arm in ("A2_standard", "A4_primary"):
        cols = [F.IDX[n] for n in arm_channels(arm)]
        surf = np.zeros(foot.shape, np.float32)
        for k, fold in enumerate(folds):
            idx, y = fold_samples[k]
            surf += train_predict(stack, cols, idx, y, fold["region"], f"s21-{arm}/{fold['name']}")
        np.save(PROC / f"oof_s21_{arm}.npy", surf)
        dots = emit_pooled(surf)
        per, terms = score_dots(dots)
        terms_by_arm[arm] = terms
        out["arms"][arm] = {"dots": int(dots.sum()), "per_fold": per,
                            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        np.save(PROC / f"terms_s21_{arm}.npy", terms)
        log(f"s21 {arm}: pooled HOLDOUT-DTI={out['arms'][arm]['pooled_holdout_dti']:.4f}")
        del surf; gc.collect()

    summary21 = EH.pooled_summary(terms_by_arm, draws=1000, seed=770021, candidate="A4_primary")
    out["pooled_summary_seed21"] = summary21

    # two-draw pooled: sum Exp1 (seed30) terms by re-scoring saved Exp1 dots is NOT
    # valid (different folds). Instead reload Exp1 terms via its saved dot masks
    # scored on seed-30 folds again through identical evaluator, then add terms.
    folds30, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    pooled_terms = {}
    for arm in ("A2_standard", "A4_primary"):
        dots30 = np.load(PROC / f"dots_holdout_{arm}.npy")
        t30 = None
        for fold in folds30:
            _, t = EH.evaluate(dots30.astype(np.float32), fold, valid=foot, block_side=200)
            t30 = t if t30 is None else t30 + t
        pooled_terms[arm] = t30 + terms_by_arm[arm]
        log(f"two-draw terms pooled for {arm}")
    summary2 = EH.pooled_summary(pooled_terms, draws=1000, seed=770030, candidate="A4_primary")
    out["pooled_summary_two_draw"] = summary2
    d21 = summary21["paired_differences"]["A2_standard"]
    d2 = summary2["paired_differences"]["A2_standard"]
    out["frozen_rule"] = {
        "seed21_delta": d21["delta"], "seed21_ci95": d21["ci95"],
        "seed21_pass": bool(d21["ci95"][0] > 0),
        "two_draw_delta": d2["delta"], "two_draw_ci95": d2["ci95"],
        "two_draw_pass": bool(d2["ci95"][0] > 0),
        "promote_under_amendment": bool(d21["ci95"][0] > 0 and d2["ci95"][0] > 0),
    }
    out["elapsed_s"] = round(time.time() - t0, 1)
    (ROOT / "evidence" / "holdout_exp3_replication.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"DONE {out['elapsed_s']}s; frozen rule: {out['frozen_rule']}")


if __name__ == "__main__":
    main()
