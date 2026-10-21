#!/usr/bin/env python3
"""H80-OTA Experiment 2 of 3: paired HGB arms A_std (33ch) vs A_ota (+4 OTA ch).

Same folds/sampling/emitter/evaluator as the XSEN lane for comparability.
OTA channels come from data/processed/ota_visible.npy (Exp 1; visible-orphan).
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems77 import features77 as F, emit77  # noqa: E402
from gems77h80 import fields as OTA  # noqa: E402

PROC = ROOT / "data" / "processed"
HGB_KW = dict(max_iter=180, learning_rate=0.09, max_leaf_nodes=31,
              min_samples_leaf=40, early_stopping=False, random_state=77)
BUDGET, EXCL = 37_654, 2


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    grid = G.load_grid(); foot, cat = grid.footprint, grid.catalogue
    folds, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]; visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    log(f"holdout seed30: withheld={int(hidden.sum())}")

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

    # stack33: same features77 definitions, only the 33 used channels
    # (full builder OOM-killed; see evidence/h80_stack33_receipt.json)
    stack = np.load(PROC / "stack33.npy", mmap_mode="r")
    ota = np.load(PROC / "ota_visible.npy", mmap_mode="r")
    names33 = F.BLOCKS["raw"] + F.BLOCKS["terrain"] + F.BLOCKS["rad"]
    assert stack.shape == (*foot.shape, len(names33)), stack.shape
    assert ota.shape == (*foot.shape, 4)
    # fail-closed: refuse to train on a degenerate (e.g. truncated/zeroed) stack
    probe = np.asarray(stack[::37, ::37, :], dtype=np.float32)
    pfoot = foot[::37, ::37]
    pin = probe[pfoot]
    stds = np.nanstd(pin, axis=0)
    assert np.isfinite(pin).mean() > 0.9 and (stds > 0).all(), \
        f"stack degenerate finite_in_foot={np.isfinite(pin).mean()} stds={stds}"
    cols_std = list(range(len(names33)))
    log(f"A_std channels={len(cols_std)} A_ota channels={len(cols_std) + 4}")

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
    flat_ota = ota.reshape(-1, ota.shape[2])

    def train_predict(use_ota, idx, y, region):
        X = np.asarray(flat_stack[idx][:, cols_std], dtype=np.float32)
        if use_ota:
            X = np.concatenate([X, np.asarray(flat_ota[idx], dtype=np.float32)], axis=1)
        clf = HistGradientBoostingClassifier(**HGB_KW); clf.fit(X, y); del X; gc.collect()
        out = np.zeros(foot.shape, np.float32)
        ys, xs = np.nonzero(region); r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pred = np.zeros((r1 - r0, c1 - c0), np.float32); step = 100
        nfeat = len(cols_std) + (4 if use_ota else 0)
        for i in range(0, pred.shape[0], step):
            j = min(i + step, pred.shape[0])
            blk_s = np.asarray(stack[r0 + i:r0 + j, c0:c1][:, :, cols_std], dtype=np.float32)
            if use_ota:
                blk_o = np.asarray(ota[r0 + i:r0 + j, c0:c1], dtype=np.float32)
                blk = np.concatenate([blk_s, blk_o], axis=2).reshape(-1, nfeat)
                del blk_s, blk_o
            else:
                blk = blk_s.reshape(-1, nfeat); del blk_s
            pred[i:j] = clf.predict_proba(blk)[:, 1].reshape(j - i, c1 - c0)
            del blk
        out[r0:r1, c0:c1] = np.where(region[r0:r1, c0:c1], pred, 0.0)
        del pred, clf; gc.collect(); return out

    terms_by = {}
    res = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION, "seed": 30,
           "withheld_positive_pixels": int(hidden.sum()), "budget": BUDGET, "exclusion_px": EXCL,
           "arms": {}}
    for name in ("S_ota_xridge", "A0_random"):
        p = PROC / f"h80_dots_{name}.npy"
        if p.exists():
            _, terms_by[name] = score(np.load(p))
    for arm, use_ota in (("A_std", False), ("A_ota", True)):
        surf = np.zeros(foot.shape, np.float32)
        for k, fold in enumerate(folds):
            idx, y = fold_samples[k]
            surf += train_predict(use_ota, idx, y, fold["region"])
            log(f"{arm}/{fold['name']} predicted ({time.time()-t0:.0f}s)")
        np.save(PROC / f"h80_oof_{arm}.npy", surf)
        assert np.isfinite(surf).all() and surf.max() > 0.40, f"{arm} OOF degenerate max={surf.max()}"
        dots = emit_pooled(surf); np.save(PROC / f"h80_dots_{arm}.npy", dots)
        per, terms = score(dots); terms_by[arm] = terms
        res["arms"][arm] = {"dots": int(dots.sum()), "per_fold": per,
                            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
                            "n_channels": len(cols_std) + (4 if use_ota else 0),
                            "oof_max": float(surf.max()), "oof_mean_footprint": float(surf[foot].mean())}
        log(f"{arm}: HOLDOUT-DTI={res['arms'][arm]['pooled_holdout_dti']:.4f}")
    res["pooled_summary"] = EH.pooled_summary(terms_by, draws=1000, seed=520810, candidate="A_ota")
    res["seconds"] = round(time.time() - t0, 1)
    (ROOT / "evidence" / "h80_exp2.json").write_text(json.dumps(res, indent=2) + "\n")
    log("wrote evidence/h80_exp2.json")


if __name__ == "__main__":
    main()
