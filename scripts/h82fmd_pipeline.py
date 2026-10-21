#!/usr/bin/env python3
"""H82-FMD lane driver: cache -> E1 canary -> E2 holdout -> E3 build.

Shared instruments are used unchanged: ``vendor/gems57`` supplies the grid, the
hide-and-recover holdout, the DTI metric, the evaluator
(``gems57-pooled-hide-v2``), the validator and the submission writer;
``src/gems77d/submodular.py`` supplies the 300 m greedy allocator.  Only the
ranking surface is new (``src/gems82``).

Usage::

    python scripts/h82fmd_pipeline.py cache
    python scripts/h82fmd_pipeline.py e1
    python scripts/h82fmd_pipeline.py e2 --seed 20
    python scripts/h82fmd_pipeline.py e2 --seed 30 --arms A_std,A_fmd
    python scripts/h82fmd_pipeline.py e3
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH            # noqa: E402
from gems57 import holdout as HO                     # noqa: E402
from gems57 import grid as G                         # noqa: E402
from gems77d.submodular import allocate_submodular_300m  # noqa: E402
from gems82fmd import deficit as DEF                    # noqa: E402
from gems82fmd import fields as F                       # noqa: E402

RAW = ROOT / "data" / "raw"
TF = RAW / "training_features.tif"
CACHE = ROOT / ".cache" / "h82fmd"
EVID = ROOT / "evidence"

BUDGET = 37654
EXCL_HOLDOUT_PX = 3.0
EXCL_FINAL_PX = 2.0
ALLOC_KW = dict(support_quantile=0.968, gamma=3.5, min_sep_px=2.20)
HGB_KW = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=48,
              min_samples_leaf=50, l2_regularization=2.0,
              early_stopping=False, random_state=8202)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_grid():
    return G.load_grid(ROOT / "data" / "bridge")


def get_stack(footprint, ext: bool = False):
    CACHE.mkdir(parents=True, exist_ok=True)
    base = F.build_stack(TF, footprint, CACHE / "stack_fp.npy")
    if not ext:
        return base
    extra = F.build_ext_stack(RAW, footprint, CACHE / "stack_ext.npy")
    return _Joined(base, extra)


class _Joined:
    """Row-indexable view over two footprint-compacted memmaps (no 1 GB copy)."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.shape = (a.shape[0], a.shape[1] + b.shape[1])

    def __getitem__(self, rows):
        if isinstance(rows, tuple):          # stack[rows, j]
            r, j = rows
            na = self.a.shape[1]
            return self.a[r, j] if j < na else self.b[r, j - na]
        return np.hstack([np.asarray(self.a[rows], np.float32),
                          np.asarray(self.b[rows], np.float32)])


# --------------------------------------------------------------------------- E1
def cmd_cache(_args):
    grid = load_grid()
    t0 = time.time()
    stack = get_stack(grid.footprint)
    desc = F.band_descriptions(TF)
    out = {
        "channels": F.CHANNELS,
        "n_footprint_px": int(grid.footprint.sum()),
        "n_catalogue_px": int(grid.catalogue.sum()),
        "stack_shape": list(stack.shape),
        "embedded_band_descriptions": desc,
        "finite_fraction_per_channel": {
            n: float(np.isfinite(np.asarray(stack[:, i])).mean())
            for i, n in enumerate(F.CHANNELS)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (EVID / "h82fmd_cache.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"cache ready {stack.shape} in {out['elapsed_s']}s")
    return 0


def cmd_e1(args):
    """Leakage canary: AUC of every channel alone on the withheld positives."""
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    stack = get_stack(foot)
    fp_flat = np.flatnonzero(foot)
    lut = np.full(foot.size, -1, np.int32)
    lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    folds, _ = HO.buffered_component_draw(grid, seed=args.seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    region = np.zeros_like(foot)
    for f in folds:
        region |= f["region"]
    d_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)

    pos_px = np.flatnonzero((hidden & region & foot).ravel())
    neg_pool = np.flatnonzero((foot & region & ~cat & (d_vis > 3.0)).ravel())
    rng = np.random.default_rng(820 + args.seed)
    neg_px = rng.choice(neg_pool, min(200_000, neg_pool.size), replace=False)
    pos_rows = lut[pos_px]
    neg_rows = lut[neg_px]
    y = np.concatenate([np.ones(pos_rows.size, np.int8), np.zeros(neg_rows.size, np.int8)])
    rows = np.concatenate([pos_rows, neg_rows])

    aucs = {}
    for j, name in enumerate(F.CHANNELS):
        v = np.asarray(stack[rows, j], np.float64)
        ok = np.isfinite(v)
        if ok.sum() < 1000 or len(np.unique(y[ok])) < 2:
            aucs[name] = None
            continue
        a = float(roc_auc_score(y[ok], v[ok]))
        aucs[name] = round(max(a, 1.0 - a), 4)   # direction-free discriminability

    # deficit weight canary (same LOQO structure as E2, fold by fold)
    blk, (nby, nbx) = DEF.block_ids(foot.shape)
    n_blocks = nby * nbx
    blk_fp = blk[foot]
    bfeat = DEF.aggregate(stack, blk_fp, n_blocks)
    n_fp_blk = DEF.block_counts(foot, blk, n_blocks)
    n_vis_blk = DEF.block_counts(visible & foot, blk, n_blocks)
    wmap = np.zeros(foot.shape, np.float32)
    for k, fold in enumerate(folds):
        tr_blocks = np.zeros(n_blocks, bool)
        for j2, f2 in enumerate(folds):
            if j2 != k:
                tr_blocks |= DEF.block_counts(f2["region"] & foot, blk, n_blocks) > 100
        te_blocks = DEF.block_counts(fold["region"] & foot, blk, n_blocks) > 0
        tr_blocks &= ~te_blocks
        w = DEF.deficit_field(bfeat, n_fp_blk, n_vis_blk, tr_blocks, te_blocks,
                              blk, foot.shape, foot)
        wmap[fold["region"] & foot] = w[fold["region"] & foot]
    np.save(CACHE / f"wmap_seed{args.seed}.npy", wmap)
    wv = np.concatenate([wmap.ravel()[pos_px], wmap.ravel()[neg_px]])
    a = float(roc_auc_score(y, wv))
    aucs["mapping_deficit_weight"] = round(max(a, 1.0 - a), 4)

    flagged = {k: v for k, v in aucs.items() if v is not None and v > 0.90}
    out = {"experiment": "H82-E1", "evidence_class": "canary-diagnostic",
           "seed": args.seed, "n_withheld_positive_px": int(pos_rows.size),
           "n_negative_px": int(neg_rows.size),
           "auc_single_channel_direction_free": aucs,
           "leakage_threshold": 0.90, "channels_above_threshold": flagged,
           "verdict": "no leakage" if not flagged else "LEAKAGE SUSPECTED"}
    (EVID / "h82fmd_e1_canary.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"E1 done. max AUC = {max(v for v in aucs.values() if v is not None):.4f}; flagged={list(flagged)}")
    return 0


# --------------------------------------------------------------------------- E2
def _train_surface(stack, fp_lut, fp_flat, foot, visible, folds, shape, rng):
    """OOF per-pixel physics surface S_phys (LOQO over the four quadrants)."""
    h, w = shape
    d_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    surf = np.zeros((h, w), np.float32)
    for k, fold in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        vis_tr = visible & tr_reg
        d_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
        pos = fp_lut[np.flatnonzero((vis_tr & foot).ravel())]
        if pos.size > 48_000:
            pos = rng.choice(pos, 48_000, replace=False)
        neg_base = foot & tr_reg & ~vis_tr & (d_tr > 3.0)
        hard = fp_lut[np.flatnonzero((neg_base & (d_tr <= 16.0)).ravel())]
        easy = fp_lut[np.flatnonzero((neg_base & (d_tr > 16.0)).ravel())]
        n_neg = int(2.5 * pos.size)
        nh = min(hard.size, int(0.55 * n_neg))
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False),
                              rng.choice(easy, ne, replace=False)])
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        te_idx = fp_lut[np.flatnonzero((fold["region"] & foot).ravel())]
        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(np.asarray(stack[tr_idx], np.float32), y_tr)
        preds = np.empty(te_idx.size, np.float32)
        for s in range(0, te_idx.size, 250_000):
            sl = slice(s, min(s + 250_000, te_idx.size))
            preds[sl] = clf.predict_proba(np.asarray(stack[te_idx[sl]], np.float32))[:, 1]
        surf.ravel()[fp_flat[te_idx]] = preds
        del clf
        gc.collect()
        log(f"  S_phys fold {fold['name']}: train={tr_idx.size} test={te_idx.size}")
    return surf, d_vis


def _emit(surface, folds, foot, visible, d_vis, budget=BUDGET, excl=EXCL_HOLDOUT_PX):
    allowed, tot = [], 0
    for fold in folds:
        a = fold["region"] & foot & ~visible & (d_vis > excl)
        allowed.append(a)
        tot += int(a.sum())
    dots = np.zeros(foot.shape, bool)
    for a in allowed:
        b = int(round(budget * int(a.sum()) / max(tot, 1)))
        if b > 0:
            dots |= allocate_submodular_300m(surface, a, target_n=b, **ALLOC_KW)
    return dots


def cmd_e2(args):
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    stack = get_stack(foot)
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(foot.size, -1, np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    seed = args.seed
    arms = args.arms.split(",")
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    log(f"=== SEED {seed}: withheld={int(hidden.sum())} visible={int(visible.sum())} ===")

    rng = np.random.default_rng(1000 + seed)
    sp_cache = CACHE / f"sphys_seed{seed}.npy"
    if sp_cache.exists():
        s_phys = np.load(sp_cache)
        d_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
        log("  S_phys loaded from cache")
    else:
        s_phys, d_vis = _train_surface(stack, fp_lut, fp_flat, foot, visible,
                                       folds, (h, w), rng)
        np.save(sp_cache, s_phys)

    w_cache = CACHE / f"wmap_seed{seed}.npy"
    if w_cache.exists():
        wmap = np.load(w_cache)
        log("  deficit weight loaded from cache")
    else:
        blk, (nby, nbx) = DEF.block_ids(foot.shape)
        n_blocks = nby * nbx
        bfeat = DEF.aggregate(stack, blk[foot], n_blocks)
        n_fp_blk = DEF.block_counts(foot, blk, n_blocks)
        n_vis_blk = DEF.block_counts(visible & foot, blk, n_blocks)
        wmap = np.zeros(foot.shape, np.float32)
        for k, fold in enumerate(folds):
            tr_blocks = np.zeros(n_blocks, bool)
            for j2, f2 in enumerate(folds):
                if j2 != k:
                    tr_blocks |= DEF.block_counts(f2["region"] & foot, blk, n_blocks) > 100
            te_blocks = DEF.block_counts(fold["region"] & foot, blk, n_blocks) > 0
            tr_blocks &= ~te_blocks
            wf = DEF.deficit_field(bfeat, n_fp_blk, n_vis_blk, tr_blocks, te_blocks,
                                   blk, foot.shape, foot)
            wmap[fold["region"] & foot] = wf[fold["region"] & foot]
            log(f"  deficit fold {fold['name']} done")
        np.save(w_cache, wmap)

    surfaces = {
        "A_std": s_phys,
        "A_fmd": DEF.apply_weight(s_phys, wmap),
        "A_def": (wmap + 1e-6 * np.random.default_rng(7).random((h, w)).astype(np.float32)) * foot,
        "A0_random": np.random.default_rng(999 + seed).random((h, w), dtype=np.float32) * foot,
    }

    terms_by_arm, details = {}, {}
    for arm in arms:
        t0 = time.time()
        dots = _emit(surfaces[arm], folds, foot, visible, d_vis)
        per, terms = {}, None
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": float(res["dti"]), "n_truth": int(res["n_truth"]),
                                 "emitted": int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        terms_by_arm[arm] = terms
        details[arm] = {"dots": int(dots.sum()), "per_fold": per,
                        "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
                        "alloc_s": round(time.time() - t0, 1)}
        np.save(CACHE / f"dots_{arm}_seed{seed}.npy", dots)
        log(f" [seed {seed}] {arm}: HOLDOUT-DTI={details[arm]['pooled_holdout_dti']:.6f} "
            f"dots={int(dots.sum())} ({details[arm]['alloc_s']}s)")

    pooled = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810,
                               candidate="A_fmd" if "A_fmd" in terms_by_arm else arms[0])
    out_path = EVID / "h82fmd_e2_holdout.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "experiment": "H82-E2", "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION, "evaluator_hashes": EH.implementation_hashes(),
        "alpha": 0.2, "beta": 0.8, "kernel_radius_m": 300.0,
        "budget_dots": BUDGET, "exclusion_px": EXCL_HOLDOUT_PX,
        "allocator": "gems77d.submodular.allocate_submodular_300m", "allocator_kw": ALLOC_KW,
        "seeds": {}}
    out["seeds"][str(seed)] = {
        "seed": seed, "withheld_positive_pixels": int(hidden.sum()),
        "fold_receipts": [f["receipt"] for f in folds],
        "arms": details, "pooled_summary": pooled}
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"E2 seed {seed} written -> {out_path}")
    return 0


# -------------------------------------------------------------------- E3 (amend)
def cmd_e2b(args):
    """Amendment E3: shared allocation-geometry fix (across-ridge NMS crests).

    Declared in registry/preregistration_h82fmd_amendment_e3.json BEFORE running.
    Every arm uses the identical allocator, budget and exclusion; the only
    change is that the ranking surface is restricted to its across-ridge crest
    (``gems82fmd.fields.crest_mask``).  ``B_fmd_nms`` re-tests the refuted lane
    hypothesis under the corrected geometry.
    """
    grid = load_grid()
    foot = grid.footprint
    h, w = foot.shape
    seed = args.seed
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    d_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    s_phys = np.load(CACHE / f"sphys_seed{seed}.npy")
    wmap = np.load(CACHE / f"wmap_seed{seed}.npy")

    crest = F.crest_mask(s_phys, foot)
    log(f"  crest pixels = {int(crest.sum())} ({100*crest.sum()/foot.sum():.1f}% of footprint)")
    s_nms = (s_phys * crest).astype(np.float32)

    arms = {
        "A_std": (s_phys, ALLOC_KW),
        "B_std_nms": (s_nms, ALLOC_KW),
        "B_std_nms_sep3": (s_nms, {**ALLOC_KW, "min_sep_px": 3.0}),
        "B_fmd_nms": (DEF.apply_weight(s_nms, wmap), ALLOC_KW),
    }
    wanted = args.arms.split(",")
    terms_by_arm, details = {}, {}
    for arm in wanted:
        surface, kw = arms[arm]
        t0 = time.time()
        allowed, tot = [], 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (d_vis > EXCL_HOLDOUT_PX)
            allowed.append(a)
            tot += int(a.sum())
        dots = np.zeros((h, w), bool)
        for a in allowed:
            b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= allocate_submodular_300m(surface, a, target_n=b, **kw)
        per, terms = {}, None
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": float(res["dti"]), "n_truth": int(res["n_truth"]),
                                 "emitted": int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        terms_by_arm[arm] = terms
        details[arm] = {"dots": int(dots.sum()), "per_fold": per,
                        "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
                        "alloc_s": round(time.time() - t0, 1), "allocator_kw": kw}
        np.save(CACHE / f"dots_{arm}_seed{seed}.npy", dots)
        log(f" [seed {seed}] {arm}: HOLDOUT-DTI={details[arm]['pooled_holdout_dti']:.6f} "
            f"dots={int(dots.sum())} ({details[arm]['alloc_s']}s)")

    pooled = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810,
                               candidate="B_std_nms" if "B_std_nms" in terms_by_arm else wanted[0])
    out_path = EVID / "h82fmd_e3_amendment_nms.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "experiment": "H82-E3-amendment", "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION, "budget_dots": BUDGET,
        "exclusion_px": EXCL_HOLDOUT_PX, "seeds": {}}
    out["seeds"][str(seed)] = {"seed": seed, "withheld_positive_pixels": int(hidden.sum()),
                               "crest_px": int(crest.sum()), "arms": details,
                               "pooled_summary": pooled}
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"E3 amendment seed {seed} written -> {out_path}")
    return 0


# ------------------------------------------------------- E4 (full shared stack)
def cmd_e4(args):
    """Rule-2 completion: same two arms on the FULL mirrored channel stack.

    E1-E3 used only the 19 shipped bands plus nine derivatives.  The template's
    cached stack also contains twenty mirrored external channels (3DEP lidar
    scarp morphology, GeoDAWN radiometrics).  This run adds them to the
    identical pipeline so the lane's conclusion is not an artefact of a
    deliberately thin feature stack.  Declared in
    registry/preregistration_h82fmd_amendment_e4.json before execution.
    """
    grid = load_grid()
    foot = grid.footprint
    h, w = foot.shape
    seed = args.seed
    stack = get_stack(foot, ext=True)
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(foot.size, -1, np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    rng = np.random.default_rng(1000 + seed)
    cache = CACHE / f"sphys_ext_seed{seed}.npy"
    if cache.exists():
        s_ext = np.load(cache)
        d_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
        log("  S_ext loaded from cache")
    else:
        s_ext, d_vis = _train_surface(stack, fp_lut, fp_flat, foot, visible, folds, (h, w), rng)
        np.save(cache, s_ext)
    crest = F.crest_mask(s_ext, foot)
    s_ext_nms = (s_ext * crest).astype(np.float32)
    s_phys = np.load(CACHE / f"sphys_seed{seed}.npy")
    crest0 = F.crest_mask(s_phys, foot)

    arms = {
        "A_std": (s_phys, ALLOC_KW),
        "B_std_nms_sep3": ((s_phys * crest0).astype(np.float32), {**ALLOC_KW, "min_sep_px": 3.0}),
        "C_ext": (s_ext, ALLOC_KW),
        "C_ext_nms_sep3": (s_ext_nms, {**ALLOC_KW, "min_sep_px": 3.0}),
    }
    terms_by_arm, details = {}, {}
    for arm in args.arms.split(","):
        surface, kw = arms[arm]
        t0 = time.time()
        allowed, tot = [], 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (d_vis > EXCL_HOLDOUT_PX)
            allowed.append(a)
            tot += int(a.sum())
        dots = np.zeros((h, w), bool)
        for a in allowed:
            b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= allocate_submodular_300m(surface, a, target_n=b, **kw)
        per, terms = {}, None
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": float(res["dti"]), "n_truth": int(res["n_truth"]),
                                 "emitted": int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        terms_by_arm[arm] = terms
        details[arm] = {"dots": int(dots.sum()), "per_fold": per,
                        "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
                        "alloc_s": round(time.time() - t0, 1), "allocator_kw": kw}
        np.save(CACHE / f"dots_{arm}_seed{seed}.npy", dots)
        log(f" [seed {seed}] {arm}: HOLDOUT-DTI={details[arm]['pooled_holdout_dti']:.6f} "
            f"dots={int(dots.sum())} ({details[arm]['alloc_s']}s)")

    pooled = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810,
                               candidate="C_ext_nms_sep3" if "C_ext_nms_sep3" in terms_by_arm
                               else args.arms.split(",")[0])
    out_path = EVID / "h82fmd_e4_extstack.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "experiment": "H82-E4-full-stack", "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION, "budget_dots": BUDGET,
        "exclusion_px": EXCL_HOLDOUT_PX, "channels": F.CHANNELS + F.EXT_NAMES, "seeds": {}}
    out["seeds"][str(seed)] = {"seed": seed, "withheld_positive_pixels": int(hidden.sum()),
                               "crest_px": int(crest.sum()), "arms": details,
                               "pooled_summary": pooled}
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"E4 seed {seed} written -> {out_path}")
    return 0


# --------------------------------------------------------------------------- E3
def cmd_e3(args):
    """Final build on the FULL catalogue: surface -> crest -> dots -> GeoTIFF.

    ``--surface ext`` (default) uses the 48-channel stack; ``phys`` uses the
    28-channel stack.  The refuted mapping-deficit weight is NOT applied: E2
    showed it costs -0.023 / -0.027 HOLDOUT-DTI on the two seeds.
    """
    from gems57 import submission_writer as SW
    from gems57 import validate as V

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    use_ext = args.surface in ("auto", "ext")
    stack = get_stack(foot, ext=use_ext)
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(foot.size, -1, np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    quad = HO.quadrant_ids(foot)
    folds = [{"name": n, "region": (quad == i) & foot} for i, n in enumerate(HO.FOLD_NAMES)]
    rng = np.random.default_rng(82_000)

    tag = "ext" if use_ext else "phys"
    sp_cache = CACHE / f"sphys_final_{tag}.npy"
    if sp_cache.exists():
        surf = np.load(sp_cache)
        d_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
        log("  final surface loaded from cache")
    else:
        surf, d_cat = _train_surface(stack, fp_lut, fp_flat, foot, cat, folds, (h, w), rng)
        np.save(sp_cache, surf)

    crest = F.crest_mask(surf, foot)
    thin = (surf * crest).astype(np.float32)
    np.save(CACHE / "surface_final.npy", thin)
    log(f"  crest pixels = {int(crest.sum())}")

    allowed, tot = [], 0
    for fold in folds:
        a = fold["region"] & foot & ~cat & (d_cat > EXCL_FINAL_PX)
        allowed.append(a)
        tot += int(a.sum())
    dots = np.zeros((h, w), bool)
    for a in allowed:
        b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
        if b > 0:
            dots |= allocate_submodular_300m(thin, a, target_n=b,
                                             **{**ALLOC_KW, "min_sep_px": 3.0})
    n_dots = int(dots.sum())
    log(f"  final dots = {n_dots}")

    pred = np.zeros((h, w), np.float32)
    pred[dots] = 1.0
    pred[~foot] = 0.0
    pred[cat] = 0.0

    name, note = args.name, args.note
    out_tif = ROOT / "docs" / "downloads" / f"{name}.tif"
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    receipt = SW.write_submission(out_tif, pred,
                                  sample=ROOT / "data" / "bridge" / "sample_submission.tif",
                                  footprint=foot, catalogue=cat, note=note, name=name,
                                  metadata={"lane": "H82-FMD",
                                            "surface": tag,
                                            "evaluator": EH.VERSION,
                                            "budget_dots": BUDGET,
                                            "exclusion_px": EXCL_FINAL_PX})
    rec = V.validate(out_tif, footprint=foot, catalogue=cat)
    (ROOT / "docs" / "downloads" / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    d_on_cat = int((dots & cat).sum())
    summary = {"experiment": "H82-E3-build", "name": name, "note": note,
               "surface": tag, "n_dots": n_dots, "dots_on_catalogue": d_on_cat,
               "min_dist_to_catalogue_px": float(d_cat[dots].min()) if n_dots else None,
               "crest_px": int(crest.sum()),
               "writer_receipt": receipt, "validator": rec,
               "exclusion_px": EXCL_FINAL_PX, "budget_dots": BUDGET,
               "allocator_kw": {**ALLOC_KW, "min_sep_px": 3.0}}
    (EVID / "h82fmd_e3_build.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log(f"E3 wrote {out_tif} ({n_dots} dots, {d_on_cat} on catalogue)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache").set_defaults(fn=cmd_cache)
    p1 = sub.add_parser("e1"); p1.add_argument("--seed", type=int, default=20); p1.set_defaults(fn=cmd_e1)
    p2 = sub.add_parser("e2")
    p2.add_argument("--seed", type=int, default=20)
    p2.add_argument("--arms", default="A_std,A_fmd,A_def,A0_random")
    p2.set_defaults(fn=cmd_e2)
    p2b = sub.add_parser("e2b")
    p2b.add_argument("--seed", type=int, default=20)
    p2b.add_argument("--arms", default="A_std,B_std_nms,B_std_nms_sep3,B_fmd_nms")
    p2b.set_defaults(fn=cmd_e2b)
    p4 = sub.add_parser("e4")
    p4.add_argument("--seed", type=int, default=20)
    p4.add_argument("--arms", default="A_std,B_std_nms_sep3,C_ext,C_ext_nms_sep3")
    p4.set_defaults(fn=cmd_e4)
    p3 = sub.add_parser("e3")
    p3.add_argument("--surface", default="auto",
                    help="auto|phys|ext : which final surface to thin and allocate")
    p3.add_argument("--name", default="gems77-h82-mapdeficit-37654dots-20261021-zeros")
    p3.add_argument("--note", default="H82 mapping-deficit allocation, 37654 dots, >2px off catalogue")
    p3.set_defaults(fn=cmd_e3)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
