#!/usr/bin/env python3
"""GEMSDOE77 Exp 1: leakage canaries + whole-component quadrant hide-and-recover.

Evaluator: vendor gems57.evaluate_holdout (gems57-pooled-hide-v2), alpha=0.2,
beta=0.8, 300 m triangular kernel. Every score written here is HOLDOUT-DTI
with withheld-positive counts and paired 95% CIs from pooled_summary.
Arms (preregistered): A0 random, A2 standard-block HGB, A3 new-physics HGB,
A4 combined (primary, no lidar), A5 combined+lidar (ablation), single-feature
physics arms, REF archived H33 dots (reference only).
"""
from __future__ import annotations
import argparse, gc, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems57 import metric as M  # noqa: E402
from gems77 import features77 as F, emit77  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SEED = 30
HGB_KW = dict(max_iter=180, learning_rate=0.09, max_leaf_nodes=31,
              min_samples_leaf=40, early_stopping=False, random_state=77)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def arm_channels(name):
    std = F.BLOCKS["raw"] + F.BLOCKS["terrain"] + F.BLOCKS["rad"]
    return {
        "A2_standard": std,
        "A3_newphysics": list(F.BLOCKS["new"]),
        "A4_primary": std + list(F.BLOCKS["new"]),
        "A5_primary_lidar": std + list(F.BLOCKS["new"]) + list(F.BLOCKS["lidar"]),
    }[name]


def train_predict(stack, cols, train_idx, y, region, logtag):
    flat = stack.reshape(-1, stack.shape[2])
    X = np.asarray(flat[train_idx][:, cols], dtype=np.float32)
    clf = HistGradientBoostingClassifier(**HGB_KW)
    t = time.time(); clf.fit(X, y); log(f"{logtag}: fit {time.time()-t:.0f}s ({X.shape[0]} rows x {X.shape[1]} ch)")
    del X; gc.collect()
    out = np.zeros(stack.shape[:2], np.float32)
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return out
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = np.asarray(stack[r0:r1, c0:c1][:, :, cols], dtype=np.float32)
    pred = np.zeros(sub.shape[:2], np.float32)
    step = 200
    for i in range(0, sub.shape[0], step):
        blk = sub[i:i + step].reshape(-1, sub.shape[2])
        pred[i:i + step] = clf.predict_proba(blk)[:, 1].reshape(min(step, sub.shape[0] - i), sub.shape[1])
    m = region[r0:r1, c0:c1]
    out[r0:r1, c0:c1] = np.where(m, pred, 0.0)
    del sub, pred, clf; gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A2_standard,A3_newphysics,A4_primary,A5_primary_lidar")
    ap.add_argument("--skip-scan", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    grid = G.load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    folds, quad = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]; visible = folds[0]["visible"]
    receipts = [f["receipt"] for f in folds]
    log(f"holdout: withheld positives={int(hidden.sum())} visible={int(visible.sum())} catalogue={int(cat.sum())}")
    for r in receipts:
        log(" fold receipt:", {k: r[k] for k in ("fold", "held_components", "withheld_positive_pixels")})

    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    flat = stack.reshape(-1, stack.shape[2])
    rng = np.random.default_rng(1234)

    # ---------- leakage canary: every channel alone ----------
    pos_idx = np.flatnonzero(hidden)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    neg_pool = np.flatnonzero(foot & ~cat & (dist_cat > 3))
    neg_idx = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
    canary_idx = np.concatenate([pos_idx, neg_idx])
    y_can = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])
    Xc = np.asarray(flat[canary_idx], dtype=np.float32)
    canaries = {}
    for j, name in enumerate(F.CHANNELS):
        v = Xc[:, j]; m = np.isfinite(v)
        auc = float(roc_auc_score(y_can[m], np.nan_to_num(v[m]))) if m.sum() > 100 and len(np.unique(y_can[m])) == 2 else None
        canaries[name] = auc
    worst = max((abs(a - 0.5), n) for n, a in canaries.items() if a is not None)
    new_worst = max((abs(canaries[n] - 0.5), n) for n in F.BLOCKS["new"] if canaries[n] is not None)
    log(f"canary worst overall: {worst[1]} symAUC={0.5+worst[0]:.4f}; worst NEW: {new_worst[1]} symAUC={0.5+new_worst[0]:.4f}")
    leakage = [n for n, a in canaries.items() if a is not None and max(a, 1 - a) > 0.90]
    del Xc; gc.collect()

    results = {
        "evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA, "beta": M.BETA, "kernel_radius_m": M.R_M,
        "holdout_seed": SEED, "withheld_positive_pixels_total_draw": int(hidden.sum()),
        "fold_receipts": receipts, "canary_auc": canaries,
        "canary_worst_new_symmetric_auc": 0.5 + float(new_worst[0]),
        "leakage_channels_gt_0p90_symmetric": leakage,
        "arms": {}, "runtime_note": "CPU sandbox 2 vCPU",
    }
    if leakage:
        results["verdict"] = "STOP-LEAKAGE"
        (ROOT / "evidence" / "holdout_exp1.json").write_text(json.dumps(results, indent=2) + "\n")
        log("LEAKAGE CANARY FAILED:", leakage); return

    # ---------- per-fold training samples (shared across arms) ----------
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
        neg = np.concatenate([rng.choice(hard, nh, replace=False) if nh else np.array([], int),
                              rng.choice(easy, ne, replace=False) if ne else np.array([], int)])
        idx = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        fold_samples.append((idx, y))
        log(f"fold {fold['name']}: train pos={pos.size} neg={neg.size} test-region px={int(fold['region'].sum())} truth={int(fold['truth'].sum())}")

    # ---------- OOF surfaces ----------
    arms = args.arms.split(",")
    oof = {}
    for arm in arms:
        path = PROC / f"oof_{arm}.npy"
        if path.exists():
            oof[arm] = np.load(path); log(f"{arm}: loaded cached OOF"); continue
        cols = [F.IDX[n] for n in arm_channels(arm)]
        surf = np.zeros(foot.shape, np.float32)
        for k, fold in enumerate(folds):
            idx, y = fold_samples[k]
            surf += train_predict(stack, cols, idx, y, fold["region"], f"{arm}/{fold['name']}")
        np.save(path, surf); oof[arm] = surf
        log(f"{arm}: OOF done ({time.time()-t0:.0f}s)")

    # ---------- emission helpers ----------
    def emit_pooled(p_surface, budget, excl, seed_offset=0):
        allowed_folds = []
        tot = 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > excl)
            allowed_folds.append(a); tot += int(a.sum())
        dots = np.zeros(foot.shape, bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= emit77.emit_dots(p_surface, a, b)
        return dots

    def score_dots(dots):
        terms = None; per = {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": res["dti"], "n_truth": res["n_truth"], "emitted": res["n_emitted"]}
            terms = t if terms is None else terms + t
        return per, terms

    # ---------- operating-point scan on primary arm ----------
    primary = "A4_primary" if "A4_primary" in oof else arms[0]
    scan = {}
    best = None
    if not args.skip_scan:
        for B in (20_000, 30_000, 37_654, 44_090):
            for E in (0, 1, 2, 3):
                dots = emit_pooled(oof[primary], B, E)
                per, terms = score_dots(dots)
                pooled = float(EH.from_terms(terms.sum(axis=0)))
                scan[f"B{B}_E{E}"] = {"budget": B, "exclusion_px": E, "dots": int(dots.sum()),
                                      "pooled_holdout_dti": pooled, "per_fold": per}
                log(f"scan B={B} E={E}: pooled HOLDOUT-DTI={pooled:.4f} dots={int(dots.sum())}")
                key = (pooled, -B, -E)
                if best is None or key > best[0]:
                    best = (key, B, E)
        selB, selE = best[1], best[2]
    else:
        selB, selE = 37_654, 2
    results["scan_primary_arm"] = scan
    results["selected_operating_point"] = {"budget": selB, "exclusion_px": selE}
    log(f"selected operating point: B={selB} E={selE}")

    # ---------- final arm comparison at selected cell ----------
    terms_by_arm = {}
    dot_masks = {}
    for arm in arms:
        dots = emit_pooled(oof[arm], selB, selE)
        dot_masks[arm] = dots
        per, terms = score_dots(dots)
        terms_by_arm[arm] = terms
        results["arms"][arm] = {"dots": int(dots.sum()), "per_fold": per,
                                "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        np.save(PROC / f"dots_holdout_{arm}.npy", dots)
        log(f"{arm}: pooled HOLDOUT-DTI={results['arms'][arm]['pooled_holdout_dti']:.4f}")
    # random control
    rng2 = np.random.default_rng(999)
    rand_surf = rng2.random(foot.shape, dtype=np.float32) * foot
    dots_r = emit_pooled(rand_surf, selB, selE)
    per, terms = score_dots(dots_r); terms_by_arm["A0_random"] = terms
    results["arms"]["A0_random"] = {"dots": int(dots_r.sum()), "per_fold": per,
                                    "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
    log(f"A0_random: pooled HOLDOUT-DTI={results['arms']['A0_random']['pooled_holdout_dti']:.4f}")
    del rand_surf
    # single-feature physics arms at selected cell
    for fname, invert in (("contact_confluence", False), ("pc_mag", False),
                          ("frontness", False), ("mag_as_log", False)):
        ch = np.asarray(stack[:, :, F.IDX[fname]], np.float32)
        p = F.rank01(np.nan_to_num(ch, nan=-1e30), foot)
        if invert:
            p = 1.0 - p
        p = np.where(foot, p, 0).astype(np.float32)
        dots = emit_pooled(p, selB, selE)
        per, terms = score_dots(dots); terms_by_arm[f"S_{fname}"] = terms
        results["arms"][f"S_{fname}"] = {"dots": int(dots.sum()), "per_fold": per,
                                         "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        log(f"S_{fname}: pooled HOLDOUT-DTI={results['arms'][f'S_{fname}']['pooled_holdout_dti']:.4f}")
        del ch, p; gc.collect()
    # archived H33 reference (static; full-catalogue pruning caveat)
    h33p = ROOT / "data" / "registry_rasters" / "gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif"
    if h33p.exists():
        with rasterio.open(h33p) as s:
            h33 = s.read(1)
        h33 = np.where(np.isfinite(h33), h33, 0).astype(np.float32)
        per, terms = score_dots(h33 > 0)
        terms_by_arm["REF_H33_static"] = terms
        results["arms"]["REF_H33_static"] = {"dots": int((h33 > 0).sum()), "per_fold": per,
            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
            "caveat": "archived dots pruned with FULL catalogue incl. withheld segments; reference only"}
        log(f"REF_H33_static: pooled HOLDOUT-DTI={results['arms']['REF_H33_static']['pooled_holdout_dti']:.4f}")

    summary = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate=primary)
    results["pooled_summary"] = summary
    # SGMC off-catalogue proxy (secondary, NOT holdout)
    with rasterio.open(RAW / "external" / "derived_sgmc_faults_100m_u8.tif") as s:
        sgmc = s.read(1) > 0
    sgmc_off = sgmc & ~cat & (dist_cat > 3) & foot
    results["sgmc_offcatalogue_proxy"] = {"n_truth_px": int(sgmc_off.sum()), "note": "PROXY-DTI vs SGMC\\catalogue (d>3px); not HOLDOUT-DTI, not an organizer score"}
    for arm, dots in dot_masks.items():
        r = M.dti_binary(dots, sgmc_off, valid=foot, known=cat)
        results["sgmc_offcatalogue_proxy"][arm] = {"dti": r["dti"], "coverage": r["coverage"]}
    results["elapsed_s"] = round(time.time() - t0, 1)
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "holdout_exp1.json").write_text(json.dumps(results, indent=2) + "\n")
    log(f"DONE {results['elapsed_s']}s -> evidence/holdout_exp1.json")
    for name, s in summary["scores"].items():
        log(f"  {name}: HOLDOUT-DTI {s['dti']:.4f} CI95 {s['ci95']} (withheld {s['withheld_positive_pixels']})")


if __name__ == "__main__":
    main()
