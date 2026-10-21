#!/usr/bin/env python3
"""Lane H79 -- attribute-conditioned hanging-wall strand prior: canaries (E1) + fold-fair holdout (E2).

Harness is the H77D one, unchanged in every scoring-relevant respect:
  * vendor/gems57 holdout.buffered_component_draw(grid, seed, hide_frac=0.20, buffer_px=3)
  * 4 quadrant folds; the model for fold k is trained on the other three quadrants only
  * context features are computed from VISIBLE catalogue pixels only; training positives are
    produced by *sub-hiding* 20 % of the visible components inside the training quadrants
    (with the same 3 px collar) so that their context features come from OTHER traces,
    exactly as withheld components do at evaluation time
  * emission: emit77.emit_dots (greedy, min_sep 2.8) with the fold-proportional budget and
    the > 3 px visible-catalogue exclusion; scoring: evaluate_holdout.evaluate per fold,
    summed terms, pooled_summary paired 20 km block bootstrap (2,000 draws, seed 520810)

Arms
  A2_standard  : exact H77D legacy baseline (19 bands + terrain + rad, HGB_LEGACY_KW, H77D sampling)
  A2_subhide   : same static columns, sub-hide sampling (bridge between sampling schemes)
  A2_DIST      : A2_subhide + attribute-free isotropic context (ctx_dvis, ctx_dens10, ctx_dens30)
  H79_full     : A2_DIST + all attribute-conditioned context channels (candidate)
  ATTR_only    : context channels only (diagnostic)
  A0_h195_foldfair, A0_random : the H77D controls, byte-identical construction

Every number printed is HOLDOUT-DTI (gems57-pooled-hide-v2). Nothing here is a live score.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems77 import features77 as F77  # noqa: E402
from gems79 import attrs as AT  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
WORK = ROOT / ".cache" / "work" / "h79"
EXT = ROOT / ".cache" / "ext"

HGB_LEGACY_KW = dict(max_iter=180, learning_rate=0.09, max_leaf_nodes=31, min_samples_leaf=40,
                     early_stopping=False, random_state=77)
STATIC_CHANNELS = F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["rad"]
ARMS = {
    "A2_standard": ("static", "h77d"),
    "A2_subhide": ("static", "subhide"),
    "A2_DIST": ("static+iso", "subhide"),
    "H79_full": ("static+ctx", "subhide"),
    "H79_noHW": ("static+ctx-hw", "subhide"),      # ablation: drop the two signed hanging-wall channels
    "H79_noATTR": ("static+geom", "subhide"),      # ablation: isotropic + geometry only, no catalogue attributes
    "ATTR_only": ("ctx", "subhide"),
}
HW_NAMES = ["ctx_hw_attr", "ctx_hw_topo"]
GEOM_NAMES = ["ctx_dvis", "ctx_dens10", "ctx_dens30", "ctx_along", "ctx_across", "ctx_strike_sin2", "ctx_strike_cos2"]


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_attr_rasters(grid) -> dict:
    cache = WORK / "attr_tables.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        A = {k: z[k] for k in z.files if k != "receipt"}
        A["receipt"] = json.loads(str(z["receipt"]))
        return A
    with rasterio.open(RAW / "training_features.tif") as src:
        det = src.read(F77.BAND["det_elev"])
    A = AT.build_attribute_rasters(grid.footprint.shape, grid.catalogue, det, EXT / "trace_segments_utm11.csv",
                                   EXT / "qfault_attributes.csv")
    np.savez_compressed(cache, **{k: v for k, v in A.items() if k != "receipt"}, receipt=json.dumps(A["receipt"]))
    return A


def h195_support(grid) -> np.ndarray:
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    name = "h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan"
    blob = [e["blob"] for e in idx["scored"] if e["submission_name"] == name][0]
    with rasterio.open(ROOT / ".cache" / "registry" / f"{blob}.tif") as src:
        p = src.read(1).astype(np.float32)
    p = np.where(np.isfinite(p), p, 0.0)
    p[grid.catalogue] = 0
    return p > 0


def static_rows(flat_stack, rows: np.ndarray, cols: list[int]) -> np.ndarray:
    out = np.empty((rows.size, len(cols)), dtype=np.float32)
    for s in range(0, rows.size, 250_000):
        sl = rows[s:s + 250_000]
        out[s:s + 250_000] = np.asarray(flat_stack[sl][:, cols], dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--budgets", default="37654,44090")
    ap.add_argument("--arms", default="A2_standard,A2_subhide,A2_DIST,H79_full,ATTR_only")
    ap.add_argument("--subdraws", type=int, default=4)
    ap.add_argument("--candidate", default="H79_full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed = args.seed
    arms = [a for a in args.arms.split(",") if a]
    budgets = [int(b) for b in args.budgets.split(",") if b]

    grid = load_grid(RAW)
    foot = grid.footprint
    cat = grid.catalogue
    h, w = foot.shape
    folds, quad = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    visible = folds[0]["visible"]
    for f in folds[1:]:
        assert np.array_equal(f["visible"], visible), "visible mask differs between folds"
    hidden_all = np.zeros_like(foot)
    for f in folds:
        hidden_all |= f["truth"]
    n_withheld = int(hidden_all.sum())
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    log(f"seed {seed}: visible px={int(visible.sum())} withheld px={n_withheld} folds={[f['name'] for f in folds]}")

    A = load_attr_rasters(grid)
    log("attribute tables:", json.dumps(A["receipt"]))

    stack = np.load(PROC / "stack.npy", mmap_mode="r")
    flat = stack.reshape(-1, stack.shape[2])
    static_cols = [F77.IDX[c] for c in STATIC_CHANNELS]
    iso_cols = [AT.CTX_NAMES.index(n) for n in AT.ISO_NAMES]
    nohw_cols = [i for i, n in enumerate(AT.CTX_NAMES) if n not in HW_NAMES]
    geom_cols = [AT.CTX_NAMES.index(n) for n in GEOM_NAMES]
    ctx_subset = {"static+iso": iso_cols, "static+ctx-hw": nohw_cols, "static+geom": geom_cols,
                  "static+ctx": list(range(len(AT.CTX_NAMES))), "ctx": list(range(len(AT.CTX_NAMES)))}

    # ---------------- E1: canaries (hidden positives vs negatives > 3 px from the FULL catalogue)
    rng = np.random.default_rng(7900 + seed)
    cat_d3 = ndimage.binary_dilation(cat, structure=AT._disk(3))
    pos_rows = np.flatnonzero((hidden_all & foot).ravel())
    neg_pool = np.flatnonzero((foot & ~cat_d3).ravel())
    neg_rows = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
    can_rows = np.concatenate([pos_rows, neg_rows])
    y_can = np.concatenate([np.ones(pos_rows.size, np.int8), np.zeros(neg_rows.size, np.int8)])
    Xc = AT.context_features(visible, A, foot, rows=can_rows)
    canary = {}
    for j, nm in enumerate(AT.CTX_NAMES):
        v = Xc[:, j].astype(np.float64)
        ok = np.isfinite(v)
        canary[nm] = float(roc_auc_score(y_can[ok], v[ok])) if ok.sum() > 1000 and len(np.unique(y_can[ok])) == 2 else None
    Xs = static_rows(flat, can_rows, static_cols)
    for j, nm in enumerate(STATIC_CHANNELS):
        v = Xs[:, j].astype(np.float64)
        ok = np.isfinite(v)
        canary[nm] = float(roc_auc_score(y_can[ok], v[ok])) if ok.sum() > 1000 else None
    del Xc, Xs
    sym = {k: max(a, 1 - a) for k, a in canary.items() if a is not None}
    worst = max(sym, key=sym.get)
    flags = [k for k, a in sym.items() if a > 0.90]
    log(f"E1 canaries: {len(sym)} channels, max symAUC={sym[worst]:.4f} ({worst}); flags={flags}")
    ctx_sorted = sorted(((k, sym[k]) for k in AT.CTX_NAMES if k in sym), key=lambda kv: -kv[1])
    log("E1 context channel symAUC: " + ", ".join(f"{k}={v:.3f}" for k, v in ctx_sorted))
    if flags:
        raise SystemExit(f"HARD STOP -- leakage canary tripped (>0.90): {flags}")
    # sign of the hanging-wall effect among withheld pixels (MEASURED, descriptive)
    hw_stats = {}
    Xp = AT.context_features(visible, A, foot, rows=pos_rows)
    Xn = AT.context_features(visible, A, foot, rows=neg_rows)
    for nm in ("ctx_hw_attr", "ctx_hw_topo"):
        j = AT.CTX_NAMES.index(nm)
        vp, vn = Xp[:, j], Xn[:, j]
        vp, vn = vp[np.isfinite(vp)], vn[np.isfinite(vn)]
        hw_stats[nm] = {
            "withheld_frac_on_hanging_wall": float(np.mean(vp > 0)) if vp.size else None,
            "negatives_frac_on_hanging_wall": float(np.mean(vn > 0)) if vn.size else None,
            "withheld_frac_on_hw_within_30px": float(np.mean(vp[np.abs(vp) <= 30] > 0)) if (np.abs(vp) <= 30).any() else None,
            "negatives_frac_on_hw_within_30px": float(np.mean(vn[np.abs(vn) <= 30] > 0)) if (np.abs(vn) <= 30).any() else None,
            "n_withheld_defined": int(vp.size), "n_negatives_defined": int(vn.size),
        }
    del Xp, Xn
    log("E1 hanging-wall asymmetry:", json.dumps(hw_stats))

    # ---------------- E2: fold-fair OOF surfaces
    surfs: dict[str, np.ndarray] = {}
    fold_rows = []
    for k, fold in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        te_rows = np.flatnonzero((fold["region"] & foot).ravel())
        fold_rows.append((fold["name"], tr_reg, te_rows, fold["region"] & foot))

    need_static = any(ARMS[a][0].startswith("static") for a in arms)
    need_ctx = any(ARMS[a][0] != "static" for a in arms)
    for arm in arms:
        cache = WORK / f"oof_{arm}_seed{seed}.npy"
        if cache.exists():
            surfs[arm] = np.load(cache)
            log(f"loaded cached {arm}")
    todo = [a for a in arms if a not in surfs]
    if todo:
        for arm in todo:
            surfs[arm] = np.zeros((h, w), dtype=np.float32)
        for k, (fname, tr_reg, te_rows, te_mask) in enumerate(fold_rows):
            t_f = time.time()
            rng_f = np.random.default_rng(1000 * seed + k)
            # --- H77D sampling (positives = visible pixels of the training quadrants)
            if "A2_standard" in todo:
                vis_tr = visible & tr_reg
                d_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
                pos = np.flatnonzero((vis_tr & foot).ravel())
                if pos.size > 48_000:
                    pos = rng_f.choice(pos, 48_000, replace=False)
                base = foot & tr_reg & ~vis_tr & (d_tr > 3.0)
                hard = np.flatnonzero((base & (d_tr <= 16.0)).ravel())
                easy = np.flatnonzero((base & (d_tr > 16.0)).ravel())
                n_neg = int(2.5 * pos.size)
                nh = min(hard.size, int(0.55 * n_neg))
                ne = min(easy.size, n_neg - nh)
                neg = np.concatenate([rng_f.choice(hard, nh, replace=False), rng_f.choice(easy, ne, replace=False)])
                rows = np.concatenate([pos, neg])
                y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
                clf = HistGradientBoostingClassifier(**HGB_LEGACY_KW)
                clf.fit(static_rows(flat, rows, static_cols), y)
                pred = np.empty(te_rows.size, np.float32)
                for s in range(0, te_rows.size, 250_000):
                    sl = te_rows[s:s + 250_000]
                    pred[s:s + 250_000] = clf.predict_proba(static_rows(flat, sl, static_cols))[:, 1]
                surfs["A2_standard"].ravel()[te_rows] = pred
                del clf, d_tr, base
                gc.collect()
                log(f"fold {fname}: A2_standard pos={pos.size} neg={neg.size} ({time.time()-t_f:.0f}s)")
            sub_arms = [a for a in todo if ARMS[a][1] == "subhide"]
            if not sub_arms:
                continue
            # --- sub-hide sampling: positives = sub-hidden components, context from the remaining visible
            Xs_parts, Xc_parts, y_parts = [], [], []
            n_pos_tot = 0
            for s_i in range(args.subdraws):
                hid, vis_s = AT.sub_hide(visible, tr_reg, rng_f, frac=0.20, buffer_px=3)
                d_s = ndimage.distance_transform_edt(~vis_s).astype(np.float32)
                pos = np.flatnonzero((hid & foot).ravel())
                base = foot & tr_reg & ~vis_s & ~hid & (d_s > 3.0)
                hard = np.flatnonzero((base & (d_s <= 16.0)).ravel())
                easy = np.flatnonzero((base & (d_s > 16.0)).ravel())
                n_neg = int(2.5 * pos.size)
                nh = min(hard.size, int(0.55 * n_neg))
                ne = min(easy.size, n_neg - nh)
                neg = np.concatenate([rng_f.choice(hard, nh, replace=False), rng_f.choice(easy, ne, replace=False)])
                rows = np.concatenate([pos, neg])
                y_parts.append(np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)]))
                n_pos_tot += pos.size
                if need_ctx:
                    Xc_parts.append(AT.context_features(vis_s, A, foot, rows=rows))
                if need_static:
                    Xs_parts.append(static_rows(flat, rows, static_cols))
                del d_s, base, hid, vis_s
            y = np.concatenate(y_parts)
            Xs_tr = np.vstack(Xs_parts) if need_static else None
            Xc_tr = np.vstack(Xc_parts) if need_ctx else None
            del Xs_parts, Xc_parts
            gc.collect()
            # test features: context from the fold's visible mask (hidden + collar excluded)
            Xc_te = AT.context_features(visible, A, foot, rows=te_rows) if need_ctx else None
            for arm in sub_arms:
                kind = ARMS[arm][0]
                if kind == "static":
                    xtr = Xs_tr
                elif kind == "ctx":
                    xtr = Xc_tr
                else:
                    xtr = np.hstack([Xs_tr, Xc_tr[:, ctx_subset[kind]]])
                clf = HistGradientBoostingClassifier(**HGB_LEGACY_KW)
                clf.fit(xtr, y)
                pred = np.empty(te_rows.size, np.float32)
                for s in range(0, te_rows.size, 250_000):
                    sl = slice(s, s + 250_000)
                    parts = []
                    if kind.startswith("static"):
                        parts.append(static_rows(flat, te_rows[sl], static_cols))
                    if kind != "static":
                        parts.append(Xc_te[sl][:, ctx_subset[kind]])
                    pred[sl] = clf.predict_proba(np.hstack(parts) if len(parts) > 1 else parts[0])[:, 1]
                surfs[arm].ravel()[te_rows] = pred
                del clf, xtr
                gc.collect()
            log(f"fold {fname}: sub-hide arms {sub_arms} pos={n_pos_tot} rows={y.size} ({time.time()-t_f:.0f}s)")
            del Xs_tr, Xc_tr, Xc_te, y
            gc.collect()
        for arm in todo:
            np.save(WORK / f"oof_{arm}_seed{seed}.npy", surfs[arm])

    # ---------------- emission + scoring (identical to H77D)
    def emit_greedy_pooled(p_surf: np.ndarray, budget: int, excl_px: float = 3.0) -> np.ndarray:
        allowed, tot = [], 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > excl_px)
            allowed.append(a)
            tot += int(a.sum())
        dots = np.zeros((h, w), dtype=bool)
        for a in allowed:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= emit77.emit_dots(p_surf, a, b)
        return dots

    def score_dots(dots: np.ndarray):
        terms, per = None, {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": float(res["dti"]), "n_truth": int(res["n_truth"]), "emitted": int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        return per, terms

    dsup = h195_support(grid)
    from gems77lia import allocate as LIA_AL  # noqa: E402  (shared greedy Poisson-disk selector)
    out_budgets = {}
    for B in budgets:
        terms_by_arm, details = {}, {}
        for arm in arms:
            d = emit_greedy_pooled(surfs[arm], B, 3.0)
            per, t = score_dots(d)
            terms_by_arm[arm] = t
            details[arm] = {"dots": int(d.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0)))}
            log(f"[seed {seed} B={B}] {arm:16s} HOLDOUT-DTI={details[arm]['pooled_holdout_dti']:.5f} dots={int(d.sum())}")
            if arm == args.candidate:
                np.save(WORK / f"dots_{arm}_seed{seed}_B{B}.npy", np.flatnonzero(d.ravel()).astype(np.int32))
        rk = np.random.default_rng(7).random(h * w)
        d_a0 = np.zeros((h, w), dtype=bool)
        for fold in folds:
            valid_f = fold["region"] & foot & ~fold["masked_known"]
            ac = np.flatnonzero((dsup & valid_f & (dist_vis > 2.0)).ravel())
            sel = LIA_AL.greedy_select(ac[np.argsort(rk[ac])], 3.0)
            d_a0.ravel()[sel] = True
        per, t = score_dots(d_a0)
        terms_by_arm["A0_h195_foldfair"] = t
        details["A0_h195_foldfair"] = {"dots": int(d_a0.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0)))}
        log(f"[seed {seed} B={B}] {'A0_h195_foldfair':16s} HOLDOUT-DTI={details['A0_h195_foldfair']['pooled_holdout_dti']:.5f} dots={int(d_a0.sum())}")
        r_surf = np.random.default_rng(999 + seed).random((h, w), dtype=np.float32) * foot
        d_r = emit_greedy_pooled(r_surf, B, 3.0)
        per, t = score_dots(d_r)
        terms_by_arm["A0_random"] = t
        details["A0_random"] = {"dots": int(d_r.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0)))}
        log(f"[seed {seed} B={B}] {'A0_random':16s} HOLDOUT-DTI={details['A0_random']['pooled_holdout_dti']:.5f} dots={int(d_r.sum())}")
        pooled = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810, candidate=args.candidate)
        for ctrl, drec in pooled["paired_differences"].items():
            log(f"  -> {args.candidate} vs {ctrl:16s}: delta={drec['delta']:+.5f} CI95=[{drec['ci95'][0]:+.5f}, {drec['ci95'][1]:+.5f}]")
        out_budgets[f"B{B}"] = {"arms": details, "pooled_summary": pooled}

    out = {
        "lane": "H79-ATTR-HANGINGWALL",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "holdout": {"draw": "buffered_component_draw", "hide_frac": 0.20, "buffer_px": 3, "seed": seed,
                    "withheld_positive_pixels": n_withheld, "fold_receipts": [f["receipt"] for f in folds]},
        "sampling": {"subdraws": args.subdraws, "neg_ratio": 2.5, "hard_frac": 0.55, "hard_px": 16, "excl_px": 3},
        "attribute_receipt": A["receipt"],
        "canary_auc": canary, "canary_max_symmetric_auc": sym[worst], "canary_max_channel": worst, "leakage_flags": flags,
        "hanging_wall_asymmetry": hw_stats,
        "static_channels": STATIC_CHANNELS, "context_channels": AT.CTX_NAMES, "iso_channels": AT.ISO_NAMES,
        "hgb_kwargs": HGB_LEGACY_KW,
        "budgets": out_budgets,
        "elapsed_seconds": round(time.time() - t0, 1),
        "inputs_sha256": {"stack.npy": None, "trace_segments_utm11.csv": sha256_file(EXT / "trace_segments_utm11.csv"),
                          "qfault_attributes.csv": sha256_file(EXT / "qfault_attributes.csv")},
    }
    out_path = Path(args.out) if args.out else ROOT / "evidence" / f"h79_holdout_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True))
    log(f"wrote {out_path} ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
