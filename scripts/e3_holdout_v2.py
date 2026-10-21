#!/usr/bin/env python3
"""E3 holdout of the FINAL (uniqueness-repaired) rule: fold-fair, template tools only.

Identical to scripts/e3_holdout.py except that arm C is rebuilt per fold with the per-raster
overlap budget (gems77lia.allocate.budgeted_selection, budget 0.66 vs every informative registry
raster).  Registry membership does not depend on the fold, so it is computed once for a
catalogue-distance-free pool; every catalogue-derived input (catalogue distance, the S0/S1
bins, the B-px prune) is recomputed from the VISIBLE catalogue of each fold.
Writes evidence/e3_holdout_final.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, spatial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import evaluate_holdout as eh  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems57.holdout import buffered_component_draw  # noqa: E402
from gems77lia import allocate as al, core  # noqa: E402
from e3_build_candidate import bins_like_e2  # noqa: E402
from e3_holdout import auc  # noqa: E402

SEED = 20


def main():
    grid = load_grid(core.DATA)
    g = core.grid()
    folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    rep = json.loads((ROOT / "evidence" / "e3_unique_repair.json").read_text())
    ev = json.loads((ROOT / "evidence" / "e2_hypothesis_test.json").read_text())
    t80, t95 = ev["sef_thresholds"]["t80"], ev["sef_thresholds"]["t95"]
    res = {r["model"][:2]: r for r in ev["results"] if r["ridge"] == 1e-3}
    dsup = np.load(core.WORK / "d_h195.npy")
    sef = np.load(core.WORK / "f_sef.npy")
    eb = np.where(sef <= 0, 0, np.where(sef <= t80, 1, np.where(sef <= t95, 2, 3)))
    valid_all = g.footprint & ~g.catalogue
    # dcat-free pool (superset of every fold's pool) + registry membership, once
    pool_on = np.flatnonzero(((dsup == 0) & g.footprint).ravel())
    pool_off = np.flatnonzero((g.footprint & (dsup > 3) & (sef > 0)).ravel())
    pool = np.concatenate([pool_on, pool_off])
    pos = np.full(core.H * core.W, -1, np.int64)
    pos[pool] = np.arange(len(pool))
    pr, pc = np.divmod(pool, core.W)
    free = json.loads((ROOT / "evidence" / "e3_registry_free.json").read_text())
    excluded = {x["raster"] for x in free["space_filling_rasters"]} | {x["raster"] for x in free["unreadable_rasters"]}
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    cols = []
    for r in idx["rasters"]:
        if r["aliases"][0] in excluded:
            continue
        try:
            p = core.read_prediction(core.REG / f"{r['blob']}.tif") > 0
        except Exception:
            continue
        rr, cc = np.nonzero(p)
        if len(rr) == 0:
            continue
        d, _ = spatial.cKDTree(np.c_[rr, cc]).query(np.c_[pr, pc], k=1, distance_upper_bound=3.0 + 1e-9)
        cols.append(np.packbits(d <= 3.0))
    Mbits = np.stack(cols, axis=1)
    print("membership:", Mbits.shape, flush=True)

    def member(f):
        i = int(pos[f])
        return ((Mbits[i >> 3] >> (7 - (i & 7))) & 1).astype(np.int64)

    rng = np.random.default_rng(7)
    rand_key = rng.random(core.H * core.W)
    terms = {k: [] for k in ("C", "A0", "E1", "R")}
    receipts = []
    canary = {k: {"pos": [], "neg": []} for k in ("h19_5_support_proximity", "straight_edge_H77_1",
                                                  "visible_catalogue_proximity", "S0_credit", "S1_credit")}
    for f in folds:
        masked = f["masked_known"]
        valid = g.footprint & ~masked
        dvis = ndimage.distance_transform_edt(~f["visible"]).astype(np.float32)
        hm0 = al.hmap_from_bins(bins_like_e2(sef, dvis, dsup, valid, t80, t95, None), res["S0"]["h"])
        hm1 = al.hmap_from_bins(bins_like_e2(sef, dvis, dsup, valid, t80, t95, eb), res["S1"]["h"])
        cr0, cr1 = al.credit_map(hm0), al.credit_map(hm1)
        vflat, dflat = valid.ravel(), dvis.ravel()
        on_o = pool_on[vflat[pool_on] & (dflat[pool_on] > rep["B_px"])]
        on_o = on_o[np.lexsort((-sef.ravel()[on_o], -cr0.ravel()[on_o]))]
        off_o = pool_off[vflat[pool_off] & (dflat[pool_off] > 4)]
        off_o = off_o[np.lexsort((-sef.ravel()[off_o], -cr1.ravel()[off_o]))]
        n_on = int(len(al.greedy_select(on_o, 3.0)) * rep["on_frac_of_greedy"])
        on_sel, off_sel, o = al.budgeted_selection(on_o, off_o, member, n_on, budget=rep["budget"])
        C = np.zeros(core.H * core.W, np.float32); C[on_sel] = 1; C[off_sel] = 1
        n_c = int(C.sum())
        L = (dsup == 0) & valid
        a_c = np.flatnonzero((L & (dvis > 2)).ravel())
        A0 = np.zeros(core.H * core.W, np.float32); A0[al.greedy_select(a_c[np.argsort(rand_key[a_c])], 3.0)] = 1
        e_c = np.flatnonzero((valid & (sef > 0) & (dvis > 4)).ravel())
        E1 = np.zeros(core.H * core.W, np.float32); E1[al.greedy_select(e_c[np.argsort(-sef.ravel()[e_c])], 3.0, limit=n_c)] = 1
        r_c = np.flatnonzero(vflat)
        R = np.zeros(core.H * core.W, np.float32)
        R[al.greedy_select(r_c[np.argsort(rand_key[r_c])][: n_c * 3], 3.0, limit=n_c)] = 1
        for k, arr in (("C", C), ("A0", A0), ("E1", E1), ("R", R)):
            _, t = eh.evaluate(arr.reshape(core.H, core.W), f, g.footprint)
            terms[k].append(t)
        region = f["region"] & g.footprint & ~masked
        truth = f["truth"] & region
        neg_idx = np.flatnonzero((region & ~truth).ravel())
        neg_idx = neg_idx[rng.choice(len(neg_idx), size=min(300_000, len(neg_idx)), replace=False)]
        pos_idx = np.flatnonzero(truth.ravel())
        feats = {"h19_5_support_proximity": -dsup, "straight_edge_H77_1": sef,
                 "visible_catalogue_proximity": -dvis, "S0_credit": cr0, "S1_credit": cr1}
        for k, a in feats.items():
            canary[k]["pos"].append(a.ravel()[pos_idx]); canary[k]["neg"].append(a.ravel()[neg_idx])
        receipts.append({**f["receipt"], "candidate_dots": n_c, "candidate_on": int(len(on_sel)),
                         "candidate_off": int(len(off_sel)), "worst_registry_overlap": float((o / n_c).max()),
                         "A0_dots": int(A0.sum())})
        print(f"fold {f['name']}: truth={int(truth.sum())} C={n_c} (on {len(on_sel)}, off {len(off_sel)}, "
              f"worst overlap {(o / n_c).max():.3f}) A0={int(A0.sum())}", flush=True)
    summ = {}
    for k in ("A0", "E1", "R"):
        summ[k] = eh.pooled_summary({"C": np.sum(terms["C"], axis=0), k: np.sum(terms[k], axis=0)},
                                    draws=2000, candidate="C")
    can = {k: {"auc": auc(np.concatenate(v["pos"]), np.concatenate(v["neg"]))} for k, v in canary.items()}
    for v in can.values():
        v["leakage_flag"] = bool(v["auc"] > 0.90)
    out = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": eh.VERSION,
           "rule": "final uniqueness-repaired rule (budget 0.66 vs 765 informative registry rasters)",
           "split": "gems57.holdout.buffered_component_draw seed=20 hide_frac=0.20 buffer_px=3 boundary_px=12",
           "fold_receipts": receipts, "pooled": summ, "leakage_canary": can,
           "implementation_sha256": eh.implementation_hashes()}
    (ROOT / "evidence" / "e3_holdout_final.json").write_text(json.dumps(out, indent=1, default=float))
    c = summ["A0"]["scores"]["C"]
    print("C  HOLDOUT-DTI", round(c["dti"], 4), [round(x, 4) for x in c["ci95"]], "withheld", c["withheld_positive_pixels"])
    for k in ("A0", "E1", "R"):
        s = summ[k]["scores"][k]; d = summ[k]["paired_differences"][k]
        print(f"{k:3s} HOLDOUT-DTI {s['dti']:.4f} {[round(x, 4) for x in s['ci95']]}  C-{k} = {d['delta']:+.4f} CI {[round(x, 4) for x in d['ci95']]}")
    print("canary", {k: round(v["auc"], 3) for k, v in can.items()})


if __name__ == "__main__":
    main()
