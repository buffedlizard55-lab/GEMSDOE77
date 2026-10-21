#!/usr/bin/env python3
"""Experiment 3 (part b): hide-and-recover HOLDOUT-DTI + leakage canary, template tools only.

Holdout: gems57.holdout.buffered_component_draw (whole 8-connected catalogue components
withheld per quadrant, 3 px feature buffer, visible faults masked pixel-exactly), scored by
gems57.evaluate_holdout (pooled DTI, alpha 0.2, beta 0.8, 300 m triangular kernel, paired
20 km spatial-block bootstrap).  Every catalogue-derived input (catalogue distance and the
bins that depend on it) is recomputed from the VISIBLE catalogue of the draw.

Arms (same holdout, same blocks):
  C    candidate rule (pre-registered E3 rule, B and N_on as selected)
  A0   champion rule rebuilt fold-fair (H19-5 line cells, >2 px from visible catalogue,
       3 px greedy thinning in a fixed random order) - the "current best" construction
  E1   H77-1 alone (straight-edge top cells, same dot count as C)
  R    random dots, same count as C, 3 px apart
Writes evidence/e3_holdout.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import evaluate_holdout as eh  # noqa: E402  (vendored, byte-identical)
from gems57.grid import load_grid  # noqa: E402
from gems57.holdout import buffered_component_draw  # noqa: E402
from gems77 import allocate as al, core  # noqa: E402
from e3_build_candidate import bins_like_e2  # noqa: E402

SEED = 20


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    grid = load_grid(core.DATA)
    g = core.grid()
    assert np.array_equal(grid.footprint, g.footprint) and np.array_equal(grid.catalogue, g.catalogue)
    folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    sel = json.loads((ROOT / "evidence" / "e3_allocation.json").read_text())["selected"]
    ev = json.loads((ROOT / "evidence" / "e2_hypothesis_test.json").read_text())
    t80, t95 = ev["sef_thresholds"]["t80"], ev["sef_thresholds"]["t95"]
    res = {r["model"][:2]: r for r in ev["results"] if r["ridge"] == 1e-3}
    dsup = np.load(core.WORK / "d_h195.npy")
    sef = np.load(core.WORK / "f_sef.npy")
    eb = np.where(sef <= 0, 0, np.where(sef <= t80, 1, np.where(sef <= t95, 2, 3)))
    rng = np.random.default_rng(7)
    rand_key = rng.random(core.H * core.W)

    terms = {k: [] for k in ("C", "A0", "E1", "R")}
    receipts = []
    canary = {k: {"pos": [], "neg": []} for k in ("h19_5_support_proximity", "straight_edge_H77_1",
                                                  "visible_catalogue_proximity", "S0_credit", "S1_credit")}
    for f in folds:
        visible = f["visible"]
        masked = f["masked_known"]
        valid = g.footprint & ~masked                      # never place on visible catalogue
        dvis = ndimage.distance_transform_edt(~visible).astype(np.float32)
        hm0 = al.hmap_from_bins(bins_like_e2(sef, dvis, dsup, valid, t80, t95, None), res["S0"]["h"])
        hm1 = al.hmap_from_bins(bins_like_e2(sef, dvis, dsup, valid, t80, t95, eb), res["S1"]["h"])
        cr0, cr1 = al.credit_map(hm0), al.credit_map(hm1)
        L = (dsup == 0) & valid
        # --- C: pre-registered candidate rule (B, frac from the selected variant)
        on_c = np.flatnonzero((L & (dvis > sel["B_px"])).ravel())
        on_all = al.greedy_select(on_c[np.lexsort((-sef.ravel()[on_c], -cr0.ravel()[on_c]))], 3.0)
        n_on = int(len(on_all) * sel["on_frac_of_greedy"])
        on_sel = on_all[:n_on]
        n_off = int(np.ceil(n_on * 0.35 / 0.65))
        off_c = np.flatnonzero((valid & (dsup > 3) & (sef > t80) & (dvis > 4)).ravel())
        off_sel = al.greedy_select(off_c[np.lexsort((-sef.ravel()[off_c], -cr1.ravel()[off_c]))], 3.0,
                                   taken=on_sel, limit=n_off)
        C = np.zeros(core.H * core.W, np.float32); C[on_sel] = 1; C[off_sel] = 1
        n_c = int(C.sum())
        # --- A0: champion construction rebuilt fold-fair
        a_c = np.flatnonzero((L & (dvis > 2)).ravel())
        A0 = np.zeros(core.H * core.W, np.float32)
        A0[al.greedy_select(a_c[np.argsort(rand_key[a_c])], 3.0)] = 1
        # --- E1: H77-1 alone, same count as C
        e_c = np.flatnonzero((valid & (sef > 0) & (dvis > 4)).ravel())
        E1 = np.zeros(core.H * core.W, np.float32)
        E1[al.greedy_select(e_c[np.argsort(-sef.ravel()[e_c])], 3.0, limit=n_c)] = 1
        # --- R: random, same count
        r_c = np.flatnonzero(valid.ravel())
        R = np.zeros(core.H * core.W, np.float32)
        R[al.greedy_select(r_c[np.argsort(rand_key[r_c])][: n_c * 3], 3.0, limit=n_c)] = 1
        for k, arr in (("C", C), ("A0", A0), ("E1", E1), ("R", R)):
            r, t = eh.evaluate(arr.reshape(core.H, core.W), f, g.footprint)
            terms[k].append(t)
        # --- leakage canary inputs (truth vs other active cells in the fold region)
        region = f["region"] & g.footprint & ~masked
        truth = f["truth"] & region
        negm = region & ~truth
        neg_idx = np.flatnonzero(negm.ravel())
        neg_idx = neg_idx[rng.choice(len(neg_idx), size=min(300_000, len(neg_idx)), replace=False)]
        pos_idx = np.flatnonzero(truth.ravel())
        feats = {"h19_5_support_proximity": -dsup, "straight_edge_H77_1": sef,
                 "visible_catalogue_proximity": -dvis, "S0_credit": cr0, "S1_credit": cr1}
        for k, a in feats.items():
            canary[k]["pos"].append(a.ravel()[pos_idx]); canary[k]["neg"].append(a.ravel()[neg_idx])
        receipts.append({**f["receipt"], "candidate_dots": n_c, "A0_dots": int(A0.sum())})
        print(f"fold {f['name']}: truth={int(truth.sum())} C={n_c} A0={int(A0.sum())}", flush=True)

    summ = {}
    for k in ("A0", "E1", "R"):
        tb = {"C": np.sum(terms["C"], axis=0), k: np.sum(terms[k], axis=0)}
        summ[k] = eh.pooled_summary(tb, draws=2000, candidate="C")
    can = {}
    for k, v in canary.items():
        a = auc(np.concatenate(v["pos"]), np.concatenate(v["neg"]))
        can[k] = {"auc": a, "leakage_flag": bool(a > 0.90)}
    out = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": eh.VERSION,
           "split": "gems57.holdout.buffered_component_draw seed=20 hide_frac=0.20 buffer_px=3 boundary_px=12",
           "fold_receipts": receipts, "pooled": summ, "leakage_canary": can,
           "implementation_sha256": eh.implementation_hashes()}
    (ROOT / "evidence" / "e3_holdout.json").write_text(json.dumps(out, indent=1, default=float))
    c = summ["A0"]["scores"]["C"]
    print("C  HOLDOUT-DTI", round(c["dti"], 4), [round(x, 4) for x in c["ci95"]], "withheld", c["withheld_positive_pixels"])
    for k in ("A0", "E1", "R"):
        s = summ[k]["scores"][k]; d = summ[k]["paired_differences"][k]
        print(f"{k:3s} HOLDOUT-DTI {s['dti']:.4f} {[round(x,4) for x in s['ci95']]}  C-{k} = {d['delta']:+.4f} CI {[round(x,4) for x in d['ci95']]}")
    print("canary", {k: round(v["auc"], 3) for k, v in can.items()})


if __name__ == "__main__":
    main()
