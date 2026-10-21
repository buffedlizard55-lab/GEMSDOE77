#!/usr/bin/env python3
"""Experiment 2 (part b): test H77-1 / H77-2 against 95 organizer scores (live inversion).

Question: among cells more than 3 px from the H19-5 line set ("off-support"), does the
hidden-truth density rise with the straight-edge field (H77-1) or the strain-catalogue
deficit (H77-2)?  Each model is fitted by NNLS on the linearised DTI identity and judged
by leave-one-out RMSE / Spearman against the organizer scores.  Writes
evidence/e2_hypothesis_test.json.  All outputs are INSTRUMENT values, not scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems77lia import core, inversion as inv  # noqa: E402
from e2_inversion import load, cover_design  # noqa: E402

CAT3 = [0, 4, 20, 1e9]


def main():
    g = core.grid()
    valid = g.footprint & ~g.catalogue
    dcat = core.catalogue_distance_px()
    dsup = np.load(core.WORK / "d_h195.npy")
    sef = np.load(core.WORK / "f_sef.npy")
    scd = np.load(core.WORK / "f_scd.npy")
    cb = np.digitize(dcat, CAT3[1:], right=True)                 # 0,1,2
    sb = np.where(dsup == 0, 0, np.where(dsup <= 3, 1, 2))        # on, near, off
    nz = sef[valid & (sef > 0)]
    t80, t95 = np.quantile(nz, [0.8, 0.95])
    eb = np.where(sef <= 0, 0, np.where(sef <= t80, 1, np.where(sef <= t95, 2, 3)))
    qs = np.quantile(scd[valid], [0.25, 0.5, 0.75])
    db = np.digitize(scd, qs)

    def system(off_split):
        # on/near support: support x catalogue (6 bins); off-support: catalogue x split
        b = np.full(sef.shape, -1, np.int32)
        base = sb * 3 + cb
        b[(sb < 2)] = base[(sb < 2)]
        if off_split is None:
            b[sb == 2] = 6 + cb[sb == 2]
        else:
            k = int(off_split.max()) + 1
            b[sb == 2] = 6 + cb[sb == 2] * k + off_split[sb == 2]
        b[~valid] = -1
        return b

    systems = {"S0 support x catalogue": system(None),
               "S1 + H77-1 straight-edge (off-support)": system(eb),
               "S2 + H77-2 strain deficit (off-support)": system(db),
               "S3 + H77-1 x H77-2 (off-support)": system(eb * 4 + db)}
    rows = load()
    y = np.array([r["score"] for r in rows])
    Cs, S = cover_design(rows, list(systems.values()))
    res = []
    rng = np.random.default_rng(0)
    for (name, b), C in zip(systems.items(), Cs):
        N = np.bincount(b.ravel()[b.ravel() >= 0], minlength=C.shape[1]).astype(float)
        for ridge in (1e-6, 1e-3):
            h = inv.fit_nnls(C, S, N, y, c=2.0, ridge=ridge)
            lo = inv.loo(C, S, N, y, c=2.0, ridge=ridge)
            # paired bootstrap of LOO squared error for model comparison later
            r = {"model": name, "ridge": ridge, "k": int(C.shape[1]),
                 "loo_rmse": float(np.sqrt(np.mean((lo - y) ** 2))),
                 "loo_spearman": float(spearmanr(lo, y)[0]),
                 "loo_sq_err": ((lo - y) ** 2).tolist(),
                 "h": h.tolist(), "cells": N.tolist(), "K": float(N @ h)}
            res.append(r)
            print(f"{name:42s} ridge={ridge:g} k={C.shape[1]:3d} LOO RMSE={r['loo_rmse']:.4f} rho={r['loo_spearman']:+.3f} K={r['K']:.0f}")
    # paired bootstrap: S1/S2/S3 vs S0 (ridge 1e-3)
    base = np.array([r for r in res if r["model"].startswith("S0") and r["ridge"] == 1e-3][0]["loo_sq_err"])
    for r in res:
        if r["ridge"] != 1e-3 or r["model"].startswith("S0"):
            continue
        e = np.array(r["loo_sq_err"])
        d = [np.mean((base - e)[rng.integers(0, len(e), len(e))]) for _ in range(4000)]
        r["delta_mse_vs_S0_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        r["delta_mse_vs_S0_mean"] = float(np.mean(base - e))
        print(f"  {r['model']}: LOO MSE reduction vs S0 = {r['delta_mse_vs_S0_mean']:.6f} CI95 {r['delta_mse_vs_S0_ci95']}")
    (ROOT / "evidence" / "e2_hypothesis_test.json").write_text(json.dumps({
        "label": "INSTRUMENT (live inversion; not a score)", "n_rasters": len(rows),
        "sef_thresholds": {"t80": float(t80), "t95": float(t95)}, "scd_quartiles": [float(x) for x in qs],
        "results": res}, indent=1))


if __name__ == "__main__":
    main()
