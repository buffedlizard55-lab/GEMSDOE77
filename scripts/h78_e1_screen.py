#!/usr/bin/env python3
"""H78 Experiment 1: live-inversion screen of candidate OFF-support fields.

Rule frozen in registry/preregistration_h78.json (committed before this script ran).

Base model S0 (LIA lane): support {on (0 px), near (<=3 px), off (>3 px)} from the H19-5 line set
x catalogue distance {0-4, 4-20, >20 px}.  Each candidate field replaces the three off-support
bins by off-support x catalogue x field level {top 3 %, 3-15 %, rest}, quantiles taken over the
valid off-support cells.  The region hypothesis (H78-3) replaces them by 9 region blocks.
NNLS ridge 1e-3, c = 2, leave-one-out over the 95 distinct organizer-scored rasters, paired
bootstrap (4000 draws) of the LOO-MSE reduction against S0.

Every output is an INSTRUMENT value (a projection from organizer scores), never a score.
Writes evidence/h78_e1_screen.json and .cache/work/h78_e1_design.npz.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems77lia import core, inversion as inv  # noqa: E402
from gems78 import fields as F  # noqa: E402
from e2_inversion import load, cover_design  # noqa: E402

CAT3 = [0, 4, 20, 1e9]
LEVEL_Q = (0.85, 0.97)       # level 2 = top 3 %, level 1 = 3-15 %, level 0 = rest
RIDGE = 1e-3
DRAWS = 4000


def base_bins(dcat, dsup, valid):
    cb = np.digitize(dcat, CAT3[1:], right=True)                    # 0, 1, 2
    sb = np.where(dsup == 0, 0, np.where(dsup <= 3, 1, 2))          # on, near, off
    return cb.astype(np.int32), sb.astype(np.int32)


def system_with_split(cb, sb, valid, split, k):
    b = np.full(cb.shape, -1, np.int32)
    near = sb < 2
    b[near] = (sb * 3 + cb)[near]
    off = sb == 2
    if split is None:
        b[off] = 6 + cb[off]
    elif k == "region":
        b[off] = 6 + split[off]
    else:
        b[off] = 6 + cb[off] * k + split[off]
    b[~valid] = -1
    return b


def levels(field, sel):
    q = np.quantile(field[sel], LEVEL_Q)
    lv = np.zeros(field.shape, np.int32)
    lv[field > q[0]] = 1
    lv[field > q[1]] = 2
    return lv, [float(x) for x in q]


def main():
    t0 = time.time()
    g = core.grid()
    valid = g.footprint & ~g.catalogue
    dcat = core.catalogue_distance_px()
    dsup = np.load(core.WORK / "d_h195.npy")
    cb, sb = base_bins(dcat, dsup, valid)
    offsel = valid & (sb == 2)

    cand = {
        "H78-1 lidar scarp consensus": F.lidar_consensus(),
        "H78-2 thermal proximity (-km to Hot/Warm)": -F.thermal_distance_km(),
        "H78-4a lidar ex_max": F.lidar_single("ex_max"),
        "H78-4b lidar step_max": F.lidar_single("step_max"),
        "H78-4c lidar upface_max": F.lidar_single("upface_max"),
        "H78-4d lidar relief": F.lidar_single("relief"),
        "CTRL H77-1 straight-edge (LIA)": np.load(core.WORK / "f_sef.npy"),
        "CTRL det_elev_slope (band 19)": np.nan_to_num(core.read_band(19), nan=-1e9),
    }
    np.save(core.WORK / "f_h78_lidar_consensus.npy", cand["H78-1 lidar scarp consensus"])
    np.save(core.WORK / "f_h78_thermal_km.npy", -cand["H78-2 thermal proximity (-km to Hot/Warm)"])
    systems = {"S0 support x catalogue": system_with_split(cb, sb, valid, None, None)}
    qinfo = {}
    for name, fld in cand.items():
        lv, q = levels(fld, offsel)
        qinfo[name] = q
        systems[name] = system_with_split(cb, sb, valid, lv, 3)
    systems["H78-3 region blocks 3x3"] = system_with_split(cb, sb, valid, F.region_blocks(3), "region")
    del cand

    rows = load()
    y = np.array([r["score"] for r in rows])
    names = [r["name"] for r in rows]
    Cs, S = cover_design(rows, list(systems.values()))
    print(f"design built in {time.time() - t0:.0f}s", flush=True)
    res = []
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(y), (DRAWS, len(y)))
    base_err = None
    for (name, b), C in zip(systems.items(), Cs):
        N = np.bincount(b.ravel()[b.ravel() >= 0], minlength=C.shape[1]).astype(float)
        h = inv.fit_nnls(C, S, N, y, c=2.0, ridge=RIDGE)
        lo = inv.loo(C, S, N, y, c=2.0, ridge=RIDGE)
        err = (lo - y) ** 2
        r = {"model": name, "k": int(C.shape[1]), "loo_rmse": float(np.sqrt(err.mean())),
             "loo_spearman": float(spearmanr(lo, y)[0]), "h": h.tolist(), "cells": N.tolist(),
             "K": float(N @ h), "K_off_support": float(N[6:] @ h[6:]),
             "loo_pred": lo.tolist(), "level_quantiles": qinfo.get(name)}
        if base_err is None:
            base_err = err
        else:
            d = (base_err - err)[boot_idx].mean(axis=1)
            r["delta_mse_vs_S0_mean"] = float((base_err - err).mean())
            r["delta_mse_vs_S0_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
            if name.startswith("H78-3"):
                top = h[6:]
            else:
                top = np.asarray(h[6:]).reshape(3, 3)[:, 2]
            r["off_support_top_h"] = [float(x) for x in np.ravel(top)]
            r["live_supported"] = bool(r["delta_mse_vs_S0_ci95"][0] > 0 and np.max(top) > 0)
        res.append(r)
        extra = "" if "delta_mse_vs_S0_mean" not in r else (
            f" dMSE={r['delta_mse_vs_S0_mean']:+.6f} CI[{r['delta_mse_vs_S0_ci95'][0]:+.6f},"
            f"{r['delta_mse_vs_S0_ci95'][1]:+.6f}] Koff={r['K_off_support']:.0f} live={r['live_supported']}")
        print(f"{name:44s} k={r['k']:2d} LOO RMSE={r['loo_rmse']:.4f} rho={r['loo_spearman']:+.3f} K={r['K']:.0f}{extra}",
              flush=True)
    cands = [r for r in res[1:] if r["model"].startswith("H78")]
    best = max(cands, key=lambda r: r["delta_mse_vs_S0_mean"])
    out = {"label": "INSTRUMENT (live inversion over organizer scores; not a score)",
           "rule": "registry/preregistration_h78.json e1_rule", "n_rasters": len(y),
           "names": names, "scores": y.tolist(), "results": res,
           "selected": best["model"], "selected_live_supported": best["live_supported"],
           "n_rows_screened": len(res) - 1, "runtime_s": round(time.time() - t0, 1)}
    (ROOT / "evidence" / "h78_e1_screen.json").write_text(json.dumps(out, indent=1))
    np.savez(core.WORK / "h78_e1_design.npz", y=y, S=S, **{f"C_{k}": C for k, C in enumerate(Cs)})
    print("selected:", best["model"], "live_supported:", best["live_supported"])


if __name__ == "__main__":
    main()
