#!/usr/bin/env python3
"""H78 Experiment 2: does an independent field rank WHICH H19-5 support pixels hold hidden truth?

Rule frozen in registry/preregistration_h78_amendment_e2.json (committed after E1, before this ran).
On-support bins (support distance 0 px) are split by catalogue {0-4, 4-20, >20 px} x field level
{top 10 %, 10-40 %, rest} (quantiles over valid on-support cells); near/off bins as in S0.
LOO over the 95 distinct organizer-scored rasters, paired bootstrap of the LOO-MSE reduction.
INSTRUMENT values only.  Writes evidence/h78_e2_within_support.json.
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
from h78_e1_screen import base_bins  # noqa: E402

LEVEL_Q = (0.60, 0.90)
RIDGE = 1e-3
DRAWS = 4000


def system_on_split(cb, sb, valid, split, k):
    """bins: on x cat x level (0..9k-1), near x cat (+3), off x cat (+3)."""
    b = np.full(cb.shape, -1, np.int32)
    on, near, off = sb == 0, sb == 1, sb == 2
    nb_on = 3 if split is None else (9 if k == "region" else 3 * k)
    if split is None:
        b[on] = cb[on]
    elif k == "region":
        # region applies to the far (>20 px) on-support bin only; near-catalogue on bins kept
        far = on & (cb == 2)
        b[on & (cb < 2)] = cb[on & (cb < 2)]
        b[far] = 2 + split[far]
        nb_on = 2 + 9
    else:
        b[on] = (cb * k + split)[on]
    b[near] = nb_on + cb[near]
    b[off] = nb_on + 3 + cb[off]
    b[~valid] = -1
    return b, nb_on


def main():
    t0 = time.time()
    g = core.grid()
    valid = g.footprint & ~g.catalogue
    dcat = core.catalogue_distance_px()
    dsup = np.load(core.WORK / "d_h195.npy")
    cb, sb = base_bins(dcat, dsup, valid)
    onsel = valid & (sb == 0)
    cand = {
        "H78-5a lidar scarp consensus": np.load(core.WORK / "f_h78_lidar_consensus.npy"),
        "H78-5b lidar upface_max": F.lidar_single("upface_max"),
        "H78-5c lidar ex_max": F.lidar_single("ex_max"),
        "H78-5d H77-1 straight-edge": np.load(core.WORK / "f_sef.npy"),
        "H78-5e det_elev_slope (band 19)": np.nan_to_num(core.read_band(19), nan=-1e9),
        "H78-5f thermal proximity": -np.load(core.WORK / "f_h78_thermal_km.npy"),
    }
    systems, meta = {}, {}
    b0, nb0 = system_on_split(cb, sb, valid, None, None)
    systems["S0 support x catalogue"] = b0
    meta["S0 support x catalogue"] = {"nb_on": nb0}
    for name, fld in cand.items():
        q = np.quantile(fld[onsel], LEVEL_Q)
        lv = np.zeros(fld.shape, np.int32)
        lv[fld > q[0]] = 1
        lv[fld > q[1]] = 2
        b, nb = system_on_split(cb, sb, valid, lv, 3)
        systems[name] = b
        meta[name] = {"nb_on": nb, "level_quantiles": [float(x) for x in q]}
    b, nb = system_on_split(cb, sb, valid, F.region_blocks(3), "region")
    systems["H78-5g region blocks (far on-support)"] = b
    meta["H78-5g region blocks (far on-support)"] = {"nb_on": nb}
    np.savez_compressed(core.WORK / "h78_e2_fields.npz", **{k.split()[0]: v for k, v in cand.items()})
    del cand
    rows = load()
    y = np.array([r["score"] for r in rows])
    names = [r["name"] for r in rows]
    Cs, S = cover_design(rows, list(systems.values()))
    print(f"design built in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(y), (DRAWS, len(y)))
    res, base_err = [], None
    watch = [i for i, n in enumerate(names) if n.startswith(("gate_ortho", "gems70", "lidarscarp-ridge",
                                                              "h33-h33-2-b2", "h64-lidar", "gems66"))]
    for (name, b), C in zip(systems.items(), Cs):
        N = np.bincount(b.ravel()[b.ravel() >= 0], minlength=C.shape[1]).astype(float)
        h = inv.fit_nnls(C, S, N, y, c=2.0, ridge=RIDGE)
        lo = inv.loo(C, S, N, y, c=2.0, ridge=RIDGE)
        err = (lo - y) ** 2
        r = {"model": name, "k": int(C.shape[1]), **meta[name], "loo_rmse": float(np.sqrt(err.mean())),
             "loo_spearman": float(spearmanr(lo, y)[0]), "h": h.tolist(), "cells": N.tolist(),
             "K": float(N @ h), "loo_pred": lo.tolist(),
             "watch_residuals": {names[i]: float(y[i] - lo[i]) for i in watch}}
        if base_err is None:
            base_err = err
        else:
            d = (base_err - err)[boot_idx].mean(axis=1)
            r["delta_mse_vs_S0_mean"] = float((base_err - err).mean())
            r["delta_mse_vs_S0_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
            if "region" in name:
                ordered = True
                hs = np.asarray(h[2:11])
                r["far_on_support_h_by_region"] = hs.tolist()
                ordered = bool(hs.max() > 1.5 * max(np.median(hs), 1e-9))
            else:
                hh = np.asarray(h[:9]).reshape(3, 3)       # cat x level
                r["on_support_h_cat_by_level"] = hh.tolist()
                ordered = bool(np.any(hh[:, 2] > hh[:, 0]))
            r["ordered"] = ordered
            r["live_supported"] = bool(r["delta_mse_vs_S0_ci95"][0] > 0 and ordered)
        res.append(r)
        extra = "" if "delta_mse_vs_S0_mean" not in r else (
            f" dMSE={r['delta_mse_vs_S0_mean']:+.6f} CI[{r['delta_mse_vs_S0_ci95'][0]:+.6f},"
            f"{r['delta_mse_vs_S0_ci95'][1]:+.6f}] live={r['live_supported']}")
        print(f"{name:40s} k={r['k']:2d} LOO RMSE={r['loo_rmse']:.4f} rho={r['loo_spearman']:+.3f} K={r['K']:.0f}{extra}",
              flush=True)
        print("    h_on:", np.round(np.asarray(h[:meta[name]['nb_on']]), 4).tolist(), flush=True)
        print("    watch:", {k[:22]: round(v, 3) for k, v in r["watch_residuals"].items()}, flush=True)
    best = max(res[1:], key=lambda r: r["delta_mse_vs_S0_mean"])
    out = {"label": "INSTRUMENT (live inversion over organizer scores; not a score)",
           "rule": "registry/preregistration_h78_amendment_e2.json", "n_rasters": len(y), "names": names,
           "scores": y.tolist(), "results": res, "selected": best["model"],
           "selected_live_supported": best["live_supported"], "n_rows_screened": len(res) - 1,
           "runtime_s": round(time.time() - t0, 1)}
    (ROOT / "evidence" / "h78_e2_within_support.json").write_text(json.dumps(out, indent=1))
    np.savez(core.WORK / "h78_e2_design.npz", y=y, S=S, **{f"C_{k}": C for k, C in enumerate(Cs)})
    print("selected:", best["model"], "live_supported:", best["live_supported"])


if __name__ == "__main__":
    main()
