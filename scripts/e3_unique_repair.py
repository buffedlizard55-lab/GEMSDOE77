#!/usr/bin/env python3
"""E3 uniqueness repair: rebuild the candidate under an explicit per-raster overlap budget.

The first E3 build capped overlap only against the H19-5 line set and was logged as a
DUPLICATE by the template audit (evidence/e3_uniqueness.json): 69 like-for-like firings,
including genuine dot fields with 16-33% halo coverage.  This rebuild keeps the same
pre-registered priorities (S0 credit for on-support, S1 credit then straight-edge value for
off-support) but enforces, for EVERY informative registry raster R (halo coverage < 0.70,
see scripts/e3_registry_free.py):

    |D within 3 px of R| <= 0.66 |D|        (protocol limit 0.70; 0.04 margin)

Off-support candidates are accepted greedily only if they keep every raster within budget,
or lie outside the halo of every raster currently over budget.  If violations remain when
the pool is exhausted, N_on is reduced by 10% and the build repeats.
Writes .cache/work/candidate_dots.npy (overwrites the duplicate build, which is kept as
.cache/work/candidate_dots_dup.npy) and evidence/e3_unique_repair.json.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems77lia import allocate as al, core  # noqa: E402
from e3_build_candidate import bins_like_e2  # noqa: E402

BUDGET = 0.66


def main():
    g = core.grid()
    valid = g.footprint & ~g.catalogue
    dcat = core.catalogue_distance_px()
    dsup = np.load(core.WORK / "d_h195.npy")
    sef = np.load(core.WORK / "f_sef.npy")
    ev = json.loads((ROOT / "evidence" / "e2_hypothesis_test.json").read_text())
    t80, t95 = ev["sef_thresholds"]["t80"], ev["sef_thresholds"]["t95"]
    eb = np.where(sef <= 0, 0, np.where(sef <= t80, 1, np.where(sef <= t95, 2, 3)))
    res = {r["model"][:2]: r for r in ev["results"] if r["ridge"] == 1e-3}
    hm0 = al.hmap_from_bins(bins_like_e2(sef, dcat, dsup, valid, t80, t95, None), res["S0"]["h"])
    hm1 = al.hmap_from_bins(bins_like_e2(sef, dcat, dsup, valid, t80, t95, eb), res["S1"]["h"])
    cr0, cr1 = al.credit_map(hm0), al.credit_map(hm1)
    sel0 = json.loads((ROOT / "evidence" / "e3_allocation.json").read_text())["selected"]
    B = sel0["B_px"]

    # pools in pre-registered priority order
    L = (dsup == 0) & valid
    on_c = np.flatnonzero((L & (dcat > B)).ravel())
    on_c = on_c[np.lexsort((-sef.ravel()[on_c], -cr0.ravel()[on_c]))]
    off_c = np.flatnonzero((valid & (dsup > 3) & (sef > 0) & (dcat > 4)).ravel())
    off_c = off_c[np.lexsort((-sef.ravel()[off_c], -cr1.ravel()[off_c]))]
    pool = np.concatenate([on_c, off_c])
    pr, pc = np.divmod(pool, core.W)

    # membership of every pool cell in every informative registry halo (Euclidean <= 3 px)
    free = json.loads((ROOT / "evidence" / "e3_registry_free.json").read_text())
    excluded = {x["raster"] for x in free["space_filling_rasters"]} | {x["raster"] for x in free["unreadable_rasters"]}
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    names, cols = [], []
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
        names.append(r["aliases"][0])
    Mbits = np.stack(cols, axis=1)             # (ceil(P/8), W) packed along the pool axis
    W = len(names)
    print("informative rasters:", W, "pool:", len(pool), flush=True)

    def member(i):
        return (Mbits[i >> 3] >> (7 - (i & 7))) & 1   # length-W 0/1 vector for pool cell i

    n_on_pool = len(on_c)
    on_greedy = al.greedy_select(on_c, 3.0)
    pos_of = {int(f): k for k, f in enumerate(pool)}
    frac = sel0["on_frac_of_greedy"]
    for attempt in range(8):
        n_on = int(len(on_greedy) * frac)
        on_sel = on_greedy[:n_on]
        o = np.zeros(W, np.int64)
        for f in on_sel:
            o += member(pos_of[int(f)])
        n = n_on
        # spacing structure
        cell = 3
        occ = {}
        for f in on_sel:
            r_, c_ = divmod(int(f), core.W)
            occ.setdefault((r_ // cell, c_ // cell), []).append((r_, c_))
        off_sel = []
        for k in range(n_on_pool, len(pool)):
            f = int(pool[k])
            r_, c_ = divmod(f, core.W)
            gr, gc = r_ // cell, c_ // cell
            ok = True
            for a in (gr - 1, gr, gr + 1):
                for b in (gc - 1, gc, gc + 1):
                    for (rr_, cc_) in occ.get((a, b), ()):
                        if (rr_ - r_) ** 2 + (cc_ - c_) ** 2 < 9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            m = member(k)
            viol = o > BUDGET * n
            if viol.any():
                if (m[viol] == 1).any():
                    continue
                if np.any((o + m)[~viol] > BUDGET * (n + 1)):
                    continue
            else:
                if np.any(o + m > BUDGET * (n + 1)):
                    continue
            o += m
            n += 1
            off_sel.append(f)
            occ.setdefault((gr, gc), []).append((r_, c_))
            # stop once compliant and off-support share reaches the dilution needed
            if not (o > BUDGET * n).any() and len(off_sel) >= int(np.ceil(n_on * (1 - BUDGET) / BUDGET)):
                break
        worst = float((o / n).max())
        print(f"attempt {attempt}: n_on={n_on} n_off={len(off_sel)} worst_overlap={worst:.4f} "
              f"violators={int((o > BUDGET * n).sum())}", flush=True)
        if not (o > BUDGET * n).any():
            break
        frac *= 0.9
    dots = np.zeros(core.H * core.W, bool)
    dots[on_sel] = True
    dots[np.asarray(off_sel, dtype=np.int64)] = True
    dots = dots.reshape(core.H, core.W)
    dup = core.WORK / "candidate_dots_dup.npy"
    if not dup.exists():
        shutil.copy(core.WORK / "candidate_dots.npy", dup)
    np.save(core.WORK / "candidate_dots.npy", dots)
    p0, p1 = al.project(dots, hm0), al.project(dots, hm1)
    top = np.argsort(-(o / n))[:10]
    out = {"label": "INSTRUMENT projections; overlap measured on pool membership (Euclidean 3 px)",
           "budget": BUDGET, "informative_rasters": W, "B_px": B, "on_frac_of_greedy": frac,
           "n_on": int(len(on_sel)), "n_off": int(len(off_sel)), "dots": int(dots.sum()),
           "on_share": len(on_sel) / dots.sum(), "worst_overlap": worst,
           "worst_rasters": [{"raster": names[i], "overlap": float(o[i] / n)} for i in top],
           "S0": p0, "S1": p1}
    (ROOT / "evidence" / "e3_unique_repair.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_on", "n_off", "dots", "on_share", "worst_overlap")}),
          "S0", round(p0["dti_projection"], 4), "S1", round(p1["dti_projection"], 4))


if __name__ == "__main__":
    main()
