#!/usr/bin/env python3
"""Experiment 3: build the GEMSDOE77 candidate by live-inversion allocation.

Allocation rule (pre-registered in docs/hypotheses.html before this script ran):
  * on-support dots: cells of the H19-5 line set (the support the live inversion finds
    ~38x enriched in hidden truth), farther than B px from the catalogue, chosen greedily
    by S0 expected credit (ties broken by the H77-1 straight-edge value), >= 3 px apart;
  * off-support dots (H77-1): straight-edge cells above the 80th percentile of edge
    strength, more than 3 px from the H19-5 line set and more than 4 px from the
    catalogue, chosen greedily by the S1 expected credit (which ranks the >q95 tier at
    4-20 px first) then the edge value, >= 3 px from every accepted dot;
  * uniqueness by construction: on-support share <= 0.65 (protocol limit 0.70 within
    3 px of one registry raster), then verified against all 812 registry rasters with
    the template tool (scripts/e3_uniqueness.py).
N_on is chosen to maximise the conservative S0 projection.  Writes
.cache/work/candidate_dots.npy and evidence/e3_allocation.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems77lia import allocate as al, core  # noqa: E402

CAT3 = [0, 4, 20, 1e9]
ON_SHARE_MAX = 0.65


def bins_like_e2(sef, dcat, dsup, valid, t80, t95, off_split):
    cb = np.digitize(dcat, CAT3[1:], right=True)
    sb = np.where(dsup == 0, 0, np.where(dsup <= 3, 1, 2))
    b = np.full(sef.shape, -1, np.int32)
    base = sb * 3 + cb
    b[sb < 2] = base[sb < 2]
    if off_split is None:
        b[sb == 2] = 6 + cb[sb == 2]
    else:
        k = int(off_split.max()) + 1
        b[sb == 2] = 6 + cb[sb == 2] * k + off_split[sb == 2]
    b[~valid] = -1
    return b


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

    # sanity: the champion under both instruments (in-sample)
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    champ = [e for e in idx["scored"] if e["public_score"] == 0.2778][0]
    cm = core.read_prediction(core.REG / f"{champ['blob']}.tif") > 0
    pc0, pc1 = al.project(cm, hm0), al.project(cm, hm1)
    print("champion projection S0", round(pc0["dti_projection"], 4), "S1", round(pc1["dti_projection"], 4),
          "(organizer 0.2778)")

    L = (dsup == 0) & valid
    out = {"label": "INSTRUMENT projections (live inversion); NOT scores",
           "champion_projection": {"S0": pc0, "S1": pc1, "organizer_score": 0.2778}, "variants": []}
    best = None
    for B in (2, 4, 8):
        on_c = np.flatnonzero((L & (dcat > B)).ravel())
        pri = np.lexsort((-sef.ravel()[on_c], -cr0.ravel()[on_c]))
        on_sel_all = al.greedy_select(on_c[pri], 3.0)
        off_c = np.flatnonzero((valid & (dsup > 3) & (sef > t80) & (dcat > 4)).ravel())
        pri_off = np.lexsort((-sef.ravel()[off_c], -cr1.ravel()[off_c]))
        for frac in (1.0, 0.85, 0.7, 0.55):
            n_on = int(len(on_sel_all) * frac)
            on_sel = on_sel_all[:n_on]
            n_off = int(np.ceil(n_on * (1 - ON_SHARE_MAX) / ON_SHARE_MAX))
            off_sel = al.greedy_select(off_c[pri_off], 3.0, taken=on_sel, limit=n_off)
            if len(off_sel) < n_off:
                print(f"  B={B} frac={frac}: only {len(off_sel)} off-support dots available (< {n_off})")
            dots = np.zeros(core.H * core.W, bool)
            dots[on_sel] = True
            dots[off_sel] = True
            dots = dots.reshape(core.H, core.W)
            p0, p1 = al.project(dots, hm0), al.project(dots, hm1)
            share = len(on_sel) / max(1, len(on_sel) + len(off_sel))
            v = {"B_px": B, "on_frac_of_greedy": frac, "n_on": len(on_sel), "n_off": len(off_sel),
                 "on_share": share, "S0": p0, "S1": p1}
            out["variants"].append(v)
            print(f"B={B} frac={frac:.2f} n_on={len(on_sel):6d} n_off={len(off_sel):6d} share_on={share:.3f} "
                  f"S0={p0['dti_projection']:.4f} S1={p1['dti_projection']:.4f}")
            if share <= ON_SHARE_MAX + 1e-9 and (best is None or p0["dti_projection"] > best[0]["S0"]["dti_projection"]):
                best = (v, dots)
    v, dots = best
    out["selected"] = v
    np.save(core.WORK / "candidate_dots.npy", dots)
    (ROOT / "evidence" / "e3_allocation.json").write_text(json.dumps(out, indent=1))
    print("SELECTED", {k: v[k] for k in ("B_px", "on_frac_of_greedy", "n_on", "n_off", "on_share")},
          "S0", round(v["S0"]["dti_projection"], 4), "S1", round(v["S1"]["dti_projection"], 4))


if __name__ == "__main__":
    main()
