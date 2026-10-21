#!/usr/bin/env python3
"""Quantitative evidence for reports/why_02778.md (S0 live-inversion instrument).

* re-measures the 0.2778 raster from its bytes (dots, spacing, catalogue distances);
* reproduces the GEMSDOE19 -> GEMSDOE32 lineage with the S0 instrument (in-sample);
* champion credit per dot by S0 bin vs the break-even bar alpha*DTI;
* break-even credit per off-support dot for a protocol-unique candidate;
* credit per dot implied by the leaderboard #1 (0.4413, ORGANIZER-CONFIRMED on the
  public leaderboard page) at the champion's dot count.
Writes evidence/why_02778.json.  Projections are INSTRUMENT values, not scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems77 import allocate as al, core  # noqa: E402
from e3_build_candidate import bins_like_e2  # noqa: E402

LINEAGE = [
    ("h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan", "H19-5 solid ridges (19GEMSDOE)"),
    ("h25-1-dotted-h19-5-d1-5-20261002-989f59505db1-nan", "dotted d=1.5 px (GEMSDOE24)"),
    ("dotted-h19-5-d2-8-20261002-e56ea318af89-nan", "dotted d=2.8 px (GEMSDOE25)"),
    ("h27-4-r1-solo-d2-8-20261003-8acb75e1f2cc-nan", "d2.8 minus dots <=1 px from catalogue (GEMSDOE28/31)"),
    ("h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros", "d2.8 minus dots <=2 px (200 m) from catalogue (GEMSDOE32)"),
]


def main():
    g = core.grid()
    valid = g.footprint & ~g.catalogue
    dcat = core.catalogue_distance_px()
    dsup = np.load(core.WORK / "d_h195.npy")
    sef = np.load(core.WORK / "f_sef.npy")
    ev = json.loads((ROOT / "evidence" / "e2_hypothesis_test.json").read_text())
    t80, t95 = ev["sef_thresholds"]["t80"], ev["sef_thresholds"]["t95"]
    S0 = [r for r in ev["results"] if r["model"].startswith("S0") and r["ridge"] == 1e-3][0]
    bins0 = bins_like_e2(sef, dcat, dsup, valid, t80, t95, None)
    hm0 = al.hmap_from_bins(bins0, S0["h"])
    cr0 = al.credit_map(hm0)
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    by = {e["submission_name"]: e for e in idx["scored"]}
    out = {"label": "INSTRUMENT (S0 live inversion, in-sample) unless marked ORGANIZER", "lineage": []}
    for nm, lab in LINEAGE:
        e = by[nm]
        d = core.read_prediction(core.REG / f"{e['blob']}.tif") > 0
        p = al.project(d, hm0)
        out["lineage"].append({"step": lab, "submission": nm, "organizer_score": e["public_score"],
                               "dots": int(d.sum()), "S0_projection": p["dti_projection"],
                               "S0_T": p["T"], "S0_credit_per_dot": p["credit_per_dot"]})
        print(f"{e['public_score']:.4f} S0={p['dti_projection']:.4f} dots={int(d.sum()):6d} cpd={p['credit_per_dot']:.4f} {lab}")
    champ = core.read_prediction(core.REG / f"{by[LINEAGE[-1][0]]['blob']}.tif") > 0
    rr, cc = np.nonzero(champ)
    tree = spatial.cKDTree(np.c_[rr, cc])
    dnn, _ = tree.query(np.c_[rr, cc], k=2)
    dc = dcat[champ]
    out["champion_bytes"] = {
        "sha256": core.sha256_file(core.REG / f"{by[LINEAGE[-1][0]]['blob']}.tif"),
        "dots": int(champ.sum()), "dots_on_catalogue": int((champ & g.catalogue).sum()),
        "nn_spacing_px_quantiles": {q: float(np.quantile(dnn[:, 1], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "min_catalogue_distance_px": float(dc.min()),
        "share_dots_by_catalogue_distance_km": {
            "0.2-0.4": float(np.mean((dc > 2) & (dc <= 4))), "0.4-2": float(np.mean((dc > 4) & (dc <= 20))),
            "2-6": float(np.mean((dc > 20) & (dc <= 60))), ">6": float(np.mean(dc > 60))},
        "share_dots_on_h19_5_lines": float(np.mean(dsup[champ] == 0))}
    pc = al.project(champ, hm0)
    bar = 0.2 * pc["dti_projection"]
    lab9 = [f"{s}/{c}" for s in ("on", "near", "off") for c in ("cat<=0.4km", "cat0.4-2km", "cat>2km")]
    per = []
    for j, l in enumerate(lab9):
        m = champ & (bins0 == j)
        if m.any():
            per.append({"bin": l, "dots": int(m.sum()), "mean_isolated_credit": float(cr0[m].mean()),
                        "h": float(S0["h"][j])})
    out["champion_S0"] = {**pc, "bar_alpha_dti": bar, "by_bin": per}
    print("champion S0", round(pc["dti_projection"], 4), "T", round(pc["T"]), "K", round(pc["K"]), "bar", round(bar, 4))
    for r in per:
        print("  ", r)
    # break-even for the unique candidate
    alloc = json.loads((ROOT / "evidence" / "e3_allocation.json").read_text())["selected"]
    dots = np.load(core.WORK / "candidate_dots.npy")
    on = dots & (dsup == 0)
    p_on = al.project(on, hm0)
    S_tot, K = float(dots.sum()), pc["K"]
    target = pc["dti_projection"]
    # solve T_tot / (0.1 T_tot + 0.2 S_tot + 0.8 K) = target  (c = 2)
    T_need = target * (0.2 * S_tot + 0.8 * K) / (1 - 0.1 * target)
    n_off = int((dots & (dsup > 3)).sum())
    cpd_need = (T_need - p_on["T"]) / n_off
    out["break_even_unique"] = {"candidate_dots": int(S_tot), "on_support_dots": int(on.sum()),
                                "off_support_dots": n_off, "S0_T_on_support": p_on["T"],
                                "T_needed_to_tie_champion_S0": T_need,
                                "credit_per_off_support_dot_needed": cpd_need,
                                "S1_estimated_credit_per_off_support_dot": None}
    print("break-even credit per off-support dot", round(cpd_need, 4))
    # leaderboard #1 at the champion's dot count
    for dti_t, lab in ((0.4413, "leaderboard #1 (ORGANIZER-CONFIRMED public LB, 2026-10-21)"),
                       (0.3195, "user-quoted 0.3195 (not on current leaderboard top 20)")):
        T = dti_t * (0.2 * pc["S"] + 0.8 * K) / (1 - 0.1 * dti_t)
        out.setdefault("targets", []).append({"target": dti_t, "label": lab, "dots": int(pc["S"]),
                                              "T_needed": T, "credit_per_dot_needed": T / pc["S"],
                                              "recall_needed": T / K})
        print(lab, "T needed", round(T), "credit/dot", round(T / pc["S"], 4), "recall", round(T / K, 3))
    out["max_possible_at_champion_dots"] = K / (0.1 * K + 0.2 * pc["S"] + 0.8 * K)
    (ROOT / "evidence" / "why_02778.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
