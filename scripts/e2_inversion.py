#!/usr/bin/env python3
"""Experiment 2 (part a): live-score inversion over 95 organizer-scored registry rasters.

Fits the truth-density model of gems77lia.inversion for several bin systems and reports
leave-one-out (LOO) and leave-one-family-out predictive accuracy against the
organizer scores.  Writes evidence/e2_inversion.json.
Every output is an INSTRUMENT value (projection), never a score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems77lia import core, inversion as inv  # noqa: E402

CAT_EDGES = [0, 2, 4, 8, 20, 60, 1e9]          # px from catalogue
SUP_EDGES = [-1, 0, 1.5, 3, 6, 1e9]           # px from H19-5 line set (explanatory only)


def load():
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    Z = np.load(core.WORK / "scored_sparse.npz")
    seen, rows = set(), []
    for e in idx["scored"]:
        if e["blob"] in seen:
            continue
        seen.add(e["blob"])
        # several rasters were scored more than once (twins); use the mean of their scores
        sc = [x["public_score"] for x in idx["scored"] if x["blob"] == e["blob"]]
        rows.append({"blob": e["blob"], "name": e["submission_name"], "score": float(np.mean(sc)),
                     "i": Z["i_" + e["blob"]], "v": Z["v_" + e["blob"]]})
    return rows


def bin_map(edges, dist, footprint, exclude):
    b = np.digitize(dist, edges[1:], right=True)
    b = b.astype(np.int16)
    b[~footprint | exclude] = -1
    return b


def cover_design(rows, bins_list):
    """C[i, j] for each bin system; S[i]."""
    nbs = [int(b.max()) + 1 for b in bins_list]
    Cs = [np.zeros((len(rows), nb)) for nb in nbs]
    S = np.zeros(len(rows))
    for k, r in enumerate(rows):
        p = np.zeros(core.H * core.W, np.float32)
        p[r["i"]] = r["v"]
        p = p.reshape(core.H, core.W)
        m = core.cover_field(p).ravel()
        S[k] = float(r["v"].sum())
        for C, b in zip(Cs, bins_list):
            bb = b.ravel()
            ok = bb >= 0
            C[k] = np.bincount(bb[ok], weights=m[ok], minlength=C.shape[1])
    return Cs, S


def evaluate(name, C, S, N, y, groups, c=2.0, ridge=1e-6):
    h = inv.fit_nnls(C, S, N, y, c=c, ridge=ridge)
    fit = inv.predict(h, C, S, N, c)
    lo = inv.loo(C, S, N, y, c=c, ridge=ridge)
    lg = inv.loo(C, S, N, y, c=c, ridge=ridge, groups=groups)
    out = {"model": name, "n_params": int(C.shape[1]), "c": c,
           "in_sample_rmse": float(np.sqrt(np.mean((fit - y) ** 2))),
           "loo_rmse": float(np.sqrt(np.mean((lo - y) ** 2))),
           "loo_spearman": float(spearmanr(lo, y)[0]), "loo_pearson": float(pearsonr(lo, y)[0]),
           "lofo_rmse": float(np.sqrt(np.mean((lg - y) ** 2))),
           "lofo_spearman": float(spearmanr(lg, y)[0]),
           "h": h.tolist(), "K_implied": float(N @ h)}
    print(f"{name:34s} k={C.shape[1]:3d} c={c:.1f} inRMSE={out['in_sample_rmse']:.4f} "
          f"LOO RMSE={out['loo_rmse']:.4f} rho={out['loo_spearman']:+.3f} r={out['loo_pearson']:+.3f} | "
          f"LOFO RMSE={out['lofo_rmse']:.4f} rho={out['lofo_spearman']:+.3f} K={out['K_implied']:.0f}")
    return out, lo, lg


def main():
    g = core.grid()
    rows = load()
    y = np.array([r["score"] for r in rows])
    dcat = core.catalogue_distance_px()
    dsup = np.load(core.WORK / "d_h195.npy")
    bc = bin_map(CAT_EDGES, dcat, g.footprint, g.catalogue)
    bs = bin_map(SUP_EDGES, dsup, g.footprint, g.catalogue)
    b2 = np.where((bc >= 0) & (bs >= 0), bc * (len(SUP_EDGES) - 1) + bs, -1).astype(np.int16)
    one = np.where(bc >= 0, 0, -1).astype(np.int16)
    systems = {"null (1 bin)": one, "catalogue distance": bc, "H19-5 support distance": bs,
               "catalogue x support": b2}
    Cs, S = cover_design(rows, list(systems.values()))
    # family groups for leave-one-family-out: H19-5 family (>=97% dots on the line set) vs others
    fam = []
    for r in rows:
        on = float(np.mean(dsup.ravel()[r["i"]] == 0)) if len(r["i"]) else 0
        fam.append(0 if on >= 0.97 else 1 + len(fam))
    fam = np.array(fam)
    results = []
    for (name, b), C in zip(systems.items(), Cs):
        N = np.bincount(b.ravel()[b.ravel() >= 0], minlength=C.shape[1]).astype(float)
        for c in (1.5, 2.0, 2.5):
            out, lo, lg = evaluate(name, C, S, N, y, fam, c=c)
            out["cells_per_bin"] = N.tolist()
            results.append(out)
    # persist design for downstream use
    np.savez(core.WORK / "e2_design.npz", y=y, S=S, fam=fam,
             **{f"C_{k}": C for k, C in enumerate(Cs)})
    (ROOT / "evidence" / "e2_inversion.json").write_text(json.dumps({
        "label": "INSTRUMENT (projection from organizer scores; not a score)",
        "n_rasters": len(rows), "cat_edges_px": CAT_EDGES, "support_edges_px": SUP_EDGES,
        "support": "H19-5 line set (19GEMSDOE, scored 0.1922) - explanatory use only",
        "names": [r["name"] for r in rows], "scores": y.tolist(), "results": results}, indent=1))


if __name__ == "__main__":
    main()
