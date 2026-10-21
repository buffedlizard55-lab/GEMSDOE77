#!/usr/bin/env python3
"""H78 Experiment 3 (final): candidate build + fold-fair HOLDOUT-DTI + leakage canary.

E1 and E2 found no live-supported field (evidence/h78_e1_screen.json,
evidence/h78_e2_within_support.json), so E3 builds the PRE-REGISTERED DEFAULT
(registry/preregistration_h78.json, e2_rule) with the a-priori rank-1 field H78-1:

  on-support dots   H19-5 line cells > B px from the (visible) catalogue, greedy (>= 3 px apart)
                    by S0 expected credit, lidar consensus as tie-break; first on_frac of them
  off-support dots  every footprint cell > 3 px from the support and > 4 px from the (visible)
                    catalogue, greedy by the H78-1 lidar scarp consensus (cells without lidar
                    coverage last, in a fixed random order) - the pre-registered pool; an earlier
                    top-half-consensus shortcut ran out of admissible cells (IR-78-03)

Allocation = capacity-constrained greedy: the total N = n_on / s_on is fixed first; every informative
registry raster R gets the capacity floor(0.66 N); a candidate (on-support dots first, by credit;
then off-support dots, by consensus) is accepted only if it is >= 3 px from accepted dots and
every raster whose 3 px halo contains it is still below capacity.  This guarantees overlap <= 0.66
for every informative raster by construction.  (The LIA lane's budgeted_selection rejects every
cell inside ANY over-budget raster's halo; on this pool that left 1 admissible off-support cell,
because 366 informative rasters cover > 30 % of the support and their halos leave only 7,334 free
lidar cells - measured, IR-78-03.)
  uniqueness        per-raster overlap cap 0.66 x N (pre-registered; protocol limit 0.70) against every
                    INFORMATIVE registry raster: the 812 indexed rasters + every TIFF published in
                    GEMSDOE73-76 at clone time + the four GEMSDOE77 lane TIFFs.  Space-filling
                    rasters (3 px halo >= 70 % of allowed cells) are reported, not budgeted.

(B, on_frac) is chosen on the FULL map by the S0 projection (INSTRUMENT) before the holdout and
then frozen.  The holdout (template gems57 evaluator, unchanged; buffered_component_draw
seed=20 hide_frac=0.20 buffer_px=3 - the same draw as the LIA lane, so arms are comparable)
rebuilds every arm per fold from the VISIBLE catalogue.

Writes .cache/work/h78_candidate_dots.npy, .cache/work/h78_surface.npy,
evidence/h78_e3_holdout.json, evidence/h78_e3_registry.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import evaluate_holdout as eh  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems57.holdout import buffered_component_draw  # noqa: E402
from gems57.uniqueness import _near  # noqa: E402
from gems77lia import allocate as al, core  # noqa: E402
from e3_holdout import auc  # noqa: E402

SEED = 20
BUDGET = 0.66
LIDAR_TOP_Q = 0.50  # reported only (median consensus of lidar-covered off-support cells)
CAT3 = [0, 4, 20, 1e9]
GRID_B = (2, 4)
GRID_FRAC = (1.0, 0.85)
GRID_SON = (0.66, 0.60)
EXTRA_GLOBS = [
    (".cache/sib/GEMSDOE73", "**/*.tif"), (".cache/sib/GEMSDOE74", "**/*.tif"),
    (".cache/sib/GEMSDOE75", "**/*.tif"), (".cache/sib/GEMSDOE76", "**/*.tif"),
    ("docs/downloads", "*.tif"),
]


def sha256(p: Path) -> str:
    return core.sha256_file(p)


def registry_files():
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    rec = json.loads((ROOT / "registry" / "registry_receipt.json").read_text())["sha256_by_blob"]
    out = []
    for r in idx["rasters"]:
        if r["blob"] in rec:
            out.append({"id": r["aliases"][0], "repo": r["repo"], "file": str(core.REG / f"{r['blob']}.tif"),
                        "sha256": rec[r["blob"]], "source": "registry_index"})
    seen = {o["sha256"] for o in out}
    for base, pat in EXTRA_GLOBS:
        for p in sorted((ROOT / base).glob(pat)):
            if ".git" in p.parts or p.name.startswith("gems77-h78"):
                continue
            s = sha256(p)
            if s in seen:
                continue
            seen.add(s)
            repo = "GEMSDOE77" if base == "docs/downloads" else base.split("/")[-1]
            rel = p.relative_to(ROOT) if base == "docs/downloads" else p.relative_to(ROOT / base)
            out.append({"id": f"{repo}:{rel}", "repo": repo,
                        "file": str(p), "sha256": s, "source": "added_h78"})
    return out


def s0_hmap(dvis, dsup, valid, h):
    cb = np.digitize(dvis, CAT3[1:], right=True)
    sb = np.where(dsup == 0, 0, np.where(dsup <= 3, 1, 2))
    b = (sb * 3 + cb).astype(np.int32)
    b[~valid] = -1
    return al.hmap_from_bins(b, h)


def cap_greedy(order, unpack, counts, cap, occ, acc, limit, min_sep=3.0):
    """Accept cells in priority order under spacing + per-raster capacity; mutates counts/occ/acc."""
    W = core.W
    cell = int(np.ceil(min_sep))
    s2 = min_sep * min_sep
    n0 = len(acc)
    for f in order:
        if len(acc) - n0 >= limit:
            break
        f = int(f)
        r, c = divmod(f, W)
        gr, gc = r // cell, c // cell
        if any((rr - r) ** 2 + (cc - c) ** 2 < s2 for a in (gr - 1, gr, gr + 1)
               for b in (gc - 1, gc, gc + 1) for (rr, cc) in occ.get((a, b), ())):
            continue
        m = unpack(f)
        if m.any() and (counts[m] >= cap).any():
            continue
        counts[m] += 1
        occ.setdefault((gr, gc), []).append((r, c))
        acc.append(f)
    return len(acc) - n0


def build(dvis, valid, B, frac, s_on, ctx, unpack, n_reg):
    """The frozen candidate rule.  Returns (on_sel, off_sel, overlap_counts, credit_map)."""
    cr0 = al.credit_map(s0_hmap(dvis, ctx["dsup"], valid, ctx["h_s0"]))
    lid = ctx["lid"].ravel()
    vflat, dflat = valid.ravel(), dvis.ravel()
    on_o = ctx["pool_on"][vflat[ctx["pool_on"]] & (dflat[ctx["pool_on"]] > B)]
    on_o = on_o[np.lexsort((-lid[on_o], -cr0.ravel()[on_o]))]
    off_o = ctx["pool_off"][vflat[ctx["pool_off"]] & (dflat[ctx["pool_off"]] > 4)]
    off_o = off_o[np.argsort(-ctx["off_key"].ravel()[off_o], kind="stable")]
    n_on = int(len(al.greedy_select(on_o, 3.0)) * frac)
    n_tot = int(np.floor(n_on / s_on))
    cap = int(np.floor(BUDGET * n_tot))
    counts = np.zeros(n_reg, np.int64)
    occ, acc = {}, []
    got_on = cap_greedy(on_o, unpack, counts, cap, occ, acc, n_on)
    got_off = cap_greedy(off_o, unpack, counts, cap, occ, acc, n_tot - got_on)
    acc = np.asarray(acc, np.int64)
    return acc[:got_on], acc[got_on:], counts, cr0


def main():
    t0 = time.time()
    g = core.grid()
    allowed = g.footprint & ~g.catalogue
    dsup = np.load(core.WORK / "d_h195.npy")
    lid = np.load(core.WORK / "f_h78_lidar_consensus.npy")
    e1 = json.loads((ROOT / "evidence" / "h78_e1_screen.json").read_text())
    h_s0 = np.asarray(e1["results"][0]["h"])
    assert e1["results"][0]["model"].startswith("S0") and len(h_s0) == 9
    lid_q = float(np.quantile(lid[allowed & (dsup > 3) & (lid > 0)], LIDAR_TOP_Q))
    pool_on = np.flatnonzero(((dsup == 0) & g.footprint).ravel())
    pool_off = np.flatnonzero((g.footprint & ~g.catalogue & (dsup > 3)).ravel())  # pre-registered pool
    pool = np.concatenate([pool_on, pool_off])
    pos = np.full(core.H * core.W, -1, np.int64)
    pos[pool] = np.arange(len(pool))
    print(f"pool on={len(pool_on)} off={len(pool_off)} (median lidar consensus {lid_q:.4f})", flush=True)
    # off-support priority: consensus (in [0,1]); no-lidar cells get -1 + random tie-break
    off_key = np.where(np.isfinite(lid) & (lid > 0), lid,
                       -1.0 + 0.5 * np.random.default_rng(78).random(lid.shape)).astype(np.float32)

    # ---------------- registry: informative vs space-filling, membership bits on the pool
    n_allowed = int(allowed.sum())
    min_n = int(np.ceil(0.70 * n_allowed / 29))
    regs = registry_files()
    informative, space_filling, unreadable = [], [], []
    Mbits = np.zeros((len(pool), (len(regs) + 7) // 8), np.uint8)
    j = 0
    for r in regs:
        try:
            p = core.read_prediction(r["file"]) > 0
        except Exception as exc:
            unreadable.append({**r, "error": str(exc)[:160]})
            continue
        n = int(p.sum())
        if n == 0:
            unreadable.append({**r, "error": "no positive cells"})
            continue
        near = _near(p)
        if n >= min_n:
            cov = float(near[allowed].mean())
            if cov >= 0.70:
                space_filling.append({**r, "positives": n, "coverage_3px": cov})
                continue
        col = near.ravel()[pool]
        Mbits[:, j >> 3] |= (col.astype(np.uint8) << np.uint8(7 - (j & 7)))
        informative.append({**r, "positives": n})
        j += 1
    n_reg = j
    Mbits = Mbits[:, : (n_reg + 7) // 8]
    print(f"registry: {len(regs)} files, informative {len(informative)}, space-filling {len(space_filling)}, "
          f"unreadable/empty {len(unreadable)}  ({time.time() - t0:.0f}s)", flush=True)
    (ROOT / "evidence" / "h78_e3_registry.json").write_text(json.dumps({
        "n_files": len(regs), "n_informative": len(informative), "space_filling": space_filling,
        "unreadable_or_empty": unreadable, "added_h78": [r for r in regs if r["source"] == "added_h78"],
        "space_filling_threshold_positives": min_n, "allowed_cells": n_allowed,
        "sibling_clone_heads": {s: (ROOT / ".cache/sib" / s / ".git" / "HEAD").exists()
                                for s in ("GEMSDOE73", "GEMSDOE74", "GEMSDOE75", "GEMSDOE76")}}, indent=1))

    def unpack(f):
        return np.unpackbits(Mbits[int(pos[f])])[:n_reg].astype(bool)

    ctx = {"off_key": off_key, "dsup": dsup, "lid": lid, "h_s0": h_s0, "pool_on": pool_on, "pool_off": pool_off}

    # ---------------- full-map choice of (B, on_frac) by the S0 projection (INSTRUMENT)
    dcat = core.catalogue_distance_px()
    hm_full = s0_hmap(dcat, dsup, allowed, h_s0)
    grid_rows, best = [], None
    for B in GRID_B:
      for frac in GRID_FRAC:
        for s_on in GRID_SON:
            on_sel, off_sel, o, _ = build(dcat, allowed, B, frac, s_on, ctx, unpack, n_reg)
            dots = np.zeros(core.H * core.W, bool)
            dots[on_sel] = True
            dots[off_sel] = True
            pj = al.project(dots.reshape(core.H, core.W), hm_full)
            n = int(dots.sum())
            row = {"B_px": B, "on_frac": frac, "s_on_target": s_on, "n_on": int(len(on_sel)), "n_off": int(len(off_sel)),
                   "on_share": len(on_sel) / n, "worst_informative_overlap": float((o / n).max()),
                   "S0": pj}
            grid_rows.append(row)
            print(f"B={B} frac={frac} s_on={s_on}: on={len(on_sel)} off={len(off_sel)} worst={row['worst_informative_overlap']:.3f} "
                  f"S0={pj['dti_projection']:.4f}", flush=True)
            row["feasible"] = bool(row["worst_informative_overlap"] <= BUDGET + 1e-9)
            if row["feasible"] and (best is None or pj["dti_projection"] > best[0]["S0"]["dti_projection"]):
                best = (row, dots.copy())
    champ = core.read_prediction(core.REG / "17a76895f68174cc93f3cb1597d25686f6d6bc67.tif") > 0
    champ_pj = al.project(champ, hm_full)
    if best is None:
        raise SystemExit("no grid row meets the 0.66 overlap cap - stop (IR)")
    row, dots = best
    np.save(core.WORK / "h78_candidate_dots.npy", dots.reshape(core.H, core.W))
    # placement surface (phase-1 uniqueness): on-pool cells 0.5 + 0.5 * credit, off-pool 0.5 * consensus
    cr_full = al.credit_map(hm_full)
    surf = np.zeros(core.H * core.W, np.float32)
    okon = pool_on[allowed.ravel()[pool_on] & (dcat.ravel()[pool_on] > row["B_px"])]
    surf[okon] = 0.5 + 0.5 * cr_full.ravel()[okon] / max(cr_full.ravel()[okon].max(), 1e-12)
    okoff = pool_off[allowed.ravel()[pool_off] & (dcat.ravel()[pool_off] > 4)]
    surf[okoff] = 0.5 * lid.ravel()[okoff]
    np.save(core.WORK / "h78_surface.npy", surf.reshape(core.H, core.W))
    print(f"chosen B={row['B_px']} frac={row['on_frac']} s_on={row['s_on_target']} S0={row['S0']['dti_projection']:.4f} "
          f"vs champion S0={champ_pj['dti_projection']:.4f}", flush=True)

    # ---------------- holdout (fold-fair), canary
    grid = load_grid(core.DATA)
    folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    rng = np.random.default_rng(7)
    rand_key = rng.random(core.H * core.W)
    thermal = -np.load(core.WORK / "f_h78_thermal_km.npy")
    slope = np.nan_to_num(core.read_band(19), nan=0.0)
    terms = {k: [] for k in ("C", "A0", "L", "R")}
    receipts = []
    cnames = ("h19_5_support_proximity", "lidar_consensus_H78_1", "visible_catalogue_proximity",
              "S0_credit", "thermal_proximity_H78_2", "det_elev_slope")
    canary = {k: {"pos": [], "neg": []} for k in cnames}
    for f in folds:
        masked = f["masked_known"]
        valid = g.footprint & ~masked
        dvis = ndimage.distance_transform_edt(~f["visible"]).astype(np.float32)
        on_sel, off_sel, o, cr0 = build(dvis, valid, row["B_px"], row["on_frac"], row["s_on_target"], ctx, unpack, n_reg)
        C = np.zeros(core.H * core.W, np.float32)
        C[on_sel] = 1
        C[off_sel] = 1
        n_c = int(C.sum())
        L_ = (dsup == 0) & valid
        a_c = np.flatnonzero((L_ & (dvis > 2)).ravel())
        A0 = np.zeros(core.H * core.W, np.float32)
        A0[al.greedy_select(a_c[np.argsort(rand_key[a_c])], 3.0)] = 1
        l_c = np.flatnonzero((valid & (lid > 0) & (dvis > 4)).ravel())
        Lr = np.zeros(core.H * core.W, np.float32)
        Lr[al.greedy_select(l_c[np.argsort(-lid.ravel()[l_c], kind="stable")], 3.0, limit=n_c)] = 1
        r_c = np.flatnonzero(valid.ravel())
        R = np.zeros(core.H * core.W, np.float32)
        R[al.greedy_select(r_c[np.argsort(rand_key[r_c])][: n_c * 3], 3.0, limit=n_c)] = 1
        for k, arr in (("C", C), ("A0", A0), ("L", Lr), ("R", R)):
            _, t = eh.evaluate(arr.reshape(core.H, core.W), f, g.footprint)
            terms[k].append(t)
        region = f["region"] & g.footprint & ~masked
        truth = f["truth"] & region
        neg_idx = np.flatnonzero((region & ~truth).ravel())
        neg_idx = neg_idx[rng.choice(len(neg_idx), size=min(300_000, len(neg_idx)), replace=False)]
        pos_idx = np.flatnonzero(truth.ravel())
        feats = {"h19_5_support_proximity": -dsup, "lidar_consensus_H78_1": lid,
                 "visible_catalogue_proximity": -dvis, "S0_credit": cr0,
                 "thermal_proximity_H78_2": thermal, "det_elev_slope": slope}
        for k, a in feats.items():
            canary[k]["pos"].append(a.ravel()[pos_idx])
            canary[k]["neg"].append(a.ravel()[neg_idx])
        receipts.append({**f["receipt"], "candidate_dots": n_c, "candidate_on": int(len(on_sel)),
                         "candidate_off": int(len(off_sel)), "worst_registry_overlap": float((o / n_c).max()),
                         "A0_dots": int(A0.sum()), "truth_px": int(truth.sum())})
        print(f"fold {f['name']}: truth={int(truth.sum())} C={n_c} (on {len(on_sel)}, off {len(off_sel)}) "
              f"A0={int(A0.sum())}  ({time.time() - t0:.0f}s)", flush=True)
    summ = {}
    for k in ("A0", "L", "R"):
        summ[k] = eh.pooled_summary({"C": np.sum(terms["C"], axis=0), k: np.sum(terms[k], axis=0)},
                                    draws=2000, candidate="C")
    can = {k: {"auc": auc(np.concatenate(v["pos"]), np.concatenate(v["neg"]))} for k, v in canary.items()}
    for v in can.values():
        v["leakage_flag"] = bool(v["auc"] > 0.90)
    out = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": eh.VERSION,
           "rule": "H78 pre-registered default (registry/preregistration_h78.json e2_rule; field H78-1)",
           "split": "gems57.holdout.buffered_component_draw seed=20 hide_frac=0.20 buffer_px=3",
           "budget": BUDGET, "lidar_top_quantile": LIDAR_TOP_Q, "lidar_threshold": lid_q,
           "full_map_grid": grid_rows, "chosen": {"B_px": row["B_px"], "on_frac": row["on_frac"], "s_on_target": row["s_on_target"]},
           "full_map_candidate": {k: row[k] for k in ("n_on", "n_off", "on_share", "worst_informative_overlap", "S0")},
           "champion_S0_projection": champ_pj,
           "s0_delta_vs_champion": row["S0"]["dti_projection"] - champ_pj["dti_projection"],
           "fold_receipts": receipts, "pooled": summ, "leakage_canary": can,
           "implementation_sha256": eh.implementation_hashes(),
           "candidate_dots_sha256": hashlib.sha256(dots.tobytes()).hexdigest(),
           "runtime_s": round(time.time() - t0, 1)}
    (ROOT / "evidence" / "h78_e3_holdout.json").write_text(json.dumps(out, indent=1, default=float))
    c = summ["A0"]["scores"]["C"]
    print("C  HOLDOUT-DTI", round(c["dti"], 4), [round(x, 4) for x in c["ci95"]], "withheld", c["withheld_positive_pixels"])
    for k in ("A0", "L", "R"):
        s = summ[k]["scores"][k]
        d = summ[k]["paired_differences"][k]
        print(f"{k:3s} HOLDOUT-DTI {s['dti']:.4f} {[round(x, 4) for x in s['ci95']]}  C-{k} = {d['delta']:+.4f} "
              f"CI {[round(x, 4) for x in d['ci95']]}")
    print("canary", {k: round(v["auc"], 3) for k, v in can.items()})


if __name__ == "__main__":
    main()
