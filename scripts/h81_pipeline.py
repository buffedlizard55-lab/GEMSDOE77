#!/usr/bin/env python3
"""H81: slip-budget fault-tip continuation cones — features, canary, holdout, artifact.

Modes (run in order; each gate fails closed):
  surface           E1: frozen features + leakage canaries + surface uniqueness
  holdout           E2: pooled HOLDOUT-DTI (seed-20 folds) vs controls
  artifact          E3a: emit dots + shared-writer artifact (zeros primary + NaN twin)
  final-uniqueness  E3b: final-dot uniqueness vs registry + local TIFFs
  runcard           assemble evidence/h81_run_card.json (no computation)

Lane definition: registry/preregistration_h81.json (written before Exp 1).
Shared code: vendor/gems57 (pin-verified, unmodified) + src/gems77lia/allocate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor")]
from gems57 import evaluate_holdout as eh  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems57.holdout import buffered_component_draw  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems57.uniqueness import compare_array_to_registry  # noqa: E402
from gems77lia.allocate import greedy_select  # noqa: E402

DATA = ROOT / ".cache" / "gems_data"
WORK = ROOT / ".cache" / "work"
REGINDEX = ROOT / "registry" / "registry_index.json"
RECEIPT = ROOT / "registry" / "registry_receipt.json"

SEED = 20
BUDGET = 37_654
CONE_LEN_PX = 20
CONE_HALFANGLE_DEG = 30.0
LOCO_PX = 5
EDGE_GUARD_PX = 10  # SUPERSEDED by IR-81-04 (grid-truncation rule); kept for audit
CANARY_SEED = 81
CANARY_NEG_MAX = 300_000
LANE = "H81-slipcone"


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank01(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, np.float32)
    v = a[valid]
    r = rankdata(v, method="average").astype(np.float64)
    out[valid] = ((r - 1.0) / max(len(r) - 1, 1)).astype(np.float32)
    return out


def load_segments(grid) -> tuple[list[dict], dict]:
    """Parse + strand-split records (IR-81-04). Returns (tips, diagnostics).

    tips: list of dicts {r, c, dr, dc, slip, sense, record_id, comp_id} with
    float row/col of the tip pixel and unit outward direction (dr, dc).
    A tip is a degree-1 node of a connected component of the per-record
    segment graph (1 m rounding): trace ends, splay tips, gap-bounded strand
    ends. Segment joints (degree-2) are excluded; closed loops yield no tips.
    """
    path = DATA / "external" / "trace_segments_utm11.csv"
    expect = json.loads((ROOT / "registry" / "preregistration_h81.json").read_text())["inputs"]["trace_segments"]
    got = sha256_file(path)
    if got != expect["sha256"]:
        raise SystemExit(f"trace_segments sha mismatch: {got} != {expect['sha256']}")
    recs: dict[str, list[tuple]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            recs[row["record_id"]].append(
                (float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]),
                 float(row["slip_mm_yr"]), (row["sense"] or "").strip()))
    ox, oy = 243350.0, 4508550.0  # grid origin (matches sample transform)

    def key(x, y):
        return (round(x), round(y))

    def in_grid(r, c):
        # rounded-pixel bounds: every accepted coordinate must be indexable
        ri, ci = int(round(r)), int(round(c))
        return 0 <= ri < grid.shape[0] and 0 <= ci < grid.shape[1]

    tips = []
    diag = {"records": len(recs), "segments": sum(len(v) for v in recs.values()),
            "components": 0, "components_truncated_by_grid": 0,
            "components_closed_loop": 0, "tips_out_of_grid": 0}
    for rid, segs in recs.items():
        adj: dict[tuple, list[tuple]] = defaultdict(list)
        for i, (x0, y0, x1, y1, _s, _n) in enumerate(segs):
            a, b = key(x0, y0), key(x1, y1)
            adj[a].append((b, i))
            adj[b].append((a, i))
        # connected components over nodes (BFS), tracking member segment ids
        seen: set[tuple] = set()
        comp_id = 0
        for node in adj:
            if node in seen:
                continue
            stack, nodes, seg_ids = [node], [], set()
            seen.add(node)
            while stack:
                cur = stack.pop()
                nodes.append(cur)
                for nb, i in adj[cur]:
                    seg_ids.add(i)
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            deg1 = [v for v in nodes if len(adj[v]) == 1]
            if not deg1:
                diag["components_closed_loop"] += 1
                continue
            comp_slips = [segs[i][4] for i in seg_ids]
            comp_senses = [segs[i][5] for i in seg_ids]
            slip = max(comp_slips)
            sense = max(set(comp_senses), key=comp_senses.count)
            # grid-truncation rule (IR-81-04): any out-of-grid vertex -> drop
            out_grid = False
            for vx, vy in nodes:
                cc = (vx - ox) / 100.0
                rr = (oy - vy) / 100.0
                if not in_grid(rr, cc):
                    out_grid = True
                    break
            if out_grid:
                diag["components_truncated_by_grid"] += 1
                continue
            diag["components"] += 1
            for tip in deg1:
                # greedy walk up to 5 vertices into the component for the tangent
                used, walk = set(), [tip]
                cur = tip
                while len(walk) < 5:
                    nxt = next(((n, i) for n, i in adj[cur] if i not in used), None)
                    if nxt is None:
                        break
                    used.add(nxt[1])
                    walk.append(nxt[0])
                    cur = nxt[0]
                if len(walk) < 2:
                    continue
                ex, ey = walk[0]
                rx, ry = walk[-1]
                vx, vy = ex - rx, ey - ry
                norm = float(np.hypot(vx, vy))
                if norm == 0:
                    continue
                c = (ex - ox) / 100.0
                r = (oy - ey) / 100.0
                if not in_grid(r, c):
                    diag["tips_out_of_grid"] += 1
                    continue
                tips.append({"r": float(r), "c": float(c),
                             "dr": float(-(vy / norm)), "dc": float(vx / norm),
                             "slip": float(slip), "sense": sense, "record_id": rid,
                             "comp_id": comp_id, "walk_verts": len(walk),
                             "comp_segs": len(seg_ids),
                             "comp_nodes": [[int(v[0]), int(v[1])] for v in nodes]})
            comp_id += 1
    return tips, diag


def rasterize_cones(tips: list[dict], weights: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.float32)
    half = np.deg2rad(CONE_HALFANGLE_DEG)
    cos_half = float(np.cos(half))
    L = CONE_LEN_PX
    for t, w in zip(tips, weights):
        if w <= 0:
            continue
        r0, c0 = t["r"], t["c"]
        r_lo, r_hi = max(0, int(r0 - L - 1)), min(shape[0], int(r0 + L + 2))
        c_lo, c_hi = max(0, int(c0 - L - 1)), min(shape[1], int(c0 + L + 2))
        rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi].astype(np.float32)
        vr, vc = rr - r0, cc - c0
        d = np.hypot(vr, vc)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_a = np.where(d > 0, (vr * t["dr"] + vc * t["dc"]) / np.maximum(d, 1e-9), 1.0)
        m = (d <= L) & (cos_a >= cos_half)
        if not m.any():
            continue
        val = (w * (1.0 - d / L)).astype(np.float32)
        sub = out[r_lo:r_hi, c_lo:c_hi]
        np.maximum(sub, np.where(m, val, sub), out=sub)
    return out


def features() -> dict:
    """Build (or load cached) frozen H81 surfaces + metadata."""
    WORK.mkdir(parents=True, exist_ok=True)
    cache = WORK / "h81_features.npz"
    meta_p = WORK / "h81_features_meta.json"
    grid = load_grid(DATA)
    folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden_all = np.zeros(grid.shape, bool)
    for f in folds:
        hidden_all |= np.asarray(f["hidden_all"], bool)
    if cache.exists() and meta_p.exists():
        z = np.load(cache)
        meta = json.loads(meta_p.read_text())
        return {"arrays": {k: z[k] for k in z.files}, "meta": meta, "grid": grid, "folds": folds,
                "hidden_all": hidden_all}
    tips, diag = load_segments(grid)
    fp = np.asarray(grid.footprint, bool)
    d_hidden = distance_transform_edt(~hidden_all)
    # component-LOCO (IR-81-04): a tip is usable iff its tip pixel lies inside
    # the footprint AND no vertex of its own component lies within LOCO_PX of
    # hidden truth. (Grid-truncated components were already dropped in parsing.)
    usable, unusable_loco, unusable_off_fp = [], 0, 0
    for t in tips:
        r, c = int(round(t["r"])), int(round(t["c"]))
        if not fp[r, c]:
            unusable_off_fp += 1
            continue
        bad = d_hidden[r, c] < LOCO_PX
        if not bad:
            for vx, vy in t["comp_nodes"]:
                cc = (vx - 243350.0) / 100.0
                rr = (4508550.0 - vy) / 100.0
                ri, ci = int(round(rr)), int(round(cc))
                if d_hidden[ri, ci] < LOCO_PX:
                    bad = True
                    break
        if bad:
            unusable_loco += 1
            continue
        usable.append(t)
    slips = np.array([t["slip"] for t in usable], np.float64)
    w_slip = ((rankdata(slips, method="average") - 1.0) / max(len(usable) - 1, 1)).astype(np.float32)
    w_uni = np.ones(len(usable), np.float32)
    w_norm = np.where([t["sense"] == "N" for t in usable], w_slip, 0.0).astype(np.float32)
    surf_slip = rasterize_cones(usable, w_slip, grid.shape)
    surf_uni = rasterize_cones(usable, w_uni, grid.shape)
    surf_norm = rasterize_cones(usable, w_norm, grid.shape)
    surf_slip[~fp] = 0
    surf_uni[~fp] = 0
    surf_norm[~fp] = 0
    # scarp-gated sensitivity (topo10 rank gate; invalid q==0 -> 0)
    import rasterio
    with rasterio.open(DATA / "external" / "topo_10m_derived_100m_u8.tif") as ds:
        curv = ds.read(6).astype(np.float32)
        hsol = ds.read(8).astype(np.float32)
    gate = _rank01((curv + hsol) / 2.0, fp)
    surf_gate = (surf_slip * gate).astype(np.float32)
    # slip-proximity canary-only surface
    d_cat = distance_transform_edt(~np.asarray(grid.catalogue, bool))
    prox = np.zeros(grid.shape, np.float32)
    # nearest usable-record slip: approximate with tip Voronoi via tip raster
    tip_r = np.array([int(round(t["r"])) for t in usable])
    tip_c = np.array([int(round(t["c"])) for t in usable])
    seed_img = np.zeros(grid.shape, np.int32)
    seed_img[tip_r, tip_c] = np.arange(1, len(usable) + 1, dtype=np.int32)
    _, nearest = distance_transform_edt(seed_img == 0, return_indices=True)
    lab = seed_img[tuple(nearest)]
    with np.errstate(divide="ignore", invalid="ignore"):
        prox = np.where(lab > 0, (w_slip[lab - 1] / (1.0 + d_cat / 10.0)).astype(np.float32), 0.0)
    prox[~fp] = 0
    arrays = {"slip_cone": surf_slip.astype(np.float32), "tip_uniform": surf_uni.astype(np.float32),
              "normal_only": surf_norm.astype(np.float32), "scarp_gated": surf_gate.astype(np.float32),
              "slip_proximity": prox.astype(np.float32)}
    meta = {"tips_total": len(tips), "tips_usable": len(usable),
            "tips_unusable_off_footprint": unusable_off_fp,
            "tips_unusable_loco": unusable_loco,
            "records_total": diag["records"], "components": diag["components"],
            "components_truncated_by_grid": diag["components_truncated_by_grid"],
            "components_closed_loop": diag["components_closed_loop"],
            "tips_out_of_grid": diag["tips_out_of_grid"],
            "withheld_positive_pixels": int(hidden_all.sum()),
            "support_px": {k: int((v > 0).sum()) for k, v in arrays.items()},
            "seed": SEED}
    np.savez_compressed(cache, **arrays)
    _json(meta_p, meta)
    return {"arrays": arrays, "meta": meta, "grid": grid, "folds": folds, "hidden_all": hidden_all}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.concatenate([pos, neg])
    ranks = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def canaries(F: dict) -> dict:
    A, grid, folds = F["arrays"], F["grid"], F["folds"]
    rng = np.random.default_rng(CANARY_SEED)
    keys = ["slip_cone", "tip_uniform", "normal_only", "scarp_gated", "slip_proximity"]
    vals = {k: {"p": [], "n": []} for k in keys}
    for fold in folds:
        region = np.asarray(fold["region"], bool) & np.asarray(grid.footprint, bool) \
            & ~np.asarray(fold["masked_known"], bool)
        truth = np.asarray(fold["truth"], bool) & region
        neg = region & ~truth
        ni = np.flatnonzero(neg)
        ni = rng.choice(ni, min(CANARY_NEG_MAX, len(ni)), replace=False)
        pi = np.flatnonzero(truth)
        for k in keys:
            vals[k]["p"].append(A[k].ravel()[pi])
            vals[k]["n"].append(A[k].ravel()[ni])
    out = {}
    for k, v in vals.items():
        a = _auc(np.concatenate(v["p"]), np.concatenate(v["n"]))
        s = max(a, 1 - a)
        out[k] = {"auc": a, "symmetric_auc": s, "leakage_flag": bool(s > 0.90)}
    return out


def registry_index() -> dict:
    idx = json.loads(REGINDEX.read_text())
    receipt = json.loads(RECEIPT.read_text())["sha256_by_blob"]
    rasters = []
    for r in idx["rasters"]:
        blob = r["blob"]
        if blob not in receipt:
            continue
        rasters.append({"repo_first": r["repo"], "sources": r["aliases"],
                        "cache_file": str(ROOT / ".cache" / "registry" / f"{blob}.tif"),
                        "sha256": receipt[blob]})
    # extend with local TIFFs (this repo's own prior lanes) for a complete comparison.
    # The H81 candidate itself (+ its NaN twin, identical inside the footprint)
    # is NEVER a valid comparator: self-comparison would fake rho=1.0.
    own = set()
    bp = ROOT / "evidence/h81_e3_build.json"
    if bp.exists():
        b = json.loads(bp.read_text())["name"]
        own = {f"{b}.tif", f"{b[:-6]}-nan.tif" if b.endswith("-zeros") else ""}
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")) + sorted((ROOT / "submissions").glob("*.tif")):
        if p.name in own:
            continue
        rasters.append({"repo_first": "GEMSDOE77-local", "sources": [p.name],
                        "cache_file": str(p), "sha256": sha256_file(p)})
    n_local = (len(list((ROOT / "docs" / "downloads").glob("*.tif")))
               + len(list((ROOT / "submissions").glob("*.tif"))) - len(own))
    return {"rasters": rasters,
            "complete_accessible_scan": len(rasters) == len(receipt) + n_local,
            "errors": [], "n_registry": len(receipt), "n_local": n_local}


def uniqueness(arr: np.ndarray, phase: str) -> dict:
    F_grid = load_grid(DATA)
    arr = np.asarray(arr, np.float32)
    arr = np.where(np.asarray(F_grid.footprint, bool), arr, 0).astype(np.float32)
    if arr.max() > 0:
        arr = (arr / arr.max()).astype(np.float32)

    def progress(i, n, row):
        if i % 100 == 0 or i == n:
            print(f"{phase}: {i}/{n}", flush=True)
    report = compare_array_to_registry(arr, registry_index(), np.asarray(F_grid.footprint, bool),
                                       include_reverse=(phase != "surface"), progress=progress,
                                       expected_grid=(F_grid.shape, F_grid.crs, F_grid.transform))
    report["phase"] = phase
    return report


def emit(surface: np.ndarray, allowed: np.ndarray, limit: int) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(allowed, bool) & (surface > 0))
    order = idx[np.argsort(-surface.ravel()[idx], kind="stable")]
    sel = greedy_select(order, 3.0, limit=limit)
    p = np.zeros(surface.shape, np.float32)
    p.ravel()[np.asarray(sel)] = 1
    return p


def run_surface() -> None:
    F = features()
    can = canaries(F)
    uq = uniqueness(F["arrays"]["slip_cone"], "surface")
    max_abs_rho = max((abs(r["spearman_full_footprint"]) for r in uq["rows"]
                       if r.get("spearman_full_footprint") is not None), default=0.0)
    # E1 capacity dry-run: primary emission per fold at full budget
    grid, folds = F["grid"], F["folds"]
    caps = {}
    for f in folds:
        allowed = np.asarray(grid.footprint, bool) & ~np.asarray(f["masked_known"], bool)
        p = emit(F["arrays"]["slip_cone"], allowed, BUDGET)
        caps[f["name"]] = int((p > 0).sum())
    out = {"experiment": "H81-E1",
           "feature_definition": json.loads((ROOT / "registry/preregistration_h81.json").read_text())["fixed_transform"],
           "feature_meta": F["meta"],
           "leakage_canary": can,
           "surface_max_abs_spearman": max_abs_rho,
           "surface_duplicate": bool(max_abs_rho > 0.90),
           "capacity_at_budget": caps,
           "registry": uq}
    _json(ROOT / "evidence/h81_e1_surface.json", out)
    print(json.dumps({"canary_max": max(v["symmetric_auc"] for v in can.values()),
                      "surface_max_abs_spearman": max_abs_rho,
                      "duplicate": out["surface_duplicate"], "capacity": caps}, indent=2))
    if out["surface_duplicate"]:
        raise SystemExit("STOP: H81 surface duplicates a registry raster")
    if any(v["leakage_flag"] for v in can.values()):
        raise SystemExit("STOP: H81 leakage canary fired")


def resolve_budget(F: dict) -> int:
    grid, folds = F["grid"], F["folds"]
    mn = BUDGET
    for f in folds:
        allowed = np.asarray(grid.footprint, bool) & ~np.asarray(f["masked_known"], bool)
        p = emit(F["arrays"]["slip_cone"], allowed, BUDGET)
        mn = min(mn, int((p > 0).sum()))
    full_allowed = np.asarray(grid.footprint, bool) & ~np.asarray(grid.catalogue, bool)
    p = emit(F["arrays"]["slip_cone"], full_allowed, BUDGET)
    mn = min(mn, int((p > 0).sum()))
    if mn < BUDGET:
        return max(100, (mn // 100) * 100)
    return BUDGET


def run_holdout() -> None:
    e1 = json.loads((ROOT / "evidence/h81_e1_surface.json").read_text())
    if e1["surface_duplicate"] or any(v["leakage_flag"] for v in e1["leakage_canary"].values()):
        raise SystemExit("E1 gate failed; no holdout")
    F = features()
    grid, folds, A = F["grid"], F["folds"], F["arrays"]
    B = resolve_budget(F)
    rng = np.random.default_rng(8101)
    random_key = rng.random(grid.shape).astype(np.float32)
    arms = {"H81_slip": A["slip_cone"], "tip_uniform": A["tip_uniform"],
            "normal_only": A["normal_only"], "scarp_gated": A["scarp_gated"],
            "random": random_key}
    terms = {k: [] for k in arms}
    receipts = []
    for fold in folds:
        allowed = np.asarray(grid.footprint, bool) & ~np.asarray(fold["masked_known"], bool)
        for k, s in arms.items():
            p = emit(s, allowed, B)
            if int((p > 0).sum()) != B:
                raise RuntimeError(f"arm {k} fold {fold['name']}: {int((p>0).sum())} != {B}")
            _, t = eh.evaluate(p, fold, np.asarray(grid.footprint, bool))
            terms[k].append(t)
        receipts.append({**fold["receipt"], "dots_per_arm": B})
    pooled = {k: np.sum(v, axis=0) for k, v in terms.items()}
    summary = eh.pooled_summary(pooled, draws=2000, candidate="H81_slip")
    out = {"experiment": "H81-E2", "evidence_class": "HOLDOUT-DTI",
           "evaluator_version": eh.VERSION, "fold_receipts": receipts,
           "budget_dots": B, "summary": summary,
           "implementation_sha256": eh.implementation_hashes(),
           "current_holdout_best": {"score": 0.152718, "source": "evidence/h77/run_card.json"}}
    _json(ROOT / "evidence/h81_e2_holdout.json", out)
    print(json.dumps({"budget": B, "summary": summary}, indent=2))


def artifact_name(B: int, dot_hash8: str, twin: str) -> str:
    return f"gems77-h81-slipcone-{B}dots-20261021-{dot_hash8}-{twin}"


def run_artifact() -> None:
    e1 = json.loads((ROOT / "evidence/h81_e1_surface.json").read_text())
    if e1["surface_duplicate"]:
        raise SystemExit("surface duplicate: protocol requires stop")
    e2 = json.loads((ROOT / "evidence/h81_e2_holdout.json").read_text())
    B = int(e2["budget_dots"])
    F = features()
    grid = F["grid"]
    # full-catalogue surface: all in-footprint strand tips (no hidden), IR-81-04
    tips_all, _ = load_segments(grid)
    fp = np.asarray(grid.footprint, bool)
    use = [t for t in tips_all if fp[int(round(t["r"])), int(round(t["c"]))]]
    slips = np.array([t["slip"] for t in use], np.float64)
    w = ((rankdata(slips, method="average") - 1.0) / max(len(use) - 1, 1)).astype(np.float32)
    surf = rasterize_cones(use, w, grid.shape)
    surf[~fp] = 0
    pred = emit(surf, fp & ~np.asarray(grid.catalogue, bool), B)
    if int((pred > 0).sum()) != B:
        raise RuntimeError(f"artifact emission {int((pred > 0).sum())} != budget {B}")
    dot8 = hashlib.sha256(np.ascontiguousarray(np.flatnonzero(pred > 0)).tobytes()).hexdigest()[:8]
    dti = float(e2["summary"]["scores"]["H81_slip"]["dti"])
    lo, hi = e2["summary"]["scores"]["H81_slip"]["ci95"]
    note = (f"H81 slip-budget tip cones {B} dots 0-on-cat; HOLDOUT-DTI {dti:.4f} [{lo:.4f},{hi:.4f}]")[:140]
    name = artifact_name(B, dot8, "zeros")
    out = ROOT / "submissions" / f"{name}.tif"
    for p in (out, out.with_suffix(".zip"), out.with_suffix(".json")):
        p.unlink(missing_ok=True)
    receipt = write_submission(out, pred, DATA / "sample_submission.tif", fp,
                               catalogue=np.asarray(grid.catalogue, bool),
                               name=name, note=note,
                               metadata={"lane": LANE, "selector_approved": False,
                                         "budget_dots": B, "withheld_positives": F["meta"]["withheld_positive_pixels"]})
    for p in (out, out.with_suffix(".zip"), out.with_suffix(".json")):
        shutil.copy2(p, ROOT / "docs" / "downloads" / p.name)
    # NaN-outside reference twin (direct write; NOT via the fail-closed writer)
    import rasterio
    twin_name = artifact_name(B, dot8, "nan")
    twin = ROOT / "docs" / "downloads" / f"{twin_name}.tif"
    with rasterio.open(out) as src:
        prof = src.profile.copy()
        arr = src.read(1)
    prof.update(nodata=float("nan"))
    twin_arr = np.where(fp, arr, np.nan).astype(np.float32)
    with rasterio.open(twin, "w", **prof) as dst:
        dst.write(twin_arr, 1)
    _json(ROOT / "evidence/h81_e3_build.json",
          {"name": name, "note": note, "budget_dots": B, "dot_hash8": dot8,
           "receipt": receipt, "nan_twin": twin.name, "nan_twin_sha256": sha256_file(twin),
           "full_tips_used": len(use), "full_tips_total": len(tips_all)})
    print(json.dumps({"name": name, "note": note, "twin": twin.name}, indent=2))


def run_final_uniqueness() -> None:
    import rasterio
    build = json.loads((ROOT / "evidence/h81_e3_build.json").read_text())
    with rasterio.open(ROOT / "submissions" / f"{build['name']}.tif") as src:
        arr = src.read(1)
    uq = uniqueness(arr, "final-dots")
    _json(ROOT / "evidence/h81_e3_uniqueness.json", uq)
    print(json.dumps({k: uq[k] for k in ("registry_rasters_checked", "worst_spearman_full_footprint",
                                        "worst_dot_overlap", "duplicate_count", "unique",
                                        "stop_required")}, indent=2))


def run_runcard() -> None:
    e1 = json.loads((ROOT / "evidence/h81_e1_surface.json").read_text())
    e2 = json.loads((ROOT / "evidence/h81_e2_holdout.json").read_text())
    build = json.loads((ROOT / "evidence/h81_e3_build.json").read_text())
    uq = json.loads((ROOT / "evidence/h81_e3_uniqueness.json").read_text())
    best = float(e2["current_holdout_best"]["score"])
    arm = e2["summary"]["scores"]["H81_slip"]
    # Amendment IR-81-03: the incumbent is a historical number, not a re-runnable
    # arm, so no paired interval vs the incumbent exists. "Beats" is conservatively
    # defined as H81 lower 95% CI above the incumbent point estimate; paired
    # intervals vs same-run controls are reported alongside.
    beats = bool(arm["ci95"][0] > best)
    literal_ok = not bool(uq["stop_required"])
    canary_ok = not any(v["leakage_flag"] for v in e1["leakage_canary"].values())
    verdict = "promote" if (beats and literal_ok and canary_ok) else "negative"
    card = {
        "hypothesis": "Slip-budget fault-tip continuation: strand-level fault tips emit slip-rate-weighted propagation cones; hidden faults continue/transfer beyond mapped tips.",
        "mechanism": "Displacement conservation + d-L scaling (Cowie & Scholz 1992) require continuation of mapped slip; tip stress concentration (Pollard & Segall 1987) localises it ahead of the tip. Deterministic cones, no ML.",
        "named_non_fault_process_that_could_mimic_it": "Grid-truncation artifacts (dropped by rule); mapping-gap strand ends from erosion/cover rather than fault termination; en-echelon overlaps already linked in the catalogue.",
        "holdout_dti": {"label": "HOLDOUT-DTI", "evidence_class": "HOLDOUT-DTI",
                        "evaluator_version": e2["evaluator_version"],
                        "withheld_positives": e1["feature_meta"]["withheld_positive_pixels"],
                        "value": arm["dti"], "dti": arm["dti"], "ci95": arm["ci95"],
                        "withheld_positive_pixels": e1["feature_meta"]["withheld_positive_pixels"],
                        "incumbent": best, "beats_current_best": bool(beats),
                        "beats_rule": "H81 ci95[0] > incumbent (IR-81-03; no paired cross-lane interval exists)",
                        "paired_differences_vs_same_run_controls": e2["summary"]["paired_differences"]},
        "correlation_overlap_vs_registry": {
            "surface_max_abs_spearman": e1["surface_max_abs_spearman"],
            "final_max_abs_spearman": uq["worst_spearman_full_footprint"],
            "final_worst_forward_3px_overlap": uq["worst_dot_overlap"],
            "final_dots_max_forward_overlap_3px": uq["worst_dot_overlap"],
            "final_worst_overlap_submission": uq["worst_overlap_submission"],
            "registry_rasters_checked": uq["registry_rasters_checked"],
            "complete_accessible_scan": uq["complete_accessible_scan"],
            "duplicate_count": uq["duplicate_count"], "unique": uq["unique"],
            "stop_required": uq["stop_required"],
            "duplicate_stop": bool(uq["stop_required"])},
        "registry": {"duplicate_count": uq["duplicate_count"],
                     "complete": bool(uq["complete_accessible_scan"]),
                     "checked": uq["registry_rasters_checked"],
                     "missing_or_invalid": uq.get("missing_or_invalid", 0)},
        "raster_sha256": sha256_file(ROOT / "submissions" / f"{build['name']}.tif"),
        "validator": build["receipt"],
        "validator_output": {"all_checks_passed": bool(build["receipt"]["validator"]["all_checks_passed"])},
        "submission_name": build["name"],
        "submission_note": build["note"],
        "submission_slots_used": 0,
        "verdict": verdict,
        "selector_approved": False,
        "safe_to_submit": False,
        "ok_to_download": True,
        "ok_to_submit": False,
        "protocol_notes": [
            "Surface-stage gate was rank-correlation (preregistered, H79-RH precedent): surface overlap vs full-footprint dense witnesses is 1.0 for ANY non-empty support and carries no lane-drift information (see reports/template_fix_dot_overlap_proposal.md). Drift was judged by rho at surface stage, rho+overlap at final-dot stage.",
            "slip_proximity (symmetric AUC 0.81) is a canary-only positive control: it uses full-catalogue distance (including hidden components), so its high AUC demonstrates canary power, not arm leakage. No arm or artifact surface uses it; all arm surfaces are component-LOCO with AUC 0.56-0.59.",
            "The final artifact uses the frozen primary (slip_cone), NOT the best holdout arm (scarp_gated, 0.0804): switching post-hoc would be tuning on holdout outcomes."
        ],
    }
    _json(ROOT / "evidence/h81_run_card.json", card)
    print(json.dumps({k: card[k] for k in ("verdict", "submission_name", "submission_note")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("surface", "holdout", "artifact", "final-uniqueness", "runcard"))
    a = ap.parse_args()
    {"surface": run_surface, "holdout": run_holdout, "artifact": run_artifact,
     "final-uniqueness": run_final_uniqueness, "runcard": run_runcard}[a.mode]()


if __name__ == "__main__":
    main()