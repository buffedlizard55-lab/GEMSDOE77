#!/usr/bin/env python3
"""H83-NETSIB 3-experiment pipeline (registry/preregistration_h83_netsib.json).

exp1: leakage canaries (seeds 20/30) + pre-placement surface Spearman of the
      label-free NETSIB score vs every fetched registry raster.
exp2: paired hide-and-recover HOLDOUT-DTI with the UNMODIFIED vendor evaluator
      (gems57-pooled-hide-v2), seeds 20/30; arms A_netsib_full (candidate),
      A_base, S_netsib_only, A0_random; identical emission for every arm.
exp3: full-catalogue LOQO surface -> surface Spearman -> 37,654-dot GeoTIFF via
      the unmodified vendor writer -> validator (run card built separately).

Catalogue use: only through gems83.netsib on a catalogue-STATE mask (visible,
synthetic re-withheld, or full for the final build), as labels of synthetic
withheld components, as the > 3 px negative rule, and as the writer mask.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems77lia.allocate import greedy_select  # noqa: E402
from gems83 import netsib as NS  # noqa: E402

RAW = ROOT / "data" / "raw"
BRIDGE = ROOT / "data" / "bridge"
PROC = ROOT / "data" / "processed" / "h83"
EVID = ROOT / "evidence"
BUDGET = 37_654
MIN_SEP = 2.8
EXCL_PX = 2.0
INCUMBENT = 0.152718   # evidence/h77/run_card.json, gems57-pooled-hide-v2 seed 20, 11,321 positives
N_SYN = 2
HGB_KW = dict(max_iter=160, learning_rate=0.075, max_leaf_nodes=45,
              min_samples_leaf=50, l2_regularization=2.5,
              early_stopping=False, random_state=83083)
PINS = {
    "training_features.tif": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
    "external/lidar_scarp_features_u8.tif": "d580bb8bdcdb941e32fefb8b38044bc5bf04e199bf2e83498c3576e6fc465568",
    "external/geodawn_rad_u8.tif": "c22420f75999030d7cc65c9e31e50d232ea6158423bca051613a18a8b20ba682",
    "external/geodawn_extensions_u8.tif": "a35a9c6d2a14786f4dab85481ee59769213072f5dab5b2535ea82ae4d9bb7d9b",
}
STD_NAMES = ["mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
             "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15", "iso_grav_anom_vg",
             "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf", "ieq_n100a15",
             "cond_surf", "iso_grav_anom_hg", "det_elev_slope"]
LIDAR_NAMES = ["lidar_ex_max", "lidar_ex_mean", "lidar_step_max", "lidar_lapneg_max",
               "lidar_lappos_max", "lidar_downface_max", "lidar_upface_max", "lidar_cross_max",
               "lidar_relief", "lidar_coh100", "lidar_strike", "lidar_valid"]
RAD_NAMES = ["rad_K", "rad_Th", "rad_U", "rad_TC", "rad_ThK", "rad_UK", "rad_UTh", "tmi_up150"]
BASE_NAMES = STD_NAMES + LIDAR_NAMES + RAD_NAMES


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_grid():
    return G.load_grid(BRIDGE)


def base_matrix(foot: np.ndarray) -> np.ndarray:
    """Footprint-row float32 matrix of the 39 base channels (cached, gitignored)."""
    PROC.mkdir(parents=True, exist_ok=True)
    p = PROC / "base39_fp.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    for rel, pin in PINS.items():
        got = sha256_file(RAW / rel)
        if got != pin:
            raise SystemExit(f"input hash mismatch {rel}: {got}")
    n = int(foot.sum())
    X = np.lib.format.open_memmap(p.with_suffix(".tmp.npy"), mode="w+", dtype=np.float32,
                                  shape=(n, len(BASE_NAMES)))
    j = 0
    with rasterio.open(RAW / "training_features.tif") as src:
        nod = src.nodata
        for b in range(1, 20):
            a = src.read(b).astype(np.float32)
            bad = ~np.isfinite(a) | (a <= -1e30) | ((a == nod) if nod is not None else False)
            a[bad] = np.nan              # HGB handles NaN natively
            X[:, j] = a[foot]
            j += 1
    for rel, nb in (("external/lidar_scarp_features_u8.tif", 12),
                    ("external/geodawn_rad_u8.tif", 4),
                    ("external/geodawn_extensions_u8.tif", 4)):
        with rasterio.open(RAW / rel) as src:
            for b in range(1, nb + 1):
                a = src.read(b).astype(np.float32)
                if not (rel.startswith("external/lidar") and b == 12):
                    a[a == 0] = np.nan   # uint8 nodata = 0 (except the 'valid' flag band)
                X[:, j] = a[foot]
                j += 1
    assert j == len(BASE_NAMES)
    X.flush()
    del X
    p.with_suffix(".tmp.npy").rename(p)
    return np.load(p, mmap_mode="r")


def ns_matrix(state: np.ndarray, foot: np.ndarray, tag: str) -> np.ndarray:
    """Footprint-row NETSIB matrix for one catalogue state (cached by tag)."""
    p = PROC / f"ns_{tag}.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    f = NS.netsib_fields(state, foot)
    M = np.empty((int(foot.sum()), len(NS.NAMES)), np.float32)
    for j, nm in enumerate(NS.NAMES):
        M[:, j] = f[nm][foot]
    np.save(p, M)
    del f, M
    gc.collect()
    return np.load(p, mmap_mode="r")


def fp_lut(foot):
    flat = np.flatnonzero(foot)
    lut = np.full(foot.size, -1, np.int64)
    lut[flat] = np.arange(flat.size)
    return flat, lut


def emit(surface: np.ndarray, allowed: np.ndarray, limit: int = BUDGET) -> np.ndarray:
    """Greedy Poisson-disk dotted emission (min-sep 2.8 px), identical for all arms."""
    surf = np.where(allowed, surface, 0.0)
    idx = np.flatnonzero(surf > 0)
    vals = surf.ravel()[idx]
    cap = min(idx.size, 600_000)
    if idx.size > cap:
        top = np.argpartition(-vals, cap - 1)[:cap]
        idx, vals = idx[top], vals[top]
    order = idx[np.argsort(-vals, kind="stable")]
    sel = greedy_select(order, MIN_SEP, limit=limit)
    if sel.size != limit:
        raise RuntimeError(f"emission reached {sel.size} != {limit}")
    p = np.zeros(surface.shape, np.float32)
    p.ravel()[sel] = 1.0
    return p


# ----------------------------------------------------------------- training rows
def training_rows(foot, cat_all, train_region, sets, lut, rng, pos_cap=40_000):
    """Rows for one LOQO model.

    ``sets`` = list of (hidden_mask, ns_matrix) synthetic catalogue states.
    Positives = synthetic-hidden pixels inside ``train_region``; negatives =
    train_region pixels > 3 px from ``cat_all`` (half within 15 px), drawn per
    state in a 2:1 ratio and represented with that state's NETSIB features.
    Returns list of (row_idx, y, set_id).
    """
    d_cat = ndimage.distance_transform_edt(~cat_all).astype(np.float32)
    neg_base = foot & train_region & (d_cat > 3.0)
    hard = lut[np.flatnonzero(neg_base & (d_cat <= 15.0))]
    easy = lut[np.flatnonzero(neg_base & (d_cat > 15.0))]
    out = []
    per_set_cap = pos_cap // len(sets)
    for sid, (hidden, _M) in enumerate(sets):
        pos = lut[np.flatnonzero(hidden & train_region & foot)]
        if pos.size > per_set_cap:
            pos = rng.choice(pos, per_set_cap, replace=False)
        n_neg = 2 * pos.size
        nh = min(hard.size, n_neg // 2)
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
        out.append((np.concatenate([pos, neg]),
                    np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)]), sid))
    return out


def gather(X_base, rows, M=None, cols="full"):
    b = np.asarray(X_base[np.sort(rows)], np.float32)
    order = np.argsort(np.argsort(rows))
    b = b[order]
    if cols == "base":
        return b
    n = np.asarray(M[np.sort(rows)], np.float32)[order]
    return n if cols == "ns" else np.hstack([b, n])


def fit_predict(X_base, train_parts, sets, test_rows, M_test, cols):
    Xs, ys = [], []
    for rows, y, sid in train_parts:
        Xs.append(gather(X_base, rows, sets[sid][1], cols))
        ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    clf = HistGradientBoostingClassifier(**HGB_KW).fit(X, y)
    del X, Xs
    preds = np.empty(test_rows.size, np.float32)
    for s in range(0, test_rows.size, 200_000):
        r = test_rows[s:s + 200_000]
        preds[s:s + 200_000] = clf.predict_proba(gather(X_base, r, M_test, cols))[:, 1]
    return preds, int(y.sum()), int(y.size)


# ----------------------------------------------------------------------- exp1
def registry_files():
    rec = json.loads((ROOT / "registry" / "registry_receipt.json").read_text())["sha256_by_blob"]
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())["rasters"]
    files = []
    for e in idx:
        p = ROOT / ".cache" / "registry" / f"{e['blob']}.tif"
        if p.exists():
            files.append((f"{e['repo']}:{e['path']}", p, rec.get(e["blob"])))
    for extra in ("dense_witness.tif", "h33.tif"):
        p = ROOT / ".cache" / "registry" / extra
        if p.exists():
            files.append((f"witness:{extra}", p, sha256_file(p)))
    for p in sorted((ROOT / "submissions").glob("*.tif")) + sorted((ROOT / "docs" / "downloads").glob("*.tif")):
        files.append((f"local:{p.relative_to(ROOT)}", p, None))
    return files


def surface_scan(surface, foot, n_sample=300_000, seed=8383):
    """Spearman of ``surface`` vs every registry raster on a FIXED random sample
    of footprint pixels (declared approximation; SE ~0.002). Rasters whose
    sampled |rho| > 0.80 are recomputed exactly on the full footprint."""
    from scipy.stats import rankdata, spearmanr
    fl = np.flatnonzero(foot)
    samp = np.sort(np.random.default_rng(seed).choice(fl, n_sample, replace=False))
    mine = surface.ravel()[samp].astype(np.float64)
    mr = rankdata(mine); mr -= mr.mean(); mn = np.linalg.norm(mr)
    rows, seen = [], set()
    for label, p, sha in registry_files():
        try:
            with rasterio.open(p) as src:
                if src.count != 1 or src.shape != foot.shape:
                    rows.append({"raster": label, "skipped": "grid mismatch"}); continue
                t = src.read(1).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            rows.append({"raster": label, "error": str(e)}); continue
        t = np.where(np.isfinite(t), t, 0.0)
        key = hashlib.sha256(t.tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        tv = t.ravel()[samp].astype(np.float64)
        if np.ptp(tv) == 0:
            rows.append({"raster": label, "rho_sampled": None}); continue
        tr = rankdata(tv); tr -= tr.mean()
        rho = float(np.dot(mr, tr) / (mn * np.linalg.norm(tr)))
        row = {"raster": label, "rho_sampled": rho}
        if abs(rho) > 0.80:
            row["rho_exact_full_footprint"] = float(spearmanr(surface[foot], t[foot]).statistic)
        rows.append(row)
    vals = [(abs(r.get("rho_exact_full_footprint", r["rho_sampled"])), r["raster"])
            for r in rows if r.get("rho_sampled") is not None]
    worst = max(vals) if vals else (None, None)
    return {"method": f"Spearman on fixed {n_sample} random footprint pixels (seed {seed}); exact recompute if |rho|>0.80",
            "rasters_compared_unique_decoded": len(vals), "max_abs_rho": worst[0],
            "worst_raster": worst[1], "gate_limit": 0.90, "pass": bool(worst[0] is not None and worst[0] <= 0.90),
            "rows": rows}


def run_exp1():
    t0 = time.time()
    grid = load_grid(); foot, cat = grid.footprint, grid.catalogue
    X = base_matrix(foot)
    flat, lut = fp_lut(foot)
    d_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    res = {}
    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden, visible = folds[0]["hidden_all"], folds[0]["visible"]
        M = ns_matrix(visible, foot, f"s{seed}_vis")
        rng = np.random.default_rng(8300 + seed)
        pos = lut[np.flatnonzero(hidden & foot)]
        negpool = lut[np.flatnonzero(foot & ~cat & (d_cat > 3.0))]
        neg = rng.choice(negpool, 200_000, replace=False)
        rows = np.concatenate([pos, neg]); y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
        A = {}
        Xr = gather(X, rows, M, "full")
        for j, nm in enumerate(BASE_NAMES + list(NS.NAMES)):
            v = Xr[:, j]; ok = np.isfinite(v)
            if ok.sum() > 100 and np.ptp(v[ok]) > 0:
                a = float(roc_auc_score(y[ok], v[ok])); A[nm] = max(a, 1 - a)
            else:
                A[nm] = None
        mx = max((v, k) for k, v in A.items() if v is not None)
        res[str(seed)] = {"withheld_positive_pixels": int(pos.size), "negatives": int(neg.size),
                          "symmetric_auc": A, "max_auc": mx[0], "max_feature": mx[1],
                          "leakage_flag_gt_0p90": bool(mx[0] > 0.90)}
        log(f"seed {seed}: max canary AUC {mx[0]:.4f} ({mx[1]})")
    fields_full = NS.netsib_fields(cat, foot)
    det = NS.deterministic_score(fields_full, foot)
    np.save(PROC / "det_score_full.npy", det)
    scan = surface_scan(det, foot)
    log(f"surface scan max|rho|={scan['max_abs_rho']} vs {scan['worst_raster']} n={scan['rasters_compared_unique_decoded']}")
    out = {"experiment": "H83-E1", "evidence_class": "CANARY-AND-REGISTRY-MEASUREMENT",
           "canaries": res, "leakage_ceiling": 0.90,
           "leakage_pass": not any(v["leakage_flag_gt_0p90"] for v in res.values()),
           "deterministic_netsib_surface_scan": scan, "elapsed_s": round(time.time() - t0, 1)}
    (EVID / "h83_e1_canary_surface.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ----------------------------------------------------------------------- exp2
def run_exp2(seeds=(20, 30)):
    t0 = time.time()
    grid = load_grid(); foot, cat = grid.footprint, grid.catalogue
    X = base_matrix(foot)
    flat, lut = fp_lut(foot)
    quad = HO.quadrant_ids(foot)
    out_seeds = {}
    for seed in seeds:
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        visible, hidden_all = folds[0]["visible"], folds[0]["hidden_all"]
        M_vis = ns_matrix(visible, foot, f"s{seed}_vis")
        sets = []
        for r in range(N_SYN):
            syn_hidden, syn_state = NS.synthetic_withhold(visible, foot, quad, seed=seed * 10 + r)
            sets.append((syn_hidden, ns_matrix(syn_state, foot, f"s{seed}_syn{r}")))
            log(f"seed {seed} syn{r}: hidden px {int(syn_hidden.sum())}")
        # Negatives must avoid every catalogue pixel known to a model of fold k:
        # visible + synthetic hidden + real hidden of the other quadrants. The
        # train region excludes fold k, so hidden_all is safe to use here.
        surfs = {a: np.zeros(foot.shape, np.float32) for a in ("A_netsib_full", "A_base", "S_netsib_only")}
        diag = []
        for k, fold in enumerate(folds):
            tr_reg = np.zeros_like(foot)
            for j, f2 in enumerate(folds):
                if j != k:
                    tr_reg |= f2["region"]
            rng = np.random.default_rng(8800 + 10 * seed + k)
            parts = training_rows(foot, cat & tr_reg | visible, tr_reg, sets, lut, rng)
            te_rows = lut[np.flatnonzero(fold["region"] & foot)]
            for arm, cols in (("A_netsib_full", "full"), ("A_base", "base"), ("S_netsib_only", "ns")):
                pr, npos, nrows = fit_predict(X, parts, sets, te_rows, M_vis, cols)
                surfs[arm].ravel()[flat[te_rows]] = pr
                diag.append({"fold": fold["name"], "arm": arm, "train_pos": npos, "train_rows": nrows})
                log(f"seed {seed} fold {fold['name']} {arm} done (pos {npos}, rows {nrows})")
                gc.collect()
        surfs["A0_random"] = (np.random.default_rng(999 + seed).random(foot.shape).astype(np.float32) * foot)
        dist_known = ndimage.distance_transform_edt(~folds[0]["masked_known"]).astype(np.float32)
        allowed = foot & (dist_known > EXCL_PX)
        terms = {}
        for name, s in surfs.items():
            np.save(PROC / f"oof_s{seed}_{name}.npy", s)
            p = emit(s, allowed)
            acc = None
            for fold in folds:
                _, t = EH.evaluate(p, fold, foot)
                acc = t if acc is None else acc + t
            terms[name] = acc
        summ = EH.pooled_summary(terms, draws=2000, candidate="A_netsib_full")
        out_seeds[str(seed)] = {"summary": summ, "fold_receipts": [f["receipt"] for f in folds],
                                "train_diag": diag, "budget_dots": BUDGET}
        sc = summ["scores"]
        log(f"seed {seed}: " + " ".join(f"{a}={sc[a]['dti']:.4f}" for a in sc))
        (EVID / "h83_e2_holdout.partial.json").write_text(json.dumps(out_seeds, indent=1) + "\n")
    out = {"experiment": "H83-E2", "evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
           "seeds": out_seeds, "implementation_sha256": EH.implementation_hashes(),
           "incumbent": {"score": INCUMBENT, "source": "evidence/h77/run_card.json (seed 20, 11,321 withheld)"},
           "elapsed_s": round(time.time() - t0, 1)}
    (EVID / "h83_e2_holdout.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ----------------------------------------------------------------------- exp3
def run_exp3():
    """Full-catalogue LOQO surface + emission (writer/validator in h83_build.py)."""
    t0 = time.time()
    grid = load_grid(); foot, cat = grid.footprint, grid.catalogue
    X = base_matrix(foot)
    flat, lut = fp_lut(foot)
    quad = HO.quadrant_ids(foot)
    M_full = ns_matrix(cat, foot, "full")
    sets = []
    for r in range(N_SYN):
        syn_hidden, syn_state = NS.synthetic_withhold(cat, foot, quad, seed=9000 + r)
        sets.append((syn_hidden, ns_matrix(syn_state, foot, f"full_syn{r}")))
    regions = [ndimage.binary_erosion(quad == q, iterations=HO.DOMAIN_ERODE) & foot for q in range(4)]
    surf = np.zeros(foot.shape, np.float32)
    for k in range(4):
        tr_reg = np.zeros_like(foot)
        for j in range(4):
            if j != k:
                tr_reg |= regions[j]
        rng = np.random.default_rng(9900 + k)
        parts = training_rows(foot, cat, tr_reg, sets, lut, rng)
        te_rows = lut[np.flatnonzero((quad == k) & foot)]
        pr, npos, nrows = fit_predict(X, parts, sets, te_rows, M_full, "full")
        surf.ravel()[flat[te_rows]] = pr
        log(f"final fold {k} done (pos {npos}, rows {nrows})")
    assert np.isfinite(surf).all() and surf.min() >= 0 and surf.max() <= 1
    np.save(PROC / "final_surface.npy", surf)
    scan = surface_scan(surf, foot)
    log(f"final surface scan max|rho|={scan['max_abs_rho']} vs {scan['worst_raster']}")
    d_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    p = emit(surf, foot & (d_cat > EXCL_PX))
    np.save(PROC / "final_dots.npy", p)
    out = {"experiment": "H83-E3", "evidence_class": "BUILD-AND-REGISTRY-MEASUREMENT",
           "surface_scan_before_placement": scan, "dots": int(p.sum()),
           "dots_on_catalogue": int((p[cat] > 0).sum()), "elapsed_s": round(time.time() - t0, 1)}
    (EVID / "h83_e3_surface.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["base", "exp1", "exp2", "exp3"])
    ap.add_argument("--seeds", default="20,30")
    a = ap.parse_args()
    if a.mode == "base":
        g = load_grid(); base_matrix(g.footprint); return 0
    if a.mode == "exp2":
        run_exp2(tuple(int(s) for s in a.seeds.split(",")))
    else:
        {"exp1": run_exp1, "exp3": run_exp3}[a.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
