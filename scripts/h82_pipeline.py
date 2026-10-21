#!/usr/bin/env python3
"""GEMSDOE77 lane H82-RRD — Expenses 1-3 (preregistered, registry/preregistration_h82.json).

Exp1: label-free RRD fields + leakage canaries + Phase-1 surface uniqueness.
Exp2: seed-20 hide-and-recover holdout (gems57-pooled-hide-v2), fixed operating point.
Exp3: full-map TIF via shared writer + Phase-2 dot uniqueness + run card.
Reuse: vendor/gems57 evaluator/writer/validator/uniqueness; src/gems77/emit77 emitter.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH, metric as M  # noqa: E402
from gems57 import uniqueness as UQ  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems57.validate import validate  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems82.fields import RRD_CHANNELS, build_rrd, clean_band, rank01  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
EV = ROOT / "evidence"
SEED = 20
BUDGET, EXCL = 37_654, 2
HGB_KW = dict(max_iter=140, learning_rate=0.09, max_leaf_nodes=31,
              min_samples_leaf=40, early_stopping=False, random_state=77)
RAW_NAMES = ["mag_anom","rtp","tmi_hg","geod_2ndinv","iso_grav_anom_slope","tc_totalcount",
             "geod_shearrate","geod_dilaterate","tmi_vg","deq_n100a15","iso_grav_anom_vg",
             "det_elev","iso_grav_anom","tmi","depth_to_base_surf","ieq_n100a15","cond_surf",
             "iso_grav_anom_hg_signed","det_elev_slope"]
CHANNELS = RAW_NAMES + RRD_CHANNELS
IDX = {n: i for i, n in enumerate(CHANNELS)}


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()


def build_stack(foot):
    """27ch memmap stack: 19 cleaned raw bands + 8 RRD channels. Returns (stack, S_rrd)."""
    PROC.mkdir(parents=True, exist_ok=True)
    path = PROC / "h82_stack.npy"
    spath = PROC / "h82_S_rrd.npy"
    if path.exists() and spath.exists():
        log("stack: cached"); return np.load(path, mmap_mode="r"), np.load(spath)
    stack = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(3730, 3292, len(CHANNELS)))
    bands = {}
    with rasterio.open(RAW / "training_features.tif") as ds:
        for b in range(1, 20):
            a = clean_band(ds.read(b), foot)
            stack[:, :, b - 1] = a
            if b in (1, 2, 14): bands[b] = a
            log(f"band {b} cached")
    rrd = build_rrd(bands[14], bands[2], bands[1], foot)
    for j, name in enumerate(RRD_CHANNELS):
        stack[:, :, 19 + j] = rrd[name]
    S = rrd["S_rrd"]; np.save(spath, S)
    stack.flush(); del stack, rrd, bands; gc.collect()
    return np.load(path, mmap_mode="r"), S


def fetch_h33():
    dest = RAW / "registry" / "gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            with dest.open("wb") as out:
                subprocess.run(["gh", "api",
                                "repos/buffedlizard55-lab/GEMSDOE32/contents/docs/downloads/"
                                "gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif",
                                "-H", "Accept: application/vnd.github.raw"],
                               stdout=out, check=True)
        except Exception as e:  # scope limitation, recorded not hidden
            log("H33 fetch failed:", e); dest.unlink(missing_ok=True); return None
    h = sha_bytes(dest.read_bytes())
    log(f"H33 champion bytes sha256={h} (site prefix c55bafc470054e82)")
    return dest if h.startswith("c55bafc4") else dest  # record actual hash either way


def local_registry_files(exclude: Path | None = None):
    """Local TIFF scope, sha-deduped. `exclude` removes the candidate itself
    by hash as well as by path, so byte-identical mirrors (docs/downloads +
    submissions) can never self-match (Pass-2 fix, IR-82-03)."""
    files, seen = [], set()
    excl_hash = sha_bytes(exclude.read_bytes()) if exclude and exclude.exists() else None
    for d in (ROOT / "docs" / "downloads", ROOT / "submissions"):
        for p in sorted(d.glob("*.tif")):
            if excl_hash and p.name == exclude.name: continue
            h = sha_bytes(p.read_bytes())
            if h in seen or h == excl_hash: continue
            seen.add(h); files.append((p, h))
    return files


def halo_frac(dots, foot):
    return float(UQ._near(dots)[foot].mean()) if dots.any() else 0.0


# ---------------- Exp 1 ----------------
def exp1():
    t0 = time.time()
    grid = G.load_grid(RAW); foot, cat = grid.footprint, grid.catalogue
    stack, S = build_stack(foot)
    folds, _ = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden, visible = folds[0]["hidden_all"], folds[0]["visible"]
    log(f"seed{SEED}: withheld={int(hidden.sum())} visible={int(visible.sum())}")

    # Band-identity re-verification on a fixed 200k footprint sample (IR-82-02).
    rng = np.random.default_rng(82)
    fp_idx = np.flatnonzero(foot)
    samp = rng.choice(fp_idx, 200_000, replace=False)
    flat = np.asarray(stack).reshape(-1, len(CHANNELS)) if False else None
    tmi = np.asarray(stack[:, :, IDX["tmi"]], np.float32)
    gy, gx = np.gradient(tmi); gmag = np.hypot(gy, gx).ravel()
    ident = {
        "spearman_band3_tmi_hg_vs_gradmag_tmi": float(spearmanr(
            np.asarray(stack[:, :, IDX["tmi_hg"]]).ravel()[samp], gmag[samp]).statistic),
        "spearman_rtp_vs_tmi_raw": float(spearmanr(
            np.asarray(stack[:, :, IDX["rtp"]]).ravel()[samp],
            np.asarray(stack[:, :, IDX["tmi"]]).ravel()[samp]).statistic),
        "n_sample": int(samp.size),
    }
    del tmi, gy, gx, gmag; gc.collect()
    log("band identities:", ident)

    # Leakage canaries: every RRD channel (+ composite) alone.
    dist_cat = ndimage.distance_transform_edt(~cat)
    pos = np.flatnonzero(hidden)
    neg_pool = np.flatnonzero(foot & ~cat & (dist_cat > 3))
    neg = rng.choice(neg_pool, min(200_000, neg_pool.size), replace=False)
    idx = np.concatenate([pos, neg]); y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    canaries = {}
    for name in RRD_CHANNELS:
        v = np.asarray(stack[:, :, IDX[name]]).ravel()[idx]
        auc = float(roc_auc_score(y, v)); canaries[name] = {"auc": auc, "sym_auc": max(auc, 1 - auc)}
    v = S.ravel()[idx]; auc = float(roc_auc_score(y, v))
    canaries["S_rrd"] = {"auc": auc, "sym_auc": max(auc, 1 - auc)}
    worst = max(c["sym_auc"] for c in canaries.values())
    log(f"canaries worst symAUC={worst:.4f} (stop if >0.90)")

    # Phase-1 surface uniqueness (sampled Spearman, documented scope).
    h33 = fetch_h33()
    mine_s = S.ravel()[samp]
    rows = []
    candidates = local_registry_files()
    if h33: candidates.append((h33, sha_bytes(h33.read_bytes())))
    for p, h in candidates:
        with rasterio.open(p) as ds: a = ds.read(1)
        a = np.where(np.isfinite(a), a, 0).astype(np.float32)
        rho = float(spearmanr(mine_s, a.ravel()[samp]).statistic) if np.ptp(a) else float("nan")
        rows.append({"file": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
                     "sha256": h, "dots": int((a > 0).sum()),
                     "spearman_sampled_surface": rho})
        del a
    rows.sort(key=lambda r: -abs(r["spearman_sampled_surface"]) if np.isfinite(r["spearman_sampled_surface"]) else 0)
    out = {"experiment": 1, "lane": "H82-RRD", "seed": SEED,
           "withheld_positives": int(hidden.sum()), "band_identities": ident,
           "canaries": canaries, "canary_worst_sym_auc": worst,
           "canary_pass_lt_090": bool(worst <= 0.90),
           "surface_uniqueness": {"scope": "local docs/downloads + submissions (sha-deduped) + fetched H33 champion; sampled Spearman n=200k seed82; NOT complete family registry",
                                  "n_compared": len(rows), "max_abs_rho": max(abs(r["spearman_sampled_surface"]) for r in rows),
                                  "top5": rows[:5]},
           "seconds": round(time.time() - t0, 1)}
    (EV / "h82_e1_canary_surface.json").write_text(json.dumps(out, indent=2) + "\n")
    log("Exp1 -> evidence/h82_e1_canary_surface.json")
    return out


# ---------------- Exp 2 ----------------
def _train_predict(stack, cols, idx, y, region):
    flat_rows = np.asarray(stack).reshape(-1, stack.shape[2])[idx][:, cols] \
        if not hasattr(stack, "filename") else np.array(stack.reshape(-1, stack.shape[2])[idx][:, cols], np.float32)
    clf = HistGradientBoostingClassifier(**HGB_KW); clf.fit(flat_rows, y)
    del flat_rows; gc.collect()
    out = np.zeros(stack.shape[:2], np.float32)
    ys, xs = np.nonzero(region)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = np.array(stack[r0:r1, c0:c1][:, :, cols], np.float32)
    pred = np.zeros(sub.shape[:2], np.float32)
    for i in range(0, sub.shape[0], 200):
        blk = sub[i:i+200].reshape(-1, sub.shape[2])
        pred[i:i+200] = clf.predict_proba(blk)[:, 1].reshape(blk.shape[0] // sub.shape[1], sub.shape[1])
    out[r0:r1, c0:c1] = np.where(region[r0:r1, c0:c1], pred, 0.0)
    del sub, pred, clf; gc.collect()
    return out


def exp2():
    t0 = time.time()
    grid = G.load_grid(RAW); foot, cat = grid.footprint, grid.catalogue
    stack, S = build_stack(foot)
    folds, _ = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden, visible = folds[0]["hidden_all"], folds[0]["visible"]
    dist_cat = ndimage.distance_transform_edt(~cat)
    dist_vis = ndimage.distance_transform_edt(~visible)
    rng = np.random.default_rng(1234)
    samples = []
    for k, fold in enumerate(folds):
        train_region = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k: train_region |= f2["region"]
        pos = np.flatnonzero(visible & train_region)
        if pos.size > 40_000: pos = rng.choice(pos, 40_000, replace=False)
        neg_mask = train_region & ~cat & (dist_cat > 3)
        hard = np.flatnonzero(neg_mask & (dist_cat <= 15)); easy = np.flatnonzero(neg_mask & (dist_cat > 15))
        nh = min(hard.size, pos.size); ne = min(easy.size, 2 * pos.size - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
        samples.append((np.concatenate([pos, neg]),
                        np.r_[np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)]))
        log(f"fold {fold['name']}: pos={pos.size} neg={neg.size} truth={int(fold['truth'].sum())}")

    arms_cols = {"A_rrd": [IDX[n] for n in RRD_CHANNELS],
                 "A_std19": [IDX[n] for n in RAW_NAMES],
                 "A_std19_rrd": list(range(len(CHANNELS)))}
    surfaces = {"S_rrd": S}
    for arm, cols in arms_cols.items():
        cache = PROC / f"h82_oof_{arm}.npy"
        if cache.exists(): surfaces[arm] = np.load(cache); log(f"{arm}: cached OOF"); continue
        surf = np.zeros(foot.shape, np.float32)
        for k, fold in enumerate(folds):
            idx, y = samples[k]
            surf += _train_predict(stack, cols, idx, y, fold["region"])
            log(f"{arm}/{fold['name']} done ({time.time()-t0:.0f}s)")
        np.save(cache, surf); surfaces[arm] = surf

    def emit_pooled(p):
        allowed = [f["region"] & foot & ~visible & (dist_vis > EXCL) for f in folds]
        tot = sum(int(a.sum()) for a in allowed); dots = np.zeros(foot.shape, bool)
        for a in allowed:
            b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
            if b > 0: dots |= emit77.emit_dots(np.clip(p, 0, 1), a, b)
        return dots

    def score(dots):
        terms, per = None, {}
        for f in folds:
            res, t = EH.evaluate(dots.astype(np.float32), f, valid=foot, block_side=200)
            per[f["name"]] = {"dti": res["dti"], "n_truth": res["n_truth"], "emitted": res["n_emitted"]}
            terms = t if terms is None else terms + t
        return per, terms

    terms_by_arm, results = {}, {}
    order = ["S_rrd", "A_rrd", "A_std19", "A_std19_rrd"]
    for arm in order:
        dots = emit_pooled(surfaces[arm]); per, terms = score(dots)
        terms_by_arm[arm] = terms
        results[arm] = {"dots": int(dots.sum()), "per_fold": per,
                        "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        np.save(PROC / f"h82_dots_{arm}.npy", dots)
        log(f"{arm}: HOLDOUT-DTI={results[arm]['pooled_holdout_dti']:.4f}")
    rng2 = np.random.default_rng(999)
    dots = emit_pooled((rng2.random(foot.shape, dtype=np.float32) * foot))
    per, terms = score(dots); terms_by_arm["A0_random"] = terms
    results["A0_random"] = {"dots": int(dots.sum()), "per_fold": per,
                            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
    log(f"A0_random: HOLDOUT-DTI={results['A0_random']['pooled_holdout_dti']:.4f}")

    summary = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_std19_rrd")
    winner = max(("S_rrd", "A_rrd", "A_std19_rrd"), key=lambda a: results[a]["pooled_holdout_dti"])
    out = {"experiment": 2, "evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
           "alpha": M.ALPHA, "beta": M.BETA, "kernel_radius_m": M.R_M, "seed": SEED,
           "withheld_positive_pixels_total_draw": int(hidden.sum()),
           "fold_receipts": [f["receipt"] for f in folds],
           "operating_point_fixed": {"budget": BUDGET, "exclusion_px": EXCL, "min_sep_px": 2.8, "scan": "none (preregistered)"},
           "arms": results, "pooled_summary": summary, "preregistered_winner": winner,
           "incumbent": {"dti": 0.152718, "evaluator": "gems57-pooled-hide-v2", "seed": 20,
                         "withheld": 11321, "ci95": [0.131548, 0.175185]},
           "beats_incumbent_point_estimate": bool(results[winner]["pooled_holdout_dti"] > 0.152718),
           "seconds": round(time.time() - t0, 1)}
    (EV / "h82_e2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    log("Exp2 -> evidence/h82_e2_holdout.json; winner:", winner)
    return out


# ---------------- Exp 3 ----------------
def exp3():
    t0 = time.time()
    e2 = json.loads((EV / "h82_e2_holdout.json").read_text())
    winner = e2["preregistered_winner"]
    grid = G.load_grid(RAW); foot, cat = grid.footprint, grid.catalogue
    stack, S = build_stack(foot)
    dist_cat = ndimage.distance_transform_edt(~cat)
    if winner == "S_rrd":
        surf = S; fit_note = "pure physics composite (no refit)"
    else:
        cols = {"A_rrd": [IDX[n] for n in RRD_CHANNELS],
                "A_std19_rrd": list(range(len(CHANNELS)))}[winner]
        rng = np.random.default_rng(77)
        cache = PROC / "h82_final_surface.npy"
        if cache.exists(): surf = np.load(cache); fit_note = "cached full-catalogue refit"
        else:
            pos = np.flatnonzero(cat)
            neg_mask = foot & ~cat & (dist_cat > 3)
            hard = np.flatnonzero(neg_mask & (dist_cat <= 15)); easy = np.flatnonzero(neg_mask & (dist_cat > 15))
            neg = np.concatenate([rng.choice(hard, min(hard.size, pos.size), replace=False),
                                  rng.choice(easy, min(easy.size, pos.size), replace=False)])
            idx = np.concatenate([pos, neg]); y = np.r_[np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)]
            surf = _train_predict(stack, cols, idx, y, foot)
            np.save(cache, surf); fit_note = f"full-catalogue refit pos={pos.size} neg={neg.size}"
        log(fit_note)
    allowed = foot & (dist_cat > EXCL)
    dots = emit77.emit_dots(np.clip(surf, 0, 1).astype(np.float32), allowed, BUDGET)
    pred = dots.astype(np.float32)
    h8 = sha_bytes(dots.tobytes())[:8]
    name = f"gems77-h82-rrd-discord-37654dots-20261021-{h8}-zeros"
    note = "H82-RRD RTP-TMI remanence-discordance breaks; 37654 binary dots >200m off catalogue; holdout-gated, see run card"
    assert len(note) <= 140 and len(name) <= 140
    dest = ROOT / "docs" / "downloads" / f"{name}.tif"
    sample = RAW / "sample_submission.tif"
    if dest.exists():
        receipt = json.loads(dest.with_suffix(".json").read_text()); log("TIF exists; reusing receipt")
    else:
        receipt = write_submission(dest, pred, sample, foot, note=note, name=name, catalogue=cat,
                                   metadata={"lane": "H82-RRD", "winner_arm": winner, "fit": fit_note})
    # Mirror into submissions/ (byte-identical copies for repo convention).
    for suf in (".tif", ".zip", ".json"):
        src = dest.with_suffix(suf); dst = ROOT / "submissions" / src.name
        if src.exists() and not dst.exists(): dst.write_bytes(src.read_bytes())
    val = validate(dest, foot, cat)
    # Phase-2 final-dot uniqueness vs local scope + H33.
    rows = []
    for p, h in local_registry_files(exclude=dest):
        with rasterio.open(p) as ds: a = ds.read(1)
        a = np.where(np.isfinite(a), a, 0).astype(np.float32)
        theirs = a > 0
        rows.append({"file": p.name, "sha256": h, "their_dots": int(theirs.sum()),
                     "halo_bg_frac": halo_frac(theirs, foot),
                     "forward_overlap_3px": UQ.dot_overlap(pred, a)})
        del a
    h33p = fetch_h33(); h33_row = None
    if h33p:
        with rasterio.open(h33p) as ds: a = np.where(np.isfinite(ds.read(1)), ds.read(1), 0) if False else ds.read(1)
        a = np.where(np.isfinite(a), a, 0).astype(np.float32)
        h33_row = {"file": "GEMSDOE32 " + h33p.name, "sha256": sha_bytes(h33p.read_bytes()),
                   "their_dots": int((a > 0).sum()), "halo_bg_frac": halo_frac(a > 0, foot),
                   "forward_overlap_3px": UQ.dot_overlap(pred, a)}
        rows.append(h33_row); del a
    informative = [r for r in rows if r["halo_bg_frac"] < 0.70]
    uniq = {"scope": "local docs/downloads + submissions (sha-deduped) + fetched H33 champion; NOT complete family registry",
            "n_compared": len(rows), "n_informative_lt70_halo": len(informative),
            "max_overlap_literal": max(r["forward_overlap_3px"] for r in rows),
            "max_overlap_informative": max((r["forward_overlap_3px"] for r in informative), default=0.0),
            "literal_firings_gt070": sum(r["forward_overlap_3px"] > 0.70 for r in rows),
            "informative_firings_gt070": sum(r["forward_overlap_3px"] > 0.70 for r in informative),
            "vs_h33": h33_row,
            "top5_informative": sorted(informative, key=lambda r: -r["forward_overlap_3px"])[:5],
            "rows": rows}
    out = {"experiment": 3, "winner_arm": winner, "fit": fit_note, "dots": int(dots.sum()),
           "on_catalogue": int((dots & cat).sum()), "min_dist_to_catalogue_px": float(dist_cat[dots].min()) if dots.any() else None,
           "submission_name": name, "note": note, "note_chars": len(note),
           "file": f"docs/downloads/{name}.tif", "sha256": receipt["sha256"], "bytes": receipt["bytes"],
           "zip_sha256": receipt.get("zip_sha256"), "validator": val, "writer_receipt": receipt,
           "uniqueness_final_dots": uniq, "seconds": round(time.time() - t0, 1)}
    (EV / "h82_e3_build.json").write_text(json.dumps(out, indent=2) + "\n")
    log("Exp3 ->", dest.name, "sha", receipt["sha256"][:12])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["exp1", "exp2", "exp3"])
    a = ap.parse_args()
    {"exp1": exp1, "exp2": exp2, "exp3": exp3}[a.stage]()
