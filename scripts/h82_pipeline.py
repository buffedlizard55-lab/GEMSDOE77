#!/usr/bin/env python3
"""H82-TCSTEP 3-experiment pipeline (registry/preregistration_h82.json).

Exp 1: leakage canaries (seeds 20/30) + pre-placement surface Spearman vs
       every locally present sibling/registry raster. Stops on AUC > 0.90
       or |rho| > 0.90.
Exp 2: paired hide-and-recover HOLDOUT-DTI (unmodified vendor
       gems57-pooled-hide-v2), seeds 20/30. Arms: A_std_plus_tcstep
       (candidate), A_std (paired control), S_tcstep_physics, A0_random.
Exp 3: full-catalogue LOQO candidate surface -> surface Spearman re-check
       -> 37,654-dot research GeoTIFF via unmodified vendor
       submission_writer -> final-dot overlap + validator + run card.

Shared vendor code is used unchanged (verify registry/template_pin.json).
No catalogue-derived feature is used anywhere: the catalogue enters only
as (a) training labels on VISIBLE pixels in holdout training quadrants,
(b) the >2 px emission exclusion, (c) the exact writer mask.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
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
from gems57 import submission_writer as SW  # noqa: E402
from gems57 import uniqueness as UQ  # noqa: E402
from gems57 import validate as VAL  # noqa: E402
from gems77h82.fields import H82_NAMES, build_h82_fields, clean  # noqa: E402
from gems77lia.allocate import greedy_select  # noqa: E402

RAW = ROOT / "data" / "raw"
BRIDGE = ROOT / "data" / "bridge"
PROC = ROOT / "data" / "processed" / "h82"
EVID = ROOT / "evidence"
BUDGET = 37_654
MIN_SEP = 2.8
EXCL_PX = 2.0
INCUMBENT = 0.152718  # accepted repo holdout best (evidence/h81_e2_holdout.json)

BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope",
    "tc_totalcount", "geod_shearrate", "geod_dilaterate", "tmi_vg",
    "deq_n100a15", "iso_grav_anom_vg", "det_elev", "iso_grav_anom",
    "tmi", "depth_to_base_surf", "ieq_n100a15", "cond_surf",
    "iso_grav_anom_hg_signed", "det_elev_slope",
]
HGB_KW = dict(max_iter=160, learning_rate=0.075, max_leaf_nodes=45,
              min_samples_leaf=50, l2_regularization=2.5,
              early_stopping=False, random_state=82082)


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


def ensure_feature_matrices(foot: np.ndarray):
    """Footprint-row matrices: std19 (19 core bands) and h82 (4 channels).

    Cached as .npy under data/processed/h82 (gitignored). Returns memmaps
    plus the full-grid deterministic tc_blind surface.
    """
    PROC.mkdir(parents=True, exist_ok=True)
    std_p, h82_p, blind_p = PROC / "std19_fp.npy", PROC / "h82_fp.npy", PROC / "tc_blind_full.npy"
    if std_p.exists() and h82_p.exists() and blind_p.exists():
        return (np.load(std_p, mmap_mode="r"), np.load(h82_p, mmap_mode="r"),
                np.load(blind_p))
    src_path = RAW / "training_features.tif"
    got = sha256_file(src_path)
    assert got == "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5", got
    n_fp = int(foot.sum())
    std = np.lib.format.open_memmap(std_p, mode="w+", dtype=np.float32,
                                    shape=(n_fp, 19))
    tc_full = slope_full = None
    with rasterio.open(src_path) as src:
        assert src.count == 19 and (src.height, src.width) == foot.shape
        for b in range(1, 20):
            band = clean(src.read(b))
            std[:, b - 1] = band[foot]
            if b == 6:
                tc_full = band
            if b == 19:
                slope_full = band
            log(f"  band {b}/19 {BAND_NAMES[b-1]} cached")
    std.flush()
    fields = build_h82_fields(tc_full, slope_full, foot)
    h82 = np.lib.format.open_memmap(h82_p, mode="w+", dtype=np.float32,
                                    shape=(n_fp, len(H82_NAMES)))
    for j, name in enumerate(H82_NAMES):
        h82[:, j] = fields[name][foot]
    h82.flush()
    np.save(blind_p, fields["tc_blind"])
    diag = {name: {"min": float(fields[name][foot].min()),
                   "max": float(fields[name][foot].max()),
                   "mean": float(fields[name][foot].mean())} for name in H82_NAMES}
    (PROC / "field_diag.json").write_text(json.dumps(diag, indent=2) + "\n")
    return (np.load(std_p, mmap_mode="r"), np.load(h82_p, mmap_mode="r"),
            fields["tc_blind"])


def local_registry_rasters() -> list[Path]:
    """Every sibling submission raster present in this checkout, sha-deduped."""
    seen, out = set(), []
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")) + \
             sorted((ROOT / "submissions").glob("*.tif")):
        s = sha256_file(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def surface_spearman_scan(surface: np.ndarray, foot: np.ndarray) -> dict:
    rows = []
    for p in local_registry_rasters():
        with rasterio.open(p) as src:
            theirs = src.read(1)
        if theirs.shape != surface.shape:
            rows.append({"file": str(p.relative_to(ROOT)), "skipped": "shape mismatch"})
            continue
        r = UQ.surface_rho(surface, theirs, foot)
        rho = r["spearman_full_footprint"]
        rows.append({"file": str(p.relative_to(ROOT)), "sha256": sha256_file(p),
                     "spearman_full_footprint": None if rho != rho else float(rho)})
        del theirs
    finite = [r for r in rows if r.get("spearman_full_footprint") is not None]
    worst = max(finite, key=lambda r: abs(r["spearman_full_footprint"]), default=None)
    return {"scope": "locally present rasters only (docs/downloads + submissions); "
                      "complete accessible registry scan NOT performed in this session",
            "complete": False, "checked": len(finite), "rows": rows,
            "max_abs_spearman": abs(worst["spearman_full_footprint"]) if worst else None,
            "worst_file": worst["file"] if worst else None}


def emit(surface: np.ndarray, allowed: np.ndarray, limit: int = BUDGET) -> np.ndarray:
    """Greedy Poisson-disk dotted emission, min-sep 2.8 px (h81 pattern)."""
    surf = np.where(allowed, surface, 0.0)
    idx = np.flatnonzero(surf > 0)
    if idx.size == 0:
        raise RuntimeError("empty emission candidate set")
    vals = surf.ravel()[idx]
    cap = min(idx.size, 600_000)
    if idx.size > cap:  # bound the python greedy loop; verified below
        top = np.argpartition(-vals, cap - 1)[:cap]
        idx, vals = idx[top], vals[top]
    order = idx[np.argsort(-vals, kind="stable")]
    sel = greedy_select(order, MIN_SEP, limit=limit)
    if sel.size != limit:
        raise RuntimeError(f"emission reached {sel.size} != {limit} dots "
                           f"(candidate cap binding or surface too sparse)")
    p = np.zeros(surface.shape, np.float32)
    p.ravel()[sel] = 1.0
    return p


def fp_lut(foot: np.ndarray):
    h, w = foot.shape
    flat = np.flatnonzero(foot)
    lut = np.full(h * w, -1, dtype=np.int32)
    lut[flat] = np.arange(flat.size, dtype=np.int32)
    return flat, lut


def run_exp1() -> int:
    t0 = time.time()
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h82_fp, blind = ensure_feature_matrices(foot)
    log(f"matrices std={std_fp.shape} h82={h82_fp.shape}")
    fp_flat, lut = fp_lut(foot)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    canary_by_seed = {}
    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden = folds[0]["hidden_all"]
        rng = np.random.default_rng(8200 + seed)
        pos = lut[np.flatnonzero(hidden & foot)]
        neg_pool = lut[np.flatnonzero(foot & ~cat & (dist_cat > 3.0))]
        neg = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
        rows = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])

        def eval_can(mat, names):
            Xc = np.asarray(mat[rows], dtype=np.float32)
            out = {}
            for j, nm in enumerate(names):
                v = Xc[:, j]
                m = np.isfinite(v)
                if m.sum() > 100 and np.ptp(v[m]) > 0:
                    a = float(roc_auc_score(y[m], v[m]))
                    out[nm] = {"auc": a, "symmetric_auc": max(a, 1.0 - a)}
                else:
                    out[nm] = {"auc": None, "symmetric_auc": None}
            return out

        c_h82, c_std = eval_can(h82_fp, H82_NAMES), eval_can(std_fp, BAND_NAMES)
        mx_h = max((v["symmetric_auc"] or 0.0, k) for k, v in c_h82.items())
        mx_s = max((v["symmetric_auc"] or 0.0, k) for k, v in c_std.items())
        canary_by_seed[str(seed)] = {
            "withheld_positive_pixels": int(hidden.sum()),
            "negative_sample_pixels": int(neg.size),
            "h82_canaries": c_h82, "std19_canaries": c_std,
            "max_h82_symmetric_auc": mx_h[0], "max_h82_channel": mx_h[1],
            "max_std_symmetric_auc": mx_s[0], "max_std_channel": mx_s[1],
            "leakage_flagged": bool(mx_h[0] > 0.90 or mx_s[0] > 0.90)}
        log(f"seed {seed} canary max H82={mx_h[0]:.4f} ({mx_h[1]}) std={mx_s[0]:.4f} ({mx_s[1]})")

    log("surface Spearman scan (tc_blind, pre-placement) vs local rasters...")
    scan = surface_spearman_scan(blind, foot)
    log(f"surface max |rho|={scan['max_abs_spearman']} vs {scan['worst_file']} "
        f"({scan['checked']} checked)")
    any_leak = any(v["leakage_flagged"] for v in canary_by_seed.values())
    rho_fail = scan["max_abs_spearman"] is not None and scan["max_abs_spearman"] > 0.90
    out = {"experiment": "H82-E1", "evidence_class": "CANARY-AND-SURFACE-DIAGNOSTIC",
           "canaries_by_seed": canary_by_seed, "leakage_ceiling_auc": 0.90,
           "leakage_pass": not any_leak,
           "phase1_tc_blind_surface_spearman": scan,
           "surface_spearman_gate_pass": not rho_fail,
           "literal_overlap_gate_note": "docs/evidence/registry_saturation.json: a dense "
             "registry witness covers 100% of allowed pixels, so the literal 0.70 "
             "forward-overlap gate fails for ANY non-empty candidate by construction. "
             "Reported, not relitigated; overlap vs sparse local rasters is measured in Exp 3.",
           "elapsed_seconds": round(time.time() - t0, 2)}
    EVID.mkdir(exist_ok=True)
    (EVID / "h82_e1_canary_surface.json").write_text(json.dumps(out, indent=2) + "\n")
    if any_leak or rho_fail:
        raise SystemExit("STOP: leakage canary or surface Spearman gate failed")
    return 0


def _fold_splits(grid, folds, foot, lut, labels_pos, dist_labels, rng):
    splits = []
    for k, fold in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        pos_mask = labels_pos & tr_reg & foot
        d_tr = ndimage.distance_transform_edt(~(labels_pos & tr_reg)).astype(np.float32)
        pos = lut[np.flatnonzero(pos_mask)]
        if pos.size > 40_000:
            pos = rng.choice(pos, 40_000, replace=False)
        neg_base = foot & tr_reg & ~labels_pos & (d_tr > 3.0) & (dist_labels > 3.0)
        hard = lut[np.flatnonzero(neg_base & (d_tr <= 15.0))]
        easy = lut[np.flatnonzero(neg_base & (d_tr > 15.0))]
        n_neg = 2 * pos.size
        nh = min(hard.size, n_neg // 2)
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False),
                              rng.choice(easy, ne, replace=False)])
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        splits.append((fold, tr_idx, y_tr))
    return splits


def _oof_surface(X, splits, te_masks, foot, fp_flat, lut, tag, cache_dir):
    cache = cache_dir / f"oof_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    surf = np.zeros(foot.shape, np.float32)
    for (fold, tr_idx, y_tr), te_mask in zip(splits, te_masks):
        te_idx = lut[np.flatnonzero(te_mask & foot)]
        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(np.asarray(X[tr_idx], np.float32), y_tr)
        preds = np.empty(te_idx.size, np.float32)
        for s in range(0, te_idx.size, 250_000):
            sl = slice(s, min(s + 250_000, te_idx.size))
            preds[sl] = clf.predict_proba(np.asarray(X[te_idx[sl]], np.float32))[:, 1]
        surf.ravel()[fp_flat[te_idx]] = preds
        del clf, preds
        gc.collect()
        log(f"  OOF {tag} fold {fold['name']} done")
    np.save(cache, surf)
    return surf


def run_exp2() -> int:
    t0 = time.time()
    e1 = json.loads((EVID / "h82_e1_canary_surface.json").read_text())
    if not e1["leakage_pass"] or not e1["surface_spearman_gate_pass"]:
        raise SystemExit("Cannot run Exp 2: Exp 1 gate failed")
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h82_fp, blind = ensure_feature_matrices(foot)
    fp_flat, lut = fp_lut(foot)
    X_std = std_fp
    X_comb_p = PROC / "comb23_fp.npy"
    if not X_comb_p.exists():
        comb = np.lib.format.open_memmap(X_comb_p, mode="w+", dtype=np.float32,
                                         shape=(std_fp.shape[0], 23))
        comb[:, :19] = np.asarray(std_fp)
        comb[:, 19:] = np.asarray(h82_fp)
        comb.flush()
    X_comb = np.load(X_comb_p, mmap_mode="r")

    seeds_out = {}
    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        visible = folds[0]["visible"]
        dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
        log(f"=== Exp2 seed {seed}: withheld={int(folds[0]['hidden_all'].sum())} "
            f"visible={int(visible.sum())} ===")
        rng = np.random.default_rng(8800 + seed)
        splits = _fold_splits(grid, folds, foot, lut, visible, dist_vis, rng)
        te_masks = [f["region"] for f in folds]
        sdir = PROC / f"seed{seed}"
        sdir.mkdir(exist_ok=True)
        surfs = {
            "A_std_plus_tcstep": _oof_surface(X_comb, splits, te_masks, foot, fp_flat, lut,
                                              "comb", sdir),
            "A_std": _oof_surface(X_std, splits, te_masks, foot, fp_flat, lut,
                                  "std", sdir),
            "S_tcstep_physics": blind,
            "A0_random": np.random.default_rng(999 + seed).random(foot.shape).astype(np.float32) * foot,
        }
        dist_known = ndimage.distance_transform_edt(~folds[0]["masked_known"]).astype(np.float32)
        allowed = foot & (dist_known > EXCL_PX)
        terms = {k: None for k in surfs}
        for name, surf in surfs.items():
            p = emit(np.asarray(surf, np.float32), allowed)
            assert int((p > 0).sum()) == BUDGET
            acc = None
            for fold in folds:
                _, t = EH.evaluate(p, fold, foot)
                acc = t if acc is None else acc + t
            terms[name] = acc
            log(f"  arm {name} emitted+evaluated")
        summary = EH.pooled_summary(terms, draws=2000, candidate="A_std_plus_tcstep")
        seeds_out[str(seed)] = {"summary": summary,
                                "fold_receipts": [f["receipt"] for f in folds],
                                "budget_dots": BUDGET}
        sc = summary["scores"]
        log(f"seed {seed}: candidate DTI={sc['A_std_plus_tcstep']['dti']:.4f} "
            f"std={sc['A_std']['dti']:.4f} physics={sc['S_tcstep_physics']['dti']:.4f} "
            f"random={sc['A0_random']['dti']:.4f}")
    out = {"experiment": "H82-E2", "evidence_class": "HOLDOUT-DTI",
           "evaluator_version": EH.VERSION, "seeds": seeds_out,
           "implementation_sha256": EH.implementation_hashes(),
           "current_holdout_best": {"score": INCUMBENT,
                                    "source": "evidence/h81_e2_holdout.json current_holdout_best"},
           "elapsed_seconds": round(time.time() - t0, 2)}
    (EVID / "h82_e2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


def run_exp3() -> int:
    t0 = time.time()
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h82_fp, blind = ensure_feature_matrices(foot)
    fp_flat, lut = fp_lut(foot)
    X_comb = np.load(PROC / "comb23_fp.npy", mmap_mode="r")

    # Full-catalogue LOQO: train regions = eroded quadrants of the other
    # folds; prediction masks = uneroded quadrants (full coverage, no
    # train/test pixel overlap).
    folds, _ = HO.buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
    quad = HO.quadrant_ids(foot)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    rng = np.random.default_rng(9900)
    splits = _fold_splits(grid, folds, foot, lut, cat, dist_cat, rng)
    te_masks = [quad == k for k in range(4)]
    surf = _oof_surface(X_comb, splits, te_masks, foot, fp_flat, lut,
                        "fullcat", PROC)
    assert np.isfinite(surf).all() and surf.min() >= 0.0 and surf.max() <= 1.0
    assert (surf[foot] > 0).mean() > 0.99, "OOF surface has unpredicted footprint pixels"

    log("candidate-surface Spearman re-check before placement...")
    scan = surface_spearman_scan(surf, foot)
    if scan["max_abs_spearman"] is not None and scan["max_abs_spearman"] > 0.90:
        (EVID / "h82_e3_surface_duplicate.json").write_text(
            json.dumps({"experiment": "H82-E3", "duplicate": True, "scan": scan}, indent=2) + "\n")
        raise SystemExit("STOP: candidate surface duplicates a local registry raster "

                         f"(|rho|={scan['max_abs_spearman']:.4f} vs {scan['worst_file']})")

    allowed = foot & (dist_cat > EXCL_PX)
    p = emit(surf, allowed)
    assert int((p > 0).sum()) == BUDGET and not (p[cat] > 0).any()
    sel = np.flatnonzero(p)
    dot_hash = hashlib.sha256(np.sort(sel).astype("<i8").tobytes()).hexdigest()
    name = f"gems77-h82-tcstep-{BUDGET}dots-20261021-{dot_hash[:8]}-zeros"
    note = ("H82-TCSTEP band6 TC step x coherence x low-slope + std19 HGB LOQO; "
            "37654 dots sep2.8px >2px from catalogue")
    assert len(note) <= 140 and len(name) <= 140
    out_path = ROOT / "submissions" / f"{name}.tif"
    receipt = SW.write_submission(out_path, p, BRIDGE / "sample_submission.tif",
                                  foot, note=note, name=name, catalogue=cat,
                                  metadata={"lane": "H82-TCSTEP", "dots": BUDGET,
                                            "dot_index_sha256": dot_hash})
    for ext in (".tif", ".zip", ".json"):
        shutil.copy2(ROOT / "submissions" / f"{name}{ext}",
                     ROOT / "docs" / "downloads" / f"{name}{ext}")

    # Final-dot overlap vs locally present rasters (vendor dot_overlap).
    rows = []
    for rp in local_registry_rasters():
        if rp.resolve() == out_path.resolve():
            continue
        with rasterio.open(rp) as src:
            theirs = src.read(1)
        if theirs.shape != p.shape:
            continue
        rows.append({"file": str(rp.relative_to(ROOT)), "sha256": sha256_file(rp),
                     "forward_overlap_3px": UQ.dot_overlap(p, theirs),
                     "their_positive_px": int((np.isfinite(theirs) & (theirs > 0)).sum())})
        del theirs
    sparse = [r for r in rows if r["their_positive_px"] < 0.5 * int(foot.sum())]
    worst_sparse = max(sparse, key=lambda r: r["forward_overlap_3px"], default=None)
    val = VAL.validate(out_path, foot, cat)
    build = {"experiment": "H82-E3", "submission_name": name, "note": note,
             "note_chars": len(note), "dots": BUDGET, "dot_index_sha256": dot_hash,
             "file_sha256": sha256_file(out_path), "bytes": out_path.stat().st_size,
             "writer_receipt_file": f"submissions/{name}.json",
             "validator": val,
             "candidate_surface_spearman": scan,
             "final_dot_overlap": {
                 "scope": "locally present rasters only; complete registry NOT scanned",
                 "complete": False, "checked": len(rows), "rows": rows,
                 "max_forward_overlap_3px_sparse_rasters":
                     worst_sparse["forward_overlap_3px"] if worst_sparse else None,
                 "worst_sparse_file": worst_sparse["file"] if worst_sparse else None,
                 "literal_gate": "FAIL by construction (dense witness, "
                                 "docs/evidence/registry_saturation.json)"},
             "elapsed_seconds": round(time.time() - t0, 2)}
    (EVID / "h82_e3_build.json").write_text(json.dumps(build, indent=2) + "\n")
    log(f"built {name}.tif sha256={build['file_sha256']} validator_ok={val.get('ok', val)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["exp1", "exp2", "exp3"])
    args = ap.parse_args()
    return {"exp1": run_exp1, "exp2": run_exp2, "exp3": run_exp3}[args.mode]()


if __name__ == "__main__":
    raise SystemExit(main())
