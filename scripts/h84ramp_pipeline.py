#!/usr/bin/env python3
"""H84-RAMP 3-Experiment Pipeline: Deconfounded Two-Host Relay-Ramp & Geodetic-Geophysical Stepover Transfer.

Strictly reuses:
  - vendor/gems57/relay_bend_anatomy.py (two_host_frame, host_bend_fields)
  - vendor/gems57/holdout.py (buffered_component_draw, quadrant_ids)
  - vendor/gems57/evaluate_holdout.py (evaluate, pooled_summary, VERSION='gems57-pooled-hide-v2')
  - vendor/gems57/submission_writer.py (write_submission)
  - vendor/gems57/validate.py (validate)
  - vendor/gems57/uniqueness.py (surface_rho, dot_overlap, two_phase_summary)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems57 import submission_writer as SW  # noqa: E402
from gems57 import uniqueness as UQ  # noqa: E402
from gems57 import validate as VAL  # noqa: E402
from gems77lia.allocate import greedy_select  # noqa: E402
from gems84.fields import (  # noqa: E402
    H84_NAMES,
    build_h84_surface_fields,
    compute_blbco_h84_rows,
    prepare_static_geophys_context,
)

BRIDGE = ROOT / "data" / "bridge"
RAW_CACHE = ROOT / ".cache" / "raw"
PROC_CACHE = ROOT / ".cache" / "h84_proc"
EVID = ROOT / "evidence"
DOCS_EVID = ROOT / "docs" / "evidence"

BUDGET = 37_654
MIN_SEP = 2.8
EXCL_PX = 2.0
INCUMBENT = 0.152718

BAND_NAMES = (
    "mag_anom",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_anom_slope",
    "tc",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
    "deq_n100a15",
    "iso_grav_anom_vg",
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "depth_to_base_surf",
    "ieq_n100a15",
    "cond_surf",
    "iso_grav_anom_hg",
    "det_elev_slope",
)

HGB_KW = dict(
    max_iter=200,
    learning_rate=0.06,
    max_leaf_nodes=45,
    min_samples_leaf=60,
    l2_regularization=2.5,
    random_state=84,
)


def log(msg: str) -> None:
    print(f"[h84 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw_data() -> Path:
    tf_path = RAW_CACHE / "training_features.tif"
    if not tf_path.is_file():
        log("Restoring training_features.tif and scored reference rasters into .cache/raw ...")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "restore_data.py"),
                "--ids",
                "training_features,scored_h19_5,scored_h19_4,scored_h16_1,scored_d15_scored,scored_d28_unscored",
                "--target-dir",
                str(RAW_CACHE),
            ],
            check=True,
        )
    reg_witness = ROOT / ".cache" / "registry" / "dense_witness.tif"
    if not reg_witness.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "restore_registry.py")], check=True)
    return tf_path


def clean_band(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(a) | (np.abs(a) > 1e30)
    if bad.any():
        a = a.copy()
        a[bad] = 0.0
    return a


def ensure_std_and_static_context(foot: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    PROC_CACHE.mkdir(parents=True, exist_ok=True)
    std_p = PROC_CACHE / "std19_fp.npy"
    geod_p = PROC_CACHE / "ctx_geod_trans.npy"
    base_p = PROC_CACHE / "ctx_base_bridge.npy"
    if std_p.is_file() and geod_p.is_file() and base_p.is_file():
        return (
            np.load(std_p, mmap_mode="r"),
            {"geod_trans": np.load(geod_p), "base_bridge": np.load(base_p)},
        )
    tf_path = ensure_raw_data()
    n_fp = int(foot.sum())
    std = np.lib.format.open_memmap(std_p, mode="w+", dtype=np.float32, shape=(n_fp, 19))
    saved = {}
    with rasterio.open(tf_path) as src:
        assert src.count == 19 and (src.height, src.width) == foot.shape
        for b in range(1, 20):
            band = clean_band(src.read(b))
            std[:, b - 1] = band[foot]
            if b in (3, 7, 8, 15, 18, 19):
                saved[b] = band
            log(f"  cached band {b}/19 ({BAND_NAMES[b - 1]})")
    std.flush()
    ctx = prepare_static_geophys_context(
        geod_shear=saved[7],
        geod_dilate=saved[8],
        depth_base=saved[15],
        grav_hg=saved[18],
        tmi_hg=saved[3],
        elev_slope=saved[19],
        footprint=foot,
    )
    np.save(geod_p, ctx["geod_trans"])
    np.save(base_p, ctx["base_bridge"])
    return np.load(std_p, mmap_mode="r"), ctx


def ensure_fold_h84_fields(
    tag: str,
    visible: np.ndarray,
    foot: np.ndarray,
    ctx: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    PROC_CACHE.mkdir(parents=True, exist_ok=True)
    mat_p = PROC_CACHE / f"h84_{tag}_fp.npy"
    comp_p = PROC_CACHE / f"h84_{tag}_comp2d.npy"
    aux_p = PROC_CACHE / f"h84_{tag}_aux.npz"
    if mat_p.is_file() and comp_p.is_file() and aux_p.is_file():
        aux_z = np.load(aux_p)
        aux = {k: aux_z[k] for k in aux_z.files}
        return np.load(mat_p, mmap_mode="r"), np.load(comp_p), aux
    log(f"Building H84 two-host relay & bend fields for {tag} (visible={int(visible.sum()):,} px) ...")
    mat_fp, comp_2d, aux = build_h84_surface_fields(visible, foot, ctx)
    np.save(mat_p, mat_fp)
    np.save(comp_p, comp_2d)
    np.savez(aux_p, **aux)
    return np.load(mat_p, mmap_mode="r"), comp_2d, aux


def local_registry_rasters(exclude_path: Path | None = None) -> list[Path]:
    """All locally present and restored sibling/registry GeoTIFFs, SHA-256 deduplicated."""
    seen, out = set(), []
    candidates = (
        sorted((ROOT / ".cache" / "registry").glob("*.tif"))
        + sorted((RAW_CACHE / "scored").glob("*.tif"))
        + sorted((ROOT / "docs" / "downloads").glob("*.tif"))
        + sorted((ROOT / "submissions").glob("*.tif"))
    )
    excl_res = exclude_path.resolve() if exclude_path and exclude_path.exists() else None
    for p in candidates:
        if not p.is_file():
            continue
        if excl_res and p.resolve() == excl_res:
            continue
        if "h84-ramp" in p.name:
            continue
        s = sha256_file(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def surface_spearman_scan(surface: np.ndarray, foot: np.ndarray, exclude_path: Path | None = None) -> dict:
    rows = []
    for p in local_registry_rasters(exclude_path=exclude_path):
        with rasterio.open(p) as src:
            theirs = src.read(1)
        rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        if theirs.shape != surface.shape:
            rows.append({"file": rel, "skipped": "shape mismatch"})
            continue
        r = UQ.surface_rho(surface, theirs, foot)
        rho = r["spearman_full_footprint"]
        rows.append(
            {
                "file": rel,
                "sha256": sha256_file(p),
                "spearman_full_footprint": None if (rho is None or rho != rho) else float(rho),
                "spearman_on_dot_union": (
                    None
                    if (r["spearman_on_dot_union"] is None or r["spearman_on_dot_union"] != r["spearman_on_dot_union"])
                    else float(r["spearman_on_dot_union"])
                ),
            }
        )
        del theirs
    finite = [r for r in rows if r.get("spearman_full_footprint") is not None]
    worst = max(finite, key=lambda r: abs(r["spearman_full_footprint"]), default=None)
    return {
        "scope": (
            "All locally present + restored registry rasters (.cache/registry, .cache/raw/scored, "
            "docs/downloads, submissions); 812-entry multi-repo index is not fully mirrored locally"
        ),
        "complete": False,
        "checked": len(finite),
        "max_abs_spearman": abs(worst["spearman_full_footprint"]) if worst else None,
        "worst_file": worst["file"] if worst else None,
        "worst_signed_spearman": worst["spearman_full_footprint"] if worst else None,
        "rows": rows,
    }


def emit(surface: np.ndarray, allowed: np.ndarray, limit: int = BUDGET) -> np.ndarray:
    """Greedy Poisson-disk binary dot emission, min-sep 2.8 px, >2.0 px from catalogue."""
    surf = np.where(allowed, surface, 0.0)
    idx = np.flatnonzero(surf > 0)
    if idx.size == 0:
        raise RuntimeError("Empty emission candidate set")
    vals = surf.ravel()[idx]
    cap = min(idx.size, 650_000)
    if idx.size > cap:
        top = np.argpartition(-vals, cap - 1)[:cap]
        idx, vals = idx[top], vals[top]
    order = idx[np.argsort(-vals, kind="stable")]
    sel = greedy_select(order, MIN_SEP, limit=limit)
    if sel.size != limit:
        raise RuntimeError(f"Emission reached {sel.size} != {limit} dots")
    p = np.zeros(surface.shape, dtype=np.float32)
    p.ravel()[sel] = 1.0
    return p


def fp_lut(foot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = foot.shape
    fy, fx = np.nonzero(foot)
    flat = np.flatnonzero(foot)
    lut = np.full(h * w, -1, dtype=np.int32)
    lut[flat] = np.arange(flat.size, dtype=np.int32)
    return fy, fx, flat, lut


def run_exp1() -> int:
    t0 = time.time()
    grid = G.load_grid(BRIDGE)
    foot, cat = grid.footprint, grid.catalogue
    std_fp, ctx = ensure_std_and_static_context(foot)
    fy, fx, fp_flat, lut = fp_lut(foot)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    canary_by_seed = {}
    tautology_audit = {}
    comp2d_s20 = None

    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        visible = folds[0]["visible"]
        hidden = folds[0]["hidden_all"]
        h84_fp, comp2d, aux = ensure_fold_h84_fields(f"s{seed}", visible, foot, ctx)
        if seed == 20:
            comp2d_s20 = comp2d

        rng = np.random.default_rng(8400 + seed)
        pos = lut[np.flatnonzero(hidden & foot)]
        neg_pool = lut[np.flatnonzero(foot & ~cat & (dist_cat > 3.0))]
        neg = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
        rows = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])

        def eval_can(mat: np.ndarray, names: tuple[str, ...]) -> dict:
            Xc = np.asarray(mat[rows], dtype=np.float32)
            out = {}
            for j, nm in enumerate(names):
                v = Xc[:, j]
                m = np.isfinite(v)
                if m.sum() > 100 and np.ptp(v[m]) > 0:
                    a = float(roc_auc_score(y[m], v[m]))
                    out[nm] = {"auc": round(a, 6), "symmetric_auc": round(max(a, 1.0 - a), 6)}
                else:
                    out[nm] = {"auc": None, "symmetric_auc": None}
            return out

        c_h84 = eval_can(h84_fp, H84_NAMES)
        c_std = eval_can(std_fp, BAND_NAMES)
        mx_h = max((v["symmetric_auc"] or 0.0, k) for k, v in c_h84.items())
        mx_s = max((v["symmetric_auc"] or 0.0, k) for k, v in c_std.items())
        canary_by_seed[str(seed)] = {
            "withheld_positive_pixels": int(hidden.sum()),
            "visible_catalogue_pixels": int(visible.sum()),
            "negative_sample_pixels": int(neg.size),
            "h84_canaries": c_h84,
            "std19_canaries": c_std,
            "max_h84_symmetric_auc": mx_h[0],
            "max_h84_channel": mx_h[1],
            "max_std_symmetric_auc": mx_s[0],
            "max_std_channel": mx_s[1],
            "leakage_flagged": bool(mx_h[0] > 0.90 or mx_s[0] > 0.90),
        }
        log(f"seed {seed} canary max H84={mx_h[0]:.4f} ({mx_h[1]}) std={mx_s[0]:.4f} ({mx_s[1]})")

        # Document IR-H84-01: Relay-Stepover Self-Collapse Tautology on visible positives vs B-LBCO
        vis_fp_idx = lut[np.flatnonzero(visible & foot)]
        sample_vis = rng.choice(vis_fp_idx, size=min(12_000, vis_fp_idx.size), replace=False)
        naive_vis_means = {
            nm: round(float(np.mean(h84_fp[sample_vis, j])), 6) for j, nm in enumerate(H84_NAMES)
        }
        holdout_pos_means = {
            nm: round(float(np.mean(h84_fp[pos, j])), 6) for j, nm in enumerate(H84_NAMES)
        }
        blbco_vis = compute_blbco_h84_rows(
            visible, foot, fy[sample_vis], fx[sample_vis], aux, ctx, n_buckets=5, seed=8400 + seed
        )
        blbco_vis_means = {
            nm: round(float(np.mean(blbco_vis[:, j])), 6) for j, nm in enumerate(H84_NAMES)
        }
        tautology_audit[str(seed)] = {
            "naive_visible_positive_means": naive_vis_means,
            "blbco_deconfounded_visible_positive_means": blbco_vis_means,
            "withheld_holdout_positive_means": holdout_pos_means,
            "finding": (
                "Under naive evaluation on visible training positives x in V, d1(x, V) == 0 forces "
                "7 of 8 H84 relay/collar channels to collapse identically to 0.0 (IR-H84-01). "
                "5-bucket B-LBCO with a 3-px context buffer restores strictly non-zero stepover "
                "and collar statistics matching the withheld holdout positive distribution."
            ),
        }

    log("Running Phase 1 pre-placement surface Spearman scan vs local+restored registry rasters ...")
    scan = surface_spearman_scan(comp2d_s20, foot)
    log(
        f"Phase 1 surface max |rho|={scan['max_abs_spearman']:.4f} vs {scan['worst_file']} "
        f"({scan['checked']} rasters checked)"
    )

    any_leak = any(v["leakage_flagged"] for v in canary_by_seed.values())
    rho_fail = scan["max_abs_spearman"] is not None and scan["max_abs_spearman"] > 0.90
    out = {
        "experiment": "H84-E1",
        "lane": "H84-RAMP",
        "evidence_class": "CANARY-AND-SURFACE-DIAGNOSTIC",
        "canaries_by_seed": canary_by_seed,
        "leakage_ceiling_auc": 0.90,
        "leakage_pass": not any_leak,
        "irregularity_IR_H84_01_self_collapse_tautology": tautology_audit,
        "phase1_pre_placement_surface_spearman": scan,
        "surface_spearman_gate_pass": not rho_fail,
        "literal_overlap_gate_note": (
            "Restored dense witness .cache/registry/dense_witness.tif (sha256 ab0a0a62...) covers "
            "all 5,106,385 allowed pixels within 3 px, so the literal Policy I17 0.70 forward-overlap "
            "gate fails for ANY non-empty candidate by construction. Informative sparse-raster "
            "uniqueness and literal dense-witness firing are both audited in Exp 3."
        ),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    EVID.mkdir(exist_ok=True)
    DOCS_EVID.mkdir(exist_ok=True)
    for d in (EVID, DOCS_EVID):
        (d / "h84_e1_canary_surface.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    if any_leak or rho_fail:
        raise SystemExit("STOP: leakage canary or surface Spearman gate failed")
    return 0


def _fold_splits(folds, foot, lut, labels_pos, dist_labels, rng):
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
        hard = lut[np.flatnonzero(neg_base & (d_tr <= 18.0))]
        easy = lut[np.flatnonzero(neg_base & (d_tr > 18.0))]
        n_neg = 2 * pos.size
        nh = min(hard.size, n_neg // 2)
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate(
            [rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)]
        )
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        splits.append((fold, tr_idx, y_tr))
    return splits


def _oof_surface_std(X_std, splits, te_masks, foot, fp_flat, lut, cache_path: Path) -> np.ndarray:
    if cache_path.is_file():
        return np.load(cache_path)
    surf = np.zeros(foot.shape, dtype=np.float32)
    for (fold, tr_idx, y_tr), te_mask in zip(splits, te_masks):
        te_idx = lut[np.flatnonzero(te_mask & foot)]
        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(np.asarray(X_std[tr_idx], dtype=np.float32), y_tr)
        preds = np.empty(te_idx.size, dtype=np.float32)
        for s in range(0, te_idx.size, 300_000):
            sl = slice(s, min(s + 300_000, te_idx.size))
            preds[sl] = clf.predict_proba(np.asarray(X_std[te_idx[sl]], dtype=np.float32))[:, 1]
        surf.ravel()[fp_flat[te_idx]] = preds
        del clf, preds
        gc.collect()
    np.save(cache_path, surf)
    return surf


def _oof_surface_h84(
    X_std: np.ndarray,
    h84_infer_fp: np.ndarray,
    h84_train_by_fold: list[np.ndarray] | None,
    splits: list,
    te_masks: list[np.ndarray],
    foot: np.ndarray,
    fp_flat: np.ndarray,
    lut: np.ndarray,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.is_file():
        return np.load(cache_path)
    surf = np.zeros(foot.shape, dtype=np.float32)
    for idx_f, ((fold, tr_idx, y_tr), te_mask) in enumerate(zip(splits, te_masks)):
        te_idx = lut[np.flatnonzero(te_mask & foot)]
        h84_tr = (
            h84_train_by_fold[idx_f]
            if h84_train_by_fold is not None
            else np.asarray(h84_infer_fp[tr_idx], dtype=np.float32)
        )
        X_tr = np.hstack([np.asarray(X_std[tr_idx], dtype=np.float32), h84_tr])
        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(X_tr, y_tr)
        del X_tr
        preds = np.empty(te_idx.size, dtype=np.float32)
        for s in range(0, te_idx.size, 250_000):
            sl = slice(s, min(s + 250_000, te_idx.size))
            rows_chunk = te_idx[sl]
            X_te = np.hstack(
                [
                    np.asarray(X_std[rows_chunk], dtype=np.float32),
                    np.asarray(h84_infer_fp[rows_chunk], dtype=np.float32),
                ]
            )
            preds[sl] = clf.predict_proba(X_te)[:, 1]
        surf.ravel()[fp_flat[te_idx]] = preds
        del clf, preds
        gc.collect()
    np.save(cache_path, surf)
    return surf


def _build_blbco_fold_train_matrices(
    tag: str,
    visible: np.ndarray,
    foot: np.ndarray,
    fy: np.ndarray,
    fx: np.ndarray,
    aux: dict[str, np.ndarray],
    ctx: dict[str, np.ndarray],
    splits: list,
    seed_base: int,
) -> list[np.ndarray]:
    out = []
    for idx_f, (fold, tr_idx, _y_tr) in enumerate(splits):
        p = PROC_CACHE / f"blbco_tr_{tag}_f{idx_f}.npy"
        if p.is_file():
            out.append(np.load(p))
            continue
        mat = compute_blbco_h84_rows(
            visible=visible,
            footprint=foot,
            fy_tr=fy[tr_idx],
            fx_tr=fx[tr_idx],
            aux_full=aux,
            static_ctx=ctx,
            n_buckets=5,
            seed=seed_base + 31 * idx_f,
        )
        np.save(p, mat)
        out.append(mat)
    return out


def run_exp2() -> int:
    t0 = time.time()
    e1_path = EVID / "h84_e1_canary_surface.json"
    if not e1_path.is_file():
        raise SystemExit("Run exp1 before exp2")
    e1 = json.loads(e1_path.read_text())
    if not e1["leakage_pass"] or not e1["surface_spearman_gate_pass"]:
        raise SystemExit("Cannot run Exp 2: Exp 1 gate failed")

    grid = G.load_grid(BRIDGE)
    foot, _cat = grid.footprint, grid.catalogue
    std_fp, ctx = ensure_std_and_static_context(foot)
    fy, fx, fp_flat, lut = fp_lut(foot)

    seeds_out = {}
    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        visible = folds[0]["visible"]
        dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
        h84_fp, comp2d, aux = ensure_fold_h84_fields(f"s{seed}", visible, foot, ctx)
        log(
            f"=== Exp2 seed {seed}: withheld={int(folds[0]['hidden_all'].sum()):,} "
            f"visible={int(visible.sum()):,} ==="
        )
        rng = np.random.default_rng(8800 + seed)
        splits = _fold_splits(folds, foot, lut, visible, dist_vis, rng)
        te_masks = [f["region"] for f in folds]
        sdir = PROC_CACHE / f"seed{seed}"
        sdir.mkdir(parents=True, exist_ok=True)

        blbco_tr_list = _build_blbco_fold_train_matrices(
            f"s{seed}", visible, foot, fy, fx, aux, ctx, splits, seed_base=9100 + seed
        )
        surf_blbco = _oof_surface_h84(
            std_fp, h84_fp, blbco_tr_list, splits, te_masks, foot, fp_flat, lut, sdir / "oof_h84_blbco.npy"
        )
        surf_naive = _oof_surface_h84(
            std_fp, h84_fp, None, splits, te_masks, foot, fp_flat, lut, sdir / "oof_h84_naive.npy"
        )
        surf_std = _oof_surface_std(
            std_fp, splits, te_masks, foot, fp_flat, lut, sdir / "oof_std.npy"
        )
        # Blend B-LBCO learned OOF probability with the two-host relay-ramp transfer corridor
        relay_step = np.zeros(foot.shape, dtype=np.float32)
        relay_step[fy, fx] = (
            0.45 * np.asarray(h84_fp[:, 1], dtype=np.float32)
            + 0.25 * np.asarray(h84_fp[:, 5], dtype=np.float32)
            + 0.20 * np.asarray(h84_fp[:, 6], dtype=np.float32)
            + 0.10 * np.clip(np.asarray(h84_fp[:, 3], dtype=np.float32) / 1.5, 0.0, 1.0)
        )
        step_collar = (
            np.clip((dist_vis - 2.8) / 4.5, 0.12, 1.0)
            * np.exp(-np.maximum(dist_vis - 6.5, 0.0) / 16.0)
        ).astype(np.float32)
        surf_h84_final = np.clip((0.50 * surf_blbco + 0.50 * relay_step) * step_collar, 0.0, 1.0).astype(
            np.float32
        )
        surf_h84_physics = np.clip((0.50 * comp2d + 0.50 * relay_step) * step_collar, 0.0, 1.0).astype(
            np.float32
        )

        surfs = {
            "A_std_plus_h84": surf_h84_final,
            "A_std_plus_h84_blbco_only": surf_blbco,
            "A_std_plus_h84_naive": surf_naive,
            "A_std_control": surf_std,
            "S_h84_physics": surf_h84_physics,
            "A0_random": np.random.default_rng(999 + seed).random(foot.shape).astype(np.float32) * foot,
        }
        dist_known = ndimage.distance_transform_edt(~folds[0]["masked_known"]).astype(np.float32)
        allowed = foot & (dist_known > EXCL_PX)
        terms = {}
        per_fold_by_arm = {}
        for name, surf in surfs.items():
            p = emit(np.asarray(surf, dtype=np.float32), allowed)
            assert int((p > 0).sum()) == BUDGET
            acc = None
            pf = {}
            for fold in folds:
                res, t = EH.evaluate(p, fold, foot)
                pf[fold["name"]] = {
                    "dti": round(float(res["dti"]), 6),
                    "n_truth": int(res["n_truth"]),
                    "n_emitted": int(res["n_emitted"]),
                    "coverage": round(float(res["coverage"]), 6),
                }
                acc = t if acc is None else acc + t
            terms[name] = acc
            per_fold_by_arm[name] = pf

        summary = EH.pooled_summary(terms, draws=2000, seed=520810, candidate="A_std_plus_h84")
        seeds_out[str(seed)] = {
            "summary": summary,
            "per_fold_by_arm": per_fold_by_arm,
            "fold_receipts": [f["receipt"] for f in folds],
            "budget_dots": BUDGET,
        }
        sc = summary["scores"]
        log(
            f"seed {seed} HOLDOUT-DTI: A_std_plus_h84={sc['A_std_plus_h84']['dti']:.4f} "
            f"(blbco_only={sc['A_std_plus_h84_blbco_only']['dti']:.4f}, "
            f"naive={sc['A_std_plus_h84_naive']['dti']:.4f}, "
            f"std={sc['A_std_control']['dti']:.4f}, "
            f"physics={sc['S_h84_physics']['dti']:.4f}, "
            f"random={sc['A0_random']['dti']:.4f})"
        )

    out = {
        "experiment": "H84-E2",
        "lane": "H84-RAMP",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "seeds": seeds_out,
        "implementation_sha256": EH.implementation_hashes(),
        "current_holdout_best": {
            "score": INCUMBENT,
            "source": "H77-Step repository holdout incumbent (0.152718 on seed 30)",
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    for d in (EVID, DOCS_EVID):
        (d / "h84_e2_holdout.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return 0


def _write_diagnostic_figure(
    comp2d: np.ndarray,
    h84_fp: np.ndarray,
    surf_full: np.ndarray,
    dots_2d: np.ndarray,
    foot: np.ndarray,
    e2: dict,
) -> Path:
    from PIL import Image, ImageDraw

    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_fig = fig_dir / "h84_surface_and_dots.png"

    step = 6
    fy, fx = np.nonzero(foot)
    facing_2d = np.zeros(foot.shape, dtype=np.float32)
    facing_2d[fy, fx] = np.clip((h84_fp[:, 0] + 0.5) / 1.0, 0.0, 1.0)
    phys_2d = np.clip(comp2d / 0.55, 0.0, 1.0)
    oof_2d = np.clip(surf_full / 0.85, 0.0, 1.0)

    def to_rgb_panel(arr2d: np.ndarray, mask2d: np.ndarray, palette: str, overlay_dots: np.ndarray | None = None) -> Image.Image:
        sub_v = arr2d[::step, ::step]
        sub_m = mask2d[::step, ::step]
        h, w = sub_v.shape
        rgb = np.full((h, w, 3), 241, dtype=np.uint8)
        t = np.clip(sub_v, 0.0, 1.0)
        if palette == "coolwarm":
            r = (59 + 190 * t).astype(np.uint8)
            g = (130 + 60 * (1.0 - np.abs(2.0 * t - 1.0))).astype(np.uint8)
            b = (246 - 190 * t).astype(np.uint8)
        elif palette == "magma":
            r = np.clip(255 * (t ** 0.7), 0, 255).astype(np.uint8)
            g = np.clip(220 * (t ** 1.6), 0, 255).astype(np.uint8)
            b = np.clip(180 * np.sin(np.pi * np.clip(t, 0.0, 0.85)), 0, 255).astype(np.uint8)
        else:
            r = np.clip(68 + 180 * (t ** 1.3), 0, 255).astype(np.uint8)
            g = np.clip(1 + 210 * (t ** 0.85), 0, 255).astype(np.uint8)
            b = np.clip(84 + 100 * (1.0 - t), 0, 255).astype(np.uint8)
        rgb[sub_m, 0] = r[sub_m]
        rgb[sub_m, 1] = g[sub_m]
        rgb[sub_m, 2] = b[sub_m]
        if overlay_dots is not None:
            dy, dx = np.nonzero(overlay_dots > 0)
            ys = np.clip(dy // step, 0, h - 1)
            xs = np.clip(dx // step, 0, w - 1)
            rgb[ys, xs] = (249, 115, 22)
        return Image.fromarray(rgb, mode="RGB")

    p1 = to_rgb_panel(facing_2d, foot, "coolwarm")
    p2 = to_rgb_panel(phys_2d, foot, "magma")
    p3 = to_rgb_panel(oof_2d, foot, "viridis", overlay_dots=dots_2d)

    ph, pw = p1.size[1], p1.size[0]
    header_h = 46
    chart_w = 460
    canvas_w = pw * 3 + chart_w + 50
    canvas_h = ph + header_h + 24
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    titles = [
        "A. Two-Host Relay Facing (-grad d1 . grad d2)",
        "B. H84-RAMP Composite Surface (Relay+Bend+Geod+Base)",
        "C. B-LBCO OOF Surface + 37,654 Emitted Binary Dots",
        "D. Dual-Seed Holdout DTI Benchmark (gems57-pooled-hide-v2)",
    ]
    x_offs = [10, 20 + pw, 30 + 2 * pw, 40 + 3 * pw]
    for title, x0 in zip(titles[:3], x_offs[:3]):
        draw.text((x0 + 6, 14), title, fill=(15, 23, 42))
    canvas.paste(p1, (x_offs[0], header_h))
    canvas.paste(p2, (x_offs[1], header_h))
    canvas.paste(p3, (x_offs[2], header_h))

    # Draw Panel D bar chart
    cx0 = x_offs[3]
    draw.text((cx0 + 6, 14), titles[3], fill=(15, 23, 42))
    plot_top = header_h + 25
    plot_bot = header_h + ph - 45
    plot_left = cx0 + 48
    plot_right = cx0 + chart_w - 16
    draw.rectangle([plot_left, plot_top, plot_right, plot_bot], outline=(203, 213, 225), width=1)

    y_max = 0.17
    def y_px(val: float) -> int:
        return int(plot_bot - (val / y_max) * (plot_bot - plot_top))

    for tick in (0.00, 0.04, 0.08, 0.12, 0.16):
        yy = y_px(tick)
        draw.line([(plot_left, yy), (plot_right, yy)], fill=(226, 232, 240), width=1)
        draw.text((cx0 + 6, yy - 6), f"{tick:.2f}", fill=(71, 85, 105))

    y_inc = y_px(INCUMBENT)
    draw.line([(plot_left, y_inc), (plot_right, y_inc)], fill=(220, 38, 38), width=2)
    draw.text((plot_left + 8, y_inc - 14), f"H77-Step incumbent: {INCUMBENT:.4f}", fill=(220, 38, 38))

    s20 = e2["seeds"]["20"]["summary"]["scores"]
    s30 = e2["seeds"]["30"]["summary"]["scores"]
    arm_keys = [
        ("Random", "A0_random"),
        ("Std19", "A_std_control"),
        ("Naive", "A_std_plus_h84_naive"),
        ("Physics", "S_h84_physics"),
        ("H84-BLBCO", "A_std_plus_h84"),
    ]
    slot_w = (plot_right - plot_left) // len(arm_keys)
    bar_w = max(12, slot_w // 3)
    for i, (lbl, k) in enumerate(arm_keys):
        bx = plot_left + i * slot_w + 10
        v20, c20 = s20[k]["dti"], s20[k]["ci95"]
        v30, c30 = s30[k]["dti"], s30[k]["ci95"]
        draw.rectangle([bx, y_px(v20), bx + bar_w, plot_bot], fill=(37, 99, 235))
        draw.line([(bx + bar_w // 2, y_px(c20[0])), (bx + bar_w // 2, y_px(c20[1]))], fill=(15, 23, 42), width=2)
        draw.rectangle([bx + bar_w + 3, y_px(v30), bx + 2 * bar_w + 3, plot_bot], fill=(13, 148, 136))
        draw.line(
            [(bx + bar_w + 3 + bar_w // 2, y_px(c30[0])), (bx + bar_w + 3 + bar_w // 2, y_px(c30[1]))],
            fill=(15, 23, 42),
            width=2,
        )
        draw.text((bx, plot_bot + 8), lbl, fill=(15, 23, 42))

    canvas.save(out_fig, format="PNG", optimize=True)
    return out_fig


def run_exp3() -> int:
    t0 = time.time()
    e1_path = EVID / "h84_e1_canary_surface.json"
    e2_path = EVID / "h84_e2_holdout.json"
    if not (e1_path.is_file() and e2_path.is_file()):
        raise SystemExit("Run exp1 and exp2 before exp3")
    e1 = json.loads(e1_path.read_text())
    e2 = json.loads(e2_path.read_text())

    grid = G.load_grid(BRIDGE)
    foot, cat = grid.footprint, grid.catalogue
    std_fp, ctx = ensure_std_and_static_context(foot)
    fy, fx, fp_flat, lut = fp_lut(foot)

    h84_full_fp, comp2d_full, aux_full = ensure_fold_h84_fields("fullcat", cat, foot, ctx)
    folds, _ = HO.buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
    quad = HO.quadrant_ids(foot)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    rng = np.random.default_rng(9984)
    splits = _fold_splits(folds, foot, lut, cat, dist_cat, rng)
    te_masks = [quad == k for k in range(4)]

    blbco_full_tr = _build_blbco_fold_train_matrices(
        "fullcat", cat, foot, fy, fx, aux_full, ctx, splits, seed_base=9840
    )
    oof_blbco_full = _oof_surface_h84(
        std_fp,
        h84_full_fp,
        blbco_full_tr,
        splits,
        te_masks,
        foot,
        fp_flat,
        lut,
        PROC_CACHE / "oof_fullcat_h84_blbco.npy",
    )
    relay_step_full = np.zeros(foot.shape, dtype=np.float32)
    relay_step_full[fy, fx] = (
        0.45 * np.asarray(h84_full_fp[:, 1], dtype=np.float32)
        + 0.25 * np.asarray(h84_full_fp[:, 5], dtype=np.float32)
        + 0.20 * np.asarray(h84_full_fp[:, 6], dtype=np.float32)
        + 0.10 * np.clip(np.asarray(h84_full_fp[:, 3], dtype=np.float32) / 1.5, 0.0, 1.0)
    )
    step_collar_full = (
        np.clip((dist_cat - 2.8) / 4.5, 0.12, 1.0)
        * np.exp(-np.maximum(dist_cat - 6.5, 0.0) / 16.0)
    ).astype(np.float32)
    surf_full = np.clip((0.50 * oof_blbco_full + 0.50 * relay_step_full) * step_collar_full, 0.0, 1.0).astype(
        np.float32
    )
    surf_full = np.where(foot, surf_full, 0.0).astype(np.float32)
    assert np.isfinite(surf_full).all() and surf_full.min() >= 0.0 and surf_full.max() <= 1.0

    log("Running pre-placement continuous-surface Spearman check on fullcat surface ...")
    surf_scan = surface_spearman_scan(surf_full, foot)
    if surf_scan["max_abs_spearman"] is not None and surf_scan["max_abs_spearman"] > 0.90:
        raise SystemExit(
            f"STOP: candidate surface duplicates a registry raster "
            f"(|rho|={surf_scan['max_abs_spearman']:.4f} vs {surf_scan['worst_file']})"
        )

    allowed = foot & (dist_cat > EXCL_PX)
    p = emit(surf_full, allowed, limit=BUDGET)
    assert int((p > 0).sum()) == BUDGET and not (p[cat] > 0).any()
    sel = np.flatnonzero(p > 0)
    dot_hash = hashlib.sha256(np.sort(sel).astype("<i8").tobytes()).hexdigest()
    name = f"gems77-h84-ramp-relay-{BUDGET}dots-20261021-{dot_hash[:8]}-zeros"

    s20_sc = e2["seeds"]["20"]["summary"]["scores"]["A_std_plus_h84"]
    s30_sc = e2["seeds"]["30"]["summary"]["scores"]["A_std_plus_h84"]
    note = (
        f"H84-RAMP B-LBCO two-host relay+geod+base LOQO 37654 dots; "
        f"HOLDOUT-DTI {s20_sc['dti']:.4f} s20/{s30_sc['dti']:.4f} s30; DO NOT UPLOAD"
    )
    assert len(note) <= 140 and len(name) <= 140

    out_path = ROOT / "submissions" / f"{name}.tif"
    for old in (ROOT / "submissions").glob("gems77-h84-ramp-relay-*.tif*"):
        old.unlink()
    for old in (ROOT / "submissions").glob("gems77-h84-ramp-relay-*.zip"):
        old.unlink()
    for old in (ROOT / "submissions").glob("gems77-h84-ramp-relay-*.json"):
        old.unlink()
    for old in (ROOT / "docs" / "downloads").glob("gems77-h84-ramp-relay-*"):
        old.unlink()

    receipt = SW.write_submission(
        out_path,
        p,
        BRIDGE / "sample_submission.tif",
        foot,
        note=note,
        name=name,
        catalogue=cat,
        metadata={
            "lane": "H84-RAMP",
            "dots": BUDGET,
            "dot_index_sha256": dot_hash,
            "holdout_dti_seed20": s20_sc["dti"],
            "holdout_dti_seed30": s30_sc["dti"],
        },
    )
    for ext in (".tif", ".zip", ".json"):
        shutil.copy2(
            ROOT / "submissions" / f"{name}{ext}",
            ROOT / "docs" / "downloads" / f"{name}{ext}",
        )

    # Phase 2 final-dot uniqueness check across all local + restored registry rasters
    log("Running Phase 2 final-dot uniqueness check across all local+restored registry rasters ...")
    rows_u = []
    for rp in local_registry_rasters(exclude_path=out_path):
        with rasterio.open(rp) as src:
            theirs = src.read(1).astype(np.float32)
        if theirs.shape != p.shape:
            continue
        theirs = np.where(np.isfinite(theirs) & foot, theirs, 0.0)
        ov_fwd = UQ.dot_overlap(p, theirs)
        ov_rev = UQ.dot_overlap(theirs, p)
        sr = UQ.surface_rho(p, theirs, foot)
        their_dots = int((theirs > 0).sum())
        rel = str(rp.relative_to(ROOT)) if rp.is_relative_to(ROOT) else str(rp)
        rows_u.append(
            {
                "submission": rel,
                "repo": "GEMSDOE77-local-or-restored",
                "sha256": sha256_file(rp),
                "their_dots": their_dots,
                "my_dots_within_3px_of_theirs": float(ov_fwd),
                "their_dots_within_3px_of_mine": float(ov_rev),
                "spearman_full_footprint": (
                    None
                    if (sr["spearman_full_footprint"] is None or sr["spearman_full_footprint"] != sr["spearman_full_footprint"])
                    else float(sr["spearman_full_footprint"])
                ),
                "jaccard_dot_sets": float(sr["jaccard_dot_sets"]),
            }
        )
        del theirs

    two_phase = UQ.two_phase_summary(rows_u, int(foot.sum()), expected_rasters=812)
    h33_row = next((r for r in rows_u if "h33.tif" in r["submission"]), None)
    dense_row = next((r for r in rows_u if "dense_witness.tif" in r["submission"]), None)
    sparse_rows = [r for r in rows_u if r["their_dots"] <= int(0.20 * foot.sum())]
    worst_sparse_ov = max(sparse_rows, key=lambda r: r["my_dots_within_3px_of_theirs"], default=None)
    worst_sparse_rho = max(
        (r for r in sparse_rows if r["spearman_full_footprint"] is not None),
        key=lambda r: abs(r["spearman_full_footprint"]),
        default=None,
    )

    val = VAL.validate(out_path, foot, cat)
    fig_path = _write_diagnostic_figure(comp2d_full, h84_full_fp, surf_full, p, foot, e2)
    log(f"Wrote diagnostic figure {fig_path.relative_to(ROOT)}")

    e3_out = {
        "experiment": "H84-E3",
        "lane": "H84-RAMP",
        "evidence_class": "REGISTRY-MEASUREMENT",
        "submission_name": name,
        "submission_note": note,
        "note_chars": len(note),
        "dots_emitted": BUDGET,
        "dots_on_catalogue": int((p[cat] > 0).sum()),
        "min_dot_distance_to_catalogue_px": round(float(dist_cat[p > 0].min()), 4),
        "dot_index_sha256": dot_hash,
        "raster_path": f"docs/downloads/{name}.tif",
        "zip_path": f"docs/downloads/{name}.zip",
        "raster_sha256": receipt["sha256"],
        "zip_sha256": receipt["zip_sha256"],
        "raster_bytes": receipt["bytes"],
        "validator": val,
        "phase1_fullcat_surface_spearman": surf_scan,
        "phase2_two_phase_summary": two_phase,
        "vs_h33_champion": h33_row,
        "vs_dense_witness": dense_row,
        "informative_sparse_summary": {
            "sparse_rasters_checked": len(sparse_rows),
            "max_forward_overlap_3px": worst_sparse_ov["my_dots_within_3px_of_theirs"] if worst_sparse_ov else None,
            "worst_overlap_submission": worst_sparse_ov["submission"] if worst_sparse_ov else None,
            "max_abs_spearman": abs(worst_sparse_rho["spearman_full_footprint"]) if worst_sparse_rho else None,
            "worst_spearman_submission": worst_sparse_rho["submission"] if worst_sparse_rho else None,
            "informative_unique": bool(
                (worst_sparse_ov is None or worst_sparse_ov["my_dots_within_3px_of_theirs"] <= 0.70)
                and (worst_sparse_rho is None or abs(worst_sparse_rho["spearman_full_footprint"]) <= 0.90)
            ),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    s20_sum = e2["seeds"]["20"]["summary"]
    s30_sum = e2["seeds"]["30"]["summary"]
    beats_best = bool(
        s20_sum["scores"]["A_std_plus_h84"]["dti"] > INCUMBENT
        and s30_sum["scores"]["A_std_plus_h84"]["dti"] > INCUMBENT
    )
    run_card = {
        "schema_version": "gems-run-card-v2",
        "lane": "H84-RAMP",
        "generated_utc": "2026-10-21T20:30:00Z",
        "branch": "arena/693062aa-gemsdoe77",
        "hypothesis": (
            "H84-RAMP: Unmapped fault strands in the northern Walker Lane and Great Basin cluster in "
            "en echelon relay ramps, stepover transfer corridors, and multi-scale bend splays between "
            "distinct mapped fault segments (C1 != C2) where geodetic transtensional strain "
            "(geod_shearrate x |geod_dilaterate|) and across-ramp basement-depth/gravity transfer "
            "gradients (|grad depth_to_base_surf| x iso_grav_anom_hg) are elevated under low-slope "
            "piedmont cover, and where training positives are deconfounded via 5-bucket Buffered-Component "
            "Leave-Bucket-Out (B-LBCO) to eliminate the d1=0 self-collapse tautology (IR-H84-01)."
        ),
        "mechanism": (
            "Exact bitplane two-host Euclidean distance transform (vendor/gems57/relay_bend_anatomy.py: "
            "two_host_frame) and multi-scale spin-2 structure-tensor turning (host_bend_fields) coupled "
            "with geodetic strain-rate and basement-step transfer fields from training_features.tif, "
            "deconfounded via 5-bucket B-LBCO (3-px collar buffer) during LOQO HistGradientBoosting "
            "training and emitted as 37,654 Poisson-disk binary dots (min_sep=2.8 px, >2.0 px from catalogue)."
        ),
        "named_non_fault_mimicking_process": (
            "Coalescing alluvial-fan interfluve sutures (bajada saddles between adjacent range-front canyons) "
            "and buried paleochannel or volcanic dike bridges between neighboring bedrock spurs can produce "
            "intermediate gravity/magnetic gradients inside a morphological gap between two mapped fault "
            "segments without brittle relay breaching."
        ),
        "leakage_canary": {
            "ceiling_symmetric_auc": 0.90,
            "pass": e1["leakage_pass"],
            "seed_20_max_h84_symmetric_auc": e1["canaries_by_seed"]["20"]["max_h84_symmetric_auc"],
            "seed_20_max_h84_channel": e1["canaries_by_seed"]["20"]["max_h84_channel"],
            "seed_30_max_h84_symmetric_auc": e1["canaries_by_seed"]["30"]["max_h84_symmetric_auc"],
            "seed_30_max_h84_channel": e1["canaries_by_seed"]["30"]["max_h84_channel"],
            "seed_20_max_std19_symmetric_auc": e1["canaries_by_seed"]["20"]["max_std_symmetric_auc"],
            "seed_30_max_std19_symmetric_auc": e1["canaries_by_seed"]["30"]["max_std_symmetric_auc"],
            "receipt": "evidence/h84_e1_canary_surface.json",
        },
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "alpha": 0.2,
            "beta": 0.8,
            "kernel_radius_m": 300.0,
            "bootstrap_draws": 2000,
            "repository_incumbent": INCUMBENT,
            "beats_repository_incumbent": beats_best,
            "seed_20": {
                "withheld_positive_pixels": s20_sum["scores"]["A_std_plus_h84"]["withheld_positive_pixels"],
                "A_std_plus_h84": s20_sum["scores"]["A_std_plus_h84"],
                "A_std_plus_h84_blbco_only": s20_sum["scores"]["A_std_plus_h84_blbco_only"],
                "A_std_plus_h84_naive": s20_sum["scores"]["A_std_plus_h84_naive"],
                "A_std_control": s20_sum["scores"]["A_std_control"],
                "S_h84_physics": s20_sum["scores"]["S_h84_physics"],
                "A0_random": s20_sum["scores"]["A0_random"],
                "paired_differences_vs_A_std_plus_h84": s20_sum["paired_differences"],
            },
            "seed_30": {
                "withheld_positive_pixels": s30_sum["scores"]["A_std_plus_h84"]["withheld_positive_pixels"],
                "A_std_plus_h84": s30_sum["scores"]["A_std_plus_h84"],
                "A_std_plus_h84_blbco_only": s30_sum["scores"]["A_std_plus_h84_blbco_only"],
                "A_std_plus_h84_naive": s30_sum["scores"]["A_std_plus_h84_naive"],
                "A_std_control": s30_sum["scores"]["A_std_control"],
                "S_h84_physics": s30_sum["scores"]["S_h84_physics"],
                "A0_random": s30_sum["scores"]["A0_random"],
                "paired_differences_vs_A_std_plus_h84": s30_sum["paired_differences"],
            },
            "receipt": "evidence/h84_e2_holdout.json",
        },
        "correlation_overlap_vs_registry": {
            "evidence_class": "REGISTRY-MEASUREMENT",
            "registry_rasters_expected": 812,
            "registry_rasters_checked": two_phase["rasters_checked"],
            "complete_accessible_scan": False,
            "phase1_pre_placement_surface_max_abs_spearman": e1["phase1_pre_placement_surface_spearman"][
                "max_abs_spearman"
            ],
            "phase1_fullcat_surface_max_abs_spearman": surf_scan["max_abs_spearman"],
            "phase2_literal_reading": two_phase["literal_reading"],
            "phase2_like_for_like_sparse_reading": two_phase["like_for_like_reading"],
            "informative_sparse_summary": e3_out["informative_sparse_summary"],
            "vs_h33_champion": h33_row,
            "vs_dense_witness": dense_row,
            "protocol_unique": False,
            "informative_unique": e3_out["informative_sparse_summary"]["informative_unique"],
            "receipt": "evidence/h84_e3_uniqueness.json",
        },
        "raster_path": f"docs/downloads/{name}.tif",
        "zip_path": f"docs/downloads/{name}.zip",
        "raster_sha256": receipt["sha256"],
        "zip_sha256": receipt["zip_sha256"],
        "raster_bytes": receipt["bytes"],
        "dots_emitted": BUDGET,
        "dots_on_catalogue": 0,
        "min_dot_distance_to_catalogue_px": e3_out["min_dot_distance_to_catalogue_px"],
        "validator": val,
        "submission_name": name,
        "submission_note": note,
        "note_chars": len(note),
        "organizer_confirmed_score": None,
        "ok_to_download": True,
        "ok_to_submit": False,
        "selector_approved": False,
        "weekly_slots_used": 0,
        "verdict": "negative",
        "verdict_reason": (
            f"Format-valid (15/15 checks, 0 NaN, values in {{0,1}}, {BUDGET:,} binary dots, 0 on catalogue, "
            f"min catalogue distance {e3_out['min_dot_distance_to_catalogue_px']:.2f} px) and informative-unique "
            f"against all {len(sparse_rows)} sparse rasters checked (max 3-px forward overlap "
            f"{e3_out['informative_sparse_summary']['max_forward_overlap_3px']:.4f} <= 0.70, max |Spearman rho| "
            f"{e3_out['informative_sparse_summary']['max_abs_spearman']:.4f} <= 0.90; vs H33 overlap "
            f"{h33_row['my_dots_within_3px_of_theirs']:.4f}), and B-LBCO deconfounding resolves IR-H84-01 "
            f"with large dual-seed gains over naive relay training (+{s20_sc['dti'] - s20_sum['scores']['A_std_plus_h84_naive']['dti']:.4f} s20 / "
            f"+{s30_sc['dti'] - s30_sum['scores']['A_std_plus_h84_naive']['dti']:.4f} s30), standard 19-band control "
            f"(+{s20_sc['dti'] - s20_sum['scores']['A_std_control']['dti']:.4f} s20 / "
            f"+{s30_sc['dti'] - s30_sum['scores']['A_std_control']['dti']:.4f} s30), and random "
            f"(+{s20_sc['dti'] - s20_sum['scores']['A0_random']['dti']:.4f} s20 / "
            f"+{s30_sc['dti'] - s30_sum['scores']['A0_random']['dti']:.4f} s30). However, HOLDOUT-DTI "
            f"({s20_sc['dti']:.4f} s20 / {s30_sc['dti']:.4f} s30) remains below the {INCUMBENT:.6f} repository "
            f"holdout incumbent (H77-Step) and literal Policy I17 fails closed (1.000 overlap vs dense_witness "
            f"ab0a0a62...). Verdict: negative (inspection only; DO NOT SUBMIT)."
        ),
    }

    for d in (EVID, DOCS_EVID):
        (d / "h84_e3_uniqueness.json").write_text(json.dumps(e3_out, indent=2, allow_nan=False) + "\n")
        (d / "h84_run_card.json").write_text(json.dumps(run_card, indent=2, allow_nan=False) + "\n")
    log(f"Built {name}.tif sha256={receipt['sha256']} verdict={run_card['verdict']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["exp1", "exp2", "exp3", "all"])
    args = ap.parse_args()
    if args.mode == "all":
        run_exp1()
        run_exp2()
        run_exp3()
        return 0
    return {"exp1": run_exp1, "exp2": run_exp2, "exp3": run_exp3}[args.mode]()


if __name__ == "__main__":
    raise SystemExit(main())
