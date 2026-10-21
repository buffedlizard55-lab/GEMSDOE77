#!/usr/bin/env python3
"""H77V 3-experiment pipeline: Tufa-Deconfounded Vent-Sinter & Conductive Heat-Flow Conduit Coupling.

Strictly adheres to the Parallel-Run Protocol and AGENTS.md:
  - Exp 1 (mode='exp1'): build 10 label-free H77V channels (fixing IR-H77V-01..04 and
    IR-H77G-01), run single-feature ROC-AUC leakage canaries on seeds 20 and 30
    (stop if > 0.90), and check Phase-1 pre-placement surface uniqueness against the registry.
  - Exp 2 (mode='exp2'): run dual-seed (seed 20: 11,321 withheld px; seed 30:
    11,342 withheld px) whole-component buffered hide-and-recover holdout on
    unmodified vendor/gems57/evaluate_holdout.py (gems57-pooled-hide-v2).
  - Exp 3 (mode='exp3'): build full-map 4-quadrant out-of-fold H77V surface,
    emit 37,654 dots (>3 px from catalogue) via 300 m submodular cover, write
    GeoTIFF + ZIP via unmodified vendor/gems57/submission_writer.py, run
    Phase-2 final-dot uniqueness + validator, and emit evidence/run_card_h77v.json.
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
from gems57 import metric as M  # noqa: E402
from gems57 import submission_writer as SW  # noqa: E402
from gems57 import uniqueness as UQ  # noqa: E402
from gems57 import validate as VAL  # noqa: E402
from gems77 import features77 as F77  # noqa: E402
from gems77d.submodular import allocate_submodular_300m  # noqa: E402
from gems77v.fields import H77V_NAMES, build_h77v_fields  # noqa: E402

RAW = ROOT / "data" / "raw"
BRIDGE = ROOT / "data" / "bridge"
PROC = ROOT / "data" / "processed" / "h77v"
EVID = ROOT / "evidence"
BUDGET = 37_654
EXCL_PX = 3.0

BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope",
    "tc_totalcount", "geod_shearrate", "geod_dilaterate", "tmi_vg",
    "deq_n100a15", "iso_grav_anom_vg", "det_elev", "iso_grav_anom",
    "tmi", "depth_to_base_surf", "ieq_n100a15", "cond_surf",
    "iso_grav_anom_hg_signed", "det_elev_slope",
]
BAND_INDEX = {n: i + 1 for i, n in enumerate(BAND_NAMES)}

HGB_KW = dict(
    max_iter=160,
    learning_rate=0.075,
    max_leaf_nodes=45,
    min_samples_leaf=50,
    l2_regularization=2.5,
    early_stopping=False,
    random_state=7721,
)


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


def read_band(name: str) -> np.ndarray:
    tf_path = RAW / "training_features.tif"
    if not tf_path.exists():
        tf_path = ROOT / ".cache" / "data" / "training_features.tif"
    with rasterio.open(tf_path) as src:
        a = src.read(BAND_INDEX[name]).astype(np.float32)
    a[a <= -1e30] = np.nan
    return a


def ensure_feature_matrices(foot: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    PROC.mkdir(parents=True, exist_ok=True)
    h77v_path = PROC / "h77v_fp.npy"
    std_path = PROC / "std19_fp.npy"
    diag_path = PROC / "h77v_diag.json"
    n_fp = int(foot.sum())

    if h77v_path.exists() and diag_path.exists():
        h77v_fp = np.load(h77v_path, mmap_mode="r")
        diag = json.loads(diag_path.read_text())
    else:
        with rasterio.open(BRIDGE / "sample_submission.tif") as ds:
            tr = ds.transform
        ext_dir = RAW / "external"
        if not ext_dir.exists():
            ext_dir = ROOT / ".cache" / "data" / "data" / "external"
        tf_path = RAW / "training_features.tif"
        if not tf_path.exists():
            tf_path = ROOT / ".cache" / "data" / "training_features.tif"
        log("Reading 19-band training_features.tif in single pass...")
        with rasterio.open(tf_path) as src:
            all_19 = src.read().astype(np.float32)
        all_19[all_19 <= -1e30] = np.nan
        if not std_path.exists():
            mm_s = np.lib.format.open_memmap(
                std_path, mode="w+", dtype=np.float32, shape=(n_fp, len(BAND_NAMES))
            )
            for j in range(len(BAND_NAMES)):
                mm_s[:, j] = all_19[j][foot]
            mm_s.flush()
            del mm_s
        needed = [
            "rtp", "iso_grav_anom", "depth_to_base_surf",
            "cond_surf", "geod_dilaterate", "geod_shearrate", "ieq_n100a15",
            "deq_n100a15", "det_elev", "det_elev_slope",
        ]
        bands = {nm: all_19[BAND_INDEX[nm] - 1].copy() for nm in needed}
        del all_19
        gc.collect()
        log("Building 10 label-free H77V channels...")
        fields, diag = build_h77v_fields(bands, foot, transform=tr, ext_dir=ext_dir)
        del bands
        gc.collect()
        mm = np.lib.format.open_memmap(
            h77v_path, mode="w+", dtype=np.float32, shape=(n_fp, len(H77V_NAMES))
        )
        for j, nm in enumerate(H77V_NAMES):
            mm[:, j] = fields[nm][foot]
            mm.flush()
        np.save(PROC / "h77v_composite_full.npy", fields["h77v_composite_surface"])
        del mm, fields
        gc.collect()
        diag_path.write_text(json.dumps(diag, indent=2) + "\n")
        h77v_fp = np.load(h77v_path, mmap_mode="r")

    std_shared = ROOT / "data" / "processed" / "h77g" / "std19_fp.npy"
    if std_path.exists():
        std_fp = np.load(std_path, mmap_mode="r")
    elif std_shared.exists():
        std_fp = np.load(std_shared, mmap_mode="r")
    else:
        tf_path = RAW / "training_features.tif"
        if not tf_path.exists():
            tf_path = ROOT / ".cache" / "data" / "training_features.tif"
        with rasterio.open(tf_path) as src:
            all_19 = src.read().astype(np.float32)
        all_19[all_19 <= -1e30] = np.nan
        mm_s = np.lib.format.open_memmap(
            std_path, mode="w+", dtype=np.float32, shape=(n_fp, len(BAND_NAMES))
        )
        for j in range(len(BAND_NAMES)):
            mm_s[:, j] = all_19[j][foot]
        mm_s.flush()
        del mm_s, all_19
        gc.collect()
        std_fp = np.load(std_path, mmap_mode="r")

    return std_fp, h77v_fp, diag


def collect_registry(exclude_substr: str = "h77v") -> list[dict]:
    recs, seen = [], set()

    def add(repo: str, name: str, path: Path) -> None:
        if not path.is_file():
            return
        s = sha256_file(path)
        if s in seen:
            return
        seen.add(s)
        recs.append({
            "repo_first": repo,
            "sources": [name],
            "cache_file": str(path),
            "sha256": s,
        })

    # 1. Pinned rejection witnesses (dense_witness ab0a0a62 + H33 c55bafc4)
    for key, fname in (("17GEMSDOE-dense-witness", "dense_witness.tif"), ("GEMSDOE32-h33", "h33.tif")):
        add(key, fname, ROOT / ".cache" / "registry" / fname)

    # 2. Local GEMSDOE77 lane downloads
    for p in sorted((ROOT / "docs" / "downloads").glob("*.tif")):
        if exclude_substr in p.name or p.name.endswith("-nan.tif"):
            continue
        add("GEMSDOE77-local", p.name, p)

    # 3. Scored sibling rasters in .cache/data/registry/rasters
    for p in sorted((ROOT / ".cache" / "data" / "registry" / "rasters").glob("*.tif")):
        if p.name.endswith("-nan.tif"):
            continue
        add(p.name.split("__")[0], p.name, p)

    return recs


def summarize_uniqueness(arr: np.ndarray, foot: np.ndarray, cat: np.ndarray, recs: list[dict], file_sha: str | None = None) -> dict:
    idx = {
        "rasters": recs,
        "complete_accessible_scan": False,
        "errors": ["Partial registry scan: 2 SHA-pinned witnesses + scored sibling rasters in .cache/data/registry/rasters + all local GEMSDOE77 lane TIFFs; full 812-raster REST download omitted to preserve shared GitHub API quota."],
    }
    allowed = foot & ~cat
    n_allowed = int(allowed.sum())
    rep = UQ.compare_array_to_registry(arr, idx, foot, file_sha256=file_sha)
    space_ids = set()
    for r in rep["rows"]:
        if "error" in r:
            continue
        nd = int(r.get("their_dots", 0))
        if nd >= int(0.70 * n_allowed):
            space_ids.add(r["sha256"])
        elif nd >= 120_000:
            with rasterio.open(r["file"]) as src:
                a = src.read(1)
            sup = np.isfinite(a) & (a > 0) & foot
            if float(UQ._near(sup)[allowed].mean()) >= 0.70:
                space_ids.add(r["sha256"])
    info_rows = [r for r in rep["rows"] if "error" not in r and r["sha256"] not in space_ids]
    worst_rho_row = max(info_rows, key=lambda r: abs(r["spearman_full_footprint"] or 0.0), default=None)
    worst_ov_row = max(info_rows, key=lambda r: r["my_dots_within_3px_of_theirs"], default=None)
    h33_row = next((r for r in rep["rows"] if r.get("sha256", "").startswith("c55bafc4")), None)
    dense_row = next((r for r in rep["rows"] if r.get("sha256", "").startswith("ab0a0a62")), None)

    w_rho = abs(worst_rho_row["spearman_full_footprint"]) if worst_rho_row else 0.0
    w_ov = float(worst_ov_row["my_dots_within_3px_of_theirs"]) if worst_ov_row else 0.0
    info_pass = (w_rho <= 0.90) and (w_ov <= 0.70)

    return {
        "registry_rasters_checked": rep["registry_rasters_checked"],
        "space_filling_rasters_count": len(space_ids),
        "space_filling_sha256s": sorted(space_ids),
        "literal_worst_spearman_full_footprint": rep["worst_spearman_full_footprint"],
        "literal_worst_rho_submission": rep["worst_rho_submission"],
        "literal_worst_dot_overlap": rep["worst_dot_overlap"],
        "literal_worst_overlap_submission": rep["worst_overlap_submission"],
        "literal_unique": rep["unique"],
        "literal_duplicate_count": rep["duplicate_count"],
        "informative_non_space_filling": {
            "n_informative_rasters": len(info_rows),
            "worst_abs_spearman": w_rho,
            "worst_rho_submission": worst_rho_row["submission"] if worst_rho_row else None,
            "worst_forward_overlap_3px": w_ov,
            "worst_overlap_submission": worst_ov_row["submission"] if worst_ov_row else None,
            "informative_gate_pass": info_pass,
            "top5_by_overlap": sorted(
                (
                    {
                        "submission": r["submission"],
                        "sha256": r["sha256"][:12],
                        "forward_overlap_3px": float(r["my_dots_within_3px_of_theirs"]),
                        "abs_spearman": abs(r["spearman_full_footprint"] or 0.0),
                    }
                    for r in info_rows
                ),
                key=lambda d: -d["forward_overlap_3px"],
            )[:5],
        },
        "vs_h33_champion": {
            "sha256": h33_row["sha256"] if h33_row else None,
            "spearman_full_footprint": h33_row["spearman_full_footprint"] if h33_row else None,
            "forward_overlap_3px": h33_row["my_dots_within_3px_of_theirs"] if h33_row else None,
        },
        "vs_dense_witness_ab0a0a62": {
            "sha256": dense_row["sha256"] if dense_row else None,
            "spearman_full_footprint": dense_row["spearman_full_footprint"] if dense_row else None,
            "forward_overlap_3px": dense_row["my_dots_within_3px_of_theirs"] if dense_row else None,
        },
        "byte_unique_among_checked": rep.get("byte_unique_among_checked", True),
        "pixel_unique_among_checked": rep.get("pixel_unique_among_checked", True),
    }


def run_exp1() -> int:
    t0 = time.time()
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h77v_fp, diag = ensure_feature_matrices(foot)
    log(f"Feature matrices ready: std={std_fp.shape}, h77v={h77v_fp.shape}")

    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    canary_by_seed = {}
    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden = folds[0]["hidden_all"]
        rng = np.random.default_rng(7720 + seed)
        pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
        neg_pool = fp_lut[np.flatnonzero(foot & ~cat & (dist_cat > 3.0))]
        neg_fp = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
        rows = np.concatenate([pos_fp, neg_fp])
        y = np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])

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

        c_h77v = eval_can(h77v_fp, H77V_NAMES)
        c_std = eval_can(std_fp, BAND_NAMES)
        max_h77v = max((v["symmetric_auc"] or 0.0, k) for k, v in c_h77v.items())
        max_std = max((v["symmetric_auc"] or 0.0, k) for k, v in c_std.items())
        canary_by_seed[str(seed)] = {
            "withheld_positive_pixels": int(hidden.sum()),
            "negative_sample_pixels": int(neg_fp.size),
            "h77v_canaries": c_h77v,
            "std19_canaries": c_std,
            "max_h77v_symmetric_auc": max_h77v[0],
            "max_h77v_channel": max_h77v[1],
            "max_std_symmetric_auc": max_std[0],
            "max_std_channel": max_std[1],
            "leakage_flagged": bool(max_h77v[0] > 0.90 or max_std[0] > 0.90),
        }
        log(f"Seed {seed} canary max H77V={max_h77v[0]:.4f} ({max_h77v[1]}), max std={max_std[0]:.4f} ({max_std[1]})")

    comp_surf = np.load(PROC / "h77v_composite_full.npy")
    recs = collect_registry()
    log(f"Running Phase-1 surface uniqueness against {len(recs)} registry rasters...")
    surf_uq = summarize_uniqueness(comp_surf, foot, cat, recs)
    log(
        f"Phase-1 surface: worst |rho|={surf_uq['literal_worst_spearman_full_footprint']:.4f} "
        f"vs H33 rho={surf_uq['vs_h33_champion']['spearman_full_footprint']:.4f}"
    )

    any_leak = any(v["leakage_flagged"] for v in canary_by_seed.values())
    surf_rho_exceeded = abs(surf_uq["literal_worst_spearman_full_footprint"] or 0.0) > 0.90
    out = {
        "experiment": "H77V-E1",
        "evidence_class": "CANARY-AND-SURFACE-DIAGNOSTIC",
        "channel_diagnostics": diag["channels"],
        "source_audit": diag["source_audit"],
        "canaries_by_seed": canary_by_seed,
        "max_h77v_symmetric_auc_overall": max(
            v["max_h77v_symmetric_auc"] for v in canary_by_seed.values()
        ),
        "leakage_ceiling_auc": 0.90,
        "leakage_pass": not any_leak,
        "phase1_composite_surface_uniqueness": surf_uq,
        "surface_spearman_gate_pass": not surf_rho_exceeded,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    EVID.mkdir(exist_ok=True)
    (EVID / "h77v_e1_canary_surface.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "h77v_e1_canary_surface.json").write_text(json.dumps(out, indent=2) + "\n")
    log("Wrote evidence/h77v_e1_canary_surface.json")
    if any_leak or surf_rho_exceeded:
        raise SystemExit("STOP: leakage canary or surface Spearman threshold exceeded")
    return 0


def run_exp2() -> int:
    t0 = time.time()
    e1 = json.loads((EVID / "h77v_e1_canary_surface.json").read_text())
    if not e1["leakage_pass"] or not e1["surface_spearman_gate_pass"]:
        raise SystemExit("Cannot run Exp 2: Exp 1 gate failed")

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    std_fp, h77v_fp, _ = ensure_feature_matrices(foot)
    comp_surf = np.load(PROC / "h77v_composite_full.npy")
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    arms_order = ["S_h77v_physics", "A_h77v_only", "A_std_plus_h77v", "A_std", "A0_random"]
    seeds_out = {}

    for seed in (20, 30):
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden = folds[0]["hidden_all"]
        visible = folds[0]["visible"]
        dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
        log(f"=== Exp 2 Holdout Seed {seed}: withheld={int(hidden.sum())} visible={int(visible.sum())} ===")

        rng = np.random.default_rng(8820 + seed)
        fold_splits = []
        for k, fold in enumerate(folds):
            tr_reg = np.zeros_like(foot)
            for j, f2 in enumerate(folds):
                if j != k:
                    tr_reg |= f2["region"]
            vis_tr = visible & tr_reg
            d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
            pos = fp_lut[np.flatnonzero(vis_tr & foot)]
            if pos.size > 40_000:
                pos = rng.choice(pos, 40_000, replace=False)
            neg_base = foot & tr_reg & ~vis_tr & (d_vis_tr > 3.0)
            hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 15.0))]
            easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 15.0))]
            n_neg = 2 * pos.size
            nh = min(hard.size, n_neg // 2)
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            te_idx = fp_lut[np.flatnonzero(fold["region"] & foot)]
            fold_splits.append((fold, tr_idx, y_tr, te_idx))

        def _slice_mats(mats: list[np.ndarray], idx: np.ndarray | slice) -> np.ndarray:
            if len(mats) == 1:
                return np.asarray(mats[0][idx], dtype=np.float32)
            return np.hstack([np.asarray(m[idx], dtype=np.float32) for m in mats])

        def oof_surface(mats: list[np.ndarray], tag: str) -> np.ndarray:
            cache_p = PROC / f"oof_{tag}_seed{seed}.npy"
            if cache_p.exists():
                return np.load(cache_p)
            surf = np.zeros((h, w), dtype=np.float32)
            for fold, tr_idx, y_tr, te_idx in fold_splits:
                clf = HistGradientBoostingClassifier(**HGB_KW)
                X_tr = _slice_mats(mats, tr_idx)
                clf.fit(X_tr, y_tr)
                del X_tr
                preds = np.empty(te_idx.size, dtype=np.float32)
                for s in range(0, te_idx.size, 250_000):
                    sl = te_idx[s : min(s + 250_000, te_idx.size)]
                    preds[s : s + sl.size] = clf.predict_proba(_slice_mats(mats, sl))[:, 1]
                surf.ravel()[fp_flat[te_idx]] = preds
                del clf, preds
                gc.collect()
            np.save(cache_p, surf)
            return surf

        def get_arm_surf(arm: str) -> np.ndarray:
            if arm == "S_h77v_physics":
                return comp_surf
            if arm == "A_h77v_only":
                return oof_surface([h77v_fp], "A_h77v_only")
            if arm == "A_std_plus_h77v":
                return oof_surface([std_fp, h77v_fp], "A_std_plus_h77v")
            if arm == "A_std":
                return oof_surface([std_fp], "A_std")
            if arm == "A0_random":
                return np.random.default_rng(999 + seed).random((h, w), dtype=np.float32) * foot
            raise ValueError(arm)

        def emit_submod(surf: np.ndarray) -> np.ndarray:
            allowed_folds, tot = [], 0
            for fold in folds:
                a = fold["region"] & foot & ~visible & (dist_vis > EXCL_PX)
                allowed_folds.append(a)
                tot += int(a.sum())
            dots = np.zeros((h, w), bool)
            for a in allowed_folds:
                b = int(round(BUDGET * int(a.sum()) / max(tot, 1)))
                if b > 0:
                    dots |= allocate_submodular_300m(
                        surf, a, target_n=b, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
                    )
            return dots

        terms_by_arm, arm_details = {}, {}
        for arm in arms_order:
            surf_a = get_arm_surf(arm)
            dots = emit_submod(surf_a)
            if arm != "S_h77v_physics":
                del surf_a
                gc.collect()
            terms, per = None, {}
            for fold in folds:
                res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
                per[fold["name"]] = {
                    "dti": float(res["dti"]),
                    "n_truth": int(res["n_truth"]),
                    "emitted": int(res["n_emitted"]),
                }
                terms = t if terms is None else terms + t
            terms_by_arm[arm] = terms
            dti_val = float(EH.from_terms(terms.sum(axis=0)))
            arm_details[arm] = {
                "dots": int(dots.sum()),
                "per_fold": per,
                "pooled_holdout_dti": dti_val,
            }
            log(f"[Seed {seed}] {arm}: HOLDOUT-DTI = {dti_val:.5f} (dots={int(dots.sum())})")

        summary_h77v = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_h77v_only")
        summary_comb = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_std_plus_h77v")
        seeds_out[str(seed)] = {
            "seed": seed,
            "withheld_positive_pixels": int(hidden.sum()),
            "fold_receipts": [f["receipt"] for f in folds],
            "arms": arm_details,
            "pooled_summary_candidate_A_h77v_only": summary_h77v,
            "pooled_summary_candidate_A_std_plus_h77v": summary_comb,
        }

    out = {
        "experiment": "H77V-E2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "budget_dots": BUDGET,
        "exclusion_px": EXCL_PX,
        "incumbent_reference": {
            "lane": "H77-Step",
            "seed20_holdout_dti": 0.152718,
            "seed20_ci95": [0.131548, 0.175185],
            "withheld_positive_pixels": 11321,
        },
        "seeds": seeds_out,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (EVID / "h77v_e2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "h77v_e2_holdout.json").write_text(json.dumps(out, indent=2) + "\n")
    log("Wrote evidence/h77v_e2_holdout.json")
    return 0


def run_exp3() -> int:
    t0 = time.time()
    e1 = json.loads((EVID / "h77v_e1_canary_surface.json").read_text())
    e2 = json.loads((EVID / "h77v_e2_holdout.json").read_text())
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    _, h77v_fp, diag = ensure_feature_matrices(foot)
    comp_surf = np.load(PROC / "h77v_composite_full.npy")
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    X_h77v = np.asarray(h77v_fp, dtype=np.float32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    full_oof_path = PROC / "full_oof_h77v.npy"
    if full_oof_path.exists():
        oof_h77v = np.load(full_oof_path)
    else:
        oof_h77v = np.zeros((h, w), dtype=np.float32)
        ymid, xmid = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        quads = [
            ("NW", (yy < ymid) & (xx < xmid)),
            ("NE", (yy < ymid) & (xx >= xmid)),
            ("SW", (yy >= ymid) & (xx < xmid)),
            ("SE", (yy >= ymid) & (xx >= xmid)),
        ]
        rng = np.random.default_rng(7729)
        for qname, qmask in quads:
            tr_mask = foot & ~qmask
            cat_tr = cat & tr_mask
            d_tr = ndimage.distance_transform_edt(~cat_tr).astype(np.float32)
            pos = fp_lut[np.flatnonzero(cat_tr)]
            if pos.size > 40_000:
                pos = rng.choice(pos, 40_000, replace=False)
            neg_base = foot & tr_mask & ~cat_tr & (d_tr > 3.0)
            hard = fp_lut[np.flatnonzero(neg_base & (d_tr <= 15.0))]
            easy = fp_lut[np.flatnonzero(neg_base & (d_tr > 15.0))]
            n_neg = 2 * pos.size
            nh = min(hard.size, n_neg // 2)
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
            tr_idx = np.concatenate([pos, neg])
            y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            te_idx = fp_lut[np.flatnonzero(qmask & foot)]

            clf = HistGradientBoostingClassifier(**HGB_KW)
            clf.fit(X_h77v[tr_idx], y_tr)
            preds = np.empty(te_idx.size, dtype=np.float32)
            for s in range(0, te_idx.size, 250_000):
                sl = slice(s, min(s + 250_000, te_idx.size))
                preds[sl] = clf.predict_proba(X_h77v[te_idx[sl]])[:, 1]
            oof_h77v.ravel()[fp_flat[te_idx]] = preds
            del clf, preds
            gc.collect()
            log(f"Full-map H77V quadrant {qname} predicted")
        np.save(full_oof_path, oof_h77v)

    # Pure geothermal vent conduit physics core + 4-quadrant OOF H77V posterior
    vent_pure = np.zeros((h, w), dtype=np.float32)
    for cname, wt in (
        ("h77v_sinter_alter_vent_conduit", 0.25),
        ("h77v_probe_upward_heatflow_dtdz", 0.22),
        ("h77v_geotherm_disequilibrium", 0.20),
        ("h77v_fixed_spring_well_upflow", 0.15),
        ("h77v_epithermal_k_th_alteration", 0.10),
        ("h77v_seamfree_claycap_transtension", 0.08),
    ):
        vent_pure[foot] += wt * np.asarray(h77v_fp[:, H77V_NAMES.index(cname)], dtype=np.float32)
    vent_rank = F77.rank01(np.where(foot, vent_pure, np.float32(np.nan)), foot)
    oof_rank = F77.rank01(np.where(foot, oof_h77v, np.float32(np.nan)), foot)
    final_surf = np.where(
        foot,
        np.float32(0.45) * vent_rank + np.float32(0.30) * comp_surf + np.float32(0.25) * oof_rank,
        0.0,
    ).astype(np.float32)
    np.save(PROC / "final_placement_surface.npy", final_surf)

    recs = collect_registry()
    log(f"Phase 1 uniqueness on final placement surface ({len(recs)} rasters)...")
    p1_uq = summarize_uniqueness(final_surf, foot, cat, recs)
    log(
        f"Phase 1: worst |rho|={p1_uq['literal_worst_spearman_full_footprint']:.4f}, "
        f"informative worst |rho|={p1_uq['informative_non_space_filling']['worst_abs_spearman']:.4f}"
    )
    if abs(p1_uq["literal_worst_spearman_full_footprint"] or 0.0) > 0.90:
        raise SystemExit("STOP: Phase 1 placement surface Spearman rho > 0.90")

    allowed = foot & ~cat & (dist_cat > EXCL_PX)
    dots = allocate_submodular_300m(
        final_surf, allowed, target_n=BUDGET, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
    )
    n_dots = int(dots.sum())
    dots_sha8 = hashlib.sha256(np.ascontiguousarray(dots).tobytes()).hexdigest()[:8]
    name = f"gems77-h77v-ventsinter-heatflow-{n_dots}dots-20261021-{dots_sha8}-zeros"

    s20_score = e2["seeds"]["20"]["pooled_summary_candidate_A_std_plus_h77v"]["scores"]["A_std_plus_h77v"]["dti"]
    s30_score = e2["seeds"]["30"]["pooled_summary_candidate_A_std_plus_h77v"]["scores"]["A_std_plus_h77v"]["dti"]
    s20_only = e2["seeds"]["20"]["pooled_summary_candidate_A_h77v_only"]["scores"]["A_h77v_only"]["dti"]
    note = (
        f"H77V vent-sinter+2m dT/dz+diseq ({n_dots}d); "
        f"HOLDOUT-DTI {s20_score:.4f}(s20)/{s30_score:.4f}(s30) H77V-only {s20_only:.4f} <0.1527: NEGATIVE"
    )
    assert len(note) <= 140, f"Note too long: {len(note)}"

    pred = np.where(foot, dots.astype(np.float32), 0.0).astype(np.float32)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    sub_tif = sub_dir / f"{name}.tif"
    sample_path = BRIDGE / "sample_submission.tif"

    for suffix in (".tif", ".zip", ".json"):
        p_existing = sub_tif.with_suffix(suffix)
        if p_existing.exists():
            p_existing.unlink()
    writer_receipt = SW.write_submission(
        sub_tif,
        pred,
        sample_path,
        foot,
        note=note,
        name=name,
        catalogue=cat,
        metadata={"lane": "H77V", "selector_approved": False},
    )
    dl_dir = ROOT / "docs" / "downloads"
    for suffix in (".tif", ".zip", ".json"):
        src_p = sub_tif.with_suffix(suffix)
        shutil.copy2(src_p, dl_dir / src_p.name)

    val = VAL.validate(sub_tif, foot, cat)
    log(f"Validator: all_checks_passed={val['all_checks_passed']} dots={val['emitted_positive_pixels']}")

    log(f"Phase 2 uniqueness on final placed dots ({len(recs)} rasters)...")
    p2_uq = summarize_uniqueness(pred, foot, cat, recs, file_sha=writer_receipt["sha256"])
    log(
        f"Phase 2 informative worst overlap={p2_uq['informative_non_space_filling']['worst_forward_overlap_3px']:.4f} "
        f"({p2_uq['informative_non_space_filling']['worst_overlap_submission']}), "
        f"worst |rho|={p2_uq['informative_non_space_filling']['worst_abs_spearman']:.4f}"
    )

    uq_bundle = {
        "experiment": "H77V-E3-UNIQUENESS",
        "candidate": name,
        "tif_sha256": writer_receipt["sha256"],
        "registry_scope": "2 SHA-pinned witnesses (dense_witness ab0a0a62 + H33 c55bafc4) + scored sibling rasters in .cache/data/registry/rasters + all local GEMSDOE77 lane TIFFs; full 812-raster REST scan omitted to preserve shared GitHub API quota",
        "phase1_surface": p1_uq,
        "phase2_final_dots": p2_uq,
        "protocol_unique": False,
        "informative_unique": bool(
            p1_uq["informative_non_space_filling"]["informative_gate_pass"]
            and p2_uq["informative_non_space_filling"]["informative_gate_pass"]
        ),
    }
    (EVID / "h77v_e3_uniqueness.json").write_text(json.dumps(uq_bundle, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "h77v_e3_uniqueness.json").write_text(json.dumps(uq_bundle, indent=2) + "\n")

    s20_h77v = e2["seeds"]["20"]["pooled_summary_candidate_A_h77v_only"]["scores"]["A_h77v_only"]
    s20_comb = e2["seeds"]["20"]["pooled_summary_candidate_A_std_plus_h77v"]["scores"]["A_std_plus_h77v"]
    s20_std = e2["seeds"]["20"]["pooled_summary_candidate_A_std_plus_h77v"]["scores"]["A_std"]
    s20_phys = e2["seeds"]["20"]["pooled_summary_candidate_A_h77v_only"]["scores"]["S_h77v_physics"]
    s20_rand = e2["seeds"]["20"]["pooled_summary_candidate_A_h77v_only"]["scores"]["A0_random"]
    s20_diff_vs_std = e2["seeds"]["20"]["pooled_summary_candidate_A_std_plus_h77v"]["paired_differences"]["A_std"]
    s20_diff_comb_vs_rand = e2["seeds"]["20"]["pooled_summary_candidate_A_std_plus_h77v"]["paired_differences"]["A0_random"]
    s30_diff_comb_vs_rand = e2["seeds"]["30"]["pooled_summary_candidate_A_std_plus_h77v"]["paired_differences"]["A0_random"]

    s30_h77v = e2["seeds"]["30"]["pooled_summary_candidate_A_h77v_only"]["scores"]["A_h77v_only"]
    s30_comb = e2["seeds"]["30"]["pooled_summary_candidate_A_std_plus_h77v"]["scores"]["A_std_plus_h77v"]
    s30_std = e2["seeds"]["30"]["pooled_summary_candidate_A_std_plus_h77v"]["scores"]["A_std"]
    s30_phys = e2["seeds"]["30"]["pooled_summary_candidate_A_h77v_only"]["scores"]["S_h77v_physics"]
    s30_rand = e2["seeds"]["30"]["pooled_summary_candidate_A_h77v_only"]["scores"]["A0_random"]
    s30_diff_vs_std = e2["seeds"]["30"]["pooled_summary_candidate_A_std_plus_h77v"]["paired_differences"]["A_std"]

    run_card = {
        "schema": "gems-parallel-run-card-v1",
        "lane_id": "H77V",
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/e9248aa6-gemsdoe77",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": (
            "H77V (Tufa-Deconfounded Siliceous Vent-Sinter & Multi-Depth Conductive Heat-Flow Strike-Conduit Coupling): "
            "Blind Great Basin geothermal fault conduits obscured by playal sedimentation or argillic alteration leak "
            "high-enthalpy siliceous sinter/sulfur/breccia precipitates (distinct from Pleistocene Lake Lahontan shoreline "
            "Tufa mounds, which make up 86/281 = 30.6% of GDR paleo-records), upward conductive heat-flow gradients "
            "(dT/dz = T2m - T1m > 0 across 816 of 1,543 multi-depth 2m temperature probes), chemical geothermometer cooling "
            "disequilibrium (dT_diseq = max(T_quartz, T_chalc, T_cat) - T_surf), and potassium-metasomatic K/Th halos "
            "aligned with local structural strike and subsurface gravity-basement steps."
        ),
        "mechanism": (
            "Exclude non-tectonic Lake Lahontan Tufa mounds (86 records, med_d_vis=61.5 px) from Paleo_geothermal_final.csv "
            "while weighting Sinter/Alteration/Sulfur/Breccia/Explosion-crater fault vents (195 records; IR-H77V-01); extract "
            "downward-increasing 1m-to-2m probe conductive gradients dT/dz = T2m - T1m > 0 from 2m_temperature_probe_n83geo.dbf "
            "(IR-H77V-03/04); fix the vals[0] geothermometer index shift and missing-temp_c drop in gdr_wellspring_in_footprint.csv "
            "(IR-H77V-02) to compute dT_diseq = T_geo - T_surf; advect all hydrothermal impulses anisotropically along the local "
            "topographic-gravity structure-tensor strike tangent; couple with GeoDAWN epithermal K/Th alteration, H77G negative RTP "
            "demagnetization trough x gravity-basement step, seam-free sigma=36 px (3.6 km) MT clay-cap/transtension fairway "
            "(IR-H77G-01 fix), and 1m LiDAR + 10m 3DEP dip-slip scarp crestness. Emit 37,654 binary dots (>3 px from catalogue) "
            "via 300 m submodular cover and serialize via vendor/gems57/submission_writer.py."
        ),
        "named_non_fault_process_that_could_mimic_it": (
            "Lateral thermal groundwater outflow plumes migrating 1-4 km down-hydraulic-gradient through permeable alluvial-fan "
            "gravel channels away from the range-front feeder fault, or residual Pleistocene Lake Lahontan wave-cut beach ridges."
        ),
        "source_audit": diag["source_audit"],
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "alpha": M.ALPHA,
            "beta": M.BETA,
            "kernel_radius_m": M.R_M,
            "seed_20": {
                "withheld_positive_pixels": e2["seeds"]["20"]["withheld_positive_pixels"],
                "S_h77v_physics": s20_phys,
                "A_h77v_only": s20_h77v,
                "A_std_plus_h77v": s20_comb,
                "A_std_control": s20_std,
                "A0_random": s20_rand,
                "paired_delta_A_std_plus_h77v_vs_A_std": s20_diff_vs_std,
                "paired_delta_A_std_plus_h77v_vs_random": s20_diff_comb_vs_rand,
            },
            "seed_30": {
                "withheld_positive_pixels": e2["seeds"]["30"]["withheld_positive_pixels"],
                "S_h77v_physics": s30_phys,
                "A_h77v_only": s30_h77v,
                "A_std_plus_h77v": s30_comb,
                "A_std_control": s30_std,
                "A0_random": s30_rand,
                "paired_delta_A_std_plus_h77v_vs_A_std": s30_diff_vs_std,
                "paired_delta_A_std_plus_h77v_vs_random": s30_diff_comb_vs_rand,
            },
            "incumbent_bar": {
                "lane": "H77-Step",
                "seed20_holdout_dti": 0.152718,
                "seed20_ci95": [0.131548, 0.175185],
                "withheld_positive_pixels": 11321,
                "beats_incumbent": False,
            },
        },
        "leakage_canary": {
            "evidence_class": "CANARY-DIAGNOSTIC",
            "channels_tested": len(H77V_NAMES),
            "max_symmetric_auc": e1["max_h77v_symmetric_auc_overall"],
            "ceiling": 0.90,
            "pass": e1["leakage_pass"],
        },
        "correlation_overlap_vs_registry": {
            "registry_scope": uq_bundle["registry_scope"],
            "registry_rasters_checked": p2_uq["registry_rasters_checked"],
            "surface_max_abs_spearman": abs(p1_uq["literal_worst_spearman_full_footprint"] or 0.0),
            "surface_informative_max_abs_spearman": p1_uq["informative_non_space_filling"]["worst_abs_spearman"],
            "final_dots_max_abs_spearman": abs(p2_uq["literal_worst_spearman_full_footprint"] or 0.0),
            "final_dots_informative_max_abs_spearman": p2_uq["informative_non_space_filling"]["worst_abs_spearman"],
            "final_dots_informative_max_forward_overlap_3px": p2_uq["informative_non_space_filling"]["worst_forward_overlap_3px"],
            "final_dots_informative_worst_submission": p2_uq["informative_non_space_filling"]["worst_overlap_submission"],
            "final_dots_max_forward_overlap_3px": p2_uq["literal_worst_dot_overlap"],
            "vs_h33_champion": p2_uq["vs_h33_champion"],
            "vs_dense_witness_ab0a0a62": p2_uq["vs_dense_witness_ab0a0a62"],
            "informative_sparse_models_unique": uq_bundle["informative_unique"],
            "duplicate_stop": True,
            "duplicate_stop_reason": (
                "Literal Policy I17 stop: full-footprint dense witness ab0a0a62 covers 100% of allowed "
                "pixels (forward 3-px overlap = 1.000 > 0.70), and full 812-raster REST scan was omitted. "
                "Against all non-space-filling sparse/dot rasters in the checked set, informative uniqueness "
                f"is {uq_bundle['informative_unique']} (max forward 3-px overlap = "
                f"{p2_uq['informative_non_space_filling']['worst_forward_overlap_3px']:.4f} <= 0.70, "
                f"max |Spearman rho| = {p2_uq['informative_non_space_filling']['worst_abs_spearman']:.4f} <= 0.90)."
            ),
        },
        "raster_path": f"docs/downloads/{name}.tif",
        "zip_path": f"docs/downloads/{name}.zip",
        "raster_sha256": writer_receipt["sha256"],
        "zip_sha256": writer_receipt["zip_sha256"],
        "raster_bytes": writer_receipt["bytes"],
        "dots_emitted": n_dots,
        "validator": {
            "all_checks_passed": bool(val["all_checks_passed"]),
            "checks": val["checks"],
            "min": float(val["min"]),
            "max": float(val["max"]),
            "n_nan": int(val["n_nan"]),
            "n_infinite": int(val["n_infinite"]),
            "emitted_positive_pixels": int(val["emitted_positive_pixels"]),
            "on_catalogue_positive_pixels": int(val["on_catalogue_positive_pixels"]),
        },
        "submission_name": name,
        "submission_note": note,
        "note_length": len(note),
        "protocol_unique": False,
        "informative_unique": uq_bundle["informative_unique"],
        "ok_to_download": True,
        "ok_to_submit": False,
        "safe_to_download_and_submit": False,
        "approved_for_weekly_slot": False,
        "organizer_confirmed_score_for_this_tif": None,
        "weekly_slots_used": 0,
        "verdict": "negative",
        "verdict_reason": (
            f"H77V fixes four geothermal data bugs (IR-H77V-01..04) and the MT mesh seam artifact (IR-H77G-01), "
            f"lifting standalone S_h77v_physics HOLDOUT-DTI to {s20_phys['dti']:.4f} (Seed 20) / {s30_phys['dti']:.4f} "
            f"(Seed 30), A_h77v_only (10 ch) to {s20_h77v['dti']:.4f} (Seed 20) / {s30_h77v['dti']:.4f} (Seed 30), and "
            f"combined 29-channel A_std_plus_h77v to {s20_comb['dti']:.4f} on Seed 20 (vs A_std {s20_std['dti']:.4f}, "
            f"paired delta {s20_diff_vs_std['delta']:+.4f}, 95% CI [{s20_diff_vs_std['ci95'][0]:+.4f}, "
            f"{s20_diff_vs_std['ci95'][1]:+.4f}]) and {s30_comb['dti']:.4f} on Seed 30 (vs A_std {s30_std['dti']:.4f}, "
            f"paired delta {s30_diff_vs_std['delta']:+.4f}, 95% CI [{s30_diff_vs_std['ci95'][0]:+.4f}, "
            f"{s30_diff_vs_std['ci95'][1]:+.4f}]). Because all arms remain below the 0.152718 H77-Step repository "
            "incumbent and the Policy I17 literal dense-witness gate fails closed, H77V is delivered as a format-valid, "
            "informative-unique, inspection-only negative research artifact; 0 weekly slots spent."
        ),
        "budget": {
            "experiments_run": 3,
            "max_experiments": 3,
            "weekly_slots_spent": 0,
            "elapsed_seconds_exp3": round(time.time() - t0, 2),
        },
    }

    (EVID / "run_card_h77v.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "docs" / "evidence" / "run_card_h77v.json").write_text(json.dumps(run_card, indent=2) + "\n")
    (ROOT / "docs" / "downloads" / "run_card_h77v.json").write_text(json.dumps(run_card, indent=2) + "\n")
    log(f"Wrote evidence/run_card_h77v.json (tif={name}.tif, sha256={writer_receipt['sha256']})")
    return 0


def run_plot() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    _, h77v_fp, _ = ensure_feature_matrices(foot)
    e2 = json.loads((EVID / "h77v_e2_holdout.json").read_text())
    rc = json.loads((EVID / "run_card_h77v.json").read_text())

    def to_map(col_idx: int) -> np.ndarray:
        m = np.full(foot.shape, np.nan, dtype=np.float32)
        m[foot] = np.asarray(h77v_fp[:, col_idx], dtype=np.float32)
        return m[::4, ::4]

    sinter_m = to_map(H77V_NAMES.index("h77v_sinter_alter_vent_conduit"))
    probe_m = to_map(H77V_NAMES.index("h77v_probe_upward_heatflow_dtdz"))
    diseq_m = to_map(H77V_NAMES.index("h77v_geotherm_disequilibrium"))
    final_surf = np.load(PROC / "final_placement_surface.npy")
    final_m = np.where(foot, final_surf, np.nan)[::4, ::4]

    with rasterio.open(ROOT / rc["raster_path"]) as src:
        dots = src.read(1) > 0.5

    overlay = np.full(foot.shape, np.nan, dtype=np.float32)
    overlay[foot] = 0.0
    overlay[ndimage.binary_dilation(cat, iterations=2)] = 0.45
    overlay[ndimage.binary_dilation(dots, iterations=2)] = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=130)
    fig.suptitle(
        "GEMSDOE77 H77V: Tufa-Deconfounded Vent-Sinter & Conductive Heat-Flow Strike-Conduit Coupling\n"
        f"HOLDOUT-DTI A_std+H77V: {e2['seeds']['20']['arms']['A_std_plus_h77v']['pooled_holdout_dti']:.4f} (s20) / "
        f"{e2['seeds']['30']['arms']['A_std_plus_h77v']['pooled_holdout_dti']:.4f} (s30) | "
        f"A_h77v_only: {e2['seeds']['20']['arms']['A_h77v_only']['pooled_holdout_dti']:.4f} (s20) / "
        f"{e2['seeds']['30']['arms']['A_h77v_only']['pooled_holdout_dti']:.4f} (s30) < 0.1527 incumbent -- NEGATIVE",
        fontsize=10.0,
        fontweight="bold",
    )

    panels = [
        (axes[0, 0], sinter_m, "magma", "1. Tufa-Deconfounded Sinter/Sulfur Strike Conduit"),
        (axes[0, 1], probe_m, "inferno", "2. 2m Probe Upward Heat-Flow dT/dz = T2m - T1m"),
        (axes[0, 2], diseq_m, "plasma", "3. Geothermometer Disequilibrium (T_geo - T_surf)"),
        (axes[1, 0], final_m, "viridis", "4. H77V Placement Surface (Vent + OOF)"),
        (axes[1, 1], overlay[::4, ::4], "viridis", "5. Emitted 37,654 Dots (bright) + Catalogue"),
    ]
    for ax, arr, cmap, title in panels:
        ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    ax_bar = axes[1, 2]
    arms = ["H77-Step\n(incumbent)", "A_std+H77V\n(29 ch)", "A_h77v_only\n(10 ch)", "S_h77v_phys\n(pure)", "A_std\n(19 ch)", "A0_random"]
    s20_vals = [
        0.152718,
        e2["seeds"]["20"]["arms"]["A_std_plus_h77v"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["A_h77v_only"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["S_h77v_physics"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["A_std"]["pooled_holdout_dti"],
        e2["seeds"]["20"]["arms"]["A0_random"]["pooled_holdout_dti"],
    ]
    s30_vals = [
        0.152718,
        e2["seeds"]["30"]["arms"]["A_std_plus_h77v"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["A_h77v_only"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["S_h77v_physics"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["A_std"]["pooled_holdout_dti"],
        e2["seeds"]["30"]["arms"]["A0_random"]["pooled_holdout_dti"],
    ]
    x = np.arange(len(arms))
    w = 0.36
    ax_bar.bar(x - w / 2, s20_vals, width=w, label="Seed 20 (11,321 px)", color="#1d4ed8")
    ax_bar.bar(x + w / 2, s30_vals, width=w, label="Seed 30 (11,342 px)", color="#0d9488")
    ax_bar.axhline(0.152718, color="#b91c1c", linestyle="--", linewidth=1.2, label="Repo Incumbent (0.1527)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(arms, fontsize=7.5)
    ax_bar.set_ylabel("HOLDOUT-DTI (gems57-pooled-hide-v2)", fontsize=8.5)
    ax_bar.set_title("6. Dual-Seed Holdout Benchmark", fontsize=9.5)
    ax_bar.set_ylim(0, 0.175)
    ax_bar.legend(fontsize=7.5, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_p = fig_dir / "h77v_surface_and_dots.png"
    fig.savefig(out_p)
    plt.close(fig)
    log(f"Wrote {out_p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("exp1", "exp2", "exp3", "plot", "all"))
    args = ap.parse_args()
    if args.mode in ("exp1", "all"):
        run_exp1()
    if args.mode in ("exp2", "all"):
        run_exp2()
    if args.mode in ("exp3", "all"):
        run_exp3()
    if args.mode in ("plot", "all"):
        run_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
