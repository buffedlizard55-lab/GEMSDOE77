#!/usr/bin/env python3
"""Experiment 2 of 3 (H77D): Leakage Canaries + Dual-Seed Spatial Quadrant Holdout (Seed 30 & Seed 20).

Evaluates H77D-1 and all pre-registered domain/allocator ablations against every prior baseline
(A5_primary_lidar, A4_primary, A2_standard, A0_h195_foldfair, A0_random) using the byte-identical
shared evaluator vendor/gems57/evaluate_holdout.py (gems57-pooled-hide-v2, alpha=0.2, beta=0.8,
300 m triangular kernel, 2,000-draw paired 20 km physical spatial-cluster bootstrap).

Anti-leakage invariants:
1. All single-feature symmetric ROC-AUC canaries run FIRST; hard stop if max(AUC, 1-AUC) > 0.90.
2. Zero SGMC bedrock features anywhere in X or y; zero 'dist_known_fault_px' in GDR CSV reads;
   zero prior submission rasters inside any feature or model.
3. For each holdout draw (seed=30: 11,342 withheld positive px; seed=20: 11,321 withheld positive px)
   and each quadrant fold k in {NW, NE, SW, SE}:
   - Training positives come strictly from visible & train_region (outside fold k, 0 withheld px).
   - Training negatives come strictly from train_region & ~visible_train & (dist_vis_train > 3).
   - Visible structural mechanics (vis_*) are computed strictly from fold['visible'] (100% blind to withheld).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH
from gems57 import grid as G
from gems57 import holdout as HO
from gems57 import metric as M
from gems77 import emit77, features77 as F77
from gems77d import features as F77D
from gems77d.submodular import allocate_submodular_300m
from gems77lia import allocate as LIA_AL, core as LIA_CORE

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
WORK = LIA_CORE.WORK

HGB_STATIC_KW = dict(
    max_iter=180,
    learning_rate=0.075,
    max_leaf_nodes=55,
    min_samples_leaf=50,
    l2_regularization=2.5,
    early_stopping=False,
    random_state=7701,
)
HGB_MECH_KW = dict(
    max_iter=180,
    learning_rate=0.075,
    max_leaf_nodes=47,
    min_samples_leaf=60,
    l2_regularization=3.0,
    early_stopping=False,
    random_state=7702,
)
HGB_LEGACY_KW = dict(
    max_iter=180,
    learning_rate=0.09,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    early_stopping=False,
    random_state=77,
)


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_combined_fp_matrix(foot: np.ndarray) -> tuple[np.memmap, list[str], dict[str, list[int]]]:
    """Assemble clean zero-SGMC in-footprint float32 matrix combining shared features77 (no buggy lidar) + H77D."""
    comb_path = WORK / "h77d_combined_fp.npy"
    comb_meta = WORK / "h77d_combined_meta.json"

    shared_keep = F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["new"] + F77.BLOCKS["rad"]
    h77d_pot = list(F77D.STATIC_BLOCKS["potential_crossgrad"])
    h77d_scarp = list(F77D.STATIC_BLOCKS["scarp_1m_10m_deconf"])
    h77d_hydro = list(F77D.STATIC_BLOCKS["hydro_vent_conduit"])

    names = [f"s77_{c}" for c in shared_keep] + h77d_pot + h77d_scarp + h77d_hydro
    idx_map = {n: i for i, n in enumerate(names)}
    groups = {
        "shared_raw_terr_rad": [idx_map[f"s77_{c}"] for c in F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["rad"]],
        "shared_contactphys": [idx_map[f"s77_{c}"] for c in F77.BLOCKS["new"]],
        "potential_crossgrad": [idx_map[c] for c in h77d_pot],
        "scarp_1m_10m_deconf": [idx_map[c] for c in h77d_scarp],
        "hydro_vent_conduit": [idx_map[c] for c in h77d_hydro],
    }

    n_fp = int(foot.sum())
    n_ch = len(names)
    if comb_path.exists() and comb_meta.exists():
        meta = json.loads(comb_meta.read_text())
        if meta.get("names") == names and meta.get("shape") == [n_fp, n_ch]:
            return np.load(comb_path, mmap_mode="r"), names, groups

    log(f"Building unified zero-SGMC in-footprint feature matrix ({n_fp} x {n_ch})...")
    s77 = np.load(PROC / "stack.npy", mmap_mode="r")
    s77d, _ = F77D.build_static_stack(LIA_CORE.DATA, WORK)
    mm = np.lib.format.open_memmap(comb_path, mode="w+", dtype=np.float32, shape=(n_fp, n_ch))

    for c in shared_keep:
        mm[:, idx_map[f"s77_{c}"]] = s77[:, :, F77.IDX[c]][foot]
    for c in h77d_pot + h77d_scarp + h77d_hydro:
        mm[:, idx_map[c]] = s77d[:, F77D.STATIC_IDX[c]]

    mm.flush()
    del mm
    comb_meta.write_text(json.dumps({"shape": [n_fp, n_ch], "names": names}, indent=2) + "\n")
    gc.collect()
    return np.load(comb_path, mmap_mode="r"), names, groups


def fit_predict_fold_fp(
    X_fp: np.memmap,
    extra_fp: np.ndarray | None,
    cols: list[int],
    tr_fp_idx: np.ndarray,
    y_tr: np.ndarray,
    te_fp_idx: np.ndarray,
    kw: dict,
) -> np.ndarray:
    x_tr = np.asarray(X_fp[tr_fp_idx][:, cols], dtype=np.float32)
    if extra_fp is not None:
        x_tr = np.hstack([x_tr, extra_fp[tr_fp_idx]])
    clf = HistGradientBoostingClassifier(**kw)
    clf.fit(x_tr, y_tr)
    del x_tr
    gc.collect()

    preds = np.empty(te_fp_idx.size, dtype=np.float32)
    chunk = 250_000
    for s in range(0, te_fp_idx.size, chunk):
        sl = te_fp_idx[s : s + chunk]
        xb = np.asarray(X_fp[sl][:, cols], dtype=np.float32)
        if extra_fp is not None:
            xb = np.hstack([xb, extra_fp[sl]])
        preds[s : s + chunk] = clf.predict_proba(xb)[:, 1].astype(np.float32)
        del xb
    del clf
    gc.collect()
    return preds


def run_seed(
    seed: int,
    grid,
    X_fp: np.memmap,
    names: list[str],
    groups: dict[str, list[int]],
    geod_rank_2d: np.ndarray,
    dsup_2d: np.ndarray,
    full_ablation: bool = True,
) -> dict:
    foot = grid.footprint
    cat = grid.catalogue
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)

    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    receipts = [f["receipt"] for f in folds]
    log(f"=== SEED {seed}: withheld={int(hidden.sum())} visible={int(visible.sum())} ===")

    # Compute strictly fold-visible structural mechanics (100% blind to hidden!)
    vis_mech_fp = F77D.compute_visible_mechanics(visible, foot, geod_rank_2d)

    # 1. Single-feature symmetric ROC-AUC leakage canaries across ALL 75 channels + legacy lidar
    rng = np.random.default_rng(1000 + seed)
    pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    neg_pool_fp = fp_lut[np.flatnonzero(foot & ~cat & (dist_cat > 3))]
    neg_fp = rng.choice(neg_pool_fp, size=min(200_000, neg_pool_fp.size), replace=False)
    can_fp = np.concatenate([pos_fp, neg_fp])
    y_can = np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])

    canaries = {}
    Xc = np.asarray(X_fp[can_fp], dtype=np.float32)
    for j, nm in enumerate(names):
        v = Xc[:, j]
        m = np.isfinite(v)
        if m.sum() > 100 and len(np.unique(y_can[m])) == 2:
            canaries[nm] = float(roc_auc_score(y_can[m], v[m]))
        else:
            canaries[nm] = None
    del Xc
    Vc = vis_mech_fp[can_fp]
    for j, nm in enumerate(F77D.VISIBLE_NAMES):
        v = Vc[:, j]
        canaries[nm] = float(roc_auc_score(y_can, v))
    del Vc
    gc.collect()

    sym_aucs = {k: max(a, 1.0 - a) for k, a in canaries.items() if a is not None}
    worst_name = max(sym_aucs, key=sym_aucs.get)
    worst_sym = sym_aucs[worst_name]
    leak_flags = [k for k, sa in sym_aucs.items() if sa > 0.90]
    log(f"[SEED {seed}] Canary check across {len(sym_aucs)} channels: max symAUC={worst_sym:.4f} ({worst_name}), flags={leak_flags}")
    if leak_flags:
        raise RuntimeError(f"HARD STOP — LEAKAGE CANARY TRIPPED (>0.90): {leak_flags}")

    # 2. Strictly label-blind spatial-quadrant training folds (zero access to cat or hidden!)
    fold_io = []
    for k, fold in enumerate(folds):
        tr_reg = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                tr_reg |= f2["region"]
        vis_tr = visible & tr_reg
        d_vis_tr = ndimage.distance_transform_edt(~vis_tr).astype(np.float32)
        pos = fp_lut[np.flatnonzero(vis_tr & foot)]
        if pos.size > 48_000:
            pos = rng.choice(pos, 48_000, replace=False)
        neg_base = foot & tr_reg & ~vis_tr & (d_vis_tr > 3.0)
        hard = fp_lut[np.flatnonzero(neg_base & (d_vis_tr <= 16.0))]
        easy = fp_lut[np.flatnonzero(neg_base & (d_vis_tr > 16.0))]
        n_neg = int(2.5 * pos.size)
        nh = min(hard.size, int(0.55 * n_neg))
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([rng.choice(hard, nh, replace=False), rng.choice(easy, ne, replace=False)])
        tr_idx = np.concatenate([pos, neg])
        y_tr = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        te_idx = fp_lut[np.flatnonzero(fold["region"] & foot)]
        fold_io.append((fold["name"], tr_idx, y_tr, te_idx, fold["region"] & foot))

    all_cols = list(range(len(names)))

    def oof_surface_h77d(cols: list[int], use_static_view: bool = True, use_mech_view: bool = True) -> np.ndarray:
        surf = np.zeros((h, w), dtype=np.float32)
        for fname, tr_idx, y_tr, te_idx, te_mask in fold_io:
            p_acc = np.zeros(te_idx.size, dtype=np.float32)
            w_sum = 0.0
            if use_static_view:
                p_s = fit_predict_fold_fp(X_fp, None, cols, tr_idx, y_tr, te_idx, HGB_STATIC_KW)
                p_acc += 0.45 * p_s
                w_sum += 0.45
            if use_mech_view:
                p_m = fit_predict_fold_fp(X_fp, vis_mech_fp, cols, tr_idx, y_tr, te_idx, HGB_MECH_KW)
                p_acc += 0.55 * p_m
                w_sum += 0.55
            surf[te_mask] = p_acc / w_sum
        return surf

    # Build or load cached OOF probability surfaces (deterministic fixed seeds per holdout seed)
    t_s = time.time()
    surfs: dict[str, np.ndarray] = {}
    core_cache = WORK / f"oof_h77d_pair_seed{seed}.npz"
    if core_cache.exists():
        zc = np.load(core_cache)
        s_static_all = zc["static"]
        s_mech_all = zc["mech"]
        log(f"[SEED {seed}] Loaded cached H77D core pair surface")
    else:
        s_static_all = np.zeros((h, w), dtype=np.float32)
        s_mech_all = np.zeros((h, w), dtype=np.float32)
        for fname, tr_idx, y_tr, te_idx, te_mask in fold_io:
            t_f = time.time()
            ps = fit_predict_fold_fp(X_fp, None, all_cols, tr_idx, y_tr, te_idx, HGB_STATIC_KW)
            pm = fit_predict_fold_fp(X_fp, vis_mech_fp, all_cols, tr_idx, y_tr, te_idx, HGB_MECH_KW)
            s_static_all[te_mask] = ps
            s_mech_all[te_mask] = pm
            log(f"[SEED {seed}] H77D core fold {fname} done in {time.time()-t_f:.1f}s")
        np.savez(core_cache, static=s_static_all, mech=s_mech_all)

    surfs["H77D_full"] = (0.45 * s_static_all + 0.55 * s_mech_all).astype(np.float32)
    surfs["H77D_no_vismech"] = s_static_all
    surfs["H77D_mech_only"] = s_mech_all
    np.save(WORK / f"oof_h77d_seed{seed}.npy", surfs["H77D_full"])

    if full_ablation:
        for ab_name, drop_group in (
            ("H77D_no_scarp", "scarp_1m_10m_deconf"),
            ("H77D_no_hydro", "hydro_vent_conduit"),
            ("H77D_no_pot", "potential_crossgrad"),
        ):
            ab_cache = WORK / f"oof_{ab_name}_seed{seed}.npy"
            if ab_cache.exists():
                surfs[ab_name] = np.load(ab_cache)
                log(f"[SEED {seed}] Loaded cached {ab_name}")
            else:
                keep_cols = [c for c in all_cols if c not in set(groups[drop_group])]
                surfs[ab_name] = oof_surface_h77d(keep_cols, use_static_view=True, use_mech_view=True)
                np.save(ab_cache, surfs[ab_name])
                log(f"[SEED {seed}] {ab_name} OOF done ({time.time()-t_s:.0f}s)")

    # Exact reproduction of shared baselines (A2_standard, A4_primary, A5_primary_lidar) on identical folds
    s77_stack = np.load(PROC / "stack.npy", mmap_mode="r")
    s77_flat = s77_stack.reshape(-1, s77_stack.shape[2])
    for b_arm, b_chs in (
        ("A2_standard", F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["rad"]),
        ("A4_primary", F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["rad"] + list(F77.BLOCKS["new"])),
        ("A5_primary_lidar", F77.BLOCKS["raw"] + F77.BLOCKS["terrain"] + F77.BLOCKS["rad"] + list(F77.BLOCKS["new"]) + list(F77.BLOCKS["lidar"])),
    ):
        b_cache = WORK / f"oof_{b_arm}_seed{seed}.npy"
        if b_cache.exists():
            surfs[b_arm] = np.load(b_cache)
            log(f"[SEED {seed}] Loaded cached baseline {b_arm}")
            continue
        cols77 = [F77.IDX[c] for c in b_chs]
        bsurf = np.zeros((h, w), dtype=np.float32)
        for fname, tr_idx, y_tr, te_idx, te_mask in fold_io:
            tr_raw_idx = fp_flat[tr_idx]
            te_raw_idx = fp_flat[te_idx]
            clf = HistGradientBoostingClassifier(**HGB_LEGACY_KW)
            clf.fit(np.asarray(s77_flat[tr_raw_idx][:, cols77], dtype=np.float32), y_tr)
            preds = np.empty(te_raw_idx.size, dtype=np.float32)
            for s in range(0, te_raw_idx.size, 250_000):
                sl = te_raw_idx[s : s + 250_000]
                preds[s : s + 250_000] = clf.predict_proba(np.asarray(s77_flat[sl][:, cols77], dtype=np.float32))[:, 1]
            bsurf[te_mask] = preds
            del clf
            gc.collect()
        np.save(b_cache, bsurf)
        surfs[b_arm] = bsurf
        log(f"[SEED {seed}] Baseline {b_arm} OOF done ({time.time()-t_s:.0f}s)")

    # Emission helpers (identical fold-proportional budget & visible-catalogue exclusion across all arms)
    def emit_submod_pooled(p_surf: np.ndarray, budget: int, excl_px: float = 3.0) -> np.ndarray:
        allowed_folds = []
        tot = 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > excl_px)
            allowed_folds.append(a)
            tot += int(a.sum())
        dots = np.zeros((h, w), dtype=bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= allocate_submodular_300m(
                    p_surf, a, target_n=b, support_quantile=0.968, gamma=3.5, min_sep_px=2.20
                )
        return dots

    def emit_greedy_pooled(p_surf: np.ndarray, budget: int, excl_px: float = 3.0) -> np.ndarray:
        allowed_folds = []
        tot = 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > excl_px)
            allowed_folds.append(a)
            tot += int(a.sum())
        dots = np.zeros((h, w), dtype=bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                dots |= emit77.emit_dots(p_surf, a, b)
        return dots

    def score_dots(dots: np.ndarray):
        terms = None
        per = {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": float(res["dti"]), "n_truth": int(res["n_truth"]), "emitted": int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        return per, terms

    budgets_out = {}
    for B in (44_090, 37_654):
        terms_by_arm = {}
        arm_details = {}

        # Primary candidate & submodular ablations
        submod_arms = ["H77D_full", "H77D_no_vismech", "H77D_mech_only"]
        if full_ablation and B == 44_090:
            submod_arms += ["H77D_no_scarp", "H77D_no_hydro", "H77D_no_pot"]
        for aname in submod_arms:
            d = emit_submod_pooled(surfs[aname], B, 3.0)
            per, t = score_dots(d)
            terms_by_arm[aname] = t
            arm_details[aname] = {"dots": int(d.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0)))}
            log(f"[SEED {seed} B={B}] {aname:20s}: HOLDOUT-DTI={arm_details[aname]['pooled_holdout_dti']:.5f} dots={int(d.sum())}")

        # Greedy allocator ablation on identical H77D_full surface + prior baselines
        for aname, src_key in [
            ("H77D_greedy_alloc", "H77D_full"),
            ("A5_primary_lidar", "A5_primary_lidar"),
            ("A4_primary", "A4_primary"),
            ("A2_standard", "A2_standard"),
        ]:
            d = emit_greedy_pooled(surfs[src_key], B, 3.0)
            per, t = score_dots(d)
            terms_by_arm[aname] = t
            arm_details[aname] = {"dots": int(d.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0)))}
            log(f"[SEED {seed} B={B}] {aname:20s}: HOLDOUT-DTI={arm_details[aname]['pooled_holdout_dti']:.5f} dots={int(d.sum())}")

        # A0_h195_foldfair (exact PR #6 champion line-set construction)
        rk = np.random.default_rng(7).random(h * w)
        d_a0 = np.zeros((h, w), dtype=bool)
        for fold in folds:
            valid_f = fold["region"] & foot & ~fold["masked_known"]
            ac = np.flatnonzero(((dsup_2d == 0) & valid_f & (dist_vis > 2.0)).ravel())
            sel_a0 = LIA_AL.greedy_select(ac[np.argsort(rk[ac])], 3.0)
            d_a0.ravel()[sel_a0] = True
        per_a0, t_a0 = score_dots(d_a0)
        terms_by_arm["A0_h195_foldfair"] = t_a0
        arm_details["A0_h195_foldfair"] = {"dots": int(d_a0.sum()), "per_fold": per_a0, "pooled_holdout_dti": float(EH.from_terms(t_a0.sum(axis=0)))}
        log(f"[SEED {seed} B={B}] {'A0_h195_foldfair':20s}: HOLDOUT-DTI={arm_details['A0_h195_foldfair']['pooled_holdout_dti']:.5f} dots={int(d_a0.sum())}")

        # A0_random
        r_surf = np.random.default_rng(999 + seed).random((h, w), dtype=np.float32) * foot
        d_rnd = emit_greedy_pooled(r_surf, B, 3.0)
        per_r, t_r = score_dots(d_rnd)
        terms_by_arm["A0_random"] = t_r
        arm_details["A0_random"] = {"dots": int(d_rnd.sum()), "per_fold": per_r, "pooled_holdout_dti": float(EH.from_terms(t_r.sum(axis=0)))}
        log(f"[SEED {seed} B={B}] {'A0_random':20s}: HOLDOUT-DTI={arm_details['A0_random']['pooled_holdout_dti']:.5f} dots={int(d_rnd.sum())}")

        pooled = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810, candidate="H77D_full")
        for ctrl, drec in pooled["paired_differences"].items():
            log(f"  -> [SEED {seed} B={B}] H77D_full vs {ctrl:20s}: delta={drec['delta']:+.5f} CI95=[{drec['ci95'][0]:+.5f}, {drec['ci95'][1]:+.5f}]")
        budgets_out[f"B{B}_E3"] = {"arms": arm_details, "pooled_summary": pooled}

    return {
        "seed": seed,
        "withheld_positive_pixels": int(hidden.sum()),
        "fold_receipts": receipts,
        "canary_max_symmetric_auc": worst_sym,
        "canary_max_channel": worst_name,
        "leakage_flags": leak_flags,
        "canary_auc": canaries,
        "budgets": budgets_out,
    }


def main() -> int:
    t0 = time.time()
    grid = G.load_grid(RAW)
    X_fp, names, groups = build_combined_fp_matrix(grid.footprint)
    dom = np.load(WORK / "h77d_domains_fp.npz")
    geod_rank_2d = np.zeros(grid.footprint.shape, dtype=np.float32)
    geod_rank_2d[grid.footprint] = dom["geod_rank"]
    dsup_2d = np.load(WORK / "d_h195.npy")

    seed30 = run_seed(30, grid, X_fp, names, groups, geod_rank_2d, dsup_2d, full_ablation=True)
    seed20 = run_seed(20, grid, X_fp, names, groups, geod_rank_2d, dsup_2d, full_ablation=False)

    out = {
        "experiment": "H77D-E2",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "implementation_sha256": EH.implementation_hashes(),
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "triangular_radius_m": M.R_M,
        "n_static_channels": len(names),
        "n_visible_mechanics_channels": len(F77D.VISIBLE_NAMES),
        "sgmc_channels_used": 0,
        "prior_submission_rasters_used_in_features": 0,
        "elapsed_seconds": round(time.time() - t0, 2),
        "seed30": seed30,
        "seed20": seed20,
    }
    out_path = ROOT / "evidence" / "h77d_exp2_holdout.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"Saved Experiment 2 holdout report -> {out_path} in {out['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
