#!/usr/bin/env python3
"""GEMSDOE77 Lane H77-HSL — Experiment 1:
Leakage canaries + Seed-30 whole-component quadrant hide-and-recover holdout.

Evaluator: vendor gems57.evaluate_holdout (gems57-pooled-hide-v2), alpha=0.2,
beta=0.8, 300 m triangular kernel. Every score written here is HOLDOUT-DTI
with withheld-positive counts and paired 95% CIs from pooled_summary.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems57 import metric as M  # noqa: E402
from gems77 import features77 as F0  # noqa: E402
from gems77hsl import features_hsl as FH, emit_hsl as EM  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / ".cache" / "hsl_processed"
SEED = 30
HGB_KW = dict(
    max_iter=180,
    learning_rate=0.09,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    early_stopping=False,
    random_state=77,
)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def arm_channels(name: str) -> list[str]:
    std = FH.BLOCKS["raw"] + FH.BLOCKS["terrain"] + FH.BLOCKS["rad"]
    cphys = std + FH.BLOCKS["new"]
    cphys_l5 = cphys + FH.BLOCKS["lidar"]
    hsl_l12 = cphys_l5 + FH.BLOCKS["hsl1_lidar12"]
    hsl_nolabel = hsl_l12 + FH.BLOCKS["hsl2_mag_hydro"] + FH.BLOCKS["hsl3_gdr_vents"]
    return {
        "C2_standard": std,
        "C3_contactphys": cphys,
        "C4_contactphys_lidar5": cphys_l5,
        "HSL_lidar12": hsl_l12,
        "HSL_nolabel": hsl_nolabel,
    }[name]


def extract_rows(stack_chw: np.ndarray, cols: list[int], flat_idx: np.ndarray) -> np.ndarray:
    """Memory-frugal extraction from channel-first (C, H, W) memmap."""
    H, W = stack_chw.shape[1], stack_chw.shape[2]
    ys, xs = np.divmod(flat_idx, W)
    X = np.empty((flat_idx.size, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, j] = stack_chw[c, ys, xs]
    return X


def train_predict_chw(
    stack_chw: np.ndarray,
    cols: list[int],
    train_idx: np.ndarray,
    y: np.ndarray,
    region: np.ndarray,
    logtag: str,
) -> np.ndarray:
    t = time.time()
    X = extract_rows(stack_chw, cols, train_idx)
    clf = HistGradientBoostingClassifier(**HGB_KW)
    clf.fit(X, y)
    t_fit = time.time() - t
    del X
    gc.collect()

    H, W = stack_chw.shape[1], stack_chw.shape[2]
    out = np.zeros((H, W), dtype=np.float32)
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return out
    step = 400_000
    preds = np.empty(ys.size, dtype=np.float32)
    for i in range(0, ys.size, step):
        i1 = min(ys.size, i + step)
        by, bx = ys[i:i1], xs[i:i1]
        blk = np.empty((i1 - i, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            blk[:, j] = stack_chw[c, by, bx]
        preds[i:i1] = clf.predict_proba(blk)[:, 1].astype(np.float32)
    out[ys, xs] = preds
    log(f"{logtag}: fit {t_fit:.1f}s + pred {time.time()-t-t_fit:.1f}s ({ys.size} px x {len(cols)} ch)")
    del preds, clf
    gc.collect()
    return out


def make_inner_component_split(visible: np.ndarray, seed: int, hide_frac: float = 0.25, buffer_px: int = 3):
    """Split visible fault components into (context_vis, target_vis) with a 3-px collar."""
    lbl, ncomp = ndimage.label(visible, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(lbl.ravel())
    eligible = np.flatnonzero((np.arange(ncomp + 1) > 0) & (sizes >= 4))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eligible)
    tot_px = int(visible.sum())
    target_px = int(round(hide_frac * tot_px))
    chosen = []
    acc = 0
    for cid in perm:
        chosen.append(int(cid))
        acc += int(sizes[cid])
        if acc >= target_px:
            break
    target_vis = np.isin(lbl, chosen)
    collar = ndimage.distance_transform_edt(~target_vis) <= buffer_px
    context_vis = visible & ~collar
    return context_vis, target_vis


STRUCT_PHYS_COLS = [
    "dem_ridge1", "dem_ridge2", "dem_ridge3p5", "dem_curv1", "dem_relief5",
    "frontness", "st_coherence", "mag_as_log", "grav_thdr_log", "contact_confluence",
    "base_grad_log", "pc_mag", "pc_grav", "lidar_ex_max", "lidar_step_max",
    "lidar_coh100", "lidar_valid", "lidar_crest_toe_dipole", "lidar_facing_polarity",
    "lidar_strike_coh_s2", "lidar_strike_coh_s5", "lidar_scarp_ridge_s2",
    "mag_shallow_grad", "hydro_demag_conduit", "hydro_temp_excess_s12", "hydro_hot_spring_logdist",
]


def train_predict_structural(
    stack_chw: np.ndarray,
    phys_oof: np.ndarray,
    V_test: np.ndarray,
    folds: list[dict],
    visible: np.ndarray,
    foot: np.ndarray,
    u2_lid: np.ndarray,
    v2_lid: np.ndarray,
    seed_base: int = 400,
) -> np.ndarray:
    """Train H77-HSL-4 structural continuation model using internal component splits of visible."""
    H, W = foot.shape
    cols_phys = [FH.IDX[n] for n in STRUCT_PHYS_COLS]
    rng = np.random.default_rng(seed_base + 99)

    fold_X = [[] for _ in range(len(folds))]
    fold_y = [[] for _ in range(len(folds))]

    for s_idx, s_seed in enumerate((seed_base + 1, seed_base + 2)):
        ctx_vis, tgt_vis = make_inner_component_split(visible, seed=s_seed, hide_frac=0.25, buffer_px=3)
        V_in = FH.compute_visible_fault_features(ctx_vis, foot, u2_lid, v2_lid)
        dist_ctx = ndimage.distance_transform_edt(~ctx_vis).astype(np.float32)
        log(f"inner split {s_idx+1}: target_vis={int(tgt_vis.sum())} px, context_vis={int(ctx_vis.sum())} px")

        for k, fold in enumerate(folds):
            train_region = np.zeros_like(foot)
            for j, f2 in enumerate(folds):
                if j != k:
                    train_region |= f2["region"]
            pos = np.flatnonzero(tgt_vis & train_region & (dist_ctx > 3))
            if pos.size > 25_000:
                pos = rng.choice(pos, 25_000, replace=False)
            neg_pool = train_region & foot & ~visible & (dist_ctx > 3)
            hard = np.flatnonzero(neg_pool & (dist_ctx <= 25))
            easy = np.flatnonzero(neg_pool & (dist_ctx > 25))
            n_neg = 2 * pos.size
            nh = min(hard.size, n_neg // 2)
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([
                rng.choice(hard, nh, replace=False) if nh else np.array([], int),
                rng.choice(easy, ne, replace=False) if ne else np.array([], int),
            ])
            idx = np.concatenate([pos, neg])
            y_s = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            Xp = extract_rows(stack_chw, cols_phys, idx)
            Xv = extract_rows(V_in, list(range(len(FH.HSL4_VIS_STRUCT_NAMES))), idx)
            fold_X[k].append(np.hstack([Xp, Xv]))
            fold_y[k].append(y_s)
            del Xp, Xv
        del V_in, dist_ctx, ctx_vis, tgt_vis
        gc.collect()

    out = np.zeros((H, W), dtype=np.float32)
    n_p = len(cols_phys)
    n_v = len(FH.HSL4_VIS_STRUCT_NAMES)

    for k, fold in enumerate(folds):
        X_tr = np.vstack(fold_X[k])
        y_tr = np.concatenate(fold_y[k])
        fold_X[k].clear()
        fold_y[k].clear()
        gc.collect()

        t = time.time()
        clf = HistGradientBoostingClassifier(**HGB_KW)
        clf.fit(X_tr, y_tr)
        log(f"HSL_structural/{fold['name']}: fit {time.time()-t:.1f}s ({X_tr.shape[0]} rows x {X_tr.shape[1]} ch)")
        del X_tr, y_tr
        gc.collect()

        ys, xs = np.nonzero(fold["region"] & foot)
        step = 400_000
        preds = np.empty(ys.size, dtype=np.float32)
        for i in range(0, ys.size, step):
            i1 = min(ys.size, i + step)
            by, bx = ys[i:i1], xs[i:i1]
            blk = np.empty((i1 - i, n_p + n_v), dtype=np.float32)
            for j, c in enumerate(cols_phys):
                blk[:, j] = stack_chw[c, by, bx]
            for j in range(n_v):
                blk[:, n_p + j] = V_test[j, by, bx]
            preds[i:i1] = clf.predict_proba(blk)[:, 1].astype(np.float32)
        out[ys, xs] = preds
        del preds, clf
        gc.collect()

    blended = (np.float32(0.45) * phys_oof + np.float32(0.55) * out).astype(np.float32)
    return blended


def build_champion_vis_dots(stack_chw: np.ndarray, folds: list[dict], visible: np.ndarray, foot: np.ndarray, budget: int, excl: int) -> np.ndarray:
    """Fold-fair visible-fault-only H19-5/H33 style ridge rule (Selective-U Next Step 3)."""
    # Use multi-scale terrain + magnetic + gravity ridge score, pruned ONLY by visible (>excl px)
    c_r2 = stack_chw[FH.IDX["dem_ridge2"]]
    c_r35 = stack_chw[FH.IDX["dem_ridge3p5"]]
    c_mag = stack_chw[FH.IDX["mag_as_log"]]
    c_grv = stack_chw[FH.IDX["grav_thdr_log"]]
    score = (
        np.float32(0.40) * F0.rank01(np.nan_to_num(c_r2, nan=-1e30), foot)
        + np.float32(0.25) * F0.rank01(np.nan_to_num(c_r35, nan=-1e30), foot)
        + np.float32(0.20) * F0.rank01(np.nan_to_num(c_mag, nan=-1e30), foot)
        + np.float32(0.15) * F0.rank01(np.nan_to_num(c_grv, nan=-1e30), foot)
    ).astype(np.float32)
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    dots = np.zeros(foot.shape, dtype=bool)
    tot = sum(int((f["region"] & foot & ~visible & (dist_vis > excl)).sum()) for f in folds)
    for f in folds:
        a = f["region"] & foot & ~visible & (dist_vis > excl)
        b = int(round(budget * int(a.sum()) / max(tot, 1)))
        if b > 0:
            dots |= EM.emit_dots_standard(score, a, b, min_sep=3.0)
    return dots


def main():
    t0 = time.time()
    stack_path = FH.build_hsl_stack(ROOT, force=False)
    stack_chw = np.load(stack_path, mmap_mode="r")
    grid = G.load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    H, W = foot.shape

    folds, quad = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    visible = folds[0]["visible"]
    receipts = [f["receipt"] for f in folds]
    log(f"Seed {SEED} holdout: withheld={int(hidden.sum())} visible={int(visible.sum())} cat={int(cat.sum())}")

    # Precompute lidar spin-2 director fields for H77-HSL-4
    lid_path = RAW / "external" / "lidar_scarp_features_u8.tif"
    with rasterio.open(lid_path) as ds:
        step_u8 = ds.read(3).astype(np.float32) / np.float32(255.0)
        coh_u8 = ds.read(10).astype(np.float32) / np.float32(255.0)
        strike_u8 = ds.read(11).astype(np.float32)
        lvalid = (ds.read(12) > 0).astype(np.float32)
    theta_lid = (strike_u8 / np.float32(255.0)) * np.float32(np.pi)
    w_lid = coh_u8 * step_u8 * lvalid
    u2_lid = (w_lid * np.cos(np.float32(2.0) * theta_lid)).astype(np.float32)
    v2_lid = (w_lid * np.sin(np.float32(2.0) * theta_lid)).astype(np.float32)
    del step_u8, coh_u8, strike_u8, lvalid, theta_lid, w_lid
    gc.collect()

    # 1. Leakage canary across all 77 label-free channels + 7 fold-visible channels
    rng = np.random.default_rng(1234)
    pos_idx = np.flatnonzero(hidden)
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    neg_pool = np.flatnonzero(foot & ~visible & (dist_vis > 3))
    neg_idx = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
    canary_idx = np.concatenate([pos_idx, neg_idx])
    y_can = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])

    ys_c, xs_c = np.divmod(canary_idx, W)
    canaries = {}
    for j, name in enumerate(FH.CHANNELS):
        v = stack_chw[j, ys_c, xs_c]
        m = np.isfinite(v)
        auc = float(roc_auc_score(y_can[m], v[m])) if m.sum() > 100 else None
        canaries[name] = round(auc, 5) if auc is not None else None

    # Compute V_test on visible to check canary on all 7 H77-HSL-4 channels as well
    V_test = FH.compute_visible_fault_features(visible, foot, u2_lid, v2_lid)
    for j, name in enumerate(FH.HSL4_VIS_STRUCT_NAMES):
        v = V_test[j, ys_c, xs_c]
        m = np.isfinite(v)
        auc = float(roc_auc_score(y_can[m], v[m])) if m.sum() > 100 else None
        canaries[name] = round(auc, 5) if auc is not None else None

    worst_all = max((abs(a - 0.5), n) for n, a in canaries.items() if a is not None)
    hsl_new_names = (
        FH.HSL1_LIDAR12_NAMES
        + FH.HSL2_MAG_HYDRO_NAMES
        + FH.HSL3_GDR_VENTS_NAMES
        + FH.HSL4_VIS_STRUCT_NAMES
    )
    worst_hsl = max((abs(canaries[n] - 0.5), n) for n in hsl_new_names if canaries[n] is not None)
    leakage = [n for n, a in canaries.items() if a is not None and max(a, 1.0 - a) > 0.90]
    log(
        f"Canary worst overall: {worst_all[1]} symAUC={0.5+worst_all[0]:.4f} | "
        f"worst HSL-new: {worst_hsl[1]} symAUC={0.5+worst_hsl[0]:.4f} | leakage>0.90: {leakage}"
    )
    if leakage:
        raise RuntimeError(f"STOP-LEAKAGE: channels exceeded 0.90 symmetric AUC: {leakage}")

    # 2. Shared per-fold training samples (strictly using visible & dist_vis > 3, never cat!)
    fold_samples = []
    for k, fold in enumerate(folds):
        train_region = np.zeros_like(foot)
        for j, f2 in enumerate(folds):
            if j != k:
                train_region |= f2["region"]
        pos = np.flatnonzero(visible & train_region)
        if pos.size > 40_000:
            pos = rng.choice(pos, 40_000, replace=False)
        neg_mask = train_region & foot & ~visible & (dist_vis > 3)
        hard = np.flatnonzero(neg_mask & (dist_vis <= 15))
        easy = np.flatnonzero(neg_mask & (dist_vis > 15))
        n_neg = 2 * pos.size
        nh = min(hard.size, n_neg // 2)
        ne = min(easy.size, n_neg - nh)
        neg = np.concatenate([
            rng.choice(hard, nh, replace=False) if nh else np.array([], int),
            rng.choice(easy, ne, replace=False) if ne else np.array([], int),
        ])
        idx = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
        fold_samples.append((idx, y))

    # 3. Train out-of-fold probability surfaces
    arm_list = ["C2_standard", "C3_contactphys", "C4_contactphys_lidar5", "HSL_lidar12", "HSL_nolabel"]
    oof = {}
    for arm in arm_list:
        p_cache = PROC / f"hsl_oof_s{SEED}_{arm}.npy"
        if p_cache.exists():
            oof[arm] = np.load(p_cache)
            log(f"{arm}: loaded cached OOF")
            continue
        cols = [FH.IDX[n] for n in arm_channels(arm)]
        surf = np.zeros((H, W), dtype=np.float32)
        for k, fold in enumerate(folds):
            idx, y = fold_samples[k]
            surf += train_predict_chw(stack_chw, cols, idx, y, fold["region"], f"{arm}/{fold['name']}")
        np.save(p_cache, surf)
        oof[arm] = surf

    # Train HSL_structural (HSL_nolabel + H77-HSL-4 visible-fault continuation)
    p_struct_cache = PROC / f"hsl_oof_s{SEED}_HSL_structural.npy"
    if p_struct_cache.exists():
        oof["HSL_structural"] = np.load(p_struct_cache)
        log("HSL_structural: loaded cached OOF")
    else:
        surf_st = train_predict_structural(
            stack_chw, oof["HSL_nolabel"], V_test, folds, visible, foot, u2_lid, v2_lid, seed_base=400 + SEED
        )
        np.save(p_struct_cache, surf_st)
        oof["HSL_structural"] = surf_st
    del V_test, u2_lid, v2_lid
    gc.collect()

    # 4. Evaluate arms at matched operating point (B=37654, E=2) and (B=44090, E=3)
    selB, selE = 37_654, 2

    def emit_pooled(p_surface: np.ndarray, budget: int, excl: int, submodular: bool = False) -> np.ndarray:
        allowed_folds = []
        tot = 0
        for fold in folds:
            a = fold["region"] & foot & ~visible & (dist_vis > excl)
            allowed_folds.append(a)
            tot += int(a.sum())
        dots = np.zeros(foot.shape, dtype=bool)
        for a in allowed_folds:
            b = int(round(budget * int(a.sum()) / max(tot, 1)))
            if b > 0:
                if submodular:
                    dots |= EM.emit_dots_submodular_bayes(p_surface, a, b)
                else:
                    dots |= EM.emit_dots_standard(p_surface, a, b)
        return dots

    def score_dots(dots: np.ndarray):
        terms = None
        per = {}
        for fold in folds:
            res, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            per[fold["name"]] = {"dti": res["dti"], "n_truth": res["n_truth"], "emitted": res["n_emitted"]}
            terms = t if terms is None else terms + t
        return per, terms

    terms_by_arm = {}
    dot_masks = {}
    arms_out = {}

    # Controls C0_random and C1_champion_vis
    rng2 = np.random.default_rng(999)
    rand_surf = rng2.random(foot.shape, dtype=np.float32) * foot
    dots_r = emit_pooled(rand_surf, selB, selE, submodular=False)
    per_r, terms_r = score_dots(dots_r)
    terms_by_arm["C0_random"] = terms_r
    dot_masks["C0_random"] = dots_r
    arms_out["C0_random"] = {
        "dots": int(dots_r.sum()),
        "per_fold": per_r,
        "pooled_holdout_dti": float(EH.from_terms(terms_r.sum(axis=0))),
    }
    log(f"C0_random: HOLDOUT-DTI={arms_out['C0_random']['pooled_holdout_dti']:.4f}")

    dots_c1 = build_champion_vis_dots(stack_chw, folds, visible, foot, selB, selE)
    per_c1, terms_c1 = score_dots(dots_c1)
    terms_by_arm["C1_champion_vis"] = terms_c1
    dot_masks["C1_champion_vis"] = dots_c1
    arms_out["C1_champion_vis"] = {
        "dots": int(dots_c1.sum()),
        "per_fold": per_c1,
        "pooled_holdout_dti": float(EH.from_terms(terms_c1.sum(axis=0))),
    }
    log(f"C1_champion_vis: HOLDOUT-DTI={arms_out['C1_champion_vis']['pooled_holdout_dti']:.4f}")

    for arm in ["C2_standard", "C3_contactphys", "C4_contactphys_lidar5", "HSL_lidar12", "HSL_nolabel", "HSL_structural"]:
        dots = emit_pooled(oof[arm], selB, selE, submodular=False)
        per, terms = score_dots(dots)
        terms_by_arm[arm] = terms
        dot_masks[arm] = dots
        arms_out[arm] = {
            "dots": int(dots.sum()),
            "per_fold": per,
            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
        }
        log(f"{arm} (standard emit): HOLDOUT-DTI={arms_out[arm]['pooled_holdout_dti']:.4f}")

    # Test Bayes-calibrated submodular 300 m kernel coverage emission on HSL_lidar12, HSL_nolabel, and HSL_structural
    for base_arm in ("HSL_lidar12", "HSL_nolabel", "HSL_structural"):
        sub_name = f"{base_arm}_submod"
        dots = emit_pooled(oof[base_arm], selB, selE, submodular=True)
        per, terms = score_dots(dots)
        terms_by_arm[sub_name] = terms
        dot_masks[sub_name] = dots
        arms_out[sub_name] = {
            "dots": int(dots.sum()),
            "per_fold": per,
            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
        }
        log(f"{sub_name} (Bayes submodular emit): HOLDOUT-DTI={arms_out[sub_name]['pooled_holdout_dti']:.4f}")

    # Also evaluate single-feature physics arms for our new hypotheses
    for fname in ("lidar_crest_toe_dipole", "lidar_facing_polarity", "lidar_strike_coh_s5", "mag_shallow_grad", "hydro_demag_conduit", "hydro_temp_excess_s12"):
        ch = stack_chw[FH.IDX[fname]]
        p = F0.rank01(np.nan_to_num(ch, nan=-1e30), foot)
        p = np.where(foot, p, 0).astype(np.float32)
        dots = emit_pooled(p, selB, selE, submodular=False)
        per, terms = score_dots(dots)
        sname = f"S_{fname}"
        terms_by_arm[sname] = terms
        arms_out[sname] = {
            "dots": int(dots.sum()),
            "per_fold": per,
            "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0))),
        }
        log(f"{sname}: HOLDOUT-DTI={arms_out[sname]['pooled_holdout_dti']:.4f}")

    # Pick best HSL candidate arm and compute paired bootstrap summary against all controls
    best_cand = max(
        ("HSL_nolabel", "HSL_nolabel_submod", "HSL_structural", "HSL_structural_submod"),
        key=lambda a: arms_out[a]["pooled_holdout_dti"],
    )
    summary_best = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate=best_cand)
    summary_nolabel = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="HSL_nolabel_submod")

    # SGMC off-catalogue proxy (secondary diagnostic)
    with rasterio.open(RAW / "external" / "derived_sgmc_faults_100m_u8.tif") as s:
        sgmc = s.read(1) > 0
    sgmc_off = sgmc & ~cat & (dist_cat > 3) & foot
    sgmc_proxy = {
        "n_truth_px": int(sgmc_off.sum()),
        "note": "PROXY-DTI vs SGMC\\catalogue (d>3px); secondary off-catalogue structural lineament proxy, not HOLDOUT-DTI",
    }
    for arm, dots in dot_masks.items():
        r = M.dti_binary(dots, sgmc_off, valid=foot, known=cat)
        sgmc_proxy[arm] = {"dti": round(float(r["dti"]), 5), "coverage": round(float(r["coverage"]), 5)}

    results = {
        "evidence_class": "HOLDOUT-DTI",
        "lane": "H77-HSL",
        "preregistration_id": "GEMSDOE77-PREREG-HSL-1",
        "evaluator_version": EH.VERSION,
        "alpha": M.ALPHA,
        "beta": M.BETA,
        "kernel_radius_m": M.R_M,
        "holdout_seed": SEED,
        "withheld_positive_pixels_total_draw": int(hidden.sum()),
        "operating_point": {"budget": selB, "exclusion_px": selE},
        "fold_receipts": receipts,
        "canary_auc": canaries,
        "canary_worst_overall_symmetric_auc": round(0.5 + float(worst_all[0]), 5),
        "canary_worst_hsl_new_symmetric_auc": round(0.5 + float(worst_hsl[0]), 5),
        "leakage_channels_gt_0p90_symmetric": leakage,
        "arms": arms_out,
        "best_candidate": best_cand,
        "pooled_summary_best_candidate": summary_best,
        "pooled_summary_hsl_nolabel_submod": summary_nolabel,
        "sgmc_offcatalogue_proxy": sgmc_proxy,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (ROOT / "evidence").mkdir(exist_ok=True)
    out_json = ROOT / "evidence" / "hsl_holdout_exp1.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    log(f"EXP 1 DONE in {results['elapsed_s']}s -> {out_json}")
    log(f"Best candidate: {best_cand} = {arms_out[best_cand]['pooled_holdout_dti']:.4f}")
    for comp, d in summary_best["paired_differences"].items():
        if comp in ("C0_random", "C1_champion_vis", "C2_standard", "C3_contactphys", "C4_contactphys_lidar5", "HSL_nolabel", "HSL_nolabel_submod"):
            log(f"  Paired delta ({best_cand} - {comp}): {d['delta']:+.4f} CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
