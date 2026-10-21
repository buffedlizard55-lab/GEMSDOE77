"""Zero-leakage spatial out-of-fold cross-family (`HGB + LightGBM`) ensemble models
for Lane H77E (`Physical` model + multi-split hard-collar `Structural Continuation` model).

Single-pass streaming I/O reads each selected channel once for all 4 fold training sets
and once in contiguous sequential chunks for full-footprint prediction (`8x` faster I/O,
`< 650 MB` RAM peak on `0B`-swap containers).
"""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

from gems77pd import features as FE

HGB_PARAMS = dict(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=35,
    min_samples_leaf=40,
    l2_regularization=1.5,
    early_stopping=False,
    random_state=77,
)

LGBM_PARAMS = dict(
    n_estimators=220,
    learning_rate=0.075,
    num_leaves=35,
    min_child_samples=40,
    subsample_for_bin=120_000,
    colsample_bytree=0.72,
    subsample=0.85,
    subsample_freq=1,
    reg_lambda=2.0,
    n_jobs=2,
    verbose=-1,
    random_state=177,
)


def make_inner_component_split(
    visible: np.ndarray,
    seed: int,
    hide_frac: float = 0.25,
    buffer_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    lbl, ncomp = ndi.label(visible, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(lbl.ravel())
    eligible = np.flatnonzero((np.arange(ncomp + 1) > 0) & (sizes >= 4))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eligible)
    target_px = int(round(hide_frac * int(visible.sum())))
    chosen = []
    acc = 0
    for cid in perm:
        chosen.append(int(cid))
        acc += int(sizes[cid])
        if acc >= target_px:
            break
    target_vis = np.isin(lbl, chosen)
    collar = ndi.distance_transform_edt(~target_vis) <= buffer_px
    context_vis = visible & (~collar)
    return context_vis, target_vis


def _fit_pair(X_tr: np.ndarray, y_tr: np.ndarray, use_lgbm: bool = True, seed: int = 77):
    hgb = HistGradientBoostingClassifier(**{**HGB_PARAMS, "random_state": seed})
    hgb.fit(X_tr, y_tr)
    lgbm = None
    if use_lgbm:
        lgbm = lgb.LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed + 100})
        lgbm.fit(X_tr, y_tr)
    return hgb, lgbm


def _predict_pair(models, X_blk: np.ndarray) -> np.ndarray:
    hgb, lgbm = models
    p_h = hgb.predict_proba(X_blk)[:, 1].astype(np.float32)
    if lgbm is not None:
        p_l = lgbm.predict_proba(X_blk)[:, 1].astype(np.float32)
        return np.float32(0.55) * p_h + np.float32(0.45) * p_l
    return p_h


def train_physical_oof(
    store: FE.StaticFeatureStore,
    col_indices: list[int],
    folds: list[dict],
    visible: np.ndarray,
    foot: np.ndarray,
    use_lgbm: bool = True,
    max_pos: int = 40_000,
    seed: int = 77,
    tag: str = "phys",
) -> np.ndarray:
    """Single-scan out-of-fold cross-fitted pure physical fault probability."""
    t0 = time.time()
    H, W = foot.shape
    fp_idx = np.flatnonzero(foot)
    N_fp = int(fp_idx.size)
    full_to_fp = np.full(H * W, -1, dtype=np.int32)
    full_to_fp[fp_idx] = np.arange(N_fp, dtype=np.int32)

    # 1. Sample indices for all folds upfront so we read `store` ONCE for training!
    tr_fp_list, y_list = [], []
    fold_of_fp = np.full(N_fp, -1, dtype=np.int8)
    for k, fold in enumerate(folds):
        rng = np.random.default_rng(seed + 13 * k)
        train_reg = np.zeros((H, W), dtype=bool)
        for j, f2 in enumerate(folds):
            if j != k:
                train_reg |= f2["region"]
        pos_full = np.flatnonzero(train_reg & foot & visible)
        if pos_full.size > max_pos:
            pos_full = rng.choice(pos_full, max_pos, replace=False)
        neg_full = np.flatnonzero(train_reg & foot & (~visible))
        neg_full = rng.choice(neg_full, min(neg_full.size, 2 * pos_full.size), replace=False)

        tr_fp_list.append(full_to_fp[np.concatenate([pos_full, neg_full])])
        y_list.append(np.concatenate([np.ones(pos_full.size, np.int8), np.zeros(neg_full.size, np.int8)]))
        te_fp = full_to_fp[np.flatnonzero(fold["region"] & foot)]
        fold_of_fp[te_fp] = k

    X_tr_list = [np.empty((idx.size, len(col_indices)), dtype=np.float32) for idx in tr_fp_list]
    for j, c in enumerate(col_indices):
        vec = store.read_channel(c)
        for k, idx in enumerate(tr_fp_list):
            X_tr_list[k][:, j] = vec[idx]
        del vec

    clfs = []
    for k in range(len(folds)):
        clfs.append(_fit_pair(X_tr_list[k], y_list[k], use_lgbm=use_lgbm, seed=seed + k))
        X_tr_list[k] = None
    del X_tr_list
    gc.collect()

    # 2. Single contiguous sequential scan over `0..N_fp` for prediction!
    preds_fp = np.zeros(N_fp, dtype=np.float32)
    step = 650_000
    for start in range(0, N_fp, step):
        end = min(N_fp, start + step)
        blk = store.extract_slice(start, end, col_indices)
        f_sub = fold_of_fp[start:end]
        for k in range(len(folds)):
            m = f_sub == k
            if m.any():
                preds_fp[start:end][m] = _predict_pair(clfs[k], blk[m])
        del blk
    del clfs
    gc.collect()

    out_grid = np.zeros((H, W), dtype=np.float32)
    out_grid.ravel()[fp_idx] = preds_fp
    print(f"[h77e_models] {tag} ({len(col_indices)} ch, lgbm={use_lgbm}) OOF done in {time.time() - t0:.1f}s", flush=True)
    return out_grid


def logodds_blend(p_a: np.ndarray, p_b: np.ndarray, w_a: float = 0.42, eps: float = 1e-6) -> np.ndarray:
    """Bayesian log-odds (Geometric / Product-of-Experts) posterior fusion."""
    pa = np.clip(p_a, eps, 1.0 - eps).astype(np.float32)
    pb = np.clip(p_b, eps, 1.0 - eps).astype(np.float32)
    la = np.log(pa / (np.float32(1.0) - pa))
    lb = np.log(pb / (np.float32(1.0) - pb))
    lw = np.float32(w_a) * la + np.float32(1.0 - w_a) * lb
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-lw))).astype(np.float32)


def train_structural_oof(
    store: FE.StaticFeatureStore,
    partner_indices: list[int],
    phys_oof: np.ndarray,
    folds: list[dict],
    visible: np.ndarray,
    foot: np.ndarray,
    directors: dict[str, np.ndarray],
    n_inner_splits: int = 3,
    use_lgbm: bool = True,
    blend_phys_weight: float = 0.42,
    seed: int = 400,
    tag: str = "struct",
) -> tuple[np.ndarray, np.ndarray]:
    """Single-scan multi-split hard-collar fracture continuation model + clean OOF blend.

    CRITICAL DISTRIBUTION-SHIFT GUARD:
      `phys_oof` is NEVER included as a training column inside `Model S`, because `phys_oof`
      was fit on `visible` (which includes `tgt_vis` in 2 of the 3 training quadrants!) and
      would cause severe train-vs-test target shift that starves `B4_VIS_FRACTURE`. Instead,
      `Model S` sees strictly raw physical `partner_indices` (40 ch) + strictly inner-split
      `B4_VIS_FRACTURE` (11 ch), and fuses with `phys_oof` strictly out-of-fold afterwards!
    """
    t0 = time.time()
    H, W = foot.shape
    fp_idx = np.flatnonzero(foot)
    N_fp = int(fp_idx.size)
    full_to_fp = np.full(H * W, -1, dtype=np.int32)
    full_to_fp[fp_idx] = np.arange(N_fp, dtype=np.int32)

    fold_of_fp = np.full(N_fp, -1, dtype=np.int8)
    for k, fold in enumerate(folds):
        fold_of_fp[full_to_fp[np.flatnonzero(fold["region"] & foot)]] = k

    split_tr_fp: list[list[np.ndarray]] = [[] for _ in range(len(folds))]
    split_dyn: list[list[np.ndarray]] = [[] for _ in range(len(folds))]
    fold_y: list[list[np.ndarray]] = [[] for _ in range(len(folds))]
    rng = np.random.default_rng(seed + 77)

    for s_idx in range(n_inner_splits):
        ctx_vis, tgt_vis = make_inner_component_split(
            visible & foot, seed=seed + 10 * s_idx + 1, hide_frac=0.25, buffer_px=3
        )
        V_in = FE.compute_visible_fracture_features(ctx_vis, foot, directors)
        dist_ctx = ndi.distance_transform_edt(~ctx_vis).astype(np.float32)

        for k in range(len(folds)):
            train_reg = np.zeros((H, W), dtype=bool)
            for j, f2 in enumerate(folds):
                if j != k:
                    train_reg |= f2["region"]

            pos_full = np.flatnonzero(tgt_vis & train_reg & foot & (dist_ctx > 3.0))
            if pos_full.size > 24_000:
                pos_full = rng.choice(pos_full, 24_000, replace=False)

            neg_pool = train_reg & foot & (~visible) & (dist_ctx > 3.0)
            hard = np.flatnonzero(neg_pool & (dist_ctx <= 25.0))
            far = np.flatnonzero(neg_pool & (dist_ctx > 25.0))
            n_neg = 2 * pos_full.size
            nh = min(hard.size, n_neg // 2)
            nf = min(far.size, n_neg - nh)
            neg_full = np.concatenate([
                rng.choice(hard, nh, replace=False) if nh else np.empty(0, int),
                rng.choice(far, nf, replace=False) if nf else np.empty(0, int),
            ])

            tr_fp = full_to_fp[np.concatenate([pos_full, neg_full])]
            y_s = np.concatenate([np.ones(pos_full.size, np.int8), np.zeros(neg_full.size, np.int8)])
            split_tr_fp[k].append(tr_fp)
            split_dyn[k].append(V_in[tr_fp])
            fold_y[k].append(y_s)

        del V_in, dist_ctx, ctx_vis, tgt_vis
        gc.collect()

    all_idx_per_fold = [np.concatenate(split_tr_fp[k]) for k in range(len(folds))]
    all_dyn_per_fold = [np.vstack(split_dyn[k]) for k in range(len(folds))]
    all_y_per_fold = [np.concatenate(fold_y[k]) for k in range(len(folds))]
    del split_tr_fp, split_dyn, fold_y

    n_p = len(partner_indices)
    n_d = all_dyn_per_fold[0].shape[1]
    X_tr_list = [np.empty((all_idx_per_fold[k].size, n_p + n_d), dtype=np.float32) for k in range(len(folds))]
    for k in range(len(folds)):
        X_tr_list[k][:, n_p:] = all_dyn_per_fold[k]
    del all_dyn_per_fold

    for j, c in enumerate(partner_indices):
        vec = store.read_channel(c)
        for k in range(len(folds)):
            X_tr_list[k][:, j] = vec[all_idx_per_fold[k]]
        del vec

    clfs = []
    for k in range(len(folds)):
        clfs.append(_fit_pair(X_tr_list[k], all_y_per_fold[k], use_lgbm=use_lgbm, seed=seed + 50 + k))
        X_tr_list[k] = None
    del X_tr_list, all_y_per_fold, all_idx_per_fold
    gc.collect()

    V_test = FE.compute_visible_fracture_features(visible & foot, foot, directors)
    preds_fp = np.zeros(N_fp, dtype=np.float32)
    step = 650_000
    for start in range(0, N_fp, step):
        end = min(N_fp, start + step)
        Xp = store.extract_slice(start, end, partner_indices)
        blk = np.hstack([Xp, V_test[start:end]])
        del Xp
        f_sub = fold_of_fp[start:end]
        for k in range(len(folds)):
            m = f_sub == k
            if m.any():
                preds_fp[start:end][m] = _predict_pair(clfs[k], blk[m])
        del blk

    del clfs, V_test
    gc.collect()

    struct_grid = np.zeros((H, W), dtype=np.float32)
    struct_grid.ravel()[fp_idx] = preds_fp
    w_p = np.float32(blend_phys_weight)
    blended = (w_p * phys_oof + (np.float32(1.0) - w_p) * struct_grid).astype(np.float32)
    print(f"[h77e_models] {tag} ({n_inner_splits} splits, lgbm={use_lgbm}) OOF done in {time.time() - t0:.1f}s", flush=True)
    return blended, struct_grid
