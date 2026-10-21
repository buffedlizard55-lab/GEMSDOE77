"""Cross-family fold-strict fault propensity learners & single-feature leakage canaries (`H77H`)."""
from __future__ import annotations

import gc
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier

from gems77e.models import (
    _assemble_matrix,
    _make_lgbm_or_fallback,
    _predict_chunked,
    _seam_align_probabilities,
    bayes_prior_shift_calibrate,
    run_single_feature_leakage_canaries,
)

__all__ = [
    "fit_predict_h77h_fold",
    "run_single_feature_leakage_canaries",
]


def fit_predict_h77h_fold(
    X16: np.ndarray,
    X_mech_train: np.ndarray,
    X_mech_infer: np.ndarray,
    has_lidar_fp: np.ndarray,
    crest_deconf_fp: np.ndarray,
    fp: np.ndarray,
    visible_train_2d: np.ndarray,
    train_region_2d: np.ndarray,
    eval_region_2d: np.ndarray | None,
    static_names: list[str] | None = None,
    static_cols: list[int] | None = None,
    use_mechanics: bool = True,
    seed: int = 7781,
    crest_alpha: float = 0.28,
) -> dict[str, Any]:
    """Fit 3-specialist H77H cross-family ensemble strictly on ``train_region_2d`` using ONLY ``visible_train_2d``."""
    rng = np.random.default_rng(seed)
    n_fp, n_static_total = X16.shape
    cols = list(range(n_static_total)) if static_cols is None else list(static_cols)

    vis_tr = np.asarray(visible_train_2d, dtype=bool) & fp & train_region_2d
    d_vis_tr = distance_transform_edt(~vis_tr).astype(np.float32)

    fp_idx = np.flatnonzero(fp)
    vis_tr_fp = vis_tr.ravel()[fp_idx]
    tr_reg_fp = (train_region_2d & fp).ravel()[fp_idx]
    d_tr_fp = d_vis_tr.ravel()[fp_idx]
    del d_vis_tr

    pos_all = np.flatnonzero(vis_tr_fp)
    pos_sub = pos_all if pos_all.size <= 54_000 else rng.choice(pos_all, 54_000, replace=False)

    neg_elig = tr_reg_fp & (~vis_tr_fp) & (d_tr_fp > 3.0)
    hard_pool = np.flatnonzero(neg_elig & (d_tr_fp <= 16.0))
    med_pool = np.flatnonzero(neg_elig & (d_tr_fp > 16.0) & (d_tr_fp <= 48.0))
    far_pool = np.flatnonzero(neg_elig & (d_tr_fp > 48.0))

    n_neg_target = int(3.2 * pos_sub.size)
    n_hard = min(hard_pool.size, int(0.45 * n_neg_target))
    n_med = min(med_pool.size, int(0.30 * n_neg_target))
    n_far = min(far_pool.size, n_neg_target - n_hard - n_med)

    neg_sub = np.concatenate(
        [
            rng.choice(hard_pool, n_hard, replace=False),
            rng.choice(med_pool, n_med, replace=False),
            rng.choice(far_pool, n_far, replace=False),
        ]
    )
    neg_rate = float(neg_sub.size) / float(max(int(neg_elig.sum()), 1)) * (
        float(pos_all.size) / float(max(pos_sub.size, 1))
    )

    perm = rng.permutation(pos_sub.size + neg_sub.size)
    tr_fp_pos = np.concatenate([pos_sub, neg_sub])[perm]
    y_tr = np.concatenate(
        [np.ones(pos_sub.size, dtype=np.int8), np.zeros(neg_sub.size, dtype=np.int8)]
    )[perm]

    eval_fp_pos = (
        np.arange(n_fp, dtype=np.int64)
        if eval_region_2d is None
        else np.flatnonzero((eval_region_2d & fp).ravel()[fp_idx])
    )
    bg_tr_mask_fp = neg_elig

    # Specialist 3: Pure Zero-Catalogue Multi-Physics LightGBM (0 SLOCO features)
    X_tr_stat = _assemble_matrix(X16, None, cols, tr_fp_pos)
    clf_stat = _make_lgbm_or_fallback(seed=seed + 31, colsample=0.60, n_estimators=230)
    clf_stat.fit(X_tr_stat, y_tr)
    del X_tr_stat
    p_stat_fp = np.zeros(n_fp, dtype=np.float32)
    p_stat_fp[eval_fp_pos] = _predict_chunked(clf_stat, X16, None, cols, eval_fp_pos)
    del clf_stat

    if not use_mechanics:
        X_tr_stat2 = _assemble_matrix(X16, None, cols, tr_fp_pos)
        clf_hgb0 = HistGradientBoostingClassifier(
            max_iter=190,
            learning_rate=0.068,
            max_leaf_nodes=63,
            max_depth=10,
            min_samples_leaf=90,
            l2_regularization=7.0,
            max_features=0.58,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.10,
            random_state=seed + 17,
        )
        clf_hgb0.fit(X_tr_stat2, y_tr)
        del X_tr_stat2
        p_hgb_fp = np.zeros(n_fp, dtype=np.float32)
        p_hgb_fp[eval_fp_pos] = _predict_chunked(clf_hgb0, X16, None, cols, eval_fp_pos)
        del clf_hgb0
        p_blend_fp = 0.55 * p_stat_fp + 0.45 * p_hgb_fp
    else:
        # Specialist 1: Full Mesh-Safe Stack + Coupled SLOCO LightGBM
        X_tr_full = _assemble_matrix(X16, X_mech_train, cols, tr_fp_pos)
        clf_lgbm = _make_lgbm_or_fallback(seed=seed + 11, colsample=0.52, n_estimators=250)
        clf_lgbm.fit(X_tr_full, y_tr)

        # Specialist 2: Full Mesh-Safe Stack + Coupled SLOCO HistGradientBoostingClassifier
        clf_hgb = HistGradientBoostingClassifier(
            max_iter=190,
            learning_rate=0.068,
            max_leaf_nodes=63,
            max_depth=10,
            min_samples_leaf=90,
            l2_regularization=7.0,
            max_features=0.55,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.10,
            random_state=seed + 23,
        )
        clf_hgb.fit(X_tr_full, y_tr)
        del X_tr_full

        p_lgbm_fp = np.zeros(n_fp, dtype=np.float32)
        p_hgb_fp = np.zeros(n_fp, dtype=np.float32)
        p_lgbm_fp[eval_fp_pos] = _predict_chunked(clf_lgbm, X16, X_mech_infer, cols, eval_fp_pos)
        p_hgb_fp[eval_fp_pos] = _predict_chunked(clf_hgb, X16, X_mech_infer, cols, eval_fp_pos)
        del clf_lgbm, clf_hgb

        z_blend = (
            0.46 * logit(np.clip(p_lgbm_fp, 1e-6, 1.0 - 1e-6))
            + 0.32 * logit(np.clip(p_hgb_fp, 1e-6, 1.0 - 1e-6))
            + 0.22 * logit(np.clip(p_stat_fp, 1e-6, 1.0 - 1e-6))
        )
        p_blend_fp = expit(z_blend).astype(np.float32)
        del p_lgbm_fp, p_hgb_fp, z_blend

    if eval_region_2d is None:
        p_blend_fp = _seam_align_probabilities(p_blend_fp, has_lidar_fp, bg_tr_mask_fp)

    pi_cal_fp = bayes_prior_shift_calibrate(p_blend_fp, neg_rate)

    pi_2d = np.zeros(fp.shape, dtype=np.float32)
    pi_2d.ravel()[fp_idx] = pi_cal_fp
    pi_smooth = gaussian_filter(pi_2d, sigma=0.75)
    pi_sharp_fp = (0.72 * pi_cal_fp + 0.28 * pi_smooth.ravel()[fp_idx]).astype(np.float32)
    utility_fp = (pi_sharp_fp * (1.0 + crest_alpha * crest_deconf_fp)).astype(np.float32)

    score_2d = np.zeros(fp.shape, dtype=np.float32)
    score_2d.ravel()[fp_idx] = utility_fp
    raw_2d = np.zeros(fp.shape, dtype=np.float32)
    raw_2d.ravel()[fp_idx] = p_blend_fp
    static_2d = np.zeros(fp.shape, dtype=np.float32)
    static_2d.ravel()[fp_idx] = p_stat_fp

    gc.collect()
    return {
        "utility_2d": score_2d,
        "prob_blend_2d": raw_2d,
        "prob_static_2d": static_2d,
        "subsample_neg_rate": neg_rate,
        "n_train_pos": int(pos_sub.size),
        "n_train_neg": int(neg_sub.size),
    }
