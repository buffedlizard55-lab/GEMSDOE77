"""Cross-family fold-strict fault propensity learners & single-feature leakage canaries (`H77E`)."""
from __future__ import annotations

import gc
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def run_single_feature_leakage_canaries(
    X16: np.ndarray,
    static_names: list[str],
    X_mech_infer: np.ndarray,
    mech_names: list[str],
    pos_fp_idx: np.ndarray,
    neg_fp_idx: np.ndarray,
    ceiling: float = 0.90,
    seed: int = 7705,
) -> dict[str, Any]:
    """Audit every single static + SLOCO feature against the symmetric ROC-AUC <= 0.90 gate."""
    rng = np.random.default_rng(seed)
    p_sub = pos_fp_idx if pos_fp_idx.size <= 25_000 else rng.choice(pos_fp_idx, 25_000, replace=False)
    n_sub = neg_fp_idx if neg_fp_idx.size <= 90_000 else rng.choice(neg_fp_idx, 90_000, replace=False)
    idx = np.concatenate([p_sub, n_sub])
    y = np.concatenate([np.ones(p_sub.size, dtype=np.int8), np.zeros(n_sub.size, dtype=np.int8)])

    rows: list[dict[str, Any]] = []
    n_static = len(static_names)
    for j, name in enumerate(static_names):
        col = X16[idx, j].astype(np.float32)
        fin = np.isfinite(col)
        if int(fin.sum()) < 200 or np.unique(y[fin]).size < 2 or float(np.ptp(col[fin])) == 0.0:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y[fin], col[fin]))
        sym = float(max(auc, 1.0 - auc))
        rows.append({"feature": name, "group": "static_physics", "auc": round(auc, 6), "symmetric_auc": round(sym, 6)})

    for j, name in enumerate(mech_names):
        col = X_mech_infer[j, idx].astype(np.float32)
        fin = np.isfinite(col)
        auc = float(roc_auc_score(y[fin], col[fin])) if int(fin.sum()) >= 200 else 0.5
        sym = float(max(auc, 1.0 - auc))
        rows.append({"feature": name, "group": "sloco_mechanics", "auc": round(auc, 6), "symmetric_auc": round(sym, 6)})

    rows.sort(key=lambda r: -r["symmetric_auc"])
    max_all = rows[0]
    max_phys = next(r for r in rows if r["group"] == "static_physics")
    violators = [r for r in rows if r["symmetric_auc"] > ceiling]
    return {
        "n_features_checked": n_static + len(mech_names),
        "n_static_checked": n_static,
        "n_mech_checked": len(mech_names),
        "ceiling": ceiling,
        "passed": len(violators) == 0,
        "violator_count": len(violators),
        "violators": violators,
        "max_overall": max_all,
        "max_static_physics": max_phys,
        "top_15_features": rows[:15],
    }


def _make_lgbm_or_fallback(seed: int, colsample: float, n_estimators: int = 240):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.065,
            num_leaves=63,
            max_depth=9,
            min_child_samples=85,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=colsample,
            reg_lambda=8.0,
            reg_alpha=1.5,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=210,
            learning_rate=0.065,
            max_leaf_nodes=63,
            max_depth=9,
            min_samples_leaf=85,
            l2_regularization=8.0,
            max_features=colsample,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.10,
            random_state=seed,
        )


def _assemble_matrix(
    X16: np.ndarray,
    X_mech: np.ndarray | None,
    static_cols: list[int],
    fp_pos: np.ndarray,
) -> np.ndarray:
    n_s = len(static_cols)
    n_m = 0 if X_mech is None else int(X_mech.shape[0])
    out = np.empty((fp_pos.size, n_s + n_m), dtype=np.float32)
    out[:, :n_s] = X16[fp_pos][:, static_cols].astype(np.float32)
    if X_mech is not None:
        out[:, n_s:] = X_mech[:, fp_pos].T
    return out


def _predict_chunked(
    clf: Any,
    X16: np.ndarray,
    X_mech_infer: np.ndarray | None,
    static_cols: list[int],
    te_fp_pos: np.ndarray,
    chunk: int = 262_144,
) -> np.ndarray:
    preds = np.empty(te_fp_pos.size, dtype=np.float32)
    for s in range(0, te_fp_pos.size, chunk):
        e = min(s + chunk, te_fp_pos.size)
        Xb = _assemble_matrix(X16, X_mech_infer, static_cols, te_fp_pos[s:e])
        preds[s:e] = clf.predict_proba(Xb)[:, 1].astype(np.float32)
        del Xb
    return preds


def _seam_align_probabilities(
    p_fp: np.ndarray,
    has_lidar_fp: np.ndarray,
    bg_train_mask_fp: np.ndarray,
) -> np.ndarray:
    """Align non-LiDAR marginal quantiles to LiDAR marginal quantiles strictly on training background."""
    lid_bg = p_fp[bg_train_mask_fp & has_lidar_fp]
    gap_bg = p_fp[bg_train_mask_fp & (~has_lidar_fp)]
    if lid_bg.size < 2000 or gap_bg.size < 2000:
        return p_fp
    qs = np.concatenate([np.linspace(0.0, 0.90, 91), np.linspace(0.901, 0.9995, 200), [1.0]])
    q_gap = np.maximum.accumulate(np.quantile(gap_bg, qs))
    q_lid = np.maximum.accumulate(np.quantile(lid_bg, qs))
    uk, ui = np.unique(q_gap, return_index=True)
    if uk.size < 4:
        return p_fp
    out = p_fp.copy()
    gap_all = ~has_lidar_fp
    out[gap_all] = np.interp(out[gap_all], uk, q_lid[ui]).astype(np.float32)
    return out


def bayes_prior_shift_calibrate(p_raw: np.ndarray, subsample_neg_rate: float) -> np.ndarray:
    """Exact Elkan (2001) Bayes posterior calibration under negative downsampling."""
    p = np.clip(p_raw.astype(np.float32), 1e-6, 1.0 - 1e-6)
    s = float(max(subsample_neg_rate, 1e-5))
    num = s * p
    return (num / (num + (1.0 - p))).astype(np.float32)


def fit_predict_h77e_fold(
    X16: np.ndarray,
    X_mech_train: np.ndarray,
    X_mech_infer: np.ndarray,
    has_lidar_fp: np.ndarray,
    crest_deconf_fp: np.ndarray,
    fp: np.ndarray,
    visible_train_2d: np.ndarray,
    train_region_2d: np.ndarray,
    eval_region_2d: np.ndarray | None,
    static_cols: list[int] | None = None,
    use_mechanics: bool = True,
    seed: int = 7701,
    crest_alpha: float = 0.28,
) -> dict[str, Any]:
    """Fit H77E dual-specialist cross-family ensemble strictly on ``train_region_2d``
    using ONLY ``visible_train_2d`` labels, and predict calibrated propensity across footprint.
    """
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

    # Specialist 3: Pure Zero-Catalogue Multi-Physics LightGBM (0 SLOCO features!)
    X_tr_stat = _assemble_matrix(X16, None, cols, tr_fp_pos)
    clf_stat = _make_lgbm_or_fallback(seed=seed + 31, colsample=0.60, n_estimators=230)
    clf_stat.fit(X_tr_stat, y_tr)
    del X_tr_stat
    p_stat_fp = np.zeros(n_fp, dtype=np.float32)
    p_stat_fp[eval_fp_pos] = _predict_chunked(clf_stat, X16, None, cols, eval_fp_pos)
    del clf_stat

    if not use_mechanics:
        # Also fit pure-physics HistGBDT for fair cross-family capacity without mechanics
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
        # Specialist 1: SLOCO + Multi-Physics LightGBM (colsample=0.52 forces physical split diversity)
        X_tr_full = _assemble_matrix(X16, X_mech_train, cols, tr_fp_pos)
        clf_lgbm = _make_lgbm_or_fallback(seed=seed + 11, colsample=0.52, n_estimators=250)
        clf_lgbm.fit(X_tr_full, y_tr)

        # Specialist 2: SLOCO + Multi-Physics HistGradientBoostingClassifier
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

        # Log-odds + probability synthesis preserving both off-end SLOCO and detached pure-physics discoveries
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

    # Road-deconfounded subpixel scarp crest sharpening in probability space
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
