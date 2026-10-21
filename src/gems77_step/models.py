"""Zero-leakage spatial ensemble with Voronoi Leave-Bucket-of-Components-Out (`V-LBCO`)
and SAR-PU low-relief piedmont reweighting (`GEMSDOE77`).

Architectural Firewall:
  Every training and prediction routine in this module accepts ONLY ``(X16, X_mech_train,
  X_mech_infer, all_names, visible_fp, ...)`` — withheld ``truth`` and global ``existing_faults``
  never enter scope anywhere in this file.
"""
from __future__ import annotations

import gc
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# Regional absolute-level bands excluded from the oblique intra-province head so trees learn
# strictly local physical signatures (step purity, Radon continuity, Pollard cones, heat flow).
REGIONAL_LEVEL_FEATURES = frozenset({
    "mag_anom",
    "rtp",
    "tc",
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "depth_to_base_surf",
    "cond_surf",
    "topo10_dem_mean",
    "chi_zmean_m",
})


def quantile_align_by_quadrant(
    raw_prob_fp: np.ndarray,
    has_lidar_fp: np.ndarray,
    quad_fp: np.ndarray,
) -> np.ndarray:
    """Align non-LiDAR probability distribution in each quadrant to LiDAR tail percentiles."""
    out = raw_prob_fp.astype(np.float32, copy=True)
    q_grid = np.linspace(0.0, 1.0, 2001)
    for q_id in range(4):
        m_lid = (quad_fp == q_id) & has_lidar_fp
        m_gap = (quad_fp == q_id) & (~has_lidar_fp)
        if m_lid.sum() > 100 and m_gap.sum() > 100:
            src_q = np.quantile(raw_prob_fp[m_gap], q_grid)
            dst_q = np.quantile(raw_prob_fp[m_lid], q_grid)
            out[m_gap] = np.interp(raw_prob_fp[m_gap], src_q, dst_q).astype(np.float32)
    return out


def gather_rows(
    X16: np.ndarray,
    X_mech: np.ndarray | None,
    col_indices: np.ndarray,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Ultra-fast row-major float16 -> contiguous float32 gather (~30 ms per fold)."""
    n_static = X16.shape[1]
    static_cols = col_indices[col_indices < n_static]
    mech_cols = col_indices[col_indices >= n_static] - n_static
    if mech_cols.size == 0 or X_mech is None:
        return X16[row_indices][:, static_cols].astype(np.float32)
    out = np.empty((row_indices.size, col_indices.size), dtype=np.float32)
    n_s = static_cols.size
    out[:, :n_s] = X16[row_indices][:, static_cols]
    out[:, n_s:] = X_mech[mech_cols][:, row_indices].T
    return out


def fit_loqo_baseline_ensemble(
    X16: np.ndarray,
    col_indices: np.ndarray,
    visible_fp: np.ndarray,
    d_vis_fp: np.ndarray,
    quad_fp: np.ndarray,
    boundary_excl_fp: np.ndarray,
    has_lidar_fp: np.ndarray,
    seed_base: int = 77100,
    max_neg: int = 240000,
) -> np.ndarray:
    """Experiment 77.1 baseline: strict 4-quadrant LOQO HGB on 74 localized physical features."""
    n_fp = X16.shape[0]
    oof = np.zeros(n_fp, dtype=np.float32)
    chunk_sz = 900000

    for q_id in range(4):
        in_tr = (quad_fp != q_id) & (~boundary_excl_fp)
        pos_idx = np.flatnonzero(in_tr & visible_fp)
        neg_pool = np.flatnonzero(in_tr & (~visible_fp) & (d_vis_fp > 3.5))
        rng = np.random.default_rng(seed_base + 97 * q_id)
        neg_idx = np.sort(rng.choice(neg_pool, size=min(max_neg, neg_pool.size), replace=False))
        tr_idx = np.concatenate([pos_idx, neg_idx])
        y_tr = np.concatenate([np.ones(pos_idx.size, np.int32), np.zeros(neg_idx.size, np.int32)])

        X_tr = gather_rows(X16, None, col_indices, tr_idx)
        clf1 = HistGradientBoostingClassifier(
            max_iter=240, max_leaf_nodes=63, min_samples_leaf=50,
            learning_rate=0.055, l2_regularization=2.5, random_state=seed_base + q_id * 13 + 1,
        )
        clf2 = HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=31, min_samples_leaf=90,
            learning_rate=0.05, l2_regularization=4.0, random_state=seed_base + q_id * 13 + 2,
        )
        clf1.fit(X_tr, y_tr)
        clf2.fit(X_tr, y_tr)
        del X_tr, y_tr
        gc.collect()

        val_idx = np.flatnonzero(quad_fp == q_id)
        preds = np.empty(val_idx.size, dtype=np.float32)
        for s in range(0, val_idx.size, chunk_sz):
            sl = val_idx[s : s + chunk_sz]
            xb = gather_rows(X16, None, col_indices, sl)
            preds[s : s + chunk_sz] = (0.55 * clf1.predict_proba(xb)[:, 1] + 0.45 * clf2.predict_proba(xb)[:, 1]).astype(np.float32)
            del xb
        oof[val_idx] = preds
        del clf1, clf2, preds
        gc.collect()

    return quantile_align_by_quadrant(oof, has_lidar_fp, quad_fp)


def fit_h77_deconfounded_ensemble(
    X16: np.ndarray,
    X_mech_train: np.ndarray | None,
    X_mech_infer: np.ndarray | None,
    all_names: list[str],
    visible_fp: np.ndarray,
    d_vis_fp: np.ndarray,
    quad_fp: np.ndarray,
    boundary_excl_fp: np.ndarray,
    has_lidar_fp: np.ndarray,
    row_fp: np.ndarray,
    col_fp: np.ndarray,
    allow_intra_province: bool = True,
    seed_base: int = 77200,
    max_neg: int = 260000,
) -> np.ndarray:
    """Experiment 77.2 / 77.3 Multi-Family + V-LBCO + SAR-PU Piedmont-Weighted Ensemble.

    Key innovations:
      1. Training rows use ``X_mech_train`` (5-bucket Voronoi Leave-Bucket-of-Components-Out),
         completely eliminating the self-distance tautology ``d(x, V) == 0`` on training positives!
         Inference rows use ``X_mech_infer`` (full fold-visible ``V``).
      2. Cross-family ensembling (HistGradientBoosting + feature-subsampled LightGBM with
         ``colsample_bytree=0.58``) forces splits across H77-1 dip-slip step purity, dip-shifted
         gravity/basement concordance, Radon worms, and geothermal upflow channels.
      3. Gentle SAR-PU sample weighting (1.30x on low-relief Quaternary faults) lifts subtle
         alluvial piedmont and intrabasin scarps without suppressing range-front continuations.
    """
    n_fp = X16.shape[0]
    import lightgbm as lgb

    full_cols = np.arange(len(all_names), dtype=np.int32)
    dereg_cols = np.array(
        [i for i, nm in enumerate(all_names) if nm not in REGIONAL_LEVEL_FEATURES],
        dtype=np.int32,
    )
    slope_col_idx = all_names.index("det_elev_slope")
    chunk_sz = 900000

    # Head 1: 4-Quadrant Out-of-Fold Cross-Family Head (HGB + Colsample LightGBM)
    oof_loqo = np.zeros(n_fp, dtype=np.float32)
    for q_id in range(4):
        in_tr = (quad_fp != q_id) & (~boundary_excl_fp)
        pos_idx = np.flatnonzero(in_tr & visible_fp)
        neg_pool = np.flatnonzero(in_tr & (~visible_fp) & (d_vis_fp > 3.5))
        rng = np.random.default_rng(seed_base + 131 * q_id)
        neg_idx = np.sort(rng.choice(neg_pool, size=min(max_neg, neg_pool.size), replace=False))
        tr_idx = np.concatenate([pos_idx, neg_idx])
        y_tr = np.concatenate([np.ones(pos_idx.size, np.int32), np.zeros(neg_idx.size, np.int32)])

        slope_tr = X16[pos_idx, slope_col_idx].astype(np.float32)
        w_pos = np.where(slope_tr <= float(np.percentile(slope_tr, 60.0)), 1.30, 1.0).astype(np.float32)
        sw_tr = np.concatenate([w_pos, np.ones(neg_idx.size, dtype=np.float32)])

        X_tr = gather_rows(X16, X_mech_train, full_cols, tr_idx)
        hgb = HistGradientBoostingClassifier(
            max_iter=190, max_leaf_nodes=63, min_samples_leaf=50,
            learning_rate=0.055, l2_regularization=3.0, random_state=seed_base + q_id * 19 + 1,
        )
        lgbm = lgb.LGBMClassifier(
            n_estimators=260,
            num_leaves=63,
            min_child_samples=60,
            learning_rate=0.05,
            colsample_bytree=0.58,
            subsample=0.80,
            subsample_freq=1,
            reg_alpha=1.5,
            reg_lambda=4.0,
            n_jobs=2,
            random_state=seed_base + q_id * 19 + 2,
            verbose=-1,
        )
        hgb.fit(X_tr, y_tr, sample_weight=sw_tr)
        lgbm.fit(X_tr, y_tr, sample_weight=sw_tr)
        del X_tr, y_tr, sw_tr
        gc.collect()

        val_idx = np.flatnonzero(quad_fp == q_id)
        preds_q = np.empty(val_idx.size, dtype=np.float32)
        for s in range(0, val_idx.size, chunk_sz):
            sl = val_idx[s : s + chunk_sz]
            xb = gather_rows(X16, X_mech_infer, full_cols, sl)
            preds_q[s : s + chunk_sz] = (
                0.45 * hgb.predict_proba(xb)[:, 1] + 0.55 * lgbm.predict_proba(xb)[:, 1]
            ).astype(np.float32)
            del xb
        oof_loqo[val_idx] = preds_q
        del hgb, lgbm, preds_q
        gc.collect()

    p_loqo_cal = quantile_align_by_quadrant(oof_loqo, has_lidar_fp, quad_fp)
    if not allow_intra_province:
        return p_loqo_cal

    # Head 2: 45-degree Rotated Oblique Macro-Quadrants with 35 px (3.5 km) Guard Band
    u_coord = 0.70710678 * (row_fp.astype(np.float32) + col_fp.astype(np.float32))
    v_coord = 0.70710678 * (row_fp.astype(np.float32) - col_fp.astype(np.float32))
    u_med = float(np.median(u_coord))
    v_med = float(np.median(v_coord))
    oblique_q = ((u_coord >= u_med).astype(np.int8) * 2 + (v_coord >= v_med).astype(np.int8))
    oblique_guard = (np.abs(u_coord - u_med) <= 35.0) | (np.abs(v_coord - v_med) <= 35.0)
    del u_coord, v_coord

    oof_vis = np.zeros(n_fp, dtype=np.float32)
    for b_id in range(4):
        in_tr = (oblique_q != b_id) & (~oblique_guard)
        pos_idx = np.flatnonzero(in_tr & visible_fp)
        neg_pool = np.flatnonzero(in_tr & (~visible_fp) & (d_vis_fp > 4.0))
        rng = np.random.default_rng(seed_base + 503 + 71 * b_id)
        neg_idx = np.sort(rng.choice(neg_pool, size=min(max_neg, neg_pool.size), replace=False))
        tr_idx = np.concatenate([pos_idx, neg_idx])
        y_tr = np.concatenate([np.ones(pos_idx.size, np.int32), np.zeros(neg_idx.size, np.int32)])

        slope_tr = X16[pos_idx, slope_col_idx].astype(np.float32)
        w_pos = np.where(slope_tr <= float(np.percentile(slope_tr, 60.0)), 1.30, 1.0).astype(np.float32)
        sw_tr = np.concatenate([w_pos, np.ones(neg_idx.size, dtype=np.float32)])

        X_tr = gather_rows(X16, X_mech_train, dereg_cols, tr_idx)
        lgbm_v = lgb.LGBMClassifier(
            n_estimators=250,
            num_leaves=55,
            min_child_samples=70,
            learning_rate=0.05,
            colsample_bytree=0.56,
            subsample=0.80,
            subsample_freq=1,
            reg_alpha=2.0,
            reg_lambda=5.0,
            n_jobs=2,
            random_state=seed_base + 800 + b_id,
            verbose=-1,
        )
        lgbm_v.fit(X_tr, y_tr, sample_weight=sw_tr)
        del X_tr, y_tr, sw_tr
        gc.collect()

        val_idx = np.flatnonzero(oblique_q == b_id)
        preds_v = np.empty(val_idx.size, dtype=np.float32)
        for s in range(0, val_idx.size, chunk_sz):
            sl = val_idx[s : s + chunk_sz]
            xb = gather_rows(X16, X_mech_infer, dereg_cols, sl)
            preds_v[s : s + chunk_sz] = lgbm_v.predict_proba(xb)[:, 1].astype(np.float32)
            del xb
        oof_vis[val_idx] = preds_v
        del lgbm_v, preds_v
        gc.collect()

    p_vis_cal = quantile_align_by_quadrant(oof_vis, has_lidar_fp, quad_fp)
    return (0.52 * p_loqo_cal + 0.48 * p_vis_cal).astype(np.float32)
