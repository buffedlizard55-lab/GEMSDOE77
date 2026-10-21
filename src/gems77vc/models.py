"""Zero-leakage Spatially-Blocked Out-of-Fold Geothermal Vent & Breach Ensemble (`Lane H77V`).

Fast, low-memory (<1.5 GB peak RSS, ~2.5 min per seed) native float32 implementation:
  1. Uses zero-copy `fast_gather_rows` when all static columns are selected.
  2. Uses native float32 C++ LightGBM multi-depth/multi-subspace boosters (`LGBM-Deep-63`,
     `LGBM-Reg-39`, and `LGBM-Vent-Specialist-55`), avoiding sklearn HGB's internal float64 conversion.
  3. Fits and predicts each booster sequentially in 150k-row chunks, immediately freeing all buffers.
"""
from __future__ import annotations

import gc
import lightgbm as lgb
import numpy as np

from gems77_step.models import (
    REGIONAL_LEVEL_FEATURES,
    quantile_align_by_quadrant,
)


def fast_gather_rows(
    X16: np.ndarray,
    X_mech: np.ndarray | None,
    col_indices: np.ndarray,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Fast low-memory float16 -> float32 row/column gather without double intermediate copies."""
    n_static = X16.shape[1]
    static_cols = col_indices[col_indices < n_static]
    mech_cols = col_indices[col_indices >= n_static] - n_static
    n_s = int(static_cols.size)
    out = np.empty((row_indices.size, col_indices.size), dtype=np.float32)
    if n_s == n_static and np.array_equal(static_cols, np.arange(n_static, dtype=static_cols.dtype)):
        out[:, :n_s] = X16[row_indices]
    else:
        sub_r = X16[row_indices]
        out[:, :n_s] = sub_r[:, static_cols]
        del sub_r
    if mech_cols.size > 0 and X_mech is not None:
        out[:, n_s:] = X_mech[mech_cols][:, row_indices].T
    return out


def _select_vent_specialist_cols(all_names: list[str]) -> np.ndarray:
    """Select the H77V hydrothermal vent + step purity + Radon + potential-field + V-LBCO subset."""
    keep = []
    for i, nm in enumerate(all_names):
        if (
            nm.startswith("h77v_")
            or nm.startswith("vis_")
            or nm.startswith("hwfw_")
            or nm.startswith("radon_")
            or nm.startswith("hessian_")
            or nm.startswith("ctx2d_")
            or nm.startswith("hydro_")
            or nm.startswith("probe_")
            or nm.startswith("paleo_")
            or nm.startswith("quat_")
            or nm.startswith("rad_")
            or nm.startswith("ext_")
            or nm.startswith("chi_")
            or nm in (
                "tmi_hg",
                "iso_grav_anom_slope",
                "iso_grav_anom_hg",
                "geod_2ndinv",
                "geod_shearrate",
                "geod_dilaterate",
                "deq_n100a15",
                "ieq_n100a15",
                "det_elev_slope",
                "grav_hg_worm",
                "mag_hg_worm",
                "basement_grad",
                "basement_curv",
                "cond_grad",
                "tmi_demagnetization_trough",
                "topo10_slope_p90",
                "topo10_curv_prof_absmax",
                "topo10_aspect_coherence",
                "topo10_hs_lineament",
                "lidar_step_max",
                "lidar_coh100",
                "lidar_has_coverage",
            )
        ):
            keep.append(i)
    return np.asarray(keep, dtype=np.int32)


def _predict_in_chunks(
    clf: object,
    X16: np.ndarray,
    X_mech_infer: np.ndarray | None,
    cols: np.ndarray,
    val_idx: np.ndarray,
    chunk_sz: int = 160000,
) -> np.ndarray:
    out = np.empty(val_idx.size, dtype=np.float32)
    for s in range(0, val_idx.size, chunk_sz):
        sl = val_idx[s : s + chunk_sz]
        xb = fast_gather_rows(X16, X_mech_infer, cols, sl)
        out[s : s + chunk_sz] = clf.predict_proba(xb)[:, 1].astype(np.float32)
        del xb
    return out


def fit_h77v_vent_ensemble(
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
    max_neg: int = 200000,
) -> np.ndarray:
    """Fit Dual-Geometry Out-of-Fold H77V Geothermal-Vent & Structural-Mechanics Ensemble."""
    n_fp = X16.shape[0]
    full_cols = np.arange(len(all_names), dtype=np.int32)
    dereg_cols = np.array(
        [i for i, nm in enumerate(all_names) if nm not in REGIONAL_LEVEL_FEATURES],
        dtype=np.int32,
    )
    vent_cols = _select_vent_specialist_cols(all_names)
    if X_mech_train is None:
        n_static = X16.shape[1]
        full_cols = full_cols[full_cols < n_static]
        dereg_cols = dereg_cols[dereg_cols < n_static]
        vent_cols = vent_cols[vent_cols < n_static]

    slope_col_idx = all_names.index("det_elev_slope")
    vent_col_idx = all_names.index("h77v_master_vent_potential")

    # -------------------------------------------------------------------------
    # Head 1: 4-Quadrant Out-of-Fold Multi-Subspace Head (Deep-63 + Reg-39 + Vent-55)
    # -------------------------------------------------------------------------
    oof_loqo = np.zeros(n_fp, dtype=np.float32)

    for q_id in range(4):
        in_tr = (quad_fp != q_id) & (~boundary_excl_fp)
        pos_idx = np.flatnonzero(in_tr & visible_fp)
        neg_pool = np.flatnonzero(in_tr & (~visible_fp) & (d_vis_fp > 3.5))
        rng = np.random.default_rng(seed_base + 131 * q_id)
        neg_idx = np.sort(rng.choice(neg_pool, size=min(max_neg, neg_pool.size), replace=False))
        tr_idx = np.concatenate([pos_idx, neg_idx])
        y_tr = np.concatenate([np.ones(pos_idx.size, np.int32), np.zeros(neg_idx.size, np.int32)])

        slope_tr = np.asarray(X16[pos_idx, slope_col_idx], dtype=np.float32)
        vent_tr = np.asarray(X16[pos_idx, vent_col_idx], dtype=np.float32)
        low_relief = slope_tr <= float(np.percentile(slope_tr, 60.0))
        high_vent = vent_tr >= float(np.percentile(vent_tr, 60.0))
        w_pos = np.where(low_relief & high_vent, 1.36, np.where(low_relief, 1.30, 1.0)).astype(np.float32)
        sw_tr = np.concatenate([w_pos, np.ones(neg_idx.size, dtype=np.float32)])
        val_idx = np.flatnonzero(quad_fp == q_id)

        X_tr = fast_gather_rows(X16, X_mech_train, full_cols, tr_idx)
        lgb_reg = lgb.LGBMClassifier(
            n_estimators=185,
            num_leaves=39,
            min_child_samples=85,
            learning_rate=0.055,
            colsample_bytree=0.72,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=1.0,
            reg_lambda=5.0,
            n_jobs=2,
            random_state=seed_base + q_id * 19 + 1,
            verbose=-1,
        )
        lgb_reg.fit(X_tr, y_tr, sample_weight=sw_tr)
        p_reg = _predict_in_chunks(lgb_reg, X16, X_mech_infer, full_cols, val_idx)
        del lgb_reg
        gc.collect()

        lgb_deep = lgb.LGBMClassifier(
            n_estimators=210,
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
        lgb_deep.fit(X_tr, y_tr, sample_weight=sw_tr)
        del X_tr
        gc.collect()
        p_deep = _predict_in_chunks(lgb_deep, X16, X_mech_infer, full_cols, val_idx)
        del lgb_deep
        gc.collect()

        X_tr_v = fast_gather_rows(X16, X_mech_train, vent_cols, tr_idx)
        lgb_vent = lgb.LGBMClassifier(
            n_estimators=175,
            num_leaves=55,
            min_child_samples=65,
            learning_rate=0.05,
            colsample_bytree=0.62,
            subsample=0.82,
            subsample_freq=1,
            reg_alpha=1.8,
            reg_lambda=4.5,
            n_jobs=2,
            random_state=seed_base + q_id * 19 + 3,
            verbose=-1,
        )
        lgb_vent.fit(X_tr_v, y_tr, sample_weight=sw_tr)
        del X_tr_v, y_tr, sw_tr
        gc.collect()
        p_vent = _predict_in_chunks(lgb_vent, X16, X_mech_infer, vent_cols, val_idx)
        del lgb_vent
        gc.collect()

        oof_loqo[val_idx] = (0.36 * p_reg + 0.46 * p_deep + 0.18 * p_vent).astype(np.float32)
        del p_reg, p_deep, p_vent
        gc.collect()

    p_loqo_cal = quantile_align_by_quadrant(oof_loqo, has_lidar_fp, quad_fp)
    del oof_loqo

    if not allow_intra_province:
        return p_loqo_cal

    # -------------------------------------------------------------------------
    # Head 2: 45-degree Rotated Oblique Macro-Quadrants with 35 px (3.5 km) Guard Band
    # -------------------------------------------------------------------------
    u_coord = 0.70710678 * (row_fp.astype(np.float32) + col_fp.astype(np.float32))
    v_coord = 0.70710678 * (row_fp.astype(np.float32) - col_fp.astype(np.float32))
    u_med = float(np.median(u_coord))
    v_med = float(np.median(v_coord))
    oblique_q = ((u_coord >= u_med).astype(np.int8) * 2 + (v_coord >= v_med).astype(np.int8))
    oblique_guard = (np.abs(u_coord - u_med) <= 35.0) | (np.abs(v_coord - v_med) <= 35.0)
    del u_coord, v_coord

    oof_oblique = np.zeros(n_fp, dtype=np.float32)
    for b_id in range(4):
        in_tr = (oblique_q != b_id) & (~oblique_guard)
        pos_idx = np.flatnonzero(in_tr & visible_fp)
        neg_pool = np.flatnonzero(in_tr & (~visible_fp) & (d_vis_fp > 4.0))
        rng = np.random.default_rng(seed_base + 503 + 71 * b_id)
        neg_idx = np.sort(rng.choice(neg_pool, size=min(max_neg, neg_pool.size), replace=False))
        tr_idx = np.concatenate([pos_idx, neg_idx])
        y_tr = np.concatenate([np.ones(pos_idx.size, np.int32), np.zeros(neg_idx.size, np.int32)])

        slope_tr = np.asarray(X16[pos_idx, slope_col_idx], dtype=np.float32)
        vent_tr = np.asarray(X16[pos_idx, vent_col_idx], dtype=np.float32)
        low_relief = slope_tr <= float(np.percentile(slope_tr, 60.0))
        high_vent = vent_tr >= float(np.percentile(vent_tr, 60.0))
        w_pos = np.where(low_relief & high_vent, 1.36, np.where(low_relief, 1.30, 1.0)).astype(np.float32)
        sw_tr = np.concatenate([w_pos, np.ones(neg_idx.size, dtype=np.float32)])
        val_idx = np.flatnonzero(oblique_q == b_id)

        X_tr = fast_gather_rows(X16, X_mech_train, dereg_cols, tr_idx)
        lgbm_o = lgb.LGBMClassifier(
            n_estimators=195,
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
        lgbm_o.fit(X_tr, y_tr, sample_weight=sw_tr)
        del X_tr
        gc.collect()
        p_o = _predict_in_chunks(lgbm_o, X16, X_mech_infer, dereg_cols, val_idx)
        del lgbm_o
        gc.collect()

        X_tr_v = fast_gather_rows(X16, X_mech_train, vent_cols, tr_idx)
        lgbm_ov = lgb.LGBMClassifier(
            n_estimators=165,
            num_leaves=55,
            min_child_samples=70,
            learning_rate=0.05,
            colsample_bytree=0.60,
            subsample=0.82,
            subsample_freq=1,
            reg_alpha=2.0,
            reg_lambda=5.0,
            n_jobs=2,
            random_state=seed_base + 900 + b_id,
            verbose=-1,
        )
        lgbm_ov.fit(X_tr_v, y_tr, sample_weight=sw_tr)
        del X_tr_v, y_tr, sw_tr
        gc.collect()
        p_ov = _predict_in_chunks(lgbm_ov, X16, X_mech_infer, vent_cols, val_idx)
        del lgbm_ov
        gc.collect()

        oof_oblique[val_idx] = (0.80 * p_o + 0.20 * p_ov).astype(np.float32)
        del p_o, p_ov
        gc.collect()

    p_oblique_cal = quantile_align_by_quadrant(oof_oblique, has_lidar_fp, quad_fp)
    del oof_oblique

    return (0.52 * p_loqo_cal + 0.48 * p_oblique_cal).astype(np.float32)
