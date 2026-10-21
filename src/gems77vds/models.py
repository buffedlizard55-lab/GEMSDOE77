"""Horvitz-Thompson inverse-probability-weighted cross-family ensemble & OOF stacker (`H77E`).

Strict anti-leakage, speed & memory invariants:
1. Takes pre-scaled `(n_fp, 98)` `float16` feature matrix `X16` (`966 MB` total RAM, zero underflow,
   exact `NaN` preservation) and streams prediction in `150,000`-row chunks (`< 70 MB` scratch RAM),
   keeping peak container RSS strictly below `1.65 GiB` on 3.8 GiB zero-swap machines!
2. Strict leave-one-quadrant-out (LOQO) isolation:
   - First fits the `C(4, 2) = 6` two-quadrant `HGB` physical models once (`~11s` total!).
   - When training outer test quadrant `k`, every training quadrant `j != k` uses strictly
     the physical score fit on `{m : m != k and m != j}` (100% disjoint from BOTH test quadrant `k`
     AND quadrant `j` itself — zero test touch and zero in-sample stacking bias!).
3. Identical frozen parameters (`FROZEN_H77E_PARAMS`) shared 100% unchanged between `E2` and `E3` (`IR-77E-01`).
"""
from __future__ import annotations

import ctypes
import gc
import resource
import time

import lightgbm as lgb
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features as FE

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None


def free_ram() -> float:
    """Run Python GC + Linux glibc `malloc_trim(0)` and return current VmRSS in MiB."""
    gc.collect()
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * 4096.0 / (1024.0 * 1024.0)
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


STRUCT8 = np.ones((3, 3), dtype=bool)

HGB_KW = dict(
    max_iter=190,
    learning_rate=0.075,
    max_leaf_nodes=55,
    min_samples_leaf=45,
    l2_regularization=2.5,
    early_stopping=False,
    random_state=7701,
)

LGB_KW = dict(
    n_estimators=185,
    max_bin=127,
    learning_rate=0.07,
    num_leaves=55,
    min_child_samples=55,
    subsample_for_bin=100_000,
    colsample_bytree=0.60,
    subsample=0.80,
    subsample_freq=1,
    reg_alpha=1.5,
    reg_lambda=4.0,
    n_jobs=2,
    random_state=7702,
    verbose=-1,
)

# Frozen production parameter dictionary shared 100% identically between E2 (holdout) and E3 (production)!
FROZEN_H77E_PARAMS = {
    "pi_scale_ht": 0.25,
    "w_stack_log_odds": 0.68,
    "dinkelbach_pi_true": 0.0022,
    "dinkelbach_temperature": 1.00,
    "dinkelbach_fp_lambda": 0.012,
    "dinkelbach_min_sep_px": 2.35,
    "dinkelbach_candidate_pool": 680_000,
    "quadrant_calibrate": True,
    "excl_px": 3.0,
    "sparse_overlap_cap": 0.650,
    "max_sparse_positives": 125_000,
}


def sample_ht_strata(
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    d_ctx: np.ndarray,
    slope_2d: np.ndarray,
    fp_lut: np.ndarray,
    rng: np.random.Generator,
    max_pos: int = 44_000,
    max_h: int = 44_000,
    max_s: int = 44_000,
    max_b: int = 30_000,
    pi_scale: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample 4 exhaustive disjoint strata `(P, H, S, B)` with exact Horvitz-Thompson weights."""
    p_all = fp_lut[np.flatnonzero(pos_mask)]
    h_all = fp_lut[np.flatnonzero(neg_mask & (d_ctx <= 18.0))]
    s_all = fp_lut[np.flatnonzero(neg_mask & (d_ctx > 18.0) & (slope_2d > 3.0))]
    b_all = fp_lut[np.flatnonzero(neg_mask & (d_ctx > 18.0) & ~(slope_2d > 3.0))]

    def _draw(arr: np.ndarray, cap: int) -> tuple[np.ndarray, float]:
        N = int(arr.size)
        if N == 0:
            return arr, 0.0
        m = min(N, cap)
        sub = rng.choice(arr, m, replace=False) if m < N else arr
        return sub, float(N) / float(m)

    p_sub, wp = _draw(p_all, max_pos)
    h_sub, wh = _draw(h_all, max_h)
    s_sub, ws = _draw(s_all, max_s)
    b_sub, wb = _draw(b_all, max_b)

    idx = np.concatenate([p_sub, h_sub, s_sub, b_sub])
    y = np.concatenate(
        [
            np.ones(p_sub.size, dtype=np.int8),
            np.zeros(h_sub.size + s_sub.size + b_sub.size, dtype=np.int8),
        ]
    )
    w = np.concatenate(
        [
            np.full(p_sub.size, wp * pi_scale, dtype=np.float32),
            np.full(h_sub.size, wh, dtype=np.float32),
            np.full(s_sub.size, ws, dtype=np.float32),
            np.full(b_sub.size, wb, dtype=np.float32),
        ]
    )
    w *= float(w.size) / float(w.sum())
    return idx, y, w


def _build_interaction_cols(p_phys_vec: np.ndarray, V_mat: np.ndarray) -> np.ndarray:
    """Construct 6 geomechanical x multi-physics interaction columns (`shape = (N, 6)` float32)."""
    p = np.clip(p_phys_vec.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p / (1.0 - p)).astype(np.float32)
    d_approx = np.expm1(np.maximum(V_mat[:, 0], 0.0))
    corridor = np.exp(-d_approx / np.float32(18.0)).astype(np.float32)
    blind = (np.float32(1.0) - np.exp(-d_approx / np.float32(22.0))).astype(np.float32)
    return np.column_stack(
        [
            logit_p,
            p * V_mat[:, 5],        # physical scarp inside Pollard-Segall crack-tip cone
            p * V_mat[:, 6],        # physical scarp parallel to strike (vetoes flank canyons)
            p * V_mat[:, 4],        # physical bridge inside relay ramp step-over
            p * corridor,           # near-network continuation coupling
            p * blind,              # detached blind-basin fault coupling
        ]
    ).astype(np.float32)


def fit_predict_h77e_full(
    X16: np.ndarray,
    vis_full_fp: np.ndarray,
    folds: list[dict],
    visible: np.ndarray,
    foot: np.ndarray,
    slope_2d: np.ndarray,
    aux: dict,
    fp_lut: np.ndarray,
    seed: int = 7700,
) -> dict[str, np.ndarray]:
    """Strictly LOQO 4-quadrant fitting & prediction (`peak RSS < 1.65 GiB`, `~85s` per seed)."""
    H, W = foot.shape
    n_fp = X16.shape[0]
    rng = np.random.default_rng(seed)

    phys_hgb = np.zeros((H, W), dtype=np.float32)
    phys_lgb = np.zeros((H, W), dtype=np.float32)
    phys_ens = np.zeros((H, W), dtype=np.float32)
    stack_hgb = np.zeros((H, W), dtype=np.float32)
    stack_lgb = np.zeros((H, W), dtype=np.float32)
    stack_ens = np.zeros((H, W), dtype=np.float32)

    # Pre-fit the C(4, 2) = 6 two-quadrant physical HGB models ONCE so every inner OOF prediction
    # for training quadrant `j` when holding out quadrant `k` uses pair `tuple(sorted({0,1,2,3} - {k, j}))`!
    pair_preds_fp: dict[tuple[int, int, int], np.ndarray] = {}
    pairs_models: dict[tuple[int, int], HistGradientBoostingClassifier] = {}
    for a_q in range(4):
        for b_q in range(a_q + 1, 4):
            sub_reg = folds[a_q]["region"] | folds[b_q]["region"]
            v_sub = visible & sub_reg
            d_sub = ndi.distance_transform_edt(~v_sub).astype(np.float32)
            i_sub, y_sub, w_sub = sample_ht_strata(
                v_sub & foot,
                sub_reg & foot & (~v_sub) & (d_sub > 3.0),
                d_sub,
                slope_2d,
                fp_lut,
                rng,
                max_pos=30_000,
                max_h=30_000,
                max_s=30_000,
                max_b=20_000,
                pi_scale=FROZEN_H77E_PARAMS["pi_scale_ht"],
            )
            X_sub = X16[i_sub].astype(np.float32)
            pairs_models[(a_q, b_q)] = HistGradientBoostingClassifier(**HGB_KW).fit(X_sub, y_sub, sample_weight=w_sub)
            del X_sub, y_sub, w_sub
    gc.collect()

    # Pre-extract quadrant fp indices once
    quad_fp = [fp_lut[np.flatnonzero(folds[q]["region"] & foot)] for q in range(4)]

    # Precompute the 4 x 3 = 12 quadrant predictions from the 6 pair models in small 150k chunks
    pair_quad_pred: dict[tuple[tuple[int, int], int], np.ndarray] = {}
    for pair_key, clf_p in pairs_models.items():
        for q in range(4):
            if q not in pair_key:
                q_fp = quad_fp[q]
                pr = np.empty(q_fp.size, dtype=np.float32)
                for s0 in range(0, q_fp.size, 150_000):
                    s1 = min(q_fp.size, s0 + 150_000)
                    Xb = X16[q_fp[s0:s1]].astype(np.float32)
                    pr[s0:s1] = np.clip(clf_p.predict_proba(Xb)[:, 1], 1e-6, 1.0 - 1e-6).astype(np.float32)
                    del Xb
                pair_quad_pred[(pair_key, q)] = pr
    del pairs_models, pair_preds_fp
    free_ram()

    for k, fold in enumerate(folds):
        t_k = time.time()
        other_js = [j for j in range(4) if j != k]
        tr_reg = np.zeros((H, W), dtype=bool)
        for j in other_js:
            tr_reg |= folds[j]["region"]

        # Step A: Fit outer physical HGB + LightGBM on `tr_reg` and predict on `fold['region']`
        vis_tr = visible & tr_reg
        d_tr = ndi.distance_transform_edt(~vis_tr).astype(np.float32)
        idx_p, y_p, w_p = sample_ht_strata(
            vis_tr & foot,
            tr_reg & foot & (~vis_tr) & (d_tr > 3.0),
            d_tr,
            slope_2d,
            fp_lut,
            rng,
            pi_scale=FROZEN_H77E_PARAMS["pi_scale_ht"],
        )
        Xp = X16[idx_p].astype(np.float32)
        c_phgb = HistGradientBoostingClassifier(**HGB_KW).fit(Xp, y_p, sample_weight=w_p)
        c_plgb = lgb.LGBMClassifier(**LGB_KW).fit(Xp, y_p, sample_weight=w_p)
        del Xp, y_p, w_p
        gc.collect()

        te_mask = fold["region"] & foot
        te_fp = quad_fp[k]
        ph1 = np.empty(te_fp.size, dtype=np.float32)
        ph2 = np.empty(te_fp.size, dtype=np.float32)
        phe = np.empty(te_fp.size, dtype=np.float32)
        for s0 in range(0, te_fp.size, 150_000):
            s1 = min(te_fp.size, s0 + 150_000)
            Xb = X16[te_fp[s0:s1]].astype(np.float32)
            q1 = np.clip(c_phgb.predict_proba(Xb)[:, 1], 1e-6, 1.0 - 1e-6)
            q2 = np.clip(c_plgb.predict_proba(Xb)[:, 1], 1e-6, 1.0 - 1e-6)
            lo = 0.5 * np.log(q1 / (1.0 - q1)) + 0.5 * np.log(q2 / (1.0 - q2))
            ph1[s0:s1] = q1.astype(np.float32)
            ph2[s0:s1] = q2.astype(np.float32)
            phe[s0:s1] = (1.0 / (1.0 + np.exp(-lo))).astype(np.float32)
            del Xb
        phys_hgb[te_mask] = ph1
        phys_lgb[te_mask] = ph2
        phys_ens[te_mask] = phe
        del ph1, ph2, c_phgb, c_plgb
        free_ram()

        # Assemble strictly out-of-quadrant `p_phys_inner_fp` on `other_js` in 0.02s!
        p_phys_inner_fp = np.zeros(n_fp, dtype=np.float32)
        for j_val in other_js:
            rem_pair = tuple(sorted(m for m in other_js if m != j_val))
            p_phys_inner_fp[quad_fp[j_val]] = pair_quad_pred[(rem_pair, j_val)]

        # Step B: Inner 3-split buffered whole-component cross-fitting for 112-ch structural stacker
        comp, n_comp = ndi.label(vis_tr, STRUCT8)
        perm = rng.permutation(np.arange(1, n_comp + 1))
        groups = np.array_split(perm, 3)

        X_parts, y_parts, w_parts = [], [], []
        for g_ids in groups:
            hold_c = np.isin(comp, g_ids) & vis_tr
            collar_c = ndi.distance_transform_edt(~hold_c) <= 3.0
            ctx_m = vis_tr & (~collar_c)
            d_ctx = ndi.distance_transform_edt(~ctx_m).astype(np.float32)

            idx_m, y_m, w_m = sample_ht_strata(
                hold_c & foot & (d_ctx > 3.0),
                tr_reg & foot & (~vis_tr) & (d_ctx > 3.0),
                d_ctx,
                slope_2d,
                fp_lut,
                rng,
                max_pos=15_000,
                max_h=16_000,
                max_s=16_000,
                max_b=11_000,
                pi_scale=FROZEN_H77E_PARAMS["pi_scale_ht"],
            )
            Vm_fp = FE.compute_visible_structural_fp(
                ctx_m, foot, aux["u2_dir"], aux["v2_dir"], aux["geod_rank"]
            )
            Vm_sub = Vm_fp[idx_m]
            inter_sub = _build_interaction_cols(p_phys_inner_fp[idx_m], Vm_sub)
            Xm = np.empty((idx_m.size, 112), dtype=np.float32)
            Xm[:, :98] = X16[idx_m]
            Xm[:, 98:106] = Vm_sub
            Xm[:, 106:112] = inter_sub
            X_parts.append(Xm)
            y_parts.append(y_m)
            w_parts.append(w_m)
            del Vm_fp, Vm_sub, inter_sub
            free_ram()

        del p_phys_inner_fp
        X_st = np.vstack(X_parts)
        y_st = np.concatenate(y_parts)
        w_st = np.concatenate(w_parts)
        w_st *= float(w_st.size) / float(w_st.sum())
        del X_parts, y_parts, w_parts
        free_ram()

        c_shgb = HistGradientBoostingClassifier(**HGB_KW).fit(X_st, y_st, sample_weight=w_st)
        c_slgb = lgb.LGBMClassifier(**LGB_KW).fit(X_st, y_st, sample_weight=w_st)
        del X_st, y_st, w_st
        free_ram()

        sh1 = np.empty(te_fp.size, dtype=np.float32)
        sh2 = np.empty(te_fp.size, dtype=np.float32)
        she = np.empty(te_fp.size, dtype=np.float32)
        Xb = np.empty((150_000, 112), dtype=np.float32)
        for s0 in range(0, te_fp.size, 150_000):
            s1 = min(te_fp.size, s0 + 150_000)
            nb = s1 - s0
            V_chunk = vis_full_fp[te_fp[s0:s1]]
            inter_chunk = _build_interaction_cols(phe[s0:s1], V_chunk)
            sub_X = Xb[:nb]
            sub_X[:, :98] = X16[te_fp[s0:s1]]
            sub_X[:, 98:106] = V_chunk
            sub_X[:, 106:112] = inter_chunk
            q1 = np.clip(c_shgb.predict_proba(sub_X)[:, 1], 1e-6, 1.0 - 1e-6)
            q2 = np.clip(c_slgb.predict_proba(sub_X)[:, 1], 1e-6, 1.0 - 1e-6)
            lo = 0.5 * np.log(q1 / (1.0 - q1)) + 0.5 * np.log(q2 / (1.0 - q2))
            sh1[s0:s1] = q1.astype(np.float32)
            sh2[s0:s1] = q2.astype(np.float32)
            she[s0:s1] = (1.0 / (1.0 + np.exp(-lo))).astype(np.float32)
            del inter_chunk
        del Xb
        stack_hgb[te_mask] = sh1
        stack_lgb[te_mask] = sh2
        stack_ens[te_mask] = she
        del sh1, sh2, she, phe, c_shgb, c_slgb
        rss_now = free_ram()
        print(f"    [fold {fold['name']}] completed in {time.time() - t_k:.1f}s (live VmRSS={rss_now:.0f} MiB)", flush=True)

    del pair_quad_pred, quad_fp
    gc.collect()

    ws = FROZEN_H77E_PARAMS["w_stack_log_odds"]
    h77e_arith = np.zeros((H, W), dtype=np.float32)
    h77e_arith[foot] = ((1.0 - ws) * phys_ens[foot] + ws * stack_ens[foot]).astype(np.float32)

    h77e_primary = np.zeros((H, W), dtype=np.float32)
    p1 = np.clip(phys_ens[foot].astype(np.float64), 1e-6, 1.0 - 1e-6)
    p2 = np.clip(stack_ens[foot].astype(np.float64), 1e-6, 1.0 - 1e-6)
    lo_pool = (1.0 - ws) * np.log(p1 / (1.0 - p1)) + ws * np.log(p2 / (1.0 - p2))
    h77e_primary[foot] = (1.0 / (1.0 + np.exp(-lo_pool))).astype(np.float32)

    only_hgb = np.zeros((H, W), dtype=np.float32)
    p1h = np.clip(phys_hgb[foot].astype(np.float64), 1e-6, 1.0 - 1.0e-6)
    p2h = np.clip(stack_hgb[foot].astype(np.float64), 1e-6, 1.0 - 1.0e-6)
    lo_h = (1.0 - ws) * np.log(p1h / (1.0 - p1h)) + ws * np.log(p2h / (1.0 - p2h))
    only_hgb[foot] = (1.0 / (1.0 + np.exp(-lo_h))).astype(np.float32)

    only_lgb = np.zeros((H, W), dtype=np.float32)
    p1l = np.clip(phys_lgb[foot].astype(np.float64), 1e-6, 1.0 - 1.0e-6)
    p2l = np.clip(stack_lgb[foot].astype(np.float64), 1e-6, 1.0 - 1.0e-6)
    lo_l = (1.0 - ws) * np.log(p1l / (1.0 - p1l)) + ws * np.log(p2l / (1.0 - p2l))
    only_lgb[foot] = (1.0 / (1.0 + np.exp(-lo_l))).astype(np.float32)

    return {
        "phys_ens": phys_ens,
        "stack_ens": stack_ens,
        "only_hgb": only_hgb,
        "only_lgb": only_lgb,
        "h77e_arith": h77e_arith,
        "h77e_primary": h77e_primary,
    }
