#!/usr/bin/env python3
"""Experiment E2 (2 of 3): Multi-Seed Hide-and-Recover, Detached & 4-Quadrant Holdout Benchmark.

Pre-registered under `registry/preregistration_h77e.json` (`H77E-AUTOCTX-DINKELBACH`).
Evaluates `H77E-1` and all pre-registered feature/stage/solver/covariate-matching ablations via
unmodified `vendor/gems57/evaluate_holdout.py` (`gems57-pooled-hide-v2`, `alpha=0.2`, `beta=0.8`,
`300 m` triangular kernel, `2,000` paired 20 km physical spatial-block bootstrap draws) across:
  1. Seed 30 (`11,342` withheld positive px), Seed 20 (`11,321` px), Seed 21 (`11,572` px)
     -> `34,235` pooled withheld positive pixels (`12` spatial folds).
  2. Far-from-catalogue `detached` strand evaluation (`d_vis >= 8 px`, `> 800 m` from any visible trace)
     and Controlled Feature-Block & Covariate-Shift Ablations on Seed 30.
  3. Blind 4-Quadrant Province Holdout (`57,448` positive px; 100% of catalogue inside each held-out
     quarter of Nevada hidden simultaneously).
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems57 import evaluate_holdout as EH, grid as G, holdout as HO  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems77lia import allocate as LIA_AL, core as LIA_CORE  # noqa: E402
from gems77e.autoctx import (  # noqa: E402
    _fit_fast_dual_lgb,
    _gather_f32,
    _safe_logit,
    _sample_balanced_fold_rows,
    compute_filament_autocontext,
    fit_h77e_twostage_oof,
    quantile_bridge_lidar_gap,
)
from gems77e.dinkelbach import allocate_dinkelbach_1swap, tip_gated_allowed_mask  # noqa: E402
from gems77e.features import compute_visible_mechanics_pair, load_h77e_static_cache  # noqa: E402


def log(*a) -> None:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.strftime('%H:%M:%S')} RSS={rss_mb:6.0f}M]", *a, flush=True)


def _data_dir() -> Path:
    for p in (ROOT / ".cache" / "data", ROOT / ".cache" / "gems_data", ROOT / "data" / "raw"):
        if (p / "training_features.tif").exists():
            return p
    raise FileNotFoundError("Missing data directory")


def _score_across_folds(dots_f32: np.ndarray, folds: list[dict], foot: np.ndarray) -> tuple[dict, np.ndarray]:
    terms = None
    per = {}
    for fold in folds:
        res, t = EH.evaluate(dots_f32, fold, valid=foot, block_side=200)
        per[fold["name"]] = {
            "dti": round(float(res["dti"]), 6),
            "n_truth": int(res["n_truth"]),
            "emitted": int(res["n_emitted"]),
            "tpw": round(float(res["tpw"]), 3),
            "fpw": round(float(res["fpw"]), 3),
        }
        terms = t if terms is None else terms + t
    return per, terms


def _single_block_quick_oof(
    X16: np.ndarray,
    X_mech_tr: np.ndarray | None,
    X_mech_inf: np.ndarray | None,
    keep_cols: np.ndarray,
    visible_fp: np.ndarray,
    d_vis_fp: np.ndarray,
    quad_fp: np.ndarray,
    boundary_excl_fp: np.ndarray,
    slope_fp: np.ndarray,
    seed: int,
    include_collar_neg: bool = True,
) -> np.ndarray:
    """Single controlled model family pass for clean single-factor ablations (~28s, <250MB RSS)."""
    n_fp = X16.shape[0]
    oof_cal = np.zeros(n_fp, dtype=np.float32)
    chunk = 700_000

    for q_id in range(4):
        in_tr = (quad_fp != q_id) & (~boundary_excl_fp)
        pos_idx = np.flatnonzero(in_tr & visible_fp)
        if include_collar_neg:
            neg_pool = np.flatnonzero(in_tr & (~visible_fp))
        else:
            neg_pool = np.flatnonzero(in_tr & (~visible_fp) & (d_vis_fp > 3.5))
        rng = np.random.default_rng(seed + 17 * q_id)
        neg_idx = np.sort(rng.choice(neg_pool, size=min(230_000, neg_pool.size), replace=False))
        tr_idx = np.concatenate([pos_idx, neg_idx])
        y_tr = np.concatenate([np.ones(pos_idx.size, np.int32), np.zeros(neg_idx.size, np.int32)])
        p60 = float(np.percentile(slope_fp[pos_idx], 60.0))
        sw = np.concatenate([
            np.where(slope_fp[pos_idx] <= p60, 1.35, 1.0).astype(np.float32),
            np.ones(neg_idx.size, dtype=np.float32),
        ])

        X_tr = _gather_f32(X16, X_mech_tr, keep_cols, tr_idx)
        m = lgb.LGBMClassifier(
            n_estimators=240, num_leaves=63, min_child_samples=60, learning_rate=0.052,
            colsample_bytree=0.58, subsample=0.80, subsample_freq=1,
            reg_alpha=1.5, reg_lambda=4.0, n_jobs=2, random_state=seed + q_id, verbose=-1,
        )
        m.fit(X_tr, y_tr, sample_weight=sw)
        del X_tr, y_tr, sw

        cal_pool = np.flatnonzero(in_tr)[::5]
        Xc = _gather_f32(X16, X_mech_tr, keep_cols, cal_pool)
        rc = np.clip(m.predict_proba(Xc)[:, 1], 1e-6, 1.0 - 1e-6)
        del Xc
        lgc = np.log(rc / (1.0 - rc))
        pl = LogisticRegression(C=0.5, max_iter=300).fit(
            np.column_stack([lgc, np.maximum(lgc, 0.0) ** 2]), visible_fp[cal_pool].astype(np.int32)
        )

        val_idx = np.flatnonzero(quad_fp == q_id)
        pv = np.empty(val_idx.size, dtype=np.float64)
        for s in range(0, val_idx.size, chunk):
            sl = val_idx[s : s + chunk]
            Xv = _gather_f32(X16, X_mech_inf, keep_cols, sl)
            pv[s : s + chunk] = np.clip(m.predict_proba(Xv)[:, 1], 1e-6, 1.0 - 1e-6)
            del Xv
        lgv = np.log(pv / (1.0 - pv))
        oof_cal[val_idx] = pl.predict_proba(np.column_stack([lgv, np.maximum(lgv, 0.0) ** 2]))[:, 1].astype(np.float32)
        del m, pl, pv, lgv
        gc.collect()
    return oof_cal


def main() -> None:
    t0 = time.time()
    dd = _data_dir()
    work_dir = ROOT / ".cache" / "work"
    X16, aux, meta = load_h77e_static_cache(dd, work_dir, force=False)
    static_names = meta["channels"]
    blocks = meta["blocks"]

    grid = G.load_grid(dd)
    foot = grid.footprint
    h, w = foot.shape
    quad_2d = HO.quadrant_ids(foot)
    quad_fp = quad_2d[foot].astype(np.int8)

    geod_rank_2d = np.zeros((h, w), dtype=np.float32)
    scarp_rank_2d = np.zeros((h, w), dtype=np.float32)
    crest_2d = np.zeros((h, w), dtype=np.float32)
    geod_rank_2d[foot] = aux["geod_rank_fp"]
    scarp_rank_2d[foot] = aux["scarp_rank_fp"]
    crest_2d[foot] = aux["crest_fp"]
    has_lidar_fp = aux["has_lidar_fp"] > 0.5
    slope_fp = np.nan_to_num(aux["slope_fp"], nan=0.0).astype(np.float32)

    dsup_zero_mask = np.zeros((h, w), dtype=bool)
    sgmc_p = LIA_CORE.WORK / "sgmc_d_sup_m.npy"
    h195_p = ROOT / ".cache" / "registry" / "5d1a920f40fa3a589e91248c1309761e44d45da8.tif"
    if sgmc_p.is_file():
        dsup_zero_mask = np.load(sgmc_p) == 0
    elif h195_p.is_file():
        import rasterio
        with rasterio.open(h195_p) as s_h195:
            a_h195 = s_h195.read(1)
        dsup_zero_mask = np.isfinite(a_h195) & (a_h195 > 0) & foot
        del a_h195

    seed_results = {}
    pooled_terms_all_seeds: dict[str, np.ndarray] = {}

    only_seeds_arg = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [30, 20, 21]
    for seed in only_seeds_arg:
        seed_done_json = work_dir / f"h77e_seed{seed}_summary.json"
        seed_done_terms = work_dir / f"h77e_seed{seed}_terms.npz"
        if seed_done_json.is_file() and seed_done_terms.is_file():
            seed_results[f"seed_{seed}"] = json.loads(seed_done_json.read_text())
            zt = np.load(seed_done_terms)
            for k in zt.files:
                pooled_terms_all_seeds[k] = (
                    zt[k].copy() if k not in pooled_terms_all_seeds else pooled_terms_all_seeds[k] + zt[k]
                )
            log(f"[SEED {seed}] Loaded cached complete seed summary ({seed_results[f'seed_{seed}']['arm_details']['H77E_full_B37654']['pooled_holdout_dti']:.5f}).")
            continue
        ts = time.time()
        folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
        visible = folds[0]["visible"]
        hidden = folds[0]["hidden_all"]
        visible_fp = visible[foot]
        d_vis_2d = ndi.distance_transform_edt(~visible).astype(np.float32)
        d_vis_fp = d_vis_2d[foot]

        eval_union = np.zeros((h, w), dtype=bool)
        for f in folds:
            eval_union |= f["region"]
        boundary_excl_fp = (~eval_union)[foot]

        log(f"=== SEED {seed}: withheld={int(hidden.sum()):,} px, visible={int(visible.sum()):,} px ===")
        mech_cache = work_dir / f"h77e_mech_seed{seed}.npz"
        if mech_cache.is_file():
            zm = np.load(mech_cache)
            X_mech_tr, X_mech_inf = zm["tr"], zm["inf"]
            del zm
        else:
            X_mech_tr, X_mech_inf = compute_visible_mechanics_pair(
                visible, foot, geod_rank_2d, scarp_rank_2d, n_buckets=5, buffer_px=3.0, seed=seed
            )
            np.savez(mech_cache, tr=X_mech_tr, inf=X_mech_inf)
        tip_sig_fp = np.maximum(X_mech_inf[2], X_mech_inf[3]) + 0.5 * X_mech_inf[4]

        oof_cache = work_dir / f"h77e_oof_v2_seed{seed}.npz"
        if oof_cache.is_file():
            zo = np.load(oof_cache)
            oof = {k: zo[k] for k in zo.files}
            del zo
            log(f"[SEED {seed}] Loaded cached H77E v2 OOF surfaces.")
        else:
            oof = fit_h77e_twostage_oof(
                X16=X16,
                X_mech_train=X_mech_tr,
                X_mech_infer=X_mech_inf,
                static_names=static_names,
                footprint=foot,
                visible_fp=visible_fp,
                d_vis_fp=d_vis_fp,
                quad_fp=quad_fp,
                boundary_excl_fp=boundary_excl_fp,
                has_lidar_fp=has_lidar_fp,
                slope_fp=slope_fp,
                crest_fp=aux["crest_fp"],
                seed_base=77000 + seed * 10,
            )
            np.savez(oof_cache, **oof)
            log(f"[SEED {seed}] Fitted H77E Two-Stage Auto-Context in {time.time() - ts:.1f}s.")

        p_cal_2d = np.zeros((h, w), dtype=np.float32)
        p_s1_2d = np.zeros((h, w), dtype=np.float32)
        p_phys_2d = np.zeros((h, w), dtype=np.float32)
        p_cal_2d[foot] = oof["p_autoctx_cal"]
        p_s1_2d[foot] = oof["p_stage1"]
        p_phys_2d[foot] = oof["p_phys"]

        allow_tip = eval_union & tip_gated_allowed_mask(
            foot, visible, tip_sig_fp, oof["p_autoctx_raw"], base_collar_px=3.00, tip_collar_px=2.45
        )
        allow_d3 = eval_union & foot & (~visible) & (d_vis_2d > 3.00)
        allow_phys = eval_union & foot & (~visible)

        far_mask = hidden & (d_vis_2d >= 8.0)
        folds_far = [
            {
                "fold": f["fold"],
                "name": f["name"],
                "region": f["region"],
                "truth": far_mask & f["region"],
                "visible": visible,
                "masked_known": grid.catalogue & (~(far_mask & f["region"])),
            }
            for f in folds if (far_mask & f["region"]).any()
        ]

        terms_seed: dict[str, np.ndarray] = {}
        terms_far: dict[str, np.ndarray] = {}
        details_seed: dict[str, dict] = {}
        tot_a = sum(int((f["region"] & allow_d3).sum()) for f in folds)

        def _register_arm(aname: str, d_arr: np.ndarray, st: dict, score_far: bool = False) -> None:
            per, t = _score_across_folds(d_arr, folds, foot)
            terms_seed[aname] = t
            details_seed[aname] = {
                "emitted_dots": int((d_arr > 0).sum()),
                "pooled_holdout_dti": round(float(EH.from_terms(t.sum(axis=0))), 6),
                "per_fold": per,
                "solver_stats": st,
            }
            if aname in (
                "H77E_full_B37654",
                "H77E_tipgated_B37654",
                "H77E_full_B44090",
                "H77E_auto_dinkelbach",
                "H77E_no_1swap_B37654",
                "H77E_foldprop_B37654",
                "H77E_stage1_only_B37654",
                "H77E_greedy_nms_B37654",
                "H77E_pure_physics_B37654",
                "A0_h195_foldfair",
                "A0_random_B37654",
            ):
                pooled_terms_all_seeds[aname] = (
                    t.copy() if aname not in pooled_terms_all_seeds else pooled_terms_all_seeds[aname] + t
                )
            if score_far and folds_far:
                _, tf = _score_across_folds(d_arr, folds_far, foot)
                terms_far[aname] = tf
            log(f"  [SEED {seed}] {aname:<28s} -> HOLDOUT-DTI = {details_seed[aname]['pooled_holdout_dti']:.5f} ({details_seed[aname]['emitted_dots']:,} dots)")

        # Score core pre-registered arms sequentially (constant <900 MB RSS!)
        d, st = allocate_dinkelbach_1swap(p_cal_2d, p_cal_2d, crest_2d, allow_d3, target_dots=37_654, n_swap_rounds=2)
        _register_arm("H77E_full_B37654", d, st, score_far=True); del d

        d, st = allocate_dinkelbach_1swap(p_cal_2d, p_cal_2d, crest_2d, allow_tip, target_dots=37_654, n_swap_rounds=2)
        _register_arm("H77E_tipgated_B37654", d, st); del d

        d, st = allocate_dinkelbach_1swap(p_cal_2d, p_cal_2d, crest_2d, allow_d3, target_dots=44_090, n_swap_rounds=2)
        _register_arm("H77E_full_B44090", d, st); del d

        d, st = allocate_dinkelbach_1swap(
            p_cal_2d, p_cal_2d, crest_2d, allow_d3, target_dots=None,
            auto_dinkelbach_bounds=(34_000, 44_090), n_swap_rounds=2
        )
        _register_arm("H77E_auto_dinkelbach", d, st); del d

        d, st = allocate_dinkelbach_1swap(p_cal_2d, p_cal_2d, crest_2d, allow_d3, target_dots=37_654, n_swap_rounds=0)
        _register_arm("H77E_no_1swap_B37654", d, st); del d

        d_fprop = np.zeros((h, w), dtype=np.float32)
        for f in folds:
            af = f["region"] & allow_d3
            bf = int(round(37_654 * int(af.sum()) / max(tot_a, 1)))
            df_sub, _ = allocate_dinkelbach_1swap(p_cal_2d, p_cal_2d, crest_2d, af, target_dots=bf, n_swap_rounds=2)
            np.maximum(d_fprop, df_sub, out=d_fprop); del df_sub
        _register_arm("H77E_foldprop_B37654", d_fprop, {"emitted_dots": int((d_fprop > 0).sum())}); del d_fprop

        d, st = allocate_dinkelbach_1swap(p_s1_2d, p_s1_2d, crest_2d, allow_d3, target_dots=37_654, n_swap_rounds=2)
        _register_arm("H77E_stage1_only_B37654", d, st, score_far=True); del d

        d_gnms = np.zeros((h, w), dtype=bool)
        for f in folds:
            af = f["region"] & allow_d3
            bf = int(round(37_654 * int(af.sum()) / max(tot_a, 1)))
            d_gnms |= emit77.emit_dots(p_cal_2d, af, bf)
        _register_arm("H77E_greedy_nms_B37654", d_gnms.astype(np.float32), {"emitted_dots": int(d_gnms.sum())}); del d_gnms

        d, st = allocate_dinkelbach_1swap(p_phys_2d, p_phys_2d, crest_2d, allow_phys, target_dots=37_654, n_swap_rounds=2)
        _register_arm("H77E_pure_physics_B37654", d, st, score_far=True); del d

        rk = np.random.default_rng(7).random(h * w)
        d_a0 = np.zeros((h, w), dtype=bool)
        if np.any(dsup_zero_mask):
            for f in folds:
                vf = f["region"] & foot & (~visible) & (d_vis_2d > 2.0)
                ac = np.flatnonzero((dsup_zero_mask & vf).ravel())
                sel_a0 = LIA_AL.greedy_select(ac[np.argsort(rk[ac])], 3.0)
                d_a0.ravel()[sel_a0] = True
        _register_arm("A0_h195_foldfair", d_a0.astype(np.float32), {"emitted_dots": int(d_a0.sum())}); del d_a0, rk

        rng_r = np.random.default_rng(900 + seed)
        d_rnd = np.zeros((h, w), dtype=np.float32)
        d_rnd.ravel()[rng_r.choice(np.flatnonzero(allow_d3.ravel()), size=37_654, replace=False)] = 1.0
        _register_arm("A0_random_B37654", d_rnd, {"emitted_dots": 37_654}, score_far=True); del d_rnd
        del p_cal_2d, p_s1_2d, p_phys_2d, oof
        gc.collect()

        # Controlled Single-Factor Covariate-Shift & Feature-Block Ablations on Primary Seed 30
        if seed == 30:
            all_c = np.arange(len(static_names), dtype=np.int32)
            c_full1 = _single_block_quick_oof(
                X16, X_mech_tr, X_mech_inf, all_c, visible_fp, d_vis_fp, quad_fp, boundary_excl_fp, slope_fp,
                seed=3100, include_collar_neg=True
            )
            p2 = np.zeros((h, w), np.float32); p2[foot] = c_full1; del c_full1
            d, st = allocate_dinkelbach_1swap(p2, p2, crest_2d, allow_d3, target_dots=37_654, n_swap_rounds=1)
            _register_arm("Abl_singlehead_covmatched", d, st); del d, p2
            gc.collect()

            for blk_name in ("graben_scarp_1m_10m", "chi_knickpoint", "euler_blakely_hydro"):
                drop_set = set(blocks[blk_name])
                keep_c = np.array([i for i, nm in enumerate(static_names) if nm not in drop_set], dtype=np.int32)
                cc = _single_block_quick_oof(
                    X16, X_mech_tr, X_mech_inf, keep_c, visible_fp, d_vis_fp, quad_fp, boundary_excl_fp, slope_fp,
                    seed=3100, include_collar_neg=True
                )
                p2 = np.zeros((h, w), np.float32); p2[foot] = cc; del cc
                d, st = allocate_dinkelbach_1swap(p2, p2, crest_2d, allow_d3, target_dots=37_654, n_swap_rounds=1)
                _register_arm(f"Abl_drop_{blk_name}", d, st); del d, p2
                gc.collect()

            # Legacy unbuffered component-LOCO ablation (5 lightweight EDTs -> <100 MB peak RAM!)
            del X_mech_tr
            gc.collect()
            lbl_v, nc_v = ndi.label(visible, structure=np.ones((3, 3), dtype=int))
            rng_l = np.random.default_rng(30)
            b_l = rng_l.integers(0, 5, size=nc_v + 1)
            X_mech_legacy = X_mech_inf.copy()
            for bb in range(5):
                d_unbuf = ndi.distance_transform_edt(~(visible & (b_l[lbl_v] != bb))).astype(np.float32)
                in_comp = (visible & (b_l[lbl_v] == bb))[foot]
                d_sub = d_unbuf[foot][in_comp]
                X_mech_legacy[0, in_comp] = np.log1p(d_sub)
                X_mech_legacy[1, in_comp] = np.minimum(d_sub, 60.0)
                del d_unbuf, d_sub
            del lbl_v
            c_leg1 = _single_block_quick_oof(
                X16, X_mech_legacy, X_mech_inf, all_c, visible_fp, d_vis_fp, quad_fp, boundary_excl_fp, slope_fp,
                seed=3100, include_collar_neg=False
            )
            del X_mech_legacy, X_mech_inf
            p2 = np.zeros((h, w), np.float32); p2[foot] = c_leg1; del c_leg1
            d, st = allocate_dinkelbach_1swap(p2, p2, crest_2d, allow_d3, target_dots=37_654, n_swap_rounds=1)
            _register_arm("Abl_singlehead_legacy_loco", d, st); del d, p2
            gc.collect()
        else:
            del X_mech_tr, X_mech_inf
            gc.collect()

        boot_seed = EH.pooled_summary(terms_seed, draws=2000, seed=520810 + seed, candidate="H77E_full_B37654")
        detached_summary = (
            EH.pooled_summary(terms_far, draws=2000, seed=7700 + seed, candidate="H77E_full_B37654")
            if terms_far else {}
        )
        if detached_summary:
            log(
                f"  [SEED {seed} DETACHED d>=800m ({int(far_mask.sum()):,} px)] "
                f"H77E={detached_summary['scores']['H77E_full_B37654']['dti']:.5f}, "
                f"PurePhys={detached_summary['scores']['H77E_pure_physics_B37654']['dti']:.5f}, "
                f"Random={detached_summary['scores']['A0_random_B37654']['dti']:.5f}"
            )
        seed_results[f"seed_{seed}"] = {
            "seed": seed,
            "withheld_positive_pixels": int(hidden.sum()),
            "arm_details": details_seed,
            "bootstrap_2000_draws": boot_seed,
            "detached_gt800m_evaluation": detached_summary,
        }
        seed_done_json.write_text(json.dumps(seed_results[f"seed_{seed}"], indent=2) + "\n")
        np.savez(
            seed_done_terms,
            **{
                k: terms_seed[k]
                for k in (
                    "H77E_full_B37654",
                    "H77E_tipgated_B37654",
                    "H77E_full_B44090",
                    "H77E_auto_dinkelbach",
                    "H77E_no_1swap_B37654",
                    "H77E_foldprop_B37654",
                    "H77E_stage1_only_B37654",
                    "H77E_greedy_nms_B37654",
                    "H77E_pure_physics_B37654",
                    "A0_h195_foldfair",
                    "A0_random_B37654",
                )
                if k in terms_seed
            },
        )
        log(f"[SEED {seed}] Checkpoint saved -> {seed_done_json.name}")

    if len(only_seeds_arg) < 3:
        log(f"Partial seed run {only_seeds_arg} completed cleanly.")
        return

    pooled_3seed = EH.pooled_summary(
        pooled_terms_all_seeds, draws=2000, seed=520810, candidate="H77E_full_B37654"
    )
    log("=== 3-SEED POOLED SUMMARY (34,235 WITHHELD POSITIVE PIXELS) ===")
    for k, v in pooled_3seed["scores"].items():
        log(f"  {k:<28s}: HOLDOUT-DTI = {v['dti']:.5f}  95% CI [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]")

    # 4-Quadrant Full Blind Province Evaluation (`57,448` positive pixels; zero catalogue inside held-out province!)
    log("=== BLIND 4-QUADRANT PROVINCE HOLDOUT (57,448 POSITIVE PIXELS) ===")
    all_c = np.arange(len(static_names), dtype=np.int32)
    cat_fp = grid.catalogue[foot]
    border_prov = np.zeros((h, w), dtype=bool)
    for q_id in range(4):
        er = ndi.binary_erosion(quad_2d == q_id, iterations=12) & foot
        border_prov |= (quad_2d == q_id) & foot & (~er)
    b_excl_prov = border_prov[foot]

    oof_prov_raw = np.zeros(X16.shape[0], dtype=np.float32)
    for q_id in range(4):
        in_tr = (quad_fp != q_id) & (~b_excl_prov)
        tr_idx, y_tr, sw_tr = _sample_balanced_fold_rows(
            in_tr, cat_fp, np.full(cat_fp.size, 99.0, np.float32), slope_fp, seed=9900 + 31 * q_id, max_neg=250_000
        )
        X_tr = _gather_f32(X16, None, all_c, tr_idx)
        m1, m2 = _fit_fast_dual_lgb(X_tr, y_tr, sw_tr, seed=9950 + 13 * q_id)
        del X_tr, y_tr, sw_tr
        val_idx = np.flatnonzero(quad_fp == q_id)
        preds = np.empty(val_idx.size, dtype=np.float32)
        for s in range(0, val_idx.size, 700_000):
            sl = val_idx[s : s + 700_000]
            xb = _gather_f32(X16, None, all_c, sl)
            preds[s : s + 700_000] = (0.58 * m1.predict_proba(xb)[:, 1] + 0.42 * m2.predict_proba(xb)[:, 1]).astype(np.float32)
            del xb
        oof_prov_raw[val_idx] = preds
        del m1, m2, preds
        gc.collect()

    p_prov_bridged = quantile_bridge_lidar_gap(oof_prov_raw, has_lidar_fp, quad_fp)
    del oof_prov_raw
    delta_prov = compute_filament_autocontext(p_prov_bridged, foot, aux["crest_fp"], has_lidar_fp)
    lg_prov = _safe_logit(p_prov_bridged) + delta_prov
    del p_prov_bridged, delta_prov

    prov_cal = np.zeros(X16.shape[0], dtype=np.float32)
    for q_id in range(4):
        in_tr = (quad_fp != q_id) & (~b_excl_prov)
        cp = np.flatnonzero(in_tr)[::3]
        lr_p = LogisticRegression(C=0.5, max_iter=300).fit(
            np.column_stack([lg_prov[cp], np.maximum(lg_prov[cp], 0.0) ** 2]), cat_fp[cp].astype(np.int32)
        )
        vi = np.flatnonzero(quad_fp == q_id)
        prov_cal[vi] = lr_p.predict_proba(
            np.column_stack([lg_prov[vi], np.maximum(lg_prov[vi], 0.0) ** 2])
        )[:, 1].astype(np.float32)
    del lg_prov
    pc_prov_2d = np.zeros((h, w), dtype=np.float32)
    pc_prov_2d[foot] = prov_cal; del prov_cal

    folds_prov = [
        {
            "fold": q_id,
            "name": HO.FOLD_NAMES[q_id],
            "region": (quad_2d == q_id) & foot,
            "truth": grid.catalogue & (quad_2d == q_id) & foot,
            "visible": grid.catalogue & (quad_2d != q_id) & foot,
            "masked_known": np.zeros((h, w), dtype=bool),
        }
        for q_id in range(4)
    ]
    d_prov_37k, _ = allocate_dinkelbach_1swap(pc_prov_2d, pc_prov_2d, crest_2d, foot, target_dots=37_654)
    _, t_p37 = _score_across_folds(d_prov_37k, folds_prov, foot); del d_prov_37k
    d_prov_44k, _ = allocate_dinkelbach_1swap(pc_prov_2d, pc_prov_2d, crest_2d, foot, target_dots=44_090)
    _, t_p44 = _score_across_folds(d_prov_44k, folds_prov, foot); del d_prov_44k, pc_prov_2d
    d_prov_rnd = np.zeros((h, w), dtype=np.float32)
    d_prov_rnd.ravel()[np.random.default_rng(42).choice(np.flatnonzero(foot.ravel()), size=37_654, replace=False)] = 1.0
    _, t_prn = _score_across_folds(d_prov_rnd, folds_prov, foot); del d_prov_rnd

    prov_boot = EH.pooled_summary(
        {
            "H77E_blind_province_B37654": t_p37,
            "H77E_blind_province_B44090": t_p44,
            "A0_random_province_B37654": t_prn,
        },
        draws=2000,
        seed=520999,
        candidate="H77E_blind_province_B37654",
    )
    for k, v in prov_boot["scores"].items():
        log(f"  [PROVINCE] {k:<28s}: HOLDOUT-DTI = {v['dti']:.5f}  95% CI [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]")

    prior_benchmarks = {
        "H77D_PR12_published": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": "gems57-pooled-hide-v2",
            "seed_30_B37654": 0.15011,
            "seed_20_B37654": 0.14030,
            "mean_2seed": 0.14521,
        },
        "H77_Step_PR10_published": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": "gems57-pooled-hide-v2",
            "seed_20_loqo": 0.15272,
            "blind_4quadrant_province": 0.18583,
        },
        "H77_HSL_PR11_open": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": "gems57-pooled-hide-v2",
            "emitted_submission_seed30": 0.14660,
            "emitted_submission_seed21": 0.14800,
            "emitted_mean_2seed": 0.14730,
            "unconstrained_structural_mean_2seed": 0.17290,
        },
    }

    out = {
        "schema_version": "h77e-exp2-v1",
        "experiment_id": "E2",
        "budget_counter": "2 of 3",
        "lane": "H77E-AUTOCTX-DINKELBACH",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "elapsed_seconds": round(time.time() - t0, 2),
        "pooled_3seed_34235px": pooled_3seed,
        "per_seed_results": seed_results,
        "blind_4quadrant_province_57448px": prov_boot,
        "prior_lane_published_benchmarks": prior_benchmarks,
    }
    out_path = ROOT / "evidence" / "h77e_exp2_holdout.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"[E2] Saved complete holdout benchmark receipt -> {out_path} in {out['elapsed_seconds']}s.")


if __name__ == "__main__":
    main()
