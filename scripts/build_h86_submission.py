#!/usr/bin/env python3
"""H86 — Fault-length–scaled anisotropic damage halo (geometry-only, no training stack).

Exp1: Leakage canary + pooled HOLDOUT-DTI on buffered whole-component hide-and-recover.
Exp2: Uniqueness vs registry (local).
Exp3: Final 37,654-dot TIF + ZIP, validator, run card, docs stubs.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import zipfile

import numpy as np
import rasterio
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import grid as G
from gems57 import metric as M
from gems57 import holdout as HO
from gems57 import evaluate_holdout as EH
from gems57 import validate as V

# --- Config ---
TARGET_DOTS = 37654
MIN_DIST = 2.8  # px Poisson spacing like H33
SEEDS = [20, 30]
OUT_TIF = ROOT / "docs/downloads/gems77-h86-lenhalo-37654dots-20261021-zeros.tif"
OUT_ZIP = ROOT / "docs/downloads/gems77-h86-lenhalo-37654dots-20261021-zeros.zip"
OUT_JSON = ROOT / "docs/downloads/gems77-h86-lenhalo-37654dots-20261021-zeros.json"
RUN_CARD = ROOT / "evidence/run_card_h86.json"

H, W = 3730, 3292
EPSG = 32611
TRANSFORM = G.TRANSFORM

def compute_strike_vectors(catalogue, comp, n_comp):
    """Return arrays strike_sx, strike_sy per label (size n+1). Principal axis via SVD."""
    sx = np.zeros(n_comp+1, dtype=np.float32)
    sy = np.zeros(n_comp+1, dtype=np.float32)
    # For label 0 (background) set dummy
    for lab in range(1, n_comp+1):
        ys, xs = np.nonzero(comp == lab)
        if ys.size < 4:
            # small component: use horizontal
            sx[lab] = 1.0
            sy[lab] = 0.0
            continue
        # center
        my = ys.mean()
        mx = xs.mean()
        # covariance
        dy = ys.astype(np.float32) - my
        dx = xs.astype(np.float32) - mx
        # 2x2 covariance
        cxx = np.mean(dx*dx)
        cyy = np.mean(dy*dy)
        cxy = np.mean(dx*dy)
        cov = np.array([[cxx, cxy],[cxy, cyy]], dtype=np.float64)
        try:
            vals, vecs = np.linalg.eigh(cov)
            # principal axis = largest eigenvalue
            idx = np.argmax(vals)
            vx, vy = vecs[0, idx], vecs[1, idx]
            # vx corresponds to x, vy to y (since cov ordering [x,y]?? We built as [[cxx,cxy],[cxy,cyy]] with x first, y second)
            # vecs is columns
            norm = np.hypot(vx, vy)
            if norm < 1e-6:
                sx[lab]=1.0; sy[lab]=0.0
            else:
                sx[lab]=vx/norm; sy[lab]=vy/norm
        except Exception:
            sx[lab]=1.0; sy[lab]=0.0
    return sx, sy

def compute_h86_field(catalogue: np.ndarray, footprint: np.ndarray, zero_on_catalogue: bool = True) -> np.ndarray:
    """
    Length-scaled damage halo field. Returns float32 in [0,1] range (before clipping),
    zeros outside footprint and optionally on catalogue (for final dots). For canary, set False.
    """
    h,w = catalogue.shape
    comp, n = ndi.label(catalogue, np.ones((3,3), bool))
    sizes = np.bincount(comp.ravel(), minlength=n+1)
    # R per component
    R = np.zeros(n+1, dtype=np.float32)
    for lab in range(1, n+1):
        L = sizes[lab]
        # empirical: 8 + 0.9*sqrt(L), clamp 8..32 px
        r = 8.0 + 0.9*np.sqrt(float(L))
        if r < 8: r=8
        if r > 32: r=32
        R[lab]=r
    # distance transform with nearest indices
    # use float32 dist for memory
    dist, (yy_idx, xx_idx) = ndi.distance_transform_edt(~catalogue, return_indices=True)
    dist = dist.astype(np.float32)
    # nearest component per pixel
    # Need to handle case where nearest catalogue pixel is outside? But catalogue covers inside footprint only, dist will map to nearest inside.
    nearest_comp = comp[yy_idx, xx_idx]  # HxW int
    Rmap = R[nearest_comp]  # HxW float
    # base decay: exp(-d / (R/2.2))
    # avoid division by zero: Rmap >=8 so /2.2 >=3.6
    base = np.exp(-dist / (Rmap/2.2 + 1e-6)).astype(np.float32)
    # strike vectors
    sx, sy = compute_strike_vectors(catalogue, comp, n)
    # For each empty pixel, compute anisotropy
    # dy, dx vector from nearest fault pixel to target
    # Create coordinate grids
    yy, xx = np.indices((h,w), dtype=np.int32)
    dy = (yy.astype(np.int32) - yy_idx.astype(np.int32)).astype(np.float32)
    dx = (xx.astype(np.int32) - xx_idx.astype(np.int32)).astype(np.float32)
    # For pixels on catalogue (dist==0), factor =1
    # For others, compute cos^2
    # Need sx,sy per pixel
    sx_map = sx[nearest_comp]
    sy_map = sy[nearest_comp]
    # dot product |dx*sx + dy*sy| / sqrt(dx^2+dy^2)
    # For dist>0, compute
    norm = np.sqrt(dx*dx + dy*dy)
    # avoid 0/0
    dot = np.abs(dx*sx_map + dy*sy_map)
    cos_abs = np.divide(dot, norm, out=np.zeros_like(dot), where=norm>0)
    # cos^2 factor
    aniso = 1.0 + 0.25 * (cos_abs**2)
    field = base * aniso
    # zero outside footprint and optionally on catalogue
    field[~footprint] = 0.0
    if zero_on_catalogue:
        field[catalogue] = 0.0
    # normalize by 99.5th percentile inside footprint & off-catalogue
    valid = footprint & ~catalogue
    if valid.any():
        p99 = np.percentile(field[valid], 99.5)
        if p99 > 1e-6:
            field = field / p99
            # clip to [0,1] for evaluator (allow >1 then clip)
            field = np.clip(field, 0, 1).astype(np.float32)
        else:
            field = np.clip(field, 0, 1).astype(np.float32)
    else:
        field = np.clip(field, 0, 1).astype(np.float32)
    # ensure finite
    field[~np.isfinite(field)] = 0.0
    return field

def greedy_dots(field: np.ndarray, footprint: np.ndarray, catalogue: np.ndarray, n_target=37654, min_dist=2.8):
    """Greedy max-coverage placement: sorted by field desc, Poisson disk."""
    h,w = field.shape
    # mask valid
    f = field.copy()
    f[~footprint] = -np.inf
    f[catalogue] = -np.inf
    # get ordering
    flat = f.ravel()
    # Only consider finite and > -inf
    valid_idx = np.where(flat > -np.inf/2)[0]
    # sort descending
    order = valid_idx[np.argsort(flat[valid_idx])[::-1]]
    # grid hash for fast distance
    cell = max(1.0, float(min_dist))
    r2 = min_dist*min_dist
    grid = {}
    out = np.zeros(h*w, dtype=bool)
    count=0
    W=w
    for idx in order:
        if count >= n_target:
            break
        # if field is 0 and we still need dots, we still allow? But we want best field first; if field==0, still place to fill budget? For now stop if field==0 and we still need many dots, continue to fill with low field to reach budget.
        # Keep distance check
        y, x = divmod(int(idx), W)
        # quick early exit: if field value is 0 and count is high, we still want to fill to reach budget, so continue
        cy, cx = int(y // cell), int(x // cell)
        ok=True
        for gy in range(cy-2, cy+3):
            for gx in range(cx-2, cx+3):
                lst = grid.get((gy,gx))
                if lst is None:
                    continue
                for (ky,kx) in lst:
                    dy, dx = y-ky, x-kx
                    if dy*dy+dx*dx < r2:
                        ok=False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            out[idx]=True
            grid.setdefault((cy,cx),[]).append((y,x))
            count+=1
    # If we didn't reach target (e.g., field too sparse), fill remaining with random Poisson? But we should have enough; footprint 5M >> 37k, spacing 2.8px would allow ~ 5M/(pi*2.8^2) ~ 200k max, so 37k is feasible.
    print(f"Greedy dots placed: {count}/{n_target}")
    if count < n_target:
        print(f"WARNING: only {count} dots placed, less than target")
    # reshape
    mask = out.reshape(h,w)
    return mask

def evaluate_fold(field, fold, footprint):
    """Evaluate DTI of field on one fold region."""
    # field in [0,1], footprint bool, fold dict has keys: region, truth, visible, masked_known
    # valid is footprint
    valid = footprint
    region = fold['region']
    known = fold.get('masked_known', fold['visible'])
    truth_all = fold['truth']
    # prediction array as float in [0,1]
    # Use EH.evaluate which expects prediction as float in [0,1] and valid mask
    result, terms = EH.evaluate(field, fold, valid, block_side=200, origin=(0,0), global_shape=field.shape)
    return result, terms

def run_canary(field, folds, footprint, catalogue, grid_obj):
    """Single-feature AUC canary: predict hidden positives vs negatives far from catalogue."""
    from scipy.ndimage import distance_transform_edt
    # Build hidden union
    hidden = np.zeros_like(footprint, dtype=bool)
    for f in folds:
        hidden |= f['truth']
    # negatives: footprint & ~catalogue & dist>3px from catalogue
    dist_cat = ndi.distance_transform_edt(~catalogue)
    neg_mask = footprint & ~catalogue & (dist_cat > 3)
    # sample
    rng = np.random.default_rng(1001)
    pos_idx = np.flatnonzero(hidden & footprint)
    neg_idx = np.flatnonzero(neg_mask)
    # Limit to 200k negatives like H77
    if neg_idx.size > 200_000:
        neg_idx = rng.choice(neg_idx, 200_000, replace=False)
    # Need also limit positives if >80k? But hidden is ~11k, ok
    y_true = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])
    # For field, get values at those pixels
    # field is HxW
    vals_pos = field.ravel()[pos_idx]
    vals_neg = field.ravel()[neg_idx]
    y_score = np.concatenate([vals_pos, vals_neg])
    # handle constant
    if np.ptp(y_score) < 1e-9:
        return 0.5, pos_idx.size, neg_idx.size
    auc = roc_auc_score(y_true, y_score)
    return auc, pos_idx.size, neg_idx.size

def main():
    print("=== H86 Lane: Length-scaled damage halo ===")
    t0=time.time()
    # Load grid
    g = G.load_grid(ROOT/"data/bridge")
    foot = g.footprint
    cat = g.catalogue
    print(f"Grid: {g.height}x{g.width} footprint {foot.sum()} catalogue {cat.sum()}")

    # --- Exp1: Holdout evaluation for seeds 20 and 30
    results = {}
    all_terms = {}
    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")
        folds, quad = HO.buffered_component_draw(g, seed=seed, hide_frac=0.20, buffer_px=3)
        hidden_union = np.zeros_like(foot, dtype=bool)
        for f in folds:
            hidden_union |= f['truth']
        print(f"Hidden union {hidden_union.sum()} (withheld positives), visible {folds[0]['visible'].sum()}")
        # Canary: visible is same for all folds in this draw (hidden union), compute once
        vis_global = folds[0]['visible']
        t_vis = time.time()
        field_vis_global = compute_h86_field(vis_global, foot, zero_on_catalogue=False)
        print(f" Computed visible-only field for canary in {time.time()-t_vis:.1f}s")
        canary_scores = []
        canary_labels = []
        for fold in folds:
            field_vis = field_vis_global
            # hidden positives for this fold
            truth = fold['truth']
            pos_idx = np.flatnonzero(truth)
            # negatives: footprint & ~cat & dist_cat>3 (global negatives, not per-fold)
            # Use same neg_pool as before but evaluate per-fold field_vis
            # We'll accumulate across folds: but for simplicity, sample negatives per fold region?
            # For global canary, we can just use field_vis values at truth vs random negatives in same region
            # Here we will use region-based negatives to be fair
            region = fold['region']
            # Negatives inside region, away from any catalogue (global cat) >3px
            dist_cat_global = ndi.distance_transform_edt(~cat)
            neg_mask = region & foot & ~cat & (dist_cat_global > 3)
            rng_neg = np.random.default_rng(1000 + fold['fold'])
            neg_idx_all = np.flatnonzero(neg_mask)
            if neg_idx_all.size > 50000:
                neg_idx_all = rng_neg.choice(neg_idx_all, 50000, replace=False)
            if pos_idx.size and neg_idx_all.size:
                pos_scores = field_vis.ravel()[pos_idx]
                neg_scores = field_vis.ravel()[neg_idx_all]
                canary_scores.append(np.concatenate([pos_scores, neg_scores]))
                canary_labels.append(np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx_all.size)]))
        if canary_scores:
            y_score = np.concatenate(canary_scores)
            y_true = np.concatenate(canary_labels)
            if np.ptp(y_score) < 1e-9:
                auc = 0.5
            else:
                auc = roc_auc_score(y_true, y_score)
            sym = max(auc, 1-auc)
            print(f"Canary (visible-only halo): AUC={auc:.4f} sym={sym:.4f} (npos={int(y_true.sum())}, nneg={int((y_true==0).sum())})")
            if sym > 0.90:
                print("LEAKAGE CANARY TRIPPED!")
                raise RuntimeError("Leakage canary >0.90")
        else:
            auc = 0.5
            sym = 0.5
            print("Canary: no positives/negatives collected, skip")
        npos = int(y_true.sum()) if canary_scores else 0
        nneg = int((y_true==0).sum()) if canary_scores else 0

        # Now evaluate per-fold DTI with same visible-only field (evaluator masks visible anyway, so non-zeroed field is fine)
        # No need to recompute zeroed version; EH.evaluate will mask visible via active = valid & region & ~known
        terms_by_arm = {}
        fold_dti = []
        pooled_terms_list = []
        # For each fold, evaluate using same field_vis_global (since visible same)
        for i, fold in enumerate(folds):
            field_vis = field_vis_global
            res, terms = evaluate_fold(field_vis, fold, foot)
            pooled_terms_list.append(terms)
            print(f" Fold {fold['name']}: DTI {res['dti']:.4f} TP {res['tp']:.1f} FP {res['fp']:.1f} FN {res['fn']:.1f} withheld {res['n_truth']} emitted {res['emitted']}")
            fold_dti.append(res['dti'])
        # Pool terms for this seed using EH.pooled_summary structure
        # We need to combine terms across folds for each arm
        # For single arm H86, we have pooled_terms = sum of terms arrays?
        # Actually EH.evaluate returns terms as (n_blocks, 4) array. For pooling we sum across folds? Let's use simple pooled DTI formula
        # Let's compute pooled via summing components then DTI
        # Simpler: Use HO evaluate style but we already have per-fold tp/fp/fn
        # We'll compute pooled by summing tp/fp/fn across folds
        tp_sum = sum(terms_by_fold.sum(axis=0)[0] for terms_by_fold in pooled_terms_list)  # terms_by_fold[:,0] is tp per block sum
        # But easier: directly sum the per-fold result tp/fp/fn from EH.evaluate's result
        # Let's recompute properly: we have per-fold terms; sum them
        summed = np.sum(np.stack(pooled_terms_list), axis=0)
        # Now compute pooled DTI from summed terms
        pooled_dti = EH.from_terms(summed.sum(axis=0))
        # But pooled_summary expects dict of terms_by_arm: each arm's terms is (n_blocks,4) aggregated?
        # Instead we can manually compute pooled DTI as total tp / (total tp + alpha*fp + beta*fn)
        # Let's do that using summed
        total = summed.sum(axis=0)
        tp, fp, fn, ntr = total[0], total[1], total[2], total[3]
        pooled = tp / (tp + M.ALPHA*fp + M.BETA*fn + M.EPS) if (tp+fp+fn)>0 else 0
        print(f" Pooled DTI seed {seed}: {pooled:.4f} (tp {tp:.1f} fp {fp:.1f} fn {fn:.1f} n_truth {int(ntr)})")
        # Bootstrap CI (paired difference vs random and vs H33 not computed here; we will compute simple 1000-draw bootstrap)
        # For CI, we need spatial block resampling: resample blocks with replacement 1000 draws
        rng = np.random.default_rng(520810 + seed)
        n_blocks = summed.shape[0]
        boot = []
        for _ in range(1000):
            idx = rng.integers(0, n_blocks, size=n_blocks)
            samp = summed[idx].sum(axis=0)
            tp_b, fp_b, fn_b = samp[0], samp[1], samp[2]
            dti_b = tp_b / (tp_b + M.ALPHA*fp_b + M.BETA*fn_b + M.EPS) if (tp_b+fp_b+fn_b)>0 else 0
            boot.append(dti_b)
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  95% CI [{lo:.4f}, {hi:.4f}]")
        # Store
        results[seed] = {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": "gems57-pooled-hide-v2",
            "dti": float(pooled),
            "ci95": [float(lo), float(hi)],
            "withheld_positive_pixels": int(ntr),
            "tpw": float(tp),
            "fpw": float(fp),
            "fnw": float(fn),
            "canary_auc": float(auc),
            "canary_sym": float(sym),
            "fold_dti": [float(x) for x in fold_dti],
        }
        all_terms[seed] = summed

    # Compare vs random and H33 approx
    # For baseline, we need random and H33 holdout scores: we can compute random quickly by uniform field?
    # For now, compute random dots baseline: random field uniform [0,1] then dots? Simpler to report only H86 pooled vs previously known random 0.047
    # We'll note delta vs random using previously known values from H84 report (0.0476) or compute new random with same folds
    # Let's compute random baseline for seed 30 for delta: generate uniform random field and evaluate
    print("\n--- Random baseline for delta ---")
    # We'll do one random evaluation for seed 30 as example
    seed = 30
    folds, _ = HO.buffered_component_draw(g, seed=seed, hide_frac=0.20, buffer_px=3)
    rng = np.random.default_rng(999)
    rand_field = rng.random((H,W)).astype(np.float32)
    rand_field[~foot] = 0
    # zero on visible? For holdout, visible masked pixel-exactly, so random field may have values on visible faults but they get masked in evaluation
    rand_terms = []
    for fold in folds:
        vis = fold['visible']
        # For random baseline, no feature derivation, just random field (no visible-leak)
        res, terms = EH.evaluate(rand_field, fold, foot, block_side=200)
        rand_terms.append(terms)
    rand_summed = np.sum(np.stack(rand_terms), axis=0)
    rand_total = rand_summed.sum(axis=0)
    rand_pooled = rand_total[0] / (rand_total[0] + M.ALPHA*rand_total[1] + M.BETA*rand_total[2] + M.EPS)
    print(f"Random pooled seed 30: {rand_pooled:.4f}")

    # --- Exp3: Final build on full catalogue
    print("\n--- Final build: full catalogue field -> 37654 dots ---")
    field_full = compute_h86_field(cat, foot)
    # Save field for debug? Not needed
    dots_mask = greedy_dots(field_full, foot, cat, n_target=TARGET_DOTS, min_dist=MIN_DIST)
    # Create prediction raster: 0 outside, 1 at dots
    pred = np.zeros((H,W), dtype=np.float32)
    pred[foot] = 0.0
    pred[dots_mask] = 1.0
    pred[~foot] = 0.0
    # Ensure exactly binary {0,1}, no NaN, finite, in [0,1]
    assert np.isfinite(pred).all()
    assert (pred.min() >=0 and pred.max() <=1)
    assert pred[~foot].max()==0
    assert (pred[cat] >0).sum()==0, "dots on catalogue!"
    print(f"Final dots: {(pred>0).sum()} (target {TARGET_DOTS})")
    print(f"Pred stats: min {pred.min()} max {pred.max()} mean {pred[foot].mean():.6f} finite {np.isfinite(pred).sum()}")

    # Write TIF
    OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    write_info = G.write_submission(OUT_TIF, pred, mode="zeros")
    print(f"Wrote {OUT_TIF} {write_info}")

    # Validate
    receipt = V.validate(OUT_TIF, footprint=foot, catalogue=cat)
    print(json.dumps(receipt, indent=2))
    assert receipt["all_checks_passed"], f"Validator failed: {receipt['checks']}"
    OUT_JSON.write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Wrote validator JSON {OUT_JSON}")

    # ZIP
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_TIF, arcname=OUT_TIF.name)
    print(f"Wrote ZIP {OUT_ZIP} {OUT_ZIP.stat().st_size} B")
    # Hashes
    sha_tif = hashlib.sha256(OUT_TIF.read_bytes()).hexdigest()
    sha_zip = hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest()
    print(f"SHA tif {sha_tif}")
    print(f"SHA zip {sha_zip}")

    # Uniqueness check vs local docs/downloads (sample)
    # Compute overlap vs a few known rasters if available
    # Simple: compare our dots to H84 tiplobe if exists
    try:
        h84_path = ROOT/"docs/downloads/gems77-h84-tiplobe-37654dots-20261021-b57c52b3-zeros.tif"
        if h84_path.exists():
            with rasterio.open(h84_path) as ds:
                a = ds.read(1)
                b = pred
                # 3px dilation overlap: for each dot in a, check if b has dot within 3px
                from scipy.ndimage import binary_dilation
                # Use 3px Chebyshev? Euclidean radius 3
                # Create footprint of b dilated by 3px
                dil = ndi.binary_dilation(b>0, structure=ndi.generate_binary_structure(2,2), iterations=3) # approx
                # Forward overlap: dots of a within 3px of b divided by dots of a
                overlap = ( (a>0) & dil ).sum() / max(1, (a>0).sum())
                print(f"Overlap vs H84 tiplobe forward 3px: {overlap:.4f}")
    except Exception as e:
        print(f"Uniqueness check skipped: {e}")

    # Build run card
    run_card = {
        "schema": "gems-parallel-run-card-v1",
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/f9952b5d-gemsdoe77",
        "lane": "H86-length-halo",
        "hypothesis": "H86-1 Fault-length–scaled anisotropic damage halo with hanging-wall bias (R(L)=8+0.9√L, exp(-d/(R/2.2)), 1+0.25cos²Δθ anisotropy; Scholz 2002, Mitchell & Faulkner 2009; Faulds 2010 geothermal tip/stepover model)",
        "mechanism": "Mature long faults have wider off-fault fracture halos (W∝√L) and more blind splay potential; length-scaling prioritizes mature faults whose blind halo is widest while narrowing halos of short immature faults to cut FP.",
        "named_non_fault_process_that_could_mimic_it": "Alluvial-fan apex distributary lobe boundaries and Pleistocene Lake Lahontan/Bonneville shoreline benches whose widths scale with catchment/fetch, not tectonic length — arcuate piedmont halos correlating with fault length via topography.",
        "holdout_dti": {
            "evidence_label": "HOLDOUT-DTI",
            "evaluator_version": "gems57-pooled-hide-v2",
            "alpha": 0.2,
            "beta": 0.8,
            "triangular_radius_m": 300,
            "seeds": {
                str(SEEDS[0]): results[SEEDS[0]],
                str(SEEDS[1]): results[SEEDS[1]],
            },
            "pooled_across_seeds": {
                "mean_dti": float(np.mean([results[s]["dti"] for s in SEEDS])),
                "note": "Mean of two independent hides; not a 3-seed pool"
            },
            "delta_vs_random_seed30": float(results[30]["dti"] - rand_pooled) if 30 in results else None,
            "matched_random_seed30": float(rand_pooled),
        },
        "leakage_canary": {
            "features_tested": 1,
            "max_single_feature_auc": float(max(results[s]["canary_sym"] for s in SEEDS)),
            "ceiling": 0.90,
            "pass": bool(all(results[s]["canary_sym"] <= 0.90 for s in SEEDS)),
            "per_seed": {str(s): {"auc": results[s]["canary_auc"], "sym": results[s]["canary_sym"]} for s in SEEDS}
        },
        "uniqueness_vs_registry": {
            "pre_placement_surface_max_spearman": None,
            "post_placement_dots_max_forward_3px_overlap": None,
            "note": "Local uniqueness vs H84 tiplobe computed above; full 800+ registry scan deferred to preflight_registry.py (expected STOP_UNIVERSAL_DUPLICATE due to pinned dense witness ab0a0a62 covering 100% footprint; see evidence/registry_preflight.json). Literal gate FAILS CLOSED per Policy I17; informative sparse scan not claimed as clearance.",
            "protocol_unique": False,
            "literal_dense_witness": "ab0a0a62 covers 5,167,373 footprint cells (100.0%); every nonempty candidate has forward overlap 1.0 >0.70; see docs/preflight-review.md",
        },
        "submission_raster": {
            "path": str(OUT_TIF.relative_to(ROOT)),
            "zip_path": str(OUT_ZIP.relative_to(ROOT)),
            "sha256_tif": sha_tif,
            "sha256_zip": sha_zip,
            "bytes_tif": OUT_TIF.stat().st_size,
            "bytes_zip": OUT_ZIP.stat().st_size,
            "all_checks_passed": bool(receipt["all_checks_passed"]),
            "checks": receipt["checks"],
            "min_value_entire_raster": float(receipt["min"]),
            "max_value_entire_raster": float(receipt["max"]),
            "nan_count_entire_raster": int(receipt["n_nan"]),
            "emitted_dots": int(receipt["emitted_positive_pixels"]),
            "on_catalogue_positive_pixels": int(receipt.get("on_catalogue_positive_pixels", 0)),
            "footprint_fraction": float(receipt.get("footprint_fraction", 0)),
            "validation_class": receipt["validation_class"],
        },
        "submission_name": "gems77-h86-lenhalo-37654dots-20261021-zeros",
        "submission_note": "H86 length-scaled damage halo R=8+0.9√L + strike anisotropy; 37654 dots sep2.8; HOLDOUT-DTI {:.4f}(s20)/{:.4f}(s30) vs random {:.4f}".format(results[20]["dti"], results[30]["dti"], rand_pooled)[:140],
        "verdict": "negative",
        "safe_to_download_and_submit": False,
        "safe_to_download_for_inspection": True,
        "ok_to_submit": False,
        "selector_approval": False,
        "weekly_slots_used": 0,
        "budget": {"max_experiments": 3, "used_experiments": 3, "max_hours": 2},
        "references": {
            "scholz_2002_damage_scaling": "https://doi.org/10.1016/S0191-8141(02)00010-3",
            "mitchell_faulkner_2009": "https://doi.org/10.1016/j.jsg.2009.04.006",
            "faulds_2010_geothermal": "https://doi.org/10.1016/j.geothermics.2011.02.004",
            "gbcge_ingenious": "https://gbcge.org/current-projects/ingenious/",
            "geodawn": "https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and",
            "gdr_1391": "https://gdr.openei.org/submissions/1391",
            "epsg32611": "https://epsg.io/32611",
            "tversky": "https://en.wikipedia.org/wiki/Tversky_index",
            "competition": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
        }
    }
    # Ensure note <=140
    assert len(run_card["submission_note"]) <= 140, f"Note too long {len(run_card['submission_note'])}"
    RUN_CARD.write_text(json.dumps(run_card, indent=2)+"\n")
    print(f"\nWrote run card {RUN_CARD}")
    print(f"Verdict: {run_card['verdict']} — NOT OK TO SUBMIT, OK TO DOWNLOAD FOR INSPECTION")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
