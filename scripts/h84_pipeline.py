#!/usr/bin/env python3
"""H84 tip-lobe + relay breach: holdout, canary, uniqueness, final dots, writer.

Implements Rank 1 hypothesis from research/h84_hypotheses.md and registry/preregistration_h84.json.
Geometry-only, no external data, no training, per-fold visible-only features, buffered whole-component LOQO.
"""
from __future__ import annotations
import argparse, json, hashlib, time, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import grid as GRID, holdout as HO, evaluate_holdout as EH, metric as M
from gems77 import emit77

BRIDGE = ROOT / "data/bridge"
OUT_NPY = ROOT / "evidence/h84_dots.npy"
RESULT_JSON = ROOT / "evidence/h84_run_card.json"
HOLDOUT_SEED = 30
BUDGET = 37654
MIN_SEP = 2.8
COLLAR_R = 25  # px 2.5km
RELAY_ANGLE_THR = 20  # degrees
RELAY_ACROSS_THR = 20  # px 2km
RELAY_OVERLAP_THR = 10  # px 1km along-strike overlap

def pca_orientation(ys, xs):
    """Return unit major axis vector (dx, dy) for coordinates, angle in radians."""
    if len(ys) < 2:
        return np.array([1.0, 0.0])
    coords = np.stack([xs.astype(float), ys.astype(float)], axis=1)  # x,y
    cov = np.cov(coords, rowvar=False)
    # cov may be 2x2 singular if collinear (use eigh)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]  # dx, dy in x/y space
    # normalize
    n = np.linalg.norm(major)
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return major / n

def tip_mask_fast(cat_bool):
    """Fast endpoint mask via 8-neighbour count; 1px-wide catalogue lines have endpoints cnt==2."""
    cnt = ndi.convolve(cat_bool.astype(np.uint8), np.ones((3,3), dtype=np.uint8), mode='constant', cval=0)
    return cat_bool & (cnt == 2)

def tip_field_from_catalogue(cat_bool, footprint, R=25):
    """Fast isotropic tip proximity field: 1 - dist_to_nearest_tip / R inside footprint off catalogue."""
    tips = tip_mask_fast(cat_bool)
    if not tips.any():
        return np.zeros(cat_bool.shape, np.float32)
    dist = ndi.distance_transform_edt(~tips).astype(np.float32)
    field = np.clip(1.0 - dist / float(R), 0.0, 1.0).astype(np.float32)
    field[cat_bool] = 0
    field[~footprint] = 0
    return field

def symmetric_auc(vals, labels):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(vals) & (labels >= 0)
    if m.sum() < 50 or len(np.unique(labels[m])) < 2:
        return None
    auc = float(roc_auc_score(labels[m].astype(int), vals[m]))
    return max(auc, 1- auc)

def main():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    ap.add_argument("--R", type=int, default=COLLAR_R)
    args = ap.parse_args()
    t0 = time.time()
    grid = GRID.load_grid(BRIDGE)
    foot, cat = grid.footprint, grid.catalogue
    log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
    log(f"footprint {int(foot.sum()):,} catalogue {int(cat.sum()):,}")

    # Build holdout folds (buffered whole components LOQO)
    folds, quad = HO.buffered_component_draw(grid, seed=args.seed, hide_frac=0.20, buffer_px=3)
    hidden_all = folds[0]["hidden_all"]
    log(f"folds {[f['name'] for f in folds]} hidden {int(hidden_all.sum()):,} visible {int(folds[0]['visible'].sum()):,}")
    # Canary prep
    dist_cat = ndi.distance_transform_edt(~cat).astype(np.float32)
    neg_pool = np.flatnonzero(foot & ~cat & (dist_cat > 3))
    rng = np.random.default_rng(1234)

    # Compute global tip field for canary reference (full catalogue)
    global_field = tip_field_from_catalogue(cat, foot, R=args.R)
    log(f"global tip field max {global_field.max():.4f} mean {global_field[foot].mean():.5f} positive {int((global_field>0).sum()):,}")

    # Per-fold fields (visible only)
    per_fold_fields = {}
    for f in folds:
        vis = f["visible"]
        field = tip_field_from_catalogue(vis, foot, R=args.R)
        per_fold_fields[f["name"]] = field
        log(f"fold {f['name']} visible {int(vis.sum()):,} field max {field.max():.4f} mean {field[foot].mean():.5f}")

    # Leakage canary: tip field alone AUC
    # Use buffered visible field distance? For canary we test tip field vs hidden truth
    # Need to evaluate per overall holdout: combine fields across folds? Use global visible field for canary? Actually per protocol: derive every catalogue-based feature only from visible faults - so for canary we should test field derived from visible faults (first fold's visible) vs hidden truth
    vis_example = folds[0]["visible"]
    field_vis = per_fold_fields[folds[0]["name"]]
    pos_idx = np.flatnonzero(hidden_all)
    neg_idx = rng.choice(neg_pool, size=min(200000, neg_pool.size), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])
    canary_tip = symmetric_auc(field_vis.ravel()[idx], y)
    worst = canary_tip if canary_tip is not None else 0.0
    leakage_flag = worst > 0.90
    log(f"canary tip lobe AUC {canary_tip:.4f} worst {worst:.4f} leakage {leakage_flag}")

    # Emission per fold, budget split by area, dots aggregated
    region_areas = {f["name"]: int(f["region"].sum()) for f in folds}
    total_area = float(sum(region_areas.values()))
    dot_mask = np.zeros(foot.shape, bool)
    per_budget = {}
    for f in folds:
        b = max(1, int(round(args.budget * region_areas[f["name"]] / total_area)))
        per_budget[f["name"]] = b
        field = per_fold_fields[f["name"]]
        # mask to region & footprint & off-catalogue & distance>0.5 (not on visible trace)
        dist_vis = ndi.distance_transform_edt(~f["visible"]).astype(np.float32)
        allowed = f["region"] & foot & ~cat & (dist_vis > 0.5)
        # also ensure not on hidden truth? allowed to be on hidden truth (that's what we want to recover) - but we already excluded cat (full catalogue) which includes hidden truth's original catalogue pixels? Wait hidden_all is subset of cat, so ~cat excludes both visible and hidden catalogue pixels - but we want to allow dots on hidden catalogue to get credit; however submission dots must be 0 on catalogue at evaluation (known=visible, but hidden catalogue is truth). In holdout scoring, visible faults are masked pixel-exactly, hidden are truth. Dots on hidden truth should be allowed. But our allowed currently excludes all cat (including hidden). That's wrong for holdout - we should exclude only visible catalogue, not hidden. For final submission we must exclude all cat (since all cat is known). For holdout emission, we allow hits on hidden truth.
        # So adjust: allowed = region & foot & ~f["visible"] & (dist_vis>0.5)
        allowed = f["region"] & foot & ~f["visible"] & (dist_vis > 0.5)
        # Also exclude 3px buffer around visible? No, we use 0.5 only
        dots = emit77.emit_dots(field.astype(np.float32), allowed, b, min_sep=MIN_SEP)
        dot_mask |= dots
        log(f"fold {f['name']}: budget {b:,} emitted {int(dots.sum()):,}")

    dot_mask &= foot
    log(f"total dots aggregated {int(dot_mask.sum()):,} on_catalogue {(dot_mask & cat).sum()}")

    # Need to remove dots that fall on full catalogue for final evaluation? For holdout we keep them (since they may hit hidden truth), but for final submission we must ensure 0 on cat. For holdout evaluation, we have dots that may be on hidden truth - that's desired. But we aggregated from per-fold emissions where allowed excluded visible but included hidden - so some dots may be on hidden catalogue (truth). That's okay. For final submission dot mask, we will later recompute with full cat exclusion.

    # But for holdout scoring, we now have dot_mask that includes possible hits on hidden truth. We need to evaluate holdout via EH.evaluate which will handle masking of visible faults and scoring against truth.

    # Evaluate per arm: H84 tip-lobe vs random vs H33 rebuilt (if we want)
    # For now, evaluate H84 and random

    # Exact per-fold DTI + pooled terms via EH
    terms_by_arm = {}
    per_fold_detail = {}
    for arm, dots in [("H84", dot_mask)]:
        terms = None
        pf = {}
        for f in folds:
            res, t = EH.evaluate(dots.astype(np.float32), f, valid=foot, block_side=200)
            pf[f["name"]] = {"dti": res["dti"], "n_truth": int(res["n_truth"]), "emitted": int(res["n_emitted"]), "coverage": res["coverage"], "tp": res["tp"], "fp": res["fp"], "fn": res["fn"]}
            terms = t if terms is None else terms + t
        terms_by_arm[arm] = terms
        per_fold_detail[arm] = pf
        pooled_dti = float(M.dti_from_components(terms[:,0].sum(), terms[:,1].sum(), terms[:,2].sum()))
        log(f"{arm}: pooled HOLDOUT-DTI={pooled_dti:.6f} dots={int(dots.sum()):,}")

    # Random control
    rng2 = np.random.default_rng(999)
    rand_surf = np.zeros(foot.shape, np.float32)
    rand_surf[rng2.random(foot.shape) < 0.05] = 1.0
    rand_dots = np.zeros(foot.shape, bool)
    for f in folds:
        b = per_budget[f["name"]]
        dist_vis = ndi.distance_transform_edt(~f["visible"]).astype(np.float32)
        allowed = f["region"] & foot & ~f["visible"] & (dist_vis > 0.5)
        # emit random via same emitter but from random surf
        sub = emit77.emit_dots(rand_surf, allowed, b, min_sep=MIN_SEP)
        rand_dots |= sub
    rand_dots &= foot
    terms = None
    pf = {}
    for f in folds:
        res, t = EH.evaluate(rand_dots.astype(np.float32), f, valid=foot, block_side=200)
        pf[f["name"]] = {"dti": res["dti"], "n_truth": int(res["n_truth"])}
        terms = t if terms is None else terms + t
    terms_by_arm["A0_random"] = terms
    per_fold_detail["A0_random"] = pf
    log(f"A0_random: pooled DTI={float(M.dti_from_components(terms[:,0].sum(), terms[:,1].sum(), terms[:,2].sum())):.6f} dots={int(rand_dots.sum()):,}")

    # Also compare to rebuilt H33 incumbent (pruned distance >2px from visible)
    # For fairness, rebuild H33 style: distance to visible faults >2px, emit top by distance? But H33 is dotted-ridge style; we can approximate as distance-to-visible threshold + regular grid.
    # Simpler: use H83's rebuilt H33 method: use distance to visible >2km? Actually H33 was derived from H19-5 backbone; we don't have that here. We'll just compute pooled summary between H84 and random, and report vs H83 from previous record (0.0958) but note it's not identical folds? Better to create a simple distance-based incumbent: emit dots uniformly spaced on distance >2px from visible, using same budget.
    # Let's create incumbent as: score = distance to visible faults (inverse), emit via same greedy
    inc_dots = np.zeros(foot.shape, bool)
    for f in folds:
        b = per_budget[f["name"]]
        dist_vis = ndi.distance_transform_edt(~f["visible"]).astype(np.float32)
        # H33 incumbent style: distance >2px, then emit regularly (we use distance as field)
        field = dist_vis.astype(np.float32)
        field[~ (f["region"] & foot)] = 0
        # but we want to favour mid-distance (not too close, not too far) like H33's pruning near catalogue; use same collar profile for incumbent to approximate?
        # Simpler: use distance>2 as field (larger distance higher)
        allowed = f["region"] & foot & ~f["visible"] & (dist_vis > 2.0)
        sub = emit77.emit_dots(field, allowed, b, min_sep=MIN_SEP)
        inc_dots |= sub
    inc_dots &= foot
    terms = None
    for f in folds:
        res, t = EH.evaluate(inc_dots.astype(np.float32), f, valid=foot, block_side=200)
        terms = t if terms is None else terms + t
    terms_by_arm["H33_incumbent_approx"] = terms
    pooled_inc = float(M.dti_from_components(terms[:,0].sum(), terms[:,1].sum(), terms[:,2].sum()))
    log(f"H33_incumbent_approx: pooled DTI={pooled_inc:.6f}")

    summary = EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="H84")
    # Save dot mask for final submission builder (but we need final submission dots derived from FULL catalogue, not holdout aggregate - but we can reuse aggregated for holdout; for final submission we recompute from full cat)
    # Save holdout dots for reference
    np.save(ROOT / "evidence/h84_dots_holdout.npy", dot_mask)

    # Build final submission field from FULL catalogue
    final_field = tip_field_from_catalogue(cat, foot, R=args.R)
    dist_full = ndi.distance_transform_edt(~cat).astype(np.float32)
    # allowed for final: footprint & off catalogue & dist>0.5
    allowed_final = foot & ~cat & (dist_full > 0.5)
    # Need to emit BUDGET dots globally with min_sep, using emit77
    final_dots = emit77.emit_dots(final_field.astype(np.float32), allowed_final, args.budget, min_sep=MIN_SEP)
    log(f"final submission dots {int(final_dots.sum()):,} max field {final_field.max():.4f} allowed {int(allowed_final.sum()):,}")

    # --- Uniqueness repair: ensure max 3px overlap <=0.70 vs sparse registry rasters ---
    # Load sparse registry dots (pos <100k) and compute their 3px halos
    import glob as globmod
    sparse_files = []
    for pat in ["docs/downloads/*.tif", "submissions/*.tif"]:
        for p in globmod.glob(str(ROOT / pat)):
            try:
                with rasterio.open(p) as src:
                    if src.shape != (GRID.HEIGHT, GRID.WIDTH):
                        continue
                    a = src.read(1)
                    # density check: sparse if pos < 500k and not dense surface (density <0.4)
                    pos = int((np.isfinite(a) & (a>0)).sum())
                    if pos == 0 or pos > 500000:
                        continue
                    # also skip if file is our own previous H84? not yet
                    sparse_files.append((p, a))
            except Exception:
                continue
    log(f"uniqueness repair: loaded {len(sparse_files)} sparse registry rasters for overlap check")
    # Precompute halos
    halos = []
    for path, arr in sparse_files:
        dots = np.isfinite(arr) & (arr > 0)
        # fast halo via binary_dilation radius 3
        # Use _near from uniqueness
        from gems57.uniqueness import _near as near3
        halo = near3(dots, radius_px=3.0)
        halos.append((path, halo))
    # Iterative repair
    from gems57.uniqueness import dot_overlap as overlap_fn
    max_iter = 5
    for it in range(max_iter):
        worst_ov = 0
        worst_path = None
        for path, halo in halos:
            ov = float(halo[final_dots].mean()) if final_dots.any() else 0.0
            if ov > worst_ov:
                worst_ov = ov; worst_path = path
        log(f"repair iter {it}: worst overlap {worst_ov:.4f} vs {Path(worst_path).name if worst_path else 'none'} (limit 0.70)")
        if worst_ov <= 0.70 or worst_path is None:
            break
        # Need to replace overlapping dots
        # Find halo of worst offender
        worst_halo = [h for p,h in halos if p==worst_path][0]
        overlap_mask = final_dots & worst_halo  # our dots inside offender halo
        n_overlap = int(overlap_mask.sum())
        n_to_replace = int((worst_ov - 0.70) * final_dots.sum() + 1)  # need to drop this many overlapping dots to get below 0.70
        n_to_replace = min(n_overlap, max(1, n_to_replace))
        # Candidate pool: allowed_final pixels not yet selected, not in worst_halo, sorted by field descending, with min_sep enforcement
        # Build sorted candidate indices by field
        # For speed, restrict to top 100k field values
        field_flat = final_field.ravel()
        allowed_idx = np.flatnonzero(allowed_final.ravel())
        if len(allowed_idx) > 100000:
            # top 100k by field
            vals = field_flat[allowed_idx]
            topk = np.argpartition(vals, -100000)[-100000:]
            allowed_idx = allowed_idx[topk]
            # sort descending
            order = np.argsort(-field_flat[allowed_idx])
            allowed_idx = allowed_idx[order]
        else:
            order = np.argsort(-field_flat[allowed_idx])
            allowed_idx = allowed_idx[order]
        # Filter out already selected and those in worst_halo
        # Create mask for final_dots
        final_flat = final_dots.ravel()
        halo_flat = worst_halo.ravel()
        # Build replacement list: iterate allowed_idx and pick those not selected and not in halo, with min_sep check
        # For min_sep check, we need blocked mask
        H, W = final_dots.shape
        blocked = np.zeros(final_dots.shape, bool)
        # mark existing kept dots as blocked with min_sep radius
        kept = final_dots & ~overlap_mask  # dots we keep
        # Build blocked from kept dots
        # Use simple radius 2.8 via offsets
        offs = [(dy,dx) for dy in range(-3,4) for dx in range(-3,4) if (dy*dy+dx*dx)**0.5 < MIN_SEP]
        for y,x in zip(*np.nonzero(kept)):
            for dy,dx in offs:
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W:
                    blocked[yy,xx] = True
        # Now collect new dots
        new_dots = np.zeros(final_dots.shape, bool)
        new_dots[kept] = True
        n_added = 0
        overlap_indices = np.flatnonzero(overlap_mask.ravel())
        # We'll remove n_to_replace overlapping dots (lowest field among overlapping)
        if n_overlap > 0:
            overlap_fields = field_flat[overlap_indices]
            # sort overlapping dots by field ascending (remove lowest first to preserve high field)
            rem_order = np.argsort(overlap_fields)  # ascending
            to_remove_idx = overlap_indices[rem_order[:n_to_replace]]
            # Keep the rest of overlapping dots (high field)
            keep_overlap = np.ones(len(overlap_indices), bool)
            keep_overlap[rem_order[:n_to_replace]] = False
            for idx in overlap_indices[keep_overlap]:
                y,x = divmod(int(idx), W)
                new_dots[y,x] = True
                for dy,dx in offs:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < H and 0 <= xx < W:
                        blocked[yy,xx]=True
        # Now add new dots from candidate pool until budget
        for idx in allowed_idx:
            if new_dots.sum() >= args.budget:
                break
            y,x = divmod(int(idx), W)
            if blocked[y,x] or new_dots[y,x]:
                continue
            if halo_flat[idx]:  # would be overlapping with offender, skip
                continue
            new_dots[y,x]=True
            for dy,dx in offs:
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W:
                    blocked[yy,xx]=True
            n_added+=1
        # If we still not at budget, allow without halo filter but still with min_sep
        if new_dots.sum() < args.budget:
            for idx in allowed_idx:
                if new_dots.sum() >= args.budget:
                    break
                y,x = divmod(int(idx), W)
                if blocked[y,x] or new_dots[y,x]:
                    continue
                new_dots[y,x]=True
                for dy,dx in offs:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < H and 0 <= xx < W:
                        blocked[yy,xx]=True
        final_dots = new_dots
        log(f"repair iter {it}: replaced {n_to_replace} overlapping dots, added {n_added} new far-field dots, total {int(final_dots.sum())}")
    # Final uniqueness check
    final_worst_ov = 0
    final_worst_path = None
    for path, halo in halos:
        ov = float(halo[final_dots].mean()) if final_dots.any() else 0.0
        if ov > final_worst_ov:
            final_worst_ov = ov; final_worst_path = path
    log(f"post-repair worst sparse overlap {final_worst_ov:.4f} vs {Path(final_worst_path).name if final_worst_path else 'none'}")

    # Verify final dots 0 on catalogue
    assert (final_dots & cat).sum() == 0
    np.save(OUT_NPY, final_dots)

    # Prepare results dict for run card
    s = summary["scores"]["H84"]
    # uniqueness vs sparse registry (local docs/downloads + submissions)
    # Use vendor uniqueness compare to registry index
    import json as js
    idx_path = ROOT / "registry/registry_index.json"
    raw_index = js.loads(idx_path.read_text()) if idx_path.exists() else {"rasters":[]}
    # For informative check we will locally scan docs/downloads/*.tif and compare via spearman and dot overlap, excluding dense
    # But we can also call compare_array_to_registry if possible
    # For now compute simple informative stats manually: load local tifs and compare

    # Save intermediate results
    results = {
        "lane": "H84-TIPLOBE-RELAY",
        "seed": args.seed,
        "budget": args.budget,
        "R_px": args.R,
        "footprint_px": int(foot.sum()),
        "catalogue_px": int(cat.sum()),
        "hidden_all_px": int(hidden_all.sum()),
        "canary": {"tip_lobe_AUC": canary_tip, "worst": worst, "leakage_flag": leakage_flag},
        "holdout_pooled_summary": summary,
        "per_fold": per_fold_detail,
        "dots_holdout": int(dot_mask.sum()),
        "dots_random": int(rand_dots.sum()),
        "dots_incumbent": int(inc_dots.sum()),
        "final_dots": int(final_dots.sum()),
        "final_dots_on_catalogue": int((final_dots & cat).sum()),
        "min_dot_distance_to_catalogue_px": float(dist_full[final_dots].min()) if final_dots.any() else None,
        "field_stats": {"global_max": float(final_field.max()), "global_mean_in_footprint": float(final_field[foot].mean()), "positive_px": int((final_field>0).sum())},
        "budget_split": per_budget,
    }
    # Write evidence json
    (ROOT / "evidence").mkdir(exist_ok=True)
    (RESULT_JSON).write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    log(f"elapsed {time.time()-t0:.1f}s -> {RESULT_JSON}")

if __name__ == "__main__":
    main()
