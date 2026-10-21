#!/usr/bin/env python3
"""H80 Pipeline: Multi-Scale Tilt Persistence (E1-E3)"""
import json, time, hashlib, pathlib, sys
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"vendor"))
from gems57.grid import load_grid as load_grid57
from gems57 import holdout as HO
from gems57 import metric as M
from gems57.holdout import build_holdout
from gems77lia.core import grid as core_grid
from gems77lia.allocate import greedy_select

# Config
EVAL_SEEDS = (20,21)
BAND_TMI = 14
BAND_TMIVG = 9
BAND_GRAV = 13
BAND_GRAVVV = 11
BAND_DEPTH = 15
W = 3292
H = 3730
N_DOTS = 37654
RADIUS_PX = 3.0
CATALOGUE_BUFFER_PX = 2  # mask dots within 2px of visible catalogue (like H33 pruning)

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def clean_band(arr, footprint):
    a = np.asarray(arr, dtype=np.float32)
    # sentinel = -3.4e38
    bad = (~np.isfinite(a)) | (a < -1e30)
    # fill with median inside footprint
    vals = a[footprint & ~bad]
    med = float(np.median(vals)) if vals.size else 0
    out = np.where(bad, med, a).astype(np.float32)
    return out

def gf(arr, sigma):
    return ndimage.gaussian_filter(arr, sigma=sigma, mode='nearest')

def compute_tilt_field():
    """Compute tilt persistence surface. Returns 2D float32 array HxW normalized [0,1]"""
    log("loading grid")
    grid = load_grid57(ROOT/"data/raw")
    footprint = grid.footprint
    log(f"footprint {footprint.sum()} catalogue {grid.catalogue.sum()}")

    log("reading bands")
    with rasterio.open(ROOT/"data/raw/training_features.tif") as src:
        tmi = clean_band(src.read(BAND_TMI), footprint)
        tmi_vg = clean_band(src.read(BAND_TMIVG), footprint)
        grav = clean_band(src.read(BAND_GRAV), footprint)
        grav_vg = clean_band(src.read(BAND_GRAVVV), footprint)
        depth = clean_band(src.read(BAND_DEPTH), footprint)

    log("computing tilt persistence")
    # Helper to compute Z for one field at one sigma
    def tilt_zero_weighted(field, vg, sigma):
        # field = source field (tmi or grav)
        # smooth field for THDR
        sm = gf(field, sigma=sigma)
        gy, gx = np.gradient(sm)
        thdr = np.hypot(gy, gx).astype(np.float32)
        # smooth VG similarly
        vg_sm = gf(vg, sigma=sigma)
        # tilt
        tilt = np.arctan2(vg_sm, np.maximum(thdr, 1e-12)).astype(np.float32)
        # zero proximity: 1 - |tilt|/(pi/2)
        zero = (1.0 - np.abs(tilt) / (np.pi/2.0)).astype(np.float32)
        zero = np.clip(zero, 0, 1)
        # AS
        as_ = np.sqrt(thdr*thdr + vg_sm*vg_sm).astype(np.float32)
        as_log = np.log1p(np.maximum(as_, 0)).astype(np.float32)
        # Z = zero * logAS
        Z = (zero * as_log).astype(np.float32)
        return Z, tilt, thdr, as_log

    # Mag field at two scales
    Zmag_1p5, _, _, _ = tilt_zero_weighted(tmi, tmi_vg, sigma=1.5)
    Zmag_3p0, _, _, _ = tilt_zero_weighted(tmi, tmi_vg, sigma=3.0)
    Zgrav_1p5, _, _, _ = tilt_zero_weighted(grav, grav_vg, sigma=1.5)
    Zgrav_3p0, _, _, _ = tilt_zero_weighted(grav, grav_vg, sigma=3.0)

    # rank normalize each Z inside footprint to [0,1] for comparability
    def rank01(arr):
        out = np.zeros_like(arr, dtype=np.float32)
        v = arr[footprint]
        # percentile rank
        order = np.argsort(v)
        ranks = np.empty_like(v, dtype=np.float32)
        ranks[order] = np.linspace(0,1, v.size, dtype=np.float32)
        out[footprint] = ranks[np.argsort(order)]  # actually need mapping; simpler: use quantiles
        # Use quantiles for speed: we already have order, we can assign directly
        # But we need to map sorted values to ranks: v_sorted[order] corresponds to ranks
        # Let's do direct:
        # v[order] sorted ascending, ranks[order] = linspace
        # So out[footprint][order] = ranks
        # Actually we have v[order] sorted, we want to put ranks at those positions
        tmp = np.empty(v.size, dtype=np.float32)
        tmp[order] = np.linspace(0,1, v.size, dtype=np.float32)
        out[footprint] = tmp
        return out

    # Normalize each
    Zmag_1p5_n = rank01(Zmag_1p5)
    Zmag_3p0_n = rank01(Zmag_3p0)
    Zgrav_1p5_n = rank01(Zgrav_1p5)
    Zgrav_3p0_n = rank01(Zgrav_3p0)

    # Persistence = min across scales (requires coherence at both short and longer wavelength)
    # This suppresses shallow noise (decorrelates) and deep pluton edges that are broad only
    mag_persist = np.minimum(Zmag_1p5_n, Zmag_3p0_n)
    grav_persist = np.minimum(Zgrav_1p5_n, Zgrav_3p0_n)

    # Cross-field coincidence: product (both must be high)
    cross = (mag_persist * grav_persist).astype(np.float32)
    cross_n = rank01(cross)

    # Basement throw gating
    # Compute depth gradient and grav gradient at sigma 2.0
    sm_depth = gf(depth, sigma=2.0)
    dgy, dgx = np.gradient(sm_depth)
    db_norm = np.hypot(dgy, dgx).astype(np.float32)
    # grav THDR at sigma 2.0 for alignment
    sm_grav = gf(grav, sigma=2.0)
    ggy, ggx = np.gradient(sm_grav)
    grav_thdr_2 = np.hypot(ggy, ggx).astype(np.float32)
    # alignment: -dot / (|db||grav|) ; positive when basement deepens opposite to gravity increase
    # (basement high = gravity high, so across a step, depth gradient points toward basin, grav gradient points toward basement high -> anti-aligned)
    dot = (dgx * ggx + dgy * ggy)
    denom = (db_norm * grav_thdr_2 + 1e-9)
    align = (-dot / denom).astype(np.float32)
    align = np.clip(align, -1, 1)
    # only keep positive alignment (throw-consistent)
    align_pos = np.maximum(align, 0.0)
    # weight by depth gradient magnitude (strong basement steps matter)
    base_grad_log = np.log1p(np.maximum(db_norm, 0)).astype(np.float32)
    base_gate = (align_pos * base_grad_log).astype(np.float32)
    base_gate_n = rank01(base_gate)

    # Final field: cross * (0.5 + 0.5*base_gate) to allow cross alone but boost where throw consistent
    # Use 70% cross + 30% basement-gated
    final = (cross_n * (0.6 + 0.4*base_gate_n)).astype(np.float32)
    final[~footprint] = 0.0
    # Also rank overall final for emission (greedy needs monotonic ranking)
    final_ranked = rank01(final)
    final_ranked[~footprint] = 0

    # Save intermediates for audit
    out_dict = {
        "Zmag_1p5": Zmag_1p5,
        "Zmag_3p0": Zmag_3p0,
        "Zgrav_1p5": Zgrav_1p5,
        "Zgrav_3p0": Zgrav_3p0,
        "mag_persist": mag_persist,
        "grav_persist": grav_persist,
        "cross": cross,
        "cross_n": cross_n,
        "align": align,
        "base_gate": base_gate,
        "final": final,
        "final_ranked": final_ranked,
    }
    return final_ranked, out_dict, footprint, grid

def leakage_canary(fields_dict, footprint, catalogue):
    """Test each intermediate field alone on holdout canary. Returns dict with AUCs."""
    from sklearn.metrics import roc_auc_score
    from scipy.ndimage import distance_transform_edt
    log("building holdout for canary")
    grid = load_grid57(ROOT/"data/raw")
    # Use single seed build
    ctx = build_holdout(grid, hide_frac=0.20, seeds=(20,), modes=("all",))
    # Collect hidden truth vs far negatives
    # For each cell, get hidden vs negs, but for canary we can pool across cells
    # Simpler: use first cell's hidden as truth proxy and sample negatives outside 3px buffer
    # We will do pooled across all cells: concatenate all hidden pixels vs all far negatives
    all_pos = []
    all_neg_pool = []
    for cell in ctx.cells:
        hidden = ctx.hidden_by_cell[cell.key]
        # active region is domain & ~visible
        # For canary we want to evaluate field's ability to separate hidden truth vs background
        # Use hidden pixels inside footprint as positives, and random negatives deep outside catalogue
        pass
    # Simpler canary: use labels proxy: labels vs far background (like prior E1)
    # Use whole catalogue as truth proxy, but with leakage we want to see if field correlates with catalogue location (which would be leakage if feature derived from visible faults)
    # Since our fields are geophys-only, they should have low AUC
    h,w = footprint.shape
    fp_flat = np.flatnonzero(footprint.ravel())
    fp_lut = np.full(h*w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)
    dist_cat = ndimage.distance_transform_edt(~catalogue).astype(np.float32)
    rng = np.random.default_rng(1020)
    # hidden for canary: use ctx hidden union
    hidden_all = np.zeros((h,w), bool)
    for cell in ctx.cells:
        hidden_all |= ctx.hidden_by_cell[cell.key]
    hidden_all &= footprint
    pos_fp = fp_lut[np.flatnonzero(hidden_all)]
    neg_pool = fp_lut[np.flatnonzero(footprint & ~catalogue & (dist_cat > 3))]
    neg_fp = rng.choice(neg_pool, size=min(200000, neg_pool.size), replace=False)
    can_idx = np.concatenate([pos_fp, neg_fp])
    y = np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])
    log(f"canary pooled: pos {pos_fp.size} neg {neg_fp.size} total hidden union {int(hidden_all.sum())}")
    out = {}
    for name, arr in fields_dict.items():
        # flatten to footprint vector
        v = arr.ravel()[footprint.ravel()]
        # but we need mapping via fp_lut: arr[footprint] is already in fp order
        # Extract canary values
        flat = arr.ravel()
        # use fp_lut to index: we have flat indices, we can get values via flat[can_flat]
        # can_fp are indices into footprint vector, not flat. Let's map:
        # footprint vector idx -> flat index: fp_flat[footprint_idx] = flat location
        # So can_fp (footprint idx) -> flat via fp_flat[can_fp]
        can_flat = fp_flat[can_idx]
        vals = flat[can_flat]
        m = np.isfinite(vals)
        if m.sum() > 100 and len(np.unique(y[m]))==2 and np.ptp(vals[m])>0:
            auc = float(roc_auc_score(y[m], vals[m]))
            sym = max(auc, 1-auc)
            out[name] = {"auc": auc, "sym_auc": sym, "leakage_flag": bool(sym>0.90)}
        else:
            out[name] = {"auc": None, "sym_auc": None, "leakage_flag": False}
        log(f" canary {name}: auc {out[name]['auc']} sym {out[name]['sym_auc']}")
    return out

def holdout_evaluation(surface, footprint, grid):
    """Pooled HOLDOUT-DTI across seeds 20,21 all/detached. Compare candidate vs random vs simple control."""
    import json
    from gems57 import evaluate_holdout as EH
    from scipy.ndimage import distance_transform_edt
    log("building holdout contexts")
    ctx = build_holdout(grid, hide_frac=0.20, seeds=(20,21), modes=("all","detached"))
    # Prepare arms
    # For each cell, we need emitted binary masks
    # We will generate globally then crop per cell? But EH.evaluate expects prediction array shape same as grid, and fold region to limit scoring via cell.active
    # We can generate emitted masks once globally, then per cell evaluation will automatically limit to active region via EH.evaluate's active masking? Actually holdout score_cell uses active from cell, but EH.evaluate uses prediction and fold region.
    # Let's use vendor gems57.evaluate_holdout.evaluate which does global prediction and fold region.
    # Simpler: use ctx.cells and score_cell? But we need pooled_summary with terms.
    # Use the newer pooled evaluator: for each cell, call EH.evaluate(prediction, fold, valid) where valid = footprint, fold contains region, masked_known etc.
    # But we have ctx with hidden_by_cell and cells that already have bbox and active and truth_yx.
    # Let's try to replicate H77D's holdout via HO.build_holdout + EH.evaluate style that H80 should replicate.
    # Simpler: use the direct API: for each cell key, get fold dict from buffered_component_draw? However ctx is HoldoutContext, not dict fold.
    # We need to reconstruct fold dict for EH.evaluate: it expects keys: region, visible, truth, masked_known, etc. But HO.build_holdout returns HoldoutContext with hidden_by_cell.
    # Let's see EH.evaluate signature: evaluate(prediction, fold, valid, block_side, origin, global_shape)
    # fold must have region, visible, truth, masked_known.
    # We can synthesize fold per cell: region = cell bbox expanded? Actually cell.active is domain & ~visible cropped to bbox.
    # But EH.evaluate works on full-grid prediction arrays; we will pass full-grid arrays and let it crop via origin/global_shape? The function supports origin/global_shape for cropped folds to align blocks.
    # Let's inspect EH.evaluate implementation: it checks prediction shape vs valid, region, known, truth_all via fold dict.
    # It expects fold['region'] is full-grid bool, fold['visible'], fold['truth'], fold['masked_known'].
    # For our ctx, we have hidden_by_cell and can infer visible = catalogue & ~hidden, region = domain (eroded quadrant?) But we have cell.active = domain & ~visible, but not full region.
    # Alternative: use older simpler holdout API: scripts/e3_holdout.py used buffered_component_draw from gems57.holdout which returns folds list of dicts with keys visible, masked_known, truth, region, receipt.
    # That API is different: it returns folds as list of dicts directly usable with evaluate_holdout.evaluate.
    # Let's try that API: from gems57.holdout import buffered_component_draw
    from gems57.holdout import buffered_component_draw, score_cell
    log("using buffered_component_draw for holdout")
    folds, receipts = buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
    log(f" got {len(folds)} folds for seed 20")
    # For seed 21 also
    folds2, receipts2 = buffered_component_draw(grid, seed=21, hide_frac=0.20, buffer_px=3)
    all_folds = folds + folds2
    log(f" total folds {len(all_folds)}")

    # Generate emitted masks globally
    # Candidate: greedy on surface ranking
    def make_greedy(surf, n=N_DOTS):
        flat = surf.ravel()
        order = np.argsort(-flat)  # descending
        # filter to footprint only
        order = order[np.isfinite(flat[order]) & (flat[order]>0) & footprint.ravel()[order]]
        # also mask catalogue buffer (2px) globally for emission? We'll do per fold masking later, but for global we can mask catalogue 2px to match submission pruning
        dist_cat = ndimage.distance_transform_edt(~grid.catalogue)
        allowed = (dist_cat > CATALOGUE_BUFFER_PX) & footprint
        allowed_flat = np.flatnonzero(allowed.ravel())
        # order only allowed
        allowed_set = set(allowed_flat.tolist())  # slow but ok for 5M?
        # Instead filter via boolean mask
        mask_allowed = allowed.ravel()[order]
        order = order[mask_allowed]
        # greedy select
        sel = greedy_select(order, min_sep=RADIUS_PX, limit=n)
        mask = np.zeros(footprint.size, dtype=bool)
        mask[sel] = True
        return mask.reshape(H,W)

    cand_global = make_greedy(surface, N_DOTS)
    log(f"candidate dots {cand_global.sum()}")
    # Random control same n
    rng = np.random.default_rng(0)
    allowed = (ndimage.distance_transform_edt(~grid.catalogue) > CATALOGUE_BUFFER_PX) & footprint
    allowed_idx = np.flatnonzero(allowed.ravel())
    rand_choice = rng.choice(allowed_idx, size=N_DOTS, replace=False)
    # greedy thin random to enforce spacing
    rand_order = rand_choice[np.argsort(rng.random(len(rand_choice)))]
    rand_sel = greedy_select(rand_order, min_sep=RADIUS_PX, limit=N_DOTS)
    rand_global = np.zeros(footprint.size, dtype=bool)
    rand_global[rand_sel] = True
    rand_global = rand_global.reshape(H,W)
    log(f"random dots {rand_global.sum()}")
    # Simple control: gravity THDR alone as ranking (to test tilt adds value)
    # Compute grav THDR log quickly for control: reuse grav THDR from earlier? Let's compute simple grav THDR ranking
    # For speed, we will reuse cross_n as simple control? But we need distinct control.
    # Let's create control surface = grav THDR log rank (precomputed earlier as grav_thdr_log rank) - we can compute quickly here
    with rasterio.open(ROOT/"data/raw/training_features.tif") as src:
        grav_raw = clean_band(src.read(BAND_GRAV), footprint)
        grav_vg_raw = clean_band(src.read(BAND_GRAVVV), footprint)
    # compute grav THDR at 2.0 for control
    sm_grav = gf(grav_raw, sigma=2.0)
    ggy, ggx = np.gradient(sm_grav)
    grav_thdr = np.hypot(ggy, ggx).astype(np.float32)
    grav_thdr_log = np.log1p(np.maximum(grav_thdr,0)).astype(np.float32)
    # rank it
    tmp = grav_thdr_log[footprint]
    order = np.argsort(tmp)
    ranks = np.empty_like(tmp, dtype=np.float32)
    ranks[order] = np.linspace(0,1, tmp.size, dtype=np.float32)
    grav_rank = np.zeros((H,W), dtype=np.float32)
    grav_rank[footprint] = ranks
    ctrl_global = make_greedy(grav_rank, N_DOTS)
    log(f"control grav dots {ctrl_global.sum()}")

    # Evaluate each arm per fold, collect terms
    from gems57.evaluate_holdout import evaluate as EH_eval, pooled_summary
    terms_by_arm = {"candidate": [], "random": [], "grav_control": []}
    for fold in all_folds:
        valid = footprint
        for name, pred in [("candidate", cand_global), ("random", rand_global), ("grav_control", ctrl_global)]:
            res, terms = EH_eval(pred.astype(np.float32), fold, valid, block_side=200)
            # terms is (n_blocks,4) array
            terms_by_arm[name].append(terms)
        log(f" fold {fold['receipt']['fold'] if 'receipt' in fold else fold.get('name','?')} truth {fold['truth'].sum() if 'truth' in fold else '?'}")

    # Pool sums
    # Stack terms per arm: sum over folds then pool
    arms_summed = {k: np.sum(np.stack(v, axis=0), axis=0) if len(v)>1 else np.stack(v,axis=0)[0] for k,v in terms_by_arm.items()}
    # But pooled_summary expects dict of terms_by_arm where each is (n_blocks,4) summed? Let's check: original h77d used pooled_summary with terms dict where each value is stacked then summed?
    # In vendor evaluate_holdout.pooled_summary, it expects terms_by_arm as dict name -> (n_blocks,4) array already summed? Let's inspect: it does arrays = {n: np.asarray(v,float) for n,v in terms_by_arm.items()} and checks shape == (n_blocks,4)
    # So we need to sum folds into single (n_blocks,4) per arm.
    # Our stacking: each fold's terms is (n_blocks,4) but with many zeros outside its region; summing across folds correctly pools.
    # Our arms_summed currently is sum over folds of (n_blocks,4) -> correct pooled terms.
    # Now call pooled_summary
    summary = pooled_summary({k: v for k,v in arms_summed.items()}, draws=2000, candidate="candidate", seed=520810)
    return summary, {"candidate": cand_global, "random": rand_global, "grav_control": ctrl_global}

def main():
    t0 = time.time()
    surface, intermediates, footprint, grid = compute_tilt_field()
    log(f"surface built min {surface[footprint].min():.3f} max {surface[footprint].max():.3f} mean {surface[footprint].mean():.3f}")
    # save surface for quick inspection
    Path(".cache").mkdir(exist_ok=True)
    np.save(".cache/h80_surface.npy", surface)

    # E1 canary
    log("E1 canary")
    canaries = leakage_canary(intermediates, footprint, grid.catalogue)
    # check max sym auc
    max_sym = max([v["sym_auc"] for v in canaries.values() if v["sym_auc"] is not None] or [0])
    log(f"max sym AUC {max_sym:.4f}")
    e1_out = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": "gems57-pooled-hide-v2", "fields": list(intermediates.keys()), "canaries": canaries, "max_sym_auc": max_sym, "leakage_flag": bool(max_sym>0.90)}
    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/h80_e1_fields.json").write_text(json.dumps(e1_out, indent=2))
    if max_sym > 0.90:
        log("LEAKAGE FLAG >0.90 hard stop")
        sys.exit(2)

    # E2 holdout
    log("E2 holdout")
    summary, masks = holdout_evaluation(surface, footprint, grid)
    log(f"E2 summary: candidate DTI {summary['scores']['candidate']['dti']:.4f} CI {summary['scores']['candidate']['ci95']}")
    # write out
    Path("evidence/h80_e2_holdout.json").write_text(json.dumps(summary, indent=2))
    # also save canaries already

    # decide promotion: candidate must beat random and grav_control with CI >0 and beat repo best 0.1527
    cand = summary["scores"]["candidate"]
    rand = summary["scores"]["random"]
    grav = summary["scores"]["grav_control"]
    # paired diffs
    diff_rand = summary["paired_differences"]["random"]
    diff_grav = summary["paired_differences"]["grav_control"]
    promote = (diff_rand["delta"] > 0 and diff_rand["ci95"][0] > 0) and (diff_grav["delta"] > 0 and diff_grav["ci95"][0] > 0) and (cand["dti"] > 0.152718)
    # also require absolute > random and > grav
    log(f" promotion check: vs random {diff_rand['delta']:.4f} CI {diff_rand['ci95']} vs grav {diff_grav['delta']:.4f} CI {diff_grav['ci95']} candidate {cand['dti']:.4f} vs repo best 0.1527 => promote {promote}")
    # Even if not promoting, we still generate negative deliverable per instructions: negative results are deliverables
    # So we will generate TIF regardless but mark verdict accordingly
    # Build submission for candidate
    log("E3 building submission")
    # Use submission_writer via core grid
    from gems77lia.core import grid as cg
    g2 = cg()
    # masks candidate is already binary HxW
    pred = masks["candidate"].astype(np.float32)
    # outside footprint must be 0, inside is 0/1, no NaN
    pred[~g2.footprint] = 0
    # validate range
    assert np.isfinite(pred[g2.footprint]).all()
    assert pred[g2.footprint].min()>=0 and pred[g2.footprint].max()<=1
    # write via rasterio directly using sample_submission as template (like vendor write_submission)
    import hashlib, uuid, shutil, zipfile
    from affine import Affine
    # Get template meta from sample_submission
    with rasterio.open(ROOT/"data/raw/sample_submission.tif") as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
    # Ensure float32
    meta.update(driver="GTiff", dtype="float32", count=1, compress="deflate", tiled=True, nodata=None)
    # Generate unique id
    uid = hashlib.sha256(pred.tobytes()).hexdigest()[:8]
    name = f"gems77-h80-tilt-persist-{N_DOTS}dots-20261021-{uid}-zeros"
    note = f"H80 tilt persistence dual-scale {N_DOTS} dots; HOLDOUT-DTI {cand['dti']:.4f} [{cand['ci95'][0]:.4f},{cand['ci95'][1]:.4f}] vs rand {diff_rand['delta']:+.4f}"
    # Note must be <=140 chars
    note = note[:140]
    out_tif = ROOT / f"submissions/{name}.tif"
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(pred, 1)
    log(f"wrote {out_tif} sha {hashlib.sha256(out_tif.read_bytes()).hexdigest()[:12]} dots {int(pred[g2.footprint].sum())}")
    # also write zip
    out_zip = ROOT / f"submissions/{name}.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_tif, arcname=out_tif.name)
    # also copy to docs/downloads for site
    docs_dl = ROOT / "docs/downloads"
    docs_dl.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_tif, docs_dl / out_tif.name)
    shutil.copy(out_zip, docs_dl / out_zip.name)
    # also nan variant for reference (NaN outside)
    meta_nan = meta.copy()
    meta_nan.update(nodata=np.nan)
    pred_nan = np.where(g2.footprint, pred, np.nan).astype(np.float32)
    out_tif_nan = ROOT / f"submissions/{name.replace('-zeros','-nan')}.tif"
    with rasterio.open(out_tif_nan, "w", **meta_nan) as dst:
        dst.write(pred_nan, 1)
    shutil.copy(out_tif_nan, docs_dl / out_tif_nan.name)
    # validate via gems77lia.validate
    from gems77lia.validate import validate as validate_sub
    v = validate_sub(out_tif, mode="zeros", name=name, note=note)
    log(f"validator: {v['passed']} {v['n_pass']}/{v['n_checks']}")
    # uniqueness: quick check vs registry sample (we can't scan all 812 here fully, but do informative subset)
    # We'll compute rho vs a few registry rasters if available
    # For full audit we need to run script, but for pipeline we log verdict based on holdout
    run_card = {
        "schema": "gems-parallel-run-card-v1",
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/de858329-gemsdoe77",
        "hypothesis": "H80: Multi-Scale Tilt-Angle Persistence x AS-Weighted Zero Proximity + Signed Basement Throw",
        "mechanism": "Tilt = atan2(VG, THDR) at sigma 1.5 and 3.0; Z = (1-|tilt|/(pi/2))*log1p(AS); persistence = min(Z_1.5, Z_3.0) per field; F = Zmag_persist * Zgrav_persist * (0.6+0.4*base_throw_align) ranked",
        "named_non_fault_process_that_could_mimic_it": "Lithologic/intrusive contacts and caldera margins (tilt+AS without throw), GeoDAWN survey-block seams, remanently magnetized volcanics",
        "holdout_dti": {"evidence_label": "HOLDOUT-DTI", "evaluator_version": "gems57-pooled-hide-v2", "score": cand["dti"], "ci_95": cand["ci95"], "withheld_positive_pixels": cand["withheld_positive_pixels"], "paired_vs_random": diff_rand, "paired_vs_grav": diff_grav},
        "leakage_canary": {"max_sym_auc": max_sym, "ceiling": 0.90, "pass": bool(max_sym<=0.90)},
        "uniqueness_vs_registry": {"note": "Full 812-raster scan not run in pipeline; see e3 script", "preliminary": True},
        "submission_raster": {"path": str(out_tif), "sha256": hashlib.sha256(out_tif.read_bytes()).hexdigest(), "bytes": out_tif.stat().st_size, "dots": int(pred[g2.footprint].sum()), "validator": v},
        "submission_name": name,
        "submission_note": note,
        "verdict": "promote" if promote else "negative",
        "safe_to_download_and_submit": False
    }
    Path("evidence/h80_run_card.json").write_text(json.dumps(run_card, indent=2))
    log(f"done promote={promote} run card written")
    # print for next steps
    print(json.dumps({"candidate_dti": cand["dti"], "ci": cand["ci95"], "withheld": cand["withheld_positive_pixels"], "promote": promote, "file": str(out_tif)}, indent=2))

if __name__ == "__main__":
    main()
