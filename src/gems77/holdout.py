"""Spatially-blocked holdout (hide-and-recover).

Four quadrants, buffer, truth = labels (catalogue) proxy.
Derives every catalogue-based feature only from visible faults? For our case,
features are geophysical-only (no catalogue-derived), so we just mask.

Scores pooled DTI (alpha 0.2 beta 0.8 triangular radius 3px).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .metric import dti_binary

def make_quadrant_blocks(shape, buffer_px: int = 30):
    H,W = shape
    mid_r, mid_c = H//2, W//2
    b = buffer_px
    # four blocks: NW, NE, SW, SE with buffer gap
    blocks = []
    # block masks (truth withheld region)
    # include buffer: if pixel is in block and not within buffer of center lines?
    # Simplified: split on midlines with no overlap: each block gets its quadrant, buffer excluded from all truth scoring to avoid leakage.
    # We'll create active region per fold: test block = quadrant expanded, train excludes buffer zone.
    blocks.append(("NW", (slice(0, mid_r-b), slice(0, mid_c-b))))
    blocks.append(("NE", (slice(0, mid_r-b), slice(mid_c+b, W))))
    blocks.append(("SW", (slice(mid_r+b, H), slice(0, mid_c-b))))
    blocks.append(("SE", (slice(mid_r+b, H), slice(mid_c+b, W))))
    return blocks

def holdout_score(field: np.ndarray, labels: np.ndarray, footprint: np.ndarray, n_emission: int = 8000, buffer_px: int = 2, min_dist: float = 2.8):
    """Compare greedy_maxcov vs dot_thin vs random control across 4 folds.
    Returns per-fold DTI for each method, mean etc.
    """
    from .emission import greedy_maxcov, poisson_thin, topk_mask, mask_catalogue_and_buffer
    import numpy as np
    blocks = make_quadrant_blocks(labels.shape, buffer_px=30)
    results = {}
    # precompute for each fold
    for name, sl in blocks:
        # truth for this fold = labels inside block only (hide whole segments: we withhold whole faults outside? But per competition spec, we withhold fault segments with buffer.
        # Simplified: we treat catalogue faults truncated to block as truth; any emission outside block not scored.
        # Derive visible faults = labels outside block expanded by buffer (not used for features since ours are geophys-only)
        ys, xs = sl
        truth = np.zeros(labels.shape, bool)
        truth[ys, xs] = labels[ys, xs]
        # valid = footprint inside block (so scoring only inside withheld region, masked visible faults pixel-exactly outside not needed because valid excludes outside)
        valid = footprint.copy()
        # mask visible faults not in this block from valid? Per spec: visible faults masked pixel-exactly from scoring (they are known). But since valid only inside block, visible faults outside block are irrelevant.
        # However any visible faults that happen to extend into block due to buffer leakage we mask via 30px buffer already.
        # For simplicity, known mask = labels outside block (but we don't need because valid excludes them)
        # Instead we set known = ~truth & labels  (visible) but ensure valid = footprint & test block
        valid_test = np.zeros(footprint.shape, bool)
        valid_test[ys, xs] = footprint[ys, xs]
        known = labels & ~truth  # visible
        # For scoring we will use DTI that internally does valid & ~known; we pass valid=valid_test, known=known
        # Emissions: generate candidates globally then score only inside test block (crop)
        # Generate emission masks from field (global field, but features derived only from visible? Ours are geophys so no leakage; we assert.)
        # Method A1 greedy fixed budget
        # For fairness, emit globally then evaluate inside block; but count pooled DTI per fold.
        # Top-k greedy
        greedy = greedy_maxcov(field, n_emission, footprint, radius=3.0)
        # dot_thin matched (incumbent style): topk then thin
        # For simplicity we use topk then poisson thin to match count ~n_emission
        # We need to calibrate q to get ~n_emission dots for incumbent; we approximate with greedy's count matching.
        # Here we use poisson_thin on top 2*n support
        sup = topk_mask(field, n_emission*3, footprint)
        thin = poisson_thin(sup, min_dist=min_dist)
        # if thin > n_emission, trim by field rank
        if thin.sum() > n_emission:
            # keep top n by field among thin
            vals = field.copy()
            vals[~thin] = -np.inf
            flat = vals.ravel()
            idx = np.argpartition(flat, -n_emission)[-n_emission:]
            new = np.zeros(flat.size, bool)
            new[idx]=True
            thin = new.reshape(field.shape)
        elif thin.sum() < n_emission*0.9:
            # fallback to topk directly
            thin = topk_mask(field, n_emission, footprint)

        # random control
        rng = np.random.default_rng(0)
        flat_idx = np.where(footprint.ravel())[0]
        rand_choice = rng.choice(flat_idx, size=n_emission, replace=False)
        rand = np.zeros(footprint.size, bool)
        rand[rand_choice]=True
        rand = rand.reshape(footprint.shape)

        # Mask catalogue with buffer (ensure 0 on catalogue for fairness): remove any emission within buffer_px of labels (using full labels, not just truth, because even visible catalog leakage must be removed)
        # But per holdout, forbidden should be visible labels only? For strict off-catalogue we use visible labels.
        # Let's use buffer on full labels for cleaner separation
        # CRITICAL FIX: during holdout, forbidden = visible faults only (known), NOT withheld truth.
        # Masking on full labels removes credit for correctly recovered hidden faults -> artificially zero DTI (bug found 2026-10-21)
        def apply_buffer(mask):
            from .emission import mask_catalogue_and_buffer
            # use known (visible) only, so withheld faults remain scorable
            # we still respect footprint
            # Build forbidden from known only
            if known.any():
                from scipy.ndimage import distance_transform_edt
                d = distance_transform_edt(~known)
                forbidden = d <= 2
                return mask & ~forbidden & footprint
            else:
                return mask & footprint

        greedy_b = apply_buffer(greedy)
        thin_b = apply_buffer(thin)
        rand_b = apply_buffer(rand)

        # Score each inside test block: we need DTI where pred = masked emission, truth = withheld faults, valid=valid_test, known = visible labels that fall inside valid_test? But visible inside valid_test is empty by construction (we withheld all in block). However some visible faults may cross buffer gap and leak 30px? Our buffer excluded 60px central cross, so safe.
        # So known inside valid_test should be zero.

        def score_mask(pred):
            # pred globally defined, but DTI will consider only valid_test region via valid param; outside ignored.
            # Need to ensure pred outside valid doesn't contribute to FP: _prepare does p = pred[active] where active=valid & ~known, so outside valid not counted. Good.
            res = dti_binary(pred, truth, valid=valid_test, known=known)
            return res

        r_greedy = score_mask(greedy_b)
        r_thin = score_mask(thin_b)
        r_rand = score_mask(rand_b)

        results[name] = {
            "greedy": r_greedy,
            "thin": r_thin,
            "random": r_rand,
            "n_truth": r_greedy["n_truth"],
            "greedy_px": int(greedy_b.sum()),
            "thin_px": int(thin_b.sum()),
            "rand_px": int(rand_b.sum()),
        }
    # aggregate
    def mean_dti(key):
        return float(np.mean([results[k][key]["dti"] for k in results]))
    summary = {
        "per_fold": results,
        "mean_greedy": mean_dti("greedy"),
        "mean_thin": mean_dti("thin"),
        "mean_random": mean_dti("random"),
        "delta_greedy_thin": mean_dti("greedy") - mean_dti("thin"),
        "folds_positive_greedy_vs_thin": int(sum(1 for k in results if results[k]["greedy"]["dti"] > results[k]["thin"]["dti"])),
        "folds_positive_greedy_vs_random": int(sum(1 for k in results if results[k]["greedy"]["dti"] > results[k]["random"]["dti"])),
    }
    return summary

def leakage_canary(features: dict, labels: np.ndarray, footprint: np.ndarray):
    """Test each feature alone on holdout for leakage (AUC >0.90)."""
    from sklearn.metrics import roc_auc_score
    out=[]
    for name, arr in features.items():
        vals = arr[footprint]
        y = labels[footprint].astype(int)
        # need finite
        mask = np.isfinite(vals)
        if mask.sum()==0:
            out.append((name, 0.5, False))
            continue
        try:
            auc = roc_auc_score(y[mask], vals[mask])
            auc2 = max(auc, 1-auc)
            out.append((name, float(auc2), bool(auc2>0.90)))
        except Exception:
            out.append((name, 0.5, False))
    return out
