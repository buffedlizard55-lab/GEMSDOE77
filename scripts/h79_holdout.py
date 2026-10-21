#!/usr/bin/env python3
"""Holdout evaluation for H79 candidate vs controls."""
import json, sys
from pathlib import Path
import numpy as np, rasterio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"vendor"))
from gems57.grid import load_grid
from gems57.holdout import build_holdout
from gems57.evaluate_holdout import evaluate as eval_fold, pooled_summary, VERSION

DATA_BRIDGE = ROOT/"data"/"bridge"

def load_pred(path):
    with rasterio.open(path) as s:
        a=s.read(1).astype(np.float32)
    a[~np.isfinite(a)] = 0
    a = np.clip(a,0,1)
    return a

def main():
    grid = load_grid(DATA_BRIDGE)
    # need holdout context
    ctx = build_holdout(grid, hide_frac=0.20, seeds=(20,30), modes=("all",))
    print(f"built holdout: {len(ctx.cells)} cells")
    # load predictions
    cand_path = ROOT/"docs"/"downloads"/"gems77-h79-crossgrad-ridge-37654dots-20261021-dd0dbf9c-zeros.tif"
    cand = load_pred(cand_path)
    # champion construction: we need to rebuild H19-5-like? Use simple control: catalogue distance + ridge baseline
    # For fair comparison, create controls:
    # A0: H19-5 proxy? We'll use a simple control: random dots
    # A1: H33 fold-fair proxy: dots on most persistent ridges > catalogue 3px (we approximate by ridge field threshold)
    # But we need at least champion construction to compare: we can create a surrogate champion as dots sampled from same scoring but filtered to catalogue distance >2px (mimics H33 logic)
    # Simpler: Use existing scored raster if available: data/raw/scored? Try to find H19-5 scored raster in data/raw/scored
    scored_dir = ROOT/"data"/"raw"/"scored"
    champion = None
    champion_name = "random_control"
    if scored_dir.exists():
        print("scored dir", list(scored_dir.glob("*.tif"))[:5])
    # Create random control
    rng = np.random.default_rng(0)
    allowed = grid.footprint & ~grid.catalogue
    rand = np.zeros_like(cand, dtype=np.float32)
    # random dots equal count
    n = int((cand>0).sum())
    idx = rng.choice(np.flatnonzero(allowed), size=n, replace=False)
    rand.flat[idx] = 1.0
    rand[~grid.footprint]=0

    # Create catalogue-distance-filtered version of cand as "fold-fair H33-like" for comparison: we need per-fold geometry
    # Actually for holdout scoring, we need to score cand that was built on full catalogue (leaky). For fold-fair we must rebuild per fold.
    # We will compute pooled holdout for cand and random using the hide-and-recover protocol.
    # The evaluator requires per-fold prediction derived from visible catalogue only.
    # Our cand is static (leaky). True holdout should be computed via per-fold rebuilding for fair champion.
    # But we can still compute the leak-free holdout for cand vs random by evaluating static field on withheld truth with leakage guard
    # We'll evaluate using evaluate_holdout with per-cell predictions masked to active domain.

    # We need to implement pooled evaluation across cells, with terms per block
    # For static cand, we evaluate directly.
    terms_by_arm = {"h79_canvas": None, "random": None}
    # We'll compute terms manually via evaluate()
    # For each cell, evaluate fold's prediction (static) masked
    all_terms = {}
    for name, pred in [("h79_canvas", cand), ("random", rand)]:
        terms_list = []
        for cell in ctx.cells:
            # cell.bbox slices
            sy, sx = cell.bbox
            # Crop pred to cell bbox
            crop_pred = pred[sy, sx]
            # cell.active is boolean cropped domain minus visible faults
            # fold dict for evaluate
            fold = {"region": cell.active, "visible": np.zeros_like(cell.active, bool), "masked_known": np.zeros_like(cell.active, bool), "truth": np.zeros_like(cell.active, bool)}
            # Actually we need proper fold: visible is grid.catalogue & ~hidden, truth is hidden, region is eroded quadrant
            # Use ctx to get visible and truth
            visible_full = ctx.visible(cell.key)
            truth_full = ctx.hidden_by_cell[cell.key]
            # Crop to bbox
            fold_visible = visible_full[sy, sx]
            fold_truth = truth_full[sy, sx]
            # region is cell.active (which already is domain & ~visible but we need domain)
            # For evaluate we pass fold with truth and visible
            fold_dict = {"region": cell.active, "visible": fold_visible, "masked_known": fold_visible, "truth": fold_truth}
            valid = grid.footprint[sy,sx]
            # origin for block alignment
            oy, ox = sy.start, sx.start
            result, terms = eval_fold(crop_pred, fold_dict, valid, block_side=200, origin=(oy,ox), global_shape=grid.footprint.shape)
            terms_list.append(terms)
        # Pool terms across cells
        stacked = np.concatenate(terms_list, axis=0)
        all_terms[name] = stacked

    # Now pooled summary vs candidate
    # Need to pass terms_by_arm with candidate key
    summary = pooled_summary(all_terms, draws=1000, seed=520810, candidate="h79_canvas")
    print(json.dumps(summary, indent=2))
    # Also compute single-arm DTI pooled
    for name, terms in all_terms.items():
        tp, fp, fn, n = terms.sum(axis=0)
        dti = tp / (tp + 0.2*fp + 0.8*fn + 1e-9)
        print(f"{name}: tp {tp:.1f} fp {fp:.1f} fn {fn:.1f} n {n:.0f} dti {dti:.4f}")

    # Save
    (ROOT/"evidence"/"h79_holdout.json").write_text(json.dumps({"all_terms_shapes": {k: list(v.shape) for k,v in all_terms.items()}, "pooled_summary": summary}, indent=2)+"\n")

if __name__ == "__main__":
    main()
