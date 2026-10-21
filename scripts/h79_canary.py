#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import rasterio
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"vendor"))
from gems57.grid import load_grid
from gems57.holdout import build_holdout

DATA_BRIDGE = ROOT/"data"/"bridge"
DATA_RAW = ROOT/"data"/"raw"

def load_pred(path):
    with rasterio.open(path) as s:
        a=s.read(1).astype(np.float32)
    a[~np.isfinite(a)]=0
    return a

# Test features as surfaces: our scoring components
# We'll load contact, ridge, coh from saved npy if exists, else recompute quick?
# For canary we test the final_rank and its components

# Build holdout to get withheld truth
grid = load_grid(DATA_BRIDGE)
ctx = build_holdout(grid, hide_frac=0.20, seeds=(20,), modes=("all",))
# Collect active domain
active = np.zeros(grid.footprint.shape, bool)
truth = np.zeros(grid.footprint.shape, bool)
for cell in ctx.cells:
    if cell.key.endswith("_20_all") or "20" in cell.key:
        # cell.key format like 0_NW_20_all etc
        sy,sx = cell.bbox
        # active domain is cell.active cropped
        active[sy,sx] |= cell.active
        truth[sy,sx] |= ctx.hidden_by_cell[cell.key][sy,sx]

# Load scoring rank
score = np.load(ROOT/"docs"/"evidence"/"h79_scoring_rank.npy")
# For canary, compute AUC per feature alone on holdout
from sklearn.metrics import roc_auc_score

features = {
    "final_rank": score,
    "contact_proxy": score, # placeholder, we will recompute contact if needed
}

# Actually load contact, ridge, coh from build step? We saved only final_rank, but we have pred tif
# Use final_rank as representative
results = {}
for name, feat in features.items():
    # mask to active domain where truth is defined (active = scored domain minus visible)
    # active already excludes visible faults per cell
    mask = active & grid.footprint & np.isfinite(feat)
    y = truth[mask].astype(int)
    x = feat[mask]
    if len(np.unique(y)) <2:
        auc = None
    else:
        try:
            auc = float(roc_auc_score(y, x))
        except Exception as e:
            auc = str(e)
    results[name] = auc
    print(f"{name}: AUC {auc}")

# Also test catalogue distance as canary (should be high if leaked)
from scipy import ndimage
dist = ndimage.distance_transform_edt(~grid.catalogue).astype(np.float32)
mask = active & grid.footprint
y = truth[mask].astype(int)
x = dist[mask]
try:
    auc_dist = float(roc_auc_score(y, -x))  # closer to catalogue = more likely to be withheld? But withheld segments are inside quadrants, distance to visible may be high
    print(f"catalogue_distance_neg AUC {auc_dist}")
    results["catalogue_distance_neg"] = auc_dist
except Exception as e:
    print(e)

Path(ROOT/"evidence"/"h79_canary.json").write_text(json.dumps(results, indent=2)+"\n")
print("saved")
