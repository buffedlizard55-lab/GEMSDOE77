#!/usr/bin/env python3
"""Write H84 final TIF + ZIP + receipts, validate."""
import json, hashlib, sys
from pathlib import Path
import numpy as np, rasterio
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"vendor"))
sys.path.insert(0, str(ROOT/"src"))
from gems57.grid import load_grid, write_geotiff, read_geotiff
from gems57 import metric as M
grid=load_grid(ROOT/"data/bridge")
foot=grid.footprint
cat=grid.catalogue
# Load dots (repaired if exists else original)
p_repaired=ROOT/"evidence/h84_dots_repaired.npy"
p_orig=ROOT/"evidence/h84_dots.npy"
src_path = p_repaired if p_repaired.exists() else p_orig
dots=np.load(src_path)
print(f"loaded {src_path} dots {int(dots.sum())} on_cat {(dots & cat).sum()}")
# Create submission array: zeros outside footprint, 0/1 inside
arr=np.zeros((3730,3292), dtype=np.float32)
arr[foot] = dots[foot].astype(np.float32)  # inside footprint: 0/1
# outside footprint remains 0 (not NaN) => all finite, in [0,1]
assert np.isfinite(arr).all()
assert arr.min()>=0 and arr.max()<=1
# Also ensure 0 on catalogue? our dots already 0 on cat, but we also need to ensure arr[cat]==0 (cat subset of foot, dots 0 there, so arr 0)
assert (arr[cat]==0).all()
# Write
out_dir=ROOT/"docs/downloads"
out_dir.mkdir(parents=True, exist_ok=True)
# Unique name: include budget, date, hash
import datetime, hashlib
# Use hash of dots for uniqueness
dot_hash=hashlib.sha256(dots.astype(np.float32).tobytes()).hexdigest()[:8]
date_str="20261021"
stem=f"gems77-h84-tiplobe-37654dots-{date_str}-{dot_hash}"
tif_path=out_dir/f"{stem}-zeros.tif"
# write via grid.write_geotiff (ensures float32, EPSG, etc, zeros mode)
receipt=write_geotiff(tif_path, arr, nodata=None)  # no nodata, zeros outside
print(f"WROTE {tif_path} sha256 {receipt['sha256']} bytes {receipt['bytes']}")
# Create ZIP with single TIF
import zipfile
zip_path=out_dir/f"{stem}-zeros.zip"
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(tif_path, arcname=tif_path.name)
zip_sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
print(f"ZIP {zip_path} sha256 {zip_sha} bytes {zip_path.stat().st_size}")
# Validate readback
with rasterio.open(tif_path) as src:
    a=src.read(1)
    print(f"readback crs {src.crs} shape {src.shape} transform {src.transform[:6]} dtype {src.dtypes[0]} nodata {src.nodata}")
    print(f"inside footprint finite {np.isfinite(a[foot]).sum()}/{foot.sum()} in_range {((a[foot]>=0)&(a[foot]<=1)).sum()} min {np.nanmin(a[foot])} max {np.nanmax(a[foot])} positive {int((a>0).sum())} unique {np.unique(a[foot])[:5]}")
    print(f"outside footprint zeros {(a[~foot]==0).sum()}/{(~foot).sum()} nan {(np.isnan(a[~foot])).sum()}")
    # check catalogue
    print(f"dots on catalogue {(a>0 & cat).sum() if False else int(((a>0) & cat).sum())}")
    assert np.isfinite(a).all(), "NaN found anywhere, fails zeros policy"
    assert (a[foot]>=0).all() and (a[foot]<=1).all(), "range fail"
    assert (a[~foot]==0).all(), "outside not zeros"
    assert ((a>0) & cat).sum()==0, "dots on catalogue"
print("VALIDATOR ALL PASS (local finite [0,1] zeros-outside)")

# Build run card update with validator and submission info
run_card_path=ROOT/"evidence/h84_run_card.json"
rc=json.loads(run_card_path.read_text())
# Add submission info
rc["submission"]={
    "name": f"{stem}-zeros",
    "note": "H84 tip-lobe 37654 dots; HOLDOUT-DTI 0.1209 [0.1068,0.1354] s30 vs random +0.073; tip AUC 0.79; NOT FOR UPLOAD without selector",
    "note_chars": len("H84 tip-lobe 37654 dots; HOLDOUT-DTI 0.1209 [0.1068,0.1354] s30 vs random +0.073; tip AUC 0.79; NOT FOR UPLOAD without selector"),
    "tif_path": str(tif_path.relative_to(ROOT)),
    "zip_path": str(zip_path.relative_to(ROOT)),
    "tif_sha256": receipt["sha256"],
    "zip_sha256": zip_sha,
    "tif_bytes": receipt["bytes"],
    "zip_bytes": zip_path.stat().st_size,
    "validator": receipt,
    "ok_to_download": True,
    "ok_to_submit": "NO - negative lane, duplicate overlap 0.787 >0.70 vs sparse registry, dense-witness gate FAIL, selector not approved, 0 slots used",
    "verdict": "negative"
}
# Label numbers
rc["number_labelling"]="HOLDOUT-DTI evaluator gems57-pooled-hide-v2 11342 withheld positives 95% CI [0.1068,0.1354]; no ORGANIZER-CONFIRMED score; OWNER-REPORT 0.2778 not organizer receipt; projection never written as score"
# Add hypothesis etc
rc["hypothesis"]="Fault-tip crack-tip stress lobe + en-echelon relay breach: blind tip splay fans (22% geothermal) and relay breaching fractures (48% geothermal per Faulds 2010) under Quaternary cover, omitted from QFaults but dilational"
rc["mechanism"]="Tip mask via 8-neighbour endpoint count, distance to nearest tip R=25px/2.5km isotropic field 1-d/R, off-catalogue emission 37654 dots min_sep 2.8px, visible-only per-fold features, masked visible pixel-exactly, pooled DTI"
rc["named_non_fault_mimic"]="Lake Lahontan/Bonneville terrace risers fanning from tips, alluvial-fan apices, road/canal cuts; inter-fan swales mimicking relay corridors"
rc["correlation_overlap_vs_registry"]="Sparse registry worst forward 3px overlap ~0.787 >0.70 vs H82 faultcollar/H77E-pure (38 sparse rasters, sampled Spearman max |rho| ~0.45); dense witness 5.16M positive gives universal overlap 1.0 (literal gate FAIL, reported separately). Informative sparse check shows duplicate by protocol."
rc["raster_sha256"]=receipt["sha256"]
rc["validator_output"]=f"no NaN inside footprint, values in [0,1] min {float(a[foot].min())} max {float(a[foot].max())}, CRS EPSG:32611 shape 3730x3292 transform (100,0,243350,0,-100,4508550) match, 37654 dots 0 on catalogue"
rc["promotion_gates"]={"format_valid": True, "leakage_canary_pass": rc["canary"]["leakage_flag"]==False, "holdout_vs_random_significant": True, "holdout_vs_incumbent": "positive but below repository incumbent 0.1527, so not promotion", "uniqueness_informative": "FAIL 0.787 >0.70 duplicate", "uniqueness_dense_literal": "FAIL 1.0 universal blocker", "selector_approved": False, "verdict": "negative - inspection download only"}
run_card_path.write_text(json.dumps(rc, indent=2, allow_nan=False)+"\n")
print(f"updated run card {run_card_path}")

# Also write a simple JSON for downloads index
downloads_json=ROOT/"docs/downloads/gems77-h84-tiplobe-37654dots-20261021.json"
# reuse submission info plus holdout dti
import datetime
downloads_json.write_text(json.dumps({
    "name": f"{stem}-zeros",
    "note": "H84 tip-lobe 37654 dots; HOLDOUT-DTI 0.1209 [0.1068,0.1354] s30 vs random +0.073; tip AUC 0.79; NOT FOR UPLOAD without selector",
    "tif": str(tif_path.relative_to(ROOT/"docs")),
    "zip": str(zip_path.relative_to(ROOT/"docs")),
    "sha256": receipt["sha256"],
    "holdout_dti": rc["holdout_pooled_summary"]["scores"]["H84"]["dti"],
    "ci95": rc["holdout_pooled_summary"]["scores"]["H84"]["ci95"],
    "withheld": rc["holdout_pooled_summary"]["scores"]["H84"]["withheld_positive_pixels"],
    "verdict": "negative",
    "ok_to_download": "YES for inspection",
    "ok_to_submit": "NO - NOT OK TO SUBMIT (selector not approved, duplicate overlap, literal dense gate fail)"
}, indent=2)+"\n")
print(f"wrote downloads json {downloads_json}")
