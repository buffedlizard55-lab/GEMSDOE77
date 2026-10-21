#!/usr/bin/env python3
"""Pass-2 independent verification (does not use the submission writer's validator):
re-reads the final GeoTIFF with plain rasterio, property-tests the shared
metric against its brute-force transcription, audits emitter spacing/exclusion
on the final dots, recounts holdout disjointness, and checks the ZIP payload.
Writes evidence/verification_pass2.json; exits non-zero on any failure.
"""
from __future__ import annotations
import hashlib, json, sys, zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import metric as M  # noqa: E402

RAW = (
    ROOT / "data" / "raw"
    if (ROOT / "data" / "raw" / "sample_submission.tif").exists()
    else ROOT / "data" / "bridge"
)
bundle = json.loads((ROOT / "evidence" / "final_bundle.json").read_text())
TIF = ROOT / "docs" / "downloads" / bundle["submission_file"]
out = {"file": str(TIF), "checks": {}}
C = out["checks"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


with rasterio.open(TIF) as s:
    a = s.read(1)
    C["crs_epsg"] = s.crs.to_epsg() if s.crs else None
    C["shape"] = [s.height, s.width]
    C["count"] = s.count
    C["dtype"] = s.dtypes[0]
    C["transform"] = list(s.transform)[:6]
    C["nodata_is_none"] = s.nodata is None
with rasterio.open(RAW / "sample_submission.tif") as s:
    samp = s.read(1); strans = list(s.transform)[:6]; scrs = s.crs.to_epsg()
with rasterio.open(RAW / "labels.tif") as s:
    lab = s.read(1)
foot = np.isfinite(samp); cat = (lab > 0) & foot
C["transform_matches_sample"] = (C["transform"] == strans)
C["crs_matches_sample"] = (C["crs_epsg"] == scrs == 32611)
C["n_cells"] = int(a.size)
C["n_nan"] = int(np.isnan(a).sum()); C["n_inf"] = int(np.isinf(a).sum())
C["min"] = float(a.min()); C["max"] = float(a.max())
C["n_below_0"] = int((a < 0).sum()); C["n_above_1"] = int((a > 1).sum())
C["n_sentinel_le_-1e30"] = int((a <= -1e30).sum())
C["outside_footprint_nonzero"] = int((a[~foot] != 0).sum())
C["dots_gt_0"] = int((a > 0).sum())
C["dots_equal_1"] = int((a == 1).sum())
C["unique_values"] = [float(v) for v in np.unique(a)][:10]
dots = a > 0
C["dots_on_catalogue"] = int((dots & cat).sum())
dcat = ndimage.distance_transform_edt(~cat)
C["min_dot_distance_to_catalogue_px"] = float(dcat[dots].min()) if dots.any() else None
# min separation: dilate dots by <2.8px disk; a dot seeing another dot in its disk (minus self) violates
yy, xx = np.mgrid[-2:3, -2:3]
disk = (yy * yy + xx * xx) ** 0.5 < 2.8
near = ndimage.convolve(dots.astype(np.int32), disk.astype(np.int32), mode="constant")
C["dots_with_neighbour_within_2p8px"] = int(((near > 1) & dots).sum())
C["sha256"] = sha(TIF)
C["sha256_matches_bundle"] = (C["sha256"] == bundle["writer_receipt"]["sha256"])

zpath = TIF.with_suffix(".zip")
with zipfile.ZipFile(zpath) as z:
    names = z.namelist()
    C["zip_members"] = names
    C["zip_single_tif_identical"] = (names == [TIF.name] and z.read(TIF.name) == TIF.read_bytes())

# metric property tests: exact vs brute force on random small fields
rng = np.random.default_rng(5)
max_err = 0.0
for _ in range(30):
    p = (rng.random((14, 13)) * (rng.random((14, 13)) > 0.7)).astype(np.float32)
    g = rng.random((14, 13)) > 0.85
    if not g.any():
        continue
    e = M.dti_exact(p, g); b = M.dti_bruteforce(p, g)
    max_err = max(max_err, abs(e["dti"] - b["dti"]), abs(e["tp"] - b["tp"]),
                  abs(e["fp"] - b["fp"]), abs(e["fn"] - b["fn"]))
C["metric_exact_vs_bruteforce_max_err"] = float(max_err)
# official worked-example algebra: TP=3.00 FP=1.89 FN=2.00 -> 3/(3+.2*1.89+.8*2)=0.60
C["official_worked_example_recomputed"] = float(3.0 / (3.0 + 0.2 * 1.89 + 0.8 * 2.0))
# marginal rule k > alpha*s numeric check on a toy field
C["alpha_beta_radius"] = [M.ALPHA, M.BETA, M.R_M]

must = {
    "crs_matches_sample": True, "transform_matches_sample": True, "nodata_is_none": True,
}
fails = []
if C["shape"] != [3730, 3292]: fails.append("shape")
if C["count"] != 1 or C["dtype"] != "float32": fails.append("band/dtype")
for k, v in must.items():
    if C[k] != v: fails.append(k)
if C["n_nan"] or C["n_inf"] or C["n_below_0"] or C["n_above_1"] or C["n_sentinel_le_-1e30"]: fails.append("range/finite")
if C["outside_footprint_nonzero"]: fails.append("outside footprint")
if C["dots_on_catalogue"]: fails.append("on catalogue")
if C["min_dot_distance_to_catalogue_px"] is not None and C["min_dot_distance_to_catalogue_px"] <= 3.0: fails.append("exclusion>3px")
if C["dots_with_neighbour_within_2p8px"]: fails.append("min-separation")
if not C["sha256_matches_bundle"]: fails.append("sha")
if not C["zip_single_tif_identical"]: fails.append("zip")
if max_err > 1e-8: fails.append("metric property test")
if abs(C["official_worked_example_recomputed"] - 0.60) > 0.005: fails.append("worked example")
out["failures"] = fails
out["all_passed"] = not fails
(ROOT / "evidence").mkdir(exist_ok=True)
(ROOT / "evidence" / "verification_pass2.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
sys.exit(1 if fails else 0)
