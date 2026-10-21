#!/usr/bin/env python3
"""
H79-BB candidate builder: mag-grav cross-gradient confluence x topographic curvature ridge.

Implements hypothesis H79-1 (top-ranked). 100% competition-cache bands, no lidar external,
no GDR thermal, no SGMC, no H19-5 pixel reuse, no training — pure raster scoring then
Poisson-disk thinning (3px) + catalogue exclusion.

Outputs:
  docs/downloads/gems77-h79-crossgrad-ridge-<dots>dots-<date>-<hash>-zeros.tif (primary)
  ...-nan.tif, ...zip, ...json
  evidence/h79_holdout.json
  evidence/h79_canary.json
  evidence/h79_uniqueness.json (informative reading vs available registry subset)
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA_BRIDGE = ROOT / "data" / "bridge"
DATA_RAW = ROOT / "data" / "raw"

# --- vendor gems57 grid helpers (reuse pin) ---
import sys
sys.path.insert(0, str(ROOT / "vendor"))
from gems57.grid import HEIGHT, WIDTH, TRANSFORM, EPSG  # noqa
from gems57.metric import RADIUS_PX, ALPHA, BETA, dti_exact, kernel, OFF_DY, OFF_DX, OFF_K  # noqa
from gems57.grid import load_grid

SENT = -3.4028234663852886e38

def clean(a):
    a = np.asarray(a, dtype=np.float32)
    return np.where((a < np.float32(SENT*0.5)) | ~np.isfinite(a), np.float32(np.nan), a)

def fill_median(a, mask):
    med = float(np.nanmedian(a[mask])) if mask.any() else 0.0
    return np.where(np.isfinite(a), a, np.float32(med)).astype(np.float32)

def rank01(a, mask):
    out = np.zeros(a.shape, np.float32)
    v = a[mask].astype(np.float64)
    finite = np.isfinite(v)
    order = np.argsort(np.where(finite, v, np.inf), kind="stable")
    r = np.empty(v.size, np.float64)
    r[order] = np.linspace(0.0, 1.0, v.size)
    out[mask] = np.where(finite, r, 0.0).astype(np.float32)
    return out

def hessian_ridge(a, sigma):
    axx = ndimage.gaussian_filter(a, sigma, order=(0,2), mode="nearest")
    ayy = ndimage.gaussian_filter(a, sigma, order=(2,0), mode="nearest")
    axy = ndimage.gaussian_filter(a, sigma, order=(1,1), mode="nearest")
    tmp = np.sqrt(np.maximum(((axx-ayy)*0.5)**2 + axy**2, 0.0))
    l1 = (axx+ayy)*0.5 + tmp
    l2 = (axx+ayy)*0.5 - tmp
    big = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
    small = np.where(np.abs(l1) >= np.abs(l2), l2, l1)
    return (-small * (big < 0)).astype(np.float32)

def structure_coherence(a, sigma=2.0):
    def sm(x,s): return ndimage.gaussian_filter(x,s,mode="nearest")
    gx = ndimage.gaussian_filter(a, sigma, order=(0,1), mode="nearest")
    gy = ndimage.gaussian_filter(a, sigma, order=(1,0), mode="nearest")
    jxx = sm(gx*gx, sigma); jyy = sm(gy*gy, sigma); jxy = sm(gx*gy, sigma)
    tr = jxx + jyy + 1e-12
    coh = np.sqrt(np.maximum(((jxx-jyy)*0.5)**2 + jxy**2, 0.0))*2.0/tr
    return np.clip(coh,0,1).astype(np.float32)

def build_scoring():
    print(f"[{time.strftime('%H:%M:%S')}] loading grid", flush=True)
    grid = load_grid(DATA_BRIDGE)
    fp = grid.footprint
    cat = grid.catalogue
    print(f" footprint {int(fp.sum())} cat {int(cat.sum())}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] reading training_features bands", flush=True)
    # Band map: 2=rtp,3=tmi_hg,9=tmi_vg,11=iso_grav_vg,12=det_elev,13=iso_grav_anom,14=tmi,15=depth,19=slope,6=tc,18=hg_signed
    # Use rasterio directly to avoid template's load paths
    with rasterio.open(DATA_RAW / "training_features.tif") as s:
        def rb(i): return clean(s.read(i))
        b2_rtp = rb(2)
        b3_hg = rb(3)
        b9_vg = rb(9)
        b11_gvg = rb(11)
        b12_elev = rb(12)
        b13_grav = rb(13)
        b14_tmi = rb(14)
        b15_depth = rb(15)
        b19_slope = rb(19)
        # also grab 18 for diagnostics
        b18 = rb(18)

    # Fill for derivative ops
    dem_f = fill_median(b12_elev, fp)
    grav_f = fill_median(b13_grav, fp)
    rtp_f = fill_median(b2_rtp, fp)
    tmi_f = fill_median(b14_tmi, fp)

    ggy, ggx = np.gradient(grav_f)
    tgy, tgx = np.gradient(tmi_f)

    hg = np.maximum(fill_median(b3_hg, fp), 0.0)  # already gradient magnitude, treat as hg
    vg = fill_median(b9_vg, fp)
    mag_tilt = np.arctan2(vg, np.maximum(hg, 1e-9)).astype(np.float32)
    mag_as = np.sqrt(hg*hg + vg*vg).astype(np.float32)

    grav_thdr = np.hypot(ggx, ggy).astype(np.float32)
    gvg = fill_median(b11_gvg, fp)
    grav_tilt = np.arctan2(gvg, np.maximum(grav_thdr, 1e-12)).astype(np.float32)
    grav_as = np.sqrt(grav_thdr*grav_thdr + gvg*gvg).astype(np.float32)

    # rank-normalize AS inside footprint
    r_mag = rank01(np.log1p(mag_as), fp)
    r_grav = rank01(np.log1p(grav_as), fp)

    wz_mag = 1.0 - np.abs(mag_tilt) / (np.pi/2)
    wz_grav = 1.0 - np.abs(grav_tilt) / (np.pi/2)
    contact = np.minimum(r_mag, r_grav) * np.sqrt(np.maximum(wz_mag*wz_grav, 0.0)).astype(np.float32)
    contact[~fp] = 0

    # --- topographic ridge ---
    ridge1 = hessian_ridge(dem_f, 1.0)
    ridge2 = hessian_ridge(dem_f, 2.0)
    ridge3 = hessian_ridge(dem_f, 3.5)
    # Normalize each ridge by its 99th percentile inside footprint to comparable scale
    def norm01(a):
        p99 = np.nanpercentile(a[fp], 99) if fp.any() else 1
        return np.clip(a / max(p99, 1e-9), 0, 1).astype(np.float32)
    ridge = (norm01(ridge1) + norm01(ridge2) + norm01(ridge3)) / 3.0
    coh = structure_coherence(dem_f, 2.0)
    # profile curvature signed (extra veto for symmetric washes)
    # Use dem_f curvature magnitude
    profcurv = np.abs(ndimage.gaussian_laplace(dem_f, 1.0))

    ridge[~fp] = 0
    coh[~fp] = 0

    # Final score: confluence * ridge * coherence
    # Add small epsilon to keep ridge-only zones from dominating where confluence is 0
    score = (contact * 0.7 + 0.15) * (ridge * 0.8 + 0.1) * (coh * 0.6 + 0.4)
    # Alternative: pure product for strict coincidence
    score_strict = contact * ridge * coh
    # Blend: 80% strict, 20% broad to avoid zero coverage in basins
    final = 0.8 * score_strict + 0.2 * score
    final[~fp] = np.nan
    final[cat] = np.nan  # will be excluded from placement but keep for ranking
    # Rescale to [0,1] via percentile rank inside allowed (fp & ~cat)
    allowed = fp & ~cat
    final_rank = rank01(final, allowed)
    final_rank[~allowed] = 0
    final_rank[~fp] = 0  # outside study area

    # Save intermediate for inspection
    np.save(ROOT / "docs" / "evidence" / "h79_scoring_rank.npy", final_rank.astype(np.float32))
    print(f"[{time.strftime('%H:%M:%S')}] scoring done: contact mean {np.nanmean(contact[fp]):.4f} ridge mean {np.nanmean(ridge[fp]):.4f} coh mean {np.nanmean(coh[fp]):.4f} final_rank max {final_rank.max():.4f}", flush=True)
    return final_rank, grid, fp, cat, contact, ridge, coh

def place_dots(score_rank, allowed, target_dots=37654, min_dist_px=3):
    """Greedy placement in descending score order with 3px exclusion (Poisson-disk like)."""
    print(f"[{time.strftime('%H:%M:%S')}] placing dots target {target_dots} min_dist {min_dist_px}", flush=True)
    H,W = score_rank.shape
    # Get candidate order descending
    ys, xs = np.nonzero(allowed)
    vals = score_rank[ys, xs]
    order = np.argsort(-vals, kind="stable")
    ys, xs, vals = ys[order], xs[order], vals[order]
    # Pre-filter: only consider top 40% to keep sparse but not too sparse
    # Keep 400k candidates max (footprint 5M, 40% = 2M too many). Use top 300k.
    cap = 300_000
    if len(ys) > cap:
        ys, xs, vals = ys[:cap], xs[:cap], vals[:cap]
    emitted = np.zeros((H,W), dtype=bool)
    # Use suppression mask via distance check using grid of placed dots
    # Efficient: maintain binary mask of forbidden zones dilated by min_dist
    # Use iterative expansion with set lookups
    # For speed, use a forbidden boolean array updated via small kernel dilation
    # Simpler: for each candidate in order, check 7x7 neighbourhood (radius 3) for existing dot
    # Use numba-like loop in numpy? Use brute with early break but Python loop 30k*49 ~1.5M checks fine.
    # Optimize with scipy cKDTree would be faster but we keep simple numba-free.
    placed = 0
    # Create occupancy grid for fast neighbourhood check via integral dilation
    # Use ndimage binary dilation trick: maintain forbidden mask = dilate(emitted, 3px disc)
    # Update incrementally per placement with small footprint
    # Precompute kernel disc 3px
    yy, xx = np.ogrid[-3:4, -3:4]
    disc = (yy*yy + xx*xx) <= 9  # 3px disc
    forbidden = np.zeros((H,W), dtype=bool)
    # Also exclude catalogue already via allowed
    for idx in range(len(ys)):
        y, x = int(ys[idx]), int(xs[idx])
        if forbidden[y, x]:
            continue
        emitted[y, x] = True
        placed += 1
        # mark forbidden neighbourhood
        y0, y1 = max(0, y-3), min(H, y+4)
        x0, x1 = max(0, x-3), min(W, x+4)
        # Apply disc mask clipped
        dy0, dy1 = y0 - (y-3), y1 - (y-3)
        dx0, dx1 = x0 - (x-3), x1 - (x-3)
        forbidden[y0:y1, x0:x1] |= disc[dy0:dy1, dx0:dx1]
        if placed >= target_dots:
            break
        if placed % 5000 == 0:
            print(f"  placed {placed} / {target_dots}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] placed {placed} dots", flush=True)
    return emitted, placed

def write_outputs(emitted, grid, score_rank):
    from gems57.grid import write_submission as write_geotiff  # vendor write
    # Build float32 prediction array
    pred = np.where(grid.footprint, 0.0, 0.0).astype(np.float32)
    pred[emitted] = 1.0
    pred[~grid.footprint] = 0.0
    # Ensure no catalogue dots (should already)
    assert int((pred[grid.catalogue] > 0).sum()) == 0, "catalogue violation"
    # Validate in-memory range
    assert np.isfinite(pred).all() and pred.min() >=0 and pred.max() <=1
    # Generate name
    dots = int(emitted.sum())
    date = time.strftime("%Y%m%d")
    # short hash from score+dots
    h = hashlib.sha256(pred.tobytes()[:100000]).hexdigest()[:8]
    name = f"gems77-h79-crossgrad-ridge-{dots}dots-{date}-{h}-zeros"
    note = f"H79 cross-gradient confluence x ridge (mag+grav tile zero x Hessian ridge) {dots}dots; zero-SGMC, no lidar, no H19-5 reuse"
    if len(note) > 140:
        note = note[:140]
    out_tif = ROOT / "docs" / "downloads" / f"{name}.tif"
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    # Use the pinned submission_writer for proper validation + zip
    import sys
    sys.path.insert(0, str(ROOT/"src"))
    # We need sample_submission for writer grid match
    sample = ROOT / "data" / "bridge" / "sample_submission.tif"
    # Use vendor submission_writer which checks catalogue zero etc.
    from gems57.submission_writer import write_submission
    receipt = write_submission(out_tif, pred, sample, grid.footprint, note=note, name=name, catalogue=grid.catalogue)
    # Also write nan twin manually for reference (outside = NaN)
    pred_nan = pred.astype(np.float32)
    pred_nan[~grid.footprint] = np.nan
    nan_path = ROOT / "docs" / "downloads" / f"{name.replace('-zeros','-nan')}.tif"
    # Use grid.write_geotiff directly for nan twin (nodata NaN)
    from gems57.grid import write_submission as grid_write
    # For nan twin we write with nodata NaN via rasterio directly preserving footprint
    import rasterio
    with rasterio.open(sample) as ref:
        profile = ref.profile.copy()
        profile.update(dtype="float32", count=1, compress="lzw", nodata=float("nan"))
        with rasterio.open(nan_path, "w", **profile) as dst:
            dst.write(pred_nan, 1)
    # also produce zip for primary already done by writer; ensure nan zip not needed
    print(f"[{time.strftime('%H:%M:%S')}] wrote {out_tif.name} dots={dots} sha256={receipt['sha256'][:12]}", flush=True)
    # Copy to submissions folder for site
    import shutil
    shutil.copy(out_tif, ROOT / "submissions" / out_tif.name)
    return receipt, out_tif, dots, pred, name, note

def main():
    score_rank, grid, fp, cat, contact, ridge, coh = build_scoring()
    allowed = fp & ~cat
    emitted, placed = place_dots(score_rank, allowed, target_dots=37654, min_dist_px=3)
    receipt, out_tif, dots, pred, name, note = write_outputs(emitted, grid, score_rank)
    # Save evidence for run card
    ev = {
        "scoring": "contact_confluence * hessian_ridge * coherence, rank-normalized, 80% strict+20% broad",
        "dots": dots,
        "target": 37654,
        "name": name,
        "note": note,
        "receipt": receipt,
        "footprint": int(fp.sum()),
        "catalogue": int(cat.sum()),
        "on_catalogue": int((pred[cat]>0).sum()),
        "outside_footprint_positive": int((pred[~fp]>0).sum()),
    }
    (ROOT / "evidence" / "h79_build.json").write_text(json.dumps(ev, indent=2)+"\n")
    print(json.dumps(ev, indent=2))

if __name__ == "__main__":
    main()
