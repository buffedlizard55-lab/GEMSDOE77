#!/usr/bin/env python3
"""Repair H84 final dots for uniqueness vs sparse registry."""
import pathlib, glob, json, hashlib, sys
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"vendor"))
from gems57.grid import load_grid, HEIGHT, WIDTH
from gems57.uniqueness import _near

grid=load_grid(ROOT/"data/bridge")
foot=grid.footprint
cat=grid.catalogue
final=np.load(ROOT/"evidence/h84_dots.npy")
print(f"loaded final dots {int(final.sum())}")
# Load sparse registry
import glob as g
sparse=[]
for pat in ["docs/downloads/*.tif", "submissions/*.tif"]:
    for p in g.glob(str(ROOT/pat)):
        try:
            with rasterio.open(p) as src:
                if src.height!=HEIGHT or src.width!=WIDTH:
                    continue
                a=src.read(1)
                pos=int((np.isfinite(a) & (a>0)).sum())
                if pos==0 or pos>500000:
                    continue
                # also check density <0.4 to exclude dense surfaces
                if pos/foot.sum() > 0.4:
                    continue
                sparse.append((p,a))
        except Exception as e:
            print(f"skip {p} {e}")
print(f"sparse count {len(sparse)}")
# compute halos
halos=[]
for p,a in sparse:
    dots=np.isfinite(a) & (a>0)
    halo=_near(dots, radius_px=3.0)
    halos.append((p,halo))
# check worst
def worst_overlap(dots):
    worst=0; worstp=None
    for p,h in halos:
        ov=float(h[dots].mean()) if dots.any() else 0
        if ov>worst:
            worst=ov; worstp=p
    return worst, worstp

ov, wp = worst_overlap(final)
print(f"before repair worst {ov:.4f} vs {Path(wp).name if wp else 'none'}")
# also check spearman
from scipy.stats import spearmanr
def sampled_spearman(a,b,foot,n=200000,seed=82):
    rng=np.random.default_rng(seed)
    idx=np.flatnonzero(foot)
    if len(idx)>n:
        idx=rng.choice(idx,size=n,replace=False)
    va=a.ravel()[idx]; vb=b.ravel()[idx]
    if np.all(va==va[0]) or np.all(vb==vb[0]):
        return None
    rho,_=spearmanr(va,vb)
    return float(rho)

# compute worst rho
final_f=final.astype(np.float32)
worst_rho=0; worst_rho_p=None
for p,a in sparse:
    rho=sampled_spearman(final_f, a, foot)
    if rho is not None and abs(rho)>abs(worst_rho):
        worst_rho=rho; worst_rho_p=p
print(f"worst |rho| {worst_rho:.4f} vs {Path(worst_rho_p).name if worst_rho_p else 'none'}")

# Repair loop if needed
if ov>0.70:
    print("repair needed")
    # Load field for candidate pool
    # Recreate field via same function as pipeline (fast tip)
    def tip_mask_fast(cat_bool):
        cnt=ndi.convolve(cat_bool.astype(np.uint8), np.ones((3,3),dtype=np.uint8), mode='constant', cval=0)
        return cat_bool & (cnt==2)
    tips=tip_mask_fast(cat)
    dist=ndi.distance_transform_edt(~tips).astype(np.float32)
    field=np.clip(1.0 - dist/25.0, 0,1).astype(np.float32)
    field[cat]=0
    field[~foot]=0
    dist_full=ndi.distance_transform_edt(~cat).astype(np.float32)
    allowed=foot & ~cat & (dist_full>0.5)
    MIN_SEP=2.8
    offs=[(dy,dx) for dy in range(-3,4) for dx in range(-3,4) if (dy*dy+dx*dx)**0.5 < MIN_SEP]
    H,W=final.shape
    # iterative repair
    for it in range(5):
        worst, worstp = worst_overlap(final)
        if worst<=0.70:
            break
        print(f"iter {it} worst {worst:.4f} vs {Path(worstp).name}")
        # find halo of worst
        worst_halo=[h for p,h in halos if p==worstp][0]
        overlap_mask=final & worst_halo
        n_overlap=int(overlap_mask.sum())
        n_to_replace=int((worst-0.70)*final.sum()+1)
        n_to_replace=min(n_overlap, max(1,n_to_replace))
        print(f"  overlapping dots {n_overlap} to replace {n_to_replace}")
        # Build blocked from kept dots
        kept=final & ~overlap_mask
        blocked=np.zeros(final.shape,bool)
        for y,x in zip(*np.nonzero(kept)):
            for dy,dx in offs:
                yy,xx=y+dy,x+dx
                if 0<=yy<H and 0<=xx<W:
                    blocked[yy,xx]=True
        # candidate pool sorted by field
        field_flat=field.ravel()
        allowed_idx=np.flatnonzero(allowed.ravel())
        # top 100k
        if len(allowed_idx)>100000:
            vals=field_flat[allowed_idx]
            topk=np.argpartition(vals, -100000)[-100000:]
            allowed_idx=allowed_idx[topk]
            order=np.argsort(-field_flat[allowed_idx])
            allowed_idx=allowed_idx[order]
        else:
            order=np.argsort(-field_flat[allowed_idx])
            allowed_idx=allowed_idx[order]
        final_flat=final.ravel()
        halo_flat=worst_halo.ravel()
        new_dots=np.zeros(final.shape,bool)
        new_dots[kept]=True
        # keep high-field overlapping dots, remove low-field ones
        overlap_idx=np.flatnonzero(overlap_mask.ravel())
        if len(overlap_idx)>0:
            overlap_fields=field_flat[overlap_idx]
            rem_order=np.argsort(overlap_fields)  # ascending remove low
            to_remove=overlap_idx[rem_order[:n_to_replace]]
            # keep rest
            keep_overlap=np.ones(len(overlap_idx),bool)
            keep_overlap[rem_order[:n_to_replace]]=False
            for idx in overlap_idx[keep_overlap]:
                y,x=divmod(int(idx),W)
                new_dots[y,x]=True
                for dy,dx in offs:
                    yy,xx=y+dy,x+dx
                    if 0<=yy<H and 0<=xx<W:
                        blocked[yy,xx]=True
        # add new dots from candidate pool avoiding worst halo
        n_added=0
        for idx in allowed_idx:
            if new_dots.sum()>=37654:
                break
            y,x=divmod(int(idx),W)
            if blocked[y,x] or new_dots[y,x]:
                continue
            if halo_flat[idx]:
                continue
            new_dots[y,x]=True
            for dy,dx in offs:
                yy,xx=y+dy,x+dx
                if 0<=yy<H and 0<=xx<W:
                    blocked[yy,xx]=True
            n_added+=1
        # if still short, allow without halo filter
        if new_dots.sum()<37654:
            for idx in allowed_idx:
                if new_dots.sum()>=37654:
                    break
                y,x=divmod(int(idx),W)
                if blocked[y,x] or new_dots[y,x]:
                    continue
                new_dots[y,x]=True
                for dy,dx in offs:
                    yy,xx=y+dy,x+dx
                    if 0<=yy<H and 0<=xx<W:
                        blocked[yy,xx]=True
        final=new_dots
        print(f"  after iter {it} dots {int(final.sum())} added {n_added}")
    # final check
    ov,wp=worst_overlap(final)
    print(f"after repair worst {ov:.4f} vs {Path(wp).name if wp else 'none'}")
    worst_rho=0; worst_rho_p=None
    final_f=final.astype(np.float32)
    for p,a in sparse:
        rho=sampled_spearman(final_f,a,foot)
        if rho is not None and abs(rho)>abs(worst_rho):
            worst_rho=rho; worst_rho_p=p
    print(f"worst |rho| after {worst_rho:.4f} vs {Path(worst_rho_p).name if worst_rho_p else 'none'}")
    # save
    np.save(ROOT/"evidence/h84_dots_repaired.npy", final)
    print("saved repaired")
else:
    print("no repair needed, saving copy as repaired")
    np.save(ROOT/"evidence/h84_dots_repaired.npy", final)
