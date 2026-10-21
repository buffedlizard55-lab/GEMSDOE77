#!/usr/bin/env python3
"""H84: build the unique inspection TIF from the Exp2 (OR-gate) production surface.
Recomputes deterministically: collar(dist to FULL catalogue) * (0.5+0.5*max(rank magHG, rank gravHG)),
37,654 dots, min_sep 2.8 px, shared fail-closed writer. Verdict negative (see run card) - NOT OK TO SUBMIT."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"vendor")); sys.path.insert(0,str(ROOT/"src"))
from gems57 import grid as G  # noqa
from gems57.submission_writer import write_submission  # noqa
from gems77 import emit77  # noqa
from gems77.features77 import clean, rank01  # noqa

def collar_fn(d):
    d=np.asarray(d,np.float32); p=np.zeros_like(d)
    m1=(d>1)&(d<=3); p[m1]=np.sin(np.pi/2*(d[m1]-1)/2)**2
    m2=(d>3)&(d<=12); p[m2]=np.cos(np.pi/2*(d[m2]-3)/9)**2
    return p

def main():
    grid=G.load_grid(ROOT/"data"/"bridge"); foot,cat=grid.footprint,grid.catalogue
    with rasterio.open(ROOT/"data"/"raw"/"training_features.tif") as ds:
        mag=clean(ds.read(3)); grav=clean(ds.read(13))
    gy,gx=np.gradient(np.where(np.isfinite(grav),grav,np.nanmedian(grav[foot])))
    gravhg=np.hypot(gx,gy).astype(np.float32)
    gate=(0.5+0.5*np.maximum(rank01(mag,foot),rank01(gravhg,foot))).astype(np.float32)
    dcat=ndi.distance_transform_edt(~cat).astype(np.float32)
    score=(collar_fn(dcat)*gate).astype(np.float32); score[~foot]=0; score[cat]=0
    allowed=foot&~cat&(dcat>0.5)
    dots=emit77.emit_dots(score,allowed,37654,min_sep=2.8)
    pred=dots.astype(np.float32)
    digest=hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()
    name=f"gems77-h84-orgate-collar-37654dots-20261021-{digest[:8]}-zeros"
    note="H84 OR-gated second-strand collar 37654 dots; HOLDOUT-DTI 0.1215 vs incumbent 0.1527: NEGATIVE do not upload"
    assert len(name)<=140 and len(note)<=140, (len(name),len(note))
    out=ROOT/"docs"/"downloads"/f"{name}.tif"
    receipt=write_submission(out,pred,sample=ROOT/"data"/"bridge"/"sample_submission.tif",footprint=foot,catalogue=cat,note=note,name=name,
        metadata=dict(lane_id="H84-ORGATE-COLLAR",holdout_dti=0.12148954698767672,holdout_ci95=[0.10706251690747319,0.13550906533658974],
        withheld_positive_pixels=11342,evaluator_version="gems57-pooled-hide-v2",incumbent_holdout_dti=0.152718,verdict="negative",dots=int(pred.sum())))
    print(json.dumps({"file":receipt["file"],"sha256":receipt["sha256"],"zip":receipt["zip_file"],"zip_sha256":receipt["zip_sha256"],"name":name,"note":note},indent=1))
if __name__=="__main__": main()
