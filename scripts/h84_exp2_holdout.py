#!/usr/bin/env python3
"""H84 Exp1: geophysics-gated second-strand collar, hide-and-recover holdout.
Preregistered registry/preregistration_h84.json. Arms (matched 37,654 dots, seed 30):
H84_gated vs H83_collar (geometry-only control) vs A0_random. Evaluator gems57-pooled-hide-v2.
Saves evidence/h84_exp2_holdout.json + evidence/h84_exp2_surface.npy (full-catalogue production score)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"vendor")); sys.path.insert(0, str(ROOT/"src"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH, metric as M  # noqa
from gems77 import emit77  # noqa
from gems77.features77 import clean, rank01  # noqa
BRIDGE = ROOT/"data"/"bridge"; BUDGET = 37654; SEED = 30

def collar(d):
    d = np.asarray(d, np.float32); p = np.zeros_like(d)
    m1 = (d>1)&(d<=3); p[m1]=np.sin(np.pi/2*(d[m1]-1)/2)**2
    m2 = (d>3)&(d<=12); p[m2]=np.cos(np.pi/2*(d[m2]-3)/9)**2
    return p

def auc_sym(v,y):
    from sklearn.metrics import roc_auc_score
    a=float(roc_auc_score(y,v)); return max(a,1-a)

def main():
    t0=time.time(); log=lambda *a: print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
    grid=G.load_grid(BRIDGE); foot,cat=grid.footprint,grid.catalogue
    with rasterio.open(ROOT/"data"/"raw"/"training_features.tif") as ds:
        mag=clean(ds.read(3)); grav=clean(ds.read(13))
    gy,gx=np.gradient(np.where(np.isfinite(grav),grav,np.nanmedian(grav[foot])))
    gravhg=np.hypot(gx,gy).astype(np.float32)
    gate=(0.5+0.5*np.maximum(rank01(mag,foot),rank01(gravhg,foot))).astype(np.float32)
    log("gate built", gate.shape)
    folds,_=HO.buffered_component_draw(grid,seed=SEED,hide_frac=0.20,buffer_px=3)
    hidden=folds[0]["hidden_all"]; visible=folds[0]["visible"]
    dvis=ndi.distance_transform_edt(~visible).astype(np.float32)
    dcat=ndi.distance_transform_edt(~cat).astype(np.float32)
    scores={"H84_exp2_orgate": collar(dvis)*gate, "H83_collar": collar(dvis)}
    # canaries
    rng=np.random.default_rng(1234)
    neg=np.flatnonzero(foot&~cat&(dcat>3)); pos=np.flatnonzero(hidden)
    nidx=rng.choice(neg,size=min(200000,neg.size),replace=False); idx=np.concatenate([pos,nidx])
    y=np.concatenate([np.ones(pos.size),np.zeros(nidx.size)])
    feats={"collar_visible_only":scores["H83_collar"],"mag_hg":mag,"grav_hg":gravhg,"gate":gate,"gated_score":scores["H84_exp2_orgate"]}
    can={k:auc_sym(np.nan_to_num(v.ravel()[idx]),y) for k,v in feats.items()}
    log("canaries",can)
    # emission per fold + evaluation
    areas={f["name"]:int(f["region"].sum()) for f in folds}; tot=float(sum(areas.values()))
    per={f["name"]:max(1,int(round(BUDGET*areas[f["name"]]/tot))) for f in folds}
    arms={}
    terms_by={}
    for arm in ["H84_exp2_orgate","H83_collar","A0_random"]:
        dots=np.zeros(foot.shape,bool)
        for f in folds:
            allowed=f["region"]&foot&~cat&(dvis>0.5)
            if arm=="A0_random":
                r=np.random.default_rng(999); s=(r.random(foot.shape)<0.05).astype(np.float32)
            else: s=np.where(foot&f["region"],scores[arm],0).astype(np.float32)
            dots|=emit77.emit_dots(s,allowed,per[f["name"]],min_sep=2.8)
        dots&=foot; terms=None; pf={}
        for f in folds:
            res,t=EH.evaluate(dots.astype(np.float32),f,valid=foot,block_side=200)
            pf[f["name"]]={"dti":res["dti"],"n_truth":int(res["n_truth"]),"emitted":int(res["n_emitted"])}
            terms=t if terms is None else terms+t
        terms_by[arm]=terms
        arms[arm]={"dots":int(dots.sum()),"per_fold":pf,"pooled":float(M.dti_from_components(terms[:,0].sum(),terms[:,1].sum(),terms[:,2].sum()))}
        log(arm, arms[arm]["pooled"], arms[arm]["dots"])
        if arm=="H84_exp2_orgate": np.save(ROOT/"evidence"/"h84_exp2_dots.npy",dots)
    summ=EH.pooled_summary(terms_by,draws=1000,seed=520810,candidate="H84_exp2_orgate")
    # production surface: full catalogue
    prod=(collar(dcat)*gate).astype(np.float32); prod[~foot]=0; prod[cat]=0
    np.save(ROOT/"evidence"/"h84_exp2_surface.npy",prod)
    out={"evidence_class":"HOLDOUT-DTI","evaluator_version":EH.VERSION,"seed":SEED,"budget":BUDGET,
         "withheld_positive_pixels":int(hidden.sum()),"canary_auc":can,"canary_worst":max(can.values()),
         "leakage_flag_gt_0p90":max(can.values())>0.90,"arms":arms,"pooled_summary":summ,
         "elapsed_s":time.time()-t0}
    (ROOT/"evidence"/"h84_exp2_holdout.json").write_text(json.dumps(out,indent=1))
    log("H84 HOLDOUT-DTI",summ["scores"]["H84_exp2_orgate"])
if __name__=="__main__": main()
