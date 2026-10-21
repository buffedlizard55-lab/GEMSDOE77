#!/usr/bin/env python3
"""H80 E2b: Ensemble of tilt + geod + det_elev"""
import numpy as np, rasterio, json, time, hashlib, pathlib, sys
from pathlib import Path
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"vendor"))
from gems57.grid import load_grid as load_grid57
from gems57.holdout import buffered_component_draw
from gems77lia.allocate import greedy_select

BANDS = {1:"mag_anom",2:"rtp",3:"tmi_hg",4:"geod_2ndinv",5:"iso_grav_anom_slope",6:"tc",7:"geod_shearrate",8:"geod_dilaterate",9:"tmi_vg",10:"deq_n100a15",11:"iso_grav_anom_vg",12:"det_elev",13:"iso_grav_anom",14:"tmi",15:"depth_to_base",16:"ieq_n100a15",17:"cond_surf",18:"iso_grav_anom_hg",19:"det_elev_slope"}
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
def clean_band(arr, footprint):
    a=np.asarray(arr,dtype=np.float32)
    bad=(~np.isfinite(a))|(a<-1e30)
    vals=a[footprint & ~bad]
    med=float(np.median(vals)) if vals.size else 0
    return np.where(bad, med, a).astype(np.float32)
def gf(arr,s): return ndimage.gaussian_filter(arr, sigma=s, mode='nearest')
def rank01(arr, footprint):
    out=np.zeros_like(arr,dtype=np.float32)
    v=arr[footprint]
    order=np.argsort(v)
    ranks=np.empty_like(v,dtype=np.float32)
    ranks[order]=np.linspace(0,1,v.size,dtype=np.float32)
    out[footprint]=ranks
    return out
def hessian_ridge(a, sigma):
    axx=ndimage.gaussian_filter(a, sigma, order=(0,2), mode='nearest')
    ayy=ndimage.gaussian_filter(a, sigma, order=(2,0), mode='nearest')
    axy=ndimage.gaussian_filter(a, sigma, order=(1,1), mode='nearest')
    tmp=np.sqrt(np.maximum(((axx-ayy)*0.5)**2+axy**2,0))
    l1=(axx+ayy)*0.5+tmp
    l2=(axx+ayy)*0.5-tmp
    big=np.where(np.abs(l1)>=np.abs(l2),l1,l2)
    small=np.where(np.abs(l1)>=np.abs(l2),l2,l1)
    return (-small*(big<0)).astype(np.float32)

def compute_ensemble():
    grid=load_grid57(ROOT/"data/raw")
    fp=grid.footprint
    log(f"fp {fp.sum()}")
    with rasterio.open(ROOT/"data/raw/training_features.tif") as src:
        tmi=clean_band(src.read(14),fp)
        tmi_vg=clean_band(src.read(9),fp)
        grav=clean_band(src.read(13),fp)
        grav_vg=clean_band(src.read(11),fp)
        depth=clean_band(src.read(15),fp)
        det_elev=clean_band(src.read(12),fp)
        det_slope=clean_band(src.read(19),fp)
        dilat=clean_band(src.read(8),fp)
        shear=clean_band(src.read(7),fp)
        second_inv=clean_band(src.read(4),fp)
    log("bands read")
    def tilt_weighted(field, vg, sigma):
        sm=gf(field, sigma)
        gy,gx=np.gradient(sm)
        thdr=np.hypot(gy,gx).astype(np.float32)
        vg_sm=gf(vg, sigma)
        tilt=np.arctan2(vg_sm, np.maximum(thdr,1e-12)).astype(np.float32)
        zero=(1.0-np.abs(tilt)/(np.pi/2.0)).astype(np.float32)
        zero=np.clip(zero,0,1)
        as_=np.sqrt(thdr*thdr+vg_sm*vg_sm).astype(np.float32)
        as_log=np.log1p(np.maximum(as_,0)).astype(np.float32)
        Z=(zero*as_log).astype(np.float32)
        return Z
    Zmag1=tilt_weighted(tmi, tmi_vg, 1.5)
    Zmag3=tilt_weighted(tmi, tmi_vg, 3.0)
    Zgrav1=tilt_weighted(grav, grav_vg, 1.5)
    Zgrav3=tilt_weighted(grav, grav_vg, 3.0)
    # rank and persist
    def r(a): return rank01(a, fp)
    mag_persist=np.minimum(r(Zmag1), r(Zmag3))
    grav_persist=np.minimum(r(Zgrav1), r(Zgrav3))
    cross=r(mag_persist*grav_persist)
    # basement gate
    sm_depth=gf(depth,2.0)
    dgy,dgx=np.gradient(sm_depth); db_norm=np.hypot(dgy,dgx).astype(np.float32)
    sm_grav=gf(grav,2.0); ggy,ggx=np.gradient(sm_grav); grav_thdr=np.hypot(ggy,ggx).astype(np.float32)
    dot=(dgx*ggx+dgy*ggy); align=(-dot/(db_norm*grav_thdr+1e-9)).astype(np.float32); align=np.clip(align,-1,1); align_pos=np.maximum(align,0)
    base_gate=r(align_pos*np.log1p(np.maximum(db_norm,0)))
    tilt_field=r(cross*(0.6+0.4*base_gate))
    # det_elev ridge (hessian)
    ridge1=hessian_ridge(det_elev,1.0); ridge1=np.maximum(ridge1,0)
    ridge2=hessian_ridge(det_elev,2.5); ridge2=np.maximum(ridge2,0)
    ridge_comb=r(0.6*rank01(ridge1,fp)+0.4*rank01(ridge2,fp))
    # slope front
    # frontness: flat pixel adjacent to steep
    # simple: det_slope ridge
    slope_ridge=r(np.log1p(np.maximum(det_slope,0)))
    # geodetic dilat positive
    dilat_pos=np.maximum(dilat,0).astype(np.float32)
    dilat_n=r(np.log1p(dilat_pos))
    shear_n=r(np.abs(shear))
    second_n=r(second_inv) # second invariant
    geod_comb=r(0.5*dilat_n+0.3*shear_n+0.2*second_n)
    # ensemble: weighted average of proven components + tilt
    # H77-Step high holdout used step tensor + V-LBCO + Radon; our proxy is ridge + geod + tilt
    ensemble = (0.25*tilt_field + 0.35*ridge_comb + 0.20*geod_comb + 0.20*slope_ridge).astype(np.float32)
    ensemble[~fp]=0
    ensemble_ranked=r(ensemble)
    ensemble_ranked[~fp]=0
    log(f"ensemble built mean {ensemble_ranked[fp].mean():.3f}")
    return ensemble_ranked, fp, grid

def holdout(surface, fp, grid):
    from gems57.evaluate_holdout import evaluate as EH_eval, pooled_summary
    folds,_=buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
    folds2,_=buffered_component_draw(grid, seed=21, hide_frac=0.20, buffer_px=3)
    all_folds=folds+folds2
    def make_greedy(surf, n=37654):
        flat=surf.ravel()
        order=np.argsort(-flat)
        order=order[fp.ravel()[order] & np.isfinite(flat[order]) & (flat[order]>0)]
        dist_cat=ndimage.distance_transform_edt(~grid.catalogue)
        allowed=(dist_cat>2)&fp
        # filter order to allowed
        mask_allowed=allowed.ravel()[order]
        order=order[mask_allowed]
        sel=greedy_select(order, min_sep=3.0, limit=n)
        mask=np.zeros(fp.size,dtype=bool); mask[sel]=True
        return mask.reshape(fp.shape)
    cand=make_greedy(surface)
    log(f"cand {cand.sum()}")
    rng=np.random.default_rng(0)
    allowed=(ndimage.distance_transform_edt(~grid.catalogue)>2)&fp
    allowed_idx=np.flatnonzero(allowed.ravel())
    rand_choice=rng.choice(allowed_idx, size=37654, replace=False)
    rand_order=rand_choice[np.argsort(rng.random(len(rand_choice)))]
    rand_sel=greedy_select(rand_order, min_sep=3.0, limit=37654)
    rand=np.zeros(fp.size,dtype=bool); rand[rand_sel]=True; rand=rand.reshape(fp.shape)
    log(f"rand {rand.sum()}")
    # simple control: ridge_comb only (without tilt) to see tilt adds value
    # For now we use grav_thdr as control
    with rasterio.open(ROOT/"data/raw/training_features.tif") as src:
        grav=clean_band(src.read(13),fp)
    sm_grav=gf(grav,2.0); ggy,ggx=np.gradient(sm_grav); grav_thdr=np.hypot(ggy,ggx).astype(np.float32)
    grav_rank=rank01(np.log1p(np.maximum(grav_thdr,0)),fp)
    ctrl=make_greedy(grav_rank)
    log(f"ctrl {ctrl.sum()}")
    terms_by_arm={"candidate":[],"random":[],"grav":[]}
    for fold in all_folds:
        valid=fp
        for name,pred in [("candidate",cand),("random",rand),("grav",ctrl)]:
            res, terms=EH_eval(pred.astype(np.float32), fold, valid, block_side=200)
            terms_by_arm[name].append(terms)
    arms_summed={k: np.sum(np.stack(v,axis=0),axis=0) for k,v in terms_by_arm.items()}
    summary=pooled_summary({k:v for k,v in arms_summed.items()}, draws=2000, candidate="candidate", seed=520810)
    return summary, {"candidate":cand,"random":rand,"grav":ctrl}

def main():
    surface,fp,grid=compute_ensemble()
    np.save(".cache/h80_ensemble.npy", surface)
    # canary quick (reuse previous canary logic but for ensemble field)
    from sklearn.metrics import roc_auc_score
    from scipy.ndimage import distance_transform_edt as edt
    # canary
    from gems57.holdout import build_holdout
    ctx=build_holdout(grid, hide_frac=0.20, seeds=(20,), modes=("all",))
    h=np.zeros(fp.shape,bool)
    for cell in ctx.cells:
        h |= ctx.hidden_by_cell[cell.key]
    h &= fp
    dist_cat=edt(~grid.catalogue).astype(np.float32)
    rng=np.random.default_rng(1020)
    h_w,h_h=fp.shape
    fp_flat=np.flatnonzero(fp.ravel())
    fp_lut=np.full(h_w*h_h,-1,dtype=np.int32); fp_lut[fp_flat]=np.arange(fp_flat.size,dtype=np.int32)
    pos_fp=fp_lut[np.flatnonzero(h.ravel())]
    neg_pool=fp_lut[np.flatnonzero((fp & ~grid.catalogue & (dist_cat>3)).ravel())]
    neg_fp=rng.choice(neg_pool, size=min(200000, neg_pool.size), replace=False)
    can_idx=np.concatenate([pos_fp, neg_fp]); y=np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])
    flat=surface.ravel(); can_flat=fp_flat[can_idx]; vals=flat[can_flat]
    m=np.isfinite(vals)
    auc=float(roc_auc_score(y[m], vals[m])); sym=max(auc,1-auc)
    log(f"ensemble canary auc {auc:.3f} sym {sym:.3f}")
    max_sym=sym
    e1={"max_sym_auc":max_sym, "leakage_flag": bool(max_sym>0.90)}
    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/h80_e1_ensemble.json").write_text(json.dumps(e1,indent=2))
    summary,masks=holdout(surface,fp,grid)
    log(f"candidate DTI {summary['scores']['candidate']['dti']:.4f} CI {summary['scores']['candidate']['ci95']}")
    for k in summary["scores"]:
        log(f" {k}: {summary['scores'][k]['dti']:.4f} {summary['scores'][k]['ci95']}")
    Path("evidence/h80_e2_ensemble.json").write_text(json.dumps(summary,indent=2))
    # build submission regardless (negative deliverable)
    from gems77lia.core import grid as cg
    g2=cg()
    pred=masks["candidate"].astype(np.float32)
    pred[~g2.footprint]=0
    import hashlib, zipfile, shutil, rasterio as rio
    with rio.open(ROOT/"data/raw/sample_submission.tif") as src:
        meta=src.meta.copy(); meta.update(driver="GTiff", dtype="float32", count=1, compress="deflate", tiled=True, nodata=None)
    uid=hashlib.sha256(pred.tobytes()).hexdigest()[:8]
    name=f"gems77-h80-ensemble-{masks['candidate'].sum()}dots-20261021-{uid}-zeros"
    note=f"H80 ensemble tilt+ridge+geod {masks['candidate'].sum()} dots; HOLDOUT-DTI {summary['scores']['candidate']['dti']:.4f} [{summary['scores']['candidate']['ci95'][0]:.4f},{summary['scores']['candidate']['ci95'][1]:.4f}]"
    note=note[:140]
    out_tif=ROOT/f"submissions/{name}.tif"
    with rio.open(out_tif,"w",**meta) as dst: dst.write(pred,1)
    out_zip=ROOT/f"submissions/{name}.zip"
    with zipfile.ZipFile(out_zip,"w",compression=zipfile.ZIP_DEFLATED) as z: z.write(out_tif, arcname=out_tif.name)
    docs_dl=ROOT/"docs/downloads"; docs_dl.mkdir(parents=True,exist_ok=True)
    shutil.copy(out_tif, docs_dl/out_tif.name); shutil.copy(out_zip, docs_dl/out_zip.name)
    # nan variant
    meta_nan=meta.copy(); meta_nan.update(nodata=np.nan)
    pred_nan=np.where(g2.footprint, pred, np.nan).astype(np.float32)
    out_nan=ROOT/f"submissions/{name.replace('-zeros','-nan')}.tif"
    with rio.open(out_nan,"w",**meta_nan) as dst: dst.write(pred_nan,1)
    shutil.copy(out_nan, docs_dl/out_nan.name)
    from gems77lia.validate import validate as validate_sub
    v=validate_sub(out_tif, mode="zeros", name=name, note=note)
    log(f"validator {v['passed']} {v['n_pass']}/{v['n_checks']}")
    run_card={"schema":"gems-parallel-run-card-v1","lane_repo":"buffedlizard55-lab/GEMSDOE77","branch":"arena/de858329-gemsdoe77","hypothesis":"H80 ensemble tilt+ridge+geod","mechanism":"tilt persistence x basement throw + detrended ridge + geod dilat/shear + slope front ensemble","named_non_fault_process":"Lithologic contacts, survey seams, volcanic demag zones","holdout_dti":{"evidence_label":"HOLDOUT-DTI","evaluator_version":"gems57-pooled-hide-v2","score":summary['scores']['candidate']['dti'],"ci_95":summary['scores']['candidate']['ci95'],"withheld_positive_pixels":summary['scores']['candidate']['withheld_positive_pixels'],"vs_random":summary['paired_differences']['random'] if 'random' in summary['paired_differences'] else None},"leakage_canary":{"max_sym_auc":max_sym,"ceiling":0.90,"pass":bool(max_sym<=0.90)},"submission_raster":{"path":str(out_tif),"sha256":hashlib.sha256(out_tif.read_bytes()).hexdigest(),"bytes":out_tif.stat().st_size,"dots":int(pred[g2.footprint].sum()),"validator":v},"submission_name":name,"submission_note":note,"verdict":"negative","safe_to_download_and_submit":False}
    Path("evidence/h80_ensemble_run_card.json").write_text(json.dumps(run_card,indent=2))
    print(json.dumps({"candidate_dti":summary['scores']['candidate']['dti'],"ci":summary['scores']['candidate']['ci95'],"withheld":summary['scores']['candidate']['withheld_positive_pixels'],"file":str(out_tif)},indent=2))

if __name__=="__main__":
    main()
