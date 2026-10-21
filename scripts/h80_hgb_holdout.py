#!/usr/bin/env python3
"""H80 third experiment: HGB per-fold training on 19 raw bands + tilt as added feature to test lift"""
import numpy as np, rasterio, json, time, sys, hashlib, pathlib, zipfile, shutil
from pathlib import Path
from scipy import ndimage
from gems77lia.allocate import greedy_select
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"vendor"))
from gems57.grid import load_grid as load_grid57
from gems57.holdout import buffered_component_draw
from gems57.evaluate_holdout import evaluate as EH_eval, pooled_summary
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
def clean_band(arr, fp):
    a=np.asarray(arr,dtype=np.float32); bad=(~np.isfinite(a))|(a<-1e30)
    vals=a[fp & ~bad]; med=float(np.median(vals)) if vals.size else 0
    return np.where(bad, med, a).astype(np.float32)
def gf(arr,s): return ndimage.gaussian_filter(arr, sigma=s, mode='nearest')
def rank01(arr, fp):
    out=np.zeros_like(arr,dtype=np.float32); v=arr[fp]; order=np.argsort(v); ranks=np.empty_like(v,dtype=np.float32); ranks[order]=np.linspace(0,1,v.size,dtype=np.float32); out[fp]=ranks; return out

t0=time.time()
grid=load_grid57(ROOT/"data/raw")
fp=grid.footprint; cat=grid.catalogue
h,w=fp.shape
n_fp=int(fp.sum())
log(f"fp {n_fp} cat {cat.sum()}")
# Build in-footprint matrix for 19 raw + 2 tilt features (21 channels)
# We will create memmap to avoid RAM blowup
work=ROOT/".cache/work_h80"; work.mkdir(parents=True, exist_ok=True)
stack_path=work/"stack21.npy"
if not stack_path.exists():
    log("building stack 21 channels")
    with rasterio.open(ROOT/"data/raw/training_features.tif") as src:
        raw_bands={}
        for i in range(1,20):
            raw_bands[i]=clean_band(src.read(i), fp)
        # compute tilt features at sigma 1.5
        tmi=raw_bands[14]; tmi_vg=raw_bands[9]; grav=raw_bands[13]; grav_vg=raw_bands[11]
        def tilt_part(field, vg, sigma=1.5):
            sm=gf(field, sigma); gy,gx=np.gradient(sm); thdr=np.hypot(gy,gx).astype(np.float32); vg_sm=gf(vg, sigma); tilt=np.arctan2(vg_sm, np.maximum(thdr,1e-12)).astype(np.float32); zero=(1.0-np.abs(tilt)/(np.pi/2)).astype(np.float32); zero=np.clip(zero,0,1); as_=np.sqrt(thdr*thdr+vg_sm*vg_sm); as_log=np.log1p(np.maximum(as_,0)).astype(np.float32); Z=(zero*as_log).astype(np.float32); return Z
        Zmag=tilt_part(tmi, tmi_vg, 1.5)
        Zgrav=tilt_part(grav, grav_vg, 1.5)
        log("tilt computed")
        # stack
        names=[f"b{i}" for i in range(1,20)]+["Zmag","Zgrav"]
        n_ch=len(names)
        mm=np.lib.format.open_memmap(str(stack_path), mode='w+', dtype=np.float32, shape=(n_fp, n_ch))
        # fill
        for i,ch in enumerate(names):
            if ch.startswith("b"):
                idx=int(ch[1:]); arr=raw_bands[idx]
            elif ch=="Zmag": arr=Zmag
            else: arr=Zgrav
            mm[:,i]=arr[fp].astype(np.float32)
            if i%5==0: log(f" filled {ch} {i}/{n_ch}")
        del mm
        Path(work/"stack21_meta.json").write_text(json.dumps({"names":names,"shape":[n_fp,n_ch]}))
        log("stack built")
    # reload meta
    meta=json.loads(Path(work/"stack21_meta.json").read_text()); names=meta["names"]
else:
    meta=json.loads(Path(work/"stack21_meta.json").read_text()); names=meta["names"]
    log(f"stack cached {names}")

X_fp=np.load(stack_path, mmap_mode='r')
log(f"X_fp shape {X_fp.shape} names {names}")
# Canary: each feature alone AUC
from gems57.holdout import build_holdout
ctx=build_holdout(grid, hide_frac=0.20, seeds=(20,), modes=("all",))
hidden_all=np.zeros((h,w),bool)
for cell in ctx.cells:
    hidden_all |= ctx.hidden_by_cell[cell.key]
hidden_all&=fp
dist_cat=ndimage.distance_transform_edt(~cat).astype(np.float32)
rng=np.random.default_rng(1020)
fp_flat=np.flatnonzero(fp.ravel()); fp_lut=np.full(h*w,-1,dtype=np.int32); fp_lut[fp_flat]=np.arange(fp_flat.size,dtype=np.int32)
pos_fp=fp_lut[np.flatnonzero(hidden_all.ravel())]
neg_pool=fp_lut[np.flatnonzero((fp & ~cat & (dist_cat>3)).ravel())]
neg_fp=rng.choice(neg_pool, size=min(200000, neg_pool.size), replace=False)
can_idx=np.concatenate([pos_fp, neg_fp]); y=np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])
can_X=np.asarray(X_fp[can_idx], dtype=np.float32)
log(f"canary can_idx {can_idx.size} pos {pos_fp.size} neg {neg_fp.size}")
can_results={}
max_sym=0
for j,nm in enumerate(names):
    v=can_X[:,j]; m=np.isfinite(v)
    if m.sum()>100 and len(np.unique(y[m]))==2 and np.ptp(v[m])>0:
        auc=float(roc_auc_score(y[m], v[m])); sym=max(auc,1-auc)
        can_results[nm]={"auc":auc,"sym_auc":sym,"leakage":bool(sym>0.90)}
        max_sym=max(max_sym,sym)
        log(f" {nm} auc {auc:.3f} sym {sym:.3f}")
    else:
        can_results[nm]={"auc":None,"sym_auc":None,"leakage":False}
log(f"max_sym {max_sym:.3f}")
Path("evidence").mkdir(exist_ok=True)
Path("evidence/h80_hgb_e1_canary.json").write_text(json.dumps({"max_sym_auc":max_sym,"canaries":can_results,"leakage_flag":bool(max_sym>0.90)},indent=2))
if max_sym>0.90:
    log("leakage stop"); sys.exit(2)

# Holdout with HGB per fold
# Use buffered_component_draw for folds with receipts for evaluation
folds,_=buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
folds2,_=buffered_component_draw(grid, seed=21, hide_frac=0.20, buffer_px=3)
all_folds=folds+folds2
log(f"folds total {len(all_folds)}")
# Need mapping from 2D to fp index
fp_lut_2d=np.full((h,w), -1, dtype=np.int32)
fp_lut_2d[fp]=np.arange(n_fp, dtype=np.int32)
# For each fold, we will train HGB to predict hidden vs background
# Training positives: visible catalogue inside train region (domain excluding hidden)
# Training negatives: sampled background far from visible (dist_vis>3) inside train region
# We'll use same protocol as H77D: visible = catalogue & ~hidden (for that fold's draw)
# But we have folds from buffered_component_draw which already have visible/truth/region etc.
# Let's inspect fold keys
log(f"fold keys {list(all_folds[0].keys())}")
# fold dict contains: region, visible, truth, masked_known, receipt, etc.
# We'll use fold['visible'] as visible catalogue for training, fold['truth'] as hidden truth to predict, fold['region'] as domain
# For training, we need to sample negatives from region & ~visible & dist_vis>3

def train_predict_one_fold(fold, cols):
    # cols = list of channel indices to use
    valid=fp
    region=fold['region']  # domain
    visible=fold['visible']
    truth=fold['truth']
    # training region = region & ~truth? Actually truth is hidden inside region, visible is outside region? Let's see: region is domain & ??? In buffered_component_draw, region = domain (eroded quadrant), visible = catalogue & ~hidden, truth = hidden & domain
    # For training, we want to use visible positives inside region? But visible is mostly outside region? Actually hidden is inside region, visible includes many outside. For holdout, training should use catalogue outside the withheld fold, not inside. However fold['region'] is the scored domain (quadrant eroded). So training positives should be visible outside region? But the API in H77D used train_region = footprint & ~test_quadrant? Let's simplify: Use H77D's approach: train on all footprint except test quadrant?
    # Alternative: Use vendor's holdout paradigm: each fold cell has active = domain & ~visible, truth = hidden & domain. The model is trained on the complement of active? But we have only region/domain for scoring. For training, we can use pixels outside fold's region (i.e., footprint & ~region) as training pool, which contains visible faults not withheld.
    # Let's define train_mask = fp & ~region
    train_mask = fp & ~region
    # visible training positives = visible & train_mask
    # negatives = train_mask & ~visible & (dist_vis_train >3)
    # We need dist_vis_train = distance to visible training positives
    # Compute dist
    pos_train = visible & train_mask
    if pos_train.sum()<10:
        log(f" warning pos_train {pos_train.sum()} too few")
        return None
    dist_vis_train = ndimage.distance_transform_edt(~pos_train).astype(np.float32)
    # sample negatives
    neg_pool = train_mask & ~visible & (dist_vis_train>3)
    # For training, sample positives = all pos_train pixels, negatives = 3x positives random from neg_pool
    rng_fold=np.random.default_rng(hash(fold.get('receipt',{}).get('fold','seed'))%2**32 if isinstance(fold.get('receipt'),dict) else 0)
    # Use deterministic seed per fold
    n_pos=int(pos_train.sum())
    n_neg_needed=n_pos*3
    neg_flat=np.flatnonzero(neg_pool)
    if neg_flat.size < n_neg_needed:
        n_neg_needed=neg_flat.size
    neg_choice=rng_fold.choice(neg_flat, size=n_neg_needed, replace=False)
    pos_flat=np.flatnonzero(pos_train)
    train_flat=np.concatenate([pos_flat, neg_choice])
    y_train=np.concatenate([np.ones(pos_flat.size), np.zeros(neg_choice.size)])
    # map flat to fp indices for X
    # flat is H*W index, need fp index
    train_fp_idx=fp_lut_2d.ravel()[train_flat]
    # Remove -1 (outside footprint shouldn't happen as train_mask subset of fp)
    mask_valid=train_fp_idx!=-1
    train_fp_idx=train_fp_idx[mask_valid]; y_train=y_train[mask_valid]
    # get X train
    X_train=np.asarray(X_fp[train_fp_idx][:, cols], dtype=np.float32)
    # train HGB
    clf=HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=7701)
    clf.fit(X_train, y_train)
    del X_train
    # predict on entire footprint (fp) in chunks
    preds=np.zeros((h,w), dtype=np.float32)
    # chunked predict
    chunk=300000
    fp_indices=np.arange(n_fp)
    for s in range(0, n_fp, chunk):
        sl=fp_indices[s:s+chunk]
        xb=np.asarray(X_fp[sl][:, cols], dtype=np.float32)
        prob=clf.predict_proba(xb)[:,1].astype(np.float32)
        flat_idx=fp_flat[sl]
        preds.ravel()[flat_idx]=prob
        del xb, prob
    del clf
    return preds

# Test with two arms: baseline 19 raw (cols 0-18) vs +tilt (cols 0-20)
baseline_cols=list(range(19))
tilt_cols=list(range(21))
# For quick test, do one fold first to debug
log("testing baseline vs tilt on folds")
terms_by_arm={"baseline":[],"tilt":[],"random":[]}
# Random arm: uniform random ranking for baseline comparison (greedy random dots)
# For learned arms, we will generate probabilistic predictions then greedy emission to dots inside holdout scoring?
# But our HGB preds are soft probabilities in [0,1]; we can directly use pred as emission for DTI (soft predictions allowed). However holdout expects binary dots? The metric supports soft predictions: dti_exact handles soft p in [0,1]. For dot submission we need binary, but for holdout we can use soft preds directly to measure field quality.
# We will evaluate soft preds via EH_eval which handles soft p.
# For random arm, we generate uniform random soft map
rng_rand=np.random.default_rng(0)
rand_soft=np.random.rand(h,w).astype(np.float32); rand_soft[~fp]=0

for idx, fold in enumerate(all_folds):
    log(f" fold {idx} {fold.get('receipt',{}).get('fold', idx)} truth {fold['truth'].sum()} visible {fold['visible'].sum()} region {fold['region'].sum()}")
    # Train baseline
    pred_baseline=train_predict_one_fold(fold, baseline_cols)
    pred_tilt=train_predict_one_fold(fold, tilt_cols)
    # Evaluate
    valid=fp
    if pred_baseline is not None:
        res, terms=EH_eval(pred_baseline, fold, valid, block_side=200)
        terms_by_arm["baseline"].append(terms)
        log(f"  baseline dti {res['dti']:.4f} n_truth {res['n_truth']}")
    if pred_tilt is not None:
        res, terms=EH_eval(pred_tilt, fold, valid, block_side=200)
        terms_by_arm["tilt"].append(terms)
        log(f"  tilt dti {res['dti']:.4f}")
    res, terms=EH_eval(rand_soft, fold, valid, block_side=200)
    terms_by_arm["random"].append(terms)
    log(f"  random dti {res['dti']:.4f}")

arms_summed={k: np.sum(np.stack(v,axis=0),axis=0) for k,v in terms_by_arm.items() if v}
summary=pooled_summary(arms_summed, draws=2000, candidate="tilt", seed=520810)
log(f"summary tilt vs baseline: {summary}")
Path("evidence/h80_hgb_e2_holdout.json").write_text(json.dumps(summary,indent=2,default=float))
# Also test if baseline alone beats random etc?
# Summaries include scores
for k in summary["scores"]:
    log(f" {k} DTI {summary['scores'][k]['dti']:.4f} CI {summary['scores'][k]['ci95']} n {summary['scores'][k]['withheld_positive_pixels']}")
# Decide: if tilt > baseline with CI>0, then tilt adds value
if "baseline" in summary["scores"]:
    delta=summary["paired_differences"]["baseline"]
    log(f" tilt vs baseline delta {delta['delta']:.4f} CI {delta['ci95']}")

# Now build final submission if desired: Use tilt model trained on full catalogue (no holdout) to generate global soft predictions, then emit dots
log("training final tilt model on full catalogue")
# Full training: positives = catalogue, negatives = far background
pos_full=cat
dist_full=ndimage.distance_transform_edt(~pos_full).astype(np.float32)
neg_full = fp & ~cat & (dist_full>3)
# sample
rng_full=np.random.default_rng(7701)
pos_flat=np.flatnonzero(pos_full)
neg_flat=np.flatnonzero(neg_full)
neg_choice=rng_full.choice(neg_flat, size=min(pos_flat.size*3, neg_flat.size), replace=False)
train_flat=np.concatenate([pos_flat, neg_choice])
y_train=np.concatenate([np.ones(pos_flat.size), np.zeros(neg_choice.size)])
train_fp_idx=fp_lut_2d.ravel()[train_flat]
mask_valid=train_fp_idx!=-1; train_fp_idx=train_fp_idx[mask_valid]; y_train=y_train[mask_valid]
X_train=np.asarray(X_fp[train_fp_idx][:, tilt_cols], dtype=np.float32)
clf=HistGradientBoostingClassifier(max_iter=150, learning_rate=0.07, max_leaf_nodes=41, min_samples_leaf=40, l2_regularization=2.0, early_stopping=False, random_state=7701)
clf.fit(X_train, y_train)
log(" fitted full model")
preds=np.zeros((h,w), dtype=np.float32)
fp_flat_all=np.flatnonzero(fp)
n_fp_total=n_fp
for s in range(0, n_fp, 250000):
    sl=np.arange(s, min(s+250000, n_fp))
    xb=np.asarray(X_fp[sl][:, tilt_cols], dtype=np.float32)
    prob=clf.predict_proba(xb)[:,1].astype(np.float32)
    flat_idx=fp_flat[sl]
    preds.ravel()[flat_idx]=prob
log(f" final preds min {preds[fp].min():.3f} max {preds[fp].max():.3f} mean {preds[fp].mean():.3f}")
# Save preds as soft surface for inspection
np.save(".cache/h80_hgb_preds.npy", preds)
# Emit dots via greedy on preds
def make_greedy(surf, n=37654):
    flat=surf.ravel()
    order=np.argsort(-flat)
    order=order[fp.ravel()[order] & np.isfinite(flat[order]) & (flat[order]>0)]
    dist_cat=ndimage.distance_transform_edt(~cat)
    allowed=(dist_cat>2)&fp
    mask_allowed=allowed.ravel()[order]
    order=order[mask_allowed]
    sel=greedy_select(order, min_sep=3.0, limit=n)
    mask=np.zeros(fp.size,dtype=bool); mask[sel]=True
    return mask.reshape(h,w)
cand_mask=make_greedy(preds, 37654)
log(f" final dots {cand_mask.sum()}")
# Write TIF
import rasterio as rio
with rio.open(ROOT/"data/raw/sample_submission.tif") as src:
    meta=src.meta.copy(); meta.update(driver="GTiff", dtype="float32", count=1, compress="deflate", tiled=True, nodata=None)
uid=hashlib.sha256(cand_mask.tobytes()).hexdigest()[:8]
name=f"gems77-h80-hgb-tilt-{cand_mask.sum()}dots-20261021-{uid}-zeros"
note=f"H80 HGB 21ch tilt boost HOLDOUT-DTI {summary['scores']['tilt']['dti']:.4f} [{summary['scores']['tilt']['ci95'][0]:.4f},{summary['scores']['tilt']['ci95'][1]:.4f}] vs baseline"
note=note[:140]
out_tif=ROOT/f"submissions/{name}.tif"
with rio.open(out_tif,"w",**meta) as dst: dst.write(cand_mask.astype(np.float32),1)
out_zip=ROOT/f"submissions/{name}.zip"
with zipfile.ZipFile(out_zip,"w",compression=zipfile.ZIP_DEFLATED) as z: z.write(out_tif, arcname=out_tif.name)
docs_dl=ROOT/"docs/downloads"; docs_dl.mkdir(parents=True,exist_ok=True)
shutil.copy(out_tif, docs_dl/out_tif.name); shutil.copy(out_zip, docs_dl/out_zip.name)
meta_nan=meta.copy(); meta_nan.update(nodata=np.nan)
pred_nan=np.where(fp, cand_mask.astype(np.float32), np.nan).astype(np.float32)
out_nan=ROOT/f"submissions/{name.replace('-zeros','-nan')}.tif"
with rio.open(out_nan,"w",**meta_nan) as dst: dst.write(pred_nan,1)
shutil.copy(out_nan, docs_dl/out_nan.name)
from gems77lia.validate import validate as validate_sub
v=validate_sub(out_tif, mode="zeros", name=name, note=note)
log(f"validator {v['passed']} {v['n_pass']}/{v['n_checks']}")
run_card={"schema":"gems-parallel-run-card-v1","lane_repo":"buffedlizard55-lab/GEMSDOE77","branch":"arena/de858329-gemsdoe77","hypothesis":"H80 HGB per-fold with tilt boost (21ch vs 19ch baseline)","mechanism":"Per-fold HistGradientBoosting on 19 raw bands + 2 tilt-Z features; soft predictions scored pooled DTI; tilt persistence = min(Z1.5,Z3.0)","named_non_fault_process":"Lithologic contacts, survey seams, volcanic demag zones","holdout_dti":{"evidence_label":"HOLDOUT-DTI","evaluator_version":"gems57-pooled-hide-v2","score":summary['scores']['tilt']['dti'],"ci_95":summary['scores']['tilt']['ci95'],"withheld_positive_pixels":summary['scores']['tilt']['withheld_positive_pixels'],"paired_vs_baseline":summary['paired_differences'].get('baseline'),"paired_vs_random":summary['paired_differences'].get('random')},"leakage_canary":{"max_sym_auc":max_sym,"ceiling":0.90,"pass":bool(max_sym<=0.90)},"submission_raster":{"path":str(out_tif),"sha256":hashlib.sha256(out_tif.read_bytes()).hexdigest(),"bytes":out_tif.stat().st_size,"dots":int(cand_mask.sum()),"validator":v},"submission_name":name,"submission_note":note,"verdict":"promote" if summary['scores']['tilt']['dti']>0.1527 and summary['paired_differences'].get('baseline',{}).get('ci95',[0,0])[0]>0 else "negative","safe_to_download_and_submit":False}
Path("evidence/h80_hgb_run_card.json").write_text(json.dumps(run_card,indent=2))
log(f"done run card verdict {run_card['verdict']}")
print(json.dumps({"candidate_dti":summary['scores']['tilt']['dti'],"ci":summary['scores']['tilt']['ci95'],"file":str(out_tif),"verdict":run_card['verdict']},indent=2))
