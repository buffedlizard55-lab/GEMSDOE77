#!/usr/bin/env python3
"""H83-PHASE 3-experiment pipeline — magnetic phase-congruency buried contacts"""
import argparse, gc, hashlib, json, sys, time, subprocess
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH, metric as M, uniqueness as UQ
from gems57.validate import validate
from gems57.submission_writer import write_submission
from gems77 import emit77
from gems83_phase.fields import H83_PHASE_NAMES, build_phase_fields, clean
from gems83_phase.fields import fill_nearest as gf_fill

RAW = ROOT / "data" / "raw"
BRIDGE = ROOT / "data" / "bridge"
PROC = ROOT / "data" / "processed" / "h83_phase"
EVID = ROOT / "evidence"
BUDGET, EXCL = 37654, 2
HGB_KW = dict(max_iter=140, learning_rate=0.09, max_leaf_nodes=31, min_samples_leaf=40, early_stopping=False, random_state=83)
BAND_NAMES = ["mag_anom","rtp","tmi_hg","geod_2ndinv","iso_grav_anom_slope","tc","geod_shearrate","geod_dilaterate","tmi_vg","deq_n100a15","iso_grav_anom_vg","det_elev","iso_grav_anom","tmi","depth_to_base_surf","ieq_n100a15","cond_surf","iso_grav_anom_hg","det_elev_slope"]
SEED = 20

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_grid():
    return G.load_grid(BRIDGE)

def build_stack(foot):
    PROC.mkdir(parents=True, exist_ok=True)
    path = PROC / "h83_stack.npy"
    spath = PROC / "h83_phase_stack_full.npy"
    if path.exists() and spath.exists():
        # verify shape matches footprint
        try:
            stack = np.load(path, mmap_mode="r")
            surf = np.load(spath)
            if stack.shape[:2]==foot.shape and surf.shape==foot.shape:
                log(f"stack cached {stack.shape} phase_stack {surf.shape}")
                return stack, surf
        except Exception as e:
            log(f"cache miss {e}")
    log("building stack from training_features.tif")
    stack = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(3730,3292,26))
    bands_full = {}
    band_map = {"mag_anom":1,"rtp":2,"tmi_hg":3,"tc":6,"tmi_vg":9,"tmi":14}
    with rasterio.open(RAW/"training_features.tif") as ds:
        assert ds.count==19
        for b in range(1,20):
            arr = clean(ds.read(b))
            # store first 19 channels in stack 0..18
            stack[:,:,b-1] = np.where(np.isfinite(arr), arr, np.nanmedian(arr[foot&np.isfinite(arr)]) if np.isfinite(arr[foot]).any() else 0)
            if b in band_map.values():
                for name, idx in band_map.items():
                    if idx==b:
                        bands_full[name]=arr
                        log(f"  band {b} {BAND_NAMES[b-1]} kept for phase")
            else:
                log(f"  band {b} {BAND_NAMES[b-1]} cached")
    # build phase fields from full grids
    fields = build_phase_fields(bands_full, foot)
    for j, name in enumerate(H83_PHASE_NAMES):
        stack[:,:,19+j] = fields[name]
        log(f"  phase {name} mean {float(fields[name][foot].mean()):.4f}")
    surf = fields["phase_stack"]
    # fill NaN in stack where outside footprint? Keep as 0 outside
    stack[~np.repeat(foot[:,:,None],26,axis=2)]=0
    stack.flush()
    np.save(spath, surf)
    log(f"stack saved {stack.shape} phase_stack saved")
    # return memmap + full surface
    return np.load(path, mmap_mode="r"), surf

def fetch_h33():
    dest = RAW/"registry"/"gemsdoe32-h33.tif"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            with dest.open("wb") as out:
                subprocess.run(["gh","api","repos/buffedlizard55-lab/GEMSDOE32/contents/docs/downloads/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif","-H","Accept: application/vnd.github.raw"], stdout=out, check=True)
        except Exception as e:
            log(f"H33 fetch failed {e}"); dest.unlink(missing_ok=True); return None
    h = sha_file(dest)
    log(f"H33 sha {h[:12]} expected c55bafc4")
    return dest

def local_registry_files(exclude=None):
    files, seen = [], set()
    excl_hash = sha_file(exclude) if exclude and exclude.exists() else None
    for d in (ROOT/"docs"/"downloads", ROOT/"submissions"):
        for p in sorted(d.glob("*.tif")):
            if excl_hash and p.name==exclude.name:
                continue
            h = sha_file(p)
            if h in seen or h==excl_hash:
                continue
            seen.add(h); files.append((p,h))
    return files

def halo_frac(dots, foot):
    return float(UQ._near(dots)[foot].mean()) if dots.any() else 0.0

# Exp1
def exp1():
    t0=time.time()
    grid=load_grid(); foot,cat=grid.footprint, grid.catalogue
    stack, phase_stack = build_stack(foot)
    folds,_ = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden, visible = folds[0]["hidden_all"], folds[0]["visible"]
    log(f"seed{SEED} withheld {int(hidden.sum())} visible {int(visible.sum())}")
    rng = np.random.default_rng(83)
    fp_idx = np.flatnonzero(foot)
    samp = rng.choice(fp_idx, 200_000, replace=False)
    # band identity check: tmi_hg vs gradmag tmi
    tmi = np.asarray(stack[:,:,BAND_NAMES.index("tmi")], np.float32)
    gy,gx = np.gradient(tmi); gmag=np.hypot(gy,gx).ravel()
    ident = {"spearman_tmi_hg_vs_gradmag": float(spearmanr(np.asarray(stack[:,:,2]).ravel()[samp], gmag[samp]).statistic), "n_sample": int(samp.size)}
    log(f"band identities {ident}")
    dist_cat = ndimage.distance_transform_edt(~cat)
    pos = np.flatnonzero(hidden); neg_pool = np.flatnonzero(foot & ~cat & (dist_cat>3))
    neg = rng.choice(neg_pool, min(200_000, neg_pool.size), replace=False)
    idx = np.concatenate([pos,neg]); y=np.r_[np.ones(pos.size), np.zeros(neg.size)]
    canaries={}
    for j,name in enumerate(BAND_NAMES+H83_PHASE_NAMES):
        col = j if j <19 else 19+(j-19)
        # stack is (3730,3292,26) need to index correctly: for phase names, index 19..25
        if name in H83_PHASE_NAMES:
            v = np.asarray(stack[:,:, 19+H83_PHASE_NAMES.index(name)]).ravel()[idx]
        else:
            v = np.asarray(stack[:,:, BAND_NAMES.index(name)]).ravel()[idx]
        auc=float(roc_auc_score(y,v)); canaries[name]={"auc":auc, "sym_auc": max(auc,1-auc)}
    worst = max(c["sym_auc"] for c in canaries.values())
    log(f"canaries worst symAUC {worst:.4f}")
    if worst>0.90:
        raise SystemExit(f"STOP leakage canary {worst:.4f}>0.90")
    # Phase-1 surface uniqueness
    h33 = fetch_h33()
    mine_s = phase_stack.ravel()[samp]
    rows=[]
    candidates = local_registry_files()
    if h33:
        candidates.append((h33, sha_file(h33)))
    for p,h in candidates:
        with rasterio.open(p) as ds: a=ds.read(1)
        a=np.where(np.isfinite(a),a,0).astype(np.float32)
        rho=float(spearmanr(mine_s, a.ravel()[samp]).statistic) if np.ptp(a) else float("nan")
        rows.append({"file": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p), "sha256":h, "dots": int((a>0).sum()), "spearman_sampled_surface": rho})
        del a
    rows.sort(key=lambda r: -abs(r["spearman_sampled_surface"]) if np.isfinite(r["spearman_sampled_surface"]) else 0)
    max_rho = max(abs(r["spearman_sampled_surface"]) for r in rows if np.isfinite(r["spearman_sampled_surface"]))
    log(f"surface rho max |rho| {max_rho:.4f}")
    if max_rho>0.90:
        raise SystemExit(f"STOP surface duplicate |rho| {max_rho:.4f}")
    out={"experiment":1, "lane":"H83-PHASE", "seed": SEED, "withheld_positives": int(hidden.sum()), "band_identities": ident, "canaries": canaries, "canary_worst_sym_auc": worst, "canary_pass_lt_090": bool(worst<=0.90), "surface_uniqueness": {"scope":"local docs/downloads+submissions+ H33 sampled n=200k", "n_compared": len(rows), "max_abs_rho": max_rho, "top5": rows[:5]}, "seconds": round(time.time()-t0,1)}
    (EVID/"h83_phase_e1_canary_surface.json").write_text(json.dumps(out, indent=2)+"\n")
    log("Exp1 done")
    return out

def _train_predict(stack, cols, idx, y, region):
    # stack is memmap (3730,3292,26)
    flat_rows = np.array([stack[r,c,cols] for r,c in zip(*np.unravel_index(idx, stack.shape[:2]))], dtype=np.float32) if len(cols)>0 else np.zeros((len(idx),0))
    # alternative faster: use stack reshape
    # But memmap reshape is heavy; use direct indexing
    # Let's do proper: stack reshaped to (3730*3292,26) then idx
    # idx are flat indices into foot? Actually pos/neg are flat indices of full grid (3730*3292) where foot true. Our earlier idx from np.flatnonzero(hidden) are flat indices of full grid. So we can use them to index stack flat.
    # Recreate flat view
    flat_stack = stack.reshape(-1, stack.shape[2])
    X = flat_stack[idx][:, cols].astype(np.float32)
    clf = HistGradientBoostingClassifier(**HGB_KW); clf.fit(X, y)
    del X; gc.collect()
    out=np.zeros(stack.shape[:2], np.float32)
    ys,xs = np.nonzero(region)
    if ys.size==0:
        return out
    r0,r1,c0,c1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    sub = np.array(stack[r0:r1, c0:c1][:,:,cols], np.float32)
    pred=np.zeros(sub.shape[:2], np.float32)
    for i in range(0, sub.shape[0], 200):
        blk = sub[i:i+200].reshape(-1, sub.shape[2])
        if blk.shape[0]==0: continue
        pred[i:i+200] = clf.predict_proba(blk)[:,1].reshape(blk.shape[0]//sub.shape[1], sub.shape[1])
    out[r0:r1,c0:c1]=np.where(region[r0:r1,c0:c1], pred, 0.0)
    del sub, pred, clf; gc.collect()
    return out

def exp2():
    t0=time.time()
    grid=load_grid(); foot,cat=grid.footprint, grid.catalogue
    stack, phase_stack = build_stack(foot)
    folds,_ = HO.buffered_component_draw(grid, seed=SEED, hide_frac=0.20, buffer_px=3)
    hidden, visible = folds[0]["hidden_all"], folds[0]["visible"]
    dist_cat = ndimage.distance_transform_edt(~cat)
    dist_vis = ndimage.distance_transform_edt(~visible)
    rng=np.random.default_rng(1234)
    samples=[]
    for k,fold in enumerate(folds):
        train_region=np.zeros_like(foot)
        for j,f2 in enumerate(folds):
            if j!=k: train_region |= f2["region"]
        pos=np.flatnonzero(visible & train_region)
        if pos.size>40000: pos=rng.choice(pos,40000,replace=False)
        neg_mask = train_region & ~cat & (dist_cat>3)
        hard=np.flatnonzero(neg_mask & (dist_cat<=15)); easy=np.flatnonzero(neg_mask & (dist_cat>15))
        nh=min(hard.size, pos.size); ne=min(easy.size, 2*pos.size - nh)
        neg=np.concatenate([rng.choice(hard, nh, replace=False) if nh>0 else np.array([],dtype=int), rng.choice(easy, ne, replace=False) if ne>0 else np.array([],dtype=int)])
        samples.append((np.concatenate([pos,neg]), np.r_[np.ones(pos.size,np.int8), np.zeros(neg.size,np.int8)]))
        log(f"fold {fold['name']} pos {pos.size} neg {neg.size} truth {int(fold['truth'].sum())}")
    # define arms
    IDX={n:i for i,n in enumerate(BAND_NAMES+H83_PHASE_NAMES)}
    arms_cols={"A_phase": [IDX[n] for n in H83_PHASE_NAMES], "A_std19": [IDX[n] for n in BAND_NAMES], "A_cand": list(range(26))}
    surfaces={"S_phase": phase_stack}
    for arm, cols in arms_cols.items():
        cache=PROC/f"h83_oof_{arm}.npy"
        if cache.exists():
            surfaces[arm]=np.load(cache); log(f"{arm} cached OOF")
            continue
        surf=np.zeros(foot.shape, np.float32)
        for k,fold in enumerate(folds):
            idx,y = samples[k]
            surf += _train_predict(stack, cols, idx, y, fold["region"])
            log(f"{arm}/{fold['name']} done {time.time()-t0:.0f}s")
        np.save(cache, surf); surfaces[arm]=surf
    def emit_pooled(p):
        allowed=[f["region"] & foot & ~visible & (dist_vis>EXCL) for f in folds]
        tot=sum(int(a.sum()) for a in allowed); dots=np.zeros(foot.shape,bool)
        for a in allowed:
            b=int(round(BUDGET * int(a.sum())/max(tot,1)))
            if b>0: dots |= emit77.emit_dots(np.clip(p,0,1), a, b)
        return dots
    def score(dots):
        terms, per = None, {}
        for f in folds:
            res,t = EH.evaluate(dots.astype(np.float32), f, valid=foot, block_side=200)
            per[f["name"]]={"dti": res["dti"], "n_truth": res["n_truth"], "emitted": res["n_emitted"]}
            terms = t if terms is None else terms + t
        return per, terms
    terms_by_arm, results = {}, {}
    order=["S_phase","A_phase","A_std19","A_cand"]
    for arm in order:
        dots=emit_pooled(surfaces[arm]); per, terms = score(dots)
        terms_by_arm[arm]=terms
        results[arm]={"dots": int(dots.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        np.save(PROC/f"h83_dots_{arm}.npy", dots)
        log(f"{arm} HOLDOUT-DTI {results[arm]['pooled_holdout_dti']:.4f}")
    rng2=np.random.default_rng(999)
    dots=emit_pooled((rng2.random(foot.shape,dtype=np.float32)*foot))
    per, terms=score(dots); terms_by_arm["A0_random"]=terms
    results["A0_random"]={"dots": int(dots.sum()), "per_fold": per, "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
    log(f"A0_random HOLDOUT-DTI {results['A0_random']['pooled_holdout_dti']:.4f}")
    summary=EH.pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate="A_cand")
    winner=max(("S_phase","A_phase","A_cand"), key=lambda a: results[a]["pooled_holdout_dti"])
    out={"experiment":2, "evidence_class":"HOLDOUT-DTI", "evaluator_version": EH.VERSION, "alpha": M.ALPHA, "beta": M.BETA, "kernel_radius_m": M.R_M, "seed": SEED, "withheld_positive_pixels_total_draw": int(hidden.sum()), "fold_receipts": [f["receipt"] for f in folds], "operating_point_fixed": {"budget": BUDGET, "exclusion_px": EXCL, "min_sep_px":2.8, "scan":"none"}, "arms": results, "pooled_summary": summary, "preregistered_winner": winner, "incumbent": {"dti":0.152718, "evaluator":"gems57-pooled-hide-v2","seed":20,"withheld":11321,"ci95":[0.131548,0.175185]}, "beats_incumbent_point_estimate": bool(results[winner]["pooled_holdout_dti"]>0.152718), "seconds": round(time.time()-t0,1)}
    (EVID/"h83_phase_e2_holdout.json").write_text(json.dumps(out, indent=2)+"\n")
    log(f"Exp2 winner {winner}")
    return out

def exp3():
    t0=time.time()
    e2=json.loads((EVID/"h83_phase_e2_holdout.json").read_text())
    winner=e2["preregistered_winner"]
    grid=load_grid(); foot,cat=grid.footprint, grid.catalogue
    stack, phase_stack = build_stack(foot)
    dist_cat=ndimage.distance_transform_edt(~cat)
    if winner=="S_phase":
        surf=phase_stack; fit_note="pure phase_stack physics (no refit)"
    else:
        cols={"A_phase": [BAND_NAMES.index(n) if n in BAND_NAMES else 19+H83_PHASE_NAMES.index(n) for n in H83_PHASE_NAMES], "A_cand": list(range(26))}[winner] if winner!="A_cand" else list(range(26))
        # Actually need correct cols: for A_phase it's phase only indices 19..25, for A_cand 0..25
        if winner=="A_phase":
            cols=list(range(19,26))
        elif winner=="A_cand":
            cols=list(range(26))
        else:
            cols=list(range(19))
        rng=np.random.default_rng(77)
        cache=PROC/"h83_final_surface.npy"
        if cache.exists():
            surf=np.load(cache); fit_note="cached full-catalogue refit"
        else:
            pos=np.flatnonzero(cat)
            neg_mask=foot & ~cat & (dist_cat>3)
            hard=np.flatnonzero(neg_mask & (dist_cat<=15)); easy=np.flatnonzero(neg_mask & (dist_cat>15))
            neg=np.concatenate([rng.choice(hard, min(hard.size,pos.size),replace=False), rng.choice(easy, min(easy.size,pos.size),replace=False)])
            idx=np.concatenate([pos,neg]); y=np.r_[np.ones(pos.size,np.int8), np.zeros(neg.size,np.int8)]
            # _train_predict needs region = foot
            # Build flat training using same helper
            flat_stack=stack.reshape(-1, stack.shape[2])
            X=flat_stack[idx][:,cols].astype(np.float32)
            clf=HistGradientBoostingClassifier(**HGB_KW); clf.fit(X,y)
            # predict full region
            surf=np.zeros(foot.shape, np.float32)
            r0,c0=0,0; r1,c1=3730,3292
            sub=np.array(stack[r0:r1,c0:c1][:,:,cols], np.float32)
            # predict in chunks
            for i in range(0, sub.shape[0], 200):
                blk=sub[i:i+200].reshape(-1, sub.shape[2])
                surf[i:i+200]=clf.predict_proba(blk)[:,1].reshape(blk.shape[0]//sub.shape[1], sub.shape[1])
            surf=np.where(foot, surf, 0).astype(np.float32)
            np.save(cache, surf); fit_note=f"full-catalogue refit pos {pos.size} neg {neg.size}"
        log(fit_note)
    allowed=foot & (dist_cat>EXCL)
    dots=emit77.emit_dots(np.clip(surf,0,1).astype(np.float32), allowed, BUDGET)
    pred=dots.astype(np.float32)
    h8=sha_bytes(dots.tobytes())[:8]
    name=f"gems77-h83-phase-37654dots-20261021-{h8}-zeros"
    note=f"H83-PHASE magnetic phase-congruency + std19 HGB LOQO 37654 dots >2px from catalogue phase_stack+coh"
    assert len(note)<=140 and len(name)<=140
    dest=ROOT/"docs"/"downloads"/f"{name}.tif"
    sample=BRIDGE/"sample_submission.tif"
    if dest.exists():
        receipt=json.loads(dest.with_suffix(".json").read_text()); log("TIF exists reusing receipt")
    else:
        receipt=write_submission(dest, pred, sample, foot, note=note, name=name, catalogue=cat, metadata={"lane":"H83-PHASE","winner_arm":winner,"fit":fit_note})
    for suf in (".tif",".zip",".json"):
        src=dest.with_suffix(suf); dst=ROOT/"submissions"/src.name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
    val=validate(dest, foot, cat)
    rows=[]
    for p,h in local_registry_files(exclude=dest):
        with rasterio.open(p) as ds: a=ds.read(1)
        a=np.where(np.isfinite(a),a,0).astype(np.float32)
        theirs=a>0
        rows.append({"file": p.name, "sha256": h, "their_dots": int(theirs.sum()), "halo_bg_frac": halo_frac(theirs, foot), "forward_overlap_3px": UQ.dot_overlap(pred, a)})
        del a
    h33p=fetch_h33()
    h33_row=None
    if h33p and h33p.exists():
        with rasterio.open(h33p) as ds: a=ds.read(1)
        a=np.where(np.isfinite(a),a,0).astype(np.float32)
        h33_row={"file": "GEMSDOE32 "+h33p.name, "sha256": sha_file(h33p), "their_dots": int((a>0).sum()), "halo_bg_frac": halo_frac(a>0, foot), "forward_overlap_3px": UQ.dot_overlap(pred, a)}
        rows.append(h33_row); del a
    informative=[r for r in rows if r["halo_bg_frac"]<0.70]
    uniq={"scope":"local docs/downloads+submissions + H33; NOT complete family registry","n_compared": len(rows),"n_informative_lt70_halo": len(informative),"max_overlap_literal": max(r["forward_overlap_3px"] for r in rows) if rows else 0,"max_overlap_informative": max((r["forward_overlap_3px"] for r in informative), default=0.0),"literal_firings_gt070": sum(r["forward_overlap_3px"]>0.70 for r in rows),"informative_firings_gt070": sum(r["forward_overlap_3px"]>0.70 for r in informative),"vs_h33": h33_row, "top5_informative": sorted(informative, key=lambda r: -r["forward_overlap_3px"])[:5] if informative else [], "rows": rows}
    out={"experiment":3, "winner_arm": winner, "fit": fit_note, "dots": int(dots.sum()), "on_catalogue": int((dots & cat).sum()), "min_dist_to_catalogue_px": float(dist_cat[dots].min()) if dots.any() else None, "submission_name": name, "note": note, "note_chars": len(note), "file": f"docs/downloads/{name}.tif", "sha256": receipt["sha256"], "bytes": receipt["bytes"], "zip_sha256": receipt.get("zip_sha256"), "validator": val, "writer_receipt": receipt, "uniqueness_final_dots": uniq, "seconds": round(time.time()-t0,1)}
    (EVID/"h83_phase_e3_build.json").write_text(json.dumps(out, indent=2)+"\n")
    log(f"Exp3 {dest.name} sha {receipt['sha256'][:12]}")
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("stage", choices=["exp1","exp2","exp3","all"])
    a=ap.parse_args()
    if a.stage in ("exp1","all"): exp1()
    if a.stage in ("exp2","all"): exp2()
    if a.stage in ("exp3","all"): exp3()
