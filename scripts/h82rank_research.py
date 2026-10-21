#!/usr/bin/env python3
"""One preregistered H82 experiment. Research-only artifact; duplicate => STOP, no dots.

Uses pinned gems57 holdout/evaluator/uniqueness/writer, without private forks.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor'))
from gems57.grid import load_grid
from gems57.holdout import buffered_component_draw
from gems57.evaluate_holdout import evaluate, pooled_summary, VERSION
from gems57.uniqueness import compare_array_to_registry, saturation_certificate
from gems57.submission_writer import write_submission
from restore_registry import get_registry

NAME = 'gems77-h82-ternaryentropy-20261021-RESEARCH-ONLY'
NOTE = 'H82 radiometric ternary rank-mixture entropy deficit; research only; NOT APPROVED for submission'
OUT = ROOT/'docs'/'downloads'/f'{NAME}.tif'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    prereg = json.loads((ROOT/'registry/preregistration_h82rank.json').read_text())
    assert len(NOTE) <= 140
    manifest = json.loads((ROOT/'registry/data_manifest.json').read_text())
    pin = next(e for e in manifest['files'] if e['id']=='ext_geodawn_rad_u8')
    path = ROOT/'data'/'raw'/pin['dest']
    if sha(path) != pin['sha256']:
        raise ValueError('radiometric mirror pin mismatch')
    grid = load_grid(ROOT/'data'/'bridge')
    with rasterio.open(path) as ds:
        assert ds.shape == grid.shape and ds.crs == grid.crs and ds.transform == grid.transform
        assert ds.descriptions[:3] == ('K','Th','U') and ds.nodata == 0.0
        raw = ds.read((1,2,3))
    valid = grid.footprint & np.all(raw > 0, axis=0)
    print('radiometric valid',int(valid.sum()),flush=True)
    # Quantized values are not concentrations; only rank-mixture variation is measured.
    v = raw.astype(np.float32)/255.0
    shares = (v + .02) / (v.sum(axis=0) + .06)
    entropy = -np.sum(shares * np.log(shares), axis=0)/np.log(3)
    del shares, v
    entropy = entropy.astype(np.float32)
    entropy[~valid] = 0
    denom = gaussian_filter(valid.astype(np.float32),2.5)
    background = gaussian_filter(entropy,2.5)/np.maximum(denom,1e-6)
    deficit = np.maximum(background - entropy, 0).astype(np.float32)
    deficit[~valid] = 0
    scale = float(np.percentile(deficit[valid],99))
    if not scale > 0: raise ValueError('degenerate feature')
    surface = np.minimum(deficit/scale, 1).astype(np.float32)
    surface[~valid | grid.catalogue] = 0
    assert np.isfinite(surface).all() and surface.min()>=0 and surface.max()<=1
    del deficit, denom
    features = {'K':raw[0].astype(np.float32)/255, 'Th':raw[1].astype(np.float32)/255,
                'U':raw[2].astype(np.float32)/255, 'entropy':entropy,
                'background_entropy':background,'candidate':surface}
    del raw
    folds, _ = buffered_component_draw(grid,seed=20,hide_frac=.20,buffer_px=3)
    # No catalogue-derived features: all fields above are fixed, label-blind.
    rng=np.random.default_rng(820)
    canaries={}
    for key,a in features.items():
        fold_rows=[]
        for f in folds:
            active = valid & f['region'] & ~f['masked_known']
            y = (f['truth'] & active)
            neg = active & ~y
            pos_idx = np.flatnonzero(y)
            neg_idx = np.flatnonzero(neg)
            if len(pos_idx)==0: raise ValueError('no held-out positives')
            chosen = rng.choice(neg_idx, size=min(200_000, len(neg_idx)), replace=False)
            flat=a.ravel()
            auc=float(roc_auc_score(np.r_[np.ones(len(pos_idx)),np.zeros(len(chosen))],
                                    np.r_[flat[pos_idx],flat[chosen]]))
            fold_rows.append({'fold':f['name'],'n_pos':len(pos_idx), 'n_neg':len(chosen),
                              'auc':auc,'symmetric_auc':max(auc,1-auc)})
        canaries[key]={'folds':fold_rows,'max_symmetric_auc':max(r['symmetric_auc'] for r in fold_rows)}
        print('canary',key,canaries[key]['max_symmetric_auc'],flush=True)
    terms={'candidate':[],'permuted_control':[]}
    # Fold-cropped and globally aligned 20km physical clusters, identical masks and truth.
    # Spatial permutation retains the candidate's exact one-pixel value distribution.
    ctrl=np.zeros(grid.shape,np.float32)
    indices=np.flatnonzero(valid & ~grid.catalogue)
    ctrl.ravel()[indices]=np.random.default_rng(820).permutation(surface.ravel()[indices])
    fold_receipts=[]
    for f in folds:
        r,c = np.where(f['region'])
        if not len(r): raise ValueError('empty fold')
        # 6 px padding to preserve 3 px kernel at scored region boundaries.
        r0,r1=max(0,r.min()-6),min(grid.height,r.max()+7)
        c0,c1=max(0,c.min()-6),min(grid.width,c.max()+7)
        sl=np.s_[r0:r1,c0:c1]
        crop={k:(x[sl] if isinstance(x,np.ndarray) else x) for k,x in f.items()}
        fold_row={'fold':f['name'],'receipt':f['receipt'],'scores':{}}
        for name,field in (('candidate',surface),('permuted_control',ctrl)):
            result,t=evaluate(field[sl],crop,grid.footprint[sl],origin=(r0,c0),global_shape=grid.shape)
            terms[name].append(t)
            fold_row['scores'][name]=dict(dti=result['dti'], n_truth=result['n_truth'], n_emitted=result['n_emitted'])
        fold_receipts.append(fold_row)
        print('fold',f['name'],fold_row['scores'],flush=True)
    summary=pooled_summary({k:np.sum(v,axis=0) for k,v in terms.items()},seed=820,
                           draws=1000,candidate='candidate')
    print('HOLDOUT-DTI',summary['scores'],flush=True)
    # Binding gate before ANY dot placement. Same candidate surface is later packaged
    # solely for reproducible inspection; packaging does NOT override the stop.
    rows=get_registry()
    witness=saturation_certificate(rows[0]['file'],grid.footprint,grid.catalogue,
                                    expected_grid=(grid.shape,grid.crs,grid.transform))
    diagnostic=compare_array_to_registry(surface,rows,grid.footprint,
                     expected_grid=(grid.shape,grid.crs,grid.transform),include_reverse=False)
    duplicate=bool(diagnostic['stop_required'])
    if not duplicate: raise AssertionError('expected explicit stop / no placement')
    print('STOP: surface duplicate; NO DOT PLACEMENT', witness['uncovered_allowed_pixels'],flush=True)
    # Retain this exact stopped surface, zero outside footprint, as research-only.
    receipt=write_submission(OUT,surface,ROOT/'data'/'bridge'/'sample_submission.tif',
            grid.footprint,note=NOTE,name=NAME,catalogue=grid.catalogue,
            metadata={'warning':'Research only; literal surface duplicate, no dots placed, NOT OK TO SUBMIT'})
    # The final dots are absent by design: no spurious claim of a passed final gate.
    card={'hypothesis':prereg['lane'],'mechanism':prereg['formula'],
          'named_non_fault_process_that_could_mimic_it':prereg['mimic'],
          'evidence_class':'RESEARCH-ONLY; NOT ORGANIZER-CONFIRMED',
          'holdout_dti':summary['scores']['candidate'],
          'holdout_paired_vs_spatial_random':summary['paired_differences']['permuted_control'],
          'holdout_details':{'evaluator_version':VERSION,'folds':fold_receipts,'canaries':canaries,
                             'control':summary['scores']['permuted_control'],
                             'bootstrap':summary['bootstrap']},
          'surface_correlation_overlap_vs_registry':{
              'max_abs_rho':max(abs(r['spearman_full_footprint']) for r in diagnostic['rows'] if r.get('spearman_full_footprint') is not None),
              'worst_forward_3px_overlap':diagnostic['worst_dot_overlap'],
              'worst_overlap_submission':diagnostic['worst_overlap_submission'],
              'raster_count_checked':diagnostic['registry_rasters_checked'],
              'complete_registry':False,'duplicate_stop':duplicate,
              'witness':witness,'two_pinned_rows':diagnostic['rows']},
          'protocol_unique':False,'registry':{'complete':False,'surface_max_forward_overlap_3px':diagnostic['worst_dot_overlap']},
          'final_dots':{'count':0,'correlation_overlap_vs_registry':None,
                        'reason':'No placement after surface duplicate stop'},
          'raster_sha256':receipt['sha256'],'validator_output':receipt['validator'],
          'submission_name':NAME,'submission_note':NOTE,
          'ok_to_download_for_inspection':True,'ok_to_submit':False,
          'weekly_slots_used':0,'verdict':'negative',
          'irregularities':['Quantized channel ranks are not elemental concentrations',
                            'Dense witness makes literal gate unsatisfiable for any nonempty candidate',
                            'Two pinned rasters only: not full registry uniqueness clearance']}
    out=ROOT/'evidence'/'h82rank_run_card.json'
    out.write_text(json.dumps(card,indent=2,allow_nan=False)+'\n')
    print('research artifact',OUT,receipt['sha256'],out,flush=True)

if __name__=='__main__':main()
