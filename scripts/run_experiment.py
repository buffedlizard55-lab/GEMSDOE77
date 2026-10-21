"""One preregistered pilot; literal duplicate stop; research-only export on failure."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor'), str(ROOT/'scripts')]
import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score
from gems57 import evaluate_holdout as ev, spatial, uniqueness
from gems57.submission_writer import write_submission
from gems77.features import build
from prepare_data import prepare
from fetch_data import sha
from restore_registry import get_registry


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def crop_region(fold):
    yy, xx = np.nonzero(fold['region'])
    y0, x0 = int(yy.min()), int(xx.min())
    sl = np.s_[y0:int(yy.max())+1, x0:int(xx.max())+1]
    small = {k: fold[k][sl] for k in ('region','visible','truth')}
    return sl, small, (y0, x0)


def validate_features(responses, valid, catalogue):
    """Hide whole components, and use only visible faults for masking.

    No training or catalogue-based transform exists in this pilot. The buffered
    training domain is recorded to permit a later supervised lane, not used here.
    Canary AUC is exact over each quadrant vs catalogue-zero proxies, not true absence.
    """
    terms = {n: None for n in responses}
    folds, canaries = [], []
    for fold in spatial.folds(catalogue, valid, buffer_px=80):
        sl, small, origin = crop_region(fold)
        active = small['region'] & ~small['visible']
        truth = small['truth'][active]
        if not truth.any() or truth.all():
            raise ValueError('degenerate canary domain')
        folds.append(fold['receipt'])
        for name, p in responses.items():
            a = p[sl]
            auc = float(roc_auc_score(truth, a[active]))
            canaries.append({'feature':name, 'fold':int(fold['fold']), 'auc':auc,
                             'positive_pixels':int(truth.sum()),'negative_proxy_pixels':int((~truth).sum()),
                             'leakage_flag':bool(auc > .90),
                             'interpretation':'LEAKAGE-CANARY diagnostic, not HOLDOUT-DTI or a leaderboard score'})
            _, block_terms = ev.evaluate(a, small, valid[sl], origin=origin, global_shape=valid.shape)
            if terms[name] is None: terms[name] = block_terms
            else: terms[name] += block_terms
        print('holdout fold',fold['fold'],'complete',flush=True)
    summary = ev.pooled_summary(terms, candidate='candidate', draws=1000, seed=77021)
    summary['comparison_warning'] = ('Best control includes feature-alone ablations. This blank repository had no current holdout best. '
                                     'Archived H33 bytes are catalogue-conditioned and are NOT valid hide-and-recover predictions. '
                                     'No comparison to a historical score is made.')
    dump(ROOT/'evidence/holdout.json', summary)
    dump(ROOT/'evidence/folds.json', {'split':'label-blind-quadrants-v2','folds':folds,
         'catalogue_features':'none; buffered train masks not used for label-free transforms'})
    dump(ROOT/'evidence/canaries.json', {'evidence_class':'LEAKAGE-CANARY','threshold':.90,
         'any_flag':any(x['leakage_flag'] for x in canaries), 'rows':canaries,
         'warning':'AUC below threshold does not prove absence of leakage. Negatives are unlabelled proxies.'})
    return summary, canaries


def main():
    if (ROOT/'evidence/run_card.json').exists():
        raise SystemExit('Run already finalized. Do not rerun or tune a stopped lane.')
    started = time.monotonic()
    g = prepare()
    with rasterio.open(ROOT/'data/geodawn_rad_u8.tif') as src: raw = src.read([1,2,3])
    responses, valid, scales = build(raw, g.footprint)
    del raw
    dump(ROOT/'evidence/features.json', {'input':'data/geodawn_rad_u8.tif',
         'input_sha256':sha(ROOT/'data/geodawn_rad_u8.tif'), 'eligible_pixels':int(valid.sum()),
         'normalization_scales':scales,'feature_sha256':{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in responses.items()},
         'catalogue_derived_features':[], 'scientific_claim':'Exploratory rank-intensity transform; not physical enrichment or a calibrated probability'})
    summary, canaries = validate_features(responses, valid, g.catalogue)
    # The exact final research surface; no thinning, budget search or dot placement.
    surface = responses['candidate'].copy()
    surface[g.catalogue | ~g.footprint] = 0
    del responses
    digest = hashlib.sha256(surface.tobytes()).hexdigest()
    name = f'gems77-h77-urank-curvature-{digest[:12]}-research'
    note = 'H77 selective-U rank curvature; research-only, literal registry gate failed; do not submit.'
    registry = get_registry()
    audit = uniqueness.compare_array_to_registry(surface, registry, g.footprint,
                expected_grid=(g.shape, g.crs, g.transform))
    dump(ROOT/'evidence/surface_registry.json', audit)
    # Stop means no further model/placement experiment. Retain the tested surface
    # only as negative evidence. Shared writer validates bytes; it does NOT approve a slot.
    path = ROOT/'docs/downloads'/f'{name}.tif'
    receipt = write_submission(path, surface, ROOT/'data/sample_submission.tif', g.footprint,
           name=name,note=note,catalogue=g.catalogue,
           metadata={'research_only':True,'lane':'H77','placement':'NOT RUN: surface gate stop',
                     'decoded_float32_sha256':digest,'uniqueness_cleared':False})
    dump(ROOT/'evidence/artifact_receipt.json', receipt)
    # Post-serialization identity audit of unchanged surface, not another experiment.
    audit = uniqueness.compare_array_to_registry(surface, registry, g.footprint,
                file_sha256=receipt['sha256'], expected_grid=(g.shape, g.crs, g.transform))
    dump(ROOT/'evidence/surface_registry.json', audit)
    delta = summary['paired_differences'][summary['best_comparable_control']]
    card = {'hypothesis':'H77 selective-U curvature after K/Th common-mode subtraction',
            'mechanism':'Anisotropic curvature of a label-free U minus K-Th rank-intensity residual; no old predictions used',
            'non_fault_mimic':'Groundwater redox fronts and fluvial heavy-mineral sorting',
            'holdout':summary['scores']['candidate'],
            'best_comparable_control':summary['best_comparable_control'],
            'paired_difference_to_best_control':delta,
            'canary_any_auc_above_090':any(c['leakage_flag'] for c in canaries),
            'registry':{'scope':'Partial pinned template registry: rejection witness only, never uniqueness certification',
                        'complete':False,'checked':audit['registry_rasters_checked'],
                        'max_spearman':audit['worst_spearman_full_footprint'],
                        'max_forward_overlap_3px':audit['worst_dot_overlap'],
                        'duplicate_count':audit['duplicate_count'],
                        'surface_gate':'FAIL' if audit['stop_required'] else 'PASS',
                        'final_dot_gate':'NOT RUN — stopped before placement'},
            'raster_sha256':receipt['sha256'],
            'validator':receipt['validator']['checks'],
            'submission_name':name,'note':note,'file':str(path.relative_to(ROOT)),
            'ok_to_download':True,'ok_to_submit':False,
            'byte_novel_within_checked_registry':audit['byte_unique_among_checked'],
            'protocol_unique':False,'approved_for_weekly_slot':False,
            'weekly_slots_used':0,'organizer_confirmed_score':None,
            'verdict':'negative','reason':'Literal surface duplicate/overlap stop or incomplete registry; no slot approval. Research-only TIFF.',
            'experiments_run':1,'elapsed_seconds':round(time.monotonic()-started,2)}
    dump(ROOT/'evidence/run_card.json',card)
    print(json.dumps(card,indent=2))

if __name__ == '__main__': main()
