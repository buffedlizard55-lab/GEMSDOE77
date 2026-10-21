#!/usr/bin/env python3
"""H82 one-lane, three-stage research pipeline. No organizer submission or slot selection.

E1 individual leakage canaries and pre-placement surface; E2 pinned pooled hide-v2;
E3 write locally valid inspection TIFF then STOP on final-dot registry duplicate.
"""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor'), str(ROOT / 'src')]
from gems57.grid import load_grid
from gems57.holdout import buffered_component_draw
from gems57 import evaluate_holdout as ev
from gems57.submission_writer import write_submission
from gems57.uniqueness import compare_array_to_registry, saturation_certificate
from gems77lia.allocate import greedy_select

SOURCE = ROOT / 'data/external/gdr_wellspring_in_footprint.csv'
PIN = '122718e65bdf55aab0ee12ad20d80062f0deb1de957225a61ad880dd5dc196ea'
WORK = ROOT / '.cache/h82'
EVIDENCE = ROOT / 'evidence'
BUDGET = 37654
SEED = 20


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ensure_inputs():
    """Restore only the three required SHA-pinned owner mirrors, never hidden truth."""
    witness = json.loads((ROOT / 'registry/witness_sources.json').read_text())
    source = {'repo': 'buffedlizard55-lab/GEMSDOE24',
              'ref': '07345ea0604953d7efb858d9cfbc21e20c7aca0b',
              'path': 'data/external/gdr_wellspring_in_footprint.csv', 'sha256': PIN}
    for dest, spec in ((SOURCE, source), (WORK/'dense.tif', witness['dense_witness']),
                       (WORK/'h33.tif', witness['h33'])):
        if dest.is_file() and digest(dest) == spec['sha256']:
            continue
        # Mismatched bytes are never silently replaced with a new training input.
        if dest.exists():
            raise ValueError(f'SHA mismatch in existing input {dest}; STOP')
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + '.partial')
        try:
            with tmp.open('wb') as out:
                subprocess.run(['gh','api',f'repos/{spec["repo"]}/contents/{spec["path"]}?ref={spec["ref"]}',
                                '-H', 'Accept: application/vnd.github.raw'], stdout=out, check=True)
            if digest(tmp) != spec['sha256']:
                raise ValueError(f'SHA mismatch from pinned source {spec["repo"]}; STOP')
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)


def fields(grid):
    if digest(SOURCE) != PIN:
        raise ValueError('owner mirror changed; STOP')
    impulses = {k: np.zeros(grid.shape, np.float32) for k in ('concordance', 'presence', 'cool')}
    accepted = 0
    with SOURCE.open(newline='') as fh:
        for row in csv.DictReader(fh):
            # Never access dist_known_fault_px: it is catalogue-derived and forbidden.
            try:
                r, c = int(float(row['row'])), int(float(row['col']))
                t = float(row['temp_c'])
                q, ch, ca = (float(row[k]) for k in ('geothermquartz_c', 'geothermchalc_c', 'geothermcat_c'))
            except (ValueError, KeyError, OverflowError):
                continue
            if not (0 <= r < grid.height and 0 <= c < grid.width and grid.footprint[r, c]):
                continue
            if not (0 < t < 200 and all(np.isfinite([q, ch, ca, t])) and 0 < min(q, ch, ca)
                    and max(q, ch, ca) < 400 and min(q, ch, ca) > 100 and t <= 60):
                continue
            weight = min(1., (min(q, ch, ca) - 100.) / 100.) * np.exp(-(max(q, ch, ca)-min(q, ch, ca))/80.)
            weight *= max(0., 1. - max(0., t-35.) / 50.)
            if weight <= 0:
                continue
            impulses['concordance'][r, c] = max(impulses['concordance'][r, c], weight)
            impulses['presence'][r, c] = 1.
            impulses['cool'][r, c] = max(impulses['cool'][r, c], max(0., 1. - t/60.))
            accepted += 1
    if not accepted:
        raise ValueError('no accepted geothermometers; STOP')
    result = {}
    for key, imp in impulses.items():
        surface = gaussian_filter(imp, 20., mode='constant', truncate=4.).astype(np.float32)
        surface /= float(surface.max())
        surface[(~grid.footprint) | (surface < .02)] = 0
        result[key] = surface
    return result, {'accepted_rows': accepted, 'accepted_unique_cells': int(np.count_nonzero(impulses['presence'])),
                    'source_sha256': PIN, 'source_status': 'owner mirror, NOT organizer-authenticated',
                    'methods': 'gaussian sigma20px, truncate4, support cutoff normalized .02'}


def inventory():
    witness = json.loads((ROOT / 'registry/witness_sources.json').read_text())
    records = []
    for key in ('dense_witness', 'h33'):
        p = WORK / ('dense.tif' if key == 'dense_witness' else 'h33.tif')
        if digest(p) != witness[key]['sha256']:
            raise ValueError(f'{key} witness SHA mismatch; STOP')
        records.append({'cache_file': str(p), 'sha256': witness[key]['sha256'],
                        'repo_first': witness[key]['repo'], 'sources': [key]})
    return {'rasters': records, 'complete_accessible_scan': False,
            'errors': ['Only two pinned witnesses checked; NOT a complete registry scan']}


def compare(a, grid):
    return compare_array_to_registry(a, inventory(), grid.footprint,
        expected_grid=(grid.shape, grid.crs, grid.transform), include_reverse=False)


def canaries(surfaces, grid, folds):
    rng = np.random.default_rng(8201)
    v = {k: [[], []] for k in surfaces}
    for fold in folds:
        active = grid.footprint & fold['region'] & ~fold['masked_known']
        pos = np.flatnonzero(active & fold['truth'])
        neg = np.flatnonzero(active & ~fold['truth'])
        neg = rng.choice(neg, min(100000, len(neg)), replace=False)
        for k, a in surfaces.items():
            v[k][0].append(a.ravel()[pos])
            v[k][1].append(a.ravel()[neg])
    results = {}
    for k, pair in v.items():
        pos, neg = (np.concatenate(x) for x in pair)
        ranks = rankdata(np.concatenate((pos, neg)))
        auc = (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
        results[k] = {'auc': float(auc), 'symmetric_auc': float(max(auc, 1-auc)),
                      'leakage_flag': bool(max(auc, 1-auc) > .90)}
    return results


def emit(a, allowed, budget=BUDGET, rng=None):
    if rng is None:
        ids = np.flatnonzero(allowed & (a > 0))
        order = ids[np.argsort(-a.ravel()[ids], kind='stable')]
    else:
        order = rng.permutation(np.flatnonzero(allowed))
    selected = greedy_select(order, min_sep=3.0, limit=budget)
    out = np.zeros(a.shape, np.float32)
    out.ravel()[selected] = 1.
    return out


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    ensure_inputs()
    grid = load_grid(ROOT / 'data/bridge')
    folds, _ = buffered_component_draw(grid, seed=SEED, hide_frac=.20, buffer_px=3, boundary_px=12)
    surfaces, metadata = fields(grid)
    w = inventory()
    cert = saturation_certificate(WORK/'dense.tif', grid.footprint, grid.catalogue,
                                  expected_grid=(grid.shape, grid.crs, grid.transform))
    if cert['witness_sha256'] != w['rasters'][0]['sha256']:
        raise ValueError('witness mismatch')
    c = canaries(surfaces, grid, folds)
    surface = compare(surfaces['concordance'], grid)
    rows = surface['rows']
    rho = max(abs(x['spearman_full_footprint']) for x in rows if x.get('spearman_full_footprint') is not None)
    e1 = {'experiment':'H82-E1', 'split': [f['receipt'] for f in folds], 'metadata':metadata,
          'leakage_canaries':c, 'preplacement_registry':surface,
          'max_abs_surface_spearman_checked':rho, 'dense_witness_certificate':cert}
    save('h82_e1.json', e1)
    print('E1 surface max |rho|', rho, 'canaries', c, flush=True)
    if rho > .90 or any(x['leakage_flag'] for x in c.values()):
        raise SystemExit('STOP: E1 surface duplicate or leakage flag; no E2/E3')

    # E2: same emitted budget for all arms in each fold; no label-derived features.
    terms = {k: None for k in ('concordance', 'presence', 'random')}
    scores = {}
    counts = []
    for f in folds:
        allowed = grid.footprint & ~f['masked_known']
        pred = {k: emit(surfaces[k], allowed) for k in ('concordance','presence')}
        count = min(int(x.sum()) for x in pred.values())
        if count <= 0:
            raise ValueError('empty candidate arm')
        if any(x.sum() != count for x in pred.values()):
            pred = {k: emit(surfaces[k], allowed, count) for k in pred}
        pred['random'] = emit(surfaces['concordance'], allowed, count,
                              np.random.default_rng(82000 + f['fold']))
        counts.append(count)
        for k, p in pred.items():
            result, t = ev.evaluate(p, f, grid.footprint, origin=(0, 0), global_shape=grid.shape)
            scores.setdefault(k, []).append({'fold':f['name'], 'dti':result['dti'],
                                               'withheld_positive_pixels':result['n_truth']})
            terms[k] = t if terms[k] is None else terms[k] + t
        print('E2', f['name'], 'count', count, flush=True)
    summary = ev.pooled_summary(terms, draws=1000, seed=82021, candidate='concordance')
    save('h82_e2.json', {'experiment':'H82-E2', 'pooled':summary, 'fold_scores_diagnostic_only':scores,
                         'fold_budgets':counts})
    h = summary['scores']['concordance']
    print('E2 HOLDOUT-DTI', h['dti'], h['ci95'], h['withheld_positive_pixels'], flush=True)

    # E3 is a preflight package: locally valid bytes remain inspection-only even when
    # post-placement uniqueness fails. Do not re-tune or submit after failure.
    allowed = grid.footprint & ~grid.catalogue
    p = emit(surfaces['concordance'], allowed, min(counts))
    digest_short = hashlib.sha256(p.tobytes()).hexdigest()[:10]
    name = f'gems77-h82-cool-chemconcord-{int(p.sum())}dots-20261021-{digest_short}-zeros'
    note = 'H82 cool-outlet concordant geothermometers; research-only, NOT APPROVED'
    out = ROOT / 'docs/downloads' / (name+'.tif')
    if out.exists():
        raise FileExistsError(out)
    receipt = write_submission(out, p, ROOT/'data/bridge/sample_submission.tif', grid.footprint,
                               note=note, name=name, catalogue=grid.catalogue,
                               metadata={'lane':'H82', 'selector_approved':False})
    dots = compare(p, grid)
    worst = max(dots['rows'], key=lambda x:x.get('my_dots_within_3px_of_theirs', 0))
    duplicate = bool(any(x.get('duplicate_by_overlap') or x.get('duplicate_by_rho') for x in dots['rows']))
    e3 = {'experiment':'H82-E3','name':name,'note':note, 'sha256':digest(out), 'receipt':receipt,
          'postplacement_registry':dots, 'worst_overlap':worst['my_dots_within_3px_of_theirs'],
          'worst_overlap_submission':worst['submission'], 'duplicate_stop':duplicate,
          'registry_complete':False, 'slot_used':False}
    save('h82_e3.json',e3)
    card = {'hypothesis':'Cool outlet + three-geothermometer concordance predicts concealed hydrothermal permeability',
            'mechanism':'Concordant hotter-than-outlet estimated temperatures, pointwise maxima then 2 km Gaussian field',
            'named_non_fault_process_that_could_mimic_it':'Conductive non-fault basin aquifers and road-clustered sampling',
            'holdout_dti':h, 'paired_controls':summary['paired_differences'],
            'leakage_canary':c, 'correlation_overlap_vs_registry':{
                'surface_max_abs_spearman':rho,
                'final_max_abs_spearman':max(abs(row['spearman_full_footprint']) for row in dots['rows']),
                'final_max_forward_3px_overlap':e3['worst_overlap'],
                'vs_h33_forward_overlap_3px':next(row['my_dots_within_3px_of_theirs'] for row in dots['rows'] if row['submission']=='h33'),
                'worst_submission':worst['submission'], 'registry_rasters_checked':len(dots['rows']),
                'registry_complete':False, 'duplicate':duplicate, 'dense_witness':cert['witness_sha256']},
            'raster_sha256':digest(out), 'validator_output':{
                'no_nan_inside_footprint':receipt['validator']['checks']['in_footprint_all_finite'],
                'values_in_0_1':receipt['validator']['checks']['range_0_1_guaranteed'],
                'crs_shape_transform_match':all(receipt['validator']['checks'][k] for k in
                    ('crs_epsg_32611','dimensions_3730x3292','transform_matches_sample_submission')),
                'all_checks_passed':receipt['validator']['all_checks_passed']},
            'submission_name':name,'note':note,'note_chars':len(note),
            'verdict':'negative','ok_to_download_for_inspection':True,'ok_to_submit':False,
            'reason':'STOP: final dots duplicate full-footprint registry witness; incomplete global registry and holdout gate. No selector authorization or organizer score.',
            'experiments_used':3,'weekly_slots_used':0,'evaluator_version':ev.VERSION}
    save('h82_run_card.json',card)
    print('E3 STOP: duplicate',duplicate,'overlap',e3['worst_overlap'],name, flush=True)
    return card

if __name__ == '__main__':
    run()
