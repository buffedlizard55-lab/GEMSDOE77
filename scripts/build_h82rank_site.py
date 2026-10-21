#!/usr/bin/env python3
"""Render the stopped H82-radiometric research lane (post-run alias H82R); never grant submission clearance."""
from __future__ import annotations
import hashlib, html, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs'
def e(x): return html.escape(str(x),quote=True)

def main():
    card=json.loads((ROOT/'evidence/h82rank_run_card.json').read_text())
    h=card['holdout_dti']; cov=card['surface_correlation_overlap_vs_registry']
    path=D/'downloads'/f"{card['submission_name']}.tif"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==card['raster_sha256']
    assert card['verdict']=='negative' and not card['ok_to_submit'] and card['protocol_unique'] is False
    assert cov['duplicate_stop'] and cov['worst_forward_3px_overlap']>0.70
    assert card['validator_output']['all_checks_passed'] and len(card['submission_note'])<=140
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>H82R negative research result · GEMSDOE77</title><link rel="stylesheet" href="assets/lia.css"></head><body><header><div class="wrap"><a href="index.html">Start here</a> · <a href="executive-summary.html">Executive summary: how to submit after approval</a></div></header><main><div class="wrap">
<h1>H82R: ternary radiometric rank-mixture entropy deficit</h1><p class="verdict"><b>OK to download for research/inspection: YES. NOT OK TO SUBMIT.</b> A surface-level duplicate stop was triggered; no dots were placed, no selector approved a slot and the research TIFF must NOT be uploaded. It is a new research artifact, not a protocol-unique competition submission.</p>
<section class="card dl"><h2>Direct research download — DO NOT SUBMIT</h2><a class="btn" download href="downloads/{e(path.name)}">Download research GeoTIFF</a> <a class="btn alt" download href="downloads/{e(path.stem)}.zip">Download one-TIFF research ZIP</a><p>SHA-256: <code>{e(card['raster_sha256'])}</code></p><p>Name (reference only): <code>{e(card['submission_name'])}</code> · Note ({len(card['submission_note'])}/140): <code>{e(card['submission_note'])}</code>.</p></section>
<h2>Predeclared hypothesis and measured outcome</h2><p>Three separately quantized airborne K, Th and U intensities were turned into a local rank-mixture entropy; the surface highlights a local entropy deficit against a Gaussian background. <b>Not a measured chemical K/Th/U concentration ratio.</b> Off-catalogue composition transitions could be fractures beneath thin cover, but ordinary soil provenance, lithologic boundaries and survey striping can produce the same pattern.</p><p><b>HOLDOUT-DTI {h['dti']:.5f}</b> (evaluator {e(h['evaluator_version'])}; {h['withheld_positive_pixels']:,} withheld positives; 95% CI [{h['ci95'][0]:.5f}, {h['ci95'][1]:.5f}]). Spatially permuted same-intensity control: HOLDOUT-DTI {card['holdout_details']['control']['dti']:.5f} (95% CI [{card['holdout_details']['control']['ci95'][0]:.5f}, {card['holdout_details']['control']['ci95'][1]:.5f}], same evaluator/positives). Paired delta {card['holdout_paired_vs_spatial_random']['delta']:+.5f} (95% CI [{card['holdout_paired_vs_spatial_random']['ci95'][0]:+.5f}, {card['holdout_paired_vs_spatial_random']['ci95'][1]:+.5f}]). No evidence of improvement over the repo incumbent, let alone the public leaderboard.</p>
<h2>Mandatory gates</h2><ul><li>Six single-feature leakage canaries: maximum symmetric AUC {max(v['max_symmetric_auc'] for v in card['holdout_details']['canaries'].values()):.4f} ≤0.90.</li><li>Surface Spearman maximum |rho| {cov['max_abs_rho']:.5f} vs two SHA-pinned witnesses; forward 3 px overlap {cov['worst_forward_3px_overlap']:.3f} &gt;0.70 vs pinned dense witness. <b>Duplicate stop. No final dots exist.</b> Two witnesses are a rejection proof, not a complete registry scan.</li><li>Local writer readback: one float32 band, EPSG:32611, 3730 × 3292; exact template transform; no NaN inside or outside footprint, values in [0,1], no nodata tag. Local format validity is NOT portal acceptance.</li><li>Verdict negative; weekly feedback slots used: 0; selector not approved.</li></ul>
<h2>Sources and replication</h2><p><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">Organizer problem, metric and format</a> · <a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">USGS GeoDAWN data release</a>. Derived data are SHA-pinned owner mirrors, not organizer-authenticated downloads. <a href="https://github.com/buffedlizard55-lab/GEMSDOE77/blob/main/research/h82rank_hypotheses.md">Ranked pre-experiment hypotheses</a> · <a href="https://github.com/buffedlizard55-lab/GEMSDOE77/blob/main/registry/preregistration_h82rank.json">Preregistration</a> · <a href="https://github.com/buffedlizard55-lab/GEMSDOE77/blob/main/evidence/h82rank_run_card.json">Run card</a> · <a href="https://github.com/buffedlizard55-lab/GEMSDOE77/blob/main/scripts/h82rank_research.py">Reproduce experiment</a>.</p><p><a href="executive-summary.html">How to submit a future cleared file (not this one) →</a></p></div></main></body></html>'''
    (D/'h82rank.html').write_text(page)
    lanespath=D/'data'/'lanes.json'; lanes=json.loads(lanespath.read_text())
    # Keep a pre-existing H82R row in place so repeated site builds do not reorder
    # the public download inventory. New rows use the original H82 insertion rule.
    existing_index=next((i for i,r in enumerate(lanes['lanes']) if r['lane'].startswith('H82R:')),None)
    lanes['lanes']=[r for r in lanes['lanes'] if not r['lane'].startswith('H82R:')]
    insert_at=existing_index if existing_index is not None else (1 if any(r['lane'].startswith('H82:') for r in lanes['lanes']) else 0)
    lanes['lanes'].insert(insert_at,{
        'lane':'H82R: local ternary radiometric rank-mixture entropy (STOPPED; research only)',
        'page':'h82rank.html','summary':'executive-summary.html','file':f'downloads/{path.name}',
        'run_card':'evidence/h82rank_run_card.json','format_valid':True,'protocol_unique':False,
        'informative_unique':None,'beats_holdout_best':False,'verdict':'negative',
        'primary':False,'name':card['submission_name'],'note':card['submission_note'],
        'sha256':card['raster_sha256'],
        'upload':f"Not OK to submit. OK to download for inspection only. HOLDOUT-DTI {h['dti']:.5f} (95% CI {h['ci95'][0]:.5f}-{h['ci95'][1]:.5f}; {h['evaluator_version']}; {h['withheld_positive_pixels']} withheld positives). Surface duplicate stop 1.0 vs dense witness; no dots placed; selector not approved; zero weekly slots.",
        'why_primary':'Research-only negative deliverable; no submission clearance.'})
    note = lanes.get('note', '')
    if 'H82R:' not in note:
        note += (' ' if note else '') + 'H82R: radiometric negative research file; NOT OK TO SUBMIT.'
    import re
    lanes['note'] = re.sub(r'All \d+ listed TIFF lanes', f'All {len(lanes["lanes"])} listed TIFF lanes', note)
    lanespath.write_text(json.dumps(lanes,indent=2,ensure_ascii=True)+'\n')

if __name__=='__main__':main()
