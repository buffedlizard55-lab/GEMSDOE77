"""H82 stopped research artifact must never become a submit recommendation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_h82_card_and_download_are_negative():
    card = json.loads((ROOT/'evidence/h82_run_card.json').read_text())
    art = ROOT/'docs'/'downloads'/f"{card['submission_name']}.tif"
    assert art.is_file()
    assert hashlib.sha256(art.read_bytes()).hexdigest() == card['raster_sha256']
    assert card['verdict']=='negative' and card['ok_to_submit'] is False
    assert card['protocol_unique'] is False and card['weekly_slots_used'] == 0
    assert card['surface_correlation_overlap_vs_registry']['worst_forward_3px_overlap'] > .70
    assert card['final_dots']['count'] == 0
    assert card['validator_output']['all_checks_passed']
    assert card['validator_output']['n_nan'] == 0
    assert all(card['validator_output']['checks'].values())
    assert len(card['submission_note']) <= 140
    h = card['holdout_dti']
    assert h['evidence_class'] == 'HOLDOUT-DTI'
    assert h['evaluator_version'] == 'gems57-pooled-hide-v2'
    assert h['withheld_positive_pixels'] > 0 and h['ci95'][0] <= h['dti'] <= h['ci95'][1]
    assert h['dti'] < card['holdout_details']['control']['dti']
    assert max(v['max_symmetric_auc'] for v in card['holdout_details']['canaries'].values()) <= .90


def test_h82_pages_and_manifest_fail_closed():
    card = json.loads((ROOT/'evidence/h82_run_card.json').read_text())
    lanes = json.loads((ROOT/'docs/data/lanes.json').read_text())['lanes']
    row = next(r for r in lanes if r['lane'].startswith('H82:'))
    assert row['sha256'] == card['raster_sha256']
    assert row['protocol_unique'] is False and row['verdict'] == 'negative'
    assert row['primary'] is False and row['beats_holdout_best'] is False
    receipt = json.loads((ROOT/'docs/data/submission-authorization.json').read_text())
    assert receipt['repository_decision'] == 'NOT OK TO SUBMIT'
    matches = [r for r in receipt['artifacts'] if r['sha256'] == card['raster_sha256']]
    assert len(matches) == 1 and matches[0]['authorized_to_submit'] is False
    for p in ('index.html','h82.html','executive-summary.html'):
        text = (ROOT/'docs'/p).read_text()
        assert 'NOT OK TO SUBMIT' in text and card['submission_name'] in text
