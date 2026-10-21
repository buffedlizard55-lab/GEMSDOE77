"""Release/readback checks for the stopped H82 experiment (never rerun it)."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from release_gate import assess

ROOT = Path(__file__).resolve().parents[1]


def test_h82_frozen_artifact_and_zip():
    c = json.loads((ROOT / 'evidence/h82_texture/run_card.json').read_text())
    p = ROOT / c['raster_path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == c['raster_sha256']
    with rasterio.open(p) as ds, rasterio.open(ROOT / 'data/bridge/sample_submission.tif') as ref:
        a = ds.read(1)
        assert ds.count == 1 and ds.dtypes == ('float32',) and ds.nodata is None
        assert (ds.shape, ds.crs, ds.transform) == (ref.shape, ref.crs, ref.transform)
        assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    with zipfile.ZipFile(p.with_suffix('.zip')) as z:
        assert z.namelist() == [p.name] and z.read(p.name) == p.read_bytes()
    readme = (ROOT/'README.md').read_text()
    assert c['raster_sha256'] in readme
    assert 'NOT OK TO SUBMIT any file' in readme[:2000]
    assert len(c['submission_note']) <= 140
    assert c['verdict'] == 'negative' and c['experiment_count'] == 1
    assert c['registry']['final_dots'].startswith('NOT PLACED')
    assert c['registry']['surface_max_forward_overlap_3px'] > .70
    assert c['registry']['full_registry_certified'] is False
    assert c['leakage_canary']['max_symmetric_auc'] <= .90
    assert c['weekly_slots_used'] == 0 and not c['ok_to_submit']


def test_h82_release_and_pages_match_card():
    row = next(r for r in json.loads((ROOT/'docs/data/lanes.json').read_text())['lanes']
               if r['lane'].startswith('H82-TEXTURE:'))
    result = assess(row, ROOT/'docs')
    assert result['checks']['format_valid'] and result['checks']['sha256_matches']
    assert not result['authorized_to_submit']
    for name in ('index.html', 'executive-summary.html', 'h82-texture.html'):
        text = (ROOT/'docs'/name).read_text()
        assert text.count('id="h82-status"') == 1
        assert row['file'] in text and row['sha256'] in text
        assert 'NOT OK TO SUBMIT' in text
    h = json.loads((ROOT/'evidence/h82_texture/holdout.json').read_text())['summary']
    for arm in h['scores'].values():
        assert arm['evidence_class'] == 'HOLDOUT-DTI'
        assert arm['evaluator_version'] == 'gems57-pooled-hide-v2'
        assert arm['withheld_positive_pixels'] == 59484
        assert arm['ci95'][0] <= arm['dti'] <= arm['ci95'][1]


def test_historical_baseline_flags_do_not_clear_current_incumbent_gate():
    rows = json.loads((ROOT/'docs/data/lanes.json').read_text())['lanes']
    historical = [r for r in rows if r.get('historical_beats_baseline_flag')]
    assert len(historical) == 3
    for row in historical:
        assert row['beats_holdout_best'] is False
        assert assess(row, ROOT/'docs')['checks']['beats_accepted_holdout_best'] is False
