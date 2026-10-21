"""Freeze format, identity and negative decision for the research-only H82 lane."""
import hashlib
import json
import zipfile
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parents[1]


def test_h82_artifact_status_and_grid():
    card = json.loads((ROOT/'evidence/h82_run_card.json').read_text())
    name = card['submission_name']
    p = ROOT/'docs/downloads'/f'{name}.tif'
    assert p.is_file()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == card['raster_sha256']
    with zipfile.ZipFile(p.with_suffix('.zip')) as z:
        assert z.namelist() == [p.name]
        assert z.read(p.name) == p.read_bytes()
    with rasterio.open(p) as ds:
        assert ds.count == 1 and ds.dtypes == ('float32',) and ds.crs.to_epsg() == 32611
        with rasterio.open(ROOT/'data/bridge/sample_submission.tif') as ref:
            assert (ds.shape, ds.transform, ds.crs) == (ref.shape, ref.transform, ref.crs)
        for _, window in ds.block_windows(1):
            a = ds.read(1, window=window)
            assert (a >= 0).all() and (a <= 1).all()  # NaN fails comparisons
    assert card['verdict'] == 'negative' and card['ok_to_submit'] is False
    assert card['correlation_overlap_vs_registry']['duplicate'] is True
    assert card['correlation_overlap_vs_registry']['final_max_forward_3px_overlap'] == 1.0
    assert card['holdout_dti']['ci95'][1] < 0.152718
    assert len(card['note']) <= 140


def test_h82_links_not_authorized():
    card = json.loads((ROOT/'evidence/h82_run_card.json').read_text())
    receipt = json.loads((ROOT/'docs/data/submission-authorization.json').read_text())
    lanes = json.loads((ROOT/'docs/data/lanes.json').read_text())['lanes']
    row = next(x for x in lanes if x['name'] == card['submission_name'])
    assert row['protocol_unique'] is False and row['sha256'] == card['raster_sha256']
    assert receipt['repository_decision'] == 'NOT OK TO SUBMIT'
    release = next(x for x in receipt['artifacts'] if x['file'] == row['file'])
    assert release['authorized_to_submit'] is False
    assert release['run_card_audit']['literal_duplicate_recorded'] is True
    assert release['run_card_audit']['registry_incomplete'] is True
    for page in ('index.html','executive-summary.html','h82-executive-summary.html'):
        s = (ROOT/'docs'/page).read_text()
        assert row['file'] in s and 'NOT OK TO SUBMIT' in s
