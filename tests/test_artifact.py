import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pytest
import rasterio
from gems57 import uniqueness
from gems57.grid import HEIGHT,WIDTH,TRANSFORM
from gems57.submission_writer import write_submission
ROOT=Path(__file__).resolve().parents[1]


def test_immutable_candidate_bytes_and_metadata():
    card=json.loads((ROOT/'evidence/run_card.json').read_text());path=ROOT/card['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==card['raster_sha256']
    with rasterio.open(path) as s:
        a=s.read(1)
        assert s.count==1 and s.shape==(HEIGHT,WIDTH) and s.crs.to_epsg()==32611
        assert s.transform==TRANSFORM and s.nodata is None and s.dtypes==('float32',)
    assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
    with zipfile.ZipFile(path.with_suffix('.zip')) as z:
        assert z.namelist()==[path.name]
        assert z.read(path.name)==path.read_bytes()


def test_status_never_equates_format_with_approval():
    card=json.loads((ROOT/'evidence/run_card.json').read_text())
    manifest=json.loads((ROOT/'docs/downloads/manifest.json').read_text())
    assert all(card['validator'].values())
    assert card['verdict']=='negative' and not card['ok_to_submit'] and not card['protocol_unique']
    assert card['registry']['final_dot_gate'].startswith('NOT RUN')
    assert manifest['ok_to_download'] and not manifest['ok_to_submit']
    assert len(card['note'])<=140 and len(card['submission_name'])<=140
    assert card['weekly_slots_used']==0 and card['organizer_confirmed_score'] is None


def test_literal_dense_witness_stops_every_nonempty_set():
    a=np.zeros((25,25),np.float32);a[10,10]=1
    dense=np.full(a.shape,.001)
    assert uniqueness.dot_overlap(a,dense)==1
    cert=json.loads((ROOT/'evidence/registry_saturation.json').read_text())
    assert cert['universal_overlap_blocker'] and cert['uncovered_allowed_pixels']==0
    audit=json.loads((ROOT/'evidence/surface_registry.json').read_text())
    assert audit['stop_required'] and not audit['unique'] and not audit['complete_accessible_scan']
    assert audit['worst_dot_overlap']==1
    assert audit['candidate_file_sha256'] is not None


@pytest.mark.parametrize('bad',[np.nan,np.inf,-.1,1.1])
def test_writer_rejects_without_creating_file(tmp_path,bad):
    p=np.zeros((10,10),np.float32);p[1,1]=bad
    with pytest.raises(ValueError):
        write_submission(tmp_path/'bad.tif',p,ROOT/'data/sample_submission.tif',np.ones(p.shape,bool),note='test',name='test')
    assert not list(tmp_path.iterdir())


def test_writer_rejects_outside_footprint(tmp_path):
    p=np.zeros((10,10),np.float32);p[2,2]=1
    with pytest.raises(ValueError,match='outside'):
        write_submission(tmp_path/'bad.tif',p,'not-used',np.zeros(p.shape,bool),note='test',name='test')
    assert not list(tmp_path.iterdir())


def test_writer_rejects_mapped_fault(tmp_path):
    p=np.zeros((10,10),np.float32);p[2,2]=1
    with pytest.raises(ValueError,match='mapped'):
        write_submission(tmp_path/'bad.tif',p,'not-used',np.ones(p.shape,bool),catalogue=p>0,note='test',name='test')


def test_runner_does_not_rerun_stopped_lane():
    import subprocess,sys
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py')],capture_output=True,text=True)
    assert p.returncode!=0 and 'Do not rerun' in p.stderr


def test_inherited_generator_and_artifact_remain_quarantined():
    import subprocess,sys
    p=subprocess.run([sys.executable,str(ROOT/'scripts/generate_submission.py')],capture_output=True,text=True)
    assert p.returncode!=0 and 'quarantined' in p.stderr
    review=json.loads((ROOT/'evidence/inherited_h77c_review.json').read_text())
    assert not review['ok_to_submit'] and review['shared_registry_check']['stop_required']
    assert review['shared_registry_check']['worst_dot_overlap']==1
    data=json.loads((ROOT/'registry/submission_build.json').read_text())
    assert not data['ok_to_submit'] and not data['run_card']['submission']['portal_ready']
    assert (ROOT/'docs/archive/concurrent-h77c/generate_submission.py.txt').exists()
