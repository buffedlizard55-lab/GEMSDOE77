import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import ndimage as ndi
from gems57 import metric, spatial, evaluate_holdout as ev
from gems77.urank_features import curvature, normalized, build
ROOT=Path(__file__).resolve().parents[1]


def test_shared_code_byte_identical():
    pins=json.loads((ROOT/'registry/template_pin.json').read_text())
    assert pins['files']
    for name,digest in pins['files'].items():
        assert hashlib.sha256((ROOT/'vendor/gems57'/name).read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('seed',[7,11,13])
def test_exact_metric_matches_bruteforce(seed):
    rng=np.random.default_rng(seed)
    truth=rng.random((22,24))<.08
    p=np.where(rng.random(truth.shape)<.12,rng.random(truth.shape),0)
    a=metric.dti_exact(p,truth); b=metric.dti_bruteforce(p,truth)
    for k in ['tp','fp','fn','dti']: assert a[k]==pytest.approx(b[k],abs=1e-10)
    assert a['fn']==pytest.approx(truth.sum()-a['tp'])


@pytest.mark.parametrize('offset',[0,1,2,3])
def test_kernel_radius_and_asymmetry(offset):
    truth=np.zeros((15,15),bool);truth[7,7]=True
    p=np.zeros(truth.shape);p[7,7+offset]=1
    assert metric.dti_exact(p,truth)['dti']==pytest.approx(max(1-offset/3,0))
    assert (metric.ALPHA,metric.BETA,metric.R_M)==(.2,.8,300)


def test_whole_component_hidden_in_each_intersecting_quadrant():
    cat=np.zeros((60,60),bool);cat[12,20:44]=True;cat[40:46,10]=True;cat[45,43:50]=True
    valid=np.ones(cat.shape,bool)
    for fold in spatial.folds(cat,valid,buffer_px=3):
        labels,n=ndi.label(cat,np.ones((3,3)))
        for i in range(1,n+1):
            if (fold['truth'] & (labels==i)).any():
                assert not fold['visible'][labels==i].any()
                assert not fold['train'][labels==i].any()
        assert not (fold['visible'] & fold['truth']).any()
        assert not (fold['train'] & fold['region']).any()
        assert fold['receipt']['evaluation_region_label_blind']


def test_evaluator_masks_only_visible_and_pools_terms():
    shape=(30,30);valid=np.ones(shape,bool)
    truth=np.zeros(shape,bool);truth[10,10]=True
    visible=np.zeros(shape,bool);visible[20,20]=True
    p=np.zeros(shape);p[10,10]=1;p[20,20]=1
    fold={'truth':truth,'region':valid,'visible':visible}
    a,terms=ev.evaluate(p,fold,valid,block_side=10)
    assert a['dti']==pytest.approx(1)
    assert a['n_emitted']==1
    assert terms.sum(axis=0)[3]==1
    assert ev.from_terms(terms.sum(axis=0))==pytest.approx(a['dti'])


def test_aligned_physical_block_bootstrap_rejects_changed_truth():
    a=np.array([[1,2,3,4],[2,3,4,6]],float)
    b=a.copy();b[0,3]+=1
    with pytest.raises(ValueError,match='same integer truth'):
        ev.pooled_summary({'candidate':a,'control':b},draws=100,candidate='candidate')


@pytest.mark.parametrize('bad',[np.nan,np.inf,-.1,1.1])
def test_shared_evaluator_rejects_bad_predictions(bad):
    p=np.zeros((10,10));p[4,4]=bad
    truth=np.zeros(p.shape,bool);truth[2,2]=True
    fp=np.ones(p.shape,bool)
    with pytest.raises(ValueError,match='finite'):
        ev.evaluate(p,{'truth':truth,'visible':~fp,'region':fp},fp)


def test_hessian_directionality():
    x=np.arange(65,dtype=np.float32)
    a=np.broadcast_to(np.exp(-((x-32)/3)**2),(65,65)).copy()
    strength,anis=curvature(a)
    assert strength[32,32]>0
    assert anis[32,32]>.95
    assert np.all((anis>=0)&(anis<=1))


def test_no_feature_uses_catalogue():
    # Deliberate API boundary, no hidden labels can enter build().
    import inspect
    assert list(inspect.signature(build).parameters)==['raw','footprint']
    raw=np.full((3,60,60),150,np.uint8)
    with pytest.raises(ValueError,match='degenerate'):
        build(raw,np.ones((60,60),bool))


def test_normalization_fails_closed():
    valid=np.ones((3,3),bool)
    for a in [np.zeros((3,3)),np.full((3,3),np.nan),-np.ones((3,3))]:
        with pytest.raises(ValueError): normalized(a,valid)
    p,s=normalized(np.arange(9).reshape(3,3),valid)
    assert p.min()>=0 and p.max()<=1 and s>0


def test_diagonal_nodata_support_fails_closed_without_new_experiment():
    raw=np.full((3,80,80),150,np.uint8)
    raw[:,30:35,30:35]=0
    with pytest.raises(ValueError,match='Gaussian-stencil-safe'):
        build(raw,np.ones((80,80),bool))
    audit=json.loads((ROOT/'evidence/support_audit.json').read_text())
    assert audit['eligible_cells_with_potential_nodata_stencil_contact']==15044
    assert audit['catalogue_positive_pixels_in_risky_cells']==365
    assert not audit['score_recomputed'] and not audit['artifact_changed']
