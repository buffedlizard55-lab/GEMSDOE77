import rasterio, numpy as np, pathlib
from pathlib import Path

def test_validator_zeros():
    p = Path("docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.tif")
    if not p.exists():
        import pytest; pytest.skip("TIF not yet generated")
    with rasterio.open(p) as s:
        assert s.width==3292 and s.height==3730
        assert "32611" in str(s.crs)
        assert s.dtypes[0]=="float32" and s.count==1
        assert s.nodata is None
        a=s.read(1)
        assert np.isfinite(a).all(), "not all finite"
        assert (a>=0).all() and (a<=1).all(), "outside [0,1]"
        # footprint
        with rasterio.open("data/bridge/sample_submission.tif") as tpl:
            foot=np.isfinite(tpl.read(1))
        assert int((a>0.5).sum())==44090
        assert int((a[foot]>0.5).sum())==44090
        # 0 on catalogue
        import rasterio as rio
        with rio.open("data/bridge/labels.tif") as lab:
            labels=lab.read(1)==1
        assert int(((a>0.5) & labels).sum())==0

def test_validator_nan_twin():
    p = Path("docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-nan.tif")
    if not p.exists():
        import pytest; pytest.skip("nan twin not yet generated")
    with rasterio.open(p) as s:
        a=s.read(1)
        with rasterio.open("data/bridge/sample_submission.tif") as tpl:
            foot=np.isfinite(tpl.read(1))
        # inside footprint finite and in [0,1]
        assert np.isfinite(a[foot]).all()
        assert (a[foot]>=0).all() and (a[foot]<=1).all()
        # outside nan (at least mostly)
        assert np.isnan(a[~foot]).all() or (a[~foot]==0).all()
