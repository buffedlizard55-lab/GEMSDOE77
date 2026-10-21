import hashlib, json, zipfile
from pathlib import Path
import numpy as np, rasterio
ROOT = Path(__file__).resolve().parents[1]
NAME = "gems77-h84-orgate-collar-37654dots-20261021-2d0a731a-zeros"

def test_h84_artifact_negative_and_valid():
    card = json.loads((ROOT/"evidence/h84_run_card.json").read_text())
    assert card["verdict"] == "negative" and "NOT OK TO SUBMIT" in card["ok_to_submit"]
    assert len(card["submission_note"]) <= 140
    p = ROOT/"docs"/"downloads"/f"{NAME}.tif"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == card["raster_sha256"]
    with rasterio.open(p) as ds, rasterio.open(ROOT/"data/bridge/sample_submission.tif") as ref:
        a = ds.read(1)
        assert ds.count == 1 and ds.dtypes == ("float32",)
        assert (ds.shape, str(ds.crs), tuple(ds.transform)[:6]) == (ref.shape, str(ref.crs), tuple(ref.transform)[:6])
        assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
        assert int((a > 0).sum()) == 37654
    with zipfile.ZipFile(p.with_suffix(".zip")) as z:
        assert z.namelist() == [p.name] and z.read(p.name) == p.read_bytes()
    # The historical artifact is listed for inspection only, not as a new or approved submission.
    lanes = json.loads((ROOT/"docs/data/lanes.json").read_text())["lanes"]
    row = [x for x in lanes if x["lane"].startswith("H84:")]
    assert len(row) == 1
    row = row[0]
    assert row["sha256"] == card["raster_sha256"] and row["name"] == card["submission_name"]
    assert row["note"] == card["submission_note"] and len(row["note"]) <= 140
    assert row["verdict"] == "negative" and row["protocol_unique"] is False
    assert row["informative_unique"] is False and row["beats_holdout_best"] is False
    assert "Not OK to submit" in row["upload"]
    # The shipped holdout arm is below its matched collar control and repo incumbent.
    assert card["holdout"]["experiments"]["Exp2_OR_gate_SHIPPED"]["dti"] < 0.152718
