"""Comprehensive verification tests for Lane `H77-VDS` (`gems77vds`) across GeoTIFF bytes,
ZIP membership, NaN twin identity, dual-seed holdout CIs, 985-witness post-merge audit,
release-gate schema compliance, and Dinkelbach submodular allocator unit properties.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_h77vds_geotiff_zip_and_nan_twin_integrity():
    rc = json.loads((ROOT / "evidence" / "run_card_h77vds.json").read_text())
    sub = rc["submission_artifacts"]
    tif_path = ROOT / sub["primary_geotiff"]
    zip_path = ROOT / sub["primary_zip"]
    nan_path = ROOT / sub["nan_twin_geotiff"]
    sub_copy = ROOT / sub["submissions_dir_copy"]

    assert tif_path.is_file() and zip_path.is_file() and nan_path.is_file() and sub_copy.is_file()
    assert _sha256(tif_path) == sub["primary_geotiff_sha256"] == rc["raster_sha256"]
    assert _sha256(sub_copy) == sub["primary_geotiff_sha256"]
    assert _sha256(zip_path) == sub["primary_zip_sha256"]
    assert _sha256(nan_path) == sub["nan_twin_geotiff_sha256"]
    assert len(rc["submission_name"]) <= 120 and len(rc["submission_note"]) <= 120

    with rasterio.open(tif_path) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert ds.shape == (3730, 3292)
        assert ds.crs.to_epsg() == 32611
        assert ds.nodata is None
        arr = ds.read(1)

    assert np.isfinite(arr).all()
    assert set(np.unique(arr).tolist()) == {0.0, 1.0}
    assert int((arr == 1.0).sum()) == 44_090

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert names == [tif_path.name]
        assert hashlib.sha256(zf.read(names[0])).hexdigest() == rc["raster_sha256"]

    with rasterio.open(nan_path) as ds_nan:
        arr_nan = ds_nan.read(1)
    fin = np.isfinite(arr_nan)
    assert int(fin.sum()) == 5_167_373
    assert np.array_equal(arr_nan[fin], arr[fin])
    assert float(arr[~fin].max()) == 0.0


def test_h77vds_holdout_uniqueness_and_release_gate_compliance():
    can = json.loads((ROOT / "evidence" / "h77vds_canaries.json").read_text())
    e1 = json.loads((ROOT / "evidence" / "h77vds_exp1_ablation.json").read_text())
    e2 = json.loads((ROOT / "evidence" / "h77vds_exp2_holdout.json").read_text())
    uq = json.loads((ROOT / "evidence" / "h77vds_uniqueness_audit.json").read_text())
    rc = json.loads((ROOT / "evidence" / "run_card_h77vds.json").read_text())
    auth = json.loads((ROOT / "docs" / "data" / "submission-authorization.json").read_text())

    assert can["channels_tested"] == 106 and can["passed"] is True
    assert rc["leakage_audit"]["seed30_max_symmetric_auc"] < 0.90
    assert rc["leakage_audit"]["seed20_max_symmetric_auc"] < 0.90
    assert rc["leakage_audit"]["sgmc_channels_used"] == 0

    s_b3 = e1["pooled_summary"]["scores"]["E1_B3_full_phys98_ens"]["dti"]
    s_b0 = e1["pooled_summary"]["scores"]["E1_B0_raw19_signfixed"]["dti"]
    assert s_b3 > s_b0 + 0.04

    s30 = e2["seeds"]["30"]["budgets"]["44090"]["pooled_summary"]["scores"]["H77E_primary_regcap"]
    s20 = e2["seeds"]["20"]["budgets"]["44090"]["pooled_summary"]["scores"]["H77E_primary_regcap"]
    assert s30["dti"] > 0.174 and s20["dti"] > 0.160

    assert uq["sparse_discrete_witnesses_checked"] >= 840
    assert uq["sparse_max_forward_overlap_3px"] <= 0.6505 < 0.70
    assert uq["sibling_max_forward_overlap_3px"] < 0.60
    assert uq["surface_max_spearman"] <= 0.90
    assert uq["final_max_spearman"] <= 0.90
    assert uq["template_literal_reading_duplicate"] is True

    # Verify clean integration with `scripts/release_gate.py`
    entry = next(x for x in auth["artifacts"] if x["file"].endswith(f"{rc['submission_name']}.tif"))
    assert entry["authorized_to_submit"] is False
    rec = next(
        x for x in auth["historical_run_card_reconciliation"] if x["run_card"] == "evidence/run_card_h77vds.json"
    )
    assert rec["operative_decision"] == "NOT OK TO SUBMIT"
    assert rec["historical_promotion_conflict"] is False


def test_h77vds_dinkelbach_allocator_and_quadrant_calibration():
    from gems77vds import dinkelbach as DK

    foot = np.ones((40, 40), dtype=bool)
    raw = np.full((40, 40), 0.10, dtype=np.float32)
    raw[20:, :] = 0.05
    raw[5:8, 5:8] = 0.85
    raw[28:31, 28:31] = 0.45

    cal_q = DK.bayes_prior_calibrate(raw, foot, pi_true=0.02, quadrant_calibrate=True)
    assert np.isfinite(cal_q).all() and cal_q[6, 6] > 0.10 and cal_q[29, 29] > 0.10

    w_halo = np.zeros((1600, 1), dtype=np.uint8)
    w_halo[:800, 0] = 1
    packed = np.packbits(w_halo, axis=1)
    sel, st = DK.allocate_dinkelbach_submodular_dti(
        cal_q,
        foot,
        budget=12,
        min_sep_px=2.0,
        pi_true=0.02,
        candidate_pool=1600,
        packed_halos_fp=packed,
        fp_lut=np.arange(1600, dtype=np.int32),
        n_sparse_rasters=1,
        overlap_cap=0.50,
    )
    assert int(sel.sum()) == 12 and st["max_sparse_overlap_achieved"] <= 0.50 + 1e-9
