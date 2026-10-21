"""Unit, integration, and artifact verification tests for Lane H77H."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from gems77hdd.allocator import allocate_h77h_submodular_300m
from gems77hdd.fields import compute_h77h_loco_features, measure_mt_mesh_periodicity_suppression


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_h77h_mt_mesh_notch36_suppresses_35px_spike():
    """Verify IR-H77G-01 fix: sigma=36 px Gaussian smoothing suppresses 35-px mesh steps by >95%."""
    h, w = 560, 560
    yy, xx = np.mgrid[0:h, 0:w]
    # Synthetic piecewise-constant 35-px rectilinear mesh + smooth background
    mesh = ((xx // 35) % 2).astype(np.float32) * 0.8 + ((yy // 35) % 2).astype(np.float32) * 0.8
    smooth_bg = np.exp(-((xx - 280.0) ** 2 + (yy - 280.0) ** 2) / (2.0 * 140.0**2)).astype(np.float32)
    log_cond = (mesh + smooth_bg).astype(np.float32)
    foot = np.ones((h, w), dtype=bool)
    audit = measure_mt_mesh_periodicity_suppression(log_cond, foot)
    assert audit["mesh_35px_suppression_fraction"] > 0.95
    assert audit["mesh_35px_relative_amp_sigma36"] < 0.05 * audit["mesh_35px_relative_amp_sigma2"]


def test_h77h_segment_loco_avoids_k12_truncation_on_long_traces():
    """Verify IR-H77H-02 fix: 4-bucket Segment-LOCO recovers true inter-component distance on >=40 px traces."""
    h, w = 96, 96
    foot = np.ones((h, w), dtype=bool)
    vis = np.zeros((h, w), dtype=bool)
    # Component A: 50-px vertical fault trace at x=30 (exceeds k=12 and max_len_px=16)
    vis[20:70, 30] = True
    # Component B: parallel 50-px fault trace at x=40 (exact distance = 10 px)
    vis[20:70, 40] = True

    loco_tr, loco_inf = compute_h77h_loco_features(visible_mask=vis, footprint=foot)
    assert loco_tr.shape == (h * w, 5)
    assert loco_inf.shape == (h * w, 5)
    assert np.isfinite(loco_tr).all() and np.isfinite(loco_inf).all()

    # Check interior pixel (y=45, x=30) on Component A: distance to other bucket should be <= ~18 px (never 100 px)
    idx_interior = 45 * w + 30
    d_other_recovered = float(np.expm1(loco_tr[idx_interior, 0]))
    assert 2.0 <= d_other_recovered <= 20.0, f"Expected true inter-segment distance, got {d_other_recovered}"

    # Check midpoint (y=45, x=35) between facing strands: stepover relay should be strictly positive
    idx_mid = 45 * w + 35
    assert float(loco_inf[idx_mid, 3]) > 0.25


def test_h77h_submodular_allocator_enforces_overlap_cap():
    """Verify capacity-capped 300 m submodular allocator respects max_overlap_cap."""
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    surf = np.exp(-((yy - 32.0) ** 2 + (xx - 32.0) ** 2) / (2.0 * 14.0**2)).astype(np.float32)
    allowed = np.ones((h, w), dtype=bool)
    allowed[:4, :] = False

    # Construct a synthetic informative registry raster whose halo covers the top half (y < 32)
    ref_dots = np.zeros((h, w), dtype=bool)
    ref_dots[16, 16:48] = True
    halo = ndimage.distance_transform_edt(~ref_dots) <= 3.0
    packed = np.packbits(halo.ravel()[None, :], axis=1)

    dots = allocate_h77h_submodular_300m(
        surf,
        allowed,
        target_n=60,
        support_quantile=0.50,
        gamma=1.85,
        min_sep_px=0.0,
        packed_halos=packed,
        max_overlap_cap=0.25,
    )
    assert dots.dtype == bool
    assert int(dots.sum()) == 60
    assert not (dots & (~allowed)).any()
    ov = float((dots & halo).sum()) / float(dots.sum())
    assert ov <= 0.25 + 1e-6


def test_h77h_artifact_run_card_and_site_consistency():
    """Verify H77H-DD frozen run card, GeoTIFF/ZIP bytes, canary receipt, and site pages."""
    rc = json.loads((ROOT / "evidence/run_card_h77hdd.json").read_text())
    e1 = json.loads((ROOT / "evidence/h77hdd_e1_canary.json").read_text())
    e2 = json.loads((ROOT / "evidence/h77hdd_e2_holdout.json").read_text())
    uq = json.loads((ROOT / "evidence/h77hdd_uniqueness.json").read_text())
    lanes = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"]
    row = next(x for x in lanes if x["lane"].startswith("H77H-DD"))

    tif = ROOT / rc["artifact"]["tif_path"]
    zip_p = ROOT / rc["artifact"]["zip_path"]
    assert tif.is_file() and zip_p.is_file()
    assert _sha256(tif) == rc["artifact"]["tif_sha256"] == row["sha256"]
    assert _sha256(zip_p) == rc["artifact"]["zip_sha256"] == row["zip_sha256"]

    with zipfile.ZipFile(zip_p) as zf:
        assert zf.namelist() == [tif.name]
        assert zf.read(tif.name) == tif.read_bytes()

    with rasterio.open(ROOT / "data/bridge/sample_submission.tif") as s:
        samp = s.read(1)
        strans = list(s.transform)[:6]
        scrs = s.crs.to_epsg()
    with rasterio.open(ROOT / "data/bridge/labels.tif") as s:
        lab = s.read(1)
    foot = np.isfinite(samp)
    cat = (lab > 0) & foot

    with rasterio.open(tif) as ds:
        arr = ds.read(1)
        assert ds.crs.to_epsg() == scrs == 32611
        assert (ds.height, ds.width) == (3730, 3292)
        assert ds.count == 1 and ds.dtypes[0] == "float32"
        assert list(ds.transform)[:6] == strans
        assert ds.nodata is None

    assert not np.isnan(arr).any() and not np.isinf(arr).any()
    assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
    assert int((arr[~foot] != 0.0).sum()) == 0
    dots = arr > 0.5
    assert int(dots.sum()) == 36500 == rc["artifact"]["emitted_dots"]
    assert int((dots & cat).sum()) == 0

    # Leakage canary + MT mesh suppression checks
    assert e1["canary_seed20"]["passed"] is True and e1["canary_seed30"]["passed"] is True
    assert e1["canary_seed20"]["max_overall"]["symmetric_auc"] <= 0.90
    assert e1["canary_seed30"]["max_overall"]["symmetric_auc"] <= 0.90
    assert e1["static_diagnostics"]["mt_mesh_periodicity_audit_ir_h77g_01"]["mesh_35px_suppression_fraction"] > 0.99

    # Holdout DTI checks
    s20 = rc["holdout_dti"]["seed_20_primary"]
    s30 = rc["holdout_dti"]["seed_30_replication"]
    assert s20["withheld_positive_pixels"] == 11321
    assert s30["withheld_positive_pixels"] == 11342
    assert s20["A_h77h_full"]["dti"] > s20["A0_h33_foldfair"]["dti"] > s20["A_std_control"]["dti"]
    assert s30["A_h77h_full"]["dti"] > s30["A0_h33_foldfair"]["dti"] > s30["A_std_control"]["dti"]
    assert s20["A_h77h_unique_cap"]["dti"] > s20["A0_h33_foldfair"]["dti"]
    assert s30["A_h77h_unique_cap"]["dti"] > s30["A0_h33_foldfair"]["dti"]

    # Uniqueness & fail-closed checks
    p2_info = uq["phase2_final_dots"]["informative"]
    assert p2_info["n_informative"] == 54
    assert p2_info["informative_firings_overlap_gt_0p70"] == 0
    assert p2_info["worst_forward_overlap_3px"] <= 0.70
    assert p2_info["worst_abs_spearman"] <= 0.90
    assert rc["verdict"] == "negative"
    assert rc["protocol_unique"] is False
    assert rc["ok_to_download"] is True and rc["ok_to_submit"] is False
    assert len(rc["submission_note"]) <= 140

    # Site pages check
    for page_name in ("h77hdd.html", "h77hdd-executive-summary.html", "index.html", "executive-summary.html"):
        txt = (ROOT / "docs" / page_name).read_text()
        assert rc["submission_name"] in txt
        assert "NOT OK TO SUBMIT" in txt or "Not OK to submit" in txt


def test_lia_build_site_regression_ir_h77h_01():
    """Regression test for IR-H77H-01: ensure scripts/lia_build_site.py runs cleanly."""
    sub_json = json.loads((ROOT / "docs/data/submission.json").read_text())
    assert "zip" in sub_json and "primary" in sub_json
    assert sub_json["primary"]["n_pass"] == sub_json["primary"]["n_checks"] == 17
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lia_build_site.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "site built:" in proc.stdout
