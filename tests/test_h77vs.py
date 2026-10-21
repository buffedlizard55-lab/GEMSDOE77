"""Unit and artifact verification tests for H77V (Tufa-Deconfounded Vent-Sinter & Conductive Heat-Flow Conduit Coupling)."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from gems77v.fields import (
    FORBIDDEN_COLUMNS,
    H77V_NAMES,
    _rank01,
    anisotropic_strike_advect,
    build_h77v_fields,
    extract_paleo_sinter_vs_tufa,
    extract_wellspring_disequilibrium,
)

ROOT = Path(__file__).resolve().parents[1]


def test_tie_aware_rank01_eliminates_row_major_tie_ramp():
    """Verify IR-H77V-05 fix: tied background zeros receive identical rank, not a north-south ramp."""
    foot = np.ones((32, 32), dtype=bool)
    arr = np.zeros((32, 32), dtype=np.float32)
    arr[4, 4] = 5.0
    arr[28, 28] = 10.0
    r = _rank01(arr, foot)
    # All tied zero cells at top (row 0) and bottom (row 31) must have identical rank 0.0
    assert float(r[0, 16]) == 0.0
    assert float(r[31, 16]) == 0.0
    assert float(r[28, 28]) == 1.0
    assert 0.0 < float(r[4, 4]) < 1.0


def test_extract_paleo_sinter_vs_tufa_separates_lahontan_tufa(tmp_path: Path):
    """Verify IR-H77V-01 fix: Sinter/Alteration/Sulfur go to vent_imp while Tufa goes to tufa_imp."""
    zpath = tmp_path / "paleo_test.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(
            "Paleo_geothermal_final.csv",
            "Name,Deposit_type,Age,Lat_deg_NAD83,Long_deg_NAD83\n"
            "Beowawe,Sinter,Quaternary,40.5, -116.5\n"
            "Pyramid,Tufa,Quaternary,40.1, -119.5\n",
        )
    tr = Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
    foot = np.ones((3730, 3292), dtype=bool)
    vent_imp, tufa_imp, stats = extract_paleo_sinter_vs_tufa(zpath, tr, (3730, 3292), foot)
    assert stats["vent_records_in_footprint"] == 1
    assert stats["tufa_records_in_footprint"] == 1
    assert int((vent_imp > 0).sum()) == 1
    assert int((tufa_imp > 0).sum()) == 1
    assert not np.any((vent_imp > 0) & (tufa_imp > 0))


def test_extract_wellspring_fixes_vals0_index_shift_and_missing_temp(tmp_path: Path):
    """Verify IR-H77V-02 fix: missing temp_c does not shift geothermquartz_c into surface temp."""
    ws_csv = tmp_path / "ws_test.csv"
    with ws_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "thermalclass", "row", "col", "temp_c", "geothermquartz_c", "geothermchalc_c", "geothermcat_c"])
        # Row 1: Hot spring with missing temp_c but valid geothermquartz_c = 165.0
        w.writerow(["Thermal_Springs", "Hot", "10", "12", "", "165.0", "140.0", ""])
        # Row 2: Warm spring with missing temp_c and no geothermometers
        w.writerow(["Thermal_Springs", "Warm", "20", "22", "", "", "", ""])
    foot = np.ones((32, 32), dtype=bool)
    diseq, upflow, stats = extract_wellspring_disequilibrium(ws_csv, None, (32, 32), foot)
    assert stats["recovered_hot_warm_missing_temp"] == 2
    assert stats["fixed_geotherm_without_temp"] == 1
    assert float(diseq[10, 12]) > 0.0
    assert float(upflow[10, 12]) > float(upflow[20, 22]) > 0.0
    assert "dist_known_fault_px" in FORBIDDEN_COLUMNS


def test_anisotropic_strike_advect_elongates_along_strike():
    imp = np.zeros((48, 48), dtype=np.float32)
    imp[24, 24] = 1.0
    # Strike along y-axis (ty=1, tx=0)
    ty = np.ones((48, 48), dtype=np.float32)
    tx = np.zeros((48, 48), dtype=np.float32)
    out = anisotropic_strike_advect(imp, ty, tx, sig_cross=2.0, sig_along=10.0, decay_px=30.0)
    # Response 8 px along strike (y=32, x=24) must exceed response 8 px across strike (y=24, x=32)
    assert float(out[32, 24]) > float(out[24, 32])


def test_build_h77v_fields_all_finite_in_0_1():
    rng = np.random.default_rng(7721)
    foot = np.zeros((48, 48), dtype=bool)
    foot[6:42, 6:42] = True
    needed = [
        "rtp", "iso_grav_anom", "depth_to_base_surf", "cond_surf",
        "geod_dilaterate", "geod_shearrate", "ieq_n100a15", "deq_n100a15",
        "det_elev", "det_elev_slope",
    ]
    bands = {nm: rng.normal(size=(48, 48)).astype(np.float32) for nm in needed}
    bands["rtp"][0, 0] = -3.4e38
    out, diag = build_h77v_fields(bands, foot)
    assert list(out.keys()) == H77V_NAMES
    assert diag["source_audit"]["sgmc_in_features"] is False
    assert diag["source_audit"]["prior_submissions_in_features"] is False
    for nm in H77V_NAMES:
        arr = out[nm]
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert np.all(arr[~foot] == 0.0)
        assert diag["channels"][nm]["nan_count"] == 0


def test_h77v_artifact_matches_run_card_and_is_negative():
    card = json.loads((ROOT / "evidence" / "run_card_h77v.json").read_text())
    tif = ROOT / card["raster_path"]
    assert tif.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    assert card["verdict"] == "negative"
    assert card["ok_to_download"] is True
    assert card["ok_to_submit"] is False
    assert card["safe_to_download_and_submit"] is False
    assert card["protocol_unique"] is False
    assert card["informative_unique"] is True
    assert card["weekly_slots_used"] == 0
    assert card["organizer_confirmed_score_for_this_tif"] is None
    assert len(card["submission_note"]) <= 140
    with rasterio.open(tif) as src:
        a = src.read(1)
        assert src.count == 1 and src.dtypes == ("float32",)
        assert src.crs.to_epsg() == 32611 and src.shape == (3730, 3292)
    assert np.isfinite(a).all() and float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert int((a > 0).sum()) == 37654
    for rel in ("docs/h77vs.html", "docs/h77vs-executive-summary.html", "docs/index.html", "README.md"):
        txt = (ROOT / rel).read_text()
        assert card["submission_name"] in txt
        assert "NOT OK TO SUBMIT" in txt

