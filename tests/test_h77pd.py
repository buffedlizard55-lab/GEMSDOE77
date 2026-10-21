from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_h77pd_preregistration_lock_and_vendor_integrity():
    lock = json.loads((ROOT / "evidence" / "h77pd_prereg_lock.json").read_text())
    prereg_path = ROOT / "registry" / "preregistration_h77pd.json"
    hyp_path = ROOT / "research" / "h77pd_hypotheses.md"
    receipt_path = ROOT / "evidence" / "h77pd_data_receipt.json"
    assert prereg_path.is_file() and hyp_path.is_file() and receipt_path.is_file()
    assert lock["experiments_executed_so_far"] == 0
    assert _sha256(prereg_path) == lock["files"]["registry/preregistration_h77pd.json"]
    assert _sha256(hyp_path) == lock["files"]["research/h77pd_hypotheses.md"]
    assert _sha256(receipt_path) == lock["files"]["evidence/h77pd_data_receipt.json"]

    receipt = json.loads(receipt_path.read_text())
    assert receipt["vendor_gems57_integrity"]["ok"] is True
    assert receipt["vendor_gems57_integrity"]["verified_files"] == 23


def test_h77pd_exp1_leakage_canary_and_ablation_ladder():
    exp1 = json.loads((ROOT / "evidence" / "h77pd_exp1_ablation_holdout.json").read_text())
    assert exp1["evaluator_version"] == "gems57-pooled-hide-v2"
    can = exp1["leakage_canary"]
    assert can["status"] == "PASS"
    assert can["tested_channels"] == 107
    assert can["max_symmetric_auc"] <= 0.90

    sc = exp1["pooled_summary"]["scores"]
    assert sc["Arm0_B0_core_greedy"]["dti"] < sc["Arm2_plus_H77E2_deseam_greedy"]["dti"]
    assert sc["Arm2_plus_H77E2_deseam_greedy"]["dti"] < sc["Arm3_plus_H77E3_step_gap_greedy"]["dti"]
    assert sc["Arm3_plus_H77E3_step_gap_greedy"]["dti"] < sc["Arm3b_phys_only_bayes_submod"]["dti"]
    assert sc["Arm3b_phys_only_bayes_submod"]["dti"] < sc["Arm4_plus_H77E4_ensemble_greedy"]["dti"]
    assert sc["Arm4_plus_H77E4_ensemble_greedy"]["dti"] < sc["Arm5_H77E_rank4_submod"]["dti"]
    assert sc["Arm5_H77E_rank4_submod"]["dti"] < sc["Arm6_H77E_bayes_submod_unconstrained"]["dti"]
    assert sc["Arm6_H77E_bayes_submod_unconstrained"]["dti"] < sc["Arm7_H77E_lagrangian_knapsack_submod"]["dti"]
    assert abs(sc["Arm7_H77E_lagrangian_knapsack_submod"]["dti"] - 0.165580) < 1e-4

    for arm_val in sc.values():
        lo, hi = arm_val["ci95"]
        assert 0.0 <= lo <= arm_val["dti"] <= hi <= 1.0


def test_h77pd_exp2_multiseed_and_blind_quadrant_replication():
    exp2 = json.loads((ROOT / "evidence" / "h77pd_exp2_multiseed_replication.json").read_text())
    s20 = exp2["seed20"]["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]
    s21 = exp2["seed21"]["pooled_summary"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]
    pool3 = exp2["three_seed_pooled_34235px"]["scores"]["Arm7_H77E_lagrangian_knapsack_submod"]
    quad = exp2["blind_4_quadrant_transfer_57448px"]["pooled_summary"]["scores"]["Quad_H77E_Phys96_area_prop_submod"]

    assert s20["dti"] > 0.163 and s21["dti"] > 0.175
    assert pool3["dti"] > 0.168
    assert quad["dti"] > 0.194
    for row in (s20, s21, pool3, quad):
        assert row["ci95"][0] <= row["dti"] <= row["ci95"][1]


def test_h77pd_geotiff_zip_uniqueness_and_run_card_contract():
    rc = json.loads((ROOT / "evidence" / "run_card_h77pd.json").read_text())
    assert rc["experiments_run"] == 3 and rc["weekly_slots_used"] == 0
    assert rc["ok_to_download"] is True and rc["ok_to_submit"] is False
    assert rc["verdict"] == "negative"
    assert rc["technical_informative_uniqueness_passed"] is True

    sp = rc["submission_package"]
    assert 1 <= len(sp["submission_note"]) <= 140
    assert sp["submission_note_chars"] == len(sp["submission_note"])

    tif_sub = ROOT / sp["tif_path"]
    zip_sub = ROOT / sp["zip_path"]
    tif_dl = ROOT / sp["docs_download_tif"]
    zip_dl = ROOT / sp["docs_download_zip"]
    assert tif_sub.is_file() and zip_sub.is_file() and tif_dl.is_file() and zip_dl.is_file()
    assert _sha256(tif_sub) == sp["tif_sha256"] == _sha256(tif_dl)
    assert _sha256(zip_sub) == sp["zip_sha256"] == _sha256(zip_dl)

    with zipfile.ZipFile(zip_sub) as zf:
        assert zf.namelist() == [tif_sub.name]
        assert zf.read(tif_sub.name) == tif_sub.read_bytes()

    with rasterio.open(tif_sub) as ds:
        arr = ds.read(1)
        assert str(ds.crs) == "EPSG:32611"
        assert ds.shape == (3730, 3292)
        assert ds.dtypes == ("float32",)
        assert ds.nodata is None
        assert ds.transform == Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
        assert np.isfinite(arr).all()
        assert int(np.isnan(arr).sum()) == 0 and int(np.isinf(arr).sum()) == 0
        assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
        assert int((arr > 0).sum()) == 37654

    sample_p = ROOT / ".cache/data/sample_submission.tif"
    cat_p = ROOT / ".cache/data/existing_faults.tif"
    if sample_p.is_file() and cat_p.is_file():
        with rasterio.open(sample_p) as ref, rasterio.open(cat_p) as cat_ds:
            foot = np.isfinite(ref.read(1))
            cat = cat_ds.read(1) > 0
            assert int((arr[~foot] != 0).sum()) == 0
            assert int((arr[cat] > 0).sum()) == 0

    ua = rc["uniqueness_audit"]
    assert ua["phase1_surface_max_abs_spearman"] <= 0.90
    assert ua["phase2_dot_max_abs_spearman_all_873"] <= 0.90
    assert ua["informative_discrete_rasters_count"] == 823
    assert ua["informative_max_1way_3px_overlap"] <= 0.70
    assert ua["informative_max_sym_3px_overlap"] <= 0.70
    assert ua["pigeonhole_stride5_lattice_certificate"]["pigeonhole_min_possible_forward_overlap_at_budget_37654"] > 0.70

    for fig_name in (
        "h77pd_fig1_ablation_ladder.png",
        "h77pd_fig2_multiseed_family_benchmark.png",
        "h77pd_fig3_nevada_posterior_and_lidar_gap.png",
        "h77pd_fig4_uniqueness_and_pigeonhole_audit.png",
    ):
        assert (ROOT / "docs" / "assets" / fig_name).is_file()
