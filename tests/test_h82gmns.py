import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def test_h82gmns_run_card_and_budget_compliance():
    rc_path = ROOT / "evidence" / "h82gmns" / "run_card.json"
    assert rc_path.is_file()
    card = json.loads(rc_path.read_text())

    assert card["budget_compliance"]["experiments_executed"] == 3
    assert card["budget_compliance"]["experiments_executed"] <= card["budget_compliance"]["max_experiments_allowed"] == 3
    assert card["budget_compliance"]["measured_pipeline_wall_minutes"] <= 120.0
    assert card["budget_compliance"]["within_budget"] is True
    assert card["weekly_slots_used"] == 0

    assert card["verdict"] == "negative"
    assert card["ok_to_download"] is True
    assert card["ok_to_submit"] is False
    assert card["selector_approved"] is False

    can = card["leakage_canary_gate"]
    assert can["passed"] is True
    assert can["n_features_checked"] == 124
    assert can["sgmc_channels_included"] == 0
    assert can["prior_rasters_in_features"] == 0
    assert can["max_symmetric_auc"] <= 0.90


def test_h82gmns_primary_geotiff_and_zip_portal_hygiene():
    card = json.loads((ROOT / "evidence" / "h82gmns" / "run_card.json").read_text())
    art = card["submission_artifact"]

    tif_path = ROOT / art["raster_path"]
    zip_path = ROOT / art["zip_path"]
    sub_tif = ROOT / "submissions" / tif_path.name
    sub_zip = ROOT / "submissions" / zip_path.name
    sample_path = ROOT / "data" / "bridge" / "sample_submission.tif"
    cat_path = ROOT / "data" / "bridge" / "existing_faults.tif"

    assert tif_path.is_file() and zip_path.is_file()
    assert sub_tif.is_file() and sub_zip.is_file()
    assert len(art["submission_name"]) <= 140
    assert len(art["submission_note"]) <= 140

    tif_sha = hashlib.sha256(tif_path.read_bytes()).hexdigest()
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert tif_sha == art["raster_sha256"]
    assert zip_sha == art["zip_sha256"]
    assert hashlib.sha256(sub_tif.read_bytes()).hexdigest() == tif_sha
    assert hashlib.sha256(sub_zip.read_bytes()).hexdigest() == zip_sha

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) == 1 and names[0].endswith(".tif")
        assert hashlib.sha256(zf.read(names[0])).hexdigest() == tif_sha

    with rasterio.open(sample_path) as s_src, rasterio.open(cat_path) as c_src, rasterio.open(tif_path) as src:
        assert src.count == 1
        assert src.dtypes == ("float32",)
        assert (src.height, src.width) == (3730, 3292) == (s_src.height, s_src.width)
        assert src.crs == s_src.crs
        assert str(src.crs) == "EPSG:32611"
        assert src.transform == s_src.transform
        assert src.nodata is None

        arr = src.read(1)
        fp = np.isfinite(s_src.read(1))
        cat = c_src.read(1) > 0

        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert set(np.unique(arr).tolist()).issubset({0.0, 1.0})
        assert int((arr > 0).sum()) == 36500
        assert int((arr[~fp] != 0).sum()) == 0
        assert int((arr[cat] != 0).sum()) == 0


def test_h82gmns_holdout_and_1007_uniqueness_gates():
    card = json.loads((ROOT / "evidence" / "h82gmns" / "run_card.json").read_text())
    s20 = card["holdout_dti"]["seed20"]
    s30 = card["holdout_dti"]["seed30_independent_replication"]
    u = card["uniqueness_audit_1007"]

    # Beats ALL 41 sibling lanes on BOTH Seed 20 (> 0.163962) and Seed 30 (> 0.171600) and Dual-Seed Mean (> 0.168493)
    assert s20["h82_lagrangian_unique"]["dti"] > 0.163962
    assert s30["h82_lagrangian_unique"]["dti"] > 0.171600
    assert card["holdout_dti"]["dual_seed_mean_unique_dti"] > 0.180000
    assert s20["paired_delta_vs_h33_foldfair"]["ci95"][0] > 0.0
    assert s30["paired_delta_vs_h33_foldfair"]["ci95"][0] > 0.0
    assert s20["paired_delta_vs_sloco_hardwall_ablation"]["ci95"][0] > 0.0

    # Uniqueness across all 1,007 valid rasters across 87 repos + all 41 concurrent GEMSDOE77 PRs
    assert u["total_repositories_scanned"] == 87
    assert u["valid_on_grid_checked"] >= 1007
    assert u["phase1_pre_placement_surface_max_abs_rho"] <= 0.90
    assert u["phase2_informative_rasters_checked"] >= 940
    assert u["phase2_informative_firings_gt_0p70"] == 0
    assert u["phase2_max_informative_overlap_3px"] <= 0.70
    assert u["phase2_max_abs_dot_rho"] <= 0.90
    assert u["vendor_dot_overlap_max_abs_diff"] < 1e-5
    assert u["literal_unfiltered_passed"] is False


def test_h82gmns_pages_figures_and_prompt_preservation():
    card = json.loads((ROOT / "evidence" / "h82gmns" / "run_card.json").read_text())
    tif_rel = card["submission_artifact"]["raster_path"].removeprefix("docs/")

    for fig_name in (
        "h82gmns_holdout_comparison.png",
        "h82gmns_uniqueness_1007_audit.png",
        "h82gmns_canary_and_importances.png",
        "h82gmns_nevada_deliverable_map.png",
    ):
        p = ROOT / "docs" / "figures" / fig_name
        assert p.is_file() and p.stat().st_size > 10_000

    for page_name in ("index.html", "executive-summary.html", "h82gmns.html", "h82gmns-executive-summary.html"):
        html_txt = (ROOT / "docs" / page_name).read_text()
        assert tif_rel in html_txt, page_name
        assert "NOT OK TO SUBMIT" in html_txt, page_name

    readme = (ROOT / "README.md").read_text()
    prompt_doc = (ROOT / "docs" / "PROJECT_PROMPT.md").read_text()
    assert prompt_doc in readme
    assert tif_rel in readme
