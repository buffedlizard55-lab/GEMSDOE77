"""Comprehensive unit, holdout, GeoTIFF, uniqueness, and site tests for Lane H77E."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "vendor"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gems57.evaluate_holdout import evaluate
from gems77e.allocate import (
    PortfolioCapacityIndex,
    allocate_dinkelbach_submodular,
    allocate_peak_nms_baseline,
)
from gems77e.features import compute_sloco_mechanics_fp


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_h77e_feature_source_invariants_zero_sgmc_and_zero_forbidden():
    """Verify src/gems77e/ reads zero SGMC rasters and zero forbidden catalogue columns."""
    src_txt = (ROOT / "src/gems77e/features.py").read_text()
    assert "FORBIDDEN_POINT_COLUMNS" in src_txt
    assert '"sgmc_channels_used": 0' in src_txt
    assert '"prior_rasters_read": 0' in src_txt


def test_sloco_strict_self_component_exclusion():
    """Verify SLOCO training distance strictly excludes each segment's own bucket."""
    h, w = 60, 60
    fp = np.ones((h, w), dtype=bool)
    vis = np.zeros((h, w), dtype=bool)
    vis[10, 5:14] = True
    vis[48, 42:52] = True
    geod = np.full((h, w), 0.5, dtype=np.float32)

    x_train, x_infer = compute_sloco_mechanics_fp(vis, fp, geod, n_buckets=4)
    assert x_train.shape == (7, h * w)
    assert x_infer.shape == (7, h * w)
    assert np.all(np.isfinite(x_train)) and np.all(np.isfinite(x_infer))
    idx_seg = 48 * w + 46
    assert float(x_infer[0, idx_seg]) == 0.0
    assert float(x_train[0, idx_seg]) >= 20.0


def test_dinkelbach_submodular_beats_peak_nms_on_linear_fault():
    """Verify exact 300 m kernel cover outperforms peak-NMS when dense false plateau competes with linear fault."""
    h, w = 90, 90
    foot = np.ones((h, w), dtype=bool)
    cat = np.zeros((h, w), dtype=bool)
    truth = np.zeros((h, w), dtype=bool)
    truth[55, 8:82] = True

    rng = np.random.default_rng(7)
    prop = rng.uniform(0.005, 0.02, size=(h, w)).astype(np.float32)
    prop[55, 8:82] = 0.72
    prop[8:26, 8:26] = 0.85  # localized compact mimic patch vs 74-px linear fault

    sub_dots = allocate_dinkelbach_submodular(prop, foot & (~cat), target_n=32, support_quantile=0.70)
    nms_dots = allocate_peak_nms_baseline(prop, foot & (~cat), target_n=32, min_sep_px=1.0)

    fold = {"region": foot, "truth": truth, "visible": cat}
    dti_sub = evaluate(sub_dots.astype(np.float32), fold, foot)[0]["dti"]
    dti_nms = evaluate(nms_dots.astype(np.float32), fold, foot)[0]["dti"]
    assert int(sub_dots.sum()) == 32
    assert dti_sub > dti_nms


def test_uniqueness_constraint_strictly_enforced(tmp_path: Path):
    """Verify PortfolioCapacityIndex keeps forward 3-px overlap strictly <= max_overlap_cap."""
    h, w = 64, 64
    foot = np.ones((h, w), dtype=bool)
    cat = np.zeros((h, w), dtype=bool)
    rng = np.random.default_rng(99)
    prop = rng.uniform(0.01, 0.95, size=(h, w)).astype(np.float32)

    allow_idx = np.flatnonzero(foot & (~cat))
    n_bytes = (allow_idx.size + 7) // 8
    packed = np.zeros((2, n_bytes), dtype=np.uint8)
    packed[0, : n_bytes // 2] = 0xFF
    packed[1, n_bytes // 4 : (3 * n_bytes) // 4] = 0xFF

    packed_path = tmp_path / "packed.npy"
    meta_path = tmp_path / "meta.json"
    np.save(packed_path, packed)
    meta_path.write_text(
        json.dumps(
            {
                "n_valid": 2,
                "n_allow": int(allow_idx.size),
                "entries": [
                    {"tag": "t", "name": "r0", "halo3_frac_allow": 0.50, "is_space_filling": False},
                    {"tag": "t", "name": "r1", "halo3_frac_allow": 0.50, "is_space_filling": False},
                ],
            }
        )
    )
    cap_idx = PortfolioCapacityIndex(packed_path, meta_path, foot, cat)
    dots = allocate_dinkelbach_submodular(
        prop,
        foot & (~cat),
        target_n=120,
        support_quantile=0.50,
        capacity_index=cap_idx,
        max_overlap_cap=0.65,
    )
    max_ov, firings, _ = cap_idx.max_informative_overlap(dots)
    assert int(dots.sum()) == 120
    assert firings == 0
    assert max_ov <= 0.65 + 1e-9


def test_h77e_deliverables_run_card_geotiff_and_pages():
    """Verify full H77E GeoTIFFs, SHA-256 digests, dual-seed CIs, 816-raster audit, and Pages UI."""
    rc = json.loads((ROOT / "evidence/h77e/run_card.json").read_text())
    sub = rc["submission_artifact"]
    val = sub["validator_receipt"]
    uq = rc["uniqueness_audit"]
    can = rc["leakage_canary_audit"]

    # 1. Leakage canaries: 127/127 <= 0.90
    assert can["passed"] is True
    assert can["violator_count"] == 0
    assert can["n_features_checked"] == 127
    assert can["max_overall"]["symmetric_auc"] <= 0.90

    # 2. Primary GeoTIFF + ZIP + companion GeoTIFF byte integrity & zero-NaN [0, 1] grid checks
    tif_path = ROOT / sub["primary_geotiff"]
    zip_path = ROOT / sub["primary_zip"]
    pure_path = ROOT / sub["companion_pure_geotiff"]
    assert tif_path.exists() and zip_path.exists() and pure_path.exists()
    assert _sha256_bytes(tif_path.read_bytes()) == sub["primary_geotiff_sha256"]
    assert _sha256_bytes(zip_path.read_bytes()) == sub["primary_zip_sha256"]
    assert _sha256_bytes(pure_path.read_bytes()) == sub["companion_pure_sha256"]
    assert len(sub["submission_note"]) <= 140
    assert val["all_checks_passed"] is True

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == [tif_path.name]
        assert zf.read(tif_path.name) == tif_path.read_bytes()

    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        assert arr.shape == (3730, 3292)
        assert str(src.crs) == "EPSG:32611"
        assert src.nodata is None
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))
        assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
        assert int((arr > 0).sum()) == 36500

    # 3. Dual-seed holdout superiority over fold-fair H33, Peak-NMS, and Random
    for seed_key in ("holdout_dti_seed20_11321px", "holdout_dti_seed30_11342px"):
        sr = rc[seed_key]
        arms = sr["scores"]
        assert arms["h77e_pure_submodular"]["dti"] > 0.154
        assert arms["h77e_portfolio_unique_submodular"]["dti"] > 0.124
        for k in ("h33_foldfair", "h77e_peak_nms_control", "random_uniform_dots"):
            d_u = sr["paired_differences_unique_candidate"][k]
            d_p = sr["paired_differences_pure_candidate"][k]
            assert d_u["delta"] > 0.0 and d_u["ci95"][0] > 0.0, (seed_key, k, d_u)
            assert d_p["delta"] > 0.0 and d_p["ci95"][0] > 0.0, (seed_key, k, d_p)

    # 4. Full 861-raster audit: 0 / 816 informative duplicate firings
    assert uq["unique_valid_rasters_checked"] == 861
    assert uq["informative_rasters_checked_lt_0p70_bg"] == 816
    assert uq["phase1_pre_placement_surface"]["firings_gt_0p90"] == 0
    assert uq["phase1_pre_placement_surface"]["max_abs_spearman"] <= 0.90
    info_p2 = uq["phase2_post_placement_final_dots"]["informative_reading_lt_0p70_bg"]
    assert info_p2["firings_gt_0p70"] == 0
    assert info_p2["max_forward_3px_overlap"] <= 0.70

    # 5. Publication figures + HTML pages + README status banner + zero dead local links
    import re
    for fig in rc["diagnostic_figures"]:
        assert (ROOT / fig).exists() and (ROOT / fig).stat().st_size > 25_000
    for page_name in ("index.html", "h77e-knick.html"):
        txt = (ROOT / "docs" / page_name).read_text()
        assert sub["submission_name"] in txt
        assert sub["primary_geotiff_sha256"] in txt
        for href in re.findall(r'(?:href|src)="([^"]+)"', txt):
            if href.startswith(("http://", "https://", "#", "mailto:")):
                continue
            assert (ROOT / "docs" / href).exists(), (page_name, href)
    readme_txt = (ROOT / "README.md").read_text()
    # The top banner is the current release status, not a permanently pinned
    # historical H77E/H81 summary. H81 remains traceable after newer parallel
    # lanes are added; current authorization status must still be above the fold.
    h81 = json.loads((ROOT / "evidence/h81_run_card.json").read_text())
    h81_name = h81["submission_name"]
    assert h81_name in readme_txt
    assert h81["raster_sha256"] in readme_txt
    assert "OK TO DOWNLOAD FOR INSPECTION: YES" in readme_txt[:6000]
    assert "OK TO SUBMIT: NO" in readme_txt[:6000]
    head_md = readme_txt.split("# Original project request")[0]
    for target in re.findall(r'\]\(([^)]+)\)', head_md):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (ROOT / target).exists(), target
