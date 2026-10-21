from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

from src.gems77vldsa.allocate import PortfolioCapacityIndex, allocate_ldsa_submodular
from src.gems77vldsa.features import _hyperbolic_saddle, _tubular_demag_trough

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_h77v_run_card_and_preregistration_contract():
    prereg = json.loads((ROOT / "registry" / "preregistration_h77vldsa.json").read_text())
    assert prereg["lane_id"] == "H77V-DEALIASED-VENT-DEMAG-KNICK-LDSA"
    assert prereg["hard_invariants"]["max_experiments"] == 3
    assert len(prereg["hypotheses"]) >= 4

    rc_paths = [
        ROOT / "evidence" / "h77v" / "run_card.json",
        ROOT / "evidence" / "run_card_h77vldsa.json",
        ROOT / "docs" / "evidence" / "run_card_h77vldsa.json",
    ]
    for p in rc_paths:
        assert p.exists(), f"Missing run card at {p}"
    rc = json.loads(rc_paths[0].read_text())

    assert rc["lane_id"] == "H77V-DEALIASED-VENT-DEMAG-KNICK-LDSA"
    assert rc["verdict"] == "negative"
    assert "OK TO DOWNLOAD FOR INSPECTION" in rc["status_banner"]
    assert rc["ok_to_submit"] is False
    assert rc["protocol_unique"] is False
    assert rc["informative_unique"] is True
    assert rc["weekly_submission_slots_used"] == 0
    assert rc["experiments_executed"] == 3

    # Leakage canary contract
    can = rc["leakage_canary_gate"]
    assert can["features_checked"] == 140
    assert can["passed"] is True
    assert can["max_overall"]["symmetric_auc"] <= 0.90
    assert can["max_static_physics"]["symmetric_auc"] <= 0.90
    assert can["max_h77v_new_physics"]["symmetric_auc"] <= 0.90

    # Holdout DTI contract (gems57-pooled-hide-v2, both seeds)
    s20 = rc["holdout_dti_seed20_11321px"]["scores"]
    s30 = rc["holdout_dti_seed30_11342px"]["scores"]
    u20 = s20["h77v_portfolio_unique_submodular"]
    u30 = s30["h77v_portfolio_unique_submodular"]
    p20 = s20["h77v_pure_submodular"]
    p30 = s30["h77v_pure_submodular"]

    for item, n_pos in [(u20, 11321), (p20, 11321), (u30, 11342), (p30, 11342)]:
        assert item["evidence_class"] == "HOLDOUT-DTI"
        assert item["evaluator_version"] == "gems57-pooled-hide-v2"
        assert item["withheld_positive_pixels"] == n_pos
        assert len(item["ci95"]) == 2
        assert item["ci95"][0] < item["dti"] < item["ci95"][1]

    # Beats H77E-Knick unique (0.126664 / 0.124727) and H77-Step pure (0.152718)
    assert u20["dti"] > 0.144
    assert u30["dti"] > 0.142
    assert p20["dti"] > 0.152718
    assert p30["dti"] > 0.152718

    # Uniqueness audit contract
    uq = rc["uniqueness_audit"]
    assert uq["phase1_pre_placement_surface"]["pass_le_0p90"] is True
    assert uq["phase1_pre_placement_surface"]["max_abs_spearman"] <= 0.90
    info = uq["phase2_post_placement_final_dots"]["informative_reading_lt_0p70_bg"]
    assert info["informative_unique"] is True
    assert info["firings_gt_0p70"] == 0
    assert info["max_forward_3px_overlap"] <= 0.70
    assert info["max_abs_spearman"] <= 0.90


def test_h77v_geotiff_and_zip_artifacts():
    rc = json.loads((ROOT / "evidence" / "h77v" / "run_card.json").read_text())
    sub = rc["submission_artifact"]
    tif_name = sub["submission_name"]
    zip_name = sub["submission_zip"]
    json_name = sub["submission_json"]

    for base in [ROOT / "submissions", ROOT / "docs" / "downloads"]:
        tif_path = base / tif_name
        zip_path = base / zip_name
        json_path = base / json_name
        assert tif_path.exists()
        assert zip_path.exists()
        assert json_path.exists()
        assert _sha256(tif_path) == sub["tif_sha256"]
        assert _sha256(zip_path) == sub["zip_sha256"]

    assert len(sub["submission_note"]) <= 140
    assert sub["emitted_positive_dots"] == 36500

    tif_path = ROOT / "docs" / "downloads" / tif_name
    with rasterio.open(tif_path) as ds:
        assert ds.crs.to_epsg() == 32611
        assert (ds.height, ds.width) == (3730, 3292)
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert ds.nodata is None
        arr = ds.read(1)

    assert int(np.isnan(arr).sum()) == 0
    assert int(np.isinf(arr).sum()) == 0
    assert float(arr.min()) == 0.0
    assert float(arr.max()) == 1.0
    assert int((arr > 0).sum()) == 36500
    assert int((arr == 1.0).sum()) == 36500

    zip_path = ROOT / "docs" / "downloads" / zip_name
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert names == [tif_name]
        assert hashlib.sha256(zf.read(tif_name)).hexdigest() == sub["tif_sha256"]


def test_h77v_html_and_figures_exist():
    for fig in [
        "fig1_h77v_physics_panels.png",
        "fig2_h77v_holdout_bootstrap.png",
        "fig3_h77v_uniqueness_spectrum.png",
        "fig4_h77v_submission_map.png",
    ]:
        p = ROOT / "docs" / "figures" / "h77v" / fig
        assert p.exists() and p.stat().st_size > 20_000

    for page in ["h77vldsa.html", "h77vldsa-executive-summary.html", "index.html", "executive-summary.html"]:
        txt = (ROOT / "docs" / page).read_text()
        assert "gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.tif" in txt
        assert "861c880e7a56203d197182075765480efaf224ad6cfde342fab511f5f908fa4c" in txt


def test_h77v_operators_and_ldsa_allocator_synthetic(tmp_path: Path):
    yy, xx = np.mgrid[-20:21, -20:21].astype(np.float32)
    # Synthetic demagnetization trough along x = 0: u(x,y) = -exp(-x^2 / 8)
    u_trough = -np.exp(-(xx * xx) / 8.0).astype(np.float32)
    trough_resp = _tubular_demag_trough(u_trough, 2.0)
    assert trough_resp.shape == u_trough.shape
    assert float(trough_resp[20, 20]) > float(trough_resp[20, 5]) * 5.0

    # Synthetic hyperbolic saddle u(x,y) = x^2 - y^2 -> L1 * L2 < 0
    u_saddle = ((xx * xx - yy * yy) / 50.0).astype(np.float32)
    sad_resp = _hyperbolic_saddle(u_saddle, 2.0)
    assert sad_resp.shape == u_saddle.shape
    assert float(sad_resp[20, 20]) > 0.0

    # Synthetic LDSA allocation with uniqueness constraint via PortfolioCapacityIndex
    H, W = 40, 40
    prob = np.zeros((H, W), dtype=np.float32)
    prob[5:35, 5:35] = 0.8
    foot = np.ones((H, W), dtype=bool)
    cat = np.zeros((H, W), dtype=bool)
    d_exc = np.full((H, W), 10.0, dtype=np.float32)
    n_allow = int((foot & (~cat)).sum())
    halo_bits = np.zeros(n_allow, dtype=np.uint8)
    halo_bits[: n_allow // 2] = 1
    packed = np.packbits(halo_bits)[None, :]
    packed_path = tmp_path / "packed.npy"
    meta_path = tmp_path / "meta.json"
    np.save(packed_path, packed)
    meta_path.write_text(
        json.dumps(
            {
                "n_valid": 1,
                "n_allow": n_allow,
                "entries": [
                    {
                        "tag": "SYN",
                        "name": "synthetic_ref.tif",
                        "sha256": "0" * 64,
                        "pos_dots": 100,
                        "halo3_bg_frac": 0.5,
                        "is_space_filling": False,
                    }
                ],
            }
        )
    )
    port_idx = PortfolioCapacityIndex(packed_path, meta_path, foot, cat)
    dots = allocate_ldsa_submodular(
        utility_2d=prob,
        allowed_2d=foot,
        target_n=12,
        support_quantile=0.50,
        capacity_index=port_idx,
        max_overlap_cap=0.65,
    )
    max_ov, firings, _ = port_idx.max_informative_overlap(dots)
    assert int(dots.sum()) == 12
    assert firings == 0
    assert max_ov <= 0.65 + 1e-6
