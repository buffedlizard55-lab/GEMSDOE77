"""Comprehensive unit & integration test suite for GEMSDOE77 Lane H77D:
Coupled Zero-SGMC Tectono-Hydrothermal Conduits + 300 m Submodular Kernel Cover.
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gems77d.features import compute_visible_mechanics
from gems77d.submodular import KERNEL_300M, allocate_submodular_300m


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kernel_cover_score(dots: np.ndarray, weights: np.ndarray) -> float:
    h, w = dots.shape
    cov = np.zeros((h, w), dtype=np.float32)
    dy_arr, dx_arr = np.nonzero(dots)
    for dy, dx, k in KERNEL_300M:
        rr = dy_arr + dy
        cc = dx_arr + dx
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        np.maximum.at(cov, (rr[ok], cc[ok]), k)
    return float(np.sum(cov * weights))


def test_zero_sgmc_and_zero_label_leakage_source_signatures():
    """Verify no H77D module or script reads SGMC contacts or GDR dist_known_fault_px."""
    checked = [
        ROOT / "src" / "gems77d" / "features.py",
        ROOT / "src" / "gems77d" / "submodular.py",
        ROOT / "scripts" / "h77d_exp1_live_inversion.py",
        ROOT / "scripts" / "h77d_exp2_holdout.py",
        ROOT / "scripts" / "h77d_exp3_build_and_audit.py",
    ]
    for path in checked:
        code_lines = [
            ln
            for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith('"')
        ]
        joined = "\n".join(code_lines)
        assert '["dist_known_fault_px"]' not in joined, f"Leakage key access in {path}"
        assert "['dist_known_fault_px']" not in joined, f"Leakage key access in {path}"
        assert "sgmc_prebaked" not in joined, f"SGMC dependency found in {path}"
        assert "sgmc_contacts" not in joined, f"SGMC dependency found in {path}"


def test_12bit_loco_mechanics_eliminates_self_component_zero_distance():
    """Verify 12-bit color-coding LOCO gives strictly positive distance on every connected component."""
    h, w = 80, 80
    vis = np.zeros((h, w), dtype=bool)
    # Two distinct 8-connected N26.6E fault segments well separated in space
    for i in range(12):
        vis[10 + 2 * i, 15 + i] = True
        vis[11 + 2 * i, 15 + i] = True
        vis[42 + 2 * i, 48 + i] = True
        vis[43 + 2 * i, 48 + i] = True
    foot = np.ones((h, w), dtype=bool)
    geod_rank = np.full((h, w), 0.5, dtype=np.float32)

    feats = compute_visible_mechanics(vis, foot, geod_rank)
    assert feats.shape == (h * w, 4)
    assert np.all(np.isfinite(feats))

    d_grid = feats[:, 0].reshape(h, w)
    lobe_grid = feats[:, 1].reshape(h, w)
    # Every visible-fault pixel MUST measure distance to OTHER components (> 20 px), never 0!
    assert float(d_grid[vis].min()) > 20.0
    # Off-tip propagation lobe beyond segment tip exceeds far-field background
    assert float(lobe_grid[35, 27]) > float(lobe_grid[5, 70])


def test_submodular_300m_allocator_invariants_and_knight_move_gain():
    """Verify exact dot count, allowed-mask containment, min separation, and kernel cover dominance."""
    h, w = 96, 96
    allowed = np.ones((h, w), dtype=bool)
    allowed[:4, :] = False
    p = np.full((h, w), 0.01, dtype=np.float32)

    # Plant two N26.6E (2:1 knight's-move) range-front fault scarps
    scarp_weights = np.zeros((h, w), dtype=np.float32)
    for k in range(36):
        r1, c1 = 12 + 2 * k, 18 + k
        r2, c2 = 12 + 2 * k, 56 + k
        for dc in (-1, 0, 1):
            p[r1, c1 + dc] = 0.85 if dc == 0 else 0.45
            p[r2, c2 + dc] = 0.80 if dc == 0 else 0.42
            scarp_weights[r1, c1 + dc] = p[r1, c1 + dc]
            scarp_weights[r2, c2 + dc] = p[r2, c2 + dc]

    budget = 28
    dots = allocate_submodular_300m(
        p,
        allowed,
        target_n=budget,
        min_sep_px=2.2,
        gamma=3.5,
        support_quantile=0.80,
    )
    assert dots.dtype == bool and dots.shape == (h, w)
    assert int(dots.sum()) == budget
    assert not np.any(dots & (~allowed))

    # Pairwise Euclidean spacing strictly >= 2.2 px
    ys, xs = np.nonzero(dots)
    dy = ys[:, None] - ys[None, :]
    dx = xs[:, None] - xs[None, :]
    dmat = np.sqrt(dy * dy + dx * dx)
    np.fill_diagonal(dmat, 999.0)
    assert float(dmat.min()) >= 2.2 - 1e-6

    # Compare exact 300 m kernel cover against standard 3.0 px greedy disk thinning
    order = np.argsort(-p.ravel())
    greedy = np.zeros((h, w), dtype=bool)
    blocked = ~allowed.copy()
    picked = 0
    for idx in order:
        r, c = divmod(int(idx), w)
        if blocked[r, c]:
            continue
        greedy[r, c] = True
        picked += 1
        if picked == budget:
            break
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr * dr + dc * dc <= 9:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        blocked[rr, cc] = True

    assert _kernel_cover_score(dots, scarp_weights) > _kernel_cover_score(greedy, scarp_weights)


def test_h77d_all_three_experiments_and_run_card_receipts():
    """Verify pre-registered E1, E2, E3 evidence files, CIs, uniqueness, and GeoTIFF bytes."""
    e1 = json.loads((ROOT / "evidence" / "h77d_exp1_live_inversion.json").read_text())
    e2 = json.loads((ROOT / "evidence" / "h77d_exp2_holdout.json").read_text())
    u3 = json.loads((ROOT / "evidence" / "h77d_exp3_uniqueness.json").read_text())
    rc = json.loads((ROOT / "evidence" / "run_card_h77d.json").read_text())

    # E1 live inversion checks
    assert e1["n_distinct_scored_rasters"] == 95
    assert any("1272913" in json.dumps(e1) for _ in [0])
    assert e1["models"]["M_pot_crossgrad_H77D4_offsupport"]["delta_mse_reduction_all95_ci95"][1] < 0.0

    # E2 leakage canaries & dual-seed paired cluster bootstrap dominance
    assert e2["n_static_channels"] == 88
    assert e2["n_visible_mechanics_channels"] == 4
    assert e2["sgmc_channels_used"] == 0
    assert e2["seed30"]["canary_max_symmetric_auc"] <= 0.90
    assert e2["seed20"]["canary_max_symmetric_auc"] <= 0.90
    assert e2["seed30"]["leakage_flags"] == []
    assert e2["seed20"]["leakage_flags"] == []

    for skey, min_dti in [("seed30", 0.145), ("seed20", 0.135)]:
        ps = e2[skey]["budgets"]["B37654_E3"]["pooled_summary"]
        assert ps["scores"]["H77D_full"]["dti"] > min_dti
        for ctrl in [
            "H77D_greedy_alloc",
            "H77D_mech_only",
            "H77D_no_vismech",
            "A5_primary_lidar",
            "A4_primary",
            "A2_standard",
            "A0_h195_foldfair",
            "A0_random",
        ]:
            delta = ps["paired_differences"][ctrl]
            assert delta["delta"] > 0.0, (skey, ctrl)
            assert delta["ci95"][0] > 0.0, f"95% CI includes 0 for {skey} vs {ctrl}: {delta}"

    # E3 uniqueness & GeoTIFF verification
    assert u3["total_prior_rasters_checked"] == 865
    assert u3["phase1_surface_max_spearman_abs"] <= 0.90
    assert u3["phase2_dot_max_spearman_abs"] <= 0.90
    assert u3["phase2_informative_max_mutual_overlap_3px"] <= 0.70
    assert u3["phase2_informative_max_forward_overlap_3px"] <= 0.70

    assert rc["verdict"] == "promote"
    assert "not approved" in rc["selector_status"].lower()
    sub = rc["submission_artifacts"]
    assert len(sub["submission_note_le140"]) <= 140

    tif_path = ROOT / sub["primary_geotiff"]
    zip_path = ROOT / sub["primary_zip"]
    nan_path = ROOT / sub["nan_twin_geotiff"]
    assert tif_path.exists() and zip_path.exists() and nan_path.exists()
    assert _sha256(tif_path) == sub["primary_geotiff_sha256"]
    assert _sha256(zip_path) == sub["primary_zip_sha256"]
    assert _sha256(nan_path) == sub["nan_twin_geotiff_sha256"]

    with zipfile.ZipFile(zip_path, "r") as zf:
        assert zf.namelist() == [tif_path.name]

    with rasterio.open(tif_path) as src:
        assert str(src.crs) == "EPSG:32611"
        assert (src.height, src.width) == (3730, 3292)
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert tuple(src.transform)[:6] == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
        arr = src.read(1)
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))
        assert float(arr.min()) == 0.0 and float(arr.max()) == 1.0
        assert int((arr > 0).sum()) == 37654


def test_h77d_featured_on_home_and_orthogonal_to_all_siblings():
    """Verify H77D is featured clearly on site pages and orthogonal to all GEMSDOE77 sibling lanes."""
    rc = json.loads((ROOT / "evidence" / "run_card_h77d.json").read_text())
    u3 = json.loads((ROOT / "evidence" / "h77d_exp3_uniqueness.json").read_text())
    sub = rc["submission_artifacts"]

    for page_name in ["index.html", "h77d.html", "h77d-executive-summary.html"]:
        html_txt = (ROOT / "docs" / page_name).read_text()
        assert sub["submission_name"] in html_txt
        assert sub["primary_geotiff_sha256"] in html_txt
        assert Path(sub["primary_geotiff"]).name in html_txt
        assert Path(sub["primary_zip"]).name in html_txt

    sibs = u3["gemsdoe77_head_to_head_siblings"]
    assert len(sibs) >= 4
    for sib_name, stats in sibs.items():
        assert stats["forward_overlap_3px"] <= 0.50, (sib_name, stats)
        assert stats["mutual_overlap_3px"] <= 0.50, (sib_name, stats)
        assert stats["spearman_abs"] <= 0.10, (sib_name, stats)
        assert stats["surface_spearman_abs"] <= 0.15, (sib_name, stats)

