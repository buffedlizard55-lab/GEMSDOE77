"""Comprehensive unit, integrity, leakage, uniqueness, and GitHub Pages link tests for Lane H77E:
Covariate-Matched Zero-SGMC Multi-Modal Ensemble + Residual Filament Auto-Context + Exact Dinkelbach 1-Swap.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gems77e.autoctx import compute_filament_autocontext
from gems77e.dinkelbach import allocate_dinkelbach_1swap
from gems77e.features import compute_visible_mechanics_pair

KERNEL_29 = [
    (dy, dx, float(1.0 - np.hypot(dy, dx) / 3.0))
    for dy in range(-3, 4)
    for dx in range(-3, 4)
    if dy * dy + dx * dx < 9
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fractional_dti(dots: np.ndarray, weights: np.ndarray, k_tot: float = 50.0) -> float:
    h, w = dots.shape
    cov = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.nonzero(dots > 0.5)
    for dy, dx, kw in KERNEL_29:
        rr = ys + dy
        cc = xs + dx
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        np.maximum.at(cov, (rr[ok], cc[ok]), kw)
    tp = float(np.sum(cov * weights))
    s = float(np.sum(dots > 0.5))
    return tp / (0.8 * tp + 0.2 * s + 0.8 * k_tot)


def test_h77e_zero_sgmc_and_zero_label_leakage_source_audit():
    """Verify no H77E source file or experiment script reads SGMC channels or GDR dist_known_fault_px."""
    checked = [
        ROOT / "src" / "gems77e" / "features.py",
        ROOT / "src" / "gems77e" / "autoctx.py",
        ROOT / "src" / "gems77e" / "dinkelbach.py",
        ROOT / "scripts" / "h77e_exp1_canaries_and_live_inversion.py",
        ROOT / "scripts" / "h77e_exp2_holdout.py",
        ROOT / "scripts" / "h77e_exp3_build_and_audit.py",
    ]
    for path in checked:
        assert path.is_file(), f"Missing file: {path}"
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


def test_h77e_buffered_bucket_mechanics_covariate_match():
    """Verify 5-bucket 3-px buffered mechanics never produces <= 3.0 px self/sub-component leakage."""
    h, w = 84, 84
    vis = np.zeros((h, w), dtype=bool)
    for i in range(14):
        vis[8 + 2 * i, 12 + i] = True
        vis[9 + 2 * i, 12 + i] = True
        vis[46 + 2 * i, 52 + i] = True
        vis[47 + 2 * i, 52 + i] = True
    foot = np.ones((h, w), dtype=bool)
    rk = np.full((h, w), 0.5, dtype=np.float32)
    mech_tr, mech_inf = compute_visible_mechanics_pair(vis, foot, rk, rk, n_buckets=5, buffer_px=3.0)
    assert mech_tr is not None and mech_tr.shape == (8, h * w)
    assert mech_inf.shape == (8, h * w)
    assert np.all(np.isfinite(mech_tr)) and np.all(np.isfinite(mech_inf))
    d_vis_tr = mech_tr[0].reshape(h, w)
    assert float(d_vis_tr[vis].min()) > 3.0


def test_h77e_filament_context_and_dinkelbach_overlap_cap_invariants():
    """Verify pure-physics 8-azimuth filament auto-context and exact Dinkelbach + 1-swap + overlap_cap."""
    h, w = 96, 96
    foot = np.ones((h, w), dtype=bool)
    p = np.full((h, w), 0.005, dtype=np.float32)
    # Collinear scarp segments with an eroded alluvial wash gap at rows 42..47
    for r in list(range(16, 42)) + list(range(48, 74)):
        p[r, 40] = 0.85
    ctx = compute_filament_autocontext(p[foot], foot, (p[foot] > 0.5).astype(np.float32), np.ones(h * w, dtype=bool))
    assert ctx.shape == (int(foot.sum()),) and np.all(np.isfinite(ctx))
    # Wash gap between collinear scarps must receive far higher filament/bridge log-odds delta than background
    assert float(ctx[45 * w + 40]) > float(ctx[45 * w + 10]) + 0.20

    allowed = np.ones((h, w), dtype=bool)
    allowed[:3, :] = False
    crest = (p > 0.5).astype(np.float32)
    prior_halo = np.zeros((h, w), dtype=bool)
    prior_halo[:55, :] = True

    dots_no_swap, _ = allocate_dinkelbach_1swap(
        p, p, crest, allowed, target_dots=20, n_swap_rounds=0, halo_masks_to_cap=[prior_halo], overlap_cap=0.50
    )
    dots_swap, stats = allocate_dinkelbach_1swap(
        p, p, crest, allowed, target_dots=20, n_swap_rounds=2, halo_masks_to_cap=[prior_halo], overlap_cap=0.50
    )
    assert int((dots_swap > 0.5).sum()) == 20
    assert not np.any((dots_swap > 0.5) & (~allowed))
    assert float(prior_halo[dots_swap > 0.5].mean()) <= 0.50 + 1e-6
    assert _fractional_dti(dots_swap, p) >= _fractional_dti(dots_no_swap, p) - 1e-6
    assert isinstance(stats, dict) and len(stats) > 0


def test_h77e_experiments_run_card_geotiff_and_pages_integrity():
    """Verify all 3 H77E experiments, CIs, canaries, uniqueness, 17 GeoTIFF checks, and 0 broken Pages links."""
    e1 = json.loads((ROOT / "evidence" / "h77e_exp1_canaries_and_live_inversion.json").read_text())
    e2 = json.loads((ROOT / "evidence" / "h77e_exp2_holdout.json").read_text())
    rc = json.loads((ROOT / "evidence" / "run_card_h77e.json").read_text())

    # E1: 99 leakage canaries <= 0.90 across all 3 seeds & covariate-shift proof
    assert e1["leakage_canary_gate_0p90"]["status"] == "ALL_SEEDS_PASS"
    for s in ("seed_30", "seed_20", "seed_21"):
        assert e1["leakage_canary_gate_0p90"]["per_seed"][s]["max_symmetric_auc"] <= 0.90
    assert e1["covariate_shift_root_cause_audit"]["legacy_loco_pos_frac_le_3p5px"] > 0.10
    assert e1["covariate_shift_root_cause_audit"]["legacy_loco_neg_frac_le_3p5px"] == 0.0

    # E2: 3-seed pooled holdout record (> 0.180 over 34,235 withheld positive pixels)
    pooled = e2["pooled_3seed_34235px"]
    assert pooled["scores"]["H77E_full_B37654"]["withheld_positive_pixels"] == 34235
    assert pooled["scores"]["H77E_full_B37654"]["dti"] > 0.184
    assert pooled["paired_differences"]["A0_h195_foldfair"]["ci95"][0] > 0.060
    assert pooled["paired_differences"]["H77E_stage1_only_B37654"]["delta"] > 0.005
    assert pooled["paired_differences"]["H77E_greedy_nms_B37654"]["delta"] > 0.010

    # E3: Run card, uniqueness across 860 rasters, and GeoTIFF/ZIP SHA-256 + format checks
    assert rc["verdict"] == "promote"
    assert rc["experiments_used"] == 3 and rc["experiments_cap"] == 3
    assert rc["leakage_audit"]["sgmc_channels_used"] == 0
    assert rc["leakage_audit"]["in_sample_predicted_pixels"] == 0
    assert rc["leakage_audit"]["pixels_copied_from_prior_rasters"] == 0

    uq = rc["uniqueness_audit"]
    assert uq["prior_rasters_checked"] >= 850
    assert uq["surface_max_spearman_abs"] <= 0.90
    assert uq["dot_max_spearman_abs"] <= 0.90
    assert uq["dot_max_forward_overlap_3px_informative"] <= 0.70

    sub = rc["submission_artifacts"]
    assert len(sub["submission_note_le140"]) <= 140
    assert all(sub["validator_17_checks"].values())

    tif_path = ROOT / sub["primary_geotiff"]
    repo_copy = ROOT / sub["repo_copy_geotiff"]
    zip_path = ROOT / sub["primary_zip"]
    nan_path = ROOT / sub["nan_twin_geotiff"]
    for p in (tif_path, repo_copy, zip_path, nan_path):
        assert p.is_file(), f"Missing artifact: {p}"
    assert _sha256(tif_path) == sub["primary_geotiff_sha256"]
    assert _sha256(repo_copy) == sub["primary_geotiff_sha256"]
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

    # Site pages + zero broken relative links check
    for page_name in ["index.html", "executive-summary.html", "h77e.html", "h77e-executive-summary.html"]:
        p_html = ROOT / "docs" / page_name
        assert p_html.is_file()
        txt = p_html.read_text()
        assert sub["submission_name"] in txt
        assert sub["primary_geotiff_sha256"] in txt
        for ref in re.findall(r'(?:href|src)="([^"#?:]+)"', txt):
            target = (p_html.parent / ref).resolve()
            assert target.exists(), f"Broken relative link in {page_name}: {ref} -> {target}"

    # Verify original prompt remains verbatim in README.md
    readme_txt = (ROOT / "README.md").read_text()
    assert "h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros" in readme_txt
    assert "Run this task through multiple passes." in readme_txt
