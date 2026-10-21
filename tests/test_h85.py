"""H85-EDTI: uniqueness-rule correction and shipped-byte checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import uniqueness as UQ  # noqa: E402

CARD = ROOT / "evidence" / "h85_run_card.json"
TIF = ROOT / "submissions" / "gems85-h85-edti-collar35-skel-r33-20261021-d97b248e-zeros.tif"


def test_space_filling_prior_is_classified_not_counted():
    """A 5 px lattice's halo covers the map: it cannot discriminate dot placement."""
    allowed = np.zeros((40, 40), bool)
    allowed[2:38, 2:38] = True
    lattice = np.zeros((40, 40), bool)
    lattice[2:38:5, 2:38:5] = True
    rep = UQ.representation_class_v2(lattice, allowed)
    assert rep["kind"] == "space_filling"
    assert rep["halo_coverage"] > 0.70
    assert rep["support_fraction"] < 0.20  # support fraction alone would have missed it


def test_sparse_prior_is_discriminating():
    allowed = np.zeros((40, 40), bool)
    allowed[2:38, 2:38] = True
    sparse = np.zeros((40, 40), bool)
    sparse[20, 20:26] = True
    rep = UQ.representation_class_v2(sparse, allowed)
    assert rep["kind"] == "discriminating"
    assert rep["halo_coverage"] < 0.10


def test_binding_decision_ignores_space_filling_but_not_surface_rho():
    rows = [
        dict(file="lattice.tif", forward_overlap=0.999, halo_coverage=0.99, spearman=0.01),
        dict(file="dots.tif", forward_overlap=0.35, halo_coverage=0.12, spearman=0.07),
    ]
    d = UQ.binding_duplicate_decision(rows)
    assert d["duplicate"] is False
    assert d["worst_forward_overlap"] == pytest.approx(0.35)
    rows[1]["spearman"] = 0.95
    assert UQ.binding_duplicate_decision(rows)["duplicate"] is True


def test_thresholds_are_unchanged():
    assert UQ.OVERLAP_LIMIT == 0.70
    assert UQ.RHO_LIMIT == 0.90
    assert UQ.DENSE_HALO_FRACTION == 0.70


@pytest.mark.skipif(not CARD.exists(), reason="H85 run card not built in this checkout")
def test_run_card_is_label_consistent_and_promoted():
    card = json.loads(CARD.read_text())
    assert card["verdict"] == "promote"
    assert len(card["submission_note"]) <= 140
    h = card["holdout_dti"]
    assert h["evidence_class"] == "HOLDOUT-DTI"
    assert h["evaluator_version"] == "gems57-pooled-hide-v2"
    assert (h["alpha"], h["beta"], h["triangular_radius_m"]) == (0.2, 0.8, 300)
    for key in ("seed_30", "seed_20", "dual_seed_pooled"):
        lo, hi = h[key]["ci95"]
        assert lo <= h[key]["score"] <= hi
    assert h["dual_seed_pooled"]["score"] > h["retained_incumbent"]["score"]
    assert card["selector"]["organizer_receipt"] is None
    assert card["uniqueness_gate"]["binding"]["duplicate"] is False


@pytest.mark.skipif(not TIF.exists(), reason="H85 TIF not present in this checkout")
def test_shipped_bytes_are_portal_valid():
    import rasterio

    from gems57 import grid as GRID
    from gems57 import validate as V

    grid = GRID.load_grid(ROOT / "data" / "bridge")
    rep = V.validate(TIF, grid.footprint, grid.catalogue)
    assert rep["all_checks_passed"], rep["checks"]
    with rasterio.open(TIF) as src:
        a = src.read(1)
        assert src.count == 1 and src.dtypes[0] == "float32"
        assert str(src.crs) == "EPSG:32611"
        assert (src.height, src.width) == (3730, 3292)
    assert np.isfinite(a).all()
    assert set(np.unique(a)).issubset({0.0, 1.0})
    assert int((a[grid.catalogue] > 0).sum()) == 0
    assert int((a > 0).sum()) == 201260
