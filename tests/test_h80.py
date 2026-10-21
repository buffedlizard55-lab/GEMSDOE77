"""Synthetic unit tests for H80-OTA fields (no competition data needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems77h80.fields import collinearity_votes, rank01, splat_lines  # noqa: E402


def test_collinear_trio_votes_and_scatter_does_not():
    # 4 points on an east-west line (row 50), 15 px apart, weight 1 each
    rows = np.array([50, 50, 50, 50])
    cols = np.array([50, 65, 80, 95])
    w = np.ones(4)
    v, az, npairs = collinearity_votes(rows, cols, w)
    # interior points see 3 neighbours within 40 px along the line (dist 15,30; 45 excluded)
    assert v[1] >= 2 and v[2] >= 2
    assert npairs >= 5
    # azimuth near 0 (east, x=col)
    assert min(abs(az[1]), abs(az[1] - np.pi)) < np.pi / 36 + 1e-9
    # scattered points: no pair within [5,40] px
    v2, _, np2 = collinearity_votes(np.array([0, 300, 600]), np.array([0, 300, 600]), np.ones(3))
    assert (v2 == 0).all() and np2 == 0


def test_weighted_vote_uses_tier_weights():
    rows = np.array([10, 10]); cols = np.array([10, 25]); w = np.array([2.0, 1.0])
    v, _, _ = collinearity_votes(rows, cols, w)
    assert v[0] == 1.0 and v[1] == 2.0  # support excludes own weight


def test_rank01_ties_share_one_rank_no_scan_gradient():
    # regression (IR-H80-04): 98% zeros must not become a row-order gradient
    a = np.zeros((50, 40), np.float32)
    a[10, 10] = 5.0
    mask = np.ones_like(a, bool)
    r = rank01(a, mask)
    assert len(np.unique(r[a == 0])) == 1
    assert r[10, 10] == 1.0
    assert r[49, 39] == r[0, 0]  # south-east zero cell == north-west zero cell


def test_splat_stays_on_line_and_bounded():
    rows = np.array([50]); cols = np.array([50]); v = np.array([3.0]); az = np.array([0.0])
    out = splat_lines(rows, cols, v, az, (100, 140))
    assert out.max() > 0
    assert out[50, 80] > 0          # along-strike east
    assert out[20, 50] == 0         # nothing 30 px perpendicular
    assert np.isfinite(out).all()
