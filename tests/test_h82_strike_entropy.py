import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from h82se_strike_entropy import strike_entropy


def test_crossing_has_higher_local_orientation_entropy_than_straight_trace():
    footprint = np.ones((101, 101), dtype=bool)
    horizontal = np.zeros_like(footprint)
    horizontal[50, 10:91] = True
    crossing = horizontal.copy()
    crossing[10:91, 50] = True
    straight_feature = strike_entropy(horizontal, footprint)
    crossing_feature = strike_entropy(crossing, footprint)
    assert np.isfinite(straight_feature).all()
    assert np.isfinite(crossing_feature).all()
    assert 0 <= crossing_feature.min() <= crossing_feature.max() <= 1
    assert crossing_feature[50, 50] > straight_feature[50, 50]


def test_feature_is_zero_outside_footprint_and_checks_shapes():
    footprint = np.zeros((41, 41), dtype=bool)
    footprint[2:-2, 2:-2] = True
    visible = np.zeros_like(footprint)
    visible[20, 5:36] = True
    out = strike_entropy(visible, footprint)
    assert np.all(out[~footprint] == 0)
    assert np.isfinite(out).all()
    with pytest.raises(ValueError, match="matching 2-D"):
        strike_entropy(visible[:10], footprint)
    visible[0, 0] = True
    with pytest.raises(ValueError, match="outside footprint"):
        strike_entropy(visible, footprint)
