"""H82 mathematical/support checks, independent of large cached inputs."""
import numpy as np
import pytest
from scipy.ndimage import maximum_filter
from gems77.texture82 import build, local_sd, normalize


def test_constant_has_no_interior_variance():
    a = np.full((40, 50), 0.5)
    assert np.max(local_sd(a)[4:-4, 4:-4]) == 0


def test_sd_matches_direct_window():
    a = np.random.default_rng(82).random((40, 50))
    sd = local_sd(a)
    assert sd[20, 25] == pytest.approx(np.std(a[16:25, 21:30]), abs=1e-12)


def test_square_support_covers_diagonal_missing_and_gradient_stencil():
    a = np.full((40, 50), 100, np.uint8)
    a[20, 25] = 0
    raw, fields, eligible, scales = build(a, a, np.ones_like(a, bool))
    bad = maximum_filter((a == 0).astype(int), size=11, mode='constant', cval=1)
    assert not (eligible & (bad > 0)).any()
    assert not eligible[:5].any() and not eligible[-5:].any()
    for field in fields.values():
        assert np.isfinite(field).all() and field.min() >= 0 and field.max() <= 1
        assert not field[~eligible].any()


def test_texture_boundary_not_uniform_region():
    rng = np.random.default_rng(82)
    a = np.full((100, 100), 100, np.uint8)
    a[:, 50:] = rng.integers(50, 150, size=(100, 50), dtype=np.uint8)
    raw, _, eligible, _ = build(a, a, np.ones_like(a, bool))
    field = raw['candidate']
    assert field[10:90, 46:54].mean() > field[10:90, 65:85].mean()
    assert field[10:90, 10:30].max() < 1e-8


def test_normalize_degenerate_and_empty():
    a = np.zeros((10, 10))
    out, scale = normalize(a, np.ones_like(a, bool))
    assert scale == 0 and not out.any()
    with pytest.raises(ValueError, match='empty'):
        normalize(a, np.zeros_like(a, bool))


def test_invalid_input_rejected():
    with pytest.raises(ValueError, match='grid mismatch'):
        build(np.ones((4, 5)), np.ones((5, 4)), np.ones((4, 5), bool))
    with pytest.raises(ValueError, match='odd'):
        local_sd(np.ones((10, 10)), 4)
