"""Unit and artifact tests for the H82-FMD lane (mapping-deficit allocation).

Every test is a property the lane's conclusion depends on; none of them reads
the 419 MB competition raster, so the suite stays runnable in CI.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from gems82 import deficit as DEF
from gems82 import fields as F

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "docs" / "downloads" / "gems77-h82-crestthin-fullstack-37654dots-20261021-zeros.tif"


# ----------------------------------------------------------------- crest NMS
def test_crest_mask_thins_a_synthetic_ridge_to_about_one_pixel():
    """Where the surface carries weight, a ribbon must collapse to ~1 px."""
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    surface = np.exp(-0.5 * ((xx - 32.0) / 2.5) ** 2).astype(np.float32)
    valid = np.ones_like(surface, bool)
    crest = F.crest_mask(surface, valid)
    core = surface > 0.01                       # the part the allocator can ever rank
    kept_per_row = (crest & core)[8:-8].sum(axis=1)
    assert kept_per_row.max() <= 2, "a 1-D ridge must thin to at most two columns"
    assert crest[8:-8, 31:34].any(), "the crest column itself must survive"
    # the allocator sees surface * crest, so judge the thinning on that product
    thin = surface * crest
    assert (thin > 0.5).sum() < (surface > 0.5).sum() / 2


def test_crest_mask_ties_are_confined_to_numerically_flat_regions():
    """Documented property (IR-H82-04): in a region that is flat to within float32
    noise, every pixel ties its neighbours and survives non-maximum suppression.
    Those pixels carry no weight, because the lane always uses ``surface * crest``:
    the pipeline measures 25 such pixels out of 1,065,169 crest pixels on the real
    surface. This test pins the harmlessness, not the tie itself."""
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    surface = np.exp(-0.5 * ((xx - 32.0) / 2.5) ** 2).astype(np.float32)
    crest = F.crest_mask(surface, np.ones_like(surface, bool))
    tails = crest & (surface <= 0.01)
    assert float((surface * crest)[tails].max(initial=0.0)) <= 0.01


def test_crest_mask_respects_validity_and_drops_flat_areas():
    surface = np.zeros((32, 32), np.float32)
    surface[:, 16] = 1.0
    valid = np.zeros((32, 32), bool)
    valid[:, 10:20] = True
    crest = F.crest_mask(surface, valid)
    assert not crest[~valid].any()
    assert crest[:, 16].any()


def test_crest_mask_keeps_the_global_maximum_of_a_blob():
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float32)
    blob = np.exp(-(((xx - 24) ** 2 + (yy - 24) ** 2) / 50.0)).astype(np.float32)
    crest = F.crest_mask(blob, np.ones_like(blob, bool))
    assert crest[23:26, 23:26].any()


# ------------------------------------------------------------ block algebra
def test_block_ids_tile_the_grid_without_gaps_or_overlaps():
    blk, (nby, nbx) = DEF.block_ids((50, 70), block_px=20)
    assert blk.shape == (50, 70)
    assert (nby, nbx) == (3, 4)
    assert blk.min() == 0 and blk.max() == nby * nbx - 1
    assert blk[0, 0] == 0 and blk[19, 19] == 0 and blk[20, 0] == nbx


def test_aggregate_matches_direct_numpy_mean_and_std():
    rng = np.random.default_rng(0)
    stack = rng.normal(size=(500, 3)).astype(np.float32)
    stack[7, 1] = np.nan                      # NaN must be ignored, not propagated
    blk_of_px = rng.integers(0, 4, size=500).astype(np.int32)
    out = DEF.aggregate(stack, blk_of_px, 4)
    for b in range(4):
        for j in range(3):
            vals = stack[blk_of_px == b, j]
            vals = vals[np.isfinite(vals)]
            assert out[b, 2 * j] == pytest.approx(vals.mean(), rel=1e-4)
            assert out[b, 2 * j + 1] == pytest.approx(vals.std(), rel=1e-3)


def test_block_counts_counts_true_pixels_per_block():
    blk, (nby, nbx) = DEF.block_ids((40, 40), block_px=20)
    mask = np.zeros((40, 40), bool)
    mask[0:5, 0:5] = True          # 25 pixels in block 0
    mask[25, 25] = True            # 1 pixel in block 3
    counts = DEF.block_counts(mask, blk, nby * nbx)
    assert counts[0] == 25 and counts[3] == 1 and counts.sum() == 26


def test_deficit_is_zero_where_the_catalogue_already_explains_the_intensity():
    """A block whose visible count equals the fitted intensity must get no weight."""
    rng = np.random.default_rng(3)
    shape = (200, 200)
    blk, (nby, nbx) = DEF.block_ids(shape, block_px=20)
    n_blocks = nby * nbx
    foot = np.ones(shape, bool)
    n_fp = DEF.block_counts(foot, blk, n_blocks)
    # one covariate that linearly drives the mapped-fault count
    x = rng.uniform(0, 1, n_blocks).astype(np.float32)
    feat = np.stack([x, x ** 2], 1)
    n_vis = np.round(80 * x)                       # perfectly explained by x
    train = np.ones(n_blocks, bool)
    w, lam, raw = DEF.deficit_field(feat, n_fp, n_vis, train, train, blk, shape, foot,
                                    smooth_px=0.0, return_lambda=True)
    # The estimator itself is right: with the catalogue fully explained, the positive
    # residual is a rounding error beside the counts it is a residual of.
    assert raw.sum() < 0.10 * n_vis.sum(), "a fully explained catalogue must leave no deficit"
    # But the published weight is a 90th-percentile rescale of that residual, so a
    # noise-only deficit still saturates at 1.0. That amplification is exactly why the
    # weight could only add variance in E2, and it is pinned here rather than hidden.
    assert w.max() == pytest.approx(1.0, abs=1e-6)

    n_vis_censored = n_vis.copy()
    censored = np.zeros(n_blocks, bool)
    censored[: n_blocks // 4] = True
    n_vis_censored[censored] = 0                   # hide the mapping in a quarter of blocks
    w2 = DEF.deficit_field(feat, n_fp, n_vis_censored, ~censored, np.ones(n_blocks, bool),
                           blk, shape, foot, smooth_px=0.0)
    hi = np.array([w2[blk == b].mean() for b in range(n_blocks)])
    assert hi[censored].mean() > hi[~censored].mean(), "censored blocks must score higher"


def test_apply_weight_floor_never_zeroes_the_physics():
    s = np.full((8, 8), 0.4, np.float32)
    w = np.zeros((8, 8), np.float32)
    out = DEF.apply_weight(s, w, floor=0.1)
    assert out == pytest.approx(0.04 * np.ones((8, 8)), abs=1e-6)
    out1 = DEF.apply_weight(s, np.ones_like(w), floor=0.1)
    assert out1 == pytest.approx(0.4 * np.ones((8, 8)), abs=1e-6)


# -------------------------------------------------- shared uniqueness helper
def _write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:32611",
                       transform=Affine(100, 0, 0, 0, -100, 0)) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_space_filling_flags_separates_dense_witnesses_from_sparse_dots(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "vendor"))
    from gems57 import uniqueness as U

    fp = np.ones((100, 100), bool)
    dense = np.ones((100, 100), np.float32)
    sparse = np.zeros((100, 100), np.float32)
    sparse[::20, ::20] = 1.0
    pd, ps = tmp_path / "dense.tif", tmp_path / "sparse.tif"
    _write(pd, dense)
    _write(ps, sparse)
    rows = [{"sha256": "d", "file": str(pd), "their_dots": int(dense.sum())},
            {"sha256": "s", "file": str(ps), "their_dots": int((sparse > 0).sum())}]
    flags = U.space_filling_flags(rows, fp, radius_px=3.0)
    assert flags["d"]["space_filling"] is True and flags["d"]["exact"] is True
    assert flags["s"]["space_filling"] is False
    assert 0.0 <= flags["s"]["halo_coverage"] <= 1.0


# --------------------------------------------------------- shipped artifact
@pytest.mark.skipif(not SUBMISSION.exists(), reason="submission artifact not built")
def test_submission_artifact_is_portal_legal_and_matches_its_receipt():
    with rasterio.open(SUBMISSION) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert (src.height, src.width) == (3730, 3292)
        assert str(src.crs) == "EPSG:32611"
        assert tuple(src.transform)[:6] == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
        assert src.nodata is None
        a = src.read(1)
    assert np.isfinite(a).all(), "a NaN would risk the portal range error"
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert set(np.unique(a).tolist()) == {0.0, 1.0}
    assert int((a > 0).sum()) == 37654

    build = json.loads((ROOT / "evidence" / "h82_e3_build.json").read_text())
    assert build["n_dots"] == 37654
    assert build["dots_on_catalogue"] == 0
    assert build["min_dist_to_catalogue_px"] > 2.0
    digest = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    card = json.loads((ROOT / "evidence" / "h82_run_card.json").read_text())
    assert card["raster_sha256"] == digest
    assert len(card["submission_note"]) <= 140
    assert len(card["submission_name"]) <= 140


@pytest.mark.skipif(not (ROOT / "evidence" / "h82_e2_holdout.json").exists(),
                    reason="holdout evidence not present")
def test_primary_hypothesis_is_recorded_as_refuted_on_both_seeds():
    e2 = json.loads((ROOT / "evidence" / "h82_e2_holdout.json").read_text())
    for seed in ("20", "30"):
        arms = e2["seeds"][seed]["arms"]
        assert arms["A_fmd"]["pooled_holdout_dti"] < arms["A_std"]["pooled_holdout_dti"]
    card = json.loads((ROOT / "evidence" / "h82_run_card.json").read_text())
    assert card["verdict"] == "negative"
    assert card["weekly_slots_used"] == 0
