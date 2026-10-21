"""Label-free OTA fields from GDR 1391 well/spring points.

Frozen at preregistration (registry/preregistration_h80.json):
- tier A (weight 2): finite dT = max(Tqtz,Tchalc,Tcat) - temp_c >= 30 C and
  max geothermometer >= 90 C
- tier B (weight 1): temp_c >= 40 C
- orphan: distance to the reference catalogue mask > 10 px (the caller passes
  the *visible* mask inside the holdout, the full catalogue for the final map)
- vote: for 36 azimuth bins (5 deg over [0,180)), weighted count of other
  orphans at along-strike distance in [5,40] px whose pair azimuth falls in the
  bin (bin half-width 2.5 deg => perpendicular offset <= 1.8 px at 40 px,
  matching the frozen <=2 px corridor); vote = max over bins
- splat: vote along best azimuth +/-40 px, step 2 px, perpendicular Gaussian
  sigma 1.5 px

The CSV column dist_known_fault_px is label-derived and is never read.
No label raster is read here; the catalogue enters only via the caller-supplied
boolean reference mask used for orphan definition.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CHANNELS = ["ota_line", "ota_x_ridge", "ota_orphan_prox", "ota_hot_prox"]
N_AZ = 36
ALONG_MIN, ALONG_MAX = 5.0, 40.0
ORPHAN_DIST_PX = 10.0


def load_points(csv_path: str | Path, shape: tuple[int, int]):
    """Dedup points by cell keeping max tier weight. Returns rows, cols, weight."""
    best: dict[tuple[int, int], float] = {}
    n_rows = 0
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            n_rows += 1
            try:
                row, col = int(r["row"]), int(r["col"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= row < shape[0] and 0 <= col < shape[1]):
                continue

            def f(key):
                try:
                    return float(r.get(key) or "nan")
                except ValueError:
                    return float("nan")

            temp = f("temp_c")
            geotherms = [f("geothermquartz_c"), f("geothermchalc_c"), f("geothermcat_c")]
            gmax = max([g for g in geotherms if math.isfinite(g)], default=float("nan"))
            w = 0.0
            if math.isfinite(temp) and math.isfinite(gmax) and (gmax - temp) >= 30.0 and gmax >= 90.0:
                w = 2.0
            elif math.isfinite(temp) and temp >= 40.0:
                w = 1.0
            if w > 0.0:
                key = (row, col)
                if w > best.get(key, 0.0):
                    best[key] = w
    if not best:
        raise ValueError("no tier A/B points found")
    rc = np.array(sorted(best.keys()), dtype=np.int64)
    w = np.array([best[(int(r), int(c))] for r, c in rc], dtype=np.float64)
    return rc[:, 0], rc[:, 1], w, n_rows


def collinearity_votes(rows, cols, weights):
    """Max weighted azimuth-bin support per point + best azimuth (radians, x-east)."""
    xy = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])
    tree = cKDTree(xy)
    pairs = tree.query_pairs(ALONG_MAX, output_type="ndarray")
    votes = np.zeros(len(rows), np.float64)
    best_az = np.zeros(len(rows), np.float64)
    if len(pairs) == 0:
        return votes, best_az, 0
    d = xy[pairs[:, 1]] - xy[pairs[:, 0]]
    dist = np.hypot(d[:, 0], d[:, 1])
    keep = (dist >= ALONG_MIN) & (dist <= ALONG_MAX)
    pairs, d, dist = pairs[keep], d[keep], dist[keep]
    az = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)  # undirected line azimuth
    bins = np.minimum((az / math.pi * N_AZ).astype(int), N_AZ - 1)
    # accumulate support[i, bin] += w[j] and support[j, bin] += w[i] into ONE
    # array (azimuth is undirected, so a pair contributes to the same bin at
    # both endpoints); vote = max over bins of the summed support.
    support = np.zeros((len(rows), N_AZ), np.float64)
    np.add.at(support, (pairs[:, 0], bins), weights[pairs[:, 1]])
    np.add.at(support, (pairs[:, 1], bins), weights[pairs[:, 0]])
    b = np.argmax(support, axis=1)
    votes = support[np.arange(len(rows)), b]
    best_az = (b + 0.5) * math.pi / N_AZ
    return votes, best_az, int(len(pairs))


def splat_lines(rows, cols, votes, best_az, shape):
    out = np.zeros(shape, np.float32)
    perp_k = np.arange(-4, 5)
    perp_w = np.exp(-(perp_k.astype(np.float64) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
    H, W = shape
    ts = np.arange(-ALONG_MAX, ALONG_MAX + 1, 2.0)
    for r, c, v, az in zip(rows, cols, votes, best_az):
        if v <= 0:
            continue
        ux, uy = math.cos(float(az)), math.sin(float(az))  # x=col, y=row
        px, py = -uy, ux
        for t in ts:
            cx, cy = c + t * ux, r + t * uy
            for k, wk in zip(perp_k, perp_w):
                xx = int(round(cx + k * px)); yy = int(round(cy + k * py))
                if 0 <= yy < H and 0 <= xx < W:
                    out[yy, xx] += np.float32(v) * wk
    return out


def rank01(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tie-averaged percentile rank in [0,1] over mask.

    An earlier version broke ties by scan order, which turned the ~98%-zero
    OTA line field into a spurious north-south gradient (IR-H80-04; first
    H80 build invalidated). scipy rankdata(method='average') gives every
    tied cell the same rank.
    """
    from scipy.stats import rankdata
    out = np.zeros(a.shape, np.float32)
    v = a[mask].astype(np.float64)
    finite = np.isfinite(v)
    r = np.zeros(v.size, np.float64)
    if finite.any():
        r[finite] = (rankdata(v[finite], method="average") - 1.0) / max(finite.sum() - 1, 1)
    out[mask] = r.astype(np.float32)
    return out


def build_ota_fields(csv_path, footprint, reference_catalogue, band3, band5):
    """Returns (HxWx4 float32 channels in CHANNELS order, diagnostics dict).

    band3/band5: raw shipped bands (tmi_hg, iso_grav_anom_slope), NaN-cleaned.
    """
    shape = footprint.shape
    rows, cols, weights, n_rows = load_points(csv_path, shape)
    in_foot = footprint[rows, cols]
    rows, cols, weights = rows[in_foot], cols[in_foot], weights[in_foot]

    dist_ref = ndimage.distance_transform_edt(~reference_catalogue)
    orphan = dist_ref[rows, cols] > ORPHAN_DIST_PX
    orow, ocol, ow = rows[orphan], cols[orphan], weights[orphan]

    votes, best_az, n_pairs = collinearity_votes(orow, ocol, ow)
    ota_line_raw = splat_lines(orow, ocol, votes, best_az, shape)
    ota_line = rank01(np.where(footprint, ota_line_raw, 0.0), footprint)

    ridge = np.maximum(rank01(band3, footprint), rank01(band5, footprint))
    ota_x_ridge = np.sqrt(np.clip(ota_line, 0, 1) * ridge).astype(np.float32)

    orphan_mask = np.zeros(shape, bool); orphan_mask[orow, ocol] = True
    hot_mask = np.zeros(shape, bool); hot_mask[rows, cols] = True
    ota_orphan_prox = np.exp(-ndimage.distance_transform_edt(~orphan_mask) / 10.0).astype(np.float32)
    ota_hot_prox = np.exp(-ndimage.distance_transform_edt(~hot_mask) / 10.0).astype(np.float32)

    stack = np.stack([ota_line, ota_x_ridge, ota_orphan_prox, ota_hot_prox], axis=2).astype(np.float32)
    stack[~footprint] = 0.0
    diag = {
        "csv_rows": int(n_rows),
        "tier_points_in_footprint": int(len(rows)),
        "tierA_points": int((weights == 2).sum()),
        "tierB_points": int((weights == 1).sum()),
        "orphan_points_gt10px_from_reference": int(len(orow)),
        "pairs_within_5_40px": int(n_pairs),
        "orphans_vote_ge2": int((votes >= 2).sum()),
        "orphans_vote_ge3": int((votes >= 3).sum()),
        "vote_max": float(votes.max()) if len(votes) else 0.0,
        "ota_line_nonzero_cells": int((ota_line_raw > 0).sum()),
    }
    return stack, diag
