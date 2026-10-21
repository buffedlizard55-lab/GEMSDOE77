"""Non-authorizing halo-coverage diagnostics; shared uniqueness policy is unchanged.

Moved out of the pinned template after integration exposed a hash mismatch.
Uses the shared dilation helper; this is not an evaluator or uniqueness gate.
"""
import numpy as np
import rasterio
from gems57.uniqueness import RADIUS_PX, OVERLAP_LIMIT, _near

def _disk_area(radius_px=RADIUS_PX):
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return int((yy * yy + xx * xx <= radius_px * radius_px).sum())


def space_filling_flags(rows, footprint, radius_px=RADIUS_PX):
    """Mark registry rasters whose own halo cannot discriminate anything.

    The parallel-run duplicate rule is "more than 70 % of your dots fall within
    3 px of one registry raster's dots".  Against a raster whose 3 px halo
    already covers >= 70 % of the scored footprint, a *uniformly random* dot
    field also exceeds the limit, so the test has no power there: firing
    against such a raster is evidence about the comparator, not about the
    candidate.  This helper measures that property so both readings can be
    reported; it does **not** change any measurement, limit, or verdict, and it
    never drops a raster from the literal reading.

    Returns ``{sha256: {"space_filling": bool, "halo_coverage": float,
    "exact": bool}}``.  Exact dilation is only computed for rasters that could
    possibly reach the limit (``their_dots * disk_area >= limit * |footprint|``);
    for the rest the cheap upper bound already proves they cannot, which keeps
    a several-hundred-raster audit affordable.
    """
    fp = np.asarray(footprint, bool)
    n_fp = int(fp.sum())
    area = _disk_area(radius_px)
    need = OVERLAP_LIMIT * n_fp / max(area, 1)
    out = {}
    for row in rows:
        if 'error' in row or 'sha256' not in row:
            continue
        nd = int(row.get('their_dots', 0) or 0)
        if nd < need:
            out[row['sha256']] = {"space_filling": False,
                                  "halo_coverage": round(min(1.0, nd * area / n_fp), 6),
                                  "exact": False}
            continue
        with rasterio.open(row['file']) as src:
            a = src.read(1).astype(np.float32)
        a = np.where(np.isfinite(a), a, 0.0)
        support = (a > 0) & fp
        cov = float((_near(support, radius_px) & fp).sum() / n_fp)
        out[row['sha256']] = {"space_filling": bool(cov >= OVERLAP_LIMIT),
                              "halo_coverage": round(cov, 6), "exact": True}
    return out
