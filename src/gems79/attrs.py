"""Catalogue-attribute rasters and fold-blind context features for lane H79.

Inputs (all free/official, receipts in .cache/ext/qfaults_receipt.json and
registry/data_manifest.json):
  * .cache/ext/trace_segments_utm11.csv  -- qfaults traces as UTM-11 line segments
    (record_id = 0-based row of qfault_attributes.csv = shapefile FID)
  * .cache/ext/qfault_attributes.csv     -- INGENIOUS 2023 qfaults attribute table
  * existing_faults.tif via vendor.gems57.grid (catalogue pixels)
  * training_features.tif band 12 (det_elev) for the downthrown-side inference

Grid mapping (verified in session E0): col = (x - 243350)/100, row = (4508550 - y)/100.

Everything in `context_features` is computed from the *visible* pixel mask that
the caller passes in; hidden components and their 3 px collar never enter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

X0, Y0, RES = 243350.0, 4508550.0, 100.0

# dip-direction compass label -> unit vector in grid (col, row) coordinates (row grows south)
_DIPVEC = {
    "E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, -1.0), "S": (0.0, 1.0),
    "NE": (0.7071, -0.7071), "NW": (-0.7071, -0.7071), "SE": (0.7071, 0.7071), "SW": (-0.7071, 0.7071),
}

CTX_NAMES = [
    "ctx_dvis",          # distance (px) to nearest visible catalogue pixel, clipped 60
    "ctx_dens10",        # gaussian density of visible pixels, sigma 10 px
    "ctx_dens30",        # gaussian density of visible pixels, sigma 30 px
    "ctx_hw_attr",       # signed distance: + on the DIPDIRECT hanging wall of the nearest visible trace (NaN if unspecified)
    "ctx_hw_topo",       # signed distance: + on the topographically downthrown side of the nearest visible trace
    "ctx_along",         # |offset . strike| of nearest visible pixel (px)
    "ctx_across",        # |offset . strike-normal| (px)
    "ctx_strike_sin2",   # sin(2*strike) of nearest visible pixel
    "ctx_strike_cos2",   # cos(2*strike)
    "ctx_rcode_near",    # recency class 1 (historic) .. 5 (<1.6 Ma) of nearest visible trace
    "ctx_scode_near",    # slip-rate class of nearest visible trace
    "ctx_fcode_near",    # certainty class 1 well .. 3 inferred
    "ctx_mapscale_near", # 100 or 250 (x1000) map scale of nearest visible trace
    "ctx_logslip_near",  # log10 slip rate (mm/yr) of nearest visible trace
    "ctx_dvis_active",   # distance to visible pixels with RCODE<=3 (<130 ka), clipped 60
    "ctx_dvis_holo",     # distance to visible pixels with RCODE<=2 (<15 ka), clipped 60
    "ctx_dens_active30", # density of visible RCODE<=3 pixels, sigma 30
]
ISO_NAMES = ["ctx_dvis", "ctx_dens10", "ctx_dens30"]  # attribute-free isotropic control set
CLIP_PX = 60.0


def _bresenham_pixels(c0: float, r0: float, c1: float, r1: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(max(abs(c1 - c0), abs(r1 - r0)) * 2) + 2
    t = np.linspace(0.0, 1.0, n)
    cc = np.round(c0 + (c1 - c0) * t).astype(np.int64)
    rr = np.round(r0 + (r1 - r0) * t).astype(np.int64)
    key = rr * 10_000_000 + cc
    _, keep = np.unique(key, return_index=True)
    return rr[np.sort(keep)], cc[np.sort(keep)]


def build_attribute_rasters(shape: tuple[int, int], catalogue: np.ndarray, det_elev: np.ndarray,
                            seg_csv: Path, attr_csv: Path) -> dict:
    """Per-catalogue-pixel attribute rasters (float32 H x W; NaN off-catalogue or unknown).

    Returns dict with keys: tcol, trow (unit strike), n_attr_c, n_attr_r (unit normal toward
    the DIPDIRECT hanging wall; NaN if unspecified), n_topo_c, n_topo_r (unit normal toward
    the lower det_elev side), rcode, scode, fcode, mapscale, logslip, and a receipt dict.
    """
    h, w = shape
    seg = pd.read_csv(seg_csv)
    att = pd.read_csv(attr_csv, low_memory=False)
    seg = seg[(seg.x0 > X0 - 2e4) & (seg.x0 < X0 + (w + 200) * RES) & (seg.y0 < Y0 + 2e4) & (seg.y0 > Y0 - (h + 200) * RES)]

    # --- rasterise segments carrying record_id + strike; longest segment wins per pixel
    rec_r = np.full((h, w), -1, dtype=np.int32)
    tc = np.full((h, w), np.nan, dtype=np.float32)
    tr = np.full((h, w), np.nan, dtype=np.float32)
    best_len = np.zeros((h, w), dtype=np.float32)
    c0 = (seg.x0.values - X0) / RES
    c1 = (seg.x1.values - X0) / RES
    r0 = (Y0 - seg.y0.values) / RES
    r1 = (Y0 - seg.y1.values) / RES
    rid = seg.record_id.values.astype(np.int32)
    ln = np.hypot(c1 - c0, r1 - r0).astype(np.float32)
    for i in range(len(seg)):
        if ln[i] <= 0:
            continue
        rr, cc = _bresenham_pixels(c0[i], r0[i], c1[i], r1[i])
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        if not ok.any():
            continue
        rr, cc = rr[ok], cc[ok]
        upd = ln[i] > best_len[rr, cc]
        rr, cc = rr[upd], cc[upd]
        best_len[rr, cc] = ln[i]
        rec_r[rr, cc] = rid[i]
        tc[rr, cc] = (c1[i] - c0[i]) / ln[i]
        tr[rr, cc] = (r1[i] - r0[i]) / ln[i]
    seg_px = rec_r >= 0

    # --- transfer to the official catalogue pixels (nearest segment pixel; 100 % within 1 px, E0)
    _, (ir, ic) = ndi.distance_transform_edt(~seg_px, return_indices=True)
    cat = np.asarray(catalogue, bool)
    ry, rx = np.nonzero(cat)
    rec = rec_r[ir[ry, rx], ic[ry, rx]].astype(np.int32)
    tcol = tc[ir[ry, rx], ic[ry, rx]].astype(np.float32)
    trow = tr[ir[ry, rx], ic[ry, rx]].astype(np.float32)
    del rec_r, tc, tr, ir, ic, best_len
    pix = np.full((h, w), -1, dtype=np.int32)
    pix[ry, rx] = np.arange(ry.size, dtype=np.int32)

    # --- attribute lookup tables (index = record_id)
    n_rec = len(att)
    rc = pd.to_numeric(att["RCODE2023"], errors="coerce").fillna(5).values.astype(np.float32)
    sc = pd.to_numeric(att["SCODE2023"], errors="coerce").fillna(4).values.astype(np.float32)
    fc = pd.to_numeric(att["FCODE2023"], errors="coerce").fillna(2).values.astype(np.float32)
    ms = pd.to_numeric(att["MAPSCALE"], errors="coerce").fillna(250).values.astype(np.float32)
    sl = np.log10(pd.to_numeric(att["SLIPRTNUM"], errors="coerce").fillna(0.01).clip(lower=1e-4).values).astype(np.float32)
    dd = att["DIPDIRECT"].astype(str).str.strip().str.upper().values
    dvc = np.full(n_rec, np.nan, np.float32)
    dvr = np.full(n_rec, np.nan, np.float32)
    for i, lab in enumerate(dd):
        v = _DIPVEC.get(lab)
        if v is not None:
            dvc[i], dvr[i] = v

    ok = rec >= 0
    def lut(table: np.ndarray) -> np.ndarray:
        out = np.full(ry.size, np.nan, dtype=np.float32)
        out[ok] = table[rec[ok]]
        return out

    rcode, scode, fcode, mapscale, logslip = lut(rc), lut(sc), lut(fc), lut(ms), lut(sl)

    # --- unit strike-normal n = (-trow, tcol); orient toward the attribute dip direction
    nc, nr = -trow, tcol
    dc, dr = lut(dvc), lut(dvr)
    dot_attr = nc * dc + nr * dr
    sgn_attr = np.where(np.isnan(dot_attr), np.nan, np.where(dot_attr >= 0, 1.0, -1.0)).astype(np.float32)
    n_attr_c = (nc * sgn_attr).astype(np.float32)
    n_attr_r = (nr * sgn_attr).astype(np.float32)

    # --- topographic downthrown side: mean det_elev at +k n vs -k n (k = 4..12 px)
    de = np.asarray(det_elev, np.float32)
    de = np.where(np.isfinite(de) & (de > -1e30), de, np.nan)
    med = float(np.nanmedian(de))
    de = np.where(np.isnan(de), med, de)
    plus = np.zeros(ry.size, np.float32)
    minus = np.zeros(ry.size, np.float32)
    for k in range(4, 13):
        pr = np.clip(np.round(ry + k * nr), 0, h - 1).astype(np.int64)
        pc = np.clip(np.round(rx + k * nc), 0, w - 1).astype(np.int64)
        mr = np.clip(np.round(ry - k * nr), 0, h - 1).astype(np.int64)
        mc = np.clip(np.round(rx - k * nc), 0, w - 1).astype(np.int64)
        plus += de[pr, pc]
        minus += de[mr, mc]
    sgn_topo = np.where(plus <= minus, 1.0, -1.0).astype(np.float32)  # + side lower => + side is hanging wall
    n_topo_c = (nc * sgn_topo).astype(np.float32)
    n_topo_r = (nr * sgn_topo).astype(np.float32)

    # --- agreement of the topographic inference with the attribute where both exist (MEASURED check)
    both = np.isfinite(n_attr_c) & np.isfinite(n_topo_c)
    agree = float(np.mean((n_attr_c[both] * n_topo_c[both] + n_attr_r[both] * n_topo_r[both]) > 0)) if both.any() else float("nan")
    receipt = {
        "segments_rasterised": int(len(seg)),
        "catalogue_pixels": int(cat.sum()),
        "catalogue_pixels_with_record": int(ok.sum()),
        "pixels_with_dipdirect": int(np.isfinite(n_attr_c).sum()),
        "pixels_with_strike": int(np.isfinite(tcol).sum()),
        "topo_vs_attr_hangingwall_agreement": agree,
        "n_pixels_in_agreement_test": int(both.sum()),
    }
    # NaN-safe sentinel row for the lookups (index -1 -> last row)
    def tab(a):
        return np.append(np.asarray(a, np.float32), np.float32(np.nan))
    return dict(pix=pix, rec=np.append(rec, -1), tcol=tab(tcol), trow=tab(trow), n_attr_c=tab(n_attr_c),
                n_attr_r=tab(n_attr_r), n_topo_c=tab(n_topo_c), n_topo_r=tab(n_topo_r), rcode=tab(rcode),
                scode=tab(scode), fcode=tab(fcode), mapscale=tab(mapscale), logslip=tab(logslip), receipt=receipt)


def context_features(visible: np.ndarray, A: dict, foot: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Return (N, len(CTX_NAMES)) float32 features for footprint pixels (or the given flat row idx).

    `visible` is the only label-bearing input; attribute rasters are read at the nearest
    visible pixel only, so hidden traces (and their collar) never contribute.
    """
    h, w = visible.shape
    vis = np.asarray(visible, bool) & foot & (A["pix"] >= 0)
    if rows is None:
        rows = np.flatnonzero(foot.ravel())
    ry, rx = np.divmod(rows, w)

    d, (iy, ix) = ndi.distance_transform_edt(~vis, return_indices=True)
    d = d.astype(np.float32)
    qy, qx = iy[ry, rx], ix[ry, rx]
    dd = d[ry, rx]
    vc = (rx - qx).astype(np.float32)
    vr = (ry - qy).astype(np.float32)
    del iy, ix

    pidx = A["pix"][qy, qx]  # -1 (sentinel NaN row) cannot occur: q is a visible catalogue pixel

    def at(name):
        return A[name][pidx]

    tcol, trow = at("tcol"), at("trow")
    nc, nr = -trow, tcol
    along = np.abs(vc * tcol + vr * trow)
    across = np.abs(vc * nc + vr * nr)
    s_attr = vc * at("n_attr_c") + vr * at("n_attr_r")
    s_topo = vc * at("n_topo_c") + vr * at("n_topo_r")
    hw_attr = np.where(np.isnan(s_attr), np.nan, np.sign(s_attr) * dd)
    hw_topo = np.where(np.isnan(s_topo), np.nan, np.sign(s_topo) * dd)
    strike = np.arctan2(trow, tcol)
    dens10 = ndi.gaussian_filter(vis.astype(np.float32), 10.0, mode="constant")[ry, rx] * 1e3
    dens30 = ndi.gaussian_filter(vis.astype(np.float32), 30.0, mode="constant")[ry, rx] * 1e4
    rc_r = A["rcode"][A["pix"]]  # H x W, NaN off-catalogue
    act = vis & (rc_r <= 3)
    holo = vis & (rc_r <= 2)
    del rc_r
    d_act = ndi.distance_transform_edt(~act).astype(np.float32)[ry, rx] if act.any() else np.full(rows.size, CLIP_PX, np.float32)
    d_holo = ndi.distance_transform_edt(~holo).astype(np.float32)[ry, rx] if holo.any() else np.full(rows.size, CLIP_PX, np.float32)
    dens_act = ndi.gaussian_filter(act.astype(np.float32), 30.0, mode="constant")[ry, rx] * 1e4

    cols = [
        np.minimum(dd, CLIP_PX),
        dens10, dens30,
        np.clip(hw_attr, -CLIP_PX, CLIP_PX), np.clip(hw_topo, -CLIP_PX, CLIP_PX),
        np.minimum(along, CLIP_PX), np.minimum(across, CLIP_PX),
        np.sin(2 * strike), np.cos(2 * strike),
        at("rcode"), at("scode"), at("fcode"), at("mapscale"), at("logslip"),
        np.minimum(d_act, CLIP_PX), np.minimum(d_holo, CLIP_PX), dens_act,
    ]
    X = np.column_stack([np.asarray(c, np.float32) for c in cols])
    assert X.shape[1] == len(CTX_NAMES)
    return X


def sub_hide(visible: np.ndarray, region: np.ndarray, rng: np.random.Generator, frac: float = 0.20,
             buffer_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Hide `frac` of the visible 8-connected components inside `region` (+ collar).

    Returns (hidden_mask, visible_after) -- the training analogue of the holdout draw so
    that training positives see context features computed from OTHER traces, exactly as
    withheld components do at evaluation time.
    """
    lbl, n = ndi.label(visible & region, structure=np.ones((3, 3), np.int8))
    if n == 0:
        return np.zeros_like(visible), visible.copy()
    pick = rng.random(n + 1) < frac
    pick[0] = False
    hidden = pick[lbl]
    collar = ndi.binary_dilation(hidden, structure=_disk(buffer_px))
    vis_after = visible & ~hidden & ~collar
    return hidden, vis_after


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r
