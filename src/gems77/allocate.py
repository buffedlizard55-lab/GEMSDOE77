"""Live-inversion allocation (GEMSDOE77 lane): place binary dots where the live-calibrated
expected credit per dot is highest, under the parallel-run uniqueness constraint.

Projection (INSTRUMENT, never a score):
    T = sum_y h(y) M(y),  Mk ~= T / c,  K = sum_y h(y),
    DTI_proj = T / (0.2 (T + S - T/c) + 0.8 K)
with h(y) the per-cell hidden-truth density fitted to 95 organizer scores
(scripts/e2_inversion.py, scripts/e2_hypothesis_test.py).
"""
from __future__ import annotations

import numpy as np
from scipy import spatial

from . import core


def hmap_from_bins(bins: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = np.zeros(bins.shape, np.float32)
    ok = bins >= 0
    out[ok] = np.asarray(h, np.float32)[bins[ok]]
    return out


def credit_map(hm: np.ndarray) -> np.ndarray:
    """Expected credit of one isolated binary dot at x: sum_o k(|o|) h(x + o)."""
    hgt, wid = hm.shape
    pad = np.pad(hm, 3)
    out = np.zeros_like(hm)
    for dy, dx, k in core._offsets():
        out += np.float32(k) * pad[3 + dy:3 + dy + hgt, 3 + dx:3 + dx + wid]
    return out


def project(dots: np.ndarray, hm: np.ndarray, c: float = 2.0) -> dict:
    p = dots.astype(np.float32)
    m = core.cover_field(p)
    T = float((hm * m).sum())
    S = float(p.sum())
    K = float(hm.sum())
    dti = T / (0.2 * (T + S - T / c) + 0.8 * K)
    return {"T": T, "S": S, "K": K, "dti_projection": dti, "credit_per_dot": T / max(S, 1)}


def greedy_select(order_idx: np.ndarray, min_sep: float = 3.0, taken: np.ndarray | None = None,
                  limit: int | None = None) -> np.ndarray:
    """Greedy Poisson-disk selection along a priority order (flat indices).

    A candidate is accepted if no accepted dot (including ``taken``) lies closer than
    ``min_sep`` pixels (Euclidean).  Uses a coarse occupancy grid for speed.
    """
    W = core.W
    cell = int(np.ceil(min_sep))
    occ: dict[tuple[int, int], list[tuple[int, int]]] = {}
    acc = []
    if taken is not None:
        for f in taken:
            r, c = divmod(int(f), W)
            occ.setdefault((r // cell, c // cell), []).append((r, c))
    s2 = min_sep * min_sep
    for f in order_idx:
        r, c = divmod(int(f), W)
        gr, gc = r // cell, c // cell
        ok = True
        for a in (gr - 1, gr, gr + 1):
            for b in (gc - 1, gc, gc + 1):
                for (rr, cc) in occ.get((a, b), ()):
                    if (rr - r) ** 2 + (cc - c) ** 2 < s2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            occ.setdefault((gr, gc), []).append((r, c))
            acc.append(int(f))
            if limit is not None and len(acc) >= limit:
                break
    return np.asarray(acc, dtype=np.int64)


def overlap_fraction(mine_idx: np.ndarray, theirs_mask: np.ndarray, radius: float = 3.0) -> float:
    """Fraction of my dots within ``radius`` px (Euclidean) of any positive cell of theirs."""
    rr, cc = np.nonzero(theirs_mask)
    if len(rr) == 0 or len(mine_idx) == 0:
        return 0.0
    tree = spatial.cKDTree(np.c_[rr, cc])
    mr, mc = np.divmod(mine_idx, core.W)
    d, _ = tree.query(np.c_[mr, mc], k=1, distance_upper_bound=radius + 1e-6)
    return float(np.mean(d <= radius))


def budgeted_selection(on_order: np.ndarray, off_order: np.ndarray, member, n_on: int,
                       budget: float = 0.66, min_sep: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """On-support dots (first ``n_on`` accepted by spacing) + off-support dots under a per-raster
    overlap budget: for every registry raster R, |D within 3 px of R| <= budget * |D|.

    ``member(cell)`` returns the 0/1 membership vector of a flat cell index in every registry
    halo.  Off-support candidates are accepted (in priority order) only if they keep every
    raster within budget, or lie outside the halo of every raster currently over budget.
    Returns (on_sel, off_sel, overlap_counts).
    """
    on_sel = greedy_select(on_order, min_sep, limit=n_on)
    o = None
    for f in on_sel:
        m = member(int(f))
        o = m.astype(np.int64) if o is None else o + m
    n = len(on_sel)
    W = core.W
    cell = int(np.ceil(min_sep))
    s2 = min_sep * min_sep
    occ: dict = {}
    for f in on_sel:
        r, c = divmod(int(f), W)
        occ.setdefault((r // cell, c // cell), []).append((r, c))
    off = []
    need = int(np.ceil(n * (1 - budget) / budget))
    for f in off_order:
        f = int(f)
        r, c = divmod(f, W)
        gr, gc = r // cell, c // cell
        if any((rr - r) ** 2 + (cc - c) ** 2 < s2 for a in (gr - 1, gr, gr + 1)
               for b in (gc - 1, gc, gc + 1) for (rr, cc) in occ.get((a, b), ())):
            continue
        m = member(f)
        viol = o > budget * n
        if viol.any():
            if (m[viol] == 1).any() or np.any((o + m)[~viol] > budget * (n + 1)):
                continue
        elif np.any(o + m > budget * (n + 1)):
            continue
        o = o + m
        n += 1
        off.append(f)
        occ.setdefault((gr, gc), []).append((r, c))
        if not (o > budget * n).any() and len(off) >= need:
            break
    return on_sel, np.asarray(off, dtype=np.int64), o
