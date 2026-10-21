"""Uniqueness-capped greedy dot emission (lane H79, amendment E3b).

Same ordering and spacing as ``gems77.emit77.emit_dots`` (expected credit, min_sep 2.8 px),
plus capacity constraints: for constraint k with 3 px halo H_k, at most ``caps[k]`` accepted
dots may lie inside H_k. Halos are packed into per-pixel bit words so a candidate is tested
with one lookup per word. Precedent: H78 lane (IR-78-03). Nothing here changes the surface.
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

from gems77.emit77 import expected_credit

RADIUS_PX = 3.0


def halo_of_raster(path, footprint: np.ndarray, radius_px: float = RADIUS_PX) -> np.ndarray:
    """3 px halo of a registry raster's literal support (finite & > 0), restricted to the footprint.
    Mirrors gems57.uniqueness._dots/_near exactly (the audit's definition)."""
    with rasterio.open(path) as src:
        if src.shape != footprint.shape:
            raise ValueError("grid mismatch")
        a = src.read(1)
    positive = np.isfinite(a) & (a > 0) & footprint
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return binary_dilation(positive, structure=yy * yy + xx * xx <= radius_px * radius_px)


def pack_halos(halos: list[np.ndarray]) -> np.ndarray:
    """(n_words, H, W) uint64 bit words; bit k of word k//64 set where halo k covers the pixel."""
    if not halos:
        return np.zeros((0,) + (1, 1), np.uint64)
    H, W = halos[0].shape
    n_words = (len(halos) + 63) // 64
    words = np.zeros((n_words, H, W), np.uint64)
    for k, hmask in enumerate(halos):
        words[k // 64] |= np.where(hmask, np.uint64(1) << np.uint64(k % 64), np.uint64(0))
    return words


def emit_dots_capped(p: np.ndarray, allowed: np.ndarray, budget: int, words: np.ndarray, caps: np.ndarray,
                     min_sep: float = 2.8, candidate_factor: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Greedy emission under per-halo capacities. Returns (dot mask, counts per constraint)."""
    p = np.clip(np.asarray(p, np.float32), 0.0, 1.0)
    allowed = np.asarray(allowed, bool)
    K = int(caps.size)
    counts = np.zeros(K, np.int64)
    if budget <= 0 or not allowed.any():
        return np.zeros(p.shape, bool), counts
    ek = expected_credit(p)
    key = np.where(allowed, ek, np.float32(-np.inf))
    flat = key.ravel()
    n_cand = int(min(flat.size, max(budget * candidate_factor, budget + 1000)))
    cand = np.argpartition(flat, -n_cand)[-n_cand:]
    cand = cand[np.argsort(-flat[cand], kind="stable")]
    cand = cand[np.isfinite(flat[cand])]
    H, W = p.shape
    blocked = np.zeros((H, W), bool)
    out = np.zeros((H, W), bool)
    offs = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if (dy * dy + dx * dx) ** 0.5 < min_sep]
    n_words = words.shape[0] if K else 0
    full = [0] * n_words  # bit set once a constraint is saturated
    caps_l = [int(c) for c in caps]
    cnt = [0] * K
    n = 0
    for fidx in cand:
        if n >= budget:
            break
        y, x = divmod(int(fidx), W)
        if blocked[y, x] or not allowed[y, x]:
            continue
        bits = [int(words[wi, y, x]) for wi in range(n_words)]
        if any(b & f for b, f in zip(bits, full)):
            continue
        out[y, x] = True
        n += 1
        for wi, b in enumerate(bits):
            while b:
                low = b & -b
                k = wi * 64 + low.bit_length() - 1
                cnt[k] += 1
                if cnt[k] >= caps_l[k]:
                    full[wi] |= low
                b ^= low
        for dy, dx in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                blocked[yy, xx] = True
    counts[:] = cnt
    return out, counts
