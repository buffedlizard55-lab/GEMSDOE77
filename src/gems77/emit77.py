"""Fixed-budget rank-ordered minimum-distance dot emitter (GEMSDOE77).

Candidates are ranked by the shared template's expected kernel credit
``gems57.emit.expected_credit(p)`` and accepted greedily subject to a minimum
separation (default 2.8 px, matched to the official 300 m triangular kernel).
Fixed budgets make arm comparisons mass-matched by construction.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
from gems57.emit import expected_credit  # noqa: E402

_OFFSETS_28 = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
               if 0 < (dy * dy + dx * dx) ** 0.5 < 2.8]


def emit_dots(p: np.ndarray, allowed: np.ndarray, budget: int,
              min_sep: float = 2.8, candidate_factor: int = 10) -> np.ndarray:
    """Binary dot mask: top-`budget` allowed cells by expected credit, >=min_sep apart."""
    p = np.clip(np.asarray(p, np.float32), 0.0, 1.0)
    allowed = np.asarray(allowed, bool)
    if budget <= 0 or not allowed.any():
        return np.zeros(p.shape, bool)
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
    offs = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)
            if (dy * dy + dx * dx) ** 0.5 < min_sep]
    n = 0
    for fidx in cand:
        if n >= budget:
            break
        y, x = divmod(int(fidx), W)
        if blocked[y, x] or not allowed[y, x]:
            continue
        out[y, x] = True
        n += 1
        for dy, dx in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                blocked[yy, xx] = True
    return out
