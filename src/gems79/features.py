"""Scale-space orientation-persistence features for lane H79.

Scientific hypothesis
---------------------
A blind fault beneath basin fill can be a weak, laterally persistent contact
rather than a single high-amplitude edge.  A real structural boundary should
therefore survive a modest scale change and have a stable *unoriented* edge
normal in multiple independent fields (reduced-to-pole magnetics, isostatic
gravity, and detrended elevation).  The implementation measures that statement
with Gaussian derivative states and second-angle circular concentration.  It
is deliberately different from a scalar gradient, a scarp maximum, a Radon
line integral, or a catalogue-distance feature.

The functions in this module accept a cached feature stack and a footprint only.
They do not accept labels, an existing-fault mask, a holdout truth mask, or any
prior submission raster.  All output cells are finite and in [0, 1]; cells
outside the footprint are zeroed by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class OrientationPersistenceConfig:
    """Frozen transform settings for reproducibility and pre-registration."""

    sigmas: tuple[float, ...] = (1.5, 3.0, 6.0)
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    epsilon: float = 0.03


SOURCE_NAMES = ("rtp", "iso_grav_anom", "det_elev")
OUTPUT_NAMES = (
    "h79_rtp_edge_mean",
    "h79_rtp_edge_persistence",
    "h79_rtp_orientation_coherence",
    "h79_grav_edge_mean",
    "h79_grav_edge_persistence",
    "h79_grav_orientation_coherence",
    "h79_dem_edge_mean",
    "h79_dem_edge_persistence",
    "h79_dem_orientation_coherence",
    "h79_multisensor_persistence",
    "h79_multisensor_orientation_confluence",
    "h79_scale_survival",
    "h79_edge_disagreement",
    "h79_structural_persistence",
)


def _robust_unit(a: np.ndarray, mask: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Robustly scale a finite field to [0, 1] without using labels."""
    v = np.asarray(a, dtype=np.float32)[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(a.shape, dtype=np.float32)
    q0, q1 = np.percentile(v, [lo, hi]).astype(np.float32)
    if not np.isfinite(q0) or not np.isfinite(q1) or q1 <= q0:
        return np.zeros(a.shape, dtype=np.float32)
    return np.clip((np.nan_to_num(a, nan=float(q0)) - q0) / (q1 - q0), 0.0, 1.0).astype(np.float32)


def _filled(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median-fill only for convolution; output is masked later."""
    x = np.asarray(a, dtype=np.float32)
    v = x[mask & np.isfinite(x)]
    med = float(np.median(v)) if v.size else 0.0
    return np.where(mask & np.isfinite(x), x, np.float32(med)).astype(np.float32)


def _state(field: np.ndarray, mask: np.ndarray, config: OrientationPersistenceConfig) -> dict[str, np.ndarray]:
    """Return edge magnitude, persistence and double-angle coherence for one field.

    Geological lineations have an axis, not an arrow.  The doubled angle
    representation (cos 2θ, sin 2θ) consequently treats θ and θ+π as equal.
    ``persistence`` is a geometric mean across scales, so one isolated response
    cannot dominate it.  The calculations are label-blind.
    """
    f = _filled(field, mask)
    strengths: list[np.ndarray] = []
    cos_terms: list[np.ndarray] = []
    sin_terms: list[np.ndarray] = []
    for sigma in config.sigmas:
        gx = ndimage.gaussian_filter(f, sigma=sigma, order=(0, 1), mode="nearest")
        gy = ndimage.gaussian_filter(f, sigma=sigma, order=(1, 0), mode="nearest")
        mag = np.hypot(gx, gy).astype(np.float32)
        strength = _robust_unit(mag, mask, config.percentile_low, config.percentile_high)
        den = mag * mag + np.float32(1e-12)
        cos_terms.append(((gx * gx - gy * gy) / den * strength).astype(np.float32))
        sin_terms.append(((2.0 * gx * gy) / den * strength).astype(np.float32))
        strengths.append(strength)
        del gx, gy, mag, den

    s = np.stack(strengths, axis=0).astype(np.float32)
    weight = s + np.float32(config.epsilon)
    mean_strength = np.mean(s, axis=0).astype(np.float32)
    persistence = np.clip(np.exp(np.mean(np.log(weight), axis=0)), 0.0, 1.0).astype(np.float32)
    c = np.sum(np.stack(cos_terms, axis=0), axis=0)
    q = np.sum(np.stack(sin_terms, axis=0), axis=0)
    coherence = np.clip(np.hypot(c, q) / (np.sum(weight, axis=0) + 1e-12), 0.0, 1.0).astype(np.float32)
    # response ratio between the broad and narrow scales; positive means a
    # broad feature survives better than a pixel-scale edge.
    scale_survival = np.clip(s[-1] / (s[0] + np.float32(config.epsilon)), 0.0, 1.0).astype(np.float32)
    for out in (mean_strength, persistence, coherence, scale_survival):
        out[~mask] = 0.0
    return {
        "mean": mean_strength,
        "persistence": persistence,
        "coherence": coherence,
        "scale_survival": scale_survival,
        "cos": np.clip(c / (np.sum(weight, axis=0) + 1e-12), -1.0, 1.0).astype(np.float32),
        "sin": np.clip(q / (np.sum(weight, axis=0) + 1e-12), -1.0, 1.0).astype(np.float32),
    }


def build(stack: np.ndarray, channels: Mapping[str, int], footprint: np.ndarray,
          config: OrientationPersistenceConfig = OrientationPersistenceConfig()) -> tuple[np.ndarray, list[str]]:
    """Build the H79 feature matrix from the cached stack.

    Parameters
    ----------
    stack:
        H×W×C float32 memmap from ``scripts/build_features.py``.
    channels:
        Name-to-column mapping for the cached stack.
    footprint:
        Official finite sample-submission footprint.  It is not a target mask.
    config:
        Pre-registered Gaussian scales and robust normalization settings.
    """
    fp = np.asarray(footprint, dtype=bool)
    if stack.ndim != 3 or stack.shape[:2] != fp.shape:
        raise ValueError("stack and footprint shape mismatch")
    missing = [n for n in (*SOURCE_NAMES, "depth_to_base_surf") if n not in channels]
    if missing:
        raise KeyError(f"cached stack is missing H79 source channels: {missing}")

    states: dict[str, dict[str, np.ndarray]] = {}
    for name in SOURCE_NAMES:
        field = np.asarray(stack[:, :, int(channels[name])], dtype=np.float32)
        states[name] = _state(field, fp, config)
        del field

    r, g, d = (states[n] for n in SOURCE_NAMES)
    persistence = np.stack([r["persistence"], g["persistence"], d["persistence"]], axis=0)
    multisensor_persistence = np.clip(
        np.exp(np.mean(np.log(persistence + np.float32(config.epsilon)), axis=0)), 0.0, 1.0
    ).astype(np.float32)

    # A fault-related edge is more credible when independent fields agree on
    # its axis.  The vector sum is weighted by each field's edge persistence.
    cos_stack = np.stack([r["cos"], g["cos"], d["cos"]], axis=0)
    sin_stack = np.stack([r["sin"], g["sin"], d["sin"]], axis=0)
    denom = np.sum(persistence, axis=0) + 1e-12
    confluence = np.clip(np.hypot(np.sum(cos_stack * persistence, axis=0),
                                  np.sum(sin_stack * persistence, axis=0)) / denom, 0.0, 1.0).astype(np.float32)
    edge_disagreement = np.clip(1.0 - confluence, 0.0, 1.0).astype(np.float32)
    scale_survival = np.mean(
        np.stack([r["scale_survival"], g["scale_survival"], d["scale_survival"]], axis=0),
        axis=0,
    ).astype(np.float32)
    structural_persistence = np.clip(multisensor_persistence * confluence, 0.0, 1.0).astype(np.float32)

    out = np.stack([
        r["mean"], r["persistence"], r["coherence"],
        g["mean"], g["persistence"], g["coherence"],
        d["mean"], d["persistence"], d["coherence"],
        multisensor_persistence, confluence, scale_survival, edge_disagreement,
        structural_persistence,
    ], axis=-1).astype(np.float32)
    out[~fp] = 0.0
    if not np.isfinite(out).all() or out.min() < 0.0 or out.max() > 1.0:
        raise ValueError(
            f"H79 transform did not produce finite unit-range features: "
            f"finite={int(np.isfinite(out).sum())}/{out.size} min={float(np.nanmin(out))} max={float(np.nanmax(out))}"
        )
    return out, list(OUTPUT_NAMES)
