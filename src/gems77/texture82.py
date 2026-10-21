"""Frozen H82 rank-intensity texture-boundary hypothesis; no label inputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter


def local_sd(a: np.ndarray, size: int = 9) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("local_sd needs finite 2-D data")
    if size < 3 or size % 2 != 1:
        raise ValueError("moment window must be odd and at least 3")
    mean = uniform_filter(a, size=size, mode="constant", cval=0)
    variance = uniform_filter(a * a, size=size, mode="constant", cval=0) - mean * mean
    return np.sqrt(np.maximum(variance, 0))


def normalize(a: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, float]:
    if not eligible.any():
        raise ValueError("empty feature support")
    scale = float(np.percentile(a[eligible], 99))
    out = np.zeros(a.shape, np.float32)
    if scale > 0:
        out[eligible] = np.clip(a[eligible] / scale, 0, 1)
    return out, scale


def build(k: np.ndarray, th: np.ndarray, footprint: np.ndarray):
    k, th, footprint = np.asarray(k), np.asarray(th), np.asarray(footprint, bool)
    if k.ndim != 2 or k.shape != th.shape or k.shape != footprint.shape:
        raise ValueError("feature grid mismatch")
    if not np.isfinite(k).all() or not np.isfinite(th).all():
        raise ValueError("cache contains nonfinite values")
    if min(k.min(), th.min()) < 0 or max(k.max(), th.max()) > 255:
        raise ValueError("cache must have uint8 rank-intensity range")
    valid = footprint & (k > 0) & (th > 0)
    # Square support, not taxicab erosion: every needed stencil location exists.
    eligible = minimum_filter(valid.astype(np.uint8), size=11, mode="constant", cval=0).astype(bool)
    raw = {}
    gradients = []
    for name, channel in (("K", k), ("Th", th)):
        values = channel.astype(np.float64) / 255
        sd = local_sd(values)
        dy, dx = np.gradient(sd)
        grad = np.hypot(dx, dy)
        raw[name] = values.astype(np.float32)
        raw[name + "_sd"] = sd.astype(np.float32)
        raw[name + "_texture_gradient"] = grad.astype(np.float32)
        gradients.append(grad)
    raw["candidate"] = np.sqrt(gradients[0] * gradients[1]).astype(np.float32)
    fields, scales = {}, {}
    for name, values in raw.items():
        fields[name], scales[name] = normalize(values, eligible)
    fields["constant"] = eligible.astype(np.float32)
    raw["constant"] = fields["constant"]
    return raw, fields, eligible, scales
