"""Label-free transforms of the template's cached rank-intensity radiometrics."""
import numpy as np
from scipy import ndimage as ndi


def normalized(a, valid, percentile=99):
    """Explicit label-free scaling, not a repair at serialization time."""
    if not valid.any():
        raise ValueError('empty feature domain')
    a = np.asarray(a, np.float32)
    if not np.isfinite(a[valid]).all() or (a[valid] < 0).any():
        raise ValueError('non-finite or negative response')
    scale = float(np.percentile(a[valid], percentile))
    if scale <= 0:
        raise ValueError('degenerate response')
    out = np.zeros_like(a)
    out[valid] = np.minimum(a[valid] / scale, 1)
    return out, scale


def curvature(a, sigma=2):
    """Absolute principal Hessian curvature and eigenvalue anisotropy."""
    yy = ndi.gaussian_filter(a, sigma, order=(2, 0), mode='nearest') * sigma**2
    xx = ndi.gaussian_filter(a, sigma, order=(0, 2), mode='nearest') * sigma**2
    xy = ndi.gaussian_filter(a, sigma, order=(1, 1), mode='nearest') * sigma**2
    gap = np.sqrt((yy - xx)**2 + 4 * xy**2)
    hi = np.abs((xx + yy + gap) / 2)
    lo = np.abs((xx + yy - gap) / 2)
    strongest, weakest = np.maximum(hi, lo), np.minimum(hi, lo)
    anisotropy = (strongest - weakest) / (strongest + weakest + 1e-8)
    return strongest.astype(np.float32), anisotropy.astype(np.float32)


def build(raw, footprint):
    """No catalogue argument: hidden/visible faults cannot enter any feature.

    Zero is nodata, valid raw values are 1..255. Rank-intensities are bounded
    re-expressions of independently quantized channels, NOT K/Th/U ppm ratios.
    Erosion exceeds the 4*sigma Gaussian support to avoid nodata-edge responses.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[1:] != footprint.shape:
        raise ValueError('expected K,Th,U on footprint grid')
    supported = footprint & (raw > 0).all(axis=0)
    if not supported.any():
        raise ValueError('no radiometric support')
    valid = ndi.binary_erosion(supported, iterations=10)
    ranks = (raw.astype(np.float32) - 1) / 254
    # Constants cancel in U - (K+Th)/2, matching the registered U/254 contrast.
    ranks[:, ~supported] = 0
    k, th, u = ranks
    common = (k + th) / 2
    residual = u - common
    principal, anisotropy = curvature(residual)
    common_curve, _ = curvature(common)
    responses = {'K':k, 'Th':th, 'U':u,
                 'absolute_selective_U':np.abs(residual),
                 'principal_curvature':principal, 'anisotropy':anisotropy,
                 'common_mode_curvature':common_curve,
                 'candidate':principal * anisotropy}
    scales = {}
    for name in responses:
        responses[name], scales[name] = normalized(responses[name], valid)
    responses['constant_001'] = np.where(valid, .01, 0).astype(np.float32)
    return responses, valid, scales
