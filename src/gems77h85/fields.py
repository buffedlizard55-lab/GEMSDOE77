"""H85-XWC fields: multi-scale magnetic x gravity cross-field coherence.

Label-blind by construction: only ``training_features.tif`` band 2 (``rtp``)
and band 13 (``iso_grav_anom``) enter the surface.  The catalogue is used
nowhere in this module (holdout splits and emission masking live in the
pipeline script, using fold-visible faults only).

Method (preregistered in ``research/h85_hypotheses.md``):
high-pass (field minus Gaussian sigma=8 px) -> robust-z in footprint ->
FFT log-Gabor-style bandpass at wavelengths 4/8/16 px -> per-scale local
coherence ``|S(XY)|/sqrt(S(X^2) S(Y^2))`` (Gaussian smoothing S, sigma =
wavelength/2) times rank-normalised joint edge energy -> mean over scales,
weighted by cross-scale phase-sign agreement -> rank-normalised surface.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

WAVELENGTHS_PX = (4.0, 8.0, 16.0)
HIGHPASS_SIGMA_PX = 8.0
_BANDWIDTH_OCT = 0.75  # log2-frequency Gaussian width of each bandpass


def clean_field(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Scrub non-finite / sentinel cells to the in-footprint median (float64)."""
    v = np.asarray(a, dtype=np.float64)
    ok = footprint & np.isfinite(v) & (np.abs(v) <= 1e18)
    if not ok.any():
        raise ValueError("field has no usable in-footprint cells")
    fill = float(np.median(v[ok]))
    out = np.where(np.isfinite(v) & (np.abs(v) <= 1e18), v, fill)
    return out


def robust_z(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    v = a[footprint]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) * 1.4826)
    if not np.isfinite(mad) or mad <= 0:
        mad = max(float(np.std(v)), 1e-12)
    return np.clip((a - med) / mad, -10.0, 10.0)


def highpass(a: np.ndarray, sigma: float = HIGHPASS_SIGMA_PX) -> np.ndarray:
    return a - ndi.gaussian_filter(a, sigma, mode="nearest")


def bandpass(a: np.ndarray, wavelength_px: float) -> np.ndarray:
    """FFT radial log-Gabor bandpass centred at ``1/wavelength_px`` cycles/px."""
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fy * fy + fx * fx)
    f0 = 1.0 / float(wavelength_px)
    with np.errstate(divide="ignore"):
        lf = np.log2(np.where(f > 0, f / f0, 1.0))
    filt = np.exp(-0.5 * (lf / _BANDWIDTH_OCT) ** 2)
    filt[f == 0] = 0.0
    spec = np.fft.rfft2(a)
    out = np.fft.irfft2(spec * filt, s=a.shape)
    return out


def rank01(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Percentile rank inside the footprint mapped to [0,1]; 0 outside."""
    out = np.zeros(a.shape, dtype=np.float32)
    vals = a[footprint].astype(np.float64)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(vals.size, dtype=np.float64)
    ranks[order] = np.arange(vals.size, dtype=np.float64)
    if vals.size > 1:
        ranks /= (vals.size - 1)
    out[footprint] = ranks.astype(np.float32)
    return out


def coherence_surface(rtp_raw: np.ndarray, grav_raw: np.ndarray,
                      footprint: np.ndarray) -> dict[str, np.ndarray]:
    """Return candidate surface plus single-field control surfaces.

    Keys: ``xwc`` (candidate), ``rtp_only``, ``grav_only`` (controls),
    ``coherence_mean`` and ``phase_agreement`` (canary components).
    All are float32 in [0,1], zero outside the footprint.
    """
    fp = np.asarray(footprint, bool)
    x = robust_z(highpass(clean_field(rtp_raw, fp)), fp)
    y = robust_z(highpass(clean_field(grav_raw, fp)), fp)

    coh_scales, e_rank_scales, ex_scales, ey_scales, signs = [], [], [], [], []
    for wl in WAVELENGTHS_PX:
        bx = bandpass(x, wl)
        by = bandpass(y, wl)
        sig = wl / 2.0
        sxy = ndi.gaussian_filter(bx * by, sig, mode="nearest")
        sxx = ndi.gaussian_filter(bx * bx, sig, mode="nearest")
        syy = ndi.gaussian_filter(by * by, sig, mode="nearest")
        denom = np.sqrt(np.maximum(sxx * syy, 1e-300))
        coh = np.clip(np.abs(sxy) / denom, 0.0, 1.0)
        energy = np.sqrt(np.maximum(sxx + syy, 0.0))
        coh_scales.append(coh.astype(np.float32))
        e_rank_scales.append(rank01(energy, fp))
        ex_scales.append(rank01(np.sqrt(np.maximum(sxx, 0.0)), fp))
        ey_scales.append(rank01(np.sqrt(np.maximum(syy, 0.0)), fp))
        signs.append(np.sign(sxy))
        del bx, by, sxy, sxx, syy, energy

    coh_mean = np.mean(coh_scales, axis=0)
    joint = np.mean([c * e for c, e in zip(coh_scales, e_rank_scales)], axis=0)
    sgn = np.stack(signs)  # (n_scale, H, W)
    phase_agreement = np.abs(np.mean(sgn, axis=0)).astype(np.float32)  # 1/3..1 for 3 scales
    surface = rank01(joint * (0.5 + 0.5 * phase_agreement), fp)
    return {
        "xwc": surface,
        "rtp_only": np.mean(ex_scales, axis=0).astype(np.float32),
        "grav_only": np.mean(ey_scales, axis=0).astype(np.float32),
        "coherence_mean": rank01(coh_mean, fp),
        "phase_agreement": np.where(fp, phase_agreement, 0.0).astype(np.float32),
    }
