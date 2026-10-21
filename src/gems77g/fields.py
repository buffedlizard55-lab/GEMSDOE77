"""Label-free H77G geophysical feature block for blind hydrothermal fault conduits.

Preregistered in ``registry/preregistration_h77g.json`` before Experiment 1.
All channels are computed strictly from geophysical bands in ``training_features.tif``
within the valid study footprint of ``sample_submission.tif``:
  - zero fault-catalogue inputs,
  - zero SGMC bedrock-contact inputs,
  - zero prior submission rasters.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from gems77 import features77 as F77

H77G_NAMES = [
    "h77g_demag_trough_s1p5",
    "h77g_demag_trough_s3p0",
    "h77g_demag_persistence",
    "h77g_grav_base_step",
    "h77g_grav_over_mag_decoupling",
    "h77g_mt_claycap_margin",
    "h77g_transtension_index",
    "h77g_declustered_swarm_frac",
    "h77g_basement_saddle",
    "h77g_composite_surface",
]


def hessian_trough(field: np.ndarray, sigma: float, hp_sigma: float = 12.0) -> np.ndarray:
    """Extract narrow linear negative-anomaly troughs (valleys) at scale ``sigma``.

    High-pass filters ``field`` by subtracting a Gaussian regional of scale
    ``hp_sigma`` and computes the scale-normalized principal positive Hessian
    eigenvalue across the valley (``lambda_1 > 0``) weighted by linear
    tubularity ``1 - min(|lambda_2|, lambda_1) / (lambda_1 + eps)``.
    Note (IR-H77G-02): ``F77.hessian_ridge`` in ``src/gems77/features77.py``
    computes ``-small * (big < 0)`` (along-axis curvature rather than
    across-axis curvature), which vanishes on straight 1-D ridges/troughs;
    this function uses the true across-trough eigenvalue ``lambda_1``.
    """
    f = np.asarray(field, dtype=np.float32)
    hp = f - ndimage.gaussian_filter(f, hp_sigma, mode="nearest")
    axx = ndimage.gaussian_filter(hp, sigma, order=(0, 2), mode="nearest")
    ayy = ndimage.gaussian_filter(hp, sigma, order=(2, 0), mode="nearest")
    axy = ndimage.gaussian_filter(hp, sigma, order=(1, 1), mode="nearest")
    tr = (axx + ayy) * 0.5
    disc = np.sqrt(np.maximum(((axx - ayy) * 0.5) ** 2 + axy * axy, 0.0))
    l1 = tr + disc  # largest signed eigenvalue (across-valley curvature > 0 in a trough)
    l2 = tr - disc  # orthogonal eigenvalue (along-valley curvature ~ 0 on a linear trough)
    pos = np.maximum(l1, 0.0)
    tubularity = 1.0 - np.minimum(np.abs(l2), pos) / (pos + np.float32(1e-9))
    return (np.float32(sigma * sigma) * pos * tubularity).astype(np.float32)


def hessian_saddle(field: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Opposite-sign Hessian eigenvalue saddle detector (lambda_1 > 0, lambda_2 < 0)."""
    f = np.asarray(field, dtype=np.float32)
    axx = ndimage.gaussian_filter(f, sigma, order=(0, 2), mode="nearest")
    ayy = ndimage.gaussian_filter(f, sigma, order=(2, 0), mode="nearest")
    axy = ndimage.gaussian_filter(f, sigma, order=(1, 1), mode="nearest")
    tr = (axx + ayy) * 0.5
    disc = np.sqrt(np.maximum(((axx - ayy) * 0.5) ** 2 + axy * axy, 0.0))
    l1 = tr + disc
    l2 = tr - disc
    prod = l1 * l2
    return np.where(prod < 0.0, np.sqrt(np.maximum(-prod, 0.0)), 0.0).astype(np.float32)


def _grad(field: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy = ndimage.gaussian_filter(field, sigma, order=(1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(field, sigma, order=(0, 1), mode="nearest")
    mag = np.hypot(gx, gy).astype(np.float32)
    return gy.astype(np.float32), gx.astype(np.float32), mag


def build_h77g_fields(
    bands: dict[str, np.ndarray],
    footprint: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute all 10 preregistered H77G channels and summary diagnostics.

    Parameters
    ----------
    bands : dict[str, np.ndarray]
        Cleaned float32 arrays keyed by `` BAND_NAMES ``:
        ``rtp``, ``tmi_hg``, ``tmi_vg``, ``iso_grav_anom``, ``depth_to_base_surf``,
        ``cond_surf``, ``geod_dilaterate``, ``geod_shearrate``, ``ieq_n100a15``,
        ``deq_n100a15``.
    footprint : np.ndarray of bool
        Valid study-area mask from ``sample_submission.tif``.
    """
    foot = np.asarray(footprint, dtype=bool)
    rk = lambda a: F77.rank01(np.where(foot, a, np.float32(np.nan)), foot)  # noqa: E731

    rtp = F77.fill_median(F77.clean(bands["rtp"]), foot)
    tmi_hg = F77.fill_median(F77.clean(bands["tmi_hg"]), foot)
    tmi_vg = F77.fill_median(F77.clean(bands["tmi_vg"]), foot)
    grav = F77.fill_median(F77.clean(bands["iso_grav_anom"]), foot)
    depth = F77.fill_median(F77.clean(bands["depth_to_base_surf"]), foot)
    cond = F77.fill_median(F77.clean(bands["cond_surf"]), foot)
    dilat = F77.fill_median(F77.clean(bands["geod_dilaterate"]), foot)
    shear = F77.fill_median(F77.clean(bands["geod_shearrate"]), foot)
    ieq = F77.fill_median(F77.clean(bands["ieq_n100a15"]), foot)
    deq = F77.fill_median(F77.clean(bands["deq_n100a15"]), foot)

    # 1-3: Multi-scale RTP hydrothermal demagnetization trough
    t15_raw = hessian_trough(rtp, sigma=1.5, hp_sigma=12.0)
    t30_raw = hessian_trough(rtp, sigma=3.0, hp_sigma=12.0)
    t15 = rk(t15_raw)
    t30 = rk(t30_raw)
    demag_persist = rk(np.sqrt(np.clip(t15 * t30, 0.0, 1.0)))

    # 4: Oriented gravity-basement normal-fault step concordance
    ggy, ggx, gmag = _grad(grav, sigma=2.0)
    dgy, dgx, dmag = _grad(depth, sigma=2.0)
    cos_align = np.abs(ggx * dgx + ggy * dgy) / (gmag * dmag + np.float32(1e-9))
    rg = rk(gmag)
    rd = rk(dmag)
    grav_base_step = rk(np.sqrt(np.clip(rg * rd, 0.0, 1.0)) * np.clip(cos_align, 0.0, 1.0))

    # 5: Gravity-step over magnetic analytic-signal decoupling (demagnetized step)
    mag_as = np.hypot(np.maximum(tmi_hg, 0.0), tmi_vg).astype(np.float32)
    r_as = rk(mag_as)
    decouple = rk(np.maximum(0.0, rg - np.float32(0.5) * r_as))

    # 6: MT electrical conductance clay-cap lateral margin
    _, _, cmag = _grad(cond, sigma=2.0)
    r_cond = rk(cond)
    r_cgrad = rk(cmag)
    mt_margin = rk(np.sqrt(np.clip(r_cond * r_cgrad, 0.0, 1.0)))

    # 7: Geodetic transtensional opening index
    r_dil = rk(np.maximum(dilat, 0.0))
    r_shr = rk(np.maximum(shear, 0.0))
    transtension = rk(np.sqrt(np.clip(r_dil * r_shr, 0.0, 1.0)))

    # 8: Declustered independent earthquake swarm fraction
    r_ieq = rk(ieq)
    r_deq = rk(deq)
    swarm_frac = rk(r_ieq * (r_ieq / (r_ieq + r_deq + np.float32(0.05))))

    # 9: Subsurface basement-depth Hessian saddle (buried relay transfer zone)
    base_saddle = rk(hessian_saddle(depth, sigma=2.5))

    # 10: Preregistered composite surface
    regime_term = np.float32(0.30) + np.float32(0.35) * transtension + np.float32(0.35) * swarm_frac
    mt_term = np.float32(0.25) + np.float32(0.75) * mt_margin
    raw_comp = (
        np.power(np.clip(demag_persist, 1e-6, 1.0), np.float32(0.35))
        * np.power(np.clip(grav_base_step, 1e-6, 1.0), np.float32(0.35))
        * np.power(np.clip(mt_term, 1e-6, 1.0), np.float32(0.20))
        * np.power(np.clip(regime_term, 1e-6, 1.0), np.float32(0.10))
    )
    composite = rk(raw_comp)

    out = {
        "h77g_demag_trough_s1p5": np.where(foot, t15, 0.0).astype(np.float32),
        "h77g_demag_trough_s3p0": np.where(foot, t30, 0.0).astype(np.float32),
        "h77g_demag_persistence": np.where(foot, demag_persist, 0.0).astype(np.float32),
        "h77g_grav_base_step": np.where(foot, grav_base_step, 0.0).astype(np.float32),
        "h77g_grav_over_mag_decoupling": np.where(foot, decouple, 0.0).astype(np.float32),
        "h77g_mt_claycap_margin": np.where(foot, mt_margin, 0.0).astype(np.float32),
        "h77g_transtension_index": np.where(foot, transtension, 0.0).astype(np.float32),
        "h77g_declustered_swarm_frac": np.where(foot, swarm_frac, 0.0).astype(np.float32),
        "h77g_basement_saddle": np.where(foot, base_saddle, 0.0).astype(np.float32),
        "h77g_composite_surface": np.where(foot, composite, 0.0).astype(np.float32),
    }
    diag = {
        nm: {
            "in_footprint_mean": float(arr[foot].mean()),
            "in_footprint_std": float(arr[foot].std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "nan_count": int(np.isnan(arr).sum()),
        }
        for nm, arr in out.items()
    }
    return out, diag
