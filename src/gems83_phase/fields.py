"""H83-PHASE label-free magnetic phase-congruency buried-contact fields.

Physical signature: buried basement offsets under alluvial cover produce
magnetic contacts with weak amplitude but stable phase across wavelengths.
Amplitude-independent phase congruency detects them where Hessian/gradient
magnitude thresholds miss.

Inputs: 5 magnetic bands from training_features (1 mag_anom, 2 rtp, 3 tmi_hg, 9 tmi_vg, 14 tmi)
Operators: Gaussian derivatives at 1.5, 3.0, 6.0 px (150/300/600 m at 100 m grid)
  Phase vectors exp(j*theta) per scale, resultant length R, amplitude-weighted persistence
  Structure-tensor coherence at 3 px gates strike-persistence
All deterministic, label-free, footprint-ranked tie-averaged.
No catalogue, no prior raster enters.

SENT sentinel handling: training_features nodata is -3.4028235e38; we treat < -1e30 as bad.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

SENT = -3.4028234663852886e38
H83_PHASE_NAMES = [
    "phase_rtp",
    "phase_tmi",
    "phase_mag",
    "phase_gradhg",
    "phase_gradvg",
    "phase_stack",
    "phase_coherence",
]

def clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    bad = (~np.isfinite(a)) | (a < np.float32(-1e30)) | (a > np.float32(1e30)) | (a < np.float32(SENT * 0.5))
    if bad.any():
        # fill with footprint median later; for now mark NaN
        out = a.copy()
        out[bad] = np.nan
        return out
    return a

def fill_nearest(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a.astype(np.float32)
    if bad.all():
        raise ValueError("cannot nearest-fill all-NaN field")
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float32)

def rank01_tieavg(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, np.float32)
    v = np.asarray(a, dtype=np.float64)[mask]
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return out
    r = np.zeros(v.size, np.float64)
    r[finite] = rankdata(v[finite], method="average") / float(finite.sum())
    out[mask] = r.astype(np.float32)
    return out

def grad_at_scale(a: np.ndarray, sigma: float):
    """Gaussian derivative gradients gx, gy at sigma."""
    gx = ndimage.gaussian_filter(a, sigma, order=(0,1), mode="nearest")
    gy = ndimage.gaussian_filter(a, sigma, order=(1,0), mode="nearest")
    return gx.astype(np.float32), gy.astype(np.float32)

def phase_congruency_single_band(band: np.ndarray, footprint: np.ndarray, scales=(1.5, 3.0, 6.0)) -> np.ndarray:
    """
    For one magnetic band, compute amplitude-weighted phase congruency.
    Theta per scale = atan2(gy,gx), amp = hypot(gx,gy)
    Phase congruency: R = |sum exp(j theta_k)| / K ; amp-weighted persistence = sum amp_k * R_k_local? Simplified:
    PC = (sum_k amp_k * cos(theta_k - theta_mean)) / sum amp_k approx = resultant length weighted by amp.
    We compute per-pixel:
      - gather theta_k, amp_k across scales
      - compute vector sum of phase unit vectors: sum_cos = sum cos(theta_k), sum_sin = sum sin(theta_k)
      - R = hypot(sum_cos, sum_sin) / K  (0..1)  -> phase alignment
      - mean_amp = mean amp_k
      - PC = R * (mean_amp / (mean_amp + 1e-3))  # soft threshold so weak-amp noise not amplified
      Actually we want amplitude-independent: R alone is amplitude independent, but we weight by mean_amp softly to suppress pure noise.
    """
    K = len(scales)
    thetas = []
    amps = []
    for sigma in scales:
        gx, gy = grad_at_scale(band, sigma)
        theta = np.arctan2(gy, gx).astype(np.float32)
        amp = np.hypot(gx, gy).astype(np.float32)
        thetas.append(theta)
        amps.append(amp)
    # stack
    cos_sum = np.zeros(band.shape, np.float32)
    sin_sum = np.zeros(band.shape, np.float32)
    for th in thetas:
        cos_sum += np.cos(th)
        sin_sum += np.sin(th)
    R = np.hypot(cos_sum, sin_sum) / float(K)  # 0..1
    mean_amp = np.mean(np.stack(amps, axis=0), axis=0).astype(np.float32)
    # Soft amplitude gate: suppress very low amp noise but keep weak contacts
    # Use rank-independent: normalize mean_amp by its footprint 90th percentile
    vals = mean_amp[footprint]
    p90 = float(np.percentile(vals[np.isfinite(vals)], 90)) if vals.size else 1.0
    soft = np.clip(mean_amp / (p90 + 1e-6), 0, 1).astype(np.float32)
    # Phase congruency strength: R * soft^0.5  (balance amplitude independence vs noise)
    pc = (R * np.sqrt(np.clip(soft, 0, 1))).astype(np.float32)
    return pc

def coherence_field(band: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    gx = ndimage.gaussian_filter(band, sigma, order=(0,1), mode="nearest")
    gy = ndimage.gaussian_filter(band, sigma, order=(1,0), mode="nearest")
    jxx = ndimage.gaussian_filter(gx*gx, sigma, mode="nearest")
    jyy = ndimage.gaussian_filter(gy*gy, sigma, mode="nearest")
    jxy = ndimage.gaussian_filter(gx*gy, sigma, mode="nearest")
    tr = jxx + jyy + 1e-12
    coh = 2.0 * np.sqrt(np.maximum(((jxx - jyy)*0.5)**2 + jxy**2, 0.0)) / tr
    return np.clip(coh, 0, 1).astype(np.float32)

def build_phase_fields(bands: dict[str, np.ndarray], footprint: np.ndarray) -> dict[str, np.ndarray]:
    """
    bands: dict with keys mag_anom, rtp, tmi_hg, tmi_vg, tmi, tc
    Each is 3730x3292 float32 already cleaned/filled.
    """
    foot = np.asarray(footprint, bool)
    pcs = {}
    for key in ["rtp", "tmi", "mag_anom", "tmi_hg", "tmi_vg"]:
        if key not in bands:
            continue
        arr = bands[key]
        # Fill NaNs via nearest
        arr_filled = fill_nearest(arr)
        pc = phase_congruency_single_band(arr_filled, foot)
        pcs[key] = pc
    # Map to output names
    out = {}
    out["phase_rtp"] = pcs.get("rtp", np.zeros_like(next(iter(pcs.values()))))
    out["phase_tmi"] = pcs.get("tmi", np.zeros_like(out["phase_rtp"]))
    out["phase_mag"] = pcs.get("mag_anom", np.zeros_like(out["phase_rtp"]))
    out["phase_gradhg"] = pcs.get("tmi_hg", np.zeros_like(out["phase_rtp"]))
    out["phase_gradvg"] = pcs.get("tmi_vg", np.zeros_like(out["phase_rtp"]))
    # Stack mean rank: cross-band phase persistence
    stack_vals = np.stack([v for v in pcs.values()], axis=0) if pcs else np.zeros((1, *foot.shape), dtype=np.float32)
    stack_mean = np.mean(stack_vals, axis=0).astype(np.float32)
    # Coherence gate from rtp (most diagnostic)
    coh = coherence_field(fill_nearest(bands["rtp"]) if "rtp" in bands else stack_mean, sigma=3.0)
    # Final stack field: sqrt(rank(pc_stack) * rank(coh)) -> strike-persistent phase ridges
    rank_pc = rank01_tieavg(stack_mean, foot)
    rank_coh = rank01_tieavg(coh, foot)
    phase_stack = np.sqrt(np.maximum(rank_pc * (0.3 + 0.7*rank_coh), 0)).astype(np.float32)
    out["phase_stack"] = np.where(foot, phase_stack, 0).astype(np.float32)
    out["phase_coherence"] = np.where(foot, coh, 0).astype(np.float32)
    # Ensure [0,1] and finite
    for name, arr in out.items():
        if not np.isfinite(arr).all() or arr.min() < -1e-6 or arr.max() > 1.0+1e-6:
            raise ValueError(f"H83 field {name} violates [0,1] finite: min {arr.min()} max {arr.max()} finite {np.isfinite(arr).all()}")
        out[name] = np.clip(arr, 0, 1).astype(np.float32)
    return out
