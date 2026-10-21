"""Label-free H77E fields from the cached 19-band competition stack.

Physics (see research/h77e_hypotheses.md):
- Fourier upward continuation exp(-|k| h), h = 0/200/400/800 m, of RTP
  (band 2) and isostatic gravity (band 13). At each height the analytic-signal
  edge strength sqrt(Tx^2+Ty^2+Tz^2) is 99th-percentile normalised inside the
  footprint; persistence = mean across heights, deep = 800 m edge.
- Euler deconvolution (Reid et al. 1990, Geophysics 55:80-91,
  doi:10.1190/1.1442774), structural index 0 (contact), 7 px windows on a
  stride-2 lattice, solved from normal equations built with integral images.
  Solutions with depth 50-1500 m and low relative residual are rasterised as
  density, mean depth and depth coherence.
- Texture block: multi-scale RTP roughness suppression and its anisotropy
  (magnetite-destruction / damage-zone smoothing proxy).

No catalogue, label, or prior-submission raster is read here. Bands are
identified by their embedded GDAL descriptions in scripts, never by the
band-6 tag (band 6 is radiometric total count, not tilt; see
evidence/band6_resolution.json).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

HEIGHTS_M = (0.0, 200.0, 400.0, 800.0)
CELL_M = 100.0

NEW_NAMES = [
    "ucp_mag_persist", "ucp_mag_deep800",
    "ucp_grav_persist", "ucp_grav_deep800",
    "ucp_cross_min",
    "euler_density", "euler_depth_coh", "euler_depth_mean",
    "tex_supp", "tex_aniso", "tex_supp_aniso",
]


def nearest_fill(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid cells with the nearest valid value (EDT indices)."""
    if valid.all():
        return a.astype(np.float32, copy=False)
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float32, copy=False)


def _kgrid(shape):
    h, w = shape
    ky = 2.0 * np.pi * np.fft.fftfreq(h)[:, None].astype(np.float32)
    kx = 2.0 * np.pi * np.fft.rfftfreq(w)[None, :].astype(np.float32)
    kmag = np.sqrt(kx * kx + ky * ky)
    return kx, ky, kmag


def upward_derivatives(filled: np.ndarray, heights_m=HEIGHTS_M):
    """Yield (Tx, Ty, Tz_up, T) at each continuation height (per-pixel units).

    Coordinates: x = column (east), y = row (south), z = up. A harmonic field
    above its sources obeys T(k, z) = T(k, 0) * exp(-|k| z) for z > 0 upward,
    hence Tz_up = IFFT(-|k| * That) and upward continuation to height h
    multiplies by exp(-|k| h). Derivative magnitudes are invariant to the
    row-south reflection; Euler is solved consistently in (x, y_row, z_up)
    so source depth is -z0.
    """
    F = np.fft.rfft2(filled.astype(np.float32))
    kx, ky, kmag = _kgrid(filled.shape)
    out = []
    for hm in heights_m:
        att = np.exp(-kmag * (float(hm) / CELL_M)).astype(np.complex64)
        Fa = F * att
        tx = np.fft.irfft2(Fa * (1j * kx), s=filled.shape).astype(np.float32)
        ty = np.fft.irfft2(Fa * (1j * ky), s=filled.shape).astype(np.float32)
        tz = np.fft.irfft2(Fa * (-kmag), s=filled.shape).astype(np.float32)
        t = np.fft.irfft2(Fa, s=filled.shape).astype(np.float32)
        out.append((tx, ty, tz, t))
    return out


def _norm99(edge: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    v = edge[footprint]
    q = float(np.quantile(v, 0.99)) if v.size else 1.0
    if not np.isfinite(q) or q <= 0:
        q = 1.0
    return np.clip(np.log1p(edge) / np.log1p(q), 0.0, 1.0).astype(np.float32)


def analytic_edges(filled: np.ndarray, footprint: np.ndarray, heights_m=HEIGHTS_M):
    """Percentile-normalised analytic-signal edge at each height."""
    edges = []
    native = None
    for tx, ty, tz, t in upward_derivatives(filled, heights_m):
        e = np.sqrt(tx * tx + ty * ty + tz * tz)
        edges.append(_norm99(e, footprint))
        if native is None:
            native = (tx, ty, tz, t)
    return edges, native


def _box_sum(integral: np.ndarray, cy: np.ndarray, cx: np.ndarray, half: int):
    """Sum over (2*half+1)^2 windows centred at (cy, cx) from an integral image."""
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def euler_solve_windows(tx, ty, tz, t, centers_y, centers_x, half=3, si=0.0):
    """Vectorised windowed Euler least squares via normal equations.

    Solves, per window, for (x0, y0, z0, B) in
        -Tx*x0 - Ty*y0 - Tz*z0 + SI*B = -SI*T - x*Tx - y*Ty   (z = 0 plane)
    with x, y local pixel coordinates relative to the window centre.
    Returns (x0_local, y0_local, z0, rel_residual) per centre.
    """
    h, w = t.shape
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    xf, yf = xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)
    fields = {"tx": tx.astype(np.float64), "ty": ty.astype(np.float64),
              "tz": tz.astype(np.float64), "t": t.astype(np.float64)}
    # Local-coordinate weighted fields: x*Tx etc. as full grids is memory-heavy;
    # instead accumulate window sums of Tx, Tx*x_off is handled by expanding
    # (x_c + dx): sum(Tx * x) = x_c * sum(Tx) + sum(Tx * dx). The dx-weighted
    # sums are translation-variant, so we compute them with fixed-offset
    # shifted box sums: sum over window of Tx(p) * dx(p) where dx depends only
    # on offset within the window. We therefore build, for each field, the
    # plain box sum and two first-moment box sums via three integral images
    # of Tx, Tx*dx_kernel is not separable per-window... use direct shifted
    # sums instead (window is only 7x7): gather 49 shifted views.
    n = len(centers_y)
    # Unknowns are (x0, y0, z0) plus a background B only when SI != 0:
    # with SI = 0 the term SI*B vanishes from Euler's equation, so B is
    # unidentifiable and a 4-unknown system is exactly singular.
    nu = 3 if float(si) == 0.0 else 4
    ata = np.zeros((n, nu, nu), dtype=np.float64)
    atb = np.zeros((n, nu), dtype=np.float64)
    b2 = np.zeros(n, dtype=np.float64)
    # coefficient vectors per pixel: c = [-Tx, -Ty, -Tz, SI]; rhs b = -SI*T - x*Tx - y*Ty
    # x, y here are ABSOLUTE pixel coordinates; recentering is done by the
    # solver caller through local offsets: we use absolute coords and let the
    # intercept-like conditioning be handled in float64 (grid <= 3730 px).
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            ys = centers_y + dy
            xs = centers_x + dx
            vx = fields["tx"][ys, xs]
            vy = fields["ty"][ys, xs]
            vz = fields["tz"][ys, xs]
            vt = fields["t"][ys, xs]
            c0, c1, c2 = -vx, -vy, -vz
            b = -float(si) * vt - xs * vx - ys * vy
            cols = (c0, c1, c2, np.full_like(vx, float(si)))[:nu]
            for i in range(nu):
                atb[:, i] += cols[i] * b
                for j in range(i, nu):
                    ata[:, i, j] += cols[i] * cols[j]
            b2 += b * b
    for i in range(nu):
        for j in range(i):
            ata[:, i, j] = ata[:, j, i]
    # Batched solve; singular windows are solved by lstsq and flagged by a
    # large residual / non-finite check below rather than aborting the batch.
    try:
        sol = np.linalg.solve(ata, atb[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(ata) @ atb[..., None]
        sol = sol[..., 0]
    rss = b2 - 2.0 * np.einsum("ni,ni->n", sol, atb) + np.einsum("ni,nij,nj->n", sol, ata, sol)
    rss = np.maximum(rss, 0.0)
    scale = np.maximum(b2, 1e-300)
    rel = np.sqrt(rss / scale)
    ok = np.isfinite(sol).all(axis=1)
    return sol, rel, ok


def euler_fields(tx, ty, tz, t, footprint, window_px=7, stride_px=2,
                 depth_lo_m=50.0, depth_hi_m=1500.0, max_rel_resid=0.35):
    """Rasterise Euler (SI=0) solutions: density, mean depth, coherence."""
    h, w = t.shape
    half = window_px // 2
    cy = np.arange(half + 1, h - half - 1, stride_px)
    cx = np.arange(half + 1, w - half - 1, stride_px)
    CY, CX = np.meshgrid(cy, cx, indexing="ij")
    cyf, cxf = CY.ravel(), CX.ravel()
    inside = footprint[cyf, cxf]
    cyf, cxf = cyf[inside], cxf[inside]
    sol, rel, ok = euler_solve_windows(tx, ty, tz, t, cyf, cxf, half=half, si=0.0)
    x0, y0, z0 = sol[:, 0], sol[:, 1], sol[:, 2]
    depth_m = -z0 * CELL_M
    keep = (ok & (rel <= max_rel_resid)
            & (depth_m >= depth_lo_m) & (depth_m <= depth_hi_m)
            & (np.abs(x0 - cxf) <= 4 * window_px) & (np.abs(y0 - cyf) <= 4 * window_px))
    dens = np.zeros((h, w), np.float32)
    dsum = np.zeros((h, w), np.float64)
    d2sum = np.zeros((h, w), np.float64)
    wsum = np.zeros((h, w), np.float64)
    ky_ = np.clip(np.rint(y0[keep]).astype(int), 0, h - 1)
    kx_ = np.clip(np.rint(x0[keep]).astype(int), 0, w - 1)
    wt = (1.0 / (1.0 + rel[keep])).astype(np.float64)
    np.add.at(dens, (ky_, kx_), wt.astype(np.float32))
    np.add.at(dsum, (ky_, kx_), wt * depth_m[keep])
    np.add.at(d2sum, (ky_, kx_), wt * depth_m[keep] ** 2)
    np.add.at(wsum, (ky_, kx_), wt)
    stats = dict(n_windows=int(len(cyf)), n_kept=int(keep.sum()),
                 kept_fraction=float(keep.mean()) if len(cyf) else 0.0)
    dens_s = ndi.gaussian_filter(dens, 1.5)
    wsum_s = ndi.gaussian_filter(wsum, 1.5)
    mean_s = np.where(wsum_s > 1e-9, ndi.gaussian_filter(dsum, 1.5) / np.maximum(wsum_s, 1e-9), 0.0)
    var_s = np.where(wsum_s > 1e-9,
                     ndi.gaussian_filter(d2sum, 1.5) / np.maximum(wsum_s, 1e-9) - mean_s ** 2, 0.0)
    std_s = np.sqrt(np.maximum(var_s, 0.0))
    coh = np.where(dens_s > 0, np.exp(-std_s / 300.0), 0.0).astype(np.float32)
    dens_n = np.clip(np.log1p(dens_s) / max(np.log1p(float(np.quantile(dens_s[footprint], 0.99))), 1e-9), 0, 1)
    mean_n = np.clip(mean_s / depth_hi_m, 0, 1).astype(np.float32)
    return dens_n.astype(np.float32), coh, mean_n, stats


def texture_fields(rtp_filled: np.ndarray, footprint: np.ndarray):
    """Roughness suppression + anisotropy of the roughness field."""
    def _std(a, size):
        m = ndi.uniform_filter(a, size=size)
        m2 = ndi.uniform_filter(a * a, size=size)
        return np.sqrt(np.maximum(m2 - m * m, 0.0))
    r5 = _std(rtp_filled, 5)
    r11 = _std(rtp_filled, 11)
    r51 = _std(rtp_filled, 51)
    supp = np.clip(1.0 - r11 / (r51 + 1e-6), 0.0, 1.0).astype(np.float32)
    # Structure tensor of the 11px roughness field -> coherence (anisotropy).
    gy, gx = np.gradient(r11)
    sxx = ndi.gaussian_filter(gx * gx, 3.0)
    syy = ndi.gaussian_filter(gy * gy, 3.0)
    sxy = ndi.gaussian_filter(gx * gy, 3.0)
    tr = sxx + syy
    disc = np.sqrt(np.maximum((sxx - syy) ** 2 + 4.0 * sxy * sxy, 0.0))
    aniso = np.where(tr > 1e-12, disc / np.maximum(tr, 1e-12), 0.0).astype(np.float32)
    return supp, aniso, (supp * aniso).astype(np.float32)


def build_new_fields(rtp: np.ndarray, grav: np.ndarray, footprint: np.ndarray):
    """Return (dict name->2D float32 field, diagnostics dict)."""
    rtp_f = nearest_fill(rtp, footprint & np.isfinite(rtp) & (rtp > -1e30))
    grav_f = nearest_fill(grav, footprint & np.isfinite(grav) & (grav > -1e30))
    mag_edges, mag_native = analytic_edges(rtp_f, footprint)
    grav_edges, _ = analytic_edges(grav_f, footprint)
    mag_persist = np.mean(mag_edges, axis=0).astype(np.float32)
    grav_persist = np.mean(grav_edges, axis=0).astype(np.float32)
    fields = {
        "ucp_mag_persist": mag_persist,
        "ucp_mag_deep800": mag_edges[-1],
        "ucp_grav_persist": grav_persist,
        "ucp_grav_deep800": grav_edges[-1],
        "ucp_cross_min": np.minimum(mag_persist, grav_persist).astype(np.float32),
    }
    tx, ty, tz, t = mag_native
    dens, coh, mean_d, stats = euler_fields(tx, ty, tz, t, footprint)
    fields["euler_density"] = dens
    fields["euler_depth_coh"] = coh
    fields["euler_depth_mean"] = mean_d
    supp, aniso, both = texture_fields(rtp_f, footprint)
    fields["tex_supp"] = supp
    fields["tex_aniso"] = aniso
    fields["tex_supp_aniso"] = both
    for k in fields:
        fields[k] = np.where(footprint, fields[k], 0.0).astype(np.float32)
    diag = {"euler": stats,
            "heights_m": list(HEIGHTS_M),
            "mag_edge_q99_raw": None}
    return fields, diag
