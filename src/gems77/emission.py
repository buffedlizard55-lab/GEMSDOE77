"""Emitters — turning a field into a legal binary raster.
Mirrors GEMSDOE32 logic with our own implementation.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

def topk_mask(field: np.ndarray, n: int, footprint: np.ndarray) -> np.ndarray:
    f = field.copy().astype(np.float32)
    f[~footprint] = -np.inf
    flat = f.ravel()
    # argpartition to get top n
    if n >= int(footprint.sum()):
        out = footprint.copy()
        return out
    idx = np.argpartition(flat, -n)[-n:]
    out = np.zeros(flat.size, bool)
    out[idx] = True
    return out.reshape(field.shape)

def poisson_thin(support: np.ndarray, min_dist: float = 2.8) -> np.ndarray:
    """Raster-order Poisson disk thinning (incumbent rule for comparison)."""
    ys, xs = np.nonzero(support)
    if ys.size == 0:
        return np.zeros(support.shape, bool)
    r2 = min_dist*min_dist
    cell = max(1.0, float(min_dist))
    grid = {}
    out = np.zeros(support.shape, bool)
    for i in range(ys.size):
        y,x = int(ys[i]), int(xs[i])
        cy, cx = int(y // cell), int(x // cell)
        ok=True
        for gy in range(cy-2, cy+3):
            for gx in range(cx-2, cx+3):
                for (ky,kx) in grid.get((gy,gx),()):
                    dy, dx = y-ky, x-kx
                    if dy*dy+dx*dx < r2:
                        ok=False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            out[y,x]=True
            grid.setdefault((cy,cx),[]).append((y,x))
    return out

def greedy_maxcov(field: np.ndarray, n: int, footprint: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """Greedy maximum coverage of the field's high-response region.
    Sorts by field value descending, then keeps pixel if it adds new coverage
    under triangular kernel (approximated by Euclidean exclusion of radius/2 to avoid redundancy).
    Simplified: field-sorted Poisson disk with min_dist = radius.
    """
    f = field.copy()
    f[~footprint] = -np.inf
    # sort indices by field descending
    flat = f.ravel()
    order = np.argsort(flat)[::-1]
    # only consider finite and positive field
    out = np.zeros(flat.size, bool)
    # grid for fast distance check
    cell = max(1.0, radius)
    H,W = field.shape
    grid = {}
    r2 = radius*radius
    count=0
    for idx in order:
        if count >= n:
            break
        if not np.isfinite(flat[idx]) or flat[idx] <= 0:
            # still allow if we need to fill? break if field is zeros everywhere? Continue to allow lower
            if flat[idx]==-np.inf:
                break
        y, x = divmod(int(idx), W)
        if not footprint[y,x]:
            continue
        cy, cx = int(y//cell), int(x//cell)
        ok=True
        for gy in range(cy-2, cy+3):
            for gx in range(cx-2, cx+3):
                for (ky,kx) in grid.get((gy,gx),()):
                    dy,dx = y-ky, x-kx
                    if dy*dy+dx*dx < r2:
                        ok=False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            out[idx]=True
            grid.setdefault((cy,cx),[]).append((y,x))
            count+=1
    return out.reshape(field.shape)

def buffer_mask_no_catalogue(labels: np.ndarray, footprint: np.ndarray, buffer_px: int = 2) -> np.ndarray:
    """Mask of pixels within buffer_px of any catalogue fault (Euclidean)."""
    # distance to catalogue
    # invert: distance to nearest fault pixel
    if not labels.any():
        return np.zeros(labels.shape, bool)
    d = distance_transform_edt(~labels)
    return d <= buffer_px

def mask_catalogue_and_buffer(emission: np.ndarray, labels: np.ndarray, footprint: np.ndarray, buffer_px: int = 2) -> np.ndarray:
    """Ensure emission has zero pixels on catalogue or within buffer."""
    # distance to catalogue
    if labels.any():
        d = distance_transform_edt(~labels)
        forbidden = d <= buffer_px
        emission = emission & ~forbidden
    # also ensure inside footprint
    emission = emission & footprint
    return emission
