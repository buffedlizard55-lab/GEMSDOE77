#!/usr/bin/env python3
"""Build the D77 lane submission: blind-distance band + metric-aware greedy dots.

Single-method lane paragraph (preregistered separately):

  From the locally bridged catalogue only: compute Euclidean distance to the
  nearest mapped fault; restrict emission to the blind band 250 m-1 000 m from
  the catalogue (beyond the H33 flank prune at <= 200 m, targeting faults the
  catalogue misses); weight each allowed cell by a triangular kernel peaking at
  600 m; place dots with the shared gems57.emit.greedy_allocate under the exact
  two-sided DTI bar (alpha = 0.2, beta = 0.8); write the dot field with the
  shared gems57.submission_writer (float32, [0, 1], EPSG:32611, 3730 x 3292,
  official transform, all finite, zeros outside the footprint, nothing on the
  mapped catalogue). Score on the shared gems57.holdout hide-and-recover
  (whole fault segments withheld per buffered quadrant fold; catalogue-derived
  feature built from visible faults only; visible faults masked pixel-exactly;
  pooled DTI; bootstrap 95% CI). Check uniqueness against every locally
  accessible registry raster (Spearman |rho| <= 0.90 on the placement surface;
  dots within 3 px <= 0.70 of any one raster on the final dots).

This lane is new work: it is not H77 (stopped, negative per AGENTS.md), it is
not a copy of any published file, and it is built only from data committed in
this repository (data/bridge/). No DrivenData auth is required.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from gems57.grid import load_grid  # noqa: E402
from gems57.holdout import build_holdout, evaluate  # noqa: E402
from gems57.emit import greedy_allocate  # noqa: E402
from gems57.metric import ALPHA, BETA, RADIUS_PX  # noqa: E402
from gems57.validate import validate  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems57.grid import TRANSFORM, EPSG, HEIGHT, WIDTH  # noqa: E402

DATA = ROOT / "data" / "bridge"
OUT = ROOT / "docs" / "downloads"
EVID = ROOT / "evidence"
STAMP = "20261021"
RNG_SEED = 77

# --- Lane parameters (frozen before running) --------------------------------
BAND_LO_PX = 2.5          # 250 m: beyond H33's <= 200 m flank prune
BAND_HI_PX = 10.0         # 1 000 m: blind structures beyond the mapped flank
BAND_PEAK_PX = 6.0        # 600 m: one kernel radius, max expected credit
K_TRUTH_GUESS = 12_000    # family lattice-calibrated hidden-truth order (7.9k-12.7k px)
MAX_DOTS = 60_000
FLOOR = 0.02
N_BOOT = 2000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_posterior(catalogue: np.ndarray) -> np.ndarray:
    """Posterior p(x) in [0,1]: triangular blind band around the catalogue."""
    d = ndi.distance_transform_edt(~catalogue).astype(np.float32)  # px
    band = (d >= BAND_LO_PX) & (d <= BAND_HI_PX)
    p = np.zeros_like(d)
    p[band] = np.clip(1.0 - np.abs(d[band] - BAND_PEAK_PX) / 4.0, 0.0, 1.0).astype(np.float32)
    return p


def rank_spearman(a: np.ndarray, b: np.ndarray, footprint: np.ndarray):
    """Spearman rho of two rasters on the shared finite footprint."""
    m = footprint & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 100:
        return None
    ra = a[m].astype(np.float64)
    rb = b[m].astype(np.float64)
    if np.ptp(ra) <= 1e-12 or np.ptp(rb) <= 1e-12:
        return None
    return float(spearmanr(ra, rb).statistic)


def dot_overlap(mine: np.ndarray, theirs: np.ndarray, radius_px: float = RADIUS_PX) -> float:
    """Fraction of my dots within radius_px of any of their dots."""
    m = np.asarray(mine, bool)
    t = np.asarray(theirs, bool)
    if not m.any() or not t.any():
        return 0.0
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    struct = yy * yy + xx * xx <= radius_px * radius_px
    near = ndi.binary_dilation(t, structure=struct)
    return float(near[m].mean())


def bootstrap_ci(cells: list, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """Resample holdout cells; pooled DTI is additive over cells."""
    rng = np.random.default_rng(seed)
    rows = [(float(c["tp"]), float(c["fp"]), float(c["fn"])) for c in cells]
    out = np.empty(n_boot)
    n = len(rows)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tp = sum(rows[i][0] for i in idx)
        fp = sum(rows[i][1] for i in idx)
        fn = sum(rows[i][2] for i in idx)
        out[b] = tp / (tp + ALPHA * fp + BETA * fn + 1e-12)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    t0 = time.time()
    grid = load_grid(DATA)
    footprint, catalogue = grid.footprint, grid.catalogue
    print(f"grid {HEIGHT}x{WIDTH} EPSG:{EPSG} footprint={int(footprint.sum()):,} "
          f"catalogue={int(catalogue.sum()):,} px")

    # --- Surface (placement) --------------------------------------------------
    p = build_posterior(catalogue)
    allowed = footprint & ~catalogue & (p > 0)
    print(f"posterior: allowed={int(allowed.sum()):,} cells, max p={float(p.max()):.3f}")

    # --- Placement-stage uniqueness vs locally accessible registry rasters ----
    import rasterio  # noqa: E402

    registry_rows = []
    worst_rho = None
    for tif in sorted(OUT.glob("*.tif")):
        try:
            with rasterio.open(tif) as ds:
                other = ds.read(1)
        except Exception as exc:  # diagnostic
            registry_rows.append(dict(file=tif.name, error=str(exc)))
            continue
        rho = rank_spearman(p, other, footprint)
        registry_rows.append(dict(file=tif.name, surface_spearman=rho,
                                  shape=[int(other.shape[0]), int(other.shape[1])],
                                  finite=int(np.isfinite(other).sum())))
        if rho is not None and (worst_rho is None or abs(rho) > abs(worst_rho[1])):
            worst_rho = (tif.name, float(rho))
    print(f"surface worst |rho| = {worst_rho}")

    # --- Dots -----------------------------------------------------------------
    alloc = greedy_allocate(p, allowed, K_TRUTH_GUESS, floor=FLOOR, max_dots=MAX_DOTS)
    dots = alloc.emitted
    print(f"dots={int(dots.sum())} rounds={alloc.rounds}")
    assert not (dots & ~footprint).any(), "dots outside footprint"
    assert not (dots & catalogue).any(), "dots on catalogue"

    # --- Final-dots uniqueness -----------------------------------------------
    # A whole-footprint *soft* registry surface (e.g. a continuous curvature
    # field) dilates to cover the footprint, which makes the 3-px dot-overlap
    # test degenerate against it: that is the repo's known space-filling
    # witness condition (Policy I17), reported separately, never silently
    # exempted. Drift is measured against real dot fields.
    dot_rows = []
    worst_ov = None
    worst_ov_dense = None
    for tif in sorted(OUT.glob("*.tif")):
        try:
            with rasterio.open(tif) as ds:
                other = ds.read(1)
        except Exception as exc:
            dot_rows.append(dict(file=tif.name, error=str(exc)))
            continue
        pos = np.isfinite(other) & (other > 0)
        support_frac = float(pos.mean())
        ov = dot_overlap(dots, other)
        row = dict(file=tif.name, dots_within_3px=float(ov),
                   positive_support_fraction=support_frac,
                   dense_space_filling=bool(support_frac > 0.5))
        dot_rows.append(row)
        if support_frac > 0.5:
            target = worst_ov_dense
        else:
            target = worst_ov
        if target is None or ov > target[1]:
            if support_frac > 0.5:
                worst_ov_dense = (tif.name, float(ov), support_frac)
            else:
                worst_ov = (tif.name, float(ov), support_frac)
    print(f"dots worst within-3px overlap (dot fields) = {worst_ov}")
    print(f"dots worst within-3px overlap (dense surfaces) = {worst_ov_dense}")

    # Binary dot rasters only: continuous curvature-style surfaces (like the
    # selective-U research TIFF) dilate to broad support, which makes their
    # 3-px dot overlap degenerate; they are reported as dense witnesses, not
    # as dot-set identity (same space-filling-witness treatment as Policy I17).
    def _is_binary(arr):
        vals = arr[np.isfinite(arr)]
        return vals.size and float(np.mean((vals == 0) | (vals == 1))) > 0.95

    binary_worst = None
    for r in dot_rows:
        if r.get("dense_space_filling") or "error" in r:
            continue
        try:
            with rasterio.open(OUT / r["file"]) as ds:
                arr = ds.read(1)
        except Exception:
            continue
        if _is_binary(arr):
            if binary_worst is None or r["dots_within_3px"] > binary_worst[1]:
                binary_worst = (r["file"], float(r["dots_within_3px"]))
    print(f"dots worst within-3px overlap (binary dot rasters) = {binary_worst}")

    drift = (worst_rho is not None and abs(worst_rho[1]) > 0.90) or \
        (worst_ov is not None and worst_ov[1] > 0.70)
    print(f"LANE DRIFT (protocol stop condition): {drift} -- logged explicitly; "
          f"the session brief requires a generated TIF, so the build continues "
          f"with the drift flagged in the run card and on the site.")

    # --- Holdout --------------------------------------------------------------
    ctx = build_holdout(grid)
    res = evaluate(dots, ctx)
    cell_rows = [dict(tp=r["tp"], fp=r["fp"], fn=r["fn"]) for r in res["folds"].values()]
    lo, hi = bootstrap_ci(cell_rows)
    holdout = {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": "gems57-pooled-hide-v2",
        "dti": float(res["pooled_dti"]),
        "ci95": [lo, hi],
        "withheld_positive_pixels": int(res["pooled_n_truth"]),
        "tpw": float(res["pooled_tp"]), "fpw": float(res["pooled_fp"]), "fnw": float(res["pooled_fn"]),
        "coverage": float(res["pooled_coverage"]),
        "n_dots": int(res["emitted_pixels"]),
        "on_visible_catalogue": int(res["on_visible_catalogue"]),
        "per_fold_dti": res["per_fold_dti"],
    }
    print(f"HOLDOUT-DTI = {holdout['dti']:.5f} 95% CI [{lo:.5f}, {hi:.5f}] "
          f"withheld={holdout['withheld_positive_pixels']} dots={holdout['n_dots']}")

    # --- Leakage canary: distance band alone vs withheld truth ----------------
    from sklearn.metrics import roc_auc_score  # noqa: E402

    withheld = np.zeros(footprint.shape, bool)
    active_domain = np.zeros(footprint.shape, bool)
    for cell in ctx.cells:
        truth_full = np.zeros(footprint.shape, bool)
        tyx = cell.truth_yx
        truth_full[(tyx[0] + cell.bbox[0].start, tyx[1] + cell.bbox[1].start)] = True
        withheld |= truth_full
        full = np.zeros(footprint.shape, bool)
        full[cell.bbox] = cell.active
        active_domain |= full
    scored = active_domain
    y = withheld[scored]
    s = p[scored]
    if int(y.sum()) > 0 and int((~y).sum()) > 0:
        auc = float(roc_auc_score(y, s))
    else:
        auc = None
    print(f"leakage canary (distance band alone, AUC vs withheld): {auc}")
    canary = {"feature": "distance-to-catalogue blind band",
              "auc_vs_withheld": auc,
              "leakage": bool(auc is not None and auc > 0.90)}
    if canary["leakage"]:
        print("STOP: leakage canary AUC > 0.90")
        return 3

    # --- Write submission (fail-closed shared writer) -------------------------
    pred = np.zeros((HEIGHT, WIDTH), np.float32)
    pred[dots] = 1.0
    digest = hashlib.sha256(pred.tobytes()).hexdigest()
    name = f"gems77-d77-blind-band-{int(dots.sum())}dots-{STAMP}-{digest[:8]}-zeros"
    note = (f"GEMSDOE77 D77 blind band 250m-1km off catalogue; greedy dots, {int(dots.sum())} dots, "
            f"0 on catalogue; HOLDOUT-DTI {holdout['dti']:.4f} [{lo:.4f},{hi:.4f}]")
    assert len(name) <= 140 and len(note) <= 140, (len(name), len(note))

    tif_path = OUT / f"{name}.tif"
    receipt = write_submission(tif_path, pred, DATA / "sample_submission.tif", footprint,
                               note=note, name=name, catalogue=catalogue)
    print(f"wrote {tif_path} sha256={receipt['sha256']} bytes={receipt['bytes']}")

    # --- Re-validate the on-disk file ----------------------------------------
    val = validate(tif_path, footprint, catalogue)
    print(f"validator: all_checks_passed={val['all_checks_passed']} "
          f"min={val['min']} max={val['max']} nan={val['n_nan']} "
          f"positive={val['emitted_positive_pixels']}")

    # --- Run card -------------------------------------------------------------
    dense_list = [r for r in dot_rows if r.get("dense_space_filling")]
    dot_fields = [r for r in dot_rows if not r.get("dense_space_filling") and "error" not in r]
    worst_dot_field = max(dot_fields, key=lambda r: r["dots_within_3px"], default=None)
    card = {
        "schema": "gems-parallel-run-card-v1",
        "lane_repo": "buffedlizard55-lab/GEMSDOE77",
        "branch": "arena/95c2d269-gemsdoe77",
        "lane": "D77 blind-distance band + metric-aware greedy dots",
        "lane_file": str(tif_path.relative_to(ROOT)),
        "hypothesis": ("Hidden geothermal faults missed by the USGS/INGENIOUS catalogue lie in the blind "
                       "band 250 m-1 000 m from mapped traces (beyond the H33 flank prune), not on the "
                       "catalogue itself; dots spaced ~one kernel radius apart pay the minimum false-positive "
                       "tax under DTI (alpha 0.2, beta 0.8)."),
        "mechanism": ("Euclidean distance to nearest catalogue fault from visible faults only; triangular "
                      "posterior peaking at 600 m; exact two-sided DTI bar greedy allocation; zeros outside "
                      "the footprint and on the mapped catalogue."),
        "named_non_fault_mimic": ("Pluvial Lake Lahontan shoreline benches and alluvial fan toes, which also "
                                  "sit at moderate distance from range-front faults without a seismogenic fault."),
        "holdout_dti": holdout,
        "leakage_canary": canary,
        "registry_comparison": {
            "scope": "locally accessible registry subset: TIFFs published in this repository (docs/downloads/); "
                     "the full 812-raster inventory is not accessible from the sandbox",
            "registry_rasters_checked": len(registry_rows),
            "surface_worst_spearman": worst_rho,
            "rho_limit": 0.90,
            "dots_worst_overlap_3px": worst_ov,
            "dots_worst_overlap_3px_binary_dot_rasters": binary_worst,
            "dots_worst_overlap_3px_dense_surfaces": worst_ov_dense,
            "overlap_limit": 0.70,
            "space_filling_dense_rasters": dense_list,
            "lane_drift": drift,
            "surface_rows": registry_rows,
            "dot_rows": dot_rows,
        },
        "validator": val["checks"],
        "validator_all_checks_passed": bool(val["all_checks_passed"]),
        "raster": {"file": tif_path.name, "sha256": receipt["sha256"], "bytes": receipt["bytes"],
                   "dots": int(dots.sum())},
        "submission_name": name,
        "submission_note": note,
        "note_chars": len(note),
        "zip_file": receipt["zip_file"], "zip_sha256": receipt["zip_sha256"],
        "receipt_file": receipt["file"],
        "ok_to_download": True,
        "ok_to_submit": False,
        "submission_status": {
            "ok_to_download_for_inspection": True,
            "ok_to_submit": ("Format-valid and not drifting into another lane vs dot fields; NOT organizer-accepted "
                             "and not selector-approved. The literal uniqueness rule fails closed against dense "
                             "space-filling witness rasters (Policy I17) -- the same condition as every lane in "
                             "this repository. Spending a weekly slot is a separate selector decision."),
            "literal_protocol_uniqueness": "FAIL (dense space-filling witness rasters; Policy I17)",
            "informative_uniqueness": ("PASS" if (worst_rho is None or abs(worst_rho[1]) <= 0.90)
                                       and (worst_dot_field is None or worst_dot_field["dots_within_3px"] <= 0.70)
                                       else "FAIL"),
            "portal_acceptance": "NOT ORGANIZER-CONFIRMED",
            "selector_approved": False,
        },
        "verdict": "negative",
        "verdict_reason": ("No organizer score exists; local holdout is measured, not projected. Not approved for a "
                           "weekly slot; delivered as a format-valid, locally non-drifting, downloadable research TIFF."),
        "champion_reference": {"submission": "GEMSDOE32 h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros",
                               "owner_reported_score": 0.2778, "note": "OWNER-REPORT; no organizer receipt inspected"},
        "budget": {"experiments_used": 1, "limit": 3, "wall_seconds": round(time.time() - t0, 1),
                   "weekly_slots_used_by_this_run": 0},
    }
    card_path = EVID / "run_card_d77.json"
    card_path.write_text(json.dumps(card, indent=1, allow_nan=False) + "\n")
    print(f"wrote {card_path}")
    print(f"verdict={card['verdict']} ok_to_download={card['ok_to_download']} ok_to_submit={card['ok_to_submit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
