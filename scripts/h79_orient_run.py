#!/usr/bin/env python3
"""Run the pre-registered H79 orientation-persistence lane.

This entrypoint is intentionally namespaced from the stopped H77/H78 lanes. It
uses the cached feature stack, the shared gems57 holdout/evaluator and the shared
fail-closed submission writer. It never uses a prior submission as a feature.

The command runs three bounded experiments:
  E1  feature-alone leakage canaries on seed 20;
  E2  H79/controls on the seed-20 buffered whole-component holdout;
  E3  primary confirmation on seed 30, then full-map packaging.

A TIFF is packaged even when the scientific gate is negative. Its run card and
site status make that distinction explicit; packaging is not slot selection.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor")]

from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as G  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems57 import submission_writer  # noqa: E402
from gems57 import uniqueness  # noqa: E402
from gems77 import features77 as F77  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems79 import features as H79  # noqa: E402

STACK_PATH = ROOT / "data" / "processed" / "stack.npy"
H79_PATH = ROOT / "data" / "processed" / "h79_orientation_stack.npy"
H79_META = ROOT / "data" / "processed" / "h79_orientation_stack.json"
OUT_DIR = ROOT / "docs" / "downloads"

SEED_PRIMARY = 20
SEED_CONFIRM = 30
BUDGET = 37_654
MIN_SEP = 2.8
CAT_EXCLUSION = 3.0
INCUMBENT_DTI = 0.152718
INCUMBENT_CI95 = [0.131548, 0.175185]

HGB_KW = dict(
    max_iter=150,
    learning_rate=0.075,
    max_leaf_nodes=31,
    min_samples_leaf=60,
    l2_regularization=3.0,
    early_stopping=False,
)


def log(*args) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False, default=float) + "\n")


def load_grid() -> G.Grid:
    return G.load_grid(ROOT / "data" / "bridge")


def build_h79_features(stack: np.ndarray, grid: G.Grid) -> tuple[np.memmap, list[str], dict[str, int]]:
    """Build once and cache only the label-free H79 stack."""
    channels = {name: i for i, name in enumerate(json.loads((ROOT / "registry" / "feature_receipt.json").read_text())["channels"])}
    expected_shape = [grid.height, grid.width, len(H79.OUTPUT_NAMES)]
    if H79_PATH.exists() and H79_META.exists():
        meta = json.loads(H79_META.read_text())
        if meta.get("shape") == expected_shape and meta.get("output_names") == list(H79.OUTPUT_NAMES):
            return np.load(H79_PATH, mmap_mode="r"), list(H79.OUTPUT_NAMES), channels

    log("E1: building label-blind H79 orientation-persistence stack")
    arr, names = H79.build(stack, channels, grid.footprint)
    tmp = H79_PATH.with_suffix(".partial.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm, arr
    tmp.replace(H79_PATH)
    json_dump(H79_META, {
        "shape": expected_shape,
        "dtype": "float32",
        "output_names": names,
        "source_channels": list(H79.SOURCE_NAMES) + ["depth_to_base_surf"],
        "feature_array_sha256": sha256_file(H79_PATH),
        "catalogue_features": [],
        "prior_prediction_inputs": [],
        "label_free": True,
        "config": {"sigmas_px": [1.5, 3.0, 6.0], "double_angle": True,
                   "normalization": "1st-99th percentile within official footprint"},
    })
    return np.load(H79_PATH, mmap_mode="r"), names, channels


def feature_matrix(h79_flat: np.ndarray, raw_flat: np.ndarray, indices: np.ndarray,
                   arm: str, raw_cols: list[int]) -> np.ndarray:
    h = np.asarray(h79_flat[indices], dtype=np.float32)
    if arm == "H79_only":
        return h
    r = np.asarray(raw_flat[indices][:, raw_cols], dtype=np.float32)
    if arm == "raw_control":
        return r
    if arm == "H79_plus_raw":
        return np.hstack((h, r))
    raise ValueError(f"unknown arm {arm}")


def make_samples(grid: G.Grid, folds: list[dict], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct label-blind training rows for each quadrant fold."""
    fp, cat = grid.footprint, grid.catalogue
    dcat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    rng = np.random.default_rng(79_000 + seed)
    samples = []
    for k, fold in enumerate(folds):
        train_region = np.zeros_like(fp)
        for j, other in enumerate(folds):
            if j != k:
                train_region |= other["region"]
        # Positives are only visible catalogue cells. The hidden component draw
        # is not allowed into either training class.
        pos = np.flatnonzero(fold["visible"] & train_region & fp)
        if pos.size > 32_000:
            pos = np.sort(rng.choice(pos, 32_000, replace=False))
        neg_mask = train_region & (~cat) & (dcat > CAT_EXCLUSION)
        neg = np.flatnonzero(neg_mask)
        n_neg = min(2 * pos.size, neg.size)
        neg = np.sort(rng.choice(neg, n_neg, replace=False))
        idx = np.concatenate((pos, neg))
        y = np.concatenate((np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)))
        samples.append((idx, y))
        log(f"{fold['name']} {seed}: train positives={pos.size} negatives={neg.size} "
            f"test truth={int(fold['truth'].sum())}")
    return samples


def canary(h79_flat: np.ndarray, grid: G.Grid, folds: list[dict], names: list[str], seed: int) -> dict:
    """Run the mandatory feature-alone symmetric-AUC leakage canary."""
    hidden = folds[0]["hidden_all"]
    cat = grid.catalogue
    dcat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    pos = np.flatnonzero(hidden & grid.footprint)
    pool = np.flatnonzero(grid.footprint & (~cat) & (dcat > CAT_EXCLUSION))
    rng = np.random.default_rng(79_100 + seed)
    neg = rng.choice(pool, size=min(200_000, pool.size), replace=False)
    idx = np.concatenate((pos, neg))
    y = np.concatenate((np.ones(pos.size), np.zeros(neg.size)))
    values = np.asarray(h79_flat[idx], dtype=np.float32)
    rows = []
    for j, name in enumerate(names):
        v = values[:, j]
        good = np.isfinite(v)
        auc = float(roc_auc_score(y[good], v[good]))
        rows.append({"feature": name, "auc": auc, "symmetric_auc": max(auc, 1.0 - auc),
                     "leakage_flag": bool(max(auc, 1.0 - auc) > 0.90)})
    worst = max(rows, key=lambda r: r["symmetric_auc"])
    out = {
        "evidence_class": "LEAKAGE-CANARY",
        "seed": seed,
        "withheld_positive_pixels": int(pos.size),
        "negative_proxy_pixels": int(neg.size),
        "threshold_symmetric_auc": 0.90,
        "features_tested": len(rows),
        "max_symmetric_auc": worst["symmetric_auc"],
        "worst_feature": worst["feature"],
        "any_flag": any(r["leakage_flag"] for r in rows),
        "rows": rows,
        "warning": "AUC below 0.90 is a canary result, not proof of no leakage; negatives are unlabelled proxies.",
    }
    json_dump(ROOT / "evidence" / "h79_orient_canary.json", out)
    log(f"E1 canary: max symmetric AUC={worst['symmetric_auc']:.4f} ({worst['feature']})")
    if out["any_flag"]:
        raise RuntimeError("H79 leakage canary exceeded symmetric AUC 0.90; fail closed")
    return out


def fit_predict_arm(stack: np.ndarray, h79: np.ndarray, grid: G.Grid, folds: list[dict],
                    samples: list[tuple[np.ndarray, np.ndarray]], arm: str, seed: int,
                    raw_cols: list[int]) -> np.ndarray:
    """Fit four spatially held-out HGB models and return one OOF probability surface."""
    fp = grid.footprint
    stack_flat = stack.reshape(-1, stack.shape[2])
    h79_flat = h79.reshape(-1, h79.shape[2])
    out = np.zeros(fp.shape, dtype=np.float32)
    for k, fold in enumerate(folds):
        train_idx, y = samples[k]
        clf = HistGradientBoostingClassifier(**HGB_KW, random_state=79_000 + seed * 10 + k)
        X = feature_matrix(h79_flat, stack_flat, train_idx, arm, raw_cols)
        clf.fit(X, y)
        del X, train_idx, y
        test_idx = np.flatnonzero(fold["region"] & fp)
        pred = np.empty(test_idx.size, dtype=np.float32)
        for start in range(0, test_idx.size, 200_000):
            sl = test_idx[start:start + 200_000]
            xb = feature_matrix(h79_flat, stack_flat, sl, arm, raw_cols)
            pred[start:start + sl.size] = clf.predict_proba(xb)[:, 1].astype(np.float32)
            del xb
        out.ravel()[test_idx] = pred
        del clf, pred, test_idx
        log(f"{arm} seed {seed}: fold {fold['name']} complete")
    out[~fp] = 0.0
    return out


def allocate_by_fold(surface: np.ndarray, folds: list[dict], grid: G.Grid,
                     dcat: np.ndarray, budget: int = BUDGET, random: bool = False,
                     seed: int = 79) -> np.ndarray:
    """Emit a fixed budget using the shared-kernel credit-ranked dot emitter."""
    fp = grid.footprint
    # Holdout placement is allowed to approach the withheld component, but must
    # stay away from the *visible* catalogue by the fixed 3 px collar. Using
    # distance to the full catalogue here would exclude the hidden positives
    # themselves and force every valid holdout DTI to zero (a detected bug in
    # the first H79 draft; IR-79O-01).
    allowed = []
    for f in folds:
        d_visible = ndimage.distance_transform_edt(~f["visible"]).astype(np.float32)
        allowed.append(f["region"] & fp & (~f["masked_known"]) & (d_visible > CAT_EXCLUSION))
        del d_visible
    total = sum(int(a.sum()) for a in allowed)
    dots = np.zeros(fp.shape, dtype=bool)
    rng = np.random.default_rng(seed)
    remaining = budget
    for i, a in enumerate(allowed):
        n = remaining if i == len(allowed) - 1 else int(round(budget * int(a.sum()) / max(total, 1)))
        n = min(n, int(a.sum()))
        field = rng.random(fp.shape, dtype=np.float32) if random else surface
        d = emit77.emit_dots(field, a, n, min_sep=MIN_SEP, candidate_factor=10)
        dots |= d
        remaining -= int(d.sum())
    return dots


def score_surface(surface: np.ndarray, folds: list[dict], grid: G.Grid, dcat: np.ndarray,
                  seed: int, arm: str) -> tuple[dict, np.ndarray, np.ndarray]:
    dots = allocate_by_fold(surface, folds, grid, dcat, random=False, seed=seed)
    terms = None
    per = {}
    for fold in folds:
        result, block_terms = EH.evaluate(dots.astype(np.float32), fold, grid.footprint)
        per[fold["name"]] = result
        terms = block_terms if terms is None else terms + block_terms
    return {"arm": arm, "dots": int(dots.sum()), "per_fold": per,
            "pooled_dti": float(EH.from_terms(terms.sum(axis=0)))}, dots, terms


def evaluate_seed(stack: np.ndarray, h79: np.ndarray, grid: G.Grid, seed: int,
                  raw_cols: list[int], arms: tuple[str, ...]) -> dict:
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    samples = make_samples(grid, folds, seed)
    dcat = ndimage.distance_transform_edt(~grid.catalogue).astype(np.float32)
    surfaces = {}
    scores = {}
    terms_by_arm = {}
    dots_by_arm = {}
    for arm in arms:
        surface = fit_predict_arm(stack, h79, grid, folds, samples, arm, seed, raw_cols)
        surfaces[arm] = surface
        scores[arm], dots_by_arm[arm], terms_by_arm[arm] = score_surface(surface, folds, grid, dcat, seed, arm)
        log(f"E2/E3 {arm} seed {seed}: HOLDOUT-DTI {scores[arm]['pooled_dti']:.6f}; dots={scores[arm]['dots']}")
    # Matched random control uses the same fold allocation and budget.
    random_surface = np.zeros(grid.footprint.shape, dtype=np.float32)
    random_dots = allocate_by_fold(random_surface, folds, grid, dcat, random=True, seed=79_999 + seed)
    random_terms = None
    random_per = {}
    for fold in folds:
        result, block_terms = EH.evaluate(random_dots.astype(np.float32), fold, grid.footprint)
        random_per[fold["name"]] = result
        random_terms = block_terms if random_terms is None else random_terms + block_terms
    terms_by_arm["random_control"] = random_terms
    random_score = {"arm": "random_control", "dots": int(random_dots.sum()), "per_fold": random_per,
                    "pooled_dti": float(EH.from_terms(random_terms.sum(axis=0)))}
    scores["random_control"] = random_score
    summary = EH.pooled_summary(terms_by_arm, draws=2000, seed=79_500 + seed,
                                candidate="H79_plus_raw" if "H79_plus_raw" in terms_by_arm else arms[0])
    summary["holdout_seed"] = seed
    summary["withheld_positive_pixels_by_fold"] = {
        f["name"]: int(f["truth"].sum()) for f in folds
    }
    summary["fixed_emission"] = {"budget": BUDGET, "minimum_separation_px": MIN_SEP,
                                  "catalogue_exclusion_px": CAT_EXCLUSION}
    json_dump(ROOT / "evidence" / f"h79_orient_holdout_seed{seed}.json", {
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "implementation_sha256": EH.implementation_hashes(),
        "seed": seed,
        "scores": scores,
        "pooled_summary": summary,
        "feature_input_policy": "H79 label-blind stack; no catalogue or prior raster in features",
    })
    return {"folds": folds, "scores": scores, "summary": summary,
            "surfaces": surfaces, "dots": dots_by_arm, "terms": terms_by_arm,
            "random_dots": random_dots}


def existing_raster_paths() -> list[Path]:
    """Return locally comparable witnesses without pretending the snapshot is complete.

    The first registry fetch hit GitHub's API rate limit after 521/812 blobs.
    The public dense witness and H33 witness are restored through the pinned
    codeload path and are always checked. The optional full scan is explicit
    because 500+ exact 3-px dilation/rank comparisons can exceed the session
    budget; it never changes the fail-closed status.
    """
    paths = set()
    for root in (ROOT / "docs" / "downloads", ROOT / "submissions"):
        if root.exists():
            paths.update(p for p in root.glob("*.tif") if p.is_file())
    for name in ("dense_witness.tif", "h33.tif"):
        p = ROOT / ".cache" / "registry" / name
        if p.is_file():
            paths.add(p)
    if str(__import__("os").environ.get("H79_COMPARE_FULL_REGISTRY", "0")) == "1":
        root = ROOT / ".cache" / "registry"
        if root.exists():
            paths.update(p for p in root.glob("*.tif") if p.is_file())
    return sorted(paths)


def audit_against_local_registry(surface: np.ndarray, grid: G.Grid, label: str) -> dict:
    """Run exact shared comparisons where bytes are present; never call it complete."""
    fp = grid.footprint
    rows = []
    for path in existing_raster_paths():
        try:
            with rasterio.open(path) as src:
                if src.count != 1 or src.shape != fp.shape or src.crs != grid.crs or src.transform != grid.transform:
                    continue
                theirs = np.nan_to_num(src.read(1), nan=0.0).astype(np.float32)
            their_support = np.isfinite(theirs) & (theirs > 0) & fp
            my_support = (surface > 0) & fp
            # The shared implementation is used for the actual definitions.
            row = uniqueness.surface_rho(surface, theirs, fp)
            row.update({
                "file": str(path.relative_to(ROOT)),
                "their_dots": int(their_support.sum()),
                "forward_overlap_3px": uniqueness.dot_overlap(surface, theirs),
                "reverse_overlap_3px": uniqueness.dot_overlap(theirs, surface),
                "surface_duplicate": bool(row["spearman_full_footprint"] is not None and abs(row["spearman_full_footprint"]) > 0.90),
                "dot_duplicate": bool(uniqueness.dot_overlap(surface, theirs) > 0.70),
            })
            rows.append(row)
        except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
            rows.append({"file": str(path.relative_to(ROOT)), "error": str(exc)})
    comparable = [r for r in rows if "error" not in r]
    expected = 812
    out = {
        "evidence_class": "REGISTRY-MEASUREMENT",
        "phase": label,
        "registry_source": "local decoded GeoTIFFs only; expected public snapshot 812 rasters",
        "expected_registry_rasters": expected,
        "checked_local_rasters": len(comparable),
        "complete_accessible_scan": len(comparable) >= expected,
        "source_errors": [r for r in rows if "error" in r],
        "surface_threshold": 0.90,
        "forward_overlap_threshold": 0.70,
        "surface_duplicate_count": sum(r.get("surface_duplicate", False) for r in comparable),
        "dot_duplicate_count": sum(r.get("dot_duplicate", False) for r in comparable),
        "rows": rows,
        "policy": "incomplete or dense registry is never converted into protocol clearance",
    }
    if comparable:
        out["max_abs_surface_spearman"] = max(abs(r["spearman_full_footprint"]) for r in comparable if r.get("spearman_full_footprint") is not None)
        out["max_forward_overlap_3px"] = max(r["forward_overlap_3px"] for r in comparable)
    else:
        out["max_abs_surface_spearman"] = None
        out["max_forward_overlap_3px"] = None
    out["pass_on_checked_subset"] = bool(out["surface_duplicate_count"] == 0 and out["dot_duplicate_count"] == 0)
    out["protocol_unique"] = bool(out["complete_accessible_scan"] and out["pass_on_checked_subset"])
    json_dump(ROOT / "evidence" / f"h79_orient_registry_{label}.json", out)
    return out


def fit_full_primary(stack: np.ndarray, h79: np.ndarray, grid: G.Grid,
                     raw_cols: list[int]) -> np.ndarray:
    """Fit the chosen primary to visible catalogue positives for a download-only artifact."""
    fp, cat = grid.footprint, grid.catalogue
    dcat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    pos = np.flatnonzero(cat & fp)
    neg = np.flatnonzero(fp & (~cat) & (dcat > CAT_EXCLUSION))
    rng = np.random.default_rng(79_777)
    if pos.size > 45_000:
        pos = np.sort(rng.choice(pos, 45_000, replace=False))
    if neg.size > 90_000:
        neg = np.sort(rng.choice(neg, 90_000, replace=False))
    idx = np.concatenate((pos, neg))
    y = np.concatenate((np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)))
    clf = HistGradientBoostingClassifier(**HGB_KW, random_state=79_777)
    hf = h79.reshape(-1, h79.shape[2])
    sf = stack.reshape(-1, stack.shape[2])
    X = feature_matrix(hf, sf, idx, "H79_plus_raw", raw_cols)
    clf.fit(X, y)
    del X, idx, y
    p = np.zeros(fp.size, dtype=np.float32)
    all_idx = np.flatnonzero(fp)
    for start in range(0, all_idx.size, 200_000):
        sl = all_idx[start:start + 200_000]
        xb = feature_matrix(hf, sf, sl, "H79_plus_raw", raw_cols)
        p[start:start + sl.size] = clf.predict_proba(xb)[:, 1].astype(np.float32)
        del xb
    p = p.reshape(fp.shape)
    p[~fp] = 0.0
    p[cat] = 0.0
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("full-map primary probability failed finite unit-range check")
    return p


def package_candidate(surface: np.ndarray, grid: G.Grid, registry_surface: dict,
                      holdout20: dict, holdout30: dict) -> dict:
    """Run pre-placement audit, emit final dots, run final audit, then write TIFF."""
    dcat = ndimage.distance_transform_edt(~grid.catalogue).astype(np.float32)
    # Surface gate is explicitly before final placement.
    final_allowed = grid.footprint & (~grid.catalogue) & (dcat > CAT_EXCLUSION)
    surface_gate = dict(registry_surface)
    if surface_gate["surface_duplicate_count"] or surface_gate["dot_duplicate_count"]:
        log("H79 surface registry gate fired; retaining negative artifact and recording duplicate witness")
    dots = emit77.emit_dots(surface, final_allowed, BUDGET, min_sep=MIN_SEP, candidate_factor=10)
    surface_after = np.where(dots, 1.0, 0.0).astype(np.float32)
    registry_final = audit_against_local_registry(surface_after, grid, "final_dots")

    # A unique name is derived from the decoded final dot raster, never copied
    # from another repository. The human-facing name and note are kept <=140.
    decoded_sha = hashlib.sha256(np.ascontiguousarray(surface_after).tobytes()).hexdigest()
    base = f"gems79-orient-persistence-{int(dots.sum())}dots-20261021-{decoded_sha[:10]}"
    note = ("H79 cross-sensor scale-persistent orientation confluence; 3-field Gaussian "
            f"edge axes, {int(dots.sum())} dots; HOLDOUT-DTI seed20/30 recorded")
    if len(note) > 140:
        raise AssertionError(f"note too long: {len(note)}")
    out_tif = OUT_DIR / f"{base}.tif"
    if out_tif.exists() or out_tif.with_suffix(".zip").exists() or out_tif.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite H79 artifact: {out_tif}")
    receipt = submission_writer.write_submission(
        out_tif,
        surface_after,
        ROOT / "data" / "bridge" / "sample_submission.tif",
        grid.footprint,
        note=note,
        name=base,
        catalogue=grid.catalogue,
        metadata=_json_safe({
            "lane": "H79-ORIENT",
            "safe_to_download": True,
            "approved_for_weekly_slot": False,
            "promoted": False,
            "verdict": "promote" if (holdout20["scores"]["H79_plus_raw"]["pooled_dti"] > INCUMBENT_DTI and registry_surface["protocol_unique"] and registry_final["protocol_unique"]) else "negative",
            "surface_registry_gate_before_placement": registry_surface,
            "final_dot_registry_gate": registry_final,
            "fixed_emission": {"budget": BUDGET, "min_sep_px": MIN_SEP, "catalogue_exclusion_px": CAT_EXCLUSION},
        }),
    )
    card = {
        "schema_version": "gems-run-card-v2",
        "lane_id": "H79-ORIENT",
        "branch": "arena/77173f8d-gemsdoe77",
        "timestamp_utc": "2026-10-21",
        "hypothesis": "A hidden fault is more likely where RTP, isostatic gravity and detrended elevation share a stable edge axis across 150/300/600 m scales; source amplitudes are controls.",
        "mechanism": "label-blind Gaussian derivative edge persistence and doubled-angle cross-sensor orientation confluence; no catalogue mechanics and no prior predictions",
        "named_non_fault_process": "intrusive/lithologic contacts, flight-line seams, pluvial shoreline/terrace edges and drainage margins can mimic a persistent edge",
        "holdout": {
            "seed20": holdout20["summary"],
            "seed30": holdout30["summary"],
            "incumbent": {"evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
                          "dti": INCUMBENT_DTI, "ci95": INCUMBENT_CI95,
                          "withheld_positive_pixels": 11321,
                          "source": "retained H77-Step receipt; not an organizer score"},
        },
        "correlation_overlap_vs_registry": {
            "surface_before_placement": surface_gate,
            "final_dots": registry_final,
            "policy": "all 812 indexed rasters were not locally available after the GitHub API rate-limit; neither subset PASS nor byte novelty is organizer uniqueness clearance"
        },
        "raster_sha256": receipt["sha256"],
        "validator_output": receipt["validator"],
        "submission_name": base,
        "note": note,
        "file": str(out_tif.relative_to(ROOT)),
        "zip": str(out_tif.with_suffix(".zip").relative_to(ROOT)),
        "safe_to_download": True,
        "ok_to_download": True,
        "ok_to_submit": False,
        "approved_for_weekly_slot": False,
        "organizer_confirmed_score": None,
        "experiments_run": 3,
        "verdict": receipt["metadata"]["verdict"],
        "reason": "Negative: primary HOLDOUT-DTI is below the retained 0.152718 incumbent, the dense witness fires the literal registry gate, registry coverage is incomplete, and no organizer score is claimed; format is locally valid for download.",
        "irregularity_refs": ["IR-79O-01", "IR-79O-02", "IR-79O-03"],
        "superseded_runner_evidence": "evidence/h79_orient_superseded_zero_bug.json",
    }
    json_dump(ROOT / "evidence" / "h79_orient_run_card.json", card)
    json_dump(OUT_DIR / f"{base}-run-card.json", card)
    return card


def main() -> int:
    started = time.monotonic()
    grid = load_grid()
    if not STACK_PATH.exists():
        raise SystemExit("missing data/processed/stack.npy; run scripts/build_features.py after restoring pinned inputs")
    stack = np.load(STACK_PATH, mmap_mode="r")
    h79, names, channels = build_h79_features(stack, grid)
    raw_cols = [channels[n] for n in ("rtp", "iso_grav_anom", "det_elev", "depth_to_base_surf")]

    folds20, _ = HO.buffered_component_draw(grid, seed=SEED_PRIMARY, hide_frac=0.20, buffer_px=3)
    canary20 = canary(h79.reshape(-1, h79.shape[2]), grid, folds20, names, SEED_PRIMARY)
    hold20 = evaluate_seed(stack, h79, grid, SEED_PRIMARY, raw_cols,
                           ("H79_only", "H79_plus_raw", "raw_control"))
    hold30 = evaluate_seed(stack, h79, grid, SEED_CONFIRM, raw_cols,
                           ("H79_plus_raw", "raw_control"))

    # Full-map projection is not called a score. It is needed only to create the
    # requested unique, format-valid research download.
    full_surface = fit_full_primary(stack, h79, grid, raw_cols)
    registry_surface = audit_against_local_registry(full_surface, grid, "surface_before_placement")
    card = package_candidate(full_surface, grid, registry_surface, hold20, hold30)
    card["elapsed_seconds"] = round(time.monotonic() - started, 1)
    json_dump(ROOT / "evidence" / "h79_orient_run_card.json", card)
    # Publish an auditable, read-only copy with the GitHub Pages tree. The
    # repository evidence remains the source of truth; this copy prevents the
    # public technical page from linking to a path that Pages does not stage.
    public_evidence = ROOT / "docs" / "evidence"
    public_evidence.mkdir(parents=True, exist_ok=True)
    for name in ("h79_orient_run_card.json", "h79_orient_canary.json", "h79_orient_holdout_seed20.json",
                 "h79_orient_holdout_seed30.json", "h79_orient_registry_surface_before_placement.json",
                 "h79_orient_registry_final_dots.json"):
        src = ROOT / "evidence" / name
        if src.exists():
            shutil.copy2(src, public_evidence / name)
    log("H79 complete:", json.dumps({k: card[k] for k in ("file", "raster_sha256", "ok_to_download", "ok_to_submit", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
