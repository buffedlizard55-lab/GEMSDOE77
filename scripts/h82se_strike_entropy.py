#!/usr/bin/env python3
"""H82SE: pre-registered fault-strike-entropy lane using shared gems57 tools.

Run in order: e1 (feature-only leakage + pre-placement registry), e2 (dual-seed
HOLDOUT-DTI), e3 (shared-writer TIFF + final-dot registry). The feature is
computed only from the visible fault mask for each holdout. See
registry/preregistration_h82se.json and research/h82se_hypotheses.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src")]
from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import grid as GG  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems57.uniqueness import compare_array_to_registry  # noqa: E402
from gems77.emit77 import emit_dots  # noqa: E402

BUDGET = 37_654
MIN_SEP_PX = 2.8
SEEDS = (20, 30)
INCUMBENT = 0.152718  # local holdout reference in the H77-Step frozen card
REGISTRY = ROOT / "registry" / "registry_index.json"
REGISTRY_RECEIPT = ROOT / "registry" / "registry_receipt.json"
OUT = ROOT / "evidence"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strike_entropy(visible: np.ndarray, footprint: np.ndarray, *,
                   line_sigma: float = 1.0, tensor_sigma: float = 10.0,
                   density_sigma: float = 10.0) -> np.ndarray:
    """Spin-2 local orientation entropy, gated by visible-trace density.

    This takes only the fold-visible binary fault mask and footprint. The
    entropy term is 1 - structure-tensor coherence; the density gate suppresses
    arbitrary orientation in empty space. Output is finite float32 in [0,1].
    """
    visible = np.asarray(visible, dtype=bool)
    footprint = np.asarray(footprint, dtype=bool)
    if visible.ndim != 2 or visible.shape != footprint.shape:
        raise ValueError("visible fault mask and footprint must be matching 2-D arrays")
    if (visible & ~footprint).any():
        raise ValueError("visible catalogue extends outside footprint")
    line = ndi.gaussian_filter(visible.astype(np.float32), line_sigma,
                               mode="constant", cval=0.0, truncate=3.0)
    gy, gx = np.gradient(line)
    jxx = ndi.gaussian_filter(gx * gx, tensor_sigma, mode="constant", cval=0.0, truncate=3.0)
    jyy = ndi.gaussian_filter(gy * gy, tensor_sigma, mode="constant", cval=0.0, truncate=3.0)
    jxy = ndi.gaussian_filter(gx * gy, tensor_sigma, mode="constant", cval=0.0, truncate=3.0)
    trace = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / np.maximum(trace, 1e-12)
    entropy = np.clip(1.0 - coherence, 0.0, 1.0)
    density = ndi.gaussian_filter(visible.astype(np.float32), density_sigma,
                                  mode="constant", cval=0.0, truncate=3.0)
    scale_values = density[footprint]
    if not scale_values.size:
        raise ValueError("empty footprint")
    q95 = float(np.quantile(scale_values, 0.95))
    if not np.isfinite(q95) or q95 <= 0:
        return np.zeros(visible.shape, np.float32)
    gate = -np.expm1(-density / q95)
    result = np.where(footprint, entropy * gate, 0.0).astype(np.float32)
    peak = float(result[footprint].max(initial=0.0))
    if peak > 0:
        result /= np.float32(peak)
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1.000001).any():
        raise ValueError("strike entropy transform produced invalid values")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _registry_records(exclude_paths: set[Path] | None = None) -> dict:
    """Build an immutable inventory view from pinned blobs + current local TIFFs."""
    idx = json.loads(REGISTRY.read_text())
    receipt = json.loads(REGISTRY_RECEIPT.read_text())
    sha_by_blob = receipt["sha256_by_blob"]
    records = []
    missing_pins = []
    for row in idx["rasters"]:
        blob = row["blob"]
        path = ROOT / ".cache" / "registry" / f"{blob}.tif"
        digest = sha_by_blob.get(blob)
        if not digest or not path.is_file():
            missing_pins.append(blob)
            continue
        records.append({"repo_first": row["repo"], "sources": row["aliases"],
                        "cache_file": str(path), "sha256": digest})
    exclude = {p.resolve() for p in (exclude_paths or set())}
    local_paths = sorted((ROOT / "docs" / "downloads").glob("*.tif")) + sorted((ROOT / "submissions").glob("*.tif"))
    local_count = 0
    for path in local_paths:
        if path.resolve() in exclude:
            continue
        records.append({"repo_first": "GEMSDOE77-local", "sources": [path.name],
                        "cache_file": str(path), "sha256": sha256(path)})
        local_count += 1
    index = {
        "rasters": records,
        "complete_accessible_scan": bool(not missing_pins and len(records) == len(idx["rasters"]) + local_count),
        "errors": [],
        "n_registry": len(idx["rasters"]),
        "n_local": local_count,
        "missing_pins": missing_pins,
    }
    return index


def _canary(surface: np.ndarray, hidden: np.ndarray, grid) -> dict:
    """Symmetric AUC for this one feature; no model or other channels."""
    foot = np.asarray(grid.footprint, bool)
    cat = np.asarray(grid.catalogue, bool)
    distance = ndi.distance_transform_edt(~cat)
    pos = np.flatnonzero(np.asarray(hidden, bool) & foot)
    neg_pool = np.flatnonzero(foot & ~cat & (distance > 3.0))
    rng = np.random.default_rng(8201)
    n_neg = min(200_000, int(neg_pool.size))
    neg = rng.choice(neg_pool, n_neg, replace=False)
    if not pos.size or not neg.size:
        raise ValueError("canary requires positive and negative samples")
    scores = np.concatenate([surface.ravel()[pos], surface.ravel()[neg]])
    labels = np.concatenate([np.ones(pos.size, np.uint8), np.zeros(neg.size, np.uint8)])
    auc = float(roc_auc_score(labels, scores))
    return {"feature": "h82se_strike_entropy", "evidence_class": "LEAKAGE-CANARY",
            "auc": auc, "symmetric_auc": max(auc, 1.0 - auc),
            "threshold": 0.90, "pass": max(auc, 1.0 - auc) <= 0.90,
            "withheld_positive_pixels": int(pos.size), "negative_pixels": int(neg.size),
            "negative_definition": "footprint, outside full catalogue, >3 px from full catalogue"}


def _uniqueness(surface: np.ndarray, phase: str, exclude_paths: set[Path] | None = None) -> dict:
    grid = GG.load_grid()
    inventory = _registry_records(exclude_paths)

    def progress(i, n, row):
        if i % 100 == 0 or i == n:
            print(f"{phase}: registry {i}/{n}", flush=True)

    result = compare_array_to_registry(
        surface, inventory, np.asarray(grid.footprint, bool), include_reverse=(phase != "surface"),
        progress=progress, expected_grid=(grid.shape, grid.crs, grid.transform))
    result["phase"] = phase
    result["inventory_complete_by_local_manifest"] = inventory["complete_accessible_scan"]
    result["indexed_owner_rasters"] = inventory["n_registry"]
    result["local_tiffs_compared"] = inventory["n_local"]
    result["unfetched_blob_count"] = len(inventory["missing_pins"])
    return result


def run_e1() -> None:
    grid = GG.load_grid()
    folds, _ = HO.buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
    visible = np.asarray(folds[0]["visible"], bool)
    hidden = np.asarray(folds[0]["hidden_all"], bool)
    feature = strike_entropy(visible, grid.footprint)
    canary = _canary(feature, hidden, grid)
    full_surface = strike_entropy(grid.catalogue, grid.footprint)
    uq = _uniqueness(full_surface, "h82-preplacement-surface")
    max_abs_rho = max((abs(r["spearman_full_footprint"]) for r in uq["rows"]
                       if r.get("spearman_full_footprint") is not None), default=0.0)
    surface_dup = bool(max_abs_rho > 0.90)
    e1 = {
        "experiment": "H82SE-E1",
        "feature_definition": "1-sigma smoothed visible-fault structure-tensor entropy times density gate; 10-px tensor/density scales; q95 footprint scale",
        "input_catalogue_sha256": sha256(GG.DATA_DIR / "existing_faults.tif"),
        "input_footprint_sha256": sha256(GG.DATA_DIR / "sample_submission.tif"),
        "cached_feature_stack_available": (ROOT / "data" / "processed" / "stack.npy").is_file(),
        "cached_feature_stack_used": False,
        "canary": canary,
        "preplacement_uniqueness": uq,
        "surface_max_abs_spearman": max_abs_rho,
        "surface_duplicate": surface_dup,
        "registry_complete": uq.get("complete_accessible_scan", False),
        "decision": "STOP" if (not canary["pass"] or surface_dup) else "CONTINUE_TO_E2",
    }
    write_json(OUT / "h82se_e1_surface.json", e1)
    print(json.dumps({k: e1[k] for k in ("canary", "surface_max_abs_spearman", "surface_duplicate", "registry_complete", "decision")}, indent=2))
    if not canary["pass"] or surface_dup:
        raise SystemExit("H82SE E1 stop gate fired; no holdout or emission")


def _evaluate_seed(seed: int) -> dict:
    grid = GG.load_grid()
    folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
    visible = np.asarray(folds[0]["visible"], bool)
    surface = strike_entropy(visible, grid.footprint)
    dist = ndi.distance_transform_edt(~visible).astype(np.float32)
    # The distance control is a pre-registered comparator, not a fitted channel.
    distance_control = np.where(grid.footprint, 1.0 / (1.0 + dist), 0.0).astype(np.float32)
    rng = np.random.default_rng(8200 + seed)
    random_surface = np.where(grid.footprint, rng.random(grid.shape, dtype=np.float32), 0.0).astype(np.float32)
    arms = {"H82SE_entropy": surface, "visible_distance_control": distance_control,
            "random_control": random_surface}
    terms: dict[str, list[np.ndarray]] = {k: [] for k in arms}
    receipts = []
    for fold in folds:
        allowed = np.asarray(grid.footprint, bool) & ~np.asarray(fold["masked_known"], bool)
        for name, ranking in arms.items():
            support_allowed = allowed & (ranking > 0)
            dots = emit_dots(ranking, support_allowed, BUDGET, min_sep=MIN_SEP_PX)
            n_dots = int(dots.sum())
            if n_dots != BUDGET:
                raise RuntimeError(f"{name}/{fold['name']}: emitted {n_dots}, expected {BUDGET}")
            _, block_terms = EH.evaluate(dots.astype(np.float32), fold, grid.footprint,
                                         block_side=200, global_shape=grid.shape)
            terms[name].append(block_terms)
        receipts.append({**fold["receipt"], "dots_per_arm_per_fold": BUDGET})
    pooled_terms = {name: np.sum(parts, axis=0) for name, parts in terms.items()}
    summary = EH.pooled_summary(pooled_terms, draws=2000, seed=82_000 + seed,
                                candidate="H82SE_entropy")
    hidden_n = int(folds[0]["hidden_all"].sum())
    return {"evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
            "holdout_seed": seed, "withheld_positive_pixels": hidden_n,
            "alpha": 0.2, "beta": 0.8, "triangular_radius_m": 300.0,
            "summary": summary, "fold_receipts": receipts,
            "implementation_sha256": EH.implementation_hashes()}


def run_e2() -> None:
    e1 = json.loads((OUT / "h82se_e1_surface.json").read_text())
    if e1["decision"] != "CONTINUE_TO_E2":
        raise SystemExit("E1 did not pass")
    seed20 = _evaluate_seed(20)
    seed30 = _evaluate_seed(30)
    result = {"experiment": "H82SE-E2", "evaluator_version": EH.VERSION,
              "seed20": seed20, "seed30": seed30,
              "incumbent_reference": {"evidence_class": "HOLDOUT-DTI",
                  "evaluator_version": EH.VERSION, "score": INCUMBENT,
                  "withheld_positive_pixels": 11321,
                  "ci95": [0.131548, 0.175185],
                  "source": "evidence/h77/run_card.json; same evaluator and primary holdout seed; not a paired cross-lane interval"}}
    write_json(OUT / "h82se_e2_holdout.json", result)
    print(json.dumps({"seed20": seed20["summary"], "seed30": seed30["summary"]}, indent=2))


def run_e3() -> None:
    e1 = json.loads((OUT / "h82se_e1_surface.json").read_text())
    e2 = json.loads((OUT / "h82se_e2_holdout.json").read_text())
    if e1["decision"] != "CONTINUE_TO_E2":
        raise SystemExit("E1 stop gate fired")
    grid = GG.load_grid()
    full_surface = strike_entropy(grid.catalogue, grid.footprint)
    allowed = np.asarray(grid.footprint, bool) & ~np.asarray(grid.catalogue, bool) & (full_surface > 0)
    dots = emit_dots(full_surface, allowed, BUDGET, min_sep=MIN_SEP_PX)
    n_dots = int(dots.sum())
    if n_dots != BUDGET:
        raise RuntimeError(f"final emission produced {n_dots}, expected {BUDGET}")
    pred = dots.astype(np.float32)
    dot_digest = hashlib.sha256(np.ascontiguousarray(np.flatnonzero(dots), dtype=np.int64).tobytes()).hexdigest()[:8]
    name = f"gems77-h82se-strike-entropy-{BUDGET}dots-20261021-{dot_digest}-zeros"
    path = ROOT / "submissions" / f"{name}.tif"
    s20 = e2["seed20"]["summary"]["scores"]["H82SE_entropy"]
    note = (f"H82SE local fault-strike entropy; {BUDGET} dots; HOLDOUT-DTI {s20['dti']:.4f} "
            f"({EH.VERSION}, {s20['withheld_positive_pixels']} positives, CI {s20['ci95'][0]:.4f}-{s20['ci95'][1]:.4f}); NOT SUBMIT")
    if len(note) > 140:
        raise ValueError(f"submission note too long ({len(note)} chars)")
    receipt = write_submission(path, pred, GG.DATA_DIR / "sample_submission.tif",
                               grid.footprint, note=note, name=name,
                               catalogue=grid.catalogue,
                               metadata={"lane": "H82SE-STRIKE-ENTROPY", "selector_approved": False,
                                         "weekly_slots_used": 0, "budget_dots": BUDGET,
                                         "holdout_seed20": s20["dti"], "holdout_seed30": e2["seed30"]["summary"]["scores"]["H82SE_entropy"]["dti"]})
    download_path = ROOT / "docs" / "downloads" / path.name
    shutil.copy2(path, download_path)
    shutil.copy2(path.with_suffix(".zip"), download_path.with_suffix(".zip"))
    shutil.copy2(path.with_suffix(".json"), download_path.with_suffix(".json"))
    # Exclude only this exact artifact path from the local comparison inventory.
    uq = _uniqueness(pred, "h82-final-dots", exclude_paths={path, download_path})
    comparisons = uq["rows"]
    max_overlap = max((r["my_dots_within_3px_of_theirs"] for r in comparisons
                       if "error" not in r), default=0.0)
    max_abs_rho = max((abs(r["spearman_full_footprint"]) for r in comparisons
                       if r.get("spearman_full_footprint") is not None), default=0.0)
    duplicate_rows = [r for r in comparisons if r.get("duplicate_by_rho") or r.get("duplicate_by_overlap")
                      or r.get("identical_bytes") or r.get("identical_decoded_predictions")]
    scores20 = e2["seed20"]["summary"]["scores"]["H82SE_entropy"]
    scores30 = e2["seed30"]["summary"]["scores"]["H82SE_entropy"]
    beats = scores20["ci95"][0] > INCUMBENT
    gates_pass = bool(not duplicate_rows and uq["complete_accessible_scan"] and beats)
    card = {
        "schema_version": "gems-run-card-v2",
        "lane_id": "H82SE-STRIKE-ENTROPY",
        "hypothesis": "Locally mixed fault-strike orientation around visible mapped strands marks transfer/intersection zones from which unrecorded faults may continue.",
        "mechanism": "Spin-2 Gaussian structure-tensor entropy times local visible trace density; computed fold-blind from visible catalogue only, no training on held-out labels.",
        "named_non_fault_process": "Digitization bends, dense mapped swarms, or non-geothermal fault intersections.",
        "holdout": {"seed20": scores20, "seed30": scores30,
                    "evaluator_version": EH.VERSION,
                    "reference_incumbent": {"evidence_class": "HOLDOUT-DTI",
                        "evaluator_version": EH.VERSION, "score": INCUMBENT,
                        "withheld_positive_pixels": 11321,
                        "ci95": [0.131548, 0.175185],
                        "source": "evidence/h77/run_card.json; same evaluator and primary holdout seed; not a paired cross-lane interval"},
                    "comparison_note": "seed20 lower CI must exceed the historical point estimate; no paired cross-lane CI is claimed"},
        "correlation_overlap_vs_registry": {
            "surface_max_abs_spearman": e1["surface_max_abs_spearman"],
            "surface_registry_rasters_checked": e1["preplacement_uniqueness"]["registry_rasters_checked"],
            "surface_registry_complete": e1["registry_complete"],
            "final_max_abs_spearman": max_abs_rho,
            "final_worst_forward_3px_overlap": max_overlap,
            "final_registry_rasters_checked": uq["registry_rasters_checked"],
            "final_registry_complete": uq["complete_accessible_scan"],
            "duplicate_firings": len(duplicate_rows),
            "duplicate_examples": [{"repo": r.get("repo"), "source": r.get("submission"),
                                     "rho": r.get("spearman_full_footprint"),
                                     "forward_overlap": r.get("my_dots_within_3px_of_theirs")}
                                    for r in duplicate_rows[:10]],
            "complete_public_inventory_scope_only": True,
            "unique_certified": bool(uq["complete_accessible_scan"] and not duplicate_rows)},
        "raster_sha256": receipt["sha256"],
        "validator": receipt["validator"],
        "validator_output": {"no_nan_inside_footprint": receipt["validator"]["checks"]["in_footprint_all_finite"],
                             "values_in_0_1": receipt["validator"]["checks"]["in_footprint_range_0_1"],
                             "crs_shape_transform_match": bool(receipt["validator"]["checks"]["crs_epsg_32611"] and
                                 receipt["validator"]["checks"]["dimensions_3730x3292"] and
                                 receipt["validator"]["checks"]["transform_matches_sample_submission"])},
        "submission_name": name,
        "submission_note": note,
        "verdict": "promote" if gates_pass else "negative",
        "selector_approved": False,
        "safe_to_submit": False,
        "ok_to_download": True,
        "ok_to_submit": False,
        "submission_slots_used": 0,
        "historical_incumbent_beaten": bool(beats),
        "status": "OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT",
        "writer_receipt": receipt,
        "preplacement_uniqueness": e1["preplacement_uniqueness"],
        "final_uniqueness": uq,
        "irregularities": [
            "No assigned single-method paragraph was identifiable in the supplied prompt; H82SE was self-declared before E1.",
            "The declared cached feature stack is absent. H82SE is catalogue-only and did not rebuild or privately fork it.",
            "The public owner-repository raster inventory is not the organizer's full/private registry; completeness only applies to the indexed accessible inventory.",
            "The shared release policy has a known full-support dense-raster witness; literal uniqueness is expected to fail for any nonempty dot field under Policy I17.",
            "No organizer upload/receipt or weekly slot was used; local TIFF validation is not portal acceptance."
        ]
    }
    write_json(OUT / "h82se_run_card.json", card)
    write_json(OUT / "h82se_e3_uniqueness.json", uq)
    print(json.dumps({"submission_name": name, "sha256": receipt["sha256"],
                      "seed20_dti": scores20["dti"], "seed30_dti": scores30["dti"],
                      "max_overlap": max_overlap, "max_abs_rho": max_abs_rho,
                      "registry_complete": uq["complete_accessible_scan"],
                      "ok_to_download": True, "ok_to_submit": False,
                      "verdict": card["verdict"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("e1", "e2", "e3"))
    args = parser.parse_args()
    {"e1": run_e1, "e2": run_e2, "e3": run_e3}[args.mode]()


if __name__ == "__main__":
    main()
