#!/usr/bin/env python3
"""One frozen H77L experiment: label-free feature, buffered hide/recover, duplicate stop.

This script intentionally never calls the DrivenData submission endpoint and never
sets a selector approval. It writes a *research-only* GeoTIFF on a failed surface
gate; the shared pinned evaluator/writer/gates are imported, never forked.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor"), str(ROOT / "scripts")]
from gems57 import evaluate_holdout, grid, spatial, uniqueness  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402
from gems77lag.features import FIELDS, build  # noqa: E402
from restore_registry import get_registry  # noqa: E402

PREREG = ROOT / "registry/preregistration_h77l.json"
CARD = ROOT / "evidence/h77l_run_card.json"


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> dict:
    manifest = json.loads((ROOT / "registry/data_manifest.json").read_text())
    required = {
        "sample_submission": ROOT / "data/bridge/sample_submission.tif",
        "existing_faults": ROOT / "data/bridge/existing_faults.tif",
        "ext_lidar_scarp_features_u8": ROOT / "data/raw/external/lidar_scarp_features_u8.tif",
        "ext_geodawn_extensions_u8": ROOT / "data/raw/external/geodawn_extensions_u8.tif",
    }
    entries = {entry["id"]: entry for entry in manifest["files"]}
    receipt = {"evidence_class": "INPUT-INTEGRITY", "organizer_authenticated": False, "files": []}
    for ident, path in required.items():
        expected = entries[ident]
        if not path.is_file() or sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"SHA256 / byte-count mismatch or missing owner mirror: {ident}")
        receipt["files"].append({"id": ident, "path": str(path.relative_to(ROOT)),
                                 "sha256": expected["sha256"], "bytes": expected["bytes"],
                                 "pinned_public_mirror": f"https://github.com/{expected['repo']}/blob/{expected['ref']}/{expected.get('path', expected['dest'])}"})
    return receipt


def read_bands(g):
    paths = (ROOT / "data/raw/external/lidar_scarp_features_u8.tif",
             ROOT / "data/raw/external/geodawn_extensions_u8.tif")
    with rasterio.open(paths[0]) as ds:
        if (ds.shape, ds.crs, ds.transform) != (g.shape, g.crs, g.transform):
            raise ValueError("LiDAR feature grid differs from sample")
        if (ds.descriptions[2], ds.descriptions[9], ds.descriptions[11]) != ("step_max", "coh100", "valid") or ds.nodata != 0:
            raise ValueError("LiDAR band meaning/nodata changed")
        step, coh, lid_valid = ds.read([3, 10, 12])
    with rasterio.open(paths[1]) as ds:
        if (ds.shape, ds.crs, ds.transform) != (g.shape, g.crs, g.transform):
            raise ValueError("GeoDAWN extension grid differs from sample")
        if ds.descriptions[0] != "ThK" or ds.nodata != 0:
            raise ValueError("ThK band meaning/nodata changed")
        thk = ds.read(1)
    return step, coh, lid_valid, thk


def crop(fold):
    ys, xs = np.nonzero(fold["region"])
    if not ys.size:
        raise ValueError("empty evaluation region")
    y0, x0 = int(ys.min()), int(xs.min())
    sl = np.s_[y0:int(ys.max()) + 1, x0:int(xs.max()) + 1]
    return sl, {key: fold[key][sl] for key in ("region", "visible", "truth")}, (y0, x0)


def run_holdout(raw, arms, eligible, catalogue):
    terms = {name: None for name in FIELDS}
    canaries, folds = [], []
    for fold in spatial.folds(catalogue, eligible, buffer_px=80):
        sl, small, origin = crop(fold)
        active = small["region"] & ~small["visible"]
        truth = small["truth"][active]
        if not truth.any() or truth.all():
            raise ValueError("degenerate fold canary positives/negative proxies")
        if (small["truth"] & small["visible"]).any():
            raise AssertionError("hidden labels are still in the visible fault mask")
        folds.append(fold["receipt"])
        for name in FIELDS:
            # AUC uses the unthresholded single feature, not its sparse emitted
            # field: top-tail clipping must not conceal a >0.90 leakage canary.
            auc = float(roc_auc_score(truth, raw[name][sl][active]))
            canaries.append({"feature": name, "fold": int(fold["fold"]), "auc": auc,
                             "positive_pixels": int(truth.sum()),
                             "negative_proxy_pixels": int((~truth).sum()),
                             "leakage_flag": bool(auc > 0.90),
                             "evidence_class": "LEAKAGE-CANARY (not a leaderboard score)"})
            result, blocks = evaluate_holdout.evaluate(
                arms[name][sl], small, eligible[sl], origin=origin,
                global_shape=eligible.shape, block_side=200)
            if result["n_truth"] != int(truth.sum()):
                raise AssertionError("shared evaluator changed held-out truth count")
            terms[name] = blocks if terms[name] is None else terms[name] + blocks
        print(f"H77L fold {fold['fold']} withheld {int(truth.sum())} catalogue px", flush=True)
    summary = evaluate_holdout.pooled_summary(terms, candidate="candidate", draws=1000, seed=77077)
    return {"evidence_class": "HOLDOUT-DTI", "scores": summary,
            "split_receipts": folds, "leakage_canaries": canaries,
            "canary_flagged": any(x["leakage_flag"] for x in canaries),
            "note": "Labels are public known faults hidden as whole components, not undiscovered organizer truth. Validity restricted to filter-safe LiDAR+ThK coverage. Not comparable to other lanes unless fold/positive masks match."}


def main() -> int:
    if CARD.exists():
        raise SystemExit("H77L already finalized; no second experiment or re-tuning")
    if not PREREG.is_file():
        raise SystemExit("Missing before-experiment preregistration")
    started = time.monotonic()
    prereg_sha = sha(PREREG)
    inputs = verify_inputs()
    g = grid.load_grid(ROOT / "data/bridge")
    stack = read_bands(g)
    raw, arms, eligible, scaling = build(*stack, g.footprint)
    del stack
    gc.collect()
    if not (arms["candidate"] > 0).any():
        raise ValueError("H77L response has no positive signal: no meaningful research raster")
    print("H77L label-free eligible", int(eligible.sum()),
          "candidate positive", scaling["candidate"]["positive_after"], flush=True)
    holdout = run_holdout(raw, arms, eligible, g.catalogue)
    save(ROOT / "evidence/h77l_holdout.json", holdout)

    # Surface gate BEFORE any point placement; rejection is binding even if
    # ranks are dissimilar. A single pinned complete-footprint witness proves
    # this candidate's negative verdict without inventing missing registries.
    surface = arms["candidate"].copy()
    surface[g.catalogue] = 0.0  # final known-fault mask; NEVER applied in folds
    del raw, arms
    gc.collect()
    if not (surface > 0).any():
        raise ValueError("no off-catalogue research response after masking")
    witnesses = get_registry()
    certificate = uniqueness.saturation_certificate(
        Path(witnesses[0]["file"]), g.footprint, g.catalogue,
        expected_grid=(g.shape, g.crs, g.transform))
    pre = uniqueness.compare_array_to_registry(
        surface, witnesses, g.footprint, expected_grid=(g.shape, g.crs, g.transform),
        include_reverse=False)
    save(ROOT / "evidence/h77l_surface_registry.json", {"certificate": certificate,
                                                         "comparison": pre})
    if not certificate["universal_overlap_blocker"] or pre["duplicate_count"] < 1:
        raise AssertionError("pinned blocker/protocol did not match preregistration; stop for review")

    # Retain the rejected *surface* only for review, not a post-stop optimized
    # dot set. Shared fail-closed writer ensures all values on disk are [0,1]
    # despite a prior portal range error; it does not grant a weekly slot.
    decoded_sha = hashlib.sha256(np.ascontiguousarray(surface).tobytes()).hexdigest()
    name = f"gems77-h77l-thk-flank-{decoded_sha[:12]}-RESEARCH-ONLY"
    note = "H77L displaced ThK / lidar scarp; literal overlap fails; research only; DO NOT SUBMIT"
    out = ROOT / "docs/downloads" / f"{name}.tif"
    receipt = write_submission(out, surface, ROOT / "data/bridge/sample_submission.tif",
                               g.footprint, name=name, note=note, catalogue=g.catalogue,
                               metadata={"lane": "H77L", "research_only": True,
                                         "status": "NOT OK TO SUBMIT", "dot_placement": "NOT RUN: surface duplicate stop",
                                         "preregistration_sha256": prereg_sha,
                                         "decoded_float32_sha256": decoded_sha})
    post = uniqueness.compare_to_registry(
        out, witnesses, g.footprint, expected_grid=(g.shape, g.crs, g.transform),
        include_reverse=False)
    with rasterio.open(out) as ds:
        written = ds.read(1)
    if not np.array_equal(written, surface) or post["candidate_decoded_sha256"] != decoded_sha:
        raise AssertionError("serialized research surface differs from the checked pre-placement surface")
    save(ROOT / "evidence/h77l_postwrite_registry.json", post)
    summary = holdout["scores"]
    score = summary["scores"]["candidate"]
    delta = summary["paired_differences"]["lidar_step"]
    head_to_head = next(r for r in post["rows"] if r["sha256"] == witnesses[1]["sha256"])
    card = {
        "lane": "H77L", "preregistration_sha256": prereg_sha, "experiment_count": 1,
        "hypothesis": "A 3-pixel strike-normal displaced GeoDAWN ThK rank-intensity flank at a USGS 3DEP lidar micro-scarp can signal a not-yet-mapped structure",
        "mechanism": "Local structure-tensor scarp normal, positive 300 m ThK flank contrast minus 1.25x zero-lag gradient, weighted by lidar step and coherence; raw inputs only, no previous submission reused",
        "named_non_fault_process": "road/canal embankment or Pleistocene lake shoreline bench across changing alluvium",
        "holdout_dti": score,
        "paired_vs_lidar_step": delta,
        "best_comparable_control_this_experiment": summary["best_comparable_control"],
        "current_repository_best_warning": "H77-Step 0.152718 HOLDOUT-DTI (gems57-pooled-hide-v2, 11,321 withheld positives, 95% CI [0.131548, 0.175185]) uses different folds/support; not comparable. No evidence this candidate beats the repository incumbent.",
        "leakage_canary": {"threshold_auc": 0.90, "flagged": holdout["canary_flagged"],
                           "n_feature_fold_tests": len(holdout["leakage_canaries"]),
                           "highest_auc": max(x["auc"] for x in holdout["leakage_canaries"])},
        "registry": {"scope": "two independently SHA-pinned rejection witnesses ONLY; not the full registry",
                     "surface_gate": "FAIL — one prior has universal >0 support; stop BEFORE dot placement",
                     "surface_max_spearman": pre["worst_spearman_full_footprint"],
                     "surface_max_forward_overlap_3px": pre["worst_dot_overlap"],
                     "postwrite_positive_support_max_spearman": post["worst_spearman_full_footprint"],
                     "postwrite_positive_support_max_forward_overlap_3px": post["worst_dot_overlap"],
                     "vs_h33_surface_rank_correlation": head_to_head["spearman_full_footprint"],
                     "vs_h33_positive_support_overlap_3px": head_to_head["my_dots_within_3px_of_theirs"],
                     "duplicate_rasters_in_checked_subset": post["duplicate_count"],
                     "full_registry_certified": False, "final_dot_placement_audit": "NOT RUN; pre-placement gate failed. Postwrite is unchanged surface, not placed dots.",
                     "universal_overlap_certificate": certificate},
        "raster_sha256": receipt["sha256"], "raster_path": str(out.relative_to(ROOT)),
        "zip_path": str(out.with_suffix(".zip").relative_to(ROOT)),
        "validator": {"all_checks_passed": receipt["validator"]["all_checks_passed"],
                      "checks": receipt["validator"]["checks"],
                      "no_nan_inside_footprint": receipt["validator"]["checks"]["in_footprint_all_finite"],
                      "values_in_0_1": receipt["validator"]["checks"]["range_0_1_guaranteed"],
                      "crs_shape_transform_match": bool(receipt["template_report"]["ok"]),
                      "min": receipt["validator"]["min"], "max": receipt["validator"]["max"],
                      "nan_count_anywhere": receipt["validator"]["n_nan"]},
        "submission_name": name, "submission_note": note, "note_length": len(note),
        "byte_distinct_vs_checked_witnesses": post["byte_unique_among_checked"],
        "pixel_distinct_vs_checked_witnesses": post["pixel_unique_among_checked"],
        "ok_to_download_for_research": True, "ok_to_submit": False,
        "weekly_slots_spent": 0, "organizer_confirmed_score_for_this_tif": None,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "verdict": "negative",
        "reason": "Literal registry overlap exceeds 0.70 on a confirmed 100%-support witness; stopped before placement. No matched-current-best improvement/selector approval or organizer score."
    }
    if len(note) > 140 or score["withheld_positive_pixels"] < 1:
        raise AssertionError("missing score population or overlong note")
    save(CARD, card)
    print(json.dumps({"research_tif": card["raster_path"], "raster_sha256": card["raster_sha256"],
                      "holdout": score, "literal_overlap": post["worst_dot_overlap"],
                      "verdict": card["verdict"], "ok_to_submit": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
