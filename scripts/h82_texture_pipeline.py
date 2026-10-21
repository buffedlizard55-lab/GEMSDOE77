#!/usr/bin/env python3
"""One H82 feasibility experiment; fixed continuous surface, no slot and no dots.

Uses the pinned evaluator/writer/uniqueness tools, not a private scoring fork.
A failed surface gate is retained as research evidence, never converted to dots.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT / "scripts")]
from gems57 import evaluate_holdout, grid, spatial, uniqueness
from gems57.submission_writer import write_submission
from gems77.texture82 import build
from fetch_data import FILES, sha
from restore_registry import get_registry


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run():
    prereg_path = ROOT / "registry/preregistration_h82_texture.json"
    prereg = json.loads(prereg_path.read_text())
    out_dir = ROOT / "evidence/h82_texture"
    if (out_dir / "run_card.json").exists():
        raise FileExistsError("H82 is closed after its first recorded experiment; refusing rerun")
    pins = json.loads((ROOT / "registry/template_pin.json").read_text())
    for name, expected in pins["files"].items():
        if sha(ROOT / "vendor/gems57" / name) != expected:
            raise ValueError(f"shared template changed: {name}")
    inputs = []
    for name, (_, expected) in FILES.items():
        path = ROOT / "data" / name
        if sha(path) != expected:
            raise ValueError(f"input pin mismatch: {name}")
        inputs.append(dict(file=str(path.relative_to(ROOT)), sha256=expected))
    g = grid.load_grid(ROOT / "data/bridge")
    with rasterio.open(ROOT / "data/geodawn_rad_u8.tif") as ds:
        if (ds.shape, ds.crs, ds.transform) != (g.shape, g.crs, g.transform):
            raise ValueError("radiometric grid mismatch")
        if ds.descriptions[:2] != ("K", "Th") or ds.nodata != 0:
            raise ValueError("cache band metadata changed")
        k, th = ds.read([1, 2])
    raw, fields, eligible, scales = build(k, th, g.footprint)
    terms = {name: None for name in fields}
    canaries, split_receipts = [], []
    amplitude = prereg["parameters"]["continuous_amplitude"]
    for fold in spatial.folds(g.catalogue, eligible, buffer_px=80):
        split_receipts.append(fold["receipt"])
        yy, xx = np.nonzero(fold["region"])
        y0, x0 = int(yy.min()), int(xx.min())
        sl = np.s_[y0:int(yy.max()) + 1, x0:int(xx.max()) + 1]
        small = {key: fold[key][sl] for key in ("region", "visible", "truth")}
        active = small["region"] & ~small["visible"]
        truth = small["truth"][active]
        for name in fields:
            auc = float(roc_auc_score(truth, raw[name][sl][active]))
            symmetric = max(auc, 1 - auc)
            canaries.append(dict(feature=name, fold=fold["fold"], auc=auc,
                symmetric_auc=symmetric, leakage_flag=symmetric > 0.90,
                positive_pixels=int(truth.sum()), negative_proxy_pixels=int((~truth).sum()),
                evidence_class="LEAKAGE-CANARY, not a score"))
            _, blocks = evaluate_holdout.evaluate(fields[name][sl] * amplitude,
                small, eligible[sl], origin=(y0, x0), global_shape=g.shape)
            terms[name] = blocks if terms[name] is None else terms[name] + blocks
        print(f"fold {fold['fold']} done: {truth.sum()} withheld positives", flush=True)
    summary = evaluate_holdout.pooled_summary(terms, candidate="candidate", draws=1000, seed=82077)
    save(out_dir / "holdout.json", dict(summary=summary, split_receipts=split_receipts,
        canaries=canaries, normalization_scales=scales, eligible_pixels=int(eligible.sum()),
        normalization="label-free eligible-domain percentile; not trained probability calibration"))
    prediction = (fields["candidate"] * amplitude).astype(np.float32)
    prediction[g.catalogue] = 0
    witnesses = get_registry()
    # List inventory is incomplete by contract; never a full-registry clearance.
    pre = uniqueness.compare_array_to_registry(prediction, witnesses, g.footprint,
        expected_grid=(g.shape, g.crs, g.transform))
    save(out_dir / "surface_registry.json", pre)
    if not pre["stop_required"]:
        raise AssertionError("unexpected surface clearance; no placement authorized by this script")
    certificate = uniqueness.saturation_certificate(witnesses[0]["file"], g.footprint,
        g.catalogue, expected_grid=(g.shape, g.crs, g.transform))
    decoded = hashlib.sha256(prediction.tobytes()).hexdigest()
    name = f"gems77-h82-texture-kt-{decoded[:12]}-RESEARCH-ONLY"
    note = "H82 K-Th local-variance boundary; fixed continuous surface, duplicate-stop before dots; research only DO NOT SUBMIT"
    path = ROOT / "docs/downloads" / (name + ".tif")
    receipt = write_submission(path, prediction, ROOT / "data/bridge/sample_submission.tif",
        g.footprint, catalogue=g.catalogue, name=name, note=note,
        metadata=dict(lane="H82-TEXTURE", research_only=True, protocol_unique=False))
    with rasterio.open(path) as ds:
        readback = ds.read(1)
    if not np.array_equal(prediction, readback):
        raise AssertionError("surface changed during serialization")
    post = uniqueness.compare_array_to_registry(readback, witnesses, g.footprint,
        file_sha256=receipt["sha256"], expected_grid=(g.shape, g.crs, g.transform))
    save(out_dir / "readback_registry.json", post)
    # Exact decoded comparison to the prior local download inventory. This is
    # distinction-only, not a replacement for the failed literal registry gate.
    local_rows = []
    for other in sorted((ROOT / "docs/downloads").glob("*.tif")):
        if other == path:
            continue
        with rasterio.open(other) as ds:
            if ds.count != 1 or (ds.shape, ds.crs, ds.transform) != (g.shape, g.crs, g.transform):
                local_rows.append(dict(file=other.name, error="grid mismatch"))
                continue
            a = ds.read(1)
        local_rows.append(dict(file=other.name, sha256=sha(other),
            identical_pixels_inside_footprint=bool(np.array_equal(a[g.footprint], readback[g.footprint]))))
    save(out_dir / "local_distinction.json", dict(scope="local downloads only, not uniqueness clearance", rows=local_rows))
    card = dict(schema="gems-parallel-run-card-v1", lane="H82-TEXTURE", experiment_count=1,
        preregistration_sha256=sha(prereg_path), inputs=inputs, organizer_authenticated_inputs=False,
        hypothesis=prereg["hypothesis"],
        mechanism="9x9 K/Th local SD, central-difference magnitude, geometric mean, label-free p99 normalization, fixed amplitude 0.02",
        named_non_fault_process=prereg["mimic"],
        holdout_dti=summary["scores"]["candidate"], paired_vs_constant=summary["paired_differences"]["constant"],
        holdout_comparability="Fixed-quadrant eligible-support continuous-output feasibility test; no matched incumbent rerun, not a leaderboard score",
        leakage_canary=dict(tests=len(canaries), max_symmetric_auc=max(r["symmetric_auc"] for r in canaries),
            flagged=any(r["leakage_flag"] for r in canaries)),
        registry=dict(scope="two SHA-pinned rejection witnesses only", full_registry_certified=False,
            surface_max_spearman=pre["worst_spearman_full_footprint"],
            surface_max_forward_overlap_3px=pre["worst_dot_overlap"],
            duplicate_rasters_in_checked_subset=pre["duplicate_count"],
            postwrite_max_forward_overlap_3px=post["worst_dot_overlap"],
            final_dots="NOT PLACED: failed surface gate; postwrite checks same continuous surface",
            saturation_certificate=certificate),
        raster_sha256=receipt["sha256"], raster_path=str(path.relative_to(ROOT)),
        validator=receipt["validator"], unchanged_surface_readback=True,
        local_inventory_compared=len(local_rows),
        pixel_distinct_from_checked_local_downloads=all(not r.get("identical_pixels_inside_footprint", True) for r in local_rows),
        submission_name=name, submission_note=note,
        verdict="negative", protocol_unique=False, ok_to_download=True, ok_to_submit=False,
        selector_approved=False, weekly_slots_used=0, organizer_confirmed_score=None,
        reason="Literal overlap exceeds 0.70; stop before placement. No matched-current-best win or selector approval.")
    save(out_dir / "run_card.json", card)
    print(json.dumps({key: card[key] for key in ("raster_path", "raster_sha256", "holdout_dti", "verdict")}, indent=2))


if __name__ == "__main__":
    run()
