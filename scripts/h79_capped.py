#!/usr/bin/env python3
"""Lane H79 amendment E3b -- uniqueness-capped emission (registry/preregistration_h79_amendment_e3b.json).

  --constraints : choose the discriminating registry rasters to cap against from the latest phase-2
                  rows (forward overlap > --min-fwd), dedupe identical halos, write the list.
  --holdout     : HOLDOUT-DTI of the capped emission on the cached OOF surfaces (seeds 20/30),
                  paired against the uncapped greedy and A2_DIST on the same folds.
  --build       : capped emission on the full surface -> TIF/ZIP via the shared writer.
Nothing here is a score; all numbers are HOLDOUT-DTI (gems57-pooled-hide-v2) or measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import evaluate_holdout as EH  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems57 import submission_writer as SW  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems79 import capped_emit as CE  # noqa: E402

RAW = ROOT / "data" / "raw"
WORK = ROOT / ".cache" / "work" / "h79"
REG = ROOT / ".cache" / "registry"
EVID = ROOT / "evidence"
CAP_FRAC = 0.66


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def choose_constraints(min_fwd: float, extra_rows_file: Path | None) -> dict:
    halo = json.loads((ROOT / "registry" / "registry_halo_coverage.json").read_text())["rasters"]
    rows = json.loads((EVID / "h79_uniqueness_phase2_rows.json").read_text())
    if extra_rows_file and extra_rows_file.exists():
        rows += json.loads(extra_rows_file.read_text())
    prev = ROOT / "registry" / "h79_capped_constraints.json"
    keep = {c["blob"]: c for c in json.loads(prev.read_text())["constraints"]} if prev.exists() else {}
    seen_sig = {(c["their_dots"], round(c["halo_coverage_allowed"], 6)) for c in keep.values()}
    for r in sorted(rows, key=lambda r: -r["my_dots_within_3px_of_theirs"]):
        cov = halo.get(r["blob"], {}).get("covered_allowed_fraction")
        if cov is None or cov >= 0.70 or r["my_dots_within_3px_of_theirs"] <= min_fwd:
            continue
        sig = (r["their_dots"], round(cov, 6))
        if r["blob"] in keep or sig in seen_sig:  # identical halo (nan/zero/allfinite twins)
            continue
        seen_sig.add(sig)
        keep[r["blob"]] = {"blob": r["blob"], "submission": r["submission"], "repo": r["repo"], "their_dots": r["their_dots"],
                           "halo_coverage_allowed": cov, "first_build_forward_overlap": r["my_dots_within_3px_of_theirs"]}
    out = {"cap_fraction": CAP_FRAC, "min_forward_overlap_to_constrain": min_fwd, "n_constraints": len(keep),
           "constraints": sorted(keep.values(), key=lambda c: -c["first_build_forward_overlap"])}
    prev.write_text(json.dumps(out, indent=1))
    log(f"{len(keep)} distinct halo constraints -> {prev}")
    return out


def load_words(cons: dict, foot: np.ndarray):
    halos = [CE.halo_of_raster(REG / f"{c['blob']}.tif", foot) for c in cons["constraints"]]
    words = CE.pack_halos(halos)
    del halos
    return words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", action="store_true")
    ap.add_argument("--min-fwd", type=float, default=0.60)
    ap.add_argument("--extra-rows", default=None)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--seeds", default="20,30")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--budget", type=int, default=37654)
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--candidate-factor", type=int, default=40, help="candidate pool = factor x budget; 10 (emit77 default) leaves the budget unfilled under the caps, so the halo fraction of EMITTED dots exceeds the cap")
    args = ap.parse_args()
    if args.constraints:
        choose_constraints(args.min_fwd, Path(args.extra_rows) if args.extra_rows else None)
    cons = json.loads((ROOT / "registry" / "h79_capped_constraints.json").read_text())
    grid = load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    h, w = foot.shape
    words = load_words(cons, foot)
    K = cons["n_constraints"]
    log(f"packed {K} halos into {words.shape[0]} words")

    if args.holdout:
        for seed in [int(s) for s in args.seeds.split(",")]:
            folds, _ = HO.buffered_component_draw(grid, seed=seed, hide_frac=0.20, buffer_px=3)
            visible = folds[0]["visible"]
            dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
            allowed, tot = [], 0
            for fold in folds:
                a = fold["region"] & foot & ~visible & (dist_vis > 3.0)
                allowed.append(a)
                tot += int(a.sum())

            def emit(p, capped):
                dots = np.zeros((h, w), bool)
                counts = np.zeros(K, np.int64)
                for a in allowed:
                    b = int(round(args.budget * int(a.sum()) / max(tot, 1)))
                    if b <= 0:
                        continue
                    if capped:
                        d, c = CE.emit_dots_capped(p, a, b, words, np.floor(CAP_FRAC * b) * np.ones(K), candidate_factor=args.candidate_factor)
                        counts += c
                    else:
                        d = emit77.emit_dots(p, a, b)
                    dots |= d
                return dots, counts

            def score(dots):
                terms = None
                for fold in folds:
                    _, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
                    terms = t if terms is None else terms + t
                return terms

            terms_by_arm, details = {}, {}
            for arm in ("H79_full", "A2_DIST"):
                p = np.load(WORK / f"oof_{arm}_seed{seed}.npy")
                for capped in (False, True):
                    d, counts = emit(p, capped)
                    t = score(d)
                    key = f"{arm}@{'capped' if capped else 'greedy'}"
                    terms_by_arm[key] = t
                    nd = int(d.sum())
                    details[key] = {"dots": nd, "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0))),
                                    "max_halo_fraction_among_constraints": float(counts.max() / max(nd, 1)) if capped else None}
                    log(f"[seed {seed} B={args.budget}] {key:18s} HOLDOUT-DTI={details[key]['pooled_holdout_dti']:.5f} dots={nd}")
                    if arm == "H79_full" and capped:
                        np.save(WORK / f"dots_H79_full_capped_seed{seed}_B{args.budget}.npy", np.flatnonzero(d.ravel()).astype(np.int32))
            pooled = EH.pooled_summary(terms_by_arm, draws=2000, seed=520830 + seed, candidate="H79_full@capped")
            for ctrl, drec in pooled["paired_differences"].items():
                log(f"  -> H79_full@capped vs {ctrl:18s}: delta={drec['delta']:+.5f} CI95=[{drec['ci95'][0]:+.5f}, {drec['ci95'][1]:+.5f}]")
            out = {"lane": "H79-ATTR-HANGINGWALL", "amendment": "E3b uniqueness-capped emission", "evidence_class": "HOLDOUT-DTI",
                   "evaluator_version": EH.VERSION, "seed": seed, "budget": args.budget, "cap_fraction": CAP_FRAC, "n_constraints": K, "candidate_factor": args.candidate_factor,
                   "withheld_positive_pixels": int(sum(int(f["truth"].sum()) for f in folds)), "arms": details, "pooled_summary": pooled,
                   "status": "declared in registry/preregistration_h79_amendment_e3b.json before running; placement-only change"}
            (EVID / f"h79_capped_holdout_seed{seed}.json").write_text(json.dumps(out, indent=1, sort_keys=True))
            log(f"wrote evidence/h79_capped_holdout_seed{seed}.json")

    if args.build:
        assert args.name and args.note
        surf = np.load(WORK / "surface_full.npy")
        dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
        allowed = foot & ~cat & (dist_cat > 3.0)
        dots, counts = CE.emit_dots_capped(surf, allowed, args.budget, words, np.floor(CAP_FRAC * args.budget) * np.ones(K), candidate_factor=args.candidate_factor)
        final = np.zeros((h, w), np.float32)
        final[dots] = 1.0
        np.save(WORK / "final_dots_capped_flat.npy", np.flatnonzero(dots.ravel()).astype(np.int32))
        nd = int(dots.sum())
        log(f"capped emission: {nd} dots; max halo fraction among constraints {counts.max()/nd:.4f}; min dist to catalogue {float(dist_cat[dots].min()):.2f} px")
        out_tif = ROOT / "docs" / "downloads" / f"{args.name}.tif"
        receipt = SW.write_submission(out_tif, final, RAW / "sample_submission.tif", foot, note=args.note, name=args.name, catalogue=cat,
                                      metadata={"lane": "H79-ATTR-HANGINGWALL", "budget": args.budget,
                                                "emission": f"gems79.capped_emit greedy min_sep 2.8, excl dist_cat > 3 px, cap {CAP_FRAC} vs {K} halos"})
        log(f"wrote {out_tif} sha256={receipt['sha256']} bytes={receipt['bytes']}")
        res = {"tif": str(out_tif.relative_to(ROOT)), "sha256": receipt["sha256"], "zip_sha256": receipt["zip_sha256"], "dots": nd,
               "budget": args.budget, "name": args.name, "note": args.note, "note_chars": len(args.note), "validator": receipt["validator"],
               "template_report": receipt["template_report"], "surface_sha256": hashlib.sha256(np.ascontiguousarray(surf).tobytes()).hexdigest(),
               "cap_fraction": CAP_FRAC, "n_constraints": K, "candidate_factor": args.candidate_factor, "max_halo_fraction_emitted": float(counts.max() / nd), "halo_fraction_per_constraint": {c["blob"]: float(counts[i] / nd) for i, c in enumerate(cons["constraints"])},
               "emission": "capped greedy (amendment E3b)"}
        (EVID / "h79_build_capped.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
        log("wrote evidence/h79_build_capped.json")


if __name__ == "__main__":
    main()
