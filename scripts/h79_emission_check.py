#!/usr/bin/env python3
"""Post-hoc (NOT pre-registered) emission comparison for lane H79 on cached OOF surfaces.

The pre-registered E2 emission is the greedy `emit77.emit_dots`. The two incumbents whose
numbers the decision rule cites (H77D 0.1403 / H77-Step 0.1527, seed 20) were emitted with
the shared submodular 300 m allocator (`gems77d.submodular.allocate_submodular_300m`,
H77D parameters). To compare like with like this script scores BOTH emissions for the
candidate and its paired control on the same folds. Results are labelled post-hoc in the
run card; the greedy numbers remain the pre-registered ones.
"""
from __future__ import annotations

import argparse
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
from gems57.grid import load_grid  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems77d.submodular import allocate_submodular_300m  # noqa: E402

RAW = ROOT / "data" / "raw"
WORK = ROOT / ".cache" / "work" / "h79"


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--arms", default="H79_full,A2_DIST")
    ap.add_argument("--budget", type=int, default=37654)
    args = ap.parse_args()
    grid = load_grid(RAW)
    foot = grid.footprint
    h, w = foot.shape
    folds, _ = HO.buffered_component_draw(grid, seed=args.seed, hide_frac=0.20, buffer_px=3)
    visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    allowed, tot = [], 0
    for fold in folds:
        a = fold["region"] & foot & ~visible & (dist_vis > 3.0)
        allowed.append(a)
        tot += int(a.sum())

    def emit(p, method):
        dots = np.zeros((h, w), bool)
        for a in allowed:
            b = int(round(args.budget * int(a.sum()) / max(tot, 1)))
            if b <= 0:
                continue
            if method == "greedy":
                dots |= emit77.emit_dots(p, a, b)
            else:
                dots |= allocate_submodular_300m(p, a, target_n=b, support_quantile=0.968, gamma=3.5, min_sep_px=2.20)
        return dots

    def score(dots):
        terms = None
        for fold in folds:
            _, t = EH.evaluate(dots.astype(np.float32), fold, valid=foot, block_side=200)
            terms = t if terms is None else terms + t
        return terms

    terms_by_arm, details = {}, {}
    for arm in args.arms.split(","):
        p = np.load(WORK / f"oof_{arm}_seed{args.seed}.npy")
        for method in ("greedy", "submod"):
            d = emit(p, method)
            t = score(d)
            key = f"{arm}@{method}"
            terms_by_arm[key] = t
            details[key] = {"dots": int(d.sum()), "pooled_holdout_dti": float(EH.from_terms(t.sum(axis=0)))}
            log(f"[seed {args.seed} B={args.budget}] {key:22s} HOLDOUT-DTI={details[key]['pooled_holdout_dti']:.5f} dots={int(d.sum())}")
            if arm == "H79_full" and method == "submod":
                np.save(WORK / f"dots_H79_full_submod_seed{args.seed}_B{args.budget}.npy", np.flatnonzero(d.ravel()).astype(np.int32))
    pooled = EH.pooled_summary(terms_by_arm, draws=2000, seed=520810, candidate="H79_full@submod")
    for ctrl, drec in pooled["paired_differences"].items():
        log(f"  -> H79_full@submod vs {ctrl:22s}: delta={drec['delta']:+.5f} CI95=[{drec['ci95'][0]:+.5f}, {drec['ci95'][1]:+.5f}]")
    out = {"lane": "H79-ATTR-HANGINGWALL", "evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
           "status": "POST-HOC emission comparison (not pre-registered); greedy numbers are the pre-registered ones",
           "seed": args.seed, "budget": args.budget, "withheld_positive_pixels": int(sum(int(f["truth"].sum()) for f in folds)),
           "arms": details, "pooled_summary": pooled}
    path = ROOT / "evidence" / f"h79_emission_check_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
