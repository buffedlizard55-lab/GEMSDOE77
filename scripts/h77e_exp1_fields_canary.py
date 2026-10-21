#!/usr/bin/env python3
"""H77E Experiment 1 of 3: build label-free UCP/Euler/texture fields + leakage canaries.

Canary protocol (parallel-run rule 4): every new field alone, symmetric
ROC-AUC of withheld-positive vs catalogue-far negatives on the seed-20
buffered whole-component draw. Symmetric AUC > 0.90 = leakage, hard stop.
Also records raw-band canaries for context. Writes evidence/h77e_e1_fields_canary.json
and caches footprint-compacted field matrices in .cache/h77e/.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

import h77e_common as C  # noqa: E402
from gems57 import holdout as HO  # noqa: E402
from gems77e import fields as F  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    t0 = time.time()
    grid = C.load_grid()
    foot, cat = grid.footprint, grid.catalogue
    got = C.verify_band_order()
    log("band descriptions:", got)

    C.CACHE.mkdir(parents=True, exist_ok=True)
    cache_new = C.CACHE / "new_fp.npy"
    names_new = C.CACHE / "new_names.json"
    diag = {}
    if cache_new.exists() and names_new.exists() and json.loads(names_new.read_text()) == F.NEW_NAMES:
        log("new fields: cache present")
        new_fp = np.load(cache_new, mmap_mode="r")
    else:
        rtp = C.read_band("rtp")
        grav = C.read_band("iso_grav_anom")
        fields, diag = F.build_new_fields(rtp, grav, foot)
        del rtp, grav
        gc.collect()
        n_fp = int(foot.sum())
        mm = np.lib.format.open_memmap(cache_new, mode="w+", dtype=np.float32,
                                       shape=(n_fp, len(F.NEW_NAMES)))
        for j, nm in enumerate(F.NEW_NAMES):
            mm[:, j] = fields[nm][foot]
            mm.flush()
        del mm, fields
        gc.collect()
        names_new.write_text(json.dumps(F.NEW_NAMES) + "\n")
        new_fp = np.load(cache_new, mmap_mode="r")
        log("new fields built:", diag)

    std_fp, std_names = C.build_std_fp(foot)
    log("std matrix:", std_fp.shape)

    # --- leakage canaries on seed-20 draw ---
    folds, _ = HO.buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]
    h, w = foot.shape
    fp_flat = np.flatnonzero(foot)
    fp_lut = np.full(h * w, -1, dtype=np.int32)
    fp_lut[fp_flat] = np.arange(fp_flat.size, dtype=np.int32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    rng = np.random.default_rng(1020)
    pos_fp = fp_lut[np.flatnonzero(hidden & foot)]
    neg_pool = fp_lut[np.flatnonzero(foot & ~cat & (dist_cat > 3))]
    neg_fp = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
    can = np.concatenate([pos_fp, neg_fp])
    y = np.concatenate([np.ones(pos_fp.size), np.zeros(neg_fp.size)])
    log(f"canary rows: pos={pos_fp.size} neg={neg_fp.size} withheld_total={int(hidden.sum())}")

    def canaries(mat, names):
        Xc = np.asarray(mat[can], dtype=np.float32)
        out = {}
        for j, nm in enumerate(names):
            v = Xc[:, j]
            m = np.isfinite(v)
            if m.sum() > 100 and len(np.unique(y[m])) == 2 and np.ptp(v[m]) > 0:
                a = float(roc_auc_score(y[m], v[m]))
                out[nm] = {"auc": a, "sym_auc": max(a, 1.0 - a)}
            else:
                out[nm] = {"auc": None, "sym_auc": None}
        return out

    can_new = canaries(new_fp, F.NEW_NAMES)
    can_std = canaries(std_fp, std_names)
    worst_new = max((v["sym_auc"] or 0, k) for k, v in can_new.items())
    worst_std = max((v["sym_auc"] or 0, k) for k, v in can_std.items())
    flags = [k for k, v in can_new.items() if (v["sym_auc"] or 0) > 0.90]
    log(f"worst new canary: {worst_new}; worst std canary: {worst_std}; flags={flags}")

    res = {
        "experiment": "H77E-E1",
        "evidence_class": "CANARY-DIAGNOSTIC (not a score)",
        "evaluator_version": "gems57-pooled-hide-v2 (split only, for canary sampling)",
        "split": "buffered_component_draw seed=20 hide_frac=0.20 buffer_px=3",
        "withheld_positive_pixels": int(hidden.sum()),
        "canary_negatives": int(neg_fp.size),
        "band_order_verified": got,
        "euler_diag": diag.get("euler") if diag else "cached (see first build)",
        "canaries_new": can_new,
        "canaries_std": can_std,
        "worst_new_sym_auc": worst_new[0], "worst_new_channel": worst_new[1],
        "worst_std_sym_auc": worst_std[0], "worst_std_channel": worst_std[1],
        "leakage_flags_gt_090": flags,
        "elapsed_s": time.time() - t0,
    }
    (ROOT / "evidence" / "h77e_e1_fields_canary.json").write_text(json.dumps(res, indent=2) + "\n")
    log("wrote evidence/h77e_e1_fields_canary.json")
    if flags:
        print("HARD STOP: leakage canary > 0.90", flags)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
