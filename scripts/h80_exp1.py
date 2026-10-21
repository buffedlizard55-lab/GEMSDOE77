#!/usr/bin/env python3
"""H80-OTA Experiment 1 of 3: OTA fields (visible-orphan), leakage canaries,
surface-alone holdout screens. Frozen design: registry/preregistration_h80.json.

Evaluator/emitter are the shared template tools (vendor gems57, gems77.emit77),
unchanged. Every catalogue-derived quantity uses fold['visible'] only.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH  # noqa: E402
from gems57 import metric as M  # noqa: E402
from gems77 import emit77  # noqa: E402
from gems77h80 import fields as OTA  # noqa: E402

RAW = ROOT / "data" / "raw"; PROC = ROOT / "data" / "processed"
BUDGET, EXCL = 37_654, 2
SENT = -3.4028234663852886e38


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_band(i):
    with rasterio.open(RAW / "training_features.tif") as ds:
        a = ds.read(i).astype(np.float32)
    return np.where((a < SENT * 0.5) | ~np.isfinite(a), np.float32(np.nan), a)


def main():
    t0 = time.time()
    grid = G.load_grid()  # data/bridge pinned files
    foot, cat = grid.footprint, grid.catalogue
    folds, _ = HO.buffered_component_draw(grid, seed=30, hide_frac=0.20, buffer_px=3)
    hidden = folds[0]["hidden_all"]; visible = folds[0]["visible"]
    dist_vis = ndimage.distance_transform_edt(~visible).astype(np.float32)
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    log(f"holdout seed30: withheld={int(hidden.sum())} visible={int(visible.sum())}")

    band3 = read_band(3); band5 = read_band(5)
    ota, diag = OTA.build_ota_fields(RAW / "external" / "gdr_wellspring_in_footprint.csv",
                                     foot, visible, band3, band5)
    del band3, band5; gc.collect()
    np.save(PROC / "ota_visible.npy", ota)
    log("OTA diag:", json.dumps(diag))

    # ---- leakage canaries (single-feature AUC, withheld pos vs sampled neg) ----
    rng = np.random.default_rng(1234)
    pos_idx = np.flatnonzero(hidden)
    neg_pool = np.flatnonzero(foot & ~cat & (dist_cat > 3))
    neg_idx = rng.choice(neg_pool, size=min(200_000, neg_pool.size), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])
    flat = ota.reshape(-1, ota.shape[2])
    Xc = np.asarray(flat[idx], dtype=np.float32); del flat
    can = {}
    for j, n in enumerate(OTA.CHANNELS):
        v = Xc[:, j]; m = np.isfinite(v)
        auc = float(roc_auc_score(y[m], v[m])) if m.sum() > 100 else None
        can[n] = {"auc": auc, "symmetric_auc": max(auc, 1 - auc) if auc is not None else None}
        log(f"canary {n}: AUC={auc}")
    del Xc; gc.collect()
    leak = [n for n, r in can.items() if r["symmetric_auc"] and r["symmetric_auc"] > 0.90]

    def emit_pooled(p):
        tots = [int((f["region"] & foot & ~visible & (dist_vis > EXCL)).sum()) for f in folds]
        tot = max(sum(tots), 1); dots = np.zeros(foot.shape, bool)
        for f, a_n in zip(folds, tots):
            b = int(round(BUDGET * a_n / tot))
            if b > 0:
                dots |= emit77.emit_dots(p, f["region"] & foot & ~visible & (dist_vis > EXCL), b)
        return dots

    def score(dots):
        terms = None; per = {}
        for f in folds:
            res, t = EH.evaluate(dots.astype(np.float32), f, valid=foot, block_side=200)
            per[f["name"]] = {"dti": res["dti"], "n_truth": res["n_truth"], "emitted": res["n_emitted"]}
            terms = t if terms is None else terms + t
        return per, terms

    arms = {
        "S_ota_line": np.asarray(ota[:, :, 0], np.float32),
        "S_ota_xridge": np.asarray(ota[:, :, 1], np.float32),
        "S_orphan_prox": np.asarray(ota[:, :, 2], np.float32),
        "A0_random": np.random.default_rng(999).random(foot.shape, dtype=np.float32) * foot,
    }
    terms_by = {}
    res = {"evidence_class": "HOLDOUT-DTI", "evaluator_version": EH.VERSION,
           "alpha": M.ALPHA, "beta": M.BETA, "kernel_radius_m": M.R_M,
           "seed": 30, "withheld_positive_pixels": int(hidden.sum()),
           "budget": BUDGET, "exclusion_px": EXCL, "ota_diagnostics": diag,
           "canaries": can, "leakage_gt_0p90": leak, "arms": {}}
    for name, surf in arms.items():
        dots = emit_pooled(np.where(foot, surf, 0).astype(np.float32))
        per, terms = score(dots); terms_by[name] = terms
        res["arms"][name] = {"dots": int(dots.sum()), "per_fold": per,
                             "pooled_holdout_dti": float(EH.from_terms(terms.sum(axis=0)))}
        np.save(PROC / f"h80_dots_{name}.npy", dots)
        log(f"{name}: HOLDOUT-DTI={res['arms'][name]['pooled_holdout_dti']:.4f} dots={int(dots.sum())}")
    res["pooled_summary"] = EH.pooled_summary(terms_by, draws=1000, seed=520810, candidate="S_ota_xridge")
    res["verdict"] = "STOP-LEAKAGE" if leak else "screen-complete"
    res["seconds"] = round(time.time() - t0, 1)
    (ROOT / "evidence" / "h80_exp1.json").write_text(json.dumps(res, indent=2) + "\n")
    log("wrote evidence/h80_exp1.json")


if __name__ == "__main__":
    main()
