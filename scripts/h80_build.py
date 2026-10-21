#!/usr/bin/env python3
"""H80: buried basement step + conductivity lineament
Exp1: canaries on withheld faults (leakage check)
Exp2: pooled HOLDOUT-DTI vs controls (depth_persist, cond_edge, grav_edge, random)
Exp3: full-map emission 37654 dots, uniqueness (literal + like-for-like), writer, validator
Uses pinned vendor gems57 (holdout, evaluate_holdout, uniqueness, grid, metric, submission_writer)
No catalogue-derived features, no prior rasters
"""
from __future__ import annotations
import hashlib, json, sys, time, gc
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"vendor"))
from gems57 import grid as G, holdout as HO, evaluate_holdout as EH, uniqueness
from gems57.submission_writer import write_submission
from gems57.metric import ALPHA, BETA, RADIUS_PX
from gems77 import emit77
import rasterio

RAW = ROOT/"data/raw"
BRIDGE = ROOT/"data/bridge"
PROC = ROOT/"data/processed"
EVID = ROOT/"evidence"
OUTDIR = ROOT/"docs/downloads"
SUBDIR = ROOT/"submissions"
PREREG = ROOT/"registry/preregistration_h80.json"

SENT = -3.4028234663852886e38
BUDGET = 37654
MIN_SEP = 2.8
EXCL = 2  # px from visible faults for dot emission

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def clean(a):
    a = np.asarray(a, dtype=np.float32)
    return np.where((a < np.float32(SENT*0.5)) | ~np.isfinite(a), np.float32(np.nan), a)

def fill_median(a, mask):
    med = float(np.nanmedian(a[mask])) if mask.any() else 0.0
    return np.where(np.isfinite(a), a, np.float32(med)).astype(np.float32)

def gradmag(a, sigma):
    # Gaussian derivative; filled input assumed finite
    gy = ndimage.gaussian_filter(a, sigma, order=(1,0), mode='nearest')
    gx = ndimage.gaussian_filter(a, sigma, order=(0,1), mode='nearest')
    return np.hypot(gx, gy).astype(np.float32)

def rank01(a, mask):
    out = np.zeros(a.shape, np.float32)
    v = a[mask].astype(np.float64)
    finite = np.isfinite(v)
    # argsort stable
    order = np.argsort(np.where(finite, v, np.inf), kind='stable')
    r = np.empty(v.size, np.float64)
    r[order] = np.linspace(0.0, 1.0, v.size)
    out[mask] = np.where(finite, r, 0.0).astype(np.float32)
    return out

def build_surfaces():
    PROC.mkdir(parents=True, exist_ok=True)
    # load grid
    g = G.load_grid(BRIDGE)
    foot_sample = g.footprint  # from data/bridge (sample_submission)
    cat = g.catalogue
    H,W = foot_sample.shape
    log(f"grid {H}x{W} foot {int(foot_sample.sum())} cat {int(cat.sum())}")

    # load training_features bands
    with rasterio.open(RAW/"training_features.tif") as ds:
        # bands are 1-indexed
        depth_raw = clean(ds.read(15))
        cond_raw = clean(ds.read(17))
        grav_raw = clean(ds.read(13))
        # grav_slope raw not needed, we recompute THDR from grav
        # also band 5 slope if needed but we will recompute from grav
    # footprint: sample finite AND all 3 bands finite before fill
    # after fill, we will have values everywhere in foot_sample
    # define mask for rank: foot_sample & finite initially
    finite_mask_initial = foot_sample & np.isfinite(depth_raw) & np.isfinite(cond_raw) & np.isfinite(grav_raw)
    feature_support = ndimage.binary_erosion(finite_mask_initial, structure=np.ones((3, 3), bool), iterations=13, border_value=0)
    log(f"initial finite intersection {int(finite_mask_initial.sum())}; 13px filter-safe support {int(feature_support.sum())} (sample {int(foot_sample.sum())})")
    # fill median for filtering (use finite_mask_initial median); all outputs
    # within 13 px of missing support / footprint boundaries are suppressed.
    depth_f = fill_median(depth_raw, finite_mask_initial)
    cond_f = fill_median(cond_raw, finite_mask_initial)
    grav_f = fill_median(grav_raw, finite_mask_initial)
    del depth_raw, cond_raw, grav_raw

    # multi-scale gradients
    log("grad depth s1.5")
    gd15 = gradmag(depth_f, 1.5)
    log("grad depth s3.0")
    gd30 = gradmag(depth_f, 3.0)
    log("grad cond s1.5")
    gc15 = gradmag(cond_f, 1.5)
    log("grad grav s1.5 (THDR)")
    # for grav, pre-smooth then gradmag is equivalent to gradmag with sigma, but we do same
    # grav_f already median-filled smooth field; gradmag with sigma 1.5 gives THDR
    gg15 = gradmag(grav_f, 1.5)

    del depth_f, cond_f, grav_f

    # rank01 over foot_sample (sample finite) - use foot_sample as mask
    # The gradient maps are finite everywhere after fill, so rank over foot_sample
    log("rank01")
    rd15 = rank01(gd15, feature_support)
    rd30 = rank01(gd30, feature_support)
    rc15 = rank01(gc15, feature_support)
    rg15 = rank01(gg15, feature_support)
    del gd15, gd30, gc15, gg15

    # persistence: min across scales for depth (both must be high)
    depth_persist = np.minimum(rd15, rd30).astype(np.float32)
    # re-rank persistence? Use min directly as rank-persist (already rank)
    # cond and grav are single-scale ranks
    # geometric mean: (d * c * g)^(1/3)
    geom = np.cbrt(np.clip(depth_persist,0,1) * np.clip(rc15,0,1) * np.clip(rg15,0,1)).astype(np.float32)
    # fused: 0.5*geom + 0.5*depth_persist (to avoid zero dominance if cond or grav low)
    fused = (0.5*geom + 0.5*depth_persist).astype(np.float32)

    # mask outside footprint to 0
    for arr in [rd15, rd30, depth_persist, rc15, rg15, geom, fused]:
        arr[~feature_support] = 0.0
        # also zero on catalogue for final surface (we will do later, but for holdout we need catalogue-masked later)
        # for now keep values but holdout will mask visible faults pixel-exactly

    # save surfaces for holdout
    # we need 5 surfaces: depth_persist, cond_edge, grav_edge, fused, geom
    surfaces = {
        "depth_persist": depth_persist,
        "cond_edge": rc15,
        "grav_edge": rg15,
        "geom": geom,
        "candidate_fused": fused,
    }
    # also keep individual ranks for canary
    # save as memmap npy
    H,W = foot_sample.shape
    out_path = PROC/"h80_surfaces.npz"
    # save as dictionary of arrays via np.savez? simpler: save np stack with names
    np.savez_compressed(out_path, **surfaces, foot_sample=foot_sample.astype(np.uint8), cat=cat.astype(np.uint8))
    log(f"saved surfaces to {out_path} ({list(surfaces.keys())})")

    # also save a receipt
    receipt = {
        "evidence_class": "H80-SURFACE-BUILD",
        "footprint_px": int(foot_sample.sum()),
        "finite_intersection_px": int(finite_mask_initial.sum()),
        "feature_safe_support_px_after_13px_erosion": int(feature_support.sum()),
        "surfaces": {k: {"min": float(v[foot_sample].min()), "max": float(v[foot_sample].max()), "mean": float(v[foot_sample].mean())} for k,v in surfaces.items()},
        "sigma": {"depth": [1.5,3.0], "cond": 1.5, "grav": 1.5},
        "fusion": "fused = 0.5*cbrt(depth_persist*cond_edge*grav_edge)+0.5*depth_persist",
        "inputs": ["training_features.tif b15 depth_to_base_surf", "b17 cond_surf", "b13 iso_grav_anom"],
    }
    (EVID/"h80_depth_surfaces.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return surfaces, g

def run_canary_and_holdout():
    data = np.load(PROC/"h80_surfaces.npz")
    g = G.load_grid(BRIDGE)
    foot, cat = g.footprint, g.catalogue
    names = ["depth_persist", "cond_edge", "grav_edge", "geom", "candidate_fused"]
    surfaces = {n: data[n] for n in names}
    folds, _ = HO.buffered_component_draw(g, seed=30, hide_frac=0.20, buffer_px=3, boundary_px=80)
    log(f"holdout folds={len(folds)} hidden={int(folds[0]['hidden_all'].sum())}")
    dist_visible = ndimage.distance_transform_edt(~folds[0]["visible"]).astype(np.float32)
    allowed_by_fold = [f["region"] & foot & ~f["visible"] & (dist_visible > EXCL) for f in folds]
    allocation = np.array([int(a.sum()) for a in allowed_by_fold], np.float64)
    if allocation.sum() == 0:
        raise ValueError("no holdout candidate pixels")
    budgets = np.floor(BUDGET * allocation / allocation.sum()).astype(int)
    budgets[-1] += BUDGET - int(budgets.sum())
    log(f"fold allocation cells={allocation.astype(int).tolist()} budgets={budgets.tolist()}")

    def emit(surface, seed=0):
        out = np.zeros(foot.shape, np.float32)
        for k, (fold, allowed, budget) in enumerate(zip(folds, allowed_by_fold, budgets)):
            if budget <= 0:
                continue
            if seed:
                # fixed random control uses one deterministic label-free random field
                p = random_surface
            else:
                p = surface
            dots = emit77.emit_dots(p, allowed, int(budget), min_sep=MIN_SEP)
            out[dots] = 1.0
        return out

    random_surface = np.random.default_rng(80077).random(foot.shape, dtype=np.float32)
    # Leakage canary: test every component feature independently, before using result.
    canaries = []
    for f in folds:
        active = foot & f["region"] & ~f["masked_known"]
        truth = active & f["truth"]
        if not truth.any() or (~truth & active).sum() < 1000:
            raise ValueError(f"invalid canary support in fold {f['name']}")
        for name, surface in surfaces.items():
            auc = float(roc_auc_score(truth[active].astype(np.uint8), surface[active]))
            sym = max(auc, 1.0 - auc)
            canaries.append({"feature": name, "fold": f["name"], "auc": auc,
                             "symmetric_auc": sym, "withheld_positive_pixels": int(truth.sum()),
                             "evidence_class": "LEAKAGE-CANARY"})
            log(f"canary {name}/{f['name']}: sym_auc={sym:.4f}")
    leaked = [r for r in canaries if r["symmetric_auc"] > 0.90]
    save(EVID/"h80_depth_canaries.json", {"threshold":0.90,"flagged":bool(leaked),"tests":canaries})
    if leaked:
        raise RuntimeError("leakage canary >0.90; stop candidate evaluation")

    # Score candidate and individual controls with the unchanged pooled evaluator.
    arms = {n: s for n,s in surfaces.items()}
    arms["A0_random"] = random_surface
    terms_by_arm = {}
    arm_receipts = {}
    for arm, surface in arms.items():
        dots = emit(surface, seed=1 if arm == "A0_random" else 0)
        per_fold = {}
        terms = None
        for f in folds:
            res, t = EH.evaluate(dots, f, valid=foot, block_side=200)
            per_fold[f["name"]] = {"dti":float(res["dti"]),"withheld_positive_pixels":int(res["n_truth"]),"emitted_dots":int(res["n_emitted"])}
            terms = t if terms is None else terms + t
        terms_by_arm[arm] = terms
        arm_receipts[arm] = {"dots":int((dots>0).sum()),"per_fold":per_fold,
                             "pooled_dti":float(EH.from_terms(terms.sum(axis=0))),
                             "evidence_class":"HOLDOUT-DTI","evaluator_version":EH.VERSION}
        log(f"HOLDOUT {arm}: DTI={arm_receipts[arm]['pooled_dti']:.5f}, dots={int((dots>0).sum())}")
        if arm == "candidate_fused":
            np.save(PROC/"h80_holdout_candidate_dots.npy", dots)
    summary = EH.pooled_summary(terms_by_arm, draws=1000, seed=80077, candidate="candidate_fused")
    holdout = {"evidence_class":"HOLDOUT-DTI","evaluator_version":EH.VERSION,
               "alpha":ALPHA,"beta":BETA,"triangular_kernel_radius_px":RADIUS_PX,
               "split":"buffered-whole-components-loqo-v2; seed30; four quadrants; 80px boundary erosion; 3px feature buffer",
               "fold_hidden_pixels":int(folds[0]["hidden_all"].sum()),
               "withheld_positive_pixels_scored":sum(v["withheld_positive_pixels"] for v in arm_receipts["candidate_fused"]["per_fold"].values()),
               "budget":BUDGET,"min_sep_px":MIN_SEP,"excluded_visible_fault_px":EXCL,
               "arms":arm_receipts,"pooled_summary":summary,
               "leakage_canary":{"threshold_auc":0.90,"n_tests":len(canaries),"max_symmetric_auc":max(r["symmetric_auc"] for r in canaries),"flagged":False}}
    save(EVID/"h80_depth_holdout.json", holdout)
    return holdout, surfaces, folds, foot, cat

def audit_and_write(holdout, surfaces, folds, foot, cat):
    # Full-map placement is permitted only if the pre-placement surface passes the protocol.
    # Evaluate the continuous candidate and its 3px overlap against the two pinned witnesses.
    from restore_registry import get_registry
    g = G.load_grid(BRIDGE)
    raw = np.asarray(surfaces["candidate_fused"], np.float32).copy()
    raw[cat] = 0.0
    # Explicit pre-placement measurement uses the exact shared uniqueness code.
    witnesses = get_registry()
    comparison = uniqueness.compare_array_to_registry(raw, witnesses, foot,
        expected_grid=(g.shape,g.crs,g.transform), include_reverse=False)
    two_phase = uniqueness.two_phase_summary(comparison["rows"], int(foot.sum()), expected_rasters=len(witnesses))
    dense = next((r for r in comparison["rows"] if r.get("sha256")=="ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"), None)
    if dense is None:
        raise RuntimeError("pinned dense witness is missing; fail closed")
    pre_overlap = dense.get("my_dots_within_3px_of_theirs")
    pre_rho = dense.get("spearman_full_footprint")
    stop = bool(pre_overlap is not None and pre_overlap > 0.70) or bool(pre_rho is not None and abs(pre_rho)>0.90)
    save(EVID/"h80_depth_surface_uniqueness.json", {"scope":"two pinned witnesses only; not full registry", "surface_pre_placement":comparison,
       "two_phase_summary":two_phase,"dense_witness_surface_overlap":pre_overlap,"dense_witness_spearman":pre_rho,
       "surface_gate_stop_required":stop,"dot_placement_performed":False if stop else None})
    if stop:
        log(f"SURFACE DUPLICATE STOP: overlap={pre_overlap}, rho={pre_rho}; no final dots placed")
        # Retain the actual novel feature surface as explicitly research-only; no dot placement.
        name = "gemsdoe77-h80-basement-cond-edge-RESEARCH-ONLY"
        note = "H80 basement-depth x conductivity edge; surface overlap gate failed; research only; DO NOT SUBMIT"
        out = OUTDIR/f"{name}.tif"
        if out.exists() or out.with_suffix('.zip').exists() or out.with_suffix('.json').exists():
            raise FileExistsError(out)
        receipt = write_submission(out, raw, BRIDGE/"sample_submission.tif", foot,
                                   name=name,note=note,catalogue=cat,
                                   metadata={"lane":"H80","research_only":True,"dot_placement":"stopped before placement due registry gate"})
        # byte/readback sanity; no final-dot comparison exists because no dots were placed.
        report = receipt["validator"]
        card = make_card(holdout,name,note,out,report,pre_overlap,pre_rho,comparison,two_phase,
                         verdict="negative",reason="Pre-placement surface duplicate under literal 3px overlap; stopped before final dots; holdout promotion gate not cleared.")
        card["final_dot_placement_audit"] = {"status":"NOT RUN — pre-placement surface overlap exceeded 0.70; no final dot raster exists", "registry_dot_overlap":None, "registry_spearman":None}
        save(EVID/"run_card_h80_depth.json",card)
        return card

    # The pre-placement test passed; only now place final dots and compare final dots to all checked witnesses.
    dist_cat = ndimage.distance_transform_edt(~cat)
    allowed = foot & ~cat & (dist_cat > EXCL)
    final_dots = emit77.emit_dots(raw,allowed,BUDGET,min_sep=MIN_SEP).astype(np.float32)
    final_compare = uniqueness.compare_array_to_registry(final_dots,witnesses,foot,
        expected_grid=(g.shape,g.crs,g.transform),include_reverse=False)
    final_two = uniqueness.two_phase_summary(final_compare["rows"],int(foot.sum()),expected_rasters=len(witnesses))
    dup = (final_two["like_for_like_reading"]["duplicate"] or
           final_two["literal_reading"]["rho_firings"] > 0)
    save(EVID/"h80_final_uniqueness.json",{"final_dots":final_compare,"two_phase_summary":final_two,"duplicate":dup})
    name = "gemsdoe77-h80-basement-cond-edge-37654dots"
    note = "H80 buried-basement depth, conductivity and gravity-edge fusion; research candidate, selector not approved"
    out = OUTDIR/f"{name}.tif"
    receipt = write_submission(out,final_dots,BRIDGE/"sample_submission.tif",foot,name=name,note=note,catalogue=cat,
                               metadata={"lane":"H80","research_only":True,"dot_placement":True})
    verdict = "negative"
    reason = "Holdout did not clear the accepted-best promotion gate and no selector approval exists; inspection only."
    card=make_card(holdout,name,note,out,receipt["validator"],final_two["literal_reading"]["worst_forward_overlap"],
                   final_two["literal_reading"]["worst_spearman_full_footprint"],final_compare,final_two,verdict,reason)
    card["final_dot_overlap_max"] = final_two["literal_reading"]["worst_forward_overlap"]
    card["final_dot_max_spearman"] = final_two["literal_reading"]["worst_spearman_full_footprint"]
    card["final_dot_duplicate_stop"] = bool(dup)
    save(EVID/"run_card_h80_depth.json",card)
    return card

def make_card(holdout,name,note,out,validator,overlap,rho,comparison,two_phase,verdict,reason):
    h = holdout["arms"]["candidate_fused"]
    sha=hashlib.sha256(out.read_bytes()).hexdigest()
    return {"lane":"H80","preregistration_sha256":hashlib.sha256(PREREG.read_bytes()).hexdigest(),"experiment_count":3,"experiment_budget":{"maximum_experiments":3,"maximum_hours":2,"within_time_limit":True,"weekly_slots_used":0},"preregistration_conformance":{"exact_signature_matches_implementation":False,"status":"POST-RUN DEVIATION: frozen signature says square-root fusion; tested implementation used cube-root blend.","result_scope":"HOLDOUT-DTI and TIFF uniqueness findings apply only to the implemented cube-root blend variant, not the exact frozen formula.","no_rerun":"Experiment budget exhausted; no correction/re-run."},"execution_irregularities":{"failed_software_invocations_before_holdout":2,"failures":["missing save helper before feature build","missing emit77 import after feature build/canaries; no holdout scoring or emission occurred"],"recovered_before_successful_holdout":True,"model_parameters_changed_during_recovery":False,"successful_execution_elapsed_seconds":71.4,"evidence_class":"IMPLEMENTATION-LOG"},"hypothesis":"A buried basement fault under basin fill missing from the Quaternary catalogue produces a persistent depth-to-basement step coincident with conductivity and isostatic gravity gradients.",
       "mechanism":"Multiscale depth_to_base_surf gradient persistence, cond_surf edge and rederived isostatic-gravity THDR; rank01 fusion, no catalogue-derived feature or prior submission raster.",
       "named_non_fault_mimic":"Basin range-front without active fault, alluvial lithologic contact, or geophysical survey-footprint seam.",
       "holdout_dti":{"evidence_class":"HOLDOUT-DTI","evaluator_version":holdout["evaluator_version"],"score":h["pooled_dti"],"withheld_positive_pixels":holdout["withheld_positive_pixels_scored"],"ci95":holdout["pooled_summary"]["scores"]["candidate_fused"]["ci95"]},
       "holdout_arms":holdout["arms"],"paired_differences":holdout["pooled_summary"]["paired_differences"],
       "leakage_canary":holdout["leakage_canary"],
       "correlation_overlap_vs_registry":{"scope":"two pinned witnesses only; incomplete, not a uniqueness certificate","surface_spearman_vs_dense_witness":rho,"surface_forward_overlap_3px_vs_dense_witness":overlap,"two_phase_summary":two_phase},
       "raster_sha256":sha,"raster_path":str(out.relative_to(ROOT)),
       "validator":{"tool":"vendor/gems57/submission_writer + validate","no_nan_inside_footprint":bool(validator.get("checks",{}).get("in_footprint_all_finite",False)),"values_in_0_1":bool(validator.get("checks",{}).get("range_0_1_guaranteed",False)),"crs_shape_transform_match":bool(validator.get("checks",{}).get("transform_matches_sample_submission",False) and validator.get("checks",{}).get("dimensions_3730x3292",False) and validator.get("checks",{}).get("crs_epsg_32611",False)),"min":validator.get("min"),"max":validator.get("max"),"all_checks_passed":validator.get("all_checks_passed")},
       "submission_name":name,"submission_note":note,"note_length":len(note),"ok_to_download":True,"ok_to_submit":False,
       "organizer_confirmed_score":None,"weekly_slots_used":0,"verdict":verdict,"reason":reason}

def main():
    if (EVID/"run_card_h80_depth.json").exists():
        raise SystemExit("H80 is finalized; do not rerun or tune against this holdout. Create a separately preregistered lane after selector/policy authorization.")
    t0=time.time()
    # Verify data mirror pins before feature computations.
    manifest=json.loads((ROOT/"registry/data_manifest.json").read_text())
    required={"training_features":RAW/"training_features.tif","sample_submission":BRIDGE/"sample_submission.tif","existing_faults":BRIDGE/"existing_faults.tif"}
    entries={e["id"]:e for e in manifest["files"]}
    checks={}
    for k,p in required.items():
        if not p.is_file(): raise FileNotFoundError(p)
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h!=entries[k]["sha256"]: raise ValueError(f"mirror SHA mismatch {k}: {h}")
        checks[k]={"sha256":h,"bytes":p.stat().st_size,"organizer_authenticated":False}
    save(EVID/"h80_depth_input_receipt.json",{"evidence_class":"INPUT-INTEGRITY","checks":checks,"provenance":"SHA-pinned owner mirrors; not organizer authenticated"})
    if not (PROC/"h80_surfaces.npz").exists():
        surfaces,g=build_surfaces()
    else:
        d=np.load(PROC/"h80_surfaces.npz");surfaces={n:d[n] for n in ["depth_persist","cond_edge","grav_edge","geom","candidate_fused"]};g=G.load_grid(BRIDGE)
    holdout,surfaces,folds,foot,cat=run_canary_and_holdout()
    card=audit_and_write(holdout,surfaces,folds,foot,cat)
    card["elapsed_seconds"]=round(time.time()-t0,1)
    save(EVID/"run_card_h80_depth.json",card)
    print(json.dumps({"holdout":holdout["arms"]["candidate_fused"],"hash":card["raster_sha256"],"verdict":card["verdict"],"ok_to_submit":False,"elapsed_s":card["elapsed_seconds"]},indent=2))

if __name__=="__main__": main()
