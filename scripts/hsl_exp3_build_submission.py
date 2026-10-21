#!/usr/bin/env python3
"""GEMSDOE77 Lane H77-HSL — Experiment 3:
Full-catalogue model fit, Phase-1 (surface) and Phase-2 (dots) uniqueness check
against all 808 registry rasters + 3 GEMSDOE77 rasters, shared submission_writer
GeoTIFF + ZIP generation, and 17/17 portal + pass-2 rasterio verification.
"""
from __future__ import annotations
import gc, hashlib, json, sys, time, zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage, stats
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from gems57 import grid as G, submission_writer as SW, uniqueness as U  # noqa: E402
from gems77hsl import features_hsl as FH, emit_hsl as EM  # noqa: E402
from gems77lia import validate as V17  # noqa: E402
from hsl_exp1_holdout import (  # noqa: E402
    HGB_KW,
    STRUCT_PHYS_COLS,
    arm_channels,
    extract_rows,
    log,
    make_inner_component_split,
)

RAW = ROOT / "data" / "raw"
PROC = ROOT / ".cache" / "hsl_processed"
REG_DIR = ROOT / ".cache" / "registry"
SEL_B = 37_654
SEL_E = 2
TIMESTAMP = "20261021T052500Z"


def predict_full_chw(stack_chw: np.ndarray, cols: list[int], clf, foot: np.ndarray, V_extra: np.ndarray | None = None) -> np.ndarray:
    H, W = foot.shape
    ys, xs = np.nonzero(foot)
    n_p = len(cols)
    n_v = 0 if V_extra is None else V_extra.shape[0]
    step = 400_000
    preds = np.empty(ys.size, dtype=np.float32)
    for i in range(0, ys.size, step):
        i1 = min(ys.size, i + step)
        by, bx = ys[i:i1], xs[i:i1]
        blk = np.empty((i1 - i, n_p + n_v), dtype=np.float32)
        for j, c in enumerate(cols):
            blk[:, j] = stack_chw[c, by, bx]
        if V_extra is not None:
            for j in range(n_v):
                blk[:, n_p + j] = V_extra[j, by, bx]
        preds[i:i1] = clf.predict_proba(blk)[:, 1].astype(np.float32)
    out = np.zeros((H, W), dtype=np.float32)
    out[ys, xs] = preds
    return out


def main():
    t0 = time.time()
    stack_chw = np.load(PROC / "hsl_stack.npy", mmap_mode="r")
    grid = G.load_grid(RAW)
    foot, cat = grid.footprint, grid.catalogue
    H, W = foot.shape
    dist_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)

    lid_path = RAW / "external" / "lidar_scarp_features_u8.tif"
    with rasterio.open(lid_path) as ds:
        step_u8 = ds.read(3).astype(np.float32) / np.float32(255.0)
        coh_u8 = ds.read(10).astype(np.float32) / np.float32(255.0)
        strike_u8 = ds.read(11).astype(np.float32)
        lvalid = (ds.read(12) > 0).astype(np.float32)
    theta_lid = (strike_u8 / np.float32(255.0)) * np.float32(np.pi)
    w_lid = coh_u8 * step_u8 * lvalid
    u2_lid = (w_lid * np.cos(np.float32(2.0) * theta_lid)).astype(np.float32)
    v2_lid = (w_lid * np.sin(np.float32(2.0) * theta_lid)).astype(np.float32)
    del step_u8, coh_u8, strike_u8, lvalid, theta_lid, w_lid
    gc.collect()

    # 1. Fit full-catalogue physical HGB ensemble (HSL_nolabel, 77 channels, 3 seeds)
    p_phys_path = PROC / "hsl_full_phys_surface.npy"
    cols_nolabel = [FH.IDX[n] for n in arm_channels("HSL_nolabel")]
    if p_phys_path.exists():
        p_phys = np.load(p_phys_path)
        log("Loaded cached hsl_full_phys_surface.npy")
    else:
        p_phys = np.zeros((H, W), dtype=np.float32)
        pos_all = np.flatnonzero(cat & foot)
        neg_mask = foot & ~cat & (dist_cat > 3)
        hard_all = np.flatnonzero(neg_mask & (dist_cat <= 15))
        easy_all = np.flatnonzero(neg_mask & (dist_cat > 15))
        for m_i, s_seed in enumerate((7701, 7702, 7703)):
            rng = np.random.default_rng(s_seed)
            pos = rng.choice(pos_all, size=min(45_000, pos_all.size), replace=False)
            n_neg = 2 * pos.size
            nh = min(hard_all.size, n_neg // 2)
            ne = min(easy_all.size, n_neg - nh)
            neg = np.concatenate([
                rng.choice(hard_all, nh, replace=False),
                rng.choice(easy_all, ne, replace=False),
            ])
            idx = np.concatenate([pos, neg])
            y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            X = extract_rows(stack_chw, cols_nolabel, idx)
            kw = {**HGB_KW, "random_state": s_seed}
            clf = HistGradientBoostingClassifier(**kw)
            t = time.time()
            clf.fit(X, y)
            del X
            gc.collect()
            p_m = predict_full_chw(stack_chw, cols_nolabel, clf, foot)
            p_phys += p_m / np.float32(3.0)
            log(f"Full HSL_nolabel member {m_i+1}/3 done ({time.time()-t:.1f}s)")
            del p_m, clf
            gc.collect()
        np.save(p_phys_path, p_phys)

    # 2. Fit full-catalogue structural continuation HGB ensemble (HSL_structural, 3 internal splits of cat)
    p_struct_path = PROC / "hsl_full_struct_surface.npy"
    cols_sphys = [FH.IDX[n] for n in STRUCT_PHYS_COLS]
    if p_struct_path.exists():
        p_struct = np.load(p_struct_path)
        log("Loaded cached hsl_full_struct_surface.npy")
    else:
        log("Computing V_full from full catalogue...")
        V_full = FH.compute_visible_fault_features(cat, foot, u2_lid, v2_lid)
        X_parts = []
        y_parts = []
        rng = np.random.default_rng(8800)
        for s_idx, s_seed in enumerate((8801, 8802, 8803)):
            ctx_c, tgt_c = make_inner_component_split(cat, seed=s_seed, hide_frac=0.22, buffer_px=3)
            V_in = FH.compute_visible_fault_features(ctx_c, foot, u2_lid, v2_lid)
            dist_ctx = ndimage.distance_transform_edt(~ctx_c).astype(np.float32)
            pos = np.flatnonzero(tgt_c & foot & (dist_ctx > 3))
            if pos.size > 30_000:
                pos = rng.choice(pos, 30_000, replace=False)
            neg_pool = foot & ~cat & (dist_ctx > 3)
            hard = np.flatnonzero(neg_pool & (dist_ctx <= 25))
            easy = np.flatnonzero(neg_pool & (dist_ctx > 25))
            n_neg = 2 * pos.size
            nh = min(hard.size, n_neg // 2)
            ne = min(easy.size, n_neg - nh)
            neg = np.concatenate([
                rng.choice(hard, nh, replace=False),
                rng.choice(easy, ne, replace=False),
            ])
            idx = np.concatenate([pos, neg])
            y_s = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
            Xp = extract_rows(stack_chw, cols_sphys, idx)
            Xv = extract_rows(V_in, list(range(len(FH.HSL4_VIS_STRUCT_NAMES))), idx)
            X_parts.append(np.hstack([Xp, Xv]))
            y_parts.append(y_s)
            del Xp, Xv, V_in, dist_ctx, ctx_c, tgt_c
            gc.collect()
            log(f"Full structural split {s_idx+1}/3 extracted")

        X_tr = np.vstack(X_parts)
        y_tr = np.concatenate(y_parts)
        del X_parts, y_parts
        gc.collect()

        p_struct = np.zeros((H, W), dtype=np.float32)
        for m_i, s_seed in enumerate((8901, 8902)):
            kw = {**HGB_KW, "random_state": s_seed}
            clf = HistGradientBoostingClassifier(**kw)
            t = time.time()
            clf.fit(X_tr, y_tr)
            p_m = predict_full_chw(stack_chw, cols_sphys, clf, foot, V_extra=V_full)
            p_struct += p_m / np.float32(2.0)
            log(f"Full HSL_structural member {m_i+1}/2 done ({time.time()-t:.1f}s)")
            del p_m, clf
            gc.collect()
        del X_tr, y_tr, V_full
        gc.collect()
        np.save(p_struct_path, p_struct)

    del u2_lid, v2_lid
    gc.collect()

    p_final = (np.float32(0.45) * p_phys + np.float32(0.55) * p_struct).astype(np.float32)
    p_final = np.where(foot, np.clip(p_final, 0.0, 1.0), np.float32(0.0))

    # 3. Load or emit uniqueness-budgeted stratified submodular candidate dots
    allowed = foot & ~cat & (dist_cat > SEL_E)
    unique_dots_path = PROC / "hsl_unique_dots.npy"
    if unique_dots_path.exists():
        dots = np.load(unique_dots_path)
        log(f"Loaded cached uniqueness-budgeted dots: {int(dots.sum())} dots")
    else:
        dots = EM.emit_dots_submodular_bayes(p_final, allowed, budget=SEL_B, min_sep=2.8)
    n_dots = int(dots.sum())
    log(f"Emitted {n_dots} dots (budget={SEL_B}, excl={SEL_E} px, min_sep=2.8 px)")

    # 4. Full Uniqueness Audit (Phase 1 surface + Phase 2 final dots) against all 808 registry rasters + 3 GEMSDOE77 rasters
    rng_u = np.random.default_rng(20261021)
    foot_idx = np.flatnonzero(foot)
    sub_idx = rng_u.choice(foot_idx, size=250_000, replace=False)

    p_sub = p_final.ravel()[sub_idx].astype(np.float64)
    d_sub = dots.ravel()[sub_idx].astype(np.float64)

    # Precompute 3-px EDT around our emitted dots and 29-cell disk shifts for exact O(N_dots) 3-px lookup
    edt_cand = ndimage.distance_transform_edt(~dots) <= 3.0
    cand_ys, cand_xs = np.nonzero(dots)
    disk29 = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy * dy + dx * dx <= 9]
    cand_shifts = [
        (np.clip(cand_ys + dy, 0, H - 1), np.clip(cand_xs + dx, 0, W - 1))
        for dy, dx in disk29
    ]
    sub_ys, sub_xs = np.divmod(sub_idx, W)
    sub_shifts = [
        (np.clip(sub_ys + dy, 0, H - 1), np.clip(sub_xs + dx, 0, W - 1))
        for dy, dx in disk29
    ]

    def cand_in_pos_3px(pos_mask: np.ndarray) -> float:
        hit = np.zeros(cand_ys.size, dtype=bool)
        for ky, kx in cand_shifts:
            hit |= pos_mask[ky, kx]
        return float(hit.mean())

    def est_halo_frac(pos_mask: np.ndarray) -> float:
        hit = np.zeros(sub_ys.size, dtype=bool)
        for sy, sx in sub_shifts:
            hit |= pos_mask[sy, sx]
        return float(hit.mean())

    # Check dense_witness.tif and h33.tif first
    witness_path = REG_DIR / "dense_witness.tif"
    h33_path = REG_DIR / "h33.tif"

    def spearman_sub(arr_2d: np.ndarray, ref_sub: np.ndarray) -> float:
        v = np.nan_to_num(arr_2d.ravel()[sub_idx], nan=0.0).astype(np.float64)
        if np.std(v) < 1e-12:
            return 0.0
        return float(stats.spearmanr(ref_sub, v).statistic)

    with rasterio.open(witness_path) as ds:
        w_arr = ds.read(1)
    rho_surf_witness = spearman_sub(w_arr, p_sub)
    rho_dots_witness = spearman_sub(w_arr, d_sub)
    del w_arr

    with rasterio.open(h33_path) as ds:
        h33_arr = ds.read(1)
    h33_pos = np.isfinite(h33_arr) & (h33_arr > 0)
    ov_cand_in_h33 = cand_in_pos_3px(h33_pos)
    ov_h33_in_cand = float(edt_cand[h33_pos].mean())
    rho_surf_h33 = spearman_sub(h33_arr, p_sub)
    rho_dots_h33 = spearman_sub(h33_arr, d_sub)
    del h33_arr

    log(
        f"vs H33 (0.2778): surf_rho={rho_surf_h33:+.4f}, dots_rho={rho_dots_h33:+.4f}, "
        f"cand_in_h33_3px={ov_cand_in_h33:.4f}, h33_in_cand_3px={ov_h33_in_cand:.4f}"
    )
    log(f"vs dense_witness: surf_rho={rho_surf_witness:+.4f}, dots_rho={rho_dots_witness:+.4f}")

    # Check against all 3 sibling GEMSDOE77 lane rasters
    sibling_paths = {
        "contactphys_ca70a971": ROOT / "docs" / "downloads" / "gems77-contactphys-44090dots-20261021T030258Z-ca70a971-zeros.tif",
        "lia_aa5c9331": ROOT / "docs" / "downloads" / "gems77-h77-lia-38323dots-20261021-aa5c9331-zeros.tif",
        "urank_0d868dc271bb": ROOT / "docs" / "downloads" / "gems77-h77-urank-curvature-0d868dc271bb-research.tif",
    }
    sibling_audit = {}
    for sname, spath in sibling_paths.items():
        if not spath.exists():
            continue
        with rasterio.open(spath) as ds:
            s_arr = ds.read(1)
        s_pos = np.isfinite(s_arr) & (s_arr > 0) & foot
        n_s = int(s_pos.sum())
        rho_s_surf = spearman_sub(s_arr, p_sub)
        rho_s_dots = spearman_sub(s_arr, d_sub)
        halo_frac = est_halo_frac(s_pos)
        ov_c_in_s = cand_in_pos_3px(s_pos)
        ov_s_in_c = float(edt_cand[s_pos].mean()) if n_s > 0 else 0.0
        sibling_audit[sname] = {
            "path": str(spath.relative_to(ROOT)),
            "nonzero_px": n_s,
            "halo_3px_footprint_fraction": round(halo_frac, 4),
            "spearman_vs_hsl_surface": round(rho_s_surf, 4),
            "spearman_vs_hsl_dots": round(rho_s_dots, 4),
            "cand_dots_within_3px_of_sibling": round(ov_c_in_s, 4),
            "sibling_within_3px_of_cand_dots": round(ov_s_in_c, 4),
            "mutual_min_overlap_3px": round(min(ov_c_in_s, ov_s_in_c), 4),
        }
        log(f"vs sibling {sname}: surf_rho={rho_s_surf:+.4f}, dots_rho={rho_s_dots:+.4f}, cand_in_sib={ov_c_in_s:.4f}, sib_in_cand={ov_s_in_c:.4f}")
        del s_arr, s_pos

    # Scan all 812 registry rasters in .cache/registry/ (with disk cache)
    scan_cache_path = PROC / "hsl_uniqueness_scan.json"
    if scan_cache_path.exists():
        sc = json.loads(scan_cache_path.read_text())
        n_informative = sc["n_informative"]
        n_space_filling = sc["n_space_filling"]
        max_inf_ov = tuple(sc["max_inf_ov"])
        max_inf_mutual = tuple(sc["max_inf_mutual"])
        max_inf_rho_surf = tuple(sc["max_inf_rho_surf"])
        max_inf_rho_dots = tuple(sc["max_inf_rho_dots"])
        max_all_ov = tuple(sc["max_all_ov"])
        top_informative = sc["top_informative"]
        log(f"Loaded cached 812-raster uniqueness scan: {n_informative} informative, {n_space_filling} space-filling")
    else:
        free_reg = json.loads((ROOT / "evidence" / "e3_registry_free.json").read_text())
        sf_names = {x["raster"] for x in free_reg["space_filling_rasters"]}
        reg_index = json.loads((ROOT / "registry" / "registry_index.json").read_text())
        reg_entries = reg_index["rasters"]
        log(f"Scanning all {len(reg_entries)} registry rasters in .cache/registry/...")

        max_inf_ov = (-1.0, "")
        max_inf_mutual = (-1.0, "")
        max_inf_rho_surf = (-1.0, "")
        max_inf_rho_dots = (-1.0, "")
        max_all_ov = (-1.0, "")
        n_informative = 0
        n_space_filling = 0
        top_informative = []

        n_unreadable = 0
        for idx_e, ent in enumerate(reg_entries):
            rpath = REG_DIR / f"{ent['blob']}.tif"
            if not rpath.exists():
                continue
            try:
                with rasterio.open(rpath) as ds:
                    if ds.shape != (H, W):
                        n_unreadable += 1
                        continue
                    arr = ds.read(1)
            except Exception:
                n_unreadable += 1
                continue
            with np.errstate(invalid="ignore"):
                pos = (arr > 0) & foot
            n_pos = int(pos.sum())
            if n_pos == 0:
                continue
            rho_sf = abs(spearman_sub(arr, p_sub))
            rho_dt = abs(spearman_sub(arr, d_sub))
            halo_frac = est_halo_frac(pos)
            ov_c_in_r = cand_in_pos_3px(pos)
            ov_r_in_c = float(edt_cand[pos].mean())
            mut = min(ov_c_in_r, ov_r_in_c)
            label_id = f"{idx_e:03d}:{ent['repo']}/{Path(ent['path']).name}"

            if ov_c_in_r > max_all_ov[0]:
                max_all_ov = (ov_c_in_r, label_id)

            if any(a in sf_names for a in ent["aliases"]) or halo_frac >= 0.70:
                n_space_filling += 1
            else:
                n_informative += 1
                if ov_c_in_r > max_inf_ov[0]:
                    max_inf_ov = (ov_c_in_r, label_id)
                if mut > max_inf_mutual[0]:
                    max_inf_mutual = (mut, label_id)
                if rho_sf > max_inf_rho_surf[0]:
                    max_inf_rho_surf = (rho_sf, label_id)
                if rho_dt > max_inf_rho_dots[0]:
                    max_inf_rho_dots = (rho_dt, label_id)
                top_informative.append({
                    "id": label_id,
                    "nonzero_px": n_pos,
                    "halo_frac": round(halo_frac, 4),
                    "cand_in_reg_3px": round(ov_c_in_r, 4),
                    "reg_in_cand_3px": round(ov_r_in_c, 4),
                    "mutual_min_3px": round(mut, 4),
                    "abs_rho_surface": round(rho_sf, 4),
                    "abs_rho_dots": round(rho_dt, 4),
                })
            if (idx_e + 1) % 200 == 0:
                log(f"  scanned {idx_e+1}/{len(reg_entries)} rasters (max informative ov so far={max_inf_ov[0]:.4f})")

        top_informative.sort(key=lambda d: -d["cand_in_reg_3px"])
        scan_cache_path.write_text(json.dumps({
            "n_informative": n_informative,
            "n_space_filling": n_space_filling,
            "max_inf_ov": list(max_inf_ov),
            "max_inf_mutual": list(max_inf_mutual),
            "max_inf_rho_surf": list(max_inf_rho_surf),
            "max_inf_rho_dots": list(max_inf_rho_dots),
            "max_all_ov": list(max_all_ov),
            "top_informative": top_informative[:25],
        }, indent=2) + "\n")

    log(
        f"Registry scan complete: {n_informative} informative + {n_space_filling} space-filling = {n_informative+n_space_filling} non-empty rasters"
    )
    log(f"  Max informative 1-way overlap (cand in reg <=3px): {max_inf_ov[0]:.4f} ({max_inf_ov[1]})")
    log(f"  Max informative mutual overlap (min 2-way <=3px):  {max_inf_mutual[0]:.4f} ({max_inf_mutual[1]})")
    log(f"  Max informative |Spearman rho| on surface:         {max_inf_rho_surf[0]:.4f} ({max_inf_rho_surf[1]})")
    log(f"  Max informative |Spearman rho| on final dots:      {max_inf_rho_dots[0]:.4f} ({max_inf_rho_dots[1]})")

    # 5. Write unique GeoTIFF submission via shared gems57.submission_writer.write_submission
    sub_arr = np.where(dots, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    raw_bytes = hashlib.sha256(sub_arr.tobytes()).hexdigest()[:8]
    fname_base = f"gems77-hsl-{n_dots}dots-{TIMESTAMP}-{raw_bytes}-zeros"
    note_str = f"GEMSDOE77 H77-HSL: 12b 1m lidar dipole/strike+shallow mag+GDR1391+vis-continuation, Bayes-submod {n_dots} dots, holdout 0.1729/0.1473"
    assert len(note_str) <= 140
    sub_dir = ROOT / "submissions"
    dl_dir = ROOT / "docs" / "downloads"
    sub_dir.mkdir(parents=True, exist_ok=True)
    dl_dir.mkdir(parents=True, exist_ok=True)

    tif_path = dl_dir / f"{fname_base}.tif"
    zip_path = dl_dir / f"{fname_base}.zip"
    json_receipt_path = dl_dir / f"{fname_base}.json"
    for p_old in (tif_path, zip_path, json_receipt_path):
        if p_old.exists():
            p_old.unlink()

    writer_receipt = SW.write_submission(
        tif_path,
        sub_arr,
        RAW / "sample_submission.tif",
        foot,
        note=note_str,
        name=fname_base,
        catalogue=cat,
    )
    for p_new in (tif_path, zip_path, json_receipt_path):
        if p_new.exists():
            p_new.chmod(0o644)
    # Also copy to submissions/ for convenience
    (sub_dir / f"{fname_base}.tif").write_bytes(tif_path.read_bytes())
    (sub_dir / f"{fname_base}.zip").write_bytes(zip_path.read_bytes())

    # 6. Run 17/17 portal-equivalence validator + independent pass-2 rasterio verification
    val17 = V17.validate(
        tif_path,
        mode="zeros",
        name=fname_base,
        note=note_str,
    )
    val_zip = V17.validate_zip(zip_path, tif_path)
    val17["zip_validation"] = val_zip
    log(f"17-check portal validator: passed={val17['passed']} ({val17['n_pass']}/{val17['n_checks']}), zip_passed={val_zip['passed']}")

    with rasterio.open(tif_path) as ds, rasterio.open(RAW / "sample_submission.tif") as sds:
        r_arr = ds.read(1)
        pass2_checks = {
            "driver_GTiff": ds.driver == "GTiff",
            "count_1": ds.count == 1,
            "dtype_float32": ds.dtypes[0] == "float32",
            "crs_epsg32611": str(ds.crs) == "EPSG:32611" and ds.crs == sds.crs,
            "shape_3730x3292": ds.shape == (3730, 3292) and ds.shape == sds.shape,
            "transform_exact": ds.transform == sds.transform,
            "nodata_none": ds.nodata is None,
            "all_finite": bool(np.all(np.isfinite(r_arr))),
            "min_ge_0": float(r_arr.min()) >= 0.0,
            "max_le_1": float(r_arr.max()) <= 1.0,
            "zero_outside_footprint": float(np.abs(r_arr[~foot]).max()) == 0.0,
            "zero_on_catalogue": int((r_arr[cat] != 0).sum()) == 0,
            "exact_dot_count": int((r_arr > 0).sum()) == n_dots,
        }
    assert all(pass2_checks.values()), f"Pass-2 check failed: {pass2_checks}"

    uniqueness_summary = {
        "total_registry_rasters_scanned": n_informative + n_space_filling,
        "informative_registry_rasters": n_informative,
        "space_filling_registry_rasters_ge_70pct_halo": n_space_filling,
        "phase1_surface_vs_dense_witness_spearman": round(rho_surf_witness, 4),
        "phase1_surface_vs_h33_spearman": round(rho_surf_h33, 4),
        "phase1_max_informative_abs_spearman": round(max_inf_rho_surf[0], 4),
        "phase1_max_informative_abs_spearman_id": max_inf_rho_surf[1],
        "phase2_dots_vs_dense_witness_spearman": round(rho_dots_witness, 4),
        "phase2_dots_vs_h33_spearman": round(rho_dots_h33, 4),
        "phase2_dots_within_3px_of_h33": round(ov_cand_in_h33, 4),
        "phase2_h33_within_3px_of_dots": round(ov_h33_in_cand, 4),
        "phase2_max_informative_1way_overlap_3px": round(max_inf_ov[0], 4),
        "phase2_max_informative_1way_overlap_id": max_inf_ov[1],
        "phase2_max_informative_mutual_overlap_3px": round(max_inf_mutual[0], 4),
        "phase2_max_informative_mutual_overlap_id": max_inf_mutual[1],
        "phase2_max_informative_abs_spearman_dots": round(max_inf_rho_dots[0], 4),
        "phase2_max_informative_abs_spearman_dots_id": max_inf_rho_dots[1],
        "phase2_literal_max_all_rasters_overlap_3px": round(max_all_ov[0], 4),
        "phase2_literal_max_all_rasters_overlap_id": max_all_ov[1],
        "informative_gate_passed": bool(
            max_inf_rho_surf[0] <= 0.90
            and max_inf_rho_dots[0] <= 0.90
            and max_inf_ov[0] <= 0.70
        ),
        "sibling_gemsdoe77_lanes": sibling_audit,
        "top_10_informative_registry_overlaps": top_informative[:10],
    }

    exp3_out = {
        "lane": "H77-HSL",
        "preregistration_id": "GEMSDOE77-PREREG-HSL-1",
        "submission_name": f"{fname_base}.tif",
        "zip_name": f"{fname_base}.zip",
        "download_tif_path": str(tif_path.relative_to(ROOT)),
        "download_zip_path": str(zip_path.relative_to(ROOT)),
        "note_140char": note_str,
        "writer_receipt": writer_receipt,
        "portal_17check_validator": val17,
        "pass2_independent_rasterio": pass2_checks,
        "uniqueness_audit": uniqueness_summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    assert len(exp3_out["note_140char"]) <= 140
    (ROOT / "evidence" / "hsl_exp3_submission.json").write_text(json.dumps(exp3_out, indent=2) + "\n")
    log(f"EXP 3 DONE in {exp3_out['elapsed_s']}s -> evidence/hsl_exp3_submission.json")
    log(f"Generated unique TIF: {tif_path.relative_to(ROOT)} (SHA256={val17['sha256']})")


if __name__ == "__main__":
    main()
