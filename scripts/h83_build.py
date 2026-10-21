#!/usr/bin/env python3
"""H83-NETSIB: write the final GeoTIFF with the UNMODIFIED vendor writer, run the
UNMODIFIED vendor validator, and measure final-dot uniqueness against every
fetched registry raster using the vendor's own support/dilation primitives.

Reported per registry raster (REGISTRY-MEASUREMENT, not a score):
* forward overlap  = share of my dots within 3 px of their positive support
                     (vendor ``_near(_dots(theirs))[mine].mean()``, literal reading);
* informative      = their 3-px dilated support covers < 70 % of the allowed domain
                     (if it covers more, uniformly random dots would exceed 0.70, so
                     the test cannot discriminate drift - see IR-H83N-01);
* matched-budget overlap for dense rasters = overlap with their top-37,654
                     allowed pixels (a dot-equivalent at our budget);
* exact Spearman (full footprint) of my binary dots vs their values.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import grid as G  # noqa: E402
from gems57 import submission_writer as SW  # noqa: E402
from gems57 import validate as VAL  # noqa: E402
from gems57.uniqueness import OVERLAP_LIMIT, RHO_LIMIT, _dots, _near  # noqa: E402
from h83_netsib_pipeline import BUDGET, PROC, registry_files, sha256_file  # noqa: E402

EVID = ROOT / "evidence"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def registry_dot_scan(p, foot, cat, exclude=()):
    allowed = foot & ~cat
    n_allowed = int(allowed.sum())
    mine = (p > 0) & foot
    n_mine = int(mine.sum())
    mine_fp = mine[foot]
    n = int(foot.sum())
    rows, seen = [], {}
    excl = {Path(x).resolve() for x in exclude}
    for label, path, _sha in registry_files():
        if Path(path).resolve() in excl:
            continue  # never compare the candidate with itself (IR-H83N-02)
        try:
            with rasterio.open(path) as src:
                if src.count != 1 or src.shape != foot.shape:
                    rows.append({"raster": label, "skipped": "grid mismatch"}); continue
                t = src.read(1).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            rows.append({"raster": label, "error": str(e)}); continue
        t = np.where(np.isfinite(t), t, 0.0)
        t[~foot] = 0.0
        key = hashlib.sha256(t.tobytes()).hexdigest()
        if key in seen:
            rows.append({"raster": label, "same_decoded_as": seen[key]}); continue
        seen[key] = label
        sup = _dots(t) & foot
        their = int(sup.sum())
        if their == 0:
            rows.append({"raster": label, "their_dots": 0}); continue
        near = _near(sup)
        fwd = float(near[mine].mean())
        cover = float(near[allowed].sum() / n_allowed)
        tv = t[foot]
        if np.ptp(tv) == 0:
            rho = None
        else:
            tr = rankdata(tv); tr -= tr.mean()
            mr = mine_fp.astype(np.float64); mr -= mr.mean()
            rho = float(np.dot(mr, tr) / (np.linalg.norm(mr) * np.linalg.norm(tr)))
        row = {"raster": label, "sha256_decoded": key, "their_dots": their,
               "forward_overlap_3px": fwd, "their_3px_cover_of_allowed": cover,
               "informative": bool(cover < OVERLAP_LIMIT), "spearman_full_footprint": rho,
               "identical_pixels": bool(np.array_equal(sup, mine))}
        if their > 2 * BUDGET:
            v = np.where(allowed, t, -np.inf).ravel()
            top = np.argpartition(-v, BUDGET - 1)[:BUDGET]
            topm = np.zeros(foot.size, bool); topm[top] = True
            row["matched_budget_overlap_3px"] = float(_near(topm.reshape(foot.shape))[mine].mean())
        rows.append(row)
        del t, near, sup
    return summarize(rows, n_mine)


def summarize(rows, n_mine):
    ok = [r for r in rows if "forward_overlap_3px" in r]
    inf = [r for r in ok if r["informative"]]
    w = lambda rs, k: max(rs, key=lambda r: abs(r.get(k) or 0.0), default=None)  # noqa: E731
    wi, wl, wr = w(inf, "forward_overlap_3px"), w(ok, "forward_overlap_3px"), w(ok, "spearman_full_footprint")
    mb = [r for r in ok if "matched_budget_overlap_3px" in r]
    wm = w(mb, "matched_budget_overlap_3px")
    summary = {
        "evidence_class": "REGISTRY-MEASUREMENT",
        "my_dots": n_mine, "unique_decoded_rasters_compared": len(ok),
        "rows_total_including_duplicates_and_errors": len(rows),
        "errors": [r for r in rows if "error" in r or "skipped" in r],
        "rho_limit": RHO_LIMIT, "overlap_limit": OVERLAP_LIMIT,
        "max_abs_spearman": abs(wr["spearman_full_footprint"]) if wr else None,
        "max_abs_spearman_raster": wr["raster"] if wr else None,
        "informative_rasters": len(inf),
        "max_forward_overlap_informative": wi["forward_overlap_3px"] if wi else None,
        "max_forward_overlap_informative_raster": wi["raster"] if wi else None,
        "max_forward_overlap_literal_all": wl["forward_overlap_3px"] if wl else None,
        "max_forward_overlap_literal_all_raster": wl["raster"] if wl else None,
        "literal_firings_noninformative": sum(1 for r in ok if r["forward_overlap_3px"] > OVERLAP_LIMIT and not r["informative"]),
        "max_matched_budget_overlap_dense": wm["matched_budget_overlap_3px"] if wm else None,
        "max_matched_budget_overlap_dense_raster": wm["raster"] if wm else None,
        "identical_pixels_any": any(r["identical_pixels"] for r in ok),
    }
    summary["duplicate"] = bool(
        (summary["max_abs_spearman"] or 0) > RHO_LIMIT
        or (summary["max_forward_overlap_informative"] or 0) > OVERLAP_LIMIT
        or (summary["max_matched_budget_overlap_dense"] or 0) > OVERLAP_LIMIT
        or summary["identical_pixels_any"])
    return summary, rows


def resummarize():
    """Recompute the summary from saved rows after dropping self-comparisons."""
    b = json.loads((EVID / "h83_e3_build.json").read_text())
    rows = json.loads((EVID / "h83_e3_registry_rows.json").read_text())
    name = b["submission_name"]
    self_labels = {f"local:submissions/{name}.tif", f"local:docs/downloads/{name}.tif"}
    kept = [r for r in rows if r["raster"] not in self_labels and r.get("same_decoded_as") not in self_labels]
    summ, kept = summarize(kept, b["final_dot_uniqueness"]["my_dots"])
    summ["self_comparisons_removed"] = len(rows) - len(kept)
    b["final_dot_uniqueness"] = summ
    (EVID / "h83_e3_build.json").write_text(json.dumps(b, indent=1, default=str) + "\n")
    (EVID / "h83_e3_registry_rows.json").write_text(json.dumps(kept, indent=0) + "\n")
    log("resummarized:", {k: v for k, v in summ.items() if k != "errors"})


def main():
    if "--resummarize" in sys.argv:
        resummarize(); return 0
    grid = G.load_grid(ROOT / "data" / "bridge")
    foot, cat = grid.footprint, grid.catalogue
    p = np.load(PROC / "final_dots.npy").astype(np.float32)
    assert p.shape == foot.shape and int((p > 0).sum()) == BUDGET and not (p[cat] > 0).any()
    assert set(np.unique(p).tolist()) <= {0.0, 1.0} and not (p[~foot] > 0).any()
    sel = np.flatnonzero(p)
    dot_hash = hashlib.sha256(np.sort(sel).astype("<i8").tobytes()).hexdigest()
    name = f"gems77-h83-netsib-{BUDGET}dots-20261021-{dot_hash[:8]}-zeros"
    note = "H83-NETSIB fault-network sibling/gap/continuation geometry, synthetic re-withholding HGB; 37654 dots sep2.8 >2px off catalogue"
    assert len(name) <= 140 and len(note) <= 140, (len(name), len(note))
    out = ROOT / "submissions" / f"{name}.tif"
    receipt = SW.write_submission(out, p, ROOT / "data" / "bridge" / "sample_submission.tif", foot,
                                  note=note, name=name, catalogue=cat,
                                  metadata={"lane": "H83-NETSIB", "dots": BUDGET, "dot_index_sha256": dot_hash})
    dl = ROOT / "docs" / "downloads"
    for ext in (".tif", ".zip", ".json"):
        src = ROOT / "submissions" / f"{name}{ext}"
        if src.exists():
            shutil.copy2(src, dl / f"{name}{ext}")
    val = VAL.validate(out, foot, cat)
    # read back actual bytes: must equal the in-memory dots
    with rasterio.open(out) as ds:
        back = ds.read(1)
    assert np.array_equal(back, p), "readback mismatch"
    log("validator all_checks_passed:", val["all_checks_passed"], "sha256:", val["sha256"])
    log("scanning registry (final dots)...")
    summ, rows = registry_dot_scan(p, foot, cat, exclude=[out] + [dl / f"{name}{x}" for x in (".tif",)])
    log("uniqueness summary:", {k: v for k, v in summ.items() if k != "errors"})
    build = {"experiment": "H83-E3-build", "submission_name": name, "note": note, "note_chars": len(note),
             "file": f"submissions/{name}.tif", "download": f"docs/downloads/{name}.tif",
             "zip": f"docs/downloads/{name}.zip", "file_sha256": sha256_file(out),
             "zip_sha256": sha256_file(ROOT / "submissions" / f"{name}.zip") if (ROOT / "submissions" / f"{name}.zip").exists() else None,
             "bytes": out.stat().st_size, "dot_index_sha256": dot_hash, "writer_receipt": receipt,
             "validator": val, "readback_equals_dots": True, "final_dot_uniqueness": summ}
    (EVID / "h83_e3_build.json").write_text(json.dumps(build, indent=1, default=str) + "\n")
    (EVID / "h83_e3_registry_rows.json").write_text(json.dumps(rows, indent=0) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
