"""Portal-equivalence validator for GEMSDOE77 submissions (fails closed).

Checks the published rules (problem page 967 "Submission format"; rules PDF section 3.2)
against the OFFICIAL footprint = finite cells of sample_submission.tif, not a feature band.
Why: the template validator derives the footprint from feature band 1, which differs from
the official footprint by 3,061 + 1,540 cells (measured); a file built on that footprint
leaves NaN inside the scored region - one of the two mechanisms behind the portal message
"Predicted values must be in range [0, 1]".
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from . import core


def validate(path: str | Path, *, mode: str, name: str, note: str) -> dict:
    import rasterio
    path = Path(path)
    g = core.grid()
    with rasterio.open(core.resolve_core_file("sample_submission.tif")) as ref:
        ref_t, ref_crs, ref_shape = ref.transform, ref.crs, (ref.height, ref.width)
    checks = []

    def chk(n, ok, detail):
        checks.append({"check": n, "status": "PASS" if ok else "FAIL", "detail": detail})

    with rasterio.open(path) as src:
        chk("GeoTIFF driver", src.driver == "GTiff", src.driver)
        chk("single band", src.count == 1, f"count={src.count}")
        chk("dtype float32", src.dtypes[0] == "float32", src.dtypes[0])
        chk("CRS EPSG:32611", src.crs is not None and src.crs.to_epsg() == 32611, str(src.crs.to_epsg() if src.crs else None))
        chk("CRS equals sample_submission", src.crs == ref_crs, "")
        chk("shape 3730x3292 equals sample_submission", (src.height, src.width) == ref_shape, f"{(src.height, src.width)}")
        chk("transform equals sample_submission", tuple(src.transform)[:6] == tuple(ref_t)[:6], str(tuple(src.transform)[:6]))
        chk("resolution 100 m", tuple(map(abs, src.res)) == (100.0, 100.0), str(src.res))
        nod = src.nodata
        chk("nodata tag none (zeros) or NaN (nan)", (nod is None) if mode == "zeros" else (nod is None or np.isnan(nod)), str(nod))
        a = src.read(1)
    fin = np.isfinite(a)
    chk("no NaN/inf inside official footprint", bool(fin[g.footprint].all()), f"nonfinite_inside={int((~fin & g.footprint).sum())}")
    if mode == "zeros":
        chk("all 12,279,160 cells finite (no NaN anywhere)", bool(fin.all()), f"nonfinite={int((~fin).sum())}")
        chk("zeros outside footprint", bool((a[~g.footprint] == 0).all()), "")
    else:
        chk("NaN exactly outside footprint", bool(np.isnan(a[~g.footprint]).all()), f"finite_outside={int(fin[~g.footprint].sum())}")
    v = a[fin]
    chk("values within [0, 1]", bool(v.size and v.min() >= 0 and v.max() <= 1), f"min={float(v.min())} max={float(v.max())}")
    inside = a[g.footprint]
    chk("binary values {0,1} inside footprint", bool(np.isin(inside, [0.0, 1.0]).all()), "")
    chk("0 dots on masked catalogue pixels", int((a[g.catalogue] > 0).sum()) == 0, f"on_catalogue={int((a[g.catalogue] > 0).sum())}")
    chk("name non-empty, <= 120 chars", 0 < len(name) <= 120, f"{len(name)} chars")
    chk("note <= 140 chars", len(note) <= 140, f"{len(note)} chars")
    failed = [c for c in checks if c["status"] == "FAIL"]
    return {"file": path.name, "sha256": core.sha256_file(path), "bytes": path.stat().st_size,
            "mode": mode, "dots": int((inside > 0).sum()), "checks": checks,
            "n_checks": len(checks), "n_pass": len(checks) - len(failed), "passed": not failed}


def validate_zip(zpath: str | Path, tif_path: str | Path) -> dict:
    zpath, tif_path = Path(zpath), Path(tif_path)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        tifs = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
        same = len(tifs) == 1 and z.read(tifs[0]) == tif_path.read_bytes()
    return {"zip": zpath.name, "members": names, "exactly_one_geotiff": len(tifs) == 1 and len(names) == 1,
            "member_bytes_identical_to_tif": bool(same), "sha256": core.sha256_file(zpath),
            "bytes": zpath.stat().st_size, "passed": bool(len(names) == 1 and same)}


if __name__ == "__main__":
    import sys
    print(json.dumps(validate(sys.argv[1], mode=sys.argv[2], name=sys.argv[3], note=sys.argv[4]), indent=1))
