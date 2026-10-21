#!/usr/bin/env python3
"""H82: write the fail-closed submission GeoTIFF + one-TIFF ZIP + receipt.

Loads the deterministic emission mask from Experiment 1 (evidence/h82_dots_ranked.npy,
re-derivable by re-running scripts/h82_holdout.py with the preregistered seed) and
packages it via the shared fail-closed writer (vendor/gems57/submission_writer.py):
single-band float32, EPSG:32611, 3730x3292, pinned transform, values exactly {0,1},
no nodata tag, zeros outside the footprint, zero dots on the mapped catalogue.

The writer validates in memory and on disk and refuses to overwrite existing outputs;
it never clips or repairs. Local format validity is NOT uniqueness clearance or
organizer acceptance.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import grid as G  # noqa: E402
from gems57.submission_writer import write_submission  # noqa: E402

BRIDGE = ROOT / "data" / "bridge"
DOTS_NPY = ROOT / "evidence" / "h82_dots_ranked.npy"
OUT_DIR = ROOT / "docs" / "downloads"
NAME_STEM = "gems77-h82-faultcollar-37553dots-20261021"

SUBMISSION_NAME = "gems77-h82-faultcollar-37553dots-20261021-zeros"
SUBMISSION_NOTE = ("H82 fault-damage collar 37553 dots; HOLDOUT-DTI 0.1227 [0.1080,0.1372] s30; "
                   "below 0.152718 incumbent: NEGATIVE, do not upload")
assert len(SUBMISSION_NAME) <= 140 and len(SUBMISSION_NOTE) <= 140, (len(SUBMISSION_NAME), len(SUBMISSION_NOTE))


def main() -> None:
    t0 = time.time()
    if not DOTS_NPY.is_file():
        raise FileNotFoundError(f"run Experiment 1 first: {DOTS_NPY}")
    dots = np.load(DOTS_NPY)
    if dots.shape != (G.HEIGHT, G.WIDTH):
        raise ValueError(f"dots shape {dots.shape} != {(G.HEIGHT, G.WIDTH)}")
    prediction = dots.astype(np.float32)
    grid = G.load_grid(BRIDGE)

    digest = __import__("hashlib").sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()
    stem = f"{NAME_STEM}-{digest[:8]}"
    tif_path = OUT_DIR / f"{stem}-zeros.tif"

    receipt = write_submission(
        tif_path, prediction,
        sample=BRIDGE / "sample_submission.tif",
        footprint=grid.footprint,
        catalogue=grid.catalogue,
        note=SUBMISSION_NOTE, name=SUBMISSION_NAME,
        metadata=dict(
            lane_id="H82-FAULTCOLLAR",
            holdout_dti=0.1226552189170721,
            holdout_ci95=[0.10799672421127436, 0.1372130738097347],
            withheld_positive_pixels=11342,
            evaluator_version="gems57-pooled-hide-v2",
            dots_emitted=int((prediction > 0).sum()),
            dots_on_catalogue=int(((prediction > 0) & grid.catalogue).sum()),
            sha256_dots=digest,
        ),
    )
    print("WROTE", tif_path)
    print("zip:", receipt["zip_file"], "zip_sha256:", receipt["zip_sha256"])
    print("tif_sha256:", receipt["sha256"], "bytes:", receipt["bytes"])
    print("validator:", json.dumps(receipt["validator"], indent=1)[:800])
    print(f"elapsed_s {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
