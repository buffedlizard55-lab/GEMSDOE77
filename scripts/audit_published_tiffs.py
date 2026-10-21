#!/usr/bin/env python3
"""Read-only primary TIFF audit; format PASS never grants submission approval."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    lanes = json.loads((root / 'docs/data/lanes.json').read_text())['lanes']
    results = []
    with rasterio.open(root / 'data/bridge/sample_submission.tif') as reference:
        for row in lanes:
            path = root / 'docs' / row['file']
            checks = {}
            try:
                checks['sha256_matches_manifest'] = hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
                with rasterio.open(path) as src:
                    checks.update(single_band=src.count == 1, shape_matches=src.shape == reference.shape,
                                  crs_matches=src.crs == reference.crs,
                                  transform_matches=src.transform == reference.transform,
                                  nodata_none=src.nodata is None)
                    # Check raw values, not masked arrays that can hide sentinels.
                    checks['all_finite'] = True
                    checks['all_values_in_0_1'] = True
                    for _, window in src.block_windows(1):
                        values = src.read(1, window=window)
                        checks['all_finite'] &= bool(np.isfinite(values).all())
                        checks['all_values_in_0_1'] &= bool(((values >= 0) & (values <= 1)).all())
            except (OSError, ValueError, rasterio.errors.RasterioError) as error:
                checks['readable'] = False
                checks['error'] = str(error)
            results.append(dict(file=row['file'], checks=checks,
                                format_pass=all(v is True for v in checks.values()),
                                authorized_to_submit=False))
    pin = json.loads((root / 'registry/template_pin.json').read_text())
    pin_checks = {name: hashlib.sha256((root / 'vendor/gems57' / name).read_bytes()).hexdigest() == digest
                  for name, digest in pin['files'].items()}
    return dict(schema='gems77-primary-tiff-audit-v1', evidence_class='LOCAL-FORMAT-AND-INTEGRITY',
                reference='data/bridge/sample_submission.tif', template_pin_checks=pin_checks,
                artifacts=results, all_passed=all(r['format_pass'] for r in results) and all(pin_checks.values()),
                caveat='No new model experiment, holdout estimate, uniqueness clearance, or organizer acceptance.')


if __name__ == '__main__':
    result = audit()
    (ROOT / 'evidence/review_primary_tiff_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['all_passed'] else 1)
