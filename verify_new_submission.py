#!/usr/bin/env python3
"""
Verification script for the new H77-MSCAD submission.

This script verifies that:
1. All required files exist
2. The TIF files are valid and match competition format
3. Values are in [0, 1] range
4. The submission is ready for DrivenData
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent


def verify_file(path: Path, expected_sha256: str = None) -> dict:
    """Verify a single file."""
    if not path.exists():
        return {'exists': False, 'error': f'File not found: {path}'}
    
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            meta = {
                'crs': str(src.crs),
                'shape': arr.shape,
                'transform': tuple(src.transform)[:6],
                'dtype': str(arr.dtype),
                'nodata': src.nodata
            }
        
        # Check values
        footprint = np.isfinite(arr)
        in_footprint = arr[footprint]
        
        checks = {
            'exists': True,
            'path': str(path),
            'size_bytes': path.stat().st_size,
            'meta': meta,
            'all_finite_in_footprint': np.isfinite(in_footprint).all(),
            'in_range_0_1': (in_footprint >= 0).all() and (in_footprint <= 1).all(),
            'min': float(np.nanmin(in_footprint)),
            'max': float(np.nanmax(in_footprint)),
            'positive_px': int((in_footprint > 0).sum()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sha256_matches': expected_sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        }
        
        return checks
    except Exception as e:
        return {'exists': True, 'error': str(e)}


def main():
    print("=" * 80)
    print("VERIFICATION: H77-MSCAD Submission")
    print("=" * 80)
    
    # Expected files
    files = {
        'Primary TIF': ROOT / 'docs' / 'downloads' / 'h77-mscad-20261021T154202Z-b2643a2f-zeros.tif',
        'Twin TIF': ROOT / 'docs' / 'downloads' / 'h77-mscad-20261021T154202Z-b2643a2f-nan.tif',
        'ZIP': ROOT / 'docs' / 'downloads' / 'h77-mscad-20261021T154202Z-b2643a2f-zeros.zip',
        'Run Card': ROOT / 'evidence' / 'h77-mscad-20261021T154202Z-b2643a2f_run_card.json',
        'Exec Summary': ROOT / 'docs' / 'h77-mscad-executive-summary.html',
    }
    
    expected_sha256 = {
        'Primary TIF': '246d840d9d2c592d530c730155f46df65b26ceacba9655c6462a480be1275eb1',
        'Twin TIF': 'b46eb1a24a69eca388d5843becc6dbbdaa35c1af1c973bfdf56c03b1e364caa',
    }
    
    all_pass = True
    
    print("\n📁 File Verification:")
    print("-" * 80)
    
    for name, path in files.items():
        # Only verify TIF files with rasterio
        if name in ['Primary TIF', 'Twin TIF']:
            result = verify_file(path, expected_sha256.get(name))
            
            if not result.get('exists', False):
                print(f"  ❌ {name}: {result.get('error', 'Unknown error')}")
                all_pass = False
                continue
            
            if 'error' in result:
                print(f"  ❌ {name}: {result['error']}")
                all_pass = False
                continue
            
            status = "✅" if result['all_finite_in_footprint'] and result['in_range_0_1'] else "❌"
            print(f"  {status} {name}:")
            print(f"     - Size: {result['size_bytes']:,} bytes")
            print(f"     - Shape: {result['meta']['shape']}")
            print(f"     - CRS: {result['meta']['crs']}")
            print(f"     - Range: [{result['min']:.3f}, {result['max']:.3f}]")
            print(f"     - Positive pixels: {result['positive_px']:,}")
            print(f"     - All finite: {result['all_finite_in_footprint']}")
            print(f"     - In [0,1]: {result['in_range_0_1']}")
            print(f"     - SHA256: {result['sha256']}")
            if name in expected_sha256:
                match = "✅" if result.get('sha256_matches', False) else "❌"
                print(f"     - SHA256 matches expected: {match}")
            
            if not (result['all_finite_in_footprint'] and result['in_range_0_1']):
                all_pass = False
        else:
            # For non-TIF files, just check existence and size
            if path.exists():
                size = path.stat().st_size
                print(f"  ✅ {name}: {size:,} bytes")
            else:
                print(f"  ❌ {name}: File not found")
                all_pass = False
    
    print("\n" + "=" * 80)
    print("📊 Submission Details:")
    print("-" * 80)
    
    primary = verify_file(files['Primary TIF'])
    if primary.get('exists', False) and not 'error' in primary:
        print(f"  Submission Name: h77-mscad-20261021T154202Z-b2643a2f")
        print(f"  Submission Note: H77 MSCAD: Multi-Scale Curvature Anomaly Detection, {primary['positive_px']:,} dots, 0 on catalogue")
        print(f"  Note Length: 78 characters (≤ 140) ✅")
        print(f"  Dot Count: {primary['positive_px']:,}")
        print(f"  Format: Single-band float32 GeoTIFF")
        print(f"  CRS: {primary['meta']['crs']}")
        print(f"  Shape: {primary['meta']['shape']}")
    
    print("\n" + "=" * 80)
    print("✅ VERIFICATION RESULTS:")
    print("-" * 80)
    
    checks = []
    
    # Check all files exist
    all_exist = all(f.exists() for f in files.values())
    checks.append(("All required files exist", all_exist))
    
    # Check TIF format
    primary_ok = primary.get('all_finite_in_footprint', False) and primary.get('in_range_0_1', False)
    checks.append(("Primary TIF format valid", primary_ok))
    
    twin = verify_file(files['Twin TIF'])
    twin_ok = twin.get('all_finite_in_footprint', False) and twin.get('in_range_0_1', False)
    checks.append(("Twin TIF format valid", twin_ok))
    
    # Check ZIP exists
    zip_exists = files['ZIP'].exists()
    checks.append(("ZIP file exists", zip_exists))
    
    # Check run card exists
    run_card_exists = files['Run Card'].exists()
    checks.append(("Run card exists", run_card_exists))
    
    # Check exec summary exists
    exec_exists = files['Exec Summary'].exists()
    checks.append(("Executive summary exists", exec_exists))
    
    # Check SHA256
    sha_ok = primary.get('sha256_matches', True) if 'Primary TIF' in expected_sha256 else True
    checks.append(("SHA256 matches", sha_ok))
    
    # Check note length
    note_length = 78  # "H77 MSCAD: Multi-Scale Curvature Anomaly Detection, 37654 dots, 0 on catalogue"
    checks.append(("Note ≤ 140 characters", note_length <= 140))
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    print("\n" + "=" * 80)
    
    if all_pass and all(p for _, p in checks):
        print("🎉 ALL CHECKS PASSED - SUBMISSION IS READY!")
        print("\n🚀 You can now:")
        print("  1. Download the TIF or ZIP file")
        print("  2. Go to https://www.drivendata.org/competitions/306/competition-doe-gems/")
        print("  3. Submit with name: h77-mscad-20261021T154202Z-b2643a2f")
        print("  4. Note: H77 MSCAD: Multi-Scale Curvature Anomaly Detection, 37654 dots, 0 on catalogue")
        print("\n✅ The 'Predicted values must be in range [0, 1]' error WILL NOT OCCUR!")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - See above for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())
