"""CLI format validation only; it never approves competition submission."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from gems57.grid import load_grid
from gems57.validate import validate

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('path',type=Path);args=ap.parse_args()
    # The pinned public bridge is the repository's canonical lightweight grid.
    # Large restored inputs live under .cache and are not required for validation.
    g=load_grid(ROOT/'data'/'bridge')
    r=validate(args.path,g.footprint,g.catalogue)
    r['ok_to_submit']=False
    r['approval_warning']='Format validation is not protocol uniqueness, selector approval or organizer acceptance.'
    print(json.dumps(r,indent=2,allow_nan=False))
    return 0 if r['all_checks_passed'] else 1

if __name__=='__main__':sys.exit(main())
