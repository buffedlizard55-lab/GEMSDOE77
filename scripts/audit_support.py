"""Post-run kernel-support audit only; no transforms, fit, score or placement."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'scripts')]
import numpy as np
import rasterio
from scipy import ndimage as ndi
from gems57.grid import load_grid
from run_experiment import dump


def main():
    g=load_grid(ROOT/'data')
    with rasterio.open(ROOT/'data/geodawn_rad_u8.tif') as s:raw=s.read([1,2,3])
    supported=g.footprint & (raw>0).all(0)
    eligible=ndi.binary_erosion(supported,iterations=10)
    safe=ndi.binary_erosion(supported,structure=np.ones((3,3)),iterations=8)
    risky=eligible & ~safe
    report={'evidence_class':'POST-RUN SUPPORT DIAGNOSTIC, NOT A NEW EXPERIMENT OR SCORE',
            'original_erosion':'10 iterations default cross structure (taxicab metric)',
            'gaussian_stencil':'sigma=2, truncate=4; square Chebyshev radius 8 pixels',
            'original_eligible_pixels':int(eligible.sum()),
            'eligible_cells_with_potential_nodata_stencil_contact':int(risky.sum()),
            'catalogue_positive_pixels_in_risky_cells':int((risky & g.catalogue).sum()),
            'risk_fraction_of_eligible':float(risky.sum()/eligible.sum()),
            'score_recomputed':False,'artifact_changed':False,
            'original_implementation_commit':'1e5c994',
            'consequence':'Historical HOLDOUT-DTI retained as observed, confounded and not valid promotion evidence. No rerun after duplicate stop.',
            'fix':'Own feature builder now rejects registered domain if its square Gaussian stencil can touch nodata; shared tools unchanged.'}
    dump(ROOT/'evidence/support_audit.json',report)
    card=json.loads((ROOT/'evidence/run_card.json').read_text())
    card['holdout_valid_for_promotion']=False
    card['holdout_caveat']='Post-run audit: 10px cross erosion insufficient for diagonal Gaussian stencil. Retained historical result only; no rerun.'
    card['support_audit']='evidence/support_audit.json'
    card['reason']='Literal surface duplicate stop; incomplete registry; boundary-support confound. Research-only TIFF, no slot approval.'
    dump(ROOT/'evidence/run_card.json',card)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
