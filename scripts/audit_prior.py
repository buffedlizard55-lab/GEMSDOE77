"""Independent forensic reread of prior submissions, education only."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'vendor'),str(ROOT/'scripts')]
import numpy as np
import rasterio
from scipy import ndimage as ndi
from fetch_data import sha
from gems57.grid import load_grid
from run_experiment import dump
from restore_registry import get_registry


def audit(folder):
    g = load_grid(ROOT/'data')
    distance = ndi.distance_transform_edt(~g.catalogue)
    arrays, rows = {}, {}
    for key, filename in [('d28','GEMSDOE25__dotted-h19-5-d2-8.tif'),
                          ('h27','GEMSDOE28__h27-4-r1-solo-d2-8.tif'),
                          ('h33','GEMSDOE32__h33-2-b2-zeros.tif')]:
        path = Path(folder)/filename
        with rasterio.open(path) as src:
            if (src.shape,src.crs,src.transform) != (g.shape,g.crs,g.transform):
                raise ValueError('prior grid mismatch')
            a = src.read(1)
        support = g.footprint & np.isfinite(a) & (a > 0)
        arrays[key] = support
        rows[key] = {'source_path':filename,'sha256':sha(path), 'positive_pixels':int(support.sum()),
                     'on_catalogue':int((support & g.catalogue).sum()),
                     'minimum_catalogue_distance_pixels':float(distance[support].min()),
                     'median_catalogue_distance_m':float(np.median(distance[support])*100),
                     'finite_values':np.unique(a[np.isfinite(a)]).tolist()}
    assert rows['h33']['sha256'] == get_registry()[1]['sha256']
    out = {'evidence_class':'RASTER-FORENSICS, NOT SCORE EVIDENCE',
           'source_repository':'https://github.com/buffedlizard55-lab/60GEMSDOE',
           'source_commit':json.loads((ROOT/'registry/template_pin.json').read_text())['commit'],
           'files':rows,'comparisons':{},'organizer_confirmed_scores':None}
    for old,new in [('d28','h27'),('h27','h33'),('d28','h33')]:
        removed = arrays[old] & ~arrays[new]
        out['comparisons'][old+'_to_'+new] = {
            'removed':int(removed.sum()),'added':int((arrays[new] & ~arrays[old]).sum()),
            'removed_distances_px':np.unique(distance[removed]).tolist(),
            'exact_old_filtered_distance_gt_2':bool(np.array_equal(arrays[new],arrays[old] & (distance > 2)))}
    dump(ROOT/'evidence/prior_forensics.json', out)
    print(json.dumps(out,indent=2))

if __name__ == '__main__':
    if len(sys.argv) != 2: raise SystemExit('usage: audit_prior.py PATH_TO_PINNED_TEMPLATE/registry/rasters')
    audit(sys.argv[1])
