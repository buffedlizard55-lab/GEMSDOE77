"""Verify cached input metadata; never relabel catalogue as hidden truth."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'vendor'),str(ROOT/'scripts')]
import rasterio
from gems57.grid import load_grid
from fetch_data import FILES, sha

def prepare():
 for name, (_, expected) in FILES.items():
  if sha(ROOT/'data'/name) != expected: raise ValueError('input changed: '+name)
 g = load_grid(ROOT/'data')
 with rasterio.open(ROOT/'data/geodawn_rad_u8.tif') as s:
  if (s.shape,s.crs,s.transform) != (g.shape,g.crs,g.transform): raise ValueError('feature grid mismatch')
  if s.descriptions != ('K','Th','U','TC') or s.nodata != 0: raise ValueError('unexpected bands/nodata')
  receipt = {'shape':list(g.shape),'crs':str(g.crs),'transform':list(tuple(g.transform)[:6]),
             'feature_bands':list(s.descriptions),'footprint_pixels':int(g.footprint.sum()),
             'catalogue_pixels':int(g.catalogue.sum()),'training_labels':'known catalogue, NOT hidden organizer truth',
             'radiometric_units':'independently percentile-quantized channels, not concentrations'}
 (ROOT/'evidence/prepared_data.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
 return g

if __name__ == '__main__': prepare()
