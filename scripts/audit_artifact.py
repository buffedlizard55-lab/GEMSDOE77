"""Recheck immutable output bytes; never train, place dots, or change the run."""
import json
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'vendor'),str(ROOT/'scripts')]
from gems57 import uniqueness
from gems57.grid import load_grid
from gems57.validate import validate, assert_submittable
from restore_registry import get_registry
from run_experiment import dump
from fetch_data import sha


def main():
    card = json.loads((ROOT/'evidence/run_card.json').read_text())
    path = ROOT/card['file']
    if sha(path) != card['raster_sha256']: raise ValueError('immutable artifact changed')
    g = load_grid(ROOT/'data')
    report = validate(path, g.footprint, g.catalogue)
    assert_submittable(report)
    with zipfile.ZipFile(path.with_suffix('.zip')) as z:
        if z.namelist() != [path.name] or z.read(path.name) != path.read_bytes():
            raise ValueError('ZIP must contain the same single TIFF')
    audit = uniqueness.compare_to_registry(path, get_registry(), g.footprint,
                          expected_grid=(g.shape,g.crs,g.transform))
    before = json.loads((ROOT/'evidence/surface_registry.json').read_text())
    if before['candidate_decoded_sha256'] != audit['candidate_decoded_sha256']:
        raise ValueError('export changed tested surface')
    dump(ROOT/'evidence/surface_registry.json', audit)
    cert = uniqueness.saturation_certificate(get_registry()[0]['file'],g.footprint,g.catalogue,
                          expected_grid=(g.shape,g.crs,g.transform))
    dump(ROOT/'evidence/registry_saturation.json',cert)
    dump(ROOT/'evidence/artifact_recheck.json',{'format':report,'zip_roundtrip':True,
        'decoded_surface_unchanged':True,'protocol_unique':audit['unique'],
        'ok_to_submit':False,'organizer_acceptance':None})
    print('Immutable artifact, single-TIFF ZIP, format, and stop witness rechecked. NOT OK TO SUBMIT.')

if __name__ == '__main__': main()
