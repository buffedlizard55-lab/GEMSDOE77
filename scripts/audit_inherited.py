"""Review concurrently merged H77C bytes; no training, new hypothesis or slot."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'scripts')]
from gems57.grid import load_grid
from gems57.validate import validate
from gems57.uniqueness import compare_to_registry
from restore_registry import get_registry
from run_experiment import dump
from fetch_data import sha


def main():
    g=load_grid(ROOT/'data/bridge')
    path=ROOT/'docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.tif'
    result=compare_to_registry(path,get_registry(),g.footprint,expected_grid=(g.shape,g.crs,g.transform))
    report={'reviewed_base_commit':'f06695d','inherited_from_pr':2,
            'purpose':'Post-merge immutable-byte safety audit, not an H77-U experiment',
            'inherited_file':str(path.relative_to(ROOT)),'sha256':sha(path),
            'shared_validator':validate(path,g.footprint,g.catalogue),
            'shared_registry_check':result,'ok_to_submit':False,'verdict':'negative',
            'findings':['Original registry PASS text admits values would be computed if registry files fetched; not verified clearance.',
                        'Generator code contained numeric fallback overlap/correlation when measurements missing.',
                        'Measured shared literal gate fails. Inherited holdout is not validated as shared pooled whole-component hide/recover.',
                        'Mechanism text and named winning claycap hypothesis mismatch in inherited run card.',
                        'Original source/claims retained as archived text. Generation entry point disabled, not rerun.'],
            'full_registry_scanned':False,'weekly_slots_used_by_this_review':0}
    dump(ROOT/'evidence/inherited_h77c_review.json',report)
    dump(ROOT/'registry/inherited_h77c_status.json',{'status':'quarantined; not approved',
           'ok_to_submit':False,'source_commit':'f06695d','review':'evidence/inherited_h77c_review.json',
           'raw_source_archive':'docs/archive/concurrent-h77c','original_tiff_sha256':sha(path)})
    for dest in [ROOT/'registry/submission_build.json',path.with_name(path.stem.replace('-zeros','-audit')+'.json')]:
        original=json.loads(dest.read_text())
        original['current_review_status']='QUARANTINED — not approved for submission'
        original['superseding_review']='evidence/inherited_h77c_review.json'
        original['ok_to_submit']=False
        original['run_card']['verdict']='negative'
        original['run_card']['approved_for_weekly_slot']=False
        original['run_card']['submission']['portal_ready']=False
        original['run_card']['submission']['verdict_download_ok']='Research-only historical artifact. NOT OK TO SUBMIT; no guarantee of portal acceptance.'
        original['run_card']['holdout_valid_for_promotion']=False
        original['run_card']['correlation_overlap_vs_registry']['original_claim_status']='Unverified; superseded by shared measured rejection'
        original['run_card']['correlation_overlap_vs_registry']['verdict']='FAIL — shared literal dense-witness gate; see superseding review'
        original['run_card']['correlation_overlap_vs_registry']['correlation']=None
        original['run_card']['correlation_overlap_vs_registry']['overlap_3px']=result['worst_dot_overlap']
        dump(dest,original)
    print('Inherited H77C preserved and quarantined; literal overlap',result['worst_dot_overlap'])

if __name__=='__main__':main()
