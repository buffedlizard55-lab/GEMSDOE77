import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, unquote
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs'

class Links(HTMLParser):
    def __init__(self): super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src'):self.links.append(v)


def test_no_broken_local_html_links():
    for path in DOC.glob('*.html'):
        parser=Links();parser.feed(path.read_text())
        for link in parser.links:
            u=urlparse(link)
            if not u.scheme and not u.netloc and u.path:
                assert (path.parent/unquote(u.path)).exists(),(path,link)


def test_unambiguous_download_status_and_accessibility_basics():
    for path in DOC.glob('*.html'):
        s=path.read_text()
        assert 'lang="en"' in s and 'name="viewport"' in s and '<h1>' in s
        assert 'Not approved' in s or 'not approved' in s
    for name in ['urank.html','urank-executive-summary.html']:
        s=(DOC/name).read_text()
        assert 'Not OK to submit' in s and 'download href="downloads/' in s
        assert 'Do not' in s and 'research TIFF' in s


def test_historical_build_utility_cannot_delete_or_overwrite_other_lane_artifacts():
    script=(ROOT/'scripts/build_submission.py').read_text()
    assert 'for old in out.glob' not in script
    assert 'Refusing to overwrite existing artifact(s)' in script
    assert 'portal accepted and scored' not in script


def test_submission_status_obeys_literal_uniqueness_gate():
    lanes=json.loads((DOC/'data/lanes.json').read_text())
    lia=lanes['lanes'][0]
    assert lia['protocol_unique'] is False
    assert 'not ok to submit' in lia['upload'].lower()
    assert (DOC/lia['file']).is_file()
    page=(DOC/'lia-executive-summary.html').read_text()
    assert 'Literal protocol uniqueness: FAIL' in page
    assert ('Can I submit it?</b> <b>No' in page or 'Can I upload it?</b> <b>No' in page)
    assert 'Can I upload it?</b> Yes.' not in page
    assert 'do not upload' in (DOC/'lia-executive-summary.html').read_text().lower()
    contact=lanes['lanes'][1]
    assert contact['protocol_unique'] is False
    assert 'not ok to submit' in contact['upload'].lower()


def test_feed_is_not_score_verification():
    feed=json.loads((DOC/'source-health.json').read_text())
    assert len(feed['repos'])==78
    assert feed['checked_utc']
    assert all(not r.get('authenticated_score_receipt',False) for r in feed['repos'])
    assert not any(r['score_verified'] for r in feed['official_sources'])


def test_all_published_scores_have_version_count_ci():
    hold=json.loads((ROOT/'evidence/holdout.json').read_text())
    for r in hold['scores'].values():
        assert r['evidence_class']=='HOLDOUT-DTI' and r['evaluator_version']
        assert r['withheld_positive_pixels']>0 and len(r['ci95'])==2
        assert 0<=r['ci95'][0]<=r['dti']<=r['ci95'][1]<=1


def test_original_prompt_preserved_and_postrun_caveat_visible():
    assert (ROOT/'README.md').read_text().endswith((DOC/'PROJECT_PROMPT.md').read_text())
    for name in ['urank.html','research.html']:
        s=(DOC/name).read_text()
        assert 'confound' in s and ('not valid for promotion' in s or 'not valid promotion' in s)
    card=json.loads((ROOT/'evidence/run_card.json').read_text())
    assert not card['holdout_valid_for_promotion']


def test_cross_lane_release_status_fails_closed():
    """A diagnostic subset PASS must never become literal protocol clearance."""
    lanes=json.loads((DOC/'data/lanes.json').read_text())
    assert lanes['lanes']
    assert all(x['protocol_unique'] is False for x in lanes['lanes'])
    assert all('Not OK to submit' in x['upload'] for x in lanes['lanes'])

    lia=json.loads((DOC/'data/run-card.json').read_text())
    registry=lia['registry_comparison']
    assert registry['template_literal_reading_duplicate'] is True
    assert registry['space_filling_firings_in_template_audit'] > 0
    assert registry['unique_informative_reading'] is True  # narrower diagnostic only

    assert 'Not OK to submit' in (DOC/'index.html').read_text()
    lia_page=(DOC/'lia.html').read_text()
    assert '<b>Can I upload it?</b> <b>No' in lia_page
    assert 'do not upload' in (DOC/'lia-executive-summary.html').read_text()
    assert '<b>Can I upload it?</b> Yes' not in lia_page


def test_pages_staging_preserves_legacy_docs_urls_and_excludes_secrets(tmp_path):
    from stage_pages import stage
    out=stage(tmp_path/'public')
    assert 'docs/index.html' in (out/'index.html').read_text()
    assert (out/'docs/index.html').exists() and (out/'docs/assets/site.css').exists()
    assert not (out/'.git').exists() and not (out/'data').exists()
    assert set(p.name for p in out.iterdir())=={'.generated-public-pages','.nojekyll','index.html','docs'}
    # Re-running replaces only a marked generated directory.
    assert stage(out)==out


def test_home_pages_put_download_and_per_file_status_first():
    """docs/index.html is the repository-wide entry page (scripts/build_home.py), not one lane's view."""
    lanes=json.loads((DOC/'data/lanes.json').read_text())['lanes']
    home=(DOC/'index.html').read_text()
    head=home[:home.index('Every file in this repository')]
    assert 'Download .tif' in head and 'Can I submit it?' in head and 'Will it beat 0.2778?' in head
    assert 'Not OK to submit' in head and 'OK to download' in head
    assert 'Can I submit it?</b> Yes' not in home and 'Can I upload it?</b> Yes' not in home
    for x in lanes:
        assert (DOC/x['file']).exists(), x['file']
        assert Path(x['file']).name in home
        assert (DOC/x['page']).exists(), x['page']
    guide=(DOC/'executive-summary.html').read_text()
    assert 'single GeoTIFF' in guide and 'Note' in guide and 'EPSG:32611' in guide and '[0, 1]' in guide
    assert 'Not OK to submit' in guide
    prim=[x for x in lanes if x.get('primary')]
    assert len(prim)==1 and prim[0]['format_valid'] and prim[0]['protocol_unique'] is False
    assert prim[0]['name'] in home and prim[0]['note'] in home and len(prim[0]['note'])<=140
