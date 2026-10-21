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
    lia=next(x for x in lanes['lanes'] if x['lane'].startswith('LIA:'))
    assert lia['protocol_unique'] is False
    assert 'not ok to submit' in lia['upload'].lower()
    assert (DOC/lia['file']).is_file()
    page=(DOC/'lia-executive-summary.html').read_text()
    assert 'Literal protocol uniqueness: FAIL' in page
    assert ('Can I submit it?</b> <b>No' in page or 'Can I upload it?</b> <b>No' in page)
    assert 'Can I upload it?</b> Yes.' not in page
    assert 'do not upload' in (DOC/'lia-executive-summary.html').read_text().lower()
    contact=next(x for x in lanes['lanes'] if x['lane'].startswith('Contact-physics'))
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
    h80=(out/'docs/h80.html').read_text()
    assert (out/'docs/evidence/h80_run_card.json').is_file()
    assert (out/'docs/research/h80_hypotheses.md').is_file()
    assert 'href="evidence/h80_run_card.json"' in h80
    assert 'href="research/h80_hypotheses.md"' in h80
    # Re-running replaces only a marked generated directory.
    assert stage(out)==out


def test_home_pages_put_download_and_per_file_status_first():
    """docs/index.html is the repository-wide entry page (scripts/build_home.py), not one lane's view."""
    lanes=json.loads((DOC/'data/lanes.json').read_text())['lanes']
    home=(DOC/'index.html').read_text()
    head=home[:home.index('Reviewed downloadable candidates')]
    assert 'Download .tif' in head and 'Can I submit it?' in head and 'Will it beat 0.2778?' in head
    assert 'Not OK to submit' in head and 'OK to download' in head
    assert 'Can I submit it?</b> Yes' not in home and 'Can I upload it?</b> Yes' not in home
    for x in lanes:
        assert (DOC/x['file']).exists(), x['file']
        assert Path(x['file']).name in home
        assert (DOC/x['page']).exists(), x['page']
    guide=(DOC/'executive-summary.html').read_text()
    assert 'single-band float32 GeoTIFF' in guide and 'Note' in guide and 'EPSG:32611' in guide and '[0, 1]' in guide
    assert 'Not OK to submit' in guide
    prim=[x for x in lanes if x.get('primary')]
    assert len(prim)==1 and prim[0]['lane'].startswith('H80:')
    assert prim[0]['format_valid'] and prim[0]['protocol_unique'] is False
    assert prim[0]['name'] in home and prim[0]['note'] in home and len(prim[0]['note'])<=140
    assert 'HOLDOUT-DTI 0.09579668 (gems57-pooled-hide-v2; 11,319 withheld positives; 95% CI 0.08818346–0.10324797)' in home
    assert 'Will it beat 0.2778?</b> No claim that this file beats' in home


def test_new_research_raster_is_visible_but_never_called_a_submittable_entry():
    import hashlib
    card=json.loads((ROOT/'evidence/h77l_run_card.json').read_text())
    lane=next(x for x in json.loads((DOC/'data/lanes.json').read_text())['lanes']
              if x['lane'].startswith('H77L:'))
    tif=ROOT/card['raster_path']
    home=(DOC/'index.html').read_text()
    guide=(DOC/'executive-summary.html').read_text()
    detail=(DOC/'h77l.html').read_text()
    assert tif.is_file() and tif.stat().st_size>1000
    assert hashlib.sha256(tif.read_bytes()).hexdigest()==card['raster_sha256']==lane['sha256']
    assert lane['primary'] is False and lane['protocol_unique'] is False and lane['verdict']=='negative'
    assert card['validator']['all_checks_passed'] and not card['ok_to_submit']
    assert card['registry']['surface_max_forward_overlap_3px']>0.70
    assert card['registry']['final_dot_placement_audit'].startswith('NOT RUN')
    assert card['holdout_dti']['withheld_positive_pixels']>0
    assert card['holdout_dti']['evaluator_version']=='gems57-pooled-hide-v2'
    assert len(card['holdout_dti']['ci95'])==2
    assert card['leakage_canary']['flagged'] is False
    assert card['leakage_canary']['highest_auc']<card['leakage_canary']['threshold_auc']==0.90
    assert card['organizer_confirmed_score_for_this_tif'] is None and card['weekly_slots_spent']==0
    assert card['note_length']==len(card['submission_note'])<=140
    assert tif.name in home and tif.name in detail and 'RESEARCH ONLY' in guide
    assert 'NOT OK TO SUBMIT' in home and 'NOT OK TO SUBMIT' in detail
    assert '0.0 or 1.0' not in home[:home.index('Reviewed downloadable candidates')]
    assert 'row #48' in home and 'OWNER-REPORT' in home
    assert (DOC/'evidence/h77l_run_card.json').is_file()
    assert 'independent re-read audit' in detail and 'three-pass review' in detail
    audit=json.loads((DOC/'evidence/h77l_independent_audit.json').read_text())
    assert '68.4%' in detail and f"{audit['stats']['local_prior_tifs_checked']} other local TIFFs" in detail
    assert audit['all_checks_passed'] and audit['checks']['pixel_distinct_from_all_local_prior_tifs']
    assert audit['stats']['local_prior_tifs_checked']>=14
    assert 'Unknown. H77L' not in home and 'Will it beat 0.2778?</b> No claim that this file beats' in home


def test_h80_featured_homepage_is_generated_from_its_negative_run_card():
    import hashlib
    lanes=json.loads((DOC/'data/lanes.json').read_text())['lanes']
    lane=next(x for x in lanes if x['lane'].startswith('H80:'))
    card=json.loads((ROOT/'evidence/h80_run_card.json').read_text())
    tif=DOC/lane['file']
    home=(DOC/'index.html').read_text()
    assert lane['primary'] is True and lane['verdict']=='negative'
    assert lane['file']==card['download'].removeprefix('docs/')
    assert hashlib.sha256(tif.read_bytes()).hexdigest()==card['raster_sha256']==lane['sha256']
    hold=card['holdout_dti']
    assert hold['evidence_class']=='HOLDOUT-DTI'
    assert hold['evaluator_version']=='gems57-pooled-hide-v2'
    assert hold['n_withheld_positives']==11319
    assert hold['value'] < 0.152718
    assert len(hold['ci95'])==2
    assert card['validator']['ok'] and card['validator']['n_dots']==12000
    overlap=card['correlation_overlap_vs_registry']
    assert overlap['n_compared']==19 and overlap['max_3px_overlap']>0.70
    assert card['safe_to_submit'] is False
    assert 'Can I submit it?</b> <b>No. NOT OK TO SUBMIT.' in home
    assert 'HOLDOUT-DTI 0.09579668' in home and '11,319 withheld positives' in home
    assert '0.981583' in home and '19 local rasters' in home
    assert 'Pre-placement surface rank-correlation is not recorded' in home
    assert 'Will it beat 0.2778?</b> No claim that this file beats' in home


def test_ago_quick_proxy_is_not_promoted_as_a_score_or_submission():
    lanes=json.loads((DOC/'data/lanes.json').read_text())['lanes']
    ago=[x for x in lanes if x['lane'].startswith('AGO-')]
    assert len(ago)==3 and all(x['verdict']=='negative' and x['primary'] is False for x in ago)
    assert all(x['protocol_unique'] is False for x in ago)
    assert all('Not OK to submit' in x['upload'] and 'inspection only' in x['upload'] for x in ago)
    assert all('quick 4-quadrant' in x['upload'] and 'exact withheld-positive count and 95% CI' in x['upload'] for x in ago)
    page=(DOC/'ago.html').read_text()
    assert 'NOT OK TO SUBMIT' in page and 'DO NOT SUBMIT' in page
    assert 'not reportable HOLDOUT-DTI scores' in page
    assert 'exact withheld-positive count and a 95% confidence interval are absent' in page
    assert 'How to submit this file' not in page and 'RECOMMENDED for portal' not in page
    assert '0.1034' not in page and '0.0894' not in page and '0.1444' not in page
    assert 'portal acceptance' in page.lower()
    builder=(ROOT/'scripts/build_ago_page.py').read_text()
    assert '/tmp/ago_final_info.json' not in builder
    assert 'evidence/ago_run_card.json' in builder
    assert 'No submission steps apply' in builder and 'NOT OK TO SUBMIT' in builder


def test_negative_lanes_are_not_recommended_or_promoted_on_entry_page():
    """The aggregate landing page must not recommend files whose overall verdict is negative."""
    lanes=json.loads((DOC/'data/lanes.json').read_text())
    assert all(x['verdict']=='negative' and x['protocol_unique'] is False for x in lanes['lanes'])
    assert 'not ok to submit' in lanes['note'].lower()
    for lane in lanes['lanes']:
        status=lane['upload'].lower()
        assert 'not ok to submit' in status
        assert 'inspection only' in status
        assert 'recommended' not in status and 'co-primary' not in status
    for lane in lanes['lanes']:
        if 'selector_recommendation' in lane:
            assert 'not approved' in lane['selector_recommendation'].lower()
    for path in ('index.html','executive-summary.html'):
        page=(DOC/path).read_text().lower()
        assert 'top candidates for human selector review' not in page
        assert 'recommended for human selector sign-off' not in page
        assert 'recommended co-primary file' not in page
    for subpage in ('h77d.html','h77d-executive-summary.html','h77-step.html','h77-step-executive-summary.html','urank.html','urank-executive-summary.html','h80t.html','h80t-executive-summary.html'):
        txt=(DOC/subpage).read_text().lower()
        assert 'not ok to submit' in txt, subpage
        assert 'not approved' in txt, subpage
        assert 'lane verdict: promote' not in txt, subpage
        assert 'recommended for drivendata upload' not in txt, subpage


def test_h80t_lane_artifact_and_run_card_fail_closed():
    import hashlib
    card = json.loads((ROOT / 'evidence/h80t_run_card.json').read_text())
    uniq = json.loads((ROOT / 'evidence/h80t_uniqueness.json').read_text())
    lanes = json.loads((DOC / 'data/lanes.json').read_text())['lanes']
    row = next(x for x in lanes if x['lane'].startswith('H80T:'))
    tif = ROOT / card['submission_raster']['pages_download']
    zip_p = ROOT / card['submission_raster']['zip_download']
    assert tif.is_file() and zip_p.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card['raster_sha256'] == row['sha256']
    assert card['verdict'] == 'negative' and not card['safe_to_download_and_submit'] and not card['ok_to_submit']
    assert row['verdict'] == 'negative' and row['protocol_unique'] is False and row['informative_unique'] is True
    assert card['leakage_canary']['pass'] is True and card['leakage_canary']['max_sym_auc'] <= 0.90
    assert card['holdout_dti']['evaluator_version'] == 'gems57-pooled-hide-v2'
    assert card['holdout_dti']['withheld_positive_pixels'] == 22893
    assert len(card['submission_note']) <= 140
    assert uniq['phase1_surface_before_placement']['worst_spearman_full_footprint'] <= 0.90
    assert uniq['phase2_final_dots']['worst_dot_overlap_informative_dot_fields'] <= 0.70
    assert uniq['phase2_final_dots']['worst_dot_overlap_literal'] > 0.70

