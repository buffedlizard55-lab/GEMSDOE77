"""Timestamped source health, not score verification; never auto-promote a raster."""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# Repo spellings follow the supplied URLs, not guessed aliases.
REPOS = ['GEMSDOE','GEMSDOE2','GEMSDOE3','GEMSDOE4','5GEMSDOE','6GEMSDOE','7GEMSDOE','8GEMSDOE',
 'GEMSDOE9','GEMSDOE10','11GEMSDOE','12GEMSDOE','13GEMSDOE','14GEMSDOE','15GEMSDOE','16GEMSDOE',
 '17GEMSDOE','18GEMSDOE','19GEMSDOE','20GEMSDOE','GEMSDOE21','GEMSDOE22','GEMSDOE23','GEMSDOE24',
 'GEMSDOE25','GEMSDOE26','GEMSDOE27','GEMSDOE28','GEMSDOE29','GEMSDOE30','GEMSDOE31','GEMSDOE32',
 'GEMSDOE33','GEMSDOE34','GEMSDOE35','GEMSDOE36','GEMSDOE37','GEMSDOE38','GEMSDOE39','GEMSDOE40',
 'GEMSDOE41','GEMSDOE42','GEMSDOE43','GEMSDOE44','GEMSDOE45','GEMSDOE46','GEMSDOE47','GEMSDOE48',
 'GEMSDOE49','GEMSDOE50','GEMSDOE51','GEMSDOE52','GEMSDOE53','GEMSDOE54','55GEMSDOE','56GEMSDOE',
 '57GEMSDOE','58GEMSDOE','59GEMSDOE','60GEMSDOE','61GEMSDOE','62GEMSDOE','GEMSDOE63','GEMSDOE64',
 'GEMSDOE65','GEMSDOE66','GEMSDOE67','GEMSDOE68','GEMSDOE69','GEMSDOE70','GEMSDOE71','GEMSDOE72',
 'GEMSDOE73','GEMSDOE74','GEMSDOE75','GEMSDOE76','GEMSDOE77','GEMSDOE78',
 'GEMSDOE79','GEMSDOE80','GEMSDOE81','GEMSDOE82']
OFFICIAL = [
 ('Organizer reference','https://github.com/drivendataorg/gems-prize-reference-solution'),
 ('Competition problem','https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/'),
 ('Leaderboard','https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/'),
 ('Submission/data portal','https://www.drivendata.org/competitions/306/competition-doe-gems/data/'),
 ('GeoDAWN official release','https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and'),
 ('INGENIOUS GDR','https://gdr.openei.org/submissions/1391'),
 ('Competition report','https://docs.nlr.gov/docs/fy26osti/96647.pdf'),
 ('INGENIOUS project','https://gbcge.org/current-projects/ingenious/'),
 ('CRS reference','https://epsg.io/32611')]

def gh_json(endpoint):
    p = subprocess.run(['gh','api',endpoint],capture_output=True,text=True,timeout=40)
    if p.returncode: raise RuntimeError(p.stderr.strip()[:400])
    return json.loads(p.stdout)


def repo_health(name):
    row = {'repo':name,'url':f'https://github.com/buffedlizard55-lab/{name}',
           'claim_class':'OWNER-MIRROR; not organizer score verification'}
    try:
        commit = gh_json(f'repos/buffedlizard55-lab/{name}/commits/main')['sha']
        x = gh_json(f'repos/buffedlizard55-lab/{name}/git/trees/{commit}?recursive=1')
        row.update(status='retrieved',commit=commit,tree_sha=x['sha'],truncated=x.get('truncated',False))
        blobs = [f for f in x['tree'] if f['type']=='blob']
        page = next((f for p in ['docs/index.html','index.html','README.md'] for f in blobs if f['path']==p),None)
        row['review_page'] = f'{row["url"]}/blob/{commit}/{page["path"]}' if page else None
        row['review_page_blob_sha'] = page['sha'] if page else None
        row['tif_paths'] = [f['path'] for f in blobs if f['path'].lower().endswith('.tif') and 'download' in f['path']]
        row['download_tif_count'] = len(row['tif_paths'])
        # Read actual page bytes for the supplied family sites; don't infer receipts from regex scores.
        if page:
            p = subprocess.run(['gh','api',f'repos/buffedlizard55-lab/{name}/git/blobs/{page["sha"]}',
                                '-H','Accept: application/vnd.github.raw'],capture_output=True,timeout=40)
            if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[:200])
            row['page_content_sha256'] = hashlib.sha256(p.stdout).hexdigest()
            text = p.stdout.decode('utf-8',errors='replace')
            terms = ['0.2778','0.3195','0.4413','ORGANIZER-CONFIRMED','HOLDOUT-DTI']
            row['page_mentions'] = {t:(t in text) for t in terms}
        row['authenticated_score_receipt'] = False
    except Exception as e: row.update(status='error',error=str(e))
    return row


def official_health(pair):
    name,url=pair
    row={'name':name,'url':url,'score_verified':False}
    if url.startswith('https://github.com/'):
        try:
            data=gh_json('repos/drivendataorg/gems-prize-reference-solution/commits/HEAD')
            row.update(status='retrieved via GitHub API',commit=data['sha'],
                       scope='Official reference code only; not portal, DTI-distance spec or leaderboard receipt')
        except Exception as e: row.update(status='error',error=str(e))
    else:
        # Do not evade the sandbox allowlist or mislabel lack of access as source unavailability.
        row.update(status='not fetched by scheduled CI runner (restricted outbound hosts)',
                   verification='Automated feed has no official-page check. Dated manual web-fetch review: docs/review-2026-10-21-h77l.md. Do not infer current scores from that snapshot.')
    return row


def main():
    with cf.ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(repo_health,REPOS))
    report={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'purpose':'Repository/page freshness and artifact inventory, NOT scores, scientific validation, or full raster uniqueness',
            'repos':rows,'official_sources':[official_health(p) for p in OFFICIAL],
            'score_policy':'No organizer-confirmed receipt is available. Never treat webpage mentions as verified scores.',
            'weekly_cap':None,'weekly_cap_status':'CI does not re-check official rules PDF. Dated manual review 2026-10-21 observed three feedback submissions/week; see docs/review-2026-10-21-h77l.md. Zero slots consumed by this pipeline.'}
    for path in [ROOT/'evidence/source_health.json',ROOT/'docs/source-health.json']:
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    print('Retrieved',sum(r['status']=='retrieved' for r in rows),'of',len(rows),'repository page inventories')

if __name__ == '__main__': main()
