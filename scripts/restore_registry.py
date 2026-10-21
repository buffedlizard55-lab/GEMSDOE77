"""Restore two pinned rejection witnesses, never assert full registry coverage."""
import json
import subprocess
from pathlib import Path
from fetch_data import ROOT, sha


def get_registry():
    pins = json.loads((ROOT/'registry/witness_sources.json').read_text())
    rows = []
    for key, meta in pins.items():
        path = ROOT/'.cache/registry'/('dense_witness.tif' if key == 'dense_witness' else 'h33.tif')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or sha(path) != meta['sha256']:
            tmp = path.with_suffix('.partial')
            try:
                with tmp.open('wb') as out:
                    subprocess.run(['gh','api',f'repos/{meta["repo"]}/contents/{meta["path"]}?ref={meta["ref"]}',
                                   '-H','Accept: application/vnd.github.raw'],stdout=out,check=True)
                if sha(tmp) != meta['sha256']: raise ValueError('registry pin mismatch')
                tmp.replace(path)
            finally: tmp.unlink(missing_ok=True)
        rows.append(dict(repo=meta['repo'],submission=meta['path'],file=str(path),sha256=meta['sha256']))
    return rows

if __name__ == '__main__': print(json.dumps(get_registry(),indent=2))
