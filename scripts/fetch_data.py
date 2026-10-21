"""Restore only pinned cached inputs from the reusable public template."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / 'registry/template_pin.json').read_text())
FILES = {
 'sample_submission.tif': ('data/bridge/sample_submission.tif', '2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc'),
 'existing_faults.tif': ('data/bridge/existing_faults.tif', '7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093'),
 'geodawn_rad_u8.tif': ('data/external/geodawn_rad_u8.tif', 'c22420f75999030d7cc65c9e31e50d232ea6158423bca051613a18a8b20ba682'),
}

def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
 return h.hexdigest()

def restore():
 rows = []
 for name, (remote, expected) in FILES.items():
  path = ROOT / 'data' / name
  path.parent.mkdir(exist_ok=True)
  if not path.exists() or sha(path) != expected:
   tmp = path.with_suffix('.partial')
   try:
    with tmp.open('wb') as out:
     subprocess.run(['gh','api',f'repos/buffedlizard55-lab/60GEMSDOE/contents/{remote}?ref={PIN["commit"]}',
                     '-H','Accept: application/vnd.github.raw'], check=True, stdout=out)
    if sha(tmp) != expected: raise ValueError(f'hash mismatch: {name}')
    tmp.replace(path)
   finally: tmp.unlink(missing_ok=True)
  rows.append({'file':name,'sha256':sha(path),'bytes':path.stat().st_size,
               'url':f'{PIN["repository"]}/blob/{PIN["commit"]}/{remote}'})
 receipt = {'evidence_class':'INPUT-INTEGRITY', 'organizer_authenticated':False,
            'caveat':'Pinned public owner mirrors, not independent official-origin authentication', 'files':rows}
 (ROOT/'evidence/data_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))

if __name__ == '__main__': restore()
