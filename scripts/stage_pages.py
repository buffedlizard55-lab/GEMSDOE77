"""Stage only public files, preserving both / and the prompt's /docs/ URLs.

The repository's existing legacy Pages source uses a root redirect to docs/.
The scheduled Actions deployment must preserve that same layout, not move docs
contents to root and silently break all existing /docs/index.html links.
"""
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def stage(destination=None):
    destination=Path(destination) if destination is not None else ROOT/'.cache/pages'
    marker=destination/'.generated-public-pages'
    if destination.exists():
        if not marker.is_file(): raise FileExistsError('refusing to replace a directory not owned by Pages staging')
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    marker.write_text('generated public site only\n')
    shutil.copytree(ROOT/'docs',destination/'docs')
    shutil.copyfile(ROOT/'index.html',destination/'index.html')
    (destination/'.nojekyll').write_text('')
    return destination

if __name__=='__main__':print(stage())
