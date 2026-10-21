#!/usr/bin/env python3
"""Re-integrate origin/main into the H82-FMD branch, deterministically.

Several sessions publish lanes into this repository in parallel, so main moves
while a lane is being written. The conflicts are always the same three kinds,
and each has one correct resolution:

* generated pages and manifests (``docs/**.html``, ``docs/data/*.json``)
  -> take main's copy, then re-run the site builders, which re-emit this lane's
  pages from its own frozen run card;
* ``scripts/build_home.py`` -> take main's copy and re-apply the H82-FMD
  renderers additively (they are a self-contained block plus two call lines and
  two NAV entries);
* ``README.md`` / ``docs/irregularities.md`` -> keep main's text whole and
  re-insert this lane's banner/section, preserving main's "first 2,000
  characters" contract asserted by tests/test_h82_deliverable.py.

Running this script never edits a number, a receipt or the shipped raster.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNER_START = "<!-- H82 NEWEST LANE STATUS BANNER -->"
IRR_HEADING = "## H82-FMD updates — 2026-10-21 (mapping-deficit allocation"


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check).stdout


def show(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")


def conflicted() -> list[str]:
    return [l.split("\t")[-1] for l in git("diff", "--name-only", "--diff-filter=U").splitlines() if l.strip()]


def resolve_readme(mine: str, theirs: str) -> str:
    start = mine.index(BANNER_START)
    end = mine.index("\n# GEMSDOE77: DOE GEMS Geothermal Fault Detection") if \
        "\n# GEMSDOE77: DOE GEMS Geothermal Fault Detection" in mine[start:] else len(mine)
    banner = mine[start:end].strip("\n")
    base = theirs.replace(banner, "").replace(BANNER_START, "")
    cuts = [m.start() for m in re.finditer(r"\n(?=(#|> \*\*|<!--))", base) if m.start() > 2500]
    pos = cuts[0] if cuts else len(base)
    return base[:pos] + "\n\n" + banner + "\n" + base[pos:]


def resolve_irregularities(mine: str, theirs: str) -> str:
    section = mine[mine.index(IRR_HEADING):]
    base = theirs.split(IRR_HEADING)[0].rstrip("\n")
    return base + "\n\n" + section


def resolve_build_home(mine: str, theirs: str) -> str:
    start = mine.index("# ---------------------------------------------------------------------------\n# H82-FMD")
    block = mine[start:]
    if 'if __name__ == "__main__":' in block:
        block = block[:block.index('if __name__ == "__main__":')]
    assert "_write_h82fmd" in block and "_inject_h82fmd_hero" in block
    out = theirs
    if "_write_h82fmd" not in out:
        anchor = 'if __name__ == "__main__":'
        head, tail = out.split(anchor, 1)
        out = head.rstrip("\n") + "\n\n\n" + block.rstrip("\n") + "\n\n\n" + anchor + tail
        out = out.rstrip("\n") + "\n    _write_h82fmd()\n    _inject_h82fmd_hero()\n"
    if '("h82fmd.html"' not in out:
        m = re.search(r"NAV = \[\(", out)
        out = (out[:m.end() - 1] + '("h82fmd.html", "H82-FMD · negative"),\n       '
               '("h82fmd-executive-summary.html", "H82-FMD guide"),\n       (' + out[m.end():])
    compile(out, "build_home.py", "exec")
    return out


def main() -> int:
    git("fetch", "origin", "main")
    r = subprocess.run(["git", "merge", "--no-commit", "--no-ff", "origin/main"],
                       cwd=ROOT, capture_output=True, text=True)
    files = conflicted()
    print("conflicts:", len(files))
    for f in files:
        mine, theirs = show("HEAD", f), show("origin/main", f)
        if f == "README.md":
            Path(ROOT / f).write_text(resolve_readme(mine, theirs))
        elif f == "docs/irregularities.md":
            Path(ROOT / f).write_text(resolve_irregularities(mine, theirs))
        elif f == "scripts/build_home.py":
            Path(ROOT / f).write_text(resolve_build_home(mine, theirs))
        else:                      # generated artifacts: main's copy, then rebuild
            Path(ROOT / f).write_text(theirs)
        git("add", "--", f)
        print("  resolved", f)
    if r.returncode and not files:
        print(r.stdout, r.stderr, file=sys.stderr)
        return r.returncode
    for step in (["scripts/build_site.py"], ["scripts/lia_build_site.py"],
                 ["scripts/h82fmd_run_card.py"], ["scripts/build_home.py"],
                 ["scripts/release_gate.py", "--verify-fail-closed"]):
        p = subprocess.run([str(ROOT / ".venv/bin/python"), *[str(ROOT / s) if s.endswith(".py") else s
                                                              for s in step]],
                           cwd=ROOT, capture_output=True, text=True)
        print(f"  {step[0]} -> {p.returncode}")
        if p.returncode:
            print(p.stdout[-800:], p.stderr[-800:], file=sys.stderr)
            return p.returncode
    git("add", "-A")
    print("merged; review `git diff --cached --stat` and run pytest before committing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
