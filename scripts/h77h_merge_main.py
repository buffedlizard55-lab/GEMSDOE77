#!/usr/bin/env python3
"""Deterministic re-integration of origin/main into the H77H branch."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check).stdout


def show(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")


def main() -> int:
    git("fetch", "origin", "main")
    if not (ROOT / ".git" / "MERGE_HEAD").exists():
        r = subprocess.run(["git", "merge", "--no-commit", "--no-ff", "origin/main"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0 and not git("diff", "--name-only", "--diff-filter=U").strip():
            print("Already up to date or fast merged.")
            return 0
    files = [l.split("\t")[-1] for l in git("diff", "--name-only", "--diff-filter=U").splitlines() if l.strip()]
    print("Conflicted files:", files)

    mine_lanes = json.loads(show("HEAD", "docs/data/lanes.json"))
    h77h_entry = [x for x in mine_lanes["lanes"] if x.get("page") == "h77h.html"][0]
    mine_bh = show("HEAD", "scripts/build_home.py")
    start_fn = mine_bh.index("def _write_h77h():")
    end_fn = mine_bh.index('if __name__ == "__main__":', start_fn)
    h77h_fn_block = mine_bh[start_fn:end_fn].rstrip()

    for f in files:
        (ROOT / f).write_text(show("origin/main", f))

    lanes_data = json.loads((ROOT / "docs/data/lanes.json").read_text())
    lanes_data["lanes"] = [h77h_entry] + [x for x in lanes_data["lanes"] if x.get("page") != "h77h.html"]
    (ROOT / "docs/data/lanes.json").write_text(json.dumps(lanes_data, indent=2) + "\n")

    theirs_bh = (ROOT / "scripts/build_home.py").read_text()
    if "def _write_h77h():" not in theirs_bh:
        anchor = 'if __name__ == "__main__":'
        head, tail = theirs_bh.split(anchor, 1)
        theirs_bh = head.rstrip() + "\n\n\n" + h77h_fn_block + "\n\n\n" + anchor + tail + "\n    _write_h77h()\n"
    if '("h77h.html"' not in theirs_bh:
        theirs_bh = theirs_bh.replace(
            '("index.html", "Start here"), ',
            '("index.html", "Start here"), ("h77h.html", "H77H · piedmont-relay"), ("h77h-executive-summary.html", "H77H guide"), ',
            1,
        )
    compile(theirs_bh, "scripts/build_home.py", "exec")
    (ROOT / "scripts/build_home.py").write_text(theirs_bh)

    readme = (ROOT / "README.md").read_text()
    h77h_line = (
        '\n> **Current session — branch `arena/2ebd0138-gemsdoe77` (`H77H`, 2026-10-21):** '
        '**OK TO DOWNLOAD FOR INSPECTION: YES** ([`gems77-h77h-piedmont-relay-37654dots-20261021-56b7c98c-zeros.tif`](docs/downloads/gems77-h77h-piedmont-relay-37654dots-20261021-56b7c98c-zeros.tif), '
        'SHA-256 `e4c718f450130e2e07e14ef506f051ea974c53382740974e758858519e866fe4`, '
        '[`.zip`](docs/downloads/gems77-h77h-piedmont-relay-37654dots-20261021-56b7c98c-zeros.zip) `749d80028191124f209c00201f322fb66a389f55b900ddff0501e3150e23da00`) · '
        '**OK TO SUBMIT: NO (`verdict: "negative"` under Policy `I17`; `0` weekly slots spent).** '
        '`H77H` achieves **0.1579 HOLDOUT-DTI** `[0.1411, 0.1749]` (Seed 20) / **0.1570** `[0.1405, 0.1727]` (Seed 30) on capacity-guarded deliverable (`0.1709`/`0.1684` unguarded), '
        'beating `H33` (`0.0961`/`0.0990`), `H77-Step` (`0.1527`), and `H77E` (`0.1556`) and passing informative uniqueness (`max 3-px overlap = 0.6500 <= 0.70`, `max |ρ| = 0.0777 <= 0.90`). '
        '([`docs/h77h.html`](docs/h77h.html) · [`research/h77h_hypotheses.md`](research/h77h_hypotheses.md) · [`evidence/run_card_h77h.json`](evidence/run_card_h77h.json)).\n'
    )
    if "e4c718f450130e2e07e14ef506f051ea974c53382740974e758858519e866fe4" not in readme:
        idx = readme.index("c571af22cb08db9cb8cfd6edf0108364d95b0c3bec8316d09d3dfd116531790e")
        nl = readme.index("\n", idx)
        readme = readme[: nl + 1] + h77h_line + readme[nl + 1 :]
    (ROOT / "README.md").write_text(readme)

    for step in (
        ["scripts/build_site.py"],
        ["scripts/lia_build_site.py"],
        ["scripts/build_home.py"],
        ["scripts/release_gate.py", "--verify-fail-closed"],
    ):
        subprocess.run([str(ROOT / ".venv/bin/python"), *step], cwd=ROOT, check=True)
    git("add", "-A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
