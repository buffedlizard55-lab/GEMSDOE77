#!/usr/bin/env python3
"""Reassert the current H85-FTMR fail-closed Pages ordering.

This compatibility entry point is kept because the Pages workflow historically
called ``build_h85_site.py`` for an H85 promotion card.  The current session's
binding protocol stopped H85 before final-dot placement, so this script must not
re-run an experiment, change registry semantics, or inject an approval card.
It only verifies that ``build_home.py`` already rendered the inspection-only
H85-FTMR card first and re-applies the idempotent ordering helper.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_home import _ensure_h85_first  # noqa: E402


def main() -> int:
    _ensure_h85_first()
    for name in ("index.html", "executive-summary.html"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        start = text.index('<main><div class="wrap">')
        first = text.index('<section class="hero"', start)
        if 'id="h85-latest"' not in text[first:text.find('>', first) + 1]:
            raise SystemExit(f"{name}: fail-closed H85-FTMR card is not first")
        if "NOT OK TO SUBMIT" not in text[first:first + 2500]:
            raise SystemExit(f"{name}: H85 card does not state NOT OK TO SUBMIT")
    print("H85-FTMR fail-closed card verified; no promotion card injected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
