#!/usr/bin/env python3
"""Direct pass-through CLI entrypoint to shared upstream ``vendor/gems57/submission_writer.py``."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from gems57.submission_writer import write_submission  # noqa: E402, F401

if __name__ == "__main__":
    print("Shared upstream submission_writer ready (gems57.submission_writer.write_submission).")
