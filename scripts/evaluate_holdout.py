#!/usr/bin/env python3
"""Direct pass-through CLI entrypoint to shared upstream ``vendor/gems57/evaluate_holdout.py``."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from gems57.evaluate_holdout import (  # noqa: E402, F401
    EVALUATOR_VERSION,
    evaluate,
    implementation_hashes,
    main,
    pooled_summary,
)

if __name__ == "__main__":
    raise SystemExit(main())
