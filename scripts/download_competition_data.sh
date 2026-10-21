#!/usr/bin/env bash
set -euo pipefail

# Autonomous Competition & External Data Restoration Script for GEMSDOE77
# Restores SHA-256-verified official GeoDAWN competition inputs + external USGS/DOE GDR layers
# into .cache/data (excluded from git commits & snapshots) with zero manual intervention.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/restore_data.py"
python3 "${ROOT_DIR}/scripts/prepare_data.py"
