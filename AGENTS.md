# Start here each session

1. Read README.md, including the original project request, then docs/irregularities.md and registry/preregistration.json.
2. Read evidence/run_card.json. H77 is stopped and negative. Never tune/re-run a failed lane or spend a competition slot without a separately authorized lane/selector decision. Also negative and closed (do not re-tune): LIA (docs/data/run-card.json), contact-physics, H78 (evidence/h78_run_card.json).
3. Use vendor/gems57 unchanged; verify registry/template_pin.json. Shared fixes belong upstream, not in a private evaluator or gate.
4. Distinguish HOLDOUT-DTI, ORGANIZER-CONFIRMED receipts, canary diagnostics, format/input measurements and unverified claims. Never promise a leaderboard gain.
5. Keep downloads and approval status unambiguous. Format PASS is not uniqueness or approval.

6. Concurrent H77C code/artifacts are preserved for audit only; their generation entrypoint is quarantined. Do not use inherited unverified registry values or private holdout as clearance. See evidence/inherited_h77c_review.json.

7. H79 (2026-10-21, `evidence/h79_run_card.json`, `registry/preregistration_h79.json`) is CLOSED with a positive fold-fair result; do not re-tune it. Its two signed hanging-wall channels are measured useless (seed-30 ablation); the keeper is strike-aware context geometry + catalogue attributes rebuilt from visible faults (`src/gems79/attrs.py`). Catalogue-increment learning is DEAD: `labels.tif` == legacy USGS traces exactly (IR-79-04).
8. Submission status is a three-state policy (`docs/policy_uniqueness.md`, `scripts/build_home.py::status_of`, CI): CLEARED_FOR_SELECTOR / DOWNLOAD_ONLY / QUARANTINED. The 3 px overlap test binds only against discriminating rasters (`registry/registry_halo_coverage.json`); the literal reading is always published. Revert in those three places if the protocol owner disagrees; never hide the literal reading.
9. Fetch the uniqueness registry with `scripts/fetch_registry_git.py` (git, unmetered), not the REST API (shared 5,000/h quota, IR-79-02).
