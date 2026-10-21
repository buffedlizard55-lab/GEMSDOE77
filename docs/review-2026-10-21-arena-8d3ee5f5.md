# Session review — branch `arena/8d3ee5f5-gemsdoe77` (2026-10-21)

**Result: NO new TIFF generated. NOT OK TO SUBMIT. The unique-submission step is blocked by one definitional question that only the project owner can decide.**

Evidence classes used below: **REGISTRY-MEASUREMENT** (geometry of pinned rasters), **TEST RESULT** (repo test suite), **ORGANIZER-CONFIRMED** (official leaderboard page read today), **OWNER-REPORT** (family-site or brief claim, unauthenticated), **HOLDOUT-DTI** (none produced this session).

## What was checked, line by line

| # | Check | Result | Class / source |
|---|---|---|---|
| 1 | Repo state: `git status` on the session branch | clean; branched from `main` @ `71c56db` | git |
| 2 | Agent gate files read: `README.md`, `PROMPT.md`, `AGENTS.md`, `registry/witness_sources.json`, `registry/template_pin.json`, `docs/evidence/registry_preflight.json`, `docs/data/submission-authorization.json` | all read; authorization `repository_authorized_file_count: 0` | repo files |
| 3 | Test suite `pytest` (216 tests) | **216 passed** | TEST RESULT |
| 4 | Restore pinned witnesses with the repo's own `scripts/restore_registry.py` (raw.githubusercontent.com failed TLS in this sandbox; the script's `gh api` route succeeded) | `dense_witness.tif` sha256 `ab0a0a62…872be` and `h33.tif` sha256 `c55bafc4…ab6fa9` — both match the pins | REGISTRY-MEASUREMENT |
| 5 | Re-run `scripts/preflight_registry.py` | reproduces the recorded result: `STOP_UNIVERSAL_DUPLICATE`, exit code 2, 5,106,385 / 5,106,385 allowed pixels within 3 px of the dense witness | REGISTRY-MEASUREMENT |
| 6 | Witness raster facts (rasterio) | float32, 3730×3292, EPSG:32611, transform (100, 0, 243350, 0, −100, 4508550) — matches `data/bridge/sample_submission.tif` grid; 5,167,373 finite positive pixels; ~4.7 M distinct values (continuous surface, not a binary dot set) | REGISTRY-MEASUREMENT |
| 7 | Support-semantics sensitivity (new script `scripts/support_semantics_sensitivity.py`, receipt `evidence/support_semantics_sensitivity.json`) | see table below | REGISTRY-MEASUREMENT |
| 8 | Official leaderboard, DrivenData page read today | #1 xiaofanhu **0.4413**; #2 cfld__ 0.3904; #3 Batik Shirt Brothers 0.3894. **No 0.3195 entry appears on the page's first 23 rows (the fetched page was chunked at #24).** | ORGANIZER-CONFIRMED for the listed values only |
| 9 | H33 (`GEMSDOE32 … e5eb6e7e-zeros`, owner-reported 0.2778) | The owner's own page states "NO ORGANISER SCORE EXISTS for this or any artifact in this repository." The 0.2778 is therefore **unverified**. The prior analysis is in `research/why-h33-scored-02778.md`; its marginal-DTI algebra is a derivation, not a score. | OWNER-REPORT |

### Sensitivity table (fraction of the allowed footprint within 3 px of each witness)

| Witness | Shared definition (every finite positive pixel = dot) | Binary top-37,654 convention | Exceeds 0.70 under shared? | Exceeds 0.70 under top-N? |
|---|---|---|---|---|
| Dense 17GEMSDOE surface (`ab0a0a62`) | **1.000** | 0.0189 | yes | no |
| H33 binary dots (`c55bafc4`) | 0.1519 | 0.1519 | no | no |

## The blocker, stated precisely

The shared overlap function `_dots()` in `vendor/gems57/uniqueness.py` defines a dot as any finite positive pixel. A continuous probability-like surface therefore has every pixel as a "dot". Any candidate then has 100% of its dots within 3 px of that witness, so the 0.70 rule rejects every nonempty candidate before any placement. The preflight's own `next_step` says the same thing: resolve support semantics; do not silently exempt dense rasters.

Two readings are possible:

- **Literal shared code** (current): the dense surface is a registry dot set → every candidate is blocked → no unique TIFF is possible under this rule.
- **Binary convention** (the protocol's word "dots" implies point sets): the dense surface must first be reduced to a dot set. Under a top-37,654 reduction it covers 1.9% of the footprint, so a normal candidate is far below the 0.70 limit.

Choosing between these changes the outcome, and the choice belongs to the owner who wrote the protocol. I did not change `_dots()`, did not exempt the witness, and did not generate a candidate under either reading.

## Why no new TIFF was generated

- A TIFF generated now would fall under the current literal gate, so it could not be marked OK to submit.
- Generating an inspection-only file would add another negative artifact to a site that already has dozens, with no decision value.
- The repo's stop rule, `AGENTS.md` item 7, says exit 2 is the expected stop and must not be overridden.

## Answers to the prompt's questions

- **Why did H33 score highest (0.2778)?** Not verifiable here. The only evidence is the owner's site and their file bytes. Per `research/why-h33-scored-02778.md`, the H33 mirror has 37,654 positive pixels, none on the mapped catalogue, nearest catalogue distance √5 px (REGISTRY-MEASUREMENT on the pinned mirror, not a score). Its claimed causal story is not established.
- **Can we beat 0.2778?** Unknown until a candidate is validated. The gate has to be resolved first, and every holdout number would still need a HOLDOUT-DTI label with evaluator version, withheld-positive count and 95% CI.
- **0.3195 vs 0.4413?** The official page shows 0.4413 as #1 today. The 0.3195 figure in the brief does not appear on the current leaderboard and is flagged as an irregularity.

## Irregularities to review

1. **Dense witness vs. "dots"** — the blocker above. Decision needed from the owner.
2. **0.3195 "highest score" in the brief** conflicts with the official leaderboard (#1 = 0.4413; 0.3195 not present in the first 23 rows). Flagged, not used.
3. **Owner-reported family scores** (GEMSDOE32 0.2778 and others) have no organizer receipts; the family's own page says so.
4. **Raw-host TLS failure** (`raw.githubusercontent.com` gave SSL_ERROR_SYSCALL); `gh api` works. The repo's `restore_registry.py` already uses `gh api`, so no code change was needed.
5. **Brief lists a submission error** "Predicted values must be in range [0, 1]" with unknown cause (repo README). This session did not test the portal.
6. **Data restore**: `data/raw/` (training features, 418 MB) is still absent in this checkout, so no new feature stack was built.

## Hypotheses

The repo already ranks five untested H83 hypotheses (`research/h83_hypotheses.md`). I did not add new hypotheses in this session, because each one would need the same unresolved gate and holdout before it could be ranked, and inventing unvalidated ideas here would not be useful. The ranking and implementation-cost questions are listed under next steps.

## Run card (JSON)

```json
{
  "schema": "gems-run-card-arena-8d3ee5f5-v1",
  "lane": "none assigned; registry gate review",
  "hypothesis": "No new hypothesis tested. Gate question: does a continuous witness surface count as a dot set for the 0.70 overlap rule?",
  "mechanism": "Shared _dots() treats every finite positive pixel as a dot; dense surface covers 100% of the footprint within 3 px.",
  "non_fault_process_that_could_mimic_it": "Not applicable: no candidate was generated.",
  "holdout_dti": null,
  "holdout_dti_ci95": null,
  "withheld_positives": null,
  "evaluator_version": "gems57-pooled-hide-v2 (not run)",
  "overlap_vs_registry": {
    "dense_witness_shared_definition": 1.0,
    "dense_witness_top37654": 0.018870884196941672,
    "h33_binary": 0.15188024404740338,
    "stop_threshold": 0.70,
    "label": "REGISTRY-MEASUREMENT"
  },
  "rank_correlation_vs_registry": null,
  "raster_sha256": null,
  "validator_output": "No candidate raster generated. Preflight re-run: STOP_UNIVERSAL_DUPLICATE, exit 2 (reproduced).",
  "submission_name": null,
  "submission_note": null,
  "verdict": "negative",
  "authorized_to_submit": false,
  "weekly_slots_used": 0,
  "experiments_run": 0
}
```

## Next steps (ranked by impact ÷ cost)

1. **Owner decision (blocking, cost: one answer):** choose the support semantics for continuous registry rasters — literal (blocked) or binary top-N (candidate can proceed). Record it in `registry/` and the protocol file, not in code.
2. Once decided: re-run `scripts/preflight_registry.py` and `scripts/support_semantics_sensitivity.py`, then pick the highest-ranked H83 candidate and run it through the shared evaluator on the spatial holdout first. A candidate must beat the holdout incumbent before any slot is considered.
3. Restore `data/raw/` via the existing `scripts/restore_data.py` (about 418 MB) if the feature stack is needed.
4. Obtain an official organizer receipt for any score claimed on the site; owner-reported numbers stay labelled.
5. Verify the 0.3195 figure against the official leaderboard or drop it from the brief.

## Links for manual review

- Leaderboard (fetched 2026-10-21): https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
- Competition overview: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution
- Rules PDF: https://docs.nlr.gov/docs/fy26osti/96647.pdf
- Pinned dense witness: https://github.com/buffedlizard55-lab/17GEMSDOE/blob/6fc71d19d49dca4960d5434bd81b6dd06fcc030a/docs/downloads/17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif (owner mirror, not organizer-authenticated)
- Pinned H33 mirror: https://github.com/buffedlizard55-lab/60GEMSDOE/blob/b99591eede63c6d45c8a896255b8cd34824fa00b/registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif (owner mirror)
- Family site cited for H33: https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html
