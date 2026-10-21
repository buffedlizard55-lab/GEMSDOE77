# Repository review — `arena/e7aa30c3-gemsdoe77` (2026-10-21)

## Executive result

**No new TIF generated; no weekly slot used; no file is authorized to submit.** This review did not run a geological experiment or choose an existing artifact. The current instruction does not identify a single assigned method paragraph. The repository's current pinned preflight receipt says `STOP_UNIVERSAL_DUPLICATE`; the preflight implementation explains that its pinned dense witness covers all allowed cells under the literal 3-pixel overlap rule. Do not bypass it or treat an existing inspection TIFF as an approved submission.

## Pass 1 — scope and evidence

- Read the repository's `README.md`, full `PROMPT.md`, `AGENTS.md`, registry preregistration, irregularities, preflight review, shared-template pin, and committed `evidence/registry_preflight.json`.
- Confirmed the working branch is `arena/e7aa30c3-gemsdoe77`, clean, at `40075f6`, and equal to `origin/main` (`0` commits ahead/behind). There are no new changes to open as a meaningful PR or merge.
- The repo contains numerous prior TIFFs in `docs/downloads/`; these are historical research/inspection artifacts with differing lane gates. Their existence is not evidence of a newly generated, selected, or organizer-approved submission.
- The provided old leaderboard values and page inventories are owner-reported repository material, not organizer-confirmed submission receipts. No claim that H33's 0.2778 score is attributable to a specific file is established here.

## Pass 2 — reproduce gates and testability

Attempted, without changing files:

- `python scripts/preflight_registry.py` — **not executed**: this fresh environment lacks `numpy` (`ModuleNotFoundError`). The committed preflight receipt and code were inspected, but this turn did not independently reproduce its raster measurement.
- `python scripts/audit_downloads.py` — **not executed**: same missing `numpy` dependency.
- `python -m pytest -q` — **not executed**: `pytest` is not installed.

Therefore no fresh raster validation, correlation, dot-overlap audit, holdout, or test result is claimed. Reproduction instructions are documented in `docs/preflight-review.md`; setup requires the declared dependencies and hash-pinned registry witness. Installing software would not resolve the lane assignment or literal uniqueness blocker, so this review stops without experiment or file generation.

## Pass 3 — consistency, risk, and next actions

1. The opening README contains many interleaved banners from prior branches and lanes, some phrased as “current,” plus historical performance claims. This creates a risk that readers mistake old outputs or lane state for this checkout's current authorization. An authoritative banner was added above them, explicitly subordinate to the historical material below.
2. The protocol's literal overlap gate and its universal-witness claim conflict with the practical objective of producing any candidate. The shared protocol owner must resolve support/overlap semantics; this review does not change the threshold, ignore dense rasters, or create a local exception.
3. No unique-method lane is assigned in this invocation. Do not infer one from an older run card or hypothesize a new lane and consume holdout/experiment budget.
4. Do not generate or recommend a TIF for portal submission until a lane is assigned, valid inputs and pinned tools are restored, required holdout and canary checks pass, registry uniqueness is cleared, and the separate selector authorizes a particular hash.
5. Downloading an existing file for **inspection** is different from permission to **submit** it. Current decision: no new deliverable; existing files retain their own receipt-specific status; repository-wide selector authorization is zero unless a current receipt says otherwise.

## H33 score analysis: what is and is not supported

The 0.2778 value appears in the owner's supplied list and prior notes, but no official receipt tying score, submission bytes, and evaluation to one another is present in this review. Therefore it cannot support a causal conclusion about why that particular TIF scored well. General DTI mechanics in the prior H33 research are hypotheses about how sparse dots can balance truth coverage against predicted mass, not verified attribution of that reported score. A higher competition score is possible in principle but cannot be claimed or predicted from local holdout results.

## Ranked candidate directions from prior research (conditional, not validated here)

The prior preflight/research documents propose InSAR deformation gradients, microseismic lineaments, repeat Landsat thermal edges, and DEM channel-deflection diagnostics, each with non-fault mimics. They are not executable candidates in this turn: external host/data availability and current lane clearance are unresolved, and the preflight blocks output before a new feature is tested. See `docs/preflight-review.md` for the recorded details and source URLs. They must not be described as beating the current HOLDOUT-DTI best.

## Machine-readable run card

```json
{
  "hypothesis": "No new hypothesis tested; this is a repository and release-readiness review.",
  "mechanism": "No new raster or dots generated; inherited universal-witness receipt is a preflight rejection, not a scientific score.",
  "named_non_fault_mimic": "Not applicable: no geological signal was tested.",
  "holdout_dti": null,
  "holdout_ci_95": null,
  "holdout_status": "NOT RUN; evaluator version, withheld-positive count, and CI unavailable in this review",
  "correlation_overlap_vs_registry": "NOT MEASURED for a new candidate; pinned inherited preflight receipt reports universal overlap blocker; no new candidate",
  "raster_sha256": null,
  "validator_output": "NOT RUN for a new raster; no new TIFF",
  "submission_name": null,
  "submission_note": "No new candidate generated; do not submit existing inspection artifacts based on this review.",
  "verdict": "negative"
}
```

The nulls are intentional: no score, CI, registry correlation, raster hash, validator pass, or submission receipt was generated by this review. Evidence class: **REPOSITORY-REVIEW**, not HOLDOUT-DTI or ORGANIZER-CONFIRMED.
