# GEMSDOE77 independent repository review — 2026-10-21

## Decision at a glance

**No new competition submission was created in this review. Do not submit any file from this repository automatically.** The active H78 lane is already closed after its preregistered three-experiment budget and a negative holdout. Its existing GeoTIFF is downloadable for inspection, but it is not cleared for competition submission. The repository's literal uniqueness rule also flags overlap against dense registry rasters; do not bypass that rule or invent a private exception.

This is a review and evidence reconciliation, not a new experiment, not a slot-selection action, and not an organizer submission receipt.

## Scope and reproducibility

- **Branch:** `arena/914310ca-gemsdoe77`; starting worktree clean at `80dce02`.
- Read `AGENTS.md`, `PROMPT.md`, README, H78 preregistration/run card, evidence and validator code, the site entry page, and irregularity ledger.
- Re-read the lane rule: H78 was preregistered as the lane; its E1/E2/E3 budget is exhausted and the lane is explicitly closed. No fourth experiment, retuning, or promotion was undertaken.
- Installed the versions pinned in `requirements.txt` into ignored `.venv/` solely to run existing checks.
- **Tests:** `.venv/bin/python -m pytest -q` → **79 passed, 0 skipped** (including `tests/test_submission_files.py`, `tests/test_release_gate.py`, and subpage fail-closed status checks).
- **Independent full artifact script:** `.venv/bin/python scripts/verify_all.py` passes all checks (`all_passed: true`) using the SHA-256-pinned `data/bridge/` rasters (`labels.tif`, `existing_faults.tif`, `sample_submission.tif`) when `data/raw/` is absent. Note that `data/bridge/` is an integrity-pinned mirror rather than an organizer-authenticated download from login-walled DrivenData.
- Manually read H78 `.tif` with Rasterio and compared its grid to `data/bridge/sample_submission.tif`: single-band float32, EPSG:32611, 3730×3292, matching affine transform; all cells finite, min/max 0/1, no NaN in the bridge footprint, 38,301 positive cells; ZIP contains the same single TIFF byte-for-byte. **These are local format measurements against a bridge sample, not organizer authentication or submission approval.** The pinned H78 run card records the same file SHA-256: `4571e767757be95e3fb70b4afb05e45528b1cfd74147e168c36253b1247f1c47`.
- `scripts/validate_submission.py` and the full verifier require the official/raw data layout; they were not represented as passing in this review.

## The historical H33 `0.2778` question

The `0.2778` value for `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` is an **OWNER-REPORT** in the supplied family score ledger, not an organizer receipt verified in this checkout. The repository's H33 forensic note describes a plausible mechanism: spacing predictions to reduce false-positive mass under distance-weighted Tversky, then removing dots close to already mapped catalogue traces. This is a sound *metric-level explanation* for why that construction could improve over its denser ancestors; it is not proof that the exact H33 file received that organizer score or that the proposed pruning caused the reported difference.

The practical lesson is conservative: with alpha 0.2, beta 0.8 and a 300 m triangular kernel (as recorded by the project's evaluator), compact dots near hidden traces can improve the score relative to redundant dense output. Further thinning alone cannot discover new fault evidence. An improved field is needed. Whether an entirely new candidate can exceed `0.2778` is **unknown**; no leaderboard outcome can be inferred from the holdout or a projection.

Official sources supplied for manual review (not independently fetched from their non-allowlisted hosts in this session):

- [DrivenData problem description / metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [DrivenData competition rules](https://www.drivendata.org/competitions/306/competition-doe-gems/)
- [DrivenData leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- [DrivenData data page](https://www.drivendata.org/competitions/306/competition-doe-gems/data/)
- [Official reference solution repository](https://github.com/drivendataorg/gems-prize-reference-solution)
- [USGS GeoDAWN data listing](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)
- [USGS 3DEP](https://www.usgs.gov/3d-elevation-program)
- [GDR INGENIOUS regional compilation, submission 1391](https://gdr.openei.org/submissions/1391)
- [GEMS report PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf)

## H78 pre-registered candidate hypotheses and outcomes

The ranking below is copied from `evidence/h78_run_card.json`; these are the previously investigated H78 options, not a new experiment proposal. H78's target was the lidar scarp-consensus allocation. Three experiments were run; the lane is closed.

| Rank | Hypothesis and layers | Target signature / rationale | Difference recorded from prior repo work | Outcome in recorded evidence |
|---|---|---|---|---|
| 1 | **Lidar scarp consensus:** `lidar_scarp_features_u8` (seven cached 1 m DEM derivatives: extrema/step, Laplacian curvature, face and cross-slope channels) | Linear scarp with a slope break, free face, crest/toe curvature; potential small young traces absent from catalogue | Combines seven operators to rank off-support dots under a capped allocation rather than emitting a dense, single-operator raster | Negative. `HOLDOUT-DTI` `gems57-pooled-hide-v2`, 11,321 withheld positive pixels: 0.070803, 95% CI [0.060045, 0.081588], below its rebuilt champion construction at 0.095829 [0.082964, 0.108400]. Paired difference −0.025026 [−0.030103, −0.020154]. See `evidence/h78_e3_holdout.json`. |
| 2 | **Thermal manifestations:** proximity to warm/hot wells and springs from the GDR 1391 INGENIOUS compilation | Manifestations can mark blind permeable upflow structures without a surface scarp | Repository H78 preregistration says this temperature-distance layer had not been used in earlier family lanes | E1/E2 did not support it in the recorded live-score screening; the canary AUC is 0.492. GDR availability/license claims are not independently reverified in this runtime. |
| 3 | **Regional mapping heterogeneity:** 3×3 footprint blocks | Under-mapped regions might have lower catalogue completeness | Adds a spatial completeness prior instead of a local geomorphic/geophysical detector | E1 recorded zero live-supported weight. This is not evidence that a specific map campaign actually missed faults. |
| 4 | **Single lidar-operator sensitivity rows:** selected individual cached derivatives | Checks whether a single surface-break operator outperforms the seven-channel consensus | Sensitivity rows, not a separate submission lane | E1 recorded no live-supported single operator. |

Named mimic for H78-1 (from its run card): pluvial Lake Lahontan shorelines/wave-cut risers; fluvial terraces and fan-incision banks; road cuts and canal banks. These can produce linear slope breaks unlike tectonic scarps. A lidar shape is not diagnostic of fault origin by itself.

**Leakage canary:** all listed feature-only AUCs in the H78 evidence are below the protocol's 0.90 threshold; maximum recorded is visible-catalogue proximity AUC 0.7553. This says no canary crossed the preset leakage threshold; it does not establish absence of all leakage.

## Registry-uniqueness stop and artifact status

The H78 registry audit reports 839 indexed rasters, 833 locally checked and six unreadable/off-grid, so its comparison is not exhaustive of all historical/private files. Informative dot-field comparison: maximum rank correlation 0.6392 and maximum overlap 0.6600. However, the binding literal reading reports 44 overlap firings, all against space-filling rasters; its own run card sets `protocol_unique: false`, `ok_to_download: true`, `ok_to_submit: false`, and verdict `negative`. Do not relabel the TIFF “unique” or “ready to submit” based only on its SHA or its informative-comparison subset.

The website's first screen currently says the global dense-witness rule makes every non-empty candidate fail and labels H78 as inspection-only. That prominent distinction is important: **download for inspection is not permission to upload**. Existing H77D/H77-Step table descriptions also say selector approval is pending and literal uniqueness fails; language such as “co-primary” or “recommended” in older narrative sections must not be read as authorization.

## Review passes

### Pass 1 — implementation/evidence inventory

Confirmed the repository already includes the cached feature pipeline, vendored `gems57` holdout/evaluator/writer, preregistration, H78 evidence, candidate TIFF/ZIP, website, and tests. Thus the prior-session pipeline is materially different from the stale “empty repository/no TIFF” account previously recorded in this same file.

### Pass 2 — adversarial checks

- Ran 71 tests successfully (3 skipped).
- Re-read the H78 run card rather than assuming the downloadable artifact is promotable.
- Independently checked the TIFF's range, finite values, bridge grid, footprint, SHA and ZIP membership.
- Identified the missing `data/raw/sample_submission.tif` requirement preventing the full independent bundle script from running.
- Identified scope limits: 6 registry rasters unreadable/off-grid; 6 of 839 registry entries not checked locally; later sibling updates after the recorded registry snapshot are not covered; official/competition sites could not be fetched in this environment.
- Confirmed no competition slot was used and no organizer score was created.

### Pass 3 — requirements and risk re-check

The requested unique, submit-ready artifact cannot be produced responsibly in the closed lane because the actual candidate failed both the scientific holdout comparison and the binding literal uniqueness gate. Bypassing either would violate the user's own protocol. The existing H78 TIF is a unique file bytewise, but that fact is not “competition-unique” clearance. No new file or score was fabricated; no candidate was selected.

## Next actions — ordered by risk reduction

1. **Resolve the common dense-witness rule upstream.** Establish one shared rule that is meaningful for dot fields versus continuous/space-filling surfaces, with protocol-owner approval. Do not privately mask registry rasters or change the 70% threshold.
2. **Restore the official/raw competition pack to `data/raw/`** via the authorized download path; verify checksums and provenance, then rerun `scripts/verify_all.py` and the official-grid validator. Current `data/bridge/` is not a substitute for organizer-authenticated data.
3. **Resolve concurrent work before the next lane.** PR #11 (H77-HSL) was open at review time; inspect its final merge state and reconcile any shared artifacts/registry additions before selecting another lane.
4. **Get an unambiguous, separately assigned method paragraph.** Do not reuse closed H78 or another stopped lane. Preregister any new hypothesis with a 3-experiment cap, controls, feature-only canary, fold/rebuild rules and stopping criterion before execution.
5. **Verify official access and current contest state.** Authenticate the competition data/rules/leaderboard, submission cap and any organizer receipts. The cited numbers `0.3195`, `0.4413`, and family site scores are not all verified in this run; don't write them as current organizer truth without an official receipt or fresh accessible source.
6. **Only consider a real slot after evidence clears all gates:** candidate beats the current holdout best with a paired spatial CI, passes per-feature leakage canaries, matches the official grid and [0,1] format, passes the shared registry protocol on both surface and final dots, and receives separate selector approval.

## Run card (this review; no new experiment)

```json
{
  "hypothesis": "H78 lidar scarp consensus and uniqueness-capped off-support allocation (closed, negative lane)",
  "mechanism": "Seven cached 1 m DEM derivatives rank candidates with a linear scarp morphology: slope break, free face, and crest/toe curvature.",
  "named_non_fault_process_that_could_mimic_it": "Lake Lahontan shoreline benches/wave-cut risers, fluvial terraces and fan-incision banks, road cuts and canal banks.",
  "holdout_dti": {
    "label": "HOLDOUT-DTI",
    "evaluator_version": "gems57-pooled-hide-v2",
    "withheld_positive_count": 11321,
    "value": 0.07080326116269187,
    "95_percent_ci": [0.06004507090448665, 0.08158776425472165],
    "paired_delta_vs_champion_construction": -0.025025971293106353,
    "paired_delta_95_percent_ci": [-0.030103195617506226, -0.020153739950290638]
  },
  "correlation_overlap_vs_registry": {
    "comparison": "H78 run-card registry snapshot; 833 locally compared rasters of 839 indexed; 6 unreadable/off-grid",
    "informative_dot_field_max_rank_correlation": 0.639212732620308,
    "informative_dot_field_max_forward_overlap": 0.6599827680739406,
    "literal_overlap_firings": 44,
    "literal_firings_all_space_filling": true,
    "binding_protocol_unique": false
  },
  "raster_sha256": "4571e767757be95e3fb70b4afb05e45528b1cfd74147e168c36253b1247f1c47",
  "validator": {
    "independent_local_bridge_check": "single-band float32, EPSG:32611, shape 3730x3292 and affine match data/bridge/sample_submission.tif; finite all cells; range [0,1]; zero NaN inside bridge footprint; one-TIFF ZIP byte-identical",
    "official_raw_grid_validation": "NOT RUN — data/raw/sample_submission.tif absent",
    "format_result": "local format checks pass; not organizer-authenticated",
    "submission_approval": "NO — negative holdout and literal uniqueness failure"
  },
  "submission_name": "gems77-h78-lidarscarp-alloc-38301dots-20261021-429a7a12-zeros",
  "submission_note": "H78 lidar scarp consensus, 38301 binary dots; inspection only, NOT APPROVED for competition upload",
  "verdict": "negative"
}
```
