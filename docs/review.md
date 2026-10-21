# Three-pass implementation review

## Pass 1 — implement and measure

- Audited the actual empty base repository and accessible related repositories.
- Registered four hypotheses before implementing the top one. Reused pinned cached radiometric inputs and unchanged shared evaluator/writer.
- Executed one exploratory experiment: whole-component buffered spatial holdout, feature-alone canaries, pooled DTI and aligned physical-block conditional intervals.
- Surface gate failed under the literal support rule. Stopped before dot placement; no tuning, second experiment or competition slot.
- Preserved the new surface as a readback-validated, research-only TIFF and one-TIFF ZIP. Built a prominently labelled site and submission guide.

## Pass 2 — attack assumptions and edge cases

- Removed absolute-path operational registry manifests; witness sources now restore from pinned repository refs and hashes.
- Rechecked actual serialized bytes, rather than claiming byte uniqueness from a pre-serialization null file hash. Confirmed the decoded surface is unchanged.
- Recomputed a whole-footprint saturation certificate: the dense witness blocks every nonempty allowed candidate under the literal rule.
- Separated score evidence from AUC diagnostics, input/format measurements, family webpage mentions and user assertions.
- Tested metric against brute force, exact kernel boundary behavior, entire-component hiding, buffered split, visible-only masking, bootstrap alignment, degenerate inputs, invalid writer values, stop guard, TIFF/ZIP bytes, site links and status.
- Test output: [tests-pass2.txt](evidence/tests-pass2.txt). Any failures and fixes are described in the final handoff/PR.

## Pass 3 — request-by-request recheck

- Recheck source permalinks against commit SHA, not tree SHA; retain timestamp/error states for all supplied family repositories.
- Preserve the standing request in README and a separate linked document; add next-session instructions and honest capability boundaries.
- Add CI and scheduled Pages publication with no auto-training, auto-promotion or portal action.
- Independently validate the immutable TIFF and single-file ZIP again; run the full suite and static-link checks.
- Found and diagnosed diagonal nodata-stencil contact in the historical feature domain. Added a fail-closed guard and regression test; retained the negative artifact/observed score without rerunning the stopped experiment. The historical computation is preserved in commit `1e5c994`, for audit only.
- Test output: [tests-pass3.txt](evidence/tests-pass3.txt).

## Concurrent-main merge safety review

Main advanced via PRs #1/#2 during this run. Preserved their code, mirrored inputs, tests and TIFFs; renamed the U-lane feature module to avoid a namespace collision. Archived original incoming pages/code/claims as inert text. The inherited generator contained numerical fallbacks for absent registry measurements, and its metadata claimed PASS while saying values would be computed later. Independently audited incoming TIFF with the unchanged shared gate: rejection. Disabled the unsafe generation entrypoint and superseded approval claims without changing TIFF bytes. This is integration/immutable-byte review, not another geological experiment. Re-ran the combined suite.

## Publication compatibility review

PR #3 merged and the scheduled Pages workflow deployed successfully. The Pages-settings update API was denied by integration permissions, so existing legacy settings remain. Both publishing paths now intentionally use the same root redirect plus `/docs/` layout, preserving the original prompt’s deep links. Only public docs and the redirect are staged; repository internals and raw data are not uploaded. The deployment workflow’s Node runtime/runner migration warnings were non-fatal.

## Not fulfilled / not claimed

A protocol-unique, submission-approved TIFF has **not** been achieved; the measured universal overlap blocker prevents that under the supplied literal rule. No improved organizer score, current leaderboard record, complete family novelty, direct official PDF verification, full GPU training pipeline, or authenticated external-data eligibility is claimed. The negative research artifact is the deliverable required by the stop protocol. It is downloadable, but NOT OK TO SUBMIT.

---

## Lane H77-HSL Three-Pass Implementation & Verification Review (2026-10-21)

### Pass 1 — Implement and verify (`GEMSDOE77-PREREG-HSL-1`)
- Restored and SHA-256-verified all competition and external rasters plus all 812 registry rasters into `.cache/`.
- Formulated and ranked 5 geological hypotheses (`H77-HSL-1` through `H77-HSL-5` in `research/hypotheses_hsl.md`) and preregistered `registry/preregistration_hsl.json` prior to holdout evaluation.
- Implemented the 77-channel label-free feature stack (`src/gems77hsl/features_hsl.py`) plus 7 fold-visible structural continuation channels and the Bayes-calibrated submodular 300 m kernel coverage emitter (`src/gems77hsl/emit_hsl.py`).
- Executed Experiment 1 (`scripts/hsl_exp1_holdout.py` -> `evidence/hsl_holdout_exp1.json` on Seed 30, `11,342` withheld positives) and Experiment 2 (`scripts/hsl_exp2_replication.py` -> `evidence/hsl_holdout_exp2_replication.json` on Seed 21, `11,572` withheld positives, and Two-Draw Pooled across `22,914` withheld positives):
  - All 84 channels passed the single-feature leakage canary (`max symmetric AUC = 0.7982 <= 0.90`; label-free `max AUC = 0.7186 <= 0.90`).
  - Two-Draw Pooled (`22,914` withheld positives) `HOLDOUT-DTI`: `HSL_nolabel_submod = 0.1197 [0.1075, 0.1319]` (`+0.0117 [+0.0042, +0.0194]` over `C3_contactphys`), `HSL_stratified_submod = 0.1473 [0.1359, 0.1588]` (`+0.0393 [+0.0304, +0.0490]` over `C3_contactphys`), and `HSL_structural_submod = 0.1729 [0.1610, 0.1848]` (`+0.0649 [+0.0540, +0.0758]` over `C3_contactphys`).
- Executed Experiment 3 (`scripts/hsl_exp3_build_submission.py` -> `evidence/hsl_exp3_submission.json`): generated `docs/downloads/gems77-hsl-37654dots-20261021T052500Z-7b13aa7e-zeros.tif` (`SHA-256: 31c123237c6301faf0aa2b7ddec7e13727832f423c89b4352ef51d3ed28d45e0`) and `.zip` via shared `vendor/gems57/submission_writer.py`, passing `17/17` portal checks and achieving `max 1-way 3-px dot overlap = 0.6550 <= 0.70` and `max |Spearman rho| = 0.6945 <= 0.90` across all `765` informative registry rasters + sibling lanes.

### Pass 2 — Bug, edge-case, and artifact-hygiene review
- Audited and fixed Band 6 (`tc` = radiometric Total Count, `IR-77-01`) and Band 18 (`iso_grav_anom_hg` = signed x-derivative, `IR-77-03`) documentation in `knowledge/01_geothermal-vents.md` (`IR-77-10`).
- Fixed CSS overwrite bug in `scripts/build_site_contactphys.py` so it writes to `docs/assets/contactphys.css` rather than clobbering `docs/assets/site.css` (`IR-77-11`).
- Documented the 4 corrupt registry TIFFs (`MissingRequired: StripOffsets`) and 43 space-filling continuous probability surfaces in `research/irregularities.md` (`IR-77-12`) and `docs/irregularities.md` (`I19`).
- Fixed `0600` temp-file permissions from `submission_writer.py` to `0644` in `docs/downloads/gems77-hsl-*` and moved all large intermediate `.npy` caches into `.cache/hsl_processed/` so the repository working tree stays compact and clean.

### Pass 3 — Re-check against the original request
- Verified that `README.md`, `docs/index.html`, `docs/executive-summary.html`, `docs/hsl.html`, and `docs/hsl-executive-summary.html` make it immediately obvious at the top whether the generated TIF is OK to download and submit, how to submit it, its unique submission name (`gems77-hsl-37654dots-20261021T052500Z-7b13aa7e-zeros`), and its `<= 140`-character note (`130` chars).
- Verified that `README.md` retains `docs/PROJECT_PROMPT.md` verbatim at the end, that all `vendor/gems57/*.py` files match `registry/template_pin.json` byte-for-byte, and that all `58` unit and integration tests pass (`pytest -v`).

