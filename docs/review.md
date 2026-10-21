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

## Follow-up repository review — 2026-10-21 (no new experiment)

### Pass 1 — cross-check public-facing status against immutable receipts

- Found a contradiction: the LIA landing/summary described the TIFF as "safe to upload" and "unique" after excluding 43 dense rasters, while the same run card records `template_literal_reading_duplicate: true`, `template_like_for_like_reading_duplicate: true`, and 35 space-filling-raster overlap firings.
- The user's protocol says **any** registry raster exceeding 70% is a duplicate and gives no density exemption. Therefore the informative-only overlap (0.660) is a diagnostic, not a protocol pass.
- Found another overclaim: local format checks had been phrased as certainty that the external portal would accept the file, without an organizer receipt. Reclassified this as local validation only.

### Pass 2 — fix source data, generator, and visible interface

- Changed the LIA run-card interpretation to `protocol_unique: false` and `ok_to_submit: false`; retained historical measured overlap values as-is.
- Updated `docs/data/lanes.json`, the README, the generated LIA overview/submission/analysis pages, and the root overview/guide so download is explicitly **for inspection only**, and submission is **not approved**.
- Updated `scripts/make_run_card.py` to use the literal all-registry uniqueness gate in promotion criteria. Updated the site generator to distinguish the informative-only subset from literal protocol uniqueness and to avoid asserting portal acceptance.
- Added a regression test so future site builds cannot reintroduce the contradictory approval language.

### Pass 3 — revalidate and edge-case review

- Rebuilt both static-site families using their source scripts; static link/status suite and all repository tests pass: **55 passed, 3 skipped**.
- Read the actual LIA TIFF back with rasterio. Its SHA-256 is `a8bfb844ab44b5d4157f75ec914a51655c27ae54727f7b4a088eca9d6f396898`; it is one float32 band, EPSG:32611, 3730×3292, matching the local sample mirror's transform; every cell is finite, min/max 0/1, and 38,323 positive cells. One-TIFF ZIP has byte-identical contents. It has zero positive overlap with the locally mirrored catalogue raster.
- This is **format verification against owner-held local mirrors**, not independent organizer authentication or portal acceptance. No model rerun, holdout experiment, uniqueness exemption, or competition slot was used. No unique candidate is cleared for upload; keep the existing TIFF as a negative research artifact only.

### Remaining blocker

The supplied literal uniqueness rule and the currently measured space-filling registry witnesses are incompatible with the observed candidate: this TIFF fires 35 overlaps even though random placement also exceeds the threshold for those dense witnesses. The correct next action is selector/protocol-owner policy clarification or a registry representation fix in the shared template—not a private exemption and not another submission.
