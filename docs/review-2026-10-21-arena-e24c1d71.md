# GEMSDOE77 review — branch `arena/e24c1d71-gemsdoe77` — 2026-10-21

## Decision at a glance

**No new TIFF was generated. No candidate holdout was run. No competition slot was used. No file is cleared to submit.** Historical files linked on the repository site are inspection-only under the current release gate. This review deliberately does not rename or repackage an old TIFF as new, bypass the duplicate stop, or create a meaningless all-zero raster.

The request asks for a unique TIFF, but the binding parallel-run protocol says to stop if a candidate overlaps a registry raster beyond the stated threshold. A restored, SHA-pinned registry witness proves that **every nonempty candidate in the allowed domain would exceed that threshold**. The run stopped before calculating any candidate feature or placing dots. The failure is a representation/registry-policy contradiction, not evidence that all geological hypotheses are alike.

**Submission status:** **OK TO DOWNLOAD FOR INSPECTION: existing files only, as individually marked. NOT OK TO SUBMIT: every file. No new submission TIFF is available from this review.** The independent release receipt authorizes zero files; this is not an organizer score or portal receipt. See [`evidence/arena_e24c1d71_run_card.json`](evidence/arena_e24c1d71_run_card.json) and the [published authorization receipt](data/submission-authorization.json).

## Pass 1 — inspect the checkout and establish the gate

- Read the complete standing request in `README.md` and `PROMPT.md`, `AGENTS.md`, repository preregistrations, the active release/readiness material, the shared-template pin, and historical run cards. The request supplies the holdout protocol but does not assign a fresh scientific method paragraph. `registry/preregistration.json` describes the already-run H77 selective-U curvature lane; its negative run card is historical, not authorization to restart or retune it.
- Inspected actual files rather than relying on earlier restore receipts. `data/bridge/` contains the reference/grid bridge rasters, but this checkout has no `data/raw/training_features.tif`, `data/processed/stack.npy`, or `.cache/gems_data/training_features.tif`. The tracked `data/restore_receipt.json` is not proof that those raw/cache files are present now.
- Restored the two pinned rejection witnesses with `scripts/restore_registry.py`; their SHA-256 values matched `registry/witness_sources.json`. Ran the unchanged `scripts/preflight_registry.py`. It exited with the documented stop code `2` and wrote `evidence/registry_preflight.json`.
- **Preflight decision: `STOP_UNIVERSAL_DUPLICATE`.** This is a candidate-independent registry rejection, not a geological score or a full-registry clearance.
- **REGISTRY-MEASUREMENT (not HOLDOUT-DTI):** the finite allowed domain contains `5,106,385` cells; the pinned witness's three-pixel dilation covers all `5,106,385`. For any nonempty candidate subset `C` of that allowed domain `A`, the mandated forward overlap is therefore `|C ∩ dilation(W)| / |C| = 1.0`, strictly above the stop threshold `0.70`. This one-witness proof rejects a candidate before placement and would also reject any final dot set. It is a sufficient rejection witness, **not a full-registry uniqueness pass**.
- The shared-template source hashes for `evaluate_holdout.py`, `submission_writer.py`, and `uniqueness.py` match `registry/template_pin.json`. No shared template, evaluator, writer, or threshold was changed.

## Pass 2 — investigate the H33 claim and screen candidates

### What the H33 bytes and source actually establish

I re-fetched the pinned owner-repository page for [GEMSDOE32 at commit `f62038b`](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/f62038b54dcb7767d4d997f80b033cdc3d42bef9/docs/index.html), and restored the H33 TIFF from its pinned GitHub source. That page explicitly says **“NO ORGANISER SCORE EXISTS for this or any artifact in this repository.”** It labels its own `0.2747` figure as a model projection, not a score. The user-supplied `0.2778` for H33 must therefore remain **OWNER-REPORT** unless an organizer receipt ties that exact score to that exact file. This is the principal irregularity: the premise that H33 received an organizer-confirmed `0.2778` is not established by the linked GEMSDOE32 source.

I independently fetched the D2.8 owner-mirror raster pinned in `registry/data_manifest.json` and compared binary supports on the matching grid:

- **RASTER-MEASUREMENT:** H33 has `37,654` positive cells; D2.8 has `44,090`.
- **RASTER-MEASUREMENT:** H33 is an exact subset of D2.8: `6,436` positives were removed and none were added. Every removed positive was within `2` grid pixels of the mapped catalogue; the closest retained H33 positive is `√5` pixels from it.
- H33 SHA-256 is `c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9`; the fetched D2.8 mirror SHA-256 is `91eae1ca42ec845eaa8c2ba32da49806e24751743459b8a10017c479bbe639b8`. These are owner-mirror byte measurements, not score receipts. The calculation is reproducible from the pinned sources and the bridge catalogue.

**Scientific interpretation, explicitly conditional:** if the owner-reported association is correct, the observed change is consistent with sparse-emission/pruning improving a distance-weighted overlap objective: deleting near-catalogue predictions can reduce redundant/false-positive mass while leaving most candidate support intact. It does **not** prove that the deleted pixels earned zero credit, that pruning caused any particular score, or that more aggressive pruning would help. A hidden fault may lie near a mapped fault; thinning can also remove useful coverage. The linked H33 page's live-model figure is not an organizer score. The prompt's `0.2778`, `0.3195`, and `0.4413` references are not used here as current confirmed leaderboard results.

The local pinned metric and the user-provided protocol describe an asymmetric distance-weighted Tversky objective. Under that setup, sparse placement can improve the ratio only when removed mass contributes little unique truth coverage relative to the false-positive cost; it is not a universal consequence of dot thinning. This is a plausible mechanism for the *reported* result, not a verified explanation of an organizer score. Catalogue hide-and-recover remains a proxy for fault recovery, not direct validation of geothermal vents, reservoir productivity, or economic potential.

### Candidate hypotheses

Five untested computational hypotheses, their target signatures, contrasts with repository implementations, named mimics, and source/availability caveats are recorded in [`research/hypotheses-arena-e24c1d71.md`](research/hypotheses-arena-e24c1d71.md). They are proposals only; all expected HOLDOUT-DTI changes are unknown, no numerical score projection is made, and some ideas appeared in earlier written research but have not been run as the exact test described.

The conditional rank-1 proposal is scale-dependent cross-wavelet phase coherence between magnetic RTP and isostatic gravity, with basement-depth-step corroboration. The exact cross-wavelet operator was not found in executable code, but related cross-sensor methods have been tested and this idea is not new to the repository's written backlog. It is the least expensive candidate in principle because it would use the competition feature stack rather than an external dataset. It is **not currently viable**: the raw/cached feature stack is absent, its existing mirrors are not organizer-authenticated, the current method paragraph is unassigned, and the universal registry witness stops all nonempty surfaces.

The external-data candidates name official manual-review sources (USGS ComCat, USGS Landsat Collection 2, NASA LP DAAC ASTER, Copernicus Data Space / NASA ASF). Their source pages and Great Basin scene/product coverage were not fetched in this session because their hosts are outside the sandbox's outbound allowlist. They are explicitly **blocked, not proposed as obtainable**. No data were fabricated or represented as downloaded.

### Required validation was not reached

The user requested a leakage canary for every feature and a spatially blocked holdout for the top candidate. Neither can be reported: the feature raster/cache is absent, and the protocol's universal duplicate preflight fires before feature generation. Therefore there is no new feature-alone AUC, no new HOLDOUT-DTI value or confidence interval, no candidate-specific rank correlation, no candidate surface, no final dots, and no new TIFF hash. No holdout number is projected or substituted.

## Pass 3 — independent release, format, and software checks

- `scripts/audit_downloads.py` independently opened every primary TIFF listed in `docs/data/lanes.json`: **FORMAT-INTEGRITY-MEASUREMENT:** all `44` listed primary files passed the local single-band/range/grid readback against the tracked bridge reference. This does not establish complete-registry uniqueness, official input provenance, portal acceptance, or upload clearance.
- `scripts/release_gate.py --verify-fail-closed` returned **NOT OK TO SUBMIT** and zero authorized files. A format PASS does not override the duplicate stop or separate selector approval.
- The full software suite completed: **SOFTWARE-TEST MEASUREMENT (not a score):** `307` tests passed. The immutable scientific inputs/evaluator were not modified.
- Only the candidate-independent registry preflight was executed; it is not a geological experiment. The experiment count for this review is zero and no weekly competition slot was spent.
- Existing historical TIFs remain individually identified for inspection. No file is newly designated “OK TO SUBMIT,” and the failed user upload's original root cause remains unknown because the rejected bytes and organizer diagnostics are not available here.

## Verified facts, official links, and access limitations

| Topic | Source | Verification boundary in this review |
|---|---|---|
| H33 page / attribution | [Pinned GEMSDOE32 source page](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/f62038b54dcb7767d4d997f80b033cdc3d42bef9/docs/index.html) | Freshly retrieved from GitHub. It says the artifacts have no organizer score; this is an owner repository page, not an organizer receipt. |
| H33 and D2.8 bytes | [H33 pinned source](https://github.com/buffedlizard55-lab/60GEMSDOE/blob/b99591eede63c6d45c8a896255b8cd34824fa00b/registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif) · [D2.8 pinned mirror directory](https://github.com/buffedlizard55-lab/GEMSDOE24/tree/07345ea0604953d7efb858d9cfbc21e20c7aca0b/docs/downloads) | Retrieved file `gems24-h25-1-dotted-h19-5-d2-8-20261002-e56ea318af89-nan.tif` from the pinned owner mirror; SHA-256 checked. Not an official source. |
| Problem/metric/official data | [DrivenData problem page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) · [data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) · [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) | Manual-review links supplied in the prompt. Not freshly fetched; no current leaderboard claim or organizer score is asserted. |
| Public USGS geophysics | [USGS GeoDAWN release](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and) | Official manual-review link; not freshly fetched in this session. |
| External hypothesis sources | [USGS ComCat FDSN](https://earthquake.usgs.gov/fdsnws/event/1/query) · [USGS Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2) · [NASA LP DAAC ASTER](https://lpdaac.usgs.gov/products/ast_l1tv003/) · [Copernicus Data Space](https://dataspace.copernicus.eu/) · [NASA ASF Search](https://search.asf.alaska.edu/) | Official links for manual review, not live availability confirmations. Their hosts are outside the sandbox allowlist; the AOI coverage, licensing/access, and acquisition were not verified here. |
| Shared method and artifact readiness | `registry/template_pin.json`, `vendor/gems57/`, `evidence/registry_preflight.json`, `docs/data/submission-authorization.json` | Locally verified hashes and receipts. These establish template identity, a rejection witness, and release-gate state—not organizer acceptance. |

## Limitations and next session's critical path

1. Do not generate or upload a candidate while the literal one-witness overlap rule remains binding. Have the protocol owner resolve whether dense continuous probability rasters are valid “dot” witnesses. Do not silently filter them, modify the shared gate locally, or reinterpret the threshold.
2. Assign one new method paragraph. Do not restart the already-run selective-U lane or another negative lane.
3. Restore the full feature stack only from an authorized source and record whether it is an organizer download or an owner mirror. Confirm the input SHA-256 and exact sample grid before feature work.
4. After the first two blockers are resolved, preregister one candidate. Run each feature alone as a leakage canary; then use the pinned buffered whole-segment holdout and report evaluator version, withheld-positive count, and 95% CI. Compare with a fold-rebuilt matched incumbent. Do not spend a weekly slot before that result beats the incumbent and passes both surface and final-dot registry tests.
5. Keep the submission writer unchanged. Require a byte-bound format receipt, complete uniqueness evidence, separate selector approval, and a submission-page receipt tied to the exact SHA before ever calling a file OK TO SUBMIT or reporting ORGANIZER-CONFIRMED.
6. The official-source web pages and current live leaderboard were not accessible to this sandbox, and there is no DrivenData login/portal receipt here. These are access limitations, not reasons to infer a score.

## Review run card

See [`evidence/arena_e24c1d71_run_card.json`](evidence/arena_e24c1d71_run_card.json) for the single machine-readable run card. It records a negative result, no new raster, no holdout value, the measured universal overlap blocker, local format-audit scope, and the absence of selector/portal approval.

## PR and mainline reconciliation

This review and its preflight were run against the session checkout based on `40075f6f73db9f2844c478da5670a193713e6e23`. After committing and pushing the review, `origin/main` advanced to `b0b26754d354dae6e797ea7d57f3b7c4a40fcdcb` and now includes PR #100. That mainline adds H85 and an amended shared duplicate-policy helper (`representation_class_v2` / `binding_duplicate_decision`); it also retains a candidate-independent `scripts/preflight_registry.py` and committed receipt with `STOP_UNIVERSAL_DUPLICATE`.

The upstream status needs explicit reconciliation before this PR is merged: main's README labels H85 “OK TO SUBMIT,” while the checked-in release receipt still reports zero repository-authorized files, H85's run card records `approved_for_weekly_slot: false`, and the receipt's operator-decision scope says the complete 836-entry registry was not certified. Those are distinct scopes/decisions, not a result this review can reconcile by inference. This E24 evidence does not authorize H85 or alter the upstream policy.

PR #106 is open, but GitHub reports it non-mergeable. A local merge of `origin/main` produced conflicts in the README, shared page builder, tests, and generated landing pages; it was aborted without resolving by overwriting either side. `gh pr merge 106 --merge` was attempted and GitHub refused because a clean merge commit cannot be created. The next step is a deliberate owner review of the mainline policy/status, then regenerate the entry pages and resolve the conflicts; do not force-merge or silently revert the upstream work.
