# GEMSDOE77 session review — 2026-10-21 · `arena/eacf8dd2-gemsdoe77`

## Decision at a glance

**No new TIFF was generated, no geological experiment was run, and no competition slot was used. Every repository download remains inspection-only; the current authorization receipt authorizes 0 files.** This is an intentional fail-closed result, not a successful new-submission build. There is no newly generated file to upload or to call unique.

The current prompt did not assign one identifiable method paragraph. More importantly, the mandatory registry preflight rejects every nonempty candidate under the currently pinned literal overlap rule, before feature generation. The requested top-candidate holdout was therefore **not run**. Running it or creating a raster anyway would evade the explicit stop rule. A file name, byte hash, format PASS, or a zero-valued placeholder cannot repair that contradiction.

The landing page and this report distinguish **OK TO DOWNLOAD FOR INSPECTION** from **OK TO SUBMIT**. Do not upload any of the historical TIFFs based on their being downloadable or locally format-valid.

## What was checked

1. Re-read the standing project brief, `AGENTS.md`, this branch's inherited instructions, repository reviews, the corrected H33 analysis, frozen run cards, release authorization, registry pins, shared-template pins, source feed, site generator, and current files.
2. Checked the active branch and worktree: `arena/eacf8dd2-gemsdoe77`. The pre-existing worktree change on continuation was the prior local `evidence/listed_download_audit.json` format-audit receipt; no candidate code or raster change was present. No experiment branch or alternative branch was created.
3. Restored the two SHA-pinned registry witnesses with `scripts/restore_registry.py` and ran `scripts/preflight_registry.py` using the unchanged `vendor/gems57` implementation. The required expected stop is recorded in `evidence/registry_preflight.json`.
4. Ran `scripts/audit_downloads.py` on the current manifest. This is a local format/integrity check only, not a submission or uniqueness test.
5. Refreshed the scheduled GitHub inventory with `scripts/refresh_sources.py`. This feed is repository/page freshness, **not** score verification.
6. Directly checked the pinned GEMSDOE32 source page via GitHub API. Its content says “NO ORGANISER SCORE EXISTS for this or any artifact in this repository.” The page also preserves a dated leaderboard snapshot; it does not provide a submission receipt tying a score to the H33 TIFF.

No private copy of the evaluator, writer, or uniqueness gate was used. No full feature stack was rebuilt. `vendor/gems57` hashes match `registry/template_pin.json`.

## H33: what the bytes support, and what they do not

### Score attribution

The user's **0.2778** association with `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` is **OWNER-REPORT / user-supplied**, not ORGANIZER-CONFIRMED. The pinned GEMSDOE32 page explicitly says no organizer score exists for any artifact in that repository. No receipt linking those TIFF bytes or SHA-256 to a DrivenData submission ID/score was found. Therefore the literal question “why did this TIFF get that organizer score?” cannot be answered as an established fact.

The user also supplies **0.3195** and **0.4413** as different leaderboard reference values. Neither was re-read from the official leaderboard in this session. The pinned GEMSDOE32 page contains a **dated 2026-10-04 owner-page snapshot** that lists 0.3195 in its leaderboard table, while the same page disclaims organizer scores for its own artifacts. That is historical owner-page content, not current leaderboard verification or a TIFF-to-score receipt. The current source feed records the exact page commit and content hash; see `docs/source-health.json`.

### Reproducible raster forensics

The pinned owner mirror `GEMSDOE32/registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif` has SHA-256 `c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9`. The retained forensics receipt `evidence/prior_forensics.json` records 37,654 positive pixels in H33, versus 44,090 in its dotted D2.8 predecessor: 6,436 pixels were removed, none added. The removed points were at catalogue distances of 1, √2, or 2 pixels. These are **RASTER-MEASUREMENTS**, not performance scores or proof of why an organizer awarded any value.

For the pinned local implementation, let `T` be total maximum truth-cell kernel credit, `S` predicted mass, `M` predicted mass weighted by proximity to truth, and `K` the truth-cell count. With `α=0.2`, `β=0.8`, `D = α(T + S − M) + βK` and `score = T/D`. Adding a binary dot at an empty pixel with nearest-truth kernel weight `k` and total truth-credit change `ΔT` changes the denominator by `ΔD = α(ΔT + 1 − k)`; the score improves iff `ΔT > α·score·(ΔT + 1 − k)`. The shortcut `k > α·score` is valid only if `ΔT=k`; a dot may cover several truth cells or add no new coverage. Therefore there is no universal “each dot costs α” explanation. These are local metric equations, not an H33 score receipt.

A plausible mechanism is that sparse, spaced points retain some 300 m-kernel coverage while reducing redundant prediction mass; deleting pixels near the already-known catalogue could reduce false-positive mass if those locations are excluded from scoring. The opposing risk is equally real: a genuinely withheld fault may lie close to a mapped fault, and deleting nearby predictions can remove useful coverage. See `research/why-h33-scored-02778.md` and its tests for the derivation. This is a plausible explanation of raster construction, **not** causal attribution of the OWNER-REPORT value.

**Can we beat the reported value? Unknown.** A different geological ranking might improve a properly matched hide-and-recover test, but neither the owner's reported score nor a historical holdout number predicts a future leaderboard result. Catalogue holdout measures recovery of withheld mapped faults; it is not direct validation of geothermal vents or truly unmapped faults.

### Comparable local reference (historical, not rerun here)

The repository's H77-Step reference is **HOLDOUT-DTI 0.152718** (`gems57-pooled-hide-v2`; 11,321 withheld positives; 95% CI `[0.131548, 0.175185]`). This is from the frozen receipt `evidence/h77/official_hide_v2_summary.json`, not a result from this review and not an ORGANIZER-CONFIRMED leaderboard score. Its folds/support differ from other lane results, so values from other run cards must not be compared as though they were a matched trial.

## Current hard stop and input status

### Registry preflight

`evidence/registry_preflight.json` is a **REGISTRY-MEASUREMENT**, not a score. **Decision: `STOP_UNIVERSAL_DUPLICATE`.** The run verified the shared-template pins and grid alignment, then measured an allowed domain of 5,106,385 pixels. The SHA-pinned continuous witness is positive over all 5,106,385 allowed pixels; its 3-pixel dilation also covers every allowed pixel. Under the current literal forward-overlap rule, every nonempty candidate subset therefore has a lower-bound forward overlap of 1.0, above the strict 0.70 stop threshold. This mathematical rejection applies before placement and to any final nonempty dots. No candidate-specific rank correlation exists because no candidate was built; the preflight correctly records it as null.

This proves a rejection from one pinned witness. It is **not** a full registry scan, a uniqueness clearance, or proof that geological methods are alike. The policy owner must resolve the continuous-surface-versus-dot-support semantics in the shared protocol. Do not privately exclude dense rasters or change the threshold.

### Inputs and local format check

The current checkout contains only the bridge `existing_faults.tif`, `labels.tif`, and `sample_submission.tif` under `data/`. `data/raw/training_features.tif`, `data/processed/stack.npy`, and `.cache/gems_data/training_features.tif` are absent. The tracked `data/restore_receipt.json` describes a prior restore and is not proof that the large feature stack is present now. Consequently there was no active cached feature stack to reuse for canaries or holdout. Rebuilding or downloading hundreds of megabytes after a candidate-independent stop would not change the release blocker and was not done.

The fresh `evidence/listed_download_audit.json` read back all 29 primary TIFFs listed in `docs/data/lanes.json`; all 29 pass the local single-band/grid/hash/finiteness/range checks. This is **FORMAT-INTEGRITY-MEASUREMENT** only. It does not check the full archive, uniqueness, scientific performance, portal acceptance, or selector approval. `docs/data/submission-authorization.json` still authorizes 0 files.

### Historical authorization conflict

The frozen H77-Step `evidence/h77/run_card.json` retains its original `verdict: promote` / `safe_to_download_and_submit: true` fields. The current repository release receipt reconciles that as a historical promotion conflict and says **NOT OK TO SUBMIT**. Use the current fail-closed authorization receipt, not that historical field, as the release decision. The run card is retained for audit and was not edited.

## Ranked untested hypotheses — conditional backlog, not authorization

The ranking is a qualitative prior on expected fault-recovery benefit versus implementation cost. **The numeric HOLDOUT-DTI effect is unknown for every idea; no score projection is supplied.** The relative expected-gain labels below are uncalibrated expert judgments, not measured outcomes. “Untested” means no executable implementation/test was found in this repository review; it is not a claim of global novelty. Some ideas appear in older hypothesis notes. Each proposal is blocked from implementation in this session by the universal duplicate stop, missing active stack and lack of a fresh lane assignment.

| Rank | Inputs/layers and physical signature | Why it could find a fault absent from the surface-fault catalogue | Difference from this repository; named non-fault mimic | Qualitative expected HOLDOUT-DTI gain / implementation cost; source availability |
|---:|---|---|---|---|
| 1 | **Event-level earthquake geometry:** USGS ComCat event locations, depth and location uncertainty, magnitudes and focal mechanisms where supplied; test uncertainty-weighted 3-D plane/lineament coherence and project a defensible up-dip trace. | A blind or recently active fault can generate a coherent event plane without a mapped surface scarp; event-level geometry is materially more local than the shipped 100 km-smoothed seismicity bands. | No event-level ComCat plane-fitting implementation was found. The broad seismicity channels and earlier written proposals are related but not this test. **Mimics:** volcanic/geothermal swarms, quarry blasts, aftershock sequences, and uneven station detection. | **Unquantified; relative expected DTI gain: medium; cost: high; unvalidated.** Official free source: [USGS ComCat FDSN event service](https://earthquake.usgs.gov/fdsnws/event/1/query). Host is outside this sandbox's outbound allowlist; no AOI events were fetched. **Not obtainable/viable here until access and completeness are checked.** |
| 2 | **Multi-temporal InSAR deformation:** Sentinel-1 ascending/descending line-of-sight velocity, coherence and uncertainty from repeated interferograms; test persistent cross-track deformation gradients with seasonal controls. | A creeping, pressurized or otherwise active blind fault may deform under cover without a preserved scarp. | No Sentinel-1 time-series displacement/coherence processing was found; static geodetic summary bands are not a radar time series. **Mimics:** groundwater pumping/aquifer compaction, volcanic deformation, mining, and atmospheric phase delay. | **Unquantified; relative expected DTI gain: medium; cost: very high; unvalidated.** Official route: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/). Host is outside the allowlist; Great Basin product availability was not checked. **Not viable here.** |
| 3 | **Hydrologic channel-network offsets:** official 3DEP elevation, conditioned flow paths, channel azimuths, and channel-profile residuals; test repeated, same-sense deflection/knickzone pairs across neighboring catchments. | Distributed deformation can bend or offset several channels where individual scarps have been eroded, buried, or masked by alluvium. | Existing code already contains extensive scarp, curvature, knickpoint and channel-derived features; a cross-catchment topology test is distinct in scale, but has substantial conceptual overlap and needs a full novelty review. **Mimics:** lithologic controls, stream capture, fan avulsion, shorelines, and engineered drainage. | **Unquantified; relative expected DTI gain: low–medium; cost: high; unvalidated.** Official source: [USGS 3DEP](https://www.usgs.gov/3d-elevation-program). The raw AOI DEM is absent in this checkout; USGS is outside the allowlist. **Not viable here.** |
| 4 | **Persistent satellite thermal anomalies:** Landsat Collection 2 Level-2 surface temperature, QA/cloud flags, emissivity/land-cover controls and acquisition dates; test recurrent thermal residuals/edges rather than one-date hot pixels. | Recurrent, spatially coherent heat may indicate an unmapped hydrothermal discharge zone whose conduit follows a fault; this is more directly relevant to vents than to every catalogue fault. | No Landsat thermal time-series implementation was found. Existing H80/H82 lanes use point temperatures/geothermometers, not repeated satellite surface-temperature observations. **Mimics:** irrigation, wildfire, emissivity/land-cover differences, urban heat and residual cloud contamination. | **Unquantified; relative expected DTI gain: low for mapped-fault recovery; cost: high; unvalidated.** Official source: [USGS Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2). The source host is outside the allowlist; AOI scenes and temporal depth were not checked. **Not viable here.** |

No item above is selected for a run. In particular, the top-ranked event-level candidate cannot be validated without the USGS event feed, and the external source cannot be reached from this sandbox. The universal preflight would still require policy resolution before any new field or dot generation.

## Source and claim ledger

| Claim/source | Evidence and scope |
|---|---|
| GEMSDOE32 artifact's claimed score | [Pinned GEMSDOE32 page at commit `f62038b54dcb7767d4d997f80b033cdc3d42bef9`](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/f62038b54dcb7767d4d997f80b033cdc3d42bef9/docs/index.html), freshly retrieved by GitHub API. Owner-repository page, not organizer receipt; it explicitly states no organizer score exists for its artifacts. |
| H33 raster contents and lineage | `registry/witness_sources.json`, `evidence/prior_forensics.json`, and the pinned owner TIFF hash above. Reproducible mirror bytes, not organizer authentication. |
| Metric implementation | `vendor/gems57/metric.py`, pinned by `registry/template_pin.json`; this session's preflight verified the template hashes. Official public page [link](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) is supplied for manual review; it was not fetched in this sandbox session. |
| Current family-site feed | [`docs/source-health.json`](source-health.json): checked `2026-10-21T17:55:10Z`; 82/82 configured owner-repository inventories were retrieved, with no retrieval errors. This is not an authenticated score feed, a complete raster-registry audit, or a portal submission receipt. |
| Competition rules/data/leaderboard | [Problem](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/), [data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/), [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/), and [rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf) are official manual-review links. The non-GitHub hosts are outside the sandbox allowlist; no fresh portal/leaderboard/rules verification or authentication occurred. |
| External data proposals | Official links are given in the hypotheses table for manual review. USGS, Copernicus and DOE GDR pages were not fetched here; no AOI availability, license, or completeness is claimed. |

## Pass 1 — implement and verify

- Refreshed the source-inventory feed and current listed-download format audit.
- Added this branch-specific review/run card and wired the current review into the generated site.
- Added an above-the-fold status notice: **no new TIFF; no file authorized to submit**.
- Marked the older `reports/why_02778.md` analysis as superseded so its live-score inversion and earlier score attribution are not presented as current verified facts.
- Did not create a TIFF, model, feature surface, holdout prediction, or slot selection.

## Pass 2 — bugs, edge cases and irregularities

- The current review must not describe a 29-file format pass as upload acceptance or uniqueness clearance; the audit schema and page say format/integrity only.
- The source feed's 82/82 retrieval count covers configured GitHub repositories, not scores. The official leaderboard/problem/data pages remain explicitly unfetched.
- A stale H77-Step `promote` field conflicts with the operative authorization. The release gate remains authoritative and authorizes zero files.
- The pinned universal witness means a new nonempty artifact cannot satisfy the literal overlap gate. The only safe action is to stop and report; no private dense-raster exemption or all-zero pseudo-submission was made.
- The generated homepage previously repeated the listed-file count instead of deriving the format-pass count from each audit row. It now renders the actual per-file pass count and links to the audit/authorization receipts.
- `build_h82rank_site.py` previously moved H82R to lane index 1 and rewrote `lanes.json` with a different indent on each rebuild. It now preserves stable placement immediately after H77V and the committed JSON formatting; a regression check covers the order.
- The older H33 report/script and generated LIA pages associated 0.4413 with an organizer-confirmed leaderboard row and presented live-score inversion. The report/script and all LIA pages now carry a superseded-status warning; score rows are labeled owner-reported, target values are sensitivity-only, and no current leaderboard claim remains. Calculations were not rerun or treated as evidence.
- The 904a24d3 review page called itself current. Its source and generated title now identify it as historical/superseded; the current branch review is the landing-page target.
- Data restore receipts are historical. Current filesystem checks, not old receipts, determine whether the cache exists.

## Pass 3 — original-request recheck

The request asks for a unique TIFF, a validated top-candidate holdout, official source checking, simple submission download/status, a current feed, and a PR merge. This review completes the safe repository/site/source-inventory work, but **cannot honestly claim the new TIFF or holdout requirements are completed**: the assigned lane is absent, the full active cache is missing, external official sources are outside the network allowlist, and the shared literal registry gate proves a universal duplicate. No candidate was promoted or uploaded. The follow-up section below is the required next-session handoff, not a hidden authorization.

## Next work, in order

1. Obtain an explicit single-method lane assignment and have the shared protocol/registry owner resolve continuous-surface versus dot-support semantics uniformly upstream; do not change the local gate.
2. Obtain organizer-authenticated competition inputs through a permitted account workflow, and verify the feature stack is actually present before declaring it cached. Never put credentials in chat or in Git.
3. Check official source availability and AOI/metadata coverage for one candidate before preregistering it. The event-level USGS candidate is blocked in this sandbox; do not call it obtainable based only on its URL.
4. Only after the universal preflight is resolved: use the pinned `vendor/gems57` feature/evaluator/writer path, run every feature-alone leakage canary, hide buffered whole fault segments, derive catalogue-based features from visible segments only, mask visible faults pixel-exactly, and compare pooled DTI with paired block-bootstrap CI against the matched incumbent.
5. Require surface and final-dot uniqueness checks, exact-grid/range validation, a positive evidence-backed run card, and separate selector approval before making an artifact **OK TO SUBMIT**. Then record any portal receipt tied to the exact TIFF hash. Do not use a competition slot based on the owner's 0.2778 claim or a projection.

## Review run card (this review; no experiment)

See `evidence/review_eacf8dd2_run_card.json`. Its new HOLDOUT-DTI fields are explicitly **NOT RUN**, its TIFF SHA-256 is null, and the verdict is negative.
