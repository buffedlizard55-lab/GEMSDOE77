# Three-pass repository review and H82-TEXTURE result — 2026-10-21

## Download first; submission decision first

**[Download newly generated H82 research TIFF](downloads/gems77-h82-texture-kt-bf0a3028b4fa-RESEARCH-ONLY.tif)** · [single-TIFF ZIP](downloads/gems77-h82-texture-kt-bf0a3028b4fa-RESEARCH-ONLY.zip) · [executive summary](executive-summary.html) · [technical report](h82-texture.html).

**OK TO DOWNLOAD: YES, for inspection. NOT OK TO SUBMIT.** The new file is not a copied or renamed old prediction. It is pixel-distinct from all 36 prior local TIFF downloads checked, but it **fails the binding uniqueness rule**. No claim of full-registry uniqueness or leaderboard improvement is made. No competition slot was used.

## Pass 1 — repository inspection and frozen feasibility experiment

Read the complete README (including preserved standing request), PROMPT, AGENTS, irregularities, base preregistration, shared-template pins, prior release reviews, and lane evidence. Baseline: 124 software tests passed. Restored the existing cached radiometric feature stack with `scripts/fetch_data.py`; verified SHA-256 and metadata with `scripts/prepare_data.py`. These are public owner-mirror inputs, not newly organizer-authenticated data. The assertion that the only remaining task is downloading full training data is not accurate for this checkout: the usable cache can be restored, but official-origin authentication, matched incumbent validation, uniqueness semantics and slot selection remain separate blockers. No GPU is needed for this small CPU feasibility test. The official reference README also explicitly supports CPU environments.

The request contains no identifiable single-method assignment. Declared H82-TEXTURE explicitly rather than claiming an assignment. Three ranked geological hypotheses were written before implementation in [the preregistered research note](research/h82_texture_hypotheses.md) (the published copy is linked from the technical report). The new narrow hypothesis is local K/Th second-moment boundary concurrence, rather than mean/rank residual curvature, lagged halos, LiDAR scarps or mapped-tip geometry used by older lanes. Novelty is checkout-local and qualified, not an exhaustive family-wide claim.

One experiment only: use a 9×9 local standard deviation and its central-difference gradient per channel; geometric-mean concurrence; label-free percentile normalization; frozen continuous amplitude 0.02. This is a bounded decision surface, not a calibrated posterior. No trained model, previous submission, fault-distance feature, external-label surrogate or unknown hidden truth is used. Every input and derived feature is tested separately. Reused unchanged pinned `gems57.spatial.folds`, `evaluate_holdout.py`, `pooled_summary`, `uniqueness.py`, and `submission_writer.py`. All template-file pins match. No private evaluator or gate exemption.

**Result: 0.005969454 HOLDOUT-DTI**, evaluator `gems57-pooled-hide-v2`, 59,484 withheld positive pixels, 95% CI `[0.005734959, 0.006191659]`. Matched constant-output control: **0.017675473 HOLDOUT-DTI**, same evaluator/population, 95% CI `[0.017091391, 0.018186017]`. Paired difference: **−0.011706019 HOLDOUT-DTI**, same evaluator/population, 95% CI `[−0.012236802, −0.011116822]`. Conditional physical 20 km-block percentile bootstrap, 1000 draws, seed 82077. Not directly comparable to historic sparse-dot incumbents on different support. The fixed low amplitude can suppress DTI; this does not disprove every texture-based method or justify a post-hoc amplitude sweep after the stop.

Leakage diagnostics, **not scores**: 32 feature-fold canaries; maximum symmetric AUC 0.618363; no test exceeds 0.90. Whole intersecting connected fault components withheld; fixed spatial quadrants; 80 px training exclusion; no training occurs. Visible catalogue masking is pixel-exact per fold. Full catalogue masking occurs only on research export. The scientific target is faults, not discovered geothermal vents or economic resources.

Surface test against two SHA-pinned witnesses: maximum measured Spearman rho 0.010780345, literal forward 3-pixel support overlap **1.0 > 0.70**. One dense witness covers every allowed cell; full scan unnecessary for rejection but required for any pass. **Duplicate logged; stopped before placement. No final dots were generated.** Exactly the checked continuous surface was retained with RESEARCH-ONLY filename through the shared writer and rechecked after readback. The instruction to produce a protocol-cleared unique submission cannot be satisfied together with the currently binding universal-witness overlap gate; renaming, feature changes and different packing do not solve that contradiction.

## Pass 2 — independent verification and concrete fixes

- Six new synthetic tests check local-SD arithmetic against direct windows, constant fields, texture steps, invalid inputs, normalization degeneracy and full **square** missing-data stencil support. The 11×11 support covers both the 9×9 moment window and its gradient; avoids the taxicab/diagonal support problem found in an older lane.
- Independent `scripts/verify_h82_texture.py` reopens the TIFF/ZIP with plain rasterio/zipfile, without importing the writer: 15 checks pass, exact sample grid, all finite, [0,1], catalogue/outside zero, SHA match, exact one-TIFF ZIP, frozen preregistration hash, negative/no-slot status.
- Readback: single float32 band, EPSG:32611, 3730×3292, `(100,0,243350,0,-100,4508550)`; min 0, max 0.01999999955; zero NaN/Inf; nodata absent. Shared validator: 15/15 pass. Local format validation cannot certify portal acceptance or diagnose the historical range error.
- All 36 pre-existing local TIFF downloads differ in decoded footprint pixels. This distinction-only audit is separate from the literal duplicate rejection. The same two-witness surface test is repeated on the unchanged export, **not misreported as a final-dot audit**.
- Fixed misleading `beats_holdout_best=true` release inputs on H77-Step, H77D and the constrained H77E-Knick file. Those were historical comparisons against older baselines (or the wrong twin), not a newly matched current-incumbent win for the exact file. Original flags are retained under `historical_beats_baseline_flag`; frozen run cards and old scientific results are unchanged. Authorization now fails that gate too. Existing files were never approved by this change.
- Added direct research-only download/status to the first screen and executive summary. The build generates H82's evidence page without running experiments. A missing review link was found by the first expanded test run and fixed before release.
- Added explicit ignores for restored cache TIFFs so a data restore cannot accidentally stage large input files.

## Pass 3 — end-to-end checks and remaining limitations

The full CI build sequence, local verification scripts, expanded software suite, generated-link checks, exact artifact/hash/ZIP checks and fail-closed authorization are re-run before PR merge. Test and GitHub check outcomes are reported in the PR/final response, not pre-claimed here. The entry and executive summary pages state the file is inspection-only; name and <=140-character note are supplied for traceability, **not permission to upload**. No submission endpoint is called.

A new format-valid research TIFF and negative evidence have been delivered. A protocol-cleared, validated high-scoring competition entry has **not** been delivered; it would be dishonest to claim otherwise. Downloading this TIFF is safe for inspection. Uploading it is prohibited by the current project protocol.

## H33: what can and cannot be inferred

[Retained byte forensics](evidence/prior_forensics.json) show H33 retains 37,654 dots from a 44,090-dot precursor by deleting 6,436 points at catalogue distance <=2 px. No new fault locations were added. Sparse emission can reduce redundant prediction mass while retaining partial triangular-kernel coverage; pruning can improve DTI if lost truth coverage is small relative to removed false-positive mass. This is an explanatory hypothesis supported by construction and metric algebra, **not a controlled organizer causal experiment**. It is not proof that the 200 m exclusion is optimal, that hidden faults cannot occur there, or that all SGMC contacts are irrelevant.

The named-file **0.2778 is OWNER-REPORT**, not a freshly verified exact-file organizer receipt. The request's **0.3195 and 0.4413 targets conflict**; neither is freshly confirmed here. Historical repository pages asserting ORGANIZER-CONFIRMED retain their past context; this review does not recertify them. A public leaderboard row without a file identity cannot authenticate a particular TIFF. Nothing here predicts a better public result.

## Verified sources and access limits

1. [Official reference README at aebe92f7](https://github.com/drivendataorg/gems-prize-reference-solution/blob/aebe92f7c8a990f0e3443451b7a825d9afd6336b/README.md) — freshly retrieved via GitHub API; explicitly has CPU-only/GPU installation routes. It does not validate H82 or authenticate the cache.
2. [Pinned template/cache source](https://github.com/buffedlizard55-lab/60GEMSDOE/tree/b99591eede63c6d45c8a896255b8cd34824fa00b) — fetched from owner GitHub mirror, verified by hash. Mirror is not an official source of geological ground truth.
3. [USGS GeoDAWN](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and), [DOE GDR 1391](https://gdr.openei.org/submissions/1391), [competition problem](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/), [data](https://www.drivendata.org/competitions/306/competition-doe-gems/data/), [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/), [rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf) — official manual-review anchors, not freshly accessed sources. This sandbox allows outbound access only to GitHub and package registries; no DrivenData session/receipt was available. No full literature review or current weekly-cap verification is claimed.

## Next session, in order

1. Protocol owner must resolve dense-surface versus sparse-dot semantics **in the shared template/protocol**. Do not privately waive the literal gate or replace forward with mutual overlap.
2. Stop H82; do not retune its output after duplicate rejection. Preserve the negative result. Do not train further blind to the universal blocker.
3. Establish one canonical incumbent artifact, fold/support definition and matched re-runnable baseline. Keep constrained/pure twins distinct. Do not authorize based on a stale baseline flag.
4. Obtain authoritative data/receipt provenance through an approved authenticated workflow (no credentials in chat). Name an explicit new method lane, preregister canaries/ablation/control, and require matched improvement before selector consideration.
5. Only after all gates pass: generate with the shared writer, validate exact bytes, run complete surface/final-dot registry audits, seek separate selector approval, observe the portal's current weekly cap, and attach the organizer receipt to the exact artifact SHA. No slot is selected by this session.

## Parallel reconciliation and final local verification

Main advanced through PRs #48/#49 during this session. Merged `origin/main` into the same session branch, preserving the independently generated H28 artifact, corrected H33 marginal-DTI analysis, stricter release-card schema checks, and earlier H81 review. Relabeled the older “no new TIFF generated” banner as an earlier parallel review so it does not contradict H82. Frozen H82 evidence still describes the 36-file inventory at experiment time, not the later H28 addition; additional current-inventory readback comparisons are in the independent audit. No experiment was rerun during reconciliation.

Final merged local suite: **138 software tests passed**. Full CI build order, `scripts/verify_all.py`, independent H82 audit, local-link checks, ZIP/readback checks and release gate pass; authorization remains **0 files**. These are software/format checks, not organizer acceptance.
