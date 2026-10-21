# GEMSDOE77 current-session repository review — 2026-10-21

## Executive decision

**The repository contains a newly generated, locally format-valid H81 GeoTIFF, but it is not a competition-cleared submission.** It is safe to download for inspection and **not safe to upload** under the recorded protocol. The fail-closed authorization receipt authorizes **0 files**. I did not regenerate, rename, or re-place dots for another TIF: the assigned H81 lane has already used its three experiments, its HOLDOUT-DTI is below the retained best, and its final dots violate the explicit 3-pixel overlap stop against a registry raster. Generating another artifact in that lane would evade the stop/budget rules rather than improve the evidence.

This is a repository/code/artifact review, not an official competition submission or a newly run scientific experiment. The family score `0.2778` for the named GEMSDOE32 H33 file is **OWNER-REPORT**, not an organizer-confirmed score tied to the file by a receipt. The user-provided public leaderboard value `0.4413` is not independently checked in this session; no current leaderboard claim is made here.

## Immediate user-facing download status

- **OK TO DOWNLOAD FOR INSPECTION: YES.** [H81 TIFF](downloads/gems77-h81-slipcone-37654dots-20261021-0fbeaf41-zeros.tif) · [one-TIFF ZIP](downloads/gems77-h81-slipcone-37654dots-20261021-0fbeaf41-zeros.zip) · [method/evidence page](h81.html) · [how-to-submit guide](executive-summary.html).
- **OK TO SUBMIT: NO — DO NOT UPLOAD.** The file's SHA-256 is `f3eb9e5e3bf03da0285b48b5bb7d41f15b0e450d16bc6b8943aeaa63efadc1f7`. Byte distinction is not the same as protocol uniqueness or selector authorization.
- The updated README opens with this status. The generated site start page also lists H81 as inspection-only; the machine-readable source of repository-wide authorization remains [`data/submission-authorization.json`](data/submission-authorization.json).

## Three-pass review

### Pass 1 — requirements and evidence inventory

- Read the standing brief in `PROMPT.md`, `README.md`, `AGENTS.md`, `docs/irregularities.md`, the H81 preregistration and amendments, the H81 hypotheses, its frozen E1/E2/E3/run-card evidence, the submission authorization receipt, and the relevant site/build and shared-evaluator files.
- The current checkout already contains the prior H81 implementation, TIF, ZIP, and evidence. H81 was preregistered as slip-budget fault-tip continuation cones, using Qfault-derived trace segments and a catalogue mask, with one optional topography-gated sensitivity arm. Its assigned experiment budget is three; its run card reports three experiments used and zero weekly submission slots used.
- Reused shared repository components remain pinned in `registry/template_pin.json`; H81 imports `gems57.evaluate_holdout`, `gems57.submission_writer`, and the shared grid/holdout/uniqueness APIs. No private evaluator/writer fork was introduced in this review.
- The H81 research register already contains five ranked candidates. H81-1 (slip-weighted tip cones) was tested and is closed; H81-2 through H81-5 were listed as future/unrun candidates. Repeating H81 or promoting a different arm selected after looking at the holdout would violate the frozen protocol.

### Pass 2 — execution, integrity, and scientific-gate checks

**Software tests:** installed the pinned requirements into the ignored local `.venv` and ran `.venv/bin/python -m pytest -q`: **124 passed**. This verifies software tests only; it is not evidence of predictive performance.

**Independent H81 artifact readback:** reopened the linked H81 TIFF with Rasterio and compared it directly with the tracked bridge sample grid. Checks passed for one band, `float32`, `EPSG:32611`, shape `3730 × 3292`, exact affine transform match, all finite cells, values `[0,1]`, no NaN in the sample footprint, and 37,654 positive pixels. Its ZIP contains exactly one TIFF whose uncompressed bytes equal the linked TIFF. This confirms local file integrity/format relative to the tracked bridge sample—not organizer provenance, portal acceptance, or permission to upload.

**Release gate:** ran `scripts/release_gate.py` to an ignored `.cache` receipt with `--verify-fail-closed`. It returned `NOT OK TO SUBMIT`, `0` authorized files, across the 22 listed lanes. The tracked authorization receipt also records zero authorized files. The gate is fail-closed and distinguishes format validity from science, uniqueness, and selector approval.

**Frozen scientific evidence (not recomputed here):**

- **0.067071 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.055232, 0.077908]`) for the preregistered H81 slip-cone arm.
- The retained comparison is **0.152718 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.131548, 0.175185]`). H81's upper confidence limit is below this incumbent's point estimate; H81 does not satisfy the registered improvement criterion.
- H81's primary feature canary has symmetric AUC `0.5924`, below the `0.90` leakage alarm. The `0.8134` slip-proximity diagnostic is explicitly a canary-only positive control that uses hidden-catalogue proximity; it is not a model arm and was not used to make the prediction. Do not treat it as clean predictive evidence.
- **Registry uniqueness:** H81 pre-placement surface max absolute Spearman correlation is `0.5403` (below `0.90`). At final dots, max absolute Spearman is `0.1333`, but forward 3-px dot overlap is `1.0` against a dense witness, greater than the protocol threshold `0.70`. Thus the duplicate stop fires. Only 832 of 836 indexed registry rasters were readable; four were invalid/mismatched, so the registry scan is incomplete and cannot certify uniqueness. A single confirmed duplicate is sufficient to reject; completeness would be required to pass.
- The stored run card reports the H81 raster byte-unique among its checked inventory. That narrower property does not contradict the overlap failure: a raster can be different in bytes and pixels while placing its dots too close to another raster's dots.

**Important protocol/data irregularity:** the H81 amendment IR-81-04 transparently changed the tip unit from record ends to connected-component strand tips before any outcome was recorded, after discovering that record IDs can span large multi-trace zones and that the initial record-end design had insufficient support. This is auditable and disclosed; it is not a hidden outcome-driven change. The resulting strand-level design was tested and failed its promotion/uniqueness gates.

### Pass 3 — original-request and safety review

- The user asked for a unique downloadable TIF and an obvious yes/no about submission. The repository already has H81's newly generated TIFF. Its correct status is **download for inspection: yes; submit: no**. Calling it a cleared competition submission would be inaccurate. No new TIFF was created because that would violate the explicit stop/budget rules and would not remove the full-support overlap issue.
- The reported H33 value `0.2778` is explained in `research/why-h33-scored-02778.md` and related forensic receipts as a plausible consequence of emission-side thinning and pruning predictions close to the existing-fault catalogue: a sparse, less-redundant prediction field can waste less mass under the asymmetric distance-weighted Tversky metric. This is a mechanistic explanation, not causal proof of why that exact organizer score occurred. The TIFF-to-score link is not supported here by a submission-page receipt.
- The remaining H33 gains cannot be attributed to ever-larger pruning alone: pruning can remove redundant/low-value prediction mass but cannot relocate surviving predictions onto unseen-fault evidence. A genuinely better ranked field would be necessary to improve substantially; a holdout result must establish this before any selector considers a slot. H81 is an example of a different mechanism that did **not** pass that test.
- The old README top banner described H77E as the newest primary deliverable and foregrounded its format evidence, which was stale relative to H81 and potentially confusing. Replaced that top banner with current H81 download/submit status and the exact authorization/holdout/uniqueness rationale. The historical lane writeups and original long prompt remain preserved.
- No official non-GitHub website was fetched during this review. Official links below are supplied for manual review; no fresh assertion about current portal rules, official data availability, leaderboard state, licensing, or live competition acceptance is made here. The local bridge data and SHA-pinned owner-repository mirrors are not organizer authentication.

## H33 reported score: reasoned interpretation and limits

The file named `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` is reported as `0.2778` by the user and sibling-site ledger (**OWNER-REPORT**). The local explanation files describe the family lineage as repeated thinning of an earlier H19-5 ridge surface, followed by removal of dots near catalogue traces. That pathway is plausible under distance-weighted Tversky: removing predictions whose closest match is a known/masked fault can reduce false-positive mass, and spatial thinning can reduce redundant mass around the 300 m kernel radius. This analysis does not prove those operations caused the reported score, because the submission-page receipt and matched organizer evaluation are absent.

The exact marginal rule in the repository's metric analysis is conditional on the official metric definition and model assumptions: adding one unit of prediction mass with realized kernel credit `k` improves a current DTI `D` only if `k > αD`; with `α=0.2`, predictions of low credit can lower the score. The equations and brute-force local checks are in `research/why-h33-scored-02778.md` and `scripts/verify_all.py`. This is useful explanatory algebra, not a projection of H81 or any future score.

**Can we beat 0.2778?** In principle, yes, if a better field concentrates prediction mass near genuinely unseen faults. This repo does not establish that it has done so: H81's specified **HOLDOUT-DTI 0.067071** (`gems57-pooled-hide-v2`, 11,321 withheld positives, 95% CI `[0.055232, 0.077908]`) is well below the retained best. No leaderboard improvement is predicted or claimed.

## Untried geological hypotheses retained for a future, separately assigned lane

These are the unrun candidates from the preregistered H81 research screen. Ranking is qualitative expected usefulness relative to implementation cost—not a numerical DTI projection. Candidate 1 (H81 slip cones) is excluded from the untried list because it was already evaluated. None is authorization to spend a slot.

| Rank | Layers and physical signature | Why it may reveal an uncatalogued fault | Named non-fault mimic | Difference from implemented repository lanes | Cost / viability and official source |
|---:|---|---|---|---|---|
| 1 | `existing_faults.tif` visible-only per fold; windowed local strike entropy / fault-strike heterogeneity at transfer zones. | New faults may nucleate or transfer at bends/intersections where mapped traces change orientation; the test asks about complexity around visible structures, not simply extending a mapped line. | Digitization jitter, anastomosing swarms, and rasterization noise. | No strike-entropy surface was found in the H81 preregistered code/history search; unlike H77-Step/H77D tip mechanics, this is a windowed orientation statistic. Must still run a fresh full-code novelty check before assignment. | **Low implementation cost; expected signal uncertain/low-to-moderate.** Uses competition catalogue data; no new external data required. Official competition data access is login-walled according to existing repo notes, and bridge provenance is not organizer-authenticated. |
| 2 | 10 m-derived topography (`hs_lineament`, `curv_prof_absmax`) plus a roads-distance exclusion layer; long 5–30 km N–S Hough lineaments. | A basin-crossing buried range-front structure may leave a long coherent lineament without an exposed scarp and therefore be absent from a surface-fault catalogue. | Roads, canals, survey seams, and field boundaries. | Hough-scale, directional line detection with explicit infrastructure rejection differs from H77-Step's Radon worms and H78's LiDAR scarp consensus. | **Medium-to-high cost; conditional/blocked.** Official roads source proposed in H81 notes: U.S. Census TIGER/Line (`https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html`). Obtainability for this runtime/AOI was not checked here; prior H81 notes say the required road-distance raster was absent from the owner mirror. Do not call viable until source/tile and licensing are checked. |
| 3 | Official GDR 1391 wellspring chemistry (`gdr.openei.org/submissions/1391`), particularly silica/chalcedony geothermometry, plus thermal/wellspring context. | Reservoir-temperature-filtered hydrothermal discharge may prioritize deep circulation pathways that are structurally controlled but lack a mapped scarp. | Deep non-fault aquifers, irrigation return flow, and mixing/alteration that biases geothermometers. | Existing lanes use thermal proximity/counts; chemistry-derived reservoir-temperature inference is a different physical filter. | **Medium cost; source existence is linked but not freshly checked here, and usable in-footprint chemistry coverage is unknown.** Treat as blocked until official metadata, license, coordinates, and completeness are verified. |
| 4 | USGS ComCat FDSN event locations, depths, magnitude/uncertainty and moment-tensor metadata, with competition geophysics as context; robust sub-km seismic lineaments. | A depth-filtered fault-plane cluster may indicate active blind structures not present in a surface catalogue; event geometry is independent evidence from static topography. | Quarry blasts, volcanic swarms, aftershock sequences, and uneven network detection. | Shipped coarse seismic aggregates are not equivalent to event-level local plane fitting; no event-level lane is recorded in this repository. | **High cost; BLOCKED for this sandbox.** Specific free official endpoint: `https://earthquake.usgs.gov/fdsnws/event/1/query`. H81's logged access probe returned HTTP 000 / TLS exit 35 under the sandbox egress restriction, and no event-catalog mirror was found in its scan. No external data were fabricated. |

The scientifically strongest next development is not to run one of these immediately: first resolve the shared uniqueness-policy contradiction created by dense/full-support witnesses, obtain a complete accessible registry and organizer-authenticated input provenance, and assign one specific method paragraph. Then preregister one lane, run its feature-alone canaries and spatially blocked holdout using the pinned evaluator, and stop if it fails to beat the comparable incumbent. This review itself used no experiment slot.

## Verified facts, manual-review sources, and access limits

| Subject | Source link | What this review can say |
|---|---|---|
| Competition problem, metric and submission format | [DrivenData problem page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) | Manual-review link from the prompt/repository. Not fetched during this review. |
| Competition data | [DrivenData data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) | Existing repo notes say it redirects to login; no auth was available or attempted. |
| Rules / submission instructions | [DOE/NLR PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf) | Manual-review link; not freshly retrieved. |
| Reference method | [DrivenData reference solution](https://github.com/drivendataorg/gems-prize-reference-solution) | Official project repository link; not freshly fetched in this review. |
| Regional geophysics | [USGS GeoDAWN release](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and) | Official source link; exact band interpretation is not reverified here. |
| Topography | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | Official source link for the future drainage/topography/road-context hypotheses; AOI-specific availability not checked here. |
| Seismicity | [USGS ComCat FDSN](https://earthquake.usgs.gov/fdsnws/event/1/query) | Specific free official endpoint; sandbox retrieval failed in the prior H81 evidence, so the hypothesis is blocked here. |
| Geothermal springs and wells | [DOE GDR submission 1391](https://gdr.openei.org/submissions/1391) | Official data-portal link; coverage/license/chemistry columns not independently audited here. |
| Projection | [EPSG:32611](https://epsg.io/32611) | EPSG reference link; local raster metadata measured as EPSG:32611. |

## Next work (separate session / selector decision required)

1. Resolve the dense-witness overlap rule in the shared template/protocol, not in a private fork. If the literal rule is retained, report clearly that it makes nonempty artifacts collide with dense full-support witnesses; do not quietly exclude them.
2. Obtain a complete, current, auditable registry and organizer-authenticated competition inputs. Reconcile the duplicate invalid/off-grid registry entries rather than calling an incomplete scan a uniqueness pass.
3. Assign a fresh method paragraph and preregister only after checking its code/data novelty against all concurrent lanes. Do not retune closed H81/H80/H79/H77 lanes.
4. For a fresh candidate, use the pinned cached inputs/evaluator and shared writer, hide buffered whole fault segments, derive catalogue-based features only from visible faults, mask visible faults exactly, run feature-alone leakage canaries, and report pooled HOLDOUT-DTI with evaluator, withheld-positive count, and 95% CI.
5. Require improvement over the comparable holdout incumbent and both surface and final-dot uniqueness checks before a separate selector considers a slot. Preserve the unique submission name and ≤140-character note, then seek an organizer receipt before describing any live score.
6. Refresh the documentation/site only from a measured run card. Keep the first screen explicit: **OK to download for inspection** and **OK to submit** are separate states.

## Run card — repository-review outcome

```json
{
  "hypothesis": "Review H81 slip-budget fault-tip continuation cones, the retained explanation of the H33 family score, submission integrity, and repository release safety; no new scientific experiment was run.",
  "mechanism": "H81 uses strand-level mapped-fault tips to emit slip-ranked outward cones. The intended mechanism is continuation/transfer beyond mapped tips; the named non-fault mimics are map gaps, digitization fragments, en-echelon overlap, and grid truncation.",
  "named_non_fault_process_that_could_mimic_it": "Mapping-gap strand ends caused by erosion/cover or digitization; grid-truncated traces; for the H33 thinning explanation, catalogue-adjacent false positives and redundant mass are the alternative scoring mechanism, not proof of fault geology.",
  "holdout_dti": {
    "evidence_class": "HOLDOUT-DTI",
    "evaluator_version": "gems57-pooled-hide-v2",
    "withheld_positive_pixels": 11321,
    "value": 0.0670711968860764,
    "ci95": [0.05523215485018934, 0.07790796160479564],
    "comparison": {
      "evidence_class": "HOLDOUT-DTI",
      "evaluator_version": "gems57-pooled-hide-v2",
      "withheld_positive_pixels": 11321,
      "incumbent_value": 0.152718,
      "incumbent_ci95": [0.131548, 0.175185],
      "beats_incumbent": false
    }
  },
  "correlation_overlap_vs_registry": {
    "evidence_class": "REGISTRY-MEASUREMENT, frozen H81 receipts",
    "preplacement_surface_max_abs_spearman": 0.5402830747429385,
    "final_dots_max_abs_spearman": 0.13332225246271873,
    "final_dots_worst_forward_3px_overlap": 1.0,
    "overlap_threshold": 0.7,
    "registry_indexed": 836,
    "registry_readable": 832,
    "scan_complete": false,
    "verdict": "duplicate-stop; not protocol-unique"
  },
  "raster_sha256": "f3eb9e5e3bf03da0285b48b5bb7d41f15b0e450d16bc6b8943aeaa63efadc1f7",
  "validator_output": {
    "evidence_class": "independent local readback against tracked bridge sample grid",
    "no_nan_inside_footprint": true,
    "all_values_finite": true,
    "values_in_0_1": true,
    "crs_match": "EPSG:32611",
    "shape_match": [3730, 3292],
    "transform_match": true,
    "single_band_float32": true,
    "zip_contains_exactly_one_identical_tiff": true,
    "release_gate_authorized_files": 0
  },
  "submission_name": "gems77-h81-slipcone-37654dots-20261021-0fbeaf41-zeros",
  "submission_note": "H81 slip-budget tip cones 37654 dots 0-on-cat; HOLDOUT-DTI 0.0671 [0.0552,0.0779]",
  "verdict": "negative"
}
```
