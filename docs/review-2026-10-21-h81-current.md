# GEMSDOE77 current checkout review — H81 status and release-site correction

## Decision at a glance

**No new experiment was run and no new GeoTIFF was generated in this review. H81 is stopped; do not resume or upload it.** Its already-generated TIF is available for inspection only. The release receipt authorizes zero files. This is the correct outcome under the registered parallel-run protocol: H81's final-dot audit exceeded the 0.70 overlap limit, recorded 69 duplicate firings, and marked the registry scan incomplete. A byte-new filename is not a uniqueness pass.

The review did make a **site/documentation correction**: the current landing page and README had stale featured-file messaging for H80/H77E-Knick, despite a newer H81 negative artifact in the inventory. The generated first screen now features H81 with explicit `OK TO DOWNLOAD FOR INSPECTION: YES` / `NOT OK TO SUBMIT: NO`, its download links, exact local holdout metadata, validator summary, duplicate-stop result, and incomplete registry scope. Source of truth is `docs/data/lanes.json` plus `scripts/build_home.py`; generated pages are not hand-edited.

## Pass 1 — prompt, lane, and repository inspection

- Read the standing brief (`PROMPT.md`), README, agent notes, irregularity ledger, H81 preregistration/amendments, H81 evidence, submission authorization, shared-template pin, current site sources and prior review files.
- `registry/preregistration_h81.json` defines the active H81 method as slip-rate-weighted continuation cones from strand-level trace tips. `research/h81_hypotheses.md` already records five ranked candidates, their layers/signatures, mimics, implementation distinctions, and source/access notes. This review did not invent another lane or extend the three-experiment budget.
- The H81 run card records `verdict=negative`, `ok_to_download=true`, `ok_to_submit=false`, `submission_slots_used=0`, and a final-dot uniqueness stop. `docs/data/submission-authorization.json` records zero authorized files.
- This checkout has only the tracked bridge rasters under `data/bridge`; the H81 pipeline's full restored feature cache is not part of the checkout. Therefore no independent rerun of H81's holdout is claimed here.
- No current official competition, leaderboard, USGS, DOE GDR, or NLR page was freshly fetched during this review. The official links are retained for manual source review; H81's prior data/source receipt is an owner-mirror and page-read record, not organizer authentication. No new external source claim is made.

## Pass 2 — evidence and generated-page review

Frozen H81 values below are read from `docs/evidence/h81_run_card.json` and associated evidence, not recalculated in this review:

- **HOLDOUT-DTI 0.067071** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.055232, 0.077908]`). This is below the repository's recorded **0.152718 HOLDOUT-DTI** incumbent; the latter is a separate historical measurement, not a paired interval here. Neither is an organizer score.
- Surface-stage maximum absolute Spearman rank-correlation was `0.540283` (below the registered `0.90` stop threshold). The final-dot audit reports maximum absolute Spearman `0.133322`, but maximum forward 3-px dot overlap `1.000` (> `0.70`) against `GEMSDOE48:docs/downloads/GEMSDOE48-unassigned-mass-20261006.tif`. It recorded `69` duplicate firings, `832` rasters checked, and `complete_accessible_scan=false`. Duplicate stop is sufficient; the incomplete scan cannot certify uniqueness.
- The run card's local validator says one float32 band, `EPSG:32611`, shape `3730×3292`, sample transform match, no NaN/Inf, values in `[0,1]`, and zero positive pixels on mapped catalogue. This is a frozen local format receipt, not a fresh organizer validation or portal acceptance.
- Raster SHA-256 in the frozen run card is `f3eb9e5e3bf03da0285b48b5bb7d41f15b0e450d16bc6b8943aeaa63efadc1f7`; the named submission note is within 140 characters. The H81 `.tif` and one-TIFF `.zip` are already present under `docs/downloads/`.
- Fixed the current landing-page source to feature H81 rather than stale H80, and added a matching H81 manifest row with its format/holdout/uniqueness evidence. The H80 lane remains in the inventory as historical, not as the latest featured research result. The executive guide calls out the featured lane dynamically.

### Protocol/holdout caveat identified for follow-up

The H81 feature builder creates a fold-union `hidden_all` mask and uses it to censor trace components whose vertices are near hidden truth (`component-LOCO`). Its separate `slip_proximity` canary explicitly uses full-catalogue distances and is documented as canary-only, not an arm. The primary cones do not use that canary, but the feature-selection rule itself is conditioned on withheld-label locations. The preregistration calls this a conservative LOCO discipline; nevertheless, the standing protocol says catalogue-derived features must be built only from visible faults. A strict independent audit should establish whether the trace CSV is a separate public catalogue, whether filtering using hidden-label coordinates is permitted, and whether the holdout must rebuild visible catalogue geometry independently per fold. Until this is resolved, treat the recorded H81 HOLDOUT-DTI as a frozen local diagnostic with a protocol caveat, not as promotion evidence. This concern does not change the decision: its result is negative and the duplicate stop independently prohibits promotion.

## Pass 3 — corrected outputs and remaining limits

- `docs/index.html` now puts H81's unambiguous inspection-only download/status box first. The generated executive summary names H81 as the current featured research artifact while retaining the no-upload instructions.
- `README.md` now begins with H81's actual current status and records the H81 result, checksum, and final-dot stop; stale H77E-Knick primary wording has been replaced. The broad historical inventory remains archival and should not override the current banner or machine-readable authorization receipt.
- `AGENTS.md` now points to the current paths (`docs/evidence/...`) and explicitly says H81 is closed, the three-experiment budget is exhausted, and no TIF may be represented as approved.
- No changes were made to `vendor/gems57`, evaluator/writer code, the H81 raster bytes, any holdout score, or the weekly submission selector. No new competing candidate was selected.

## Official references retained for manual review

These links are anchors from the project brief/prior evidence; **not freshly fetched during this review**:

- Competition problem/metric/submission format: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Competition data (login required): https://www.drivendata.org/competitions/306/competition-doe-gems/data/
- Leaderboard: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
- Official Rules PDF: https://docs.nlr.gov/docs/fy26osti/96647.pdf
- USGS GeoDAWN data release: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
- USGS 3DEP: https://www.usgs.gov/3d-elevation-program
- DOE GDR INGENIOUS submission: https://gdr.openei.org/submissions/1391
- INGENIOUS project: https://gbcge.org/current-projects/ingenious/
- H81 preregistration/evidence: [`registry/preregistration_h81.json`](../registry/preregistration_h81.json), [`research/h81_hypotheses.md`](../research/h81_hypotheses.md), [`docs/evidence/h81_data_receipt.json`](evidence/h81_data_receipt.json), [`docs/evidence/h81_run_card.json`](evidence/h81_run_card.json).

## Recommended next steps

1. Resolve the dense/full-footprint witness interpretation once in the shared protocol and registry; do not waive it privately or adjust a candidate to evade it.
2. Audit H81's holdout feature provenance: the Qfault-derived trace CSV source, hidden-mask culling, fold-union censoring, and whether all catalogue-derived features are truly fold-local.
3. Assign the next method paragraph explicitly; do not start another lane until H81's budget/stop is closed and the registry/protocol questions are resolved.
4. Restore authenticated competition data and a complete current registry snapshot, then use the pinned shared evaluator/writer unchanged. Require feature-alone canaries, fold-local features, matched blocked holdout improvement, uniqueness at both surface and final dots, and independent local format validation before a separate selector considers any slot.
5. Tie any eventual ORGANIZER-CONFIRMED result to an actual submission-page receipt and exact raster checksum. Never translate holdout performance or sibling-site owner reports into a leaderboard projection.

## Review run card

```json
{
  "hypothesis": "Audit only; no new scientific experiment. Current lane is H81 slip-budget strand-tip continuation cones, negative and stopped.",
  "mechanism": "Existing H81: deterministic 2 km, 30-degree outward cones at strand-level trace tips weighted by slip-rate rank; catalogue geometry from a pinned owner-mirror trace table.",
  "named_non_fault_process_that_could_mimic_it": "Mapping-gap strand ends from erosion/cover; grid truncation; en-echelon links already represented in the catalogue.",
  "holdout_dti": {
    "evidence_class": "HOLDOUT-DTI (frozen prior evidence; not recalculated in this review)",
    "evaluator_version": "gems57-pooled-hide-v2",
    "withheld_positive_pixels": 11321,
    "value": 0.0670711968860764,
    "ci95": [0.05523215485018934, 0.07790796160479564],
    "protocol_caveat": "feature eligibility uses coordinates from the union of held-out truth; strict fold-local compliance needs independent resolution"
  },
  "correlation_overlap_vs_registry": {
    "surface_max_abs_spearman": 0.5402830747429385,
    "final_max_abs_spearman": 0.13332225246271873,
    "final_max_forward_3px_dot_overlap": 1.0,
    "duplicate_firings": 69,
    "registry_rasters_checked": 832,
    "complete_accessible_scan": false,
    "verdict": "negative; stop required"
  },
  "raster_sha256": "f3eb9e5e3bf03da0285b48b5bb7d41f15b0e450d16bc6b8943aeaa63efadc1f7",
  "validator_output": {
    "evidence_class": "frozen local validator receipt; not organizer acceptance",
    "no_nan_inside_footprint": true,
    "values_in_0_1": true,
    "crs_shape_transform_match": true,
    "crs": "EPSG:32611",
    "shape": [3730, 3292],
    "transform": [100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0]
  },
  "submission_name": "gems77-h81-slipcone-37654dots-20261021-0fbeaf41-zeros",
  "submission_note": "H81 slip-budget tip cones 37654 dots 0-on-cat; HOLDOUT-DTI 0.0671 [0.0552,0.0779]",
  "verdict": "negative",
  "ok_to_download_for_inspection": true,
  "ok_to_submit": false,
  "weekly_submission_slots_used": 0
}
```
