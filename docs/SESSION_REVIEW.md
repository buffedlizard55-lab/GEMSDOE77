# GEMSDOE77 session review — 2026-10-21

## Scope and evidence

Reviewed the entire task request, the local checkout, the cited GEMSDOE32 page, the official DrivenData problem/about/data/leaderboard pages, the official reference-solution repository endpoint, and the public GDR INGENIOUS listing.

### Local checkout facts

- Branch: `arena/3f4dbd12-gemsdoe77`.
- Starting commit: `55ad721` (`Initial commit`).
- Starting worktree was clean.
- Exactly one tracked file existed at review start: `README.md` (11 bytes, heading only).
- No scripts, tests, app/site, inputs, rasters, experiment outputs, or model artifacts were present.
- Python imports checked in the active environment: `rasterio`, `numpy`, `scipy`, and `pytest` are all missing.

### Source checks and irregularities

| Claim/source | Observed evidence | Interpretation |
|---|---|---|
| Official problem description | Identifies fault detection, incomplete public USGS labels, test labels, listed competition feature layers, the DTI formula, EPSG:32611/100 m and single-band float32 submission constraints. | Primary authority for task facts and file-format requirements. |
| Official data page | Resolves to a DrivenData login page in this unauthenticated session. | Competition rasters cannot be downloaded from this environment. No access-control bypass attempted. |
| Official leaderboard | On 2026-10-21 page read, first row is `xiaofanhu`, `0.4413`. | Resolves the prompt's inconsistent `0.3195` versus `0.4413`: `0.4413` is the visible public leader at this check time. It can change. |
| H33/GEMSDOE32 page | Describes a H33-2-B2 candidate, a 0.2708 anchor, reported holdout/proxy comparisons and a `0.2747` live-model estimate explicitly called a projection. It states no artifact in that repository has an organizer score. | Does not independently establish the request's `0.2778` as an organizer-confirmed score. The candidate mechanism may be discussed as owner-reported, not as verified cause of a confirmed score. |
| Reference-solution repository | `git ls-remote` and GitHub API returned a public repository and commit/tree metadata. The regular HTML page parser returned a 404 template on the first fetch, which is a renderer/access irregularity; the repository itself exists. | Do not infer the repo is missing from the API check. The reference solution is not present in this checkout. |
| GDR INGENIOUS 1391 | Public page reports DOI `10.15121/1881483`, public access and CC BY 4.0, and lists a collection of regional datasets. | Official public research data are potentially obtainable; each resource's license/citation still needs review. Not a replacement for the competition's sample grid or stack. |
| Submission request vs available files | No sample/template or grid metadata exists locally. Prior sites host older candidate files, but reusing those would violate the uniqueness requirement. | No valid, unique competition GeoTIFF can responsibly be generated from this checkout. No TIF was generated. |
| Parallel-run protocol lane | It refers to “the single method paragraph below,” but no separately delimited method paragraph is present. | Candidate 1 is only a tentative research lane; explicit lane specification is ambiguous. Do not mistake planning for completed experiment. |

The competing score values supplied in the prompt and historical results across sibling sites are not organizer receipts in this checkout. They are not all independently verified. The leaderboard value above is the only current public-score observation made during this session.

## Why the historical H33 value cannot yet be explained as a fact

The official metric is distance-weighted Tversky, with the problem page specifying alpha `0.2`, beta `0.8` and a 300 m triangular kernel. This makes omission of nearby truth more costly than an equal amount of false-positive mass, while predictions close to truth receive distance-weighted credit. That explains the *general incentive* to place a compact prediction near likely traces; it does not prove why a particular file got any particular organizer score.

The H33 page gives an owner-described thinning/pruning mechanism and reports its own holdout/projection tables. Its `0.2747` is explicitly a modeled estimate, and the page says it has no organizer score for its artifacts. The alleged `0.2778` is not shown as an organizer receipt in the material reviewed. To establish a causal explanation for `0.2778`, one would need the exact submitted raster, the matching organizer receipt/score record, and reproducible scoring/ablation evidence. Those are absent. Therefore this review does **not** assert that H33 scored 0.2778, or that any reported feature caused a confirmed score.

## Three review passes

### Pass 1 — inventory and implement the safe project foundation

- Replaced the placeholder README with a durable project charter: objective, evidence labels, official sources, data gate, submission workflow, uniqueness tests, holdout/leakage protocol, core values, historical-score caveats, candidate register, and current blockers.
- Added a five-hypothesis research ranking. All are explicitly tagged as proposals; no fabricated holdout values or external-data availability claims are presented as results.
- Named the non-fault mimic for each hypothesis and identified that the top candidate uses competition-provided layers if they are available.

### Pass 2 — adversarial review for assumptions and safety

- Cross-checked the live official leaderboard instead of relying on stale numbers in the long prompt.
- Compared the cited H33 page's candidate description with its own disclaimer. Discovered a material irregularity: a supposed `0.2778` result is not supported there as an organizer score; a `0.2747` number is explicitly a projection.
- Confirmed no local inputs/outputs/evaluator/template and no raster libraries. Rejected the unsafe alternatives of fabricating a grid, submitting a zero raster, or copying an older candidate.
- Confirmed the missing “single method paragraph” makes the lane instruction ambiguous. Recorded it rather than silently inventing a binding method.

### Pass 3 — requirements re-check and final validation

- Checked that the README states the TIF availability outcome plainly, gives a valid submission process, includes official review links, and differentiates official facts from owner-reported claims and hypotheses.
- Confirmed that no submission slot was spent and that no performance score, CI, correlation, overlap, validator result, or SHA-256 was invented.
- Final validation of any TIFF and actual holdout experiment remains **blocked** because required inputs, code, and dependencies are absent. No unique TIF can be responsibly offered as download-and-submit-ready in this session.

## Next work (gated order)

1. Add or obtain the authorized `training_features.tif`, labels/fault vectors, and `sample_submission.tif`, plus the official 1 m DEM link table if it is part of the planned experiment. Store data outside Git if large or restricted; record filenames, hashes, source, and licenses.
2. Obtain the actual shared-template feature stack, whole-segment buffered holdout evaluator, submission writer, and registry rasters. Verify evaluator version and mask behavior before running any new candidate.
3. Add automated unit tests for grid parity, mask/nodata behavior, range and finite-value validation, distance-weighted metric correctness, and train/holdout leakage boundaries.
4. Preregister Candidate 1 and its fixed-budget comparison. Run per-feature canaries first; stop if any feature-only AUC exceeds `0.90` without an explained non-leakage mechanism.
5. Run the holdout and duplicate thresholds on surface and final dots, including paired 95% CI and withheld-positive count. Do not generate/offer a submission file unless it beats the incumbent holdout under the agreed gate and passes independent validation.
6. Build the requested user-facing executive summary/download page only after a passing artifact exists. Show file, unique competition name, <=140-character note, checksum, validation receipt, and a prominent “download and submit” state. Show “no validated candidate” otherwise.
7. Treat the weekly slot as a separate selection action. Obtain an actual organizer receipt before labeling anything organizer-confirmed.

## Machine-readable run card

```json
{
  "hypothesis": "Candidate 1: depth-stratified MT conductance breaks corroborated by compatible magnetic-source-depth/TMI boundaries",
  "mechanism": "A buried fault-related basement discontinuity may align an alteration/conductance transition with a distinct magnetic structural boundary; proposed, not tested.",
  "named_non_fault_mimic": "A lithologic contact or basin edge producing coincident conductivity and magnetic contrasts without a fault-related geothermal pathway.",
  "holdout_dti": {
    "label": "HOLDOUT-DTI",
    "evaluator_version": "NOT AVAILABLE",
    "withheld_positive_count": "NOT RUN",
    "value": "NOT RUN",
    "95_percent_ci": "NOT RUN"
  },
  "correlation_overlap_vs_registry": {
    "rank_correlation": "NOT RUN — registry rasters absent",
    "fraction_of_dots_within_3px": "NOT RUN — no final dots or registry rasters"
  },
  "raster_sha256": "NOT GENERATED",
  "validator": {
    "no_nan_inside_footprint": "NOT RUN — no raster",
    "values_in_0_1": "NOT RUN — no raster",
    "crs_shape_transform_match": "NOT RUN — no template/grid metadata"
  },
  "submission_name": "NONE — no submission artifact",
  "submission_note": "No validated candidate; no TIF was generated.",
  "verdict": "negative",
  "verdict_reason": "Blocked by absent authorized rasters, evaluator/writer, registry rasters, and local geospatial dependencies; no evidence supports promotion."
}
```
