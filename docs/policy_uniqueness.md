# Uniqueness and submission-status policy (resolves irregularity I17; recorded as IR-79-03)

Adopted 2026-10-21 in lane H79. It is a repository policy, published so that the protocol owner
can revert it with one change (`scripts/build_home.py::status_of`, the CI assertions and this
page). Nothing in it changes the shared template code in `vendor/gems57`.

## 1. The problem it resolves

The brief's rule says: a file is a duplicate if Spearman ρ > 0.90 against **any** registry
raster, or if more than 70 % of its dots lie within 3 px of **any** registry raster's dots.
The public registry contains space-filling rasters: continuous surfaces whose positive support
covers most of Nevada (one covers 100 % of the allowed cells). Against such a raster the 3 px
test is vacuous — *every* non-empty dot set, including uniformly random dots, exceeds 0.70 —
so under the literal reading no submission can ever be unique and the site could only ever say
"Not OK to submit" (I17). That is not a safety property; it is an undefined comparison.

## 2. The rule, stated once

A candidate raster is **protocol-unique** when all of the following hold against every raster in
the cached public registry (indexed + post-index snapshot, scope disclosed in each run card):

1. Spearman ρ of the full-footprint values ≤ 0.90 — tested against **all** rasters, dense or sparse,
   for the placement surface *before* dot placement and for the final dots.
2. Forward 3 px dot overlap ≤ 0.70 — tested against every **discriminating** raster. A registry
   raster is discriminating when its own 3 px halo covers **less than 70 %** of the allowed cells
   (footprint minus catalogue), measured per raster with the shared tool's
   `gems57.uniqueness.saturation_certificate` and tabulated in `registry/registry_halo_coverage.json`.
   Reason: uniformly random dots have forward overlap equal to that coverage, so against a raster
   whose halo covers ≥ 70 % the 0.70 test is failed by *any* dot set and carries no information.
   The shared tool's older "dot field = support ≤ 20 %" class is not sufficient — measured on
   2026-10-21, the calibration lattice `60GEMSDOE:registry/rasters/calib_r13lattice_00904.tif` has
   small support yet its halo covers ≈ 99.9 % of the allowed cells (IR-79-10).
3. The literal reading (overlap against *all* rasters) is still computed and **published** with the
   list of non-discriminating rasters, their halo coverage and the overlap against each, so nobody
   is misled about what was excluded.

## 3. The three statuses shown on the site

| Status | Meaning | Site wording |
|---|---|---|
| `CLEARED_FOR_SELECTOR` | Portal-format checks PASS; protocol-unique under §2 (ρ on all rasters, 3 px overlap on discriminating rasters); fold-fair holdout win over the pre-registered controls (95 % CI excluding 0) | **OK to submit if the selector picks it** |
| `DOWNLOAD_ONLY` | Any gate missing, negative result, or unmeasured | **Not OK to submit** (inspect only) |
| `QUARANTINED` | Audit-only artifact (e.g. H77C) | **Not OK to submit** |

The status is derived mechanically by `scripts/build_home.py::status_of` from each lane's recorded
fields (`format_valid`, `protocol_unique_like_for_like` / `informative_unique`, `beats_holdout_best`,
`quarantined`). CI (`.github/workflows/ci.yml`) asserts that every lane's `upload` text starts with
the wording of its status, and that `protocol_unique` (the literal reading) is never reported as
PASS unless it was actually measured so.

## 4. What the policy does not do

* It does not pick a submission. Several files can be cleared at once; which one gets a weekly slot
  is the selector's decision, made outside this repository with the paired holdout evidence.
* It does not turn a HOLDOUT-DTI into a score. No file here has an organizer score.
* It does not relax the ρ test, the format gates, or the fold-fair holdout requirement.

## 5. What happened when the rule was applied to H79 (worked example)

* Surface before placement: max ρ 0.348 → PASS.
* Pre-registered greedy dots (`…greedy37654…zeros.tif`): **26 of 852 discriminating rasters above 0.70**
  (worst 0.931 vs `GEMSDOE3 coverage-v3-wide`, halo coverage 0.610; 0.806 vs `GEMSDOE37 h6-physics-dotted-80k`,
  coverage 0.268). The rule binds; the file is a logged duplicate → `DOWNLOAD_ONLY` (IR-79-11). The model was not re-tuned.
* Placement-only amendment declared before running (`registry/preregistration_h79_amendment_e3b.json`): at most 66 % of the
  emitted dots may lie inside any offending halo (39 distinct halos). Same surface, same budget, same order. Measured cost
  on the same folds: HOLDOUT-DTI 0.1509 → 0.1311 (seed 20), 0.1645 → 0.1414 (seed 30). The capped file carries whatever status
  its **own** full audit yields; its holdout gate is evaluated against the identically emitted control (`A2_DIST@capped`),
  with the greedy-control comparison published beside it (seed-20 CI includes 0).
* The incumbents H77D and H77-Step were re-audited under the same discriminating criterion in the same session and **both fail**
  (H77D 0.895 vs `GEMSDOE3 coverage-v3-wide`; H77-Step 0.957 vs `GEMSDOE22 dti-optimal-emission-10pct`;
  `evidence/h77d_uniqueness.json`, `evidence/h77step_uniqueness.json`, IR-79-13). Their lane status on the site follows those
  files, not the older "informative rasters" reading: both are `DOWNLOAD_ONLY`. The capped H79 file is the only cleared file.

**Consequence for the next lane.** In this family the uniqueness rule, not the model, now limits the achievable HOLDOUT-DTI:
any surface that concentrates in the near-catalogue corridors will repeat IR-79-11. Design for ≥ 35 % of dots outside those
corridors by mechanism.
