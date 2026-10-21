# Irregularities register — GEMSDOE77 (2026-10-21)

Each item: what was found, evidence, impact, disposition. **[MEASURED]** =
computed this session from SHA-pinned bytes.

## IR-77-01 — Band 6 tag is wrong: "tilt" band is radiometric total count
- **Evidence [MEASURED]:** embedded tag says *"tc — Tilt angle or total
  curvature — magnetic field derivative"*; Spearman(band 6, external GeoDAWN
  TC grid)=+0.99995, vs derived tilt atan2(tmi_vg,tmi_hg)=+0.020 (200k sample);
  values ∈[2.95, 88.57], median 18.48. `evidence/band6_resolution.json`.
- **Impact:** any sibling feature consuming band 6 as tilt actually used total
  count. The shipped stack has **no** tilt derivative.
- **Disposition:** renamed locally (`tc_totalcount`); all tilt channels here
  derived from bands 9/3 and 11/13. Family-wide re-audit recommended.

## IR-77-02 — Sample submission is not "total fault absence"
- **Evidence [MEASURED]:** competition page calls the sample "predicts total
  fault absence"; its 60,988 ones **exactly equal** `labels.tif` positives
  (pixel-for-pixel, verified).
- **Impact:** anyone using the sample as a zero baseline inherits the
  catalogue itself (which scores ~0 live by construction and would break
  off-catalogue logic).
- **Disposition:** used only as grid/footprint template (finite cells =
  5,167,373 px footprint), never as predictions.

## IR-77-03 — Band 18 is a signed x-derivative, not gradient magnitude
- **Evidence [MEASURED]:** tag "horizontal gradient"; Spearman vs ∂g/∂x=+0.945,
  vs ∂g/∂y=+0.026, vs |∇g|=−0.004; min −15.44 in footprint.
- **Impact:** magnitude-style uses (thresholds, analytic signal denominators)
  on band 18 are invalid; gravity THDR must be recomputed from band 13.
- **Disposition:** recomputed here; channel named `iso_grav_anom_hg_signed`.

## IR-77-04 — Family-best 0.2778 has no organizer receipt in inspected repos
- **Evidence:** GEMSDOE32's site (fetched 2026-10-21) states no organizer score
  exists in that repository for h33-2-b2; the brief assigns 0.2778.
- **Impact:** all family scores remain OWNER-REPORT; only the public
  leaderboard (0.4413 #1, fetched 2026-10-21) is ORGANIZER-CONFIRMED.
- **Disposition:** labelled OWNER-REPORT everywhere; mechanism analysis in
  `research/why-h33-scored-02778.md` is conditional on the reported ordering.

## IR-77-05 — This repository arrived empty (no template, cache, or data)
- **Evidence:** initial checkout contained only `README.md` ("# GEMSDOE77"),
  despite the protocol's "reuse the template's cached stack /
  evaluate_holdout.py / submission_writer.py".
- **Impact/disposition:** template restored byte-identical from GEMSDOE70's
  vendor cache (`vendor/gems57`, upstream 60GEMSDOE per `vendor/README.md`);
  data restored SHA-256 fail-closed from the family manifest. Dropbox mirrors
  in the brief unreachable from sandbox (github.io/dropbox egress blocked;
  github.com/PyPI work). No template defect was found, so nothing was fixed
  upstream this session; the metric was property-tested vs brute force
  (max err 7×10⁻¹⁵).

## IR-77-06 — Brief assigns no identifiable single method-paragraph lane
- **Evidence:** the prompt's lane paragraph slot contains the parallel-run
  protocol itself, not a method (same irregularity GEMSDOE70 recorded).
- **Disposition:** lane declared in `registry/preregistration_contactphys.json` before
  Exp 1 (contact-physics + front-base, no SGMC population, no lidar primary);
  uniqueness gates enforced against 43 local registry rasters on surface +
  dots (0 duplicates). Full-inventory uniqueness remains a selector step —
  the template's `unique` flag is False here *only* for missing completeness
  certification, disclosed wherever uniqueness is quoted.

## IR-77-07 — Official worked example displays rounded components
- **Evidence:** page 967 quotes TP=3.00 FP=1.89 FN=2.00 → 0.60; recomputing
  gives 0.6027. Consistent with 2-dp display rounding of the components, not
  a formula discrepancy.
- **Disposition:** none needed; formula implemented from the equations and
  brute-force-verified.

## IR-77-09 — Shared uniqueness tool has diverged between sibling templates
- **Evidence [MEASURED, post-merge]:** `vendor/gems57/uniqueness.py` on `main`
  (merged via PRs #1–#4) is the literal version (dot support = any finite
  value > 0; forward-overlap gate). The copy this lane restored from
  GEMSDOE70 carries a 2026-10-20 fix (binary 1.0-dot support; mutual-overlap
  gate) written because a whole-footprint continuous registry surface makes
  *every* nonempty candidate a literal "duplicate" (forward overlap ≈ 1).
  This merge kept main's version byte-identical (its hashes are pinned by
  `registry/template_pin.json` tests).
- **Measured consequence:** vs the concurrently merged urank research surface
  (4,961,936 positive px — a dense continuous field, not a dot placement),
  the literal tool reports forward overlap 0.979 ("duplicate"); the
  binary-dot reading reports forward 0.110 / reverse 0.134, ρ = 0.0013.
  Vs the H77C claycap zeros raster both readings agree: distinct
  (0.108 / 0.145, ρ = 0.0027). Full record:
  `evidence/uniqueness_supplement_postmerge.json`
  (`scripts/check_uniqueness_postmerge.py`).
- **Disposition:** this lane's Exp-2 uniqueness numbers (binary-dot tool,
  43 rasters, 0 duplicates) and the post-merge supplement (both tools,
  2 further rasters) are both on record; the lane verdict is negative under
  either reading, so nothing turns on the divergence here. **The family
  should converge on one template version** — recommended: adopt the
  binary-dot/mutual fix upstream with a saturation certificate retained for
  continuous priors, since the literal gate cannot distinguish duplication
  from density.

## IR-79-01 — Initial H79 holdout placement excluded the hidden component
- **Evidence [MEASURED]:** the first runner version required holdout dots to be
  more than 3 px from the full catalogue. Because the hidden positives are
  catalogue components withheld from training, all emitted dots were excluded
  from their credit halo and every arm returned `0.000000`.
- **Impact:** those zero logs were evaluator/runner misuse, not a scientific
  H79 result. They are retained as superseded evidence for audit, not quoted
  as scores.
- **Disposition:** corrected placement to use distance from visible catalogue
  cells only, while still masking visible cells and using the shared
  `evaluate_holdout.py`/`submission_writer.py`. Re-ran the fixed three-
  experiment plan; corrected scores are in `evidence/h79_orient_run_card.json`.

## IR-79-02 — Registry snapshot incomplete and dense witness universal
- **Evidence [MEASURED]:** local exact comparisons covered 13 TIFFs versus 812
  expected. The pinned dense witness has positive support across the allowed
  footprint and produces forward 3-pixel overlap `1.000` before placement and
  after final dots.
- **Impact:** H79 is not protocol-unique; byte novelty or low surface
  correlation against the sparse subset cannot clear it.
- **Disposition:** fail closed. The artifact is safe to download for inspection,
  not OK to submit, and no weekly slot was used. The GitHub API limitation and
  owner-mirror provenance are disclosed in the website and run card.

## IR-79-03 — H79 is a negative research artifact, not a promotion
- **Evidence [MEASURED]:** primary `H79_plus_raw` is `0.089063 HOLDOUT-DTI`
  (seed 20; 95% CI `[0.072384, 0.105355]`; 11,321 withheld positives) and
  `0.094507 HOLDOUT-DTI` (seed 30; 95% CI `[0.076558, 0.112844]`; 11,342
  withheld positives), below retained `0.152718 HOLDOUT-DTI`.
- **Impact/disposition:** no projection, leaderboard improvement or organizer
  confirmation is claimed. The fixed budget is closed at three experiments;
  weekly slots used remains zero.

## IR-77-08 — Operating-point selection bias (self-flagged)
- **Evidence:** budget/exclusion (44,090, >3 px) chosen as the max of 16
  Exp-1 scan cells on the same draw used for arm comparison.
- **Impact:** selected-cell point estimates are optimistically biased;
  paired-CI gate logic is unaffected (all arms share the cell), and the
  seed-21 replication re-used the frozen cell without re-scanning.
- **Disposition:** all 16 cells published in `evidence/holdout_exp1.json`;
  live-optimal mass likely lower (hidden truth sparser than holdout truth) —
  noted for any future selector.
