# H80-OTA knowledge note: orphan thermal alignment — what was learned (2026-10-21)

All scores are HOLDOUT-DTI (`gems57-pooled-hide-v2`, 11,342 withheld px, seed 30) unless
labelled otherwise. Evidence: `evidence/h80_exp1.json`, `evidence/h80_exp2.json`,
`evidence/h80_exp3_build.json`, `evidence/h80_exp3_compare.json`, `evidence/run_card_h80.json`.

## Findings (reusable by later lanes)

1. **Off-catalogue hot wells exist in useful numbers but chemistry is sparse.**
   GDR 1391 in-footprint file: 27,092 rows / 12,570 cells, but only 1,002 deduped
   tier A/B hot points (412 with geothermometer excess ΔT ≥ 30 °C & max ≥ 90 °C;
   590 with temp ≥ 40 °C); 654 (holdout) / 630 (full map) orphans > 1 km from the
   reference catalogue. Any point-process lane on this file is support-thin by
   construction — budget accordingly.
2. **Collinearity of orphan hot points is, at best, very weak fault evidence at
   100 m.** Canary AUCs: line field 0.502 (chance), ridge-intersected 0.559,
   orphan proximity 0.501, hot proximity 0.518. As an emitter surface it ties/loses
   to random (0.0475 vs 0.0500). Springs/wells cluster in developed corridors
   (roads, valley axes, geothermal fields), so collinearity mostly measures
   settlement geometry, not blind faults — the named mimic won.
3. **Inside a tabular model the channels add a small, non-significant amount**
   (+0.0068 [−0.0007, +0.0149]). Do not spend another lane re-testing the same
   four channels with different constants; a revival needs new point data
   (e.g., 2 m probe grids or seismicity hypocentres), not re-weighting.
4. **Rank ties are a leakage-grade trap for sparse fields (IR-H80-04).**
   Scan-order tie-breaking on a 98%-zero field manufactured a latitude gradient
   worth a fake “significant” +0.0117. Any future rank transform in this family
   MUST use tie-averaged ranks; `features77.rank01` shares the scan-order pattern
   and should be audited where its inputs are sparse (its current uses are dense
   continuous fields, where ties are rare — flagged, not changed here to avoid
   forking shared code mid-lane).
5. **Final-dot placement is dominated by the shared 33-channel stack.**
   Surface ρ vs siblings is low (≤0.32) yet 87–92% of emitted dots fall within
   3 px of earlier sparse dot rasters. Channel-level novelty does not imply
   dot-level uniqueness; check overlap early (cheap subset) before Exp 3.
6. **Control replication works:** A_std reproduced XSEN’s 0.0971 to 4 decimals
   from a separately assembled 33-channel stack — the shared evaluator/holdout/
   emitter path is deterministic across sessions when seeds are held fixed.

## Standing question refresh (why 0.2778; can we beat it?)

- Verified 2026-10-21: GEMSDOE32’s site still offers
  `gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif` (sha prefix
  c55bafc4, matching `evidence/why_02778.json`) and states **no organizer score
  exists for that file**; the 0.2778 attached to it remains **OWNER-REPORTED**.
  Public leaderboard #1 = **0.4413 ORGANIZER-CONFIRMED** (xiaofanhu), #2 0.3904,
  read 2026-10-21 from
  https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/.
- Mechanism (measured, `reports/why_02778.md`): 3-px Poisson-style thinning of the
  H19-5 ridges (kernel credit is a max, so solid lines waste ~2/3 of FP-weighted
  mass) + deleting dots ≤ 200 m from the catalogue (organizer scores only
  unmapped faults). Credit/dot 0.125 vs α·DTI bar ≈ 0.056; S0 inversion finds
  hidden-truth density ≈ 38× footprint average on H19-5 pixels > 2 km from the
  catalogue. Beating it requires higher top-of-ranking credit density off the
  H19-5 support (≥ 0.056/dot from off-support dots just to tie under the
  uniqueness rule); reaching 0.4413 needs ≈ 0.194/dot and ≈ 70% recall.
  H80 does not supply that. Next credible directions (from
  `research/hypotheses-h80-2026-10-21.md`): nonlinear cross-field residual ridges
  (H80-XRES design), learned 2D segmentation on GPU with metric-aware emission,
  or new point data (hypocentres / 2 m probes) — each must beat 0.152718 on the
  frozen holdout before any slot discussion.
