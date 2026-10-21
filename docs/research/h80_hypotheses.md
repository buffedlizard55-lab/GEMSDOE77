# GEMSDOE77 H80 — pre-experiment hypotheses (2026-10-21)

Evidence classes: **[OFFICIAL]** DrivenData #306 / USGS · **[MEASURED]** this session.

Competition: predict faults **not** in the USGS/INGENIOUS catalogue; DTI α=0.2, β=0.8, 300 m triangular kernel; EPSG:32611 100 m 3730×3292 **[OFFICIAL]**.

Family live high `h33-h33-2-b2…-zeros` = **0.2778 OWNER-REPORT** (GEMSDOE32 site; not an organizer receipt in this repo). Public #1 **0.4413 ORGANIZER-CONFIRMED** (leaderboard read 2026-10-21).

## What the available H33 evidence does—and does not—show

Archived raster forensics describe H33 as a **kernel-efficiency construction**: a 1-px multi-physics ridge → Poisson dots at about 300 m → pruning dots within 2 px of the mapped catalogue. Under DTI, reducing redundant false-positive mass while retaining weighted true-positive credit is a plausible way to improve a score. This is a **metric-level mechanism hypothesis**, not a measured causal effect: the repository has no controlled ablation linking those operations to the owner-reported 0.2778, and no inspected organizer receipt linking that leaderboard row to the H33 bytes. A future method could improve by increasing true-positive credit per dot, reducing false-positive mass, or both; the evidence does not establish that new true-positive mass off the H19-5 ridge is the only route.

## Ranked untried hypotheses (this lane)

### H80-1 (implemented) — Riedel R / R′ wing cracks at degree-1 tips
- **Layers:** mapped catalogue raster only (`existing_faults.tif` / labels), local strike from `gems57.network.local_strike`.
- **Signature:** pixels of degree 1 (trace tips). From each tip, emit two rays at **+15° (R)** and **−75° (R′)** from local strike, 4–10 px (400–1000 m), Gaussian falloff, **zero on the visible catalogue**.
- **Why catalogue-missing:** Qfaults/INGENIOUS omit short wing cracks and conjugate Riedel shears at mapped tips; experts drawing hidden truth from lidar still mark them (Pollard & Segall 1987; Riedel 1929).
- **Named non-fault mimic:** linear drainages leaving canyon mouths; cadastral/road alignments.
- **Differs from this repo:** H33/H19-5 are ridge *centerlines*; H77E Euler/UCP; H79 radiometric lag; D77 annulus; H77L ThK lag. None emit **fixed 15°/75° Coulomb rays from degree-1 tips**.
- **Cost:** low. Validatable: yes (visible-only geometry).

### H80-2 — Dilatation / shear-rate sign change (GeoDAWN geod_dilaterate × geod_shearrate)
Needs `training_features.tif` bands 7–8. Deferred: 419 MB restore not required for H80-1.

### H80-3 — Conductivity Laplacian ridges (`cond_surf` band 17)
Same data dependency. Deferred.

### H80-4 — χ-index knickpoint corridors
Needs hydrology DEM; not in bridge. **[BLOCKED]** without 3DEP hydro.

Rank for this session: **H80-1 only** (budget: 3 experiments).

## Post-run clarification (2026-10-21)

The preceding H33 mechanism and H80 target rationales are hypotheses, not causal or label-proven results. The 0.2778 file-to-score attribution remains **OWNER-REPORT**: the official leaderboard lists a 0.2778 result for `extradr19` at #48 but does not identify a TIFF or hash, and no organizer receipt links those bytes to the H33 artifact. Sparse spacing and near-catalogue pruning could reduce redundant false-positive mass under the distance-weighted metric, but no controlled experiment here establishes that they caused the reported score. H80 itself is negative: **0.09580 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,319 withheld positives, 95% CI `[0.08818, 0.10325]`). Do not interpret the pre-run rationale as confirmation that the proposed wing-crack geometry is present in the final expert-labeled targets.
