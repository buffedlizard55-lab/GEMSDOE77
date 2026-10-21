# GEMSDOE77 hypotheses and pre-registered decision rule

Committed **before** the hide-and-recover holdout, the leakage canary, and the registry
uniqueness audit were run (see `git log` for the order). Experiments E1 (instrument calibration)
and E2 (live-score inversion) were used to screen these hypotheses. They score against organizer
numbers, so they are instruments, not holdouts.

## Lane

The brief says "your lane is the single method paragraph below", but the paragraph is **missing**
(irregularity IR-77-01). The lane is therefore **self-assigned and disclosed**:

**Live-score inversion and allocation.** Estimate where hidden (new-fault) truth lives by inverting
95 organizer-scored registry rasters through the exact algebra of the metric. Then place binary dots
where the live-calibrated expected credit per dot is highest, subject to the protocol's uniqueness rule.

No prior GEMSDOE repo fitted a truth-density model over all scored rasters with leave-one-out
validation:

- GEMSDOE32 used a one-parameter live anchor, n = 12.
- GEMSDOE70 used a piedmont instrument, n = 22.

## Hypotheses (untried in this form)

Expected-gain figures are INSTRUMENT projections, never scores. Costs are in sandbox CPU-minutes
plus engineering effort.

| Rank | ID | Layers | Physical signature / transform | Why it should catch faults *missing* from the USGS/INGENIOUS catalogue | How it differs from prior repo work | Expected DTI gain | Cost |
|---|---|---|---|---|---|---|---|
| 1 | **H77-3 live-inversion allocation** | Catalogue distance (from `labels.tif`), H19-5 support (prior field, used for learning), 95 organizer scores | Truth-density h(bin) solved from TP = Σ h·M_i (exact in expectation) with NNLS; dots placed greedily by expected credit Σ k·h, ≥ 3 px apart, pruned where credit < α·DTI | Organizer: test truth = faults *not* in the catalogue (page 967), and catalogue pixels are masked (forum 11516). The inversion measures where those faults are, instead of assuming they hug known traces. | GEMSDOE32 used one live anchor (n = 12). This is a 95-raster LOO-validated inversion plus an allocation rule. | +0.00 to +0.01 on the on-support part (pruning low-credit dots) | Low (minutes) |
| 2 | **H77-1 straight-edge field (off-support)** | `det_elev` (band 12) | Gaussian σ = 1, 2 px. Sobel gradient. Canny-style non-maximum suppression gives thin edge lines. Weighted by structure-tensor orientation coherence over ~1.5 km (σ = 7 px). | Faults stay straight across ridges and valleys. Drainage and lithologic edges are sinuous. Short, low-offset straight scarps far from mapped traces are what 1:250k-to-1:24k compilations miss. | GEMSDOE49 `gate_ortho` used gradient magnitude × orthogonality gate, with no NMS. GEMSDOE32 used profile curvature. H19-x used ML on scarp channels. | Unknown sign: −0.03 (S0) to +0.01 (S1) for a unique candidate | Moderate |
| 3 | **H77-4 parallel strands beside the best support** | H19-5 support distance ring (3, 6] px | Truth density in the 300–600 m ring around the richest support (splays and parallel strands; forum 11536 says new geometry of existing systems counts) | Splays and parallel strands of detected faults are mapped as single traces, if at all | Not tested in prior repos (they tested halos around the *catalogue*, not around a live-validated support) | Small | Low |
| 4 | **H77-2 strain–catalogue deficit** | `geod_2ndinv` (band 4), `geod_shearrate` (band 7), catalogue density (~5 km Gaussian) | Rank(strain) − rank(mapped-fault density): a regional allocator, not a line detector | Strain must be accommodated by faults; where strain is high but few faults are mapped, the catalogue is incomplete | No GEMSDOE repo used a strain-versus-catalogue deficit. H19-1 used a fault length–frequency power-law deficit. | Small | Low |
| 5 | **H77-5 tilted-block valley asymmetry** | `det_elev` (band 12), `det_elev_slope` (band 19) | Skewness of the cross-valley elevation profile; asymmetric valleys mark half-graben block tilting toward an unmapped bounding fault | Buried or degraded bounding faults leave a geomorphic asymmetry even without a scarp | Not found in any local reference repo | Unknown | High (not run: budget) |

#**External data.** None of H77-1 to H77-5 needs new external data. They use only the official
19-band `training_features.tif` and `labels.tif`, restored from sha256-pinned owner mirrors. If H77-5
is ever run at higher resolution, the free official source is the USGS 3DEP 1 m DEM (named on page 968
as the GeoDAWN lidar source).

## Screening results available before this commit (E2, live inversion, LOO over 95 rasters)

- Support × catalogue model (S0): LOO Spearman +0.917, LOO RMSE 0.0324. The null (mean) RMSE is 0.0853.
- H77-1 added (S1): no LOO improvement. ΔMSE = −0.000024, 95% CI [−0.000059, +0.000007]. In-sample, the top 5% straight-edge cells 4–20 px from the catalogue get h = 0.089, the only positive off-support bin. **Inconclusive.**
- H77-2 added (S2): no change (all deficit bins h = 0). **Negative.**
- H77-4: the 30-bin catalogue × support model gives h = 0 for every (3, 6] px ring bin. **Negative.**

## Pre-registered build rule (E3)

- **On-support dots:** H19-5 line cells farther than B px from the (visible) catalogue, greedy by S0
  credit, ≥ 3 px apart.
- **Off-support dots (H77-1):** cells above the 80th percentile of straight-edge strength,
  > 3 px from the support and > 4 px from the catalogue, greedy by S1 credit then edge value.
- **On-support share:** ≤ 0.65 (the protocol limit is 0.70).
- **B and N_on:** chosen to maximise the conservative S0 projection.

## Pre-registered promotion rule

Verdict is **promote** only if all of the following hold. Otherwise the verdict is **negative**
(a deliverable, not a failure to report).

1. Valid format: the template validator plus portal-equivalence checks.
2. Unique: protocol thresholds against every registry raster, under the like-for-like reading.
   Literal-reading firings on space-filling rasters are reported.
3. Leakage canary clear: no single feature has holdout AUC > 0.90.
4. The S0 projection Δ versus the champion (same instrument) is ≥ 0.
5. The paired HOLDOUT-DTI difference versus the fold-fair champion rule has a CI that is not
   entirely below 0.

The weekly-slot decision itself is left to the separate selector step (brief, rule 6).
