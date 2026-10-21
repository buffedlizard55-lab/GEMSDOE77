# H84 candidate hypotheses (generated 2026-10-21, this session; ranked by expected HOLDOUT-DTI gain ÷ cost)

All layers below are already restored/verified locally unless stated. No score is projected for any of them.

| Rank | ID | Layers | Physical signature | Why an *uncatalogued* fault, not a catalogued one | Non-fault mimic | Difference from repo lanes | Cost |
|---|---|---|---|---|---|---|---|
| 1 | H84-1 | Catalogue (visible only per fold) + GeoDAWN TMI horizontal gradient (training_features band 3) + isostatic gravity (band 13, gradient recomputed) | Kernel-scale (300 m-peaked, ≤1.2 km) off-trace collar **gated by coincident magnetic+gravity gradient rank** `sqrt(rank(magHG)·rank(gravHG))` — a second strand must have both a damage-zone position and a subsurface density/magnetisation step | Blind sub-parallel splays in a fault's damage zone sit 300–900 m off the mapped trace under Quaternary cover (never entered the Quaternary catalogue) but still offset basement, so potential fields step across them | Lithologic contacts parallel to range fronts; survey seams | H83 collar was geometry-only (0.1227 HOLDOUT-DTI s30); H77-Step used DEM step tensors + full HGB; no lane gates a collar by coincident dual potential-field gradients | Very low (CPU, restored 19-band stack) |
| 2 | H84-2 | Depth-to-basement surface (band 15) | Ridge of `|∇ depth_to_base|` (buried basement step) dotted alone, catalogue-masked | A buried fault with no surface scarp still throws the basement surface; catalogue faults are primarily surface-mapped | Sedimentary onlap / palaeo-channel margins | H77G used a mag-trough × gravity-step product inside an HGB; no lane emits the basement-step ridge alone | Low |
| 3 | H84-3 | Geodetic dilatation/shear (bands 8, 7) + 2nd invariant (band 4) | Local maxima of dilatation rate coincident with shear — actively straining, creeping blind faults | Creeping/active blind faults strain the surface before a mappable scarp forms | Aquifer compaction, landslide strain | No lane uses the geodetic strain bands as the primary emitter (they were HGB inputs only) | Low |
| 4 | H84-4 | Conductivity (band 17) + depth (band 15) | Conductive lineament (clay/fluid-filled fault gouge) where basement is shallow | Fluid-filled fracture zones are conductive even where cover hides the trace | Palaeo-lake clay flats, saline playas | H77G/H82 used MT/clay-cap margins as a gate inside ensembles, never as primary dotted emitter | Low |
| 5 | H84-5 | DEM (band 12) drainage asymmetry | Signed valley-floor asymmetry reversals paired across adjacent catchments | Distributed warping from blind faults tilts catchments without a discrete scarp | Stream capture, fan avulsion, lithologic erosion asymmetry | Requires a conditioned drainage graph not built in this repo; H77E used knickpoints, not paired asymmetry | High — deferred |

**Implemented this session: H84-1 only (Experiment 1).** Promotion rule (standing): a candidate may be marked
promote only if its pooled HOLDOUT-DTI (gems57-pooled-hide-v2, seed 30) beats the repository incumbent
0.152718 with a paired CI, every single-feature canary AUC ≤ 0.90, the final dots pass the local
uniqueness check vs accessible registry rasters (rank-corr ≤ 0.90, ≤70% within 3 px), and the writer
validator passes. Otherwise verdict = negative, file is inspection-only, 0 weekly slots spent.
