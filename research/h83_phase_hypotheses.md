# H83-PHASE Candidate Geological Hypotheses — preregistered 2026-10-21

**Status:** pre-registration for lane H83-PHASE (arena/9a46b655-gemsdoe77), not a claim of fault or leaderboard gain. Current retained incumbent is 0.152718 HOLDOUT-DTI (gems57-pooled-hide-v2, 11,321 withheld positives, 95% CI [0.131548,0.175185], Seed 20, H77-Step). A DrivenData organizer score is different evidence class. Zero weekly slots used by this work.

## Why H33 scored 0.2778 and how to beat 0.4413 — PhD forensics

**H33 anatomy (byte-verified):** h33-h33-2-b2 (SHA c55bafc4, 37,654 dots) is the 44,090-dot Poisson-scattered H19-5 ridge thinned by removing 6,436 dots within 200 m of mapped catalogue. Per metric DTI = TP_w / (TP_w +0.2 FP_w +0.8 FN_w) with 300 m triangular kernel k(d)=max(1-d/300m,0), sparse 300 m Poisson spacing keeps 77.8% of TP_w while cutting FP_w mass 3×, and collar pruning deletes mass whose nearest truth is already-mapped faults (FP_w without TP_w).

**Why it can't reach 0.4413:** H33 is emission-optimal on a 0.19-class field. Re-thinning further would discard TP_w (proved by GEMSDOE33 clipping 200-300 m losing -0.0131). To reach 0.4413 requires a field that puts mass where hidden faults actually are — detector improvement, not just packing.

**Path to beat:** Prior pure-ML plateaued at 0.2478 (GEMSDOE70) due to SGMC bedrock contamination, self-distance tautology, isotropic curvature mimics (canyons, lake shorelines), and masked-collar ringing. H83-PHASE targets the blind-basin fault blind spot: buried normal faults under alluvial cover with no scarp (det_elev weak) but persistent basement offset detectable as magnetic phase discontinuities.

## Candidate matrix (ranked by expected DTI gain / cost)

| Rank | Candidate layers | Physical signature (transform) | Why off-catalogue; difference vs repo | Non-fault mimic | Cost |
|---|---|---|---|---|---|
| **1** | **H83-PHASE: magnetic phase-congruency buried contacts** — mag_anom (1), rtp (2), tmi_hg (3), tmi_vg (9), tmi (14), tc (6) | Multi-scale (150/300/600 m) Gaussian-derivative gradient phase congruency: amplitude-independent edge strength PC = \|sum exp(jθ_scale)\|/n × amp-weighted persistence, plus structure-tensor orientation coherence at 300 m. Cross-band rank-average. | Buried basement offsets under fill have weak amplitude but stable phase across wavelengths — amplitude-threshold detectors miss them, phase-congruency finds them. Differs from H80T tilt-persistence (phase of Hilbert-transform tilt), H77G trough strength, H79-orient orientation-concentration, and H82rrd RTP/TMI regression breaks. No prior code implements log-Gabor/gradient-phase congruency. | Intrusive/sedimentary magnetic contacts, flight-line levelling seams, cultural noise | Medium; uses cached 19-band stack, no new external data. CPU-only HGB feasible. |
| 2 | Geodetic anisotropic dilatation + shear alignment — geod_2ndinv (4), shearrate (7), dilaterate (8), iso_grav_anom (13), det_elev (12) | Tensor dot-product of geodetic strain eigenvectors with gravity-gradient and topographic lineament orientations; dilatant quadrant >0 persistence. | Active blind accommodation zones show extensional strain even without scarp; aligned strain+gravity corridor indicates hidden fault. Differs from H78 slope alone and H19 openness. | Landslide/subsidence strain, hydrologic deformation | Medium-high; needs careful tensor handling. |
| 3 | MT clay-cap breach via conductivity + well thermal lineaments — cond_surf (17), GDR wells (1391), rtp (2) | Conductivity ridge breach (high cond along fault-perpendicular profile interrupted) aligned with anisotropic thermal excess after regional conductive depth-detrending. | Hydrothermal conduit breaches clay cap leaving conductivity gap + thermal plume; surface catalogue misses sealed/buried conduits. Differs from H77V vent advection and H77 hydrothermal ensemble (point proximity vs breach geometry). | Saline basin fill, clay-rich playa, well-selection bias | Medium; requires GDR well depth/metadata check (source https://gdr.openei.org/submissions/1391 verified accessible). |
| 4 | Basement hinge + gravity trough odd/even asymmetry — depth_to_base_surf (15), iso_grav_anom (13), iso_grav_vg (11) | Signed odd (step) vs even (ridge) derivative ratio along magnetic gradient axes; cross-scale sign stability. | Fault offsets basement thickness producing one-sided depth step with coincident gravity low; symmetric folds/anticlines lack signed throw. Differs from H77G trough and H79-depth single-scale step. | Paleochannels, pluton edges, caldera margins | Low-medium; reuses depth model. |
| 5 | Seismic density lineation transfer — ieq_n100a15 (16), deq_n100a15 (10), tmi_hg (3) | Epicentre density structure-tensor lineation coherence transferred to magnetic edge field via cross-correlation. | Distributed swarms along blind faults show linear seismic fabric even when individual events sparse. | Aseismic swarms, catalogue completeness gradients, quarry blasts | High; seismic bands sparse, ComCat FDSN outside sandbox allowlist (verified blocked). Not viable now. |

## Selection

Build **H83-PHASE rank-1** from cached stack with no labels/catalogue/prior rasters in feature construction. Validate via leakage canary (AUC ≤0.90), then paired whole-segment hide-and-recover vs 19-band control. Emit 37,654 dots at 2.8 px min-sep, >2 px from catalogue (greedy Poisson-disk, no budget tuning after holdout). Require beating 0.152718 incumbent on Seed 20 for promotion eligibility. Report literal and like-for-like uniqueness (dense vs dot_field classification per vendor/gems57/uniqueness.py DENSE_SUPPORT_FRACTION=0.20).

## Verified sources

- [OFFICIAL] Problem page 967: DTI metric, EPSG:32611, 100 m, float32 [0,1], 3730×3292, transform (100,0,243350,0,-100,4508550): https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Leaderboard #1 0.4413 ORGANIZER-CONFIRMED (xiaofanhu): https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
- GeoDAWN magnetic/radiometric DOI 10.5066/P93LGLVQ: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
- GDR INGENIOUS 1391 CC BY 4.0: https://gdr.openei.org/submissions/1391
- Reference solution CPU/GPU: https://github.com/drivendataorg/gems-prize-reference-solution
- Rules PDF §§3.2/3.4: https://docs.nlr.gov/docs/fy26osti/96647.pdf
- Tversky index: https://en.wikipedia.org/wiki/Tversky_index
- Faulds et al. 2010 structural controls (Geothermics): https://doi.org/10.1016/j.geothermics.2011.02.004 — step-overs/relay ramps localize permeability under cover.
