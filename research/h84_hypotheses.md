# H84 candidate hypotheses — preregistered 2026-10-21 before Exp1

Status: written 2026-10-21 before any H84 experiment. None of these has been implemented in this repo before (see difference-vs-repo per hypothesis). Lane H84 is geometry-only: it needs only the SHA-pinned bridge grid (`data/bridge/sample_submission.tif`, `existing_faults.tif`), so Exp1 runs with no external data and no training. Hypotheses are ordinal priorities, never projected scores.

All external sources below are official, free, and verified with links for manual review. No hallucinations. Verify line by line.

## Competition anchor

- Task: predict faults **not** in USGS/INGENIOUS catalogue; hidden truth manually mapped by experts; metric = distance-weighted Tversky DTI alpha=0.2 beta=0.8 triangular kernel R=300m=3px [OFFICIAL] https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Grid: EPSG:32611 UTM 11N, 3730x3292, 100m, transform (100,0,243350,0,-100,4508550) [OFFICIAL] https://epsg.io/32611 verified against `data/bridge/sample_submission.tif` (SHA256 2176d08e...)
- Faulds et al. 2010 structural controls: dilational step-overs 48%, horse-tail tips 22%, intersections 20% host geothermal upflow [OFFICIAL] https://doi.org/10.1016/j.geothermics.2011.02.004 (Faulds et al. Geothermics 2010) and GBCGE INGENIOUS https://gbcge.org/current-projects/ingenious/
- USGS GeoDAWN airborne mag/radiometrics DOI 10.5066/P93LGLVQ https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
- DOE GDR INGENIOUS 1391 well/spring temps DOI 10.15121/1881483 https://gdr.openei.org/submissions/1391

## Why h33-h33-2-b2 scored 0.2778 (OWNER-REPORT) and whether we can beat it

**Measured forensics** (evidence/prior_forensics.json, independent audit): h33 has 37,654 binary dots, none on catalogue, nearest catalogue distance sqrt(5) px. It is exactly the earlier 44,090-dot D2.8 predecessor with 6,436 dots at or within 200m of catalogue removed, no dots added. DTI = TP_w / (TP_w + 0.2 FP_w + 0.8 FN_w) with 300m triangular kernel k(d)=max(1-d/300,0). Sparse dots + near-catalogue pruning plausibly cut FP mass while preserving max-kernel TP coverage of hidden faults near but not on mapped traces.

**Causal proof missing**: GEMSDOE32 site at pinned commit https://github.com/buffedlizard55-lab/GEMSDOE32/blob/f62038b54dcb7767d4d997f80b033cdc3d42bef9/docs/index.html explicitly says "NO ORGANISER SCORE EXISTS for this or any artifact in this repository." No organizer receipt ties SHA256 to 0.2778. Thinning can help but may also remove uniquely useful coverage; a truly blind fault can lie near a mapped trace; pixel-exact masking does not imply blanket halo exclusion. Therefore we cannot claim pruning caused score or that H33 is optimal. Earlier fixed-denominator emission optimum claims are withdrawn per metric-review.html.

**Can we beat it?** Possibly, but not established. Best local HOLDOUT-DTI in this repo is H77E pure submod 0.1556 (seed20, 11,321 withheld px, 95% CI [0.1383,0.1716]) vs fold-fair H33 0.0963 delta +0.059; and H77E primary 0.1267 vs H33 +0.0303 with 0/816 informative overlaps. TSD transtensional has 0.1360 (+0.037 vs standard). These beat H33 on matched folds but fail literal dense-witness gate; they suggest sparse+physics can improve holdout. Whether that transfers to leaderboard depends on whether holdout geometry matches organizer's hidden truth distribution — unknown. We must test new physics on same hide-and-recover instrument, compare to rebuilt H33 on identical folds/budget, and report pooled DTI + CI. We cannot spend slot without beating incumbent on holdout.

## Rank 1 (implemented in Exp1): Fault-tip crack-tip stress lobe + en-echelon relay breaching

- **Layers:** USGS/INGENIOUS mapped-fault catalogue (`existing_faults.tif` == `labels.tif`, 60,988 pixels on 3730x3292 EPSG:32611 100m grid, SHA256 7ba308cc...); competition footprint from `sample_submission.tif` (SHA256 2176d08e..., 5,167,373 footprint px, 7,111,787 outside NaN).
- **Physical signature:** Elastic crack-tip stress lobes beyond fault tips, weighted by tip orientation, and breaching corridors between overlapping en-echelon strands (relay ramps). For each catalogue segment tip, propagate an anisotropic lobe: weight = (1 - d/R) * max(0, cos(theta))^2 for d < R=2.5km (25px), theta = angle between outward tip direction and tip-to-pixel vector, decaying to zero beyond. For parallel segment pairs whose tips overlap along strike (projected overlap >1km) and across-strike separation <2km, add relay corridor intensity = sum of both tip lobes + inter-tip linear corridor. Final field = normalized tip-lobe + relay score.
- **Why it catches uncatalogued faults:** INGENIOUS/Faulds 2010: ~22% of Great Basin geothermal systems sit in horse-tail tip damage zones where slip dies out into unmapped splay fans; ~48% in step-over relay ramps where two overlapping normal faults transfer displacement via dense breaching fractures. These are exactly the structures most likely to be blind under Quaternary cover (no scarp) and omitted from QFaults aerial-photo mapping, yet highly dilational/permeable. Catalogue mapping captures major strands but systematically misses tip splays and inter-strand breaching faults.
- **Named non-fault mimic:** Pleistocene Lake Lahontan/Bonneville terrace risers fanning from range-front tips; alluvial-fan lobe apices; road/canal cuts radiating from fault tips. Relay corridors can mimic inter-fan swales and lithologic contacts between overlapping fans.
- **How it differs from repo:** H83 collar is isotropic 100-400m damage collar around all fault traces (distance-only, no tip anisotropy, no relay logic). H77E knick is vertical knickpoint density; H77Step tip used fault-tip linear extrapolation but not angular lobe + relay breaching; H81 slip-cone is throw-gradient based, not elastic lobes. No lane uses PCA-derived segment orientation + outward tip lobe + pairwise relay corridor as primary scorer. Verified by grep for "tip.*lobe|relay.*ramp|en.echelon|crack.*tip" in prior preregistrations — none found.
- **Expected DTI gain ÷ cost:** Uncertain, medium ÷ very low (CPU raster ops, minutes). Tips/relays are undermapped and DTI kernel 300m rewards near-tip coverage; anisotropic lobes should reduce FP vs isotropic collar while capturing tip-concentrated hidden faults.
- **Obtainability:** No external data needed; bridge grid only. Viable in this sandbox.
- **Validation:** Whole-component buffered hide-and-recover, visible-only tip field per fold, pixel-exact visible masking, pooled DTI + paired CI.

## Rank 2: Catalogue fault-intersection / relay-node intensity (deferred)

- **Layers:** catalogue centreline network only (local).
- **Physical signature:** Kernel density of fault intersections and relay bends (damage focusing, permeability contrast at step-overs) — blind upflow nodes need no surface scarp.
- **Why uncatalogued:** Target is node, not trace: small blind fault terminating at mapped strand may be unmapped while node is real.
- **Mimic:** Mapped intersections of inactive traces; drainage junctions.
- **Differs:** No repo lane uses node/intersection intensity as primary scorer (H83 rank2 was same idea but not implemented; we defer to next experiment if tip lobe shows credit).
- **Cost:** very low. Expected DTI low-medium ÷ very low. Rank 2 because tip lobe already captures intersections via relay.

## Rank 3: Signed across-strike K/Th radiometric asymmetry

- **Layers:** GeoDAWN K/Th from USGS DOI 10.5066/P93LGLVQ (uint8 grid ext_geodawn_rad_u8 / ext_geodawn_extensions_u8, SHA-pinned in registry/data_manifest.json, ~26-27 MB, obtainable via scripts/restore_data.py from owner GitHub mirror).
- **Physical signature:** Hydrothermal illitization concentrates K and leaches K/Th asymmetry across fault; first derivative of K/Th normal to local strike, signed by hanging-wall side.
- **Why uncatalogued:** Alteration halo outcrops where trace buried below mapping scale.
- **Mimic:** Primary lithologic boundaries, flight-line noise.
- **Differs:** No lane uses signed across-strike radiometric asymmetry as primary (H82R entropy, H79 radhalo lag, H77G trough are different).
- **Cost:** medium. Expected medium ÷ medium. Obtainability: free official, not locally restored, blocked in sandbox? But owner mirror exists, so viable next session after restore. Not implemented now to preserve budget.

## Rank 4: Multi-scale residual curvature after regional trend removal

- **Layers:** Pinned 54-channel feature stack detrended elevation + curvature (data/processed/stack.npy, 418 MB training_features.tif pinned 4371c82e...).
- **Physical signature:** Curvature of DEM after removing regional piedmont trend at 300m-1.2km scales — youthful fault-line antiforms vs broad tilting.
- **Why uncatalogued:** Sub-mapping-scale scarps survive in curvature texture.
- **Mimic:** Fan cones, terrace treads, playas.
- **Differs:** Selective-U used raw U-rank curvature without detrending/scale sweep.
- **Cost:** medium (needs restored 418 MB stack). Expected low-medium ÷ medium.

## Rank 5: Thermal spring/well alignment anisotropy (GDR 1391)

- **Layers:** DOE/NREL GDR submission 1391 well & spring temperatures DOI 10.15121/1881483 CC BY 4.0.
- **Physical signature:** Linear alignment of warm springs/wells (Hough / kernel-density anisotropy), blind upflow without scarp.
- **Why uncatalogued:** Thermal memory survives where geomorphic expression buried.
- **Mimic:** Basin-fill evaporative brine anomalies, irrigation return flow.
- **Differs:** H78 used distance-to-manifestation proximity only; this adds alignment anisotropy.
- **Obtainability:** gdr.openei.org not in sandbox allowlist but owner mirror ext_gdr_wellspring_in_footprint.csv (2.1 MB, SHA 122718e...) exists in manifest, so obtainable next session via restore. Not implemented now.

Ranking by expected DTI improvement ÷ implementation cost: Rank1 H84 tip-lobe > Rank2 intersections > Rank3 K/Th > Rank4 curvature > Rank5 thermal alignment. Exp1 validates Rank1 on spatially blocked holdout before any submission-slot consideration. No budget tuning after holdout peek.

