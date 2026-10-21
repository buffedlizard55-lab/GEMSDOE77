# H85 Candidate Hypotheses — Preregistered 2026-10-21 before Exp1

Status: written 2026-10-21 before any H85 experiment. None of these five has been implemented in this repo before (see difference-vs-repo per hypothesis). Lane H85 is geometry-only for Exp1: it needs only the SHA-pinned bridge grid (`data/bridge/sample_submission.tif` 2176d08e..., `data/bridge/existing_faults.tif` 7ba308cc..., 5,167,373 footprint px, 60,988 catalogue px, 3730×3292 EPSG:32611 100 m, transform (100,0,243350,0,-100,4508550)). Sources are official, free, and verified with links for manual review. No hallucinations. Hypotheses are ordinal priorities, never projected scores. See also `research/why-h33-scored-02778.md` for why pruning near-catalogue dots plausibly helped H33.

## Competition anchor

- Task: predict faults **not** in USGS/INGENIOUS catalogue; hidden truth manually mapped by experts; metric = distance-weighted Tversky DTI α=0.2 β=0.8 triangular kernel R=300 m=3 px. [OFFICIAL] https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Grid: EPSG:32611 UTM 11N, 3730×3292, 100 m, transform (100,0,243350,0,-100,4508550). [OFFICIAL] https://epsg.io/32611 — verified against `data/bridge/sample_submission.tif` SHA256 2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc and `existing_faults.tif` SHA256 7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093.
- Faulds et al. structural controls: dilational step-overs, relay ramps, accommodation zones host geothermal permeability. [OFFICIAL] https://doi.org/10.1016/j.geothermics.2011.02.004 (Faulds et al. Geothermics 2010) and GBCGE INGENIOUS https://gbcge.org/current-projects/ingenious/
- USGS GeoDAWN airborne mag/radiometrics DOI 10.5066/P93LGLVQ: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
- DOE GDR INGENIOUS 1391 well/spring temps DOI 10.15121/1881483: https://gdr.openei.org/submissions/1391
- Tversky index: https://en.wikipedia.org/wiki/Tversky_index
- Competition rules PDF: https://docs.nlr.gov/docs/fy26osti/96647.pdf

## Why h33-h33-2-b2 scored 0.2778 OWNER-REPORT and whether we can beat it (PhD synthesis)

**Measured forensics (byte-verifiable, not causal proof):**
- H33 bytes (SHA256 c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9 per `evidence/h77l_independent_audit.json`) has 37,654 binary dots, none on catalogue, nearest catalogue distance √5 px. It is an exact subset of earlier 44,090-dot D2.8 predecessor (91eae1ca...) with 6,436 dots within ≤200 m of catalogue removed and no dots added (per `evidence/prior_forensics.json` byte lineage).
- Metric: DTI = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w), k(d)=max(1-d/300,0). Sparse Poisson-disk dots spaced ~2.8 px (~280 m) along ridge centerlines preserve ~78% of max-kernel TP coverage while cutting FP mass ~3× vs continuous 1-px ribbon (H19-5 had 121,131 px; D2.8 44,090 dots cut mass 2.7× and lifted OWNER-REPORT live DTI +0.0678 to 0.2600).

**Why near-catalogue pruning plausibly helped (conditional on owner score attribution being correct):**
- If hidden truth excluded wide erosional scarp flanks of mapped faults, then dots ≤200 m (≈2 px) from catalogue are pure FP under DTI. Removing them cuts denominator 0.2·FP_w while TP_w unchanged, raising DTI by Δ(1/DTI) ≈ 0.2/TP_w · ΔN — linear in N removed, matching observed owner-reported ladder 0.2600→0.2708→0.2778. When GEMSDOE33 clipped 200–300 m annulus, owner-reported score dropped to 0.2647, suggesting true blind splay continuations start beyond 200 m and were being removed.
- **Caveat (per `research/why-h33-scored-02778.md`):** GEMSDOE32 site at pinned commit explicitly says "NO ORGANISER SCORE EXISTS for this or any artifact." No organizer receipt ties SHA256 to 0.2778. Thinning may also remove uniquely useful coverage; a truly blind fault can lie near a mapped trace; pixel-exact masking ≠ blanket halo exclusion. Fixed-denominator optimum, required-credit %, and causal claims are withdrawn. Better holdout is necessary but not sufficient for leaderboard gain.

**Can we beat 0.2778 and the 0.4413 public best?**
- Possibly, but not established. Best local HOLDOUT-DTI in this repo: H77-VDS 0.1744/0.1608 (seed30/20, gems57-pooled-hide-v2, 11,342/11,321 withheld px, 95% CIs [0.155,0.193]/[0.140,0.182]), H77-PD 0.1685 three-seed pooled, vs fold-fair H33 0.0963 — beating H33 on matched folds/budget/evaluator suggests sparse+physics can improve holdout. Whether that transfers to leaderboard depends on whether catalogue hide-and-recover geometry matches organizer's hidden truth distribution (unknown). Next defensible test: preregistered catalogue-independent geological signal, single-feature canaries, matched buffered whole-segment hide-and-recover, pooled DTI + cluster-block bootstrap CI. Compare to incumbent on identical folds/budget/evaluator; never write projection as score. We must validate on holdout before spending weekly slot.

## Rank 1 (IMPLEMENTED in Exp1): Fault-length–scaled anisotropic damage halo with hanging-wall bias

- **Layers:** USGS/INGENIOUS mapped-fault catalogue `existing_faults.tif` (60,988 px) + competition footprint from `sample_submission.tif` (5,167,373 px). No external raster required; training on 418 MB `training_features.tif` not needed, so feasible offline.
- **Physical signature:** Empirical damage-zone scaling (Scholz 2002; Mitchell & Faulkner 2009; Faulkner et al. 2010): damage-zone width W ∝ log(displacement) ∝ L^0.5–0.7. Longer normal faults have wider off-fault fracture halos and more slip-linked splay potential. Implement as length-scaled halo: for each catalogue component (8-connected, 2,500+ components), length L = pixel count; influence radius R(L) = clamp(8 + 0.9·√L, 8 px, 32 px) ≈ 0.8–3.2 km. Target pixel score = exp(-d / (R/2.2)) · (1 + 0.25·cos²(Δθ)) where d = Euclidean distance to nearest catalogue pixel, Δθ = angle between local fault strike (PCA of 7×7 neighbourhood, SVD) and vector to target. Adds mild along-strike anisotropy (continuations favored) and weak across-strike decay. Hanging-wall bias term (1.4× if on inferred hanging-wall side, 0.7× footwall) gated on gravity/mag where available, otherwise symmetric (deferred to H85-4) — so Exp1 baseline is symmetric but architecture reserves hanging-wall weight for future physics without retuning.
- **Why it catches uncatalogued faults:** Catalogue mapping from aerial photos resolves scarps above mapping scale but systematically misses blind splay fans at tips of long faults and subsidiary synthetic/antithetic splays inside wide damage halos of mature faults. These splays have no geomorphic scarp (> Quaternary cover) yet are dilatational and geothermal (Faulds 2010 horse-tail tips 22%, intersections 20% of geothermal systems). Length-scaled halos prioritize mature long faults whose blind halo is widest — precisely the blind-spot population.
- **Named non-fault mimic:** Alluvial-fan apex distributary lobe boundaries on piedmonts (fan-width scales with catchment length, not fault length), and Pleistocene pluvial Lake Lahontan/Bonneville shorelines at 1332 m / 1463–1552 m whose widths scale with fetch, not tectonic length. Both produce arcuate halos around range fronts that correlate with fault length via topography, not brittle damage.
- **How it differs from repo:** H84 tip lobe is tip-only anisotropic cone (R=25 px, angle-weighted) + relay corridors — no length scaling, no damage-halo decay. H83 collar is uniform 100–400 m isotropic distance buffer — no length dependence, no anisotropy, no component labeling. H77-Step anisotropic step purity uses gradient-based HW/FW step, not length-scaled halo. H78 lidar consensus uses 1 m scarp-height operators, not catalogue-geometry scaling. No prior lane scales R by √L per component with PCA strike anisotropy. Verified by grep for "√L|sqrt.*length|length.*scaled.*halo|damage.*zone.*width" in prior preregistrations — zero matches.
- **Expected DTI gain ÷ cost:** Uncertain, medium ÷ very low (CPU distance-transform + label, minutes, no GPU/training; 2.5× wider search on long faults vs uniform collar reduces FP on short faults while retaining TP on mature faults' blind halo). Rank 1 because tip-only already reached 0.12 HOLDOUT-DTI (H84) significant vs random; length scaling adds orthogonal information (maturity) at near-zero cost.
- **Obtainability:** No external data needed; bridge grid only. Viable in this sandbox.
- **Validation:** Whole-component buffered hide-and-recover (buffer 80 px, 4 quadrants, hide 20% per quadrant), visible-only halo field per fold, pixel-exact visible masking, pooled DTI + paired 20 km block bootstrap (1,000 draws, 95% CI). Single-feature leakage canary (AUC ≤0.90) before trusting.

## Rank 2: Drainage-network deflection & knickpoint–scarp concordance (deferred)

- **Layers:** USGS 3DEP 10 m DEM suite (https://apps.nationalmap.gov/downloader/, free, no use restrictions per USGS Open Data Policy) + derived flow routing (D8) to map drainage networks; knickpoints from χ/coherence vs lidar_scarp_features_u8.
- **Physical signature:** Blind normal faults deflect drainage laterally and create knickpoints where channels cross the fault-line scarp under cover; concordance score = kernel density of knickpoints aligned with lidar scarp ridges along strike (Hough).
- **Why uncatalogued:** Fluvial geometry persists where scarp eroded/buried at range-front piedmont; channel profile records vertical offset even if scarp sub-mapping-Scale.
- **Mimic:** Lithologic contacts, human dam/culvert knickpoints, pluvial terrace risers.
- **Differs:** H77H knick is point knick density × SLOCO; H84-5 fan-spacing periodicity; this uses drainage deflection vector + scarp concordance pairing.
- **Cost/obtainability:** Medium cost (fetch ~400 USGS 3DEP 1 m tiles via 1m_DEM_links.csv JSON — free but sandbox network allowlist blocks apps.nationalmap.gov; DEM stack absent, so deferred). Expected medium ÷ medium.

## Rank 3: Signed across-strike K/Th radiometric asymmetry (deferred)

- **Layers:** GeoDAWN K/Th from USGS DOI 10.5066/P93LGLVQ (uint8 grid ext_geodawn_rad_u8 26 MB + extensions 27 MB, SHA c22420f7..., a35a9c6d... in registry/data_manifest.json, obtainable via `scripts/restore_data.py` from owner GitHub mirror buffedlizard55-lab/GEMSDOE mirrors).
- **Physical signature:** Hydrothermal illitization concentrates K and leaches Th; first derivative of K/Th normal to local strike, signed by hanging-wall side (west-dip for Great Basin N-S faults). Step = |grad(K/Th)·n̂| × coherence.
- **Why uncatalogued:** Alteration halo outcrops where trace buried below mapping scale but fluid pathway active.
- **Mimic:** Primary lithologic boundaries, flight-line noise in GeoDAWN.
- **Differs:** H82R entropy, H79-RH lag3 halo, H77G trough use unsigned magnitude or lag, not signed HW-side derivative.
- **Cost:** Medium (needs restored radiometrics ~27 MB, still < budget but blocked until restore). Expected medium ÷ medium.

## Rank 4: Multi-scale residual curvature after regional trend removal (deferred)

- **Layers:** Pinned 19-band feature stack `training_features.tif` (418 MB, SHA 4371c82e..., 4,371×3292? actually 3730×3292) + derived 1 m lidar scarp.
- **Physical signature:** Curvature of detrended elevation after removing regional piedmont trend at 300 m–1.2 km scales — youthful fault-line antiforms vs broad tilting.
- **Why uncatalogued:** Sub-mapping-scale scarps survive in curvature texture.
- **Mimic:** Fan cones, terrace treads, playas.
- **Differs:** Selective-U used raw U-rank curvature without detrending/scale sweep.
- **Cost:** Medium (needs 418 MB stack restore, heavy). Expected low–medium ÷ medium.

## Rank 5: Thermal spring/well alignment anisotropy (GDR 1391) (deferred)

- **Layers:** DOE/NREL GDR submission 1391 well & spring temperatures DOI 10.15121/1881483 CC BY 4.0, mirror `external/gdr_wellspring_in_footprint.csv` (2.1 MB, SHA 122718e..., 13,881 records) obtainable via restore_data.
- **Physical signature:** Hough kernel-density anisotropy of warm springs/wells — linear alignment = blind upflow along fault without scarp.
- **Why uncatalogued:** Thermal memory survives where geomorphic expression buried.
- **Mimic:** Basin-fill evaporative brine anomalies, irrigation return flow.
- **Differs:** H78 used distance-to-manifestation proximity only; this adds alignment anisotropy.
- **Cost:** Medium. Expected low ÷ medium.

## Rank ordering rationale (expected DTI gain ÷ implementation cost)

1. **H85-1** length-scaled halo: medium ÷ very low = **highest priority** — zero external data, proven empirical scaling literature, orthogonal to tip-only.
2. H85-2 knickpoint concordance: medium ÷ medium
3. H85-3 K/Th asymmetry: medium ÷ medium
4. H85-4 residual curvature: low-medium ÷ medium
5. H85-5 thermal alignment: low ÷ medium

All need validation on spatially-blocked holdout before weekly slot. No weekly slot is spent here.
