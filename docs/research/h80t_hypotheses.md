# GEMSDOE77 H80 — Five Ranked Geological Hypotheses for the Next Lane (2026-10-21)

*Preregistered before any H80 experiment. Evidence classes: **[OFFICIAL]** organizer/USGS source · **[MEASURED]** computed from SHA-pinned bytes · **[OWNER-REPORT]** family ledger without organizer receipt · **[OBTAINABLE]** free official source checked as reachable.*

**Competition facts:** task = predict faults **not** in USGS/INGENIOUS catalogue; hidden faults hand-mapped by experts from same GeoDAWN/DEM/lidar evidence; metric = DTI α=0.2 β=0.8 triangular R=300 m = 3 px **[OFFICIAL]** (DrivenData #306 page 967). Public leaderboard #1 = 0.4413 xiaofanhu **[OFFICIAL]** 2026-10-21. Family best h33-2-b2 = 0.2778 **[OWNER-REPORT]** no organizer receipt.

**Band-6 correction [MEASURED]:** `training_features.tif` band 6 tag claims "Tilt angle" but Spearman vs external radiometric TC = 1.0000, vs derived tilt = 0.0200 → band 6 is radiometric TC, not tilt (evidence/band6_resolution.json). All tilt channels here are correctly derived from tmi_vg/tmi_hg and grav_vg/grav_thdr, not from band 6. Band 18 is signed directional derivative, not HGM; HGM recomputed from iso_grav_anom (band 13).

**Why the holdout/live distinction matters:** the file-to-score attribution of 0.2778 to H33 remains **OWNER-REPORT**; the same value in the archived public leaderboard row for `extradr19` does not identify a TIFF or hash. The named H33 raster's archived first-draw result is **0.004313 HOLDOUT-DTI** (`gems57-pooled-hide-v2`; 11,342 withheld positives; 95% CI `[0.003508, 0.005188]`). Separate fold-fair H19-5/H33-derived controls near 0.095 are not measurements of that named file and should not be conflated with it. The catalogue hide-and-recover proxy withholds mapped fault segments, so its target distribution differs from the organizer's hidden labels. Neither proxy outcome authenticates the file-to-score link or proves why the owner-reported live value occurred. New methods should be evaluated on the prescribed pooled blocked holdout and described as HOLDOUT-DTI, not as a live-score projection.

---

## H80-1 — Multi-Scale Tilt-Angle Persistence × AS-Weighted Zero Proximity + Signed Basement Throw (PRIMARY LANE)

- **Specific layers:** `tmi` (14) / `rtp` (2) as magnetic source, `tmi_vg` (9), `tmi_hg` (3 → HGM recomputed from `tmi` via THDR), `iso_grav_anom` (13), `iso_grav_anom_vg` (11), `depth_to_base_surf` (15). Derived transforms: magnetic tilt `atan2(vg, THDR)`, gravity tilt `atan2(vg, THDR)`, analytic signal `AS = sqrt(THDR²+VG²)`, THDR recomputed isotropically from `iso_grav_anom` (not band 18), all at σ=1.5 and σ=3.0 (two scales).
- **Physical signature:** *Tilt-angle phase* is zero directly over a buried contact and ±90° over source interior, independent of susceptibility/density contrast amplitude. A buried intra-basin basement fault under 50–300 m alluvial cover has weak amplitude but preserves tilt phase. **Persistence** = product of AS-weighted zero proximity at two scales: `Z = (1 - |tilt|/(π/2))` × `log1p(AS)`, normalized, then `Z_mag_persist = min(Z_mag_σ1.5, Z_mag_σ3.0)`. Same for gravity. Final field `F = rank01(Z_mag_persist × Z_grav_persist × max(0, basement_throw_alignment))` where throw alignment = `-∇depth · ∇grav / (|∇depth||∇grav|)` gated to downthrown side (deeper basement on hanging wall).
- **Why it catches catalogue-missing faults:** Expert mappers add hidden faults where GeoDAWN magnetics/gravity show buried basement offsets with no surface scarp. Amplitude-only HGM misses weak buried contacts (lithologic amplitude bias → pluton edges dominate). Phase-normalized tilt equalizes weak and strong contacts; multi-scale persistence suppresses shallow pedogenic/Quaternary noise that decorrelates between wavelengths, while true basement offsets persist.
- **Named non-fault mimic:** Lithologic contacts and caldera margins (tilt+AS without basement throw); GeoDAWN survey-block seams (straight, acquisition-parallel, no throw); remanently magnetized volcanics (tilt sign reversal).
- **How it differs from implemented lanes:** H77-1 used single-scale tilt zero coincidence × AS ridge × basement step (no persistence, no AS-weighting, no min-scale persistence). H77E used upward continuation persistence + Euler depth, not tilt phase. H77-Step used anisotropic step tensor, not phase-normalized tilt. No prior lane uses *dual-scale tilt persistence* (min across σ) multiplied by AS-weighted zero proximity. Verified by `grep -r "persistence.*tilt"` returning 0 in repo 2026-10-21.
- **Cost:** LOW — gradient + Gaussian filters on cached 19-band TIF (no FFT, no external data). Build <15 min CPU, <2 GB RAM. Validatable locally: YES via buffered_component_draw hide/recover.

## H80-2 — Geodetic Tensor Extensional-Shear Coupling (SECONDARY)

- **Layers:** `geod_shearrate` (7), `geod_dilaterate` (8), `geod_2ndinv` (4) (GNSS-derived strain tensor invariants, N100A15 smoothing), `depth_to_base_surf` (15), `det_elev` (12) as context.
- **Signature:** Decompose horizontal strain tensor into dilatational `Δ` and max shear `Γ`. Predicted normal-fault strike = azimuth of minimum horizontal strain (σ3) + 90° (Andersonian). Score = `dilat_pos × shear_norm × strike_alignment` where `shear_norm = |Γ|/(|Δ|+|Γ|)`, `strike_alignment = |cos(φ_fault - φ_predicted)|` where φ_fault from `∇depth` strike. High where active extension is localized and basement fabric aligns with Andersonian prediction.
- **Why catalogue-missing:** Walker Lane is actively extending at ~10 mm/yr; blind faults accommodating present strain may lack Quaternary scarp (buried by fan) and thus absent from Qfaults/INGENIOUS catalogue that keys to surface offsets, but visible to GNSS strain. Geodetic bands already in training_features but never used tensorially.
- **Mimic:** Anthropogenic subsidence bowls (also dilatational but with distinct 2nd invariant sign and no shear), seasonal hydrologic loading.
- **Difference:** Prior dilational intersection lane used `dilat_pos × basement_ridge` magnitude product only, no shear partitioning or tensor orientation. Grep `shearrate.*tensor` = 0.
- **Cost:** LOW-MEDIUM — tensor arithmetic on 19 bands, <10 min.

## H80-3 — Fluvial Knickpoint Alignment Network (TERTIARY, HIGH COST)

- **Layers:** `det_elev` (12), `det_elev_slope` (19), `iso_grav_anom_hg` (18) context, `depth_to_base_surf` (15) baselevel.
- **Signature:** D8 flow accumulation → channel network → longitudinal profile curvature → knickpoint extraction. Cluster knickpoints within 500 m elevation tolerance across adjacent catchments; lineament through aligned knickpoints at shared contour = fault trace. Inter-catchment alignment score = number of catchments sharing knickpoint elevation within 30 m / total catchments crossed.
- **Why catalogue-missing:** Blind fault under alluvium still offsets base level, producing knickzones in multiple drainages crossing fault at same structural contour but no continuous scarp to map; expert with lidar DEM could trace front-base asymmetry that catalogue mappers missed.
- **Mimic:** Resistant lithologic bed (also produces aligned knickpoints but follows stratigraphic contour, not basement structure; distinguished by lack of gravity/magnetic co-linearity), climatically driven terrace.
- **Difference:** Prior blind-saddle used pixel Hessian curvature, not inter-catchment network alignment. No prior lane extracts channel network. Verified via `grep "knickpoint\|flow_accum"` = 0.
- **Cost:** VERY HIGH — hydrological conditioning + D8 on 12M cells, >2 hr, >4 GB RAM. Not viable within 2-hr budget. Deferred.

## H80-4 — Shallow Micro-Seismicity Hough Lineament Density (EXTERNAL DATA REQUIRED)

- **Layers needed (specific free official source):** USGS ComCat FDSN event service `https://earthquake.usgs.gov/fdsnws/event/1/query` (free, public, no key; format=geojson, minmagnitude=1.0, maxdepth=15 km, starttime=1985-01-01, bounding box 37.0–42.5°N, -120.5–-117.0°E). Also NBMG fault database for blast filtering. Context layers: `iso_grav_anom_hg` (18), `tmi_hg` (3), `depth_to_base_surf` (15).
- **Signature:** Spatio-temporal de-clustered (Reasenberg) M≥1.0 events → kernel density along Hough lineaments (ρ,θ) → lineament persistence score = event density per km along strike × orientation coherence with potential-field fabric (cross-gradient alignment >0.6).
- **Why catalogue-missing:** Seismogenic blind faults active in last 40 yr produce micro-seismicity halos without surface rupture/scarp, absent from geomorphic catalogue but detectable seismically. Great Basin has diffuse seismicity; hidden faults inferred from INGENIOUS magnetotellurics + seismic.
- **Mimic:** Quarry blasts (daytime, shallow, clustered at known mine locations — filtered via USGS Mineral Resources Data System), aftershock swarms (temporal clustering), volcanic swarms.
- **Difference:** Shipped seismic bands (10,16) are 100 km-radius aggregates smoothed at N100A15, cannot localize fault at 300 m. No prior lane uses event-level catalog. Verified `grep -r "comcat\|fdsn"` = 0 in repo.
- **Obtainability check 2026-10-21:** Host `earthquake.usgs.gov` is outside sandbox egress allow-list (`github.com` only); `curl` returns 000 exit 35. No owner mirror of event catalog found (`grep -r "earthquake"` in registry = 0). Named but **BLOCKED here** — not viable until allow-list expanded or mirror cached. No synthetic catalog fabricated.

## H80-5 — Hydrothermal Alteration Halo Lag Pair (RADIOMETRIC-HYDRO)

- **Layers:** External `geodawn_rad_u8.tif` bands K (1), Th (2), U (3), extensions Th/K, U/K (via `geodawn_extensions_u8.tif`), `cond_surf` (17), `det_elev` (12), `depth_to_base_surf` (15). In-core proxies: `mag_anom` low as demagnetization alter indicator.
- **Signature:** Fault damage zone has K-depleted clay core (low K, low Th/K variance) flanked 300 m (3 px) downdip by U-enriched halo (high U/K). Score = `(K_low × ThK_stability)` at pixel ⊕ `U_high` at 3 px lag along `∇depth` direction (hanging wall), then confined to `det_elev < median` (basin lows where geothermal upflow ponds). Lag cross-correlation via directional shift, not same-pixel product.
- **Why catalogue-missing:** INGENIOUS geothermal focus: hidden faults are often geothermal conduits with clay alteration halos that have radiometric + conductivity signature but weak magnetic/topographic scarp; expert integrating magnetotellurics + geochemistry could map them while Qfaults omits.
- **Mimic:** Pedogenic carbonate vs volcanic cover change (also K/Th shift but not linear or fault-aligned; distinguished by lack of directional lag and topographic confinement), agricultural potash.
- **Difference:** H79-RH used simultaneous K–Th edge × U residual edge (no lag, no confinement, no K-depletion core). H77-C used conductivity gradient × magnetic low (no K/Th ratios). No prior lane uses *lagged* K-core/U-halo pair. Verified `grep "lag.*300m\|U.*halo"` limited to H79-RH simultaneous.
- **Cost:** MEDIUM — needs external rad u8 (available cached in owner mirror, ~60 MB, obtainable via ghost api within allow-list as github blob); processing ~20 min.

---

## Ranking by Expected DTI Improvement / Implementation Cost

| Rank | Hypothesis | Expected DTI Δ vs incumbent H77-Step (0.1527 HOLDOUT-DTI) | Cost | Expected Ratio | Validatable? | Verdict |
|------|------------|----------------------------------------------------------|------|----------------|--------------|---------|
| **1** | **H80-1 Tilt Persistence** | +0.02 to +0.04 (tilt phase equalizes weak buried contacts under cover; multi-scale persistence is textbook Blakely & Simpson 1986 for basement edge detection and is unrepresented in family) | LOW (15 min CPU, 2 GB, no external data) | **HIGH** | YES (hide/recover) | **VALIDATE FIRST** |
| 2 | H80-2 Tensor Coupling | +0.01 to +0.03 (GNSS strain is directly tied to active normal faulting but N100A15 smoothing is coarse) | LOW-MEDIUM (10 min) | MEDIUM-HIGH | YES | Next if H80-1 fails |
| 3 | H80-5 Alteration Halo Lag | +0.01 to +0.02 (prior H79-RH was 0.047 < random, so conservative; but lag+confinement is materially different and geothermal-specific) | MEDIUM (20 min + rad restore) | LOW-MEDIUM | YES (after rad restore) | Deferred |
| 4 | H80-3 Knickpoint Network | +0.02 to +0.05 (potentially highest — fluvial imprint is underused and directly sensitive to blind offsets) | VERY HIGH (>2 hr, >4 GB) | LOW | YES but exceeds budget | Deferred beyond budget |
| 5 | H80-4 Seismicity Lineament | +0.02 to +0.03 | HIGH + BLOCKED (host not in allow-list, needs blast filtering) | LOW | NO (blocked) | Named but not viable |

**Decision:** Validate H80-1 (tilt persistence) on spatially-blocked holdout before touching weekly slot. If pooled HOLDOUT-DTI beats current holdout best 0.152718 [0.131548,0.175185] with paired 95% CI excluding 0 vs both standard block and random, proceed to TIF build + uniqueness; else report negative and do not spend slot.

*Sources for manual review:*
- Tilt method: Blakely & Simpson 1986 “Approximating edges of source bodies from magnetic or gravity anomalies” (Geophysics 51, p.1494) — tilt = atan2(VG, THDR) locates edge at 0° contour; doi:10.1190/1.1442197 https://library.seg.org/doi/10.1190/1.1442197
- Analytic signal: Nabighian 1972 analytic signal energy peaks over contact; Roest et al. 1992 https://doi.org/10.1190/1.1443271
- DTI metric: DrivenData GEMS page 967 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ ; forum masking https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516
- Grid EPSG:32611 https://epsg.io/32611
- Strain bands: Kreemer et al. 2014 GNSS strain (source for N100A15 smoothing cited in training_features metadata)
