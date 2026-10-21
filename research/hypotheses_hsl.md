# Preregistered Geological Hypotheses — Lane `H77-HSL` (Hydrothermal–Scarp–Lineament & Visible-Fault Continuation)

**Preregistration ID:** `GEMSDOE77-PREREG-HSL-1` (`registry/preregistration_hsl.json`, registered `2026-10-21T04:10:00Z` before running any experiment in this lane)  
**Session branch:** `arena/2dd85e1d-gemsdoe77`  
**Evaluator pin:** `vendor/gems57/evaluate_holdout.py` (`gems57-pooled-hide-v2`, $\alpha=0.2$, $\beta=0.8$, $R=300\text{ m}$ triangular kernel, visible faults masked pixel-exactly, whole connected fault components withheld with a 3-px buffer)

---

## 1. Execution of Previous Sessions' Next Steps

Per the Parallel-Run Protocol (`Work on the next steps from the previous sessions first`), this lane directly executes the concrete next steps left open by all three merged lanes in `GEMSDOE77`:

1. **From the Contact-Physics Lane (PR #5, `docs/contactphys.html` §6 & `README.md`)**:
   - *Next step (a)*: Promote the **1 m USGS 3DEP lidar scarp block** (`external/lidar_scarp_features_u8.tif`) from an unpromoted 5-band ablation (`A5_primary_lidar`, which reached `0.1130` vs `A4_primary = 0.1090` and `A2_standard = 0.1043` on seed 30) into a primary preregistered candidate, and expand it to **all 12 lidar bands** plus physically derived crest–toe curvature dipole, unilateral facing polarity, and multi-cell sub-pixel spin-2 strike director continuity.
   - *Next step (b)*: Test the candidate on both the two-draw whole-component holdout (`seeds 30 and 21`, `22,914` withheld positive pixels) and the secondary **off-catalogue SGMC structural lineament proxy** (`SGMC \ catalogue`, $d > 3\text{ px}$, `62,122` proxy pixels).
2. **From the LIA Lane (PR #6, `reports/why_02778.md` §7 & `README.md`)**:
   - *Next step*: Search for an off-support physical field combining **1 m lidar scarp mapping, multi-physics potential-field gradients, and hydrothermal/geothermal indicators** that earns genuine credit away from the saturated H19-5 support while keeping 3-px dot overlap $\le 0.70$ against all `765` informative registry rasters.
3. **From the Selective-U Curvature Lane (PR #3, `docs/irregularities.md` & `README.md`)**:
   - *Next step*: Evaluate every candidate against **fold-recomputed, visible-fault-only incumbents** (`C1_champion_vis`, `C2_standard`, `C3_contactphys`) on identical spatial validation folds with paired 20 km physical-block bootstrap confidence intervals, never scoring a static full-catalogue-pruned TIFF on a holdout split.
4. **From the Standing Prompt Geothermal Research Directive**:
   - *Directive*: Conduct deep scientific research into **geothermal vents, blind hydrothermal upflow zones, and structural permeability pathways in the Great Basin** using official verified sources (`INGENIOUS GDR 1391` well/spring temperatures and chemical geothermometers `gdr_wellspring_in_footprint.csv`, `gdr_volcanic_vents_in_footprint.csv`, and `geodawn_extensions_u8.tif`), and test their contribution on the spatially-blocked holdout.

---

## 2. Five Ranked Untried Geological Hypotheses

Each hypothesis below names:
- **Specific layer(s) involved**,
- **Physical signature targeted (mathematical transform)**,
- **Why it catches a fault missing from the USGS/INGENIOUS catalogue rather than one already in it**,
- **Named non-fault geological/anthropogenic mimic**, and
- **How it differs from everything already implemented in `GEMSDOE77`**.

### Summary Ranking Table

| Rank | ID | Hypothesis | Specific Layers | Expected HOLDOUT-DTI Lift | Implementation Cost | Status |
|---:|---|---|---|---:|---|---|
| **1** | **H77-HSL-1** | **1 m USGS 3DEP Lidar Scarp Crest–Toe Curvature Dipole, Unilateral Facing Polarity & Sub-Pixel Spin-2 Strike Continuity** | `lidar_scarp_features_u8.tif` bands 1–12 (`ex_max`, `ex_mean`, `step_max`, `lapneg_max`, `lappos_max`, `downface_max`, `upface_max`, `cross_max`, `relief`, `coh100`, `strike`, `valid`) + `det_elev` (B12), `det_elev_slope` (B19) | **+0.008 to +0.016** | Low (in-repo verified external raster) | **Tested in Exp 1–3 (`HSL_nolabel`, `HSL_structural`)** |
| **2** | **H77-HSL-2** | **Shallow Vertical Magnetic Continuation Residual ($\text{TMI} - \text{TMI}_{\uparrow 150\text{ m}}$) & Hydrothermal Demagnetization / Smectite Clay-Cap Conduit** | `geodawn_extensions_u8.tif` bands 3 (`UTh`) & 4 (`TMI_up150`) + `training_features.tif` bands 2 (`rtp`), 3 (`tmi_hg`), 9 (`tmi_vg`), 14 (`tmi`), 17 (`cond_surf`) | **+0.003 to +0.007** | Low (in-repo verified external raster) | **Tested in Exp 1–3 (`HSL_nolabel`, `HSL_structural`)** |
| **3** | **H77-HSL-3** | **INGENIOUS GDR 1391 Geothermal Outflow (Hot/Warm Springs, Quartz/Chalcedony/Na-K-Ca Geothermometers) & Quaternary Volcanic Vent Alignment** | `gdr_wellspring_in_footprint.csv` (27,092 records; `dist_known_fault_px` strictly excluded) + `gdr_volcanic_vents_in_footprint.csv` (21 vents; `dist_known_fault_px` strictly excluded) | **+0.002 to +0.005** | Low (in-repo verified CSVs) | **Tested in Exp 1–3 (`HSL_nolabel`, `HSL_structural`)** |
| **4** | **H77-HSL-4** | **Visible-Fault-Only Multi-Scale Damage-Zone Density, Stepover Relay Deficit, and 1 m Lidar-to-Host Strike Alignment + Submodular 300 m Kernel Emission** | `fold["visible"]` fault mask (buffered $>3\text{ px}$ from withheld truth) + Band 11 (`lidar strike`) | **+0.025 to +0.045** | Medium (fold-recomputed EDT, local strike orientation & submodular 300 m kernel cover) | **Tested in Exp 1–3 (`HSL_structural`, `HSL_structural_submod`)** |
| **5** | **H77-HSL-5** | **Relocated Sub-km Microseismicity Lineaments (USGS ComCat / UNR NSL / 2020 $M_w 6.5$ Monte Cristo Range Rupture)** | External free official source: `https://earthquake.usgs.gov/fdsnws/event/1/query` (USGS ComCat API) | **+0.005 to +0.012** | Medium (FDSN query + anisotropic kernel density) | **DATA-BLOCKED** (sandbox egress allowlist blocks `earthquake.usgs.gov`; shipped B10/B16 are 100 km smoothed grids) |

---

### Detailed Hypothesis Specifications

#### Rank 1 — `H77-HSL-1`: 1 m USGS 3DEP Lidar Scarp Crest–Toe Curvature Dipole, Unilateral Facing Polarity, and Sub-Pixel Spin-2 Strike Director Continuity
- **Specific layers**: All 12 bands of `external/lidar_scarp_features_u8.tif` derived from USGS 3DEP 1 m bare-earth lidar (`(1) ex_max`, `(2) ex_mean`, `(3) step_max`, `(4) lapneg_max`, `(5) lappos_max`, `(6) downface_max`, `(7) upface_max`, `(8) cross_max`, `(9) relief`, `(10) coh100`, `(11) strike`, `(12) valid`) paired with 100 m `det_elev` (Band 12) and `det_elev_slope` (Band 19).
- **Physical signature targeted**:
  1. **Crest–Toe Curvature Dipole**: A degraded normal or strike-slip fault scarp in unconsolidated alluvium satisfies the morphological diffusion equation $\partial z/\partial t = \kappa \partial^2 z/\partial x^2$ ([Hanks et al., 1984, *JGR*](https://doi.org/10.1029/JB089iB07p05771)), producing a tightly paired **convex scarp crest** ($\nabla^2 z < 0$, captured by `lapneg_max`, Band 4) and **concave scarp toe** ($\nabla^2 z > 0$, captured by `lappos_max`, Band 5) across a cross-strike elevation step (`cross_max`, Band 8). We compute:
     $$D_{\text{scarp}} = \sqrt{\text{lapneg\_max} \cdot \text{lappos\_max}} \cdot \text{cross\_max}$$
  2. **Unilateral Facing Polarity**: Fault scarps drop elevation consistently in one facing direction (synthetic or antithetic to the range front), whereas erosional interfluves and stream channels have symmetric opposing slopes (`downface_max` $\approx$ `upface_max`). We compute:
     $$P_{\text{face}} = \frac{|\text{downface\_max} - \text{upface\_max}|}{\text{downface\_max} + \text{upface\_max} + 10^{-4}} \cdot \text{step\_max}$$
  3. **Multi-Cell Sub-Pixel Spin-2 Strike Director Continuity**: Band 11 (`strike`) encodes the sub-pixel 1 m scarp azimuth $\theta \in [0, \pi)$ inside each 100 m cell. Because fault strike is an axial (headless) director ($\theta \equiv \theta + \pi$), we project onto the spin-2 complex director $(u_2, v_2) = (c \cos 2\theta, c \sin 2\theta)$ weighted by sub-pixel coherence $c = \text{coh100} \cdot \text{valid}$, and compute multi-cell director coherence at $\sigma \in \{2\text{ px (200 m)}, 5\text{ px (500 m)}\}$:
     $$C_{\text{strike}}(\sigma) = \sqrt{(G_\sigma * u_2)^2 + (G_\sigma * v_2)^2}$$
- **Why it catches faults missing from the USGS/INGENIOUS catalogue**: Pre-lidar Quaternary fault compilations (1:250,000 to 1:24,000 low-sun-angle aerial photography) systematically missed (i) low-throw ($<1\text{–}2\text{ m}$) distributed piedmont and intra-basin scarps, (ii) antithetic graben-bounding scarps, and (iii) scarps breached every 200–400 m by alluvial-fan washes. 1 m bare-earth lidar resolves both the crest–toe Laplacian dipole inside surviving interfluve remnants and the collinear sub-pixel strike across breached washes.
- **Named non-fault mimic**: Pleistocene pluvial Lake Lahontan wave-cut shoreline terraces (constant elevation contours ringing basin margins), fluvial terrace risers along the Humboldt/Carson/Walker rivers, and highway/railroad embankments.
- **How it differs from existing repo implementations**: PR #5 (`contactphys`) excluded lidar from its primary model `A4_primary` and only tested 5 raw uint8 bands (`ex_max`, `step_max`, `downface_max`, `coh100`, `valid`) in an unpromoted ablation (`A5_primary_lidar`). Bands 2, 4, 5, 7, 8, 9, 11 and all dipole, facing-polarity, and spin-2 director transforms have never been computed or evaluated in `GEMSDOE77`.

#### Rank 2 — `H77-HSL-2`: Shallow Vertical Magnetic Continuation Residual ($\text{TMI} - \text{TMI}_{\uparrow 150\text{ m}}$) & Hydrothermal Demagnetization Conduit
- **Specific layers**: `external/geodawn_extensions_u8.tif` Band 3 (`UTh`) and Band 4 (`TMI_up150`), combined with `training_features.tif` Band 2 (`rtp`), Band 3 (`tmi_hg`), Band 9 (`tmi_vg`), Band 14 (`tmi`), and Band 17 (`cond_surf`).
- **Physical signature targeted**:
  1. **Shallow Vertical Magnetic Continuation Residual**: Upward continuation of Total Magnetic Intensity by $\Delta z = 150\text{ m}$ (`TMI_up150`, Band 4 of `geodawn_extensions_u8.tif`) attenuates horizontal wavenumbers $k$ by $e^{-k \Delta z}$ ([Blakely, 1995](https://doi.org/10.1017/CBO9780511549816)). The rank-normalized residual:
     $$R_{\text{mag,150}} = \text{rank}_{[0,1]}(\text{tmi}) - \text{rank}_{[0,1]}(\text{TMI\_up150}), \qquad G_{\text{mag,150}} = \log(1 + \|\nabla R_{\text{mag,150}}\|)$$
     acts as a vertical-dipole bandpass filter isolating shallow ($<150\text{–}300\text{ m}$ depth) faulted Miocene–Quaternary basalts and welded tuffs buried just beneath young basin fill while canceling deep plutonic basement highs.
  2. **Hydrothermal Magnetite-Destruction & Smectite Clay-Cap Conduit**: Hydrothermal fluids circulating along permeable fault zones alter primary titanomagnetite to non-magnetic pyrite/leucoxene/hematite (creating a localized magnetic analytic-signal trough) while argillic alteration precipitates electrically conductive smectite/illite clays (`cond_surf`, Band 17) and mobilizes uranium relative to thorium (`UTh`, Band 3 of `geodawn_extensions_u8.tif`). We compute:
     $$I_{\text{demag}} = \text{rank}_{[0,1]}(\text{cond\_surf}) \cdot \bigl(1 - \text{rank}_{[0,1]}(\text{AS}_{\text{mag}})\bigr) \cdot \text{rank}_{[0,1]}(\|\nabla \text{cond\_surf}\|)$$
- **Why it catches faults missing from the USGS/INGENIOUS catalogue**: Blind intra-basin faults covered by Holocene playa silt or active alluvial fans have zero topographic expression, so airphoto compilers could not map them. However, shallow buried volcanic layers offset across these faults and hydrothermal alteration corridors along their damage zones produce sharp shallow magnetic continuation residuals and conductive demagnetization troughs.
- **Named non-fault mimic**: Saline playa evaporite pans (high near-surface electrical conductivity and low magnetic gradient without fault throw) and stratabound hydrothermally altered ash-flow tuffs.
- **How it differs from existing repo implementations**: No lane in `GEMSDOE77` read Band 3 (`UTh`) or Band 4 (`TMI_up150`) of `geodawn_extensions_u8.tif`.

#### Rank 3 — `H77-HSL-3`: INGENIOUS GDR 1391 Geothermal Outflow (Thermal Springs, Quartz/Chalcedony/Cation Geothermometers) & Quaternary Volcanic Vent Alignment
- **Specific layers**: `external/gdr_wellspring_in_footprint.csv` (`27,092` Great Basin well and spring records from [GDR Submission 1391](https://gdr.openei.org/submissions/1391), using ONLY `row`, `col`, `thermalclass`, `temp_c`, `geothermquartz_c`, `geothermchalc_c`, `geothermcat_c` and **strictly excluding `dist_known_fault_px`**) and `external/gdr_volcanic_vents_in_footprint.csv` (`21` Quaternary basalt/rhyolite volcanic vents, using ONLY `row`, `col` and **strictly excluding `dist_known_fault_px`**).
- **Physical signature targeted**:
  1. Log-Euclidean distance $-\log(1 + d_{\text{thermal}})$ to the `4,239` `Hot` ($\ge 50^\circ\text{C}$, `1,994` records) and `Warm` ($20\text{–}50^\circ\text{C}$, `2,245` records) springs/wells or records with measured $T \ge 35^\circ\text{C}$.
  2. Multi-scale Gaussian thermal anomaly fields $G_\sigma * \max(T - 20^\circ\text{C}, 0)$ at $\sigma \in \{4\text{ px (400 m)}, 12\text{ px (1.2 km)}\}$.
  3. Chemical geothermometer anomaly field $G_{\sigma=8} * \max(T_{\text{chem}} - 70^\circ\text{C}, 0)$, where $T_{\text{chem}} = \max(T_{\text{quartz}}, T_{\text{chalcedony}}, T_{\text{Na-K-Ca}})$ captures deep reservoir equilibration temperatures even where rising fluids cool conductively before reaching the surface.
  4. Log-Euclidean distance $-\log(1 + d_{\text{vent}})$ to Quaternary volcanic vents (`21` vents inside the footprint).
- **Why it catches faults missing from the USGS/INGENIOUS catalogue**: As documented by [Faulds et al. (2011)](https://pubs.geoscienceworld.org/gsw/geosphere/article/7/3/675/132459) and [Coolbaugh et al. (2006)](https://www.unr.edu/nbmg), $>70\%$ of Great Basin geothermal systems are **blind** (lacking mapped surface fault scarps directly at the thermal anomaly because travertine/sinter mounds or young playa sediments conceal the trace), yet deep fluid upflow is strictly controlled by high-dilation structural settings: fault-tip horse-tail splays, releasing stepovers/relay ramps, and synthetic/antithetic fault intersections.
- **Named non-fault mimic**: Lateral thermal-water outflow plumes migrating 1–5 km down-gradient through permeable alluvial fan gravels away from the range-front conduit.
- **How it differs from existing repo implementations**: No previous lane in `GEMSDOE77` loaded `gdr_wellspring_in_footprint.csv` or `gdr_volcanic_vents_in_footprint.csv`.

#### Rank 4 — `H77-HSL-4`: Visible-Fault-Only Multi-Scale Damage-Zone Density, Stepover Relay Deficit, and 1 m Lidar-to-Host Strike Alignment + Submodular 300 m Kernel Emission
- **Specific layers**: The fold-specific `visible` fault mask (`fold["visible"]` in holdout evaluation, `catalogue` in the final full-catalogue submission fit — **never** touching withheld faults and strictly masked within $\le 3\text{ px}$ during evaluation) combined with Band 11 (`lidar strike`).
- **Physical signature targeted**:
  1. `vis_logdist` $= \log(1 + d_{\text{vis}})$ (for $d_{\text{vis}} > 3\text{ px}$),
  2. Multi-scale Gaussian visible-fault densities $D_{\text{vis}}(\sigma) = G_\sigma * \mathbb{I}_{\text{vis}}$ at $\sigma \in \{5, 15, 35\}\text{ px}$ ($0.5, 1.5, 3.5\text{ km}$),
  3. **Stepover Relay Deficit**:
     $$R_{\text{relay}} = \frac{D_{\text{vis}}(15)}{D_{\text{vis}}(5) + 0.01}$$
     which peaks inside unmapped stepover gaps and relay ramps where a 1.5 km-wide fault belt has high regional density (`D_vis(15)` high) but the local 500 m trace has an unmapped gap (`D_vis(5)` low),
  4. **Lidar-to-Host Strike Alignment**: Using the nearest visible fault's local structure-tensor strike $\theta_{\text{vis}}(\text{nearest}(x))$, we compute the spin-2 axial alignment with the 1 m lidar sub-pixel scarp strike $\theta_{\text{lidar}}(x)$:
     $$A_{\text{strike}}(x) = c_{\text{lidar}}(x) \cos\bigl(2(\theta_{\text{lidar}}(x) - \theta_{\text{vis}}(\text{nearest}(x)))\bigr)$$
  5. **Submodular 300 m Triangular-Kernel Marginal-Gain Emission**: Instead of greedy hard-disk blocking alone, dots are selected sequentially to maximize exact marginal expected DTI kernel coverage $\Delta \widetilde{\text{TP}}(x \mid M) = \sum_{\|y-x\|<3} p(y) \max(0, k(\|y-x\|) - M(y))$ where $k(r) = \max(1 - r/3, 0)$ and $M(y)$ tracks the current kernel envelope of already-placed dots.
- **Why it catches faults missing from the USGS/INGENIOUS catalogue**: Per Organizer Forum clarification (`thread 11536`), `"new fault"` in the test labels includes **newly mapped geometry of existing faults** (unmapped tip propagations, breach connectors across alluvial washes, relay ramps, and parallel synthetic/antithetic splays). Because brittle fault displacement cannot terminate instantaneously in elastic crust ($dD/dx \lesssim 10^{-2}$–$10^{-1}$; [Cowie & Scholz, 1992](https://doi.org/10.1016/0191-8141(92)90065-5)), unmapped fault traces cluster heavily in relay deficits and along strikes collinear with host fault tips.
- **Named non-fault mimic**: Piedmont contour lines and parallel distributor washes flanking a mapped range-front fault.
- **How it differs from existing repo implementations**: PR #3 (`urank`) was an unsupervised algebraic combination; PR #5 (`contactphys`) used zero visible-fault context features; PR #6 (`lia`) was an algebraic blend of H19-5 and H33 without supervised multi-physics learning or 1 m lidar strike alignment.

#### Rank 5 — `H77-HSL-5` (Documented `DATA-BLOCKED`): Relocated Sub-km Microseismicity Lineaments
- **Specific layers**: External free official source: USGS ComCat FDSN API (`https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&minlatitude=38.0&maxlatitude=41.5&minlongitude=-119.5&maxlongitude=-114.5&minmagnitude=1.0`) and UNR Nevada Seismological Laboratory relocated catalog (including the May 15, 2020 $M_w 6.5$ Monte Cristo Range sequence inside the southern Walker Lane portion of the GeoDAWN footprint).
- **Physical signature targeted**: Anisotropic epicentral lineament density and nodal-plane strike projections at 100–500 m resolution.
- **Why it catches faults missing from the USGS/INGENIOUS catalogue**: The 2020 $M_w 6.5$ Monte Cristo Range earthquake ruptured a predominantly unmapped network of sinistral and normal faults across a $\sim 30\text{ km}$ zone in the southern GeoDAWN footprint ([Koehler et al., 2021, *SRL*](https://doi.org/10.1785/0220200371)).
- **Status in this sandbox**: `DATA-BLOCKED`. The sandbox outbound allowlist permits only `github.com`, `codeload.github.com`, `api.github.com`, `registry.npmjs.org`, `pypi.org`, and `files.pythonhosted.org`; `earthquake.usgs.gov` is unreachable. Furthermore, shipped Bands 10 and 16 in `training_features.tif` are pre-smoothed with a 100 km Gaussian kernel, erasing sub-km lineaments. Documented as a clean data-blocked follow-up for an internet-enabled environment.
