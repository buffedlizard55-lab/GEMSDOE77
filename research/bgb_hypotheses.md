# Preregistered Candidate Geological Hypotheses (`H77-BGB` Session — 2026-10-21)

Before writing or running any model code, we formulate **four untried geological hypotheses** targeting unmapped Quaternary and blind hydrothermal faults in the DOE GEMS northwestern Great Basin footprint (`3730 × 3292` at `100 m × 100 m`, `EPSG:32611`). Every candidate is audited for:
1. **Specific layer(s) involved** (with verified band indices from `evidence/band_diagnostics.json`),
2. **Physical signature targeted** (differential operator / multi-field coupling),
3. **Why it catches faults missing from the USGS / INGENIOUS catalogue** rather than re-detecting mapped range-front scarps,
4. **How it differs from all 12 prior lanes** in `GEMSDOE77` (`urank`, `H77C`, `contactphys`, `LIA`, `H77-Step`, `H78`, `H77D`, `H79-BB`, `H79-RH`, `H77L`, `D77`, `H77E`, `H77-XSEN`) and open sibling PRs (`#11 H77-HSL`, `#29 H79-ORIENT`),
5. **Named non-fault geological process** that could mimic the signature, and
6. **Expected DTI improvement ÷ implementation cost** plus free official source obtainability.

---

## Ranking Summary Table

| Rank | Candidate ID & Name | Specific Layers (Verified) | Physical Signature Targeted | Expected DTI Gain | Implementation Cost | Viable in Sandbox? | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **`H77-BGB`: Buried Half-Graben Growth-Wedge Inflection × Transtensional Strain-Partitioning Gradient × Illite–Smectite Conductivity-Margin Breach** | `training_features.tif` Bands **15** (`depth_to_base_surf`), **13** (`iso_grav_anom`), **11** (`iso_grav_anom_vg`), **19** (`det_elev_slope`), **4** (`geod_2ndinv`), **7** (`geod_shearrate`), **8** (`geod_dilaterate`), **10** (`deq_n100a15`), **16** (`ieq_n100a15`), **17** (`cond_surf`), **2** (`rtp`); `geodawn_extensions_u8.tif` Band **4** (`TMI_up150`); `gdr_wellspring_in_footprint.csv` (`temp_c`, `geothermquartz_c`, `geothermchalc_c`, `geothermcat_c`) | Gravity-normal directional Hessian inflection $D_{nn} = \hat{n}_g^\top (\nabla^2 D)\hat{n}_g$ of basement depth under flat basin floors ($\exp(-\text{slope}/\sigma_s)$), gradient of kinematic transtensional mode index $|\nabla \tau_{\text{ext}}|$, and lateral log-conductivity margin $|\nabla \log(1+\sigma)|$ × shallow RTP Laplacian demagnetization trough × silica/cation geothermometer disequilibrium $\Delta T_{\text{chem}}$ | **Medium–High** (`+0.008` to `+0.025` over random / static baselines on blind intrabasin faults) | **Low–Medium** (10 vectorised `float32` channels, `<8 min` runtime, `<600 MB` RAM) | **Yes** — 100% SHA-256-verified in `.cache/gems_data/` | **SELECTED FOR HOLDOUT VALIDATION (`H77-BGB`)** |
| **2** | **`H77-CST`: Subsurface Magnetic Tilt-Worm Tangent Azimuth × Geodetic Principal Extension Slip/Dilation Tendency** | `training_features.tif` Bands **3** (`tmi_hg`), **9** (`tmi_vg`), **7** (`geod_shearrate`), **8** (`geod_dilaterate`) | Structure-tensor tangent azimuth $\theta_{\text{mag}}$ along zero-crossings of $\tan^{-1}(\text{tmi\_vg}/\text{tmi\_hg})$ weighted by Morris et al. (1996) dilation tendency $\cos^2(\theta_{\text{mag}} - \theta_{\text{ext}})$ modulated by local $\dot{\varepsilon}_{\text{dil}}/\dot{\gamma}_{\text{shear}}$ | **Medium** | **Low** | **Yes** — uses shipped bands | **Backup candidate** (do not run unless Rank 1 is aborted) |
| **3** | **`H77-PRW`: Two-Scale Wavelength-Band Poisson Ratio ($\Delta g / \Delta \text{PG}_{\text{RTP}}$) Sign-Reversal Boundaries** | `training_features.tif` Bands **2** (`rtp`) and **13** (`iso_grav_anom`) | Band-pass Difference-of-Gaussians ($\lambda_1 \in [0.4, 1.2]\text{ km}$, $\lambda_2 \in [1.5, 4.5]\text{ km}$) on $g_{\text{iso}}$ and Fourier pseudogravity $\text{PG}(\text{RTP})$ to isolate narrow ribbons where local Cordell & Grauch (1985) ratio $\partial g / \partial(\text{PG})$ switches sign along high $|\nabla g|$ | **Low–Medium** | **Medium** | **Yes** — uses shipped bands | **Deferred** (`contactphys` showed weak standalone lift from single-scale PG regression) |
| **4** | **`H77-DDM`: Relocated Double-Difference Microseismicity 3D Planar RANSAC Up-Dip Projection** | External USGS ComCat / UNR NSL relocated hypocenter catalog (`https://earthquake.usgs.gov/fdsnws/event/1/query`) + Band **15** (`depth_to_base_surf`) | 3D planar cluster fitting on relocated hypocenters ($z \in [1, 12]\text{ km}$) projected up-dip to the pre-Tertiary basement top (`depth_to_base_surf`) and ground surface | **Medium–High** (if sub-km catalog available) | **High** (external API fetch + 3D RANSAC) | **No — Blocked by network allowlist** (`earthquake.usgs.gov` not reachable in sandbox) | **Not viable in sandbox** (documented per standing instruction) |

---

## Detailed Physical & Geological Formulation of Each Candidate

### Candidate 1 (Rank 1 — Selected): `H77-BGB` — Buried Half-Graben Growth-Wedge Inflection × Transtensional Strain-Partitioning Gradient × Illite–Smectite Conductivity-Margin Breach

1. **Specific layers involved:**
   - `training_features.tif` Band **15** (`depth_to_base_surf`, depth in meters to pre-Tertiary basement from gravity-seismic constrained basin thickness inversion),
   - `training_features.tif` Band **13** (`iso_grav_anom`, isostatic residual gravity anomaly) and Band **11** (`iso_grav_anom_vg`, first vertical derivative $\partial g/\partial z$),
   - `training_features.tif` Band **19** (`det_elev_slope`, detrended surface slope magnitude),
   - `training_features.tif` Band **4** (`geod_2ndinv`), Band **7** (`geod_shearrate`), Band **8** (`geod_dilaterate`), Band **10** (`deq_n100a15`), Band **16** (`ieq_n100a15`),
   - `training_features.tif` Band **17** (`cond_surf`, surface electrical conductivity) and Band **2** (`rtp`, reduced-to-pole magnetic anomaly),
   - `external/geodawn_extensions_u8.tif` Band **4** (`TMI_up150`, 150 m upward-continued magnetic field),
   - `external/gdr_wellspring_in_footprint.csv` (`temp_c`, `geothermquartz_c`, `geothermchalc_c`, `geothermcat_c` from DOE GDR Submission #1391; the `dist_known_fault_px` column is strictly excluded).

2. **Physical signature targeted (10 channels across 3 coupled sub-blocks):**
   - **Sub-block A — Buried Half-Graben Growth-Wedge Inflection (`bgb_01` to `bgb_04`):**
     Let $D(x,y)$ be `depth_to_base_surf` (Band 15) smoothed at $\sigma = 200\text{ m}$ (`2 px`) to remove grid quantization, and let $\mathbf{g}_H = (\partial_x g, \partial_y g)$ be the true horizontal gradient of `iso_grav_anom` (Band 13; never using Band 18 directly as a magnitude because `evidence/band_diagnostics.json` proved Band 18 is signed $\partial_x g$). Define the unit gravity-gradient normal vector:
     $$\hat{n}_g = \frac{(\partial_x g, \partial_y g)}{\sqrt{(\partial_x g)^2 + (\partial_y g)^2 + \epsilon}}$$
     Along an active normal fault bounding a half-graben, $\hat{n}_g$ points perpendicular to the fault strike from the hanging-wall basin toward the dense footwall horst. Projecting the basement gradient $\nabla D$ and basement Hessian $\nabla^2 D$ onto $\hat{n}_g$ yields:
     - `bgb_01_basement_step_along_grav`: Cross-strike directional basement slope $|\nabla D \cdot \hat{n}_g|$, isolating basement steps aligned with density boundaries.
     - `bgb_02_growth_wedge_inflection`: Cross-strike directional basement second derivative (inflection magnitude) $|D_{nn}| = |\hat{n}_g^\top (\nabla^2 D)\hat{n}_g|$, which peaks at the hinge/rollover and master fault inflection of a half-graben growth wedge.
     - `bgb_03_blind_basin_basement_dipole`: The product of cross-strike basement inflection and an alluvial-flatness weight,
       $$S_{\text{blind}} = \text{rank}(|D_{nn}| \cdot |\nabla D \cdot \hat{n}_g|) \cdot \exp\!\left(-\frac{\text{det\_elev\_slope}}{P_{60}(\text{det\_elev\_slope})}\right),$$
       which selectively amplifies sharp basement fault steps buried beneath flat basin-floor playas (`det_elev_slope` near zero) while suppressing exposed mountain blocks.
     - `bgb_04_grav_basement_concordance`: Multi-scale structural concordance between the vertical gravity gradient (`iso_grav_anom_vg` Band 11) zero-crossing gradient and the basement depth Laplacian $\nabla^2 D$.
   - **Sub-block B — Kinematic Transtensional Strain-Partitioning Gradient (`bgb_05` to `bgb_07`):**
     Across the NW Nevada Walker Lane – Basin and Range transition, dextral shear ($\dot{\gamma}_{\text{shear}}$, Band 7) partitions into WNW extensional dilatation ($\dot{\varepsilon}_{\text{dil}}$, Band 8). Define the dimensionless kinematic transtensional mode index:
     $$\tau_{\text{ext}}(x,y) = \frac{\dot{\varepsilon}_{\text{dil}}}{\sqrt{\dot{\varepsilon}_{\text{dil}}^2 + \dot{\gamma}_{\text{shear}}^2 + \epsilon}} \in [-1, 1]$$
     - `bgb_05_transtensional_mode_tau`: Positively dilating oblique-extensional regime score $\text{rank}(\max(0, \dot{\varepsilon}_{\text{dil}}) \cdot \dot{\gamma}_{\text{shear}})$, peaking where both shear and dilatation coexist (transtensional stepovers and horsetail splays; Faulds & Hinz, 2015).
     - `bgb_06_strain_partition_grad`: Spatial gradient magnitude $|\nabla \tau_{\text{ext}}|$ weighted by `geod_2ndinv` (Band 4), locating domain boundaries where strike-slip deformation transfers into normal dip-slip across unmapped intrabasin relay zones.
     - `bgb_07_aseismic_dilation_excess`: Geodetic-to-seismic strain deficit $\text{rank}(\text{geod\_2ndinv}) \cdot (1 - \text{rank}(\text{ieq\_n100a15})) \cdot \text{rank}(|\nabla \text{deq\_n100a15}|)$, highlighting high geodetic strain zones along seismicity-density edges that have not released strain in large historical earthquakes.
   - **Sub-block C — Illite–Smectite Conductivity-Margin Upflow Breach (`bgb_08` to `bgb_10`):**
     In Great Basin geothermal systems (Wannamaker et al., 2004; Cumming, 2009), smectite clay alteration forms a conductive cap ($>0.05\text{ S/m}$) in the shallow outflow plume, whereas the hotter ($>200\,^\circ\text{C}$) permeable upflow fault zone converts smectite to resistive illite/chlorite and destroys primary magnetite in host volcanics:
     - `bgb_08_conductivity_margin_grad`: Lateral gradient magnitude of log-transformed electrical conductivity $|\nabla \log(1 + \max(0, \text{cond\_surf}))|$, locating the lateral margin of the smectite clay cap rather than its interior plateau.
     - `bgb_09_upflow_demag_breach`: Coupling of the conductivity-cap margin `bgb_08` with a shallow hydrothermal demagnetization trough where upward-continued `TMI_up150` exceeds shallow `rtp` AND shallow `rtp` forms a local Laplacian trough ($\nabla^2 \text{rtp} > 0$).
     - `bgb_10_geotherm_diseq_coupled`: Chemical geothermometer disequilibrium halo $\Delta T_{\text{chem}} = \max(0, \max(T_{\text{quartz}}, T_{\text{chalc}}, T_{\text{cat}}) - T_{\text{meas}})$ from DOE GDR 1391 thermal springs/wells (`σ = 3.5 km` minus `σ = 9.0 km` ring bandpass so point locations become regional thermal-upflow domains) multiplied by the subsurface basement-step + conductivity-margin breach index.

3. **Why it catches faults missing from the USGS / INGENIOUS catalogue:**
   The USGS Quaternary Fault and Fold Database maps faults primarily from subaerial geomorphic scarps on aerial photographs and hillshaded DEMs. Where intrabasin synthetic/antithetic normal faults or transtensional stepover relays cut flat playa lakebeds (Carson Sink, Dixie Valley floor, Humboldt Sink, Granite Springs Valley, Black Rock Desert), Holocene lacustrine/eolian sedimentation and argillic clay alteration obliterate surface scarps. However, cumulative Cenozoic displacement still offsets the pre-Tertiary basement (`depth_to_base_surf`) across a density boundary (`iso_grav_anom`), partitions geodetic dilatation vs shear (`geod_dilaterate`, `geod_shearrate`), and localizes hydrothermal magnetite destruction at the lateral margin of the smectite clay cap (`cond_surf`).

4. **How it differs from everything already implemented in `GEMSDOE77`:**
   - `urank` used raw uranium `rad_U` thresholding.
   - `H77C` (quarantined) multiplied `cond_surf` gradient by raw `tmi_hg` and `iso_grav_anom_slope` with no holdout evaluation.
   - `contactphys` used Kovesi phase congruency and 25×25 pseudogravity linear regression.
   - `LIA` inverted 95 historical leaderboard scores.
   - `H77-Step`, `H78`, `H77L`, and `D77` relied on 1 m LiDAR scarp surface morphology (`lidar_scarp_features_u8.tif`, missing across 24.6% of footprint).
   - `H77D` used a 92-channel multi-stack with 12-bit LOCO visible-fault mechanics, 1 m LiDAR, and 10 m DEM.
   - `H79-BB` and `H79-RH` used single analytical equations (`|grad g| * |grad RTP|` and `U/Th` halo).
   - `H77E` used FFT upward continuation (500 m / 2000 m) and Reid (1990) 3D Euler deconvolution (`SI = 0, 1`).
   - `H77-XSEN` used magnetic analytic signal × radiometric Th/K and U/Th gradients.
   - **None of the 12 merged lanes or open PRs (#11, #29) computed:** (a) the gravity-normal projected Hessian inflection $D_{nn} = \hat{n}_g^\top (\nabla^2 D)\hat{n}_g$ of `depth_to_base_surf` weighted by surface flatness, (b) the kinematic transtensional strain-mode partitioning index $\tau_{\text{ext}}$ and its spatial gradient $|\nabla \tau_{\text{ext}}|$, or (c) the illite–smectite lateral conductivity margin coupled with shallow RTP Laplacian demagnetization troughs and GDR chemical geothermometer disequilibrium $\Delta T_{\text{chem}}$.

5. **Named non-fault geological process that could mimic it:**
   Buried pre-Tertiary paleochannel incised-valley fills or Mesozoic plutonic/caldera ring margins beneath flat saline playa margins, which can produce basement depth curvature inflections along gravity gradients, sharp lateral electrical conductivity transitions at saline/freshwater interfaces, and primary low-magnetization felsic volcanic troughs without an active Quaternary fault conduit.

---

### Candidate 2 (Rank 2 — Backup): `H77-CST` — Subsurface Magnetic Tilt-Worm Tangent Azimuth × Geodetic Principal Extension Slip/Dilation Tendency
- **Layers:** `training_features.tif` Bands 3 (`tmi_hg`), 9 (`tmi_vg`), 7 (`geod_shearrate`), 8 (`geod_dilaterate`).
- **Physical signature:** Extract local strike $\theta_{\text{mag}}$ of magnetic tilt zero-contours (`atan2(tmi_vg, tmi_hg) = 0`, Miller & Singh 1994) via a structure tensor and weight by Andersonian/Coulomb dilation tendency $\cos^2(\theta_{\text{mag}} - \theta_{\text{opt}})$ under regional WNW extension modulated by local dilatation/shear ratio.
- **Why it catches missing faults:** Buried basement faults critically stressed for dilation under the modern geodetic strain field act as open fluid conduits beneath younger sediment cover.
- **Difference from repo:** `H77D` computed strike favorability only on 1 m LiDAR surface scarps (`lidar_strike`), never on subsurface magnetic tilt worms.
- **Non-fault mimic:** Pre-Tertiary dikes or steeply dipping metasedimentary bedding striking NNE parallel to modern normal faults.

---

### Candidate 3 (Rank 3 — Deferred): `H77-PRW` — Two-Scale Wavelength-Band Poisson Ratio ($\Delta g / \Delta \text{PG}_{\text{RTP}}$) Sign-Reversal Boundaries
- **Layers:** `training_features.tif` Bands 2 (`rtp`) and 13 (`iso_grav_anom`).
- **Physical signature:** Band-pass filter `iso_grav_anom` and Fourier pseudogravity $\text{PG}(\text{rtp})$ across $\lambda_1 \in [0.4, 1.2]\text{ km}$ and $\lambda_2 \in [1.5, 4.5]\text{ km}$ to isolate narrow linear ribbons where Cordell & Grauch (1985) Poisson ratio switches sign along high $|\nabla g|$.
- **Why it catches missing faults:** Hydrothermal alteration along a blind fault destroys magnetite ($\Delta M < 0$) while silicification/brecciation preserves or increases bulk density contrast ($\Delta \rho > 0$).
- **Difference from repo:** `contactphys` used a single-scale 25×25 box regression (`pg_slope`, `pg_corr`) rather than two-wavelength bandpass sign-reversal ribbons.
- **Non-fault mimic:** Non-magnetic carbonate or quartzite stratigraphy juxtaposed against mafic volcanics along a depositional unconformity.

---

### Candidate 4 (Rank 4 — Blocked by External Network Allowlist): `H77-DDM` — Relocated Double-Difference Microseismicity 3D Planar RANSAC Up-Dip Projection
- **Layers:** External USGS ComCat FDSN API (`https://earthquake.usgs.gov/fdsnws/event/1/query`) / UNR Nevada Seismological Laboratory relocated hypocenter catalog + `training_features.tif` Band 15 (`depth_to_base_surf`).
- **Physical signature:** 3D planar RANSAC cluster extraction on relocated earthquake hypocenters ($z \in [1, 12]\text{ km}$) projected up-dip to the top-of-basement surface (`depth_to_base_surf`) and ground surface.
- **Why it catches missing faults:** Active blind fault swarms illuminate sub-vertical permeable fault planes that have not ruptured the Holocene surface.
- **Obtainability / Viability Audit:** **Not viable in this sandbox** because `earthquake.usgs.gov` is outside the sandbox network allowlist (`github.com`, `codeload.github.com`, `api.github.com`, `registry.npmjs.org`, `pypi.org`, `files.pythonhosted.org`). Recorded as non-viable per the standing instruction.
