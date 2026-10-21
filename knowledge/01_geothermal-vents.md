# Geothermal Vents, Blind Hydrothermal Systems & Fault-Controlled Permeability — Curated Knowledge Base

All links are official and verified; all raster/CSV measurements below are verified directly against the local SHA-256-pinned files in `/home/user/GEMSDOE77/.cache/gems_data/`.

---

## 1. Core Science: Why Geothermal Vents and Blind Hydrothermal Upflow Localize Unmapped Great Basin Faults

- **DOE GEMS Prize Official Problem Description & Metric**
  - Official URL: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  - Metric: Distance-weighted Tversky index (`DTI`) with $\alpha = 0.2$ (false-positive weight), $\beta = 0.8$ (false-negative weight), and a $300\text{ m}$ ($3\text{ px}$) linear triangular distance kernel $k(d) = \max(1 - d / 300\text{ m}, 0)$:
    $$\text{DTI} = \frac{\widetilde{\text{TP}}}{\widetilde{\text{TP}} + 0.2\,\widetilde{\text{FP}} + 0.8\,\widetilde{\text{FN}}}$$
  - Worked example on the official page ($\widetilde{\text{TP}} = 3.00$, $\widetilde{\text{FP}} = 1.89$, $\widetilde{\text{FN}} = 2.00 \Rightarrow \text{DTI} = 0.60$) is reproduced to machine precision in `tests/test_science.py`.
  - Submission format: single-band `float32` GeoTIFF (`.tif`) or `.zip` containing a single `.tif`, `EPSG:32611`, $100\text{ m}$ pixel size, shape `(height=3730, width=3292)` (`(width=3292, height=3730)` in GDAL/rasterio notation), affine transform `[100, 0, 243350, 0, -100, 4508550]`, all values finite in $[0.0, 1.0]$.

- **Structural Controls of Great Basin Geothermal Systems (Faulds et al. Structural Typology)**
  - Official references:
    - Faulds, J. E., Hinz, N. H., Coolbaugh, M. F., Cashman, P. H., Kratt, C., Dering, G., Edwards, J., Mayhew, B., & McLachlan, H. (2011). *Assessment of favorable structural settings of geothermal systems in the Great Basin, western USA*. Geothermal Resources Council Transactions, 35, 777–783; and Geosphere, 7(3), 675–694: https://pubs.geoscienceworld.org/gsw/geosphere/article/7/3/675/132459
    - Faulds, J. E., & Hinz, N. H. (2015). *Favorable tectonic and structural settings of geothermal systems in the Great Basin region, western USA: Proxies for discovering blind geothermal systems*. Proceedings World Geothermal Congress 2015: https://www.geothermal-energy.org/pdf/IGAstandard/WGC/2015/11095.pdf
    - Coolbaugh, M., Faulds, J., Kratt, C., Oppliger, G., Shevenell, L., Calvin, W., Ehni, W., & Zehner, R. (2006). *Geothermal systems in the Great Basin, western United States: Modern analogues to ceremonial and ore-forming epithermal systems*. UNR / NBMG: https://www.unr.edu/nbmg
  - **Key physical insight for fault discovery**:
    1. Across $>250$ catalogued geothermal fields in the Great Basin, fluid upflow rarely occurs along the mid-segment of a simple planar range-front fault (where thick clay gouge reduces cross-fault permeability and maximum displacement produces a prominent, already-mapped scarp).
    2. Instead, geothermal upflow concentrates at **high-dilation quaternary structural complexities** where displacement is transferring across multiple subtle strands:
       - **Discrete stepovers / relay ramps** ($32\%$ of catalogued Great Basin systems, e.g., Desert Peak, Bradys, Steamboat),
       - **Fault-tip horse-tail splays / terminations** ($22\%$, e.g., Gerlach, San Emidio),
       - **Synthetic/antithetic fault intersections and accommodation zones** ($27\%$, e.g., Dixie Valley, McGinness Hills, Salt Wells),
       - **Pull-apart basins and transtensional displacement transfer zones** in the Walker Lane ($12\%$).
    3. Critically, **over $70\%$ of geothermal resources in the Great Basin are "blind"** (no prominent surface fault scarp or active fumarole directly over the upflow zone because young playa silt, alluvial fans, or silica sinter/travertine cementation conceal the active trace). Because pre-lidar airphoto compilers relied on prominent range-front scarps, these relay-ramp, horse-tail tip, and intra-basin stepover strands are **systematically under-represented in the USGS/INGENIOUS surface catalogue**, even though they are active Quaternary permeable faults.

- **Great Basin INGENIOUS Project (DOE Geothermal Data Repository Submission 1391)**
  - Project portal: https://gbcge.org/current-projects/ingenious/
  - Official DOE GDR dataset: https://gdr.openei.org/submissions/1391
  - USGS Quaternary Fault and Fold Database: https://www.usgs.gov/natural-hazards/earthquake-hazards/faults

---

## 2. Verified In-Footprint Geothermal & Volcanic Inventory (`H77-HSL` Lane)

We verified the local INGENIOUS GDR 1391 and GeoDAWN extension files inside `.cache/gems_data/external/` (SHA-256 pinned in `registry/lia_data_manifest.json`):

| Dataset | Local File | Verified Count / Shape | Physical observables used in `H77-HSL` | Leakage guardrail |
|---|---|---:|---|---|
| **INGENIOUS Wells & Springs** (`GDR 1391`) | `external/gdr_wellspring_in_footprint.csv` | `27,092` records (`1,994 Hot`, `2,245 Warm`, `22,853 Cold`) | `row`, `col`, `thermalclass`, `temp_c` (`8,718` finite, range $0.1\text{–}296.5^\circ\text{C}$), `geothermquartz_c` (`973` finite, up to $269.1^\circ\text{C}$), `geothermchalc_c` (`966` finite), `geothermcat_c` (`1,185` finite) | Column `dist_known_fault_px` is catalogue-derived and is **strictly excluded** |
| **Quaternary Volcanic Vents** (`GDR 1391`) | `external/gdr_volcanic_vents_in_footprint.csv` | `21` vents (`19` Basalt, `2` Rhyolite) | `row`, `col` (Quaternary dike/vent alignments along deep crustal tensile conduits) | Column `dist_known_fault_px` is **strictly excluded** |
| **GeoDAWN Radiometric Ratio & Upward-Continued TMI** | `external/geodawn_extensions_u8.tif` | `4` uint8 bands, `3730 × 3292` | Band 1 `ThK`, Band 2 `UK`, **Band 3 `UTh`** (uranium/thorium hydrothermal mobility), **Band 4 `TMI_up150`** ($150\text{ m}$ upward-continued TMI for shallow vertical continuation residual $\text{rank}(\text{tmi}) - \text{rank}(\text{TMI}_{\uparrow 150})$) | 100% label-free airborne survey |
| **USGS 3DEP 1 m Lidar Scarp Block** | `external/lidar_scarp_features_u8.tif` | `12` uint8 bands, `3730 × 3292` | All 12 bands: `(1) ex_max`, `(2) ex_mean`, `(3) step_max`, `(4) lapneg_max`, `(5) lappos_max`, `(6) downface_max`, `(7) upface_max`, `(8) cross_max`, `(9) relief`, `(10) coh100`, `(11) strike`, `(12) valid` (`87.21%` valid footprint coverage) | 100% label-free 1 m bare-earth DEM |

---

## 3. Geophysics Layers in `training_features.tif` — Verified Physical Meanings & Corrected Irregularities

> **Correction Notice (`IR-77-01`, `IR-77-03`, `IR-77-10`)**: An earlier version of this file repeated two upstream misidentifications: (1) calling Band 6 (`tc`) "Tilt angle or total curvature — magnetic derivative", and (2) treating Band 18 (`iso_grav_anom_hg`) as an unsigned horizontal gradient magnitude. Both were disproved by direct verification against USGS ScienceBase item `657e1d85d34e23d3533209f7` and finite-difference checks on `training_features.tif` (see `research/irregularities.md` `IR-77-01` and `IR-77-03`). The corrected table is below.

| Band | Official description | Verified physical identity & what it detects | Official Source |
|---:|---|---|---|
| `1` | `mag_anom` | Earth Magnetic Anomaly Model (`EMAG2v3`) regional magnetic anomaly | NOAA NCEI EMAG2v3 https://doi.org/10.7289/V5H70CVX |
| `2` | `rtp` | GeoDAWN reduced-to-pole magnetic anomaly (centers dipolar anomalies over magnetic bodies) | USGS GeoDAWN DOI `10.5066/P93LGLVQ` https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and |
| `3`, `9`, `14` | `tmi_hg`, `tmi_vg`, `tmi` | GeoDAWN Total Magnetic Intensity (`14`), horizontal gradient magnitude (`3`), and first vertical derivative $\partial\text{TMI}/\partial z$ (`9`) | USGS GeoDAWN (`651f167fd34e3bef0c9a3047`) |
| `6` | `tc` | **GeoDAWN Airborne Gamma-Ray Spectrometer Total Count (`TC_air`, nGy/h)** ($r = +0.9921$ with USGS `GeoDAWN_Rad_TC_air.tif`, range $[10.0, 290.9]$, strictly positive; **NOT** magnetic tilt angle or curvature, which we compute explicitly from Bands `2, 3, 9`) | USGS GeoDAWN Radiometrics (`657e1d85d34e23d3533209f7`) |
| `13`, `18`, `11`, `5` | `iso_grav_anom`, `iso_grav_anom_hg`, `iso_grav_anom_vg`, `iso_grav_anom_slope` | Isostatic gravity anomaly (`13`), **signed eastward derivative $\partial g/\partial x$** (`18`, range $[-1.80, +1.64]$, $r = +0.978$ with $\partial(\text{Band 13})/\partial x$; full $\|\nabla g_I\|$ must be reconstructed with $\partial(\text{Band 13})/\partial y$), vertical derivative $\partial g_I/\partial z$ (`11`), and slope (`5`) | USGS gravity compilation (`Ponce 1997` / PACES) |
| `15` | `depth_to_base_surf` | Depth to pre-Cenozoic basement surface (Cenozoic basin fill thickness; steep gradients mark basin-bounding normal faults) | USGS Great Basin depth-to-basement model |
| `12`, `19` | `det_elev`, `det_elev_slope` | Detrended 100 m elevation (`12`) and slope (`19`); zero-crossings of $\nabla^2(\text{det\_elev})$ at positive slope mark the range-front piedmont base | USGS 3DEP https://apps.nationalmap.gov/downloader/ |
| `17` | `cond_surf` | Magnetotelluric (`MT`) near-surface electrical conductivity; high values in non-playa settings trace smectite/illite hydrothermal clay caps and fluid conduits | USGS Great Basin MT compilation |
| `4`, `7`, `8` | `geod_2ndinv`, `geod_shearrate`, `geod_dilaterate` | Geodetic strain-rate second invariant (`4`), shear rate (`7`), and dilatation rate (`8`) from GPS velocity fields | Hammond et al. / UNAVCO |
| `10`, `16` | `deq_n100k_a15`, `ieq_n100k_a15` | Earthquake density (`10`) and intensity (`16`) smoothed with a $100\text{ km}$, $a=15^\circ$ kernel | USGS ComCat https://earthquake.usgs.gov/fdsnws/event/1/ |
