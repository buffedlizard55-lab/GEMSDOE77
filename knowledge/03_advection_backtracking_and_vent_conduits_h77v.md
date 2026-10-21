# Knowledge Note 03 — Hydrogeologic Advection Backtracking, Kinetic Geothermometer Fractionation, and Lake Lahontan Tufa Confounding in Blind Geothermal Vent Discovery (`H77V`)

- **Date:** `2026-10-21` (UTC)
- **Branch:** `arena/a278fbd0-gemsdoe77`
- **Scope:** Primary literature synthesis and empirical data forensic audit for Lane `H77V` (`H77V-VENT-ADVECT-CONDUIT`), storing verified scientific references and empirical discoveries for current and future DOE GEMS runs.

---

## 1. Verified Primary Scientific Sources (With Direct Official Links)

1. **DOE Geothermal Data Repository (GDR) Submission 1391 — Great Basin INGENIOUS Play Fairway Compilation:**
   - **Citation:** Faulds, J. E., Hinz, N. H., Coolbaugh, M. F., Ayling, B. F., Glen, J. M., Craig, J. W., McConville, E., Siler, D. L., Queen, J., earney, T., & DeAngelo, J. (2021). *INGENIOUS (Innovative Geothermal Exploration through Novel Investigations of Undiscovered Systems) Regional Play Fairway Data*. U.S. Department of Energy Geothermal Data Repository, Submission 1391.
   - **Official Link:** <https://gdr.openei.org/submissions/1391> | **DOI:** <https://doi.org/10.15121/1881483>
2. **USGS Data Release — GeoDAWN Airborne Magnetic and Radiometric Survey (NW Nevada / Oregon):**
   - **Citation:** Glen, J. M. G., Earney, T. E., Zielinski, L. A., Schermerhorn, W. D., Dean, B. J., & Hardwick, C. (2023). *Airborne magnetic and radiometric survey of northwest Nevada and adjacent parts of Oregon (GeoDAWN)*. U.S. Geological Survey Data Release.
   - **Official Link:** <https://doi.org/10.5066/P9X832BB>
3. **USGS 3DEP 1 m LiDAR & 1/3 Arc-Second (10 m) Seamless Digital Elevation Models:**
   - **Citation:** U.S. Geological Survey (2023). *3D Elevation Program (3DEP) 1-Meter and 1/3 Arc-Second Digital Elevation Models*.
   - **Official Link:** <https://www.usgs.gov/3d-elevation-program>
4. **Horizontal Fluid Flow & Lateral Outflow Plumes in Basin-and-Range Geothermal Systems:**
   - **Citation:** Ziagos, J. P., & Blackwell, D. D. (1986). A model for the transient temperature effects of horizontal fluid flow in geothermal systems. *Journal of Volcanology and Geothermal Research*, 27(3–4), 371–397.
   - **Official Link:** <https://doi.org/10.1016/0377-0273(86)90021-1>
   - **Citation:** Richards, M., & Blackwell, D. D. (2002). The forgotten ones: Geothermal roads less traveled in Nevada. *Geothermal Resources Council Transactions*, 26, 109–114.
   - **Official Link:** <https://www.geothermal-library.org/index.php?mode=pubs&action=view&record=1020025>
   - **Citation:** Ferguson, G., Grasby, S. E., & Hindle, S. R. (2009). What do aqueous geothermometers really tell us? *Geothermics*, 38(2), 183–194.
   - **Official Link:** <https://doi.org/10.1016/j.geothermics.2008.09.006>
5. **Chemical Geothermometry & Silica vs. Na-K-Ca Re-Equilibration Kinetics:**
   - **Citation:** Fournier, R. O., & Truesdell, A. H. (1973). An empirical Na-K-Ca geothermometer for natural waters. *Geochimica et Cosmochimica Acta*, 37(5), 1255–1275.
   - **Official Link:** <https://doi.org/10.1016/0016-7037(73)90060-4>
   - **Citation:** Fournier, R. O. (1977). Chemical geothermometers and mixing models for geothermal systems. *Geothermics*, 5(1–4), 41–50.
   - **Official Link:** <https://doi.org/10.1016/0375-6505(77)90007-4>
   - **Citation:** Giggenbach, W. F. (1988). Geothermal solute equilibria: Derivation of Na-K-Mg-Ca geoindicators. *Geochimica et Cosmochimica Acta*, 52(12), 2749–2765.
   - **Official Link:** <https://doi.org/10.1016/0016-7037(88)90143-3>
6. **Structural Controls & Critically Stressed Dilational Conduits in the Great Basin:**
   - **Citation:** Morris, A., Ferrill, D. A., & Henderson, D. B. (1996). Slip-tendency analysis and fault reactivation. *Geology*, 24(3), 275–278.
   - **Official Link:** <https://doi.org/10.1130/0091-7613(1996)024%3C0275:STAAFR%3E2.3.CO;2>
   - **Citation:** Faulds, J. E., & Hinz, N. H. (2015). Favorable tectonic and structural settings of geothermal systems in the Great Basin region, western USA: Proxies for discovering blind geothermal systems. *Proceedings, World Geothermal Congress 2015*, Melbourne, Australia, 1–6.
   - **Official Link:** <https://pangea.stanford.edu/ERE/db/WGC/papers/WGC/2015/11066.pdf>
   - **Citation:** Coolbaugh, M. F., Raines, G. L., Zehner, R. E., Shevenell, L., & Williams, C. F. (2006). Prediction and discovery of new geothermal resources in the Great Basin: Multiple evidence of a large undiscovered resource base. *Geothermal Resources Council Transactions*, 30, 867–873.
   - **Official Link:** <https://www.geothermal-library.org/index.php?mode=pubs&action=view&record=1025144>
7. **Pleistocene Lake Lahontan Highstand & Shoreline Tufa vs. Fault-Controlled Sinter:**
   - **Citation:** Adams, K. D., & Wesnousky, S. G. (1998). Shoreline processes and the age of the Lake Lahontan highstand in the Jessup embayment, Nevada. *Geological Society of America Bulletin*, 110(10), 1318–1332.
   - **Official Link:** <https://doi.org/10.1130/0016-7606(1998)110%3C1318:SPATAO%3E2.3.CO;2>

---

## 2. Three New Empirical Discoveries in the DOE GDR 1391 Footprint Data (`H77V` Audit)

### Discovery 1 (`IR-H77V-1`): `Paleo_geothermal_final.csv` Is 30.6% Pleistocene Lake Lahontan Tufa
Inspecting all `281` in-footprint records of `Paleo_geothermal_final.csv` (`paleo_geothermal_regional.zip`, DOE GDR 1391) revealed the following exact `Deposit_type` breakdown inside the GeoDAWN footprint:
- `Tufa`: **86 records (`30.6%`)**
- `Silica-cemented sand`: **73 records (`26.0%`)**
- `Sinter`: **50 records (`17.8%`)**
- `Travertine`: **36 records (`12.8%`)**
- `Alteration`: **13 records (`4.6%`)**
- `Explosion craters`: **10 records (`3.6%`)**
- `Sulfur`: **5 records (`1.8%`)**
- `Silicified fault breccia`: **2 records**, `Silica-cemented mud or ash`: **2 records**, `Sulfate crust`: **1**, `Calcite-qtz veins`: **1**, `Silica vein`: **1**, `Argillic alteration`: **1**.

Prior feature pipelines (`src/gems77_step/features.py`) treated every row of `Paleo_geothermal_final.csv` as an identical impulse (`1.0`). Because `Tufa` in NW Nevada predominantly forms along wave-agitated Pleistocene Lake Lahontan paleoshorelines ($Z_{\text{ortho}} \le 1,340\text{ m}$) rather than tectonic fault conduits, unweighted paleo-deposit rasters inject shoreline false positives into the hydrothermal prior. `H77V` separates high-temperature fault-orifice deposits (`Sinter`, `Silicified fault breccia`, `Explosion craters`, `Silica-cemented sand`, `Travertine`, `Sulfur`) from shoreline `Tufa`.

### Discovery 2 (`IR-H77V-2`): `gdr_wellspring_in_footprint.csv` Separates Drilled Outflow Wells (`24,634` rows) from Natural Surface Springs (`2,458` rows) Plus Unstripped `'Hot '` Whitespace
In `gdr_wellspring_in_footprint.csv` (`27,092` rows):
- **Drilled wells** (`well_features_20220808`: `19,456`, `well_chemistry_20220808`: `2,953`, `well_temperature_20220808`: `2,225` -> **`24,634` rows, `90.9%`**) are predominantly sited in flat agricultural basins down-gradient of range-front faults.
- **Natural surface springs** (`spring_chemistry_20220808`: `1,014`, `spring_temperature_20220808`: `945`, `spring_features_20220808`: `499` -> **`2,458` rows, `9.1%`**) mark surface groundwater/thermal discharge orifices directly controlled by fault permeability barriers or piedmont fault toes.
- In addition, the `thermalclass` column contains `121` rows with trailing whitespace `'Hot '` alongside `1,873` rows of `'Hot'` and `2,245` rows of `'Warm'`. Naive equality filtering (`thermalclass == 'Hot'`) silently drops 6.1% of hot records unless `.strip()` is applied.

### Discovery 3 (`IR-H77V-3`): Dip-Induced Hanging-Wall Offset of Gravity and Basement Gradients
In the Great Basin, range-front and intrabasin normal faults dip $50^\circ\text{–}65^\circ$ toward the basin (down-slope along $-\nabla Z_{\text{ortho}}$). Because gravity and magnetic basement inversions integrate density/susceptibility contrast over the top $0.5\text{–}2.5\text{ km}$ of basin fill, the peak horizontal gradient of `iso_grav_anom` (Band 13) and `depth_to_base_surf` (Band 15) is systematically shifted **$250\text{–}500\text{ m}$ ($2.5\text{–}5.0\text{ px}$) basinward (down-slope)** relative to the surface fault trace. Under the competition's strict $300\text{ m}$ ($3\text{ px}$) triangular DTI kernel, an unshifted gravity gradient peak falls near or outside the $3\text{ px}$ scoring halo of the surface fault trace. Transporting the gravity/basement horizontal gradient **upslope along $+\hat{\mathbf{u}}_{\text{up}}$ by $3.5\text{ px}$ ($350\text{ m}$)** realigns the subsurface density boundary with the surface piedmont fault trace.
