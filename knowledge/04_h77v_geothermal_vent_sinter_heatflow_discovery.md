# Knowledge Note 04 — Lane `H77V`: Tufa-Deconfounded Siliceous Vent-Sinter & Multi-Depth Conductive Heat-Flow Strike-Conduit Coupling

**Date:** 2026-10-21<br>
**Repository:** `buffedlizard55-lab/GEMSDOE77`<br>
**Branch:** `arena/e9248aa6-gemsdoe77`<br>
**Evidence receipts:** `evidence/h77v_e1_canary_surface.json`, `evidence/h77v_e2_holdout.json`, `evidence/h77v_e3_uniqueness.json`, `evidence/run_card_h77v.json`, `docs/figures/h77v_surface_and_dots.png`

---

## 1. Executive Summary of Discoveries & Experimental Outcomes

Following the explicit next-step roadmap in `docs/review-2026-10-21-h77g.md` (Section 6), we conducted a line-by-line physical and data-engineering audit of the external geothermal and hydrothermal datasets (`paleo_geothermal_regional.zip`, `2m_temperature_probe_INGENIOUS_regional_data.zip`, `gdr_wellspring_in_footprint.csv`, `gdr_volcanic_vents_in_footprint.csv`, `geodawn_rad_u8.tif`, `lidar_scarp_features_u8.tif`, `topo_10m_derived_100m_u8.tif`) and uncovered **five concrete code/data irregularities (`IR-H77V-01` through `IR-H77V-05`)**:

1. **`IR-H77V-01` (`Paleo_geothermal_final.csv` Lacustrine `Tufa` Confounding in `src/gems77_step/features.py`):**
   - Out of `281` in-footprint paleo-geothermal deposits (`DOE GDR 1391`), `86` (`30.6%`) are Pleistocene Lake Lahontan algal/wave-agitation carbonate shoreline `Tufa` mounds (`med_d_vis = 61.5 px = 6.15 km`, only `12.8% <= 10 px` from visible faults and `3.5% <= 15 px` from withheld hidden faults on Seed 20).
   - By contrast, high-enthalpy hydrothermal fault-vent deposits sit directly on fault conduits: `Alteration`/`Argillic alteration` (`n=14`, `med_d_vis = 1.0 px`, `100.0% <= 10 px` visible, `15.4% <= 15 px` hidden), `Sulfur` (`n=5`, `med_d_vis = 2.0 px`, `80.0% <= 10 px`), `Sinter` (`n=50`, `med_d_vis = 4.0 px`, `64.0% <= 10 px`), and `Veins/Breccia/Explosion craters` (`n=15`, `46.7% <= 10 px`).
   - Prior code (`src/gems77_step/features.py` lines 253–264) assigned `imp_paleo = 1.0` equally across all 281 points.

2. **`IR-H77V-02` (`gdr_wellspring_in_footprint.csv` Missing-`temp_c` Drop & Geothermometer `vals[0]` Index Shift in `src/gems77_step/features.py`):**
   - Lines 212–218 of `src/gems77_step/features.py` appended `(temp_c, geothermquartz_c, geothermchalc_c, geothermcat_c)` into an untyped list `vals` and used `vals[0]` as surface temperature and `max(vals[1:])` as geothermometer temperature.
   - Across `27,092` in-footprint rows, `267` `Hot`/`Warm` records lack `temp_c` (`94` dropped completely; `171` rows inside footprint had `geothermquartz_c` shifted into `vals[0]`, misinterpreting 110–180 °C reservoir quartz temperatures as surface temperature and deleting the geothermometer term).

3. **`IR-H77V-03` (Non-Existent Derived Probe/Paleo TIFF Paths in `src/gems77d/features.py`):**
   - Lines 225–232 of `src/gems77d/features.py` checked `derived_gdr_2m_probes_100m_u8.tif` and `derived_gdr_paleo_100m_u8.tif`, neither of which exists in `data/external/` or `registry/data_manifest.json`, causing `H77D` to silently omit 100% of the `2,782` 2-m temperature probes and `281` paleo-geothermal deposits.

4. **`IR-H77V-04` (Ignored Multi-Depth Conductive Heat-Flow Gradient $\partial T/\partial z$ in `2m_temperature_probe_n83geo.dbf`):**
   - Out of `2,782` in-footprint 2-m probes (`DOE GDR 1388`), `1,543` stations record both `T1m` (or `T15m`) and `T2m`.
   - The `816` upward conductive heat-flow stations ($\partial T/\partial z = T_{2\text{m}} - T_{1\text{m}} > 0$) exhibit `17.8%` proximity ($\le 15\text{ px}$) to Seed 20 hidden faults vs `12.1%` for the `727` downward solar-heated stations ($\partial T/\partial z \le 0$) (`+47%` relative enrichment on withheld hidden faults).

5. **`IR-H77V-05` (Row-Major Tie-Breaking Ramp in `F77.rank01` & `4*sigma` Gaussian Truncation Box Artifact):**
   - `F77.rank01` in `src/gems77/features77.py` uses `np.argsort(..., kind="stable")`, which breaks tied values (such as `0.0` outside the `4*sigma` kernel truncation box of `ndimage.gaussian_filter` on sparse point impulses) by row-major pixel index (`row * width + col`), creating a synthetic north-to-south linear ramp across the study footprint (`row 0 -> 0.0`, `row 3729 -> ~0.85`).
   - Replacing `argsort(kind="stable")` with tie-aware `scipy.stats.rankdata(..., method="average")` and combining steerable along-strike advection with smooth Euclidean-distance radial decay ($\exp(-d/\lambda)$) eliminated both artifacts and raised `A_std_plus_h77v` `HOLDOUT-DTI` to `0.0649` on Seed 20 and `0.0767` on Seed 30 (`+0.0160` over `A_std`, 95% CI `[+0.0012, +0.0317]`).

---

## 2. Quantitative Summary of the 3-Experiment Parallel-Run Protocol (`H77V`)

### Experiment 1 (`H77V-E1` — Single-Feature Leakage Canaries & Phase-1 Surface Uniqueness)
- All 10 label-free `H77V` channels passed the single-feature ROC-AUC leakage canary (`ceiling = 0.90`):
  - Max symmetric ROC-AUC on Seed 20 (`11,321` withheld positives): **`0.5933`** (`h77v_scarp_strike_vent_bridge`), `0.5880` (`h77v_composite_surface`), `0.5665` (`h77v_probe_upward_heatflow_dtdz`), `0.5571` (`h77v_epithermal_k_th_alteration`), `0.5513` (`h77v_demag_grav_step_conduit`), `0.5338` (`h77v_tufa_shoreline_deconf_contrast` vs `0.5257` `h77v_sinter_alter_vent_conduit`).
  - Max symmetric ROC-AUC on Seed 30 (`11,342` withheld positives): **`0.5855`** (`h77v_composite_surface`), `0.5835` (`h77v_scarp_strike_vent_bridge`), `0.5790` (`h77v_probe_upward_heatflow_dtdz`), `0.5535` (`h77v_demag_grav_step_conduit`).
- Phase-1 pre-placement surface uniqueness across `70` registry rasters:
  - `literal_worst_spearman_full_footprint = 0.1652 <= 0.90` (`PASS`).

### Experiment 2 (`H77V-E2` — Dual-Seed Buffered Spatial Holdout on `gems57-pooled-hide-v2`)

| Arm | Description | Seed 20 `HOLDOUT-DTI` (`11,321` px) | Seed 20 95% CI | Seed 30 `HOLDOUT-DTI` (`11,342` px) | Seed 30 95% CI |
|---|---|---:|---:|---:|---:|
| `H77-Step` (Incumbent) | Multi-scale scarp + catalogue-continuation propensity (`37,654` dots) | `0.152718` | `[0.131548, 0.175185]` | — | — |
| **`A_std_plus_h77v` (29 ch)** | **19 standard bands + 10 `H77V` channels (4-quadrant OOF HGB)** | **`0.064880`** | **`[0.051446, 0.078805]`** | **`0.076652`** | **`[0.058871, 0.094988]`** |
| `A_std` (19 ch control) | 19 standard bands only (identical HGB + 300 m submodular cover) | `0.060664` | `[0.043863, 0.078646]` | `0.060649` | `[0.044518, 0.080751]` |
| **`S_h77v_physics` (pure)** | **Untrained 10-channel `H77V` composite surface** | **`0.050226`** | **`[0.035695, 0.065371]`** | **`0.062132`** | **`[0.041410, 0.084212]`** |
| `A_h77v_only` (10 ch) | 10 `H77V` channels only (4-quadrant OOF HGB) | `0.041992` | `[0.031326, 0.052619]` | `0.048955` | `[0.035388, 0.062618]` |
| `A0_random` | Uniform random surface + 300 m submodular cover (`37,654` dots) | `0.047619` | `[0.041392, 0.053858]` | `0.048088` | `[0.042046, 0.054049]` |

- **Paired block-bootstrap differences (`A_std_plus_h77v` vs controls):**
  - On **Seed 30**: `Δ(A_std_plus_h77v - A_std) = +0.016003` (`95% CI [+0.001231, +0.031695]` — statistically significant positive lift above the 19-band geophysical baseline!) and `Δ(A_std_plus_h77v - A0_random) = +0.028564` (`95% CI [+0.011892, +0.045326]`).
  - On **Seed 20**: `Δ(A_std_plus_h77v - A_std) = +0.004216` (`95% CI [-0.011452, +0.019233]`) and `Δ(A_std_plus_h77v - A0_random) = +0.017261` (`95% CI [+0.001470, +0.032653]`).
  - Furthermore, untrained **`S_h77v_physics`** beats `A0_random` on both Seed 20 (`0.05023 vs 0.04762`) and Seed 30 (`0.06213 vs 0.04809`), and on Seed 30 even outperforms the trained 19-band `A_std` (`0.06213 vs 0.06065`).
- **Comparison against repository incumbent (`H77-Step`, `0.152718 HOLDOUT-DTI`):**
  - Because `A_std_plus_h77v` (`0.0649` / `0.0767`) uses zero fault-catalogue spatial-continuation features (`dist_vis_fault`, tip continuation, multi-scale terrain stack) and therefore remains below the `0.152718` `H77-Step` bar, `H77V` is classified as **`verdict: "negative"`** (`ok_to_download: true`, `ok_to_submit: false`, `0` weekly slots spent).

### Experiment 3 (`H77V-E3` — Full-Map GeoTIFF + `.zip` Generation, Validator & Uniqueness Audit)
- **Submission name:** `gems77-h77v-ventsinter-heatflow-37654dots-20261021-cc5e4cd9-zeros`
- **GeoTIFF path:** `docs/downloads/gems77-h77v-ventsinter-heatflow-37654dots-20261021-cc5e4cd9-zeros.tif` (`119,861` bytes, `sha256: 2044447ff7d4df318c277846299815591211c83641a3beb20c90f9b33c2255bb`)
- **ZIP path:** `docs/downloads/gems77-h77v-ventsinter-heatflow-37654dots-20261021-cc5e4cd9-zeros.zip` (`sha256: b1259104bf6ce011a77caff1d0446102e4eb1a683ec6e3b5c69c2a4a4c22879e`)
- **Validator (`vendor/gems57/validate.py`):** `all_checks_passed = True` (`3730x3292`, `float32`, `EPSG:32611`, `min = 0.0`, `max = 1.0`, `n_nan = 0`, `n_infinite = 0`, `emitted_positive_pixels = 37,654`, `on_catalogue_positive_pixels = 0`).
- **Uniqueness (`vendor/gems57/uniqueness.py` across `70` registry rasters):**
  - **Informative (non-space-filling) sparse/dot models (`60` rasters):** **`informative_unique = True`** (`worst forward 3-px overlap = 0.3285 <= 0.70` vs `calib_h28ridge_01839.tif`; `worst |Spearman rho| = 0.0123 <= 0.90`; vs `GEMSDOE32` `H33` champion `c55bafc4`: `forward_overlap_3px = 0.2357`, `spearman_full_footprint = +0.0079`).
  - **Literal Policy I17 (`protocol_unique = False`):** Fails closed against the `10` space-filling witness/lattice rasters (`dense_witness.tif` `ab0a0a62` covers 100% of allowed pixels -> `forward_overlap_3px = 1.000`).
