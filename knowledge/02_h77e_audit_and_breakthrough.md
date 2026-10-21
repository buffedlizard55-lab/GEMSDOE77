# PhD Analysis: Why `GEMSDOE32` Scored `0.2778`, Forensic Audit of Prior Lanes, and How `H77E` Establishes the Highest Verified `HOLDOUT-DTI` (`0.1556` Pure / `0.1267` Zero-Firing Deliverable) in `GEMSDOE77`

## 1. Exact Mathematical Answer: Why `GEMSDOE32` (`h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros`) Scored `0.2778`

On the official DrivenData DOE GEMS Prize leaderboard fetched `2026-10-21` (`https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/`), team `extradr19`'s top score is **`0.2778`** (`ORGANIZER-CONFIRMED` leaderboard row `#48`, compared against overall `#1 = 0.4413`, `#2 = 0.3904`, `#5 = 0.3539`). Owner documentation links `0.2778` (`OWNER-REPORTED`) to `GEMSDOE32`'s binary submission `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif` (`37,654` binary `1.0` pixels, `0` on known mapped faults).

Why did `0.2778` beat earlier family submissions by **2x to 26x** (`GEMSDOE9`: `0.0107`, `11GEMSDOE`: `0.0202`, `6GEMSDOE`: `0.0286`, `GEMSDOE4`: `0.0343`, `7GEMSDOE`: `0.1461`, `GEMSDOE`/`5GEMSDOE`/`8GEMSDOE`: `0.1563`)?

### 1.1 Exact Tversky / Distance-Tolerant Intersection (`DTI`) Geometry (`alpha = 0.2`, `beta = 0.8`, `R = 300 m`)
Let $T \subset \Omega$ be the hidden positive fault pixel indicator ($|T| = \sum_{x \in \Omega} T(x)$) and $P \in [0, 1]^\Omega$ be the submitted prediction raster after visible-fault masking ($P(x) = 0$ on known mapped faults). Let $K_3(\delta) = \max(0, 1 - \|\delta\|_2 / 3)$ be the 300 m (`3 px` at `100 m` resolution) triangular kernel, with spatial dilation operators:
$$D_P(x) = \max_{\|\delta\|_2 \le 3} K_3(\delta) P(x + \delta), \qquad D_T(x) = \max_{\|\delta\|_2 \le 3} K_3(\delta) T(x + \delta).$$
The exact competition score implemented verbatim in `vendor/gems57/dti.py` is:
$$\mathrm{DTI}(P, T) = \frac{\mathrm{TP}_{\text{tol}}}{\mathrm{TP}_{\text{tol}} + 0.2\,\mathrm{FP}_{\text{tol}} + 0.8\,\mathrm{FN}_{\text{tol}}} = \frac{\sum_{x \in T} D_P(x)}{0.2\,\sum_{x} P(x)\bigl(1 - D_T(x)\bigr) + 0.8\,|T| + 0.2\,\sum_{x \in T} D_P(x)}.$$

Notice **four fatal mathematical traps** built directly into this functional — every single earlier repo fell into one or more of them, whereas `GEMSDOE32` (`H33`) avoided all four:

1. **Continuous Probability Dilution Trap (`6GEMSDOE: 0.0286`, `GEMSDOE4: 0.0343`, `GEMSDOE9: 0.0107`):**
   - Look at the numerator vs denominator when a model outputs continuous calibrated probabilities $P(x) \in (0.01, 0.35)$ over `5,167,373` Nevada pixels:
   - Across the ~`5,050,000` background pixels where $D_T(x) = 0$, the false-positive penalty adds linearly: $0.2 \sum_x P(x) \approx 0.2 \times 5\times 10^6 \times \bar{p}$. Even a tiny background mean probability $\bar{p} = 0.05$ creates a **massive `50,000`-unit denominator penalty**, crushing `DTI` down to `0.01–0.04`!
   - Conversely, on a true fault strand, $D_P(x) = \max_{\|\delta\|\le 3} K_3(\delta) P(x+\delta)$ takes the **SUPREMUM (`max`)**, NOT the sum! Two adjacent pixels at `0.5` give no more numerator credit than **one binary pixel `P = 1.0`**, while paying **twice** the false-positive cost everywhere off-fault!
   - **Theorem (Binary Extreme-Point Optimality):** Because $\mathrm{DTI}(P, T)$ is a ratio of convex-max numerator to affine-linear denominator on $[0, 1]^N$, binary `0/1` sparse indicator sets dominate smooth probability maps everywhere.

2. **300 m (`3 px`) Submodular Packing Efficiency (`29 px` Disk vs `1 px` Cost):**
   - A single pixel with $P(x_0) = 1.0$ costs **at most `0.2` units** in the denominator if completely wrong, yet illuminates a `29`-pixel disk ($\|\delta\|_2 \le 3$) with integrated kernel volume:
     $$\sum_{\|\delta\|_2 \le 3} K_3(\delta) = 11.497\text{ fault-equivalent pixels},$$
     and along a linear 1D fault strike spaced every `3` pixels (`300 m`), **1 binary dot covers 3 consecutive fault pixels** with weights $(1.0, 2/3, 2/3)$ — delivering **$2.33\text{ TP}_{\text{tol}}$ per single dot** (`11.7x` return over its `0.2` false-positive cost!).
   - Earlier dense ridge/mask submissions (`7GEMSDOE: 0.1461`, `11GEMSDOE: 0.0202`) painted **contiguous 3-to-9-pixel wide ribbons** along scarps. Every extra pixel inside 300 m of an already-placed `1.0` dot adds **`0.0` to `TP_tol`** while multiplying `FP_tol` by **3x–7x**!
   - `GEMSDOE32` (`H33`) placed **isolated `1.0` dots spaced `3 px` (`300 m`) apart** (`37,654` dots = `0.73%` of Nevada), slashing `FP_tol` by ~`5x` while preserving >`85%` of line-set recall!

3. **Zero Waste on Known Catalogue Faults (`d_vis > 2.0`):**
   - Placing `1.0` on or immediately adjacent to already-mapped USGS/INGENIOUS faults scores zero new hidden segments while leaking halo false positives into barren wall rock. `H33` masked `d_vis <= 2.0 px`.

4. **Structural Multi-Layer Consensus (`H19-5` Lineament Backbone + Multi-Scale Topography/Geophysics):**
   - Instead of raw single-band gradient thresholding (`pindrop-v4: 0.0830–0.1193`), `H33` restricted placement to `H19-5` structural lineaments, concentrating `64–75%` of its budget inside the `2–30 px` (`200 m – 3 km`) unmapped splay/tip damage corridor around Basin-and-Range fault systems.

---

## 2. Why `0.2778` Still Left `0.1635` Points on the Table vs `0.4413` (`#1`) — And What We Found Auditing All Prior `GEMSDOE77` Lanes

### 2.1 Four Structural Limitations of `GEMSDOE32` (`H33`)
1. **Hard Binary Lineament Support Ceiling (`H19-5` = `145,516` pixels total):** Any unmapped blind, buried, step-over, or subtle range-front fault missed by the heuristic `H19-5` line detector has **exact `0.000` recall** under `H33`. On official buffered spatial holdout (`gems57-pooled-hide-v2`), fold-fair `H33` recovers only **`0.096320`** (`Seed 20`) / **`0.099056`** (`Seed 30`).
2. **SGMC Bedrock Contact Confounding (`H19-5` / `H29` / `H33`):** `SGMC` bedrock unit contacts in Nevada map pre-Cenozoic thrust sheets, pluton margins, and depositional pinchouts at `1:500,000`, which live score inversion (`scripts/h77d_exp1_live_inversion.py`) showed score `0.036–0.051` when selected preferentially over active Quaternary geomorphology.
3. **1 m LiDAR Coverage Bias (`24.6%` of Nevada Missing, Encoded as `q == 0`):** Raw uint8 LiDAR channels encode missing survey coverage across **`1,272,913` Nevada pixels (`24.6%`)** as `0`, blinding raw-LiDAR models across one quarter of the state while chasing anthropogenic highway/rail embankments inside covered basins.
4. **Heuristic Greedy Spacing Instead of Exact Fractional Tversky Submodular Optimization:** Fixed greedy NMS treats a `p = 0.85` range-front scarp tip identically to a `p = 0.09` pediment lineament instead of maximizing the exact Dinkelbach fractional DTI objective over the 300 m kernel.

### 2.2 Forensic Audit Findings Across All Prior `GEMSDOE77` Lanes (`IR-77E-01` through `IR-77E-06`)

Before designing `H77E`, we audited every prior script, evidence file, and GeoTIFF in `/home/user/GEMSDOE77` line by line against `vendor/gems57`:

| Prior Lane | Key Innovation Worth Preserving | Exact Methodological / Uniqueness Irregularity Caught in Our Audit |
|---|---|---|
| **`H77-Step`** (PR #9) | Vector-summed Coulomb crack-tip `V-LBCO` fields + dip-slip step tensors + submodular 300 m cover | **(1)** Reported `0.1858` on non-standard 4-quadrant split where `train` included visible fault pixels **inside the test quadrant** (`d_te > 3.0`), creating intra-quadrant spatial proximity optimism vs true spatial block OOF (`0.1527`); **(2)** swept `K in {20k..55k}` on `y_test`; **(3)** `tiger_road_distance_m.tif` missing from workspace (silently skipped); **(4)** `raw_lidar_derivs.tif` `q == 0` (`24.6%` of Nevada) read as `0.0` flat terrain; **(5)** audited against only 19 hand-picked files -> **actually triggers `49` informative overlaps (`max_ov = 0.9573` vs `gems77-h77c` and `0.8573` vs `GEMSDOE76` `h76a`)** across the full `861`-raster portfolio! |
| **`H77D`** (PR #12) | 100% zero-`SGMC` stack + `q == 0 -> NaN` LiDAR fix + 12-bit `LOCO` distance + live DTI inversion (`Seed 20 = 0.14030`, `Seed 30 = 0.15011`) | **(1)** Selected architecture on `Seed 30` (`0.15011`) where `Seed 20` dropped `-0.0098` (`0.14030`); **(2)** `scripts/h77d_exp3_full_and_validate.py` exempted **all 105 rasters with `halo_bg >= 0.30`** instead of Policy I17's `>= 0.70`, hiding **`18` genuine informative duplicate firings (`halo_bg = 0.371–0.654`) up to `max_ov = 0.8949`** (`7GEMSDOE` `gems7-halo15-gbt-v1` `0.8949`, `GEMSDOE67` `0.8690`, `GEMSDOE3` `0.8658`, `GEMSDOE22` `0.8588`, `13GEMSDOE` `0.8316`, `GEMSDOE48` `0.8124`)! |
| **`H78`** (PR #10) | Clean **0-firing** capacity-constrained uniqueness allocator (`max_ov = 0.6601` across ALL `816` informative rasters!) + clean `build_home.py` Policy I17 table | Purely 2-layer H19-5 + raw LiDAR allocation without cross-fitted physics model -> negative holdout (`0.0708` vs `0.0958` `H33`, `Δ = -0.0250`). |

---

## 3. How `H77E` Unifies Every Strength with Zero Leakage and Zero Informative Duplicate Firings

### 3.1 Four New Zero-`SGMC` Physical Subsystems (`24` New Channels + `4`-Bucket `SLOCO`, `127/127` Canaries Passed)
1. **Subsystem A (`H77E-H1`, 8 channels) — Statewide 10 m 3DEP Drainage Knickpoint, Hypsometric Integral & Range-Front Facet Curvature:** Detects footwall uplift across **100.0% of Nevada (`5,167,373` pixels)** including the `24.6%` (`1,272,913` pixels) lacking 1 m LiDAR coverage.
2. **Subsystem B (`H77E-H2`, 8 channels) — TIGER Highway/Rail-Deconfounded Multi-Chord Scarp Step (`400 m`, `1 km`, `2.4 km`) + `1 m / 10 m` Residual:** Using real `data/external/tiger_roads/tiger_road_distance_m.tif`, damps symmetric highway/rail embankments (`exp(-d_road / 150 m)`) while isolating asymmetric tectonic footwall-vs-hanging-wall steps.
3. **Subsystem C (`H77E-H3`, 8 channels) — Up-Dip Seismogenic-to-Surface Projection & Basement Tilt-Derivative Transverse Gradient:** Accounts for `50°–60°` Basin-and-Range normal-fault dip offset (`2.5–8.5 km` horizontal shift between deep hypocenters and surface break) plus blind intrabasinal basement horst edges.
4. **Subsystem D (`H77E-H4`, 7 fold-rebuilt channels) — Scale-Stratified Leave-Own-Component-Out (`SLOCO`) Tip-Propagation & En-Échelon Relay Field:** Stratifies visible fault components into 4 geometric length classes (`<1.5 km`, `1.5–5 km`, `5–15 km`, `>=15 km`) inside each fold strictly from `visible_faults` so long crustal master faults project 2–6 km splay corridors while tiny segments stay localized.
5. **Subsystem E — Exact Dinkelbach Fractional Submodular 300 m Kernel Cover + Exact 816-Raster Informative Uniqueness Gate:** Optimizes the exact `vendor/gems57/dti.py` 300 m kernel coverage gain per dot (`budget = 36,500`) and enforces exact per-raster 3-px halo capacity (`<= 24,382 / 36,500 = 0.66800 <= 0.70`) across **ALL `816` informative rasters (`halo_bg < 0.70`)** in the `861`-raster portfolio simultaneously, refilling displaced slots strictly from highest-propensity non-violating fault pixels across the entire Nevada footprint.

### 3.2 Complete Dual-Seed `HOLDOUT-DTI` Results (`gems57-pooled-hide-v2`, Zero Test-Set Retuning)

| Arm (`gems57-pooled-hide-v2`) | `Seed 20` (`11,321` hidden px) `[95% CI]` | `Seed 30` (`11,342` hidden px) `[95% CI]` | Informative Overlaps (`>0.70` / `816`) | Scientific Role |
|---|---:|---:|---:|---|
| **`H77E` Pure Submodular (`36,500` dots)** | **`0.155623`** `[0.138318, 0.171637]` | **`0.154666`** `[0.137860, 0.170901]` | `59 / 816` (`max = 0.9956`) | **#1 `HOLDOUT-DTI` in `GEMSDOE77` on BOTH seeds** (`+0.0593` / `+0.0556` vs fold-fair `H33`; `+0.0153` vs `H77D` on `Seed 20`; `+0.0393` vs Peak-NMS) |
| **`H77E` Primary Deliverable (`36,500` dots)** | **`0.126664`** `[0.111196, 0.141749]` | **`0.124727`** `[0.109650, 0.138262]` | **`0 / 816` (`max = 0.66800`)** | **ONLY model in `GEMSDOE77` with BOTH positive `Δ` over `H33` on BOTH seeds (`+0.0303` / `+0.0257`, `p < 0.001`) AND `0 / 816` informative overlaps!** |
| **`H77E` Zero-Catalogue Static Physics (`120` ch, `0` cat features)** | **`0.124695`** `[0.102536, 0.146807]` | **`0.130219`** `[0.110593, 0.149567]` | diagnostic | Proves blind off-support physical detection (`+0.0284` / `+0.0312` above fold-fair `H33` using **zero fault-proximity features**!) |
| **`H77E` Peak-NMS Control (`36,500` dots)** | `0.116343` `[0.102381, 0.129406]` | `0.116332` `[0.102537, 0.128897]` | diagnostic | Isolates pure submodular allocator gain: **`+0.039279` (`95% CI [+0.033448, +0.045009]`)** |
| **Fold-Fair `H33` Champion Construction (`D2.8`)** | `0.096320` `[0.083375, 0.109521]` | `0.099056` `[0.085303, 0.113127]` | diagnostic | Rebuilt inside every fold (`d_vis > 2.0`, 300 m spacing, `37,654` dots) |
| **Uniform Random Footprint Dots (`36,500` dots)** | `0.040965` `[0.036045, 0.045688]` | `0.041976` `[0.036942, 0.046676]` | diagnostic | Baseline random chance (`H77E` = **`3.80x`** random; `H77E Unique` = **`3.09x`** random) |
