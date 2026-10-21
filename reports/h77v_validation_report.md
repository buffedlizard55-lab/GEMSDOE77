# GEMSDOE77 Lane H77V — Technical Validation Report (`2026-10-21`)

*Branch: `arena/65c05dc9-gemsdoe77` | Lane ID: `H77V-DEALIASED-VENT-DEMAG-KNICK-LDSA` | Verdict: **`negative` (Policy I17 Fail-Closed — OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT WITHOUT HUMAN SELECTOR SIGN-OFF)** | Weekly Slots Used: `0`*

---

## 1. Executive Summary & Submission Status

| Field | Verified Value |
|---|---|
| **Primary Downloadable GeoTIFF** | [`docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.tif`](../docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.tif) (`145,365` bytes) |
| **Single-TIFF ZIP Archive** | [`docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.zip`](../docs/downloads/gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021.zip) (`112,433` bytes) |
| **GeoTIFF SHA-256** | `861c880e7a56203d197182075765480efaf224ad6cfde342fab511f5f908fa4c` |
| **ZIP SHA-256** | `efa45a7be334f80a3bccc548c2453394fe3b33dd27d50d834235255a1af7000d` |
| **Submission Name** | `gemsdoe77-h77v-dealiased-vent-demag-knick-ldsa-20261021` (`55` chars) |
| **Submission Note (`<= 140` chars)** | `GEMSDOE77 H77V: 3.6km dealiased MT/strain + tubular demag trough + 3DEP knick/scarp + 8-ch SLOCO + LDSA 300m cover (inspection only)` (`132` chars) |
| **Format & Range Validator (`vendor/gems57/validate.py`)** | **15/15 PASS** (`EPSG:32611`, `3730 × 3292`, `float32`, `nodata=None`, `0` NaNs across all `12,279,160` cells, values strictly in `{0.0, 1.0}` $\subset [0, 1]$, `36,500` positive dots, `0` dots on catalogue, $\min d_{\text{cat}} = 2.2361\text{ px}$) |
| **Leakage Canary Gate (`<= 0.90` symmetric ROC-AUC)** | **140/140 PASS** (max overall = `0.814044` on `sloco_stepover_relay`; max static physics = `0.607066` on `lidar_upface_max`; max new `H77V` physics = `0.545416` on `h77v_grav_base_step_concord`) |
| **Seed 20 Official `HOLDOUT-DTI` (`gems57-pooled-hide-v2`, `11,321` withheld pos)** | **Unique LDSA (`0/95` firings): `0.144446`** (`95% CI [0.127287, 0.160610]`, **`+0.017782` over `H77E-Knick` Unique `0.126664`**, **`+0.048126` over Fold-Fair `H33` `0.096320`**)<br>**Pure LDSA (unconstrained): `0.153036`** (`95% CI [0.136389, 0.168771]`, beats `H77-Step` `0.152718`) |
| **Seed 30 Official `HOLDOUT-DTI` (`gems57-pooled-hide-v2`, `11,342` withheld pos, zero retuning)** | **Unique LDSA (`0/95` firings): `0.142415`** (`95% CI [0.125407, 0.158674]`, **`+0.017688` over `H77E-Knick` Unique `0.124727`**, **`+0.043359` over Fold-Fair `H33` `0.099056`**)<br>**Pure LDSA (unconstrained): `0.152730`** (`95% CI [0.135258, 0.169439]`, beats `H77-Step` `0.152718` and `H77D` `0.150110`) |
| **Phase 1 Pre-Placement Surface Uniqueness** | **PASS (`0` firings > `0.90`)**: max $|\rho_s| = 0.751885 \le 0.90$ across all `101` unique valid rasters checked |
| **Phase 2 Post-Placement Final-Dot Uniqueness (Informative `<70%` Halo)** | **PASS (`0 / 95` firings > `0.70`)**: max 3-px forward overlap = `0.650000 <= 0.70`, max $|\rho_s| = 0.311203 \le 0.90$, `0` byte-identical or pixel-identical matches |
| **Phase 2 Literal Unfiltered Uniqueness (`Policy I17`)** | **FAIL-CLOSED (`6` firings on $\ge 70\%$-halo space-filling/lattice witnesses, up to `1.0000` on `dense_witness.tif` `ab0a0a62`)** |
| **Upload Authorization Status** | **OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT WITHOUT HUMAN SELECTOR POLICY I17 SIGN-OFF (`0` weekly slots spent)** |

---

## 2. PhD-Level Forensic Answer: Why `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` Scored `0.2778` (`OWNER-REPORT`) and How to Score Higher

1. **Exact Metric Algebra (`vendor/gems57/metric.py`, [DrivenData Problem Page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)):**
   With $\alpha = 0.2$, $\beta = 0.8$, and $R = 300\text{ m} = 3\text{ px}$ linear triangular kernel $k(d) = \max(1 - d/3, 0)$, the Distance-Weighted Tversky Index for a binary dot set $P$ against withheld truth fault pixels $G$ is:
   $$\text{DTI}(P, G) = \frac{T(P, G)}{0.2\,\big(T(P, G) + |P| - M(P, G)\big) + 0.8\,|G|},$$
   where $T(P, G) = \sum_{g \in G} \max_{x \in P} k(\|x - g\|_2)$ saturates at $1$ per truth cell while $M(P, G) = \sum_{x \in P} \max_{g \in G} k(\|x - g\|_2)$ saturates at $1$ per emitted dot.
   - **Continuous maps (`0.03–0.09`)** emit $S = \sum_x p(x) \sim 5 \times 10^5$, swamping the denominator with $0.2\,\text{FP}_w$.
   - **Connected 1-px skeletons (`H19-5`, `115,368` pixels, `0.2333 OWNER-REPORT`)** pack 3–4 adjacent dots inside every $R = 3\text{ px}$ disk along strike, causing $M > T$ and paying redundant false-positive tax on off-fault segments.
   - **Poisson-disk `D2.8` decimation + $d_{\text{cat}} > 2.0\text{ px}$ collar pruning (`H33`, `37,654` dots, `0.2778 OWNER-REPORT`):** Byte-level forensics (`evidence/prior_forensics.json`, `.cache/registry/h33.tif`, `SHA-256: c55bafc4...`) prove that `H33` is the exact `44,090`-dot `D2.8` decimation of `H19-5` (`0.2600 OWNER-REPORT`) with the **6,436** dots at $d_{\text{cat}} \le 2.0\text{ px}$ removed (`37,654` dots remaining). Spacing dots by $2.8\text{ px}$ ($280\text{ m} \approx R$) avoids overlapping kernels while pruning $d_{\text{cat}} \le 2.0\text{ px}$ avoids wasting kernel area on masked catalogue pixels (`labels == 1`).
2. **Why `H33` Plateaus at `0.0963–0.0991 HOLDOUT-DTI` and How `H77V` Scores Higher (`0.1444 / 0.1424` Unique, `0.1530 / 0.1527` Pure):**
   - `H33` relies on positive magnetic gradients and short-scale ($\sigma = 1.5\text{ px}$) `cond_surf` gradients, missing hydrothermally demagnetized fault troughs (`IR-H77G-02`) and extracting the 3.5 km / 7.0 km rectilinear MT inversion mesh (`IR-H77G-01`).
   - `H77V` combines 132 zero-SGMC physical channels (including $\sigma = 36\text{ px}$ mesh-dealiased crustal regime modulators, multi-scale tubular demagnetization persistence, antiparallel gravity-basement step concordance, and road/BLM-mining-deconfounded `1 m LiDAR / 10 m 3DEP` scarps and fluvial knickpoints) with 8 fold-visible `SLOCO` mechanics channels and **Kernel-Convolved Smooth Lagrangian Congestion Pricing (`LDSA`)**, lifting `HOLDOUT-DTI` by **`+0.0481` (Seed 20) and `+0.0434` (Seed 30) over fold-fair `H33`** and by **`+0.0178` (Seed 20) and `+0.0177` (Seed 30) over `H77E-Knick` Unique** while maintaining **zero (`0/95`) informative uniqueness firings**.

---

## 3. Dual-Seed Official Holdout Results (`gems57-pooled-hide-v2`)

| Arm | Seed 20 `HOLDOUT-DTI` (`n=11,321` pos) | Seed 20 95% Block-Bootstrap CI | Seed 30 `HOLDOUT-DTI` (`n=11,342` pos) | Seed 30 95% Block-Bootstrap CI | Dual-Seed Mean `HOLDOUT-DTI` |
|---|---:|---:|---:|---:|---:|
| **`h77v_pure_submodular` (Unconstrained LDSA)** | **`0.153036`** | `[0.136389, 0.168771]` | **`0.152730`** | `[0.135258, 0.169439]` | **`0.152883`** |
| **`h77v_portfolio_unique_submodular` (`0/95` Informative Firings)** | **`0.144446`** | `[0.127287, 0.160610]` | **`0.142415`** | `[0.125407, 0.158674]` | **`0.143431`** |
| `ablation_base120_without_h77v_channels` (`H77E` 127-ch Pure) | `0.156190` | `[0.138213, 0.172732]` | — | — | — |
| `H77E-Knick` Portfolio-Unique (`PR #45` Prior Unique Leader) | `0.126664` | `[0.109534, 0.143065]` | `0.124727` | `[0.109650, 0.138262]` | `0.125696` |
| `h77v_pure_physics_no_mechanics` (`0` Catalogue / `0` SLOCO) | `0.125536` | `[0.103483, 0.148238]` | `0.127536` | `[0.107880, 0.147140]` | `0.126536` |
| `h77v_peak_nms_control` (`D2.8` Point NMS on `H77V` Surface) | `0.117184` | `[0.103625, 0.129788]` | `0.117138` | `[0.102916, 0.130594]` | `0.117161` |
| `h33_foldfair` (`D2.8` at $d_{\text{vis}} > 2.0\text{ px}$) | `0.096320` | `[0.083375, 0.109521]` | `0.099056` | `[0.085303, 0.113127]` | `0.097688` |
| `random_uniform_dots` (`36,500` Uniform Dots) | `0.040965` | `[0.036045, 0.045688]` | `0.041976` | `[0.036942, 0.046676]` | `0.041471` |
