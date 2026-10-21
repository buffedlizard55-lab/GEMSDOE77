# H84 Geological Hypothesis Preregistration & PhD-Level Analysis of `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` (`0.2778`)

**Date:** 2026-10-21 (UTC)  
**Branch:** `arena/693062aa-gemsdoe77`  
**Preregistration Receipt:** [`registry/preregistration_h84.json`](../registry/preregistration_h84.json)  
**Shared Evaluator:** `vendor/gems57/evaluate_holdout.py` (`gems57-pooled-hide-v2`, $\alpha=0.2$, $\beta=0.8$, $R=300\text{ m}$ triangular kernel)

---

## Part 1: PhD-Level Analysis of Why `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` Scored `0.2778` and How to Score Higher

### 1.1 Exact Functional Mechanics of the Distance-Weighted Tversky Index (DTI)
On the pinned $3730 \times 3292$ competition grid ($\text{EPSG:32611}$, $100\text{ m}$ pixel spacing, $5,167,373$ active footprint pixels $\Omega$, $60,988$ mapped USGS/INGENIOUS fault pixels $C$), the organizer evaluation masks known catalogue pixels $C$ and scores a predicted probability raster $P: \Omega \setminus C \to [0, 1]$ against expert-mapped uncatalogued ground-truth fault pixels $G \subset \Omega \setminus C$ using the Distance-Weighted Tversky Index ($\alpha = 0.2$, $\beta = 0.8$) with a $300\text{ m}$ ($R = 3.0\text{ px}$) radial triangular kernel $K(r) = \max(0, 1 - r/3.0)$:

$$\text{DTI}(P, G) = \frac{\text{TP}_w(P, G)}{\text{TP}_w(P, G) + 0.2\,\text{FP}_w(P, G) + 0.8\,\text{FN}_w(P, G)}$$

where:
- **Weighted True Positives ($\text{TP}_w$):** For each uncatalogued truth pixel $g \in G$, coverage is the supremum of kernel-weighted predictions within $300\text{ m}$:
  $$c_P(g) = \max_{x \in \Omega \setminus C:\, \|x - g\|_2 < 3} P(x)\,K(\|x - g\|_2), \qquad \text{TP}_w(P, G) = \sum_{g \in G} c_P(g)$$
- **Weighted False Negatives ($\text{FN}_w$):** $\text{FN}_w(P, G) = |G| - \text{TP}_w(P, G)$.
- **Weighted False Positives ($\text{FP}_w$):** Each predicted pixel $x \in \Omega \setminus C$ is penalized in proportion to $P(x)$ times its proximity deficit to the nearest uncatalogued truth pixel:
  $$q_G(x) = \max_{g \in G:\, \|x - g\|_2 < 3} K(\|x - g\|_2), \qquad \text{FP}_w(P, G) = \sum_{x \in \Omega \setminus C} P(x)\,\bigl(1 - q_G(x)\bigr)$$

Notice the fundamental asymmetry between the numerator and denominator operators:
1. $\text{TP}_w$ is a **local $L^\infty$ (supremum) operator** over the 29-pixel open disc $B_3(g) = \{u \in \mathbb{Z}^2 : \|u\|_2 < 3\}$ (whose kernel weights sum to $\sum_{\|u\|_2 < 3} K(\|u\|_2) = 10.6742$). Once a single binary dot $P(x)=1$ sits near $g$, placing additional nonzero predictions inside $B_3(g)$ yields **zero or diminishing marginal $\text{TP}_w$**.
2. $\text{FP}_w$ is a **global $L^1$ (summation) operator** over all $5,106,385$ allowed off-catalogue pixels. Every nonzero prediction $P(x) > 0$ placed at distance $\ge 300\text{ m}$ from $G$ ($q_G(x) = 0$) adds $+0.2\,P(x)$ directly to the denominator.

### 1.2 Why the Historical Progression Moved from `0.0884` $\to$ `0.1817` $\to$ `0.2337` $\to$ `0.2778`
Tracing the SHA-verified rasters in `registry/data_manifest.json` and `.cache/registry/h33.tif` reveals the three exact mathematical mechanisms that drove `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` to its owner-reported `0.2778` (in `GEMSDOE32`; note per `IR-01` that `0.2778` is owner-reported in the project brief while `0.0884`, `0.1817`, `0.2337`, and the `0.4413` leaderboard top are `ORGANIZER-CONFIRMED`):

1. **Continuous Surface (`h19-5`, `0.0884` ORGANIZER-CONFIRMED) $\to$ Poisson-Disk Sparsification (`D1.5`, `0.1817` $\to$ `D2.8`, `0.2337` ORGANIZER-CONFIRMED):**
   - `h19-5` emitted $1,780,829$ positive float pixels across the footprint with $\sum_x P(x) \approx 2.41 \times 10^5$. Because the supremum $c_P(g)$ is saturated by the single highest pixel within $300\text{ m}$ of each fault, $>95\%$ of that continuous probability integral contributed zero $\text{TP}_w$ while adding $\sim 0.2 \times 2.2 \times 10^5 \approx 44,000$ units of $\text{FP}_w$ to the denominator.
   - Replacing the continuous surface with $N = 37,654$ binary $\{0,1\}$ dots separated by $d_{\min} = 2.8\text{ px}$ ($2\sqrt{2}\text{ px} = 282.8\text{ m}$, matching the $R = 3.0\text{ px}$ kernel radius) preserved almost the entire supremum coverage $\text{TP}_w$ while collapsing worst-case $0.2\,\text{FP}_w$ from $\sim 44,000$ to $\le 0.2 \times 37,654 = 7,530.8$, lifting the live score by $+0.1453$ (`0.0884` $\to$ `0.2337`).

2. **The $B = 2\text{ px}$ ($200\text{ m}$) Near-Catalogue Exclusion Moat (`D2.8`, `0.2337` $\to$ `H33`, `0.2778` owner-reported):**
   - Direct inspection of the restored `.cache/registry/h33.tif` (`sha256=c55bafc4...`) against `D2.8` (`sha256=91eae1ca...`) proves that both rasters contain exactly $37,654$ binary $\{0, 1\}$ dots with minimum separation $2.828\text{ px}$ and $0$ dots on $C$, and share $25,479$ identical dots ($67.67\%$).
   - Their sole structural difference is the near-catalogue distance distribution $d_1(x, C)$:
     - In `D2.8`, **$12,175$ dots ($32.33\%$)** lie in the $0 < d_1(x, C) \le 2.0\text{ px}$ ($100\text{–}200\text{ m}$) collar immediately bordering mapped USGS faults.
     - In `H33`, **$0$ dots ($0.00\%$)** lie at $d_1(x, C) \le 2.0\text{ px}$ ($\min d_1 = 2.236\text{ px} = \sqrt{5}\text{ px}$). All $12,175$ moat dots were stripped and reallocated to off-catalogue continuations at $d_1(x, C) > 2.0\text{ px}$.
   - **Why did stripping the $200\text{ m}$ moat gain $+0.0441$?** Because any classifier trained on $y = \mathbf{1}_{x \in C}$ learns the $100\text{–}200\text{ m}$ rasterization/topographic halo of *already-mapped* faults, where $P(x)$ is artificially highest. Yet by Organizer Forum Thread 11516, known faults $C$ are masked and ground truth $G$ consists of *newly mapped uncatalogued faults*, which do not duplicate an existing trace $100\text{–}200\text{ m}$ away. Those $12,175$ halo dots therefore generated almost zero $\text{TP}_w$ while adding $0.2 \times 12,175 = 2,435$ denominator penalty units; reallocating them to genuine stepovers and tip continuations both cut $\text{FP}_w$ and added new $\text{TP}_w$.

3. **Exact Marginal Emission Criterion (Correcting `IR-02`):**
   - Given current dot set $D$ with score $S = \text{TP}_w / (\text{TP}_w + \alpha\,\text{FP}_w + \beta(|G| - \text{TP}_w))$, adding one binary dot at $x \notin D$ produces marginal coverage gain $\Delta \text{TP}_w(x \mid D) = \sum_{g \in G} \max\!\bigl(0,\, K(\|x - g\|_2) - c_D(g)\bigr)$ and marginal false-positive cost $\Delta \text{FP}_w(x) = 1 - q_G(x)$.
   - The new score exceeds $S$ if and only if:
     $$\frac{\Delta \text{TP}_w(x \mid D)}{1 - q_G(x)} > \frac{\alpha S}{\beta S + (1 - S)} = \frac{0.2\,S}{1 - 0.2\,S} \equiv \tau(S)$$
   - At $S = 0.2778$, $\tau(0.2778) = 0.0588$ coverage-weighted truth pixels per unit false-positive weight (and at the leaderboard high $S = 0.4413$, $\tau(0.4413) = 0.0968$).

### 1.3 Why `H33` Plateaued at `0.2778` and What Is Required to Reach `0.32+` / `0.4413`
`H33` remains limited by three physical and statistical bottlenecks:
1. **The Relay-Stepover Self-Collapse Tautology (`IR-H84-01`):** When a fault predictor is trained on $y=1 \iff x \in V$ using visible-catalogue structural features, every training positive lies *on* a visible component ($d_1(x, V) = 0$). Consequently, the two-host medial relay ratio $r(x) = d_1 / (d_1 + d_2)$ and stepover facing cosine $-\nabla d_1 \cdot \nabla d_2$ collapse to $0$ on $100\%$ of training positives. The model is structurally blind to en echelon stepover corridors ($3 < d_1 + d_2 \le 35\text{ px}$) where uncatalogued relay-ramp cross-faults actually transfer displacement.
2. **Lack of Geodetic Strain & Basement-Step Transfer Coupling:** `H33`'s backbone (`h19-5`) weighted topographic openness and thermal proximity equally inside and outside active strain corridors, emitting false positives along Pleistocene Lake Lahontan ($1332\text{ m}$) and Lake Bonneville ($1552\text{ m}$) paleoshorelines and unbroken volcanic flow margins.
3. **Deconfounded Leave-Bucket-of-Components-Out (`B-LBCO`) Relay Training:** By partitioning the $3\text{ px}$-buffered components of $V$ into $B=5$ buckets $V = \bigsqcup_{b=0}^4 V_b$ (`hide_frac=0.20`) and computing the bitplane two-host frame on $V \setminus V_b$ for training samples in bucket $b$, training positives $x \in V_b$ sit at realistic inter-component stepover distances $d_1(x, V \setminus V_b) > 3\text{ px}$ and $d_2(x, V \setminus V_b) > d_1$, matching the exact structural distribution of withheld uncatalogued fault segments.

---

## Part 2: Five Ranked Candidate Geological Hypotheses (`H84` Series)

None of the five hypotheses below has been implemented in `GEMSDOE77` prior to this session. They are ranked by expected `HOLDOUT-DTI` gain divided by implementation cost.

| Rank | ID | Hypothesis & Specific Layers | Physical Signature & Mathematical Transform | Named Non-Fault Mimic | Why It Catches Uncatalogued Faults & Differs from Repo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 (Implemented)** | **`H84-RAMP`** | **Deconfounded Two-Host En-Echelon Relay-Ramp & Geodetic-Geophysical Stepover Transfer Corridor**<br>Layers: `existing_faults.tif` (`visible` fold subset only in holdout) + `training_features.tif` Bands 7 (`geod_shearrate`), 8 (`geod_dilaterate`), 4 (`geod_2ndinv`), 18 (`iso_grav_anom_hg`), 3 (`tmi_hg`), 15 (`depth_to_base_surf`), 12 (`det_elev`), 19 (`det_elev_slope`). | Uses `vendor/gems57/relay_bend_anatomy.py` (`two_host_frame` exact $\lceil\log_2(N_c+1)\rceil$-bitplane EDT to nearest two distinct components $C_1 \neq C_2$ and `host_bend_fields` spin-2 multi-scale $\sigma=3$ vs $9\text{ px}$ turning) deconfounded via 5-bucket `B-LBCO` on training positives, coupled with transtensional geodetic strain ($z(\dot\gamma)\cdot z(|\dot\Delta|)$) and across-ramp basement/gravity gradient bridges ($\log(1 + |\nabla Z_{\text{base}}|\cdot |\nabla g_{\text{iso}}|)$). | Coalescing alluvial-fan interfluve sutures (bajada saddles) and buried paleochannel or volcanic dike bridges between adjacent bedrock spurs. | 1:250k–1:500k USGS maps capture master range-front segments ($C_1, C_2$) but omit lower-throw breaching cross-faults across tilted relay ramps (Faulds et al. 2006). First lane in `GEMSDOE77` to execute `vendor/gems57/relay_bend_anatomy.py` and solve `IR-H84-01` via 5-bucket `B-LBCO`. |
| **2** | **`H84-2`** | **Upward-Continuation Vertical Derivative Spectral Decay Ratio**<br>Layers: `training_features.tif` Bands 2 (`rtp`), 9 (`tmi_vg`), 13 (`iso_grav_anom`), 11 (`iso_grav_anom_vg`). | Ratio of vertical derivative to upward-continued horizontal gradient ($\text{TMI}_{\text{VG}} / |\nabla \mathcal{U}_{\Delta z}(\text{RTP})|$) isolating brittle upper-crustal fault steps ($0.2\text{–}1.5\text{ km}$) from deep plutonic roots ($>4\text{ km}$) and surface float ($<50\text{ m}$). | Shallow basaltic lava flow margins and terraced pediment gravel caps. | Catches intrabasin faults buried under $50\text{–}300\text{ m}$ of Quaternary alluvium where surface scarps are eroded. Differs from `H77G` (raw TMI troughs) by using vertical-to-upward-continued spectral decay. |
| **3** | **`H84-3`** | **Dip-Projected Microseismic Swarm & Potency Asymmetry**<br>Layers: `training_features.tif` Bands 10 (`deq_n100a15`), 16 (`ieq_n100a15`), 15 (`depth_to_base_surf`). | Cross-strike offset between earthquake density (`deq_n100a15`) and intensity (`ieq_n100a15`) back-projected up a $60^\circ$ normal-fault dip vector from basement depth $Z_{\text{base}}$ to the surface trace. | Geothermal injection/production induced seismicity clusters and caldera swarm clouds. | Catches blind seismogenic faults whose hypocentral clouds sit $2\text{–}5\text{ km}$ downdip in the hanging wall of an unmapped surface trace. Differs from `H77-TSD` (isotropic scalar difference). |
| **4** | **`H84-4`** | **Radiometric $\text{U}/\text{Th}$ Redox Mobilization Front $\times$ MT Conductance Margin**<br>Layers: `external/geodawn_rad_u8.tif` (`rad_u`, `rad_th`, `rad_u_th`) + `training_features.tif` Band 17 (`cond_surf`). | Cross-strike step in soluble hexavalent uranium vs immobile thorium ($\text{U}/\text{Th}$) co-aligned with smectite clay-cap conductance boundaries ($|\nabla \text{cond\_surf}|$). | Evaporitic playa uranium enrichment and felsic ash-flow tuff contacts. | Oxidizing geothermal/meteoric fluid upflow leaches and re-precipitates $\text{U}^{6+}$ along unmapped fault conduits lacking topographic scarps. Differs from `H79-RH` ($\text{K}/\text{Th}$) and `H82-TCSTEP` ($\text{TC}$). |
| **5** | **`H84-5`** | **Piedmont Fan-Head Trenching & Canyon-Spacing Periodicity Deficit**<br>Layers: `training_features.tif` Bands 12 (`det_elev`), 19 (`det_elev_slope`) + `external/lidar_scarp_features_u8.tif`. | Local 1-D wavelet variance of drainage incision spacing along range-front contours, detecting breakdown of regular Wallace (1978) / Hovius (1996) outlet spacing across active piedmont splays. | Lithologic sill resistance benches and Pleistocene lake highstand wave-cut notches. | Catches distributed piedmont fault splays that trench alluvial fan heads without forming continuous linear scarps. Differs from `H77E-Knick` (stream $\chi / k_{sn}$). |

---

## Part 3: Preregistered Experimental Protocol (`H84-RAMP`, Max 3 Experiments)

1. **Experiment 1 (`exp1` — Single-Feature Leakage Canaries & Pre-Placement Surface Uniqueness):**
   - Compute all 8 `H84-RAMP` channels strictly from `visible` faults (in `seed=20` and `seed=30` folds) + `training_features.tif`.
   - Test each of the 8 `H84-RAMP` channels and 19 standard bands alone on the holdout set (`seed=20`, $11,321$ withheld positives; `seed=30`, $11,342$ withheld positives); stop if any feature has symmetric $\text{AUC} > 0.90$.
   - Document `IR-H84-01` (Relay-Stepover Self-Collapse Tautology) by comparing naive vs `B-LBCO` two-host relay statistics on visible training positives vs withheld holdout positives.
   - Check pre-placement continuous surface Spearman rank-correlation against all locally accessible + restored registry rasters (stop if $|\rho| > 0.90$).

2. **Experiment 2 (`exp2` — Dual-Seed Spatially Blocked Holdout Evaluation via `vendor/gems57/evaluate_holdout.py`):**
   - Evaluate `A_std_plus_h84` (19 std + 8 `B-LBCO` deconfounded `H84-RAMP` features), `A_std_plus_h84_naive` (ablation without `B-LBCO`), `A_std_control` (matched 19-band LOQO control), `S_h84_physics` (zero-training composite), and `A0_random` (uniform random dots) on both `seed=20` ($11,321$ withheld positives) and `seed=30` ($11,342$ withheld positives) with $2,000$ paired physical spatial-cluster bootstrap draws.

3. **Experiment 3 (`exp3` — Full-Catalogue LOQO Surface, Validated GeoTIFF via `submission_writer.py`, and Final-Dot Uniqueness Audit):**
   - Fit full-catalogue LOQO `H84-RAMP` surface, verify pre-placement Spearman $|\rho| \le 0.90$, emit $37,654$ binary dots ($d > 2.0\text{ px}$ from catalogue, $d_{\min} = 2.8\text{ px}$), package via `vendor/gems57/submission_writer.py`, validate via `vendor/gems57/validate.py`, and run Phase 2 final-dot uniqueness (`vendor/gems57/uniqueness.py`) across all local + restored registry rasters (reporting both informative sparse-raster overlap and the literal Policy I17 dense-witness stop).

---

## Part 4: Executed Experimental Receipts (`H84-E1`, `H84-E2`, `H84-E3`)

| Experiment | Receipt JSON | Key Measured Outcomes (`gems57-pooled-hide-v2`) | Gate Verdict |
| :--- | :--- | :--- | :--- |
| **`H84-E1` (Canary & Surface Scan)** | [`evidence/h84_e1_canary_surface.json`](../evidence/h84_e1_canary_surface.json) | Max `H84` single-feature symmetric $\text{AUC} = 0.8076$ (`seed=20`) / $0.8073$ (`seed=30`) $\le 0.90$; max `std19` symmetric $\text{AUC} = 0.5700$ (`seed=20`) / $0.5762$ (`seed=30`) $\le 0.90$. Phase 1 pre-placement surface $\max |\rho| = 0.4211 \le 0.90$ across $52$ checked rasters. Quantifies **`IR-H84-01`**: naive visible training collapses $7/8$ relay/collar channels to $0.0000$, whereas 5-bucket `B-LBCO` restores non-zero stepover means matching withheld truth. | **PASS** |
| **`H84-E2` (Dual-Seed Holdout)** | [`evidence/h84_e2_holdout.json`](../evidence/h84_e2_holdout.json) | **Seed 20 ($11,321$ withheld):** `A_std_plus_h84` = **`0.1366`** ($95\%$ CI $[0.1237, 0.1500]$), `S_h84_physics` = **`0.1423`** ($[0.1279, 0.1564]$), `A_std_plus_h84_blbco_only` = `0.1144`, `A_std_control` = `0.0757` ($\Delta = +0.0608$, CI $[+0.0414, +0.0798]$), `A_std_plus_h84_naive` = `0.0418` ($\Delta = +0.0948$, CI $[+0.0736, +0.1167]$), `A0_random` = `0.0445` ($\Delta = +0.0921$).<br>**Seed 30 ($11,342$ withheld):** `A_std_plus_h84` = **`0.1446`** ($95\%$ CI $[0.1277, 0.1619]$), `S_h84_physics` = **`0.1476`** ($[0.1288, 0.1666]$), `A_std_plus_h84_blbco_only` = `0.1190`, `A_std_control` = `0.0672` ($\Delta = +0.0774$, CI $[+0.0528, +0.0992]$), `A_std_plus_h84_naive` = `0.0331` ($\Delta = +0.1115$, CI $[+0.0875, +0.1371]$), `A0_random` = `0.0469` ($\Delta = +0.0977$). | **Beats controls on both seeds; below `0.152718` repo incumbent (`H77-Step`)** |
| **`H84-E3` (Build & Uniqueness)** | [`evidence/h84_e3_uniqueness.json`](../evidence/h84_e3_uniqueness.json)<br>[`evidence/h84_run_card.json`](../evidence/h84_run_card.json) | `gems77-h84-ramp-relay-37654dots-20261021-1b76b4b7-zeros.tif` (`sha256=f06ada5aa834ceaf65e8700e36bbd46eddb96929b42c9041c72c8369c1022838`, $144,529$ bytes, $15/15$ validator checks PASS, $37,654$ binary dots, $0$ on catalogue, $\min d_C = 5.00\text{ px}$). Informative sparse uniqueness PASS across all $51$ sparse rasters ($\max$ 3-px forward overlap $= 0.5299 \le 0.70$, $\max |\rho| = 0.0601 \le 0.90$; vs `H33` overlap $= 0.2265$, $\rho = +0.0057$). Literal Policy I17 fails closed ($1.000$ vs `dense_witness.tif`). | **NEGATIVE (OK to download for inspection; NOT OK to submit)** |

