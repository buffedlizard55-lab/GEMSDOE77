# Knowledge Synthesis 03: Spectral Mesh De-Aliasing (`IR-H77G-01`), Hydrothermal Demagnetization Coupling (`IR-H77G-02`/`03`), and Kernel-Convolved Lagrangian Submodular Cover (`H77V`)

*Date: 2026-10-21 | Session Branch: `arena/65c05dc9-gemsdoe77` | Lane: `H77V`*

## 1. Resolution of `IR-H77G-01`: Why $\sigma = 36\text{ px}$ ($3.6\text{ km}$) Low-Pass De-Aliasing Is Essential for Regional Crustal Bands
In `training_features.tif`, five regional geophysical/geodetic/seismic bands were interpolated onto the 100 m `EPSG:32611` grid from coarse rectilinear inversion meshes:
- **Band 17 (`cond_surf`):** 3-D magnetotelluric surface conductance (`Peacock & Bedrosian 2022`, [DOI 10.5066/P9TWT2LU](https://doi.org/10.5066/P9TWT2LU)), where `30.5%` of adjacent 100 m pixel differences are identically zero and the 1-D spatial FFT exhibits sharp harmonic peaks at `35.02 px` (`3.5 km`) and `70.04 px` (`7.0 km`).
- **Bands 4, 7, 8 (`geod_2ndinv`, `geod_shearrate`, `geod_dilaterate`):** GPS strain-rate tensor invariants (`Kreemer & Young 2022`, [DOI 10.1785/0220220153](https://doi.org/10.1785/0220220153)), with `23.5%–33.1%` zero steps across mesh cell boundaries.
- **Bands 10, 16 (`deq_n100a15`, `ieq_n100a15`):** Declustered earthquake depth and intensity kernels, with `31.9%` zero steps.

Prior lanes (`H77-Step` in `src/gems77_step/features.py:269` at $\sigma = 1.5\text{ px}$, and `H77E` in `src/gems77e/features.py:409` at $\sigma = 2.5\text{ px}$) differentiated `cond_surf` at scales $\sigma \ll 35\text{ px}$, which acted as an edge detector on the rectilinear `3.5 km / 7.0 km` MT inversion mesh rather than isolating physical smectite clay-cap margins.

In **`H77V`**, we apply a Gaussian low-pass filter with $\sigma = 36.0\text{ px}$ ($3.6\text{ km} > 35.02\text{ px}$) prior to computing:
1. `h77v_mt_dealiased_3p6km`: Mesh-dealiased log-conductance $G_{36}(\log(1 + \sigma_{\text{MT}}))$;
2. `h77v_mt_claycap_margin_3p6km`: Mesh-dealiased clay-cap lateral gradient $|\nabla G_{36}(\log(1 + \sigma_{\text{MT}}))|$;
3. `h77v_transtensional_regime_3p6km`: Mesh-dealiased regional transtensional strain-seismicity modulator $\text{rank}(G_{36}(\dot{\varepsilon}_{\text{shear}})) \cdot \text{rank}(G_{36}(|\dot{\varepsilon}_{\text{dilat}}|)) \cdot \text{rank}(G_{36}(I_{\text{eq}}))$.

## 2. Resolution of `IR-H77G-02` and `IR-H77G-03`: Multi-Scale Tubular Demagnetization Troughs & Gravity-Basement Concordance Coupled with 3DEP Scarp-Knickpoint Conduits
As established in `docs/review-2026-10-21-h77g.md`:
- `src/gems77/features77.py:hessian_ridge` had an eigenvalue sign/ordering bug (`IR-H77G-02`) that multiplied the across-ridge curvature $\lambda_2$ by the *along-ridge* curvature $\lambda_1$, suppressing continuous linear troughs (`lambda_1 approx 0`) and firing on circular bulls-eyes.
- Standalone 10-channel potential-field troughs (`H77G`, `0.03915 HOLDOUT-DTI` pure / `0.06191 HOLDOUT-DTI` `19 std + 10 H77G`) lack the 100–300 m positional precision required by the $R = 300\text{ m}$ triangular DTI kernel unless coupled with 1 m LiDAR / 10 m 3DEP scarp crests, 10 m 3DEP fluvial knickpoints, and fold-visible `SLOCO` structural strike-continuation (`IR-H77G-03`).

In **`H77V`**, we compute:
- **Corrected across-valley tubular Hessian demagnetization troughs** on $-G_\sigma(\text{RTP})$ at $\sigma \in \{1.5, 3.0, 6.0\}\text{ px}$ ($150\text{ m}, 300\text{ m}, 600\text{ m}$):
  $$\tau_\sigma(x) = \max\big(\lambda_1(-H_\sigma(\text{RTP})), 0\big) \cdot \left(1 - \frac{\min(|\lambda_2|, \lambda_1)}{\lambda_1 + \epsilon}\right), \quad P_{\text{demag}}(x) = \big(\tau_{1.5}(x)\,\tau_{3.0}(x)\,\tau_{6.0}(x)\big)^{1/3}.$$
- **Antiparallel gravity-basement normal-fault step concordance** (`h77v_grav_base_step_concord`):
  $$C_{\text{GB}}(x) = \max\big(-\nabla G_2(g_{\text{iso}}) \cdot \nabla G_2(Z_{\text{base}}),\, 0\big)^{1/2},$$
  which is strictly positive where isostatic gravity drops into the basin directly down the dip of the deepening pre-Tertiary basement surface (`Band 15`), and zero where gravity and basement gradients are orthogonal or parallel (e.g. intrabasinal dense volcanic sills).
- **Direct physical interaction terms (`IR-H77G-03`):**
  - `h77v_scarp_x_demag_conduit`: Road- and mining-claim-deconfounded `1 m LiDAR / 10 m 3DEP` scarp crest multiplied by multi-scale tubular demagnetization persistence;
  - `h77v_knick_x_grav_base_step`: `10 m 3DEP` fluvial knickpoint rate (`dem10_chi_100m_u8.tif`) multiplied by antiparallel gravity-basement step concordance;
  - `h77v_vent_x_dealiased_mt_margin`: DOE GDR 1391 chemical geothermometry disequilibrium ($\Delta T_{\text{chem}}$) + 2 m temperature probe anomaly (`F2mDAB`) + paleo-geothermal sinter/travertine/tufa aprons multiplied by the de-aliased 3.6 km MT clay-cap lateral gradient;
  - `h77v_scarp_x_transtensional_regime`: Deconfounded scarp crest modulated by the 3.6 km de-aliased transtensional regime;
  - `h77v_blm_mining_deconf_dist_log`: Log Euclidean distance $\log(1 + d_{\text{BLM}})$ from historic mining claims (`blm_closed_claim_distance_m.tif`) to separate anthropogenic prospect pits/waste dumps from tectonic fault scarps.

## 3. Resolution of the Uniqueness-vs-Holdout Tradeoff: Kernel-Convolved Lagrangian-Priced Dinkelbach Submodular Cover (`LDSA`)
In `H77E-Knick` (`PR #45`), the pure submodular candidate achieved `0.155623 HOLDOUT-DTI` (Seed 20) and `0.154666 HOLDOUT-DTI` (Seed 30), whereas the portfolio-unique submodular twin dropped by `-0.0289 HOLDOUT-DTI` to `0.126664` (Seed 20) and `0.124727` (Seed 30). Forensic inspection of `src/gems77e/allocate.py` reveals three exact causes of that `-0.0289` loss:
1. **Pointwise candidate pre-screening (`line 117`):** `H77E` pre-filtered candidate dot locations by single-pixel `prob[x]` (`top 115,000`) rather than by the 29-pixel 300 m triangular kernel integral $U_0(x) = \sum_{\|y-x\|<3} p(y)\,k(\|y-x\|)$. Consequently, halo-free saddle pixels located $3.16\text{ px} = \sqrt{10}\text{ px} > 3.0\text{ px}$ from prior dot centers (which lie outside the 3-px overlap halo of prior submissions while still projecting their 300 m kernel across the fault trace!) were excluded from the candidate pool whenever their single center pixel lay $1\text{–}2\text{ px}$ off the sharp ridge crest.
2. **Hard-cliff capacity truncation (`line 168`):** `H77E` placed zero shadow price ($\mu_r = 0$) on binding prior-raster halos for the first `24,382` dots (`66.8%` of `36,500`), burning its entire overlap budget on early selections where halo-free neighbor pixels at $3.16\text{ px}$ had nearly identical kernel coverage, and then hard-banning `100%` of remaining pixels in those halos for the final `12,118` dots.
3. **Forced full-budget emission after halo exhaustion:** Once collar halos saturated, forcing `36,500` total dots compelled the greedy loop to emit low-gain dots that inflated $\alpha\,\text{FP}_w$.

In **`H77V`**, our **Kernel-Convolved Lagrangian-Priced Dinkelbach Submodular Allocator (`LDSA`)**:
1. Computes the exact 29-pixel kernel-convolved expected TP coverage $U_0(x) = \sum_{\|y-x\|<3} p_{\text{eff}}(y)\,k(\|y-x\|_2)$ and true kernel-neighborhood self-credit $k_{\text{self}}(x) = \max_{\|y-x\|<3} p_{\text{eff}}(y)\,k(\|y-x\|_2)$ across the allowed domain (`d_vis > 2.0 px`), pre-screening the top `180,000` kernel-integral candidates;
2. Applies a smooth **convex Lagrangian congestion price** $\Phi(x; \mathbf{c}) = \sum_{r \in \mathcal{R}_{\text{info}}: x \in H_r} \eta \left(\frac{c_r}{C_r}\right)^\gamma$ (plus the strict hard barrier $c_r < C_r = \lfloor \tau_{\max} N_{\text{emit}} \rfloor$ with $\tau_{\max} \le 0.65 < 0.70$) inside the lazy greedy priority queue so the solver continuously prefers halo-free saddle positions ($\|x - x_{\text{prior}}\|_2 \ge \sqrt{10} \approx 3.16\text{ px}$) whenever they deliver comparable 300 m kernel coverage;
3. Terminates when either the target dot count or the Dinkelbach fractional marginal threshold $\Delta \text{TP}_w / \Delta D \le \lambda_{\min}$ is reached, and performs a final post-audit pass guaranteeing `0` informative firings (`max forward 3-px overlap <= 0.65 <= 0.70` across all non-space-filling rasters).
