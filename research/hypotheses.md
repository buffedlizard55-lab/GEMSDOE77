# GEMSDOE77 — candidate geological hypotheses (ranked before experiments)

*Written 2026-10-21 before Exp 1. Evidence classes: **[OFFICIAL]** organizer/USGS
source · **[MEASURED]** computed this session from SHA-pinned bytes ·
**[OWNER-REPORT]** family ledger without organizer receipt ·
**[BLOCKED]** not obtainable from this sandbox, with the check stated.*

Competition facts used below: task = predict faults **not** in the USGS/INGENIOUS
catalogue; hidden truth was manually mapped by experts; metric = distance-weighted
Tversky, α=0.2, β=0.8, triangular kernel R=300 m; grid EPSG:32611, 100 m,
3730×3292 **[OFFICIAL]** (DrivenData #306 problem page 967, fetched 2026-10-21).
Public leaderboard #1 = 0.4413 (xiaofanhu) **[OFFICIAL]** (leaderboard page fetched
2026-10-21). Family best `h33-h33-2-b2…-zeros` = 0.2778 **[OWNER-REPORT]** —
no organizer receipt was found in any inspected sibling; GEMSDOE32's own site
states no organizer score exists in that repository for the file.

## The band-6 correction that reopens the potential-field lane

**[MEASURED]** Band 6 of `training_features.tif` carries the embedded tag
*"tc — Tilt angle or total curvature — magnetic field derivative for edge detection"*.
 Numerically (200k footprint sample): Spearman(band 6, atan2(tmi_vg, tmi_hg)) =
+0.0200; Spearman(band 6, external GeoDAWN **total-count** radiometric grid) =
**+1.0000**; band 6 ∈ [2.95, 88.57], median 18.48. Band 6 is radiometric total
count. The shipped stack therefore contains **no tilt derivative at all**, and any
family channel built by trusting the tag used radiometrics under a tilt name
(`evidence/band6_resolution.json`). Correctly derived tilt/phase transforms are
genuinely untried inputs, not re-derivations of a shipped band.

## Ranked hypotheses

### H77-1 (rank 1 — primary lane, with H77-2/3/4 as its feature block) — Cross-field contact confluence with signed basement throw
- **Layers:** `tmi_hg` (3), `tmi_vg` (9), `rtp` (2), `iso_grav_anom` (13),
  `iso_grav_anom_vg` (11), `depth_to_base_surf` (15). Band 18 (`iso_grav_anom_hg`)
  is signed in the file (min < 0), so it is **not** a gradient magnitude; gravity
  THDR is recomputed from band 13 **[MEASURED in feature builder diagnostics]**.
- **Physical signature:** magnetic tilt `atan2(vg, THDR)` zero-crossing +
  analytic-signal amplitude ridge **coincident within ~1 px** with a gravity-tilt
  edge, multiplied by the directional step of basement depth across the edge
  (deeper on the downthrown side). Monogenic **phase congruency** (H77-3) and
  **pseudogravity–Poisson local slope/correlation** (H77-4) enter the same block
  as contrast-invariant contact evidence.
- **Why a catalogue-missing fault:** buried intra-basin basement faults under
  alluvial cover have no scarp to map in the field, but they offset magnetic/
  dense basement and thicken fill across the trace — exactly what an expert with
  GeoDAWN aeromagnetics adds that a surface-geology catalogue cannot contain.
- **Named non-fault mimic:** intrusive/lithologic contacts and caldera margins
  (magnetic+gic edges without throw-sign basement step); GeoDAWN survey-block
  seams (straight, acquisition-parallel).
- **Difference vs implemented work:** GEMSDOE70 E1 tested unsigned RTP-gradient
  *termination* at a gravity boundary (negative, HOLDOUT-DTI 0.0102 < random
  0.0472 in that repo's record). No inspected lane uses tilt **phase**
  (zero-crossing) coincidence of two independent potential fields gated by the
  *sign* of the basement step; family "tilt" channels to date were band 6 = TC.
- **Cost:** low–medium (FFT/gradient transforms on cached bands). Validatable
  locally: yes (hide-and-recover).

### H77-2 (rank 2 — same lane's geomorphic block) — Range-front-base slope asymmetry
- **Layers:** `det_elev` (12), `det_elev_slope` (19), strain bands only as
  sampled controls.
- **Physical signature:** “frontness” = (max slope within 500 m − local slope) at
  flat pixels × structure-tensor coherence of detrended elevation, plus signed
  profile curvature. It locates the **base** of a linear mountain front
  (where the fault trace sits on the piedmont), not the scarp face itself.
- **Why a catalogue-missing fault:** young/low-slip range-front faults whose
  scarp is buried by fan gravels still hold a straight front; Qfaults mapping
  keyed to scarps/offset surfaces can omit them while lidar/DEM experts can
  still draw the front.
- **Named non-fault mimic:** pluvial Lake Lahontan shoreline benches
  (level, contour-parallel fronts); resistant-bedrock dip slopes.
- **Difference vs implemented work:** family DEM lanes use scarp curvature/step
  ridges (the scarp), unsigned Hessian ridges, or 1 m lidar step maxima. A
  flat-pixel-adjacent-to-steep-front asymmetry was not found in GEMSDOE32/70
  (grep: sinuosity/facet/front-base, 2026-10-21).
- **Cost:** low. Validatable locally: yes.

### H77-3 (rank 3 — inside H77-1 block) — Monogenic phase-congruency contacts
- **Layers:** `rtp` (2), `iso_grav_anom` (13).
- **Signature:** local phase alignment across log-Gabor scales (Riesz/monogenic),
  which locates a contact independently of its amplitude — deep, weak contacts
  localize as sharply as shallow ones.
- **Mimic:** flight-line/survey seams (also phase-congruent); coastline of the
  valid-data footprint (masked by 12 px domain erosion).
- **Difference:** no PC/monogenic/Riesz implementation in inspected repos.
- **Cost:** medium (FFT filter bank, CPU). Validatable: yes, after a synthetic
  step-edge unit test.

### H77-4 (rank 4 — inside H77-1 block) — Pseudogravity–Poisson slope breaks
- **Layers:** `rtp` (2) → pseudogravity (FFT 1/|k| integration), `iso_grav_anom` (13).
- **Signature:** where magnetized basement and density share sources, gravity ∝
  pseudogravity (Poisson relation). A fault that juxtaposes magnetic basement
  against non-magnetic fill breaks the local slope/correlation (25 px windows,
  high-passed).
- **Mimic:** remanently magnetized volcanics; intrusions with anomalous
  magnetization/density ratio.
- **Difference:** no pseudogravity/Poisson use in inspected repos.
- **Cost:** medium. Validatable: yes.

### H77-5 (rank 5) — Sub-km seismicity lineaments / 2020 Monte Cristo rupture — **[BLOCKED] here**
- **Layers needed (specific free official source):** USGS ComCat via the FDSN
  event service `https://earthquake.usgs.gov/fdsnws/event/1/query` (free,
  public; includes moment tensors), or the Nevada Seismological Laboratory
  catalogue; NBMG 2020 Monte Cristo surface-rupture mapping.
- **Obtainability check performed 2026-10-21:** `curl` to
  `earthquake.usgs.gov` from this sandbox returns HTTP 000 (exit 35) — host is
  outside the sandbox egress allow-list. GitHub code search found no owner
  mirror of an earthquake catalogue. **Named but not viable in this session**;
  no seismicity surface is fabricated. The shipped seismic bands (10, 16) are
  100 km-radius aggregates and cannot localize a fault at 300 m (family
  autocorrelation measurements, GEMSDOE32 H33-E).

## Decision rule (frozen in `registry/preregistration.json`)

Exp 1 trains fold-wise HGB rankers (whole-component quadrant holdout, seed 30)
for: standard block, new-physics block (H77-1…4), combined (primary, no lidar),
combined+lidar (ablation only), each new feature alone (leakage canary,
symmetric AUC > 0.90 stops the run), and a matched random control. The primary
candidate is promoted only if pooled HOLDOUT-DTI beats the standard block **and**
random with paired 95% CIs excluding 0, uniqueness passes (ρ ≤ 0.90, mutual
3-px overlap ≤ 0.70 vs every locally hash-verified registry raster), and the
shared writer's on-disk validation passes. Otherwise the run card verdict is
**negative** and the site must say DO NOT SUBMIT. No weekly slot is spent here.
