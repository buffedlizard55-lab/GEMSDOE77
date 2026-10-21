# H82S — candidate geological hypotheses (ranked before Exp 1)

*Written 2026-10-21, before any H82S experiment ran. Evidence classes:
**[OFFICIAL]** organizer/USGS source · **[MEASURED]** computed this session
from SHA-pinned bytes · **[OWNER-REPORT]** family ledger without an organizer
receipt · **[BLOCKED]** not obtainable from this sandbox, with the exact check
stated.*

## Why a new lane, and what it must avoid duplicating

`research/irregularities.md`, `AGENTS.md` and the per-lane run cards
(`evidence/run_card*.json`, `evidence/h7*_run_card.json`,
`evidence/h8*_run_card.json`, `evidence/run_card_xsen.json`,
`evidence/ago_run_card.json`) record roughly fifteen prior GEMSDOE77 lanes,
all measured with the same pinned evaluator (`gems57-pooled-hide-v2`) and all
**negative** against the retained local incumbent **0.152718 HOLDOUT-DTI**
(H77-Step, seed 20, 11,321 withheld positives). Already tried and closed:
RTP/gravity tilt zero-crossing + analytic-signal confluence + signed basement
step + monogenic phase congruency + pseudogravity-Poisson slope (contact-physics
lane, `registry/preregistration_contactphys.json`); front-base slope asymmetry
(same lane); Euler deconvolution (SI=0) + upward-continuation phase coherence
(H77E, `research/h77e_hypotheses.md`); magnetite-destruction trough × gravity
step × MT clay-cap margin (H77G); 1 m/10 m lidar scarp consensus off the H19-5
support (H78, H77D, H81); Riedel R/R′ wing cracks from fault tips (H80); multi
-scale tilt-angle persistence (H80T); lagged radiometric K-Th/U halo (H79-RH);
cross-field edge stability across continuation scales (H79-ORIENT); slip-budget
tip continuation cones (H81); multi-scale magnetic analytic-signal × radiometric
composite (XSEN); polarity-asymmetric grav-mag halo pruning of an ML backbone
(AGO, archived). **None of these is a frequency-domain (spectral) source-depth
estimate.** `grep -ril "curie|spectral centroid|power spectrum"` across the
whole repository returns exactly one hit before this document
(`research/hypotheses-xsen-2026-10-21.md`, which *rejected* the idea on the
mistaken premise that Curie-depth needs a new external heat-flow compilation —
corrected below).

Competition facts used: task = predict faults absent from the USGS/INGENIOUS
catalogue; hidden truth was expert-mapped; metric = distance-weighted Tversky,
alpha=0.2, beta=0.8, 300 m triangular kernel; grid EPSG:32611, 100 m,
3730x3292 **[OFFICIAL]** (problem page 967, re-read 2026-10-21). Public
leaderboard #1 = 0.4413 **[OFFICIAL]** (leaderboard page, re-read 2026-10-21).
Family `h33-h33-2-b2…zeros` = 0.2778 **[OWNER-REPORT]**, no organizer receipt
found in GEMSDOE32's own site (`research/why-h33-scored-02778.md`).

## Ranked hypotheses

### H82S-1 (rank 1 — implemented this session) — Windowed spectral magnetic source-depth gradient ("Spector & Grant" single-slope depth index)

- **Layer:** `rtp` (band 2 of the cached `training_features.tif`, already
  restored and SHA-256 verified this session, see `evidence/h82_data_receipt.json`).
  No other band, no catalogue, no lidar.
- **Physical signature:** tile the RTP grid into non-overlapping 32x32-pixel
  (3.2 km) Hann-tapered blocks; take the 2-D FFT; azimuthally average the
  power spectrum into radial wavenumber bins; fit `ln(amplitude)` vs
  wavenumber `k` over a fixed mid-band (excludes the DC bin and the noisiest
  high-frequency bins); the fitted slope's negative is the classical
  Spector & Grant (1970) **depth to the statistical ensemble of magnetic
  sources** for that block. Bilinearly resample the per-block depth grid back
  to the full 100 m grid and take its spatial gradient magnitude: this is the
  candidate surface (a step/edge in the regional source-depth field, not in
  the field's amplitude).
- **Why this can catch a fault missing from the catalogue:** this is a
  frequency-domain, ensemble-statistical estimate of how deep the *population*
  of magnetized sources sits under each 3.2 km tile. A blind normal fault
  that offsets basement (thickens/thins cover, and/or juxtaposes hotter,
  more demagnetized crust against cooler crust) produces a lateral *step in
  that depth estimate* even where the field's raw amplitude, tilt,
  zero-crossing, or Euler-style point solutions (all already tried and
  negative here) show no strong local anomaly, because the spectral estimate
  averages out the small-scale amplitude texture that those spatial-domain
  methods key on.
- **Named non-fault mimic:** volcanic edifices/calderas with a locally
  shallow magnetic/thermal source population but no bounding structural
  offset; GeoDAWN flight-line tie-spacing aliasing at the block's Nyquist
  scale (would show up as a spurious, flight-line-parallel straight edge,
  distinguishable from the fan-shaped/strike-consistent edges of structural
  steps).
- **Difference from everything already implemented here:** H77E tested
  spatial-domain Euler deconvolution (homogeneity equation, requires a
  structural index) and upward-continuation phase persistence; contact-physics
  tested spatial-domain tilt zero-crossings, analytic-signal ridges and
  monogenic phase congruency. H82S-1 is the first **frequency-domain radial
  power-spectrum** depth estimate in this repository or any inspected sibling
  script search.
- **Cost / obtainability:** **Low.** Needs only the already-cached,
  SHA-pinned RTP band; no new external data; one FFT pass over ~12,000 blocks
  runs in well under a minute on CPU. **[MEASURED] obtainable now** — this
  corrects the prior session's mistaken BLOCKED verdict
  (`research/hypotheses-xsen-2026-10-21.md` row 5), which conflated this
  single-slope estimate with the two-depth centroid Curie-point method (H82S-2
  below), which *does* need more care.

### H82S-2 (rank 2 — not implemented this session) — Two-depth centroid Curie-point-depth (Bouligand/Okubo) fused with the cached INGENIOUS probe halo

- **Layers:** `rtp` (band 2) for the centroid/top-depth fit; the already
  -cached INGENIOUS 2 m temperature-probe DAB halo and GDR 1391
  springs/geothermometers that `src/gems77_step/features.py` already parses
  (`hydro_upflow_near_400m`, `hydro_upflow_far_1400m`).
- **Signature:** fit both the top depth `Zt` and centroid depth `Zo` of the
  magnetic source ensemble per window (iterative two-parameter spectral fit,
  not H82S-1's single slope); Curie depth `Zc = 2*Zo - Zt`
  (Bouligand, Glen & Blakely, 2009, *JGR*); then require a Curie-depth edge
  to be spatially coincident with the independent, already-cached
  hydrothermal-probe halo before crediting a pixel.
- **Why uncatalogued:** couples two independent physical observables
  (basement/Curie-isotherm depth from magnetics; measured shallow heat from
  probes) that neither existing lane combines; a coincidence requirement
  should suppress the volcanic-dome mimic that threatens H82S-1 alone (calderas
  have shallow Curie depth AND often a probe signal, but a *fault-bounding
  step* in Curie depth next to the probe halo is the more specific claim).
- **Mimic:** calderas/intrusive centers with both a shallow Curie-depth dome
  and a conductive probe halo but no bounding linear structural offset.
- **Cost:** **Medium-high.** The two-parameter de-fractal spectral fit is
  iterative and numerically more fragile than H82S-1's closed-form slope
  (window size and fractal-exponent choices materially change `Zo`/`Zt` in
  the published method); needs a dedicated synthetic-source unit test before
  any holdout run. Data is already fully cached; no new external source
  needed. Deferred to a future session's budget, not attempted here.

### H82S-3 (rank 3 — not implemented this session) — Beheaded-drainage / wind-gap network asymmetry from the cached chi-network bands

- **Layers:** `chi_step_max`, `chi_log10_area`, `chi_zmean_m` (already
  computed by `src/gems77_step/features.py` from the cached 10 m 3DEP-derived
  DEM) repurposed at the **network topology** level rather than the per-pixel
  steepness-index level H77-Step/H77E-knick already use.
- **Signature:** detect asymmetric channel capture across a candidate line —
  one side's drainage network has anomalously low channel count/order for its
  catchment area (beheaded), paired with an adjacent captured channel with
  anomalously high order for its area — a classical marker of active
  range-front tilting (Wallace, 1978, *GSA Bulletin*) distinct from per-pixel
  knickpoint/steepness ridges.
- **Why uncatalogued:** targets *network-scale* drainage reorganization caused
  by blind tilting, not a single channel's local gradient break (which
  `research/h77e_knick_hypotheses.md` already tested and which failed).
- **Mimic:** climate-driven alluvial-fan avulsion with no tectonic cause.
- **Cost:** **Medium**, implementation risk from reconstructing a flow-routing
  graph correctly; needs confirmation that the *raw* 10 m DEM (not just the
  already-reduced `chi_*` scalar channels) is present in the restorable cache
  before a graph can be rebuilt. Not attempted this session; flagged for a
  future lane with an explicit synthetic-network unit test.

### H82S-4 (rank 4 — **[BLOCKED]**) — Self-potential / audio-magnetotelluric resistivity edges

- **Layers needed:** a self-potential or AMT/MT resistivity grid over the AOI.
- **Specific free official source:** GDR submission 1391
  (`https://gdr.openei.org/submissions/1391`) is the in-repo-cited GDR entry
  point for the INGENIOUS project's auxiliary geophysics; no SP/AMT grid for
  this AOI is listed in `registry/data_manifest.json`, and this host is
  outside the sandbox's allowed egress list
  (`github.com`, `codeload.github.com`, `api.github.com`, `registry.npmjs.org`,
  `pypi.org`, `files.pythonhosted.org`). **Obtainability not verified this
  session; not proposed as viable** until either the host is reachable or an
  owner GitHub mirror of a digitized SP/AMT grid is found (`gh search code` for
  "AMT" / "resistivity" / "self-potential" across `buffedlizard55-lab` returned
  no raster hit at the time of writing — see `evidence/h82spec_source_check.json`).

### H82S-5 (rank 5 — **[BLOCKED]**) — Sentinel-1 InSAR coherence/velocity gradient

- **Layers needed:** Sentinel-1 SLC/InSAR coherence or line-of-sight velocity
  over the AOI (present-day aseismic creep across a blind fault).
- **Specific free official source:** Copernicus Dataspace Ecosystem
  (`https://dataspace.copernicus.eu/`) or the ASF DAAC
  (`https://search.asf.alaska.edu/`), both free and official but outside the
  sandbox's allowed egress list. No owner GitHub mirror of an AOI InSAR
  product exists in `registry/data_manifest.json`. **Not obtainable from this
  sandbox; not proposed as viable** until that specific access gap is closed.

## Decision rule (frozen before Exp 1; see `registry/preregistration_h82spec.json`)

Exp 1 = surface + two feature-alone leakage canaries (`depth_raw` rank,
`depth_gradient` rank) on the shared buffered-whole-component holdout
(`gems57.holdout.buffered_component_draw`, seed 20). Any canary with symmetric
AUC > 0.90 stops the lane (leakage). Exp 2 = pooled HOLDOUT-DTI
(`gems57-pooled-hide-v2`) for the gradient surface vs a matched random control
at the same emitted-dot budget, with paired 95% CI from
`gems57.evaluate_holdout.pooled_summary`. Promotion requires beating the
0.152718 incumbent (not just the random control) with the paired/cross-draw
evidence disclosed, uniqueness scan, and the shared writer's validation —
**no weekly slot is spent unless Exp 2 clears that bar.** If Exp 1 or Exp 2 is
negative, the run card verdict is **negative**, the finding is the
deliverable, and H82S-2/H82S-3 remain open proposals for a future session's
budget (not retried here).
