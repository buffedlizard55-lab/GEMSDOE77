# H82 pre-experiment hypotheses (2026-10-21, branch `arena/ff72e371-gemsdoe77`)

Written **before** any model was fitted in this session. Ranked by expected
HOLDOUT-DTI improvement per unit implementation cost. Only candidate **C1** was
implemented; the others are recorded as deliverable negative/parked design work
with the reason they were not chosen.

Operating constraints that shaped the ranking (measured, not assumed):

* Outbound network in this sandbox reaches only `github.com`, `api.github.com`,
  `codeload.github.com`, `registry.npmjs.org`, `pypi.org`,
  `files.pythonhosted.org`. USGS / OpenEI / DrivenData / ScienceBase are **not**
  reachable, so any hypothesis needing *new* external rasters cannot be
  validated this session. Every candidate below is annotated with the specific
  free official source it would need.
* Compute: 2 vCPU, 3 GB RAM, no GPU.
* Reusable instruments already in the repo: `vendor/gems57` (holdout, metric,
  evaluator `gems57-pooled-hide-v2`, validator, submission writer) and
  `src/gems77d/submodular.py` (exact 300 m triangular-kernel greedy allocator).

## Metric arithmetic that motivates the ranking (derived, auditable)

With `DTI = T / (T + 0.2·FP_w + 0.8·FN_w)`, `FN_w = K − T`, and a binary dot
field of `n` dots with total self-credit `M`:

```
DTI = T / (0.2·T + 0.2·(n − M) + 0.8·K)
```

Using the organizer-confirmed GEMSDOE32 receipts (`0.2600 → 0.2708 → 0.2778`
from pruning 3,891 then 2,545 dots) the two pruning slopes
`Δ(1/DTI)/ΔN = 3.94e−5` and `3.66e−5` give `T ≈ 5,075` and, with
`n ≈ 37,654`, `K ≈ 1.3e4` withheld truth pixels. Therefore

* `∂DTI/∂T ≈ +5.3e−5` per unit of covered truth credit,
* `∂DTI/∂n ≈ −3.2e−6` per wasted (zero-credit) dot.

**A unit of recall is worth ~17 wasted dots.** Dot-geometry tricks (the main
lever used from `GEMSDOE25` to `GEMSDOE32`) are already exhausted: pruning every
remaining zero-credit dot would add at most `+0.03`. Reaching `0.32+` requires
raising coverage `T/K` from ~39 % to ~44 %, i.e. a **recall** idea, not another
pruning idea. All five candidates below are therefore recall ideas.

---

## C1 (IMPLEMENTED) — Fault mapping-deficit (censored-intensity residual) allocation

* **Layers.** All 19 bands of `training_features.tif` (`mag_anom`, `rtp`,
  `tmi_hg`, `geod_2ndinv`, `iso_grav_anom_slope`, `tc`, `geod_shearrate`,
  `geod_dilaterate`, `tmi_vg`, `deq_n100a15`, `iso_grav_anom_vg`, `det_elev`,
  `iso_grav_anom`, `tmi`, `depth_to_base_surf`, `ieq_n100a15`, `cond_surf`,
  `iso_grav_anom_hg`, `det_elev_slope`) plus eight derived channels computed
  here (Hessian across-strike curvature of `det_elev` at σ = 1.5 and 4.0 px,
  5-px local relief, magnetic tilt angle `atan2(tmi_vg, tmi_hg)` after
  Miller & Singh 1994, gravity tilt `atan2(iso_grav_anom_vg,
  |iso_grav_anom_hg|)`, log magnetic edge strength, elevation structure-tensor
  coherence, and basement-depth gradient).
* **Physical signature targeted.** Not a new edge transform — a **completeness
  residual of the mapped catalogue treated as a censored spatial point
  process.** A Poisson-loss gradient-boosted regressor predicts the *mapped*
  fault-pixel count per 2 km block from block-aggregated physics; the deficit
  `D(b) = max(0, λ̂(b) − n_visible(b))` is the number of fault pixels the
  tectonic/geophysical setting of that block implies but the catalogue does not
  contain. `D` multiplies the per-pixel physics surface before dot allocation.
* **Why it should catch a fault missing from the catalogue rather than one
  already in it.** The DTI truth set is defined as faults *absent* from the
  USGS/INGENIOUS catalogue. Every prior lane in this portfolio estimates
  `P(fault | physics)` — an estimand whose maximisers are, by construction,
  dominated by the already-mapped traces the model was trained on. C1 estimates
  `E[N_fault] − N_mapped`, which is zero exactly where the catalogue is already
  complete and large where mapping is censored (Quaternary cover, historic
  small-scale mapping, areas away from range fronts). Pruning the 0–200 m
  catalogue collar gave `+0.0178` live (GEMSDOE28 → GEMSDOE32); that is the same
  idea applied at 2 px. C1 applies it at 20 px with a *predictive* rather than a
  fixed-distance rule.
* **How it differs from everything already in this repo.** Checked against all
  25 `registry/preregistration_*.json` files and the lane list in `README.md`:
  every lane is a forward per-pixel signature detector (scarp, ridge, tilt,
  Euler depth, worm persistence, clay cap, radiometric halo, Riedel wing crack,
  knickpoint, relay ramp, hydrothermal conduit, SGMC contact). None models
  catalogue completeness, none uses block-count (Poisson) supervision, and none
  reallocates the dot budget by a residual intensity. The allocator, evaluator
  and writer are reused unchanged, so the only moving part is the surface.
* **Named non-fault process that could mimic it.** A 2 km block can have a high
  predicted intensity and no faults at all when its physics resembles a faulted
  range front for non-tectonic reasons — e.g. an eroded volcanic tableland edge,
  a playa shoreline terrace of pluvial Lake Lahontan, or a lithologic contact in
  Tertiary tuff. A second, procedural mimic is the hide-and-recover design
  itself: withholding segments manufactures exactly the deficit C1 looks for, so
  the holdout is expected to be *optimistic* relative to live. Both are reported
  with the result.
* **Expected gain / cost.** Largest expected gain of the five (direct attack on
  the estimand), lowest cost (one extra regressor, no new data).

## C2 — Geodetic slip-budget deficit corridors

* **Layers.** `geod_shearrate`, `geod_dilaterate`, `geod_2ndinv` (GPS-derived
  strain-rate bands) against mapped fault-length density.
* **Signature.** Blocks whose geodetic extension rate requires more cumulative
  fault length/throw than the catalogue provides must host unmapped structures
  (a mass-balance, not an image-processing, argument).
* **Why unmapped.** Same censoring logic as C1 but with an independent,
  physically conserved quantity (strain rate) instead of a learned intensity.
* **Difference from repo.** `geod_*` bands are used only as plain features in
  earlier lanes, never as a conserved budget.
* **Mimic.** Aseismic/distributed strain accommodated by block rotation or by
  faults outside the footprint; GPS interpolation artefacts.
* **Why not chosen.** The three `geod_*` bands are interpolated GPS fields whose
  effective resolution is tens of kilometres (visible as smooth, almost
  piecewise-linear structure in the raster). The budget residual would be nearly
  constant over 20 km, giving far coarser steering than C1 for the same cost.
  Parked, not refuted. Rank 2.

## C3 — Hydrothermal K-enrichment / Th-normalised halo on cover-concealed faults

* **Layers.** `external/geodawn_rad_u8.tif` (K, eTh, eU, TC), `tc` band 6.
* **Signature.** Adularia–sericite alteration raises K and leaves eTh flat, so
  `K/eTh` ridges mark upflow along concealed structures.
* **Why unmapped.** Alteration halos survive where the scarp is buried.
* **Difference from repo.** Partly implemented already: `GEMSDOE46`
  `r11f-scarp-radiometric-fusion` scored `0.1589` and `r12-scarp-rad-concordance`
  `0.0843` ORGANIZER-CONFIRMED; `src/gems77/features77.py` already carries
  `rad_K/rad_Th/rad_U/rad_ThK/rad_UK`. Rank 4 — low novelty.

## C4 — Lateral-uncertainty hedged "comb" dot geometry

* **Signature.** Replace one dot per 3 px of predicted centreline with a
  cross-strike comb at offsets −2/0/+2 px to hedge lateral mislocation.
* **Analytically rejected before implementation.** For lateral error σ the comb
  triples the dots per unit length, so it covers one third of the length. With
  σ = 2 px, expected per-truth-pixel credit rises only from ≈0.41 to ≈0.75
  (×1.8) against a ×3 budget cost — a net loss of ≈40 % of `T`. It only wins for
  σ > ~4 px, which is larger than the measured 300 m kernel radius itself. Kept
  as a documented negative design result. Rank 5.

## C5 — Playa-margin concealed piedmont faults (basement-step × conductance)

* **Layers.** `depth_to_base_surf`, `iso_grav_anom`, `cond_surf`.
* **Why not chosen.** This is essentially lane `H77G`
  (`demag-trough × gravity basement step × MT clay-cap margin`), already run and
  **negative** (`0.0619` HOLDOUT-DTI). Re-running it would breach the
  no-retuning rule in `AGENTS.md`. Rank 6.

## External data that would unlock the parked candidates

| Candidate | Specific free official source | Reachable from this sandbox? |
|---|---|---|
| C2 | UNAVCO/EarthScope GAGE GPS velocity field, `https://www.unavco.org/data/gps-gnss/derived-products/derived-products.html` | **No** (host not in allowlist) |
| C3 | USGS GeoDAWN radiometric grids, `https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and` | **No** — but a SHA-pinned mirror of the derived `geodawn_rad_u8.tif` is already in `registry/data_manifest.json` |
| C5 | USGS SGMC / Nevada BMG maps, `https://mrdata.usgs.gov/geology/state/` | **No** (mirror only) |

Flagged for review: every "external" layer used in this repository is an
owner-supplied GitHub mirror, integrity-pinned by SHA-256 but **not**
organizer-authenticated. See `docs/irregularities.md`.
