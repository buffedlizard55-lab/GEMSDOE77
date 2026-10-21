# H80 candidate hypotheses (generated 2026-10-21, before any H80 experiment)

Protocol: 3–5 geological hypotheses not yet tried in this repo, each naming layer(s),
physical signature, why it should catch a fault missing from the USGS/INGENIOUS
catalogue, the named non-fault mimic, and how it differs from implemented lanes.
Ranked by expected HOLDOUT-DTI improvement ÷ implementation cost. Duplicates of
existing lanes were actively rejected during drafting — two initially attractive
ideas were found already implemented and are recorded at the end.

## Verified inputs available in this sandbox (checked 2026-10-21, from file bytes)

- `training_features.tif` restored, sha256 `4371c82e…` (owner mirror, NOT
  organizer-authenticated). 19 float32 bands; in-file descriptions read directly.
  Two tags are known-wrong from prior measured evidence (`evidence/band6_resolution.json`):
  band 6 tag says "Tilt angle or total curvature" but measures as radiometric total
  count; band 18 is a signed gravity derivative, not a gradient magnitude.
- `gdr_wellspring_in_footprint.csv` (GDR 1391, DOI 10.15121/1881483): 27,092 rows,
  12,570 unique cells; only 8,718 rows carry `temp_c`, only 1,247 carry any chemical
  geothermometer; 19,456 rows are bare well locations. Its `dist_known_fault_px`
  column is label-derived and is **never used** (distances are recomputed locally).
- `gdr_volcanic_vents_in_footprint.csv`: 21 vents only.
- External rasters restored: lidar scarp u8 (12 bands), GeoDAWN rad u8 (K/Th/U/TC),
  extensions u8 (ThK/UK/UTh/TMI_up150).

## Ranked candidates

| Rank | Hypothesis | Layers | Physical signature | Why a catalogue-missing fault | Named mimic | Differs from implemented lanes | Expected gain ÷ cost |
|---|---|---|---|---|---|---|---|
| **1 — H80-OTA Orphan Thermal Alignment (SELECTED)** | Concealed geothermal conduit faults leak hot fluid to wells/springs far from any mapped fault; those **orphan hot points are collinear along the unmapped fault** | GDR 1391 well/spring points (`temp_c`, quartz/chalcedony/cation geothermometers); shipped bands 3 (`tmi_hg`) + 5 (`iso_grav_anom_slope`) only as a label-free ridge-support localizer | Weighted Hough-style collinearity: ≥3 orphan points (>1 km from visible catalogue) on one line (±2 px, 0.5–4 km), vote weighted by geothermometer excess ΔT = max(Tqtz,Tchalc,Tcat) − Tsurf / surface temperature; line field intersected with potential-field ridge support | Catalogue mapping is scarp/lineament-led; a blind fault under basin fill may have no scarp and a weak potential-field edge, but its fluid upflow still heats wells along strike. Springs cluster on faults generally, so *orphan* (off-catalogue) alignments specifically select unmapped conduits | Wells drilled along roads, canals and valley axes (anthropogenic collinearity); regional aquifer flow along basin axes. Mitigations frozen in advance: ΔT/temperature weighting (cold irrigation wells excluded), ridge-support coincidence channel | H77D used anisotropic kernel *halos around* thermal points (no collinearity test, no orphan-only restriction); H77-Step used distance halos. No repo code implements point-collinearity/Hough voting (verified by grep for hough/collinear, 2026-10-21) | Uncertain-moderate ÷ low (point process on ≤12.6k cells; KD-tree) |
| 2 — H80-XRES nonlinear cross-field residual ridge | A fault juxtaposes blocks with different density–magnetization–elevation relations, so gravity becomes *unpredictable* from the other fields along a line | Bands 13 (target), 2/14/12/15/17 (predictors) | Spatial-block OOF HGB-regression residual \|g−ĝ\|, Hessian ridge of the residual | Works where the fault has no single-field edge strong enough to map, only a relation break | Non-fault lithologic contacts also break field relations | `features77` `pg_resid` is a *linear, 2-field, 25-px-window* Poisson regression; this is nonlinear, multi-field, block-OOF. Ancestor exists ⇒ novelty moderate | Moderate ÷ medium |
| 3 — H80-SEISRIDGE seismicity-density ridge | Blind active faults host aligned microseismicity; the density field itself forms a ridge over the fault | Band 16 (`ieq_n100a15`), band 4 (`geod_2ndinv`) gate | Hessian ridge of log-density at σ=5–10 px | Seismogenic faults need no surface expression | Smooth kernel-density artefacts; geothermal-production induced seismicity clusters | H77-E used bands 10/16 only as scalar proximity/intensity multipliers, never a ridge *of* the density field. Band tags are vague ("intensity or density", "n=100km, a=15°") — flagged IR-H80-01; if the density kernel is ~100 km wide the ridge cannot localize to 300 m | Low ÷ low — deprioritized on band-definition risk |
| 4 — H80-DIPPAR depth-ordered edge parallax | A dipping fault's edge appears laterally shifted with source depth: lidar scarp → shallow magnetic edge → deep gravity edge march monotonically in the dip direction | Lidar `step_max`, band 3, band 5/13 | Monotonic cross-strike offset ordering of three edge fields | Detects dipping buried faults whose deep edge is offset from any surface trace | Dipping lithologic contacts show the same parallax | Too close to H77L (fixed-lag flank), H79-RH (lag-3 halo) and H77-Step dip-shifted concordance (+4 px): drift risk into a sibling lane, rejected as primary | Low ÷ medium |
| 5 — H80-VENTALIGN volcanic-vent alignment corridors | Quaternary vents erupt along fissure faults; vent rows trace unmapped faults | GDR vents csv | Collinearity corridors through vent points | Fissure feeders may be absent from a scarp-led catalogue | Vent rows along caldera ring fractures (not regional faults) | H77-Step has a vent *halo*, not alignments. **Not viable:** verified only 21 vents inside the footprint; a fuller free official source (Smithsonian GVP / USGS vent databases) is not reachable from this sandbox (egress limited to github/pypi hosts) — obtainability check FAILED, do not propose as viable this session | — infeasible |

*External-data note:* geodetic strain-tensor *azimuth* (needed for an extension-direction
gate) is not recoverable from the shipped scalar bands 4/7/8. The free official source
would be EarthScope/GAGE GNSS strain products (https://www.earthscope.org), unreachable
from this sandbox; recorded as a future-session candidate, not used here.

## Ideas rejected as already-implemented (duplicate-lane guard, verified by grep)

- **HW/FW polarity concordance** (DEM low side = gravity low side = deep-basement side):
  implemented in `src/gems77_step/features.py` (`hwfw_step_purity_*`,
  `hwfw_dip_basement_concord`, `hwfw_dip_grav_concord`). Rejected.
- **Magnetic texture/damage-zone suppression**: implemented and tested as H77E-2
  (`evidence/h77e_*`, alone 0.0396 HOLDOUT-DTI). Rejected.
- Dilatation intersection (H77-A), clay-cap conductivity breach (H77-C), seismic relay
  scalars (H77-E), UC-persistence/Euler (H77E), cross-gradient (H79-BB), lagged
  radiometric halos (H77L/H79-RH), analytic-signal×radiometric coincidence (XSEN),
  basement-step (family 61GEMSDOE, 0.0009 live), chi/ksn corridors (family + H77-Step).

## Frozen design for the selected candidate (H80-OTA)

See `registry/preregistration_h80.json` — written before Experiment 1. Parameters were
set from label-free marginals of the point file only (counts/quantiles above); no label
or holdout information was used. Promotion requires beating the repo incumbent
0.152718 HOLDOUT-DTI (H77-Step, `gems57-pooled-hide-v2`, 11,321 withheld px, 95% CI
[0.131548, 0.175185]) with a paired ΔCI excluding 0; otherwise verdict = negative and
the TIFF is inspection-only. No weekly slot is spent by this session regardless
(selector step is separate).
