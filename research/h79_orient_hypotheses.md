# H79 candidate hypotheses — pre-implementation ranking

**Status:** pre-registration for this session, not a claim that any candidate is a
fault or a leaderboard improvement. The current retained local incumbent is
`0.152718 HOLDOUT-DTI` under `gems57-pooled-hide-v2`, with 11,321 withheld
positive pixels and 95% CI `[0.131548, 0.175185]` (the H77-Step receipt). A
DrivenData score is a different evidence class. No weekly slot is used by this
work.

## Candidate matrix

| Rank | Candidate and layers | Physical signature | Why it can find an unmapped fault | Difference from this repo | Main non-fault mimic | External data / cost |
|---:|---|---|---|---|---|---|
| **1** | **H79-ORIENT: cross-sensor scale-space orientation persistence** — `rtp` band 2, `iso_grav_anom` band 13, `det_elev` band 12, plus `depth_to_base_surf` as a raw control | Gaussian derivative edge states at 150, 300 and 600 m; geometric persistence across scales; doubled-angle orientation concentration; a cross-sensor vector-confluence term | A buried normal-fault boundary can be weak in every one-scale amplitude map but remain a coherent axis across magnetic, density and relief fields. A stable edge below the surface-mapping threshold need not be present in the USGS/INGENIOUS surface catalogue | No prior file in this checkout implements a steerable/orientation-concentration or geometric cross-scale persistence transform. It is not the H77 Radon line integral, scalar contact confluence, phase-congruency channel, or catalogue-distance mechanic. The primary model uses only this block plus the four named source amplitudes; no previous predictions or catalogue features | Intrusive/lithologic contacts, flight-line leveling seams, drainage/terrace margins, and survey-boundary edge effects | No new external data; uses the cached stack. **Medium** CPU/memory cost. |
| 2 | **H79-RAD: displaced radiometric alteration trail** — cached K, Th, U and ratio channels, with 100–300 m signed cross-gradient lag | Cross-correlation of U residual with K/Th common-mode at ±1–3 pixels, requiring a consistent offset rather than same-pixel agreement | Fluid alteration can be displaced down-gradient from a concealed conduit, so a lagged anomaly could be absent from a surface fault catalogue | Existing H77 radiometric work uses same-pixel rank/intensity or amplitude fusion; this would test a signed spatial phase offset. It is not authorized unless H79-ORIENT is negative and a new budget is declared | Soil transport, groundwater redox fronts, ash-flow compositional boundaries | No new data; **medium** cost. |
 3 | **H79-DEPTH: multi-scale basement-throw asymmetry** — `depth_to_base_surf`, gravity anomaly and RTP | Odd/even derivative pair: a signed depth step aligned with a persistent gravity/magnetic edge, rejecting symmetric curvature | A fault may offset basin fill without a visible scarp; the signed throw is a structural discriminator rather than a generic edge | Distinct from the existing single-scale `base_step_along_maggrad`: H79-DEPTH would require cross-scale sign stability and a two-sided asymmetry score before ranking | Paleochannel fill changes, pluton contacts, caldera margins | No new data; **low–medium** cost. |
 4 | **H79-HYDRO: thermal-memory alignment** — GDR 1391 well/spring temperature and geothermometer records, radiometric U/K/Th and structural orientation field | Anisotropic, distance-decayed excess geothermometer temperature aligned with H79-ORIENT rather than point proximity alone | A blind conduit can preserve a thermal/chemical footprint even when no mappable scarp exists | Existing H78 used hot/warm point proximity and H77D used a hydrothermal ensemble; this would use signed anisotropic cross-sensor alignment, not nearest-point distance | Closed-basin evaporites, regional altered tuff, sampling-density clustering | **Viable source checked:** GDR submission 1391 is publicly accessible and CC BY 4.0. It is deliberately not used in this three-experiment budget. **Medium** cost. |
 5 | **H79-SEIS: 300 m event-lineation transfer** — USGS ComCat FDSN events/moment tensors plus cached seismic bands | Local focal-mechanism/epicentre lineation aligned with a structural edge | A distributed accommodation zone may host a blind fault with no mapped scarp | New relative to the code, but not viable here: the official FDSN host is outside the sandbox egress allow-list and no pinned event mirror was found | Aseismic geothermal/magmatic swarms, catalogue completeness gradients | **Not viable this session.** Specific free official source: USGS ComCat FDSN event service, `https://earthquake.usgs.gov/fdsnws/event/1/query`; obtainability check returned HTTP 000/exit 35 from this sandbox on 2026-10-21. No synthetic seismic data will be made. |

## Selection and validation lock

1. Build H79-ORIENT from the cached stack with no labels, catalogue mask, prior
   raster, or registry raster in feature construction.
2. Test **each H79 feature alone** on the seed-20 whole-component quadrant
   holdout. A symmetric ROC-AUC above `0.90` is a leakage stop, not a success.
3. Fit the pre-registered primary HGB using H79 features plus the four named
   source amplitudes, with H79-only and raw-only controls. Emit a fixed 37,654
   dots with 2.8 px minimum separation and a >3 px catalogue exclusion; no
   budget search is allowed after seeing the holdout.
4. The primary is eligible for a second seed confirmation only if it exceeds the
   retained `0.152718 HOLDOUT-DTI` incumbent on seed 20. A candidate that does
   not clear that bar is still written as a **negative research deliverable**
   but is not a promoted submission.
5. Perform the registry surface gate before dot placement and the final-dot gate
   after placement. The public owner registry is a snapshot, not a complete
   organizer registry; a GitHub API rate-limit or missing sibling bytes is an
   irregularity, not evidence of uniqueness.

## Verified source register

- [OFFICIAL] DrivenData problem page 967: the task is fault detection for
  geothermal-indicative structures; provided features include detrended
  elevation, gravity, magnetics, and earthquake density; the page specifies
  EPSG:32611, 100 m resolution, a single float32 layer with values in `[0,1]`,
  and the distance-weighted Tversky metric with a 300 m triangular kernel:
  <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>.
- [OFFICIAL] USGS GeoDAWN release: airborne magnetic/radiometric data were
  collected for geologic/geophysical mapping and geothermal studies; it links
  DOI `10.5066/P93LGLVQ` and states CC0 1.0 rights:
  <https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and>.
- [OFFICIAL] INGENIOUS/GDR 1391: the public compilation includes geophysics,
  elevation, seismicity, temperature/chemistry, faults and paleogeothermal
  features; the page states CC BY 4.0 and DOI `10.15121/1881483`:
  <https://gdr.openei.org/submissions/1391>.
- [OFFICIAL] Reference solution repository and CPU/GPU setup:
  <https://github.com/drivendataorg/gems-prize-reference-solution>.
- [OWNER-REPORT, not organizer-authenticated] Prior family artifacts and
  scores are indexed in `registry/registry_index.json` and `registry/live_scores.csv`.
  The H33 `0.2778` number is retained as an owner-reported family result; it is
  not used as a measured score in this lane.

## Measured outcome and stop decision

The selected H79-ORIENT primary was run through the shared evaluator before any
weekly slot. The feature-alone canary passed: maximum symmetric AUC was
`0.6242` for `h79_dem_orientation_coherence`, below the `0.90` stop threshold.
The primary `H79_plus_raw` measured `0.089063 HOLDOUT-DTI` (95% CI
`[0.072384, 0.105355]`, 11,321 withheld positives) on seed 20 and
`0.094507 HOLDOUT-DTI` (95% CI `[0.076558, 0.112844]`, 11,342 withheld
positives) on seed 30, evaluator `gems57-pooled-hide-v2`. Both are below the
retained `0.152718 HOLDOUT-DTI` incumbent, so the candidate is **negative** and
was not promoted. H79-only was `0.094507` on seed 20; the raw-only controls
were `0.040638` and `0.040465`, respectively. These are local hide-and-recover
measurements, not organizer scores.

The runner initially logged zeros for every arm because it excluded holdout
placement using distance to the full catalogue, which includes the hidden
component. That was an implementation error, recorded as IR-79O-01; the shared
evaluator and preregistered method were not changed. The corrected runner uses
distance to visible catalogue cells for holdout placement and re-ran the fixed
three-experiment plan. The final format-valid artifact is
`docs/downloads/gems79-orient-persistence-37654dots-20261021-4abff47886.tif`,
SHA-256 `745dbaa427bc1edaf477b3d0b4847b8be8d30d396106a61b1318a8726599be8e`.
It is safe to download but not approved to submit. Its before-placement and
final-dot audits each checked 13 local TIFFs; the dense witness caused a
forward overlap of `1.000`, and the incomplete 812-raster registry prevents
any uniqueness clearance.

The experiment budget is closed at three experiments and zero weekly slots.
No H79 projection is reported as a score, and H79-RAD/H79-DEPTH/H79-HYDRO/
H79-SEIS remain untested alternatives for a future separately preregistered
lane, not latent evidence for improvement.

The source pages above were fetched successfully in this session where the
sandbox allowed them. Competition data bytes remain SHA-pinned owner mirrors,
not independently authenticated organizer downloads; this limitation is
recorded in `data/README.md` and the run card.
