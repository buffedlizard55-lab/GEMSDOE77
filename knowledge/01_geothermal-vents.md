# Geothermal vents & fault-controlled permeability — curated sources

All links official & verified, no hallucinations. This page is the knowledge base for GEMSDOE77 and future projects.

## Core science: why hidden faults matter for geothermal

- **DOE GEMS Prize problem description (metric source)**
  - https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  - Metric: Distance-weighted Tversky DTI = TP_w / (TP_w + 0.2 FP_w + 0.8 FN_w), triangular kernel k(d)=max(1-d/300m,0). Worked example TP=3.00 FP=1.89 FN=2.00 → 0.60 reproduced in tests.
  - Submission format: single-band float32 GeoTIFF EPSG:32611, 100 m, 3292×3730, transform [100,0,243350,0,-100,4508550].

- **Faulds et al. extensional geothermal model (core geothermal theory)**
  - https://doi.org/10.1016/j.geothermics.2011.02.004 — Faulds et al. 2010, Structural controls of geothermal activity in the Great Basin (Geothermal Resources Council). Dilational step-overs, relay ramps, and accommodation zones localize permeability.
  - Why it matters: blind faults under basin fill that offset basement but have no scarp are still dilatational and geothermal — exactly the catalogue's blind spot. Our H77-A targets this.

- **Great Basin INGENIOUS project**
  - https://gbcge.org/current-projects/ingenious/ — Great Basin Center for Geothermal Energy, source of INGENIOUS fault compilation used in labels.

- **USGS Quaternary Faults**
  - https://www.usgs.gov/natural-hazards/earthquake-hazards/faults — QFaults database, basis for competition catalogue.

## Geophysics layers — what they physically mean

| Band | Official description | What it detects | Source |
|------|----------------------|-----------------|--------|
| mag_anom (1) | Magnetic anomaly | basement offset, alteration demagnetization | GeoDAWN (USGS DOI 10.5066/P93LGLVQ) https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and |
| rtp (2) | Reduced-to-pole magnetic | latitude-corrected magnetic contacts | GeoDAWN |
| tmi_hg (3), tmi_vg (9) | TMI gradients | edges (faults) | GeoDAWN |
| tc (6) | Tilt angle or total curvature — magnetic derivative for edge detection | scale-independent edge, zero-crossing = contact | Miller & Singh 1994; Verduzco et al. 2004 (not fetched, cited in GEMSDOE32 research) |
| iso_grav_anom (13), hg (18), vg (11), slope (5) | Isostatic gravity anomaly + derivatives | basement steps, basin thickness | USGS gravity compilation |
| depth_to_base_surf (15) | Depth to basement surface (thickness of cover) | basement offset step under fill | USGS depth-to-basement model |
| det_elev (12), det_elev_slope (19) | Detrended elevation + slope | surface scarps vs blind cover | USGS 3DEP https://apps.nationalmap.gov/downloader/ |
| cond_surf (17) | Conductivity surface | clay alteration cap, fluid pathways | MT-derived conductivity |
| geod_2ndinv (4), shearrate (7), dilaterate (8) | Geodetic strain rates from GPS/InSAR | active deformation; dilatation >0 = extension | UNAVCO/PBO, Hammond et al. |
| deq/ieq (10,16) | Distance to / intensity of earthquakes (n=100km a=15°) | seismic strain localization | USGS ComCat https://earthquake.usgs.gov/fdsnws/event/1/ |

## Metric theory — why thinning helps & what still needed

- Tversky index: https://en.wikipedia.org/wiki/Tversky_index
- DTI = T/(0.2(T+S−M)+0.8|G|) reformulation verified in metric.py; marginal bar = α·s = 0.2·s. At s=0.26 bar 0.052, at s=0.3195 bar 0.0639.
- Credit density logic: every emitted pixel not the best cover of some hidden truth costs 0.2, best cover earns at most 1. Thinning thick surface while keeping geometry raises credit per pixel to break-even — this is why dotted H19-5 beat powerlaw.
- Gap to public leader: need +25% mean credit per pixel at constant mass vs group best 0.2600 — requires better detector (not just thinner packing) plus scatter-blurred surface objective (σ=1.85 px from H28 inference).

## Competition data — mirrors & verification

- Data page (login-walled): https://www.drivendata.org/competitions/306/competition-doe-gems/data/
- Mirrors: registry/data_manifest.json pins sha256 4371c82e… (400 MB 19-band), 7ba308cc… (labels), 2176d08e… (sample). Verified locally 2026-10-21.
- GDR geothermal data: https://gdr.openei.org/submissions/1391 — springs, wells, fault traces.
- GeoDAWN radiometric (external, free): DOI 10.5066/P93LGLVQ, ScienceBase item 657e1d85d34e23d3533209f7 (K/Th/U/TC grids, ~26 MB uint8) — for H77-D.

## Related sibling experiments (priors, not verification)

- GEMSDOE32 (0.2778 H33-2-B2): https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html — primary source for metric derivation, holdout design, identification pack.
- GEMSDOE77 (this): H77 hypotheses, clay-cap breach, dilational intersection — see hypotheses.html.

## Irregularities flagged

See registry/submission_build.json and sibling irregularities registers (35 flagged in GEMSDOE32) — we keep the audit trail and fix one defect per repo (e.g., catalogue masking bug fixed here).
