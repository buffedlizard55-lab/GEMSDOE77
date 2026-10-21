# Candidate geological hypotheses not yet tried in this repo — ranked (2026-10-21, GEMSDOE77 XSEN session)

Standing instruction (PROMPT.md): generate 3–5 untried hypotheses, each naming the
layer(s), the physical signature, why it should catch a fault **missing** from the
USGS/INGENIOUS catalogue rather than one already in it, and how it differs from
anything implemented in this repo. Rank by expected DTI improvement ÷ cost.
Validate the top candidate on the spatially blocked holdout before any slot.

Sources for layer provenance (manual review):
- Competition / metric: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- GeoDAWN (magnetic + radiometric source data): https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and (DOI 10.5066/P93LGLVQ per `registry/data_manifest.json`)
- INGENIOUS / GDR 1391: https://gdr.openei.org/submissions/1391
- Grid: EPSG:32611 https://epsg.io/32611 · Metric kernel: Tversky https://en.wikipedia.org/wiki/Tversky_index
- Why the family champion scored 0.2778 (OWNER-REPORT receipts + byte forensics): `evidence/why_02778.json`, `knowledge/01_why_02778_and_path_to_04413.md`

| Rank | Hypothesis | Layers (all already restored, SHA-pinned) | Physical signature | Why a *catalogue-missing* fault | Difference from repo lanes | Cost |
|---|---|---|---|---|---|---|
| 1 | **H77-XSEN: magnetic analytic-signal ridge × radiometric compositional-boundary coincidence** | training_features bands 3/9 (tmi_hg, tmi_vg); external `geodawn_extensions_u8` ThK/UK/UTh | Product/min of rank(AS-ridge persistence σ=1.5,3 px) and rank(\|∇Th/K\|, \|∇U/K\|) | Under alluvium there is no DEM scarp (lidar lanes blind) and no mapped trace; a buried fault still juxtaposes K/Th/U domains across the fault and offsets magnetic basement. Single-sensor edges are mostly lithology/dikes/seams — the *coincidence* is the fault-specific part. | H77D cross-gradient was magnetic×**gravity**; contact-physics was mag/grav phase congruency; selective-U used U curvature alone; **no lane used mag-AS × radiometric-ratio boundaries**. | Low: 8 channels from 4 bands + HGB arm reusing `run_holdout.py` machinery |
| 2 | Gravity tilt zero-crossing × magnetic AS coincidence | bands 11/13 (gravity VG/anomaly), 3/9 | Zero-contour of tilt(gravity) co-located with AS ridge | Deep basement fault under thick cover where radiometrics (top ~30 cm) are homogenised by transported alluvium | Partially inside H77D `potential_crossgrad` block — weakened novelty; kept as ablation context, not a lane | Low |
| 3 | Radiometric ternary (K–Th–U) centroid-migration gradient | `geodawn_rad_u8` K/Th/U | Gradient of ternary centroid distance / compositional (log-ratio) distance between neighbouring cells | Hydrothermal K-metasomatism / Th-U fractionation along an unmapped conduit changes *assemblage*, not just one ratio | Selective-U lane used U-vs-(K,Th) curvature of one contrast; ternary log-ratio geometry is different but correlated with rank-1 radgrad — high duplicate-surface risk (ρ gate) | Low |
| 4 | Drainage-deflection + profile-curvature break pairs | band 12 (det_elev), band 19 (slope) | Paired channel-azimuth deflection and curvature break along a lineament | Fault-controlled drainage persists after scarp burial | Needs flow-routing on a 100 m detrended DEM (not a true hydrologic DEM); strong mimics (lithologic contacts, roads, irrigation). In-repo DEM lanes used scarp amplitude/curvature, not drainage topology | Medium–high |
| 5 | — rejected: Curie-depth / geothermal-gradient residual lineaments | would need new external heat-flow/Curie compilations | — | — | Cannot be validated from the cached stack; the specific free official source would be the USGS National Geothermal Data System / GDR (https://gdr.openei.org/submissions/1391 is already the in-repo GDR source; a Curie compilation is **not** in the manifest and obtainability was not verified) — not proposed as viable this session | High / unverified |

**Selected: Rank 1 (H77-XSEN)**, preregistered in `registry/preregistration_xsen.json`
before any experiment. Mimics (must name one): non-fault lithologic contacts,
magnetic dikes, GeoDAWN flight-line seams.

Background reading stored in-repo: `knowledge/01_geothermal-vents.md`,
`knowledge/02_hydrothermal_vent_conduits_great_basin.md`.
