# H82 candidate hypotheses (2026-10-21) — ranked before implementation

Protocol: generate 3–5 untried geological hypotheses naming layer(s), physical
signature, why each catches a catalogue-missing (USGS/INGENIOUS) fault rather
than a mapped one, and how it differs from everything already in this repo.
Rank by expected HOLDOUT-DTI gain ÷ implementation cost. Validate the top
candidate on the blocked holdout before any submission-slot discussion.

## Why did h33-h33-2-b2 score 0.2778 (the standing question)?

Verified in-repo from `docs/evidence/why_02778.json` (S0 live-inversion
instrument; scores in the brief are OWNER-REPORT, no organizer receipts in
this repo). The lineage is pure **emission geometry on one fixed surface**
(H19-5 multiline ridges), not new geology:

| step | dots | reported score | S0 credit/dot |
|---|---|---|---|
| H19-5 solid ridges | 121,131 | 0.1922 | 0.0533 |
| dotted d=1.5 px | 60,069 | 0.2477 | 0.0900 |
| dotted d=2.8 px | 44,090 | 0.2600 | 0.1091 |
| d2.8 minus dots ≤1 px from catalogue | 40,199 | 0.2708 | 0.1181 |
| d2.8 minus dots ≤2 px from catalogue (h33-2-b2) | 37,654 | 0.2778 | 0.1247 |

Mechanism (DTI: Tversky-style, α=0.2 FP weight, β=0.8 FN weight, 300 m
triangular kernel, mapped catalogue masked before scoring): a solid line
wastes budget — neighbouring dots inside one 3 px kernel footprint add full
FP weight but almost no extra covered-truth credit. Spacing dots ≈2.8 px
(≈ kernel radius) maximises covered truth per dot. Dots within 2 px of the
catalogue are the worst spend (S0 isolated credit 0.049 for the ≤0.4 km bin
vs 0.208 beyond 2 km) because their kernel overlaps already-masked known
faults. Deleting them raises credit/dot without losing reachable truth.

Consequence: to beat 0.2778 at the same 37,654-dot geometry, a surface must
raise covered-truth credit T by ≈ +11% over the champion's S0 T=4694 for
0.3195-class scores and ≈ +56% (T≈7310) for the current leaderboard top
0.4413 (ORGANIZER-CONFIRMED public LB read 2026-10-21, PROMPT.md). Geometry
is already near-optimal; only a genuinely better *surface* can move the
score. That is what the candidates below are ranked on.

## Candidates (all verified untried in this repo by source grep, 2026-10-21)

### 1. H82-TCSTEP — radiometric total-count lithologic-offset step × blind gate ★ SELECTED
- **Layers:** `training_features` band 6 (`tc_totalcount`; in-repo measurement
  `evidence/band_diagnostics.json`: Spearman 0.99995 vs external GeoDAWN TC —
  the band's shipped tag claiming tilt/curvature is wrong) + band 19
  (`det_elev_slope`) as the no-scarp gate.
- **Signature:** multi-scale |∇log TC| (σ=2, σ=5) modulated by structure-tensor
  coherence (strike persistence), multiplied by a low-slope rank — a sharp,
  straight TC step with no topographic scarp.
- **Why catalogue-missing:** the USGS/INGENIOUS catalogue is built from
  surface rupture/scarp mapping; a buried fault offsetting radiometrically
  distinct bedrock/fan units under basin fill leaves a gamma step but no
  scarp, so it is systematically absent from the catalogue.
- **Differs:** prior radiometric lanes used the *external* uint8 K/Th/U stack
  as ranks/curvature (H77 selective-U curvature, H77L Th/K flank, H79-RH
  radiometric halo lag-3). No lane has used core band 6 TC as a step
  matched-response, and none paired it against the identical 19-band HGB
  control. Grep `tc_step|totalcount.*step|tc_grad` over `src/ scripts/`: 0 hits.
- **Mimic:** fan/playa shoreline (Lake Lahontan) lithologic boundaries,
  volcanic unit contacts, survey line-levelling seams.
- **Cost:** low (2 core bands, deterministic filters, proven h77g-style HGB
  harness). **Expected gain:** low–medium — TC is already one of the 19 raw
  bands, so the HGB control sees the amplitude; the bet is that explicit
  step/coherence/blind-gate *geometry* adds what axis-aligned tree splits miss.

### 2. Seismicity-anisotropy lineaments from deq/ieq density bands
- **Layers:** bands 10 (`deq_n100a15`) + 16 (`ieq_n100a15`).
- **Signature:** structure-tensor ridge/coherence of log EQ-density — linear
  seismicity alignments on active but unmapped structures.
- **Why missing:** microseismicity does not require a surface scarp.
- **Differs:** H77E used an up-dip seismicity *channel*; no lane used density
  anisotropy as the mechanism. **Cost:** low. **Why ranked #2:** the `n100`
  density estimator is spatially very smooth (100-event neighbourhoods), so
  100 m lineaments are likely below its resolution; quarry/induced clusters
  are strong mimics.

### 3. Conductivity curvature × gravity slope (AGO-page "NOT TESTED" rank 3)
- **Layers:** band 17 (`cond_surf`, MT) + band 5 (`iso_grav_anom_slope`).
- **Signature:** curvature ridge of surface conductivity (clay-cap/alteration
  margin) coincident with a gravity step.
- **Why missing:** hydrothermal alteration along a buried conduit changes
  conductivity without surface rupture.
- **Differs:** H77G used cond as one of 10 HGB channels (clay-cap margin
  *level*), not a curvature×gravity product surface. **Why ranked #3:**
  in-repo irregularity IR-H77G-01 — band 17 carries 3.5 km / 7.0 km FFT peaks
  from its coarse regional MT mesh (30.5% zero steps), so curvature would
  first detect the mesh; needs notch-filtering work before it tests geology.

### 4. Geodetic second-invariant zero-crossings × topographic openness (AGO "NOT TESTED" rank 4)
- **Layers:** band 4 (`geod_2ndinv`), band 8 (`geod_dilaterate`), band 12.
- **Signature:** strain-rate invariant ridges/zero-crossings marking actively
  extending block boundaries.
- **Why ranked #4:** GNSS-derived strain fields have ~10s-of-km wavelengths;
  the family strain-gauge attempt (62GEMSDOE) scored 0.0000 OWNER-REPORT.
  Cost medium, expected gain very low at 100 m pixel scale.

### 5. Depth-to-basement signed-throw step polarity
- **Layers:** band 15 (`depth_to_base_surf`).
- **Signature:** signed basement step with consistent down-to-basin polarity
  along strike.
- **Why ranked #5:** basement-gradient channels already sit inside shipped
  ML stacks (`base_grad_log`, `base_step_along_maggrad` in the 54-channel
  receipt) and the family basement-step lane (61GEMSDOE) scored 0.0009
  OWNER-REPORT. Near-zero marginal novelty.

## Decision

Implement **#1 H82-TCSTEP** exactly as preregistered in
`registry/preregistration_h82.json` (3 experiments, stop rules included).
No new external data is needed; the one core input was restored this session
from the SHA-pinned owner mirror (`registry/data_manifest.json`, sha256
4371c82e… verified). Official sources for manual review: competition
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ ·
reference solution https://github.com/drivendataorg/gems-prize-reference-solution ·
GeoDAWN https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and ·
rules PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf · EPSG:32611 https://epsg.io/32611
