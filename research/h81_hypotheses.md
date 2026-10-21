# H81: candidate geological hypotheses (ranked before experiments, 2026-10-21)

*Written before Exp 1. Evidence classes: **[OFFICIAL]** organizer/USGS/DOE source
· **[MEASURED]** computed this session from SHA-pinned bytes · **[OWNER-REPORT]**
family ledger without organizer receipt · **[BLOCKED]** not obtainable from this
sandbox, with the check stated.*

Competition facts used below: task = predict faults **not** in the USGS/INGENIOUS
catalogue; hidden initial labels are expert-mapped new faults; final round rescores
the same submission against expanded labels including **all** faults; metric =
distance-weighted Tversky α=0.2, β=0.8, triangular kernel R=300 m (3 px); grid
EPSG:32611, 100 m, 3730×3292; submission single-band float32 in [0,1]
**[OFFICIAL]** (problem page 967 §metric/§submission-format, fetched 2026-10-21).
Public leaderboard #1 = **0.4413** (xiaofanhu) **[OFFICIAL]** (leaderboard page
fetched 2026-10-21; family best `h33-h33-2-b2…-zeros` = 0.2778 **[OWNER-REPORT]** —
no organizer receipt links those bytes to any leaderboard row; GEMSDOE32's own
site states no organizer score exists in that repository for the file).
Rules: three feedback submissions per week, one final submission, generative-AI
disclosure in finalist narrative **[OFFICIAL]** (GEMS Prize Official Rules PDF
§3.2, fetched 2026-10-21).

## Session-verified byte facts (used for ranking, not as scores)

- H33 raster (registry blob `17a76895…`, 219,065 B): **37,654** binary dots in
  {0,1}, all finite, **0 dots on the mapped catalogue**, 0 outside the footprint,
  minimum mapped-fault distance **√5 px ≈ 2.24 px** **[MEASURED]**.
- Sample submission: 5,167,373 finite pixels (= footprint), **60,988 pixels equal
  to 1 = exactly the 60,988 catalogue positives** — i.e. the reference file is
  NOT "total fault absence" as the page wording suggests **[MEASURED]** (matches
  the pre-existing manifest note; flagged as irregularity IR-81-02).
- `trace_segments_utm11.csv` (owner mirror of INGENIOUS Qfaults-derived
  segmentation): 84,331 segments in **1,126 records**; senses N 79.2%, RL 10.0%,
  LL 9.0%, unknown 1.7%; every segment carries slip_mm_yr > 0 (median 0.1,
  max 4.5 mm/yr) **[MEASURED]**. Zero lanes in this repo read the slip-rate or
  sense columns (`grep slip_mm src scripts`: no hits, 2026-10-21).

## Why H33-2-B2 plausibly leads the family (PhD-level mechanism; attribution remains OWNER-REPORT)

Under the official DTI algebra, adding one unit of prediction mass with realised
kernel weight k raises DTI iff **k > α·DTI** (denominator always rises by exactly
α=0.2). At DTI≈0.28 the bar is k>0.056: a dot must sit within ~283 m of a hidden
fault pixel (without duplicating better cover) to pay for itself. The H19-5 →
dotted → pruned lineage (121,131 → 44,090 → 37,654 px) is a pure emission-side
optimisation: dot spacing ≈ kernel radius removes redundant mass, and deleting
dots ≤2 px from mapped traces removes mass whose only nearby truth is the
catalogue itself (masked from initial-round scoring). The gains are exhaustible:
reaching 0.4413 at fixed mass needs ≈60% more realised credit per dot, which no
re-thinning of the same H19-5 support can supply — only a **better field** (mass
placed where hidden faults actually are) can. This session tests one such field:
slip-budget-weighted fault-tip continuation cones, whose support is disjoint by
construction from ridge-following dots. No leaderboard projection is made; the
local hide-and-recover holdout is the only gate before any slot decision.

## Ranked hypotheses (expected DTI improvement ÷ implementation cost)

### H81-1 (rank 1 — THIS LANE) — Slip-budget fault-tip continuation cones

- **Layers:** `trace_segments_utm11.csv` (record_id groups, endpoint geometry,
  slip_mm_yr, sense) + `existing_faults.tif` (mask only). No potential fields,
  no radiometrics, no lidar, no ML.
- **Physical signature:** deterministic outward propagation cone (length 2.0 km,
  half-angle 30°, linear decay) from each **strand-level** fault tip along its
  end-tangent, weighted by the strand's max-slip-rate rank. Strand-level tips
  (degree-1 nodes of each connected trace component, not segment joints) avoid
  digitisation-split false tips. *(Amendment IR-81-04: the as-first-written
  record-level definition was replaced before any outcome because record_id
  groups multi-trace fault zones/systems — same zone name across records,
  max record length 380 km — whose zone-ends usually fall outside the survey
  footprint. See `registry/preregistration_amendment_h81_04.json`.)*
- **Why a catalogue-missing fault:** displacement conservation and d–L scaling
  (Cowie & Scholz 1992, linear d∝L) imply a mapped fault's slip must continue
  or transfer beyond its mapped tip; linear-elastic tip stress concentration
  (Pollard & Segall 1987) localises the continuation/linkage fracture ahead of
  the tip. Experts mapping from lidar/GeoDAWN draw these continuations where the
  scarp-based catalogue stops. High-slip records carry the largest slip budget
  and hence the most required continuation.
- **Named non-fault mimic:** map-edge / tile-boundary cut-ends (a record simply
  ends where mapping stopped — mitigated by dropping tips within 1 km of the
  footprint edge); overlapping en-échelon tips already soft-linked in the
  catalogue.
- **Difference vs implemented work:** no lane in this repo uses slip rates or
  slip sense (grep-verified). H77-Step's Pollard cones and H77D's LOCO tip/relay
  mechanics are geometric/ML features without a slip budget; GEMSDOE33's
  analog-tip-stepover (0.2632 OWNER-REPORT) transfers analog geometry rather
  than record slip. This lane is deterministic, single-surface, catalogue-only.
- **Cost:** low–medium. Validatable locally: yes (record-LOCO hide-and-recover).

### H81-2 (rank 2 — future lane) — Strike-heterogeneity transfer zones

- **Layers:** `existing_faults.tif` (visible only per fold).
- **Signature:** circular entropy of fault-pixel strike (structure-tensor strike
  of the visible raster) in ~2 km windows; high-entropy = intersection/transfer
  complexity.
- **Why off-catalogue:** fault intersections and transfer zones localise new
  fault growth and are under-mapped relative to long planar traces.
- **Mimic:** digitisation jitter and anastomosing mapped swarms (strike noise
  without geology).
- **Difference:** strike-entropy is unused in this repo (grep-verified).
- **Cost:** low. Expected improvement: low–moderate (weak catalogue-holdout
  prospects). NOT run this session (budget).

### H81-3 (rank 3 — future lane) — Long-range N–S Hough lineaments with road rejection

- **Layers:** `topo_10m_derived_100m_u8.tif` (hs_lineament, curv_prof_absmax);
  US Census TIGER roads for rejection (conditional — see below).
- **Signature:** Hough transform for 5–30 km straight lineaments within N–S ±30°,
  keeping basin-crossing lines with low road coincidence.
- **Why off-catalogue:** range-bounding faults buried under basin fill keep
  topographic lineaments without mappable scarps.
- **Mimic:** highways/canals/survey seams (long, straight — the explicit
  rejector target).
- **Difference:** Hough long-lineament detection is unused in this repo (H77-Step
  uses Radon worms on the scarp field; H78 uses lidar consensus, not 10 m
  hillshade Hough).
- **Viability:** CONDITIONAL. H77D references `tiger_road_distance_m.tif` but the
  file is absent from every owner mirror listing checked (60GEMSDOE
  data/external, 2026-10-21); rebuilding needs US Census TIGER/Line (free,
  official) which this sandbox cannot reach. NOT run this session.

### H81-4 (rank 4 — future lane) — Silica-geothermometer-filtered upflow

- **Layers:** `gdr_wellspring_in_footprint.csv` aqueous-chemistry columns.
- **Signature:** Fournier silica/chalcedony reservoir temperature × near-field
  halo, instead of warm-well counts.
- **Why off-catalogue:** deep circulation requires a permeable fault even where
  no scarp exists; reservoir-T filters out shallow/irrigation warmth.
- **Mimic:** deep non-fault aquifers, irrigation return flow.
- **Difference:** H77D/H78 use well/spring proximity/counts; chemistry-derived
  reservoir temperature is unused.
- **Cost:** medium. Expected improvement: low (H78-2 found zero live weight for
  thermal proximity). NOT run this session.

### H81-5 (rank 5 — BLOCKED here) — Sub-km seismicity lineaments

- **Layers needed (specific free official source):** USGS ComCat earthquake
  catalog via the FDSN event service
  `https://earthquake.usgs.gov/fdsnws/event/1/query` (free, public, includes
  moment tensors), or the Nevada Seismological Laboratory catalog.
- **Obtainability check performed 2026-10-21:** `curl -m 10` to
  `earthquake.usgs.gov` from this sandbox returns HTTP 000 / exit 35 (host
  outside the sandbox egress allow-list). No owner GitHub mirror of an event
  catalog was found. **Named but not viable in this session**; no seismicity
  surface is fabricated. The shipped seismic bands are 100 km-radius aggregates
  and cannot localise a fault at 300 m.

## Decision rule (frozen in `registry/preregistration_h81.json`)

Exp 1: build frozen features + single-feature leakage canaries (symmetric AUC >
0.90 stops the run) + surface uniqueness (ρ > 0.90 stops the run). Exp 2: pooled
HOLDOUT-DTI (seed-20 whole-component hide-and-recover, same folds as H77-Step /
H79-RH) for primary vs tip-uniform vs normal-only vs scarp-gated vs random.
Exp 3: emit dots, final-dot uniqueness, shared-writer artifact. Promote iff
pooled DTI beats the 0.152718 holdout best with paired 95% CI excluding 0 AND
literal uniqueness passes AND canaries clean AND validator passes; otherwise
verdict negative. Selector approval is separate; no slot is spent by this lane.
