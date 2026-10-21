# H82 — candidate geological hypotheses (ranked BEFORE experiments, 2026-10-21)

*Written before H82 Exp 1. Evidence classes: **[OFFICIAL]** organizer/USGS source ·
**[MEASURED]** computed from SHA-pinned bytes in this repo · **[OWNER-REPORT]**
family ledger without an organizer receipt · **[BLOCKED]** not obtainable from
this sandbox, with the check stated. No leaderboard projection is made anywhere
in this document; the local hide-and-recover holdout is the only gate before any
slot decision, and slot promotion is a separate human selector step.*

## Competition anchors (verified 2026-10-21)

- Task: predict geothermal-indicative faults **not** in the USGS/INGENIOUS
  catalogue; initial-round truth was manually mapped by experts; final round
  rescores against an expanded label set **[OFFICIAL]**
  (https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/).
- Metric: distance-weighted Tversky (DTI), α=0.2, β=0.8, triangular kernel
  R=300 m; worked example TP_w=3.00, FP_w=1.89, FN_w=2.00 → 0.60 **[OFFICIAL]**
  (same page, §Performance metric — fetched this session).
- Submission: single-band float32 GeoTIFF, EPSG:32611, 100 m, same bounds,
  values in [0,1] **[OFFICIAL]** (same page, §Submission format).
- Public leaderboard #1 = **0.4413 ORGANIZER-CONFIRMED** (xiaofanhu), #2 0.3904,
  #3 0.3894 — fetched this session from
  https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/.
  The brief-era references 0.3195 / 0.3262 are superseded by this read.
- Family best `h33-h33-2-b2…-zeros` = **0.2778 [OWNER-REPORT]** — see IR-82-01
  below: the source site itself labels the file UNSCORED.
- Repo holdout incumbent to beat: **0.152718 HOLDOUT-DTI**
  (`gems57-pooled-hide-v2`, seed 20, 11,321 withheld positives, 95% CI
  [0.131548, 0.175185], H77-Step receipt in `evidence/h79_orient_run_card.json`).
  H77E-pure later reported 0.155623 on the same instrument but was stopped as a
  registry duplicate (59 informative overlaps), so 0.152718 remains the
  promotion bar *and* any new candidate must also pass the uniqueness gate.

## Why h33-2-b2 plausibly leads the family — PhD-level mechanism (attribution stays OWNER-REPORT)

**[MEASURED lineage, sibling audits]** One surface (H19-5 multiline-corroborated
ridge, 121,131 px solid, 0.1922 [OWNER-REPORT]) was progressively *thinned*,
never re-detected: dotted d1.5 (60,069 px, 0.2477) → dotted d2.8 (44,090 px,
0.2600) → delete dots 1 px from catalogue (40,199 px, 0.2708) → delete dots
≤2 px / ≤200 m from catalogue (**37,654 px, 0.2778 [OWNER-REPORT]**)
(`research/why-h33-scored-02778.md`; GEMSDOE32 site fetched this session).

The mechanism is metric algebra, not new geology. Under
DTI = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w) with TP_w using a **max** over the
300 m kernel, prediction mass next to an already-covering pixel earns ~0 extra
TP_w but pays full 0.2·FP_w. Spacing binary dots ≈3 px apart preserves most
along-strike kernel coverage (weights 1, 2/3, 2/3 per 3 truth px) while cutting
emitted mass ~3×; deleting the ≤200 m collar removes mass whose only nearby
truth is the already-mapped catalogue (masked from initial-round scoring).
GEMSDOE32's own derivation agrees: a marginal dot pays iff its realised kernel
credit k > α·DTI ≈ 0.052 at DTI≈0.26. Two consequences:

1. **The 0.2778-class result is an emission optimum on a 0.19-class field.**
   Re-thinning the same H19-5 support cannot supply the ≈60% more credit per
   dot needed to reach 0.4413 at fixed mass — only a better *field* (mass where
   hidden faults actually are) can.
2. **Catalogue hide-and-recover cannot see the live effect directly**: the same
   H33 dots score near-zero on catalogue holdout by construction (they avoid
   the catalogue the holdout hides). Holdout deltas are therefore a *necessary*
   sanity gate (the field must recover expert-style faults), never a live-score
   predictor. This lane treats them exactly that way.

**Can a unique submission beat 0.2778?** Conditionally yes — if a genuinely
different physical field places 37k dots with higher realised credit density
than H19-5's ridge, the same emission algebra multiplies it. Nothing in this
repo has yet proven such a field on the frozen holdout *and* passed uniqueness;
that is the bar this lane tests, and a negative result is the expected-value
outcome to publish, not to hide.

### IR-82-01 — 0.2778 attribution irregularity (flagged for review)
GEMSDOE32's site (fetched this session,
https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html) states for this
exact file: *"projected live DTI 0.2747 (a MODEL, not a score). NO ORGANISER
SCORE EXISTS for this or any artifact in this repository"*, and its paste-in
note ends `UNSCORED`. The brief's `0.2778` therefore has **no organizer receipt
linking score to these bytes** in any inspected source. Disposition: 0.2778 is
labelled [OWNER-REPORT] everywhere in H82 outputs; the mechanism analysis above
is conditional on the reported *ordering* (0.1922 < 0.2477 < 0.2600 < 0.2708 <
0.2778), which the algebra predicts regardless of attribution.

### IR-82-02 — Band-tag unreliability confirmed on restored bytes
`training_features.tif` (SHA-256 `4371c82e…`, restored this session) band 6 is
tagged "Tilt angle or total curvature" but is radiometric total count
(Spearman +0.99995 vs external TC, `evidence/band_diagnostics.json`); band 18
is a signed ∂g/∂x, not a gradient magnitude; bands 10/16 tags are hedged
("Distance to earthquake…" / "Earthquake intensity or density…"). Disposition:
H82 uses bands 1/2/14 whose identities (mag_anom / rtp / tmi) are corroborated
by cross-derivative checks in `evidence/band_diagnostics.json` (band 3 ≈ |∇TMI|
ρ=0.998, band 9 ≈ FFT vertical derivative ρ=0.990), and re-verifies the
RTP↔TMI relationship numerically in Exp 1 before trusting the discordance
operator.

## Ranked untried candidates

Ranking key: expected HOLDOUT-DTI improvement ÷ implementation cost, judged
*before* any H82 experiment. "Untried" = grep-verified absent as an implemented
primary in `src/`, `scripts/`, `research/` of this repo on 2026-10-21.

### Rank 1 — H82-1 (PRIMARY LANE): RTP–TMI remanence/rotation discordance breaks
- **Layers:** `training_features.tif` band 14 (`tmi`), band 2 (`rtp`),
  band 1 (`mag_anom`) only. No external data.
- **Physical signature:** RTP repositions anomalies assuming induced
  magnetization parallel to the present field. Where crustal blocks carry
  **remanent** magnetization — reversed-polarity volcanics, or fault blocks
  rotated about vertical axes in the Walker Lane / central Nevada
  transtensional system — the local linear map RTP ≈ a·TMI + b breaks down:
  local slope `a(x)`, correlation `r(x)` and regression residual change across
  the block boundary. Operator (label-free): high-pass both fields
  (σ=3 px), Gaussian-weighted (σ_w=2.5 px) local regression moments →
  `resid_rms`, `r`, `|∇r|`, `a`, `|∇a|`, signed z-difference and its gradient,
  plus the mag_anom↔RTP residual as a second pair (8 channels).
- **Why a catalogue-missing fault:** a block-bounding fault under alluvium has
  no scarp (DEM/lidar lanes blind) and may have weak *amplitude* contrast
  (similar susceptibility both sides — tilt/AS/cross-gradient lanes weak),
  while the magnetization-*direction* contrast persists. Experts re-mapping
  GeoDAWN magnetics draw exactly such domain boundaries as new faults.
- **Differs from this repo:** every prior magnetic lane uses amplitude/source
  geometry — AS ridges (XSEN), tilt zero-crossings (H80T/contactphys),
  upward-continuation persistence + Euler depths (H77E), cross-gradients
  (H77D/H79), demagnetization *troughs* (H77G), damage-zone texture
  *suppression* (H77E-2, negative at 0.0396). None uses RTP-vs-TMI local
  regression discordance; remanence appears only as a named *mimic*.
- **Named non-fault mimic:** unfaulted margins of reversely-magnetized
  volcanic flow units and GeoDAWN survey-block levelling seams.
- **Cost:** low (3 bands, Gaussian filters). Validatable: yes, cached stack.

### Rank 2 — H82-2: concordant gravity+basement flexure under low relief
- **Layers:** bands 13 (`iso_grav_anom`), 15 (`depth_to_base_surf`), 19
  (`det_elev_slope` as low-relief gate).
- **Signature:** Laplacian zero-crossing (flexure/inflection) ridges at
  σ=2.5/5 px with *sign-consistent* curvature in both fields, restricted to
  low-relief basin floors — the buried-fault hinge, not the gradient maximum.
- **Why missing:** blind intrabasin faults beneath flat playa fill; proposed
  as H79 candidate 4 but never implemented in this repo.
- **Mimic:** depositional basin-margin onlap without active faulting.
- **Cost:** low. Ranked below H82-1 because gravity/basement grids are smooth
  (effective resolution ≫300 m), capping kernel credit.

### Rank 3 — H82-3: seismicity-density structure-tensor lineaments
- **Layers:** bands 10 (`deq_n100a15`), 16 (`ieq_n100a15`) — tags hedged
  (IR-82-02), identities to be re-derived from distributions first.
- **Signature:** structure-tensor coherence ridge of the combined density
  (elongated epicentre clusters), not density amplitude (H77G used the
  ieq/(ieq+deq) fraction only as a regime scalar).
- **Why missing:** actively creeping/swarming blind faults in accommodation
  zones with no mappable scarp.
- **Mimic:** geothermal-production / injection-induced swarms; network
  detection-threshold gradients.
- **Cost:** low–medium. Ranked third: 100 km/15° kernel parameters in the band
  names suggest very coarse support vs the 300 m metric kernel.

### Rank 4 — H82-4: basement zero-hinge (bedrock–alluvium contact) continuations
- **Layers:** band 15 zero/near-zero contour + band 12 (`det_elev`).
- **Signature:** the depth-to-basement ≈0 hinge line where it extends beyond
  mapped range fronts — a buried front, traced as a contour rather than a
  gradient ridge (gradient forms are inside H77G/H80T blocks; the explicit
  hinge-contour emission is not).
- **Mimic:** depositional pinch-out of basin fill against unfaulted bedrock.
- **Cost:** low. Ranked fourth: much of the hinge coincides with the mapped
  catalogue by construction, shrinking the off-catalogue remainder.

### Rank 5 — H82-5: Quaternary volcanic-vent alignment (Hough) lines — NOT VIABLE as primary
- **Layers:** GDR 1391 volcanic vents in footprint
  (`registry/data_manifest.json` id `ext_gdr_volcanic_vents_in_footprint`,
  672 B CSV — a handful of points) [OFFICIAL free source, CC BY 4.0,
  https://gdr.openei.org/submissions/1391, DOI 10.15121/1881483; obtainable
  via the SHA-pinned mirror].
- **Signature:** collinear vent triplets → fissure/fault-control lines.
- **Verdict on viability:** obtainable but far too sparse to support a
  37,654-dot field or a powered holdout test; listed for completeness, not
  proposed as this session's lane. (ASTER SWIR alteration, the other classic
  untried data source, is **[BLOCKED]**: NASA LP DAAC host is outside the
  sandbox egress allowlist — H77G-4 disposition, reaffirmed.)

## Frozen protocol (see `registry/preregistration_h82rrd.json`)
Exp 1: build 8 RRD channels label-free; per-channel leakage canary
(symmetric AUC > 0.90 = stop); Phase-1 surface rank-correlation vs local
registry scope. Exp 2: seed-20 buffered whole-component hide-and-recover
(`gems57-pooled-hide-v2`) arms S_rrd / A_rrd / A_std19 / A_std19+rrd /
A0_random at fixed budget 37,654, exclusion >2 px, min-sep 2.8 px — **no
operating-point scan**. Exp 3: build the preregistered winner's full-map TIF
via the shared writer, Phase-2 dot uniqueness, validator, run card.
Promotion requires *all* of: beats 0.152718 with paired 95% CI excluding 0,
passes ρ≤0.90 / overlap≤0.70, human selector approval. Otherwise verdict
**negative**, file marked OK TO DOWNLOAD FOR INSPECTION; NOT OK TO SUBMIT,
zero weekly slots spent.
