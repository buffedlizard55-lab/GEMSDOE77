# H85 — Preregistered candidate hypotheses (written BEFORE implementation, 2026-10-21)

Session branch `arena/7a05d452-gemsdoe77`. No single method paragraph was assigned to this
invocation (same irregularity as prior sessions); this file preregisters one narrow,
autonomous lane and ranks the alternatives that were considered and **not** run.
Budget: ≤3 experiments, ≤2 hours. Evaluator (unchanged): `gems57-pooled-hide-v2`
(alpha 0.2, beta 0.8, 300 m triangular kernel). All scores below, if produced, are
HOLDOUT-DTI unless labelled ORGANIZER-CONFIRMED (none exist in this session).

## Why the owner-reported 0.2778 (H33) leads the family — PhD-level reading

Byte facts re-verified in this repo (`research/why-h33-scored-02778.md`, H83 addendum):
H33 (`h33-2-b2`, 37,654 dots) is an **exact subset** of the D2.8 dotted raster (44,090 dots);
all 6,436 removed dots lie 1.0–2.0 px from the mapped catalogue; the closest kept dot is
√5 px away. The 0.2778 / 0.2600 numbers are **owner-reported, with no organizer receipt**
(the GEMSDOE32 site itself states no organiser score exists for its artifacts), so the
following is a conditional, mechanistic explanation — not causal proof:

1. **DTI geometry rewards dotted, not ribbon, emission.** With a 3 px triangular kernel,
   `TP_w` saturates at 1 per truth cell (max, not sum). A contiguous ribbon along a lineament
   spends many dots re-covering the same truth cells while every dot adds FP mass
   `1 − k(d)`; thinning to ~2.8 px spacing preserves most max-kernel coverage per unit FP.
   This is why the whole dotted lineage (H28 → D1.5 0.2477 → D2.8 0.2600, owner-reported)
   climbed while continuous/ribbon fields stalled near 0.15–0.19.
2. **Near-catalogue dots are structurally unprofitable.** Mapped catalogue pixels are masked
   out of scoring, and the hidden truth is *new* faults. A dot 1–2 px from a mapped trace
   mostly covers already-mapped structure (excluded) rather than a genuinely unmapped
   strand, so it earns little `TP_w` while paying full FP cost. Deleting exactly those
   6,436 dots is therefore the cheapest possible DTI gain — pure denominator reduction —
   *if* they earned ~zero credit. Under that assumption the two reported scores imply a
   hidden-truth size K ≈ 13.2–14.1k px (≈22 % of the 60,988-px catalogue) and H33 coverage
   ≈37–40 % (derivation in `research/why-h33-scored-02778.md` §H83). If the removed dots
   did earn credit, K is not identified and this inference fails.
3. **The base surface (H19-5 multiline corroboration) already localises well.** H33 adds no
   new geology; it is emission hygiene on a good surface. Hence beating it requires either
   (a) a *better surface* (new physical signal) with equally disciplined emission, or
   (b) nothing — further pruning of H33 itself trades coverage for FP at a worsening rate.

**Can we beat 0.2778?** Unknown and not claimable in advance. The only defensible route is
(a): a catalogue-independent surface that recovers withheld whole fault components better
than matched controls on the shared holdout, emitted with H33's two hygiene rules
(spaced dots; >2 px from the catalogue). A holdout win is necessary under the user's
promotion rule; it is **not** a leaderboard projection.

## Ranked candidates (expected DTI value ÷ implementation cost)

| Rank | Layers + signature | Why it should catch an *unmapped* fault | Difference vs everything in this repo | Named non-fault mimic | Cost |
|---|---|---|---|---|---|
| **1 — SELECTED: H85-XWC** | `training_features.tif` band 2 `rtp` + band 13 `iso_grav_anom`: multi-scale bandpass **cross-field coherence** (FFT implementation of wavelet-style magnitude coherence) × joint edge energy × cross-scale phase consistency, wavelengths 4/8/16 px | A buried fault under basin fill juxtaposes basement blocks differing in **both** susceptibility and density → a coherent subsurface edge in two independent physics with no surface scarp to map. Single-field edges (dikes in magnetics only; basin-thickness gradients in gravity only) are suppressed by requiring cross-field agreement | Repo-wide grep: no wavelet / cross-spectrum / cross-field coherence implementation exists. Distinct from single-field phase congruency (contactphys/H77PD), H79 real-space orientation agreement, H82-RRD same-sensor RTP~TMI regression discordance, tilt/analytic-signal edges (amplitude-based, single-field). Proposed as rank-1 conditional in `docs/review-2026-10-21-arena-6a5eca5f.md`, blocked then only by the missing stack — now restored sha-verified (`4371c82e…`) | Lithologic / intrusive contact (shared mag+grav edge, no throw); GeoDAWN flight-line / survey-block seams | Medium; CPU FFT, no new external data |
| 2 — not run | Band 15 `depth_to_base_surf` + band 13 gravity: **basement-step / gravity-step lateral offset** (dip indicator: cross-correlate the two step edges along local strike; keep 200–600 m systematic offsets) | A dipping buried fault offsets its shallow (gravity) and deep (basement) expressions laterally; a vertical lithologic contact does not | H77G/H77H-DD used fixed +500 m downdip lag on RTP×MT, not a measured per-pixel basement-vs-gravity offset field | Unconformity / paleo-valley edge in basement topography | Medium |
| 3 — not run | Band 12 `det_elev`: drainage-network **χ-asymmetry + divide-migration** metric across candidate lineaments | An active unmapped fault tilts blocks, driving systematic cross-divide χ contrasts before a scarp is mappable | Knick/scarp lanes used local DEM derivatives; none built flow routing / χ topology (needs care at 100 m) | Regional tilting, lithologic erodibility contrast | High |
| 4 — not run | Bands 10/16 seismicity density (`deq/ieq_n100a15`) × band 4 strain 2nd invariant: **seismicity–strain colocalisation residual** after removing catalogue-distance trend | Creeping unmapped faults may host microseismicity where geodetic strain concentrates | Strain used as a gate/feature before, never as a catalogue-deconfounded colocalisation residual; seismicity bands largely unused as primary signal | Quarry blasts / geothermal-plant injection seismicity | Low–medium |
| 5 — not run | GeoDAWN radiometrics: **lagged cross-strike U–Th phase offset** (already rank-2 in `docs/hypotheses.md`, still unimplemented) | Displaced alteration halo flanking a buried structure | Tests spatial phase, not same-pixel coincidence (H79-RH used lagged halo amplitude, negative) | Hillslope-transported soil boundary | Medium |

Only rank 1 is authorised this session. Ranks 2–5 need no new external data (all layers are
in the restored pinned stack / GeoDAWN packs); none is proposed as viable until separately
preregistered and holdout-tested.

## H85-XWC preregistered protocol (fixed before seeing any holdout number)

- **Inputs (label-blind surface):** bands 2 (`rtp`) and 13 (`iso_grav_anom`) only.
  Sentinel/non-finite scrub to in-footprint median; high-pass = field − Gaussian(σ=8 px);
  robust-z inside footprint. Bandpass at wavelengths 4/8/16 px via FFT annulus (cosine-tapered).
  Per scale: coherence `C = |Sσ(X·Y)| / sqrt(Sσ(X²)·Sσ(Y²))` with σ = wavelength/2;
  joint energy `E = sqrt(Sσ(X²)+Sσ(Y²))` rank-normalised in footprint;
  phase sign `sign(Sσ(X·Y))`. Surface `S = mean_s(C_s · E_s) · (0.5 + 0.5·phase_agreement)`,
  rank-normalised to [0,1] in footprint, 0 outside. No catalogue, label, or scored-raster input.
- **Canary (Exp 1):** single-feature AUC of S and each component vs withheld truth
  (seed-20 draw) and vs full catalogue (diagnostic). AUC > 0.90 ⇒ leakage, stop.
- **Holdout (Exp 2):** `buffered_component_draw` seeds 20 and 30 (whole components,
  hide_frac 0.20, buffer 3 px, boundary 12 px). Per draw, emission uses that draw's
  **visible** catalogue only: allowed = footprint ∧ dist(visible) > 2 px.
  Emission = template `emit_decimated` (3 px tiles, E[k] ranking on p=S) at fixed budget
  **37,654 dots** (H33-matched). Arms, identical emission/budget/masks:
  `xwc` (candidate), `rtp_only` (p = rank of multi-scale RTP edge energy),
  `grav_only` (same for gravity), `random` (uniform allowed cells, seed-fixed).
  Score with `evaluate_holdout.evaluate` per fold; pool with `pooled_summary`
  (1,000 paired 20 km-block bootstrap draws). Report HOLDOUT-DTI + 95 % CI +
  withheld-positive count per seed. No parameter/budget tuning after seeing results.
- **Uniqueness:** surface Spearman ρ and final-dot 3-px forward overlap vs (i) restored
  scored lineage (D2.8, H19-5) and (ii) every sparse binary raster in `docs/downloads/`
  readable on the pinned grid. Literal-rule caveat: the known dense witness
  (`ab0a0a62…`, positive on 100 % of allowed cells) makes the literal >0.70 overlap gate
  unsatisfiable for *any* non-empty candidate including H33 itself; both readings are
  reported and the conflict is flagged as an irregularity for the protocol owner — the
  dense rasters are compared by ρ (the protocol's surface test), sparse dot rasters by
  dot overlap (the protocol's dot test). No silent exemption: the run card records both.
- **Build (Exp 3):** final dots from full-catalogue masking (dist > 2 px, 0 on catalogue),
  budget 37,654, written only via `gems57.submission_writer` (all-finite float32 in [0,1],
  zeros outside footprint, no nodata sentinel — the configuration that cannot produce the
  portal's "Predicted values must be in range [0, 1]" error from NaN/sentinel values).
- **Verdict rule:** `promote` only if Exp 2 candidate beats *all* matched controls with
  paired-CI lower bound > 0 on both seeds **and** informative uniqueness passes
  (max |ρ| ≤ 0.90, max sparse dot overlap ≤ 0.70) **and** validator passes 15/15.
  Otherwise `negative`. Either way the file is published with an unambiguous
  OK-TO-DOWNLOAD / OK-TO-SUBMIT banner; promotion to a weekly slot remains the
  separate selector step (0 slots used by this session).
