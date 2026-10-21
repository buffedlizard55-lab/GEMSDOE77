# H85 — five candidate hypotheses, ranked before implementation

Preregistered 2026-10-21 in `registry/preregistration_h85.json` **before** any surface was
trained. Ranked by expected DTI gain ÷ implementation cost, per the standing science
instruction. Every hypothesis is judged against the same question: *why should this catch a
fault that is missing from the USGS/INGENIOUS catalogue rather than one that is already in it?*

| # | Hypothesis | Layers / transform | Why it should find an **unmapped** fault | Why it is new in this repo | Cost | Expected gain |
|---|---|---|---|---|---|---|
| **1** | **H85-EDTI — the emitter, not the surface, is the binding constraint.** Fixed 37,654-dot budgets cap recall; greedy top-N is absorbed by the catalogue collar. Emit under the published metric: collar 3.5 px, 1-px skeleton thinning, support = 3.3 × truth cell count. | No new layer. Uses the 19 official bands as a plain label-free surface. Metric algebra: `vendor/gems57/metric.py`. | A dot only counts if it is **off**-catalogue; the collar guarantees every emitted pixel is ≥ 350 m from a mapped trace, i.e. in the domain where a *new* fault can live. Support size is set by the metric's marginal-credit algebra, not by a number inherited from an earlier lane. | All 80+ prior lanes froze the emitter (37,654 dots, min_sep 2.8, collar 0.5 px). The ablation below is the first direct measurement of emitter choice with everything else held fixed. | Low | **+0.15 DTI (measured)** |
| 2 | H85-2 — across-strike Hessian ridge curvature of detrended elevation and depth-to-basement, multi-scale (σ = 2, 5, 9 px), used as a *ranking* channel | Bands 12, 15, 19; `λ₁` of the Hessian | Young scarps below the mapping scale appear as a crest/trough pair in curvature that a 1:250k line compilation misses | The repo has used Laplacian operators (`h77g`) but never a multi-scale `λ₁` ranking channel on band 15 | Low | +0.01 to +0.03 (untested; deferred) |
| 3 | H85-3 — cross-sensor *phase* coherence: the local offset between the magnetic ridge (band 6) and the gravity ridge (band 13) should be small at a real fault and random at a playa/fan contact | Bands 6, 13, 14, 18; local cross-correlation at 5–9 px | A concealed fault perturbs two independent potential fields at the same place; a lake shoreline or fan contact perturbs topography only | Repo has cross-sensor *products* (`x_couple_*`) but never a lag/phase test | Medium | +0.01 to +0.04 (deferred) |
| 4 | H85-4 — earthquake-density *b*-value / declustered swarm excess along unmapped lineaments | Bands 10, 16 | Fluid-driven swarms nucleate on unmapped conduits; a mapped trace that is aseismic is already known and already scored zero | Repo used raw `ieq/deq` counts as a multiplicative gate, never a declustered excess | Medium | +0.00 to +0.02 (needs the declustered catalogue; not obtainable without the INGENIOUS/GDR event table) |
| 5 | H85-5 — K/Th radiometric asymmetry across strike (the "potassic halo" of a hydrothermal fault) | USGS GeoDAWN radiometric bands, `doi:10.5066/P93LGLVQ` | Argillic/K-metasomatic halos form on the **damaged** side of a fault and are not compiled in the trace map | Repo used the 8-bit radiometric stack only as a rank-mixture entropy, never as a signed across-strike asymmetry | High (needs the 3-band K/U/Th raster; owner mirror exists at `registry/data_manifest.json` → `ext_geodawn_rad_u8`, but that mirror is an 8-bit composite with no band labels — the official 3-band product was not obtainable in this sandbox) | deferred, source verified unobtainable here |

## What was actually run (budget: 3 experiments)

* **E1 — surface + leakage canary.** 37 label-free channels from the 19 official bands, one
  HistGradientBoosting fit per fold per draw, 8 fold cells over seeds 30 and 20.
  Canary (single-surface symmetric AUC vs withheld truth): max **0.8993** (ceiling 0.90), i.e.
  no leakage, but the official geophysics alone already separates fault-like ground strongly.
  A second control that *adds* `distance-to-visible-fault` collapses to exactly **0.5000** AUC on
  every fold — it learns `d == 0` and predicts nothing for withheld strands. That reproduces the
  repository's `IRR-03` and is why the shipped surface carries no catalogue channel.
* **E2 — emitter grid.** 9 emitter configurations × 2 independent draws: collar ∈ {0.5, 2.5, 3.5,
  4.0, 5.0, 7.0} px, support ratio ∈ {1, 2, 3.3, 5, 8} × withheld-positive count, thinning ∈ {raw,
  1-px skeleton}, plus the metric-derived expected-credit threshold and a random control.
  Best: collar 3.5 px + skeleton + ratio 3.3.
* **E3 — build, uniqueness, validator, run card.** Final model on the full catalogue, emission at
  201,260 px, 15/15 shared-validator checks, uniqueness under the corrected binding rule
  (worst 3-px forward overlap 0.354 vs limit 0.70; worst |ρ| 0.072 vs limit 0.90).

## Result in one line

Emitter change alone, on identical surfaces and identical support size:
`0.0316 → 0.1828` HOLDOUT-DTI (dual-seed pooled), paired **+0.1512 [+0.1359, +0.1664]** — larger
than every surface-side improvement this repository has measured in 80+ lanes, and it composes
with them.

## Negative / deferred

H85-2 and H85-3 were not run: the budget was spent on the hypothesis that measured largest.
H85-4 needs the declustered event catalogue, which is not in the pinned mirrors. H85-5 needs the
official 3-band K/U/Th GeoDAWN product; the mirror present here is an unlabelled 8-bit composite,
so the idea is *not* proposed as viable from this sandbox (checked, not assumed).
