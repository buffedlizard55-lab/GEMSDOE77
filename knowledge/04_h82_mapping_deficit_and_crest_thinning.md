# H82 — what a censored catalogue-intensity model actually buys you (nothing), and what geometry buys you (a lot)

*Session 2026-10-21, branch `arena/ff72e371-gemsdoe77`. Everything below is HOLDOUT-DTI from the shared
`gems57-pooled-hide-v2` hide-and-recover evaluator (whole fault segments withheld with a buffer, catalogue features
rebuilt from visible faults only, pooled α=0.2 / β=0.8 with a 300 m triangular kernel). No number here is an organizer
score, and none of them is comparable across lanes — only paired arm differences inside one run are.*

## 1. The idea that failed, and why it was worth testing

The competition scores faults **absent** from the USGS/INGENIOUS catalogue. The obvious inference is that one should
not rank pixels by "how fault-like is this?" but by "how much more fault than the catalogue admits is here?".

Implementation (`src/gems82/deficit.py`): aggregate the physics stack to 2 km blocks, fit a Poisson-loss
`HistGradientBoostingRegressor` of the *mapped* fault-pixel rate on block physics, and take the positive residual
`D(b) = max(0, λ̂(b) − n_visible(b))` as "fault length this setting implies but the catalogue does not contain".
Smooth, rescale to [0,1], multiply into the per-pixel surface with a floor of 0.10.

**Result: refuted on both seeds.**

| arm | seed 20 | seed 30 |
|---|---|---|
| `A_std` (19-band control) | 0.089519 | 0.096903 |
| `A_fmd` (control × deficit weight) | 0.066759 | 0.069948 |
| `A_def` (deficit field alone) | 0.044227 | 0.044544 |
| `A0_random` (uniform random dots) | 0.047619 | 0.048088 |

Paired `A_fmd − A_std` = **−0.022761**, CI [−0.033283, −0.012634] (s20) and ≈ −0.027 (s30). The deficit field alone is
*below* random.

**Why it fails, mechanistically.** Three reasons, all of which generalise to anyone tempted by the same idea:

1. **The residual is not where the missing faults are; it is where the model is wrong.** A GBM on 2 km block physics
   has a block-scale bias wherever the physics is unusual (basin interiors, volcanic tablelands, survey-edge
   artifacts). Those blocks get a large positive residual and no faults.
2. **Scale mismatch.** The metric credits a dot within 300 m of a fault pixel. A 2 km block weight cannot steer a dot
   at 300 m resolution; it can only re-rank whole neighbourhoods — and re-ranking by a noisy quantity strictly loses
   when the base ranking is already informative.
3. **Normalisation amplifies noise.** `deficit_field` rescales by the 90th percentile of the positive deficit, so even
   a deficit that is pure model noise is stretched to [0,1] and multiplied in at full strength. Pinned in
   `tests/test_h82.py::test_deficit_is_zero_where_the_catalogue_already_explains_the_intensity`.

The leakage canary saw this coming: the deficit weight's own single-channel AUC against withheld positives is
**0.5693** (ceiling 0.90, flagged = none). Any channel that close to 0.5 cannot carry a multiplicative re-ranking.

## 2. The idea that worked — geometry, not geology

A gradient-boosted per-pixel fault probability is **blobby**: its super-level sets are 2-D patches several pixels
across. The DTI kernel is linear in distance to a **1-pixel** truth trace, so a dot on the flank of a patch pays the
same α-penalty as one on its crest and collects less credit. Thinning the ranking surface to its across-ridge maxima
(Steger-style NMS on the Hessian's most-negative eigenvector, `fields.crest_mask`) before allocation is worth:

| comparison | seed 20 | seed 30 |
|---|---|---|
| crest NMS − unthinned | **+0.008227** [0.002425, 0.013893] | **+0.006390** [0.000146, 0.012690] |
| + 3 px minimum dot separation | +0.0023 … +0.0029 | same direction |
| 48-channel + crest − 19-band control | **+0.022584** [0.012495, 0.032946] | **+0.012414** [0.003675, 0.020865] |

This independently reproduces the **+0.0292 live DTI** that the 0.2778 family attributed to ribbon→1-px thinning. It
is an *allocator-side* fix: free, geology-agnostic, and it compounds with any surface. **If you only take one thing
from this lane: thin the surface before you allocate.**

Budget algebra for the published file (tpw ≈ 1,870, denominator ≈ 16,931): doubling to 75 k dots gives ≈ 0.099–0.114,
halving to 20 k gives ≈ 0.097. 37,654 is near-optimal; a budget sweep is not worth an experiment.

## 3. The uniqueness rule is not a duplication test against dense emitters

The published file fails Rule 1: **0.9189** of its dots lie within 3 px of the dots of a GEMSDOE22 raster that emits
**10 % of the entire map** (516,738 dots). It also "fails" against 44 rasters whose own 3 px halo covers ≥ 70 % of the
footprint, where a *uniformly random* dot field fires too.

Against the **647** registry rasters of comparable density (≤ 2× this budget) the worst overlap is **0.6745 with zero
firings**. Worst reverse overlap 0.6243, worst Jaccard 0.0407 (limit 0.50), worst |Spearman ρ| 0.1021 on the surface
and 0.0966 on the dots (limit 0.90), and no byte- or decode-identical file among 844 hash-verified rasters.

**Operational lesson for future lanes:** decide the exclusion rule *before* the audit and write it into the
preregistration. We did not, so the verdict here is the literal one (FAIL → research-only), with the density
breakdown published beside it as a diagnostic. Encoding "comparators denser than N× the candidate's budget are
reported separately" in a preregistration is legitimate; doing it after seeing 0.9189 is not.

## 4. Reusable facts

- **Hide-and-recover truth is strictly harder than the live task.** `A_std` recovers 1,495 of 11,321 withheld
  positives (13.2 %) against ~39 % implied by the live 0.2778 submission. Never compare a local DTI to a leaderboard
  number.
- **The same control reads differently in different lanes** (19-band `A_std` ≈ 0.062 in H77G, ≈ 0.13 in H77E, 0.0895
  here) because budget, allocator and fold draw differ. Only within-run paired deltas are reportable.
- **Marginal algebra** (T ≈ 5,075, n = 37,654): `∂DTI/∂T ≈ +5.3e−5`, `∂DTI/∂n ≈ −3.2e−6` → one unit of recall is worth
  about 17 wasted dots. Reaching 0.3195 needs T ≈ 5,720; 0.4413 needs T ≈ 7,900.
- **Already-mirrored external data** (`data/raw/external/`, SHA-pinned): GeoDAWN radiometrics K/Th/U/TC, LiDAR scarp
  derivatives, GeoDAWN extensions, SGMC-derived faults, GDR wells/springs/QFaults/vents. The 48-channel stack used
  here = 28 physics + 20 external. New external *fetches* remain impossible from the sandbox (egress is limited to
  github.com, codeload, api.github.com, pypi, files.pythonhosted, registry.npmjs).
- **Never** let `sgmc_faults_100m.tif` into `X` (pre-Quaternary bedrock contacts, 0.0492 live standalone), and never
  add `dist_known_fault_px` / `known_fault` / `label` as point-table columns (IRR-03 self-distance tautology).

## 5. What to try next (ranked, for the next session)

1. **Crest thinning everywhere.** Re-run the retained `H77-Step` 0.152718 pipeline with `crest_mask` + `min_sep_px=3`
   in front of its allocator. Cost: one experiment. Expected: +0.01 … +0.02 on the same harness, which would be the
   first thing in this repository to move the incumbent.
2. **C2 geodetic slip-budget deficit** (parked in `research/h82_hypotheses.md`): the same "missing fault" logic but
   anchored to a *physical* budget (strain rate vs catalogue slip) instead of a statistical residual — it fails for
   different reasons if it fails, which makes it informative.
3. **Preregister the uniqueness exclusion rule** before the next audit so a clean artifact can actually clear Rule 1.
