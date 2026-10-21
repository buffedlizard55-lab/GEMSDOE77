# Why the H33 construction plausibly explains the family's highest reported score — and what beating it would take

*Analysis, 2026-10-21. Evidence classes: **[OFFICIAL-SNAPSHOT]** a dated public organizer/source record, not freshly fetched here · **[RASTER-FORENSICS]** recorded from a hash-pinned owner-mirror or sibling audit, not independently reread in this checkout · **[OWNER-REPORT]** family file-to-score claim without an organizer receipt · **[CONDITIONAL]** algebra or interpretation under stated assumptions.*

## The question

The brief asks why `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` (GEMSDOE32)
was reported at **0.2778 [OWNER-REPORT]** — the family's best — and whether a
unique submission could beat it. A repository snapshot of the 2026-10-21 public
leaderboard recorded **0.4413 for xiaofanhu [OFFICIAL-SNAPSHOT]**; neither
snapshot figure is freshly verified in this review.

Attribution caveat first: **no inspected organizer receipt links 0.2778 to
these bytes**; GEMSDOE32's own site says no organizer score exists there for
the file. The reported ordering (0.1922 < 0.2477 < 0.2600 < 0.2708 < 0.2778)
is compatible with a metric-aware thinning/pruning explanation, but neither
that ordering nor this mechanism authenticates the score or establishes
causality.

## What the archived raster-forensics reports

**[RASTER-FORENSICS, recorded from a hash-pinned owner mirror/sibling audit]** The reported lineage is one surface (H19-5,
"powerlaw-budget multiline-corroborated", 121,131 px solid) progressively
*thinned*, never re-detected:

The pixel counts are archived raster-forensics; every score in the lineage below is **OWNER-REPORT**, not a verified file-specific organizer receipt.

| step | pixels | owner-reported score | operation |
|---|---:|---:|---|
| H19-5 solid | 121,131 | 0.1922 | — |
| dotted d1.5 | 60,069 | 0.2477 | min-distance dot thinning |
| dotted d2.8 | 44,090 | 0.2600 | wider dot spacing (≈ kernel radius) |
| H27-4 solo | 40,199 | 0.2708 | delete dots 1 px from catalogue |
| **H33-2-B2** | **37,654** | **0.2778** | delete dots ≤2 px (≤200 m) from catalogue |

The recorded B2 raster SHA-256 is `c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9`; the sibling audit reports a single-band float32 raster with 37,654 binary dots and 0 on the catalogue. The TIFF is not present in this checkout, so those raster facts were not independently reread here. They do not authenticate any score.

## Why thinning + flank pruning could raise this metric

**[OFFICIAL-SNAPSHOT]** DTI(α=0.2, β=0.8) with triangular kernel k(d)=max(1−d/300 m, 0):
TP_w = Σ_g max_x p(x)k(d), FP_w = Σ_x p(x)[1−max_g k], FN_w = |G|−TP_w
(DrivenData page 967; FN identity verified against a brute-force transcription
in `scripts/verify_all.py`, max err 7×10⁻¹⁵).

Writing T=TP_w, S=Σp (emitted mass), M=Σ_x p(x)·max_g k (mass on kernel
support), K=|G|:

```
DTI = T / (0.2·(T + S − M) + 0.8·K)         (★)
```

**[CONDITIONAL, exact algebra]** For a new binary dot, let `k` be its kernel weight to the nearest withheld-truth pixel and `dT` its *marginal* increase in weighted true-positive credit. Redundancy can make `dT < k`. With `α+β=1`, the exact improvement condition is `dT > α·DTI·(dT + 1 − k)`. Only for a non-redundant dot (`dT = k`) does this reduce to **`k > α·DTI`**. At DTI≈0.27 this special-case threshold is about `0.054`, or roughly 283 m under the 300 m triangular kernel. It is not a universal rule for redundant placements.

The lineage is **consistent with** three metric-level mechanisms, but it does not isolate them experimentally:

1. **Redundant ribbons can waste prediction mass.** If adjacent pixels cover already-served truth cells, they can add false-positive mass with little marginal `dT`. Spacing dots near the 300 m kernel radius may reduce that redundancy while retaining useful coverage. The recorded 121,131→44,090 pixel change and score increase are compatible with this explanation; they do not prove that spacing alone caused the reported score.
2. **Catalogue-distance pruning may improve precision for the initial-round target.** The problem describes hidden faults outside the existing catalogue, and the shared holdout masks visible mapped faults from scoring. Dots within 200 m of a mapped trace may therefore have lower expected credit for that target, but no file-specific organizer labels establish the truth at those locations. The archived 44,090→37,654 pixel difference is a removal count, not proof that the deleted dots were false positives.
3. **Better geology is still required for a reliable gain.** Pruning can remove predicted mass but cannot move it toward an unmapped fault. Any claim that a fixed-support construction has an exhausted optimum, or that a public leader's score requires a specific increase in hidden-truth credit, depends on unknown labels and a verified file-to-score mapping. Those conditions are not available here; no numerical leaderboard projection is warranted.

## So can 0.2778 be beaten, and how?

Beating 0.2778 is possible in principle, but no current evidence establishes how. A plausible path is a better ranking field with more kernel credit near withheld faults, combined with a metric-aware emitter. Candidate geology might include buried basement contacts, subtle scarps, or structural continuations, but these are hypotheses until tested. The leaderboard values are only a dated snapshot here; they cannot show that a particular detector or emission rule caused a rank difference.

A prior local contact-physics experiment is recorded in `evidence/holdout_exp1.json` and `evidence/holdout_exp3_replication.json`. Across its combined two-draw summary, the candidate measured **0.113714 HOLDOUT-DTI** (95% CI `[0.098774, 0.128091]`; evaluator `gems57-pooled-hide-v2`; 22,914 withheld positives) versus **0.111145 HOLDOUT-DTI** for its standard control (95% CI `[0.095771, 0.126375]`; same evaluator and withheld count). The paired difference was **+0.002569 HOLDOUT-DTI** (95% CI `[-0.003035, +0.007976]`; same evaluator and withheld count), which includes zero. The archived H33-static reference on the first draw was **0.004313 HOLDOUT-DTI** (95% CI `[0.003508, 0.005188]`; evaluator `gems57-pooled-hide-v2`; 11,342 withheld positives). These are local proxy-label measurements, not evidence explaining or reproducing the public H33 attribution. No slot was spent.

Two cautions for future work:

- **Catalogue hide-and-recover is only a proxy.** H33's near-zero first-draw result is not a public-score estimate and does not validate the owner-reported 0.2778; the hidden organizer labels may differ materially from the mapped-catalogue holdout.
- **Two shipped-band tags were reported as wrong** in the prior audit (band 6 = radiometric TC, not tilt; band 18 = ∂g/∂x, not |∇g|; see the cited band-resolution evidence). Treat that as a repository measurement to verify against current bytes before relying on the tags; it is not evidence about the H33 score.

**Bottom line:** if the OWNER-REPORT file-to-score attribution is correct, the thinning and near-catalogue pruning history offers a plausible metric-level explanation, not a proven cause or a demonstrated optimum. A future attempt should test a distinct off-catalogue evidence field on a preregistered, spatially blocked holdout with the pinned evaluator and matched control; any holdout result remains a proxy, not a leaderboard projection. No current candidate in this review has been validated or authorized for a slot.
