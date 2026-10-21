# Why h33-2-b2 got the family's highest reported score — and what beating it takes

*PhD-level analysis, 2026-10-21. Evidence classes: **[OFFICIAL]** organizer/USGS
source · **[MEASURED]** computed from hash-pinned bytes (this or sibling
checkouts, file cited) · **[OWNER-REPORT]** family ledger, no organizer receipt ·
**[CONDITIONAL]** algebra under stated assumptions.*

## The question

The brief asks why `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` (GEMSDOE32)
scored **0.2778 [OWNER-REPORT]** — the family's best — and whether a unique
submission can beat it (public leader **0.4413 [OFFICIAL]**, xiaofanhu,
leaderboard fetched 2026-10-21).

Attribution caveat first: **no organizer receipt links 0.2778 to these bytes**
in any inspected repository; GEMSDOE32's own site states no organizer score
exists in that repo for the file. What follows explains the *mechanism* that
makes the reported ordering (0.1922 < 0.2477 < 0.2600 < 0.2708 < 0.2778)
expected under the official metric; it does not authenticate the numbers.

## What the file measurably is

**[MEASURED, sibling audits + local bytes]** The lineage is one surface (H19-5,
"powerlaw-budget multiline-corroborated", 121,131 px solid) progressively
*thinned*, never re-detected:

| step | pixels | reported score | operation |
|---|---:|---:|---|
| H19-5 solid | 121,131 | 0.1922 | — |
| dotted d1.5 | 60,069 | 0.2477 | min-distance dot thinning |
| dotted d2.8 | 44,090 | 0.2600 | wider dot spacing (≈ kernel radius) |
| H27-4 solo | 40,199 | 0.2708 | delete dots 1 px from catalogue |
| **H33-2-B2** | **37,654** | **0.2778** | delete dots ≤2 px (≤200 m) from catalogue |

Local copy of the B2 raster (`data/registry_rasters/…-zeros.tif` from the
GEMSDOE32 clone): single-band float32, 37,654 binary dots, 0 on catalogue
(verified indirectly via this session's uniqueness rows: its dot count and
decoded digest match the archive).

## Why thinning + flank pruning raises this metric

**[OFFICIAL]** DTI(α=0.2, β=0.8) with triangular kernel k(d)=max(1−d/300 m, 0):
TP_w = Σ_g max_x p(x)k(d), FP_w = Σ_x p(x)[1−max_g k], FN_w = |G|−TP_w
(DrivenData page 967; FN identity verified against a brute-force transcription
in `scripts/verify_all.py`, max err 7×10⁻¹⁵).

Writing T=TP_w, S=Σp (emitted mass), M=Σ_x p(x)·max_g k (mass on kernel
support), K=|G|:

```
DTI = T / (0.2·(T + S − M) + 0.8·K)         (★)
```

**[CONDITIONAL, exact algebra]** Adding one unit of mass with realised kernel
weight k raises DTI iff **k > 0.2·DTI** — because the denominator always rises
by exactly α=0.2 (ΔT=k, ΔFP=1−k) regardless of placement. At DTI≈0.27 the bar
is k>0.054, i.e. a dot must lie within ~283 m of a hidden-truth pixel (and not
duplicate better cover) to pay for itself.

Three consequences, each matching the lineage:

1. **Solid ribbons waste mass.** Along a 1-px truth trace, one dot already
   gives max-credit ≈1 to itself and triangular credit to neighbours within
   3 px; adjacent solid pixels add denominator (0.2 each) with ΔT≈0. Spacing
   dots at ≈2.8 px keeps kernel coverage while cutting S by 2.75×
   (121,131→44,090): the observed +0.068.
2. **Near-catalogue dots are near-pure false positives live.** The hidden
   truth is defined as faults *not* in the catalogue, and known faults are
   masked from scoring (problem page: test faults are "not contained within
   the current public USGS database"; sibling metric docs cite organizer
   forum thread 11516 for pixel masking). A dot ≤200 m from a mapped trace
   most plausibly covers that mapped structure's halo/shoulder — credit ≈0
   against *new* truth, cost 0.2 guaranteed. Deleting 6,436 such dots
   (44,090→37,654 across the two prunes) raises mean realised credit per dot:
   the observed +0.018 over d2.8.
3. **The gains are emission-side and exhaustible.** Each prune can only remove
   tax; it cannot move a dot onto a real unmapped fault. Solve (★) at fixed
   S=37,654 for the leader's 0.4413 vs 0.2778 (same hidden K): required T —
   i.e. mean realised credit per emitted pixel — must rise by roughly **60%**
   (exact figure depends on the unknown K; using the family's lattice-calibrated
   K≈7.9–12.7k px gives +55–65%). No re-thinning of the H19-5 surface can
   supply that; the surface's top-ranked pixels are simply not close enough
   to hidden faults.

## So can 0.2778 be beaten, and how?

Yes in principle — only with a **better field** (higher credit density at the
top of the ranking), because the emission lever is spent. What "better" means
geologically: the hidden set was hand-mapped by experts from the same
GeoDAWN/DEM/lidar evidence, so winning pixels are places where *expert mapping
criteria* fire off-catalogue: buried basement contacts in potential fields,
fronts/scarps too subtle for the legacy catalogue, structurally required
continuations. The public leader's margin (0.4413 vs a thick pack at
0.32–0.39, [OFFICIAL] snapshot 2026-10-21) is consistent with a materially
better detector (e.g. learned segmentation over multi-physics + high-res DEM
context with metric-aware emission), not with a cleverer thinning constant.

This session tested one new-field family (derived contact physics +
front-base geomorphology). **[MEASURED]** It reached HOLDOUT-DTI 0.1090 vs
standard-block 0.1043 on catalogue hide-and-recover, but the paired advantage
is statistically indistinguishable from zero across two draws (+0.0026
[−0.0030, +0.0080], 22,914 withheld px) — so it **cannot honestly be claimed**
as the better field, and no slot was spent (`evidence/holdout_exp1.json`,
`evidence/holdout_exp3_replication.json`). Two warnings for the next attempt:

- **Catalogue holdout rewards catalogue-style faults.** The same H33 dots that
  (reportedly) lead live score 0.0043 HOLDOUT-DTI here, by construction. Any
  instrument must be calibrated against live orderings before its deltas are
  trusted (GEMSDOE70's piedmont calibration is the family's current best
  attempt — adopt and preregister, don't improvise post-hoc).
- **Two shipped-band tags are wrong** (band 6 = radiometric TC, not tilt;
  band 18 = ∂g/∂x, not |∇g| — measured, ρ≈1.00/0.95 vs the true quantities).
  Fields built by trusting tags have been training on misnamed physics;
  correcting the stack is free accuracy for every future lane.

**Bottom line:** 0.2778 is the exhaust of an emission optimum on a 0.19-class
field. Beating it — and approaching 0.4413 — requires raising top-of-ranking
credit density with genuinely better off-catalogue evidence (lidar-grade
geomorphology at scale, potential-field contacts derived correctly, sub-km
seismicity, or learned multi-context segmentation), each proven on a frozen
holdout before a slot is spent. This session's candidate did not prove that;
its verified negative, band corrections, and instrumentation are the
contribution.
