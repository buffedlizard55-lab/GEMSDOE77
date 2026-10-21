# H33: measured construction, plausible mechanism, unproven score attribution

Reviewed 2026-10-21. **The supplied 0.2778 is OWNER-REPORT for the named
H33 file, not an organizer submission-page receipt linked to its hash.**
Historical public leaderboard snapshots do not authenticate which TIFF earned
a participant's score. This session could access GitHub/package registries,
not the live DrivenData leaderboard. Conflicting target numbers in the brief
are not resolved by choosing the most convenient one.

## What the bytes show

[Retained raster forensics](../evidence/prior_forensics.json) compares the
owner-mirrored H33 file against earlier rasters. These are **RASTER
MEASUREMENTS**, not competition scores:

| Construction | Positive pixels | Change |
|---|---:|---|
| D2.8 | 44,090 | Sparse predecessor |
| H27 | 40,199 | Removes 3,891 predecessor dots; adds none |
| H33 B2 | 37,654 | Removes another 2,545 dots; adds none |

H33 equals D2.8 filtered to retain dots more than two grid pixels from the
mapped catalogue; 6,436 dots are removed in total. H33 SHA-256:
`c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9`.
The current session restored and hash-checked this exact file and independently
rechecked its dot count and grid via `scripts/verify_h77l.py`.
[Mirror provenance](../registry/witness_sources.json) is owner-repository
provenance, **not organizer authentication**.

## Metric mechanism — exact algebra, not a projected score

The pinned [shared metric implementation](../vendor/gems57/metric.py) implements
a triangular distance kernel with a 300 m radius, alpha 0.2 and beta 0.8.
[Organizer problem statement for manual review](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/).
This session verified the implementation against its brute-force property tests;
it did not freshly retrieve that external web page.

For a binary dot field, define:

- T: sum over truth pixels of their best prediction-weighted kernel credit;
- S: number of predicted dots;
- M: sum over predicted dots of their best kernel credit to truth;
- K: number of truth pixels.

Then `DTI = T / [0.2*(T + S - M) + 0.8*K]`.
Adding one dot with own nearest-truth credit `k` and **marginal total truth
credit** `dT` changes the denominator by `0.2*(dT + 1 - k)`.
The exact improvement condition is:

```
dT > 0.2 * current_DTI * (dT + 1 - k)
```

Only in the special case `dT == k` does this reduce to
`k > 0.2*current_DTI`. A dot can cover multiple truth pixels, so `dT` can
exceed `k`; it can also be zero when earlier dots already provide better
coverage. **The previous version's claim that the denominator always rises
by exactly 0.2 regardless of placement was false.** Numerical regression
examples are in `tests/test_h33_reasoning.py`. The pinned runtime metric and
writer were not changed or privately forked.

Sparse placement can reduce redundant prediction mass while retaining useful
kernel coverage. Near-catalogue pruning can remove redundant predictions on
already-mapped structures. These are plausible explanations for a better
reported result, not proof of causality. An unmapped fault can be close to a
mapped fault: near-catalogue dots are **not guaranteed** to be pure false
positives, and deleting them can lose real credit. We do not know the hidden
truth, T, M or K for this file.

## Can a new file beat it?

Possibly, but this repository has not established that a new file will. Better
structural evidence and more efficient placement both remain hypotheses.
The prior claims that thinning is mathematically exhausted, that a specific
percentage gain in hidden credit is required, or that leaderboard separation
proves a particular detector architecture were not supported by the available
evidence and have been removed.

Catalogue hide-and-recover is a proxy for missing-fault discovery, not the
organizer's hidden test set. Moreover, fault mapping is not direct geothermal
vent detection: a cold fault can still count as a fault and a hydrothermal
anomaly can be unrelated to a target trace. A candidate must beat a matched
holdout control with evaluator version, withheld-positive count and CI,
pass per-feature leakage canaries and both registry checks, then receive a
separate selector decision. No local HOLDOUT-DTI should be numerically
substituted for a leaderboard score.

## Current decisive blocker

[Fresh candidate-independent preflight](../evidence/registry_preflight.json)
shows one hash-pinned registry witness is positive at every allowed prediction
pixel. Every nonempty surface/dot set on that domain has forward overlap 1.0,
strictly above the mandatory 0.70 threshold. This is a **REGISTRY MEASUREMENT**,
not a score. Renaming a TIFF, changing the geological hypothesis, or generating
a randomized dot pattern does not resolve it. No new experiment or submission
was made in this review. Protocol/registry support semantics must be resolved
upstream before a new candidate can be cleared; this repository does not invent
an exemption.

## Verified official source and access limitations

The organizer's [reference solution at the checked commit](https://github.com/drivendataorg/gems-prize-reference-solution/tree/aebe92f7c8a990f0e3443451b7a825d9afd6336b)
was retrieved over GitHub in this session. Its README explicitly offers CPU-only
and GPU environments. Thus a GPU is helpful, not an absolute requirement of
that reference solution. DrivenData authentication and an organizer receipt
for the named H33 file are unavailable here; no credentials were requested.
