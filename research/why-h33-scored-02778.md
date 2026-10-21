# H33: measured sparsity, plausible mechanism, no verified causal score attribution

## Evidence boundary

The supplied **0.2778 is OWNER-REPORT**, not ORGANIZER-CONFIRMED: no submission-page
receipt inspected here ties that score to the named TIFF's SHA-256. Conflicting
leaderboard targets in the brief are not a basis for a performance guarantee.
An organizer score must be accompanied by its submission receipt; a local
HOLDOUT-DTI must include evaluator version, withheld-positive count and confidence
interval. This analysis reports neither a new holdout nor a new organizer score.

Pinned H33 mirror: [source](https://github.com/buffedlizard55-lab/60GEMSDOE/blob/b99591eede63c6d45c8a896255b8cd34824fa00b/registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif).
Hash: `c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9`.
The repository's independent readback records 37,654 positive pixels
(an artifact measurement, not a performance score). It is a sparse binary field,
not evidence that a particular geological detector is superior.

## Correct marginal DTI argument

Source for manual review: [organizer problem and metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/).
Local implementation: `vendor/gems57/metric.py`, pinned by
`registry/template_pin.json`. The organizer page was not freshly reachable in
this review; the equations below are checked against the pinned implementation,
not a newly authenticated portal specification.

Let T be summed maximum kernel credit over truth pixels, S total prediction mass,
M summed nearest-truth kernel credit weighted by prediction mass, and K the
number of scored truth pixels. With alpha=0.2 and beta=0.8:

```
D = alpha*(T + S - M) + beta*K
DTI = T/D
```

Add a binary dot at a previously empty eligible pixel. Let q be its maximum
kernel weight to any truth pixel, and deltaT the **sum** of newly gained truth
credits after taking the maximum against existing predictions. Then:

```
deltaS = 1
deltaM = q
deltaD = alpha*(deltaT + 1 - q)
DTI increases iff deltaT > alpha*DTI*(deltaT + 1 - q)
```

This comparison assumes positive denominators and fixed truth/evaluation domain.
A dot can improve several truth pixels, so **deltaT can exceed q**; it can also
be zero when its coverage is redundant. Only in the special case deltaT=q does
the condition reduce to q>alpha*DTI and deltaD=alpha. The earlier report treated
that special case as general and inferred a universal distance threshold. That
inference was incorrect. `tests/test_marginal_dti_review.py` verifies clustered,
isolated, redundant and randomized cases against the unchanged shared evaluator.
The shared module's prose also incorrectly states deltaT<=q; its numerical
implementation does not impose that bound. This is a documented upstream prose
issue, not grounds for a private metric fork.

## What sparsity and catalogue pruning might explain

- Removing redundant dots can reduce false-positive mass with little loss of
  truth credit. Whether thinning helps depends on the actual hidden geometry;
  a spacing constant alone does not guarantee improvement.
- Pixel-exact catalogue exclusion is not the same as removing a halo around
  catalogue traces. An off-catalogue dot near a mapped fault can still cover a
  different, unmapped fault. Therefore near-catalogue predictions are **not**
  guaranteed false positives or guaranteed fixed-cost penalties.
- The historical lineage in `evidence/prior_forensics.json` supports an
  emission/pruning explanation worth testing. The owner-reported ordering does
  not establish causality: hidden truth, receipts and controlled ablations are
  missing. Do not infer that emission optimization is exhausted, that a fixed
  percentage increase in truth credit is required, or that another team's
  model architecture can be identified from its score.

## Can a new strategy beat it?

Possibly, but no measured improvement or likelihood is established here. A
credible test needs a genuinely assigned lane, single-feature leakage canaries,
matched buffered whole-segment hide-and-recover, a comparable incumbent, and
complete surface/final-dot registry checks. Catalogue recovery is only a proxy
for unmapped faults; neither it nor this competition directly validates a
geothermal vent discovery.

Under the current literal overlap protocol a pinned dense registry witness
covers every eligible pixel. Any nonempty eligible dot set therefore overlaps
that witness completely and fails the stop threshold. This blocks a new
protocol-compliant submission regardless of geological hypothesis. Do not
manufacture a new filename or relax the shared gate to claim success.

See `docs/release-review.md` for the conditional ranked science backlog and
`docs/review-904a24d3.md` for this review's findings and next steps.
