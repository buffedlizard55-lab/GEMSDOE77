# H77E-Knick audit — corrected status and withdrawn claims

> **Correction notice (2026-10-21):** This file replaces an earlier analysis that overstated the H33 mechanism, treated a participant leaderboard row as if it identified a TIFF, called a diagnostic overlap reading a uniqueness clearance, and presented H77E as a winning/approved candidate. Those conclusions are withdrawn. Use this note and the linked run cards for current interpretation. H77E-Knick is historical research, **NOT OK TO SUBMIT**, and has no selector approval.

## H33 reported 0.2778: evidence boundary

- The archived 2026-10-21 leaderboard snapshot records a row for participant `extradr19` at `0.2778`. That is an organizer leaderboard value for that row only; the row does not identify the submitted TIFF or its hash.
- The family attribution of `0.2778` to GEMSDOE32's `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` bytes remains **OWNER-REPORT**. No inspected organizer receipt links those bytes to the row, and the sibling project page says it has no receipt for that file. The TIFF is absent from this checkout; the archived owner-mirror hash and pixel counts are not an organizer score receipt.
- The archived construction record describes a thinned H19-5-derived raster and successive catalogue-distance pruning. This sequence is compatible with a metric-aware reduction of redundant or low-credit prediction mass, but it is not a controlled experiment. It does not establish that thinning or pruning caused the reported score, that the removed dots were false positives, or that an optimum was reached.
- For `DTI = T / (α·(T + S − M) + β·K)` with `α=0.2`, `β=0.8`, let `k` be a candidate dot's nearest-truth kernel weight and `dT` its marginal increase in weighted true-positive credit. The exact improvement condition is `dT > α·DTI·(dT + 1 − k)`. Only for a nonredundant dot (`dT=k`) does it reduce to `k > α·DTI`.
- A fixed archived H33 raster measured **0.004313 HOLDOUT-DTI** on one local catalogue hide-and-recover draw (evaluator `gems57-pooled-hide-v2`; 11,342 withheld positives; 95% CI `[0.003508, 0.005188]`; `evidence/holdout_exp1.json`). This is a local proxy measurement, not a live-score estimate or validation of the file-to-score attribution.

See the [corrected H33 analysis](../research/why-h33-scored-02778.md), [forensic record](../evidence/prior_forensics.json), and [score-provenance irregularity](../docs/irregularities.md#ir-l-02). Official pages are linked there for manual checking; they were not freshly fetched for this review.

## H77E-Knick: what the frozen run card supports

The relevant record is [`evidence/h77e/run_card.json`](../evidence/h77e/run_card.json); the inspection-only page is [`docs/h77e-knick.html`](../docs/h77e-knick.html). The experiment budget was used by the historical lane; do not restart it under this review.

| Frozen arm | Result and evidence class | Registry interpretation | Disposition |
|---|---|---|---|
| Unconstrained pure-physics allocation | **0.155623 HOLDOUT-DTI** (evaluator `gems57-pooled-hide-v2`; 11,321 withheld positives; 95% CI `[0.138318, 0.171637]`) on Seed 20; **0.154666 HOLDOUT-DTI** (same evaluator; 11,342 withheld positives; 95% CI `[0.137860, 0.170901]`) on Seed 30. | Informative-subset duplicate firings: 59/816 on Seed 20; it is not the eligible deliverable. | Research diagnostic only; not promoted. |
| Portfolio-constrained primary | **0.126664 HOLDOUT-DTI** (same evaluator; 11,321 withheld positives; 95% CI `[0.111196, 0.141749]`) on Seed 20; **0.124727 HOLDOUT-DTI** (same evaluator; 11,342 withheld positives; 95% CI `[0.109650, 0.138262]`) on Seed 30. | 0/816 firings in the narrower informative subset, but literal Policy I17 still records 45 firings on 861 rasters, driven by space-filling archive rasters. Therefore `protocol_unique=false`; do not call this a uniqueness PASS. | Negative, inspection-only; **NOT OK TO SUBMIT**; selector not approved; 0 weekly slots used. |

The unrestricted arm does not clear its overlap gate. The constrained arm's local format receipt does not override the literal registry blocker, the negative verdict, or selector authorization. The blocked-lane result is the deliverable; a newly named copy or an alternate reading of the gate is not clearance.

## Review links and next action

- [README current release banner](../README.md) — zero authorized files and H33 evidence label.
- [Submission authorization receipt](../docs/data/submission-authorization.json) — authoritative repository-level status.
- [H77E-Knick page](../docs/h77e-knick.html) — research-only artifact details.
- [Current-session review](../docs/review-2026-10-21-current-session.md) — hypotheses, source/data limits, and three-pass review.
