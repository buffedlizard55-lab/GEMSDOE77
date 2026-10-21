# H33: what the bytes establish, and what the reported score does not

Updated 2026-10-21. This review supersedes the earlier causal/optimality claims in this document.
**No new HOLDOUT-DTI or ORGANIZER-CONFIRMED score was obtained.** The supplied 0.2778 is an unverified owner attribution, not a measured outcome in this review. The supplied leaderboard targets 0.3195 and 0.4413 are not current authenticated receipts here.

## Evidence ledger

| Claim | Evidence and scope |
|---|---|
| H33 mirror has 37,654 positive pixels, none on the mapped catalogue; nearest catalogue distance is √5 pixels | Fresh hash-pinned raster readback: `evidence/h77l_independent_audit.json`; measurements, not scores. The file SHA-256 is `c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9`. |
| H33 removes 6,436 dots from a 44,090-dot D2.8 mirror without adding dots | Retained `evidence/prior_forensics.json`; historical byte forensics, not freshly rerun lineage analysis. |
| The family site associates H33 with near-catalogue pruning | [GEMSDOE32 site source, pinned commit](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/f62038b54dcb7767d4d997f80b033cdc3d42bef9/docs/index.html), freshly retrieved through GitHub API. It explicitly says “NO ORGANISER SCORE EXISTS for this or any artifact in this repository.” An owner website is not an organizer receipt. |
| The organizer reference supports CPU-only execution | [Official reference README, pinned commit](https://github.com/drivendataorg/gems-prize-reference-solution/blob/aebe92f7c8a990f0e3443451b7a825d9afd6336b/README.md), freshly retrieved. GPU accelerates training; it is not universally required. |
| Local DTI uses maximum triangular-kernel credit, not additive per-dot truth credit | Pinned `vendor/gems57/metric.py` and regression tests. This is the local implementation; the [organizer problem page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) was not freshly accessible through this sandbox's allowed hosts. |

Mirror provenance: `registry/witness_sources.json`. A pinned owner mirror establishes reproducibility of those bytes, not official-origin authenticity or their leaderboard score.

## Correct marginal-emission algebra

This is a **mathematical derivation**, not a HOLDOUT-DTI result, score projection, or organizer receipt.
Let T be total maximum credit summed over truth cells, S the predicted mass, M the predicted mass weighted by proximity to truth, and K the number of truth cells. With α=0.2 and β=0.8:

```
D = α(T + S − M) + βK
s = T / D
```

For a new binary dot at an empty prediction pixel, let k be its nearest-truth kernel weight and ΔT the increase in *total* maximum truth coverage. Then ΔS=1, ΔM=k, and:

```
ΔD = α(ΔT + 1 − k)
new s > old s  iff  ΔT > α s (ΔT + 1 − k)
```

The simplified threshold `k > αs` holds **only when ΔT=k**. It is not a general inclusion rule. A dot may cover multiple truth cells, making ΔT greater than k; an overlapping dot may add no truth credit at all. Consequently the denominator does **not** always increase by α. Tests in `tests/test_marginal_explanation.py` check both counterexamples and the general identity against the unchanged shared implementation.

The pinned template's docstring also says `ΔT <= k`, which is not generally true. This is a documentation defect, not evidence that its `marginal_accept` implementation is incorrect. The pinned code is preserved; the upstream report is recorded in the session review. No private evaluator fork is introduced.

## Plausible explanation — not causal proof

Thinning a thick prediction ribbon can reduce false-positive mass while preserving much of the max-kernel truth coverage. Pruning near-catalogue dots can help if they mostly predict structures excluded from the hidden target. Both mechanisms are compatible with the retained lineage, but neither is guaranteed: thinning may remove uniquely useful coverage, and a genuinely unmapped fault can lie near a mapped fault. Pixel-exact catalogue masking does not imply a blanket exclusion of its surrounding halo.

The owner's sequence of scores is observational and lacks authenticated artifact-to-score receipts. It therefore cannot prove that pruning caused the reported improvement, that H33 is optimal, that further thinning cannot help, or that a particular detector would beat it. The previous document's fixed-denominator, “emission optimum,” required-credit percentage and leaderboard-causation claims are withdrawn.

## Can a new method beat it?

Possibly, but **not established by this review**. The defensible next test is a preregistered, catalogue-independent geological signal with single-feature canaries and a matched buffered whole-segment hide-and-recover evaluation. Rebuild all catalogue-derived features using visible segments only. Compare with the incumbent on identical folds, withheld positives, masks, dot budget and evaluator version; publish pooled DTI and a segment/block-resampled confidence interval. An existing final H33 raster is not a leakage-safe holdout baseline if it was constructed with the entire catalogue.

A catalogue holdout measures recovery of withheld mapped faults, not geothermal vents directly, and may not represent truly unmapped faults. Better holdout performance is necessary under the user's promotion rule, but not sufficient evidence of a leaderboard gain or productive geothermal resources.

## Why no new TIFF is generated in this review

A freshly restored registry witness has positive support across all 5,167,373 footprint cells. Every nonempty allowed prediction set is therefore contained in its support and its three-pixel dilation. Under the currently binding literal rule, forward overlap is exactly 1, exceeding 0.70. This is a deterministic rejection certificate, not geological similarity evidence. No low rank correlation or new filename can undo this rejection. Changing the shared protocol requires its owner's decision; this review does not exempt dense rasters privately.

Existing files remain downloadable **for inspection only**. There is no new candidate, no new holdout score, no selector approval and no competition slot use.

## Parallel-session reconciliation

The concurrent readiness review and [904a24d3 review](../docs/review-904a24d3.md) independently found the same marginal-credit error. Their additional tests (`tests/test_h33_reasoning.py`, `tests/test_marginal_dti_review.py`) and candidate-independent [preflight certificate](../evidence/registry_preflight.json) are preserved. Neither review authorizes a candidate.

## H83 addendum (2026-10-21, branch arena/e8283608-gemsdoe77)

**Byte facts (re-verified from the files; no organizer receipt).**
- The D2.8 raster (`data/raw/scored/gems24-h25-1-dotted-h19-5-d2-8-…-nan.tif`) has 44,090 dots.
- H33 (`h33-2-b2`) has 37,654 dots and is an **exact subset** of D2.8.
- All 6,436 removed dots lie 1.0–2.0 px from the catalogue. The closest kept dot is √5 px away.
- Distance from the H33 dots to the catalogue, by percentile: p10 4, p25 8.1, p50 19.6, p75 42.4, p90 68.0 px.
- The owner reports D2.8 = 0.2600 and H33 = 0.2778.

**Conditional inference.** This holds *only if* the 6,436 removed dots earned zero credit. Under that assumption, α = 0.2 / β = 0.8 algebra over the two reported scores gives a hidden-truth size K ≈ 13,200–14,100 px, about 22–23 % of the 60,988-px catalogue. H33 would then cover ≈ 37–40 % of the truth. The derivation is in `research/h83_netsib_hypotheses.md` §0.

If the removed dots did earn credit, K is not identified and this inference fails. If the inference holds, the hidden target looks like a roughly 20 % withheld subset in the catalogue's own mapping style. That matches the repository holdout's `hide_frac = 0.20` and motivated the H83 lane: predict *where a missing strand of a mapped array goes*.

**Result.** H83-NETSIB reached HOLDOUT-DTI 0.1538 (s20, 95 % CI 0.1357–0.1726, 11,321 withheld) and 0.1637 (s30, CI 0.1447–0.1847, 11,342 withheld), evaluator `gems57-pooled-hide-v2`. The incumbent is 0.152718 and the fold-fair H33 comparator 0.096086, both on the same evaluator. H83 beats the incumbent by point estimate only. The final-dot duplicate rule fired, so the verdict is negative and the file is NOT OK TO SUBMIT (`evidence/h83netsib_run_card.json`). None of these numbers is a leaderboard projection, and whether H83 would beat 0.2778 on the leaderboard is unknown.
