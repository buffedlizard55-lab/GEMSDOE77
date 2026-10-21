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
