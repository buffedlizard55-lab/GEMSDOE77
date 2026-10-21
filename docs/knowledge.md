# Scientific interpretation and evidence boundaries

## What explains the reported H33 result?

The supplied H33 result is **an unverified user claim**, not ORGANIZER-CONFIRMED evidence. No authenticated submission-page receipt or current leaderboard was accessed. The prompt also supplies inconsistent current-leader claims. We cannot establish a current record or claim that this pilot beats it.

We can establish what the rasters actually changed. The independent byte/readback audit in [prior_forensics.json](evidence/prior_forensics.json) uses the pinned template registry, not a reconstruction from filenames:

- D2.8 has 44,090 positive pixels; H27 has 40,199; H33 has 37,654. These are **raster measurements, not scores**.
- H33 is **exactly D2.8 filtered to catalogue distance greater than 2 pixels**. No points were added. H27 removes the 1-pixel-distance ring; H33 additionally removes the square-grid diagonal and 2-pixel ring.
- Thus the observed innovation in this comparison is catalogue-adjacent pruning, not evidence of a newly learned geothermal process. The H33 filename's physics-sounding label is not evidence of causal physics.

The metric explains a *possible* benefit, not a proven causal explanation of the hidden leaderboard result. Let T be weighted truth coverage, F weighted false positives, and K the number of hidden truth pixels. With the shared alpha/beta values, DTI = T / (0.2T + 0.2F + 0.8K). Removing a prediction can improve DTI if it removes appreciable false-positive weight while sacrificing little marginal truth coverage. Predictions near a mapped trace may be redundant or lie in already mapped terrain rather than the unmapped target class. That is plausible under the organizer masking description transcribed by the template, but we could not independently re-read the current portal. It is not a theorem that a 2-pixel buffer is optimal, nor that the reported gain must generalize.

Sparse placement can avoid paying repeated FP cost for redundant nearby predictions while retaining triangular-kernel coverage. However, 300 m is the kernel **radius**, not a guaranteed spatial accuracy or a positive-distance plateau: the kernel reaches zero at that radius. Coarse parameter sweeps against a public leaderboard can also overfit. There is no defensible projection of a better public score from these observations alone.

## Can we do better?

In principle yes; this run supplies **no evidence of doing so**. Catalogue-hidden recovery and hidden organizer mapping differ in sampling, mapping effort and target prevalence. A fixed catalogue-conditioned H33 file cannot serve as a fair hide-and-recover control: masking all original catalogue cells before withholding would erase the positives being tested. We do not score that file on catalogue labels or present it as an incumbent holdout best.

A future selector needs a fold-rebuilt incumbent, using visible faults only, evaluated with the same domain, mask, kernel, pooled terms and block bootstrap as each candidate. A paired improvement interval should exclude zero before spending a scarce slot. The present ablation improvement interval includes zero.

## What H77 tests — and what it does not

The exploratory idea is to suppress shared K/Th rank-intensity variation and test the **shape** of the U-specific residual instead of absolute radiometric brightness. A fracture might guide fluids or sediment redistribution that leaves a narrow contrast even without a mapped scarp. This is a **hypothesis**, not an official-source-established interpretation of these particular pixels.

The cached channels are independently percentile-quantized. `(U - (K+Th)/2)` therefore has no concentration units; it is neither a U/K ratio nor demonstrated uranium enrichment. The Hessian's strongest absolute eigenvalue times anisotropy emphasizes narrow curvature responses, including both positive and negative residual polarity. Gaussian sigma and boundary erosion were fixed before evaluating labels. No catalogue distance, fault orientation, fault density, well-to-fault distance or existing submission is a feature.

Groundwater redox fronts and fluvial heavy-mineral sorting are named alternative explanations; lithologic contacts, variable regolith cover, flight-line leveling and quantization can also generate non-fault edges. The [mirrored GeoDAWN contractor report](geodawn-report-mirror.txt) describes acquisition, altitude corrections, leveling and radiometric channels. This copy is an owner-cached text extraction, **not newly authenticated official literature**. It supports an acquisition-artifact caution, not a claim that H77 detects geothermal alteration. Direct literature verification and field interpretation remain incomplete because the official source hosts are inaccessible here.

## Validation contract

- Reuse the pinned template's `gems57.spatial.folds`, `evaluate_holdout.py`, metric and writer without changes.
- Four fixed, label-blind quadrants. Entire intersecting 8-connected catalogue components are hidden, including their out-of-quadrant tails. An 80-pixel training buffer is recorded. This label-free pilot does not fit on the training domain.
- Features and normalization depend only on radiometric covariates. The holdout uses each fold's **visible** faults for exact masking; final research export masks the full known catalogue.
- Erode radiometric support by 10 pixels. The holdout is conditional on that domain; unsupported edges are not silently scored as well-predicted.
- Test each input and derived feature alone for AUC and HOLDOUT-DTI. AUC uses catalogue-zero proxies, not verified fault absence, and is a diagnostic, **not a competition score**.
- Pool additive TP/FP/FN terms, then calculate DTI. Resample aligned physical 20 km blocks jointly across all arms for the paired conditional 95% intervals. This is not an organizer-score interval and does not include all catalogue/selection uncertainty.
- The final TIFF is not scored against the fully masked catalogue. HOLDOUT-DTI characterizes the hide/recover procedure, not a misleading readback score on the known faults.

## Literal uniqueness blocker

The re-read 17GEMSDOE continuous registry raster is positive on **every footprint cell**. Under the unchanged template's literal definition of support (`finite and >0`), any nonempty allowed candidate has forward 3-pixel overlap 1.0 with that raster. The [saturation certificate](evidence/registry_saturation.json) proves this for the full allowed footprint, not a random sample. One valid witness suffices to reject; a partial inventory can never certify acceptance.

This is a protocol-design incompatibility, not evidence that H77 and the old model have the same geological ranking. Their rank correlation is low. We do not invent a density exemption, switch to mutual overlap, exclude the witness, or generate dots after the stop. A selector/owner must resolve the representation policy in the shared template/protocol before any nonempty entry can pass this literal rule.

## Prize target versus geothermal science

The [official reference notebook](https://github.com/drivendataorg/gems-prize-reference-solution/blob/aebe92f7c8a990f0e3443451b7a825d9afd6336b/unet-mc-cv-reference-solution.ipynb) explicitly describes fault detection with U-Net and Monte Carlo cross-validation, and uses Tversky-loss weights. It does not establish vent locations, hydraulic connectivity, subsurface heat or economic resources. A fault prediction should never be advertised as a discovered geothermal vent. This repository preserves a negative structural-detection pilot; it is not a resource assessment.
