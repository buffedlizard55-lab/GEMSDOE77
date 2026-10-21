# Start here at every session

Read `README.md` from the top (current status first, then the preserved standing prompt), read all of `PROMPT.md`, and then read this file, `docs/irregularities.md`, `registry/preregistration.json`, `registry/template_pin.json`, `evidence/registry_preflight.json`, the latest review linked in README, and the relevant run cards. Historical banners and branch-specific reviews are archive records; they do not override the current Arena message or the d40 review.

## Current branch state — 2026-10-21

- Active branch: `arena/d40f78d1-gemsdoe77`, based on `fa440921424d71765a8297f6610078d14a204428`.
- No method paragraph was assigned in the active request. **Do not invent a lane, infer one from old preregistration text, or revive a closed lane.**
- The fresh `scripts/preflight_registry.py` result is `STOP_UNIVERSAL_DUPLICATE` (expected exit 2). A hash-pinned witness is positive at all allowed pixels, so every nonempty allowed candidate has a forward 3-px overlap lower bound of 1.0, above the strict 0.70 stop limit. This is a rejection proof from one witness, not complete-registry uniqueness clearance. Do not bypass it, weaken the threshold, exempt the dense raster privately, or claim uniqueness.
- The local feature stack (`data/raw/training_features.tif`, `data/processed/stack.npy`, `.cache/gems_data/training_features.tif`) is absent. Do not treat owner-mirror receipts as organizer provenance. No new feature, experiment, candidate raster, holdout score, or TIFF was generated in this review.
- Current release authorization is `NOT OK TO SUBMIT` with zero authorized files. Existing listed artifacts can be downloaded for inspection only; format validity or byte novelty is not submission approval. No weekly slot is selected or authorized.
- Latest report: `docs/review-2026-10-21-arena-d40f78d1.md`; run card: `evidence/review_d40f78d1_card.json`; source receipt: `evidence/review_sources_d40f78d1.json`.

## Scientific and scoring protocol

1. Stay within one explicitly assigned method lane. If none is assigned, stop before proposing it for implementation. Check rank correlation on the candidate surface before placement and dot overlap on the final output. A rank correlation >0.90 or >70% of dots within 3 px of any registry raster is a stop/duplicate.
2. Reuse the pinned `vendor/gems57`, cached features, `evaluate_holdout.py`, and `submission_writer.py`; do not fork or alter the shared evaluator locally. Use whole-segment buffered hide-and-recover, derive catalogue features only from visible faults, mask visible faults pixel-exactly, and score pooled DTI (α=0.2, β=0.8, 300 m triangular kernel). Run each feature alone as a leakage canary; AUC >0.90 is leakage until explained.
3. Label every score as **HOLDOUT-DTI** with evaluator version, withheld-positive count, and 95% CI, or **ORGANIZER-CONFIRMED** with the submission-page receipt. Projections, owner reports, file-format measurements, registry measurements, and test results are not scores. Never attribute a leaderboard value to a TIFF without a receipt tying the exact file/hash to the submission.
4. Before any experiment, rank 3–5 distinct, not-yet-tested hypotheses; name the layers, physical signature, reason for an uncatalogued fault, difference from prior work, and a named non-fault mimic. Rank qualitatively by expected information/DTI value per cost without inventing numerical gains. For external data, identify the free official source and verify AOI availability before calling it viable.
5. If an experiment is authorized, end with one JSON run card containing hypothesis, mechanism, mimic, holdout DTI/CI and evaluator/positive count, registry comparison, TIFF SHA-256, validator output, submission name/note (≤140 characters), and promote/negative verdict. No run card may imply a slot decision. Negative results are valid deliverables.
6. Maximum 3 experiments or 2 hours. Do not select a submission or spend a weekly slot; promotion is a separate selector decision. Never copy a prior submission as new work. Do not upload without exact-file authorization.

## Evidence and release discipline

- The H33 `0.2778` claim is **OWNER-REPORT**, not organizer-confirmed. The pinned GEMSDOE32 owner page states that no organizer score exists for its artifacts; see the current source receipt. Current leaderboard figures `0.3195` and `0.4413` are conflicting and were not freshly verified here.
- Sandbox egress is restricted to GitHub and package registries. Links to DrivenData, USGS, GDR, and the rules PDF are manual-review links unless a current receipt says otherwise. Never claim an official page was fetched when it was not.
- Preserve all existing TIFFs and negative evidence. The site must make inspection-download status distinct from submit authorization. A clean format audit never means OK TO SUBMIT.
- Run three passes: implementation, independent review/correction, end-to-end recheck. Use the repository build, audit, release-gate, verifier, and full test suite. Update the README and current review together; `scripts/build_home.py` is the source for `review-latest.html` and the entry-page status.
- For Git/GitHub work stay on `arena/d40f78d1-gemsdoe77`. Create a PR from this branch and merge only if feasible and checks pass; never switch branches or push another branch.
