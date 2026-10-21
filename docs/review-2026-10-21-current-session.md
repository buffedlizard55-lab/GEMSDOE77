# Current repository review — 2026-10-21

## Disposition

**NOT OK TO SUBMIT; 0 authorized files.** This review generated no TIFF, ran no holdout experiment, and spent no competition slot. No hypothesis was selected or validated. That is a deliberate fail-closed outcome: the raw training predictor TIFF and processed stack are absent, this session has no assigned method lane, the current authorization receipt permits zero files, and the literal registry/Policy I17 blocker remains unresolved. Do not rename, copy, or relabel a prior raster as a new candidate.

- **Download status:** existing reviewed files may be downloaded for inspection where their page says so. No file is authorized for portal upload. Format PASS is not portal acceptance, uniqueness clearance, or selector approval.
- **New-file format validation:** not applicable; no new file was built. No submission filename or ≤140-character note exists for a new candidate. Do not invent either.
- **Current data inventory:** `data/bridge/` contains `existing_faults.tif`, `labels.tif`, and `sample_submission.tif`; `data/raw/training_features.tif` and `data/processed/stack.npy` are absent. The manifest's owner-mirror receipts do not authenticate organizer provenance or licensing.
- **Official-source access:** links are provided for manual review below and in the hypothesis register. No official competition, USGS, DOE GDR, or rules page was fetched in this review; current rules, leaderboard, licensing, and product availability are therefore not re-certified.

## Three-pass review

### Pass 1 — brief, repository, and evidence inventory

- Re-read the standing instructions in `README.md`, `PROMPT.md`, and `AGENTS.md`, then inspected the manifest, authorization receipt, preregistration/run-card records, holdout/evaluator references, dataset inventory, site builders, tests, and relevant H33/H77E reports.
- Confirmed the checked-out branch is `arena/2c12246d-gemsdoe77`, based on `a72af95`; the branch remains the only authorized work branch.
- Confirmed the top-level release gate says **NOT OK TO SUBMIT** and `repository_authorized_file_count` is `0`. H80 is the sole manifest primary but is a negative inspection-only result; its **0.095797 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,319 withheld positives, 95% CI `[0.088183, 0.103248]`) is below the recorded **0.152718 HOLDOUT-DTI** local incumbent on a different holdout draw. Do not compare unlike draws as a paired result.
- Verified local data placement: bridge grid/catalogue/labels exist, but `training_features.tif` and the prepared feature stack do not. No physical predictors are available for a new blocked evaluation.

### Pass 2 — claim, hypothesis, and release-safety audit

- Reclassified the H33 claim. The family file-to-score attribution `0.2778` remains **OWNER-REPORT**. A repository record of the 2026-10-21 public leaderboard has the same value for participant `extradr19` at row #48, but the row does not identify a TIFF or hash; no inspected organizer receipt links the row to the named H33 bytes. The official page was not fetched in this review.
- Preserved the distinction between construction facts and causality. Archived owner-mirror raster forensics describe thinning and near-catalogue pruning, but the sequence is not a controlled experiment; it does not prove those changes caused the reported score, identify hidden-fault locations, show the removed dots were false positives, or establish an optimum. The exact marginal DTI condition is documented in [`research/why-h33-scored-02778.md`](../research/why-h33-scored-02778.md); the simpler `k > α·DTI` criterion is valid only for a nonredundant dot (`dT=k`).
- The archived H33-static reference records **0.004313 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,342 withheld positives, 95% CI `[0.003508, 0.005188]`). This is a local catalogue hide-and-recover proxy, not an estimate of public performance and not evidence for the file-to-score attribution.
- Replaced overconfident H33 language in the current analysis, knowledge note, H77D/H79/contact-physics/H80T pages, and the generated LIA analysis. Marked the older H77E analysis as corrected: its unconstrained arm is not submission-eligible, its informative-subset result does not override 45 literal Policy I17 firings, and its constrained artifact remains negative/inspection-only.
- Registered four ranked, evidence-qualified research hypotheses in [`research/hypotheses-2026-10-21-current-session.md`](../research/hypotheses-2026-10-21-current-session.md). The register identifies prior overlap rather than claiming novelty, records local data absence and official-source verification limits, and names no viable candidate. The provisional top-ranked event-scale seismicity idea is blocked: event-level data and its AOI coverage were not verified. No score estimate is offered.

### Pass 3 — site, tests, and final requirements check

- Rebuilt the generated home/guide (`scripts/build_home.py`), archived LIA pages (`scripts/lia_build_site.py`), H77D pages (`scripts/build_site.py`), and contact-physics page (`scripts/build_site_contactphys.py`) from their repository builders after the copy changes. Corrected the static H79 and H80T H33 sections, revised the older H77G/H80 hypothesis notes to remove unsupported causal or threshold claims, and replaced the stale `docs/leaderboard.html` table with an archival provenance/status page. The site keeps the one-glance download-versus-submit distinction and points readers to the corrected H33 analysis and this review.
- The first generated-site test pass found two real integration issues: the H77E-Knick validator assumed a missing top-level selector field, and its run-card link was not copied to the Pages tree. The builder now checks the frozen `approved_for_weekly_slot`/writer receipt fields and publishes the exact run card at the linked path. A stale LIA builder input also did not match its frozen run card; the LIA site generator was replaced with a fail-closed archive builder using the run-card schema rather than the unrelated stale submission JSON.
- **Tests:** `.venv/bin/python -m pytest -q` — **124 passed in 6.56s** after the final site rebuild. Python compilation and all four site builders completed successfully.
- Re-checked the operative request at the end of `README.md`; its original prompt remains preserved there. `PROMPT.md` contains the standing protocol and points to the current evidence.
- No candidate holdout was run, no TIFF/name/note was fabricated, no slot was spent, and no current result is labeled as a projection or organizer-confirmed score.

## Registered hypotheses and decision

The four hypotheses and per-hypothesis layers, signatures, uncatalogued-fault rationale, prior-work comparison, cost, and data-availability checks are in the linked register. Their rank is **qualitative only**: the expected DTI gain is unknown and no numeric gain/cost projection is defensible with the predictors absent. The drainage and potential-field families have substantial prior overlap; InSAR was proposed in older review material; event-scale seismology differs from coarse seismic-rate bands but its data are unverified. None should be represented as novel without a fresh full-family search and registry comparison.

**Top-candidate validation gate: BLOCKED, not passed.** There is no viable top candidate in this checkout. Before any experiment or slot decision, a separately assigned lane must first restore/authenticate required predictors, check official source availability, pass pre-placement rank-correlation screening, preregister, run feature-alone leakage canaries, and evaluate the candidate with the pinned whole-segment buffered holdout and matched control. After placement, check registry dot overlap; stop and log any duplicate at `|ρ| > 0.90` or more than 70% within 3 px of a registry raster. A holdout cannot override the literal release/selector gate.

## Remaining work and blockers

1. Obtain the missing predictor data through an authorized route and independently verify official provenance/license; the current owner-mirror receipts are not sufficient.
2. Obtain an explicit method-lane assignment and resolve the literal dense-witness policy with the protocol owner. Do not invent a density exemption or use a private fork.
3. Verify current official rules and available submissions/weekly allowance, then repeat full registry comparison and preregister one non-duplicate method.
4. Validate a viable candidate on the prescribed blocked holdout before considering any slot. A negative result is a valid deliverable.
5. Only after independent format validation, uniqueness clearance, positive matched holdout evidence, and separate selector authorization may the site say **OK TO SUBMIT**. Preserve the organizer receipt before applying **ORGANIZER-CONFIRMED** to a file-specific score.
6. The requested PR/merge is not completed by this review note; after tests/builds pass, open and merge a PR from this session branch only. This review itself does not authorize a submission or relax release policy.

## Review artifacts

- [`README.md`](../README.md) — current status and preserved project request.
- [`research/why-h33-scored-02778.md`](../research/why-h33-scored-02778.md) — corrected H33 analysis.
- [`research/hypotheses-2026-10-21-current-session.md`](../research/hypotheses-2026-10-21-current-session.md) — ranked hypotheses and verified local data status.
- [`knowledge/02_h77e_audit_and_breakthrough.md`](../knowledge/02_h77e_audit_and_breakthrough.md) — H77E correction and frozen evidence.
- [`docs/data/submission-authorization.json`](data/submission-authorization.json) — operative repository authorization.
