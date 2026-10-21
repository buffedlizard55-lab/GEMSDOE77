# GEMSDOE77 repository review — 2026-10-21 (Arena audit)

## Disposition

**No new experiment or competition submission was created in this review. No TIFF in the current release manifest is approved to submit.** H79-RH, H79-ORIENT, H80 and the subsequently merged AGO variants are negative or fail closed; the repository-wide selector receipt authorizes zero files. Re-running or changing a stopped lane would violate stored run protocols. Existing GeoTIFFs can be downloaded for local inspection only; none is a submit-ready competition entry. For the H80 release review and current per-file status, see `docs/release-review.md` and the generated `docs/index.html`.

- Branch reviewed: `arena/a5befe2a-gemsdoe77`.
- Repository decision: `NOT OK TO SUBMIT`, 0 authorized files (`docs/data/submission-authorization.json`).
- No organizer upload, submission-page receipt, or competition slot use was performed or inferred.
- The user-supplied family score table is not an organizer receipt. The particular attribution of 0.2778 to the named GEMSDOE32 TIFF remains unverified in the repository's own forensic account.

## Pass 1 — read and map the project

Read the operative README and original request, `PROMPT.md`, `AGENTS.md`, irregularity ledger, preregistrations, current authorization receipt, H79-RH and H79-ORIENT run cards and evidence, site entry pages, shared template pin, implementation/test configuration, and current artifact inventory.

The key operational instructions agree: work within the registered lane; use the pinned shared `gems57` evaluator/writer; check each feature as a leakage canary; test surface uniqueness before final placement; stop on a duplicate; label holdout values with evaluator, withheld-positive count, and interval; and keep submission-slot selection separate. The repository includes other completed negative lanes (including D77 and XSEN); this audit does not re-run or re-rank them. The present review ran no model experiment and generated no new TIFF. While this PR was open, `origin/main` advanced with the completed H79-ORIENT negative lane; that branch update was merged into this working branch to avoid an obsolete PR base and is recorded here without repeating its experiment.

The request's H33 question has a plausible metric-level explanation, not a verified causal answer. Sparse, spatially separated predictions can reduce redundant false-positive mass under a distance-tolerant DTI, and excluding pixels close to already catalogued faults may help if the evaluation emphasizes undiscovered faults. The H33 raster forensics in this repo establish byte/pixel construction facts, not that those bytes received the reported organizer score. A leaderboard row is not a raster-to-receipt link. Therefore neither a causal explanation nor a claim that this repo can exceed 0.2778 or the reported current leader is established.

## Pass 2 — executable verification and artifact review

Environment: installed the pinned `requirements.txt` into ignored `.venv/`; no tracked environment files were added.

- After merging the upstream H80/release-review changes, the first full suite run was **97 passed, 2 failed**. Both failures exposed a real source/generated-site mismatch: `docs/index.html` featured H80 while `scripts/build_home.py` still selected H77L and omitted the explicit, non-causal 0.2778 comparison required by the page test.
- Fixed the source of truth rather than patching generated HTML: added the H80 negative artifact and frozen evidence to `docs/data/lanes.json`, made it the sole featured research-only lane, changed H77L to earlier/not featured, and updated `scripts/build_home.py` to validate H80's exact name, note, SHA-256, format result and HOLDOUT-DTI metadata against its frozen run card and bytes. H80 is featured because it has complete pinned-evaluator evidence; this is not a claim that it is the chronologically newest artifact. The generated landing page exposes H80's direct inspection download, exact holdout label/CI/count, local overlap/incomplete-scan limitation, missing pre-placement rank-correlation record, explicit non-comparison to the owner-reported 0.2778, and NOT OK TO SUBMIT status.
- A later upstream merge added three already-generated AGO variants. This review did not create or independently reproduce them. Their frozen card uses a different quick 4-quadrant evaluator; the exact withheld-positive count and 95% CI are absent. The landing table therefore does not present their raw values as HOLDOUT-DTI scores, and the AGO page now labels the diagnostics unreportable, removes upload instructions and portal-acceptance guarantees, and marks every variant NOT OK TO SUBMIT. The repository release receipt now covers all 17 listed TIFF lanes and still authorizes zero files.
- Final test suite after the source-generated H80 and AGO status fixes: **101 passed** (`.venv/bin/python -m pytest -q`).
- Repository verifier: `.venv/bin/python scripts/verify_all.py` exited successfully.
- The AGO page builder formerly depended on an ephemeral `/tmp/ago_final_info.json`; it now derives its displayed artifact identity/checksum metadata from the frozen `evidence/ago_run_card.json`. I ran it successfully and verified that it contains no upload instructions or unqualified numeric quick-evaluation scores.
- Fail-closed selector: `.venv/bin/python scripts/release_gate.py --verify-fail-closed` returned `NOT OK TO SUBMIT`, `authorized_files: 0`; its published receipt covers all 17 manifest entries and authorizes zero files.
- `.venv/bin/python scripts/build_home.py` completed from the checked-in manifest and frozen evidence. Generated landing/guide updates are intentional; the H80 TIFF itself was not generated or modified.
- Shared local format validator on `docs/downloads/gems77-h79-radhalo-lag3-37654-20261021-zeros.tif`: all 15 checks passed. Single-band float32; 3,730 × 3,292; EPSG:32611; sample transform match; no NaN/Inf; values in [0,1]; no positive values on mapped catalogue. This is local format validation against the tracked bridge grid, not official-data authentication, uniqueness, portal acceptance, or permission to submit.
- The exact file SHA-256 is `c272025ee653187d09489d941613a86bfe4b88110fb455847a2256cf5699014b`. This is a byte identity for the already-existing lane artifact, not evidence that it is unique under the competition protocol.
- H79-RH's frozen run card reports **HOLDOUT-DTI 0.0477326573** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.03958336, 0.05557557]`). Its **HOLDOUT-DTI paired difference** vs same-run random is +0.00030988 (evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, 95% CI `[-0.00836770, 0.00895815]`), which includes zero. These are local holdout measurements, not leaderboard results.
- Feature-alone leakage canaries in the H79-RH receipt have maximum symmetric AUC 0.5084, below the registered 0.90 threshold. This is a diagnostic only; it does not demonstrate predictive usefulness.
- The H79-RH registry scan read 808 of 812 indexed rasters; four were invalid or mismatched. Its surface maximum absolute Spearman value was 0.1203, but final-dot forward overlap was 1.0 against a full-support raster (threshold 0.70). The scan is incomplete, and the recorded literal duplicate is already sufficient to stop; no uniqueness certificate can be claimed.
- After this PR was opened, the updated `origin/main` added the already completed H79-ORIENT lane and its negative candidate. Its frozen receipt reports **0.094507 HOLDOUT-DTI** on seed 30 (`gems57-pooled-hide-v2`, 11,342 withheld positives, 95% CI `[0.076558, 0.112844]`) and **0.089063 HOLDOUT-DTI** on seed 20 (11,321 withheld positives, 95% CI `[0.072384, 0.105355]`). The card records 13 of 812 local rasters checked, a full-support witness, and final-dot forward overlap 1.0 > 0.70. These are distinct lane-fold results, not organizer scores or a matched comparison to the incumbent.
- I independently ran the shared local validator against the H79-ORIENT TIFF: 15/15 format checks passed; SHA-256 `745dbaa427bc1edaf477b3d0b4847b8be8d30d396106a61b1318a8726599be8e`. This validates local file encoding only. It does not repair the incomplete registry or duplicate stop.
- After a further `origin/main` update during PR review, the repository added the completed H80 Riedel wing-crack TIFF. I ran the shared local validator on `docs/downloads/gems77-h80-riedel-wingcrack-tips-20261021-zeros.tif`: all 15 format checks passed; SHA-256 `bab170ed60ab7e1a8afbfad69386a8266191fa4864ac3abac5b6290b7d25bc57`, 12,000 binary dots, one float32 band, exact local sample grid, no NaN/Inf and values in `[0,1]`. Its frozen **HOLDOUT-DTI** is 0.09579668 (`gems57-pooled-hide-v2`, 11,319 withheld positives, 95% CI `[0.08818346, 0.10324797]`), below the retained 0.152718 local incumbent; its feature-only AUC canary is 0.6868 (<0.90). Its local 3-px overlap is 0.9816 (>0.70) against the U-rank dense surface and its scan is only 19 local TIFFs, not a complete registry. It is **NOT OK TO SUBMIT**. These are frozen lane data plus a fresh local format check, not an experiment run by this review.
- The H79-RH, H79-ORIENT and H80 files are existing negative research deliverables, not new candidates from this audit. No attempt was made to evade any stop through a new name, altered dot placement, or a private uniqueness interpretation.

## Pass 3 — requirements, correctness, and limitations

1. **Submission safety:** the landing page and authorization receipt clearly distinguish inspection download from permission to submit. The correct current instruction is **do not upload any file from this repository**. A local `[0,1]` check addresses the reported value-range requirement but cannot identify the historical portal error's cause or prove remote acceptance.
2. **README formatting:** the reviewed inventory table had a malformed row combining the selective-U and D77 entries and a column spill. This review fixes the table structure without changing any scientific result or artifact.
3. **Evidence freshness:** `docs/SESSION_REVIEW.md` and some historical reports describe earlier snapshots. They must not be treated as current branch, inventory, or lane-state receipts; this review is linked from the README as the current local audit. Existing frozen evidence is retained as historical record.
4. **Official-source limitation:** this review did not independently fetch DrivenData, USGS, DOE GDR, NLR, or Stanford pages. Those official/manual-review links are retained in `PROMPT.md` and the research pages, but their current contents, availability, rules, data licensing, leaderboard state, and official training-file provenance were not newly verified here. The README's local artifacts and hash-pinned owner mirrors do not substitute for organizer-authenticated inputs.
5. **Scientific status:** H80 tested catalogue-tip Riedel R/R′ wing-crack rays from visible fault tips at +15°/−75° over 400–1,000 m. Its named mimics are linear drainages at canyon mouths and road/cadastral alignments. H80-1 is negative on buffered hide-and-recover (0.09579668 HOLDOUT-DTI, 11,319 withheld positives, 95% CI `[0.08818346, 0.10324797]`; `gems57-pooled-hide-v2`), below the retained 0.152718 HOLDOUT-DTI incumbent. Its single-feature AUC canary is 0.6868 (<0.90), but its 3-px local overlap of 0.9816 exceeds the 0.70 stop threshold. Later upstream AGO variants are archived negative artifacts; their quick 4-quadrant diagnostics are not reportable HOLDOUT-DTI scores because exact withheld-positive count and CI are missing, and they are not comparable with the pinned evaluator. Their raw quick values are retained only in the frozen card, not promoted in the landing page. The H79-ORIENT five-candidate register remains documented in `research/h79_orient_hypotheses.md`; its tested top candidate was also negative. No new candidate is authorized here. The repository's best historical local holdout claim is not an organizer score and does not establish submission eligibility.
6. **Protocol blocker:** the literal uniqueness rule conflicts with at least one registered full-footprint witness. That shared policy issue must be resolved by the protocol owner; do not introduce a repo-local exemption. A future research lane also needs a clearly assigned method paragraph, aligned blocked holdout incumbent, full valid registry, and authenticated input/specification before an experiment or slot can be responsibly selected.
7. **Leaderboard goal:** there is no defensible basis to promise a score above 0.2778 or the current leaderboard leader. The exact mapping between sibling-page file names and organizer receipts is not established here. Holdout comparisons, score projections, and owner-reported sibling results are not organizer-confirmed scores.

## Recommended next steps (no slot implied)

1. Resolve the dense-witness representation/uniqueness rule once at the shared protocol level; do not privately waive it.
2. Obtain authenticated competition data/specification access and a current official leaderboard/submission receipt source. Verify which public rows correspond to which TIFFs before using sibling score tables as evidence.
3. Select an untried, explicitly assigned research lane only after screening the existing preregistrations and code for overlap. Check access and licensing for any proposed official external data before calling a hypothesis viable.
4. Rebuild every catalogue-derived feature per buffered whole-segment holdout fold, run feature-only canaries, then use the unchanged pinned evaluator and writer. Require a matched improvement over the chosen incumbent plus the uniqueness and format checks before a separate selector considers a weekly slot.
5. Preserve the machine-readable release gate, checksum receipts, named submission note, and first-screen `OK TO SUBMIT`/`NOT OK TO SUBMIT` status. Never label local validation as organizer acceptance.

## Review run card

```json
{
  "hypothesis": "Repository audit; no new scientific hypothesis or model run was performed. H80 is the latest fully documented lane under the pinned evaluator; later upstream AGO variants use a quick non-comparable proxy and remain negative/unapproved.",
  "mechanism": "Catalogue-only geometry: estimate local strike from visible fault traces, emit soft rays at +15 and -75 degrees over 400 to 1000 m from degree-1 tips, then Poisson-place binary dots with zeros on visible catalogue pixels.",
  "named_non_fault_process_that_could_mimic_it": "Linear drainages leaving canyon mouths; cadastral or road alignments.",
  "holdout_dti": {
    "evidence_class": "HOLDOUT-DTI (frozen H80 evidence; not recalculated in this audit)",
    "evaluator_version": "gems57-pooled-hide-v2",
    "withheld_positive_pixels": 11319,
    "value": 0.09579668088380372,
    "ci95": [0.08818346093522027, 0.1032479741261599]
  },
  "correlation_overlap_vs_registry": {
    "evidence_class": "local-registry overlap diagnostic; incomplete scan",
    "final_dots_max_forward_overlap_3px": 0.9815833333333334,
    "overlap_threshold": 0.7,
    "worst_checked_raster": "gems77-h77-urank-curvature-0d868dc271bb-research.tif",
    "checked_local_rasters": 19,
    "complete_registry": false,
    "preplacement_surface_rank_correlation": "NOT RECORDED in H80 summary; cannot certify this requested check",
    "verdict": "negative; literal overlap stop already exceeded; incomplete scan is not uniqueness clearance"
  },
  "raster_sha256": "bab170ed60ab7e1a8afbfad69386a8266191fa4864ac3abac5b6290b7d25bc57 (existing artifact; no new raster generated)",
  "validator_output": {
    "evidence_class": "fresh local format validation against tracked bridge sample grid",
    "checks_passed": 15,
    "no_nan_or_inf": true,
    "values_in_0_1": true,
    "crs_shape_transform_match": true,
    "single_band_float32": true,
    "mapped_catalogue_positive_pixels": 0,
    "submission_authorization": "NOT OK TO SUBMIT; 0 authorized files"
  },
  "submission_name": "gems77-h80-riedel-wingcrack-tips-20261021-zeros (existing negative research TIFF)",
  "submission_note": "H80 Riedel 15/75deg tip rays; fold-fair HOLDOUT negative expected; research only DO NOT SUBMIT",
  "verdict": "negative"
}
```
