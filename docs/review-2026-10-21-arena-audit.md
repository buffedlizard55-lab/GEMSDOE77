# GEMSDOE77 repository review — 2026-10-21 (Arena audit)

## Disposition

**No new experiment or competition submission was created in this review. No TIFF in the current release manifest is approved to submit.** This is intentional: the completed H79-RH and H79-ORIENT lanes are negative and their uniqueness scans require a stop; the repository-wide selector receipt authorizes zero files. Re-running or changing either stopped lane would violate the stored run protocols. Their existing GeoTIFFs can be downloaded for local inspection only; neither is a submit-ready competition entry.

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

- Test suite: **95 passed** (`.venv/bin/python -m pytest -q`).
- Repository verifier: `scripts/verify_all.py` exited successfully.
- Fail-closed selector: `scripts/release_gate.py --verify-fail-closed` returned `NOT OK TO SUBMIT`, `authorized_files: 0`.
- Static page builders (`build_site.py`, `lia_build_site.py`, `build_home.py`) completed. Their generated churn was reverted; the site source/artifacts were not changed as a side effect of this audit.
- Shared local format validator on `docs/downloads/gems77-h79-radhalo-lag3-37654-20261021-zeros.tif`: all 15 checks passed. Single-band float32; 3,730 × 3,292; EPSG:32611; sample transform match; no NaN/Inf; values in [0,1]; no positive values on mapped catalogue. This is local format validation against the tracked bridge grid, not official-data authentication, uniqueness, portal acceptance, or permission to submit.
- The exact file SHA-256 is `c272025ee653187d09489d941613a86bfe4b88110fb455847a2256cf5699014b`. This is a byte identity for the already-existing lane artifact, not evidence that it is unique under the competition protocol.
- H79-RH's frozen run card reports **HOLDOUT-DTI 0.0477326573** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.03958336, 0.05557557]`). Its **HOLDOUT-DTI paired difference** vs same-run random is +0.00030988 (evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, 95% CI `[-0.00836770, 0.00895815]`), which includes zero. These are local holdout measurements, not leaderboard results.
- Feature-alone leakage canaries in the H79-RH receipt have maximum symmetric AUC 0.5084, below the registered 0.90 threshold. This is a diagnostic only; it does not demonstrate predictive usefulness.
- The H79-RH registry scan read 808 of 812 indexed rasters; four were invalid or mismatched. Its surface maximum absolute Spearman value was 0.1203, but final-dot forward overlap was 1.0 against a full-support raster (threshold 0.70). The scan is incomplete, and the recorded literal duplicate is already sufficient to stop; no uniqueness certificate can be claimed.
- After this PR was opened, the updated `origin/main` added the already completed H79-ORIENT lane and its negative candidate. Its frozen receipt reports **0.094507 HOLDOUT-DTI** on seed 30 (`gems57-pooled-hide-v2`, 11,342 withheld positives, 95% CI `[0.076558, 0.112844]`) and **0.089063 HOLDOUT-DTI** on seed 20 (11,321 withheld positives, 95% CI `[0.072384, 0.105355]`). The card records 13 of 812 local rasters checked, a full-support witness, and final-dot forward overlap 1.0 > 0.70. These are distinct lane-fold results, not organizer scores or a matched comparison to the incumbent.
- I independently ran the shared local validator against the H79-ORIENT TIFF: 15/15 format checks passed; SHA-256 `745dbaa427bc1edaf477b3d0b4847b8be8d30d396106a61b1318a8726599be8e`. This validates local file encoding only. It does not repair the incomplete registry or duplicate stop.
- The H79-RH and H79-ORIENT files are existing negative research deliverables, not new candidates from this audit. No attempt was made to evade either stop through a new name, altered dot placement, or a private uniqueness interpretation.

## Pass 3 — requirements, correctness, and limitations

1. **Submission safety:** the landing page and authorization receipt clearly distinguish inspection download from permission to submit. The correct current instruction is **do not upload any file from this repository**. A local `[0,1]` check addresses the reported value-range requirement but cannot identify the historical portal error's cause or prove remote acceptance.
2. **README formatting:** the reviewed inventory table had a malformed row combining the selective-U and D77 entries and a column spill. This review fixes the table structure without changing any scientific result or artifact.
3. **Evidence freshness:** `docs/SESSION_REVIEW.md` and some historical reports describe earlier snapshots. They must not be treated as current branch, inventory, or lane-state receipts; this review is linked from the README as the current local audit. Existing frozen evidence is retained as historical record.
4. **Official-source limitation:** this review did not independently fetch DrivenData, USGS, DOE GDR, NLR, or Stanford pages. Those official/manual-review links are retained in `PROMPT.md` and the research pages, but their current contents, availability, rules, data licensing, leaderboard state, and official training-file provenance were not newly verified here. The README's local artifacts and hash-pinned owner mirrors do not substitute for organizer-authenticated inputs.
5. **Scientific status:** the latest merged H79-ORIENT lane was preregistered and tested negative; its five-candidate register is in `research/h79_orient_hypotheses.md`. The top candidate—cross-sensor scale-space edge-orientation persistence from RTP, isostatic gravity, detrended elevation, with depth-to-basement as a control—used existing cached layers and no new external data. Its named mimics include intrusive/lithologic contacts, survey seams, drainage and terrace edges. Its leakage canary maximum symmetric AUC was 0.6242 (<0.90), but the candidate's holdout result remained below the retained incumbent, and literal registry uniqueness failed. Four alternatives are not validated by this audit; the seismic candidate specifically depends on USGS ComCat access, which the preregistration says was unavailable. No new candidate is authorized in this audit. The repository's best historical local holdout claim is not an organizer score and is not evidence of present submission eligibility.
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
  "hypothesis": "Repository audit; no new scientific hypothesis was run. The latest merged H79-ORIENT lane is a completed negative research experiment.",
  "mechanism": "Existing H79-ORIENT method: label-blind cross-sensor scale-space edge persistence and doubled-angle orientation confluence across RTP, isostatic gravity and detrended elevation; no catalogue features or prior predictions.",
  "named_non_fault_process_that_could_mimic_it": "Intrusive/lithologic contacts, flight-line seams, pluvial shoreline or terrace edges, and drainage margins.",
  "holdout_dti": {
    "evidence_class": "HOLDOUT-DTI (existing frozen H79-ORIENT evidence; not recalculated in this audit)",
    "evaluator_version": "gems57-pooled-hide-v2",
    "seed": 30,
    "withheld_positive_pixels": 11342,
    "value": 0.09450695640486674,
    "ci95": [0.07655819744668374, 0.11284365420654904],
    "seed20_comparison": {
      "evidence_class": "HOLDOUT-DTI",
      "evaluator_version": "gems57-pooled-hide-v2",
      "withheld_positive_pixels": 11321,
      "value": 0.08906329728465655,
      "ci95": [0.0723842214768737, 0.10535529181959644]
    }
  },
  "correlation_overlap_vs_registry": {
    "evidence_class": "existing H79-ORIENT registry measurement",
    "surface_max_abs_spearman": 0.07976707194570443,
    "final_dots_max_abs_spearman": 0.003901796965253327,
    "final_dots_max_forward_overlap_3px": 1.0,
    "registry_indexed": 812,
    "local_rasters_checked": 13,
    "verdict": "negative; full-support witness exceeds the 0.70 overlap limit; incomplete registry is not uniqueness clearance"
  },
  "raster_sha256": "745dbaa427bc1edaf477b3d0b4847b8be8d30d396106a61b1318a8726599be8e (existing artifact; no new raster generated)",
  "validator_output": {
    "evidence_class": "local format validation only",
    "checks_passed": 15,
    "no_nan_or_inf": true,
    "values_in_0_1": true,
    "crs_shape_transform_match": true,
    "single_band_float32": true,
    "submission_authorization": "NOT OK TO SUBMIT; 0 authorized files"
  },
  "submission_name": "gems79-orient-persistence-37654dots-20261021-4abff47886 (existing negative research TIFF)",
  "submission_note": "H79 orientation persistence; HOLDOUT-DTI 0.0945 s30 / 0.0891 s20; registry gate failed; research only",
  "verdict": "negative"
}
```
