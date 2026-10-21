# GEMSDOE77 repository review — 2026-10-21 (Arena audit)

## Disposition

**No new experiment or competition submission was created in this review. No TIFF in the current release manifest is approved to submit.** This is intentional: H79-RH is already a completed negative lane, its uniqueness scan requires a stop, and the repository-wide selector receipt authorizes zero files. Re-running or changing that stopped lane would violate the stored run protocol. The existing H79-RH GeoTIFF can be downloaded for local inspection only; it is not a submit-ready competition entry.

- Branch reviewed: `arena/a5befe2a-gemsdoe77`.
- Repository decision: `NOT OK TO SUBMIT`, 0 authorized files (`docs/data/submission-authorization.json`).
- No organizer upload, submission-page receipt, or competition slot use was performed or inferred.
- The user-supplied family score table is not an organizer receipt. The particular attribution of 0.2778 to the named GEMSDOE32 TIFF remains unverified in the repository's own forensic account.

## Pass 1 — read and map the project

Read the operative README and original request, `PROMPT.md`, `AGENTS.md`, irregularity ledger, preregistrations, current authorization receipt, H79-RH run card and evidence, site entry pages, shared template pin, implementation/test configuration, and current artifact inventory.

The key operational instructions agree: work within the registered lane; use the pinned shared `gems57` evaluator/writer; check each feature as a leakage canary; test surface uniqueness before final placement; stop on a duplicate; label holdout values with evaluator, withheld-positive count, and interval; and keep submission-slot selection separate. The repository includes other completed negative lanes (including D77 and XSEN); this audit does not re-run or re-rank them. The present review ran no model experiment and generated no new TIFF.

The request's H33 question has a plausible metric-level explanation, not a verified causal answer. Sparse, spatially separated predictions can reduce redundant false-positive mass under a distance-tolerant DTI, and excluding pixels close to already catalogued faults may help if the evaluation emphasizes undiscovered faults. The H33 raster forensics in this repo establish byte/pixel construction facts, not that those bytes received the reported organizer score. A leaderboard row is not a raster-to-receipt link. Therefore neither a causal explanation nor a claim that this repo can exceed 0.2778 or the reported current leader is established.

## Pass 2 — executable verification and artifact review

Environment: installed the pinned `requirements.txt` into ignored `.venv/`; no tracked environment files were added.

- Test suite: **95 passed** (`.venv/bin/python -m pytest -q`).
- Repository verifier: `scripts/verify_all.py` exited successfully.
- Fail-closed selector: `scripts/release_gate.py --verify-fail-closed` returned `NOT OK TO SUBMIT`, `authorized_files: 0`.
- Static page builders (`build_site.py`, `lia_build_site.py`, `build_home.py`) completed. Their generated churn was reverted; the site source/artifacts were not changed as a side effect of this audit.
- Shared local format validator on `docs/downloads/gems77-h79-radhalo-lag3-37654-20261021-zeros.tif`: all 15 checks passed. Single-band float32; 3,730 × 3,292; EPSG:32611; sample transform match; no NaN/Inf; values in [0,1]; no positive values on mapped catalogue. This is local format validation against the tracked bridge grid, not official-data authentication, uniqueness, portal acceptance, or permission to submit.
- The exact file SHA-256 is `c272025ee653187d09489d941613a86bfe4b88110fb455847a2256cf5699014b`. This is a byte identity for the already-existing lane artifact, not evidence that it is unique under the competition protocol.
- H79-RH's frozen run card reports **HOLDOUT-DTI 0.0477326573** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.03958336, 0.05557557]`). Its paired difference vs same-run random is +0.00030988 (95% CI `[-0.00836770, 0.00895815]`), which includes zero. These are local holdout measurements, not leaderboard results.
- Feature-alone leakage canaries in the H79-RH receipt have maximum symmetric AUC 0.5084, below the registered 0.90 threshold. This is a diagnostic only; it does not demonstrate predictive usefulness.
- The H79-RH registry scan read 808 of 812 indexed rasters; four were invalid or mismatched. Its surface maximum absolute Spearman value was 0.1203, but final-dot forward overlap was 1.0 against a full-support raster (threshold 0.70). The scan is incomplete, and the recorded literal duplicate is already sufficient to stop; no uniqueness certificate can be claimed.
- The generated H79-RH artifact is a negative research deliverable, not a new submission candidate. No attempt was made to evade the stop through a new name, altered dot placement, or a private uniqueness interpretation.

## Pass 3 — requirements, correctness, and limitations

1. **Submission safety:** the landing page and authorization receipt clearly distinguish inspection download from permission to submit. The correct current instruction is **do not upload any file from this repository**. A local `[0,1]` check addresses the reported value-range requirement but cannot identify the historical portal error's cause or prove remote acceptance.
2. **README formatting:** the reviewed inventory table had a malformed row combining the selective-U and D77 entries and a column spill. This review fixes the table structure without changing any scientific result or artifact.
3. **Evidence freshness:** `docs/SESSION_REVIEW.md` and some historical reports describe earlier snapshots. They must not be treated as current branch, inventory, or lane-state receipts; this review is linked from the README as the current local audit. Existing frozen evidence is retained as historical record.
4. **Official-source limitation:** this review did not independently fetch DrivenData, USGS, DOE GDR, NLR, or Stanford pages. Those official/manual-review links are retained in `PROMPT.md` and the research pages, but their current contents, availability, rules, data licensing, leaderboard state, and official training-file provenance were not newly verified here. The README's local artifacts and hash-pinned owner mirrors do not substitute for organizer-authenticated inputs.
5. **Scientific status:** the top-ranked H79-RH hypothesis was already tested and negative. Five H79 hypotheses and their mimics are documented in `research/h79_radhalo_hypotheses.md`; no new candidate is authorized in this audit. The repository's best historical local holdout claim is not an organizer score and is not evidence of present submission eligibility.
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
  "hypothesis": "Repository audit; no new scientific hypothesis was run. H79-RH remains the latest reviewed preregistered negative lane.",
  "mechanism": "No new mechanism tested. Reviewed the existing H79-RH lagged K-Th edge / selective-U residual edge hypothesis and its frozen hide-and-recover results.",
  "named_non_fault_process_that_could_mimic_it": "Hillslope transport across a sediment or soil boundary (as preregistered for H79-RH).",
  "holdout_dti": {
    "evidence_class": "HOLDOUT-DTI (existing frozen H79-RH evidence; not recalculated in this audit)",
    "evaluator_version": "gems57-pooled-hide-v2",
    "withheld_positive_pixels": 11321,
    "value": 0.04773265734715734,
    "ci95": [0.03958335878724709, 0.05557557296791166]
  },
  "correlation_overlap_vs_registry": {
    "evidence_class": "existing H79-RH registry measurement",
    "surface_max_abs_spearman": 0.12034572542622088,
    "final_dots_max_abs_spearman": 0.017314978194690946,
    "final_dots_max_forward_overlap_3px": 1.0,
    "registry_indexed": 812,
    "registry_checked": 808,
    "verdict": "negative; duplicate stop; incomplete scan is not uniqueness clearance"
  },
  "raster_sha256": "c272025ee653187d09489d941613a86bfe4b88110fb455847a2256cf5699014b (existing artifact; no new raster generated)",
  "validator_output": {
    "evidence_class": "local format validation only",
    "no_nan_or_inf": true,
    "values_in_0_1": true,
    "crs_shape_transform_match": true,
    "single_band_float32": true,
    "submission_authorization": "NOT OK TO SUBMIT; 0 authorized files"
  },
  "submission_name": "gems77-h79-radhalo-lag3-37654-20261021-zeros (existing negative research TIFF)",
  "submission_note": "Inspection only; HOLDOUT negative and literal registry duplicate stop. Do not submit.",
  "verdict": "negative"
}
```
