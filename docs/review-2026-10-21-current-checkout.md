# GEMSDOE77 current-checkout review — 2026-10-21

## Disposition

**No new submission TIFF was generated, no holdout experiment was run, and no competition slot was spent. No file is cleared to submit.** This is a repo-safety and reproducibility review on branch `arena/cc014c6e-gemsdoe77`, not a scientific lane. `AGENTS.md`, `PROMPT.md`, the lane cards, and Policy I17 require stopped negative lanes to remain stopped; the release selector is a separate step. The repository's fail-closed release gate reports **NOT OK TO SUBMIT; 0 authorized files**.

The strongest historical local measurement described by the current README is **0.152718 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.131548, 0.175185]`) for H77-Step. It is not an organizer score and cannot be compared as an improvement to unmatched folds, family-reported scores, or a leaderboard entry. H33's alleged association with 0.2778 remains OWNER-REPORT in the repo; a file-specific organizer receipt is absent. This review made no fresh official competition/leaderboard access claim.

## Pass 1 — instructions, repository state, and risk inventory

- Read the operative brief and session directions, current preregistration, current run cards, template pin, inherited-artifact notice, irregularity ledger, submission pages/builders, and relevant tests.
- The operative session/lane instructions say not to rerun stopped H77/H77L/H78/H77E or the other closed negative lanes. No new assigned single-method paragraph authorizes a fresh experiment. Accordingly, no candidate-generation, holdout, registry search, or slot-selection workflow was run.
- The repository has 12 declared lanes in `docs/data/lanes.json`; each currently has `protocol_unique: false` and no selector clearance. The checked release receipt has `repository_decision: NOT OK TO SUBMIT` and `repository_authorized_file_count: 0`.
- The quick top-level site correctly distinguishes inspection downloads from submission authorization. However, several historical lane pages and README sections linked to `*-nan.tif` diagnostic variants as “reference only.” Those variants contain NaN outside the study footprint. A reference-only label was not sufficiently safe given the reported `[0,1]` portal error.

## Pass 2 — independent checks and safety corrections

### Tests and tools

- The base interpreter initially lacked NumPy and pytest. Created the ignored `.venv` and installed the exact packages in `requirements.txt`; no dependency manifest or system Python was changed.
- Full suite after reconciling the latest main and regenerating the site: **104 passed** (`.venv/bin/python -m pytest -q`).
- `scripts/verify_all.py` completed with `all_passed: true` for the selected frozen contact-physics artifact's readback, ZIP, and metric property checks. This is a selected-artifact verification, not an all-lane test or an organizer check.
- `scripts/release_gate.py --out /tmp/gems77-review-release-final.json --verify-fail-closed` returned **NOT OK TO SUBMIT; 0 authorized files** with exit status 0 for the fail-closed verification. It did not rewrite the published receipt.
- `git diff --check` passed. Static pages were rebuilt with the existing lane/home generators and link tests are included in the full suite.

### GeoTIFF inventory (expanded after main reconciliation)

I independently read every `*.tif` path in `docs/downloads/`, `docs/submissions/`, and `submissions/` with Rasterio against the tracked `data/bridge/sample_submission.tif`. This is an integrity/format measurement against the repository's local bridge raster only, not official provenance, uniqueness, scientific quality, portal acceptance, or upload approval. The audit was repeated after reconciling main, which had added concurrent H79-ORIENT/H80/AGO files.

- **38 TIFF paths total.** All match the local sample's band count, dtype, CRS, shape, and transform.
- **29 TIFF paths:** all pixels are finite and in `[0,1]`; 25 also have zeros outside the finite sample footprint. Four AGO `-zeros.tif` paths have nonzero pixels outside that local footprint; this violates the repository's conservative outside-zero convention but not the basic all-pixel numeric range check. Organizer handling is unverified.
- **9 TIFF paths:** contain NaNs outside the footprint and fail an all-raster-cells-finite / zero-outside preflight. These are historical diagnostics, not eligible submission files.
- All 38 paths remain in the repository as historical evidence; no raster bytes were modified. Removed user-facing links to NaN diagnostic TIFFs from the landing/research pages and README. Their filenames remain described as invalid diagnostics, not downloads.

### Code correction

`src/gems77lia/validate.py` and the legacy project-level `src/gems77/submission.py` audit previously allowed NaN-outside variants to appear valid when their in-footprint values were finite and bounded. That is weaker than the competition wording “Predicted values must be in range [0, 1]” and could mislead users into uploading a non-finite raster. Both local paths now require all raster cells to be finite and in `[0,1]`, with zeros outside the footprint; NaN-outside mode is explicitly diagnostic-only and never returns a passing receipt. Tests cover NaNs both inside and outside the footprint, audit the shipped diagnostic twin, and assert no HTML or README download anchor points to a `*-nan.tif` file.

This is deliberately a **conservative local policy**. It does not prove that the organizer's backend scans or rejects NaN outside its scored footprint; the original rejected file and organizer diagnostic are unavailable, so the actual cause of the reported portal error remains unknown. The fixes eliminate the local blind spot, not guarantee acceptance.

## Pass 3 — requirements and remaining risks

- No TIFF was generated because the repo protocol prohibits restarting stopped lanes and the literal registry gate has a universal dense-support witness. A new filename, format pass, or low correlation cannot clear that condition.
- No holdout DTI/CI was recalculated. No new score is asserted; all numbers in this review are explicitly format/path counts or cited historical HOLDOUT-DTI evidence, not leaderboard results.
- Existing outputs labeled “portal-equivalence” or old receipts that count only the footprint should be treated as historical evidence. The new local validator is stricter for future validation; frozen receipts have not been rewritten.
- Generated static pages may be rebuilt by CI; their source builders were updated where applicable, and tests enforce the absence of linked NaN-outside twins.
- Official inputs and rule interpretation remain limited by unavailable authenticated DrivenData access. A pinned owner mirror is not proof of organizer provenance. The portal acceptance/error cause, current submission allowance, full cross-project registry, and full-feed score receipts remain unverified.

## Handoff / highest-value next work

1. Have the protocol owner resolve the shared literal registry rule's conflict with the full-support witness. Do not patch around it locally.
2. Obtain the official competition files/specification and an authenticated file-to-score receipt for H33 before making causal claims about 0.2778; record only actual portal receipts as ORGANIZER-CONFIRMED.
3. Reconcile the all-pixel range validator with the organizer's written submission specification if authorized source access becomes available. Keep the conservative all-finite zero-outside rule until then.
4. Only after a lane is explicitly assigned and separately preregistered, run canaries and matched spatial holdouts using the pinned `gems57` evaluator/writer and cached feature stack. Do not promote without a matched incumbent, complete registry check, and separate selector decision.
5. Preserve the current fail-closed site status. A valid GeoTIFF format is not an approved competition submission.

## Review card (JSON)

```json
{
  "hypothesis": "No geological hypothesis tested; repository format-safety review only.",
  "mechanism": "Tightened local GeoTIFF validation to reject NaN values anywhere, removed direct site/README links to NaN-outside diagnostic TIFFs, and kept the release gate fail-closed.",
  "named_non_fault_process_that_could_mimic_it": "Not applicable; no geological response surface or dot placement was created.",
  "holdout_dti": {
    "evidence_class": "NOT RUN in this review",
    "evaluator_version": null,
    "withheld_positive_count": null,
    "value": null,
    "ci95": null
  },
  "correlation_overlap_vs_registry": "NOT RUN in this review; all candidate lanes remain blocked/negative under existing evidence.",
  "raster_sha256": null,
  "validator": {
    "local_bridge_grid": "38 TIFF paths match band/dtype/CRS/shape/transform; 29 are all-finite and in [0,1], of which 25 also satisfy the local outside-zero convention; 9 NaN-outside diagnostics fail strict all-pixel preflight. Four AGO zero-outside variants have nonzero values outside the local sample footprint.",
    "release_gate": "NOT OK TO SUBMIT; 0 authorized files.",
    "official_provenance_and_portal_acceptance": "NOT VERIFIED."
  },
  "submission_name": "NONE — no submission candidate generated or authorized.",
  "submission_note": "Review only; no candidate and no weekly slot used.",
  "verdict": "negative"
}
```
