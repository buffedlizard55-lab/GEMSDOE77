# GEMSDOE77 repository audit — 2026-10-21

## Decision summary

**Repository state: research-only; no file is authorized for competition upload.** The current release gate reports `NOT OK TO SUBMIT`, with zero authorized files. No new candidate was built in this audit, no competition slot was used, and no leaderboard outcome is claimed. The newest checked-in candidate selected for the H79-RH audit is retained at [`docs/downloads/gems77-h79-radhalo-lag3-37654-20261021-zeros.tif`](downloads/gems77-h79-radhalo-lag3-37654-20261021-zeros.tif); it is available for inspection only, not upload.

The apparent conflict between “generate a unique TIFF” and the run protocol is resolved conservatively: the H79-RH TIFF is a distinct raster artifact, but its final dots trip the protocol's >70% 3-pixel overlap stop against a full-support registry witness. “Unique bytes” or low rank correlation does not clear that rule. H79-RH also failed to beat the current local holdout best. Therefore a new submission build or weekly-slot recommendation would violate the stated gates. The protocol owner/selector must resolve the full-support-witness contradiction; do not create a private exemption.

## Three-pass review

### Pass 1 — inventory and declared evidence

- Read `README.md`, `PROMPT.md`, `AGENTS.md`, the irregularity ledger, template pin, preregistrations, run cards, release-gate receipt, site manifest, and the relevant artifact/build/test scripts.
- The repository retains 12 lane entries in `docs/data/lanes.json`; they are all recorded negative, and none is selector-approved. These include H77L, H79-BB, H79-RH, D77, H77D/H77-Step, H78, LIA, contact-physics, selective-U, H77E, and XSEN. Historical “promote” fields in old run cards are superseded by the fail-closed authorization receipt and must not be used as current approval.
- The local input bundle is a small `data/bridge/` set, not the full 19-band training stack. `registry/data_manifest.json` documents SHA-pinned owner mirrors and explicitly says they are not organizer-authenticated. No attempt was made to access the login-walled competition data page.
- `docs/SESSION_REVIEW.md` is older than this repository snapshot: its branch identifier and “13 TIFF” audit scope are historical, and its earlier backlog is superseded by this report. It remains unchanged as an audit record.

### Pass 2 — executable and artifact checks

- Installed the pinned `requirements.txt` in the ignored local `.venv` because the system Python had no project packages. No project or system dependency files were changed.
- `.venv/bin/python -m pytest -q`: **95 passed**.
- `.venv/bin/python scripts/release_gate.py --verify-fail-closed`: **PASS; NOT OK TO SUBMIT; 0 authorized files**.
- `.venv/bin/python scripts/verify_all.py`: completed successfully; its emitted artifact receipt reported all checks passed for the historical contact-physics TIFF. This does not establish authorization for that file or other candidates.
- Independently read the H79-RH TIFF with Rasterio and compared it with `data/bridge/sample_submission.tif`: one `float32` band, `3730×3292`, EPSG:32611, matching shape/CRS/transform, no nodata tag, finite across the complete raster, minimum 0, maximum 1, and 37,654 positive cells. SHA-256: `c272025ee653187d09489d941613a86bfe4b88110fb455847a2256cf5699014b`. The ZIP contains exactly one TIFF. These checks establish local format only; bridge-file provenance is not organizer-authenticated and format validity is not upload permission.
- The H79-RH run card records **0.047732657 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.039583359, 0.055575573]`), versus the recorded local best **0.152718 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,321 withheld positive pixels, 95% CI `[0.131548, 0.175185]`) in a separate run card. This is an unmatched comparison across candidate/fold evaluations, not evidence of a causal delta and not an organizer leaderboard score. H79-RH's paired delta against random is **+0.000310 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,321 withheld positives; 95% CI `[-0.008368, 0.008958]`), consistent with no improvement over random.
- H79-RH's recorded surface maximum absolute Spearman correlation is 0.1203 (below the 0.90 threshold); final-dot maximum forward 3-pixel overlap is 1.000 (above the 0.70 stop threshold), with 49 duplicate firings. The registry scan checked 808 of 812 indexed rasters; four were unreadable or grid-invalid, so completeness is not established. One observed witness is sufficient to fail the rule; an incomplete scan cannot certify uniqueness.
- Building the static pages and running the verifier rewrote generated HTML/receipts with a new timestamp and surfaced generated evidence files. Those generated diffs were reverted after inspection; the original tracked tree is clean apart from this report and the README pointer update. This confirms these commands are not read-only and should be run in a disposable checkout or followed by a deliberate generated-artifact review.

### Pass 3 — requirements and remaining blockers

- The submission landing page has a prominent “download for inspection / NOT OK TO SUBMIT” message and a separate how-to-submit page. That is the correct safety state while authorization is zero; do not change it to “OK” based on a validator PASS.
- The H79-RH note is 113 characters in its run card (under the portal's 140-character limit); local names and file paths are recorded there.
- `README.md` retains the full project prompt as requested. Its curated primary link is H77L by `docs/data/lanes.json`; the individual lane writeups are historical evidence, and this report is the current repository review. No official non-GitHub page was freshly fetched in this audit, and no private submission receipt or authenticated competition data was available. Official-source links below are supplied for manual verification, not cited as newly checked facts.
- User-provided score figures for individual family files are not organizer-confirmed receipts. The local H79-RH holdout is not a leaderboard score. The repo's reported top public leaderboard value must be checked on the official page before asserting it is current.

## New hypotheses to consider — not experimentally tested

The list below is a screening queue, not a claim of novelty or expected score gain. Cost/signal rankings are qualitative. The repository has already covered many static scarp, curvature, potential-field, radiometric, and lineament constructions; any proposal must be checked against the full lane registry before preregistration. No holdout result or projection is assigned to these ideas.

| Rank | Candidate and layers | Physical signature and off-catalogue rationale | Named non-fault mimic | Difference from implemented work | Signal / cost; data status |
|---:|---|---|---|---|---|
| 1 | **Drainage-network deflection and knickpoint coherence** — hydrologically conditioned high-resolution DEM, channel centerlines, local relief, and channel-gradient profiles. | Jointly score abrupt channel azimuth changes, knickpoint persistence, and aligned tributary asymmetry across neighboring catchments; a buried or degraded fault may perturb multiple channels without a preserved scarp or catalogue trace. | Resistant lithologic contacts, alluvial-fan incision, landslide dams, and engineered channels. | Moves from pixelwise curvature/scarp response to a multi-catchment network/topology test. No clear stream-network implementation was found in `src/` or `scripts/`, but grep is not proof of novelty. | **Potential signal: medium; engineering: high. BLOCKED, not viable yet.** The official source needed is USGS 3DEP elevation data (`https://www.usgs.gov/3d-elevation-program`). This host and AOI tile availability were not reachable from this audit's configured outbound network; the tracked checkout has derived LiDAR features, not a complete hydrologically suitable DEM. Obtainability is therefore **not verified**; do not preregister until checked. |
| 2 | **Cross-basin channel-profile residual alignment** — DEM-derived longitudinal profiles, basin divides, and the cached potential-field bands as independent context. | Detect a shared along-strike elevation/profile break across independent channels and ask whether its position is compatible with a basement boundary. A buried structure may produce a coherent regional base-level break even where local surface morphology is weak. | Lithologic resistance changes, basin-margin depositional transitions, and DEM seam artifacts. | Uses profile residuals across independent catchments rather than a local ridge/curvature or lineament transform. It shares DEM prerequisites with rank 1 and is not yet a separate runnable lane. | **Potential signal: low–medium; cost: high. BLOCKED** on the same unverified full DEM acquisition and hydrologic conditioning. |
| 3 | **Fault-controlled spring/well clustering with geochemical contrast** — owner-mirrored GDR well/spring chemistry coordinates, temperature, and cached radiometric/geophysical layers; exclude label-derived distance fields. | Test whether independent thermal/chemical anomalies form elongated, spatially coherent clusters along geophysically plausible structures away from catalogued faults. This may find unmapped permeable conduits, but the competition target is fault pixels, not vent locations. | Lithologic aquifers, regional groundwater flow, sampling-access bias, and spatially uneven well density. | Existing lanes use selected thermal/well context, but a preregistered spatial point-process / chemistry contrast with effort correction would differ from simple proximity fusion. Provenance and leakage safeguards are critical. | **Potential signal: medium; cost: medium–high. CONDITIONAL.** The data manifest identifies GDR 1391, DOI `10.15121/1881483`, and says the local mirrored table is owner-supplied; it does not authenticate provenance in this audit. Official review links: `https://gdr.openei.org/submissions/1391` and `https://gbcge.org/current-projects/ingenious/`. The GDR host was not reachable here, so source and license must be independently checked before use. |
| 4 | **Magnitude/depth-filtered seismic lineament consistency** — USGS ComCat event hypocenters and moment tensors, with competition geophysics for blind-fault structural context. | Test whether persistent, depth-filtered earthquake planes or hypocenter corridors align with potential-field boundaries outside the known fault catalogue. | Volcanic swarms, aftershock sequences, quarry blasts, and spatially variable event-detection completeness. | Adds a time-dependent subsurface process and explicit event-location uncertainty rather than static raster morphology. No full event catalog is checked in. | **Potential signal: medium; cost: high. BLOCKED.** Specific official source: USGS ComCat FDSN (`https://earthquake.usgs.gov/fdsnws/event/1/query`). The endpoint and study-area coverage were not checked from this network; do not call it obtainable or viable until verified. |
| 5 | **InSAR deformation-gradient persistence** — Sentinel-1 interferometric time series, coherence masks, and competition geophysics. | Seek repeatable linear deformation gradients or localized strain across suspected permeable structures, while requiring persistence across acquisition intervals. This could expose active blind faults lacking a surface scarp, but may be weak for inactive structures in the challenge labels. | Groundwater withdrawal, mining, seasonal hydrology, atmospheric phase delay, and decorrelation. | Tests temporal deformation rather than static terrain, potential-field, or radiometric gradients. | **Potential information: high; cost: very high. BLOCKED.** Official source route: Copernicus Data Space Ecosystem (`https://dataspace.copernicus.eu/`). Access, AOI coverage, coherence, and licensing were not checked here; no InSAR data is in the checkout. |

### Recommended next-session sequence

1. Resolve the missing lane assignment and the literal-uniqueness/full-support-witness policy with the shared protocol owner. Do not locally weaken the duplicate threshold.
2. Verify whether the official competition data bundle, current submission rules, registry, and weekly cap can be accessed. The official data page is `https://www.drivendata.org/competitions/306/competition-doe-gems/data/`; no credentials are available in this audit.
3. Before selecting a research lane, review the complete score/evidence ledger and rule out overlap with prior H77/H78/H79 constructions. If pursuing the drainage proposal, first verify and obtain the exact study-area DEM from the official USGS 3DEP service, then preregister; do not treat its availability as established.
4. Use only the pinned shared feature/evaluator/writer stack, with feature-alone canaries first, whole-segment buffered hide-and-recover folds, and same-support paired controls. Stay within the 3-experiment / 2-hour cap.
5. Generate a new artifact only if its pre-placement surface clears the registry gate and its holdout improves on a comparable local best. Then verify final dots, exact CRS/shape/transform, all-pixel finite range `[0,1]`, SHA, one-TIFF ZIP, and explicit selector approval. A portal receipt is the only basis for an organizer-confirmed score.

## Run card

```json
{
  "hypothesis": "Repository audit only; no geological hypothesis was experimentally tested in this review.",
  "mechanism": "Inventory, run-card, release-gate, test-suite, static-site build, and independent GeoTIFF/ZIP readback audits; ranked untested research proposals recorded above.",
  "named_non_fault_process_that_could_mimic_it": "Not applicable to an audit. Proposed mimics are listed per untested hypothesis.",
  "holdout_dti": {
    "evidence_label": "HOLDOUT-DTI",
    "evaluator_version": "gems57-pooled-hide-v2 (inherited H79-RH receipt; no new evaluation)",
    "withheld_positive_pixels": 11321,
    "score": 0.047732657,
    "ci95": [0.039583359, 0.055575573],
    "audit_status": "reported from evidence/h79_radhalo_run_card.json; not independently rerun and not a leaderboard score"
  },
  "correlation_overlap_vs_registry": {
    "label": "REGISTRY-MEASUREMENT (inherited H79-RH receipt)",
    "surface_max_abs_spearman": 0.120345725,
    "final_dots_max_abs_spearman": 0.017314978,
    "final_dots_max_forward_3px_overlap": 1.0,
    "registry_rasters_checked_of_expected": "808/812",
    "verdict": "duplicate-stop; uniqueness not cleared"
  },
  "raster_sha256": "c272025ee653187d09489d941613a86bfe4b88110fb455847a2256cf5699014b",
  "validator_output": {
    "local_readback": "PASS: one-band float32; no NaN/Inf; values [0,1]; CRS/shape/transform match data/bridge/sample_submission.tif; ZIP contains exactly one TIFF",
    "official_grid_provenance": "not authenticated in this audit",
    "portal_acceptance": "not confirmed",
    "submission_authorization": "NOT OK TO SUBMIT; 0 files authorized"
  },
  "submission_name": "gems77-h79-radhalo-lag3-37654-20261021-zeros (existing research artifact; not generated in this audit)",
  "submission_note": "H79 lag-3px U-residual halo paired to K-Th edge; 37654 spaced dots; HOLDOUT negative—research only, do not submit",
  "verdict": "negative"
}
```
