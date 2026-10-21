# GEMSDOE77 repository audit — 2026-10-21

## Decision summary

**Repository state: research-only; no file is authorized for competition upload.** The current release gate reports `NOT OK TO SUBMIT`, with zero authorized files. No new candidate was built in this audit, no competition slot was used, and no leaderboard outcome is claimed. During this review, `origin/main` advanced again and added three AGO ML×geophysics research variants; H80 remains the latest parallel geology-only lane. The AGO files are available for research inspection only, not submission, and their quick-evaluator measurements do not have the withheld-positive counts/95% CIs needed for a qualified HOLDOUT-DTI receipt.

The apparent conflict between “generate a unique TIFF” and the run protocol is resolved conservatively: H80 is a distinct file, but its final dots have 0.9816 3-pixel overlap with a checked local selective-U raster, above the 0.70 stop. “Unique bytes” or low rank correlation does not clear that rule. H80 also failed to beat the current local holdout best. Therefore a new submission build or weekly-slot recommendation would violate the stated gates. The protocol owner/selector must resolve the full-support-witness contradiction; do not create a private exemption.

The later-merged AGO-Hybrid35, AGO-Hybrid50, and AGO-Select artifacts are upstream experiments, not experiments performed or revalidated in this audit. Their stored values (0.103445, 0.089401, 0.144364) use `quick-4quadrant-spatial (30px buffer)`, lack recorded withheld-positive counts and 95% CIs, and cannot be compared with `gems57-pooled-hide-v2` results or the reported leaderboard. The two displaced variants pass only a backbone-specific overlap check; the universal full-support witness still triggers the literal protocol gate. All three remain negative with no selector approval.

## Three-pass review

### Pass 1 — inventory and declared evidence

- Read `README.md`, `PROMPT.md`, `AGENTS.md`, the irregularity ledger, template pin, preregistrations, run cards, release-gate receipt, site manifest, and the relevant artifact/build/test scripts.
- The merged repository now retains 17 lane entries in `docs/data/lanes.json`; all are recorded negative and none is selector-approved. The H80 entry was restored during this audit after the first upstream H80 merge omitted it from the generated lane manifest; the next upstream merge added three AGO variants but again omitted H80. The manifest now includes H80, H77L, H79-BB, H79-RH, H79-ORIENT, D77, H77D/H77-Step, H78, LIA, contact-physics, selective-U, H77E, XSEN, and AGO-Hybrid35/50/Select. The H80 entry is checked against its frozen run card, and a test ensures its hash remains represented in the generated fail-closed site. Historical “promote” fields in old run cards are superseded by the authorization receipt and must not be used as current approval.
- The local input bundle is a small `data/bridge/` set, not the full 19-band training stack. `registry/data_manifest.json` documents SHA-pinned owner mirrors and explicitly says they are not organizer-authenticated. No attempt was made to access the login-walled competition data page.
- `docs/SESSION_REVIEW.md` is older than this repository snapshot: its branch identifier and “13 TIFF” audit scope are historical, and its earlier backlog is superseded by this report. It remains unchanged as an audit record.

### Pass 2 — executable and artifact checks

- Installed the pinned `requirements.txt` in the ignored local `.venv` because the system Python had no project packages. No project or system dependency files were changed.
- `.venv/bin/python -m pytest -q` after reconciling H80 and the upstream AGO manifest/site: **101 passed**.
- `.venv/bin/python scripts/release_gate.py --verify-fail-closed`: **PASS; NOT OK TO SUBMIT; 0 authorized files**.
- `.venv/bin/python scripts/verify_all.py`: completed successfully; its emitted artifact receipt reported all checks passed for the historical contact-physics TIFF. This does not establish authorization for that file or other candidates.
- Independently read the merged H80 TIFF with Rasterio and compared it with `data/bridge/sample_submission.tif`: one `float32` band, `3730×3292`, EPSG:32611, matching shape/CRS/transform, no nodata tag, finite across the complete raster, minimum 0, maximum 1, and 12,000 positive cells. SHA-256: `bab170ed60ab7e1a8afbfad69386a8266191fa4864ac3abac5b6290b7d25bc57`. The ZIP contains exactly one TIFF. These checks establish local format only; bridge-file provenance is not organizer-authenticated and format validity is not upload permission.
- Independently read all three AGO zero-variant TIFFs against the same sample grid: each is one-band `float32`, `3730×3292`, matching CRS/transform, finite in `[0,1]`, with 37,654 positive pixels; each SHA-256 matches `docs/data/lanes.json`, and each ZIP contains exactly the byte-identical one TIFF. These are local format checks only, not competition authorization or scientific validation.
- The H80 run card records **0.095796681 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,319 withheld positive pixels, 95% CI `[0.088183461, 0.103247974]`) against a random control of **0.014885619 HOLDOUT-DTI** (same evaluator and positives; no CI recorded for the control). Its retained comparator is **0.152718 HOLDOUT-DTI** (`gems57-pooled-hide-v2`, 11,321 withheld positives, 95% CI `[0.131548, 0.175185]`) from H77-Step. These are local measurements on different folds, not organizer leaderboard scores or a matched causal comparison. The feature-only canary reports AUC 0.686824 on 11,319 withheld positives, below the 0.90 leakage ceiling.
- H80's run card reports a local maximum 3-pixel dot overlap of 0.981583 with the dense selective-U TIFF, above the 0.70 stop threshold, across 19 locally compared files. The current report card does not record a pre-placement surface Spearman value; that surface gate therefore cannot be independently cleared from H80's receipts. The full registry remains unchecked. H79-ORIENT and H79-RH also have separate literal overlap stops; no lane is cleared by a low correlation diagnostic.
- Static-site builders, release-gate generation, and verifier commands can rewrite generated HTML/receipts or create evidence output. The upstream AGO merge added three negative variants and a 16-lane authorization receipt but omitted H80 from `docs/data/lanes.json`; its generated start page still referenced H80. Restored H80 to the 17-entry manifest, kept all three AGO variants, added generator navigation and explicit caveats for their incomplete quick-evaluator metrics, and regenerated the start/submission pages and zero-authorization receipt. Added tests for both H80 hash/clearance and the AGO per-file fail-closed status; repaired the AGO report's broken run-card links. Treat these commands as mutating, and review generated diffs before staging.

### Pass 3 — requirements and remaining blockers

- The submission landing page has a prominent “download for inspection / NOT OK TO SUBMIT” message and a separate how-to-submit page. That is the correct safety state while authorization is zero; do not change it to “OK” based on a validator PASS.
- The H80 submission note is 94 characters and the AGO-Hybrid35 note is 99 characters in their run cards (both under the 140-character limit). AGO's run card records the two displacement variants' file names, hashes, and local validator receipts; none has selector approval.
- `README.md` retains the full project prompt as requested. Its curated primary link is H77L by `docs/data/lanes.json`; the individual lane writeups are historical evidence, and this report is the current repository review. No official non-GitHub page was freshly fetched in this audit, and no private submission receipt or authenticated competition data was available. Official-source links below are supplied for manual verification, not cited as newly checked facts.
- User-provided score figures for individual family files are not organizer-confirmed receipts. The H80 value above is local HOLDOUT-DTI, not a leaderboard score. The repo's reported top public leaderboard value must be checked on the official page before asserting it is current.

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

1. Resolve the lane-assignment question and literal-uniqueness/full-support-witness policy with the shared protocol owner. Reconcile the newly merged AGO diagnostic artifacts with their quick-evaluator limitations and the experiment assignment before any further run. Do not locally weaken the duplicate threshold.
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
    "evaluator_version": "gems57-pooled-hide-v2",
    "withheld_positive_pixels": 11319,
    "score": 0.095796681,
    "ci95": [0.088183461, 0.103247974],
    "random_control": {
      "evidence_label": "HOLDOUT-DTI",
      "evaluator_version": "gems57-pooled-hide-v2",
      "withheld_positive_pixels": 11319,
      "score": 0.014885619,
      "ci95": "not recorded"
    },
    "local_comparator": {
      "evidence_label": "HOLDOUT-DTI",
      "evaluator_version": "gems57-pooled-hide-v2",
      "withheld_positive_pixels": 11321,
      "score": 0.152718,
      "ci95": [0.131548, 0.175185],
      "note": "separate H77-Step receipt; not a leaderboard score or matched fold comparison"
    }
  },
  "upstream_ago_diagnostics": [
    {
      "evidence_label": "UNQUALIFIED EXPLORATORY HOLDOUT (upstream calls this HOLDOUT-DTI; incomplete receipt)",
      "evaluator_version": "quick-4quadrant-spatial (30px buffer)",
      "withheld_positive_count": "not recorded",
      "ci95": "not recorded",
      "organizer_confirmed": false,
      "variants": {
        "AGO-Hybrid35": 0.103445292,
        "AGO-Hybrid50": 0.089400992,
        "AGO-Select": 0.144363654
      },
      "note": "Do not compare with gems57-pooled-hide-v2 or leaderboard scores; no selector approval."
    }
  ],
  "correlation_overlap_vs_registry": {
    "label": "LOCAL-REGISTRY-MEASUREMENT (19 TIFFs checked; not full-family registry)",
    "surface_spearman": "not recorded in H80 run card",
    "final_dots_max_forward_3px_overlap": 0.981583333,
    "overlap_witness": "gems77-h77-urank-curvature-0d868dc271bb-research.tif",
    "full_registry_scan": "not performed; no uniqueness clearance"
  },
  "raster_sha256": "bab170ed60ab7e1a8afbfad69386a8266191fa4864ac3abac5b6290b7d25bc57",
  "validator_output": {
    "local_readback": "PASS: one-band float32; no NaN/Inf; values [0,1]; CRS/shape/transform match data/bridge/sample_submission.tif; ZIP contains exactly one TIFF",
    "official_grid_provenance": "not authenticated in this audit",
    "portal_acceptance": "not confirmed",
    "submission_authorization": "NOT OK TO SUBMIT; 0 files authorized"
  },
  "submission_name": "gems77-h80-riedel-wingcrack-tips-20261021-zeros (existing research artifact; not generated in this audit)",
  "submission_note": "H80 Riedel 15/75deg tip rays; fold-fair HOLDOUT negative expected; research only DO NOT SUBMIT",
  "verdict": "negative"
}
```
