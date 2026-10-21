# GEMSDOE77 repository review — 2026-10-21

## Executive disposition

**No new competition TIFF was generated, and none of the repository's current TIFFs is cleared for submission.** The checked-in/downloadable rasters are historical lane artifacts: they may be downloaded for inspection, but the current repository policy and run cards say **NOT OK TO SUBMIT**. Do not upload a file from this repository or spend a weekly slot based on this review.

The current branch is `arena/ddc5ca10-gemsdoe77`, based on `80dce02` (the repository snapshot at review start). The prior contents of this file described the first, nearly empty pilot checkout and a different branch; that is historical context and is superseded by this review. The complete user brief is retained in `PROMPT.md` and its core rules are summarized in `AGENTS.md`.

This is a local repository and artifact review, not an assertion that official competition pages, the organizer's private submission registry, or the portal were available in this session. Relevant manual-review links are maintained in `PROMPT.md`, `research/hypotheses.md`, `research/h77d_hypotheses.md`, and the run cards.

## Three-pass review

### Pass 1 — inventory and evidence map

- Read `README.md`, `PROMPT.md`, `AGENTS.md`, `docs/irregularities.md`, registrations, run cards, the submission pages/build scripts, vendored evaluator references, and test configuration.
- Found multiple completed, separately named historical lanes and an explicit stop on H78 after its registered budget. No new method paragraph was assigned in this request. The present branch must not choose another lane, restart a negative lane, or exceed its three-experiment/two-hour protocol budget.
- Confirmed the repository's fail-closed landing page says the artifacts are downloadable for inspection but are **not cleared to submit** under the current literal uniqueness policy and selector status.
- Confirmed the data manifest calls the owner-hosted GitHub mirrors integrity-pinned but **not organizer-authenticated**. The competition data page is described by the repository as login-walled; this review did not authenticate or bypass it.

### Pass 2 — independent checks and irregularities

- Installed the pinned project dependencies in an ignored local virtual environment, then ran tests: **82 passed, 0 skipped** (after wiring `src/gems77lia/core.py`, `src/gems77lia/validate.py`, `tests/test_submission_files.py`, and `scripts/verify_all.py` to fall back to the SHA-256-pinned `data/bridge/` rasters when `data/raw/` is absent). The first test attempt failed only because the base Python environment had no pytest; no system packages were changed.
- Ran the three static-site builders (`build_site.py`, `lia_build_site.py`, `build_home.py`). They completed and left the tracked worktree unchanged.
- Independently opened each of the **13 TIFF files** under `docs/downloads/`, `docs/submissions/`, and `submissions/` with Rasterio and compared it to `data/bridge/sample_submission.tif`. All 13 were single-band float32 on the local template grid (3730×3292, EPSG:32611, matching affine transform), with finite values in [0,1] inside its finite footprint. This is a **local format check against an owner-supplied bridge file**, not proof of official provenance, protocol uniqueness, portal acceptance, or permission to submit. The two NaN-outside variants have no NaN inside the template footprint. The claycap TIFF is quarantined by the repo policy; format validity does not clear it.
- The repository contains byte-identical copies of the H77-Step raster at `docs/downloads/`, `docs/submissions/`, and `submissions/` (same SHA-256 in the local audit). They are copies of one old artifact, not three new candidates; none was presented as a newly generated unique submission.
- The H77D run card's scientific `verdict: promote` is not the same as submission authorization: the same card records a literal dense-witness duplicate, and `selector_status` is not approved. The generated home page correctly says **No** to submission. Keep that distinction prominent; never translate format PASS or an informative-registry diagnostic into approval.
- H77D's statement that it beats a “champion construction” refers to its specified fold-fair holdout comparator. It does not establish that H77D is the best current lane, a leaderboard score, or a submission receipt. All such metric values must be read with their run-card evidence class, evaluator, withheld-positive count, and confidence interval.
- `gh pr list` showed PR #11 open from another branch. Its title/body make holdout claims, but those are not independently re-run in this review and its branch is outside this session's authorized branch/lane. It is not merged evidence here and was not used to select or change a candidate.
- Network/source limitation: only the repository's GitHub material was available through the configured outbound hosts. This review did not freshly fetch DrivenData, USGS, DOE GDR, NLR PDF, or other non-allowlisted endpoints. Their links remain for manual review; no live official-source availability, rule, leaderboard, license, or score claim is newly certified here.

### Pass 3 — requirements and safety re-check

- The main request is preserved in `PROMPT.md`; the README instructs contributors to read it each session. The download and submission distinction is explicit on the generated entry and executive-summary pages.
- The local GeoTIFF gate is not the scientific/uniqueness gate. The current evidence reports negative/blocked candidates, and the protocol demands duplicate checks on both surface and final dots against the full applicable registry. A dense witness currently makes literal uniqueness clearance fail closed.
- No experiment was run in this review, no TIFF bytes were changed, no holdout number was recalculated or relabelled, and no competition slot was used. This is deliberate: the H78 allocation has consumed its registered experiment budget, the request contains no newly assigned lane paragraph, and the instructed promotion threshold was not met for a currently authorized new candidate.
- The supplied GEMSDOE32 `0.2778` and family history remain **OWNER-REPORT** unless backed by an organizer receipt. The public competition leader value in old files is a dated observation, not a current score statement. A predicted/holdout metric is not an organizer-confirmed competition score.

## Research backlog for a separately authorized lane

The hypotheses below are research proposals, not validated results or permission to run them. They are ranked by qualitative expected signal per engineering cost. Their expected DTI gain is **unknown**; no projection is supplied. Candidate 1 is the only proposal that could be assessed without introducing a new scientific data source, but still needs a valid assignment, restored inputs, and a fresh preregistration. **Do not run it under this review or spend a submission slot.**

| Rank | Hypothesis and required layers | Target signature and rationale for a catalogue-missing fault | Named non-fault mimic | Difference from implemented lanes | Cost / status |
|---:|---|---|---|---|---|
| 1 | **Scale-space magnetic–gravity source-depth concordance** — competition bands for RTP/TMI, magnetic and gravity vertical/horizontal derivatives, and depth-to-basement. | Compare contact-ridge persistence and phase/gradient concordance across fixed wavelength bands, then require a spatially compatible basement-depth step. Persistent buried offsets could signal faults under basin fill without a surface scarp. | Lithologic contacts, pluton margins, and survey-line seams. | Existing contact-physics uses derived contact transforms/confluence and signed basement step; this proposal's differentiator is pre-registered multi-scale source-depth consistency, not a new feature label or a re-tuned version of that lane. A code/history grep and preregistration review would be required to prove it is truly unimplemented before use. | **Medium expected signal / medium engineering cost; conditional.** It uses competition-provided geophysics in principle, but this clone has only bridge rasters, and the full inputs are owner mirrors rather than organizer-authenticated. Restore/authorize data and assign a lane before any holdout. |
| 2 | **Fault-controlled drainage deflection and channel-profile knickzones** — detrended elevation and a hydrologically conditioned DEM/derived drainage network. | Jointly rank abrupt channel deflection, aligned tributary asymmetry, and knickzone persistence across adjacent channels; a fault can perturb several neighboring channels even where its scarp is degraded or buried. | Resistant lithologic boundaries, fan incision, and human drainage/road infrastructure. | Prior lanes emphasize scarp/edge morphology, potential fields, LiDAR, and hydrothermal proximity; drainage topology is a network-scale geomorphic test rather than a pixelwise scarp detector. Repository search found no clear implementation, but absence from grep is not a proof of novelty. | **Medium expected signal / high data-processing cost; not currently viable.** Specific free official source: USGS 3DEP elevation data (`https://www.usgs.gov/3d-elevation-program`). The source host could not be checked from this sandbox; no new DEM was downloaded, and raw DEM coverage is absent from this clone. |
| 3 | **Shallow seismicity lineament / fault-plane alignment** — event locations and, where available, moment tensors from USGS ComCat FDSN; competition geophysics only as a context/control layer. | Test whether a robust, depth- and magnitude-filtered event-plane/lineament cluster aligns with a geophysical boundary or geothermal setting, potentially exposing active blind structures not in a surface fault catalogue. | Volcanic swarms, aftershock sequences, quarry blasts, and spatially varying network detection. | Existing shipped seismic bands are coarse aggregates; no sub-kilometre event catalog is in the tracked data. This requires external event data and a distinct uncertainty-aware point-process test. | **Medium signal / high processing cost; blocked.** Specific free official source: USGS ComCat FDSN event service (`https://earthquake.usgs.gov/fdsnws/event/1/query`). This sandbox could not reach that non-allowlisted host; obtainability is therefore unverified and the idea is not viable until access is checked. |
| 4 | **InSAR deformation-gradient coherence around geothermal systems** — Sentinel-1 interferometric time series from the official Copernicus Data Space ecosystem, co-registered to the study grid and screened for coherence/seasonality. | Recurrent linear strain gradients or localized subsidence/uplift may reveal permeable active fault zones not expressed as a mapped scarp. | Groundwater withdrawal, mining, seasonal hydrology, and atmospheric phase delay. | This would test present-day deformation dynamics, rather than static terrain/potential-field or catalogue proximity. It is a substantial new-data and processing lane, not a small feature tweak. | **Potentially high information / very high cost; blocked.** Official-source route: Copernicus Data Space Ecosystem (`https://dataspace.copernicus.eu/`). The host and AOI product availability were not checked here; no InSAR product is present in this clone. Do not describe as obtainable until verified. |

### Gate for the next session

1. Read `README.md`, `PROMPT.md`, `AGENTS.md`, this review, and the current preregistration/run cards.
2. Get an explicit single-method lane assignment and protocol-owner resolution of the dense-raster uniqueness contradiction. Do not create a private exception.
3. Authenticate the official competition inputs/specification and obtain the full current registry. Reconcile the open PR #11 independently before naming an incumbent; do not import its claims as facts.
4. Select one backlog hypothesis, check prior implementation and data source availability, and preregister within a fresh three-experiment/two-hour budget. Use the unchanged pinned shared cached-feature/evaluator/writer path. Run feature-alone leakage canaries first.
5. Generate a new candidate only after holdout improvement over the selected incumbent, both surface/final-dot uniqueness checks, exact-grid/range validation, and a valid SHA receipt. Make the download status say **OK TO SUBMIT** only after uniqueness and selector clearance; otherwise explicitly say **NOT OK TO SUBMIT**.
6. Record any real portal receipt as ORGANIZER-CONFIRMED. A holdout result, projection, raster validation, or score listed on a sibling website is not that receipt.

## Review run-card summary

```json
{
  "hypothesis": "Repository audit only; no new candidate hypothesis was experimentally tested.",
  "mechanism": "Reviewed historical scientific hypotheses, protocol gates, generated pages, local artifacts, tests, and the shared-template references without changing model or raster bytes.",
  "named_non_fault_process_that_could_mimic_it": "Not applicable to an audit; the research backlog lists mimics for each untested hypothesis.",
  "holdout_dti": {
    "evidence_class": "HOLDOUT-DTI",
    "evaluator_version": "NOT RUN in this review",
    "withheld_positive_count": "NOT RUN",
    "value": "NOT RUN",
    "95_percent_ci": "NOT RUN"
  },
  "correlation_overlap_vs_registry": "NOT RUN in this review; inherited lane run cards report literal uniqueness failure/selector block. No new uniqueness certificate.",
  "raster_sha256": "No new raster generated. Existing artifacts' hashes are recorded in their respective run cards; the local format audit does not approve them.",
  "validator": {
    "local_artifact_audit": "13 TIFF paths checked against data/bridge/sample_submission.tif; single-band float32, matching local CRS/shape/transform, finite and [0,1] in finite footprint.",
    "official_grid_provenance": "Not independently authenticated in this review.",
    "portal_acceptance": "NOT CONFIRMED",
    "submission_clearance": "FAIL CLOSED: no artifact approved to submit"
  },
  "submission_name": "NONE — no new TIFF and no authorized lane",
  "submission_note": "No new candidate generated; existing downloads are inspection-only, not cleared for submission.",
  "verdict": "negative"
}
```

---

## Session Update — 2026-10-21 (Branch `arena/b2132ad0-gemsdoe77`, Lane `H77-TSD`)

### Executive Disposition
- **New Inspection Artifact Generated (`H77-TSD`):** `docs/downloads/gems77-tsd-transtens-swarm-submod-37654dots-20261021-zeros.tif` (`132,961 B`, SHA-256 `4ebb1da423d26d9823ad15c4f8a0e722edc3e79272c8e720adcd28e6224af68c`) and single-TIFF `.zip` (`94,346 B`, SHA-256 `2bbee85e6eb04ff17b3818f032fca1087a794a343b6383a8c4c97a3d27e0e884`).
- **OK to Download for Inspection:** **YES** (`15/15` local format checks `PASS`, single-band `float32`, `EPSG:32611`, `3730×3292`, exact geotransform, values strictly in `[0.0, 1.0]`, `0` NaN anywhere, `0` Inf, `nodata=None`, `37,654` binary dots, `0` dots on mapped catalogue traces).
- **OK to Submit to DrivenData:** **NO — NOT OK TO SUBMIT (Verdict: `negative` under Policy I17).** Although `H77-TSD` achieves significant holdout improvements over the standard-stack baseline on both independent whole-component holdout seeds (`Seed 30`: `0.13602` `HOLDOUT-DTI`, `95% CI [0.1213, 0.1494]`, `+0.03699` vs standard greedy; `Seed 20`: `0.13371` `HOLDOUT-DTI`, `95% CI [0.1180, 0.1492]`, `+0.02578` vs standard greedy) and passes all sparse dot-field uniqueness gates across all `675` sparse rasters (`max 3-px forward overlap = 0.6917 <= 0.70`, `Phase 1 surface max |ρ| = 0.4526 <= 0.90`, `Phase 2 dot max |ρ| = 0.0863 <= 0.90`), the full-footprint continuous witness `17GEMSDOE_E-proba-multiscale` (`ab0a0a62`) forces literal 3-px forward overlap = `1.000 > 0.70` under Policy I17, and `0.13602` remains below the `0.15272` repo holdout ceiling. Zero competition slots were spent.
- **Full Technical Report & Hypotheses:** [`docs/tsd.html`](tsd.html) · [`research/hypotheses-tsd-2026-10-21.md`](research/hypotheses-tsd-2026-10-21.md) · [`evidence/run_card_tsd.json`](evidence/run_card_tsd.json).

---

## Session Update — 2026-10-21 (Branch `arena/5c385eb9-gemsdoe77`, Lane `H83-PCD`: 3-Pass Audit & Verification)

### Pass 1 — Implementation & Verification
- Restored and SHA-256-verified all 23 pinned competition/external/calibration files in `data/raw/` (`scripts/restore_data.py --ids all`) and both pinned registry witnesses in `.cache/registry/` (`scripts/restore_registry.py` + `scripts/preflight_registry.py`, confirming exit code 2 `STOP_UNIVERSAL_DUPLICATE` on `ab0a0a62`).
- Formulated and ranked 5 untried geological hypotheses in [`research/h83pcd_hypotheses.md`](../research/h83pcd_hypotheses.md) (`H83-PCD`, `H83-WORM`, `H83-COPULA`, `H83-INSAR`, `H83-COMCAT`), verifying external-source obtainability before running any code, and locked [`registry/preregistration_h83pcd.json`](../registry/preregistration_h83pcd.json).
- Implemented `src/gems77_pcd/features.py` (8 label-free `H83-PCD` channels + 5 fold-visible 12-bit Leave-Own-Component-Out `LOCO` channels) and `src/gems77_pcd/allocator.py` (`build_sparse_halo_bitmask` + `allocate_submodular_300m_pcd` with bitwise `uint64` per-raster 3-px overlap cap `<= 0.64`).

### Pass 2 — Review for Bugs, Missing Requirements, Assumptions, Edge Cases & Fixes
- **Bug Fixed (`IR-H83PCD-01` — 4,097-Knot Quantile Tail-Compression Trap):** Initial blending applied `rank01_fp` (`4,097` quantile knots) to `s_pcd_static` and `s_pcd_mech` before adding `0.20 * (comp_2d + anti_2d)`, compressing the top `0.73%` (`37,654` dots) into `29` discrete knots (`width 0.0073`) and letting the additive composite overwhelm the classifier's tail ranking (`0.080815 HOLDOUT-DTI`). Replacing quantile-squashed blending with calibrated probability blending (`0.45 * s_pcd_static + 0.55 * s_pcd_mech`) and `early_stopping=False` (`IR-H83PCD-02`) raised `HOLDOUT-DTI` to **`0.132515`** on Seed 30 (`95% CI [0.117209, 0.145631]`, paired `+0.046276` over `A_std_greedy`) and **`0.123066`** on Seed 20 (`95% CI [0.108735, 0.138706]`, paired `+0.035720` over `A_std_greedy`).
- **Memory & EDT Optimization Fixed:** Replaced `ndi.distance_transform_edt(..., return_indices=True)` in `fill_nan_nearest` and `np.fft.rfft2` (`complex128`) with `clean_band` (footprint median fill) and single-precision `scipy.fft.rfft2`, cutting peak RAM below `400 MB`.
- **Site & Test Contract Checked:** Verified zero `-nan.tif` links exist in `docs/*.html` or `README.md`, verified `README.md` ends verbatim with `docs/PROJECT_PROMPT.md`, and verified `docs/data/lanes.json` and `docs/data/submission-authorization.json` remain fail-closed (`repository_authorized_file_count == 0`).

### Pass 3 — Final Prompt Re-Check, Accuracy, Reliability & Code Quality
- Re-ran `scripts/audit_downloads.py`, `scripts/build_home.py`, `scripts/release_gate.py --verify-fail-closed`, `scripts/verify_all.py`, and the full `pytest` suite (`220 passed`).
- **Final Disposition (`H83-PCD`):**
  - **OK to Download for Inspection:** **YES** (`docs/downloads/h83pcd-poisson-piedmont-loco-submod-20261021T183000Z-zeros.tif`, `142,109 B`, SHA-256 `1ff7bc32563b56a6b18beb0d4500afa9bad814396485aaba39b2b8cbc8a9fa59`; companion `.zip` SHA-256 `cc85430ca8eb272cf7988fbce740a9c86f011a5eadb7fedb8d9cc3fccaa95c61`; `15/15` shared validator + `17/17` independent readback checks PASS; `100%` of 32 sparse dot-field rasters pass with `max 3-px forward overlap = 0.6400 <= 0.70`, `0.3224` vs `H33`, `max |Spearman rho| = 0.2120 <= 0.90`).
  - **OK to Submit to DrivenData:** **NO — NOT OK TO SUBMIT (`verdict: "negative"`, 0 weekly slots spent)** because Policy `I17` literal all-raster overlap fails on continuous witness `17GEMSDOE_E-proba-multiscale` (`ab0a0a62`, `1.0000 > 0.70`) and `0.132515 / 0.123066 HOLDOUT-DTI` does not exceed the retained `0.152718` `H77-Step` incumbent.


