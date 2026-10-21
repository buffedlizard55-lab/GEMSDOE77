# Submission readiness review — current branch follow-up (2026-10-21)

**Branch `arena/990a6ce3-gemsdoe77`: NOT OK TO SUBMIT. No new TIFF generated.** No method paragraph/lane was assigned, no geological experiment or holdout was run, no selector decision was made, and zero competition slots were used. Existing files remain available only for inspection.

## Current evidence update

Both hash-pinned registry witnesses were restored with `scripts/restore_registry.py`, then the pinned candidate-independent check was rerun. `evidence/registry_preflight.json` returns the expected exit code **2**, `STOP_UNIVERSAL_DUPLICATE`: the witness covers **5,106,385 of 5,106,385 allowed pixels**, leaving none uncovered. Under the unchanged literal rule, every nonempty candidate has a three-pixel forward-overlap lower bound of **1.0**, greater than the strict **0.70** stop threshold. This is a **REGISTRY MEASUREMENT**, not a score, not a complete registry scan, and not uniqueness clearance. Do not privately exempt the witness or generate a candidate before shared protocol semantics are resolved.

The current independent readback covers **30/30 listed primary TIFFs** and passes local format/integrity checks; it does not establish portal acceptance or authorization. The fail-closed release receipt says **0 authorized files / NOT OK TO SUBMIT**. No new artifact, candidate-specific correlation result, canary, holdout DTI, or organizer receipt exists in this review. The machine-readable details are in `evidence/readiness_review_run_card.json`.

The 3DEP, Landsat, ComCat, Copernicus/ASF, and USGS groundwater routes named in the conditional hypothesis slate are official-source links for manual review only. Their current service behavior, AOI coverage, licensing details, and data availability were not freshly checked from this sandbox. Several external hosts are outside the sandbox's egress allowlist; those proposals are **not viable here until access and source data are verified**. They are not an assigned lane.

## Historical readiness review (prior pass; figures below are retained as historical)

The text below records an earlier readiness review. Its then-current counts and receipts are historical and are superseded where they differ from the current receipts above.

---

# Submission readiness review — 2026-10-21

**OK TO DOWNLOAD FOR INSPECTION. NOT OK TO SUBMIT. No new TIFF generated in this review.**

[Download an existing research TIFF](downloads/gems77-h77l-thk-flank-6b24305363fa-RESEARCH-ONLY.tif) ·
[How submission works](executive-summary.html) ·
[All download choices](index.html) ·
[Machine-readable preflight](evidence/registry_preflight.json) ·
[Review run card](evidence/readiness_review_run_card.json) ·
[Independent listed-download readback](evidence/listed_download_audit.json).

The requested **new protocol-unique submission is blocked**, not completed. A new name or distinct bytes
would not satisfy the literal overlap gate. This review uses the mandatory stop rule rather than
repackaging an old file or inventing an approved candidate. No competition slot was used.

## Pass 1 — review and implementation

Read the repository README and embedded original request, PROMPT, AGENTS, irregularities, preregistration,
shared-template pins, historical run cards/reviews, site builders and release controls. No identifiable new
single-method assignment was supplied. Closed lanes were not restarted; shared evaluator/writer and their
cached scientific inputs were not rebuilt or privately forked.

Added `scripts/preflight_registry.py`: a candidate-independent rejection proof using the unchanged shared
finite-positive support definition and circular three-pixel dilation. The script checks input SHA-256,
shared-template pins and exact CRS/shape/transform before deriving the allowed domain. The restored witness is
positive at **5,106,385 of 5,106,385** allowed pixels (**REGISTRY MEASUREMENT**, not a score).
Any nonempty candidate subset has forward overlap **1.0**, above the strict **0.70** stop threshold.
This covers both positive surface support before placement and any final dots. There is no candidate-specific
rank correlation because there is no new candidate; the overlap proof alone establishes rejection.

Missing/corrupt/unpinned inputs produce a fresh STOP_INPUT_ERROR receipt and a nonzero exit.
Absence of a universal witness is explicitly **not** uniqueness clearance. A partial registry scan can prove
rejection, never a pass. The two restored witnesses have pinned owner-mirror provenance, not official provenance.

## Pass 2 — independent checks and corrections

Added `scripts/audit_downloads.py`, which independently opens every primary TIFF listed in `lanes.json`,
not merely trusting the manifest format flag. The original review read 23/23 then-listed TIFFs; the current
30-file readback above supersedes that historical count. This audit checks actual hashes and grid/range/finite
properties; it does not test portal acceptance, hidden performance, full-registry uniqueness or selector approval.

Corrected [the H33 scientific analysis](metric-review.html).
The exact marginal gain is `dT > alpha*DTI*(dT + 1 - k)`; the simpler `k > alpha*DTI` requires `dT == k`.
One dot can cover multiple truth cells, and mapped/unmapped faults can be close to each other. Removed unsupported
claims of fixed denominator cost, guaranteed false positives near the catalogue, exhausted emission optimization,
and a numerically required leaderboard gain. Shared runtime code is unchanged. The shared metric module's
prose also contains an overrestrictive `dT <= k` statement; the runtime evaluator does not rely on it.
This upstream documentation irregularity is reported, not repaired in a private template fork.

Retained H33 forensics show **37,654** positive dots, obtained by removing **6,436** near-catalogue dots from the
**44,090**-dot predecessor, with none added (**RASTER MEASUREMENTS**). Sparse placement and pruning plausibly
reduce redundant mass. They do not prove why an organizer score changed. **0.2778 remains OWNER-REPORT** for the
named TIFF; no authenticated file-to-score receipt was obtained. The owner mirror explicitly says there is no
organizer score for those artifacts. The supplied leaderboard targets conflict and were not reauthenticated.
This review makes no current leaderboard or improvement claim. Fault recovery is the contest proxy, not direct
detection of geothermal vents or proof of economic permeability.

## Conditional hypothesis backlog — not authorization to implement

The following earlier backlog is historical, not an assigned lane. Expected HOLDOUT-DTI improvement was unknown
for all proposals; ranking was qualitative, not a projected score. No hypothesis below is a new result or a
submission recommendation.

| Priority | Layers and specific physical signature | Why it might find an uncatalogued fault; difference from implemented work | Named non-fault mimic | Cost / source / readiness |
|---|---|---|---|---|
| 1 | GeoDAWN magnetic anomaly and isostatic gravity: cross-wavelet phase-jump continuity after local detrending, with independent radiometric-boundary control | Could trace buried offsets despite weak surface scarps; phase discontinuity rather than H80T tilt-angle persistence or H77G trough strength | Intrusive contacts, sedimentary density contacts, survey seams | Medium; feature stack was not restored in that prior pass. No new external data needed in principle. Conditional only. |
| 2 | DEM elevation and drainage graph: signed valley-floor asymmetry reversals paired across adjacent catchments | Could reveal distributed warping away from mapped range fronts; paired graph-wide reversals, not a single knickpoint or pixel scarp | Stream capture, asymmetric lithologic erosion, fan avulsion | High; conditioned full DEM/drainage graph and access check required. Official [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) route, not freshly verified there. |
| 3 | Magnetic anomaly and gravity: local bicoherence of along-contact profiles, testing phase coupling across wavelengths | Could distinguish coherent multi-scale source offsets under fill from unrelated single-scale edges; third-order phase statistic rather than contact amplitude/Hessian curvature | Repeated layered contacts, gridding/flight-line artifacts | High; existing numerical-layer mirror route; physical interpretation uncertain. Conditional proposal only. |
| 4 | K/Th/U rank-intensity images: local empirical-copula change points along a persistent narrow boundary | Could detect changes in joint radiometric texture without assuming calibrated concentration ratios; distributional dependence rather than separate gradients | Grain sorting, moisture and soil-cover boundaries | Medium; quantized intensities are not chemical concentrations. Potential overlap with radiometric lanes; no selection. |

The existing terrain, radiometric, potential-field, tip, Riedel and tilt-persistence families are already extensively
represented. Changing a name does not create scientific novelty. Specific external official routes include
[USGS GeoDAWN](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)
and [DOE GDR INGENIOUS 1391](https://gdr.openei.org/submissions/1391); those links are not fresh access confirmations.

## Pass 3 — end-to-end requirements and source boundaries

- The prior CI-equivalent build, independent byte audit, verifier, and software tests are historical checks. The current full build/test result is recorded in `docs/review-2026-10-21-arena-990a6ce3.md` and the current run card.
- The entry page retains direct TIFF/ZIP downloads and explicit per-file NOT OK TO SUBMIT status. Current release authorization grants no uploads.
- No new TIFF, copied submission, trained model, geological experiment, HOLDOUT-DTI estimate, bootstrap CI, or organizer score was produced in this readiness review.
- Official organizer, USGS, GDR, and Copernicus links are manual-review references unless a pinned retrieval receipt explicitly says otherwise. No DrivenData login or submission-page receipt is available here.

## Reproduce the current stop without spending a slot

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/restore_registry.py
.venv/bin/python scripts/preflight_registry.py
# Expected exit 2: STOP_UNIVERSAL_DUPLICATE. Do not hide or override it.
.venv/bin/python scripts/audit_downloads.py
.venv/bin/python -m pytest -q
```

Next steps: resolve dense-surface versus dot-field overlap semantics in the **shared** protocol/registry, assign one
genuinely distinct lane, restore and pin its inputs, preregister at most three experiments, run feature-alone canaries,
then matched buffered whole-segment hide-and-recover with visible-only catalogue features and pixel-exact masking.
Only after a supported paired improvement should the shared writer and both registry checks be used; a separate
selector decision and organizer receipt are still required. No automatic upload or private exemption is introduced.
