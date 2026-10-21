# Submission readiness review — 2026-10-21 (branch `arena/d40f78d1-gemsdoe77`)

**Current branch decision:** no lane assigned; `STOP_UNIVERSAL_DUPLICATE`; no new TIFF, experiment, holdout, or slot; NOT OK TO SUBMIT. This snapshot is the current readiness gate. See the [three-pass branch review](review-latest.html) and its [run card](evidence/review_d40f78d1_card.json).

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

Re-ran the existing `scripts/preflight_registry.py`: a candidate-independent rejection proof using the unchanged shared
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

Re-ran the existing `scripts/audit_downloads.py`, which independently opens every primary TIFF listed in `lanes.json`,
not merely trusting the manifest format flag. **37 of 37** pass SHA-256, single-band, all-finite [0,1],
no-NaN-in-footprint, no-nodata, and exact reference CRS/shape/transform (**FORMAT/INTEGRITY MEASUREMENTS**).
This scope excludes unlisted archives and NaN twins. It does not test portal acceptance, hidden performance,
full-registry uniqueness or selector approval. A retained H77L independent-audit receipt reports PASS; it was not rerun in this review.

The retained [corrected H33 scientific analysis](https://github.com/buffedlizard55-lab/GEMSDOE77/blob/main/research/why-h33-scored-02778.md) was re-read.
The exact marginal gain is `dT > alpha*DTI*(dT + 1 - k)`; the simpler `k > alpha*DTI` requires `dT == k`.
One dot can cover multiple truth cells, and mapped/unmapped faults can be close to each other. Removed unsupported
claims of fixed denominator cost, guaranteed false positives near the catalogue, exhausted emission optimization,
and a numerically required leaderboard gain. Shared runtime code is unchanged. The shared metric module's
prose also contains an overrestrictive `dT <= k` statement; the runtime evaluator does not rely on it.
This upstream documentation irregularity is reported, not repaired in a private template fork.

[Retained H33 forensics](evidence/prior_forensics.json) show **37,654** positive dots, obtained by removing
**6,436** near-catalogue dots from the **44,090**-dot predecessor, with none added (**RASTER MEASUREMENTS**).
Sparse placement and pruning plausibly reduce redundant mass. They do not prove why an organizer score changed.
**0.2778 remains OWNER-REPORT for the named TIFF**; no authenticated file-to-score receipt was obtained.
The freshly retrieved [GEMSDOE32 page at its pinned commit](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/f62038b54dcb7767d4d997f80b033cdc3d42bef9/docs/index.html) explicitly says “NO ORGANISER SCORE EXISTS for this or any artifact in this repository.” Its projections are not adopted as scores. The brief supplies conflicting leader targets. This review makes no current leaderboard or improvement claim.
Fault recovery is the contest proxy, not direct detection of geothermal vents or proof of economic permeability.

## Conditional hypothesis backlog — not authorization to implement

No experiments or feature canaries were run because the universal rejection proof precedes any new field.
Expected HOLDOUT-DTI improvement is **unknown for all four**. The rank is a qualitative priority/cost judgment,
not a projected score. Local novelty means only that the specific operator was not found by the review;
no global/cross-repository novelty is certified. These proposals are conditional and not selected lanes.

| Priority | Layers and specific physical signature | Why it might find an uncatalogued fault; difference from implemented work | Named non-fault mimic | Cost / source / readiness |
|---|---|---|---|---|
| 1 | GeoDAWN magnetic anomaly and isostatic gravity: cross-wavelet phase-jump continuity after local detrending, with independent radiometric-boundary control | Could trace buried offsets despite weak surface scarps; phase discontinuity rather than existing H80T tilt persistence or H77G trough strength | Intrusive contacts, sedimentary density contacts, survey seams | Medium; existing numerical-layer mirror routes are in the data manifest, but feature stack not restored here. No external-data download needed in principle. Blocked by protocol and assignment. |
| 2 | DEM elevation and drainage graph: signed valley-floor asymmetry reversals paired across adjacent catchments | Could reveal distributed warping away from mapped range fronts; paired graph-wide reversals, not H77E's knickpoint/branch-density or pixel scarp features | Stream capture, asymmetric lithologic erosion, fan avulsion | High; requires conditioned full DEM/drainage graph. Official USGS 3DEP route below was not accessible here. **Not viable until AOI obtainability is checked.** |
| 3 | Magnetic anomaly and gravity: local bicoherence of along-contact profiles, testing phase coupling across wavelengths | Could distinguish coherent multi-scale source offsets under fill from unrelated single-scale edges; third-order phase statistic rather than contact amplitude/Hessian curvature | Repeated layered contacts, gridding/flight-line artifacts | High; same existing numerical-layer mirror route; physical interpretation especially uncertain. Conditional research proposal, no performance evidence. |
| 4 | K/Th/U rank-intensity images: local empirical-copula change points along a persistent narrow boundary | Could detect changes in joint radiometric texture without assuming calibrated concentration ratios; distributional dependence rather than the stopped selective-U curvature subtraction | Grain sorting, moisture and soil-cover boundaries | Medium; pinned `ext_geodawn_rad_u8` mirror route. Uses quantized intensities, **not** geochemical concentrations. Conditional; may duplicate existing radiometric lanes in support. |

The existing terrain, radiometric, potential-field, tip, Riedel and tilt-persistence families are already extensively
represented in this repository. Changing a name does not create scientific novelty. Specific external official
routes: [USGS 3DEP](https://www.usgs.gov/3d-elevation-program),
[USGS GeoDAWN](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and),
[DOE GDR INGENIOUS 1391](https://gdr.openei.org/submissions/1391).
These are manual-review links, not newly verified availability or scientific validation of the proposals.

## Pass 3 — end-to-end requirements and source boundaries

- Software regression tests cover finite-positive support, Euclidean-radius boundaries, empty/invalid grids,
  input pin failure, format/range/hash failure, and the exact metric algebra. The final local suite reported **305 tests passed** (software tests, not a performance score); the CI-equivalent builds and release checks also passed.
- Entry page retains one-click TIFF/ZIP downloads and explicit NOT OK TO SUBMIT status. Evidence links and
  submission instructions remain visible. Current release authorization grants no uploads.
- Restored only pinned witnesses; no new TIFF, copied submission, trained model, geological experiment,
  HOLDOUT-DTI estimate, bootstrap CI or organizer score is represented as a result of this review.
- The last recorded scheduled source-feed receipt covers **78/78** configured owner-repository inventories (**SOURCE-HEALTH MEASUREMENT**); it was not refreshed in this branch review. It records commits, page-content hashes and TIFF paths, not verified scores. [Historical freshness feed](source-health.json). This is not an exhaustive registry beyond the configured repositories.
- Official organizer [reference README](https://github.com/drivendataorg/gems-prize-reference-solution/blob/aebe92f7c8a990f0e3443451b7a825d9afd6336b/README.md)
  was freshly retrieved via GitHub; it supports CPU-only and GPU setups. GPU is helpful, not an absolute
  requirement of that reference solution. [Research source receipt](evidence/preflight_sources.json).
- Sandbox outbound hosts are limited to GitHub/package registries. DrivenData, USGS, GDR and the PDF cannot be
  freshly verified here. No DrivenData login or submission-page receipt is available. Official data provenance,
  current weekly allowance and portal acceptance are not established by owner-mirror hashes.

## Reproduce without spending a slot

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/restore_registry.py
.venv/bin/python scripts/preflight_registry.py
# Expected exit 2: STOP_UNIVERSAL_DUPLICATE. This is not a command failure to hide.
.venv/bin/python scripts/audit_downloads.py
.venv/bin/python -m pytest -q
```

Next steps: resolve support semantics in the **shared** protocol/registry, assign one genuinely distinct lane,
restore its pinned inputs, preregister at most three experiments, run individual canaries and matched buffered
whole-segment hide-and-recover with the pinned evaluator, require a supported paired improvement over the matched
incumbent, generate through the shared writer, check both surface and dots against a complete current registry,
and only then seek separate selector approval. No automatic upload or private exemption is introduced here.

## Archived parallel-merge reconciliation from earlier reviews

An earlier review session recorded that main received PRs #47 and #48. Preserved their H28 artifacts and H81 status correction.
The H28 merge had replaced `docs/data/submission.json` with a different schema, causing `lia_build_site.py` to raise
`KeyError: zip` in the CI-equivalent build. Retained H28 metadata under `docs/data/h28-submission.json`, restored the
original LIA metadata, and added LIA/run-card identity protection plus a regression test binding both metadata records
to their unchanged TIFF hashes. This is metadata repair, not a new candidate or an ownership claim over parallel work.
At that historical point the primary download inventory had 22 entries; the new unlisted H28 archive was outside that review's listed-file audit. The full build, verifier and software tests were repeated after that reconciliation.

A subsequent main merge (PR #49) independently fixed the same metadata collision and marginal-DTI prose.
Reconciled onto its canonical `h28-submission.json` path (identical metadata bytes), preserved its review and
release-gate NaN/type hardening, and retained both regression suites. All combined tests and build checks were
repeated; no shared-template source or raster bytes were changed.

PR #52 later merged the independent H82 chemistry lane while that earlier PR was under reconciliation. Its TIFFs, negative
status, source code and tests are preserved without rerunning its experiment or claiming authorship. The earlier review then re-read **23** listed primary TIFFs. The current branch readback covers **43/43** listed primary TIFFs (FORMAT/INTEGRITY-MEASUREMENT) and still authorizes zero uploads; see the linked current audit receipt.
