# GEMSDOE77 Session Findings — 2026-10-21 (branch arena/3cb42c38-gemsdoe77)

## Executive summary

**No new GeoTIFF was generated and no file is authorized for competition submission.** This is a protocol-enforced stop, not a code bug or missing data. The binding literal registry-uniqueness rule rejects every nonempty candidate before any experiment is spent.

## 1. Why h33-h33-2-b2 (reported 0.2778) — PhD-level assessment

### What is established (measurements, not scores)

| Item | Established? | Evidence |
|------|-------------|----------|
| H33 mirror has 37,654 binary positive pixels | Yes | Hash-pinned readback, SHA-256 `c55bafc4…`, `evidence/h77l_independent_audit.json` |
| None of the 37,654 pixels fall on the mapped catalogue | Yes | Nearest catalogue distance is √5 pixels |
| H33 removes 6,436 dots from a 44,090-dot D2.8 mirror, adding no new dots | Yes (historical) | `evidence/prior_forensics.json`, not freshly re-derived |
| 0.2778 is an organizer-confirmed score for these exact bytes | **No** | GEMSDOE32 site explicitly states "NO ORGANISER SCORE EXISTS for this or any artifact in this repository" (pinned commit f62038b5). Owner-reported only. |
| Public leaderboard #1 is 0.4413 (ORGANIZER-CONFIRMED, user xiaofanhu) | Yes | DrivenData leaderboard page, but does not identify underlying TIFF bytes |

### Plausible mechanism — not causal proof

Under the pinned DTI implementation (`vendor/gems57/metric.py`, α=0.2, β=0.8, 300 m triangular kernel):

```
D = α(T + S − M) + βK
s = T / D
```

For a new binary dot at pixel p with nearest-truth kernel weight k and total new truth coverage ΔT:

```
ΔD = α(ΔT + 1 − k)
new s > old s  iff  ΔT > α·s·(ΔT + 1 − k)
```

The simplified rule "include dot iff k > α·s" holds **only when ΔT = k**. A dot can cover multiple truth cells (ΔT > k), and an overlapping dot can add zero truth credit (ΔT = 0). Thinning a thick ribbon therefore reduces false-positive mass (S increases by 1 per dot, but M increases by less than 1 when k < 1) while potentially preserving much of the max-kernel truth coverage. Pruning near-catalogue dots can help if those dots mostly predict structures excluded from the hidden target.

**Neither mechanism is guaranteed**: thinning may remove uniquely useful coverage, and a genuinely unmapped fault can lie near a mapped fault. Pixel-exact catalogue masking does not imply blanket exclusion of its halo.

### Can a new method beat it?

Possibly, but **not established by this review**. The defensible next test per the standing science instruction is:

1. Preregister one catalogue-independent geological hypothesis
2. Run single-feature leakage canaries (AUC ≤ 0.90 ceiling)
3. Whole-segment buffered hide-and-recover holdout with visible-only catalogue features
4. Compare to a rebuilt incumbent on identical folds, withheld positives, masks, dot budget and evaluator version
5. Publish pooled DTI with segment/block-resampled 95% CI

An existing final H33 raster is **not** a leakage-safe holdout baseline if it was constructed with the entire catalogue.

## 2. Why a unique TIF submission cannot be generated right now

### Preflight result (candidate-independent, REGISTRY-MEASUREMENT)

```
decision: STOP_UNIVERSAL_DUPLICATE
allowed_pixels: 5,106,385
witness_positive_allowed_pixels: 5,106,385
allowed_pixels_within_3px: 5,106,385
uncovered_allowed_pixels: 0
universal_overlap_blocker: true
nonempty_subset_forward_overlap_lower_bound: 1.0
overlap_stop_threshold_strictly_greater_than: 0.70
authorized_to_submit: false
verdict: negative
```

The pinned dense witness (`ab0a0a62`, from repo 17GEMSDOE, ref `6fc71d19`, path `docs/downloads/17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif`) is positive at **every** allowed pixel. Under the binding literal rule:

> If allowed domain A is contained in the witness's 3px dilation W, every nonempty candidate C ⊆ A has |C ∩ W| / |C| = 1 > 0.70.

This applies to **both** positive surface support and final dots. No low rank correlation, new filename, or byte-level novelty can undo this rejection. It is a deterministic rejection certificate, not geological similarity evidence.

### What must change before any TIF can be generated

Per `evidence/registry_preflight.json` `next_step`:

> Resolve support semantics in shared protocol/registry, then assign a lane. Do not silently exempt dense rasters or rerun closed lanes.

The required sequence (from `docs/review-2026-10-21-60aafd45.md`, next session):

1. **Resolve dense-surface vs dot-field overlap semantics** in the shared protocol/registry; do not silently exempt the witness.
2. **Supply one assigned method lane**, restore its hash-pinned cached inputs, verify physical band meanings, freeze a complete current registry snapshot.
3. **Run single-feature leakage canaries**, then whole-segment buffered hide-and-recover with visible-only catalogue features and pixel-exact masking.
4. **Compare to a rebuilt incumbent** on identical folds; pool DTI and report withheld count, evaluator version/hash and 95% CI.
5. **Only after required holdout improvement**: scan surface and final dots, write with the shared writer, validate exact grid/finite [0,1], bind every receipt to the output hash.
6. **Obtain separate selector approval** and check current submission allowance before any upload.

## 3. Download vs submit status — obvious on every page

Every page on the site (`docs/index.html`, `docs/executive-summary.html`, every lane page) displays:

> **OK TO DOWNLOAD FOR INSPECTION: YES · NOT OK TO SUBMIT: NO**

- All linked research TIFFs are downloadable for inspection
- **Every file is NOT OK TO SUBMIT** — the machine-readable authorization receipt (`docs/data/submission-authorization.json`) says `repository_decision: NOT OK TO SUBMIT`, `repository_authorized_file_count: 0`
- Format validation PASS (single-band float32, EPSG:32611, 3730×3292, values in [0,1], 0 NaN) is **not** organizer acceptance
- The earlier portal error `"Predicted values must be in range [0, 1]"` cannot recur from format-valid files (the shared writer rejects out-of-range values), but local format validity does not prove portal acceptance

## 4. Existing inspection-only files (all NOT OK TO SUBMIT)

20+ artifacts are listed in `docs/data/submission-authorization.json`, each with SHA-256, format readback, and failed gates. Every artifact fails on at least `literal_protocol_unique`, plus `beats_accepted_holdout_best`, `scientific_verdict_promote`, `selector_approved`, and `explicit_run_card_promotion_evidence`. Notable holdout measurements (local, NOT organizer scores):

| Lane | HOLDOUT-DTI | vs incumbent 0.152718 |
|------|------------|----------------------|
| H77-Step (archived, historical 'promote' superseded) | 0.152718 [0.131548, 0.175185] | repo best (not cleared) |
| H77D (Zero-SGMC, historical 'promote' superseded) | 0.15011 (s30) / 0.14030 (s20) | below best |
| H77-TSD | 0.13602 (s30) / 0.13371 (s20) | below best |
| H77E-Knick | 0.126664 (s20) / 0.124727 (s30) | below best |
| H77V | 0.12704 (s30) / 0.12931 (s20) | below best |
| H77E (UCP-Euler) | 0.1305 (s20) / 0.1342 (s30) | below best, duplicate logged |
| H80-OTA | 0.1039 (s30) | below best, not significant |
| H79-ORIENT | 0.094507 (s30) / 0.089063 (s20) | below best |
| H80 (Riedel) | 0.09580 | below best |
| H77-XSEN | 0.0994 | below best, 93.4% dots within 3px of GEMSDOE37 (duplicate) |
| H77G | 0.0619 (s20) / 0.0618 (s30) | below best |
| H82-TCSTEP | 0.0688 (s20) / 0.0700 (s30) | below best, physics below random |
| H81 | 0.067071 | below best |
| H82-RRD | 0.0686 | below best |
| H79-RH | 0.04773 | below best |
| H77BGB | 0.058406 (lane) / 0.102390 (LOCO) | below best |
| H82 (cool-outlet) | 0.012184 | below random control |
| H82R | 0.023067 | below shuffled control |
| H82-TEXTURE | 0.005969 | below constant-output control |
| D77 | 0.0000 | negative |
| H80T | 0.006030 | below best |

## 5. Ranked research hypotheses (untried, no experiment authorized)

From `docs/review-2026-10-21-60aafd45.md`, conditional only — none trained or validated, preflight stops pipeline first:

| Rank | Hypothesis | Layers / signature | Why it may find uncatalogued faults | Mimic | Viability this session |
|------|-----------|-------------------|-----------------------------------|-------|----------------------|
| 1 | Sentinel-1 InSAR inter-time-series deformation gradients | Ascending/descending LOS velocity, coherence, uncertainty | Persistent narrow velocity gradient or decorrelation boundary exposes creeping/pressurized blind structure with no scarp | Groundwater pumping, seasonal aquifer compaction, volcanic deformation | **Not viable** — Copernicus Data Space Ecosystem host outside sandbox allowlist; AOI coverage unverified |
| 2 | Microseismic cluster lineaments | USGS ComCat event locations, depths, uncertainties, focal mechanisms | Coherent depth-consistent event plane between mapped segments indicates active blind fault | Production-induced seismicity, quarry blasts, aftershock clusters | **Not viable** — USGS ComCat host unreachable from sandbox (IR-81-06) |
| 3 | Persistent Landsat thermal anomaly edges | Landsat C2 L2 surface temperature, cloud masks, acquisition time, elevation/aspect | Repeatable compact thermal anomaly/gradient localizes unmapped hydrothermal upflow | Irrigated fields, warm springs, wildfires, urban heat, cloud artifacts | **Not viable** — USGS host outside allowlist |
| 4 | Paired tributary planform deflection | USGS 3DEP 1-m DEM flow directions, channel centerlines, profile residuals | Repeated channel deflections indicate distributed deformation where no scarp preserved | Resistant strata, fan incision, lake shorelines, engineered drainage | **Deprioritized** — high collision risk with existing DEM/scarp work |

## 6. Limitations

1. **No DrivenData authentication** — training_features.tif, labels.tif, sample_submission.tif, and 1m DEM links are login-walled; only owner-supplied GitHub mirrors are available (integrity-pinned, NOT organizer-authenticated)
2. **Universal overlap blocker** — prevents any nonempty candidate under the current literal protocol; requires shared protocol/registry decision to resolve
3. **No selector approval** — 0 files authorized; weekly slot usage requires separate selector decision
4. **No organizer-confirmed score** for any GEMSDOE77 artifact; all HOLDOUT-DTI values are local spatial holdout measurements, not leaderboard projections
5. **H33 0.2778 attribution is unverified** — no submission-page receipt ties H33 bytes to that score
6. **sklearn not installed** in sandbox — ML-dependent tests (test_gems77_suite.py) cannot run; run_experiment.py cannot execute

## 7. Next session (in order)

1. Resolve dense-witness/uniqueness semantics with the protocol owner; do not waive locally
2. Restore hash-verified source data only into ignored data/cache paths; regenerate truthful restore receipt
3. Select one explicitly assigned method lane after overlap screening; freeze hypothesis, controls, canaries and stopping rules before holdout evaluation
4. Run single-feature leakage canaries, then buffered whole-segment hide-and-recover; compare to rebuilt incumbent on identical folds
5. Only after required holdout improvement: scan surface and final dots, write with shared writer, validate grid/finite [0,1], bind receipts to output hash
6. Obtain explicit selector authorization tied to exact artifact SHA; check submission allowance on the official page
7. If selected later, retain the organizer's submission-page receipt with artifact hash

## Core values applied

- **Maximize P(Win)**: Stopping rather than producing a misleading submission that cannot pass required gates
- **Own the Outcome**: Documenting the blocker completely rather than leaving ambiguity about download/submit status