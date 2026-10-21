# Repository review — release safety, not a new model experiment

**OK to download for inspection; NOT OK to submit.**
[Start page and direct TIFF links](index.html) · [Submission instructions](executive-summary.html) ·
[Authorization receipt](data/submission-authorization.json).

## What was verified this session

- Read the repository instructions, operative prompt, README, preregistration,
  frozen lane cards and irregularity log before making changes. The full original
  request is already retained in the README; it was not replaced with a summary.
- Independently read all **13 primary manifest TIFFs** using Rasterio, including
  raw, unmasked raster blocks. All match the bridge sample CRS, shape and affine
  transform; all are single-band, have no nodata metadata, have finite raw values
  everywhere and values in [0,1], and match their manifest SHA-256.
  These are **LOCAL FORMAT / INTEGRITY measurements, not scores**.
  Reproduce with `python scripts/audit_published_tiffs.py`; the detailed receipt
  is `evidence/review_primary_tiff_audit.json` in the GitHub repository.
  Auxiliary NaN-outside TIFFs are not included in this primary-file audit.
- All **23 pinned shared-template file hashes** match `registry/template_pin.json`.
  Neither the evaluator nor writer was forked or modified.
- The existing independent `scripts/verify_all.py` completed without failures.
  It checks one historical artifact, ZIP, spacing and metric properties, not all
  lanes' scientific validity.

## Pass 1 — implement the concrete safety fix

The release gate treated an absent run card as `guardrail_clear=true`. A row
claiming format PASS, uniqueness, improvement and selector approval could then
be authorized without a run card. Missing or malformed/non-object JSON now fails
closed. A manifest `promote` also cannot override a negative or missing run-card
verdict. The regenerated authorization receipt still authorizes **zero files**.
New regression tests exercise these false-promotion cases.

## Pass 2 — scientific and edge-case review

Corrected `research/why-h33-scored-02778.md`: adding a dot does **not** generally
increase the DTI denominator by a fixed 0.2. Its marginal credit sums over all
truth pixels, whereas its self-credit is the maximum over nearby truth pixels.
Executable counterexamples cover both redundant dots and a dot benefiting
multiple truth pixels. The simplified threshold only applies when these credits
happen to be equal. The pinned evaluator computes the components correctly;
its explanatory docstring also contains the incorrect `dT <= k` assertion.
That shared **prose defect is reported here for upstream correction**, not patched
in a private template fork. No scores or old experiment outputs were changed.

Also removed a README guarantee that a portal range error “cannot recur.” Local
validity is not an organizer receipt or evidence of portal acceptance.

## Pass 3 — request reconciliation and limits

No new geological experiment, TIFF or competition upload was performed in this
review. **The request for a new protocol-unique, submission-ready TIFF remains
unfulfilled.** Generating another file and giving it a new hash would not satisfy
that request. Repository instructions explicitly close failed lanes. No single
new assigned method paragraph was supplied. Existing evidence records a dense
registry witness covering every allowed pixel, which makes the literal 70%
forward-overlap gate impossible for any nonempty candidate in that domain.
This review verified local files and pinned code, not a new download of that
external witness. No exemption, density threshold or gate relaxation was made.

The owner-reported **0.2778 is not ORGANIZER-CONFIRMED** in inspected receipts.
Thinning can reduce redundant emitted mass while retaining kernel coverage;
that plausibly explains improvement over solid ribbons. It does not identify
better geology, establish causation for this particular score or predict a gain.
The current live leaderboard and weekly cap were **not verified this session**.
The prompt supplies conflicting leaderboard-era values; none is adopted here
as current truth. No numeric leaderboard improvement is projected.

### Prioritized research backlog — not viable/approved experiments yet

Expected DTI gains are **unknown**. Priority is scientific diagnostic value per
cost, not an invented expected score. These are deferred research questions;
novelty against every sibling implementation has not been established.

| Priority | Layers and physical signature | Why possibly off-catalogue; distinction to check | Named mimic / cost |
|---|---|---|---|
| 1 | Cached DEM + Th/K/U: matched opposing-slope profiles; test whether a radiometric flank survives terrain-conditioned residualization | Would test transport/confounding rather than merely detecting another edge; must distinguish from stopped H77L and lagged-halo lanes before registration | Soil creep and lake shorelines; moderate |
| 2 | Cached magnetics + gravity: phase agreement of contact profiles across independent wavelength bands, conditional on topographic relief | Potentially buried discontinuities without a surface trace; must differ from H77E upward continuation and contact-physics | Intrusive/lithologic contacts; moderate/high |
| 3 | Cached DEM + radiometrics: repeatability of cross-valley break-in-slope evidence across several adjacent drainages | Repeated offsets could separate structure from a single erosional scarp; must distinguish from H77D/H77-Step and H79 cross-gradient methods | Differential erosion/stratigraphy; high |

All depend on restoring missing feature caches, a resolved literal gate, a
non-overlapping lane, single-feature canaries and a frozen matched holdout
comparison. No new external dataset is proposed as available or viable. Do not
spend a slot on any of these based on this table.

### Official-source audit and manual-review links

The organizer's [reference solution](https://github.com/drivendataorg/gems-prize-reference-solution/tree/aebe92f7c8a990f0e3443451b7a825d9afd6336b)
was retrieved via GitHub API in this session. Its README says to obtain data
from the competition website and documents CPU-only as well as GPU environments;
therefore a GPU is an acceleration option, **not a universal prerequisite**.
This does not prove this repository has all model inputs restored.

The following are official anchors supplied by the owner, **not newly retrieved
or verified in this network-restricted session**:

- [Competition problem and format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [Data access](https://www.drivendata.org/competitions/306/competition-doe-gems/data/)
- [Leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- [Rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf)
- [USGS GeoDAWN](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)

Outbound access is limited to GitHub and package registries. This checkout has
bridge grids and retained artifacts, not the full raw feature stack/caches.
No DrivenData session is connected; do not request or put credentials in chat.
The scheduled site refresh is a GitHub source inventory, **not a live competition
score feed**. A separately authorized selector must resolve the uniqueness policy,
restore SHA-pinned features, freeze comparable holdout baselines, obtain actual
submission receipts and check the weekly cap before a file can be promoted.

## Concurrent-merge reconciliation

PR #29 merged during this review and added H79 orientation persistence. The
latest main was merged into this session branch, the generated authorization
receipt was regenerated (not hand-resolved), and the primary audit was repeated
on all 13 files. All pass local format/integrity; none is authorized. The full
99-test suite passed again after integration. No inherited experiment was rerun.
