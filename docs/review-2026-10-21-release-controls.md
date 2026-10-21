# Release-control and source-inventory review — 2026-10-21

## Decision in one line

**No TIFF in this repository is OK TO SUBMIT.** All twelve listed GeoTIFFs are
still easy to download **for inspection only** from the [site start page](index.html),
and the exact per-file machine decision is
[`NOT OK TO SUBMIT`](data/submission-authorization.json). This review generated
no new prediction raster, ran no model experiment, placed no dots, uploaded
nothing, and spent **zero** competition slots.

This is a release-control review, not a new geological lane. The supplied brief
still contains no identifiable single-method paragraph, and the current
[H77L handoff](review-2026-10-21-h77l.md) forbids re-tuning stopped negative
lanes. Creating a new “unique” TIFF without an assigned lane, full registry
clearance, and a matched holdout improvement would contradict the parallel-run
protocol rather than maximize the probability of a valid result.

## What was checked, line by line

1. **Exact downloadable bytes, not manifest claims.**
   `scripts/release_gate.py` now invokes the pinned shared
   `vendor/gems57.validate` on every listed TIFF using the tracked
   `data/bridge` grid, then records the complete local validator receipt. The
   receipt says all **12 / 12** pass: one float32 band; EPSG:32611; 3730×3292;
   matching transform; finite cells in the footprint; values in `[0, 1]`; and
   zero outside the bridge footprint. The earlier portal error remains
   unexplained: **local format PASS is not an organizer acceptance receipt**.

2. **Hashes and portal metadata.**
   Every 12 / 12 manifest SHA-256 values matches the re-read file. Every
   artifact now has a nonempty distinct reference name, a reference note no
   longer than 140 characters, and a frozen run-card path. The release gate
   extracts the historical card spelling where needed and refuses approval if
   the manifest name or note differs. The D77 UI note had been 142 characters
   after a status suffix; it now uses the frozen 124-character reference note.
   Its **NOT OK TO SUBMIT** status remains separate and prominent.

3. **No silent gate relaxation.**
   The gate now requires an *explicit* complete-registry certificate, an
   *explicit* promotion-valid holdout declaration, and recorded Rule-1 surface
   rank plus final-dot overlap measurements in the frozen run card. It applies
   the stated `|Spearman rho| > 0.90` and forward 3-pixel overlap `> 0.70`
   stops directly, including negative rho values and fields whose names include
   `dots`. A local raster count or `complete_accessible_scan` diagnostic is not
   treated as a full registry certificate. Missing run cards, malformed TIFFs,
   absent metadata, historical `promote` fields, and missing selector approvals
   all fail closed. All current artifacts fail literal uniqueness; all retain
   `NOT OK TO SUBMIT`.

4. **The literal blocker remains.**
   This review did not recompute a registry scan. It preserves the prior,
   independently recorded rejection: a dense witness has positive support over
   all allowed cells, so a nonempty candidate has forward 3-pixel overlap 1.0
   under the literal inherited support rule (limit 0.70). The policy owner—not
   this branch—must define a shared representation rule before any candidate
   can be cleared. See the [H77L run card](evidence/h77l_run_card.json) and
   [registry saturation certificate](evidence/registry_saturation.json).

5. **Current owner-mirror feed.**
   The scheduled source script no longer ends at a hand-maintained `GEMSDOE78`
   list. It discovers all public owner repositories whose names match the
   historical family spellings (`GEMSDOE`, `GEMSDOE2`, `5GEMSDOE`, etc.) through
   the GitHub API and fails rather than publish a partial discovery result. At
   this review timestamp it found and retrieved page inventories for **92**
   public family repositories, including GEMSDOE79–82. This is a GitHub
   owner-mirror / page-freshness inventory—not a score feed, a full raster
   registry, an organizer receipt, or proof that a reported score belongs to a
   particular TIFF. See [source-health.json](source-health.json).

## H33 / 0.2778: bounded scientific conclusion

The strongest defensible explanation remains a **raster/metric mechanism, not a
verified submission attribution**. The pinned earlier forensic receipt finds
that H33 contains 37,654 positive pixels and is the 44,090-pixel D2.8 support
with 6,436 cells at mapped-catalogue distance at most two pixels removed; it
does not add new cells. Under the stated 300 m triangular DTI and its asymmetric
false-positive penalty, sparse dots separated at approximately a kernel radius
and pruning redundant mapped-fault shoulders could plausibly reduce false
prediction mass while retaining credit for unseen labels. That is why this
construction can be scientifically plausible.

It does **not** establish that these exact bytes received the owner's reported
`0.2778`, that the pruning caused the number, or that reusing the construction
would beat it. The family file-to-score attribution is **OWNER-REPORT** absent
a file/hash-linked organizer receipt. The public leaderboard row cannot identify
a TIFF by itself. See the [forensic receipt](evidence/prior_forensics.json),
[long-form H33 analysis](lia-analysis.html), and [source-boundary notes](sources.html).
No projection is called a score here.

## Hypotheses and next action

The required pre-experiment hypothesis catalog already contains four ranked,
explicitly limited proposals in [the current session handoff](SESSION_REVIEW.md)
and the frozen [H77L preregistration](evidence/preregistration_h77l.json).
They cover multi-scale potential-field source-depth concordance, drainage
knickzone topology, shallow seismicity alignment, and InSAR deformation
coherence. They are **not** new viable experiments in this session: the latter
three need external official data whose availability cannot be verified from
this restricted sandbox, while the first needs restored/authenticated inputs
and an assigned lane.

Before any new TIFF is generated, in order:

1. Obtain one explicit narrow lane assignment and a shared, protocol-owner
   decision on continuous-versus-dot registry representation.
2. Authenticate the original competition grid/data and obtain the applicable
   complete registry; do not replace either with a mirror hash.
3. Build a matched, fold-rebuilt incumbent with whole-segment buffered folds
   and visible-fault-only catalogue features.
4. Preregister one genuinely distinct hypothesis; run each feature-alone
   leakage canary, then a matched HOLDOUT-DTI comparison with CI.
5. Run full surface and final-dot registry checks and the shared writer/
   validator before a separate selector considers one weekly slot.

Official manual-review links retained by the project: [competition problem and
metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/),
[leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/),
[reference solution](https://github.com/drivendataorg/gems-prize-reference-solution),
[GeoDAWN](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and),
and [INGENIOUS GDR](https://gdr.openei.org/submissions/1391). These links are
provided for human review; this session did not assert a new live read from
non-allowlisted hosts.

## Run card

The machine-readable card is
[`evidence/release_controls_2026-10-21.json`](../evidence/release_controls_2026-10-21.json).
It labels the absent scientific evaluation as `NOT RUN`, records the 12 exact
local format checks, the 92-repository owner-mirror inventory, and its
**negative** release verdict.
