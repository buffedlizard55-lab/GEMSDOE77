# Project prompt (operative brief — re-read at the start of every session)

*Stored 2026-10-21 per the project owner's instruction ("Put this prompt into
the repo readme and read it every time we work on the project"). This file
preserves the operative instructions; the owner's long family score list is
summarised with provenance at the end rather than re-transcribed hash-by-hash
(the byte-verifiable ledger through 2026-10-03 lives in GEMSDOE32
`docs/score-ledger.csv`; later entries below are OWNER-REPORT as supplied in
the 2026-10-21 brief — no organizer receipts).*

## Current 05f9e6f8 review addendum — 2026-10-21

The full standing request is retained in README. This invocation again assigns no identifiable method paragraph. Fresh hash-pinned preflight stops all nonempty candidates under the literal registry overlap rule. No new geological experiment, TIFF or slot was used. The new review fixes contradictory recorded measurement handling in the release adapter; it does not relax the shared protocol. See `docs/review-2026-10-21-05f9e6f8.md`. GitHub Pages exists in legacy mode; switching to Actions publishing was denied by the integration's permissions.

## 2026-10-21 review addendum — arena/60aafd45-gemsdoe77

No new method paragraph is assigned in this invocation. The freshly rechecked dense registry witness blocks every nonempty candidate under the binding literal rule. No new TIFF, model experiment or slot is authorized by this review. See `docs/review-2026-10-21-60aafd45.md` and the corrected H33 analysis. Superseded earlier leaderboard observations below are historical, not current submission-page receipts.

## Current review addendum — 2026-10-21

The latest readiness review (`docs/preflight-review.md`) generated no new TIFF and ran no geological experiment.
A freshly pinned universal witness (`evidence/registry_preflight.json`) blocks every nonempty allowed candidate
under the mandatory literal overlap rule. H82 is the latest documented lane (`evidence/h82_run_card.json`):
negative HOLDOUT-DTI 0.012184, duplicate overlap 1.000, incomplete two-witness scan, and three experiments
used; do not resume it. H81 is also closed after 69 duplicate firings and an incomplete 832-raster scan.
The site shows H82 as the newest inspection-only download and H80 as the evidence-complete featured negative;
neither presentation is a promotion. Run the pinned preflight before future authorized research, preserve closed
lanes, and do not treat older execution addenda as a new lane assignment. No TIF may be uploaded and no
selector-approved slot exists. The original complete user request remains embedded in README.md. This addendum
does not relax its gates or authorize a slot.

## Historical 2026-10-21 H79 execution addendum

This session's registered lane is **H79-ORIENT**, not the historical contact-physics
lane mentioned in older repository notes. The operative preregistration is
`registry/preregistration_h79_orient.json`; the measured artifact and stop decision are
in `evidence/h79_orient_run_card.json`. The full request below remains preserved as the
standing brief. H79's result is negative, safe to download for inspection, not
approved to submit, and no weekly slot was selected.

## Highest urgency

MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. DO NOT COPY A
PREVIOUS SUBMISSION UNLESS IT'S FOR LEARNING AND EDUCATION. IT MUST BE OBVIOUS
WHETHER IT IS OK TO DOWNLOAD AND SUBMIT THE GENERATED TIF. There should be an
easy-to-download submission TIF as described by the prompt. Read the entire
prompt.

## Parallel-run protocol (read first; this session is one of several)

1. **Lane.** Your lane is the single method paragraph assigned to the session.
   Stay inside it. If your raster's rank-correlation with any registry raster
   exceeds 0.90, or more than 70% of your dots fall within 3 px of one registry
   raster's dots, you have drifted into another lane: log it as a duplicate
   and stop. Check on the surface before placement AND on the final dots.
   *(GEMSDOE77 note: the 2026-10-21 brief contained no identifiable single
   method paragraph, so the lane was declared in
   `registry/preregistration_contactphys.json` before Exp 1 — IR-77-06.)*
2. **Reuse, don't rebuild.** Use the template's cached feature stack,
   `evaluate_holdout.py` and `submission_writer.py`. Holdout = hide-and-recover:
   withhold whole fault segments with a buffer, derive every catalogue-based
   feature only from visible faults, mask visible faults pixel-exactly, score
   pooled DTI (α=0.2, β=0.8, 300 m triangular kernel). If a shared tool is
   wrong, fix it once in the template and report it; never keep a private fork.
3. **Label every number** as HOLDOUT-DTI (evaluator version, # withheld
   positives, 95% CI) or ORGANIZER-CONFIRMED (submission-page receipt). A
   projection is never written as a score.
4. **Leakage canary.** Test each feature alone on the holdout first. AUC > 0.90
   means leakage until proven otherwise.
5. **Run card.** End with one JSON card: hypothesis; mechanism; the named
   non-fault process that could mimic it; holdout DTI + CI; correlation/overlap
   vs registry; raster sha256; validator output (no NaN in footprint, values
   [0,1], CRS/shape/transform match); submission name + note ≤140 chars;
   verdict promote/negative. Negative results are deliverables.
6. **Budget.** Stop after 3 experiments or 2 hours. Do not pick submissions:
   promotion is a separate selector step within the weekly cap.

## Standing science instruction

Before implementing, generate 3–5 candidate geological hypotheses not yet
tried, each naming layer(s), physical signature, why it catches a fault missing
from the USGS/INGENIOUS catalogue, and how it differs from anything in the
repo. Rank by expected DTI improvement ÷ implementation cost. Validate the top
candidate on the spatially blocked holdout before touching a weekly slot — do
not spend a slot on an idea that hasn't beaten the current holdout best. If a
candidate needs new external data, name the specific free official source and
check obtainability before proposing it as viable. Work line by line from
official verified sources with links for manual review; flag irregularities;
no hallucinations.

## Core values (focal point for all decisions)

- **Maximize P(Win)** — weigh tradeoffs, assess risk, choose the path that
  maximizes the probability of success; set emotions aside.
- **Own the Outcome** — own results end to end; act without waiting for
  permission when we have the means; treat failure and success as signals.

## The question to answer every session

Why did `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` (GEMSDOE32) get the
family's highest reported score (0.2778, OWNER-REPORT), and can we generate a
unique submission scoring higher? Answer with PhD-level judgement, then use the
answer to build the lane's candidate. Verify line by line; no hallucinations.
(GEMSDOE77's answer: `research/why-h33-scored-02778.md`.)

## Competition anchors

- Overview/problem/metric: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Leaderboard: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/ (2026-10-21 read: #1 xiaofanhu **0.4413 ORGANIZER-CONFIRMED**; brief-era references 0.3195/0.3262 are superseded)
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution
- Rules PDF: https://docs.nlr.gov/docs/fy26osti/96647.pdf
- GeoDAWN: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
- INGENIOUS: https://gbcge.org/current-projects/ingenious/
- EPSG:32611: https://epsg.io/32611 · Tversky: https://en.wikipedia.org/wiki/Tversky_index
- Data (login-walled): https://www.drivendata.org/competitions/306/competition-doe-gems/data/
  — working sandbox route is the owner's SHA-pinned GitHub mirrors
  (`registry/data_manifest.json` + `scripts/restore_data.py`); Dropbox mirrors
  in the brief are unreachable from the sandbox.

## Submission mechanics (must stay obvious on the site)

Single-band float32 GeoTIFF (or ZIP with exactly one), same CRS/shape/
geotransform as the sample submission, values in [0, 1]. The earlier portal error
"Predicted values must be in range [0, 1]" had an unverified cause. The shared
fail-closed writer now produces an all-finite `[0,1]` readback with `nodata=None`;
that local check does not prove portal acceptance (no H77L upload/receipt). Every
candidate needs a unique name and a ≤140-char Note-field text. The site's
first screen must state download/submit status in one glance; the executive
summary explains exactly how to submit a *promoted* file.

Historical branch review `docs/review-2026-10-21-b6805ddb.md` has a pre-merge 29/29 inventory snapshot; consult its reconciliation section before citing current counts.

## Multi-pass working rule

Pass 1 implement + verify. Pass 2 review for bugs/assumptions/edge cases; fix.
Pass 3 re-check against the original request; improve; fix. Then open a PR and
merge to `main` (from the session branch only), with remaining-work and
limitation suggestions for the next session.

## Family score list (OWNER-REPORT; highlights only — full brief retained by owner)

Highest per site family as supplied 2026-10-21 (selection; all OWNER-REPORT):
GEMSDOE32 h33-2-b2 **0.2778** · GEMSDOE36 anderson-geothermal-pinn 0.2750 ·
GEMSDOE28/31 h27-4-solo 0.2708 · GEMSDOE33 analog-tip-stepover 0.2632 ·
GEMSDOE25/30 dotted-h19-5-d2.8 / poisson300m 0.2600 · GEMSDOE70
p5-dual-submod300-oof38k 0.2478 · GEMSDOE24 dotted d1.5 0.2477 · GEMSDOE49
gate_ortho 0.2376 · GEMSDOE63 cvarlw-thin 0.2303 · GEMSDOE48 h59-cover 0.2296 ·
19GEMSDOE h19-5 0.1922 · GEMSDOE10 h28-dotted-ridge 0.1839 · GEMSDOE 0.1563 ·
7GEMSDOE lidarscarp 0.1461 · GEMSDOE46 scarp-radiometric 0.1589 · all other
listed sites ≤0.1461 down to 0.0000. GEMSDOE77 (this repo): no organizer score;
verdict negative (see README).

*Post-merge namespacing note (PR #5): this lane's files that collided with concurrently merged lanes were renamed — `registry/preregistration_contactphys.json`, `evidence/run_card_contactphys.json`, `docs/contactphys*.html`, `scripts/build_site_contactphys.py`. Frozen receipts (`registry/preregistration_amendment_exp3.json`, `evidence/*.json`) retain their original internal path strings as historical record.*

## H82-TEXTURE historical execution addendum
One narrow texture-boundary feasibility experiment was completed before the parallel universal-preflight review merged. This is distinct from H82 chemistry-concordance. Results: `evidence/h82_texture/run_card.json`; review: `docs/review-h82-texture.md`. Negative, closed, no dots/slot, inspection-only TIFF. Preflight now blocks any new experiment; this addendum grants no exception.
