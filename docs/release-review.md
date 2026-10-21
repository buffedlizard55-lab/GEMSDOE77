# Release safety review — 2026-10-21

## Download versus submit

**OK to download for inspection; NOT OK TO SUBMIT.**

[Download the existing H77L research TIFF](downloads/gems77-h77l-thk-flank-6b24305363fa-RESEARCH-ONLY.tif),
[one-TIFF ZIP](downloads/gems77-h77l-thk-flank-6b24305363fa-RESEARCH-ONLY.zip),
[submission instructions](executive-summary.html),
[current authorization receipt](data/submission-authorization.json).
This review did **not** generate a new TIFF, rename a prior submission as new, run an experiment, or spend a competition slot.
The request to generate a new submission cannot currently be satisfied together with the mandatory duplicate-stop rule.

## Three passes

### Pass 1 — inspection and implementation

Read the standing README/prompt, agent instructions, irregularities, preregistration, frozen run cards and prior reviews.
The baseline test suite passed (95 tests; software checks, not performance scores).
Found a concrete authorization defect: a manifest with every flag true could pass without any run card;
a negative card with no recognized duplicate field could also be overridden by the manifest's promote flag.
Fixed the repository release gate (not the shared scientific evaluator): require a readable JSON-object card and matching promote verdict.
Unreadable/invalid-JSON/non-object cards fail closed. Added missing-card, negative/missing/mistyped verdict,
malformed-card and consistent-promotion/selector regression tests. This decision aggregator still trusts recorded format,
holdout and selector evidence; it is not an independent scientific-certification system.

### Pass 2 — independent artifact and blocker verification

Restored the two pinned registry witnesses over GitHub and checked SHA-256. Re-ran `scripts/verify_h77l.py`:
all 23 checks pass, including exact grid, finite/range readback, one-TIFF ZIP equality, pinned shared-template hashes,
and pixel distinction from 18 other local TIFFs. See [the updated audit](evidence/h77l_independent_audit.json).
Those are format/integrity measurements, **not** holdout re-evaluation or complete-registry clearance.

The dense witness has positive values at every allowed footprint pixel. The nonempty H77L surface has measured
forward overlap 1.0, exceeding the protocol's 0.70 limit. This is an independent rejection witness;
a full registry scan is unnecessary to prove failure, though it would be necessary for a pass.
For any nonempty allowed dot set A and witness support W containing the allowed domain,
A is a subset of W, hence also of its 3-pixel dilation: overlap is exactly |A|/|A|.
Changing geological features, dot count, model or file name cannot resolve that literal gate.
No final-dot placement or fresh lane was attempted. The shared evaluator/writer were not changed.

### Pass 3 — end-to-end release requirements

The expanded suite passes (99 tests; software checks). The regenerated authorization receipt authorizes zero files.
Entry and executive-summary pages retain direct downloads and explicit NOT OK TO SUBMIT status.
No TIFF bytes were changed. The featured file hash remains
`9d199f9fad93e234397a7a19a05eabf12fd77285be3007e6afe22d12a36dcbd1`.
The review preserves frozen scientific run cards rather than rebranding historical results as new experiments.

## What explains the reported H33 result?

[Prior byte forensics](evidence/prior_forensics.json) and the retained analysis support a plausible explanation:
sparse predictions reduce redundant mass, and pruning predictions near known faults avoids spending mass on
catalogue-excluded locations. H33 retains 37,654 positive pixels and removes near-catalogue dots from an earlier backbone.
This is not causal proof from a controlled organizer experiment. The supplied **0.2778 is OWNER-REPORT for the named file**;
this review obtained no submission-page receipt tying that score to those bytes. The supplied leaderboard targets
also conflict. No current leaderboard score, improvement forecast, or likelihood of winning is certified here.
Hiding catalogue segments is a useful proxy but not identical to discovering truly unmapped faults or geothermal vents.

## Ranked hypothesis backlog — conditional, not newly validated

Expected DTI improvement is **unknown** for every item. Ranking is qualitative scientific priority/cost, not a score projection.
These are retained proposals from the [earlier audit](SESSION_REVIEW.md), not claims of global novelty:

| Rank | Layers and physical signature | Why potentially uncatalogued / difference | Non-fault mimic | Cost and viability |
|---|---|---|---|---|
| 1 | Magnetic/gravity derivatives plus basement depth; multiscale phase/contact persistence with compatible depth step | Buried basin-fill offsets; source-depth consistency rather than a single contact ridge. Existing H77E continuation work makes novelty review essential. | Intrusive/lithologic contacts; flight-line seams | Medium; competition/cached inputs in principle; blocked by assignment and duplicate gate |
| 2 | Conditioned USGS 3DEP DEM; coherent drainage deflection and knickzones across adjacent channels | Network-scale deformation rather than pixelwise scarp magnitude | Fan incision, resistant strata, engineered drainage | High; no new DEM obtained; official availability for the AOI not checked, therefore not viable yet |
| 3 | USGS ComCat locations/depth/uncertainties; robust seismic event-plane alignment | Blind active structures rather than existing coarse seismic aggregates | Quarry blasts, volcanic swarms, aftershock sequences | High; event service not reachable through this sandbox's allowed hosts; not viable yet |
| 4 | Sentinel-1 time-series gradients and coherence | Present-day strain rather than static terrain/geophysical edges | Pumping, mining, atmosphere and seasonal hydrology | Very high; AOI products/access not verified; not viable yet |

No new-data proposal above is presented as obtainable or submission-ready. Do not rerun closed lanes or interpret this table
as authorization. A protocol-owner decision and explicit lane assignment must precede fresh preregistration and validation.

## Sources and access boundaries

- [Organizer reference solution](https://github.com/drivendataorg/gems-prize-reference-solution/tree/aebe92f7c8a990f0e3443451b7a825d9afd6336b): README freshly retrieved via GitHub API. It explicitly supports CPU-only as well as GPU environments; GPU is helpful, not an absolute requirement of that reference. This does not validate our model or input provenance.
- [Competition problem](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/), [data](https://www.drivendata.org/competitions/306/competition-doe-gems/data/), [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/), [NLR submission document](https://docs.nlr.gov/docs/fy26osti/96647.pdf): manual-review links, not freshly verified in this review. No DrivenData authentication/receipt is available here.
- [USGS 3DEP](https://www.usgs.gov/3d-elevation-program), [ComCat](https://earthquake.usgs.gov/fdsnws/event/1/query), [Copernicus](https://dataspace.copernicus.eu/): specific official routes for the conditional proposals; no obtainability claim.
- [Registry witness pins](https://github.com/buffedlizard55-lab/GEMSDOE77/blob/arena/49d64450-gemsdoe77/registry/witness_sources.json): owner-repository mirror provenance, not organizer authentication.

Outbound research in this review was limited to GitHub and allowed package registries. No claim of fresh deep literature
verification, complete cross-repository novelty, or live portal acceptance is made. Existing numerical HOLDOUT-DTI records
retain their evaluator version, withheld-positive count and confidence intervals in their original run cards.

## Next work, in order

1. Protocol owner resolves dense-versus-sparse overlap semantics in the shared template/registry; no local exemption.
2. Assign one genuinely distinct method lane; confirm inputs and full current registry provenance.
3. Preregister at most three experiments, run single-feature leakage canaries and matched buffered segment holdout;
   require improvement over the comparable incumbent before any slot selection.
4. Generate through the shared writer; validate exact grid/finite [0,1], scan surface and final dots, then seek separate selector approval.
5. Keep an authenticated organizer receipt tied to the exact artifact hash if a slot is eventually used.
