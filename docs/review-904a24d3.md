# Current review — 2026-10-21 — branch 904a24d3

## Executive decision: download for inspection; NOT OK TO SUBMIT

**No new TIFF was generated in this review. No experiment or competition slot was used.**
A new protocol-compliant submission is blocked, not delivered. Download the
[existing H77L research TIFF](downloads/gems77-h77l-thk-flank-6b24305363fa-RESEARCH-ONLY.tif)
for inspection only. Do not upload it. See the [submission guide](executive-summary.html)
and [per-file authorization receipt](data/submission-authorization.json).
This review does not rename, copy, perturb or relabel an old submission as new.

The prompt does not identify a single new method lane. Historical lanes are stopped.
The pinned dense registry witness is positive at every eligible footprint pixel;
any nonempty eligible dot set therefore has forward overlap exactly 1 with it.
The literal stop rule blocks generation regardless of the proposed detector.
Data placement is **not** the only blocker. This review restored the pinned sample,
catalogue and four-channel radiometric cache autonomously and ran preparation;
that small cache is not the entire numerical/LiDAR feature stack.

## Pass 1 — implement and verify

- Read the standing prompt, README, agent instructions, prior release review,
  registry pins and scientific/release code. Baseline: **124 passing software tests**.
- Found an incorrect generalization in `research/why-h33-scored-02778.md`:
  a new dot's truth-credit increment need not equal its nearest-truth kernel
  credit. Corrected the algebra, removed unsupported causal/performance claims,
  and added analytical and randomized checks against the unchanged shared metric.
- Found that malformed nested guardrail objects could crash the release gate;
  non-finite overlap values could evade comparisons. Added fail-closed type,
  finite-number, nonnegative-count and overlap-range checks with regressions.
  This aggregator still trusts recorded evidence: it is not a scientific validator.

## Pass 2 — independent review and artifact checks

Restored the two SHA-pinned registry witnesses through GitHub. Independently ran
`scripts/verify_h77l.py`: **23/23 software/artifact checks pass**. These include
all-finite values in [0,1], exact reference CRS/shape/transform, no NaN inside
footprint, hash/ZIP identity and unchanged template hashes. The existing H77L
raster differs in pixels from 35 other local TIFFs and H33, but this is **not**
protocol uniqueness. Forward overlap with the dense rejection witness is 1.0.
These are artifact measurements, not HOLDOUT-DTI or organizer scores.

A further prose error exists in the pinned shared metric: it says deltaT<=q.
One dot can cover multiple truth pixels, disproving that bound. Its numerical
implementation remains correct in the new regression cases. Shared code was
not forked or changed; this prose issue should be corrected upstream in the
shared template before repinning. No local evaluator replacement was introduced.

## Pass 3 — requirements reconciliation

Regenerated the site/authorization receipt and ran `verify_all.py`. The final
suite passes **129 software tests**, including the new algebra and gate regressions.
Pages staging was also checked before merge. The first site check caught a missing
standard approval footer on the new report; rendering it through the common
page builder fixes the issue and keeps its status and navigation consistent. Automated checks cannot prove that
all historical narrative claims or live competition requirements are accurate.
The original long brief and operative requirements already appear in README and
PROMPT; the new review is linked at the top so historical banners cannot be
mistaken for current approval. No fresh score, universal novelty claim or
leaderboard improvement is asserted.

## Why H33 may have helped — and what remains unknown

Sparse emission can reduce redundant false-positive mass while retaining useful
kernel coverage. Pruning may help if removed dots contribute little truth credit;
being near a catalogue trace alone does not prove that. No inspected organizer
submission receipt ties the owner-reported H33 result to its bytes. No controlled
ablation here proves why it performed as reported. A better result is possible
in principle but cannot be predicted from this family score list. See the
[corrected analysis](https://github.com/buffedlizard55-lab/GEMSDOE77/blob/main/research/why-h33-scored-02778.md).

## Conditional hypothesis queue — not experiments or claims of global novelty

All expected HOLDOUT-DTI improvements are **unknown**. Ranking below is scientific
priority relative to cost, not a score projection. No idea is authorized or
obtainability-certified; all require lane assignment and a protocol resolution.

| Priority | Layers / physical signature | Why potentially off-catalogue; difference to reviewed implementations | Non-fault mimic | Cost / access |
|---|---|---|---|---|
| 1 | Magnetic and gravity fields: local spectral coherence with depth-dependent phase consistency | Buried shared boundaries rather than H77G troughs or H79 gradient orientation; novelty must be checked against H77E upward continuation | Intrusive contact, survey stitching | Medium; raw layer availability must be checked beyond restored radiometric cache |
| 2 | Conditioned USGS 3DEP DEM: paired cross-channel drainage-divide migration at common elevation | Deformation expressed across a drainage network rather than H77E-Knick pixelwise knickpoints; could expose subtle mapped-scarp omissions | Stream capture, lithologic contrasts | High; official 3DEP AOI obtainability not verified here |
| 3 | USGS ComCat event depths, origin times and uncertainty: persistent planes after declustering | Blind active structures, rather than existing coarse seismic density or up-dip projections | Quarry blasts, volcanic swarms | High; official service outside allowed hosts; not viable yet |
| 4 | Sentinel-1 displacement time series plus water-level covariates: persistent differential motion after seasonal removal | Active concealed deformation rather than static edge detectors | Groundwater pumping, atmospheric delay | Very high; AOI series and covariates not obtained; not viable yet |

## Source ledger and limitations

| Source | Verified in this review | What it supports / does not support |
|---|---|---|
| [Organizer reference README](https://github.com/drivendataorg/gems-prize-reference-solution/blob/aebe92f7c8a990f0e3443451b7a825d9afd6336b/README.md) | Retrieved via GitHub API at pinned commit | Explicit CPU-only and GPU environments; does not prove our candidate's performance |
| [Witness source manifest](https://github.com/buffedlizard55-lab/GEMSDOE77/blob/main/registry/witness_sources.json) | Both hashes reproduced | Owner mirror integrity, not official data provenance or organizer receipts |
| [Competition problem](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) / [data](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) / [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) | Not freshly accessed | Manual-review anchors only; no current rank/score/weekly cap certification |
| [NLR submission document](https://docs.nlr.gov/docs/fy26osti/96647.pdf) | Not freshly accessed | No claim of full rules review |
| [USGS 3DEP](https://www.usgs.gov/3d-elevation-program), [ComCat](https://earthquake.usgs.gov/fdsnws/event/1/), [Copernicus Data Space](https://dataspace.copernicus.eu/) | Not freshly accessed | Conditional official data routes, not verified AOI availability |

Outbound hosts are restricted to GitHub and package registries. No DrivenData
session or receipt is available. The supplied score list is owner testimony, not
organizer confirmation. Fault segmentation is not direct geothermal-vent detection.

## Next session, in order

1. Resolve dense-raster versus dot overlap semantics in the **shared** protocol;
   do not add a private exemption. Current user rule remains binding.
2. Assign one method lane, check novelty against implemented features, restore its
   pinned inputs, and verify the full current registry and fold support.
3. Correct the shared metric prose upstream and repin after review (numerics unchanged).
4. Run feature canaries and at most three preregistered experiments on matched
   buffered segment holdouts; require improvement over a comparable incumbent.
5. Only then write/validate a unique TIFF, run both registry gates, and obtain a
   separate selector decision. Store any later organizer receipt against its hash.
