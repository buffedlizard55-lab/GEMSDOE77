# Knowledge note (2026-10-21): RTP–TMI remanence/rotation discordance — tested and falsified as a standalone detector (H82)

**Idea.** The shipped stack contains both total magnetic intensity (`tmi`, band 14)
and its reduction-to-pole (`rtp`, band 2) [OFFICIAL feature list:
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/].
RTP assumes induced magnetization. Blocks with remanent magnetization or
vertical-axis rotation (Walker Lane-style transtension, the INGENIOUS study
setting: https://gbcge.org/current-projects/ingenious/) violate that assumption
locally, so a Gaussian-weighted local regression RTP_hp ≈ a·TMI_hp + b should
break — slope/correlation/residual steps — at block-bounding faults, including
faults with weak susceptibility contrast and no surface scarp under alluvium.
GeoDAWN is the underlying airborne survey (USGS, DOI 10.5066/P93LGLVQ:
https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and).

**Measured facts (this repo, SHA-pinned bytes).** Global Spearman(RTP, TMI) =
0.689 on a 200k footprint sample — the fields are far from trivially identical,
so discordance has real variance. Each of the 8 discordance channels alone is
weak on withheld catalogue faults (best symmetric AUC 0.533). On the frozen
seed-20 hide-and-recover holdout (`gems57-pooled-hide-v2`, 11,321 withheld):
pure discordance surface 0.0419 HOLDOUT-DTI (below random 0.0471); HGB on RRD
channels alone 0.0490; adding RRD to the 19 raw bands moved HGB 0.0618 →
0.0686, paired Δ +0.0068 with 95% CI [−0.0036, +0.0163] (not significant).

**Interpretation.** Either (a) expert-mappable catalogue-style faults in this
footprint rarely coincide with RTP–TMI discordance breaks at 250–300 m scales,
or (b) the shipped RTP/TMI grids are too smoothed/levelled for the local
regression to isolate block boundaries at metric scale, or (c) discordance is
dominated by volcanic-unit remanence patchwork (the named mimic) rather than
fault-bounded blocks. The catalogue holdout cannot distinguish (a) from a
live-only population of discordance-only faults — but under the parallel-run
protocol an idea that cannot beat the holdout incumbent (0.152718) must not
spend a slot. **Do not retune H82** (scales, windows, blends) without a new
preregistration and a new physical argument; the cheap variants are exactly
what the canary + paired-CI design already forecloses.

**Reusable assets.** `src/gems82/fields.py` (label-free local-regression
moments, unit-tested), the fixed-operating-point holdout harness
`scripts/h82_pipeline.py` (no budget scan), and IR-82-01/02/03 in
`evidence/h82_run_card.json`.
