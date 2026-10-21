# H83 candidate hypotheses (written 2026-10-21, BEFORE any H83 experiment)

Session branch `arena/e8283608-gemsdoe77`. Evidence classes: every number in this
file is either a RASTER MEASUREMENT of bytes in this repo, an INFERENCE (arithmetic
on owner-reported scores, labelled as such), or a citation. No HOLDOUT-DTI exists
for any H83 idea at the time of writing.

## 0. What the H33 lineage tells us about the hidden truth (INFERENCE, not a score)

Owner-reported (no organizer receipt in this repo): D2.8 dotted h19-5 = 0.2600,
H33 = 0.2778. Retained byte forensics (`evidence/prior_forensics.json`): H33 = the
44,090-dot D2.8 raster minus 6,436 dots lying within 2 px of the mapped catalogue,
nothing added. If the removed dots earned ~zero truth credit (the H33 rationale),
then with `s = T/D`, removing them lowers `D` by `0.2 × 6,436 = 1,287.2` while `T`
is unchanged, so `D = 1,287.2 × 0.2778 / (0.2778 − 0.2600) ≈ 20,089` and
`T ≈ 0.2600 × 20,089 ≈ 5,223`. With `D = 0.2(T + S − M) + 0.8K` and `S = 44,090`:
`0.8K = 20,089 − 0.2(5,223 + 44,090 − M) = 10,226 + 0.2M`. For 2.8-px-spaced dots on
thin traces each dot's own credit `M`-share is ≤ 1 while it credits up to ~3 truth
pixels, so `T/3 ≲ M ≲ T`, i.e. `1,741 ≲ M ≲ 5,223`, giving a hidden-truth count
**`K ≈ 13,200–14,100` pixels — about 22–23 % of the 60,988 mapped pixels** and an
implied H33 truth coverage `T/K ≈ 0.37–0.40`.

Byte check performed this session (RASTER MEASUREMENT): the restored D2.8 mirror
(`gems24-h25-1-dotted-h19-5-d2-8…-nan.tif`, 44,090 binary dots) contains every one of
H33's 37,654 dots; the 6,436 removed dots all lie 1.0–2.0 px from the catalogue and
H33's nearest dot is √5 px away; H33's median dot-to-catalogue distance is 19.6 px.

Interpretation (hypothesis, not proof): the leaderboard truth behaves like a
~20 % withheld subset of the *same* style of fault catalogue, i.e. hidden faults are
statistically similar to mapped ones and are spatially organised by the same fault
network. That is precisely what the shared hide-and-recover holdout emulates
(`hide_frac = 0.20`), and it argues that **visible-network geometry** (where the
next fault of the network should be) is the strongest available prior, modulated by
geophysics/topography. The assumption "removed dots earned zero credit" is
untestable without organizer receipts; if they earned credit the arithmetic above does not hold and `K` is not identified.

## 1. Ranked candidates (none previously implemented in this exact form)

| Rank | Hypothesis | Layers | Physical signature / transform | Why it targets faults *missing* from the catalogue | Difference from existing lanes | Expected DTI gain / cost |
|---|---|---|---|---|---|---|
| 1 | **H83-NETSIB — sibling & gap geometry of the visible fault network, learned by synthetic re-withholding** | visible catalogue (holdout-visible only) + 19 core bands + 3DEP lidar scarp u8 (11 bands) + GeoDAWN radiometric u8 (8 bands) | Structure-tensor strike field of visible traces, binned into 12 strike classes; per class FFT-convolution with oriented kernels: (a) along-strike continuation, (b) **bilateral gap closure** `sqrt(F·B)` from forward/backward half-kernels, (c) **parallel-sibling bands** at 0.4–2.0 km perpendicular offset (characteristic Basin-and-Range normal-fault spacing), (d) multi-scale network density and log distance. A model is trained on *synthetic* withholdings of visible components (same 20 % / whole-component / 3 px collar rule as the evaluator), so features of training positives have exactly the test-time distribution. | Mapped faults are a sample of an organised array: unmapped strands sit parallel to, between, or in line with mapped ones. Synthetic re-withholding makes the model learn *where an absent strand goes*, not "pixels on mapped faults" (the d=0 tautology). | H77-Step uses only tip-cone propagation (V-LBCO) and explicitly avoids network density/distance; GEMSDOE27 gap closure is topographic, not network-bilateral; no lane uses perpendicular sibling-spacing kernels or real-hidden + synthetic positives jointly. | Highest expected (network prior is what H33/H77-Step partly exploit); cost medium (FFT features, ~9 builds). **Selected.** |
| 2 | Monogenic phase congruency of RTP across wavelengths | band 2 RTP | amplitude-independent phase agreement edges | buried basement offsets under basin fill not expressed as scarps | phase rather than tilt (H80T) / Euler (H77E) | Unknown; medium cost; mimic = intrusive contacts. Deferred. |
| 3 | Depth-corrected advective heat residual at wells/springs | GDR 1391 well & spring temperatures (DOI 10.15121/1881483, mirror pinned in `registry/data_manifest.json`) | regional conductive trend removal, line-supported positive residuals | upflow along concealed faults | residual rather than H80-OTA raw collinearity | Needs depth metadata completeness check; sparse points → low pixel coverage. Deferred. |
| 4 | Lidar scarp-face polarity asymmetry × gravity-gradient polarity concordance | lidar u8 `downface_max/upface_max` + band 18 | sign agreement between down-to-basin scarp facing and gravity step | young scarps on the hanging-wall side of basin-bounding faults | polarity concordance not used by H77-Step's step purity | Unknown; low cost; mimic = shorelines. Deferred. |

## 2. Lane rule for this session

Only rank 1 (H83-NETSIB) is run, three experiments maximum (see
`registry/preregistration_h83_netsib.json`). Ranks 2–4 are recorded for future lanes.

## 3. Outcome (2026-10-21; numbers from `evidence/h83netsib_run_card.json`)

- **Exp1 PASS.** Leakage canaries max AUC 0.8054 (s20) / 0.8092 (s30), below the 0.90 limit. The deterministic network surface reached max |ρ| 0.820 vs the registry.
- **Exp2 HOLDOUT-DTI** (`gems57-pooled-hide-v2`):
  - A_netsib_full: 0.1538 [0.1357, 0.1726] (s20, 11,321 withheld) and 0.1637 [0.1447, 0.1847] (s30, 11,342).
  - A_base: 0.0780 / 0.0840. S_netsib_only: 0.1577 / 0.1658. Random: 0.0445 / 0.0469.
  - The network geometry roughly doubles recovery over the geophysical/topographic stack (paired Δ +0.076 / +0.080, CIs exclude 0). The geophysics adds nothing measurable on top of the network.
- **Exp3.** The TIFF passes the validator and is byte- and rank-unique (max |ρ| 0.089). The final dots fall >70 % within 3 px of 16 wide dot rasters, so the duplicate rule fired and the verdict is **negative** (IR-H83N-01).
- **Interpretation.** On this holdout, "where does a missing strand of a mapped array go" is a strong signal: about 2× its own base control and above the preregistered incumbent by point estimate. It is below the concurrently merged H77-VDS (0.1608 / 0.1716). Its own failure mode is spatial: it places dots in the near-network zone that every wide coverage product also covers. If the hidden test set is a withheld subset of the catalogue style (§0, conditional), this is the right signal to chase. If the test set is truly unmapped faults, the mapping-density mimic (prereg) may dominate, and this holdout cannot tell those two cases apart.

## Sources (manual review)

- Competition problem description and metric: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ (login/host-restricted from this sandbox; transcription in `vendor/gems57/metric.py`).
- Tversky index: https://en.wikipedia.org/wiki/Tversky_index
- EPSG:32611: https://epsg.io/32611
- GeoDAWN airborne magnetic/radiometric release, DOI 10.5066/P93LGLVQ: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
- INGENIOUS / GDR 1391: https://gdr.openei.org/submissions/1391
- Normal-fault spacing/linkage background: Soliva, R., Benedicto, A., Maerten, L. (2006), *Spacing and linkage of confined normal faults: importance of mechanical thickness*, J. Geophys. Res. Solid Earth 111(B1), https://doi.org/10.1029/2004JB003507 ; Ackermann, R.V., Schlische, R.W., Withjack, M.O. (2001), *The geometric and statistical evolution of normal fault systems: an experimental study of the effects of mechanical layer thickness on scaling laws*, J. Struct. Geol. 23, 1803–1819, https://doi.org/10.1016/S0191-8141(01)00028-1 . Both pages were retrieved 2026-10-21 via their DOI links (title/authors/volume verified). They motivate the *existence* of regular spacing and relay linkage in normal-fault arrays; the 0.4–2.0 km band used here is a fixed, preregistered choice, not a value taken from those papers.
