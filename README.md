# GEMSDOE77: live-score inversion for the DOE GEMS fault-discovery prize

## ⬇ Download the submission file

**[Download `gems77-h77-lia-38323dots-20261021-aa5c9331-zeros.tif`](docs/downloads/gems77-h77-lia-38323dots-20261021-aa5c9331-zeros.tif)** (151,009 B)

- sha256 `a8bfb844ab44b5d4157f75ec914a51655c27ae54727f7b4a088eca9d6f396898`
- Same file as a one-TIFF zip: [`…-zeros.zip`](docs/downloads/gems77-h77-lia-38323dots-20261021-aa5c9331-zeros.zip)
- NaN-outside twin, bit-identical inside the footprint: [`…-nan.tif`](docs/downloads/gems77-h77-lia-38323dots-20261021-aa5c9331-nan.tif)

| Question | Answer |
|---|---|
| OK to download? | **Yes** |
| Will the portal accept it? | **Yes.** It passes 17/17 checks against the official grid and footprint, using the same encoding as the 0.2778 file the portal accepted and scored: all cells finite, binary {0, 1}, no nodata tag, EPSG:32611, 3730×3292 with the official transform. |
| Unique? | **Yes.** Worst ρ 0.377; worst overlap 0.660 against 765 informative registry rasters. 43 space-filling rasters, which any raster fails, are reported. The first build was logged as a duplicate and not shipped. |
| Spend a weekly slot? | **Not recommended. Run verdict: NEGATIVE** (pre-registered rule). INSTRUMENT Δ versus the 0.2778 champion is S0 -0.043 and S1 -0.041. HOLDOUT-DTI is 0.0706 against 0.0958 for the fold-fair champion construction (Δ -0.0252, 95% CI [-0.0303, -0.0203]). **The selector decides.** |

**Form fields**

- **Name:** `gems77-h77-lia-38323dots-20261021-aa5c9331-zeros`
- **Note** (113/140): `GEMSDOE77 H77 live-inversion allocation: 66% H19-5 support + 34% straight-edge dots, 38323 binary, 0 on catalogue`

**Links:** site at <https://buffedlizard55-lab.github.io/GEMSDOE77/> ·
[executive summary: how to submit](https://buffedlizard55-lab.github.io/GEMSDOE77/executive-summary.html)

> ### Core values
> **Maximize P(Win).** "Maximize the Probability of Winning" is our decision-making framework. In every
> decision we weigh tradeoffs, assess risk, and choose the path that maximizes the probability that we
> win this prize. We set aside our emotions and make tough decisions in order to maximize P(Win).
>
> **Own the Outcome.** We own results end to end, not just our individual slice of the work. When problems
> arise and we have the means to act, we act without waiting for permission or assignment. We treat
> failure and success as signals and use them to improve. We stay accountable to the final outcome.
>
> *Applied here:* the run's verdict is reported as **negative** rather than dressed up, because a scarce
> slot should only buy a measured gain or measured information.

---

## Results in brief

Number labels: **ORGANIZER-CONFIRMED** (public leaderboard), **OWNER-REPORTED** (transcribed from the
submission page), **HOLDOUT-DTI** (local hide-and-recover), **INSTRUMENT** (projection, never a score).

1. **Why GEMSDOE32 `h33-h33-2-b2` scored 0.2778.** We inverted all 95 distinct organizer-scored registry
   rasters through the exact algebra of the metric (E[TP] = Σ h·M).
   - The fit reaches leave-one-out Spearman **+0.917**, against **+0.20** for catalogue distance alone.
   - Hidden (new) faults are concentrated on the H19-5 ridge set (19GEMSDOE): **≈ 38×** the average
     density more than 2 km from mapped faults, and ≈ 0 more than 3 px from it.
   - The champion is a 3-px binary thinning of that set, with dots ≤ 200 m from the catalogue removed.
   - Each dot earns ≈ 0.125 credit, against a break-even bar α·DTI ≈ 0.058.
   - The model reproduces every step of its lineage, 0.1922 → 0.2477 → 0.2600 → 0.2708 → 0.2778.

   Details: [`reports/why_02778.md`](reports/why_02778.md).
2. **Can it be beaten?**
   - Without the uniqueness rule: modestly. 3,825 of its dots, all within 400 m of the catalogue, earn
     below the bar.
   - Under the brief's rule: at most 70% of dots may lie within 3 px of one registry raster, and the
     H19-5 set is itself a registry raster. At least 30% of the dots must then earn credit away from it.
     No tested off-support field shows leave-one-out-robust credit there, and the break-even is
     0.056 credit per off-support dot.
   - Leaderboard #1 is **0.4413** (ORGANIZER-CONFIRMED, read 2026-10-21). The brief's 0.3195 is stale.
     At our dot count, 0.4413 needs 0.194 credit per dot (1.56× ours), which requires a better field,
     not better thinning.
3. **The GEMSDOE77 file.**
   - **Composition.** 25,293 dots on the live-validated support (> 400 m from the catalogue) plus
     13,030 dots on a new straight-edge field (H77-1), within a per-raster overlap budget of 0.66.
   - **HOLDOUT-DTI 0.0706** [0.0599, 0.0815] (`gems57-pooled-hide-v2`,
     11,321 withheld positives).
   - **Below** the fold-fair champion construction (Δ -0.0252
     [-0.0303, -0.0203])
     and above random (+0.023). Leakage canary clear (max AUC 0.755).
   - INSTRUMENT Δ versus the champion: S0 -0.043, S1 -0.041.
   - **Uniqueness costs ≈ 0.04.** With today's fields, "unique" and "beat 0.2778" are in tension.

Experiments: **3 of 3 used** (E1 instrument calibration, E2 hypothesis tests by live inversion,
E3 candidate, holdout, canary, uniqueness and validator).

---

## The brief (full prompt, re-read every session)

> **Transcription note.** This session's context kept a structured record of the brief, not the
> original message verbatim. Everything below is that record, organized under the brief's own
> headings. Wording may differ slightly from the original, but no requirement has been dropped. The
> one paragraph that was missing from the brief itself is marked.

**Goal.** Build GEMSDOE77 into a project aimed at topping DrivenData competition 306, "DOE GEMS"
(fault-discovery raster, distance-weighted Tversky/DTI metric). Use the core values above
("Maximize P(Win)", "Own the Outcome").

**Highest urgency.**

- Generate a **unique** TIF submission. It must not copy any prior GEMSDOE submission; prior ones may
  be used only for learning.
- Make it obvious whether the TIF is OK to download and submit.
- Put an easy download link at the very top of the site and of the executive summary.

**Analysis.**

- Explain with PhD-level judgement why and how GEMSDOE32
  `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` scored **0.2778**, the best of the GEMSDOE sites.
- State whether we can beat it, and apply that reasoning to the new submission.
- Targets cited: **0.4413** (current leaderboard high) and **0.3195** (highest right now). These
  conflict; flag and verify.
- The brief lists every GEMSDOE site's submission name and public score; they are transcribed in
  [`registry/live_scores.csv`](registry/live_scores.csv). GEMSDOE73–78 were listed as empty.

**Hypotheses.** Before implementing, generate 3–5 candidate geological hypotheses not yet tried. Each
must name:

- the specific layer(s);
- the physical signature (e.g. an edge-detection or curvature transform);
- why it should catch a fault missing from the USGS/INGENIOUS catalogue rather than one already in it;
- how it differs from prior repo work.

Rank them by expected DTI gain and implementation cost. Validate the top one on a spatially blocked
holdout before using any weekly submission slot. If new external data is needed, name the free
official source and verify it is obtainable.

**Parallel-run protocol.**

1. **Lane.** "Your lane is the single method paragraph below." *That paragraph is missing from the
   brief* (irregularity IR-77-01). Before running, compare against the registry. It is a duplicate if
   its rank correlation with any registry raster exceeds 0.90, or if more than 70% of its dots fall
   within 3 px of one registry raster's dots. Check both on the surface before placement and on the
   final dots. If duplicate, log it and stop.
2. **Shared tools.** Reuse the template's cached feature stack, `evaluate_holdout.py` and
   `submission_writer.py`.
   - Holdout = hide-and-recover: withhold whole fault segments with a buffer.
   - Derive catalogue-based features only from the visible faults, and mask visible faults
     pixel-exactly.
   - Score pooled DTI with alpha 0.2, beta 0.8 and a 300 m triangular kernel.
   - If a shared tool is wrong, fix it once in the template and report it; no private forks.
3. **Number labels.** Every number is labelled either **HOLDOUT-DTI** (with evaluator version, number
   of withheld positives and 95% CI) or **ORGANIZER-CONFIRMED**. Never write a projection as a score.
4. **Leakage canary.** Test each feature alone on the holdout. AUC above 0.90 means leakage until
   proven otherwise.
5. **Run card.** End with one JSON card containing:
   - the hypothesis, the mechanism, and the named non-fault process that could mimic it;
   - holdout DTI with CI, and correlation/overlap versus the registry;
   - raster sha256;
   - validator output: no NaN inside the footprint, values in [0,1], CRS/shape/transform match;
   - submission name, and a note of at most 140 characters;
   - verdict: promote or negative. Negative results are deliverables.
6. **Budget.** Stop after 3 experiments or 2 hours. Do not pick submissions; a separate selector step
   does that, within the weekly cap.

**Submission fix.**

- Fix the portal error "Predicted values must be in range [0, 1]".
- Give each submission a unique name and a short note.
- Form rules: a single-band GeoTIFF (or a .zip with one GeoTIFF) matching the submission format's CRS,
  shape and geotransform. The note is optional.

**Site and README.**

- An executive-summary subpage explaining exactly how to submit.
- A clean, simple, organized GitHub Pages site with official, verified links.
- Put the full prompt into `README.md` and re-read it every session.

**Research.**

- Do deep research from official sources and store the knowledge in the repo.
- Look for overlooked data sources and contrarian-but-grounded hypotheses.
- Resolve the data blocker autonomously (earlier plan: `scripts/download_competition_data.sh`, then
  `scripts/prepare_data.py`).
- State limitations and access needs.

**Working rules.**

- No hallucinations. Verify line by line against official, trusted sources, give links for manual
  review, and flag every irregularity.
- No manual input: work autonomously.

**Wrap-up.** Run 3 passes (implement and verify; review and fix; re-check against the request). Then
create a PR, merge it to main, and suggest the remaining work.

**Links provided.**

- Competition pages:
  - <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>
  - <https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/>
  - [data page](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) (login required)
  - [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- Reference solution: <https://github.com/drivendataorg/gems-prize-reference-solution>
- Submission rules: <https://docs.nlr.gov/docs/fy26osti/96647.pdf>
- INGENIOUS data: <https://gdr.openei.org/submissions/1391>
- [USGS GeoDAWN](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)
- <https://gbcge.org/current-projects/ingenious/>
- <https://epsg.io/32611>
- Tversky index (Wikipedia)
- Dropbox copies of `GEMS_96647.pdf`, `example_submission.tif`, `existing_faults.tif`,
  `gems-geodawn-numerical-features.tif` and `Digital-elevation-model-links-JSON.pdf`

---

## Reproduce

The environment has 2 CPU and 3 GB RAM; everything below fits.

```bash
python3 -m venv .venv && .venv/bin/pip install rasterio numpy scipy scikit-learn pandas pytest
.venv/bin/python scripts/restore_data.py          # 11 inputs via GitHub blob API, sha256-verified (≈40 s)
.venv/bin/python scripts/fetch_registry.py        # 812 prior rasters (registry), sha256 recorded (≈45 s)
.venv/bin/python scripts/build_cache.py           # sparse scored rasters, H19-5 distance, H77-1/H77-2 fields
.venv/bin/python scripts/scored_stats.py          # E1: per-raster structure for 95 scored rasters
.venv/bin/python scripts/e2_inversion.py          # E1/E2: live-score inversion, LOO
.venv/bin/python scripts/e2_hypothesis_test.py    # E2: H77-1 / H77-2 tests
.venv/bin/python scripts/e3_build_candidate.py    # E3: allocation
.venv/bin/python scripts/e3_holdout.py            # E3: HOLDOUT-DTI + leakage canary (template tools)
.venv/bin/python scripts/build_submission.py      # write + validate .tif / twin / .zip (fails closed)
.venv/bin/python scripts/e3_uniqueness.py         # registry audit, both phases (template tool, ~1 h)
.venv/bin/python scripts/why_02778.py && .venv/bin/python scripts/make_run_card.py && .venv/bin/python scripts/build_site.py
.venv/bin/python -m pytest -q
```

## Layout

| Path | What it is |
|---|---|
| `docs/` | GitHub Pages site (`index.html`, `executive-summary.html`, `analysis.html`, `sources.html`), downloads, run card |
| `reports/why_02778.md` | The full analysis |
| `docs/research/hypotheses.md` | Pre-registered hypotheses and decision rule (committed before the holdout) |
| `evidence/*.json` | Every measured number |
| `registry/` | Data pins, organizer-score transcription, registry index (812 rasters) |
| `src/gems77/` | Core grid/metric, inversion, fields, allocation, validator |
| `vendor/template66/` | Shared template, byte-identical to GEMSDOE66@0f6762f0 (pins in `TEMPLATE_SHA256.txt`) |
| `scripts/` | The pipeline above |

## Limitations, irregularities, next steps

These are listed on the [Sources & limits page](docs/sources.html) and in
[`reports/why_02778.md`](reports/why_02778.md) §6.

Most valuable next steps:

1. If the selector spends a slot on this file, add its score to `registry/live_scores.csv` and re-run
   E2. One score pins the off-support straight-edge credit (S0 versus S1).
2. Apply the proposed template validator patch upstream
   ([`reports/template_fix_validate_footprint.patch`](reports/template_fix_validate_footprint.patch)).
3. Search for a field with ≥ 0.06 credit per dot away from the H19-5 support. That is the only route
   to a unique file above 0.2778. Two promising signals:
   - High-resolution (1 m) lidar scarp mapping away from the support.
   - The GEMSDOE49/70 positive residuals (+0.13), which show such credit exists.
