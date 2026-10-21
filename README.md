# GEMSDOE77 — Dilational Intersection & Clay-Cap Breach for DOE GEMS Prize

**One-click submission file (portal-ready ✅):**

> **⬇ [Download gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.tif](https://buffedlizard55-lab.github.io/GEMSDOE77/docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.tif)** · [.zip](https://buffedlizard55-lab.github.io/GEMSDOE77/docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.zip) · [NaN twin](https://buffedlizard55-lab.github.io/GEMSDOE77/docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-nan.tif)
>
> **sha256:** `0f86bf205eff2522fad4253d88dbd62220d06408deafb25a9672cc2e7ea0acab` · **44,090 dots** · 0 on catalogue (2 px buffer) · EPSG:32611 100m 3292×3730 single-band float32 · every cell finite, min 0.0 max 1.0 · **12/12 validator PASS — portal will ACCEPT**
>
> **Unique submission name:** `GEMSDOE77-H77C-CLAYCAP-BREACH`
>
> **Note (104/200 chars):** `GEMSDOE77 H77-C claycap_breach | 44090 dots, 0 on catalogue, holdout 0.020+/-0.014, 0.020 greedy vs thin`
>
> **Is it OK to download and submit? YES — file is valid, but holdout is NEGATIVE (0.0196 vs random 0.0184, not beating public 0.2778/0.3195) — see Run Card. Portal will not error.**

---

## Executive summary — is this submission safe?

**YES — it is safe to download and submit without fixing.** The file is single-band float32, EPSG:32611, 100 m, 3292×3730, transform `[100,0,243350,0,-100,4508550]`, every one of 12,279,160 cells finite and in `[0,1]`, 0 NaN inside footprint, no nodata tag (zeros variant) or nodata=NaN with NaN outside (nan twin). It passes all 12 validator checks by re-reading the written bytes. The DrivenData error `"Predicted values must be in range [0, 1]"` you saw before was caused by a `−3.4028235e+38` sentinel or `nodata=NaN` sentinel outside `[0,1]` — this file has **zero** such cells.

**But will it win?** Our HOLDOUT-DTI (4 quadrants, 30 px buffer, catalogue proxy, DTI α0.2 β0.8 R=300m) is **0.0196 ±0.0142 (95% CI)**, barely above random **0.0184**, far below prior group's CV GB field **0.1250** and below the bar **0.05** for promotion. Verdict **negative** — this hand-crafted clay-cap field is not meaningfully better than random on the proxy and likely **will not beat leaderboard 0.2778/0.3195**. We publish it as a **valid unique lane** (correlation 0.001 vs registry, overlap 18.7% < thresholds) and document what to try next. Do not spend a weekly slot expecting a score gain.

**Site:** https://buffedlizard55-lab.github.io/GEMSDOE77/ — see **[Executive summary → how to submit in 2 minutes](https://buffedlizard55-lab.github.io/GEMSDOE77/docs/executive-summary.html)**.

---

## The prompt we are solving (read every time we work)

> We must generate a **unique** TIF submission for the DOE GEMS Prize competition. It must be obvious whether it is OK to download and submit. The site must have an easy one-click download file matching submission format (CRS/shape/geotransform), validated to `[0,1]`.
>
> Leaderboard to beat: public #1 **0.3262** (nchuzhoy, 2026-10-04), #2 **0.3195** (DARD), group internal best **0.2778** (H33-2-B2 flank B=2 prune on 0.2708 base, 37,654 dots). Prior GEMDOE sites list many submissions from 0.00 to 0.32.
>
> We need 3–5 candidate geological hypotheses, each naming layers, physical signature (edge/curvature transform), why it catches a fault missing from USGS/INGENIOUS catalogue, how it differs from prior. Rank by expected DTI improvement & cost. Validate top candidate on spatially-blocked holdout before spending slot — do not spend slot on idea that hasn't beaten current holdout best.
>
> Work line by line verified from official trusted sources, links for manual review, no hallucinations. Flag irregularities. Maximize P(Win), Own the Outcome. Budget: 3 experiments or 2 hours. Label numbers HOLDOUT-DTI vs ORGANIZER-CONFIRMED. Test leakage canary AUC>0.90. Check rank-correlation >0.90 or 70% 3px overlap vs registry = duplicate lane. End with one JSON run card: hypothesis, mechanism, non-fault mimicker, holdout DTI+CI, correlation/overlap, raster sha256, validator output, submission name+note ≤140 chars, verdict promote/negative.

---

## Run card (HOLDOUT-DTI vs PROJECTION)

```json
{
  "hypothesis": "H77-C claycap_breach: Conductivity gradient ridge coincident with magnetic low",
  "mechanism": "Triple intersection — positive dilatation gates basement step + gravity ridge with strike alignment; only dilational relay ramps offset basement without scarp, geothermal-permeable.",
  "non_fault_mimicker": "Glacial terraces & paleo-shorelines produce basement-step anomalies without fault; they are dilatationally quiescent and misaligned, gated out, but pediment scarps could still mimic.",
  "holdout_DTI": {
    "mean": 0.0196,
    "ci95_half_width": 0.0142,
    "ci95_interval": [0.0055, 0.0338],
    "n_withheld_positives": 57407,
    "evaluator_version": "gems77 holdout v1 (4 quadrants, 30px buffer, catalogue truth proxy, DTI α0.2 β0.8 R=300m)",
    "label": "HOLDOUT-DTI"
  },
  "live_projection": "PROJECTION only — no ORGANIZER-CONFIRMED score exists for this file",
  "correlation_overlap_vs_registry": {"correlation": 0.001, "overlap_3px": 0.187, "thresholds": [0.9, 0.7], "verdict": "PASS unique lane"},
  "raster": {"filename_zeros": "gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.tif", "sha256_zeros": "0f86bf205eff2522fad4253d88dbd62220d06408deafb25a9672cc2e7ea0acab", "shape": [3730,3292], "dtype": "float32", "crs": "EPSG:32611", "emitted": 44090, "on_catalogue": 0},
  "validator_output": {"no_NaN_inside": true, "values_in_0_1": true, "crs_shape_transform_match": true, "all_checks_passed": true},
  "submission": {"name": "GEMSDOE77-H77C-CLAYCAP-BREACH", "note_140_chars": "GEMSDOE77 H77-C claycap_breach | 44090 dots, 0 on catalogue, holdout 0.020+/-0.014, 0.020 greedy vs thin", "portal_ready": true, "verdict_download_ok": "YES — portal will ACCEPT"},
  "verdict": "negative",
  "budget": "1/3 experiments used, <2h"
}
```

---

## Hypotheses ranked by measurement

| Rank | Hypothesis | Layers (official) | Signature | Holdout mean_greedy | Cost |
|------|------------|-------------------|-----------|---------------------|------|
| 1 | **H77-C clay-cap breach** | cond_surf (17) + mag_anom (1) | `\|∇ cond_surf\|` × mag low (alteration breach) | 0.0196 | low |
| 2 | H77-E seismic relay | deq (10)+ieq(16)+grav_hg(18)+depth(15) | (quake proximity+intensity) × structural ridge in gaps | 0.0179 | low |
| 3 | H77-A dilational intersection | geod_dilaterate(8)+depth(15)+grav_hg(18)+tmi_hg(3) | dilatation_pos × basement ridge × gravity ridge × alignment | 0.0163 | low-medium |
| 4 | H77-B blind saddle | det_elev(12)+slope(19) | Hessian saddle λ1>0 λ2<0, low-slope gated | 0.0132 | low |
| 5 | H77-D radiometric alteration | K/Th/U GeoDAWN external DOI 10.5066/P93LGLVQ + tmi(14) | Th/K high along lineaments (K-depletion) | not measured | medium (needs external rad ~26 MB, free official) |

H77-D requires free official USGS GeoDAWN radiometric grids (DOI 10.5066/P93LGLVQ, ScienceBase 657e1d85d34e23d3533209f7) — obtainable, ~26 MB.

---

## Why 0.2778 scored high & how to beat it

**Why H33-2-B2 scored 0.2778 (group internal best):** It's a flank B=2 prune on the 0.2708 base (40,199 dots) — removing every dot within 200 m of catalogue (37,654 kept). Metric is a budget: every pixel not best-cover costs α=0.2, best-cover earns ≤1. Thinning removes mass already covered, raises credit/pixel to break-even `k > α·s` (0.052 at 0.2600). The group measured break-even 0.0548 empirically, derived 0.052 from formula — two routes, one answer. Public #1 0.3262 needs +25% mean credit/pixel at same mass vs group best — needs better detector.

**Can we beat 0.2778?** Only with a better detector + scatter-blurred objective (σ=1.85 px per H28). Our hand-crafted ridges are currently 12× weaker than a 35-channel GB CV field (0.019 vs 0.125 holdout). Next experiment should train that GB field + radiometric, then emit via greedy on scatter field.

Full analysis: `knowledge/01_geothermal-vents.md`, `docs/hypotheses.html`, `docs/research/` in sibling GEMSDOE32.

---

## Project structure

```
GEMSDOE77/
├── docs/            # GitHub Pages site (one-click download, validator, run card, knowledge)
│   ├── index.html
│   ├── executive-summary.html
│   ├── hypotheses.html
│   ├── sources.html
│   ├── knowledge/01_geothermal-vents.html
│   ├── assets/site.css
│   └── downloads/   # submission TIFs + audit JSON + .zip
├── src/gems77/      # metric.py, grid.py, features.py, holdout.py, emission.py, submission.py
├── scripts/generate_submission.py  # full pipeline: fields → holdout → emission → validator → run card
├── registry/submission_build.json  # machine-readable feed (sha256, checks, holdout)
├── knowledge/        # curated sources
└── data/bridge/     # mirrors of competition files (pinned sha256, not organizer-authenticated)
```

---

## Core values

**Maximize P(Win)** — every slot is an experiment, not lottery. **Own the Outcome** — every claim carries evidence class, negative results published.

---

## Data: blocker to training (now solved)

Training data was login-walled at https://www.drivendata.org/competitions/306/competition-doe-gems/data/ . We mirrored via owner data_manifest.json (sha256 pinned): `training_features.tif` (19 bands, 400 MB, sha256 4371c82e…), `labels.tif` (7ba308cc…), `sample_submission.tif` (2176d08e…). Re-assembled from 5 parts in `data/bridge/` and verified footprint 5,167,373 px.

To reproduce on unrestricted machine: `bash scripts/download_competition_data.sh && python scripts/prepare_data.py` — then `python scripts/generate_submission.py` ready (GPU optional; metric/val working on CPU).

---

## Limitations & next steps

- **Stronger detector needed:** hand-crafted 0.0196 barely beats random 0.0184; GB CV field reached 0.125. Train GB on 35 channels next.
- **External radiometric:** H77-D (Th/K) viable via free GeoDAWN DOI 10.5066/P93LGLVQ, not yet fetched (CI-only egress).
- **Identification pack:** to calibrate live projection, run 3-slot experiment (S1 anchor, S2 0.5 scale, S3 null-add 2000 px far from faults).
- **Do not promote H77-C to weekly slot** — negative holdout. Next slot should be GB+claycap ensemble.

---

## Reproduce & verify

```bash
pip install --break-system-packages rasterio numpy scipy scikit-learn
python scripts/generate_submission.py  # 3-5 min, logs holdout + validator + run card
# Check download is portal-ready:
python -c "import rasterio,numpy as np; import hashlib; p='docs/downloads/gemsdoe77-h77c-claycap-breach-20261021T023803Z-58b33f22-zeros.tif'; import rasterio as rio; a=rio.open(p).read(1); print(a.min(),a.max(),np.isfinite(a).sum(), (a>0.5).sum()); print(hashlib.sha256(open(p,'rb').read()).hexdigest())"
```

---

## Sources (verified line-by-line)

- Problem/metric: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Leaderboard: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
- GeoDAWN: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and (DOI 10.5066/P93LGLVQ)
- INGENIOUS: https://gbcge.org/current-projects/ingenious/
- 3DEP: https://apps.nationalmap.gov/downloader/
- EPSG:32611: https://epsg.io/32611
- Tversky: https://en.wikipedia.org/wiki/Tversky_index
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution
- Rules PDF: https://docs.nlr.gov/docs/fy26osti/96647.pdf

See `docs/sources.html` for full table with health probes.

---

## License

MIT — use, reproduce, verify. No organizer scores claimed; all numbers HOLDOUT-DTI unless marked ORGANIZER-CONFIRMED.

