# GEMSDOE77 — contact-physics fault lane for the DOE GEMS Prize

## ⛔ Executive verdict (read first): DO NOT SUBMIT the file in this repo

| Question | Answer |
|---|---|
| Is there a downloadable GeoTIFF? | **Yes** — `docs/downloads/gems77-contactphys-44090dots-20261021T030258Z-ca70a971-zeros.tif` (150,542 B, SHA-256 `20be05cc671c85484e4cfc8ea40d2fced71d7ec70f20ab915f36fcc68fcc4851`), + single-TIFF ZIP |
| Is it format-valid (no `[0,1]` portal error)? | **Yes** — all writer checks + an independent re-read pass (`evidence/verification_pass2.json`): single-band float32, EPSG:32611, 3730×3292, transform = sample submission, 0 NaN anywhere, min 0.0 / max 1.0, no nodata tag, 0 dots on the mapped catalogue |
| Is it unique vs earlier submissions? | **Yes, within the checked subset** — 0 duplicates among 43 locally hash-verified sibling rasters (worst surface Spearman 0.160 ≤ 0.90; worst mutual 3-px dot overlap 0.460 ≤ 0.70). Full-inventory certification is a selector step |
| **Is it OK to submit?** | **NO — verdict `negative`.** It failed the preregistered holdout promotion gate: two-draw paired Δ(HOLDOUT-DTI) vs the standard-feature control = +0.0026, 95% CI [−0.0030, +0.0080] (evaluator `gems57-pooled-hide-v2`, 22,914 withheld catalogue pixels). **0 weekly slots were spent.** Download is for inspection/reproducibility only |

Site: `docs/index.html` (results) · `docs/executive-summary.html` (exact submission procedure for any future *promoted* file) · Run card: `evidence/run_card.json` (= `docs/downloads/run_card.json`).

**Core values.** **Maximize P(Win)** — every weekly slot is an experiment behind a frozen gate, not a lottery ticket. **Own the Outcome** — a failed gate is measured, replicated, and published here instead of quietly dropped. Negative results are deliverables.

---

## The brief (operative prompt — re-read at the start of every session)

Full text: [`PROMPT.md`](PROMPT.md). Operative rules, verbatim in substance:

> **Highest urgency:** generate a **unique** TIF submission; make it *obvious* whether it is OK to download and submit. Do not copy a previous submission (learning/decorrelation checks only).
>
> **Parallel-run protocol.** (1) Lane discipline: if rank-correlation vs any registry raster > 0.90, or >70% of dots within 3 px of one registry raster's dots — duplicate; log and stop. Check on the surface **and** final dots. (2) Reuse the shared template (`vendor/gems57`: `evaluate_holdout.py`, `submission_writer.py`, uniqueness); holdout = hide-and-recover (whole fault segments + buffer, catalogue features from visible only, visible masked pixel-exactly, pooled DTI α=0.2 β=0.8, 300 m triangular kernel). Fix shared tools upstream, never fork privately. (3) Label every number HOLDOUT-DTI (evaluator, withheld positives, 95% CI) or ORGANIZER-CONFIRMED (receipt) — a projection is never a score. (4) Leakage canary: each feature alone on holdout; AUC > 0.90 = leakage until proven otherwise. (5) End with one JSON run card (hypothesis; mechanism; named mimic; holdout DTI+CI; registry correlation/overlap; sha256; validator output; name + ≤140-char note; verdict promote/negative). (6) Budget ≤3 experiments / ~2 h; promotion to a real slot is a separate human selector step within the weekly cap.
>
> **Science rule:** before implementing, generate 3–5 untried geological hypotheses (layers, physical signature, why it catches catalogue-missing faults, difference vs implemented work), rank by expected DTI gain ÷ cost, validate the top candidate on the spatially blocked holdout **before** touching a slot. If a candidate needs new external data, name the specific free official source and verify obtainability first.
>
> **Competition:** DOE GEMS Prize, DrivenData #306 — https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ · leaderboard https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/ · reference solution https://github.com/drivendataorg/gems-prize-reference-solution · rules PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf
>
> **Family context (OWNER-REPORT scores, no receipts):** best family file `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` = 0.2778 (GEMSDOE32). Public leaderboard #1 = **0.4413 ORGANIZER-CONFIRMED** (xiaofanhu, read 2026-10-21). Why 0.2778 scored and what beating it requires: [`research/why-h33-scored-02778.md`](research/why-h33-scored-02778.md).

## What this session did (2026-10-21, 3/3 experiments)

**Lane (single method):** expert-criteria contact physics, supervised-ranked — HGB whose distinctive inputs are *correctly derived* potential-field contact transforms (magnetic/gravity tilt, analytic signals, tilt confluence, signed basement step, monogenic phase congruency, pseudogravity-Poisson) + range-front-base asymmetry. No catalogue-derived features, no SGMC training population, no lidar in the primary arm. Hypotheses ranked pre-experiment: [`research/hypotheses.md`](research/hypotheses.md) · frozen gates: [`registry/preregistration.json`](registry/preregistration.json) (+ Exp-3 amendment).

**Results (all HOLDOUT-DTI, `gems57-pooled-hide-v2`):**

| Arm (seed-30 draw, 11,342 withheld px) | DTI [95% CI] |
|---|---|
| A5 standard+new+lidar (ablation only) | 0.1130 [0.0961, 0.1325] |
| **A4 primary (standard+new, no lidar)** | **0.1090 [0.0921, 0.1277]** |
| A2 standard block (control) | 0.1043 [0.0866, 0.1221] |
| A3 new physics only | 0.0988 [0.0842, 0.1155] |
| Single feature `frontness` | 0.0836 [0.0689, 0.1004] |
| Random matched dots | 0.0545 [0.0478, 0.0610] |

Paired A4−A2: seed-30 +0.0047 [−0.0027, +0.0124] → seed-21 replication +0.0005 [−0.0081, +0.0093] → two-draw pooled +0.0026 [−0.0030, +0.0080]. **Gate failed twice ⇒ verdict negative.** Canaries passed (worst symmetric AUC 0.5874). Receipts: `evidence/holdout_exp1.json`, `evidence/holdout_exp3_replication.json`, `evidence/final_bundle.json`, `evidence/verification_pass2.json`.

**Reusable yield (measured, family-wide value):**
1. **Band 6 is radiometric total count, not tilt** (Spearman vs GeoDAWN TC = +0.99995; vs derived tilt = +0.020). The shipped stack contains no tilt band — audit any sibling "tilt" channel built on band 6. (`evidence/band6_resolution.json`)
2. **Band 18 is ∂g/∂x, not |∇g|** (ρ = 0.945 vs dx; −0.004 vs magnitude). (`evidence/band_diagnostics.json`)
3. `frontness` (front-base asymmetry) is the strongest single channel tested here (canary AUC 0.5874; alone HOLDOUT-DTI 0.0836).
4. New-physics-only arm leads the SGMC-off-catalogue proxy (PROXY-DTI 0.1226 vs 0.0926 standard; 62,122 px) while trailing on catalogue holdout — a preregister-next-time hypothesis, **not** a claim.

## Reproduce (CPU, no DrivenData login; needs `gh`)

```bash
pip install numpy scipy rasterio scikit-learn pillow   # sandbox used --break-system-packages
python3 scripts/restore_data.py --ids all               # SHA-256 fail-closed mirrors -> data/raw (gitignored)
python3 tests/test_features77.py                        # synthetic transform tests
python3 scripts/build_features.py                       # 54-ch stack -> data/processed/stack.npy (~8 min)
python3 scripts/run_holdout.py                          # Exp1 canaries + arms + scan (~9 min)
python3 scripts/build_final.py --arm A4_primary         # Exp2 surface -> dots -> writer + uniqueness (~2 min)
python3 scripts/run_replication.py                      # Exp3 seed-21 replication (~3 min)
python3 scripts/verify_all.py                           # independent pass-2 audit (exit 0 = pass)
python3 scripts/make_run_card_and_figures.py
python3 scripts/build_site.py
```

Data provenance: owner-supplied GitHub mirrors of the login-walled DrivenData files (manifest + hashes in `registry/data_manifest.json`); integrity-pinned, **not** organizer-authenticated. Dropbox links in the brief are unreachable from the sandbox; the GitHub mirror path above is the working route. USGS FDSN/NLR hosts also unreachable here (verified HTTP 000) — seismicity remains the largest unused free official source (fetch on an unrestricted machine).

## Limitations & next work (for the next session)

- Catalogue hide-and-recover cannot reward a genuinely new fault class (its truth *is* the catalogue); it is a screen, and this candidate did not pass even the screen vs the standard block.
- Registry uniqueness covers 43 locally present rasters, not the full family inventory — certify before any slot.
- Next preregistrations suggested by the evidence (new draw, frozen gates): (a) A5 lidar arm as primary; (b) new-physics field on a piedmont/off-catalogue instrument declared in advance; (c) sub-km seismicity lineaments once ComCat/NSL bytes are mirrored to GitHub. Until one beats the holdout best under its own preregistration, **spend slots only on sibling candidates that already passed their gates** (selector's call).

## Irregularities flagged for review

See [`research/irregularities.md`](research/irregularities.md): IR-77-01 band-6 tag wrong (TC≠tilt) · IR-77-02 sample submission ≠ "total fault absence" (its 60,988 ones exactly equal the labels) · IR-77-03 band-18 is a signed x-derivative · IR-77-04 family 0.2778 has no organizer receipt in inspected repos · IR-77-05 this repo arrived empty (no template/data) — template restored from GEMSDOE70's vendor cache · IR-77-06 prompt assigns no single method-paragraph lane — lane declared in `registry/preregistration.json`.
