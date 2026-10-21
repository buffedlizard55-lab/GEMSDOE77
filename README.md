# GEMSDOE77 — auditable DOE GEMS fault-discovery workbench

> **Start here at the beginning of every session.** Re-read this charter before research, code changes, experiments, or preparing any competition submission. The objective is evidence-led discovery of previously unmapped faults in the DOE GEMS GeoDAWN study area—not merely producing a raster or optimizing against known labels.

## Executive status (reviewed 2026-10-21)

**This checkout is an empty scaffold, not a working model or website.** It contains only this README and the initial commit. There is no cached feature stack, training/label/sample raster, DEM data, registry raster, model, holdout evaluator, submission writer, validator, test suite, website, or prior experiment history in this repository. Required Python geospatial packages (`rasterio`, `numpy`, `scipy`, `pytest`) are also absent from this environment.

**No GeoTIFF submission was generated.** The competition data page redirects to DrivenData login, and the competition CRS/bounds/shape/geotransform cannot be independently read from this checkout. Creating a guessed-grid or all-zero TIFF would not be a defensible scientific prediction and would risk the reported `[0, 1]` validation failure. We must not copy a previous submission. Do not upload or submit anything from this repository until an original model output is generated from authorized inputs and is validated against the actual sample/template raster.

The downloadable, submission-ready artifact requested by the project is therefore **not available yet**. This is an explicit blocker, not a claim of completion. No weekly competition slot was used.

## Competition question and verified facts

The task is to predict fault presence over the GeoDAWN area in a single-band GeoTIFF. The official problem description says the target is faults not present in the existing public USGS database and describes the expert-labeled private test set and later expert review. The official submission requirements specify EPSG:32611, 100 m resolution, matching bounds, a single float32 layer, confidence values in `[0,1]`, and null/NaN outside the data bounds. The metric is distance-weighted Tversky with alpha `0.2`, beta `0.8`, and a linear triangular kernel with 300 m support.

- **Current official public leaderboard, checked 2026-10-21:** the visible page lists `xiaofanhu` at `0.4413` in first place. This is a public leaderboard score, not evidence that our work has achieved it. Leaderboards are dynamic; check the official page again before decisions.
- **GEMSDOE32 / H33 claim:** the site documents an H33-2-B2 candidate and a download, but its own text also says that no artifact in that repository has an organizer score and labels its `0.2747` live estimate as a model projection, not a score. I could not verify `0.2778` as an organizer-confirmed result from the cited page. Treat `0.2778` as an **unverified claim from the request**, not as a confirmed benchmark. The site's historical assertions and held-out/projection figures are owner-reported, not independently reproduced here.
- **Contradictory/dated numbers in the request:** it cites both `0.3195` and `0.4413` as the current high score, and cites a broad history of scores without receipts. The official leaderboard currently resolves this: `0.4413` is the visible leader on the review date. Archived historical values are not silently promoted to current facts.

## Project operating rules

1. **Maximize P(Win; Own the Outcome.** Spend a scarce submission slot only on a tested, distinct candidate with evidence of improvement. Negative experiments are useful deliverables. Promotion to a real slot is a separate selection step.
2. **Use the shared template, not a private fork.** Reuse its feature cache, holdout evaluator and submission writer once they are present. Fix shared-tool defects centrally and document them.
3. **Spatial holdout is hide-and-recover.** Withhold whole fault segments with a spatial buffer; compute every catalogue-derived feature only from visible faults; mask visible faults pixel-exactly; pool the holdout DTI using alpha `0.2`, beta `0.8`, and the official 300 m triangular kernel. Evaluate every feature alone first. Any holdout AUC above `0.90` is a leakage alarm until explained.
4. **Uniqueness gate.** Before placement and on final dots, compare the candidate's rank-correlation to each available registry raster and its point overlap within 3 px. If correlation exceeds `0.90` or over `70%` of dots fall within 3 px of one registry's dots, mark it **DUPLICATE / STOP**. This repository currently has no registry rasters, so the gate cannot yet be run.
5. **Label every result.** A score must be labeled `HOLDOUT-DTI` with evaluator version, withheld-positive count, and 95% CI, or `ORGANIZER-CONFIRMED` with a copied submission-page receipt. A projection is never a score. Every reported number must include its evidence class and provenance.
6. **Submission integrity.** A TIF must be newly generated by the experiment, not copied from a sibling project. Compare its CRS, shape, transform, resolution, bounds, band count, and dtype with the real sample/template. Check finite values and `[0,1]` within the footprint; ensure outside-footprint cells use the allowed null/NaN representation. Re-open the written file and hash the final bytes. Do not infer a grid from an old submission.
7. **Budget.** At most 3 experiments or 2 hours, whichever comes first. Do not use a weekly submission slot to test an unvalidated idea.
8. **Source discipline.** Separate official-source facts, local measurements, owner-reported claims, and hypotheses. Cite the source and date. Do not claim independent verification when only a site narrative was read.
9. **Session closeout.** End each experiment cycle with a JSON run card containing hypothesis, mechanism, named non-fault mimic, holdout DTI and CI, registry correlation/overlap, raster SHA-256, validator outcome (NaN within footprint, value range, CRS/shape/transform), submission name and <=140-character note, and verdict `promote` or `negative`. If no test ran, put explicit `NOT RUN`/`BLOCKED` values—never invent measurements.

## Research candidates to preregister

These are **hypotheses, not measured results**. The repository has no implementation history, so novelty against this checkout is trivial; novelty against the large family of sibling GEMSDOE sites is **unverified**. The request itself lists previous approaches including topographic scarps/ridges, dotted/thinned fault surfaces, lineaments, geophysical fusion, conductance/hydro conduits, and fault-catalogue distance rules. The candidates below deliberately differ in the proposed combination/transform, but cannot be called unique until actual registry rasters are compared.

| Rank | Candidate hypothesis (proposed method lane) | Layers and physical signature | Why it could expose unmapped faults | Non-fault mimic / key risk | Expected DTI upside (qualitative) | Cost |
|---|---|---|---|---|---|---|
| 1 | **Depth-stratified MT conductance break + magnetic-source boundary coincidence** | Shallow surface conductivity and depth-to-conductive-base layers; compute multi-scale signed/unsigned gradients and a depth-contrast/change-point; corroborate only where a gradient in magnetic source-depth/TMI-derived structure has a compatible orientation. | A concealed basement discontinuity may juxtapose conductive alteration or fluid pathways against a distinct magnetic basement block without an obvious surface scarp. Cross-depth and cross-physics agreement is more selective than a single conductance or magnetic ridge score. | Lithologic contact, basin edge, clay cap, saline groundwater, cultural/interpolation artifacts. This is a **prior-ranked** idea, not proven. | **Highest expected potential**, because the mechanism targets buried structure and can filter broad single-sensor anomalies. No numerical score projection. | Medium-high: multi-band feature construction, orientation-aware matching, spatial folds, careful leakage audit. Uses provided competition layers; no external dataset prerequisite. |
| 2 | **Strain-tensor kinematic discontinuity at earthquake-density transitions** | Dilatation rate, shear strain rate, second invariant, earthquake-density layer; derive local strain principal-direction/coherence changes and boundaries in seismic density rather than simply high-value hotspots. | A mapped fault's mechanical boundary may be visible in strain organization or seismicity segmentation even when its trace is absent from the catalogue. | Spatially variable event detection/completeness, volcanic/tectonic swarms, interpolation edge, unrelated strain gradients. | Medium-high, but likely weak for old/aseismic structures. | Medium. |
| 3 | **Gravity–magnetic scale-space edge persistence with intersection/termination scoring** | Isostatic gravity anomaly/slope; RTP/TMI and horizontal/vertical magnetic gradients; detect edges at multiple scales and score persistent intersections/terminations, not long linear ridges alone. | Buried fault intersections and offsets can create matched density/magnetization boundaries even without terrain expression; persistence rejects some single-scale noise. | Lithologic contacts and survey seams can produce the same matched boundaries; previous repositories already explore geophysical lineaments, so duplication risk is significant. | Medium. | Medium-high. |
| 4 | **Detrended-elevation curvature/valley-network cross-scale asymmetry** | Detrended elevation and its slope; compute curvature/second derivatives and directional asymmetry at multiple spatial scales, requiring continuity but avoiding a simple scarp/ridge threshold. | Subtle distributed deformation or drainage deflection could mark previously unmapped surface traces omitted by lineament-only/ridge-thinning methods. | Lithology-controlled drainage, alluvial fan margins, roads, DEM artifacts; high overlap risk with extensive prior topo/scarp/ridge work. | Low-medium. | Low-medium. |
| 5 | **Conductive-cap geometry conditioned on gravity-defined basin margin** | Surface conductivity and depth to conductive base plus gravity anomaly/slope; detect breaks in conductive-layer thickness near independent basin-margin gradients, excluding broad uniform conductors. | Fault-controlled basin architecture and hydrothermal alteration could localize a narrow deep-to-shallow transition that generic hydro-conduit scoring misses. | Basin sediments, saline brines, non-geothermal alteration, generic basin edge. Could overlap earlier hydro/conductance experiments. | Low-medium. | Medium. |

**Selection:** Candidate 1 is the proposed next lane because it makes an explicit, testable, depth-dependent cross-physics prediction and uses the provided feature stack. This is only a research prioritization—not a validated win or an authorized submission. The parallel-run instruction also mentions “the single method paragraph below,” but no separate method paragraph was supplied in the request. Candidate 1 is the tentative lane for planning only; do not perform a score-producing experiment until the shared stack/evaluator, holdout design, and registry rasters are available. The hypothesis does not require external data beyond the official competition stack; its input availability must still be verified locally.

### Required first experiment (blocked)

Before placement, create Candidate 1 features from the visible faults/allowed provided layers only, test each feature alone in the buffered whole-segment hide-and-recover holdout, then test the preregistered combination at a fixed emission budget. Compare with the current best holdout result on identical folds and report paired uncertainty/95% CI. Run the required leakage canary, and run correlation/3-px overlap against every registry raster before and after placement. Stop on the stated duplicate thresholds. **This cannot currently be run:** there are no local inputs, stack, evaluator, writer, baseline, or registry rasters. No holdout result or CI is available.

## Data access and limitations

- The official competition data page redirects to login when accessed without DrivenData authentication. Do not bypass access controls or request secrets in chat. A team member with legitimate access can download the files into the project data directory through the official page.
- Public official research data are available through the [Geothermal Data Repository INGENIOUS dataset 1391](https://gdr.openei.org/submissions/1391) (including faults, quaternary volcanics, heat flow-related, geophysics, seismicity, geothermal features, and wells/springs). This is a possible research source, **not** a substitute for the competition sample raster or an already-obtained contest feature stack. Licenses/attribution and sharing terms must be checked per resource before use.
- External data use is permitted only subject to the competition rules and a license that allows use for this challenge and sharing with the sponsor for evaluation.
- The team's current data/authentication state is not available in this empty checkout. GPU access, local competition files, registry rasters, and any connected DrivenData account were not supplied. This environment cannot truthfully claim it has downloaded or prepared contest data.

## Submission workflow (must be made one-click once inputs exist)

1. Obtain authorized competition data and preserve source names, licenses, checksums, and metadata.
2. Inspect and pin the exact `sample_submission.tif` / training raster grid; record CRS, shape, transform, bounds, resolution, dtype, and valid footprint.
3. Build features without leakage; run buffered segment holdout and per-feature canaries; compare paired against a documented incumbent.
4. Apply duplicate tests and stop if thresholds are crossed. No result projection may be represented as a score.
5. Use the shared submission writer to create a **new** single-band float32 GeoTIFF; enforce bounds/grid and `[0,1]`, and explicitly mask outside the footprint in the competition-accepted way.
6. Independently reopen the file and run the validator. Save a JSON receipt and SHA-256 alongside it. Put the download first on any future site executive summary with a unique filename and <=140-character competition note.
7. Only after a passing validation and a holdout improvement should a separate selector consider a weekly submission slot. Copy the organizer receipt only after an actual portal result exists.

**Current workflow status:** steps 1–7 are not executable in this checkout. No output is offered as suitable to download and submit.

## Official sources for manual review

- [Competition overview / problem description, datasets, metric, and submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) — primary source for task, listed layers, DTI definition, and GeoTIFF requirements.
- [Competition about page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/) — USGS/DOE GeoDAWN background, fault context, and further literature.
- [Official live public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) — dynamic; checked 2026-10-21 for the `0.4413` first-place result.
- [Official competition data page](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) — login-walled in this environment; contains the actual sample/grid files when accessed by an authorized user.
- [Competition rules](https://www.drivendata.org/competitions/306/competition-doe-gems/rules/) — check external-data, license, and final-round requirements before using data.
- [DrivenData reference-solution source](https://github.com/drivendataorg/gems-prize-reference-solution) — public repository exists (GitHub API/remote checked); it is a starter reference, not evidence that its model is best.
- [USGS GeoDAWN data release](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and).
- [INGENIOUS project](https://gbcge.org/current-projects/ingenious/).
- [GDR INGENIOUS regional dataset compilation, DOI 10.15121/1881483](https://gdr.openei.org/submissions/1391) — page confirms public access and describes resources and licenses; resource-specific terms still apply.
- [EPSG:32611](https://epsg.io/32611) — CRS reference.
- [NLR GEMS rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf) — sponsor material linked in the project request.
- [H33-2-B2 page cited for prior work](https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html) — third-party project page; not an organizer receipt and not a primary scientific source.

## Useful reference

The official competition page explains why the visible public fault catalogue is incomplete and why holdout recovery against known catalogue segments is only a proxy: it cannot, by construction, validate discovery of a truly unmapped fault. Keep proxy performance and organizer scoring separate throughout the project. A holdout gain is necessary for internal promotion but never proves an organizer score above `0.2778`, `0.3195`, or the current leaderboard leader.
