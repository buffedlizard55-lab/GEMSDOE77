# H82-TEXTURE preregistration — 2026-10-21

Written before implementation/results. No identifiable single assigned method paragraph exists in the request; H82-TEXTURE is an autonomously declared narrow lane, not a claimed assignment. This is a one-experiment feasibility test, not a retuning of a closed lane.

## Ranked candidates

Expected DTI improvement is **unknown** throughout. The ranking is an ordinal judgement of plausibility per implementation cost, not a numerical forecast or a claim of global novelty. Search of this checkout's source/research found mean/rank contrasts, Hessian curvature, halo displacement and cross-sensor orientation tests, but no explicit radiometric *local second-moment boundary* operator. Other repositories have not been exhaustively searched.

| Rank | Hypothesis, layers and operator | Why possibly missing from a catalogue / difference here | Named non-fault mimic | Cost / obtainability |
|---|---|---|---|---|
| 1 | K and Th cached GeoDAWN rank-intensity: a boundary between smooth sediment and heterogeneous bedrock can reveal a covered structural margin. Compute local standard deviation in a 9×9 window, central-difference gradient magnitude of each deviation field, then geometric-mean concurrence. | Texture changes can remain where a mean-intensity contrast or exposed scarp is weak. Tests a distribution's second moment, not the previously tested mean/rank residual, halo offset or topographic step. It can also miss faults with no material contrast. | A depositional fan-to-lake sediment boundary without faulting; flight-line noise | Low; actual SHA-pinned cached K/Th bytes restored via the existing template fetcher. No new external source required. |
| 2 | K/Th/U local standardized skewness boundaries (9×9 third moments). | Asymmetric alteration/regolith mixtures could cross a blind structure without a mean shift. Third moments differ from the second-moment candidate and old curvature methods. | Heavy-mineral sorting and eolian dust deposition | Medium; same restored bytes; high quantization/noise risk. Proposal only; not tested. |
| 3 | K–Th local covariance normalized by local variances, then spatial covariance-transition magnitude. | A covered material boundary may change element co-variation without changing either mean; tests local joint statistics, not separate gradients. | Soil moisture, vegetation/cover variation or mixed lithology | Medium; same restored bytes. Proposal only; not tested. |

These are **testable hypotheses**, not established geothermal diagnostics. Quantization is independent by channel, so none of these quantities is a concentration ratio, elemental variance in physical units, or proof of hydrothermal alteration. A fault map is not a vent discovery or an assessment of heat/permeability.

## Frozen selected method

Use the existing `data/geodawn_rad_u8.tif` K/Th bands only as model inputs. Zero is missing per cache nodata; 255 is a valid saturated quantile. Divide by 255. For each channel, `sd = sqrt(max(mean(x²)-mean(x)²,0))` in a 9×9 square. Use central differences of sd and take the Euclidean gradient magnitude. Candidate = geometric mean of the two magnitudes. Normalize each diagnostic field by its label-free eligible-domain 99th percentile (clip [0,1]); multiply each arm by 0.02 as a frozen continuous-output operating point, **not a calibrated fault probability**. Include normalized K and Th, each sd, each texture gradient, candidate, and constant 0.02 control. No sweep, model fitting, catalogue-context feature or previous submission input.

Support is complete positive K/Th coverage in an 11×11 square: radius 4 for the moment window plus radius 1 for central differences. Constant padding is not permitted to influence eligible outputs. Holdout uses the full eligible support (including catalogue cells); mask only visible catalogue pixels in each fold. Full catalogue is masked only for final research export.

Run one experiment through the pinned `gems57.spatial.folds` (whole connected segments, fixed spatial quadrants, 80 px training buffer; no model trained), `evaluate_holdout.evaluate` and `pooled_summary` (20 km clusters; 1000 bootstrap draws, seed 82077). Test every input and derived feature alone for AUC and HOLDOUT-DTI on each fold. Symmetric AUC >0.90 triggers leakage rejection. Every score retains version, withheld count and CI. No claim of beating an incumbent evaluated on different support.

Check the surface with the unchanged template registry checker using the two pinned witnesses; one confirmed failure suffices for rejection, never clearance. On failure **stop before dot placement** and retain exactly the checked bounded surface through `submission_writer.py` as RESEARCH-ONLY TIF + ZIP. Check the unchanged file after readback. No second hypothesis/experiment after a duplicate. A small local inventory can establish byte/pixel differences only, never full-registry uniqueness. No weekly slot or selector approval.

## Source/provenance boundaries

- [Official organizer reference implementation](https://github.com/drivendataorg/gems-prize-reference-solution): README freshly retrieved via GitHub API this session; establishes a fault-detection baseline and CPU support, not H82 geological validity.
- [USGS GeoDAWN source](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and): official manual-review anchor; host inaccessible under this session's network allowlist. Not freshly authenticated.
- [Pinned reusable cache](https://github.com/buffedlizard55-lab/60GEMSDOE/tree/b99591eede63c6d45c8a896255b8cd34824fa00b/data/external): owner-processed, hash-pinned mirror; integrity verified, not independently authenticated official-origin data.
- [Competition problem/metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/), [data](https://www.drivendata.org/competitions/306/competition-doe-gems/data/), [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/), [rules](https://docs.nlr.gov/docs/fy26osti/96647.pdf): manual-review links only; no portal authentication, current-score receipt, weekly-cap verification or rule-compliance certification obtained.

The contradictory 0.3195/0.4413 targets in the supplied request remain unverified in this session. The named H33 0.2778 is OWNER-REPORT without an exact-file submission-page receipt. It is not a target that this experiment can promise to beat.
