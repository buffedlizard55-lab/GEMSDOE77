# GEMSDOE77 review — branch `arena/6a5eca5f-gemsdoe77` (2026-10-21)

## Decision at a glance

**NO NEW TIFF GENERATED · NO NEW HOLDOUT RUN · NO SLOT USED · NOT OK TO SUBMIT ANY FILE.** The user's demand for a new unique TIFF conflicts with the repository's explicit duplicate-stop protocol: a hash-pinned registry witness covers every allowed pixel after the required 3-pixel dilation. Any non-empty candidate has forward overlap `1.0`, above the strict `0.70` limit. The result was freshly reproduced by `scripts/preflight_registry.py`; its expected exit code is 2 (`STOP_UNIVERSAL_DUPLICATE`). Do not bypass this by renaming an old TIFF, creating an empty/zero file, relaxing the gate locally, or treating format validation as submission authorization.

The existing negative research TIFFs remain available only as individually marked on the website; they are **NOT OK TO SUBMIT**. The authorization receipt reports zero approved files. The requested new TIF is therefore **not available to download** from this review. The submission guide remains at [executive-summary.html](executive-summary.html), and the landing page labels downloads and submission clearance separately.

## What was inspected and reproduced

- Followed `AGENTS.md`: reviewed the full `PROMPT.md`, README status, irregularities, registry/preregistration records, current release/readiness reviews, closed H82/H80/H81/H77L run cards, shared-template pin, input manifest, and site generation path.
- The original branch already contains `research/why-h33-scored-02778.md`. Its conservative conclusion is retained: the pinned H33 mirror is a 37,654-positive binary raster; retained byte forensics say it removes 6,436 near-catalogue dots from a 44,090-dot predecessor without adding positives. This is a plausible sparse-pruning mechanism under a distance-weighted coverage metric, **not causal proof** that it received the owner-reported 0.2778. The owner-supplied score is not tied to these bytes by an organizer receipt; the pinned GEMSDOE32 page itself says it has no organizer score for its artifacts.
- `registry/template_pin.json` hashes were checked by the preflight. Input hashes matched: sample `2176d08e…d35cbc`, catalogue `7ba308cc…ae4093`, dense witness `ab0a0a62…3872be`.
- Fresh preflight measurement: all 5,106,385 allowed pixels are within 3 px of the pinned witness; uncovered count is zero. The proof rejects a nonempty candidate under the current literal overlap definition. This is a **REGISTRY-MEASUREMENT**, not a holdout score or complete-registry uniqueness pass.
- Current local feature stack status: `data/raw/training_features.tif`, `data/processed/stack.npy`, and `.cache/gems_data/training_features.tif` are absent. Existing restore manifests describe owner-supplied mirrors and explicitly do not establish organizer authentication. Therefore no cached-stack feature evaluation was possible.
- No candidate surface or final dots were constructed. The required surface-before-placement and final-dot checks are thus not candidate-specific; the universal witness already makes a placement attempt inadmissible.

## Candidate screen — ranked research priorities, not score projections

The ranking is **qualitative potential information value relative to cost**. Expected DTI change is unknown for every item. “Not found in executable code” is a bounded repository search claim, not proof of originality across sibling repositories or all participants. No candidate below was preregistered or tested in this review because the duplicate-stop gate fires before feature generation.

| Rank | Candidate and layers | Targeted physical signature / why it could reveal a missing fault | Difference from implemented work; named non-fault mimic | Data source and availability check | Cost / status |
|---|---|---|---|---|---|
| 1 | **Multiscale magnetic-gravity cross-wavelet phase coherence:** GeoDAWN RTP/TMI and gravity bands from the competition feature stack | A repeatable localized cross-wavelet coherence/phase relation between RTP/TMI and gravity could flag a shared subsurface boundary where a concealed fault juxtaposes magnetic and density contrasts beneath basin cover, even without a mapped surface trace. Require that relationship to persist across scales and spatial blocks; source-depth differences can shift phase and make this hypothesis fail. | Static search found that individual-field monogenic phase congruency already exists in `src/gems77/features77.py` and has been used in the contact-physics lane, and H79 already measures real-space cross-sensor orientation agreement. The proposed transform is specifically localized cross-wavelet magnitude-squared coherence plus relative phase between magnetic and gravity spectra; a repo-wide grep found no wavelet/cross-spectrum implementation, but that is not a global novelty claim. It differs from matching edge-axis direction, not from phase-sensitive methods in general. **Mimic:** lithologic or intrusive contact and flight-line/regridding seam. | No external dataset is required if the existing stack is restored. Its large raw feature TIFF and processed cache are absent in this checkout; mirrors are owner-supplied, not organizer-authenticated. **Not currently testable.** | Medium implementation cost; highest conditional value/cost because it adds amplitude-normalized structure to existing bands. No DTI estimate. Blocked by both missing stack and universal duplicate stop. |
| 2 | **Uncertainty-weighted 3-D microseismic planes:** event location, depth, horizontal/depth uncertainty and focal mechanism from USGS ComCat | A persistent depth-consistent plane or aligned swarm between mapped segments may indicate an active blind structure, rather than merely a high event-density cell. | Distinct from rasterized regional seismicity aggregates, but an event-lineament hypothesis exists elsewhere in prior research notes, so not globally novel. **Mimic:** quarry blasts, aftershock sequences, or geothermal-production-induced seismicity. | Official USGS [ComCat FDSN event service](https://earthquake.usgs.gov/fdsnws/event/1/query). Prior repository irregularity `IR-81-06` records the host as unreachable from this sandbox (HTTP 000 / exit 35); no events were retrieved here. **Not obtainable/validated in this environment.** | Medium–high; low data volume, high temporal completeness and event-classification burden. No DTI estimate. |
| 3 | **Repeat Landsat thermal-anomaly boundaries:** Level-2 surface temperature, QA/cloud masks and acquisition dates, controlled by elevation/aspect and land cover | A compact anomaly boundary persistent across seasons and usable acquisitions could localize active upflow or a fault-fed thermal corridor missed by static structural mapping. | A time-series radiometric response differs from existing point well/spring temperature alignment and static airborne K/Th/U features. **Mimic:** irrigation, wildfire, industrial/urban heat, cloud contamination, or a warm spring unrelated to a fault. | Official USGS [Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2). AOI scene availability and cloud-free revisit depth were not checked; USGS is outside this sandbox's outbound allowlist. **Not obtainable/validated here.** | High; acquisition/season/QA confounding and processing burden. No DTI estimate. |
| 4 | **Depth-corrected well/spring thermal residuals:** GDR 1391 temperature, measured depth, chemistry and metadata | Remove a robust regional conductive temperature-depth trend, then test whether off-catalogue positive thermal residuals form a spatially supported lineament; this targets advective upflow without treating every hot well as a fault. | Different from raw hot-point alignment / advection features already explored in H80/H77V, but the extension is not wholly untried project-wide. **Mimic:** depth-selection bias, production/pumping, shallow groundwater mixing, and volcanic heat. | Official DOE/NREL [GDR submission 1391](https://gdr.openei.org/submissions/1391). The local manifest records an owner mirror, but the actual CSV is absent here; the source host cannot be reached through the sandbox allowlist. Availability of depth metadata is **not verified**. **Not viable until checked.** | Medium–high; potentially interpretable but selection-bias corrections and complete depth metadata are essential. No DTI estimate. |

### Candidate selection and validation gate

Rank 1 is the only candidate with no *new* external-data dependency, so it is the conditional first choice **only after** the feature stack is restored with provenance, the protocol owner resolves the shared dense-support rule, and a single method lane is explicitly assigned and preregistered. It was **not** validated here; no canary AUC, holdout DTI, confidence interval, or candidate-specific correlation exists. It would be incorrect to call it viable for this checkout or predict it will outperform 0.2778 or any leaderboard entry. Candidates 2–4 require external sources that are not reachable under this environment's egress policy; they are not presented as currently obtainable.

## Why H33's reported score could be high — and what remains unknown

The available file forensics suggest a plausible metric-alignment explanation: thinning a dense backbone and excluding dots near mapped catalogue faults can reduce redundant prediction mass while retaining spatial support near withheld structures. If the competition score uses a distance-weighted Tversky-like overlap, a sparse prediction can score better than a dense raster even without adding geological information. This is a **mechanistic hypothesis**, not a demonstrated causal explanation. The exact named TIFF-to-score association is unverified, the site reports no organizer score receipt, and the scores supplied for sibling sites are owner-reported unless an organizer receipt is attached. The answer to “can this repo beat that number?” is **unknown**; the current protocol prevents a valid test, and no future score is promised. A leaderboard result is called **ORGANIZER-CONFIRMED** only when copied from an organizer submission-page receipt tied to the file; this review has no such receipt.

The appropriate comparison, when unblocked, is a matched spatial hide-and-recover test: hold out whole fault segments plus buffer; derive every catalogue feature only from visible segments; mask visible catalogue pixels exactly; run each feature-alone leakage canary; use the pinned shared evaluator/writer and matched folds/dot budget; report pooled HOLDOUT-DTI with evaluator version, withheld-positive count and 95% CI; compare against the matched current best. A projection is never a score. Only a separate selector and organizer receipt can clear a real competition slot.

## Verification passes

### Pass 1 — inspect and reproduce

Reviewed the repo instructions and evidence. Installed the exact `requirements.txt` packages into ignored `.venv` only; no dependency manifest was changed. Restored the hash-pinned registry witnesses using the existing restore script. Ran `.venv/bin/python scripts/preflight_registry.py`; it reproduced `STOP_UNIVERSAL_DUPLICATE` and the proof above (expected process exit 2). No experiment, new TIFF, or slot use.

### Pass 2 — bugs, assumptions, edge cases

Checked that the rejection argument applies to both positive surface support and any final dot set restricted to the allowed domain; because all allowed cells are within the witness dilation, every nonempty subset has overlap one. It does not prove every raster in the full registry is covered, nor does it support a passing uniqueness claim. The preflight correctly labels the scope as one rejection witness. Checked the feature-cache paths directly rather than trusting historical receipts. Reviewed source-access claims: official USGS, DOE/NREL, DrivenData and NLR pages are provided for manual review but were not freshly retrieved through the restricted sandbox network. No scientific findings are attributed to a page we did not fetch.

### Pass 3 — requirements re-check

The user asks for a unique downloadable submission TIF. This review cannot responsibly generate one while the binding rule rejects all nonempty outputs; an empty raster would not be a useful prediction or a meaningful submission. Existing downloadable files are clearly marked inspection-only and none is selector-approved. The current result is negative, with explicit limitations and a next-step order. This report and the homepage now point to each other via the generated latest-review page.

## Next work, in decision order

1. Ask the shared protocol/registry owner to resolve the dense-witness support semantics in the shared template; do not fork or weaken the gate locally.
2. Obtain and hash-verify competition data and cached stack; distinguish organizer data from owner mirrors in receipts. Confirm the actual submission grid/range specification with official material.
3. Assign one lane; preregister the cross-wavelet-coherence hypothesis and its controls before examining holdout outcomes. Run each feature alone through leakage canaries.
4. Use the unchanged pinned `gems57` hide-and-recover evaluator/writer and stop at three experiments or two hours. Require a matched HOLDOUT-DTI improvement with CI, then audit surface and final dots against the complete accessible registry.
5. Only after scientific and format gates pass, request a separate weekly selector decision. Tie any upload and score claim to a public organizer receipt and exact raster SHA-256.

## Official and project sources for manual review

- [DrivenData problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) · [competition rules](https://www.drivendata.org/competitions/306/competition-doe-gems/) · [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) · [data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/)
- [Official reference solution on GitHub](https://github.com/drivendataorg/gems-prize-reference-solution) (the pinned local source receipt covers a commit and CPU/GPU support, not score verification)
- [USGS GeoDAWN release](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and) · [USGS ComCat](https://earthquake.usgs.gov/fdsnws/event/1/query) · [USGS Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2)
- [DOE/NREL GDR 1391](https://gdr.openei.org/submissions/1391) · [NLR competition PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf) · [EPSG:32611](https://epsg.io/32611)
- Project evidence: [corrected H33 forensics](../research/why-h33-scored-02778.md) · [preflight JSON](../evidence/registry_preflight.json) · [shared template pin](../registry/template_pin.json) · [submission authorization](data/submission-authorization.json)

Links above are for review; this report does **not** claim fresh retrieval of non-GitHub official sources, organizer authentication, current leaderboard rank, source AOI product availability, or portal acceptance.

## Run card (this repository review)

```json
{
  "hypothesis": "No candidate tested. Conditional rank-1 proposal: localized multiscale cross-wavelet coherence/phase between GeoDAWN RTP/TMI and gravity may reveal a shared concealed magnetic-density boundary; no score projection.",
  "mechanism": "Repository review and candidate screen only; no feature surface, holdout, placement, or new TIFF generated.",
  "named_non_fault_process_that_could_mimic_it": "Lithologic/intrusive contact or airborne survey/regridding seam.",
  "holdout_dti": {
    "evidence_class": "NOT RUN",
    "evaluator_version": null,
    "withheld_positive_count": null,
    "value": null,
    "ci95": null
  },
  "correlation_overlap_vs_registry": {
    "evidence_class": "REGISTRY-MEASUREMENT",
    "surface_candidate": "NOT BUILT",
    "final_dots": "NOT PLACED",
    "witness_sha256": "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be",
    "allowed_pixels": 5106385,
    "allowed_pixels_within_3px": 5106385,
    "uncovered_allowed_pixels": 0,
    "nonempty_subset_forward_overlap_lower_bound": 1.0,
    "strict_stop_threshold": 0.70,
    "complete_registry_scan": false,
    "interpretation": "Candidate-independent universal rejection under current literal support rule; not a uniqueness pass."
  },
  "raster_sha256": null,
  "validator_output": {
    "candidate_tif": "NOT GENERATED; no candidate to validate",
    "no_nan_inside_footprint": "not applicable",
    "values_in_0_1": "not applicable",
    "crs_shape_transform_match": "not applicable",
    "authorization_receipt": "NOT OK TO SUBMIT; 0 files authorized"
  },
  "submission_name": null,
  "submission_note": "Review only; no new TIFF. Do not submit any existing research-only file.",
  "verdict": "negative"
}
```
