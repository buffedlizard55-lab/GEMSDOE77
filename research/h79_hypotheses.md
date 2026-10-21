# H79 — hypotheses generated before implementation (2026-10-21 session)

Ranked by expected HOLDOUT-DTI gain ÷ implementation cost, as the brief requires. The
machine-readable, time-stamped version is `registry/preregistration_h79.json`; the
decision rule there was locked before E1 ran.

Context that shaped the ranking (all MEASURED this session, see `docs/irregularities.md`
IR-79-04): the competition page says the training labels are "USGS + INGENIOUS" faults,
but inside the GeoDAWN footprint `labels.tif` is pixel-identical (100 % within 1 px, both
directions) to the rasterised **legacy** USGS Quaternary-fault traces — 1,126 traces, all
mapped at 1:100,000 (280) or 1:250,000 (846). The INGENIOUS 1:10k lidar re-mapping records
exist only in Utah. Consequences: (i) there is no "catalogue increment" to learn from
(H79-2 dead on arrival), (ii) the location accuracy of every training trace is 1:100k–1:250k,
and (iii) the per-trace *attributes* of those traces are a public layer no lane has used.

| Rank | ID | Hypothesis | Layers (free/official source) | Physical signature | Why it catches catalogue-missing faults | How it differs from the repo | Expected gain ÷ cost |
|---|---|---|---|---|---|---|---|
| 1 | **H79-1** (primary) | Attribute-conditioned hanging-wall strand prior | qfaults attributes DIPDIRECT / RCODE2023 / SCODE2023 / FCODE2023 / MAPSCALE (INGENIOUS qfaults compilation, GDR 1391, CC-BY-4.0: https://gdr.openei.org/submissions/1391); existing_faults.tif; det_elev (band 12); the 19 shipped bands | Signed hanging-wall distance to the nearest *visible* trace (dip attribute, or the topographically downthrown side where the attribute is unspecified — the two agree on 86.1 % of 25,482 attributed pixels, MEASURED), along/across-strike offset, recency/slip/certainty/scale of the nearest visible trace, activity-restricted distances | A 1:250k line stands for a fault *zone*; lidar-era mapping adds synthetic/antithetic strands on the downthrown side and along-strike continuations that the legacy map omitted by scale | H77-Step used DEM-only step asymmetry; H77D/LIA/H78 used isotropic visible distance + crack-tip lobes; nobody used dip direction, recency, slip rate, certainty or map scale | MODERATE ÷ LOW |
| 2 | H79-2 | Learn the INGENIOUS-minus-USGS increment directly | labels.tif vs rasterised qfaults traces | The generative process of the test set | By construction | No lane split labels by provenance | HIGH ÷ TRIVIAL — **but the increment is 0 px (E0, MEASURED) → dead** |
| 3 | H79-3 | Sub-km seismicity lineaments from raw hypocentres | USGS ComCat FDSN (https://earthquake.usgs.gov/fdsnws/event/1/), Nevada Seismological Lab catalogue | Strike-aligned kernel density of M ≥ 1.5 events; shipped deq/ieq bands (n100/a15) smear lineaments | Microseismicity lights up blind/buried structures without scarps | No lane used raw hypocentres | MODERATE ÷ LOW — **blocked**: sandbox egress is github/pypi only (verified), so obtainability could not be exercised; named for next session |
| 4 | H79-4 | Map-scale location-uncertainty halo | MAPSCALE of nearest visible trace | 1:250k traces carry hundreds of metres of positional error | Only exact catalogue pixels are masked (forum 11516), so offset strands count as new truth | Prior lanes pruned ≤ 200 m from the catalogue regardless of scale | LOW ÷ TRIVIAL (live evidence H27-4/H33: deleting dots ≤ 200 m *raised* the score) — kept only as an ablation column (`ctx_mapscale_near`) |
| 5 | H79-5 | Activity-weighted anisotropic strike-continuation kernel from trace tips | trace strike + RCODE/SCODE | Elliptical along-strike kernel from visible tips, weighted by recency/slip | Tips propagate; legacy mapping stops where scarps get subtle | H77D crack-tip lobes, h16-continuation (0.0461 live), tip-relay (0.0159 live) exist, never activity-weighted | LOW ÷ LOW — not run |

## What the holdout can and cannot say about H79-1

The hide-and-recover holdout withholds whole **catalogue** components and asks the model to
recover them from the rest of the catalogue plus geophysics. For H79-1 that means it measures
*catalogue-vs-catalogue* geometry: where legacy 1:250k traces sit relative to other legacy
traces. The mechanism, however, is about where **lidar-era** strands sit relative to legacy
traces. If the legacy map systematically under-represents hanging-wall piedmont strands (which
is exactly the gap lidar fills), the holdout will show a *weak or reversed* hanging-wall
asymmetry even when the mechanism is right for the test set. E1 therefore reports the
asymmetry descriptively (seed 20: withheld traces lie on the hanging wall of their nearest
visible trace 47.9 % (attribute) / 46.5 % (topographic) of the time vs 53.2 % / 50.2 % for
negatives) — a small *footwall* lean inside the catalogue. The ranked verdict for slot
promotion still follows the pre-registered holdout rule; the final-round argument (expert
re-review of all submissions, forum 11527) is recorded as a limitation, not as evidence.
