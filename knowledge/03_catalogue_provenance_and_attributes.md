# Knowledge 03 — what the GEMS training labels really are, and what the catalogue attributes add

Stored for reuse by later sessions. Every number below is MEASURED in this repository on
2026-10-21 unless labelled otherwise; reproduction commands are given.

## 1. The training labels are the legacy USGS Quaternary-fault traces — nothing more

* Source of the vector traces: INGENIOUS/USGS Quaternary-fault compilation
  `qfaults_ingenious_nad83conus117_2023-06-27` (GDR submission 1391, CC-BY-4.0,
  https://gdr.openei.org/submissions/1391), mirrored by the sibling repo 60GEMSDOE as
  `trace_segments_utm11.csv` (84,331 UTM-11 segments) + `qfault_attributes.csv` (22,956
  records; `record_id` = 0-based attribute row = shapefile FID — verified: slip-rate text
  matches 100 %, slip-sense 91.6 %).
* Inside the GeoDAWN footprint there are **1,126 traces**: MAPSCALE 1:250,000 = 846,
  1:100,000 = 280; FCODE (certainty) well-constrained 714 / moderately 331 / inferred 81;
  RCODE (recency) historic 39, <15 ka 343, <130 ka 267, <750 ka 27, <1.6 Ma 449;
  DIPDIRECT recorded for 460 traces (W 134, E 104, NW 73, SE 64, NE 33, SW 25, N 22, S 5),
  Unspecified/NaN 666; SLIPSENSE normal 868, right-lateral 87, left-lateral 76, NaN 95;
  total length 17,388 km; strike strongly N–NNE/NNW (length-weighted 61 % within ±30° of N).
* **No 1:10k INGENIOUS lidar re-mapping records exist inside the footprint** (the "ING" lidar
  interpretation records of the 2023 compilation are all in Utah).
* `data/raw/labels.tif` (60,988 positive pixels) is the rasterisation of exactly those
  traces: 100 % of label pixels lie within 1 px of a rasterised trace and 100 % of rasterised
  trace pixels lie within 1 px of a label; label pixels > 3 px from any trace = **0**.
  (`.cache/ext/label_increment_gt3px.npy` is all-False.)

Consequences for modelling:

1. There is no "INGENIOUS-minus-USGS" increment to learn the test-set generative process
   from (lane idea H79-2 is dead; do not retry).
2. The positional accuracy of every training trace is that of 1:100k–1:250k mapping
   (hundreds of metres). The organiser masks only the exact catalogue pixels (forum 11516),
   so expert re-mapping that moves or splits a legacy trace by 200–500 m lands in *unmasked*
   cells. Whether such relocated strands are counted as "new" truth is not disclosed
   (forum 11527).
3. Hide-and-recover holdouts measure *catalogue-vs-catalogue* geometry. Any hypothesis whose
   mechanism is "lidar-era mapping differs systematically from legacy mapping" cannot be
   fully validated by that holdout; this must be stated with the result.

## 2. The downthrown side can be read from the 100 m DEM

For each catalogue pixel, compare the mean detrended elevation (band 12 `det_elev`) 4–12 px
on either side of the local strike normal; the lower side is called the hanging wall.
Against the catalogue's own DIPDIRECT attribute (25,482 attributed pixels) this agrees
**86.1 %** of the time (`evidence/h79_holdout_seed20.json → attribute_receipt`). The 14 %
disagreement is a mixture of strike-slip/oblique traces (SLIPSENSE RL/LL = 163 traces), traces
inside basins with no topographic step at 100 m, and scarps whose footwall has been eroded
below the hanging-wall fan surface. So: `hw_topo` is a valid proxy for the 58 % of pixels
without a dip attribute.

## 3. Inside the catalogue, withheld traces lean slightly to the FOOTWALL of their neighbours

Holdout draw gems57-pooled-hide-v2, buffered component draw, 20 %, 3 px collar:

| seed | withheld px | on hanging wall of nearest visible trace (attribute) | negatives | (topographic) | negatives |
|---|---|---|---|---|---|
| 20 | 11,321 | 47.9 % (n = 4,424 defined) | 53.2 % | 46.5 % | 50.2 % |
| 30 | 11,342 | 48.3 % (n = 4,572) | 53.1 % | 47.5 % | 49.7 % |

Interpretation (PhD-level, hedged): legacy 1:250k mapping records range-front traces and their
*bedrock* (footwall) splays from geological maps, while piedmont strands in alluvium on the
hanging wall were rarely mapped before lidar. The catalogue therefore under-represents the
hanging wall; a hide-and-recover test on that catalogue will show a footwall lean even if
the test set (expert lidar mapping) leans the other way. The attribute channels still help the
model (see `evidence/h79_run_card.json`), mostly through recency/slip-rate conditioning of
proximity rather than through the signed hanging-wall distance alone.

## 4. Reproduction

```
.venv/bin/python scripts/build_features.py                 # shared stack (8 min, < 2.6 GB RAM)
.venv/bin/python scripts/fetch_registry_git.py             # 905 registry rasters via git, no REST quota
.venv/bin/python scripts/h79_holdout.py --seed 20          # E1 canaries + E2 arms (8 min)
.venv/bin/python scripts/h79_holdout.py --seed 30 --arms A2_standard,A2_DIST,H79_full,H79_noHW,H79_noATTR,ATTR_only
.venv/bin/python scripts/h79_emission_check.py --seed 20   # post-hoc greedy vs submodular
.venv/bin/python scripts/h79_build.py --name <name> --note "<note>"
```
