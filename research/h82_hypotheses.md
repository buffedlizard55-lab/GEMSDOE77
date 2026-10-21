# H82 candidate hypotheses (preregistered before Exp 1 in registry/preregistration_h82.json)

Status: written 2026-10-21, before any H82 experiment. None of these has been implemented in this repo
before (see difference-vs-repo per hypothesis). The lane is geometry-only: it needs only the SHA-pinned
bridge grid (`data/bridge/sample_submission.tif`, `existing_faults.tif`, `labels.tif`), so Exp 1 runs
with no external data and no training. Hypotheses are ordinal priorities, never projected scores.

## Rank 1 (implemented in Exp 1): hanging-wall fault-damage collar, kernel scale

- **Layers:** USGS/INGENIOUS mapped-fault catalogue (`existing_faults.tif` == `labels.tif`, 60,988 fault
  pixels on the 3730×3292 EPSG:32611 100 m grid); competition footprint from `sample_submission.tif`.
- **Physical signature:** active normal faults are bounded by a damage zone. Blind or subtly expressed
  strands that never entered the Quaternary fault catalogue should still cluster off the mapped trace
  at roughly the damage-zone scale. The official DTI kernel is triangular with radius 300 m (3 px), so
  the collar profile is zero on the trace, rises to a broad maximum around 100–400 m (peaking near one
  kernel radius), and decays to zero by ~1.2 km. Per holdout fold the distance transform is computed
  from the **visible** faults only; visible faults are masked pixel-exactly from scoring; dots never
  land on the mapped catalogue.
- **Why it catches uncatalogued faults:** catalogue mapping under-represents low-relief, blind, and
  young splays running parallel to mapped traces under Quaternary piedmont cover; a damage-zone collar
  targets off-trace upflow without re-predicting mapped faults (which the metric fully penalizes,
  organiser thread 11516).
- **Named non-fault mimic:** Pleistocene Lake Lahontan (1332 m) / Bonneville (1552 m) wave-cut terrace
  risers, symmetric desert washes/arroyos, road and canal cuts — all linear but not dislocation damage
  zones; the collar has no kinematic or lithologic discriminator by itself, so it can fire on these.
- **How it differs from the repo:** D77 tested a flat 250 m–1 km off-catalogue distance band (0.0000
  HOLDOUT-DTI); H82 uses a kernel-scale peaked collar with an on-trace trough, per-fold visible-faults
  only features, buffered whole-component hide-and-recover, and 2.8 px dot separation. H78/H77-Step/H77D
  use lidar, geophysics, LOCO or step tensors; H82 uses catalogue geometry only (zero SGMC, zero lidar,
  zero external data).
- **Expected DTI gain ÷ cost:** uncertain, low–medium ÷ very low (CPU raster ops, minutes).

## Rank 2 (not implemented): catalogue fault-intersection / relay-node intensity

- **Layers:** catalogue centreline network only (local).
- **Physical signature:** kernel density of fault intersections and relay bends (damage focusing,
  permeability contrast at stepovers) — blind upflow nodes need no surface scarp.
- **Why uncatalogued:** the target is the *node*, not a trace: a small blind fault terminating at a
  mapped strand may be unmapped while the node is real.
- **Mimic:** mapped intersections of inactive traces; drainage junctions.
- **Differs:** no lane has used node/intersection intensity as primary scorer.
- **Cost:** very low; needs no external data. Viable next experiment if H82's collar shows any live
  off-catalogue credit.

## Rank 3 (not implemented): K/Th radiometric asymmetry across catalogue traces

- **Layers:** GeoDAWN radiometrics (K, Th, K/Th) from USGS DOI 10.5066/P93LGLVQ (uint8 grid
  `ext_geodawn_rad_u8` / `ext_geodawn_extensions_u8`, SHA-pinned in registry/data_manifest.json, ~26–27 MB,
  obtainable from the owner's GitHub mirror — NOT locally present, not restored in this sandbox).
- **Physical signature:** hydrothermal illitization concentrates K and leaches K/Th asymmetry across a
  fault; first derivative of K/Th normal to local strike, signed by hanging-wall side.
- **Why uncatalogued:** alteration halo can outcrop where the fault trace itself is buried below
  mapping scale.
- **Mimic:** primary lithologic boundaries (K/Th contrast without displacement), flight-line noise.
- **Differs:** H77G used conductance/magnetics; H79-RH paired radiometric halos; no lane used
  signed across-strike K/Th asymmetry as primary.
- **Obtainability check:** source is free/official; not reachable from this sandbox's outbound allowlist
  (github.com only for mirrors), so marked NOT VIABLE IN THIS SESSION — name the source for the next
  session with GitHub-mirror restore (scripts/restore_data.py, registry/data_manifest.json).

## Rank 4 (not implemented): multi-scale residual curvature (detrended DEM)

- **Layers:** detrended elevation + curvature bands from the pinned 54-channel feature stack
  (`data/processed/stack.npy`, not restored locally; 418 MB training_features.tif pinned in manifest).
- **Physical signature:** curvature of the DEM after removing the regional piedmont trend at
  300 m–1.2 km scales — youthful fault-line antiforms vs broad basin tilting.
- **Why uncatalogued:** sub-mapping-scale scarps survive in curvature texture.
- **Mimic:** fan cones, terrace treads, playas.
- **Differs:** Selective-U used raw U-rank curvature; this removes the regional trend first and
  tests multiple scales.
- **Cost:** medium (needs restored stack; FFT on 12 Mpx grid).

## Rank 5 (not implemented): thermal spring/well alignment anisotropy (GDR 1391)

- **Layers:** DOE/NREL GDR submission 1391 well & spring temperatures (DOI 10.15121/1881483, CC BY 4.0).
- **Physical signature:** linear alignment of warm springs/wells perpendicular to regional extension
  (Hough/kernel-density anisotropy), blind upflow without scarp.
- **Why uncatalogued:** thermal memory survives where geomorphic expression is buried.
- **Mimic:** basin-fill evaporative brine anomalies, irrigation return flow.
- **Differs:** H78 used distance-to-manifestation proximity only; this adds alignment anisotropy.
- **Obtainability:** gdr.openei.org is not in this sandbox's outbound allowlist — NOT obtainable here;
  name the specific free official source for the next session.

Ranking by expected DTI gain ÷ implementation cost: H82 (Rank 1) > Rank 2 > Rank 3 > Rank 4 > Rank 5.
Exp 1 validates Rank 1 on the spatially blocked holdout before any submission-slot consideration.