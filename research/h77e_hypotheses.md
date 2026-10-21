# GEMSDOE77 Lane H77E — candidate geological hypotheses (ranked BEFORE experiments)

*Written 2026-10-21 before Exp 1 of lane H77E. This session's brief contained no
identifiable single method paragraph (same irregularity as IR-77-06), so the lane
is self-declared here and in `registry/preregistration_h77e.json` BEFORE any
experiment. Evidence classes: **[OFFICIAL]** organizer/USGS/peer-reviewed source ·
**[MEASURED]** computed in this repo from SHA-pinned bytes · **[OWNER-REPORT]**
family ledger without organizer receipt.*

Competition anchors (unchanged): predict faults **not** in the USGS/INGENIOUS
catalogue; metric = distance-weighted Tversky (DTI), α=0.2, β=0.8, triangular
kernel R=300 m; grid EPSG:32611, 100 m, 3730×3292 **[OFFICIAL]**
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/).
Family best `h33-h33-2-b2…-zeros` = 0.2778 **[OWNER-REPORT]**; public leader
0.4413 **[OFFICIAL leaderboard snapshot in README]**. Current holdout best in
this repo: H77-Step 0.15272 HOLDOUT-DTI (gems57-pooled-hide-v2, seed 20) and
H77D 0.15011 (seed 30) / 0.14030 (seed 20) — promotion bar for this lane.

## Ranked candidates

### H77E-1 (rank 1 — PRIMARY LANE) — Upward-continuation persistence + Euler depth coherence of potential-field source edges
- **Layers:** `training_features.tif` band 2 (`rtp`), band 14 (`tmi`), band 13
  (`iso_grav_anom`), band 3/9 (`tmi_hg`/`tmi_vg`, measured/derived gradients).
  Band 6 is **not** used as "tilt": it is radiometric total count **[MEASURED,
  evidence/band6_resolution.json]**.
- **Physical signature:** Fourier upward continuation of RTP and isostatic
  gravity to h = 200/400/800 m (`exp(-|k|h)` — continuation suppresses shallow
  short-wavelength sources and retains deep sources [1](https://wiki.seg.org/wiki/Dictionary:Upward_continuation));
  at each height, analytic-signal / horizontal-gradient edge strength,
  percentile-normalised per height; **persistence** = mean edge strength across
  heights and **deep edge** = 800 m edge strength. Euler deconvolution
  (structural index 0, contact) on RTP gradients (Reid et al. 1990,
  *Geophysics* 55, 80–91, DOI 10.1190/1.1442774 [2](https://chooser.crossref.org/?doi=10.1190%2F1.1442774)):
  windowed least-squares source locations with depth 50–1500 m, rasterised as
  solution density and depth coherence (local depth agreement). Cross-field
  term = elementwise minimum of magnetic and gravity persistence (an edge must
  survive in **both** fields).
- **Why a catalogue-missing fault:** an intrabasin normal fault buried under
  100–1500 m of alluvium has no scarp for surface/Qfaults mapping, but it offsets
  magnetic, dense basement; that offset is a **deep** source, so its edge
  persists under upward continuation while shallow lithologic/cultural edges
  fade. An expert mapping from GeoDAWN aeromagnetics can draw it; the legacy
  surface catalogue cannot contain it.
- **Named non-fault mimic:** intrusive/lithologic basement contacts and caldera
  margins (also deep, persistent, Euler-coherent — indistinguishable by this
  signature alone); GeoDAWN survey-block seams (straight, acquisition-parallel,
  also persistent). These cap the achievable precision and are stated, not hidden.
- **Difference vs implemented work in this repo:** contact-physics/H77D used
  native-resolution tilt zero-crossings, analytic signal and cross-gradients as
  channels inside a broad 88–92-channel HGB; H77-Step "worms" are unsharp-mask
  high-passes of scarp/gravity/magnetic gradient at native resolution. **No**
  in-repo lane computes Fourier upward-continued fields at multiple heights,
  persistence across heights, or an Euler solution-density/depth field.
  (Family sites outside this repo reported Euler/tilt-worm variants at low
  live scores — GEMSDOE40 0.0355–0.0984, GEMSDOE42 0.0581 **[OWNER-REPORT]** —
  which lowers the prior; the registry correlation/overlap gate below is the
  binding uniqueness test, and a duplicate reading stops this lane.)
- **Cost:** low–medium (FFT filter bank + vectorised windowed Euler on cached
  bands; no new data). Validatable locally: yes (hide-and-recover).

### H77E-2 (rank 2 — tested as an added block inside Exp 2, not a separate lane) — Fault damage-zone magnetic texture suppression
- **Layers:** band 2 (`rtp`), band 14 (`tmi`).
- **Signature:** local roughness (5×5 and 11×11 std of RTP) **suppressed** in an
  elongated corridor (structure-tensor anisotropy of the roughness field):
  hydrothermal/magnetite destruction + fractured damage zone smooths the
  magnetic texture over a 200–500 m wide zone rather than producing a line edge.
- **Why missing:** a blind fault with no basement offset step can still host a
  fractured, altered damage zone; surface mapping sees neither scarp nor contact.
- **Mimic:** thick uniform alluvial cover (also smooths magnetics — the main
  confounder; basin-wide smoothness must be normalised out by the 11×11/51×51
  roughness ratio), sedimentary lithologic units.
- **Difference:** H77D used a single `TMI_up150 − RTP` demagnetisation channel;
  no in-repo lane uses multi-scale roughness-ratio anisotropy as a block.
- **Cost:** low. Validatable: yes.

### H77E-3 (rank 3 — not implemented unless Exp 2 leaves budget) — Antisymmetric basement-step matched filter
- **Layers:** band 15 (`depth_to_base_surf`), band 13 (`iso_grav_anom`).
- **Signature:** for 8 azimuths, the normalised mean difference of basement
  depth across a 1 km line (hanging-wall minus footwall side), i.e. a matched
  template for a buried fault step, gated by gravity-step agreement.
- **Why missing / mimic / difference:** buried step with no surface expression /
  unfaulted basement palaeo-topography and intrusive margins / H77-Step's step
  tensor operates on DEM elevation and lidar scarps, contact-physics on tilt
  coincidence — neither matched-filters the basement-depth field itself.
- **Cost:** medium (8 azimuth convolutions × 2 fields). Validatable: yes.

### H77E-4 (considered, REJECTED as not new) — Drainage χ / knickpoint lineaments
Already implemented inside this repo (`CHI_BAND_NAMES` in
`src/gems77_step/features.py`: chi_step, ksn, log-area). Re-using it would be a
re-run of the H77-Step lane, which AGENTS.md forbids tuning/re-running.

### H77E-5 (considered, BLOCKED here) — Sub-km seismicity lineaments
Needs USGS ComCat/FDSN (`earthquake.usgs.gov`), outside this sandbox's egress
allow-list (measured HTTP 000 in `research/hypotheses.md`, H77-5); shipped
bands 10/16 are 100 km-radius aggregates and cannot localise a fault at 300 m.
Named, not proposed as viable in this session; no seismicity surface fabricated.

## Decision rule (frozen in `registry/preregistration_h77e.json`)
Exp 1 builds the H77E-1 (+H77E-2 block) fields label-free and runs single-feature
leakage canaries (symmetric AUC > 0.90 = leakage, stop). Exp 2 is a dual-seed
hide-and-recover holdout (gems57-pooled-hide-v2, whole-component buffered draw,
visible faults masked pixel-exactly, catalogue-based features derived only from
visible faults) comparing: standard raw-band HGB, H77E-only HGB, combined,
combined+catalogue-context (LOCO-style distance/tip distance from visible
faults only), plus fold-fair H19-5 champion construction and random controls,
all emitted at B=37,654 dots, >3 px from visible catalogue, via the shared
300 m submodular allocator. Exp 3 builds the full-map cross-fitted candidate,
checks surface (before placement) and final-dot uniqueness against the registry
(ρ ≤ 0.90, forward 3-px overlap ≤ 0.70; literal dense-witness reading reported
separately per Policy I17), and packages it with the shared fail-closed writer.
Promotion requires beating the current holdout best (0.15272) with paired CIs —
otherwise the verdict is **negative**, the TIFF is download-only research, and
no weekly slot is claimed. Selector approval is a separate human step regardless.
