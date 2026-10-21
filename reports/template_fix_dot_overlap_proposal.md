# Proposal: scope the literal 3-px dot-overlap test to placement rasters (shared-tool fix, NOT applied)

*Status: PROPOSAL for the protocol owner. `vendor/gems57` is UNMODIFIED
(pin-verified); no private exemption is claimed; every lane in this repository
remains evaluated under the literal reading. Per parallel-run rule 2 ("if a
shared tool is wrong, fix it once in the template and report it"), this file is
the report; the patch must land upstream with owner approval, never as a local
fork.*

## The defect

`vendor/gems57/uniqueness.py` defines dot support as "positive finite pixels"
(`_dots`) for BOTH the candidate and every registry raster, then applies the
protocol's "more than 70% of your dots fall within 3 px of one registry
raster's dots" test uniformly. But the protocol sentence presupposes the
registry raster HAS dots (a placement/emission raster). For a continuous
soft-surface registry raster that is positive on every footprint cell, the test
degenerates: EVERY non-empty candidate — including uniform-random dots, the
H33 champion construction, and any future lane — records forward overlap 1.0
and "fails". A gate that cannot pass is not a uniqueness test; it is a
representation-policy contradiction (this repo's Policy I17 / IR-L-03: witness
`ab0a0a62…` positive at all 5,106,385 allowed cells).

Consequences measured in this repo: every lane (H77-Step, H77D, H78, LIA, H77E,
H79-BB, H79-RH, H77L, D77, XSEN, contact-physics, selective-U) fails the
binding literal gate on dense witnesses regardless of geological dissimilarity,
so the repository authorizes 0 files and the gate carries no information about
lane drift — the thing it was built to catch.

## Proposed fix (upstream, one place)

In `summarize_registry_rows` / `compare_array_to_registry`, apply the 3-px
forward-overlap duplicate flag ONLY against registry rasters whose decoded
support is placement-like (sparse). Continuous/dense registry rasters remain
compared by full-footprint rank correlation (ρ > 0.90 flag), which is the
correct similarity measure for surfaces. Concretely:

- `placement_like = their_dots <= DOTS_MAX` with `DOTS_MAX` fixed in the
  template (suggested: 250,000 ≈ 5% of the footprint — above every binary
  dot-submission budget observed, far below any full-support surface);
- `duplicate_by_overlap` fires only when `placement_like` is true;
- dense-raster comparisons keep `duplicate_by_rho`, byte/pixel identity flags,
  and Jaccard diagnostics unchanged;
- the summary reports both counts (`dense_comparisons_skipped_for_overlap`,
  `placement_comparisons`) so the exemption is auditable, never silent.

This preserves the anti-drift intent (a copied dot pattern still fires via
overlap AND identity) while restoring informativeness (dense surfaces can no
longer veto everything).

## Why this is a tool bug, not a policy preference

1. The protocol text says "one registry raster's **dots**". A whole-footprint
   soft surface has no dots; decoding each of its 5.1M positive cells as a dot
   is the tool's invention, not the protocol's.
2. The tool ALREADY treats dots and surfaces differently for correlation (fast
   binary path vs `rankdata` path) — the overlap leg is the inconsistent one.
3. Precedent in-repo: H78/LIA/H77L run cards all record the same universal
   firing with 0.9x–1.0 overlap against dense witnesses while pairwise sparse
   overlaps stay ≤ 0.66 — the dense leg adds no discrimination.

## Effect demonstration (H81 clean-E3 numbers, `evidence/h81_e3_uniqueness.json`)

- 832 rasters compared; worst |rho| = 0.1333 (7GEMSDOE halo15 GBT surface) —
  the correlation leg PASSES with wide margin.
- Worst forward overlap overall = 1.0000 (25 full-support dense witnesses) —
  the literal overlap leg FAILS; duplicate_count = 69 (all overlap-leg, zero
  rho-leg, zero identical).
- Worst among placement-like non-lattice rasters (`their_dots <= 250000`) =
  0.8927 (59GEMSDOE loco-ipp-corridor, rho +0.04); 17 placement-like rasters
  exceed 0.70, ALL with |rho| <= 0.05. So even under the proposed scoping H81
  would still stop on the overlap leg — the dense-witness fix restores
  informativeness but does not clear this lane, because independently-built
  fault-following dot sets geometrically overlap. The operative verdict NEVER
  changes locally: this section is measurement, not clearance.
- Closest thematic neighbor: GEMSDOE53 tiprelay-ridgeconcord (ov 0.827,
  rho 0.03, different data/mechanism) — independent convergence, IR-81-11.
- Champion H33 final dots: rho +0.001, overlap 0.172, Jaccard 0.004 — the H81
  lane is decisively NOT a copy of the incumbent.

## Suggested upstream acceptance test

A uniform-random 37,654-dot raster must PASS the overlap leg against the dense
witness set (it cannot be a "copy" of anything) while a byte-shifted copy of a
registered dot raster must still FAIL (overlap + identity). Both directions are
checkable in CI without new data.
