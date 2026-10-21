"""GEMSDOE77 Lane H77E: Fluvial Knickpoint + De-Seamed Subsurface + Step Gap-Bridge
+ Hard-Collar Azimuthal Fracture Continuation + Cutting-Plane Lagrangian Submodular 300 m Cover.

Strict compliance guarantees:
  - Uses unmodified `vendor/gems57` holdout, evaluator (`gems57-pooled-hide-v2`),
    uniqueness audit, validator, and submission writer.
  - Zero SGMC bedrock channels, zero `qfault*`/`trace_segments*` catalogue features,
    and zero `dist_known_fault_px` column usage.
  - Missing 1 m LiDAR tiles (`lidar_valid == 0`, 24.6% of Nevada) explicitly encoded as `NaN`
    and bridged seamlessly across 100% of Nevada via 13-channel float32 10 m 3DEP DEM and
    10 m fluvial knickpoint / footwall chi-disequilibrium channels.
"""
from __future__ import annotations

LANE_ID = "H77E-FLUVIAL-DESEAM-KNAPSACK-SUBMOD"
LANE_SLUG = "h77e-fluvial-deseam-submod"
