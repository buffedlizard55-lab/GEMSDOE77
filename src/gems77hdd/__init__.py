"""GEMSDOE77 Lane H77H: Downdip-Lag Hanging-Wall Potential-Field Dipole,
Mesh-Deconfounded (sigma=36 px) MT/Strain/Geothermometer Regime Gate,
De-Quantized 1 m LiDAR Scarp, and Bilateral LOCO Tip-Bridge Mechanics.
"""
from __future__ import annotations

from .allocator import (
    allocate_h77h_submodular_300m,
    build_informative_halo_matrix,
    check_informative_overlaps,
    collect_checked_registry,
)
from .fields import (
    EXT_LIDAR_RAD_NAMES,
    FORBIDDEN_GDR_COLUMNS,
    H77H_LOCO_NAMES,
    H77H_STATIC_NAMES,
    build_h77h_static_stack,
    compute_h77h_loco_features,
    measure_mt_mesh_periodicity_suppression,
)

__all__ = [
    "EXT_LIDAR_RAD_NAMES",
    "FORBIDDEN_GDR_COLUMNS",
    "H77H_LOCO_NAMES",
    "H77H_STATIC_NAMES",
    "allocate_h77h_submodular_300m",
    "build_h77h_static_stack",
    "build_informative_halo_matrix",
    "check_informative_overlaps",
    "collect_checked_registry",
    "compute_h77h_loco_features",
    "measure_mt_mesh_periodicity_suppression",
]
