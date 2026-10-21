"""Lane H77E: Survey-Deconfounded Fault-Valve & Fluvial Chi-Scarp Multi-Family Ensemble
with Exact Bayes-Calibrated Dinkelbach Submodular DTI Allocation (`GEMSDOE77`).
"""
from .dinkelbach import (
    allocate_dinkelbach_submodular_dti,
    bayes_prior_calibrate,
    marginal_dti_threshold,
)
from .features import (
    ALL_CHANNEL_NAMES,
    BLOCKS,
    FORBIDDEN_COLUMNS,
    STATIC_CHANNEL_NAMES,
    VISIBLE_CHANNEL_NAMES,
    build_static_fp_memmap,
    compute_visible_structural_fp,
)

__all__ = [
    "ALL_CHANNEL_NAMES",
    "BLOCKS",
    "FORBIDDEN_COLUMNS",
    "STATIC_CHANNEL_NAMES",
    "VISIBLE_CHANNEL_NAMES",
    "allocate_dinkelbach_submodular_dti",
    "bayes_prior_calibrate",
    "build_static_fp_memmap",
    "compute_visible_structural_fp",
    "marginal_dti_threshold",
]
