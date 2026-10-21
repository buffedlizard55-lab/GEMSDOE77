"""H77H (Piedmont-Bajada Alluvial-Fan Toe Pinchout, Synthetic/Antithetic Scarp Polarity & En-Echelon Sinter-Relay Bridge)."""
from .fields import (
    H77H_LOCO_NAMES,
    H77H_STATIC_NAMES,
    build_h77h_static_fields,
    compute_h77h_loco_channels,
    elliptical_tip_relay_bridge,
    lahontan_shoreline_suppressor,
    piedmont_tail_slope_anomaly,
    scarp_polarity_ratio,
)

__all__ = [
    "H77H_STATIC_NAMES",
    "H77H_LOCO_NAMES",
    "piedmont_tail_slope_anomaly",
    "lahontan_shoreline_suppressor",
    "scarp_polarity_ratio",
    "elliptical_tip_relay_bridge",
    "compute_h77h_loco_channels",
    "build_h77h_static_fields",
]
