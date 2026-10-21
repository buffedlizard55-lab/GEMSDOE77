"""GEMSDOE79 scale-space orientation-persistence research lane.

This lane uses only label-free transforms of the cached competition layers.  It
never reads catalogue masks, labels, prior predictions, or registry rasters while
constructing features.
"""

__all__ = ["features"]
