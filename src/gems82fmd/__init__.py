"""H82-FMD: fault mapping-deficit (censored catalogue-intensity residual) lane.

The lane is self-contained on the 19 shipped competition bands plus eight
derived channels; it reads no historical submission raster and no external
layer, so the deficit model cannot inherit another lane's confounds.
"""

__all__ = ["fields", "deficit"]
