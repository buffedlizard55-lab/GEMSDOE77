## Documentation defect in the pinned gems57 template

GEMSDOE77 consumes `src/gems57/metric.py` from commit `b99591eede63c6d45c8a896255b8cd34824fa00b`. Its binary-dot explanation says marginal truth credit `dT <= k`. That inequality is not general: one dot can improve the max credit of several truth cells, whereas `k` is the nearest-truth weight for one dot.

Counterexample at R=3 px: truth at (4,3), (4,4), (4,5); no predictions; add a dot at (4,4). `k=1`, but `dT=1+2*(2/3)=7/3`. The exact denominator change is `alpha*(dT+1-k)`, not universally `alpha`. A second redundant dot next to an already perfectly covered truth cell gives `dT=0`, `k=2/3`.

The implementation of `marginal_accept` already recomputes both denominators correctly. Requested shared-template documentation correction: remove `dT <= k`; state that `dT == k` is a special case (e.g. one newly covered truth cell), not merely a consequence of non-redundancy. No evaluator/writer or template hash is privately changed in GEMSDOE77. Regression fixtures are being added to `tests/test_marginal_explanation.py` on `arena/60aafd45-gemsdoe77`.

Separately, the dense-raster literal uniqueness gate remains binding; this issue does not request a private threshold exemption.
