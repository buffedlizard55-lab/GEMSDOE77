# Shared template, not a private fork

`gems57/` is byte-identical to `src/gems57/` in
[buffedlizard55-lab/60GEMSDOE at 32024716f218d06a94e01464157392119c03cb15](https://github.com/buffedlizard55-lab/60GEMSDOE/tree/32024716f218d06a94e01464157392119c03cb15/src/gems57).

Restored through GEMSDOE69's byte-pinned vendor cache. Every file SHA-256 is
checked by `scripts/audit.py`; the upstream Git tree/blob identity was checked
independently and recorded in `evidence/template-upstream.json`.

Reuse `evaluate_holdout.py`, `submission_writer.py`, `holdout.py`, and the metric
without modification. We did not discover or apply a shared-code fix in this
session. A future shared defect must be fixed upstream once, not edited here.
Passing `data_dir` explicitly avoids the historical vendored-root ambiguity.

Inherited module docstrings contain historical reports; these are not newly
verified performance results. The metric docstring's aside that marginal truth
credit cannot exceed self-credit is not generally valid for a dot covering
multiple truth pixels. The algebra below that aside uses both terms; this audit
does not rely on that simplifying assumption or change the shared implementation.
