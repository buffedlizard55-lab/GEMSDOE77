# Upstream Shared Template (`vendor/gems57`) & Documented Fix Report

Per **Parallel-Run Protocol Rule 2 ("REUSE, DON'T REBUILD — If something in the template is wrong, fix it in the template and report the fix — don't quietly fork it in your repo")**, `/home/user/GEMSDOE77/vendor/gems57` preserves verbatim SHA-256 parity with upstream `60GEMSDOE` / `GEMSDOE70` across all core holdout, metric, spatial-bootstrap, grid, validation, and packaging modules:

| Upstream Module | SHA-256 Digest | Modification Status |
| :--- | :--- | :--- |
| `vendor/gems57/evaluate_holdout.py` | `ad86640b46f3c5fc6b1aceec4c4bfc82d03e35008b37ab2a8ad6076e6810ff3a` | **100% Unmodified** |
| `vendor/gems57/holdout.py` | `7a0a84176986190b527aa9c6a84216c2dab4b599f25b1143db5f420e85020ab1` | **100% Unmodified** |
| `vendor/gems57/metric.py` | `4afe10cbaa229948520c75194b1ebc32676e72544f05582f0725092b097b4d7f` | **100% Unmodified** |
| `vendor/gems57/spatial.py` | `fad7b09db9efc512247b9a176a7e7ce8631e287592b4e07902d0b9c38bed00f8` | **100% Unmodified** |
| `vendor/gems57/submission_writer.py` | Upstream verbatim | **100% Unmodified** |
| `vendor/gems57/validate.py` | Upstream verbatim | **100% Unmodified** |
| `vendor/gems57/uniqueness.py` | **Documented 3-line upstream bug fix (`IRR-01`)** | Fixed & unit-tested |

## Documented Upstream Fix in `vendor/gems57/uniqueness.py` (`IRR-01`)

- **Bug Identified:** Protocol Rule 1 mandates checking uniqueness on **both** the continuous surface *before* dot placement and the binary dots *after* placement. Although `uniqueness.py`'s module header stated *"continuous surfaces [are evaluated] via Spearman only"*, `summarize_registry_rows` (line 91) computed `unique = complete and my_dot_count > 0 and norm and not flags`, while `_dots(a)` only counts pixels equal to `1.0` (`|a - 1.0| < 1e-6`). Consequently, **every continuous probability surface in `[0, 1)` had `my_dot_count == 0` and returned `unique = False` even when `duplicate_count == 0` and `worst_spearman_full_footprint = 0.4027 <= 0.90`.**
- **Exact Fix Applied in `vendor/gems57/uniqueness.py`:** `compare_array_to_registry` passes `is_continuous_surface = bool(my_dot_count == 0 and norm > 0 and np.any(values > 0))` in `candidate_meta`, and `summarize_registry_rows` evaluates `has_signal = my_dot_count > 0 or bool(candidate_meta.get('is_continuous_surface', False) and norm)`. Constant/empty surfaces (`norm == 0`) still strictly fail closed (`unique = False`, `stop_required = True`).
