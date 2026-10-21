# Inputs

The competition data page is login-walled in this environment. `scripts/restore_data.py` can restore the SHA-pinned owner-hosted mirrors in `registry/data_manifest.json` to `data/raw/`; these hashes establish mirror integrity, **not organizer authentication**.

A tracked `data/restore_receipt.json` is a historical record of a previous restore operation. It does not prove that the ignored files are present in the current checkout. Before training or holdout work, run:

```bash
.venv/bin/python scripts/check_data_ready.py
```

This read-only check compares current file presence, byte counts, and SHA-256 values with both the manifest and matching receipt rows. It exits non-zero if files are missing, stale, altered, or if the receipt does not match. For one input group, pass `--ids training_features,labels,existing_faults,sample_submission`.

If the check fails and a lane has been explicitly assigned, restore the required pinned inputs with `python scripts/restore_data.py --ids <comma-separated-ids>` (or `all`), then re-run the check. The contest download script is not a credential bypass. Do not treat a receipt alone as proof that the feature stack exists; `data/processed/` is also ignored and must be rebuilt from the restored inputs when required.

## Legacy workflow note

The earlier pilot instructions were `bash scripts/download_competition_data.sh` followed by `.venv/bin/python scripts/prepare_data.py`. That path may require authenticated competition access and must not be treated as a public-data bypass. The pilot used cached GeoDAWN reductions; it did not rebuild the feature stack or use old submission predictions.
