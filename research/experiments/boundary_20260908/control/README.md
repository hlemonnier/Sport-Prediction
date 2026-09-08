# Global track-control acquisition — research paused

Status: acquisition and parser verification are complete. The integration request paused this lane **before feature engineering, model-grid freeze, fitting, selection, or predictive score inspection**. No 2024–2026 control streams were acquired. There is no candidate or predictive improvement to integrate from this directory.

The 44 discovery race streams (2022–2023) reuse the audited weather manifest's exact meeting/session mapping. Public Formula 1 timing is attempted first, followed by FastF1's configured public mirror. Raw responses and actual-URL acquisition receipts are retained. The first sandbox-only attempt failed DNS lookup; its manifest, attempt and log remain under `failed_sandbox_dns_*` and are not successful input evidence.

The successful acquisition contains 569 records: clear 254, yellow 209, safety car 39, VSC deployed 32, VSC ending 24, red flag 11. No records have absent/unknown Status. One pair of ordered status updates shares a timestamp (202216); both records are preserved. All 132 raw/CSV/receipt hashes match, and the raw plus parsed data and receipts occupy 68,285 bytes. Every timestamp, status and message matches the installed FastF1 3.8.3 native parser exactly across all 569 records.

The stream's first 12 characters encode the session clock used by native timing data; no UTC or race-start offset is applied. Status changes are persistent until a later change. Status `7` is VSC ending, still neutralized until an observed clear status. Unknown status remains unknown; an omitted Status and explicit unknown update remain distinguishable. The parser performs no future fill. Download receipts establish source provenance, **not historical consumer receipt time**. A proposed follow-up would use strict `recorded_time < checkpoint_time - 15 seconds`, with other delays treated as sensitivity assumptions; that model protocol has not been frozen or run.

Files:

- Source: `acquire.py`, `verify_acquisition.py`, `test_acquisition.py`.
- Successful manifest and verification: `artifacts/research/boundary_20260908/control/input_manifest.json` and `acquisition_verification.json`.
- Inputs: `data/f1/boundary_20260908/control/` (44 raw streams, 44 parsed CSVs, 44 source receipts).
- Acquisition scope, execution log and preserved failed attempt: the same artifact directory.

Verification (5 tests passed; pandas emitted 9 dependency deprecation warnings):

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/control/verify_acquisition.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/control/test_acquisition.py
```

Suggested commit: `research(f1): preserve verified track-control discovery inputs`.
