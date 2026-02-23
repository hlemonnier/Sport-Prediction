# Rising Qualification Prediction (Python)

This folder contains the production-style prediction code (FastF1/OpenF1).

## Quick start

```bash
cd "Rising Qualification Prediction/Python"
python run_prediction.py --mode qualifying --source fastf1 --year 2025 --round 1 --cache-dir .cache/fastf1
```

`run_prediction.py` supports `--train-policy` when `--train-seasons auto`:
- `legacy_auto` (default): `Y-2,Y-1,Y`
- `rolling`: `Y-3,Y-2,Y-1,Y`
- `strict_transfer`: `Y-4,Y-3,Y-2,Y` (excludes `Y-1`)
- `frozen_preseason`: `Y-4,Y-3,Y-2`

## Model selection
- The training step now selects the best model on historical rounds with walk-forward validation (MAE).
- Candidate models:
  - `xgboost` (if installed)
  - `hist_gradient_boosting` (scikit-learn)
  - `ridge` (scikit-learn baseline)
- If no ML dependency is available or data is too thin, the CLI falls back to heuristic ranking.

## Modes
- `qualifying`: predicts Q3 outcome (top 10) using FP1/FP2/FP3.
- `race`: predicts race top 10.
  - If qualifying results are available, race uses real qualifying + predicted qualifying context.
  - If qualifying results are not available yet, race runs in FP-only mode with predicted qualifying context.

## Data pipeline (OpenF1 + FastF1)
Build a reusable dataset for training/analysis:

```bash
cd "Rising Qualification Prediction/Python"
python run_data_pipeline.py --sources fastf1,openf1 --years 2023,2024,2025 --cache-dir .cache/f1
```

Offline-only pipeline from downloaded weekends:

```bash
cd "Rising Qualification Prediction/Python"
python run_data_pipeline.py --sources local --years 2025 --output-dir data/f1/offline_pipeline
```

Outputs (default: `data/f1/` at repo root):
- `f1_dataset.csv` (+ `f1_dataset.parquet` if parquet engine is available)
- `f1_coverage.csv` (+ `f1_coverage.parquet` if parquet engine is available)

Useful flags:
- `--max-rounds 3` for a quick smoke run
- `--output-dir /custom/path` for custom export target
- `--output-format json` for machine-readable summary

## Additional notes
- `--include-standings` adds championship standings (from previous rounds) to race predictions.
- `--meeting-name` / `--country-name` can be used for OpenF1 if round indexing is ambiguous.

## Download 5 race weekends locally (FastF1)
To build local raw data for algo tests (Race, Qualifying, Free Practice, plus Sprint and Sprint Qualifying when available):

```bash
cd "Rising Qualification Prediction/Python"
python run_weekend_data_download.py --year 2025 --start-round 1 --weekends 5 --cache-dir .cache/fastf1
```

Output folder (ignored by git): `data/f1/weekends/`

## Offline predictions (no API calls)
Once `data/f1/weekends/` exists, run predictions fully offline:

```bash
cd "Rising Qualification Prediction/Python"
python run_prediction.py --mode qualifying --source local --year 2025 --round 5 --weekends-dir data/f1/weekends
python run_prediction.py --mode race --source local --year 2025 --round 5 --include-standings --weekends-dir data/f1/weekends
```

Notes:
- `--source local` uses only local CSV/JSON files under `data/f1/weekends/`.
- No FastF1/OpenF1 API request is made in local mode.

## 2026 season operational pipeline
Goal: train on historical years, then run by race-weekend phases for the new season.

Typical flow per weekend:
1. `pre-qualifying`: only FP data available -> run qualifying + race previews.
2. `post-qualifying`: qualifying done -> rerun race prediction with real qualifying.
3. `post-race`: race done -> evaluate predictions vs real qualifying/race results.

CLI:

```bash
cd "Rising Qualification Prediction/Python"
python run_live_weekend_pipeline.py \
  --phase pre-qualifying \
  --source local \
  --year 2026 \
  --round 1 \
  --train-seasons auto \
  --train-policy strict_transfer \
  --include-standings \
  --weekends-dir data/f1/weekends
```

```bash
python run_live_weekend_pipeline.py \
  --phase post-qualifying \
  --source local \
  --year 2026 \
  --round 1 \
  --train-seasons auto \
  --train-policy strict_transfer \
  --include-standings \
  --weekends-dir data/f1/weekends
```

```bash
python run_live_weekend_pipeline.py \
  --phase post-race \
  --source local \
  --year 2026 \
  --round 1 \
  --train-seasons auto \
  --train-policy strict_transfer \
  --include-standings \
  --weekends-dir data/f1/weekends
```

Artifacts are saved under:
- `data/f1/live_pipeline/<year>/round_<XX>/`
