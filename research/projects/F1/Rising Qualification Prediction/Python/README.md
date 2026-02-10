# Rising Qualification Prediction (Python)

This folder contains the production-style prediction code (FastF1/OpenF1).

## Quick start

```bash
cd "Rising Qualification Prediction/Python"
python run_prediction.py --mode qualifying --source fastf1 --year 2025 --round 1 --cache-dir .cache/fastf1
```

## Model selection
- The training step now selects the best model on historical rounds with walk-forward validation (MAE).
- Candidate models:
  - `xgboost` (if installed)
  - `hist_gradient_boosting` (scikit-learn)
  - `ridge` (scikit-learn baseline)
- If no ML dependency is available or data is too thin, the CLI falls back to heuristic ranking.

## Modes
- `qualifying`: predicts Q3 outcome (top 10) using FP1/FP2/FP3.
- `race`: predicts race top 10 once qualifying results are available.

## Data pipeline (OpenF1 + FastF1)
Build a reusable dataset for training/analysis:

```bash
cd "Rising Qualification Prediction/Python"
python run_data_pipeline.py --sources fastf1,openf1 --years 2023,2024,2025 --cache-dir .cache/f1
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
