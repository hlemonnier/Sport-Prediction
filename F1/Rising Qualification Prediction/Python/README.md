# Rising Qualification Prediction (Python)

This folder contains the production-style prediction code (FastF1/OpenF1).

## Quick start

```bash
cd "Rising Qualification Prediction/Python"
python run_prediction.py --mode qualifying --source fastf1 --year 2025 --round 1 --cache-dir .cache/fastf1
```

## Modes
- `qualifying`: predicts Q3 outcome (top 10) using FP1/FP2/FP3.
- `race`: predicts race top 10 once qualifying results are available.

## Notes
- `--include-standings` adds championship standings (from previous rounds) to race predictions.
- `--meeting-name` / `--country-name` can be used for OpenF1 if round indexing is ambiguous.
