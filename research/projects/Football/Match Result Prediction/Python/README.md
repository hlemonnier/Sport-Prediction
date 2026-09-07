# Football research runner

The canonical implementation is in `packages/football`; this directory preserves
legacy `mrp` imports and command entrypoints. The models are research-only. Read
`packages/football/README.md` for the exact likelihood, timing, validation and
support limitations. `packages/football/maturity.json` is the machine-readable
support manifest.

From the repository root, with the football runtime dependencies installed:

```bash
python "research/projects/Football/Match Result Prediction/Python/run_experiment.py" \
  --mode match_result --league epl --season 2025 --round 1 \
  --data-source /absolute/path/to/local/football-data \
  --football_model hybrid --football_calibration auto --output-format json
```

The data directory contains `matches.csv`/parquet, `fixtures.csv`/parquet and
optionally `teams.csv`/parquet. Matches require distinct teams, nonnegative integer
scores, unique identities and dated kickoffs. Use explicit `result_available_at`
for causal intraday history; otherwise the following UTC day is the conservative
availability fallback. `xg_available_at` is required before historical xG can be
used instead of the documented goal proxy. Fixture kickoffs are mandatory.

Both `match_result` and `scoreline` return 1X2, scoreline and expected goals from a
common coherent joint distribution. They do not require or consume lineups/odds;
those feeds are unimplemented. `--weather on` attaches metadata only.
`--goal-strength-half-life-days` is an optional goal-likelihood weighting policy,
with no implied performance improvement. `--shadow_eval off` suppresses outer
metrics; it does not convert fitting diagnostics into validation.

The compatibility `run_prediction.py` forwards to `run_experiment.py`.
Run regression checks with the legacy import path on `PYTHONPATH`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=".:research/projects/Football/Match Result Prediction/Python" \
python -m pytest -p no:cacheprovider \
  "research/projects/Football/Match Result Prediction/Python/tests" -q
```

Synthetic regressions exercise timing, negative score support, tail probabilities,
optimizer stationarity, calibration separation, growing holdouts, and outer-label
invariance. They do not measure real football predictive quality.
