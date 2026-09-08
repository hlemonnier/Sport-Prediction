The frozen weather experiment failed its 2023 advancement screen. All four variants increased event-balanced next-eligible-lap MAE against the strong HGB baseline. No 2024–2026 weather was acquired or scored for this model experiment, and no production configuration changed.

The discovery input contains 44 races, 7,425 weather records and exactly the original 38,370 matched eligible-issuance forecasts: 18,363 from 2022 for fitting and 20,007 from 2023 for selection. The unmatched issuance ledger contains 39,220 forecasts. Weather features were first defined at issuance without target information; matching the weather-enriched issuance ledger reproduces every scored feature exactly.

The baseline uses the existing 15-leaf, 150-iteration HGB architecture with coefficients fitted only to 2022. Both weather HGBs also fit only 2022, using equal event total weight, the original 80 features and 56 weather features. Their target is next eligible lap time minus the most recently observed eligible lap time, clipped to ±5 seconds during fitting; predicted corrections retain the baseline's ±3-second bound. Seven- and fifteen-leaf models each produce two variants: use weather everywhere it is available, or change the baseline only after an observed rain record in the last 15 minutes. The two scopes share fitted coefficients.

| Variant | 2023 event MAE, seconds | Relative change in MAE | Paired event 95% CI for candidate minus baseline, seconds |
|---|---:|---:|---:|
| Strong HGB baseline | 0.5120807233 | — | — |
| Weather HGB, 7 leaves, all | 0.5166355396 | +0.889472% | [0.00149171, 0.00797617] |
| Weather HGB, 7 leaves, rain gate | 0.5123239426 | +0.047496% | [0, 0.000555917] |
| Weather HGB, 15 leaves, all | 0.5147839322 | +0.527887% | [0.000601041, 0.004992198] |
| Weather HGB, 15 leaves, rain gate | 0.5124098636 | +0.064275% | [−0.000297288, 0.001284708] |

The least-bad selected variant is `weather_hgb_l7_rain`. It changes forecasts in three 2023 events and worsens all three. On its 1,059 observed-recent-rain targets, event-balanced MAE rises from 1.27252314 to 1.27748811 seconds (+0.390167%). Predictions outside the rain gate equal the baseline exactly. Circular blocks of three events also give a nonnegative 95% interval, [0, 0.000555917] seconds. Its largest leave-one-event-out mean delta is +0.000254801 seconds. Re-encoding the selected, unchanged model at assumed delays of zero and 120 seconds still worsens full-population MAE, by 0.017264% and 0.068576%, respectively.

The frozen advancement screen required at least 1% full-population improvement, negative upper bounds for event and block bootstrap intervals, improvement under every event omission, changes in at least three events, and positive gain at both alternate delays. It fails. These are retrospective discovery results; selecting the least-bad variant does not create independent validation evidence. The result rejects this bounded feature/model family and does not prove that weather contains no predictive information.

Weather and lap timestamps share FastF1's session clock. The encoder uses only weather timestamps strictly earlier than issuance minus 60 seconds; it does not shift either stream by the scheduled or actual race start. This is an assumed additional delivery delay, not verified historical receipt latency. Each channel retains its own last finite observation and timestamp, with explicit missingness; a value older than 600 seconds is unavailable. All seven channels are actually present at all scored checkpoints, with ages from 60.001 to 120.012 seconds. Temperature, humidity, pressure and speed slopes use actual elapsed minutes between strictly prior observations; rain transitions and sample counts use only the observed window. Wind direction is represented circularly. No target flags, future weather, interpolation or full-event weather statistics enter the model. Original recorded lap eligibility and FastF1's retrospective corrections remain limitations inherited from the benchmark.

Verification recomputed 120,042 saved candidate/sensitivity predictions exactly, all 38,370 weather feature rows, all metrics and advancement checks, and 132 raw-weather/CSV/lap hashes. Real-prefix checks at Emilia Romagna 2022 and the Netherlands 2023 preserve 1,208 unmatched original issuances, all 80 legacy features, weather features and predictions for all four variants after truncating future laps/weather or poisoning future weather. Nine synthetic regressions cover strict cutoff, channel-specific staleness, missing data, actual-time slopes, rain windows, circular wind, exact baseline fallback and target-independent augmentation.

Source and result locations:

- `spec.json`: frozen protocol; SHA256 `9d13493c7f4c8494769f3dc101c59ddbc0a44d49bf4076bd0fc52f6aa4617d69`.
- `run_experiment.py`: frozen fitting/selection implementation; SHA256 `f21d6c3be76c875d7ee1ef9cff8d37ec3cb807ede3cb92eaae76281bc8bb70bd`.
- `artifacts/research/boundary_20260908/weather/model/results.json`: every variant, event, subset and sensitivity; SHA256 `31c9c2177bbee96d833abd1c41b555a3bc0e0ec10cd9ca25a5b5e3fdea7e3f1e`.
- `artifacts/research/boundary_20260908/weather/model/selection_lock.json`: source, model, forecast and data bindings; SHA256 `6fc31965363acb3c160583d747ffde106c644029039aa7a561eda25d4f18f184`.
- `artifacts/research/boundary_20260908/weather/model/verification.json`: exact replay and prefix evidence.
- `artifacts/research/boundary_20260908/weather/quality/review.json`: independently checked source identity, parser and clock alignment.

Run verification from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/weather/model/test_weather.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/boundary_20260908/weather/model/verify_experiment.py
```

The fitting runner refuses to replace an existing design lock. Preserve the frozen artifacts when conducting any new experiment.

Suggested commit: `research(f1-live): record rejected causal weather forecasting experiment`.
