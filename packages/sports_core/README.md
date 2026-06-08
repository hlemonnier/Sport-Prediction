# Sports Core

Shared infrastructure for every sport domain.

- `sports_core/weather`: canonical shared weather import path
- `sports_weather`: compatibility import path used by current F1/football adapters
- `weather`: canonical source files for Open-Meteo, Tomorrow.io, and historical cache
- `data`: shared data contracts, validation, and local storage
- `features`: shared time, weather, and form feature helpers
- `evaluation`: backtest, calibration, ranking, and probability metrics
- `orchestration`: pipeline, registry, and artifact helpers
