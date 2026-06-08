# F1 Package

Authoritative F1 package for the sport prediction system.

- `data/providers`: FastF1, OpenF1, and local-weekend adapters
- `data/schemas`: session, circuit, driver, and result contracts
- `features`: circuit, practice, qualifying, race, strategy, weather, and live-state features
- `models/pre_quali`: train/predict/evaluate surface for qualifying prediction
- `models/pre_race`: train/predict/evaluate surface for race prediction
- `models/live_race`: state/strategy/predict surface for live race prediction
- `models/ultimate_lap_time`: train/predict surface for theoretical best lap pace
- `orchestration`: weekend pipeline, same-season backtest, scenarios, and contracts
- `betting`: F1 betting recommendation helpers

Legacy F1 experiment scripts now import this package directly.
