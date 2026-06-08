# Football Package

Authoritative football package for the sport prediction system.

- `mrp`: current production implementation moved from the legacy research path
- `data/providers`: local, football-data API, FBref, and StatsBomb provider surfaces
- `data/schemas`: team, fixture, match, player, and venue contracts
- `features`: team-form, attack/defense, home/away, injuries, lineup, rest/travel, weather, and market features
- `models/pre_match`: train/predict/evaluate surface for 1X2 prediction
- `models/scoreline`: train/predict/evaluate surface for scoreline prediction
- `models/live_match`: state/predict/evaluate surface for live match prediction
- `models/player_props`: train/predict surface for player/props prediction
- `orchestration`: fixture pipeline, league backtest, scenarios, and contracts

Legacy imports from `research/projects/Football/Match Result Prediction/Python/mrp`
are kept through a shim.
