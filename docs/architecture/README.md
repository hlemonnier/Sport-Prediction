# Architecture

The target repo shape is now explicit: shared infrastructure lives under
`packages/sports_core`; each sport owns its data contracts, features, model
families, and orchestration under its domain package.

```text
sport-prediction/
  apps/
    web/
    api/

  packages/
    sports_core/
      weather/
      data/
      features/
      evaluation/
      orchestration/

    f1/
      data/providers/
      data/schemas/
      features/
      models/pre_quali/
      models/pre_race/
      models/live_race/
      models/ultimate_lap_time/
      orchestration/

    football/
      data/providers/
      data/schemas/
      features/
      models/pre_match/
      models/scoreline/
      models/live_match/
      models/player_props/
      orchestration/

  configs/
    shared/
    f1/
    football/

  data/
    raw/
    interim/
    processed/

  artifacts/
    predictions/
    backtests/
    reports/

  docs/
    architecture/
    reviews/
```

## Model Boundaries

- F1 pre-quali predicts qualifying before qualifying is known.
- F1 pre-race predicts race order from grid/quali or predicted qualifying plus race features.
- F1 live-race updates predictions from live state and strategy events.
- F1 ultimate-lap-time is the future theoretical best-lap pace model.
- Football pre-match predicts 1X2 before kickoff.
- Football scoreline predicts exact score and goal distribution.
- Football live-match is the future rolling event/xG model.
- Football player/props is the future player-specific model family.

## Compatibility

The backend still discovers experiments under `research/projects`. Those folders
remain runnable via shims, but new source ownership belongs under `packages`.

## Advanced Model Plans

- [F1 Advanced Model Upgrade Plan](f1_advanced_model_upgrade_plan.md): mathematical
  upgrade path for Ultimate Lap-Time and Live Race Strategy, including
  CNN/TCN/DL, RL/MPC, simulator, data contracts, losses, and validation gates.
- [F1 CNN/RL/DL Roadmap](f1_cnn_rl_dl_roadmap.md): practical phase-by-phase
  implementation roadmap from current baselines to telemetry CNN/TCN,
  simulator, DP/MPC, offline RL, and multi-agent RL.
