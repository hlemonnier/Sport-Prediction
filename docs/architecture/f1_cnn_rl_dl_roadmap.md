# F1 CNN/RL/DL Roadmap

This roadmap turns the current F1 baselines into a staged path toward advanced
CNN/DL/RL models.

The two model families are:

- `Ultimate Lap-Time`: theoretical best lap pace before or during a weekend.
- `Live Race Strategy`: live pit/compound/action policy during a race.

The current implementations are no longer empty. They are deterministic
baselines. Treat them as the floor that every advanced model must beat.

## End State

The final system should have four layers:

```text
F1 Advanced Models
+-- Data Contracts
|   +-- distance-normalized telemetry tensors
|   +-- lap/session tabular features
|   +-- live race state/action/reward records
|   +-- simulator transition records
+-- Ultimate Lap-Time
|   +-- deterministic baseline
|   +-- tabular quantile model
|   +-- CNN/TCN telemetry model
|   +-- calibrated ensemble
+-- Live Race Strategy
|   +-- deterministic strategy baseline
|   +-- race simulator
|   +-- DP/MPC planner
|   +-- behavior cloning
|   +-- offline RL
|   +-- multi-agent RL
+-- Evaluation
    +-- same-season/walk-forward tests
    +-- no-leakage live replay tests
    +-- calibration reports
    +-- promotion gates
```

## Roadmap Summary

```text
Phase 0: Lock baselines
Phase 1: Build data contracts
Phase 2: Build validation harness
Phase 3: Add strong non-DL challengers
Phase 4: Build Ultimate Lap-Time CNN/TCN
Phase 5: Build Live Race simulator
Phase 6: Add DP/MPC live strategy planner
Phase 7: Add behavior cloning and offline RL
Phase 8: Add multi-agent RL
Phase 9: Add production model registry and promotion gates
```

## Phase 0: Lock Baselines

Goal: freeze the deterministic baselines as mandatory comparison models.

Current files:

- `packages/f1/models/ultimate_lap_time/model.py`
- `packages/f1/models/live_race/strategy.py`
- `packages/f1/models/live_race/predict.py`

Deliverables:

- Keep deterministic ultimate lap-time as `ultimate_lap_time_deterministic_baseline_v1`.
- Keep deterministic live strategy policy as `deterministic_baseline_v1`.
- Add stable JSON result schemas for both model outputs.
- Add baseline backtest scripts that produce comparable artifact reports.

Done when:

- Full F1 suite passes.
- Baseline reports write to `artifacts/backtests/f1`.
- Advanced models cannot be promoted unless they beat the deterministic baseline.

## Phase 1: Build Data Contracts

Goal: create model-ready datasets before building CNN/RL code.

Add files:

```text
packages/f1/models/ultimate_lap_time/datasets.py
packages/f1/models/ultimate_lap_time/schemas.py
packages/f1/models/live_race/environment.py
packages/f1/models/live_race/action_space.py
packages/f1/models/live_race/replay_buffer.py
```

Ultimate Lap-Time dataset contract:

```text
UltimateLapTelemetryExample
  telemetry: float[channels, distance_bins]
  static_features: dict
  targets:
    lap_time_seconds
    sector1_seconds
    sector2_seconds
    sector3_seconds
    p05_target
    p50_target
    p90_target
  metadata:
    event_key
    circuit_id
    driver_id
    team_id
    session
    split_key
```

Live Race Strategy dataset contract:

```text
StrategyTransition
  state_t
  action_t
  reward_t
  state_t1
  done
  legal_action_mask
  metadata
```

Critical requirement:

- Ultimate lap telemetry must be distance-normalized, not time-normalized.
- Live strategy records must be no-leakage: state at lap `t` can only use data
  available through lap `t`.

Done when:

- Dataset builders run on local FastF1/OpenF1-style data.
- Unit tests verify shapes, masks, no-leakage splits, and empty-data behavior.
- Dataset summaries report row counts by season, circuit, driver, session, and
  target availability.

## Phase 2: Build Validation Harness

Goal: define the scoreboard before adding complex models.

Add files:

```text
packages/f1/models/ultimate_lap_time/evaluate.py
packages/f1/models/live_race/evaluate_policy.py
packages/f1/orchestration/model_promotion.py
configs/f1/profiles/ultimate_lap_time.yaml
configs/f1/profiles/live_strategy.yaml
```

Ultimate Lap-Time metrics:

- p50 MAE/RMSE.
- p05 and p90 pinball loss.
- interval coverage.
- fastest-lap winner hit rate.
- top-3 fastest-lap ranking accuracy.
- calibration curve by circuit/session/weather.

Live Race Strategy metrics:

- one-step lap-time MAE/CRPS in simulator.
- final position calibration.
- pit-loss calibration.
- strategy regret vs DP oracle in simplified settings.
- policy value by identical-seed Monte Carlo.
- illegal action rate.
- no-leakage replay invariance.

Done when:

- Every model produces the same evaluation payload shape.
- Reports are written under `artifacts/reports/f1`.
- Promotion gates fail closed if metrics are missing.

## Phase 3: Add Strong Non-DL Challengers

Goal: add serious baselines that CNN/RL must beat.

Ultimate Lap-Time:

```text
packages/f1/models/ultimate_lap_time/tabular_quantile.py
```

Model:

- GBDT quantile model if LightGBM/XGBoost is available.
- Otherwise scikit-style quantile gradient boosting if already available.
- Quantiles: p05, p50, p90.

Live Race Strategy:

```text
packages/f1/models/live_race/planner.py
```

Model:

- Dynamic programming single-car oracle.
- Simple MPC planner over the race simulator once Phase 5 exists.

Done when:

- Tabular quantile model beats deterministic ultimate-lap baseline on p05
  pinball loss or is explicitly rejected.
- DP oracle produces sensible stop windows on synthetic high-deg, low-deg,
  SC/VSC, and Monaco-style low-overtake scenarios.

## Phase 4: Build Ultimate Lap-Time CNN/TCN

Goal: learn circuit-section pace losses from distance-normalized telemetry.

Add files:

```text
packages/f1/models/ultimate_lap_time/deep.py
packages/f1/models/ultimate_lap_time/train_deep.py
packages/f1/models/ultimate_lap_time/evaluate_deep.py
```

Recommended first architecture:

```text
DistanceTelemetryTCN
  input: telemetry[batch, channels, distance_bins]
  blocks:
    Conv1d -> GELU/ReLU -> Dropout -> Conv1d -> residual
    dilations: 1, 2, 4, 8
  pooling:
    attention or adaptive average over distance
  static branch:
    MLP(driver/team/circuit/session/weather/tyre features)
  heads:
    lap_p05
    lap_p50
    lap_p90
    sector1
    sector2
    sector3
```

Loss:

```text
L = pinball(p05, p50, p90)
  + lambda_sector * sector_mae
  + lambda_rank * fastest_lap_pairwise_rank_loss
  + lambda_mono * monotonic_penalty
```

Why CNN/TCN here:

- Track distance creates local structure.
- Braking zones, throttle traces, DRS zones, and corner exits are spatially
  local patterns.
- CNN/TCN is less data-hungry than a transformer and easier to validate.

Do not promote if:

- It only beats baseline in random splits but fails walk-forward.
- It improves p50 but worsens p05 calibration.
- It creates physically impossible improvements from higher tyre age or wet
  weather.

## Phase 5: Build Live Race Simulator

Goal: create the world model required before RL.

Add files:

```text
packages/f1/models/live_race/simulator.py
packages/f1/models/live_race/simulator_calibration.py
packages/f1/models/live_race/traffic.py
packages/f1/models/live_race/pit_loss.py
```

Simulator transition:

```text
state_t, action_t -> state_t1, reward_t
```

State dynamics:

```text
lap_time =
  event_lap_baseline
  + driver_pace_state
  + tyre_degradation_state
  + compound_effect
  + fuel_proxy
  + traffic_loss
  + track_status_offset
  + weather_offset
  + random_noise
```

Pit action:

```text
pit_now(compound):
  race_time += pit_loss(track_status, circuit, traffic)
  tyre_age = 0
  compound = selected_compound
  degradation_state = compound_prior
```

Done when:

- Simulator can replay one race from lap 1 to finish.
- Simulator can start from any live replay lap.
- Calibration report shows one-step lap-time, pit-loss, track-status, and final
  order metrics.
- Same seed produces deterministic policy comparisons.

## Phase 6: Add DP/MPC Live Strategy Planner

Goal: get an optimized planner before RL.

Add files:

```text
packages/f1/models/live_race/planner.py
packages/f1/models/live_race/mcp.py
packages/f1/models/live_race/policy.py
```

Planner actions:

```text
stay_out
pit_now_soft
pit_now_medium
pit_now_hard
pit_next_soft
pit_next_medium
pit_next_hard
```

MPC loop:

```text
for each live lap:
  sample simulator scenarios
  evaluate legal action sequences over horizon H
  choose first action with best expected utility
  replan after next lap
```

Reward:

```text
R = expected_points
  - alpha * expected_race_time
  - beta * downside_cvar
  - gamma * illegal_action_penalty
```

Done when:

- Planner beats deterministic strategy baseline in same-seed simulator tests.
- Planner does not recommend illegal compounds or impossible pit windows.
- Monaco/high-overtake/wet/SC scenarios have separate diagnostics.

## Phase 7: Add Behavior Cloning and Offline RL

Goal: learn from historical behavior without blindly copying it.

Add files:

```text
packages/f1/models/live_race/rl/behavior_cloning.py
packages/f1/models/live_race/rl/offline.py
packages/f1/models/live_race/rl/replay_buffer.py
```

Behavior cloning:

- Input: historical live states.
- Target: observed team actions.
- Use as warm start only.

Offline RL:

- Conservative Q-learning or IQL-style policy.
- Must use legal action masks.
- Must be evaluated in the locked simulator, not just by historical action
  accuracy.

Do not use naive online RL on historical data:

- Public F1 data is offline.
- Historical strategy is biased by team goals, unknown car state, traffic, and
  safety-car luck.
- Direct Q-learning can exploit simulator holes.

Done when:

- Behavior cloning predicts historical action timing better than trivial
  baselines.
- Offline RL beats behavior cloning and DP/MPC only in validated simulator
  settings.
- Conservative value estimates do not explode out of distribution.

## Phase 8: Add Multi-Agent RL

Goal: model other cars as strategic actors.

Add files:

```text
packages/f1/models/live_race/rl/multi_agent_env.py
packages/f1/models/live_race/rl/mappo.py
packages/f1/models/live_race/rl/self_play.py
```

Why:

- Undercuts only matter relative to other cars.
- Traffic and DRS trains are multi-car effects.
- SC/VSC bunching changes everyone at once.
- One driver pitting can trigger other teams to respond.

Recommended algorithm:

- MAPPO-style centralized training, decentralized execution.
- Start with 4-8 cars in simplified scenarios.
- Scale to full grid only after simulator speed and calibration are acceptable.

Done when:

- Multi-agent policy beats single-agent policy in traffic-heavy scenarios.
- It remains stable under seeded scenario replay.
- It does not learn unrealistic synchronized pit patterns.

## Phase 9: Production Registry and Promotion

Goal: make model selection explicit and reversible.

Add files:

```text
packages/f1/orchestration/model_registry.py
configs/f1/profiles/ultimate_lap_time_deep.yaml
configs/f1/profiles/live_strategy_rl.yaml
```

Model registry fields:

```text
model_id
model_family
version
training_data_cutoff
feature_schema_version
artifact_path
metrics
promotion_status
fallback_model_id
```

Promotion rules:

- No random split promotion.
- No missing calibration report.
- No promotion without baseline comparison.
- No promotion if leakage tests fail.
- No promotion if simulator validation fails for RL.
- No production replacement without deterministic fallback.

## Dependency Decision

Use this order:

1. pandas/numpy/sklearn-style deterministic and tabular models.
2. Optional LightGBM/XGBoost if dependency policy allows it.
3. PyTorch for CNN/TCN and RL only after data contracts exist.
4. No TensorFlow unless there is a strong repo-level reason.

PyTorch is the right DL/RL dependency because:

- CNN/TCN and RL implementations are straightforward.
- It supports custom losses and action masks cleanly.
- It avoids committing to a heavier RL framework too early.

Avoid large RL frameworks until the simulator/environment API is stable.

## Suggested Agent Split

When implementing this roadmap, split agents by ownership:

1. `Ultimate Dataset Agent`
   Owns `ultimate_lap_time/datasets.py`, schemas, and tests.

2. `Ultimate Tabular Agent`
   Owns `ultimate_lap_time/tabular_quantile.py` and evaluation.

3. `Ultimate CNN Agent`
   Owns `ultimate_lap_time/deep.py`, `train_deep.py`, and tensor tests.

4. `Live Simulator Agent`
   Owns `live_race/simulator.py`, traffic, pit loss, and calibration.

5. `Live Planner Agent`
   Owns DP/MPC planner and policy tests.

6. `Live RL Agent`
   Owns RL replay buffer, behavior cloning, offline RL, and multi-agent RL.

Do not let two agents edit the same model file in parallel. Shared contracts
must be landed first.

## Immediate Next Step

Build Phase 1 before touching CNN/RL code:

```text
packages/f1/models/ultimate_lap_time/datasets.py
packages/f1/models/ultimate_lap_time/schemas.py
packages/f1/models/live_race/environment.py
packages/f1/models/live_race/action_space.py
```

Without those contracts, CNN/RL work will look impressive but will be fragile,
hard to validate, and likely to overfit.
