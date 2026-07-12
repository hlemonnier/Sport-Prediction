# F1 Advanced Model Upgrade Plan

This document uses historical package names. The canonical product taxonomy is
the four-mode contract in `packages/f1/orchestration/contracts.py`.
`ultimate_lap_time` is an implementation alias inside **Best Estimated Lap
Time**, whose product target is the achievable best valid Grand Prix Qualifying
lap. The compatible-sector lower bound is diagnostic only. Live strategy is a
decision layer inside **Live Race Intelligence**, not a separate user mode.

This note defines how to move the two newest F1 model families from deterministic
baselines toward serious DL/RL-grade systems without breaking the current
`packages/f1` architecture.

## Current State

### Best Estimated Lap Time (`ultimate_lap_time` compatibility package)

Current file: `packages/f1/models/ultimate_lap_time/model.py`

The retained product baseline estimates the achievable best valid Grand Prix
Qualifying lap from the target-aligned rehearsal and a source-specific median
shift learned only from earlier same-season events. The older deterministic
sector implementation estimates a compatible-sector lower bound from clean
timing rows; that is a diagnostic, not the product target.

The rehearsal-shift baseline is auditable and causal but only conditionally
scored on drivers with both an eligible rehearsal and an observed target. The
full mode remains blocked by coverage and target-observation modelling. The
diagnostic sector floor remains useful for physics checks but must never be
evaluated or served as an expected achievable lap.

### Live Race Strategy

Current files:

- `packages/f1/models/live_race/state.py`
- `packages/f1/models/live_race/strategy.py`
- `packages/f1/models/live_race/predict.py`

The implemented live model combines state-space pace/degradation updates,
Monte Carlo position rollout, telemetry feature extraction, and deterministic
strategy scoring for `stay_out`, `pit_next_lap`, and `pit_now`.

This is the right first baseline because it exposes a real policy surface while
keeping the runner deterministic and testable. It is not yet the best possible
model because the policy is heuristic, not optimized against a calibrated race
simulator or learned from counterfactual strategy value.

## Research Takeaways

- State-space degradation should remain the live-race backbone. Cappello and
  Hoegh model latent tyre degradation with pit-stop resets and rolling-origin
  validation, which matches our current live state-space direction.
  Source: https://arxiv.org/abs/2512.00640
- Pit strategy should be framed as sequential decision-making under uncertainty,
  not a static classifier. Recent F1 strategy work uses mixed-integer nonlinear
  optimization, reinforcement learning environments, and hybrid RL/MPC.
  Sources: https://arxiv.org/abs/2512.21570 and https://arxiv.org/abs/2604.00826
- Multi-agent race strategy matters because other cars are part of the state:
  traffic, undercut response, safety-car bunching, and overtaking probability
  make single-driver policies structurally incomplete.
  Source: https://arxiv.org/abs/2602.23056
- FastF1 provides the public timing/telemetry shape we need for stronger pace
  models: lap timing, sector timing, car telemetry, position, tyre data, weather,
  and session results.
  Sources: https://docs.fastf1.dev/ and https://docs.fastf1.dev/api_reference/telemetry.html
- OpenF1 can complement FastF1 for public lap, stint, pit, position, weather,
  and race-control style fields.
  Source: https://openf1.org/docs/

## Ultimate Lap-Time: Best Mathematical Form

The target should be a probabilistic lower-envelope pace model:

```text
P(best_lap_seconds | driver, car/team, circuit, session, weather, tyre,
                    fuel_proxy, track_evolution, telemetry_trace)
```

The output should not be only one scalar. It should produce:

- `p05_best_lap_seconds`: aggressive theoretical lower envelope.
- `p50_best_lap_seconds`: realistic expected best lap.
- `p90_best_lap_seconds`: conservative ceiling.
- `sector_heads`: predicted S1/S2/S3 or minisector times.
- `uncertainty`: calibrated interval and residual diagnostics.

### Data Matrix

The model needs two aligned views.

Tabular event/lap view:

```text
driver_id, team_id, circuit_id, session_type, event_year, lap_number,
compound, tyre_age, track_status, air_temp, track_temp, humidity, rainfall,
wind_speed, pressure, fp/quali/race flag, lap_time, sector1/2/3,
clean_lap flag, deleted/invalid flag
```

Distance-normalized telemetry tensor:

```text
shape = [laps, distance_bins, channels]
channels = speed, throttle, brake, rpm, gear, drs, x, y,
           delta_to_driver_best, delta_to_session_best,
           track_status, weather embeddings when available
```

Distance normalization is mandatory. Time-indexed telemetry is weaker because
the same corner appears at different time indices for faster/slower laps.
Distance-indexed telemetry lets a 1D CNN/TCN learn braking zones, traction
zones, DRS zones, and corner exit losses.

### Model Stack

Use a staged ensemble, not one magic neural net.

1. `deterministic_baseline_v1`
   Current model. Keep it as the floor and fallback.

2. `tabular_quantile_gbdt_v1`
   LightGBM/XGBoost quantile or CatBoost-style model on engineered tabular
   features. This should be the first challenger because it is strong on small
   public F1 data.

3. `telemetry_tcn_v1`
   A 1D CNN/TCN over distance-normalized telemetry:

   ```text
   X_telemetry: [B, C, D]
   z = TCN(X_telemetry)
   z_static = MLP(static_features)
   h = concat(z, z_static)
   heads = {lap_p05, lap_p50, lap_p90, sector1, sector2, sector3}
   ```

4. `telemetry_transformer_v1`
   Only after enough data exists. Transformer attention can learn circuit
   sections and tyre-state interactions, but it is easier to overfit.

5. `stacked_calibrator_v1`
   A final calibrator combines deterministic, GBDT, TCN, and transformer
   outputs using walk-forward OOF predictions only.

### Loss Functions

Primary loss should be quantile pinball loss:

```text
L_q(y, yhat) = max(q * (y - yhat), (q - 1) * (y - yhat))
```

Train at least `q = 0.05, 0.50, 0.90`. The `p05` head is the theoretical
best-lap estimate; `p50` prevents the network from learning impossible fantasy
laps.

Add sector multi-task loss:

```text
L = L_lap_quantile + lambda_sector * (MAE_s1 + MAE_s2 + MAE_s3)
                  + lambda_order * rank_loss
                  + lambda_smooth * monotonic_regularization
```

Monotonic regularization should discourage impossible effects:

- Higher tyre age should not improve expected lap time without a track-evolution
  explanation.
- Wet/rain status should not improve expected dry pace.
- Pit or non-green laps must not train the lower-envelope head.

For uncertainty, evaluate CRPS or Gaussian/Student-t NLL, but pinball loss is
the most direct for "best possible lap" because the target is a lower quantile.

### Validation

Use strict walk-forward validation:

- Train on seasons/events before the target event only.
- Also test leave-circuit-out to see whether circuit embeddings generalize.
- Report MAE/RMSE for p50, pinball loss for p05/p90, interval coverage,
  calibration slope, and fastest-lap ranking hit rate.
- Compare against the deterministic baseline and a simple fastest-clean-lap
  prior. If the DL model cannot beat both OOF, do not ship it.

## Live Race Strategy: Best Mathematical Form

The target should be a constrained partially observable Markov decision process:

```text
state_t = posterior race state after lap t
action_t = stay_out | pit_now(compound) | pit_next_lap(compound) | push/conserve
reward = -final_race_time or points utility, with risk penalties
```

The live policy should optimize expected race outcome, not "predict whether a
team pitted historically." Historical pit decisions are behavior-policy labels;
they are not necessarily optimal.

### State

The state must include:

```text
lap, laps_remaining, position, race_time, gaps_front/back,
compound, tyre_age, stint_id, degradation posterior mean/std,
pace posterior mean/std, pit_loss estimate, available compounds,
mandatory compound rule state, track status, SC/VSC/yellow/red hazard,
weather forecast/nowcast, circuit overtaking/card features,
traffic density, DRS train proxy, teammate/team coupling,
position distribution from Monte Carlo rollout
```

This is a POMDP because true tyre thermal state, fuel load, ERS, damage, and
team intent are hidden. The state-space model should provide posterior features
rather than pretending the hidden state is observed.

### Action Space

Start with a discrete action set:

```text
stay_out
pit_now_soft
pit_now_medium
pit_now_hard
pit_next_soft
pit_next_medium
pit_next_hard
```

Use action masks for illegal actions:

- cannot pit if already in pit lane,
- cannot choose unavailable compounds,
- avoid stopping too late to satisfy minimum stint logic,
- red flag/no race-running periods can force a conservative mask,
- mandatory tyre rules must be represented explicitly.

Later add continuous controls:

```text
pace_push_level in [0, 1]
tyre_save_level in [0, 1]
ERS_deploy_proxy in [0, 1] if data exists
```

### Simulator

RL is only as good as the simulator. Build this before training RL:

```text
next_lap_time = baseline_lap(circuit, lap)
              + driver_pace_state
              + tyre_degradation_state
              + compound_effect
              + traffic_loss
              + track_status_offset
              + weather_offset
              + random_noise
```

Pit stops reset tyre age/degradation state and add stochastic pit loss. SC/VSC
changes pit loss and lap-time offsets. Traffic loss must depend on circuit
overtaking difficulty, gap, and position.

The simulator should be calibrated by rolling-origin backtests and scored with:

- one-step lap-time CRPS/MAE,
- pit-loss calibration,
- SC/VSC event calibration,
- overtake/position transition calibration,
- final order/top-k calibration.

### Optimization Ladder

Do not jump straight to deep RL. Use this ladder:

1. Dynamic programming oracle
   Single-driver, no traffic. Gives an upper bound and catches simulator bugs.

2. Monte Carlo tree search or model predictive control
   Online planner over the calibrated simulator. Good production baseline.

3. Behavior cloning
   Learn historical team-like policy. Useful warm start, not optimality proof.

4. Offline RL
   Use CQL/IQL-style conservative offline RL because historical public data is
   offline and biased. Direct Q-learning can exploit simulator/data holes.

5. Multi-agent RL
   Use MAPPO/PPO in the simulator only after the simulator is calibrated. This
   is needed for traffic, undercut response, SC bunching, and game-theoretic
   interactions.

6. Hybrid RL/MPC
   Let RL choose discrete pit/compound timing and MPC optimize continuous pace
   or energy proxies. This matches recent F1 strategy research and is safer
   than end-to-end RL.

### RL Objective

For a points-focused policy:

```text
R = points(final_position)
    - alpha * expected_race_time
    - beta * tyre_failure_risk
    - gamma * illegal_or_rule_penalty
    - eta * strategy_variance
```

For a race-time policy:

```text
R = -final_race_time_seconds
    - beta * CVaR_10_bad_outcomes
    - gamma * rule_penalty
```

Train and report both. Betting/prediction usage cares about calibrated outcome
probabilities; team-strategy usage cares about expected utility under risk.

### Validation

Use counterfactual validation carefully:

- Historical outcome cannot prove an alternative strategy was better.
- Use a locked simulator and compare policies under identical seeds.
- Report regret to DP/MINLP oracle in simplified settings.
- Report policy value with uncertainty intervals, not one deterministic claim.
- Stress test Monaco, wet sessions, SC/VSC-heavy races, and high-deg races
  separately.
- Require no leakage: live policy at lap `t` can only use observations through
  lap `t`.

## Repo Implementation Plan

### Phase 1: Data Contracts

Add explicit datasets before adding neural nets:

```text
packages/f1/models/ultimate_lap_time/datasets.py
packages/f1/models/live_race/environment.py
packages/f1/models/live_race/action_space.py
```

Ultimate lap-time dataset outputs:

```text
UltimateLapTelemetryBatch:
  telemetry: np.ndarray[B, C, D]
  static_features: pd.DataFrame
  targets: lap/sector quantiles
  metadata: event, driver, circuit, split keys
```

Live strategy environment outputs:

```text
StrategyState
StrategyAction
StrategyTransition
StrategyReward
```

### Phase 2: Strong Non-DL Challengers

Before CNN/RL, add:

- GBDT quantile ultimate-lap model.
- DP/MPC live-strategy oracle on the simulator.

These are the baselines DL/RL must beat.

### Phase 3: Ultimate Lap-Time DL

Implement:

```text
packages/f1/models/ultimate_lap_time/deep.py
packages/f1/models/ultimate_lap_time/train_deep.py
packages/f1/models/ultimate_lap_time/evaluate.py
```

Recommended first neural architecture:

```text
DistanceTelemetryTCN:
  Conv1d blocks over distance bins
  static feature MLP
  multi-task quantile heads
  conformal calibration from OOF residuals
```

Only promote it if it beats deterministic and GBDT OOF on p05 pinball loss,
p50 MAE, interval coverage, and fastest-lap ranking.

### Phase 4: Live Strategy Simulator + Planner

Implement:

```text
packages/f1/models/live_race/simulator.py
packages/f1/models/live_race/planner.py
packages/f1/models/live_race/evaluate_policy.py
```

Start with DP/MPC. The live runner can use this planner before RL is ready.

### Phase 5: Offline RL and Multi-Agent RL

Implement only after simulator validation:

```text
packages/f1/models/live_race/rl/offline.py
packages/f1/models/live_race/rl/mappo.py
packages/f1/models/live_race/rl/replay_buffer.py
```

Training gates:

- behavior cloning baseline exists,
- offline policy improves simulator value without violating action masks,
- policy does not collapse to impossible stop patterns,
- sensitivity checks over pit loss, SC/VSC hazard, and traffic loss pass.

## Practical Call

CNN is strongest for ultimate lap-time only if we build distance-normalized
telemetry tensors. Without that, use GBDT/MLP first.

RL is strongest for live-race strategy only if we first build a calibrated race
simulator. Without a simulator, RL will mostly learn artifacts of historical
team decisions or exploit bad assumptions.

The next engineering move should be:

1. Build the ultimate-lap telemetry dataset contract.
2. Build the live-strategy simulator/environment contract.
3. Add GBDT quantile and DP/MPC baselines.
4. Add CNN/TCN and offline RL only after those baselines and validators exist.
