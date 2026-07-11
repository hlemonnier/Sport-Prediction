# Rising Qualification Prediction (Python)

This folder contains the F1 experiment runner compatibility layer.
Authoritative F1 package code now lives in `../../../../../packages/f1`.

The canonical experiment contract is declared in `../experiment.json` (context, snapshot root, model families, entrypoint).

Lifecycle:
- preseason proxy validation on 2025 (`artifacts/backtests/f1/preseason/holdout_2025`)
- live 2026 operations (`artifacts/predictions/f1/live`)
- shared raw weekends (`data/f1/raw/weekends`)

Runtime: Python `>=3.10` is required; Python 3.12 is recommended. Do not use
the repository's legacy Python 3.9 virtual environment. Before running:

```bash
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
```

Create a supported environment and install the executable core plus tests:

```bash
python3.12 -m venv .venv-f1
source .venv-f1/bin/activate
python -m pip install -e '.[test]'
```

Add `fastf1`, `plots`, or `deep` extras only when those paths are needed, for
example `python -m pip install -e '.[fastf1,plots]'`. The local/offline
prediction and benchmark paths do not require those optional dependencies.

## Model Architecture

Architectural target:

```text
F1 Prediction System
|-- Pre-Quali Model
|   `-- predicts qualifying
|-- Pre-Race Model
|   `-- predicts race from grid/quali + race features
|-- Live Race Model
|   `-- updates prediction and strategy during race
`-- Ultimate Lap-Time Model
    `-- predicts theoretical best lap pace
```

Executable research now:
- `Pre-Quali Model`: predicts qualifying from FP pace and team/driver form.
  Circuit features are quarantined by default; weather is scenario/uncertainty
  context. Neither is a proven source of predictive edge.
- `Pre-Race Model`: predicts the race from the official grid when available,
  qualifying fallback when only qualifying is available, or the Pre-Quali
  predicted rank when the weekend is still before qualifying. These information
  horizons are not interchangeable and require separate evaluation evidence.

Experimental, not promoted:
- `Live Race Model`: state-space, replay, simulation, strategy, and RL code
  exists, but no canonical trace model is promoted. The live platform uses
  explicitly named target-specific untrained snapshot baselines when a promoted
  model is unavailable.
- `Ultimate Lap-Time Model`: deterministic and deep candidate code exists, but
  the deep candidate is fail-closed until locked grouped validation, leakage,
  baseline-comparison, and promotion reports exist and pass.

“Executable” means the path runs; it does not mean that it beats its baseline,
has production-calibrated probabilities, or is safe for betting. See
`configs/f1/maturity.json` for the machine-readable evidence status.

Pre-quali race flow:
- Run the Pre-Quali model first.
- Merge `qualy_pred_position`, `qualy_pred_rank`, `qualy_pred_top3_proba`, and
  `qualy_pred_top10_proba` into the race feature frame.
- If real `grid_position` and `qualy_position` are missing, use
  `qualy_pred_rank` as the provisional race grid and label rows with
  `grid_source=predicted_qualifying_grid`.
- Evaluate that pre-qualifying race horizon separately from post-qualifying and
  official-grid forecasts; results from one horizon are not evidence for another.

Weather outputs:
- Every qualifying/race prediction emits two scenario payloads under
  `prediction_scenarios`.
- `base_no_weather` neutralizes weather uncertainty priors and removes the
  weather component from race-generation variance.
- `weather_integrated` keeps the available track/weather uncertainty priors.
- Current weather integration can use Open-Meteo through the shared
  `packages/sports_core` weather layer when `--weather on` is supplied;
  otherwise it falls back to local historical/track weather uncertainty priors.
- These scenarios are sensitivity views. They do not currently constitute a
  driver-specific wet-relative-pace model or prove that weather improves rank
  accuracy.

## Quick Start

```bash
cd "research/projects/F1/rising_qualification_prediction/Python"
python run_experiment.py profile --profile profiles/preseason_holdout_2025.yaml --output-format json
```

## Profile-Driven Workflows

Profiles are in `profiles/*.yaml` and define:
- `experiments.enable_dl_candidates`
- `experiments.compare_families`
- `dl.device`, `dl.arch`, `dl.hyperparams`, `dl.seed`
- `f1.mode` and `f1.live.{source,model,horizon_laps,seed,cache_dir,replay_path}`
- `evaluation.gates`

### Preseason benchmark + ablation (ML vs DL, ALL vs NO_RUNSIM)

```bash
python run_experiment.py profile --profile profiles/preseason_holdout_2025.yaml
```

This runs:
- holdout strict: train `2022,2023,2024`, evaluate `2025` rounds `6..24`
- variants:
  - `ML_ALL`
  - `ML_NO_RUNSIM`
  - `DL_ALL` (if torch available)
  - `DL_NO_RUNSIM` (if torch available)

NO_RUNSIM removes:
- `fp_quali_sim_*`
- `fp_race_sim_*`
- `fp_slow_lap_ratio`
- `fp_quali_vs_race_gap`

### Live 2026 phase runs

```bash
python run_experiment.py profile --profile profiles/live_2026_prequal.yaml --round 1 --year 2026
python run_experiment.py profile --profile profiles/live_2026_postqual.yaml --round 1 --year 2026
python run_experiment.py profile --profile profiles/live_2026_postrace.yaml --round 1 --year 2026

# Horizon B live race snapshot (state-space lap-by-lap)
python run_experiment.py profile \
  --profile profiles/live_2026_prequal.yaml \
  --phase live-race \
  --f1_mode live \
  --f1_live_source auto \
  --round 1 \
  --year 2026
```

### Horizon A vs B crossover benchmark (distance % + chaos segmentation)

Only compare rows drawn from the same driver population and information
cutoff. The current runner rejects incomplete fields; do not reuse older
top-10-only Horizon A artifacts as a full-field baseline.

```bash
python run_horizon_a_vs_b_lap_snapshots.py \
  --year 2025 \
  --horizon-a-dir artifacts/backtests/f1/compare_2025_afterfix_fullrace/horizon_a \
  --horizon-b-dir artifacts/backtests/f1/compare_2025_afterfix_fullrace/horizon_b \
  --weekends-dir data/f1/raw/weekends \
  --cutoff-mode distance_pct \
  --distance-cutoffs 5,10,20,30,40,50,60,70,80,90,100 \
  --pit-window-laps 3 \
  --clean-max-chaos-fraction 0.02 \
  --chaotic-min-chaos-fraction 0.05 \
  --epsilon-rank 0.10 \
  --epsilon-score 0.02
```

Key artifacts include:
- per-round cutoff metrics (`horizon_a_vs_b_lap_snapshots_per_round.csv`)
- crossover timing per round/metric (`horizon_a_vs_b_crossover_per_round.csv`)
- crossover distribution with `Never before finish` bucket (`horizon_a_vs_b_crossover_distribution.csv`)
- cumulative crossover curve (`horizon_a_vs_b_crossover_survival.csv`)
- full observability bundle in `observability/`:
  - data/filter/forecast diagnostics (A/B/C blocks)
  - horizon distribution + MC health diagnostics (D block)
  - pit/strategy and chaos diagnostics (E/F blocks)
  - comparative delta + crossover heatmaps (G block)

Audit regenerated benchmark JSON before using it as evidence:

```bash
python -m packages.f1.orchestration.evidence path/to/summary.json --root ../../../../../
```

The audit binds the report to the current F1 source tree, Python/package
versions, hashed weekend inputs, complete requested round/cutoff population,
and hashed upstream artifacts. A baseline-ladder artifact may be structurally
valid while its `promotion_gate.passed` remains false. Horizon evidence only
passes the auditor with locked prior calibration; hand-prior runs remain useful
research diagnostics but are deliberately non-promotable.

## Raw Data Download

```bash
python run_weekend_data_download.py --year 2025 --start-round 1 --weekends 5
```

Output folder:
- `data/f1/raw/weekends/`
- cache defaults are centralized at repo root: `.cache/fastf1` and `.cache/f1`

## Direct CLI (low-level)

Canonical direct CLI uses `run_experiment.py prediction`:

```bash
python run_experiment.py prediction \
  --mode qualifying \
  --source local \
  --year 2025 \
  --round 6 \
  --weekends-dir data/f1/raw/weekends \
  --enable-dl-candidates \
  --compare-families ml,dl \
  --dl-device auto \
  --dl-arch mlp_tabular_v1 \
  --dl-hyperparams '{"hidden_dims":[128,64],"dropout":0.15,"lr":0.001,"weight_decay":0.0001,"epochs":400,"batch_size":64,"early_stopping_patience":30}' \
  --dl-seed 42 \
  --output-format json
```

Compatibility wrappers remain available:
- `run_profile.py` -> `run_experiment.py profile`
- `run_prediction.py` -> `run_experiment.py prediction`

Additional ablation flag:
- `--disable-runsim-features`

Prediction JSON now includes:
- `rows`: top-10 table for UI/review.
- `all_prediction_rows`: full-field sorted table for betting/research consumers.
- `model_architecture`: current four-stage F1 prediction architecture and the
  active pre-quali-to-race contract.
- `prediction_scenarios`: `base_no_weather` and `weather_integrated` tables for
  both qualifying and race modes.
- `circuit_card`: the target event static research profile, including
  downforce demand, power sensitivity, corner-speed demands, tyre degradation,
  overtaking difficulty, qualifying importance, safety-car risk, strategy
  variance, and reliability. Its presence in JSON does not mean circuit
  features were enabled.
- `circuit_feature_columns`: candidate circuit-card and interaction names. Check
  `circuit_feature_state`; the default is `quarantined`.
- `proba_win`, `proba_top3`, `proba_top10`: model probability outputs. Treat
  them as calibrated only when `probability_audit.passed=true`, the audit source
  is out-of-fold, its later audit events are disjoint from the events used to
  fit PL temperature, and it covers the exact deployed score transformation
  and information horizon. In-sample, same-event calibration, or rank-based
  fallback probabilities are not validation evidence.
- When training data is unavailable, fallback ranking is a deterministic
  continuity mechanism, not a proven model and not a basis for an edge claim.

## Circuit Cards

Every known F1 event name is mapped to a numeric circuit card in `packages/f1/data/schemas/circuit.py`.
Cards are static priors for the circuit archetype, then local historical track
stats can refine them when available. Circuit features are quarantined by
default because the stored ablation did not demonstrate reliable improvement.
Enable them only for an explicit, current-code research ablation. When enabled,
the candidate feature surface includes:

- low-overtaking/high-grid-stability tracks increase qualifying/grid importance
- high-downforce, power-sensitive, traction/braking, tyre-degradation, and
  street-circuit dimensions become model features
- driver/team history on the same circuit archetype and exact circuit becomes
  additional form signal

These variables describe different circuit contexts, but description is not
predictive evidence. Static event-wide values do not differentiate drivers
unless a validated interaction or historical feature uses them, and all such
benefit must be demonstrated out of sample.

Horizon B live flags:
- `--f1_mode` (`offline` by default, `live` opt-in)
- `--f1_live_source` (`auto|local|fastf1`)
- `--f1_live_model` (`ssm_v1`)
- `--f1_live_horizon_laps` (default `10`)
- `--f1_live_seed` (default `42`)
- `--f1_live_cache_dir`
- `--f1_live_replay_path`
- `--f1_live_calibration_path` (locked filter/MC prior artifact only; this
  makes prior calibration ready, not the full live model promotion-ready)

The rollout strategy-template probabilities remain a declared heuristic.
Registry promotion fails closed until a bound calibration report also proves
those probabilities from locked replay and all simulator/baseline gates pass.

## Optional Deep Learning Dependency

Install only if you want DL candidates:

```bash
pip install -r requirements-dl.txt
```

Without torch, the pipeline still works and DL candidates are skipped with explicit notes.

## Betting Recommendation Engine

The betting layer consumes model prediction JSON plus market odds and outputs stake recommendations using expected ROI, fractional Kelly, and event exposure caps.

It is a research calculator, not evidence that the upstream probabilities have
edge. Do not interpret a positive calculated ROI or Kelly stake as actionable
unless the exact model/horizon has a passing probability audit, forward-test
record, and predeclared promotion gate.

Example odds CSV:

```csv
market,driver_name,decimal_odds,bookmaker
winner,Max Verstappen,3.40,book
podium,Lando Norris,1.85,book
top10,Alex Albon,2.20,book
```

Run:

```bash
python run_betting.py \
  --predictions artifacts/predictions/f1/live/2026/profile_weekend_pre_qualifying.json \
  --odds odds.csv \
  --bankroll 1000 \
  --fractional-kelly 0.25 \
  --min-edge-pct 3 \
  --max-bet-pct 1 \
  --max-market-pct 3 \
  --max-total-pct 5 \
  --output-format json
```

Supported markets:
- `winner` / `win` / `race_winner` -> `proba_win`
- `podium` / `top3` -> `proba_top3`
- `top10` / `points` -> `proba_top10`

The engine does not place bets with a bookmaker. It emits a deterministic bet slip (`status=bet`) and skipped candidates with reasons, which is the right boundary before adding broker/API execution.

For partner-facing forward tests, log the full recommendation record before market close:

```bash
python run_forward_bet_logger.py \
  --event-id 2026_round_06_spanish_gp \
  --information-cutoff post-qualifying \
  --market-close-utc 2026-06-07T12:00:00Z \
  --predictions artifacts/backtests/f1/rolling_2026/2026/round_06/postqual_race_prediction.json \
  --odds odds_round_06_postqual.csv \
  --log-path artifacts/predictions/f1/forward_test/f1_forward_bet_log.jsonl
```

The forward logger appends a hash-chained JSONL record containing prediction artifact hash, odds file hash, model probabilities, odds, Kelly/cap stake outputs, skipped rows, and the previous record hash. It refuses to create evidence records after market close unless `--allow-after-close` is explicitly provided for a dry-run/backfill.

Suggested commit name: `docs: correct F1 research model and evidence claims`
