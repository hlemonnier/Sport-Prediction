**Sport Prediction System**

Sport Prediction is a local multi-sport prediction system. The repo is now
organized around shared infrastructure plus domain packages for each sport.

## Runtime

Python components require Python `>=3.10`; Python 3.12 is the recommended and
container runtime. The repository `.python-version` selects 3.12 for compatible
environment managers, and each F1 service declares `requires-python` in its
`pyproject.toml`. Check an existing environment before installing or running:

```bash
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
```

Local virtual environments, FastF1 caches, and Next.js build output are
intentionally ignored and may be removed without changing the tracked project.
Recreate the recommended Python environment after a cleanup with:

```bash
python3.12 -m venv .venv-f1
.venv-f1/bin/python -m pip install -r services/f1-platform/requirements.txt
.venv-f1/bin/python -m pip install -r services/f1-prediction-service/requirements.txt
```

The web dependencies and build output are recreated independently:

```bash
cd apps/web
pnpm install
pnpm build
```

## Architecture

```text
sport-prediction/
  apps/
    api/                    # local API implementation
    web/                    # local web UI implementation

  packages/
    sports_core/
      weather/              # Open-Meteo, Tomorrow.io contract, historical cache
      data/                 # shared schemas, validation, local store helpers
      features/             # time, weather, and form transforms
      evaluation/           # backtest, calibration, ranking, probability metrics
      orchestration/        # pipeline, model registry, artifact helpers

    f1/
      data/providers/       # FastF1, OpenF1, local weekend adapters
      data/schemas/         # session, circuit, driver, result contracts
      features/             # circuit, practice, quali, race, strategy, weather, live state
      models/               # pre-quali, pre-race, live-race, ultimate-lap-time packages
      orchestration/        # weekend pipeline, same-season backtest, scenarios, contracts

    football/
      data/providers/       # local, football-data API, FBref, StatsBomb contracts
      data/schemas/         # team, fixture, match, player, venue contracts
      features/             # form, attack/defense, home/away, injuries, lineup, travel, weather, market
      models/               # pre-match, scoreline, live-match, player/props packages
      orchestration/        # fixture pipeline, league backtest, scenarios, contracts

  research/
    projects/               # experiment manifests, notebooks, legacy runners/shims
    papers/                 # research papers by sport

  configs/
    shared/                 # cross-sport config
    f1/                     # F1 profiles, circuits, seasons
    football/               # football profiles, leagues, venues

  data/
    raw/                    # source snapshots by sport
    interim/                # intermediate feature/data products by sport
    processed/              # model-ready datasets by sport

  artifacts/
    predictions/            # generated prediction outputs by sport
    backtests/              # generated backtest outputs by sport
    reports/                # generated reports by sport
  docs/
    architecture/           # architecture notes and migration contracts
    reviews/                # review docs and external audit notes
```

## Model Systems and Maturity

```text
Sport Prediction System
  Shared Core
    weather, data contracts, caching, evaluation, orchestration
  F1
    1. Qualifying Prediction
    2. Race Final Position
    3. Best Estimated Lap Time
    4. Live Race Intelligence (forecasting plus constrained strategy/RL decisions)
    Service/runtime fallback: target-specific untrained snapshot baselines;
      seconds-valued Live research forecasts are not service-integrated
  Football
    Pre-Match, Scoreline, Live Match, Player/Props
```

“Executable” does not mean “production validated.” No F1 stage should be
described as having proven predictive edge, calibrated production probabilities,
or a promoted model unless the exact current model path has a passing locked
out-of-sample promotion report. The machine-readable source for these claims is
`configs/f1/maturity.json`; README prose is not promotion evidence.

## Current Import Boundaries

- New shared imports should target `packages.sports_core`.
- New F1 imports should target `packages.f1`.
- New football imports should target `packages.football`.
- Legacy experiment runners under `research/projects/.../Python` remain supported through thin
  compatibility shims.

## Application Entrypoints

Default app paths:

```bash
cd apps/api && bun install && bun run dev
cd apps/web && pnpm install && pnpm dev
```

## Current Experiments

1. **F1 / Rising Qualification Prediction**
   - Legacy runner path: `research/projects/F1/rising_qualification_prediction/Python`
   - Authoritative package: `packages/f1`
   - Entrypoint: `run_experiment.py`

2. **Football / Match Result Prediction**
   - Legacy runner path: `research/projects/Football/Match Result Prediction/Python`
   - Authoritative package: `packages/football`
   - Entrypoint: `run_experiment.py`

## Weather

Open-Meteo is the first shared weather provider. It lives in `packages/sports_core/weather`
and is exposed to both F1 and football. F1 currently uses weather as scenario
context and uncertainty/variance priors when enabled. That is not the same as a
validated wet-relative-pace model or demonstrated weather-driven predictive
edge. Football currently attaches fixture/stadium weather context and keeps
probability adjustment gated until historical venue-weather training features
are available.

## Data vs Artifacts

- Keep reusable snapshots in `data/`.
- Keep generated predictions, backtests, and reports in `artifacts/`.
- Generated caches, old zips, local envs, and build folders should not be committed.

Suggested commit name: `docs: define F1 model maturity and supported Python runtime`
