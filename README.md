**Sport Lab**

Sport Lab is a local research cockpit for sport prediction experiments.

An experiment is the core unit of work. Each experiment has:
1. Context (`season`, `round`, `league`, `phase`, etc.)
2. Dataset snapshot (`data/...` at run time)
3. Model family (`baseline`, `ml`, `dl`)
4. Outputs + diagnostics (rankings/probabilities + calibration/rank metrics)

## Repository map

- `research/projects/<Sport>/<Project>/`
  - `Jupyter/` exploration and hypothesis testing
  - `Python/` production-style pipeline/runners
  - `experiment.json` canonical experiment metadata (kind + entrypoint + contract)
- `research/papers/` research papers by sport
- `research/sport_cli.py` local CLI explorer
- `platform/backend` API + orchestration for running experiments
- `platform/web` local UI
- `data/` local dataset snapshots used by experiments
- `outputs/` generated predictions, reports, diagnostics, and logs

## Current experiments

1. **F1 / Rising Qualification Prediction**
- Path: `research/projects/F1/rising_qualification_prediction/`
- Entrypoint: `Python/run_profile.py`
- Snapshot root: `data/f1`
- Model families: `baseline`, `ml`, `dl`

2. **Football / Match Result Prediction**
- Path: `research/projects/Football/Match Result Prediction/`
- Entrypoint: `Python/run_prediction.py`
- Snapshot root: `data/football`
- Model families: `baseline`, `ml`, `dl`

## Local platform

- Backend: `cd platform/backend && bun install && bun run dev`
- Frontend: `cd platform/web && pnpm install && pnpm dev`

## Data vs Outputs

- Keep in `data/`: reusable source snapshots (`data/f1/raw/weekends`, `data/f1/f1_dataset.*`, `data/f1/f1_coverage.*`, `data/football/*`)
- Keep in `outputs/`: generated run artifacts (live/preseason profile runs, comparison benchmarks, lap snapshot reports)
- Safe cleanup rule: if a file is under `outputs/`, you can delete it and regenerate it by rerunning the related pipeline/profile

## Add a new experiment

1. Create `research/projects/<Sport>/<Project>/`.
2. Add `Jupyter/model-research.ipynb` for research.
3. Add `Python/` with a runnable entrypoint.
4. Add `experiment.json` with `kind` and `python_entrypoint`.
5. Store related papers in `research/papers/<Sport>/`.
