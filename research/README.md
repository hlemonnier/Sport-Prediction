# Research Workspace

This workspace is organized around **experiments**.

Each experiment is a reproducible unit with:
1. Context (season/round/league/phase)
2. Dataset snapshot
3. Model family (baseline/ML/DL)
4. Outputs + diagnostics

## Structure
- `projects/` sport experiments (notebooks + Python pipelines + `experiment.json`)
- `papers/` research PDFs + index
- `sport_cli.py` CLI navigator for projects and papers

## Quick paths
- F1 experiment runner: `research/projects/F1/rising_qualification_prediction/`
- Football experiment runner: `research/projects/Football/Match Result Prediction/`
- F1 package source: `../packages/f1`
- Football package source: `../packages/football`
- Shared core source: `../packages/sports_core`
- Papers index: `research/papers/README.md`
