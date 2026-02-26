# Rising Qualification Prediction (Python)

This folder contains the F1 prediction runtime with separated lifecycle.

The canonical experiment contract is declared in `../experiment.json` (context, snapshot root, model families, entrypoint).

Lifecycle:
- preseason proxy validation on 2025 (`data/f1/preseason/holdout_2025`)
- live 2026 operations (`data/f1/live/2026`)
- shared raw weekends (`data/f1/raw/weekends`)

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

```bash
python run_horizon_a_vs_b_lap_snapshots.py \
  --year 2025 \
  --horizon-a-dir data/f1/compare_2025_afterfix_fullrace/horizon_a \
  --horizon-b-dir data/f1/compare_2025_afterfix_fullrace/horizon_b \
  --weekends-dir data/f1/raw/weekends \
  --cutoff-mode distance_pct \
  --distance-cutoffs 5,10,20,30,40,50,60,70,80,90,100 \
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

Horizon B live flags:
- `--f1_mode` (`offline` by default, `live` opt-in)
- `--f1_live_source` (`auto|local|fastf1`)
- `--f1_live_model` (`ssm_v1`)
- `--f1_live_horizon_laps` (default `10`)
- `--f1_live_seed` (default `42`)
- `--f1_live_cache_dir`
- `--f1_live_replay_path`

## Optional Deep Learning Dependency

Install only if you want DL candidates:

```bash
pip install -r requirements-dl.txt
```

Without torch, the pipeline still works and DL candidates are skipped with explicit notes.
