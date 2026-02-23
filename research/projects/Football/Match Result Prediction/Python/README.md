# Match Result Prediction (Python)

This folder contains the production-style prediction code for football match outcomes.
It currently ships a deterministic baseline and diagnostics for local experimentation.

The canonical experiment contract is declared in `../experiment.json` (context, snapshot root, model families, entrypoint).

## Quick start

```bash
cd "Match Result Prediction/Python"
python run_experiment.py --mode match_result --league epl --season 2025 --round 1
```

Compatibility wrapper:
- `run_prediction.py` forwards to `run_experiment.py`.

## Modes
- `match_result`: predicts 1X2 outcomes (home/draw/away).
- `scoreline`: predicts a likely scoreline when lineups/odds are available.

## Notes
- `--data-source` supports local folder discovery and can be extended for multiple providers (APIs, exports, or internal data).
- Current baseline is deterministic and returns 1X2 probabilities, ranked outcomes, scoreline estimate, and training diagnostics.
