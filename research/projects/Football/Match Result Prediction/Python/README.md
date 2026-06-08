# Match Result Prediction (Python)

This folder contains the football experiment runner compatibility layer.
Authoritative football package code now lives in `../../../../../packages/football`.
It currently ships deterministic and ML-backed local prediction paths for football match outcomes.

The canonical experiment contract is declared in `../experiment.json` (context, snapshot root, model families, entrypoint).

## Quick start

```bash
cd "Match Result Prediction/Python"
python run_experiment.py --mode match_result --league epl --season 2025 --round 1
```

Compatibility wrapper:
- `run_prediction.py` forwards to `run_experiment.py`.
- `mrp` imports are bridged to `packages/football/mrp`.

## Modes
- `match_result`: predicts 1X2 outcomes (home/draw/away).
- `scoreline`: predicts a likely scoreline when lineups/odds are available.

## Notes
- `--data-source` supports local folder discovery and can be extended for multiple providers (APIs, exports, or internal data).
- Current baseline is deterministic and returns 1X2 probabilities, ranked outcomes, scoreline estimate, and training diagnostics.
- `--weather on` attaches Open-Meteo fixture weather when fixture venue coordinates or fallback coordinates are available.
