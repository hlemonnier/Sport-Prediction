# F1 Prediction Service

Separate model-service process for F1 platform prediction snapshots.

The service accepts a platform session snapshot and returns the stable
prediction contract consumed by `services/f1-platform`:

```text
model_version
prediction_time
source_event_sequence
features_version
driver_number
expected_position
position_p10
position_p90
position_distribution
win_probability
podium_probability
points_probability
dnf_probability
confidence
strategy
```

It uses the existing `packages/f1/models/live_race/strategy.py`
`BaselineStrategyPolicyAdapter` when pandas/numpy are available. If the package
adapter cannot be imported, it falls back to a deterministic local policy and
marks `diagnostics.strategyPolicyEnabled=false`.

## Local Run

```bash
cd services/f1-prediction-service
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH="../..:." uvicorn f1_prediction_service.app:create_app --factory --reload --port 8002
```

Open:

```text
http://localhost:8002/health
```

## Platform Wiring

Set this on the F1 platform API:

```text
F1_PLATFORM_PREDICTION_URL=http://localhost:8002
```

In Docker Compose, `f1-platform-api` is wired to:

```text
http://f1-prediction-service:8002
```

Suggested commit name: `add-f1-prediction-model-service`
