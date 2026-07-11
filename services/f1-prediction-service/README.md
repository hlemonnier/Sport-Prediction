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

The service does **not** claim that a latest-state request ran the full
`packages/f1` live-race model. That model requires a causal lap-history trace,
which is not present in this request contract. Instead, each endpoint exposes
an explicitly named deterministic snapshot baseline:

- race: latest-state finishing-order baseline;
- qualifying: observed best-lap pace baseline;
- next lap: recent pace plus tyre-state baseline;
- strategy: strategy recommendations with current-order context.

The race, qualifying, and next-lap scores are target-specific. Position
probabilities are emitted as a jointly balanced assignment matrix: every
driver distribution sums to one and every position column sums to one. This
also guarantees one unit of win probability, three units of podium
probability, and ten units of points probability for a normal 20-driver field.

Where its input contract is satisfied, the service uses the canonical
`packages/f1/models/live_race/strategy.py`
`BaselineStrategyPolicyAdapter`. If the package adapter cannot be imported, it
falls back to a deterministic local strategy policy and records that fallback
under response diagnostics. Diagnostics also state why the canonical full
live-race runner was unavailable.

## Local Run

Python `>=3.10` is required; Python 3.12 is recommended and used by the
container. Refuse a legacy Python 3.9 environment before installation:

```bash
cd services/f1-prediction-service
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
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

Suggested commit name: `docs: declare F1 prediction-service runtime and evidence boundary`
