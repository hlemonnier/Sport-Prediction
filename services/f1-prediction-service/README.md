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
position_semantics
forecast_available
unavailable_reason
eligibility_status
participation_status
dnf_semantics
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
probabilities are emitted as a jointly balanced assignment matrix over the
target-eligible field: every available driver distribution sums to one and
every eligible-position column sums to one. Unavailable rows have an empty
distribution, null expected/P10/P90 positions, zero event probabilities, and an
explicit reason. The matrix kernel uses continuous score differences, so a
1.0-second pace gap produces a stronger separation than a 0.01-second gap with
the same ordinal ranking. Sinkhorn balancing retains one-driver/one-position
assignment coherence without inventing positions for excluded drivers.

These are conditional position marginals, not unconditional calibrated event
probabilities:

- race means classification order conditional on the latest snapshot and
  observed participation/classification status;
- qualifying means order conditional on currently observed representative pace
  and participation status;
- next lap means relative pace order conditional on the current running field;
- strategy position rows are current-classification context only, never a
  finishing-order forecast, and their win/podium/points fields are zero.

Participation and target eligibility are distinct. The normalized
`participation_status` is one of `running_or_unknown`, `retired_or_stopped`,
`finished`, `dns`, `dsq`, `withdrawn`, or `classification_ineligible`.

- A retired or stopped race driver remains classification-eligible and can
  still score points under FIA classification rules. Its race row uses
  `classification_eligible_retired`, remains in the jointly balanced field,
  and has `dnf_probability: 1` as an observed-status indicator.
- A stopped qualifying driver remains eligible when it has a valid observed
  lap. Without a valid lap, the qualifying target is explicitly unavailable.
- DNS, DSQ, withdrawn, and unclassified drivers are excluded from race and
  qualifying distributions rather than forced into a fake tail position.
- Retired, stopped, and finished drivers have no next-lap target. Their
  next-lap rows are explicitly unavailable and carry no pace distribution.

`dnf_probability` is not an independently estimated or calibrated future
retirement probability. It is `1` only for an observed retired/stopped state
and `0` for running, finished, DNS, DSQ, withdrawn, or unclassified states.

Where its input contract is satisfied, the service uses the canonical
`packages/f1/models/live_race/strategy.py`
`BaselineStrategyPolicyAdapter`. If the package adapter cannot be imported, it
falls back to a deterministic local strategy policy and records that fallback
under response diagnostics. Diagnostics also state why the canonical full
live-race runner was unavailable.

Every emitted strategy recommendation is checked against the shared
`packages/f1/models/live_race/action_space.py` legal action mask. The snapshot
must expose current lap, total laps, remaining laps, tyre age, stint, current
and used compounds, available compounds, pit-lane state, red-flag state,
wet/dry state, and whether the car is already boxing. These fields are required
individually: the service does not infer a missing horizon or replace missing
lap/tyre values with zero. If state is incomplete—or if the policy's preferred
action fails the mask—the service returns `recommendedAction: null`,
`safeToRecommend: false`, an explicit reason, and partial legality evidence
with missing values preserved as null. It does not silently substitute a
fabricated legal strategy.

## Evidence and promotion status

This service remains a deterministic, untrained, uncalibrated snapshot
heuristic. The score scales and confidence values are engineering priors, not
empirically calibrated probabilities. Response diagnostics therefore report
`uncalibrated_heuristic_not_validated_for_promotion`, and this service must not
be promoted as the canonical live-race model without chronological validation,
calibration testing, and representative race replay evidence.

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

Suggested commit name: `fix(f1-prediction-service): separate classification and live eligibility`
