# Raw telemetry at the original next-lap issuance

This experiment tests whether recent driving measurements improve the existing
next-eligible-clean-lap forecast. It keeps the original completed-lap issuance,
the original 80 predictors and anchor, and the full forecast population. It does
not issue later sector-checkpoint forecasts. The runtime model remains unchanged
until predictive evidence supports a replacement.

The [specification](specification.json) fixes one measurement family and two
augmented fits before historical feature construction. Each receives 90 new
coordinates: 27 driving measurements and 63 availability/quality indicators.
The matched control replaces only the 27 measurement coordinates with zero.
Both use all original matched 2022 rows, the same HGB settings and event weights.
A third fit reconstructs the original 2022-only 80-feature baseline. Unsupported
telemetry returns that baseline exactly; forecasts and events are never dropped.

The primary experiment uses a two-second delivery lag. A zero-second sensitivity
uses the same fitted coefficients. Windows are 30, 90 and 180 seconds, weighted
equally by packet after aggregation within each packet. The [measurement
proposal](../telemetry_pilot/future_protocol_proposal.md) defines channel validity,
including the treatment of raw code 104, jointly valid throttle/brake support,
missing channels and repeated values. [Data assembly](data_contract.md) preserves
all 39,220 original issuances, including 850 unmatched outcomes. Exact archive
prefix clocks are source-availability proxies, not certified client receipts.

Selection uses all 22 races from 2023. Advancement requires at least 1% lower
event-balanced MAE against **both** the base model and the quality-only control,
negative event and three-event-block interval upper endpoints, negative mean
differences under every leave-one-event-out comparison, and a positive gain in
the zero-second sensitivity. This is a screening threshold, not the substantial
gain required for promotion. The later, separately gated evaluation requires
10% improvement on all 48 races from 2024–2025 and 5% on all 13 declared 2026
races, against both references, with the uncertainty requirements in the spec.

The [acquisition contract](ACQUISITION.md) allows only the 44 discovery sessions,
reusing the two independently verified pilots. Acquisition, target-free feature
closure, exhaustive stream validation, model fitting, forecast closure and label
attachment are separate stages. Existing outputs are not overwritten. A failed
screen closes this family without later-season acquisition or parameter changes.

Implementation and synthetic tests are research infrastructure. No telemetry
performance gain, prospective validation, calibrated interval or strategy value
is established until a closed execution result says so.

Run from the repository root with the pinned local Python environment. The
three execution stages are separate commands; a later stage requires the closed
outputs of the previous stage. `freeze` also reruns the complete local synthetic
suite. The review file must approve the exact current source hashes.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.telemetry.run freeze --review artifacts/research/boundary_20260908/telemetry/execution_review.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.telemetry.run prepare
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.telemetry.run select
```

Outputs live under `artifacts/research/boundary_20260908/telemetry/execution`.
Historical replay requires the hash-matched local lap and raw telemetry inputs;
synthetic tests require neither provider downloads nor fitted historical models.
The runner fixes numerical libraries to one thread and saves library versions.

Suggested commit: `research(f1-live): test raw telemetry at unchanged next-lap issuance`.
