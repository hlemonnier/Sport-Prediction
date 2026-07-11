# F1 Package

Authoritative F1 package for the sport prediction system.

## Maturity

| Stage | Current status | What may be claimed |
| --- | --- | --- |
| Pre-Quali | Executable research | Produces rankings and probability fields; predictive edge and probability calibration remain run-gated. |
| Pre-Race | Executable research | Supports predicted-grid, post-qualifying, and official-grid horizons; each horizon needs its own passing evaluation. |
| Live Race | Experimental research | State-space/replay code exists; the platform currently serves explicitly named untrained snapshot baselines, not a promoted trace model. |
| Live Strategy/RL | Experimental, fail-closed | Candidate registry metrics and locked promotion reports are required before replacement of a deterministic fallback. |
| Ultimate Lap-Time | Experimental, fail-closed | Baseline and deep-model code exist; no production replacement is claimable without the locked evaluation and promotion bundle. |

The authoritative machine-readable status and evidence policy lives in
`configs/f1/maturity.json`. A runnable path, green unit tests, or an emitted
probability column is not evidence of production readiness or model edge.

- `data/providers`: FastF1, OpenF1, and local-weekend adapters
- `data/schemas`: session, circuit, driver, and result contracts
- `features`: circuit, practice, qualifying, race, strategy, weather, and live-state features
- `models/pre_quali`: train/predict/evaluate surface for qualifying prediction
- `models/pre_race`: train/predict/evaluate surface for race prediction
- `models/live_race`: state/strategy/predict surface for live race prediction
- `models/ultimate_lap_time`: train/predict surface for theoretical best lap pace
- `orchestration`: weekend pipeline, same-season backtest, scenarios, and contracts
- `betting`: F1 betting recommendation helpers

Legacy F1 experiment scripts now import this package directly.

## Provider Feature Contract

`LocalWeekendProvider`, `FastF1Provider`, and `OpenF1Provider` normalize
practice laps through `data/providers/practice_features.py` under contract
`f1_practice_lap_features_v2`. The shared contract defines:

- causal pre-qualifying session selection;
- lap validity and pit-lap filtering;
- best/top-three/median pace and variability;
- slow-lap, qualifying-simulation, race-simulation, wet-simulation, and
  qualifying-versus-race features;
- provider and contract-version provenance on the resulting rows.

This guarantees consistent feature *semantics*, not identical raw coverage or
values. FastF1, OpenF1, and local snapshots can still differ in freshness,
missing stint/tyre metadata, lap deletion flags, or session availability. A run
must retain `fp_feature_contract_version`, feature counts, and source notes so
those differences remain observable.

## Evidence Boundaries

- Circuit cards are static research priors and are quarantined by default.
  Enable them only for an explicit ablation; the stored review has not shown a
  reliable gain.
- Weather integration currently changes context and uncertainty priors. It is
  not a validated driver-specific wet-pace model.
- Probability fields are model outputs. Call them calibrated only when the
  emitted out-of-fold probability audit passes for the exact deployed score
  transformation and information horizon.
- Betting helpers are execution-free research utilities. Their presence does
  not make an unvalidated probability forecast decision-ready.

Suggested commit name: `docs: define F1 package maturity and provider contracts`
