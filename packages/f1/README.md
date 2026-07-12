# F1 Package

Authoritative F1 package for the sport prediction system.

## Maturity

| User-facing mode | Current status | What may be claimed |
| --- | --- | --- |
| Qualifying Prediction | Executable research | The 2026 causal point-ranking policy has evidence; its emitted probabilities and intervals are not promoted. |
| Race Final Position | Executable research | The post-Qualifying proxy baseline is retained for that horizon; final-grid and terminal-status components are not validated/implemented as complete products. |
| Best Estimated Lap Time | Executable research | The achievable point estimate has conditional 2026 walk-forward evidence with explicit coverage. Full-mode outcomes, intervals, and deep telemetry models remain gated. |
| Live Race Intelligence | Experimental research | The causal last-clean-lap baseline is the research-runner default after the emitted SSM blend failed its uncertainty gate. Seconds-valued next-lap output is not yet platform-integrated; RL remains decision-only and fail-closed. |

The authoritative machine-readable status and evidence policy lives in
`configs/f1/maturity.json`. A runnable path, green unit tests, or an emitted
probability column is not evidence of production readiness or model edge.

- `data/providers`: FastF1, OpenF1, and local-weekend adapters
- `data/schemas`: session, circuit, driver, and result contracts
- `features`: circuit, practice, qualifying, race, strategy, weather, and live-state features
- `models/pre_quali`: train/predict/evaluate surface for qualifying prediction
- `models/pre_race`: train/predict/evaluate surface for race prediction
- `models/live_race`: forecasting plus constrained pit/compound/pace decision research
- `models/ultimate_lap_time`: separate theoretical-sector-floor and achievable-best-lap surfaces
- `orchestration`: weekend pipeline, same-season backtest, scenarios, and contracts
- `betting`: F1 betting recommendation helpers

Legacy F1 experiment scripts now import this package directly.

## Provider Feature Contract

`LocalWeekendProvider`, `FastF1Provider`, and `OpenF1Provider` normalize
practice laps through `data/providers/practice_features.py` under contract
`f1_practice_lap_features_v3_quality_weighted`. The shared contract defines:

- season-aware, named point-in-time session selection;
- completed-session gating plus inaccurate, deleted, pit-transition, yellow,
  Safety Car, VSC, and red-flag lap filtering;
- best/top-three/median pace and variability;
- evidence-counted qualifying-simulation, race-simulation, wet-simulation,
  and raw long-run slope features without invented fallback laps or a claimed
  universal fuel correction;
- roster preservation for drivers with no representative lap;
- provider and contract-version provenance on the resulting rows.

This guarantees consistent feature *semantics*, not identical raw coverage or
values. FastF1, OpenF1, and local snapshots can still differ in freshness,
missing stint/tyre metadata, lap deletion flags, or session availability. A run
must retain `fp_feature_contract_version`, feature counts, and source notes so
those differences remain observable.

The target/cutoff matrix and 2026 regulation-era rules are documented in
`docs/architecture/f1_point_in_time_contract.md` and implemented in
`domain/weekend.py`.

## Evidence Boundaries

- Circuit cards are static research priors and are quarantined by default.
  Enable them only for an explicit ablation; the stored review has not shown a
  reliable gain.
- Weather integration currently changes context and uncertainty priors. It is
  not a validated driver-specific wet-pace model.
- Probability fields are model outputs. Call them calibrated only when the
  emitted out-of-fold probability audit passes for the exact deployed score
  transformation and information horizon, with chronological event-disjoint
  selection, calibration, and final-audit blocks.
- Historical live replay is causal only with a global event-time cutoff.
  Lap-number-only truncation is a diagnostic because different cars cross the
  same lap boundary at different timestamps.
- Ultimate Lap-Time sector minima are combined only inside compatible
  session/compound strata and remain a theoretical lower bound. They are not
  an expected achievable lap.
- Betting helpers are execution-free research utilities. Their presence does
  not make an unvalidated probability forecast decision-ready.

Suggested commit name: `feat(f1): enforce season-aware point-in-time prediction contracts`
