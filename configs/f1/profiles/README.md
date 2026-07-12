# F1 Profiles

Canonical home for new F1 model profiles. Legacy profile files are still present
under the research runner path until all backend discovery is migrated.

There are exactly four user-facing modes. Legacy file names are retained only
as compatibility aliases while callers migrate:

- `pre_quali.yaml` is **Qualifying Prediction**. It records the transparent
  causal rehearsal selector retained by the 2026 same-season walk-forward.
- `pre_race.yaml` is **Race Final Position**. It records the post-Qualifying
  order proxy for the currently evidenced horizon; causal residual/reliability
  challengers lose, while final-grid and terminal-status evidence remain absent.
- `ultimate_lap_time.yaml` is **Best Estimated Lap Time**. It separates the
  theoretical compatible-sector lower bound from the achievable session-end
  lap distribution; only the achievable point estimate has current evidence.
- `live_race.yaml` is **Live Race Intelligence**. Forecasting subcontracts and
  constrained decision subcontracts are separate. `live_strategy.yaml`
  supplies the simulator, legal action, DP, and MPC research contracts.
- `live_strategy_rl.yaml` is a candidate registry profile with empty metrics and
  a fail-closed `not_promotable_until_locked_reports_exist` gate. RL is eligible
  only for Live pit, compound, and pace decisions—not for ordinary forecasts.
- `ultimate_lap_time_deep.yaml` is a candidate deep-model registry profile and
  is not promotable until its locked reports and separate pre-Q feature/Q-target
  timestamp provenance exist and pass.

Point ranking, probabilistic calibration, intervals, terminal-status hazards,
and decision-policy value are promoted independently. A passing point metric
does not promote any of the others.

Provider names in a profile do not imply interchangeable raw data. Practice
features are comparable only after normalization through
`f1_practice_lap_features_v3_quality_weighted`; source coverage, lap-quality
counts, run-intent evidence, and missingness must still be
reported.

See `../maturity.json` for the machine-readable claim policy.

Suggested commit name: `docs: clarify F1 profile promotion and provider boundaries`
