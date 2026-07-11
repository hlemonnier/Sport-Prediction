# F1 Profiles

Canonical home for new F1 model profiles. Legacy profile files are still present
under the research runner path until all backend discovery is migrated.

Current maturity:

- `pre_quali.yaml` and `pre_race.yaml` configure executable research paths; they
  are not promoted production models and have no generally proven baseline edge.
- `live_race.yaml` and `live_strategy.yaml` define experimental state,
  simulator, planner, and deterministic-policy contracts.
- `live_strategy_rl.yaml` is a candidate registry profile with empty metrics and
  a fail-closed `not_promotable_until_locked_reports_exist` gate.
- `ultimate_lap_time.yaml` defines the deterministic research baseline contract.
- `ultimate_lap_time_deep.yaml` is a candidate deep-model registry profile and
  is not promotable until its locked reports exist and pass.

Provider names in a profile do not imply interchangeable raw data. Practice
features are comparable only after normalization through
`f1_practice_lap_features_v2`; source coverage and missingness must still be
reported.

See `../maturity.json` for the machine-readable claim policy.

Suggested commit name: `docs: clarify F1 profile promotion and provider boundaries`
