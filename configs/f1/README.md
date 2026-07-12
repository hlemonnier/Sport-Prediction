# F1 Config

F1 config home for circuits, season policies, and model profiles.

- `profiles/`: the four user-facing mode profiles plus internal Live strategy
  and deep-model candidate profiles
- `seasons/`: season-specific regime and training policy config
- `circuits.yaml`: circuit metadata and static priors
- `maturity.json`: machine-readable stage, evidence, provider-contract, and
  runtime truth used to prevent README or code-presence claims from being
  mistaken for promotion evidence

Profiles describe candidate behavior; they do not prove model quality. A
candidate is production-replaceable only when its registry artifact exists and
all required locked reports pass. Empty metrics, missing reports, or a
`not_promotable_until_locked_reports_exist` gate are explicit failures, not
documentation TODOs.

Circuit metadata is a static prior. Circuit features remain quarantined by
default and must be enabled only for a named research ablation until a current,
population-matched comparison demonstrates benefit.

The canonical four-mode contract is implemented in
`packages/f1/orchestration/contracts.py`: Qualifying Prediction, Race Final
Position, Best Estimated Lap Time, and Live Race Intelligence. `pre_quali`,
`pre_race`, `ultimate_lap_time`, and `live_race` are legacy implementation
aliases, not additional product modes.

Suggested commit name: `docs: add machine-readable F1 maturity evidence contract`
