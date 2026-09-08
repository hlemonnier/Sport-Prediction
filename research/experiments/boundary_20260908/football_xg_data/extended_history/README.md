# Older xG warmup extension: 2017–2021

This separately authorized extension adds exactly fifteen public Understat league-season lists: EPL/E0, La liga/SP1 and Serie A/I1, season-start years 2017–2021. It preserves the first twelve 2022–2025 files, manifests and report without modification. No 2026-start season was requested, and no model fitting or predictive performance evaluation occurred.

All 5,700 completed fixtures have finite nonnegative xG. Home/away fixtures and final scores match the original canonical cache completely. **5,694 rows pass the exact date/team/score join; six date discrepancies are excluded.** Four additional team aliases are explicit in this directory's `team_aliases.json`; the original sixteen aliases remain unchanged.

| Excluded canonical fixture ID | Understat day | Canonical day |
|---|---|---|
| SP1:2017:Eibar:Leganes | 2017-09-16 | 2017-09-15 |
| SP1:2017:Celta:Ath Madrid | 2017-10-23 | 2017-10-22 |
| SP1:2017:Barcelona:Sevilla | 2017-11-05 | 2017-11-04 |
| SP1:2021:Espanol:Valencia | 2022-05-15 | 2022-05-14 |
| I1:2017:Sassuolo:Atalanta | 2018-01-28 | 2018-01-27 |
| I1:2017:Chievo:Juventus | 2018-01-28 | 2018-01-27 |

These differences are reported directly from the sources. Their cause has not been established, and no dates were corrected. The original five later-season exclusions and the Fiorentina–Inter completion-day marker remain intact.

## Frozen files

- Older joined CSV: `data/football/boundary_20260908/understat/extended_history/joined_matches.csv` (5,700 rows, the same 24 columns as the first file; 1,047,740 bytes). SHA256 `daf37e223277265d3cd1bcc3dd04243437cce4231759e9bd29e79668cec07ced`.
- Acquisition manifest: `artifacts/research/boundary_20260908/football_xg_data/extended_history/acquisition_manifest.json`, SHA256 `f7f6bdde2f805ac8d1942c3324e794973e986beb168355f76c3a7a8dd7da33c2`.
- Quality report: `artifacts/research/boundary_20260908/football_xg_data/extended_history/join_quality.json`, SHA256 `b2420287213b2d239d2eab6d260528979d0e9518b003241c7728ec87ed499ee3`.
- Verification: `artifacts/research/boundary_20260908/football_xg_data/extended_history/verification.json`, SHA256 `1dde653f2ac728c6cc215ad9e7ce44c64bf3b41d56e17d64dd67fa17487de60d`.

The fifteen raw responses total 7,990,353 bytes. All 30 raw/receipt hashes, 17 original canonical CSV/manifest hashes and 16 derived CSV hashes are verified. Reconstruction reproduces every joined row and independently checks 11,400 home/away team-history xG and goal mirrors. No provider match ID overlaps the first twelve seasons.

## Combined model-input contract

Concatenate the two frozen joined CSVs listed in `artifacts/research/boundary_20260908/football_xg_data/availability_contract.json`. The result has **10,260 matches across 27 league-seasons; 10,249 accepted rows and eleven explicit exclusions**. Keep the original future prediction population; an unavailable xG history needs a separately frozen fallback rather than dropping target fixtures.

The combined contract SHA256 is `dd924f3d0b7c010f9482c782daeb20ce59c826a97e3466d8f8a6f83862569e56`. It binds both CSVs, quality reports, verification, timing markers and the new `availability.py` helper. Publication and revision times remain unknown.

The proposed availability proxy is exact: take the next local midnight after the validated completion day, convert to UTC, then add **168 hours** (primary) or **336 hours** (sensitivity). Timezones are Europe/London, Europe/Madrid and Europe/Rome. This order makes DST behavior explicit. Only accepted rows are eligible; the helper rejects a false CSV string or mismatched accepted date/goals. The delay is an assumption and must not be described as observed point-in-time publication. Use only prior matches available by the forecast cutoff, never the target's own xG.

Fourteen additional tests pass (25 across the complete data lane), including all new aliases, strict CSV boolean handling, seven/fourteen-day delays, a DST boundary and the Fiorentina–Inter completion date.

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/football_xg_data/extended_history/test_extended.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 .venv-f1/bin/python research/experiments/boundary_20260908/football_xg_data/extended_history/quality.py verify
```

Suggested commit: `research(football): extend verified xG history with explicit availability delays`.
