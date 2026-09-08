# Understat completed-match xG feasibility: verified data, no model fitting

The public source is viable. Twelve league-season lists cover EPL/E0, La liga/SP1 and Serie A/I1 for season-start labels 2022–2025 (2022/23 through 2025/26). All 4,560 fixtures have completed results and finite nonnegative home/away xG. The exact date/team/score join accepts **4,555 matches (99.89035%)**; five date disagreements are explicitly excluded. There are no missing canonical fixtures, invalid xG rows or goal disagreements.

The [public EPL2022 page](https://understat.com/league/EPL/2022) now obtains data through the [first-party league JavaScript](https://understat.com/js/league.min.js?t=1765269520), rather than embedding the old `datesData` field. Its unauthenticated jQuery request is `GET /getLeagueData/{league display name}/{season start year}` with the standard `X-Requested-With: XMLHttpRequest` header. No credentials, cookies, payment or authentication bypass was used. Earlier failed old-schema and plain-GET probes remain preserved separately. Raw pages, actual URLs, request receipts, retrieval timestamps and SHA256 hashes are retained locally.

## Data contract

Frozen joined file: `data/football/boundary_20260908/understat/joined_matches.csv`.

SHA256: `795d0a14dac04c0166ef9fc62ecfc9fa0c219b4945ee55b7af4577b27940051d`.

The CSV has 4,560 rows and 24 columns. The complete schema is in its header. Core fields are `league` (E0/SP1/I1), `season_start_year`, `canonical_match_id`, `canonical_date`, `home`, `away`, `home_goals`, `away_goals`, `home_xg`, `away_xg`, and `accepted_exact_join`. All source identities and source/canonical dates and goals remain side-by-side. `canonical_match_id` exactly follows the current cache convention: `{league}:{season_start_year}:{home}:{away}`. Only within these double-round-robin seasons is that fixture identity unique. The strict join additionally requires equal dates and equal final goals.

Use only rows whose parsed boolean `accepted_exact_join` is true. Do not use Python truthiness on CSV strings (`"False"` is truthy). Team alignment uses the explicit 16-entry `team_aliases.json`, with exact names otherwise; no fuzzy matching or score-based team identification is used. xG is in expected goals, not probabilities. Source `forecast`, odds, player records and squad features are not extracted or used.

`provider_datetime_naive` preserves the original text. Both original xG publication and last revision timestamps are unknown and stored as empty CSV fields/JSON null. Retrieval dates do not establish original availability. A later separately frozen model could assign past-match xG availability at **the next local midnight after the later validated completion day, plus seven full days**, with fourteen days as a sensitivity. This remains a conservative assumption rather than point-in-time evidence. Same-match xG must never enter a prematch prediction. Do not silently retime or accept unresolved date discrepancies.

## Date exceptions

| Fixture | Understat day | Canonical day | Status |
|---|---|---|---|
| Granada–Ath Bilbao, 2023 season | 2023-12-12 | 2023-12-11 | Excluded; official club records December11 resumption/completion |
| Valencia–Oviedo, 2025 season | 2025-09-29 | 2025-09-30 | Excluded; postponed from September29 |
| Udinese–Roma, 2023 season | 2024-04-14 | 2024-04-25 | Excluded; original interruption versus resumption day |
| Cagliari–Fiorentina, 2023 season | 2024-05-24 | 2024-05-23 | Excluded; provider day later than official matchday |
| Genoa–Atalanta, 2024 season | 2025-05-18 | 2025-05-17 | Excluded; provider day later than official matchday |

`timing_review.json` binds these observations to primary club, federation and league sources. The exact-date Fiorentina–Inter 2024-season row (2025-02-06) is also marked: both sources record completion after the December1 interruption, not the original kickoff. [Inter's official resumption notice](https://www.inter.it/en/news/rescheduled-continuation-fiorentina-inter-6-february-2025) makes that distinction explicit. No original data or dates were edited.

## Verification and artifacts

Eleven tests pass. `verify.py` reconstructs every joined row, verifies the 14 frozen canonical input/manifest files, 24 raw/receipt files and 13 derived CSVs, and independently matches all 9,120 home/away team-history xG/goals records against each match list. Each of the five exclusions remains explicit.

- Acquisition manifest: `artifacts/research/boundary_20260908/football_xg_data/public_ajax_manifest.json`, SHA256 `eb3a90a2daf5079fb92df0c66012854c6dc01ca0175a50fe9efb8b202b9c999b`.
- Quality report: `artifacts/research/boundary_20260908/football_xg_data/join_quality.json`, SHA256 `604c629eb9a5f95423b7c61cfc1d6430f78f400d656c6bfee4ef86534b9f1bde`.
- Verification: `artifacts/research/boundary_20260908/football_xg_data/verification.json`, SHA256 `560a258e0d303b5f9b4a310b9e090bfcebd93c02b619b5d795d135929a825123`.

The twelve raw JSON responses total 6,509,984 bytes; the joined CSV is 841,229 bytes. Raw responses remain local research inputs; no open redistribution license is asserted. No model was fitted and no predictive performance score was calculated.

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/football_xg_data/test_quality.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 .venv-f1/bin/python research/experiments/boundary_20260908/football_xg_data/verify.py
```

Suggested commit: `research(football): acquire and verify public historical match xG`.
