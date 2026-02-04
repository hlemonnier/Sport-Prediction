# Football Data (MVP)

Place local data files here. CSV is supported in the MVP. Parquet will be wired next.

## Required files
- `teams.csv` (or `teams.parquet`)
- `matches.csv` (or `matches.parquet`)
- `fixtures.csv` (or `fixtures.parquet`)

## Teams schema
- `team_id` (string)
- `team_name` (string)
- `league` (string, optional)
- `country` (string, optional)
- `team_aliases` (string, optional)

## Matches schema
- `match_id` (string)
- `date` (YYYY-MM-DD or ISO datetime)
- `season` (int)
- `league` (string)
- `home_team_id` (string)
- `away_team_id` (string)
- `home_goals` (int)
- `away_goals` (int)

Optional:
- `home_xg`, `away_xg` (float)
- `home_odds`, `draw_odds`, `away_odds` (float)
- `venue` (string)
- `home_red`, `away_red` (int)

## Fixtures schema
Same as Matches, but `home_goals`/`away_goals` should be empty or null.
