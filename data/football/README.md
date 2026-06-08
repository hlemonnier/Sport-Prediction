# Football Data (MVP)

Place local data files here. CSV and parquet are supported.

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
- `round` or `round_number` (int, optional but recommended)
- `home_team_id` (string)
- `away_team_id` (string)
- `home_goals` (int)
- `away_goals` (int)

Optional:
- `home_xg`, `away_xg` (float)
- `home_odds`, `draw_odds`, `away_odds` (float)
- `venue` (string)
- `venue_latitude`, `venue_longitude` (float, required for fixture-level weather unless CLI fallback coordinates are supplied)
- `timezone` (IANA timezone string, optional but recommended)
- `home_red`, `away_red` (int)

## Fixtures schema
Same as Matches, but `home_goals`/`away_goals` should be empty or null.
