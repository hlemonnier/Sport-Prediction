"""Data loading and filtering for football match prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from .config import PredictionConfig
from .constants import (
    AWAY_GOALS_COLUMNS,
    AWAY_TEAM_COLUMNS,
    AWAY_XG_COLUMNS,
    DATE_COLUMNS,
    HOME_GOALS_COLUMNS,
    HOME_TEAM_COLUMNS,
    HOME_XG_COLUMNS,
    LEAGUE_COLUMNS,
    MATCH_ID_COLUMNS,
    ROUND_COLUMNS,
    SEASON_COLUMNS,
    SUPPORTED_DATA_EXTENSIONS,
    TEAM_ID_COLUMNS,
    TEAM_NAME_COLUMNS,
)
from .utils import (
    canonical_team_id,
    discover_data_directory,
    first_non_empty,
    normalize_text,
    parse_datetime,
    parse_float,
    parse_int,
    record_sort_key,
)


@dataclass(frozen=True)
class TeamRecord:
    team_id: str
    team_name: str
    league: str | None = None


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    date: datetime | None
    season: int | None
    league: str | None
    round_number: int | None
    home_team_id: str
    away_team_id: str
    home_goals: int | None
    away_goals: int | None
    home_xg: float | None
    away_xg: float | None


@dataclass(frozen=True)
class FixtureRecord:
    match_id: str
    date: datetime | None
    season: int | None
    league: str | None
    round_number: int | None
    home_team_id: str
    away_team_id: str


@dataclass
class LocalFootballData:
    data_dir: Path | None
    teams: dict[str, TeamRecord]
    matches: list[MatchRecord]
    fixtures: list[FixtureRecord]

    def resolve_team_name(self, team_id: str) -> str:
        team = self.teams.get(team_id)
        if team is None:
            return team_id
        return team.team_name


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(key): value for key, value in row.items()} for row in reader]


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    # Try pandas first, then pyarrow directly.
    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    except Exception as pandas_exc:
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(path)
            return table.to_pylist()
        except Exception as pyarrow_exc:
            raise RuntimeError(
                f"Parquet non lisible ({pandas_exc.__class__.__name__}; {pyarrow_exc.__class__.__name__})"
            ) from pyarrow_exc


def _load_table_rows(data_dir: Path, table_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    tried_paths = [data_dir / f"{table_name}.{ext}" for ext in SUPPORTED_DATA_EXTENSIONS]
    for path in tried_paths:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                rows = _read_csv_rows(path)
            else:
                rows = _read_parquet_rows(path)
        except Exception as exc:  # pragma: no cover - error-path robustness
            notes.append(f"{table_name}: echec de lecture {path.name} ({exc}).")
            continue
        notes.append(f"{table_name}: {len(rows)} lignes chargees depuis {path.name}.")
        return rows, notes

    notes.append(f"{table_name}: fichier manquant (attendu: {', '.join(p.name for p in tried_paths)}).")
    return [], notes


def _parse_team_row(row: dict[str, Any]) -> TeamRecord | None:
    team_id_raw = first_non_empty(row, TEAM_ID_COLUMNS)
    team_name_raw = first_non_empty(row, TEAM_NAME_COLUMNS)
    if team_id_raw is None and team_name_raw is None:
        return None
    if team_id_raw is None:
        team_id_raw = team_name_raw
    if team_name_raw is None:
        team_name_raw = team_id_raw

    team_id = canonical_team_id(team_id_raw)
    team_name = str(team_name_raw).strip()
    if not team_id:
        return None
    if not team_name:
        team_name = team_id

    league_raw = first_non_empty(row, LEAGUE_COLUMNS)
    league = str(league_raw).strip() if league_raw is not None else None
    return TeamRecord(team_id=team_id, team_name=team_name, league=league)


def _parse_match_row(row: dict[str, Any], default_match_id_prefix: str, index: int) -> MatchRecord | None:
    home_team_raw = first_non_empty(row, HOME_TEAM_COLUMNS)
    away_team_raw = first_non_empty(row, AWAY_TEAM_COLUMNS)
    if home_team_raw is None or away_team_raw is None:
        return None

    home_team_id = canonical_team_id(home_team_raw)
    away_team_id = canonical_team_id(away_team_raw)
    if not home_team_id or not away_team_id:
        return None

    match_id_raw = first_non_empty(row, MATCH_ID_COLUMNS)
    if match_id_raw is None:
        match_id = f"{default_match_id_prefix}_{index}"
    else:
        match_id = str(match_id_raw).strip() or f"{default_match_id_prefix}_{index}"

    date = parse_datetime(first_non_empty(row, DATE_COLUMNS))
    season = parse_int(first_non_empty(row, SEASON_COLUMNS))
    league_raw = first_non_empty(row, LEAGUE_COLUMNS)
    league = str(league_raw).strip() if league_raw is not None else None
    round_number = parse_int(first_non_empty(row, ROUND_COLUMNS))
    home_goals = parse_int(first_non_empty(row, HOME_GOALS_COLUMNS))
    away_goals = parse_int(first_non_empty(row, AWAY_GOALS_COLUMNS))
    home_xg = parse_float(first_non_empty(row, HOME_XG_COLUMNS))
    away_xg = parse_float(first_non_empty(row, AWAY_XG_COLUMNS))

    return MatchRecord(
        match_id=match_id,
        date=date,
        season=season,
        league=league,
        round_number=round_number,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals=home_goals,
        away_goals=away_goals,
        home_xg=home_xg,
        away_xg=away_xg,
    )


def _as_fixture(match: MatchRecord) -> FixtureRecord:
    return FixtureRecord(
        match_id=match.match_id,
        date=match.date,
        season=match.season,
        league=match.league,
        round_number=match.round_number,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
    )


def _fixture_key(fixture: FixtureRecord) -> tuple[str, str, str, str]:
    return (
        fixture.home_team_id,
        fixture.away_team_id,
        fixture.date.isoformat() if fixture.date else "",
        f"{fixture.season or ''}-{fixture.round_number or ''}",
    )


def load_local_football_data(config: PredictionConfig) -> tuple[LocalFootballData, list[str]]:
    notes: list[str] = []
    data_dir, searched_candidates = discover_data_directory(config.data_source)

    if data_dir is None:
        pretty_candidates = ", ".join(str(candidate) for candidate in searched_candidates[:5])
        notes.append(
            "Aucun dossier local data/football detecte. "
            f"Recherches effectuees: {pretty_candidates if pretty_candidates else 'aucun chemin candidate'}."
        )
        return LocalFootballData(data_dir=None, teams={}, matches=[], fixtures=[]), notes

    notes.append(f"Donnees locales detectees: {data_dir}.")

    teams_rows, team_notes = _load_table_rows(data_dir, "teams")
    matches_rows, match_notes = _load_table_rows(data_dir, "matches")
    fixtures_rows, fixture_notes = _load_table_rows(data_dir, "fixtures")
    notes.extend(team_notes)
    notes.extend(match_notes)
    notes.extend(fixture_notes)

    teams: dict[str, TeamRecord] = {}
    for row in teams_rows:
        parsed = _parse_team_row(row)
        if parsed is None:
            continue
        teams[parsed.team_id] = parsed

    parsed_matches: list[MatchRecord] = []
    fixture_candidates_from_matches: list[FixtureRecord] = []
    for idx, row in enumerate(matches_rows):
        parsed = _parse_match_row(row, default_match_id_prefix="m", index=idx)
        if parsed is None:
            continue
        if parsed.home_goals is None or parsed.away_goals is None:
            fixture_candidates_from_matches.append(_as_fixture(parsed))
            continue
        parsed_matches.append(parsed)

    parsed_fixtures: list[FixtureRecord] = []
    for idx, row in enumerate(fixtures_rows):
        parsed = _parse_match_row(row, default_match_id_prefix="f", index=idx)
        if parsed is None:
            continue
        if parsed.home_goals is not None and parsed.away_goals is not None:
            continue
        parsed_fixtures.append(_as_fixture(parsed))

    deduped_fixtures: list[FixtureRecord] = []
    seen_fixture_keys: set[tuple[str, str, str, str]] = set()
    for fixture in [*parsed_fixtures, *fixture_candidates_from_matches]:
        key = _fixture_key(fixture)
        if key in seen_fixture_keys:
            continue
        seen_fixture_keys.add(key)
        deduped_fixtures.append(fixture)

    for record in [*parsed_matches, *deduped_fixtures]:
        for team_id in (record.home_team_id, record.away_team_id):
            if team_id in teams:
                continue
            teams[team_id] = TeamRecord(team_id=team_id, team_name=team_id)

    parsed_matches.sort(key=record_sort_key)
    deduped_fixtures.sort(key=record_sort_key)
    notes.append(f"Matches historiques utilisables: {len(parsed_matches)}.")
    notes.append(f"Fixtures candidates: {len(deduped_fixtures)}.")

    return (
        LocalFootballData(
            data_dir=data_dir,
            teams=teams,
            matches=parsed_matches,
            fixtures=deduped_fixtures,
        ),
        notes,
    )


TRecord = TypeVar("TRecord")


def _filter_by_league(records: list[TRecord], league: str) -> tuple[list[TRecord], bool]:
    target = normalize_text(league)
    if not records:
        return [], False
    with_league = [record for record in records if normalize_text(getattr(record, "league", None))]
    if not with_league:
        return list(records), False
    filtered = [record for record in records if normalize_text(getattr(record, "league", None)) == target]
    return filtered, True


def select_target_fixtures(
    dataset: LocalFootballData, config: PredictionConfig
) -> tuple[list[FixtureRecord], list[str]]:
    notes: list[str] = []
    fixtures = list(dataset.fixtures)
    if not fixtures:
        notes.append("Aucun fixture disponible localement.")
        return [], notes

    league_filtered, league_filter_applied = _filter_by_league(fixtures, config.league)
    if league_filter_applied:
        notes.append(f"Fixtures apres filtre league='{config.league}': {len(league_filtered)}.")
    else:
        notes.append("Fixtures sans colonne league exploitable, aucun filtre league applique.")

    season_candidates = [fixture for fixture in league_filtered if fixture.season is not None]
    if season_candidates:
        season_filtered = [fixture for fixture in league_filtered if fixture.season == config.season]
        notes.append(f"Fixtures apres filtre season={config.season}: {len(season_filtered)}.")
    else:
        season_filtered = league_filtered
        notes.append("Fixtures sans colonne season exploitable, aucun filtre season applique.")

    if not season_filtered:
        return [], notes

    with_round = [fixture for fixture in season_filtered if fixture.round_number is not None]
    if with_round:
        round_filtered = [
            fixture for fixture in season_filtered if fixture.round_number == config.round_number
        ]
        if round_filtered:
            notes.append(f"Fixtures apres filtre round={config.round_number}: {len(round_filtered)}.")
            season_filtered = round_filtered
        else:
            notes.append(
                f"Aucun fixture pour round={config.round_number}; fallback sur fixtures de la saison."
            )
    else:
        notes.append("Fixtures sans colonne round exploitable, aucun filtre round applique.")

    season_filtered.sort(key=record_sort_key)
    return season_filtered, notes


def select_training_matches(
    dataset: LocalFootballData, config: PredictionConfig, fixtures: list[FixtureRecord]
) -> tuple[list[MatchRecord], list[str]]:
    notes: list[str] = []
    matches = [
        match
        for match in dataset.matches
        if match.home_goals is not None and match.away_goals is not None
    ]
    if not matches:
        notes.append("Aucun match historique avec score final trouve.")
        return [], notes

    league_filtered, league_filter_applied = _filter_by_league(matches, config.league)
    if league_filter_applied:
        notes.append(f"Training matches apres filtre league='{config.league}': {len(league_filtered)}.")
    else:
        notes.append("Training matches sans colonne league exploitable, aucun filtre league applique.")
    matches = league_filtered

    seasons = config.train_seasons or []
    if seasons:
        with_season = [match for match in matches if match.season is not None]
        if with_season:
            season_set = set(seasons)
            season_filtered = [match for match in matches if match.season in season_set]
            notes.append(f"Training matches apres filtre train_seasons={sorted(season_set)}: {len(season_filtered)}.")
            matches = season_filtered
        else:
            notes.append("Training matches sans colonne season exploitable, aucun filtre train_seasons applique.")

    if not matches:
        notes.append("Aucun match historique apres filtres.")
        return [], notes

    round_aware: list[MatchRecord] = []
    for match in matches:
        if match.season == config.season and match.round_number is not None:
            if match.round_number >= config.round_number:
                continue
        round_aware.append(match)
    if len(round_aware) != len(matches):
        notes.append(
            f"Training matches apres anti-fuite round<{config.round_number} sur saison cible: {len(round_aware)}."
        )
    matches = round_aware

    dated_fixtures = [fixture for fixture in fixtures if fixture.date is not None]
    if dated_fixtures:
        cutoff = min(fixture.date for fixture in dated_fixtures if fixture.date is not None)
        pre_fixture = [match for match in matches if match.date is None or match.date < cutoff]
        if pre_fixture:
            matches = pre_fixture
            notes.append(f"Training matches limites aux matchs avant {cutoff.date().isoformat()}: {len(matches)}.")

    matches.sort(key=record_sort_key)
    return matches, notes
