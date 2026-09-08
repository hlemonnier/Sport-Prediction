from datetime import datetime, timezone
import pytest

from research.experiments.boundary_20260908.football_xg_data import availability
from research.experiments.boundary_20260908.football_xg_data.extended_history import quality


@pytest.mark.parametrize("league,provider,expected", [
    ("E0", "West Bromwich Albion", "West Brom"),
    ("SP1", "Deportivo La Coruna", "La Coruna"),
    ("SP1", "SD Huesca", "Huesca"),
    ("I1", "SPAL 2013", "Spal"),
])
def test_extra_aliases_are_explicit(league, provider, expected):
    payload = {"dates": [{"id": "1", "isResult": True, "datetime": "2017-08-20 18:00:00",
        "h": {"id": "1", "title": provider}, "a": {"id": "2", "title": "Other"},
        "goals": {"h": "1", "a": "0"}, "xG": {"h": "1.5", "a": "0.5"}}]}
    records, rejected = quality.parse(payload, league, 2017)
    assert not rejected and records[0]["home"] == expected


def row(day="2025-02-06", league="I1", accepted="True"):
    return {"accepted_exact_join": accepted, "league": league, "canonical_date": day, "provider_date": day,
            "home_goals": "3", "away_goals": "0", "canonical_home_goals": "3", "canonical_away_goals": "0"}


def test_completed_resumption_day_plus_full_utc_delay():
    assert availability.proxy_available_at(row()) == datetime(2025, 2, 13, 23, tzinfo=timezone.utc)
    assert availability.proxy_available_at(row(), 14) == datetime(2025, 2, 20, 23, tzinfo=timezone.utc)


@pytest.mark.parametrize("flag", ["False", "false", False, "true", "1", 1, None])
def test_non_exact_true_never_accepts_excluded_csv_row(flag):
    assert availability.proxy_available_at(row(accepted=flag)) is None


def test_delay_is_168_hours_after_midnight_even_across_dst():
    # Next London midnight is March30 00UTC; DST changes later that morning.
    assert availability.proxy_available_at(row("2025-03-29", "E0")) == datetime(2025, 4, 6, tzinfo=timezone.utc)


def test_claimed_accepted_date_mismatch_raises():
    value = row(); value["provider_date"] = "2024-12-01"
    with pytest.raises(ValueError):
        availability.proxy_available_at(value)
