from copy import deepcopy
import pytest

from research.experiments.boundary_20260908.football_xg_data import join_quality as q


def sample():
    return {"dates": [{"id": "1", "isResult": True, "datetime": "2022-08-05 19:00:00",
        "h": {"id": "10", "title": "Manchester City"}, "a": {"id": "11", "title": "Arsenal"},
        "goals": {"h": "2", "a": "1"}, "xG": {"h": "1.75", "a": "0.8"},
        "forecast": {"w": "0.8"}}]}


def canonical(day="2022-08-05"):
    return [{"league": "E0", "season_start_year": 2022, "home": "Man City", "away": "Arsenal",
        "canonical_date": day, "canonical_home_goals": 2, "canonical_away_goals": 1,
        "match_id": "E0:2022:Man City:Arsenal"}]


def test_strict_true_and_explicit_completion_exclusion():
    data = sample(); data["dates"][0]["isResult"] = "false"
    parsed, excluded = q.parse_matches(data, "E0", 2022)
    assert not parsed and len(excluded) == 1


@pytest.mark.parametrize("value", [None, True, "nan", "inf", "-0.1"])
def test_invalid_xg_is_explicitly_excluded(value):
    data = sample(); data["dates"][0]["xG"]["h"] = value
    parsed, excluded = q.parse_matches(data, "E0", 2022)
    assert not parsed and len(excluded) == 1


def test_alias_exact_join_preserves_goal_and_xg_units():
    parsed, excluded = q.parse_matches(sample(), "E0", 2022)
    assert not excluded
    joined, unmatched = q.join_rows(parsed, canonical())
    assert not unmatched and joined[0]["accepted_exact_join"] is True
    assert joined[0]["home_xg"] == 1.75 and joined[0]["home_goals"] == 2
    assert joined[0]["provider_original_published_at"] is None
    assert "forecast" not in joined[0]


def test_date_disagreement_is_not_silently_retimed():
    parsed, _ = q.parse_matches(sample(), "E0", 2022)
    joined, unmatched = q.join_rows(parsed, canonical("2022-08-16"))
    assert not unmatched and not joined[0]["accepted_exact_join"]
    assert joined[0]["canonical_minus_provider_days"] == 11
    assert joined[0]["provider_date"] == "2022-08-05"


def test_score_disagreement_excluded_even_when_date_matches():
    parsed, _ = q.parse_matches(sample(), "E0", 2022)
    rows = canonical(); rows[0]["canonical_home_goals"] = 0
    joined, _ = q.join_rows(parsed, rows)
    assert joined[0]["join_status"] == "score_disagreement_excluded"


def test_reversed_fixture_is_not_joined():
    parsed, _ = q.parse_matches(sample(), "E0", 2022)
    rows = canonical(); rows[0]["home"], rows[0]["away"] = "Arsenal", "Man City"
    joined, unmatched = q.join_rows(parsed, rows)
    assert joined[0]["join_status"] == "unmatched_team_or_fixture" and len(unmatched) == 1


def test_duplicate_provider_identity_or_fixture_rejected():
    data = sample(); data["dates"].append(deepcopy(data["dates"][0]))
    with pytest.raises(ValueError):
        q.parse_matches(data, "E0", 2022)
    parsed, _ = q.parse_matches(sample(), "E0", 2022)
    with pytest.raises(ValueError):
        q.join_rows(parsed+parsed, canonical())
