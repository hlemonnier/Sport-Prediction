"""Contracts for explicit transfer adaptation without changing the EPL model."""
from datetime import date, timedelta
import json
import pytest
from research.experiments.performance_20260907.football import transfer_run as t


def test_transfer_league_context_changes_only_timezone_and_required_fits():
    original_tz, original_dc = t.b.LONDON, t.b.DC
    with t.league_context("Europe/Rome"):
        assert t.b.LONDON.key == "Europe/Rome"
        assert t.b.DC == {"dc_equal": None, "dc_365": 365.}
        assert t.b.utc_midnight(date(2025, 8, 15)).isoformat() == "2025-08-14T22:00:00"
    assert t.b.LONDON is original_tz and t.b.DC is original_dc
    with pytest.raises(RuntimeError):
        with t.league_context("Europe/Madrid"):
            raise RuntimeError("test restoration")
    assert t.b.LONDON is original_tz and t.b.DC is original_dc


def test_transfer_parameters_match_frozen_parent_and_no_new_selection():
    spec = json.loads(t.SPEC_PATH.read_text())
    parent = json.loads((t.b.LANE / "spec.json").read_text())
    assert spec["inherited_elo"] == parent["elo"]
    assert spec["rolling_fit_calendar_days"] == parent["rolling_fit_calendar_days"]
    assert spec["inherited_refit"] == parent["parameter_refit"]
    assert spec["frozen_selected_model"] == "dc365_elo50"
    assert t.MODELS == ["dc_equal", "dc365_elo50"]


def test_transfer_loader_rejects_wrong_league_and_inconsistent_score(tmp_path):
    p = tmp_path / "transfer_example.csv"
    p.write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nSP1,15/08/2024,A,B,1,0,H\n")
    with pytest.raises(ValueError, match="league"):
        t.parse_season(p, "I1", 2024)
    p.write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nI1,15/08/2024,A,B,1,0,A\n")
    with pytest.raises(ValueError, match="score"):
        t.parse_season(p, "I1", 2024)


@pytest.mark.parametrize("league", ["SP1", "I1"])
def test_transfer_inputs_have_complete_balanced_schedules_and_local_availability(league):
    spec = json.loads(t.SPEC_PATH.read_text())
    with t.league_context(spec["leagues"][league]["timezone"]):
        rows, populations = t.load_league(league, spec)
        assert len(rows) == 3420 and len(populations) == 9
        assert all(r.match.league == league for r in rows)
        assert all(r.match.result_available_at == t.b.utc_midnight(r.day+timedelta(days=1)) for r in rows)
        assert all(r.closing is None for r in rows)
