"""Synthetic chronology, probability provenance and population regressions."""
import copy
from datetime import date, timedelta
import json
import math

import numpy as np
import pytest

from research.experiments.boundary_20260908.football_past_market import data as d


class Poison(dict):
    def get(self, *args, **kwargs):
        raise AssertionError("Unadmitted payload was inspected")


class OutcomePoison(dict):
    def get(self, key, *args):
        if key in ("FTHG", "FTAG", "FTR"):
            raise AssertionError("Outcome was inspected")
        return super().get(key, *args)


def raw(day="2022-01-01", home="A", away="B", league="E0", season=2021,
        odds=(2., 3., 4.), goals=(1, 0), payload=None):
    columns = ("BbAvH", "BbAvD", "BbAvA") if season < 2019 else ("AvgH", "AvgD", "AvgA")
    values = {key: str(value) for key, value in zip(columns, odds)}
    values.update(FTHG=str(goals[0]), FTAG=str(goals[1]), FTR="H" if goals[0] > goals[1] else "D" if goals[0] == goals[1] else "A")
    return d.RawFixture(f"{league}:{season}:{home}:{away}", league, season, home, away,
                        date.fromisoformat(day), values if payload is None else payload,
                        "synthetic.csv", "a"*64, 2, "b"*64)


def fixture(row):
    return d._metadata(row)


@pytest.mark.parametrize("league", d.LEAGUES)
@pytest.mark.parametrize("day,extra,hours", [("2022-03-26", 1, 23), ("2022-10-29", 1, 25),
                                           ("2022-03-20", 7, 167), ("2022-10-23", 7, 169)])
def test_calendar_availability_crosses_dst(league, day, extra, hours):
    day = date.fromisoformat(day)
    a0 = d.midnight(day + timedelta(days=1), league)
    assert (d.availability(day, league, extra)-a0).total_seconds() == hours*3600


@pytest.mark.parametrize("delay", [0, 2, 8, True, 1., "1"])
def test_only_two_predeclared_delays(delay):
    with pytest.raises(ValueError):
        d.availability(date(2022, 1, 1), "E0", delay)


def test_actual_utc_comparison_not_lexical_offsets():
    archive = d.Archive((raw(),))
    assert not d.quote_rows(archive, "2022-01-03T01:00:00+01:00", 1)[0]
    assert len(d.quote_rows(archive, "2022-01-03T01:00:00.000001+01:00", 1)[0]) == 1
    assert d.utc("2022-01-03T00:00:00") == d.utc("2022-01-03T01:00:00+01:00")


def test_future_current_equal_and_expired_payloads_never_accessed():
    rows = (raw("2022-01-04", home="Future", payload=Poison()),
            raw("2022-01-03", home="Current", payload=Poison()),
            raw("2022-01-01", home="Equal", payload=Poison()),
            raw("2017-01-01", home="Expired", season=2016, payload=Poison()),
            raw("2021-12-31"))
    admitted, _ = d.training_rows(d.Archive(rows), "2022-01-03T00:00:00Z", 1)
    assert [r["match_id"] for r in admitted] == [rows[-1].match_id]


def test_metadata_does_not_access_payload_and_cache_never_bypasses_clock():
    row = raw(payload=Poison())
    archive = d.Archive((row,))
    assert d.metadata_rows(archive)[0]["availability_by_delay"]["7"] == "2022-01-09T00:00:00+00:00"
    archive._quotes[row.match_id] = {"invalid": "previously seen cache"}
    assert d.quote_rows(archive, "2022-01-03", 1)[0] == []


def test_window_includes_lower_calendar_boundary_not_older():
    cutoff = date(2022, 1, 3)
    first = raw((cutoff-timedelta(days=1095)).isoformat())
    older = raw((cutoff-timedelta(days=1096)).isoformat(), home="Old", payload=Poison())
    rows, _ = d.quote_rows(d.Archive((first, older)), cutoff.isoformat(), 1)
    assert [r["match_id"] for r in rows] == [first.match_id]


def test_quote_only_preparation_and_missing_quote_skip_outcome():
    valid = raw()
    valid = raw(payload=OutcomePoison(valid.payload))
    missing = raw(home="Missing", payload=OutcomePoison(AvgH="", AvgD="3", AvgA="4"))
    archive = d.Archive((valid, missing))
    rows, diagnostics = d.quote_rows(archive, "2022-01-04", 1)
    assert len(rows) == 1 and "outcome" not in rows[0]
    assert diagnostics["excluded_missing_quotes"][0]["match_id"] == missing.match_id
    assert d.training_rows(d.Archive((missing,)), "2022-01-04", 1)[0] == []
    with pytest.raises(AssertionError, match="Outcome"):
        d.training_rows(archive, "2022-01-04", 1)


@pytest.mark.parametrize("bad", [True, "nan", "inf", "0", "1", None, "bad"])
def test_invalid_quotes_are_shared_exclusions(bad):
    payload = OutcomePoison(AvgH=bad, AvgD="3", AvgA="4")
    assert d.training_rows(d.Archive((raw(payload=payload),)), "2022-01-04", 1)[0] == []


@pytest.mark.parametrize("goals,label", [(('-1', '0'), 'A'), (('1.0', '0'), 'H'), (('1', '0'), 'A')])
def test_admitted_bad_outcome_fails(goals, label):
    row = raw()
    row.payload.update(FTHG=goals[0], FTAG=goals[1], FTR=label)
    with pytest.raises(ValueError):
        d.training_rows(d.Archive((row,)), "2022-01-04", 1)


def test_main_q_and_elo_q_are_distinct_with_shared_weights_and_population():
    archive = d.Archive(tuple(raw(league=league) for league in d.LEAGUES))
    quotes, qdiag = d.quote_rows(archive, "2022-01-04", 1)
    trained, tdiag = d.training_rows(archive, "2022-01-04", 1)
    assert qdiag["accepted_ids_sha256"] == tdiag["accepted_ids_sha256"]
    for before, after in zip(quotes, trained):
        assert before == {key: value for key, value in after.items() if key != "outcome"}
        assert after["outcome"] == 0
        assert not np.allclose(after["q"], after["q_normalized"])
        assert sum(after["q"]) == pytest.approx(1.)
        age = (d.utc(after["cutoff_utc"])-d.utc(after["a0_utc"])).total_seconds()/86400
        assert after["weight"] == 2**(-age/365)
    quotes[0]["q"][0] = -9
    assert d.quote_rows(archive, "2022-01-04", 1)[0][0]["q"][0] > 0


def test_support_preserves_all_rows_and_league_isolation():
    trained, _ = d.quote_rows(d.Archive((raw(),)), "2022-01-04", 1)
    fixtures = [fixture(raw()), fixture(raw(home="Promoted")), fixture(raw(league="I1"))]
    assert d.support_for(fixtures, trained) == [True, False, False]
    assert len(fixtures) == 3


def test_elo_one_step_and_atomic_shared_team_batch():
    a, b = raw(away="B"), raw(away="C", odds=(4., 3., 2.))
    target = fixture(raw(home="A", away="C"))
    cursor = d.EloCursor(d.Archive((a, b)), 1)
    assert cursor.query([target], "2022-01-03")["x"] == [0.]
    result = cursor.query([target], "2022-01-04")
    expected = 1/(1+10**(-80/400))
    def delta(row):
        o = [float(row.payload[k]) for k in ("AvgH", "AvgD", "AvgA")]
        inv = [1/x for x in o]; q = [x/sum(inv) for x in inv]
        return 175*(q[0]+.5*q[1]-expected)
    da, db = delta(a), delta(b)
    assert result["x"][0] == pytest.approx((da+2*db)/400)
    assert cursor.ratings['E0', 'A'] == pytest.approx(1000+da+db)
    reverse = d.EloCursor(d.Archive((b, a)), 1)
    assert reverse.query([target], "2022-01-04") == result
    assert math.fsum(cursor.ratings.values()) == pytest.approx(3000.)


def test_elo_no_outcomes_future_poison_and_no_query_roster_entry():
    past = raw()
    past = raw(payload=OutcomePoison(past.payload))
    future = raw("2022-01-05", home="Future", payload=Poison())
    cursor = d.EloCursor(d.Archive((past, future)), 1)
    target = fixture(raw(home="Unknown"))
    observed = cursor.query([target], "2022-01-04")
    baseline = d.EloCursor(d.Archive((past,)), 1).query([target], "2022-01-04")
    assert observed == baseline
    assert ('E0', 'Unknown') not in cursor.ratings
    assert cursor.query([target], "2022-01-04") == observed
    with pytest.raises(ValueError, match="backward"):
        cursor.query([target], "2022-01-03")


def test_calibration_is_prequential_and_final_cutoff_admits_before_payload():
    warmup = raw("2019-07-01", season=2018)
    current = raw("2019-08-01", season=2019)
    future = raw("2022-08-01", season=2021, home="Future", payload=Poison())
    x, y, diagnostics = d.calibration_rows(d.Archive((warmup, current, future)), 1, "2022-08-02")
    assert y == [0]
    cursor = d.EloCursor(d.Archive((warmup,)), 1)
    assert x == cursor.query([fixture(current)], d.midnight(current.day, current.league))["x"]
    assert diagnostics["rows"][0]["match_id"] == current.match_id
    assert diagnostics["rows"][0]["rating_provenance"]["processed_raw_rows"] == 1
    altered = raw("2019-08-01", season=2019, odds=(30., 2., 1.1), goals=(0, 9))
    x2, y2, _ = d.calibration_rows(d.Archive((warmup, altered)), 1, "2022-08-02")
    assert x2 == x and y2 == [2]  # Own quote/label cannot affect its predictor.


def test_calibration_missing_quote_never_interprets_its_outcome():
    row = raw("2020-01-01", payload=OutcomePoison(AvgH="", AvgD="3", AvgA="4"))
    x, y, diagnostics = d.calibration_rows(d.Archive((row,)), 1, "2022-08-01")
    assert x == y == [] and diagnostics['excluded_missing_quote_ids'] == [row.match_id]


def reference_fixture():
    r = raw("2022-08-05", season=2022)
    meta = fixture(r)
    row = {**meta, "fit_cutoff_utc": "2022-07-31T23:00:00", "fit_id": "old_fit",
           "result_available_at": meta["a0_utc"],
           "probabilities": {name: [.4134567891234567, .25, .3365432108765433] for name in d.INTERNAL}}
    row.update(label="DO NOT READ", goals="DO NOT READ")
    xg = {"match_id": r.match_id, "forecast_cutoff_utc": row["forecast_cutoff_utc"],
          "fit_cutoff_utc": row["fit_cutoff_utc"], "probabilities": {"xg_add90": [.4, .3, .3]}}
    contract = {"rows": 1, "ordered_original_match_ids_sha256": d.digest([r.match_id]),
                "season_start_years": [2022], "league": "E0", "unique_forecast_clocks": 1,
                "rows_each_league_season": 1}
    return r, row, xg, contract


def test_reference_extraction_preserves_every_float_bit_without_labels():
    _, row, xg, contract = reference_fixture()
    result = d._extract_references([row], [xg], "selection", contract, {"sha256": "a"}, {"sha256": "b"})
    assert len(result) == 1 and 'label' not in result[0] and 'goals' not in result[0]
    for name in d.INTERNAL:
        assert np.asarray(result[0]['probabilities'][name]).tobytes() == np.asarray(row['probabilities'][name]).tobytes()
    assert result[0]['probabilities']['xg_add90_selection_safeguard'] == xg['probabilities']['xg_add90']
    assert result[0]['probabilities'][d.INTERNAL[0]] is not row['probabilities'][d.INTERNAL[0]]


@pytest.mark.parametrize("field", ['match_id', 'forecast_cutoff_utc', 'fit_cutoff_utc'])
def test_xg_exact_identity_and_both_clock_guards(field):
    _, row, xg, contract = reference_fixture()
    xg[field] += "changed"
    with pytest.raises(ValueError, match="xG"):
        d._extract_references([row], [xg], "selection", contract, {}, {})


@pytest.mark.parametrize("change", ['duplicate', 'clock', 'simplex', 'season'])
def test_reference_population_and_vector_validation(change):
    _, row, xg, contract = reference_fixture()
    if change == 'duplicate':
        contract['rows'] = 2
        rows = [row, row]
    else:
        rows = [row]
        if change == 'clock': row['forecast_cutoff_utc'] = '2022-08-05T00:00:00'
        if change == 'simplex': row['probabilities'][d.INTERNAL[0]] = [.3, .3, .3]
        if change == 'season': row['season'] = 2024
    with pytest.raises(ValueError):
        d._extract_references(rows, [xg], 'selection', contract, {}, {})


def close(tmp_path, rows):
    path = tmp_path/'forecasts.json'
    path.write_bytes(d.canonical(rows))
    closure = tmp_path/'closure.json'
    closure.write_text(json.dumps({'forecasts': {'path': str(path), 'sha256': d.sha(path)},
                                   'rows': len(rows), 'labels_attached': False}))
    return closure


def test_label_attachment_requires_exact_complete_forecast_closure(tmp_path):
    rawrow, row, xg, contract = reference_fixture()
    issued = d._extract_references([row], [xg], 'selection', contract, {}, {})
    archive = d.Archive((rawrow,))
    closure = close(tmp_path, issued)
    result = d.attach_labels(issued, archive, forecast_closure_path=closure)
    assert result[0]['label'] == 0 and result[0]['goals'] == [1, 0]
    assert 'label' not in issued[0]
    assert result[0]['probabilities'] == issued[0]['probabilities']
    altered = copy.deepcopy(issued)
    altered[0]['probabilities'][d.INTERNAL[0]] = [.4, .3, .3]
    poison = d.Archive((raw(rawrow.day.isoformat(), season=2022, payload=Poison()),))
    with pytest.raises(ValueError, match="closed forecast"):
        d.attach_labels(altered, poison, forecast_closure_path=closure)
    (tmp_path/'forecasts.json').write_text('[]')
    with pytest.raises(ValueError, match="binding"):
        d.attach_labels(issued, poison, forecast_closure_path=closure)


def test_source_hash_mismatch_before_any_csv_read(tmp_path):
    path = tmp_path/'data.csv'; path.write_text('not a CSV')
    with pytest.raises(ValueError, match='binding'):
        d._bound({'path': str(path), 'sha256': '0'*64})


def test_canonical_source_day_join_precedes_forecast_inputs():
    row = raw(payload=Poison())
    archive = d.Archive((row,))
    target = fixture(row)
    d.validate_reference_metadata(archive, [target])
    target['day'] = '2022-01-02'
    with pytest.raises(ValueError, match='canonical metadata'):
        d.validate_reference_metadata(archive, [target])


def test_elo_digest_reused_when_no_updates_change(monkeypatch):
    cursor = d.EloCursor(d.Archive((raw(),)), 1)
    target = fixture(raw())
    first = cursor.query([target], '2022-01-04')
    monkeypatch.setattr(d, 'digest', lambda value: (_ for _ in ()).throw(AssertionError('rehash')))
    assert cursor.query([target], '2022-01-04') == first


def test_duplicate_raw_identity_rejected_before_payload():
    row = raw(payload=Poison())
    with pytest.raises(ValueError, match='Duplicate'):
        d.Archive((row, row))
