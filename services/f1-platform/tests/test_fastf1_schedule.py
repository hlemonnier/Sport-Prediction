from f1_platform.fastf1_schedule import FastF1ScheduleClient


def test_fastf1_schedule_resolves_current_live_session():
    client = FastF1ScheduleClient(
        schedule_provider=lambda year: [
            {
                "RoundNumber": 9,
                "Country": "Austria",
                "Location": "Spielberg",
                "EventName": "Austrian Grand Prix",
                "OfficialEventName": "FORMULA 1 AUSTRIAN GRAND PRIX 2026",
                "EventFormat": "conventional",
                "Session1": "Practice 1",
                "Session1DateUtc": "2026-06-26T11:30:00Z",
                "Session2": "Practice 2",
                "Session2DateUtc": "2026-06-26T15:00:00Z",
            }
        ]
    )

    result = client.resolve_live_or_next_session(now="2026-06-26T12:00:00Z", year=2026)

    assert result["status"] == "live"
    assert result["source"] == "fastf1-schedule"
    assert result["session"]["session_key"] == "fastf1:2026:austrian-grand-prix:fp1"
    assert result["session"]["fastf1_session_name"] == "FP1"
    assert result["nextSession"]["session_key"] == "fastf1:2026:austrian-grand-prix:fp2"
    assert result["secondsUntilStart"] == 0
    assert result["secondsUntilEnd"] == 1800


def test_fastf1_schedule_resolves_next_session_when_none_live():
    client = FastF1ScheduleClient(
        schedule_provider=lambda year: [
            {
                "RoundNumber": 9,
                "Country": "Austria",
                "Location": "Spielberg",
                "EventName": "Austrian Grand Prix",
                "Session1": "Practice 1",
                "Session1DateUtc": "2026-06-26T11:30:00Z",
                "Session2": "Qualifying",
                "Session2DateUtc": "2026-06-27T14:00:00Z",
            }
        ]
    )

    result = client.resolve_live_or_next_session(now="2026-06-26T13:00:00Z", year=2026)

    assert result["status"] == "upcoming"
    assert result["session"] is None
    assert result["nextSession"]["session_key"] == "fastf1:2026:austrian-grand-prix:q"
    assert result["nextSession"]["session_type"] == "Qualifying"
    assert result["secondsUntilStart"] == 90000


def test_fastf1_schedule_checks_next_year_when_current_year_is_finished():
    def schedule_provider(year: int):
        if year == 2026:
            return [
                {
                    "RoundNumber": 22,
                    "Country": "United Arab Emirates",
                    "Location": "Yas Marina",
                    "EventName": "Abu Dhabi Grand Prix",
                    "Session5": "Race",
                    "Session5DateUtc": "2026-12-06T13:00:00Z",
                }
            ]
        if year == 2027:
            return [
                {
                    "RoundNumber": 1,
                    "Country": "Australia",
                    "Location": "Melbourne",
                    "EventName": "Australian Grand Prix",
                    "Session1": "Practice 1",
                    "Session1DateUtc": "2027-03-05T01:30:00Z",
                }
            ]
        return []

    client = FastF1ScheduleClient(schedule_provider=schedule_provider)

    result = client.resolve_live_or_next_session(now="2026-12-07T00:00:00Z")

    assert result["status"] == "upcoming"
    assert result["nextSession"]["year"] == 2027
    assert result["nextSession"]["session_key"] == "fastf1:2027:australian-grand-prix:fp1"


def test_fastf1_schedule_returns_round_calendar_shape():
    client = FastF1ScheduleClient(
        schedule_provider=lambda year: [
            {
                "RoundNumber": 8,
                "Country": "Austria",
                "Location": "Spielberg",
                "EventName": "Austrian Grand Prix",
                "OfficialEventName": "FORMULA 1 AUSTRIAN GRAND PRIX 2026",
                "EventDate": "2026-06-28",
                "EventFormat": "conventional",
                "F1ApiSupport": True,
                "Session1": "Practice 1",
                "Session1DateUtc": "2026-06-26T11:30:00Z",
                "Session2": "Practice 2",
                "Session2DateUtc": "2026-06-26T15:00:00Z",
            }
        ]
    )

    result = client.season_schedule(year=2026)

    assert result["year"] == 2026
    assert result["source"] == "fastf1-schedule"
    assert result["roundCount"] == 1
    assert result["sessionCount"] == 2
    assert result["rounds"][0]["eventName"] == "Austrian Grand Prix"
    assert result["rounds"][0]["sessions"][1]["session_key"] == "fastf1:2026:austrian-grand-prix:fp2"


def test_fastf1_schedule_disambiguates_repeated_testing_events():
    client = FastF1ScheduleClient(
        schedule_provider=lambda year: [
            {
                "RoundNumber": 0,
                "Country": "Bahrain",
                "Location": "Bahrain",
                "EventName": "Pre-Season Testing",
                "OfficialEventName": "FORMULA 1 PRE-SEASON TESTING 1 2026",
                "EventDate": "2026-02-13",
                "EventFormat": "testing",
                "Session1": "Practice 1",
                "Session1DateUtc": "2026-02-11T07:00:00Z",
            },
            {
                "RoundNumber": 0,
                "Country": "Bahrain",
                "Location": "Bahrain",
                "EventName": "Pre-Season Testing",
                "OfficialEventName": "FORMULA 1 PRE-SEASON TESTING 2 2026",
                "EventDate": "2026-02-20",
                "EventFormat": "testing",
                "Session1": "Practice 1",
                "Session1DateUtc": "2026-02-18T07:00:00Z",
            },
        ]
    )

    result = client.season_schedule(year=2026)
    round_keys = [round_["scheduleKey"] for round_ in result["rounds"]]
    session_keys = [round_["sessions"][0]["session_key"] for round_ in result["rounds"]]

    assert len(round_keys) == len(set(round_keys))
    assert len(session_keys) == len(set(session_keys))
    assert session_keys == [
        "fastf1:2026:pre-season-testing-2026-02-13:fp1",
        "fastf1:2026:pre-season-testing-2026-02-20:fp1",
    ]
