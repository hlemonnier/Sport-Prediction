"""Small deterministic F1 replay fixture used by local API and UI development."""

from __future__ import annotations

from .replay import raw_event
from .schemas import F1Event

SAMPLE_SESSION_KEY = "sample-race"


def sample_events(session_key: int | str = SAMPLE_SESSION_KEY) -> list[F1Event]:
    base_time = "2026-06-28T13:32:18.450Z"
    drivers = [
        (1, "VER", "Max Verstappen", "Red Bull Racing", "3671C6"),
        (63, "RUS", "George Russell", "Mercedes", "27F4D2"),
        (16, "LEC", "Charles Leclerc", "Ferrari", "E80020"),
        (4, "NOR", "Lando Norris", "McLaren", "FF8000"),
        (81, "PIA", "Oscar Piastri", "McLaren", "FF8000"),
        (44, "HAM", "Lewis Hamilton", "Ferrari", "E80020"),
    ]
    events: list[F1Event] = []
    source_id = 1
    events.append(
        raw_event(
            source_id,
            "v1/sessions",
            f"session:{session_key}",
            session_key,
            {
                "meeting_key": 2026001,
                "session_key": session_key,
                "session_name": "Race",
                "session_type": "Race",
                "date_start": base_time,
                "gmt_offset": "+02:00",
                "location": "Spielberg",
                "year": 2026,
                "is_cancelled": False,
            },
            event_time=base_time,
        )
    )
    source_id += 1
    for number, acronym, full_name, team, colour in drivers:
        events.append(
            raw_event(
                source_id,
                "v1/drivers",
                f"driver:{number}",
                session_key,
                {
                    "name_acronym": acronym,
                    "full_name": full_name,
                    "team_name": team,
                    "team_colour": colour,
                },
                driver_number=number,
                event_time=base_time,
            )
        )
        source_id += 1

    positions = [(1, 1), (63, 2), (16, 3), (4, 4), (81, 5), (44, 6)]
    for number, position in positions:
        events.append(
            raw_event(
                source_id,
                "v1/position",
                f"{number}:position",
                session_key,
                {"position": position},
                driver_number=number,
                event_time=base_time,
            )
        )
        source_id += 1

    gaps = {
        1: ("Leader", "0.000"),
        63: ("+1.842", "+1.842"),
        16: ("+0.911", "+2.753"),
        4: ("+2.104", "+4.857"),
        81: ("+0.742", "+5.599"),
        44: ("+3.118", "+8.717"),
    }
    for number, (interval, gap) in gaps.items():
        events.append(
            raw_event(
                source_id,
                "v1/intervals",
                f"{number}:interval",
                session_key,
                {"interval": interval, "gap_to_leader": gap},
                driver_number=number,
                event_time=base_time,
            )
        )
        source_id += 1

    compounds = {1: "MEDIUM", 63: "MEDIUM", 16: "HARD", 4: "HARD", 81: "MEDIUM", 44: "HARD"}
    for number, compound in compounds.items():
        events.append(
            raw_event(
                source_id,
                "v1/stints",
                f"{number}:stint:2",
                session_key,
                {
                    "stint_number": 2,
                    "compound": compound,
                    "lap_start": 14,
                    "lap_end": None,
                    "tyre_age_at_start": 2,
                },
                driver_number=number,
                event_time=base_time,
            )
        )
        source_id += 1

    lap_times = {
        1: [68.451, 68.398, 68.322],
        63: [68.731, 68.590, 68.431],
        16: [68.892, 68.761, 68.640],
        4: [69.044, 68.802, 68.701],
        81: [69.020, 68.912, 68.755],
        44: [69.351, 69.124, 68.991],
    }
    for lap_offset, lap_number in enumerate((21, 22, 23)):
        for number, values in lap_times.items():
            events.append(
                raw_event(
                    source_id,
                    "v1/laps",
                    f"{number}:lap:{lap_number}",
                    session_key,
                    {
                        "lap_number": lap_number,
                        "lap_duration": values[lap_offset],
                        "duration_sector_1": round(values[lap_offset] * 0.263, 3),
                        "duration_sector_2": round(values[lap_offset] * 0.431, 3),
                        "duration_sector_3": round(values[lap_offset] * 0.306, 3),
                    },
                    driver_number=number,
                    event_time=base_time,
                )
            )
            source_id += 1

    speeds = {1: 309, 63: 306, 16: 304, 4: 307, 81: 305, 44: 302}
    for idx, (number, speed) in enumerate(speeds.items()):
        events.append(
            raw_event(
                source_id,
                "v1/car_data",
                f"{number}:car",
                session_key,
                {"speed": speed, "drs": 12 if idx in {1, 2, 3} else 8},
                driver_number=number,
                event_time=base_time,
            )
        )
        source_id += 1
        for step in range(3):
            progress = 0.10 + idx * 0.035 + step * 0.045
            events.append(
                raw_event(
                    source_id,
                    "v1/location",
                    f"{number}:loc:{step}",
                    session_key,
                    {"x": round(progress * 10_000, 3), "y": 0, "z": 0},
                    driver_number=number,
                    event_time=f"2026-06-28T13:32:{18 + step:02d}.450Z",
                )
            )
            source_id += 1

    weather_samples = [
        ("2026-06-28T13:32:18.450Z", 25.1, 40.6, 0, 2.4),
        ("2026-06-28T13:32:20.450Z", 25.3, 40.9, 0, 2.6),
        ("2026-06-28T13:32:22.450Z", 25.4, 41.2, 0, 2.7),
    ]
    for event_time, air_temperature, track_temperature, rainfall, wind_speed in weather_samples:
        events.append(
            raw_event(
                source_id,
                "v1/weather",
                f"weather:{event_time}",
                session_key,
                {
                    "date": event_time,
                    "air_temperature": air_temperature,
                    "track_temperature": track_temperature,
                    "rainfall": rainfall,
                    "wind_speed": wind_speed,
                },
                event_time=event_time,
            )
        )
        source_id += 1
    events.append(
        raw_event(
            source_id,
            "v1/race_control",
            "race-control:latest",
            session_key,
            {
                "category": "Flag",
                "flag": "GREEN",
                "message": "Track clear",
                "scope": "Track",
            },
            event_time=base_time,
        )
    )
    source_id += 1

    result_rows = [
        (1, 1, 23, 1578.214, 0.0),
        (63, 2, 23, 1580.056, 1.842),
        (16, 3, 23, 1580.967, 2.753),
        (4, 4, 23, 1583.071, 4.857),
        (81, 5, 23, 1583.813, 5.599),
        (44, 6, 23, 1586.931, 8.717),
    ]
    for number, position, laps, duration, gap in result_rows:
        events.append(
            raw_event(
                source_id,
                "v1/session_result",
                f"{number}:session_result",
                session_key,
                {
                    "position": position,
                    "number_of_laps": laps,
                    "duration": duration,
                    "gap_to_leader": gap,
                    "dnf": False,
                    "dns": False,
                    "dsq": False,
                },
                driver_number=number,
                event_time=base_time,
            )
        )
        source_id += 1
    return events
