"""Live-race state feature helpers."""

from __future__ import annotations

from typing import Mapping

from packages.f1.models.live_race.state import parse_track_status


def live_track_status_features(track_status: object) -> Mapping[str, object]:
    """Convert raw track-status codes into model feature flags."""

    flags = parse_track_status(track_status)
    return {
        "track_status_codes": "".join(sorted(flags.codes)),
        "track_status_is_red": flags.is_red,
        "track_status_is_sc_vsc": flags.is_sc_vsc,
        "track_status_is_yellow": flags.is_yellow,
        "track_status_is_greenish": flags.is_greenish,
    }
