"""Football venue schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VenueRecord:
    venue_id: str
    name: str
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None
