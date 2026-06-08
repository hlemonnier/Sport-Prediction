"""StatsBomb provider contract."""

from __future__ import annotations


class StatsBombProvider:
    def fetch_match_events(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("StatsBomb event ingestion is a future provider surface.")
