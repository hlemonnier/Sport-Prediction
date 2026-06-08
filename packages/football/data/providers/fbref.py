"""FBref provider contract."""

from __future__ import annotations


class FBrefProvider:
    def fetch_team_history(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("FBref ingestion is a future football provider surface.")
