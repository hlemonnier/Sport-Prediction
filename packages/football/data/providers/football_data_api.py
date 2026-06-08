"""football-data.org provider contract."""

from __future__ import annotations


class FootballDataApiProvider:
    """Future football-data.org adapter.

    The local CSV/parquet provider remains canonical until API credentials and
    rate-limit policy are configured.
    """

    def fetch_fixtures(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("football-data.org integration is not configured yet.")
