"""Football external and local data providers."""

from ...mrp.providers import *  # noqa: F403
from .local import LocalFootballData, load_local_football_data

__all__ = ["LocalFootballData", "load_local_football_data"]
