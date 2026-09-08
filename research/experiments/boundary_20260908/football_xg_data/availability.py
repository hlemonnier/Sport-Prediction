"""Explicit retrospective availability proxy; not a claim of original publication."""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ZONES = {"E0": "Europe/London", "SP1": "Europe/Madrid", "I1": "Europe/Rome"}


def proxy_available_at(row, delay_days=7):
    """Return next local midnight after validated completion plus7/14*24h in UTC.

    Excluded joins remain unavailable. Never interpret arbitrary CSV strings as
    true, infer a publication time, or move a resumed game to original kickoff.
    """
    if delay_days not in (7, 14) or isinstance(delay_days, bool):
        raise ValueError("only the declared 7/14-day sensitivity delays are supported")
    accepted = row["accepted_exact_join"]
    if not (accepted is True or (isinstance(accepted, str) and accepted == "True")):
        return None
    if row["canonical_date"] != row["provider_date"]:
        raise ValueError("accepted join must retain identical validated dates")
    for side in ("home", "away"):
        if float(row[f"{side}_goals"]) != float(row[f"canonical_{side}_goals"]):
            raise ValueError("accepted join must retain equal final goals")
    day = datetime.strptime(row["canonical_date"], "%Y-%m-%d").date()
    midnight = datetime.combine(day+timedelta(days=1), time.min, tzinfo=ZoneInfo(ZONES[row["league"]]))
    return midnight.astimezone(timezone.utc)+timedelta(days=delay_days)
