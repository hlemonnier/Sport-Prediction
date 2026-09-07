"""Explicit eligibility for confirmatory event-block evaluation.

Eight events is a minimum design floor, not a power calculation. The empirical
bootstrap cannot establish transportability or undo development exposure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import re
from typing import Mapping, Sequence

MIN_AUDIT_EVENTS = 8
MIN_STRATUM_EVENTS = 2
# Inspected during model development and this audit; cannot become pristine by declaration.
KNOWN_DEVELOPMENT_EXPOSED_EVENTS = frozenset(202600 + r for r in range(1, 10))


def event_ordinal(value: str) -> int | None:
    match = (re.fullmatch(r"(\d{4})[:_\-](?:R)?(\d{1,2})", str(value), re.IGNORECASE)
             or re.fullmatch(r"(\d{4})(\d{2})", str(value)))
    if match is None:
        return None
    year, round_number = map(int, match.groups())
    return year * 100 + round_number if year >= 1950 and 1 <= round_number <= 99 else None


def validate_event_partitions(*, development: Sequence[str], selection: Sequence[str],
                              calibration: Sequence[str], audit: Sequence[str]) -> tuple[str, ...]:
    named = {"development": development, "selection": selection, "calibration": calibration, "audit": audit}
    issues: list[str] = []
    normalized: dict[str, list[int]] = {}
    for name, values in named.items():
        if not values:
            issues.append(f"{name}_events_missing")
        ordinals = [event_ordinal(value) for value in values]
        if any(value is None for value in ordinals):
            issues.append(f"{name}_event_key_invalid")
        valid = [value for value in ordinals if value is not None]
        normalized[name] = valid
        if len(valid) != len(set(valid)):
            issues.append(f"{name}_events_duplicated")
        if valid != sorted(valid):
            issues.append(f"{name}_events_not_chronological")
    names = list(named)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if set(normalized[left]).intersection(normalized[right]):
                issues.append(f"event_partition_overlap:{left}:{right}")
            if normalized[left] and normalized[right] and max(normalized[left]) >= min(normalized[right]):
                issues.append(f"event_partition_order_invalid:{right}")
    return tuple(dict.fromkeys(issues))


def utc_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class EvaluationProtocol:
    development: tuple[str, ...] = ()
    selection: tuple[str, ...] = ()
    calibration: tuple[str, ...] = ()
    audit: tuple[str, ...] = ()
    candidate_sha256: str = ""
    evaluated_candidate_sha256: str = ""
    frozen_at_utc: str = ""
    audit_prediction_times_utc: Mapping[str, str] = field(default_factory=dict)
    development_exposed_events: tuple[str, ...] = ()
    development_exposure_reviewed: bool = False
    point_in_time_provenance_verified: bool = False
    minimum_audit_events: int = MIN_AUDIT_EVENTS

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EvaluationProtocol":
        if not isinstance(payload, Mapping):
            raise ValueError("evaluation protocol must be a mapping")
        fields = dict(payload)
        for key in ("development", "selection", "calibration", "audit", "development_exposed_events"):
            values = fields.get(key, ())
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{key} must be an event list")
            fields[key] = tuple(values)
        try:
            return cls(**fields)
        except TypeError as exc:
            raise ValueError("invalid evaluation protocol fields") from exc

    def issues(self, event_keys: Sequence[str], *, strata: Sequence[str] | None = None) -> tuple[str, ...]:
        try:
            return self._issues(event_keys, strata=strata)
        except (TypeError, ValueError, AttributeError, OverflowError):
            return ("evaluation_protocol_payload_invalid",)

    def _issues(self, event_keys: Sequence[str], *, strata: Sequence[str] | None = None) -> tuple[str, ...]:
        issues = list(validate_event_partitions(development=self.development, selection=self.selection,
                                               calibration=self.calibration, audit=self.audit))
        keys = [event_ordinal(key) for key in event_keys]
        audit = [event_ordinal(key) for key in self.audit]
        if keys != audit or any(key is None for key in keys):
            issues.append("audit_population_does_not_match_protocol")
        if not isinstance(self.minimum_audit_events, int) or isinstance(self.minimum_audit_events, bool) or self.minimum_audit_events < MIN_AUDIT_EVENTS:
            issues.append("minimum_audit_events_below_design_floor")
        elif len(set(keys)) < self.minimum_audit_events:
            issues.append("insufficient_independent_audit_events")
        if strata is not None:
            if len(strata) != len(keys) or any(strata.count(label) < MIN_STRATUM_EVENTS for label in ("standard", "sprint")):
                issues.append("insufficient_events_per_weekend_stratum")
        if self.development_exposure_reviewed is not True:
            issues.append("development_exposure_not_reviewed")
        exposed = [event_ordinal(key) for key in self.development_exposed_events]
        if any(key is None for key in exposed) or set(keys).intersection(exposed) or set(keys).intersection(KNOWN_DEVELOPMENT_EXPOSED_EVENTS):
            issues.append("audit_events_exposed_during_development")
        if self.point_in_time_provenance_verified is not True:
            issues.append("point_in_time_provenance_unverified")
        if re.fullmatch(r"[0-9a-f]{64}", self.candidate_sha256) is None:
            issues.append("frozen_candidate_hash_missing_or_invalid")
        elif self.candidate_sha256 != self.evaluated_candidate_sha256:
            issues.append("evaluated_candidate_changed_after_freeze")
        freeze = utc_time(self.frozen_at_utc)
        times = [utc_time(self.audit_prediction_times_utc.get(key, "")) for key in self.audit]
        if freeze is None or not times or any(time is None for time in times):
            issues.append("freeze_or_audit_timing_missing")
        elif freeze >= min(times):
            issues.append("candidate_not_frozen_before_audit")
        elif times != sorted(times) or any(time.year != event_ordinal(key) // 100 for key, time in zip(self.audit, times) if event_ordinal(key) is not None):
            issues.append("audit_prediction_timing_inconsistent_with_events")
        return tuple(dict.fromkeys(issues))
