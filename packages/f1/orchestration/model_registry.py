"""Production registry and reversible promotion workflow for F1 models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

from .model_promotion import (
    PromotionDecision,
    PromotionGateConfig,
    evaluate_model_promotion,
    live_strategy_promotion_config,
    ultimate_lap_time_promotion_config,
)


CANDIDATE_STATUS = "candidate"
FALLBACK_STATUS = "fallback"
PRODUCTION_STATUS = "production"
REJECTED_STATUS = "rejected"
ARCHIVED_STATUS = "archived"

ALLOWED_PROMOTION_STATUSES: tuple[str, ...] = (
    CANDIDATE_STATUS,
    FALLBACK_STATUS,
    PRODUCTION_STATUS,
    REJECTED_STATUS,
    ARCHIVED_STATUS,
)

REQUIRED_REGISTRY_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_family",
    "version",
    "training_data_cutoff",
    "feature_schema_version",
    "artifact_path",
    "metrics",
    "promotion_status",
    "fallback_model_id",
)


@dataclass(frozen=True)
class F1ModelRegistryEntry:
    """Single auditable model record in the production registry."""

    model_id: str
    model_family: str
    version: str
    training_data_cutoff: str
    feature_schema_version: str
    artifact_path: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    promotion_status: str = CANDIDATE_STATUS
    fallback_model_id: Optional[str] = None
    deterministic_fallback: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _required_text(self.model_id, "model_id"))
        object.__setattr__(self, "model_family", _required_text(self.model_family, "model_family"))
        object.__setattr__(self, "version", _required_text(self.version, "version"))
        object.__setattr__(
            self,
            "training_data_cutoff",
            _required_text(self.training_data_cutoff, "training_data_cutoff"),
        )
        object.__setattr__(
            self,
            "feature_schema_version",
            _required_text(self.feature_schema_version, "feature_schema_version"),
        )
        object.__setattr__(self, "artifact_path", _artifact_path(self.artifact_path, "artifact_path"))
        object.__setattr__(self, "metrics", _metrics(self.metrics))
        object.__setattr__(self, "promotion_status", _promotion_status(self.promotion_status))
        object.__setattr__(self, "fallback_model_id", _optional_text(self.fallback_model_id))
        object.__setattr__(self, "deterministic_fallback", bool(self.deterministic_fallback))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "F1ModelRegistryEntry":
        missing = [field_name for field_name in REQUIRED_REGISTRY_FIELDS if field_name not in payload]
        if missing:
            raise ValueError(f"registry entry missing required fields: {tuple(missing)}")
        return cls(
            model_id=str(payload["model_id"]),
            model_family=str(payload["model_family"]),
            version=str(payload["version"]),
            training_data_cutoff=str(payload["training_data_cutoff"]),
            feature_schema_version=str(payload["feature_schema_version"]),
            artifact_path=str(payload["artifact_path"]),
            metrics=_payload_mapping(payload.get("metrics"), "metrics"),
            promotion_status=str(payload["promotion_status"]),
            fallback_model_id=_optional_text(payload.get("fallback_model_id")),
            deterministic_fallback=bool(payload.get("deterministic_fallback", False)),
            metadata=_payload_mapping(payload.get("metadata", {}), "metadata"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "version": self.version,
            "training_data_cutoff": self.training_data_cutoff,
            "feature_schema_version": self.feature_schema_version,
            "artifact_path": self.artifact_path,
            "metrics": dict(self.metrics),
            "promotion_status": self.promotion_status,
            "fallback_model_id": self.fallback_model_id,
            "deterministic_fallback": bool(self.deterministic_fallback),
            "metadata": dict(self.metadata),
        }

    def with_status(self, promotion_status: str, *, fallback_model_id: Optional[str] = None) -> "F1ModelRegistryEntry":
        if fallback_model_id is None:
            return replace(self, promotion_status=promotion_status)
        return replace(self, promotion_status=promotion_status, fallback_model_id=fallback_model_id)


@dataclass(frozen=True)
class PromotionEvidence:
    """Evidence required before a registry entry can replace production."""

    baseline_model_id: str
    split_strategy: str
    calibration_report_path: Optional[str]
    baseline_metrics: Optional[Mapping[str, float]] = None
    leakage_issues: Sequence[str] = ()
    simulator_validation_passed: Optional[bool] = None
    deterministic_fallback_model_id: Optional[str] = None
    baseline_comparison_report_path: Optional[str] = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_model_id", _required_text(self.baseline_model_id, "baseline_model_id"))
        object.__setattr__(self, "split_strategy", str(self.split_strategy or "").strip())
        object.__setattr__(
            self,
            "calibration_report_path",
            _optional_artifact_path(self.calibration_report_path, "calibration_report_path"),
        )
        baseline_metrics = self.baseline_metrics
        object.__setattr__(self, "baseline_metrics", None if baseline_metrics is None else _metrics(baseline_metrics))
        object.__setattr__(self, "leakage_issues", tuple(str(item) for item in (self.leakage_issues or ())))
        object.__setattr__(
            self,
            "deterministic_fallback_model_id",
            _optional_text(self.deterministic_fallback_model_id),
        )
        object.__setattr__(
            self,
            "baseline_comparison_report_path",
            _optional_artifact_path(self.baseline_comparison_report_path, "baseline_comparison_report_path"),
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    def to_payload(self) -> dict[str, object]:
        return {
            "baseline_model_id": self.baseline_model_id,
            "split_strategy": self.split_strategy,
            "calibration_report_path": self.calibration_report_path,
            "baseline_metrics": None if self.baseline_metrics is None else dict(self.baseline_metrics),
            "leakage_issues": list(self.leakage_issues),
            "simulator_validation_passed": self.simulator_validation_passed,
            "deterministic_fallback_model_id": self.deterministic_fallback_model_id,
            "baseline_comparison_report_path": self.baseline_comparison_report_path,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class RegistryPromotionEvaluation:
    """Audit envelope around the base promotion decision."""

    candidate: F1ModelRegistryEntry
    evidence: PromotionEvidence
    decision: PromotionDecision
    baseline: Optional[F1ModelRegistryEntry] = None
    fallback: Optional[F1ModelRegistryEntry] = None

    @property
    def promotion_gate_passed(self) -> bool:
        return bool(self.decision.promotion_gate_passed)

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_payload(),
            "baseline": None if self.baseline is None else self.baseline.to_payload(),
            "fallback": None if self.fallback is None else self.fallback.to_payload(),
            "evidence": self.evidence.to_payload(),
            "decision": self.decision.to_payload(),
        }


@dataclass(frozen=True)
class RegistryPromotionResult:
    """Result of attempting to replace production with a candidate."""

    registry: "ModelRegistry"
    evaluation: RegistryPromotionEvaluation

    @property
    def promoted(self) -> bool:
        return bool(self.evaluation.promotion_gate_passed)

    def to_payload(self) -> dict[str, object]:
        return {
            "promoted": bool(self.promoted),
            "evaluation": self.evaluation.to_payload(),
            "registry": self.registry.to_payload(),
        }


class ModelRegistry:
    """Immutable-on-write registry with explicit production and fallback model selection."""

    registry_version = "f1_model_registry_v1"

    def __init__(self, entries: Iterable[F1ModelRegistryEntry] = ()) -> None:
        by_id: dict[str, F1ModelRegistryEntry] = {}
        production_by_family: dict[str, str] = {}
        for entry in entries:
            if entry.model_id in by_id:
                raise ValueError(f"duplicate model_id in registry: {entry.model_id}")
            if entry.promotion_status == PRODUCTION_STATUS:
                current = production_by_family.get(entry.model_family)
                if current is not None:
                    raise ValueError(f"multiple production models for family {entry.model_family}: {current}, {entry.model_id}")
                production_by_family[entry.model_family] = entry.model_id
            by_id[entry.model_id] = entry
        self._entries = by_id

    @property
    def entries(self) -> tuple[F1ModelRegistryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def get(self, model_id: str) -> Optional[F1ModelRegistryEntry]:
        return self._entries.get(str(model_id))

    def require(self, model_id: str) -> F1ModelRegistryEntry:
        entry = self.get(model_id)
        if entry is None:
            raise KeyError(f"model_id not present in registry: {model_id}")
        return entry

    def register(self, entry: F1ModelRegistryEntry) -> "ModelRegistry":
        if entry.model_id in self._entries:
            raise ValueError(f"model_id already registered: {entry.model_id}")
        return ModelRegistry((*self.entries, entry))

    def active_model(self, model_family: str) -> Optional[F1ModelRegistryEntry]:
        family = _required_text(model_family, "model_family")
        active = [entry for entry in self._entries.values() if entry.model_family == family and entry.promotion_status == PRODUCTION_STATUS]
        if len(active) > 1:
            raise ValueError(f"multiple production models for family {family}")
        return active[0] if active else None

    def select_model(self, model_family: str, *, allow_fallback: bool = True) -> Optional[F1ModelRegistryEntry]:
        active = self.active_model(model_family)
        if active is not None or not allow_fallback:
            return active
        fallbacks = [
            entry
            for entry in self._entries.values()
            if entry.model_family == model_family and entry.deterministic_fallback
        ]
        return sorted(fallbacks, key=lambda item: item.model_id)[0] if fallbacks else None

    def fallback_model_for(self, model_id: str) -> Optional[F1ModelRegistryEntry]:
        entry = self.get(model_id)
        if entry is None or not entry.fallback_model_id:
            return None
        return self.get(entry.fallback_model_id)

    def evaluate_promotion(
        self,
        candidate_model_id: str,
        evidence: PromotionEvidence,
        *,
        config: Optional[PromotionGateConfig] = None,
    ) -> RegistryPromotionEvaluation:
        candidate = self.require(candidate_model_id)
        baseline = self.get(evidence.baseline_model_id)
        fallback_model_id = evidence.deterministic_fallback_model_id or candidate.fallback_model_id
        fallback = self.get(fallback_model_id) if fallback_model_id else None

        registry_reasons = self._registry_rule_reasons(
            candidate=candidate,
            evidence=evidence,
            baseline=baseline,
            fallback_model_id=fallback_model_id,
            fallback=fallback,
        )
        baseline_metrics = evidence.baseline_metrics
        if baseline_metrics is None and baseline is not None:
            baseline_metrics = baseline.metrics

        gate_config = config or promotion_config_for_family(candidate.model_family)
        base_decision = evaluate_model_promotion(
            candidate_model_id=candidate.model_id,
            baseline_model_id=evidence.baseline_model_id,
            candidate_metrics=candidate.metrics,
            baseline_metrics=baseline_metrics,
            config=gate_config,
            leakage_issues=evidence.leakage_issues,
            simulator_validation_passed=evidence.simulator_validation_passed,
            deterministic_fallback_model_id=fallback_model_id,
        )
        reasons = tuple(dict.fromkeys((*registry_reasons, *base_decision.reasons)))
        passed = bool(not reasons)
        decision = PromotionDecision(
            candidate_model_id=base_decision.candidate_model_id,
            baseline_model_id=base_decision.baseline_model_id,
            promotion_gate_passed=passed,
            promotion_status="promoted" if passed else REJECTED_STATUS,
            reasons=reasons,
            missing_metrics=base_decision.missing_metrics,
            metric_comparisons=base_decision.metric_comparisons,
            deterministic_fallback_model_id=fallback_model_id,
            diagnostics={
                **base_decision.diagnostics,
                "registry_version": self.registry_version,
                "split_strategy": evidence.split_strategy,
                "calibration_report_path": evidence.calibration_report_path,
                "baseline_comparison_report_path": evidence.baseline_comparison_report_path,
                "baseline_model_registered": baseline is not None,
                "deterministic_fallback_registered": fallback is not None,
                "deterministic_fallback_model_id": fallback_model_id,
                "production_replacement_requires_deterministic_fallback": True,
            },
        )
        return RegistryPromotionEvaluation(
            candidate=candidate,
            evidence=evidence,
            decision=decision,
            baseline=baseline,
            fallback=fallback,
        )

    def promote_to_production(
        self,
        candidate_model_id: str,
        evidence: PromotionEvidence,
        *,
        config: Optional[PromotionGateConfig] = None,
    ) -> RegistryPromotionResult:
        evaluation = self.evaluate_promotion(candidate_model_id, evidence, config=config)
        candidate = evaluation.candidate
        if not evaluation.promotion_gate_passed:
            return RegistryPromotionResult(
                registry=self._replace_entry(candidate.with_status(REJECTED_STATUS)),
                evaluation=evaluation,
            )

        fallback_id = evaluation.decision.deterministic_fallback_model_id
        updated: list[F1ModelRegistryEntry] = []
        for entry in self._entries.values():
            if entry.model_id == candidate.model_id:
                updated.append(entry.with_status(PRODUCTION_STATUS, fallback_model_id=fallback_id))
                continue
            if entry.model_family != candidate.model_family:
                updated.append(entry)
                continue
            if entry.model_id == fallback_id:
                updated.append(entry.with_status(FALLBACK_STATUS))
                continue
            if entry.promotion_status == PRODUCTION_STATUS:
                updated.append(entry.with_status(ARCHIVED_STATUS))
                continue
            updated.append(entry)
        return RegistryPromotionResult(registry=ModelRegistry(updated), evaluation=evaluation)

    def rollback(self, model_family: str) -> "ModelRegistry":
        active = self.active_model(model_family)
        if active is None:
            raise ValueError(f"no production model for family {model_family}")
        fallback = self.fallback_model_for(active.model_id)
        if fallback is None:
            raise ValueError(f"production model {active.model_id} has no registered fallback")
        if not fallback.deterministic_fallback:
            raise ValueError(f"fallback model {fallback.model_id} is not marked deterministic")

        updated: list[F1ModelRegistryEntry] = []
        for entry in self._entries.values():
            if entry.model_id == active.model_id:
                updated.append(entry.with_status(ARCHIVED_STATUS))
            elif entry.model_id == fallback.model_id:
                updated.append(entry.with_status(PRODUCTION_STATUS))
            else:
                updated.append(entry)
        return ModelRegistry(updated)

    def to_payload(self) -> dict[str, object]:
        return {
            "registry_version": self.registry_version,
            "entries": [entry.to_payload() for entry in self.entries],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ModelRegistry":
        entries_payload = payload.get("entries")
        if not isinstance(entries_payload, Sequence) or isinstance(entries_payload, (str, bytes)):
            raise ValueError("registry payload requires an entries sequence")
        return cls(F1ModelRegistryEntry.from_payload(_payload_mapping(item, "registry entry")) for item in entries_payload)

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def read_json(cls, path: str | Path) -> "ModelRegistry":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    def _replace_entry(self, replacement: F1ModelRegistryEntry) -> "ModelRegistry":
        entries = [replacement if entry.model_id == replacement.model_id else entry for entry in self._entries.values()]
        return ModelRegistry(entries)

    def _registry_rule_reasons(
        self,
        *,
        candidate: F1ModelRegistryEntry,
        evidence: PromotionEvidence,
        baseline: Optional[F1ModelRegistryEntry],
        fallback_model_id: Optional[str],
        fallback: Optional[F1ModelRegistryEntry],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if not evidence.split_strategy:
            reasons.append("split_strategy_missing")
        elif _is_random_split(evidence.split_strategy):
            reasons.append("random_split_not_allowed")
        if evidence.calibration_report_path is None:
            reasons.append("calibration_report_missing")
        if baseline is None:
            reasons.append("baseline_model_not_registered")
        if fallback_model_id is None:
            reasons.append("deterministic_fallback_missing")
        elif fallback is None:
            reasons.append("deterministic_fallback_not_registered")
        else:
            if fallback.model_family != candidate.model_family:
                reasons.append("deterministic_fallback_family_mismatch")
            if not fallback.deterministic_fallback:
                reasons.append("fallback_model_not_deterministic")
        return tuple(reasons)


def promotion_config_for_family(model_family: str) -> PromotionGateConfig:
    family = str(model_family).strip().lower()
    if family in {"ultimate_lap_time", "ultimate_lap_time_deep"}:
        return ultimate_lap_time_promotion_config()
    if family in {"live_strategy", "live_strategy_rl", "live_race"}:
        return live_strategy_promotion_config(require_simulator_validation=True)
    raise ValueError(f"no F1 promotion config registered for model_family={model_family!r}")


def registry_entry_from_profile(profile_payload: Mapping[str, object]) -> F1ModelRegistryEntry:
    registry_payload = profile_payload.get("registry")
    if not isinstance(registry_payload, Mapping):
        raise ValueError("profile payload requires a registry mapping")
    return F1ModelRegistryEntry.from_payload(registry_payload)


def _payload_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _promotion_status(value: object) -> str:
    status = _required_text(value, "promotion_status")
    if status not in ALLOWED_PROMOTION_STATUSES:
        raise ValueError(f"promotion_status must be one of {ALLOWED_PROMOTION_STATUSES}")
    return status


def _artifact_path(value: object, field_name: str) -> str:
    text = _required_text(value, field_name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or path.parts[0] != "artifacts" or ".." in path.parts or len(path.parts) < 2:
        raise ValueError(f"{field_name} must be a relative path under artifacts/")
    return path.as_posix()


def _optional_artifact_path(value: object, field_name: str) -> Optional[str]:
    text = _optional_text(value)
    if text is None:
        return None
    return _artifact_path(text, field_name)


def _metrics(value: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("metrics must be a mapping")
    normalized: dict[str, float] = {}
    for key, raw_value in value.items():
        metric_name = _required_text(key, "metric name")
        try:
            metric_value = float(raw_value)
        except Exception as exc:
            raise ValueError(f"metric {metric_name} must be numeric") from exc
        if not math.isfinite(metric_value):
            raise ValueError(f"metric {metric_name} must be finite")
        normalized[metric_name] = metric_value
    return normalized


def _is_random_split(split_strategy: str) -> bool:
    normalized = split_strategy.strip().lower().replace("-", "_").replace(" ", "_")
    if "random" in {token for token in normalized.split("_") if token}:
        return True
    blocked = {
        "random",
        "random_split",
        "row_random",
        "iid_random",
        "train_test_random",
        "random_holdout",
        "session_row_random",
    }
    if normalized in blocked:
        return True
    blocked_fragments = (
        "row_random",
        "iid_random",
        "random_split",
        "random_holdout",
        "random_train_test",
    )
    return any(fragment in normalized for fragment in blocked_fragments)


__all__ = [
    "ALLOWED_PROMOTION_STATUSES",
    "ARCHIVED_STATUS",
    "CANDIDATE_STATUS",
    "F1ModelRegistryEntry",
    "FALLBACK_STATUS",
    "ModelRegistry",
    "PRODUCTION_STATUS",
    "PromotionEvidence",
    "REJECTED_STATUS",
    "REQUIRED_REGISTRY_FIELDS",
    "RegistryPromotionEvaluation",
    "RegistryPromotionResult",
    "promotion_config_for_family",
    "registry_entry_from_profile",
]
