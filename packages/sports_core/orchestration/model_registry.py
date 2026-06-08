"""Small in-process model registry for CLI and app orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class RegisteredModel:
    name: str
    sport: str
    family: str
    predict: Callable[..., object]


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, RegisteredModel] = {}

    def register(self, model: RegisteredModel) -> None:
        self._models[model.name] = model

    def get(self, name: str) -> RegisteredModel:
        return self._models[name]

    def all(self) -> Mapping[str, RegisteredModel]:
        return dict(self._models)
