"""Composable pipeline contracts for sport prediction workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


PipelineStep = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass
class Pipeline:
    name: str
    steps: list[PipelineStep] = field(default_factory=list)

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        state: Mapping[str, Any] = dict(context)
        for step in self.steps:
            state = step(state)
        return state
