"""Track geometry projection for OpenF1 live locations.

OpenF1 location samples are approximate X/Y points. When a FastF1-generated
centerline exists for the same session or mapped circuit, project those points
onto the polyline and expose normalized track progress to the reducer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .fastf1_analysis import FastF1ArtifactStore
from .schemas import JsonObject


@dataclass(frozen=True, slots=True)
class CenterlinePoint:
    distance: float
    progress: float
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class Centerline:
    session_key: str
    artifact_id: str
    source: str
    points: tuple[CenterlinePoint, ...]


@dataclass(frozen=True, slots=True)
class TrackProjection:
    progress: float
    distance: float
    x: float
    y: float
    z: float
    error: float
    source: str


class TrackProjectionProvider(Protocol):
    def project(self, session_key: int | str, location: JsonObject) -> TrackProjection | None:
        ...


class FastF1CenterlineProjector:
    """Project location samples onto FastF1 centerline artifacts."""

    def __init__(
        self,
        artifact_store: FastF1ArtifactStore,
        *,
        session_aliases: dict[str, str] | None = None,
        fallback_to_latest: bool = False,
        max_centerline_rows: int = 20_000,
    ) -> None:
        self.artifact_store = artifact_store
        self.session_aliases = dict(session_aliases or {})
        self.fallback_to_latest = fallback_to_latest
        self.max_centerline_rows = max(2, max_centerline_rows)
        self._centerline_by_artifact_id: dict[str, Centerline] = {}
        self._centerline_by_session_id: dict[str, Centerline | None] = {}

    def set_session_alias(self, session_key: int | str, centerline_session_key: str | None) -> None:
        session_id = str(session_key)
        if centerline_session_key:
            self.session_aliases[session_id] = str(centerline_session_key)
        else:
            self.session_aliases.pop(session_id, None)
        self._centerline_by_session_id.pop(session_id, None)

    def clear_cache(self, session_key: int | str | None = None) -> None:
        if session_key is None:
            self._centerline_by_artifact_id.clear()
            self._centerline_by_session_id.clear()
            return
        self._centerline_by_session_id.pop(str(session_key), None)

    def centerline_for_session(
        self,
        session_key: int | str,
        *,
        centerline_session_key: int | str | None = None,
    ) -> Centerline | None:
        """Return the FastF1 centerline used for a live/OpenF1 session."""

        if centerline_session_key is not None:
            centerline = self._load_latest_centerline(str(centerline_session_key))
            if centerline is not None:
                return centerline
        return self._centerline_for_session(session_key)

    def project(self, session_key: int | str, location: JsonObject) -> TrackProjection | None:
        xy = _xy_from_location(location)
        if xy is None:
            return None
        centerline = self._centerline_for_session(session_key)
        if centerline is None:
            return None
        projection = project_location_to_centerline(xy[0], xy[1], centerline.points)
        if projection is None:
            return None
        return TrackProjection(
            progress=projection.progress,
            distance=projection.distance,
            x=projection.x,
            y=projection.y,
            z=projection.z,
            error=projection.error,
            source=centerline.source,
        )

    def _centerline_for_session(self, session_key: int | str) -> Centerline | None:
        session_id = str(session_key)
        if session_id in self._centerline_by_session_id:
            return self._centerline_by_session_id[session_id]

        for candidate_session_key in self._candidate_session_keys(session_id):
            centerline = self._load_latest_centerline(candidate_session_key)
            if centerline is not None:
                self._centerline_by_session_id[session_id] = centerline
                return centerline

        if self.fallback_to_latest:
            centerline = self._load_latest_centerline(None)
            if centerline is not None:
                self._centerline_by_session_id[session_id] = centerline
                return centerline

        self._centerline_by_session_id[session_id] = None
        return None

    def _candidate_session_keys(self, session_id: str) -> tuple[str, ...]:
        candidates: list[str] = []
        alias = self.session_aliases.get(session_id)
        if alias:
            candidates.append(alias)
        candidates.append(session_id)
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return tuple(deduped)

    def _load_latest_centerline(self, session_key: str | None) -> Centerline | None:
        artifacts = self.artifact_store.list_artifacts(
            session_key=session_key,
            kind="fastf1_centerline",
            limit=1,
        )
        if not artifacts:
            return None
        artifact = artifacts[0]
        if not artifact.artifact_id:
            return None
        cached = self._centerline_by_artifact_id.get(artifact.artifact_id)
        if cached is not None:
            return cached
        _, rows, _, _ = self.artifact_store.read_table_rows(artifact.artifact_id, limit=self.max_centerline_rows)
        centerline = centerline_from_rows(
            rows,
            session_key=str(artifact.metadata.get("sessionKey") or session_key or "unknown"),
            artifact_id=artifact.artifact_id,
            source=str(artifact.relative_path or artifact.path),
        )
        if centerline is not None:
            self._centerline_by_artifact_id[artifact.artifact_id] = centerline
        return centerline


def centerline_from_rows(
    rows: Sequence[JsonObject],
    *,
    session_key: str,
    artifact_id: str,
    source: str,
) -> Centerline | None:
    raw_points = []
    for row in rows:
        distance = _finite_float(row.get("Distance", row.get("distance")))
        x = _finite_float(row.get("X", row.get("x")))
        y = _finite_float(row.get("Y", row.get("y")))
        if distance is None or x is None or y is None:
            continue
        raw_points.append(
            {
                "distance": distance,
                "progress": _finite_float(row.get("Progress", row.get("progress"))),
                "x": x,
                "y": y,
                "z": _finite_float(row.get("Z", row.get("z"))) or 0.0,
            }
        )
    raw_points.sort(key=lambda item: item["distance"])
    if len(raw_points) < 2:
        return None
    total_distance = max(1e-9, float(raw_points[-1]["distance"]))
    points = tuple(
        CenterlinePoint(
            distance=float(row["distance"]),
            progress=_clamp01(float(row["progress"]) if row["progress"] is not None else float(row["distance"]) / total_distance),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
        )
        for row in raw_points
    )
    return Centerline(session_key=session_key, artifact_id=artifact_id, source=source, points=points)


def project_location_to_centerline(
    x: float,
    y: float,
    points: Sequence[CenterlinePoint],
) -> TrackProjection | None:
    if len(points) < 2:
        return None

    best: TrackProjection | None = None
    best_error_sq = float("inf")
    for left, right in zip(points, points[1:]):
        dx = right.x - left.x
        dy = right.y - left.y
        denominator = dx * dx + dy * dy
        t = 0.0
        if denominator > 0:
            t = _clamp01(((x - left.x) * dx + (y - left.y) * dy) / denominator)
        projected_x = left.x + t * dx
        projected_y = left.y + t * dy
        error_sq = (x - projected_x) ** 2 + (y - projected_y) ** 2
        if error_sq >= best_error_sq:
            continue
        best_error_sq = error_sq
        best = TrackProjection(
            progress=_clamp01(left.progress + t * (right.progress - left.progress)),
            distance=left.distance + t * (right.distance - left.distance),
            x=projected_x,
            y=projected_y,
            z=left.z + t * (right.z - left.z),
            error=math.sqrt(error_sq),
            source="centerline",
        )
    return best


def parse_session_aliases(value: str | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not value:
        return aliases
    for item in value.split(","):
        if "=" not in item:
            continue
        session_key, centerline_session_key = item.split("=", 1)
        session_key = session_key.strip()
        centerline_session_key = centerline_session_key.strip()
        if session_key and centerline_session_key:
            aliases[session_key] = centerline_session_key
    return aliases


def _xy_from_location(location: JsonObject) -> tuple[float, float] | None:
    x = _finite_float(location.get("x", location.get("X")))
    y = _finite_float(location.get("y", location.get("Y")))
    if x is None or y is None:
        return None
    return x, y


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value
