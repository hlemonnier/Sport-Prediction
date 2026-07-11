"""FastF1 post-session analytics and artifact generation.

FastF1 is intentionally a batch/post-session source in this platform. The live
path stays on OpenF1 events; this module produces durable engineering artifacts
that can later be queried by the API, models, and frontend.
"""

from __future__ import annotations

import json
import math
import re
import base64
import binascii
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .schemas import F1Event, JsonObject
from .time import utc_now_iso

DEFAULT_DISTANCE_STEP_METERS = 5.0
DEFAULT_TELEMETRY_CHANNELS = ("Speed", "RPM", "Throttle", "Brake", "nGear", "DRS", "X", "Y", "Z")
FASTF1_DRIVER_NUMBERS: dict[str, int] = {
    "ALB": 23,
    "ALO": 14,
    "ANT": 12,
    "BEA": 87,
    "BOR": 5,
    "BOT": 77,
    "COL": 43,
    "GAS": 10,
    "HAD": 6,
    "HAM": 44,
    "HUL": 27,
    "LEC": 16,
    "LAW": 30,
    "NOR": 4,
    "OCO": 31,
    "PIA": 81,
    "RUS": 63,
    "SAI": 55,
    "STR": 18,
    "TSU": 22,
    "VER": 1,
}
FASTF1_TEAM_COLOURS: dict[str, str] = {
    "alpine": "00A1E8",
    "aston martin": "006F62",
    "aston martin racing": "006F62",
    "audi": "C0C0C0",
    "cadillac": "B8B8B8",
    "ferrari": "F91536",
    "haas f1 team": "B6BABD",
    "kick sauber": "00E701",
    "mclaren": "FF8700",
    "mercedes": "27F4D2",
    "racing bulls": "6692FF",
    "rb": "6692FF",
    "red bull racing": "3671C6",
    "williams": "64C4FF",
}


@dataclass(slots=True)
class FastF1ImportRequest:
    year: int
    event: str | int
    session_name: str = "R"
    drivers: tuple[str, ...] = ()
    include_telemetry: bool = True
    telemetry_laps_per_driver: int = 1
    distance_step_meters: float = DEFAULT_DISTANCE_STEP_METERS
    output_format: str = "parquet"


@dataclass(slots=True)
class FastF1LapTelemetry:
    driver: str
    lap_number: int
    lap_time_seconds: float | None
    telemetry: list[JsonObject]


@dataclass(slots=True)
class FastF1LoadedSession:
    year: int
    event_name: str
    session_name: str
    session_key: str
    laps: list[JsonObject]
    weather: list[JsonObject]
    race_control: list[JsonObject]
    telemetry_laps: list[FastF1LapTelemetry]


class FastF1SessionProvider(Protocol):
    def load_session(self, request: FastF1ImportRequest) -> FastF1LoadedSession:
        ...


@dataclass(slots=True)
class ArtifactRecord:
    kind: str
    path: str
    format: str
    row_count: int | None
    metadata: JsonObject
    artifact_id: str | None = None
    relative_path: str | None = None

    def to_dict(self) -> JsonObject:
        payload = asdict(self)
        payload["artifactId"] = payload.pop("artifact_id")
        payload["relativePath"] = payload.pop("relative_path")
        return payload


@dataclass(slots=True)
class FastF1ImportResult:
    session_key: str
    generated_at: str
    artifacts: list[ArtifactRecord]
    notes: list[str]
    runtime_events: list[F1Event] = field(default_factory=list, repr=False)

    def to_dict(self) -> JsonObject:
        return {
            "sessionKey": self.session_key,
            "generatedAt": self.generated_at,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "notes": list(self.notes),
            "eventCount": len(self.runtime_events),
        }


class FastF1ArtifactStore:
    """Write bounded tabular artifacts with Parquet preferred."""

    def __init__(self, root: str | Path, *, allow_json_fallback: bool = True) -> None:
        self.root = Path(root)
        self.allow_json_fallback = allow_json_fallback

    def write_table(
        self,
        rows: Sequence[JsonObject],
        relative_path_without_suffix: str | Path,
        *,
        preferred_format: str,
        metadata: JsonObject,
    ) -> ArtifactRecord:
        normalized_format = preferred_format.lower().strip()
        if normalized_format not in {"parquet", "jsonl"}:
            raise ValueError("preferred_format must be parquet or jsonl")
        if normalized_format == "parquet":
            try:
                return self._write_parquet(rows, relative_path_without_suffix, metadata=metadata)
            except ImportError:
                if not self.allow_json_fallback:
                    raise
        return self._write_jsonl(rows, relative_path_without_suffix, metadata=metadata)

    def _write_parquet(
        self,
        rows: Sequence[JsonObject],
        relative_path_without_suffix: str | Path,
        *,
        metadata: JsonObject,
    ) -> ArtifactRecord:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised through fallback tests
            raise ImportError("Install pandas and pyarrow to write FastF1 Parquet artifacts") from exc
        path = self._path(relative_path_without_suffix, ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([_json_safe(row) for row in rows])
        frame.to_parquet(path, index=False)
        record = self._record(
            kind=str(metadata.get("kind", "table")),
            path=path,
            format="parquet",
            row_count=len(rows),
            metadata=dict(metadata),
        )
        self._write_metadata(record)
        return record

    def _write_jsonl(
        self,
        rows: Sequence[JsonObject],
        relative_path_without_suffix: str | Path,
        *,
        metadata: JsonObject,
    ) -> ArtifactRecord:
        path = self._path(relative_path_without_suffix, ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":")) + "\n")
        record = self._record(
            kind=str(metadata.get("kind", "table")),
            path=path,
            format="jsonl",
            row_count=len(rows),
            metadata=dict(metadata),
        )
        self._write_metadata(record)
        return record

    def _path(self, relative_path_without_suffix: str | Path, suffix: str) -> Path:
        relative = Path(relative_path_without_suffix)
        if relative.is_absolute():
            raise ValueError("artifact path must be relative")
        return self.root / relative.with_suffix(suffix)

    def list_artifacts(
        self,
        *,
        session_key: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[ArtifactRecord]:
        bounded_limit = max(1, min(1_000, int(limit)))
        records: list[ArtifactRecord] = []
        if not self.root.exists():
            return []
        for path in self._artifact_paths():
            record = self._record_from_path(path)
            if session_key and str(record.metadata.get("sessionKey")) != str(session_key):
                continue
            if kind and record.kind != kind:
                continue
            records.append(record)
            if len(records) >= bounded_limit:
                break
        return records

    def read_artifact_rows(self, artifact_id: str, *, limit: int = 200) -> JsonObject:
        record, rows, truncated, bounded_limit = self.read_table_rows(
            artifact_id,
            limit=max(1, min(1_000, int(limit))),
        )
        return {
            "artifact": record.to_dict(),
            "columns": _columns_from_rows(rows),
            "rows": rows,
            "limit": bounded_limit,
            "truncated": truncated,
        }

    def read_table_rows(self, artifact_id: str, *, limit: int = 20_000) -> tuple[ArtifactRecord, list[JsonObject], bool, int]:
        path = self._path_from_artifact_id(artifact_id)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"FastF1 artifact not found: {artifact_id}")
        if path.name.endswith(".metadata.json") or path.suffix.lower() not in {".jsonl", ".parquet"}:
            raise ValueError("Artifact id does not point to a readable table")
        record = self._record_from_path(path)
        bounded_limit = max(1, min(100_000, int(limit)))
        if record.format == "jsonl":
            rows, truncated = self._read_jsonl_rows(path, limit=bounded_limit)
        elif record.format == "parquet":
            rows, truncated = self._read_parquet_rows(path, limit=bounded_limit)
        else:
            raise ValueError(f"Unsupported artifact format: {record.format}")
        return record, rows, truncated, bounded_limit

    def _record(
        self,
        *,
        kind: str,
        path: Path,
        format: str,
        row_count: int | None,
        metadata: JsonObject,
    ) -> ArtifactRecord:
        relative_path = self._relative_path(path)
        return ArtifactRecord(
            kind=kind,
            path=str(path),
            format=format,
            row_count=row_count,
            metadata=dict(metadata),
            artifact_id=self._artifact_id_for_relative_path(relative_path),
            relative_path=relative_path,
        )

    def _write_metadata(self, record: ArtifactRecord) -> None:
        path = Path(record.path)
        metadata_path = self._metadata_path(path)
        metadata_path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def _record_from_path(self, path: Path) -> ArtifactRecord:
        metadata_path = self._metadata_path(path)
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = payload.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                kind = str(payload.get("kind") or metadata.get("kind") or self._infer_kind(path))
                row_count = _optional_int(payload.get("row_count"))
                return self._record(
                    kind=kind,
                    path=path,
                    format=str(payload.get("format") or path.suffix.lstrip(".")).lower(),
                    row_count=row_count,
                    metadata=metadata,
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        metadata = self._infer_metadata(path)
        return self._record(
            kind=str(metadata.get("kind", self._infer_kind(path))),
            path=path,
            format=path.suffix.lstrip(".").lower(),
            row_count=None,
            metadata=metadata,
        )

    def _artifact_paths(self) -> list[Path]:
        paths = [
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and not path.name.endswith(".metadata.json")
            and path.suffix.lower() in {".jsonl", ".parquet"}
        ]
        paths.sort(key=lambda item: (item.stat().st_mtime, item.as_posix()), reverse=True)
        return paths

    def _read_jsonl_rows(self, path: Path, *, limit: int) -> tuple[list[JsonObject], bool]:
        rows: list[JsonObject] = []
        truncated = False
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    truncated = True
                    break
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(_json_safe(value))
        return rows, truncated

    def _read_parquet_rows(self, path: Path, *, limit: int) -> tuple[list[JsonObject], bool]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
                raise RuntimeError("Install pyarrow or pandas to read FastF1 Parquet artifacts") from exc
            frame = pd.read_parquet(path)
            rows = [_normalize_record(row) for row in frame.head(limit).to_dict("records")]
            return rows, len(frame) > limit

        parquet_file = pq.ParquetFile(path)
        remaining = limit
        chunks = []
        for group_index in range(parquet_file.num_row_groups):
            if remaining <= 0:
                break
            table = parquet_file.read_row_group(group_index)
            if table.num_rows > remaining:
                table = table.slice(0, remaining)
            chunks.append(table)
            remaining -= table.num_rows
        if not chunks:
            return [], False
        table = pa.concat_tables(chunks)
        rows = [_normalize_record(row) for row in table.to_pylist()]
        truncated = parquet_file.metadata.num_rows > limit if parquet_file.metadata is not None else len(rows) >= limit
        return rows, truncated

    def _metadata_path(self, path: Path) -> Path:
        return path.with_name(f"{path.name}.metadata.json")

    def _relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.name

    def _artifact_id_for_relative_path(self, relative_path: str) -> str:
        encoded = base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii")
        return encoded.rstrip("=")

    def _path_from_artifact_id(self, artifact_id: str) -> Path:
        padding = "=" * (-len(artifact_id) % 4)
        try:
            decoded = base64.urlsafe_b64decode(f"{artifact_id}{padding}".encode("ascii")).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError("Invalid FastF1 artifact id") from exc
        relative = Path(decoded)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid FastF1 artifact path")
        root = self.root.resolve()
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError("Invalid FastF1 artifact path")
        return path

    def _infer_metadata(self, path: Path) -> JsonObject:
        relative = self._relative_path(path)
        parts = Path(relative).parts
        metadata: JsonObject = {"kind": self._infer_kind(path)}
        session_key = _session_key_from_partition(parts)
        if session_key:
            metadata["sessionKey"] = session_key
        for part in parts:
            if part.startswith("driver="):
                metadata["driver"] = part.split("=", 1)[1].upper()
            if part.startswith("lap="):
                metadata["lapNumber"] = _optional_int(part.split("=", 1)[1])
        return metadata

    def _infer_kind(self, path: Path) -> str:
        relative = self._relative_path(path)
        parts = Path(relative).parts
        if not parts:
            return "fastf1_artifact"
        if parts[0] == "fastf1":
            return f"fastf1_{path.stem}"
        if parts[0] == "telemetry":
            return "fastf1_distance_aligned_telemetry"
        if parts[0] == "centerline":
            return "fastf1_centerline"
        if parts[0] == "telemetry_comparison":
            return "fastf1_telemetry_delta"
        if parts[0] == "corner_metrics":
            return "fastf1_corner_metrics"
        return "fastf1_artifact"


class FastF1ArtifactService:
    def __init__(
        self,
        artifact_store: FastF1ArtifactStore,
        *,
        provider: FastF1SessionProvider | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.provider = provider or FastF1Provider(cache_dir=cache_dir)

    def import_session(self, request: FastF1ImportRequest) -> FastF1ImportResult:
        loaded = self.provider.load_session(request)
        base = _session_partition(loaded.year, loaded.event_name, loaded.session_name)
        artifacts: list[ArtifactRecord] = []
        notes: list[str] = []
        preferred_format = request.output_format

        artifacts.append(
            self.artifact_store.write_table(
                loaded.laps,
                Path("fastf1") / base / "laps",
                preferred_format=preferred_format,
                metadata={"kind": "fastf1_laps", "sessionKey": loaded.session_key},
            )
        )
        if loaded.weather:
            artifacts.append(
                self.artifact_store.write_table(
                    loaded.weather,
                    Path("fastf1") / base / "weather",
                    preferred_format=preferred_format,
                    metadata={"kind": "fastf1_weather", "sessionKey": loaded.session_key},
                )
            )
        if loaded.race_control:
            artifacts.append(
                self.artifact_store.write_table(
                    loaded.race_control,
                    Path("fastf1") / base / "race_control",
                    preferred_format=preferred_format,
                    metadata={"kind": "fastf1_race_control", "sessionKey": loaded.session_key},
                )
            )

        resampled_laps: list[tuple[FastF1LapTelemetry, list[JsonObject]]] = []
        if request.include_telemetry:
            for lap in loaded.telemetry_laps:
                resampled = resample_telemetry(lap.telemetry, distance_step_meters=request.distance_step_meters)
                resampled_laps.append((lap, resampled))
                artifacts.append(
                    self.artifact_store.write_table(
                        resampled,
                        Path("telemetry")
                        / base
                        / f"driver={_slug(lap.driver)}"
                        / f"lap={lap.lap_number}"
                        / "part-000",
                        preferred_format=preferred_format,
                        metadata={
                            "kind": "fastf1_distance_aligned_telemetry",
                            "sessionKey": loaded.session_key,
                            "driver": lap.driver,
                            "lapNumber": lap.lap_number,
                            "distanceStepMeters": request.distance_step_meters,
                            "lapTimeSeconds": lap.lap_time_seconds,
                        },
                    )
                )
                corner_metrics = build_corner_metrics(resampled, driver=lap.driver, lap_number=lap.lap_number)
                if corner_metrics:
                    artifacts.append(
                        self.artifact_store.write_table(
                            corner_metrics,
                            Path("corner_metrics")
                            / base
                            / f"driver={_slug(lap.driver)}"
                            / f"lap={lap.lap_number}"
                            / "part-000",
                            preferred_format=preferred_format,
                            metadata={
                                "kind": "fastf1_corner_metrics",
                                "sessionKey": loaded.session_key,
                                "driver": lap.driver,
                                "lapNumber": lap.lap_number,
                                "method": "distance_window_local_speed_minima",
                            },
                        )
                    )

            centerline_source = _first_lap_with_position(resampled_laps)
            if centerline_source is not None:
                lap, telemetry = centerline_source
                centerline = build_centerline(telemetry)
                artifacts.append(
                    self.artifact_store.write_table(
                        centerline,
                        Path("centerline") / base / "canonical",
                        preferred_format=preferred_format,
                        metadata={
                            "kind": "fastf1_centerline",
                            "sessionKey": loaded.session_key,
                            "sourceDriver": lap.driver,
                            "sourceLapNumber": lap.lap_number,
                            "distanceStepMeters": request.distance_step_meters,
                        },
                    )
                )
            else:
                notes.append("No telemetry lap contained X/Y position channels for centreline generation.")

            if len(resampled_laps) >= 2:
                first_lap, first_rows = resampled_laps[0]
                second_lap, second_rows = resampled_laps[1]
                delta_rows = build_telemetry_delta(first_rows, second_rows)
                artifacts.append(
                    self.artifact_store.write_table(
                        delta_rows,
                        Path("telemetry_comparison")
                        / base
                        / f"{_slug(first_lap.driver)}-lap-{first_lap.lap_number}__{_slug(second_lap.driver)}-lap-{second_lap.lap_number}",
                        preferred_format=preferred_format,
                        metadata={
                            "kind": "fastf1_telemetry_delta",
                            "sessionKey": loaded.session_key,
                            "driverA": first_lap.driver,
                            "lapA": first_lap.lap_number,
                            "driverB": second_lap.driver,
                            "lapB": second_lap.lap_number,
                        },
                    )
                )
        else:
            notes.append("Telemetry artifacts skipped by request.")

        return FastF1ImportResult(
            session_key=loaded.session_key,
            generated_at=utc_now_iso(),
            artifacts=artifacts,
            notes=notes,
            runtime_events=fastf1_runtime_events(loaded),
        )

    def list_artifacts(
        self,
        *,
        session_key: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[ArtifactRecord]:
        return self.artifact_store.list_artifacts(session_key=session_key, kind=kind, limit=limit)

    def read_artifact_rows(self, artifact_id: str, *, limit: int = 200) -> JsonObject:
        return self.artifact_store.read_artifact_rows(artifact_id, limit=limit)

    def engineering_summary(
        self,
        *,
        session_key: str | None = None,
        telemetry_limit: int = 240,
        corner_limit: int = 12,
    ) -> JsonObject:
        telemetry_records = self.list_artifacts(
            session_key=session_key,
            kind="fastf1_telemetry_delta",
            limit=8,
        )
        corner_records = self.list_artifacts(
            session_key=session_key,
            kind="fastf1_corner_metrics",
            limit=max(1, min(50, corner_limit)),
        )

        telemetry_summary = None
        if telemetry_records:
            record = telemetry_records[0]
            _record, rows, truncated, bounded_limit = self.artifact_store.read_table_rows(
                str(record.artifact_id),
                limit=max(1, min(2_000, telemetry_limit)),
            )
            telemetry_summary = _telemetry_delta_summary(record, rows, truncated=truncated, limit=bounded_limit)

        corner_summaries = []
        for record in corner_records[: max(1, min(50, corner_limit))]:
            _record, rows, truncated, bounded_limit = self.artifact_store.read_table_rows(
                str(record.artifact_id),
                limit=100,
            )
            corner_summaries.append(_corner_metrics_summary(record, rows, truncated=truncated, limit=bounded_limit))

        resolved_session_key = session_key
        if resolved_session_key is None:
            for record in telemetry_records or corner_records:
                value = record.metadata.get("sessionKey")
                if value:
                    resolved_session_key = str(value)
                    break

        return {
            "sessionKey": resolved_session_key,
            "generatedAt": utc_now_iso(),
            "telemetryDelta": telemetry_summary,
            "cornerMetrics": corner_summaries,
            "artifactCounts": {
                "telemetryDelta": len(telemetry_records),
                "cornerMetrics": len(corner_records),
            },
        }


def fastf1_runtime_events(loaded: FastF1LoadedSession) -> list[F1Event]:
    """Convert a loaded FastF1 session into the platform reducer contract."""

    source_id = 1
    events: list[F1Event] = []
    session_key = loaded.session_key
    driver_numbers = _fastf1_driver_number_map(loaded.laps)

    events.append(
        _fastf1_runtime_event(
            source_id,
            "v1/sessions",
            f"fastf1:session:{session_key}",
            session_key,
            {
                "session_key": session_key,
                "session_name": _fastf1_session_label(loaded.session_name),
                "session_type": _fastf1_session_type(loaded.session_name),
                "event_name": loaded.event_name,
                "fastf1_event_name": loaded.event_name,
                "location": loaded.event_name,
                "year": loaded.year,
            },
        )
    )
    source_id += 1

    for driver_number, row in _fastf1_driver_rows(loaded.laps, driver_numbers):
        code = _fastf1_driver_code(row, fallback=str(driver_number))
        team = _text_value(row.get("Team"), row.get("TeamName"))
        payload: JsonObject = {
            "name_acronym": code,
            "full_name": _text_value(row.get("FullName"), row.get("DriverFullName"), row.get("BroadcastName")) or code,
        }
        if team:
            payload["team_name"] = team
            colour = FASTF1_TEAM_COLOURS.get(team.strip().lower())
            if colour:
                payload["team_colour"] = colour
        events.append(
            _fastf1_runtime_event(
                source_id,
                "v1/drivers",
                f"fastf1:driver:{driver_number}",
                session_key,
                payload,
                driver_number=driver_number,
            )
        )
        source_id += 1

    for driver_number, position in _fastf1_positions(loaded.laps, driver_numbers).items():
        events.append(
            _fastf1_runtime_event(
                source_id,
                "v1/position",
                f"fastf1:position:{driver_number}",
                session_key,
                {"position": position},
                driver_number=driver_number,
            )
        )
        source_id += 1

    for segment in _fastf1_stint_segments(loaded.laps, driver_numbers):
        driver_number = int(segment["driver_number"])
        stint_number = int(segment["stint_number"])
        events.append(
            _fastf1_runtime_event(
                source_id,
                "v1/stints",
                f"fastf1:stint:{driver_number}:{stint_number}",
                session_key,
                segment,
                driver_number=driver_number,
            )
        )
        source_id += 1

    lap_rows = sorted(
        loaded.laps,
        key=lambda row: (
            _optional_int(row.get("LapNumber")) or 10_000,
            _fastf1_driver_number(row, driver_numbers) or 10_000,
        ),
    )
    for row in lap_rows:
        driver_number = _fastf1_driver_number(row, driver_numbers)
        lap_number = _optional_int(row.get("LapNumber"))
        if driver_number is None or lap_number is None:
            continue
        lap_time = _finite_float(row.get("LapTimeSeconds"))
        sector_1 = _finite_float(row.get("Sector1TimeSeconds"))
        sector_2 = _finite_float(row.get("Sector2TimeSeconds"))
        sector_3 = _finite_float(row.get("Sector3TimeSeconds"))
        if lap_time is None and sector_1 is None and sector_2 is None and sector_3 is None:
            continue
        payload = {
            "lap_number": lap_number,
            "lap_duration": lap_time,
            "duration_sector_1": sector_1,
            "duration_sector_2": sector_2,
            "duration_sector_3": sector_3,
            "compound": _text_value(row.get("Compound")),
            "tyre_age": _optional_int(row.get("TyreLife")),
            "is_personal_best": row.get("IsPersonalBest"),
            "raw_fastf1": row,
        }
        events.append(
            _fastf1_runtime_event(
                source_id,
                "v1/laps",
                f"fastf1:lap:{driver_number}:{lap_number}",
                session_key,
                {key: value for key, value in payload.items() if value is not None},
                driver_number=driver_number,
                event_time=_fastf1_event_time(row.get("LapStartDate"), row.get("Time")),
            )
        )
        source_id += 1

    for index, row in enumerate(loaded.weather, start=1):
        payload = _fastf1_weather_payload(row)
        if not payload:
            continue
        events.append(
            _fastf1_runtime_event(
                source_id,
                "v1/weather",
                f"fastf1:weather:{index}",
                session_key,
                payload,
                event_time=_fastf1_event_time(row.get("Date"), row.get("Time")),
            )
        )
        source_id += 1

    for index, row in enumerate(loaded.race_control, start=1):
        events.append(
            _fastf1_runtime_event(
                source_id,
                "v1/race_control",
                f"fastf1:race-control:{index}",
                session_key,
                dict(row),
                event_time=_fastf1_event_time(row.get("Date"), row.get("Time")),
            )
        )
        source_id += 1

    for driver_number, position in _fastf1_positions(loaded.laps, driver_numbers).items():
        driver_laps = [
            row
            for row in loaded.laps
            if _fastf1_driver_number(row, driver_numbers) == driver_number and _optional_int(row.get("LapNumber")) is not None
        ]
        events.append(
            _fastf1_runtime_event(
                source_id,
                "v1/session_result",
                f"fastf1:result:{driver_number}",
                session_key,
                {
                    "position": position,
                    "driver_number": driver_number,
                    "number_of_laps": len(driver_laps),
                },
                driver_number=driver_number,
            )
        )
        source_id += 1

    return events


def _fastf1_runtime_event(
    source_id: int,
    topic: str,
    source_key: str,
    session_key: str,
    payload: JsonObject,
    *,
    driver_number: int | None = None,
    event_time: str | None = None,
) -> F1Event:
    return F1Event.from_payload(
        {
            "source": "fastf1",
            "topic": topic,
            "source_id": source_id,
            "source_key": source_key,
            "session_key": session_key,
            "driver_number": driver_number,
            "event_time": event_time,
            "payload": payload,
        },
        source="fastf1",
        received_at=utc_now_iso(),
    )


def _fastf1_session_label(value: str) -> str:
    normalized = value.strip().upper()
    return {
        "FP1": "Practice 1",
        "FP2": "Practice 2",
        "FP3": "Practice 3",
        "P1": "Practice 1",
        "P2": "Practice 2",
        "P3": "Practice 3",
        "Q": "Qualifying",
        "SQ": "Sprint Qualifying",
        "S": "Sprint",
        "R": "Race",
    }.get(normalized, value)


def _fastf1_session_type(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"FP1", "FP2", "FP3", "P1", "P2", "P3"}:
        return "Practice"
    if normalized in {"Q", "SQ"}:
        return "Qualifying"
    if normalized == "S":
        return "Sprint"
    if normalized == "R":
        return "Race"
    return value


def _fastf1_driver_number_map(laps: Sequence[JsonObject]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    next_generated = 900
    for row in laps:
        code = _fastf1_driver_code(row, fallback="")
        if not code:
            continue
        direct = _optional_int(row.get("DriverNumber"))
        known = direct or FASTF1_DRIVER_NUMBERS.get(code.upper())
        if known is not None:
            mapping[code.upper()] = known
            continue
        if code.upper() not in mapping:
            mapping[code.upper()] = next_generated
            next_generated += 1
    return mapping


def _fastf1_driver_number(row: JsonObject, driver_numbers: dict[str, int]) -> int | None:
    direct = _optional_int(row.get("DriverNumber"))
    if direct is not None:
        return direct
    code = _fastf1_driver_code(row, fallback="")
    if not code:
        return None
    return driver_numbers.get(code.upper()) or FASTF1_DRIVER_NUMBERS.get(code.upper())


def _fastf1_driver_code(row: JsonObject, *, fallback: str) -> str:
    value = _text_value(row.get("Driver"), row.get("Abbreviation"), row.get("DriverCode"), row.get("BroadcastName"))
    if value:
        return value.strip().upper()[:3]
    return fallback.strip().upper()


def _fastf1_driver_rows(
    laps: Sequence[JsonObject],
    driver_numbers: dict[str, int],
) -> list[tuple[int, JsonObject]]:
    rows: dict[int, JsonObject] = {}
    for row in laps:
        driver_number = _fastf1_driver_number(row, driver_numbers)
        if driver_number is None or driver_number in rows:
            continue
        rows[driver_number] = row
    return sorted(rows.items(), key=lambda item: item[0])


def _fastf1_positions(
    laps: Sequence[JsonObject],
    driver_numbers: dict[str, int],
) -> dict[int, int]:
    positions: dict[int, int] = {}
    for row in sorted(laps, key=lambda item: _optional_int(item.get("LapNumber")) or 0):
        driver_number = _fastf1_driver_number(row, driver_numbers)
        position = _optional_int(row.get("Position"))
        if driver_number is not None and position is not None:
            positions[driver_number] = position
    if positions:
        return dict(sorted(positions.items(), key=lambda item: item[1]))

    best_laps: list[tuple[float, int]] = []
    for row in laps:
        driver_number = _fastf1_driver_number(row, driver_numbers)
        lap_time = _finite_float(row.get("LapTimeSeconds"))
        if driver_number is not None and lap_time is not None:
            best_laps.append((lap_time, driver_number))
    best_by_driver: dict[int, float] = {}
    for lap_time, driver_number in best_laps:
        current = best_by_driver.get(driver_number)
        if current is None or lap_time < current:
            best_by_driver[driver_number] = lap_time
    return {
        driver_number: index + 1
        for index, (driver_number, _lap_time) in enumerate(
            sorted(best_by_driver.items(), key=lambda item: item[1])
        )
    }


def _fastf1_stint_segments(
    laps: Sequence[JsonObject],
    driver_numbers: dict[str, int],
) -> list[JsonObject]:
    segments: dict[tuple[int, int], JsonObject] = {}
    for row in laps:
        driver_number = _fastf1_driver_number(row, driver_numbers)
        lap_number = _optional_int(row.get("LapNumber"))
        if driver_number is None or lap_number is None:
            continue
        stint_number = _optional_int(row.get("Stint")) or 1
        key = (driver_number, stint_number)
        compound = _text_value(row.get("Compound")) or "UNKNOWN"
        segment = segments.setdefault(
            key,
            {
                "driver_number": driver_number,
                "stint_number": stint_number,
                "compound": compound,
                "lap_start": lap_number,
                "lap_end": lap_number,
                "tyre_age_at_start": _optional_int(row.get("TyreLife")) or 0,
            },
        )
        segment["lap_start"] = min(int(segment["lap_start"]), lap_number)
        segment["lap_end"] = max(int(segment["lap_end"]), lap_number)
        if compound != "UNKNOWN":
            segment["compound"] = compound
    return [
        segment
        for _key, segment in sorted(segments.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _fastf1_weather_payload(row: JsonObject) -> JsonObject:
    payload = {
        "air_temperature": _first_number(row.get("AirTemp"), row.get("AirTemperature"), row.get("air_temperature")),
        "track_temperature": _first_number(row.get("TrackTemp"), row.get("TrackTemperature"), row.get("track_temperature")),
        "humidity": _first_number(row.get("Humidity"), row.get("humidity")),
        "pressure": _first_number(row.get("Pressure"), row.get("pressure")),
        "rainfall": _first_number(row.get("Rainfall"), row.get("rainfall")),
        "wind_speed": _first_number(row.get("WindSpeed"), row.get("wind_speed")),
        "wind_direction": _first_number(row.get("WindDirection"), row.get("wind_direction")),
        "raw_fastf1": row,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _fastf1_event_time(*values: Any) -> str | None:
    """Return only absolute timestamps; FastF1 session durations stay in the raw payload."""

    value = _text_value(*values)
    if value is None or _duration_string_seconds(value) is not None:
        return None
    try:
        float(value)
    except ValueError:
        return value
    return None


def _text_value(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "nat", "none", "null"}:
            return text
    return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _finite_float(value)
        if number is not None:
            return number
    return None


def _telemetry_delta_summary(
    record: ArtifactRecord,
    rows: list[JsonObject],
    *,
    truncated: bool,
    limit: int,
) -> JsonObject:
    distances = [_finite_float(row.get("Distance")) for row in rows]
    deltas = [_finite_float(row.get("DeltaSeconds")) for row in rows]
    speed_deltas = [
        (_finite_float(row.get("SpeedA")) or 0.0) - (_finite_float(row.get("SpeedB")) or 0.0)
        for row in rows
        if _finite_float(row.get("SpeedA")) is not None and _finite_float(row.get("SpeedB")) is not None
    ]
    valid_distances = [value for value in distances if value is not None]
    valid_deltas = [value for value in deltas if value is not None]
    sampled_rows = _sample_rows(rows, max_points=80)
    return {
        "artifact": record.to_dict(),
        "driverA": record.metadata.get("driverA"),
        "driverB": record.metadata.get("driverB"),
        "lapA": _optional_int(record.metadata.get("lapA")),
        "lapB": _optional_int(record.metadata.get("lapB")),
        "sampleCount": len(rows),
        "limit": limit,
        "truncated": truncated,
        "distanceStart": _metric_round(min(valid_distances)) if valid_distances else None,
        "distanceEnd": _metric_round(max(valid_distances)) if valid_distances else None,
        "finalDeltaSeconds": _metric_round(valid_deltas[-1]) if valid_deltas else None,
        "maxGainDriverASeconds": _metric_round(min(valid_deltas)) if valid_deltas else None,
        "maxGainDriverBSeconds": _metric_round(max(valid_deltas)) if valid_deltas else None,
        "maxSpeedDeltaKmh": _metric_round(max(speed_deltas, key=abs)) if speed_deltas else None,
        "series": [_telemetry_delta_point(row) for row in sampled_rows],
    }


def _telemetry_delta_point(row: JsonObject) -> JsonObject:
    return {
        "distance": _metric_round(row.get("Distance")),
        "deltaSeconds": _metric_round(row.get("DeltaSeconds")),
        "speedA": _metric_round(row.get("SpeedA")),
        "speedB": _metric_round(row.get("SpeedB")),
        "throttleA": _metric_round(row.get("ThrottleA")),
        "throttleB": _metric_round(row.get("ThrottleB")),
        "brakeA": _metric_round(row.get("BrakeA")),
        "brakeB": _metric_round(row.get("BrakeB")),
        "gearA": _metric_round(row.get("GearA")),
        "gearB": _metric_round(row.get("GearB")),
        "drsA": _metric_round(row.get("DRSA")),
        "drsB": _metric_round(row.get("DRSB")),
    }


def _corner_metrics_summary(
    record: ArtifactRecord,
    rows: list[JsonObject],
    *,
    truncated: bool,
    limit: int,
) -> JsonObject:
    corner_times = [_finite_float(row.get("CornerTimeSeconds")) for row in rows]
    minimum_speeds = [_finite_float(row.get("MinimumSpeed")) for row in rows]
    valid_corner_times = [value for value in corner_times if value is not None]
    valid_minimum_speeds = [value for value in minimum_speeds if value is not None]
    return {
        "artifact": record.to_dict(),
        "driver": record.metadata.get("driver"),
        "lapNumber": _optional_int(record.metadata.get("lapNumber")),
        "cornerCount": len(rows),
        "limit": limit,
        "truncated": truncated,
        "fastestCornerTimeSeconds": _metric_round(min(valid_corner_times)) if valid_corner_times else None,
        "slowestMinimumSpeedKmh": _metric_round(min(valid_minimum_speeds)) if valid_minimum_speeds else None,
        "corners": [_corner_metric_point(row) for row in rows[:20]],
    }


def _corner_metric_point(row: JsonObject) -> JsonObject:
    return {
        "cornerIndex": _optional_int(row.get("CornerIndex")),
        "entryDistance": _metric_round(row.get("EntryDistance")),
        "apexDistance": _metric_round(row.get("ApexDistance")),
        "exitDistance": _metric_round(row.get("ExitDistance")),
        "entrySpeed": _metric_round(row.get("EntrySpeed")),
        "minimumSpeed": _metric_round(row.get("MinimumSpeed")),
        "exitSpeed": _metric_round(row.get("ExitSpeed")),
        "brakeStartDistance": _metric_round(row.get("BrakeStartDistance")),
        "throttleReapplicationDistance": _metric_round(row.get("ThrottleReapplicationDistance")),
        "fullThrottlePercent": _metric_round(row.get("FullThrottlePercent")),
        "brakingDurationSeconds": _metric_round(row.get("BrakingDurationSeconds")),
        "cornerTimeSeconds": _metric_round(row.get("CornerTimeSeconds")),
        "exitAccelerationKmhPer100m": _metric_round(row.get("ExitAccelerationKmhPer100m")),
    }


def _sample_rows(rows: list[JsonObject], *, max_points: int) -> list[JsonObject]:
    if not rows:
        return []
    if len(rows) <= max_points:
        return rows
    step = max(1, math.ceil(len(rows) / max_points))
    sampled = rows[::step]
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return sampled[:max_points]


class FastF1Provider:
    """Thin adapter around FastF1, imported only for batch jobs."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def load_session(self, request: FastF1ImportRequest) -> FastF1LoadedSession:
        try:
            import fastf1
        except ImportError as exc:
            raise RuntimeError("Install fastf1 to run FastF1 post-session imports") from exc

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(str(self.cache_dir))

        session = fastf1.get_session(request.year, request.event, request.session_name)
        session.load(laps=True, telemetry=request.include_telemetry, weather=True, messages=True)

        event_name = str(_get_nested(session, "event", "EventName") or request.event)
        session_key = f"fastf1:{request.year}:{_slug(event_name)}:{_slug(request.session_name)}"
        laps = _records_from_frame(getattr(session, "laps", None))
        weather = _records_from_frame(getattr(session, "weather_data", None))
        race_control = _records_from_frame(getattr(session, "race_control_messages", None))
        telemetry_laps = self._load_telemetry_laps(session, request)
        return FastF1LoadedSession(
            year=request.year,
            event_name=event_name,
            session_name=request.session_name,
            session_key=session_key,
            laps=laps,
            weather=weather,
            race_control=race_control,
            telemetry_laps=telemetry_laps,
        )

    def _load_telemetry_laps(self, session: Any, request: FastF1ImportRequest) -> list[FastF1LapTelemetry]:
        if not request.include_telemetry:
            return []
        laps_frame = getattr(session, "laps", None)
        if laps_frame is None:
            return []
        drivers = request.drivers or tuple(str(driver) for driver in getattr(session, "drivers", ())[:2])
        telemetry_laps: list[FastF1LapTelemetry] = []
        for driver in drivers:
            try:
                driver_laps = laps_frame.pick_drivers(driver)
            except AttributeError:
                driver_laps = laps_frame[laps_frame["Driver"].astype(str).eq(str(driver))]
            records = _records_from_frame(driver_laps)
            records = [row for row in records if _finite_float(row.get("LapTimeSeconds")) is not None]
            records.sort(key=lambda row: float(row["LapTimeSeconds"]))
            for row in records[: max(1, request.telemetry_laps_per_driver)]:
                try:
                    lap = driver_laps[driver_laps["LapNumber"].astype(int).eq(int(row["LapNumber"]))].iloc[0]
                    telemetry = lap.get_telemetry()
                    if "Distance" not in getattr(telemetry, "columns", ()):
                        telemetry = telemetry.add_distance()
                except Exception:
                    continue
                telemetry_laps.append(
                    FastF1LapTelemetry(
                        driver=str(row.get("Driver") or driver),
                        lap_number=int(float(row.get("LapNumber") or 0)),
                        lap_time_seconds=_finite_float(row.get("LapTimeSeconds")),
                        telemetry=_records_from_frame(telemetry),
                    )
                )
        return telemetry_laps


def request_from_payload(payload: JsonObject) -> FastF1ImportRequest:
    year = _required_int(payload.get("year"), "year")
    event = payload.get("event", payload.get("round", payload.get("round_number")))
    if event is None or str(event).strip() == "":
        raise ValueError("event or round is required")
    drivers = payload.get("drivers") or ()
    if isinstance(drivers, str):
        drivers = tuple(item.strip() for item in drivers.split(",") if item.strip())
    elif isinstance(drivers, Sequence):
        drivers = tuple(str(item).strip() for item in drivers if str(item).strip())
    else:
        raise ValueError("drivers must be a comma-separated string or list")
    session_name = str(payload.get("session_name", payload.get("session", "R"))).strip() or "R"
    output_format = str(payload.get("output_format", "parquet")).lower().strip()
    return FastF1ImportRequest(
        year=year,
        event=int(event) if isinstance(event, int) or str(event).isdigit() else str(event),
        session_name=session_name,
        drivers=drivers,
        include_telemetry=bool(payload.get("include_telemetry", True)),
        telemetry_laps_per_driver=max(1, _optional_int(payload.get("telemetry_laps_per_driver")) or 1),
        distance_step_meters=max(1.0, _optional_float(payload.get("distance_step_meters")) or DEFAULT_DISTANCE_STEP_METERS),
        output_format=output_format or "parquet",
    )


def resample_telemetry(
    telemetry: Sequence[JsonObject],
    *,
    distance_step_meters: float = DEFAULT_DISTANCE_STEP_METERS,
    channels: Sequence[str] = DEFAULT_TELEMETRY_CHANNELS,
) -> list[JsonObject]:
    rows = _distance_sorted_rows(telemetry)
    if not rows:
        return []
    max_distance = rows[-1]["Distance"]
    target_distances = _distance_grid(max_distance, distance_step_meters)
    by_channel = {channel: [row.get(channel) for row in rows] for channel in channels}
    distances = [row["Distance"] for row in rows]
    resampled: list[JsonObject] = []
    for distance in target_distances:
        row: JsonObject = {"Distance": round(distance, 6)}
        for channel, values in by_channel.items():
            value = _interpolate(distances, values, distance)
            if value is not None:
                row[channel] = value
        speed = _finite_float(row.get("Speed"))
        if speed is not None and speed > 0:
            row["SecondsPerMeter"] = 3.6 / speed
        resampled.append(row)
    return resampled


def build_centerline(telemetry: Sequence[JsonObject], *, smoothing_window: int = 5) -> list[JsonObject]:
    positioned = [
        {
            "Distance": _finite_float(row.get("Distance")),
            "X": _finite_float(row.get("X")),
            "Y": _finite_float(row.get("Y")),
            "Z": _finite_float(row.get("Z")) or 0.0,
        }
        for row in telemetry
    ]
    positioned = [row for row in positioned if row["Distance"] is not None and row["X"] is not None and row["Y"] is not None]
    positioned.sort(key=lambda row: float(row["Distance"]))
    if not positioned:
        return []
    smoothed = _smooth_position_rows(positioned, smoothing_window=smoothing_window)
    total = max(1e-9, float(smoothed[-1]["Distance"]))
    return [
        {
            "Distance": round(float(row["Distance"]), 6),
            "Progress": round(float(row["Distance"]) / total, 9),
            "X": round(float(row["X"]), 6),
            "Y": round(float(row["Y"]), 6),
            "Z": round(float(row["Z"]), 6),
        }
        for row in smoothed
    ]


def build_telemetry_delta(lap_a: Sequence[JsonObject], lap_b: Sequence[JsonObject]) -> list[JsonObject]:
    a_by_distance = {round(float(row["Distance"]), 6): row for row in lap_a if _finite_float(row.get("Distance")) is not None}
    b_by_distance = {round(float(row["Distance"]), 6): row for row in lap_b if _finite_float(row.get("Distance")) is not None}
    distances = sorted(set(a_by_distance).intersection(b_by_distance))
    cumulative_a = 0.0
    cumulative_b = 0.0
    previous_distance: float | None = None
    rows: list[JsonObject] = []
    for distance in distances:
        row_a = a_by_distance[distance]
        row_b = b_by_distance[distance]
        if previous_distance is not None:
            segment = max(0.0, distance - previous_distance)
            cumulative_a += segment * (_finite_float(row_a.get("SecondsPerMeter")) or _seconds_per_meter(row_a.get("Speed")))
            cumulative_b += segment * (_finite_float(row_b.get("SecondsPerMeter")) or _seconds_per_meter(row_b.get("Speed")))
        rows.append(
            {
                "Distance": distance,
                "TimeA": round(cumulative_a, 6),
                "TimeB": round(cumulative_b, 6),
                "DeltaSeconds": round(cumulative_a - cumulative_b, 6),
                "SpeedA": _finite_float(row_a.get("Speed")),
                "SpeedB": _finite_float(row_b.get("Speed")),
                "ThrottleA": _finite_float(row_a.get("Throttle")),
                "ThrottleB": _finite_float(row_b.get("Throttle")),
                "BrakeA": _finite_float(row_a.get("Brake")),
                "BrakeB": _finite_float(row_b.get("Brake")),
                "GearA": _finite_float(row_a.get("nGear")),
                "GearB": _finite_float(row_b.get("nGear")),
                "DRSA": _finite_float(row_a.get("DRS")),
                "DRSB": _finite_float(row_b.get("DRS")),
            }
        )
        previous_distance = distance
    return rows


def build_corner_metrics(
    telemetry: Sequence[JsonObject],
    *,
    driver: str,
    lap_number: int,
    min_prominence_kmh: float = 5.0,
) -> list[JsonObject]:
    rows = _distance_sorted_rows(telemetry)
    if len(rows) < 3:
        return []

    candidates: list[int] = []
    for idx in range(1, len(rows) - 1):
        previous_speed = _finite_float(rows[idx - 1].get("Speed"))
        speed = _finite_float(rows[idx].get("Speed"))
        next_speed = _finite_float(rows[idx + 1].get("Speed"))
        if previous_speed is None or speed is None or next_speed is None:
            continue
        if speed <= previous_speed and speed <= next_speed and max(previous_speed, next_speed) - speed >= min_prominence_kmh:
            candidates.append(idx)

    metrics: list[JsonObject] = []
    for corner_index, apex_idx in enumerate(candidates, start=1):
        start_idx = max(0, apex_idx - 1)
        end_idx = min(len(rows) - 1, apex_idx + 1)
        window = rows[start_idx : end_idx + 1]
        if len(window) < 2:
            continue
        entry = window[0]
        apex = rows[apex_idx]
        exit_row = window[-1]
        entry_distance = _finite_float(entry.get("Distance"))
        apex_distance = _finite_float(apex.get("Distance"))
        exit_distance = _finite_float(exit_row.get("Distance"))
        if entry_distance is None or apex_distance is None or exit_distance is None:
            continue

        braking_rows = [row for row in window if _truthy_channel(row.get("Brake"))]
        throttle_reapply = next(
            (
                row
                for row in window
                if (_finite_float(row.get("Distance")) or 0.0) >= apex_distance
                and (_finite_float(row.get("Throttle")) or 0.0) >= 70.0
            ),
            None,
        )
        full_throttle_rows = [row for row in window if (_finite_float(row.get("Throttle")) or 0.0) >= 95.0]

        metrics.append(
            {
                "Driver": driver,
                "LapNumber": lap_number,
                "CornerIndex": corner_index,
                "Method": "distance_window_local_speed_minima",
                "EntryDistance": _metric_round(entry_distance),
                "ApexDistance": _metric_round(apex_distance),
                "ExitDistance": _metric_round(exit_distance),
                "EntrySpeed": _metric_round(entry.get("Speed")),
                "MinimumSpeed": _metric_round(apex.get("Speed")),
                "ExitSpeed": _metric_round(exit_row.get("Speed")),
                "BrakeStartDistance": _metric_round(braking_rows[0].get("Distance")) if braking_rows else None,
                "ThrottleReapplicationDistance": (
                    _metric_round(throttle_reapply.get("Distance")) if throttle_reapply else None
                ),
                "FullThrottlePercent": _metric_round(100.0 * len(full_throttle_rows) / len(window)),
                "BrakingDurationSeconds": _metric_round(_channel_time_seconds(window, "Brake")),
                "CornerTimeSeconds": _metric_round(_window_time_seconds(window)),
                "ExitAccelerationKmhPer100m": _exit_acceleration(entry, exit_row),
            }
        )
    return metrics


def _distance_sorted_rows(telemetry: Sequence[JsonObject]) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for row in telemetry:
        distance = _finite_float(row.get("Distance"))
        if distance is None:
            continue
        cleaned: JsonObject = {"Distance": distance}
        for key, value in row.items():
            numeric = _finite_float(value)
            if numeric is not None:
                cleaned[str(key)] = numeric
            elif isinstance(value, bool):
                cleaned[str(key)] = 1.0 if value else 0.0
        rows.append(cleaned)
    rows.sort(key=lambda row: row["Distance"])
    deduped: list[JsonObject] = []
    for row in rows:
        if deduped and math.isclose(float(deduped[-1]["Distance"]), float(row["Distance"]), abs_tol=1e-9):
            deduped[-1] = row
        else:
            deduped.append(row)
    return deduped


def _distance_grid(max_distance: float, step: float) -> list[float]:
    if max_distance <= 0:
        return [0.0]
    distance = 0.0
    values: list[float] = []
    while distance < max_distance:
        values.append(distance)
        distance += step
    if not values or not math.isclose(values[-1], max_distance, rel_tol=0.0, abs_tol=1e-6):
        values.append(max_distance)
    return values


def _interpolate(distances: Sequence[float], values: Sequence[Any], target: float) -> float | None:
    numeric = [_finite_float(value) for value in values]
    if not distances or not numeric:
        return None
    idx = bisect_right(distances, target)
    if idx <= 0:
        return numeric[0]
    if idx >= len(distances):
        return numeric[-1]
    left_x = distances[idx - 1]
    right_x = distances[idx]
    left_y = numeric[idx - 1]
    right_y = numeric[idx]
    if left_y is None or right_y is None:
        return left_y if left_y is not None else right_y
    if math.isclose(right_x, left_x):
        return left_y
    ratio = (target - left_x) / (right_x - left_x)
    return round(left_y + (right_y - left_y) * ratio, 6)


def _smooth_position_rows(rows: Sequence[JsonObject], *, smoothing_window: int) -> list[JsonObject]:
    radius = max(0, smoothing_window // 2)
    out: list[JsonObject] = []
    for idx, row in enumerate(rows):
        start = max(0, idx - radius)
        end = min(len(rows), idx + radius + 1)
        window = rows[start:end]
        out.append(
            {
                "Distance": row["Distance"],
                "X": sum(float(item["X"]) for item in window) / len(window),
                "Y": sum(float(item["Y"]) for item in window) / len(window),
                "Z": sum(float(item.get("Z") or 0.0) for item in window) / len(window),
            }
        )
    return out


def _records_from_frame(frame: Any) -> list[JsonObject]:
    if frame is None:
        return []
    try:
        records = frame.copy().to_dict("records")
    except AttributeError:
        return []
    return [_normalize_record(record) for record in records]


def _normalize_record(record: JsonObject) -> JsonObject:
    normalized: JsonObject = {}
    for raw_key, value in record.items():
        key = str(raw_key)
        safe_value = _json_safe(value)
        normalized[key] = safe_value
        if key.endswith("Time") or key in {"LapTime", "Sector1Time", "Sector2Time", "Sector3Time"}:
            seconds = _timedelta_seconds(value) or _duration_string_seconds(safe_value)
            if seconds is not None:
                normalized[f"{key}Seconds"] = seconds
    return normalized


def _timedelta_seconds(value: Any) -> float | None:
    if hasattr(value, "total_seconds"):
        return round(float(value.total_seconds()), 6)
    return None


def _duration_string_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or value in {"", "NaT", "nan", "None"}:
        return None
    match = re.fullmatch(
        r"P(?:(?P<days>\d+(?:\.\d+)?)D)?T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?",
        value,
    )
    if match:
        days = float(match.group("days") or 0)
        hours = float(match.group("hours") or 0)
        minutes = float(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0)
        return round(days * 86_400 + hours * 3_600 + minutes * 60 + seconds, 6)
    clock_match = re.fullmatch(r"(?:(?P<days>\d+) days? )?(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)", value)
    if clock_match:
        days = float(clock_match.group("days") or 0)
        hours = float(clock_match.group("hours") or 0)
        minutes = float(clock_match.group("minutes") or 0)
        seconds = float(clock_match.group("seconds") or 0)
        return round(days * 86_400 + hours * 3_600 + minutes * 60 + seconds, 6)
    return None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "total_seconds"):
        return round(float(value.total_seconds()), 6)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _first_lap_with_position(
    telemetry_laps: Sequence[tuple[FastF1LapTelemetry, list[JsonObject]]],
) -> tuple[FastF1LapTelemetry, list[JsonObject]] | None:
    for lap, rows in telemetry_laps:
        if any(_finite_float(row.get("X")) is not None and _finite_float(row.get("Y")) is not None for row in rows):
            return lap, rows
    return None


def _session_partition(year: int, event_name: str, session_name: str) -> Path:
    return Path(f"year={year}") / f"event={_slug(event_name)}" / f"session={_slug(session_name)}"


def _session_key_from_partition(parts: Sequence[str]) -> str | None:
    year: str | None = None
    event: str | None = None
    session: str | None = None
    for part in parts:
        if part.startswith("year="):
            year = part.split("=", 1)[1]
        elif part.startswith("event="):
            event = part.split("=", 1)[1]
        elif part.startswith("session="):
            session = part.split("=", 1)[1]
    if not year or not event or not session:
        return None
    return f"fastf1:{year}:{event}:{session}"


def _columns_from_rows(rows: Sequence[JsonObject]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown"


def _required_int(value: Any, name: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError(f"{name} is required")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _seconds_per_meter(speed_kmh: Any) -> float:
    speed = _finite_float(speed_kmh)
    if speed is None or speed <= 0:
        return 0.0
    return 3.6 / speed


def _window_time_seconds(rows: Sequence[JsonObject]) -> float:
    total = 0.0
    for first, second in zip(rows, rows[1:]):
        distance = _segment_distance(first, second)
        if distance is None:
            continue
        seconds_per_meter = _average_seconds_per_meter(first, second)
        if seconds_per_meter is not None:
            total += distance * seconds_per_meter
    return total


def _channel_time_seconds(rows: Sequence[JsonObject], channel: str) -> float:
    total = 0.0
    for first, second in zip(rows, rows[1:]):
        if not (_truthy_channel(first.get(channel)) or _truthy_channel(second.get(channel))):
            continue
        distance = _segment_distance(first, second)
        if distance is None:
            continue
        seconds_per_meter = _average_seconds_per_meter(first, second)
        if seconds_per_meter is not None:
            total += distance * seconds_per_meter
    return total


def _segment_distance(first: JsonObject, second: JsonObject) -> float | None:
    first_distance = _finite_float(first.get("Distance"))
    second_distance = _finite_float(second.get("Distance"))
    if first_distance is None or second_distance is None:
        return None
    return max(0.0, second_distance - first_distance)


def _average_seconds_per_meter(first: JsonObject, second: JsonObject) -> float | None:
    first_value = _finite_float(first.get("SecondsPerMeter")) or _seconds_per_meter(first.get("Speed"))
    second_value = _finite_float(second.get("SecondsPerMeter")) or _seconds_per_meter(second.get("Speed"))
    values = [value for value in (first_value, second_value) if value > 0]
    if not values:
        return None
    return sum(values) / len(values)


def _truthy_channel(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    numeric = _finite_float(value)
    if numeric is not None:
        return numeric > 0
    return str(value).strip().lower() in {"true", "yes", "on"}


def _exit_acceleration(entry: JsonObject, exit_row: JsonObject) -> float | None:
    entry_speed = _finite_float(entry.get("Speed"))
    exit_speed = _finite_float(exit_row.get("Speed"))
    distance = _segment_distance(entry, exit_row)
    if entry_speed is None or exit_speed is None or distance is None or distance <= 0:
        return None
    return _metric_round((exit_speed - entry_speed) / distance * 100.0)


def _metric_round(value: Any) -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    return round(number, 6)


def _get_nested(obj: Any, attr: str, key: str) -> Any:
    container = getattr(obj, attr, None)
    if container is None:
        return None
    try:
        return container[key]
    except Exception:
        return getattr(container, key, None)
