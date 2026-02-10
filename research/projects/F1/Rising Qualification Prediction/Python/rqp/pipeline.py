"""Dataset ingestion pipeline for FastF1/OpenF1 historical data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from .providers import BaseProvider, FastF1Provider, OpenF1Provider


@dataclass
class PipelineConfig:
    sources: List[str]
    years: List[int]
    output_dir: str
    cache_dir: Optional[str]
    max_rounds: Optional[int] = None


@dataclass
class PipelineResult:
    dataset: pd.DataFrame
    coverage: pd.DataFrame
    dataset_csv_path: str
    dataset_parquet_path: Optional[str]
    coverage_csv_path: str
    coverage_parquet_path: Optional[str]
    notes: List[str]


def _source_cache_dir(cache_root: Optional[str], source: str) -> Optional[str]:
    if not cache_root:
        return None
    cache_dir = Path(cache_root) / source
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def _build_provider(source: str, cache_root: Optional[str]) -> BaseProvider:
    normalized = source.lower().strip()
    cache_dir = _source_cache_dir(cache_root, normalized)
    if normalized == "fastf1":
        return FastF1Provider(cache_dir=cache_dir)
    if normalized == "openf1":
        return OpenF1Provider(cache_dir=cache_dir)
    raise ValueError(f"Unsupported source: {source}")


def _fetch_frame(
    fetch_fn: Callable[[], pd.DataFrame],
    context: str,
    notes: List[str],
) -> pd.DataFrame:
    try:
        frame = fetch_fn()
    except (Exception, SystemExit) as exc:
        notes.append(f"{context}: {exc}")
        return pd.DataFrame()
    if frame is None:
        return pd.DataFrame()
    return frame


def _normalize_driver_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "driver_id" not in frame.columns:
        return pd.DataFrame(columns=["driver_id", "driver_name"])
    normalized = frame.copy()
    normalized["driver_id"] = normalized["driver_id"].astype(str)
    if "driver_name" not in normalized.columns:
        normalized["driver_name"] = normalized["driver_id"]
    normalized["driver_name"] = (
        normalized["driver_name"].fillna(normalized["driver_id"]).astype(str)
    )
    return normalized


def _merge_round_data(
    source: str,
    year: int,
    round_number: int,
    event_name: str,
    fp: pd.DataFrame,
    qualifying: pd.DataFrame,
    race: pd.DataFrame,
    standings: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        _normalize_driver_frame(fp),
        _normalize_driver_frame(qualifying),
        _normalize_driver_frame(race),
        _normalize_driver_frame(standings),
    ]
    base_parts = [f[["driver_id", "driver_name"]] for f in frames if not f.empty]
    if not base_parts:
        return pd.DataFrame()

    merged = pd.concat(base_parts, ignore_index=True)
    merged = merged.dropna(subset=["driver_id"])
    merged = merged.drop_duplicates(subset=["driver_id"], keep="last")

    fp_norm = _normalize_driver_frame(fp)
    if not fp_norm.empty:
        fp_cols = [c for c in fp_norm.columns if c != "driver_name"]
        merged = merged.merge(
            fp_norm[fp_cols].drop_duplicates(subset=["driver_id"], keep="last"),
            on="driver_id",
            how="left",
        )

    qualy_norm = _normalize_driver_frame(qualifying)
    if not qualy_norm.empty:
        qualy_cols = [c for c in ["driver_id", "position", "q3_time"] if c in qualy_norm.columns]
        if len(qualy_cols) > 1:
            qualy = qualy_norm[qualy_cols].drop_duplicates(subset=["driver_id"], keep="last")
            qualy = qualy.rename(columns={"position": "qualy_position", "q3_time": "qualy_q3_time"})
            merged = merged.merge(qualy, on="driver_id", how="left")

    race_norm = _normalize_driver_frame(race)
    if not race_norm.empty:
        race_cols = [c for c in ["driver_id", "position"] if c in race_norm.columns]
        if len(race_cols) > 1:
            race_data = race_norm[race_cols].drop_duplicates(subset=["driver_id"], keep="last")
            race_data = race_data.rename(columns={"position": "race_position"})
            merged = merged.merge(race_data, on="driver_id", how="left")

    standings_norm = _normalize_driver_frame(standings)
    if not standings_norm.empty and "position_start" in standings_norm.columns:
        standing_data = standings_norm[["driver_id", "position_start"]]
        standing_data = standing_data.drop_duplicates(subset=["driver_id"], keep="last")
        standing_data = standing_data.rename(
            columns={"position_start": "standings_position_start"},
        )
        merged = merged.merge(standing_data, on="driver_id", how="left")

    for col in [
        "qualy_position",
        "qualy_q3_time",
        "race_position",
        "standings_position_start",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged.insert(0, "event_key", (year * 100) + round_number)
    merged.insert(0, "round_number", round_number)
    merged.insert(0, "year", year)
    merged.insert(0, "event_name", event_name)
    merged.insert(0, "source", source)

    sort_cols = [
        col
        for col in ["source", "year", "round_number", "qualy_position", "race_position", "driver_id"]
        if col in merged.columns
    ]
    return merged.sort_values(sort_cols).reset_index(drop=True)


def _extract_event_name(round_meta: dict[str, object], round_number: int) -> str:
    event_name = (
        round_meta.get("event_name")
        or round_meta.get("meeting_name")
        or round_meta.get("country_name")
    )
    if event_name is None:
        return f"Round {round_number}"
    return str(event_name)


def _collect_round(
    provider: BaseProvider,
    source: str,
    year: int,
    round_meta: dict[str, object],
    notes: List[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    round_number = int(round_meta["round_number"])
    event_name = _extract_event_name(round_meta, round_number)

    fp = _fetch_frame(
        lambda: provider.get_fp_features(year, round_number),
        f"{source} {year} round {round_number} FP fetch failed",
        notes,
    )
    qualifying = _fetch_frame(
        lambda: provider.get_qualifying_results(year, round_number),
        f"{source} {year} round {round_number} qualifying fetch failed",
        notes,
    )
    race = _fetch_frame(
        lambda: provider.get_race_results(year, round_number),
        f"{source} {year} round {round_number} race fetch failed",
        notes,
    )
    def fetch_standings() -> pd.DataFrame:
        standing_frame = provider.get_standings(year, round_number)
        if standing_frame is None:
            return pd.DataFrame()
        return standing_frame

    standings = _fetch_frame(
        fetch_standings,
        f"{source} {year} round {round_number} standings fetch failed",
        notes,
    )

    merged = _merge_round_data(
        source=source,
        year=year,
        round_number=round_number,
        event_name=event_name,
        fp=fp,
        qualifying=qualifying,
        race=race,
        standings=standings,
    )

    coverage = {
        "source": source,
        "year": year,
        "round_number": round_number,
        "event_name": event_name,
        "drivers": int(merged["driver_id"].nunique()) if not merged.empty else 0,
        "fp_available": int(not fp.empty),
        "qualifying_available": int(not qualifying.empty),
        "race_available": int(not race.empty),
        "standings_available": int(not standings.empty),
    }
    return merged, coverage


def _write_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    basename: str,
    notes: List[str],
) -> tuple[str, Optional[str]]:
    csv_path = output_dir / f"{basename}.csv"
    frame.to_csv(csv_path, index=False)

    parquet_path = output_dir / f"{basename}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        parquet_output: Optional[str] = str(parquet_path)
    except Exception as exc:
        notes.append(f"{basename}.parquet non ecrit ({exc}); CSV disponible.")
        parquet_output = None

    return str(csv_path), parquet_output


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    notes: List[str] = []
    all_rows: List[pd.DataFrame] = []
    coverage_rows: List[dict[str, object]] = []

    for source in config.sources:
        normalized = source.lower().strip()
        try:
            provider = _build_provider(normalized, config.cache_dir)
        except (Exception, SystemExit) as exc:
            notes.append(f"{normalized}: provider indisponible ({exc}).")
            continue

        for year in config.years:
            try:
                rounds = provider.list_rounds(year)
            except (Exception, SystemExit) as exc:
                notes.append(f"{normalized} {year}: impossible de lister les rounds ({exc}).")
                continue

            rounds_sorted = sorted(rounds, key=lambda r: int(r.get("round_number", 0)))
            if config.max_rounds is not None:
                rounds_sorted = rounds_sorted[: config.max_rounds]

            for round_meta in rounds_sorted:
                merged, coverage = _collect_round(
                    provider=provider,
                    source=normalized,
                    year=year,
                    round_meta=round_meta,
                    notes=notes,
                )
                coverage_rows.append(coverage)
                if not merged.empty:
                    all_rows.append(merged)

    if all_rows:
        dataset = pd.concat(all_rows, ignore_index=True)
    else:
        dataset = pd.DataFrame(
            columns=[
                "source",
                "event_name",
                "year",
                "round_number",
                "event_key",
                "driver_id",
                "driver_name",
            ],
        )

    coverage = pd.DataFrame(coverage_rows)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv_path, dataset_parquet_path = _write_outputs(
        frame=dataset,
        output_dir=output_dir,
        basename="f1_dataset",
        notes=notes,
    )
    coverage_csv_path, coverage_parquet_path = _write_outputs(
        frame=coverage,
        output_dir=output_dir,
        basename="f1_coverage",
        notes=notes,
    )

    return PipelineResult(
        dataset=dataset,
        coverage=coverage,
        dataset_csv_path=dataset_csv_path,
        dataset_parquet_path=dataset_parquet_path,
        coverage_csv_path=coverage_csv_path,
        coverage_parquet_path=coverage_parquet_path,
        notes=notes,
    )
