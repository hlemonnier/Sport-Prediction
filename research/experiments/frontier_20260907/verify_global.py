"""Exercise an observed Italian GP prefix through both public prediction paths."""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from f1_platform.fastf1_analysis import (
    FastF1LoadedSession,
    fastf1_runtime_events,
    _records_from_frame,
)
from f1_platform.reducer import F1StateReducer
from f1_prediction_service.app import create_app
from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.orchestration.prediction import run_prediction
import packages.f1.models.live_race.evaluate as evaluation
from packages.f1.models.live_race.next_lap import (
    forecast_laps,
    SECONDS_COLUMNS,
    MODEL_ID,
)

ROOT = Path(__file__).resolve().parents[3]


def main():
    sources = json.loads(
        (
            ROOT
            / "artifacts/research/frontier_20260907/live/corrected_input_contract/results.json"
        ).read_text()
    )["input_manifest"]
    source = next(s for s in sources if s["event_key"] == 202613)
    path = ROOT / source["path"]
    raw = pd.read_csv(path)
    prefix = raw.loc[raw.Time <= 7800].copy()
    expected = forecast_laps(prefix, 202613).frame.groupby("driver_id").tail(1)
    references = {row["driver_id"]: row for row in expected.to_dict("records")}
    native = prefix.copy()
    for column in SECONDS_COLUMNS:
        native[column] = pd.to_timedelta(native[column], unit="s")
    loaded = FastF1LoadedSession(
        year=2026,
        event_name="Italian Grand Prix",
        session_name="R",
        session_key="verification:202613",
        laps=_records_from_frame(native),
        weather=[],
        race_control=[],
        telemetry_laps=[],
    )
    reducer = F1StateReducer(loaded.session_key, source="retrospective_verification")
    for event in fastf1_runtime_events(loaded):
        reducer.ingest(event)
    body = {"snapshot": reducer.snapshot().to_dict()}
    started = time.perf_counter()
    with TestClient(create_app()) as client:
        response = client.post("/api/f1/predict/next-lap", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    api_seconds = time.perf_counter() - started
    fallback = []
    for row in payload["predictions"]:
        point = row["lap_time_forecast"]
        reference = references.get(str(row["driver_number"]))
        if reference is None:
            assert point["status"] != "available"
            fallback.append(row["driver_number"])
        else:
            assert (
                point["model_id"] == MODEL_ID
                and point["seconds"] == reference["forecast_seconds"]
            )
    evaluation.ARIMA = (
        None  # Optional comparator only; does not enter model predictions.
    )
    config = PredictionConfig(
        source="local",
        mode="race",
        year=2026,
        round_number=13,
        train_seasons=[],
        include_standings=False,
        cache_dir=None,
        meeting_name=None,
        country_name=None,
        weekends_dir=None,
        f1_mode="live",
        f1_live_source="local",
        f1_live_replay_path=str(path),
        f1_live_replay_cutoff_time_seconds=7800.0,
    )
    result = run_prediction(config)
    for row in result.table.to_dict("records"):
        reference = references.get(row["driver_id"])
        if reference is None:
            assert row["next_lap_point_model"] != MODEL_ID
        else:
            assert (
                row["next_lap_point_model"] == MODEL_ID
                and row["next_lap_mean"] == reference["forecast_seconds"]
            )
            assert np.isnan(row["next_lap_pi90_low"])
    report = {
        "status": "passed",
        "event_key": 202613,
        "as_of_session_seconds": 7800,
        "raw_source_sha256": source["sha256"],
        "prefix_rows": len(prefix),
        "canonical_driver_rows": len(result.table),
        "http_driver_rows": len(payload["predictions"]),
        "frontier_driver_forecasts": len(references),
        "fallback_driver_numbers": fallback,
        "http_status": response.status_code,
        "http_seconds": api_seconds,
        "model_id": MODEL_ID,
        "max_forecast_difference_seconds": 0.0,
        "http_request_bytes": len(json.dumps(body)),
        "scope": "Retrospective prefix; real FastF1 hydration, reducer, HTTP endpoint and canonical run_prediction.",
        "point_forecasts": [
            {"driver": r["driver_number"], **r["lap_time_forecast"]}
            for r in payload["predictions"]
        ],
    }
    out = (
        ROOT
        / "artifacts/research/frontier_20260907/integration/global_prediction_verification.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "point_forecasts"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
