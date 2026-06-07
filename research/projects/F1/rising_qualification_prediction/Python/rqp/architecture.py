"""F1 prediction system architecture metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class F1PredictionStage:
    key: str
    name: str
    status: str
    predicts: str
    inputs: tuple[str, ...]
    output_contract: tuple[str, ...]
    next_stage: str | None = None


F1_MODEL_ARCHITECTURE: tuple[F1PredictionStage, ...] = (
    F1PredictionStage(
        key="pre_quali",
        name="Pre-Quali Model",
        status="active",
        predicts="qualifying order before qualifying is complete",
        inputs=(
            "free-practice pace features",
            "team/driver form",
            "circuit-card priors",
            "weather priors when available",
        ),
        output_contract=(
            "qualy_pred_position",
            "qualy_pred_rank",
            "qualy_pred_top3_proba",
            "qualy_pred_top10_proba",
        ),
        next_stage="pre_race",
    ),
    F1PredictionStage(
        key="pre_race",
        name="Pre-Race Model",
        status="active",
        predicts="race order from grid/quali plus race features",
        inputs=(
            "official starting grid when available",
            "qualifying result when available",
            "Pre-Quali predicted grid before qualifying",
            "free-practice race pace features",
            "strategy, reliability, circuit, and weather priors",
        ),
        output_contract=(
            "predicted race order",
            "win/top3/top10 probabilities",
            "grid_source",
            "race_stochastic_* diagnostics",
        ),
        next_stage="live_race",
    ),
    F1PredictionStage(
        key="live_race",
        name="Live Race Model",
        status="planned",
        predicts="live race order and strategy adjustments during the race",
        inputs=(
            "starting grid",
            "lap times",
            "tyre age and compound",
            "pit stops",
            "weather/live-event state",
        ),
        output_contract=(
            "updated finishing-order probabilities",
            "pit/strategy recommendations",
            "lap-by-lap trace",
        ),
        next_stage=None,
    ),
    F1PredictionStage(
        key="ultimate_lap_time",
        name="Ultimate Lap-Time Model",
        status="planned",
        predicts="theoretical best lap pace for the upcoming weekend",
        inputs=(
            "car/team pace state",
            "circuit-card demands",
            "weather/track evolution",
            "session context",
        ),
        output_contract=(
            "theoretical best lap",
            "pace envelope",
            "driver/team limiting factors",
        ),
        next_stage=None,
    ),
)


def architecture_payload() -> dict[str, Any]:
    return {
        "name": "F1 Prediction System",
        "version": "f1_prediction_architecture_v1",
        "stages": [asdict(stage) for stage in F1_MODEL_ARCHITECTURE],
        "active_flow": [
            "pre_quali",
            "pre_race",
        ],
        "pre_quali_to_race_contract": {
            "enabled": True,
            "description": (
                "Before qualifying, the race model first computes the Pre-Quali output, "
                "uses qualy_pred_rank as a provisional grid, and labels those rows with "
                "grid_source=predicted_qualifying_grid."
            ),
        },
        "weather_scenarios": {
            "enabled": True,
            "outputs": ["base_no_weather", "weather_integrated"],
            "description": (
                "Pre-Quali and Pre-Race predictions expose both a weather-neutral view "
                "and a weather-integrated view using available weather uncertainty priors."
            ),
        },
    }
