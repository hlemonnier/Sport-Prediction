"""FastAPI app for F1 prediction/model service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import FEATURES_VERSION, MODEL_VERSION, TARGET_MODELS, predict_from_snapshot


def create_app() -> FastAPI:
    app = FastAPI(
        title="F1 Prediction Service",
        version="0.1.0",
        description="Model-service boundary for platform F1 prediction snapshots.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "f1-prediction-service",
            "modelVersion": MODEL_VERSION,
            "featuresVersion": FEATURES_VERSION,
            "supportedModels": {
                kind: details["model_version"]
                for kind, details in TARGET_MODELS.items()
            },
        }

    @app.post("/api/f1/predict/race")
    async def predict_race(body: dict[str, Any]) -> dict[str, Any]:
        return _predict(body, "race")

    @app.post("/api/f1/predict/qualifying")
    async def predict_qualifying(body: dict[str, Any]) -> dict[str, Any]:
        return _predict(body, "qualifying")

    @app.post("/api/f1/predict/next-lap")
    async def predict_next_lap(body: dict[str, Any]) -> dict[str, Any]:
        return _predict(body, "next-lap")

    @app.post("/api/f1/predict/strategy")
    async def predict_strategy(body: dict[str, Any]) -> dict[str, Any]:
        return _predict(body, "strategy")

    return app


def _predict(body: dict[str, Any], prediction_kind: str) -> dict[str, Any]:
    snapshot = body.get("snapshot")
    if not isinstance(snapshot, dict):
        raise HTTPException(status_code=400, detail="snapshot JSON object is required")
    try:
        return predict_from_snapshot(snapshot, prediction_kind=prediction_kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc


app = create_app()
