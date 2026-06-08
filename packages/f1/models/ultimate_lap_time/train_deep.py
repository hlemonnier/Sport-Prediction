"""Training scaffolding for the optional Ultimate Lap-Time TCN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.deep import (
    OUTPUT_COLUMNS,
    DistanceTelemetryTCN,
    DistanceTelemetryTCNConfig,
    resolve_device,
    seed_torch,
    torch,
    torch_available,
    ultimate_lap_time_deep_loss,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    UltimateLapTelemetryBatch,
    UltimateLapTelemetryExample,
)


@dataclass(frozen=True)
class DeepTrainingConfig:
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    hidden_channels: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.10
    static_hidden_dim: int = 32
    head_hidden_dim: int = 64
    lambda_sector: float = 0.20
    lambda_rank: float = 0.02
    lambda_mono: float = 1.0


@dataclass
class DeepUltimateLapTimeModel:
    """Fitted PyTorch TCN plus preprocessing metadata."""

    network: Any
    architecture_config: DistanceTelemetryTCNConfig
    training_config: DeepTrainingConfig
    channel_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    device_used: str
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class DeepTrainingResult:
    status: str
    reason: str | None
    model: DeepUltimateLapTimeModel | None
    history: tuple[dict[str, float], ...] = ()

    @property
    def is_trained(self) -> bool:
        return self.status == "trained" and self.model is not None


def _numeric_static_feature_names(examples: Sequence[UltimateLapTelemetryExample]) -> tuple[str, ...]:
    names: set[str] = set()
    for example in examples:
        for key, value in example.static_features.items():
            if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)) and np.isfinite(float(value)):
                names.add(str(key))
    return tuple(sorted(names))


def _static_matrix(
    examples: Sequence[UltimateLapTelemetryExample],
    static_feature_names: Sequence[str],
) -> np.ndarray:
    if not static_feature_names:
        return np.empty((len(examples), 0), dtype=np.float32)
    values = np.zeros((len(examples), len(static_feature_names)), dtype=np.float32)
    for row_idx, example in enumerate(examples):
        for col_idx, name in enumerate(static_feature_names):
            raw = example.static_features.get(name, 0.0)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            values[row_idx, col_idx] = value if np.isfinite(value) else 0.0
    return values


def examples_to_deep_numpy(
    examples: Sequence[UltimateLapTelemetryExample],
    *,
    static_feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Convert examples to telemetry/static/target numpy tensors."""

    batch = UltimateLapTelemetryBatch.from_examples(examples)
    names = tuple(static_feature_names) if static_feature_names is not None else _numeric_static_feature_names(examples)
    return (
        batch.telemetry.astype(np.float32, copy=False),
        _static_matrix(examples, names),
        batch.target_matrix().astype(np.float32, copy=False),
        names,
    )


def train_ultimate_lap_time_deep(
    examples: Sequence[UltimateLapTelemetryExample],
    *,
    config: DeepTrainingConfig | None = None,
) -> DeepTrainingResult:
    """Train the optional PyTorch telemetry TCN, or return a clean skipped result."""

    cfg = config or DeepTrainingConfig()
    if not torch_available():
        return DeepTrainingResult(status="skipped", reason="PyTorch is not installed", model=None)
    if not examples:
        raise ValueError("examples must contain at least one telemetry example")

    telemetry_np, static_np, target_np, static_names = examples_to_deep_numpy(examples)
    if telemetry_np.shape[0] == 0:
        raise ValueError("examples must contain at least one telemetry example")
    seed_torch(int(cfg.seed))
    device_name = resolve_device(cfg.device)
    device = torch.device(device_name if device_name != "unavailable" else "cpu")

    architecture = DistanceTelemetryTCNConfig(
        input_channels=int(telemetry_np.shape[1]),
        distance_bins=int(telemetry_np.shape[2]),
        static_feature_dim=int(static_np.shape[1]),
        hidden_channels=int(cfg.hidden_channels),
        kernel_size=int(cfg.kernel_size),
        dilations=tuple(cfg.dilations),
        dropout=float(cfg.dropout),
        static_hidden_dim=int(cfg.static_hidden_dim),
        head_hidden_dim=int(cfg.head_hidden_dim),
        lambda_sector=float(cfg.lambda_sector),
        lambda_rank=float(cfg.lambda_rank),
        lambda_mono=float(cfg.lambda_mono),
    )
    network = DistanceTelemetryTCN(architecture).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))

    telemetry_t = torch.from_numpy(telemetry_np).to(device)
    static_t = torch.from_numpy(static_np).to(device)
    target_t = torch.from_numpy(target_np).to(device)
    indices = np.arange(telemetry_np.shape[0])
    history: list[dict[str, float]] = []

    max_epochs = int(max(1, cfg.epochs))
    batch_size = int(max(1, min(cfg.batch_size, len(indices))))
    for epoch in range(max_epochs):
        rng = np.random.default_rng(int(cfg.seed) + epoch)
        rng.shuffle(indices)
        batch_losses: list[float] = []
        network.train()
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_tensor = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = network(telemetry_t.index_select(0, batch_tensor), static_t.index_select(0, batch_tensor))
            loss = ultimate_lap_time_deep_loss(prediction, target_t.index_select(0, batch_tensor), architecture)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(batch_losses))})

    fitted = DeepUltimateLapTimeModel(
        network=network,
        architecture_config=architecture,
        training_config=cfg,
        channel_names=UltimateLapTelemetryBatch.from_examples(examples).channel_names,
        static_feature_names=static_names,
        device_used=device_name,
        history=tuple(history),
    )
    return DeepTrainingResult(status="trained", reason=None, model=fitted, history=tuple(history))


def predict_ultimate_lap_time_deep(
    model: DeepUltimateLapTimeModel,
    examples: Sequence[UltimateLapTelemetryExample],
) -> pd.DataFrame:
    """Predict lap and sector quantiles from a fitted deep model."""

    if not torch_available():
        raise RuntimeError("PyTorch is not installed")
    if not isinstance(model, DeepUltimateLapTimeModel):
        raise TypeError("model must be a DeepUltimateLapTimeModel")
    if not examples:
        return pd.DataFrame(columns=[*OUTPUT_COLUMNS, "model"])
    telemetry_np, static_np, _, _ = examples_to_deep_numpy(examples, static_feature_names=model.static_feature_names)
    device = torch.device(model.device_used if model.device_used != "unavailable" else "cpu")
    network = model.network.to(device)
    network.eval()
    with torch.no_grad():
        prediction = network(torch.from_numpy(telemetry_np).to(device), torch.from_numpy(static_np).to(device))
    output = pd.DataFrame(prediction.detach().cpu().numpy().astype(float), columns=OUTPUT_COLUMNS)
    output["model"] = "ultimate_lap_time_distance_tcn"
    return output


__all__ = [
    "DeepTrainingConfig",
    "DeepTrainingResult",
    "DeepUltimateLapTimeModel",
    "examples_to_deep_numpy",
    "predict_ultimate_lap_time_deep",
    "train_ultimate_lap_time_deep",
]
