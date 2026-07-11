"""Fail-closed training for the optional Ultimate Lap-Time telemetry TCN."""

from __future__ import annotations

import copy
import math
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
    IDEAL_LAP_TARGET_CONTRACT,
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
    validation_fraction: float = 0.20
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    require_validation: bool = True


@dataclass(frozen=True)
class DeepFeatureNormalization:
    """Train-fitted normalization persisted with the deep model."""

    channel_names: tuple[str, ...]
    telemetry_mean: tuple[float, ...]
    telemetry_std: tuple[float, ...]
    static_feature_names: tuple[str, ...]
    static_mean: tuple[float, ...]
    static_std: tuple[float, ...]
    fitted_row_count: int

    def __post_init__(self) -> None:
        if len(self.channel_names) != len(self.telemetry_mean) or len(self.channel_names) != len(self.telemetry_std):
            raise ValueError("telemetry normalization stats must match channel_names")
        if len(self.static_feature_names) != len(self.static_mean) or len(self.static_feature_names) != len(self.static_std):
            raise ValueError("static normalization stats must match static_feature_names")
        if int(self.fitted_row_count) <= 0:
            raise ValueError("fitted_row_count must be positive")
        if any(not np.isfinite(value) for value in (*self.telemetry_mean, *self.telemetry_std, *self.static_mean, *self.static_std)):
            raise ValueError("normalization statistics must be finite")
        if any(value <= 0.0 for value in (*self.telemetry_std, *self.static_std)):
            raise ValueError("normalization standard deviations must be positive")

    def transform(self, telemetry: np.ndarray, static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        telemetry_values = np.asarray(telemetry, dtype=np.float32)
        static_values = np.asarray(static, dtype=np.float32)
        if telemetry_values.ndim != 3 or telemetry_values.shape[1] != len(self.channel_names):
            raise ValueError("telemetry shape does not match fitted normalization channels")
        if static_values.ndim != 2 or static_values.shape[1] != len(self.static_feature_names):
            raise ValueError("static feature shape does not match fitted normalization features")
        telemetry_mean = np.asarray(self.telemetry_mean, dtype=np.float32)[None, :, None]
        telemetry_std = np.asarray(self.telemetry_std, dtype=np.float32)[None, :, None]
        telemetry_out = (telemetry_values - telemetry_mean) / telemetry_std

        if static_values.shape[1] == 0:
            return telemetry_out.astype(np.float32, copy=False), static_values
        static_mean = np.asarray(self.static_mean, dtype=np.float32)[None, :]
        static_std = np.asarray(self.static_std, dtype=np.float32)[None, :]
        static_imputed = np.where(np.isfinite(static_values), static_values, static_mean)
        static_out = (static_imputed - static_mean) / static_std
        return telemetry_out.astype(np.float32, copy=False), static_out.astype(np.float32, copy=False)

    def to_payload(self) -> dict[str, object]:
        return {
            "channel_names": list(self.channel_names),
            "telemetry_mean": list(self.telemetry_mean),
            "telemetry_std": list(self.telemetry_std),
            "static_feature_names": list(self.static_feature_names),
            "static_mean": list(self.static_mean),
            "static_std": list(self.static_std),
            "fitted_row_count": int(self.fitted_row_count),
            "fit_scope": "training_rows_only",
        }


@dataclass
class DeepUltimateLapTimeModel:
    """Fitted PyTorch TCN plus preprocessing and validation metadata."""

    network: Any
    architecture_config: DistanceTelemetryTCNConfig
    training_config: DeepTrainingConfig
    channel_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    normalization: DeepFeatureNormalization
    device_used: str
    history: tuple[dict[str, float], ...]
    best_epoch: int
    training_group_keys: tuple[str, ...]
    validation_group_keys: tuple[str, ...]


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
    values = np.full((len(examples), len(static_feature_names)), np.nan, dtype=np.float32)
    for row_idx, example in enumerate(examples):
        for col_idx, name in enumerate(static_feature_names):
            raw = example.static_features.get(name)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values[row_idx, col_idx] = value
    return values


def examples_to_deep_numpy(
    examples: Sequence[UltimateLapTelemetryExample],
    *,
    static_feature_names: Sequence[str] | None = None,
    normalization: DeepFeatureNormalization | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Convert examples to telemetry/static/target numpy tensors."""

    batch = UltimateLapTelemetryBatch.from_examples(examples)
    names = tuple(static_feature_names) if static_feature_names is not None else _numeric_static_feature_names(examples)
    telemetry = batch.telemetry.astype(np.float32, copy=False)
    static = _static_matrix(examples, names)
    if normalization is not None:
        if batch.channel_names != normalization.channel_names:
            raise ValueError("inference telemetry channels do not match fitted model channel order")
        if names != normalization.static_feature_names:
            raise ValueError("inference static feature order does not match fitted model")
        telemetry, static = normalization.transform(telemetry, static)
    return telemetry, static, batch.target_matrix().astype(np.float32, copy=False), names


def _fit_normalization(
    telemetry: np.ndarray,
    static: np.ndarray,
    *,
    channel_names: Sequence[str],
    static_feature_names: Sequence[str],
) -> DeepFeatureNormalization:
    telemetry_mean = np.mean(telemetry, axis=(0, 2), dtype=np.float64)
    telemetry_std = np.std(telemetry, axis=(0, 2), dtype=np.float64)
    telemetry_std = np.where(telemetry_std > 1e-6, telemetry_std, 1.0)
    if static.shape[1]:
        with np.errstate(invalid="ignore"):
            static_mean = np.nanmean(static, axis=0, dtype=np.float64)
            static_std = np.nanstd(static, axis=0, dtype=np.float64)
        static_mean = np.where(np.isfinite(static_mean), static_mean, 0.0)
        static_std = np.where(np.isfinite(static_std) & (static_std > 1e-6), static_std, 1.0)
    else:
        static_mean = np.empty((0,), dtype=float)
        static_std = np.empty((0,), dtype=float)
    return DeepFeatureNormalization(
        channel_names=tuple(channel_names),
        telemetry_mean=tuple(float(value) for value in telemetry_mean),
        telemetry_std=tuple(float(value) for value in telemetry_std),
        static_feature_names=tuple(static_feature_names),
        static_mean=tuple(float(value) for value in static_mean),
        static_std=tuple(float(value) for value in static_std),
        fitted_row_count=int(telemetry.shape[0]),
    )


def _group_key(example: UltimateLapTelemetryExample) -> str:
    metadata = example.metadata
    return "|".join(
        (
            str(metadata.season or ""),
            str(metadata.event_key),
            str(metadata.circuit_id),
            str(metadata.session),
        )
    )


def _group_sort_key(example: UltimateLapTelemetryExample) -> tuple[str, str, str, str]:
    metadata = example.metadata
    return (
        str(metadata.season or ""),
        str(metadata.event_key),
        str(metadata.circuit_id),
        str(metadata.session),
    )


def _target_contract_reason(examples: Sequence[UltimateLapTelemetryExample]) -> str | None:
    invalid = sorted({example.targets.target_contract for example in examples if example.targets.target_contract != IDEAL_LAP_TARGET_CONTRACT})
    if invalid:
        return f"deep training requires explicit {IDEAL_LAP_TARGET_CONTRACT} targets; found {invalid}"
    return None


def _split_grouped_temporal_validation(
    examples: Sequence[UltimateLapTelemetryExample],
    validation_examples: Sequence[UltimateLapTelemetryExample] | None,
    config: DeepTrainingConfig,
) -> tuple[tuple[UltimateLapTelemetryExample, ...], tuple[UltimateLapTelemetryExample, ...], str | None]:
    if validation_examples is not None:
        train_rows = tuple(examples)
        validation_rows = tuple(validation_examples)
    else:
        explicit_train = tuple(
            example for example in examples if str(example.metadata.split_key.split_name or "").lower() == "train"
        )
        explicit_validation = tuple(
            example
            for example in examples
            if str(example.metadata.split_key.split_name or "").lower() in {"validation", "val"}
        )
        if explicit_train and explicit_validation:
            train_rows, validation_rows = explicit_train, explicit_validation
        else:
            grouped: dict[str, list[UltimateLapTelemetryExample]] = {}
            first_by_group: dict[str, UltimateLapTelemetryExample] = {}
            for example in examples:
                key = _group_key(example)
                grouped.setdefault(key, []).append(example)
                first_by_group.setdefault(key, example)
            ordered_groups = sorted(grouped, key=lambda key: _group_sort_key(first_by_group[key]))
            if len(ordered_groups) < 2:
                return (), (), "grouped temporal validation requires at least two event/circuit/session groups"
            fraction = float(config.validation_fraction)
            if not 0.0 < fraction < 1.0:
                return (), (), "validation_fraction must be between zero and one"
            validation_group_count = max(1, int(math.ceil(len(ordered_groups) * fraction)))
            validation_group_count = min(validation_group_count, len(ordered_groups) - 1)
            validation_keys = set(ordered_groups[-validation_group_count:])
            train_rows = tuple(example for example in examples if _group_key(example) not in validation_keys)
            validation_rows = tuple(example for example in examples if _group_key(example) in validation_keys)

    if not train_rows:
        return (), (), "training partition is empty"
    if config.require_validation and not validation_rows:
        return (), (), "validation partition is required"
    if not validation_rows:
        return train_rows, validation_rows, None
    train_groups = {_group_key(example) for example in train_rows}
    validation_groups = {_group_key(example) for example in validation_rows}
    overlap = sorted(train_groups & validation_groups)
    if overlap:
        return (), (), f"train/validation groups overlap: {overlap}"
    latest_train = max(_group_sort_key(example) for example in train_rows)
    earliest_validation = min(_group_sort_key(example) for example in validation_rows)
    if latest_train >= earliest_validation:
        return (), (), "validation groups must be strictly later than training groups"
    return train_rows, validation_rows, None


def _group_id_array(examples: Sequence[UltimateLapTelemetryExample]) -> np.ndarray:
    keys = [_group_key(example) for example in examples]
    mapping = {key: idx for idx, key in enumerate(sorted(set(keys)))}
    return np.asarray([mapping[key] for key in keys], dtype=np.int64)


def _group_batches(group_ids: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    groups = np.unique(group_ids)
    rng.shuffle(groups)
    for group in groups:
        indices = np.flatnonzero(group_ids == group)
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    return batches


def train_ultimate_lap_time_deep(
    examples: Sequence[UltimateLapTelemetryExample],
    *,
    validation_examples: Sequence[UltimateLapTelemetryExample] | None = None,
    config: DeepTrainingConfig | None = None,
) -> DeepTrainingResult:
    """Train with grouped temporal validation and train-only normalization."""

    cfg = config or DeepTrainingConfig()
    if not examples:
        raise ValueError("examples must contain at least one telemetry example")
    if not torch_available():
        return DeepTrainingResult(status="skipped", reason="PyTorch is not installed", model=None)

    train_examples, val_examples, split_reason = _split_grouped_temporal_validation(examples, validation_examples, cfg)
    if split_reason is not None:
        return DeepTrainingResult(status="rejected", reason=split_reason, model=None)
    contract_reason = _target_contract_reason((*train_examples, *val_examples))
    if contract_reason is not None:
        return DeepTrainingResult(status="rejected", reason=contract_reason, model=None)

    train_batch = UltimateLapTelemetryBatch.from_examples(train_examples)
    static_names = _numeric_static_feature_names(train_examples)
    train_telemetry_raw, train_static_raw, train_target_np, _ = examples_to_deep_numpy(
        train_examples,
        static_feature_names=static_names,
    )
    normalization = _fit_normalization(
        train_telemetry_raw,
        train_static_raw,
        channel_names=train_batch.channel_names,
        static_feature_names=static_names,
    )
    train_telemetry_np, train_static_np = normalization.transform(train_telemetry_raw, train_static_raw)
    if val_examples:
        val_telemetry_np, val_static_np, val_target_np, _ = examples_to_deep_numpy(
            val_examples,
            static_feature_names=static_names,
            normalization=normalization,
        )
    else:
        val_telemetry_np = np.empty((0, *train_telemetry_np.shape[1:]), dtype=np.float32)
        val_static_np = np.empty((0, train_static_np.shape[1]), dtype=np.float32)
        val_target_np = np.empty((0, train_target_np.shape[1]), dtype=np.float32)

    seed_torch(int(cfg.seed))
    device_name = resolve_device(cfg.device)
    device = torch.device(device_name if device_name != "unavailable" else "cpu")
    architecture = DistanceTelemetryTCNConfig(
        input_channels=int(train_telemetry_np.shape[1]),
        distance_bins=int(train_telemetry_np.shape[2]),
        static_feature_dim=int(train_static_np.shape[1]),
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
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    train_telemetry_t = torch.from_numpy(train_telemetry_np).to(device)
    train_static_t = torch.from_numpy(train_static_np).to(device)
    train_target_t = torch.from_numpy(train_target_np).to(device)
    train_groups_np = _group_id_array(train_examples)
    train_groups_t = torch.from_numpy(train_groups_np).to(device)
    val_telemetry_t = torch.from_numpy(val_telemetry_np).to(device)
    val_static_t = torch.from_numpy(val_static_np).to(device)
    val_target_t = torch.from_numpy(val_target_np).to(device)
    val_groups_t = torch.from_numpy(_group_id_array(val_examples)).to(device) if val_examples else None

    history: list[dict[str, float]] = []
    max_epochs = int(max(1, cfg.epochs))
    batch_size = int(max(1, min(cfg.batch_size, len(train_examples))))
    patience = int(max(1, cfg.early_stopping_patience))
    min_delta = float(max(0.0, cfg.early_stopping_min_delta))
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0

    for epoch in range(max_epochs):
        rng = np.random.default_rng(int(cfg.seed) + epoch)
        batch_losses: list[float] = []
        network.train()
        for batch_idx in _group_batches(train_groups_np, batch_size, rng):
            batch_tensor = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = network(
                train_telemetry_t.index_select(0, batch_tensor),
                train_static_t.index_select(0, batch_tensor),
            )
            loss = ultimate_lap_time_deep_loss(
                prediction,
                train_target_t.index_select(0, batch_tensor),
                architecture,
                group_ids=train_groups_t.index_select(0, batch_tensor),
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses))
        network.eval()
        if val_examples:
            assert val_groups_t is not None
            with torch.no_grad():
                val_prediction = network(val_telemetry_t, val_static_t)
                validation_loss = float(
                    ultimate_lap_time_deep_loss(
                        val_prediction,
                        val_target_t,
                        architecture,
                        group_ids=val_groups_t,
                    ).detach().cpu().item()
                )
        else:
            validation_loss = train_loss
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": train_loss,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(network.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None or best_epoch <= 0 or not np.isfinite(best_validation_loss):
        return DeepTrainingResult(
            status="rejected",
            reason="validation loss never produced a finite checkpoint",
            model=None,
            history=tuple(history),
        )
    network.load_state_dict(best_state)
    fitted = DeepUltimateLapTimeModel(
        network=network,
        architecture_config=architecture,
        training_config=cfg,
        channel_names=train_batch.channel_names,
        static_feature_names=static_names,
        normalization=normalization,
        device_used=device_name,
        history=tuple(history),
        best_epoch=int(best_epoch),
        training_group_keys=tuple(sorted({_group_key(example) for example in train_examples})),
        validation_group_keys=tuple(sorted({_group_key(example) for example in val_examples})),
    )
    return DeepTrainingResult(status="trained", reason=None, model=fitted, history=tuple(history))


def predict_ultimate_lap_time_deep(
    model: DeepUltimateLapTimeModel,
    examples: Sequence[UltimateLapTelemetryExample],
) -> pd.DataFrame:
    """Predict using the exact train-fitted feature normalization."""

    if not torch_available():
        raise RuntimeError("PyTorch is not installed")
    if not isinstance(model, DeepUltimateLapTimeModel):
        raise TypeError("model must be a DeepUltimateLapTimeModel")
    if not examples:
        return pd.DataFrame(columns=[*OUTPUT_COLUMNS, "model"])
    contract_reason = _target_contract_reason(examples)
    if contract_reason is not None:
        raise ValueError(contract_reason)
    telemetry_np, static_np, _, _ = examples_to_deep_numpy(
        examples,
        static_feature_names=model.static_feature_names,
        normalization=model.normalization,
    )
    device = torch.device(model.device_used if model.device_used != "unavailable" else "cpu")
    network = model.network.to(device)
    network.eval()
    with torch.no_grad():
        prediction = network(torch.from_numpy(telemetry_np).to(device), torch.from_numpy(static_np).to(device))
    output = pd.DataFrame(prediction.detach().cpu().numpy().astype(float), columns=OUTPUT_COLUMNS)
    output["model"] = "ultimate_lap_time_distance_tcn"
    return output


__all__ = [
    "DeepFeatureNormalization",
    "DeepTrainingConfig",
    "DeepTrainingResult",
    "DeepUltimateLapTimeModel",
    "examples_to_deep_numpy",
    "predict_ultimate_lap_time_deep",
    "train_ultimate_lap_time_deep",
]
