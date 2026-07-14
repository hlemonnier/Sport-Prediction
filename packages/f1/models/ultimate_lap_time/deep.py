"""Optional PyTorch CNN/TCN model for Ultimate Lap-Time telemetry."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


OUTPUT_COLUMNS: tuple[str, ...] = (
    "lap_p05",
    "lap_p50",
    "lap_p90",
    "sector1_seconds",
    "sector2_seconds",
    "sector3_seconds",
)
MINIMUM_POSITIVE_TIME_SECONDS = 1e-4


def torch_available() -> bool:
    return torch is not None and nn is not None


def resolve_device(preferred: str = "auto") -> str:
    if torch is None:
        return "unavailable"
    pref = str(preferred).strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps" and hasattr(torch.backends, "mps"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_torch(seed: int) -> None:
    if torch is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class DistanceTelemetryTCNConfig:
    """Architecture and loss controls for the distance-normalized TCN."""

    input_channels: int
    distance_bins: int
    static_feature_dim: int = 0
    hidden_channels: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.10
    static_hidden_dim: int = 32
    head_hidden_dim: int = 64
    activation: str = "gelu"
    lambda_sector: float = 0.20
    lambda_rank: float = 0.02
    # Retained for configuration compatibility. Output monotonicity is now
    # structural, so the composite loss does not apply this redundant term.
    lambda_mono: float = 0.0


@dataclass(frozen=True)
class DistanceTelemetryResidualTCNConfig:
    """Small TCN used only for event-blocked telemetry residual research.

    The production ``DistanceTelemetryTCN`` has a six-output physical lap-time
    contract.  Pre-Qualifying telemetry bags instead have a much narrower
    estimand: a bounded driver correction on top of a train-only
    rehearsal-to-Qualifying source shift.  Keeping that adapter explicit
    prevents the residual experiment from pretending that its labels are
    sector/quantile targets.
    """

    input_channels: int
    distance_bins: int
    static_feature_dim: int = 0
    hidden_channels: int = 4
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.0
    head_hidden_dim: int = 4
    max_abs_correction_seconds: float = 2.0


if nn is not None:

    class _TCNBlock(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            channels: int,
            *,
            kernel_size: int,
            dilation: int,
            dropout: float,
            activation: str,
        ) -> None:
            super().__init__()
            padding = int(dilation) * (int(kernel_size) - 1) // 2
            act: nn.Module
            if activation.lower() == "relu":
                act = nn.ReLU()
            else:
                act = nn.GELU()
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
                act,
                nn.Dropout(float(dropout)),
                nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
                nn.Dropout(float(dropout)),
            )
            self.out_activation = nn.ReLU() if activation.lower() == "relu" else nn.GELU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            residual = x
            out = self.net(x)
            if out.shape[-1] != residual.shape[-1]:
                min_len = min(out.shape[-1], residual.shape[-1])
                out = out[..., :min_len]
                residual = residual[..., :min_len]
            return self.out_activation(out + residual)


    class DistanceTelemetryTCN(nn.Module):  # type: ignore[misc]
        """TCN over telemetry[batch, channels, distance_bins] plus static features."""

        def __init__(self, config: DistanceTelemetryTCNConfig) -> None:
            super().__init__()
            if int(config.input_channels) <= 0:
                raise ValueError("input_channels must be positive")
            if int(config.distance_bins) <= 1:
                raise ValueError("distance_bins must be greater than one")
            self.config = config
            hidden = int(max(4, config.hidden_channels))
            self.input_projection = nn.Conv1d(int(config.input_channels), hidden, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    _TCNBlock(
                        hidden,
                        kernel_size=int(config.kernel_size),
                        dilation=int(dilation),
                        dropout=float(config.dropout),
                        activation=str(config.activation),
                    )
                    for dilation in config.dilations
                ]
            )
            self.attention = nn.Linear(hidden, 1)

            static_dim = int(max(0, config.static_feature_dim))
            if static_dim > 0:
                self.static_branch = nn.Sequential(
                    nn.Linear(static_dim, int(max(4, config.static_hidden_dim))),
                    nn.GELU() if config.activation.lower() != "relu" else nn.ReLU(),
                    nn.Dropout(float(config.dropout)),
                )
                head_input_dim = hidden + int(max(4, config.static_hidden_dim))
            else:
                self.static_branch = None
                head_input_dim = hidden

            self.head = nn.Sequential(
                nn.Linear(head_input_dim, int(max(4, config.head_hidden_dim))),
                nn.GELU() if config.activation.lower() != "relu" else nn.ReLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(int(max(4, config.head_hidden_dim)), len(OUTPUT_COLUMNS)),
            )

        def forward(self, telemetry: "torch.Tensor", static_features: "torch.Tensor | None" = None) -> "torch.Tensor":  # type: ignore[override]
            if telemetry.ndim != 3:
                raise ValueError("telemetry must have shape batch x channels x distance_bins")
            x = self.input_projection(telemetry)
            for block in self.blocks:
                x = block(x)
            sequence = x.transpose(1, 2)
            weights = torch.softmax(self.attention(sequence), dim=1)
            pooled = torch.sum(weights * sequence, dim=1)

            if self.static_branch is not None:
                if static_features is None:
                    static_features = torch.zeros(
                        (telemetry.shape[0], int(self.config.static_feature_dim)),
                        dtype=telemetry.dtype,
                        device=telemetry.device,
                    )
                pooled = torch.cat([pooled, self.static_branch(static_features)], dim=1)

            raw = self.head(pooled)
            sectors = F.softplus(raw[:, 0:3]) + MINIMUM_POSITIVE_TIME_SECONDS
            p50 = torch.sum(sectors, dim=1)
            uncertainty_scale = torch.sigmoid(raw[:, 5])
            lower_fraction = torch.clamp(
                uncertainty_scale * torch.sigmoid(raw[:, 3]),
                max=1.0 - 1e-4,
            )
            upper_fraction = uncertainty_scale * F.softplus(raw[:, 4])
            p05 = p50 * (1.0 - lower_fraction)
            p90 = p50 * (1.0 + upper_fraction)
            return torch.cat(
                [p05[:, None], p50[:, None], p90[:, None], sectors],
                dim=1,
            )


    class DistanceTelemetryResidualTCN(nn.Module):  # type: ignore[misc]
        """Low-capacity dilated TCN for a bounded scalar residual correction."""

        def __init__(self, config: DistanceTelemetryResidualTCNConfig) -> None:
            super().__init__()
            if int(config.input_channels) <= 0:
                raise ValueError("input_channels must be positive")
            if int(config.distance_bins) <= 1:
                raise ValueError("distance_bins must be greater than one")
            if int(config.static_feature_dim) < 0:
                raise ValueError("static_feature_dim cannot be negative")
            if int(config.hidden_channels) <= 0:
                raise ValueError("hidden_channels must be positive")
            if int(config.head_hidden_dim) <= 0:
                raise ValueError("head_hidden_dim must be positive")
            if int(config.kernel_size) <= 0 or int(config.kernel_size) % 2 == 0:
                raise ValueError("kernel_size must be a positive odd integer")
            dilations = tuple(int(dilation) for dilation in config.dilations)
            if not dilations or any(dilation <= 0 for dilation in dilations):
                raise ValueError(
                    "dilations must be a non-empty sequence of positive integers"
                )
            if dilations != tuple(sorted(set(dilations))):
                raise ValueError("dilations must be strictly increasing and unique")
            if not np.isfinite(float(config.dropout)) or not (
                0.0 <= float(config.dropout) < 1.0
            ):
                raise ValueError("dropout must be finite and in [0, 1)")
            if (
                not np.isfinite(float(config.max_abs_correction_seconds))
                or float(config.max_abs_correction_seconds) <= 0.0
            ):
                raise ValueError(
                    "max_abs_correction_seconds must be finite and positive"
                )
            self.config = config
            hidden = int(max(2, config.hidden_channels))
            self.input_projection = nn.Conv1d(
                int(config.input_channels), hidden, kernel_size=1
            )
            self.blocks = nn.ModuleList(
                [
                    _TCNBlock(
                        hidden,
                        kernel_size=int(config.kernel_size),
                        dilation=int(dilation),
                        dropout=float(config.dropout),
                        activation="gelu",
                    )
                    for dilation in config.dilations
                ]
            )
            self.attention = nn.Linear(hidden, 1)
            head_input = hidden + int(config.static_feature_dim)
            head_hidden = int(max(2, config.head_hidden_dim))
            self.head = nn.Sequential(
                nn.Linear(head_input, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )

        def forward(
            self,
            telemetry: "torch.Tensor",
            static_features: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":  # type: ignore[override]
            if telemetry.ndim != 3:
                raise ValueError(
                    "telemetry must have shape batch x channels x distance_bins"
                )
            x = self.input_projection(telemetry)
            for block in self.blocks:
                x = block(x)
            sequence = x.transpose(1, 2)
            weights = torch.softmax(self.attention(sequence), dim=1)
            pooled = torch.sum(weights * sequence, dim=1)
            static_dim = int(self.config.static_feature_dim)
            if static_dim:
                if static_features is None:
                    raise ValueError(
                        "static_features are required by the residual TCN config"
                    )
                if static_features.ndim != 2 or static_features.shape != (
                    telemetry.shape[0],
                    static_dim,
                ):
                    raise ValueError(
                        "static_features must have shape batch x static_feature_dim"
                    )
                pooled = torch.cat([pooled, static_features], dim=1)
            correction = self.head(pooled).squeeze(1)
            return torch.tanh(correction) * float(
                self.config.max_abs_correction_seconds
            )

else:

    class DistanceTelemetryTCN:  # pragma: no cover - exercised when torch missing
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("PyTorch is not installed")


    class DistanceTelemetryResidualTCN:  # pragma: no cover - torch missing
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("PyTorch is not installed")


def trainable_parameter_count(model: object) -> int:
    """Return the exact learned scalar count for an instantiated torch model."""

    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if not hasattr(model, "parameters"):
        raise TypeError("model must expose torch parameters")
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def pinball_loss_tensor(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    quantiles: Sequence[float] = (0.05, 0.50, 0.90),
) -> "torch.Tensor":
    """Apply every quantile head to the same realized lap outcome.

    Conditional quantiles are model outputs, not independently observable
    per-row labels.  The canonical target matrix repeats the realized lap in
    its first three columns for compatibility; accepting a one-dimensional
    outcome also makes the mathematical contract explicit.
    """

    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if target.ndim == 1:
        realized = target
    elif target.ndim == 2 and target.shape[1] >= 1:
        realized = target[:, 0]
    else:
        raise ValueError("quantile target must contain one realized lap outcome per row")
    losses = []
    for idx, quantile in enumerate(quantiles):
        pred = prediction[:, idx]
        truth = realized
        mask = torch.isfinite(pred) & torch.isfinite(truth)
        if torch.any(mask):
            error = truth[mask] - pred[mask]
            q = torch.as_tensor(float(quantile), dtype=prediction.dtype, device=prediction.device)
            losses.append(torch.mean(torch.maximum(q * error, (q - 1.0) * error)))
    if not losses:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return torch.stack(losses).mean()


def sector_mae_tensor(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    pred = prediction[:, 3:6]
    truth = target[:, 3:6]
    mask = torch.isfinite(pred) & torch.isfinite(truth)
    if not torch.any(mask):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return torch.mean(torch.abs(pred[mask] - truth[mask]))


def fastest_lap_pairwise_rank_loss(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    group_ids: "torch.Tensor",
) -> "torch.Tensor":
    """Pairwise fastest-lap loss restricted to one event/circuit/session.

    Absolute lap times are not comparable across tracks. Requiring explicit
    group ids prevents a random mixed-circuit batch from creating nonsensical
    ranking pairs.
    """

    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if group_ids.ndim != 1 or group_ids.shape[0] != prediction.shape[0]:
        raise ValueError("group_ids must have one value per prediction row")
    pred = prediction[:, 1]
    truth = target[:, 0]
    mask = torch.isfinite(pred) & torch.isfinite(truth)
    pred = pred[mask]
    truth = truth[mask]
    groups = group_ids[mask]
    if pred.numel() < 2:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    pred_diff = pred[:, None] - pred[None, :]
    truth_sign = torch.sign(truth[:, None] - truth[None, :])
    same_group = groups[:, None] == groups[None, :]
    pair_mask = (truth_sign != 0) & same_group
    if not torch.any(pair_mask):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return F.softplus(-truth_sign[pair_mask] * pred_diff[pair_mask]).mean()


def monotonic_quantile_penalty(prediction: "torch.Tensor") -> "torch.Tensor":
    """Backward-compatible diagnostic for externally supplied predictions."""

    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    p05, p50, p90 = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    return torch.mean(F.relu(p05 - p50) + F.relu(p50 - p90))


def ultimate_lap_time_deep_loss(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    config: DistanceTelemetryTCNConfig,
    *,
    group_ids: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Composite pinball, sector and within-event ranking loss.

    Positivity, quantile order and median/sector coherence are imposed by the
    network parameterization, so an additional monotonicity penalty would be
    mathematically redundant.
    """

    quantile = pinball_loss_tensor(prediction, target)
    sector = sector_mae_tensor(prediction, target)
    if float(config.lambda_rank) > 0.0 and group_ids is None:
        raise ValueError("group_ids are required when lambda_rank is positive")
    rank = (
        fastest_lap_pairwise_rank_loss(prediction, target, group_ids)
        if group_ids is not None
        else torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    )
    return (
        quantile
        + float(config.lambda_sector) * sector
        + float(config.lambda_rank) * rank
    )


def deep_output_contract_issues(
    prediction: np.ndarray | Sequence[Sequence[float]],
    *,
    coherence_tolerance_seconds: float = 1e-5,
) -> tuple[str, ...]:
    """Return physical-contract violations for deep output rows."""

    values = np.asarray(prediction, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(OUTPUT_COLUMNS):
        return (f"prediction must have shape rows x {len(OUTPUT_COLUMNS)}",)
    issues: list[str] = []
    if not np.isfinite(values).all():
        issues.append("predictions contain non-finite values")
        return tuple(issues)
    if np.any(values <= 0.0):
        issues.append("lap and sector predictions must be positive")
    if np.any((values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])):
        issues.append("lap quantiles must satisfy p05 <= p50 <= p90")
    sector_sum = np.sum(values[:, 3:6], axis=1)
    if not np.allclose(
        values[:, 1],
        sector_sum,
        atol=float(coherence_tolerance_seconds),
        rtol=1e-6,
    ):
        issues.append("lap p50 must equal the sum of the three sector predictions")
    return tuple(issues)


__all__ = [
    "OUTPUT_COLUMNS",
    "DistanceTelemetryTCN",
    "DistanceTelemetryTCNConfig",
    "DistanceTelemetryResidualTCN",
    "DistanceTelemetryResidualTCNConfig",
    "deep_output_contract_issues",
    "fastest_lap_pairwise_rank_loss",
    "monotonic_quantile_penalty",
    "pinball_loss_tensor",
    "resolve_device",
    "sector_mae_tensor",
    "seed_torch",
    "torch_available",
    "trainable_parameter_count",
    "ultimate_lap_time_deep_loss",
]
