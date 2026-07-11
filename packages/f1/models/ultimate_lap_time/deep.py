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
    lambda_mono: float = 1.0


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
            p05 = raw[:, 0]
            p50 = p05 + F.softplus(raw[:, 1])
            p90 = p50 + F.softplus(raw[:, 2])
            return torch.stack([p05, p50, p90, raw[:, 3], raw[:, 4], raw[:, 5]], dim=1)

else:

    class DistanceTelemetryTCN:  # pragma: no cover - exercised when torch missing
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("PyTorch is not installed")


def pinball_loss_tensor(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    quantiles: Sequence[float] = (0.05, 0.50, 0.90),
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    losses = []
    for idx, quantile in enumerate(quantiles):
        pred = prediction[:, idx]
        truth = target[:, idx]
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
    truth = target[:, 1]
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
    """Composite loss from the roadmap: pinball + sectors + rank + monotonicity."""

    quantile = pinball_loss_tensor(prediction, target)
    sector = sector_mae_tensor(prediction, target)
    if float(config.lambda_rank) > 0.0 and group_ids is None:
        raise ValueError("group_ids are required when lambda_rank is positive")
    rank = (
        fastest_lap_pairwise_rank_loss(prediction, target, group_ids)
        if group_ids is not None
        else torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    )
    mono = monotonic_quantile_penalty(prediction)
    return (
        quantile
        + float(config.lambda_sector) * sector
        + float(config.lambda_rank) * rank
        + float(config.lambda_mono) * mono
    )


__all__ = [
    "OUTPUT_COLUMNS",
    "DistanceTelemetryTCN",
    "DistanceTelemetryTCNConfig",
    "fastest_lap_pairwise_rank_loss",
    "monotonic_quantile_penalty",
    "pinball_loss_tensor",
    "resolve_device",
    "sector_mae_tensor",
    "seed_torch",
    "torch_available",
    "ultimate_lap_time_deep_loss",
]
