"""Optional deep-learning models for tabular ranking/regression."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def torch_available() -> bool:
    return torch is not None


def resolve_device(preferred: str = "auto") -> str:
    if torch is None:
        return "unavailable"
    pref = str(preferred).strip().lower()
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    if torch is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if nn is not None:
    class _MLPRegressor(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
            super().__init__()
            dims = [int(input_dim)] + [int(max(1, h)) for h in hidden_dims]
            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(dims[-1], 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)
else:
    class _MLPRegressor:  # pragma: no cover - torch unavailable
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("PyTorch is not installed")


@dataclass
class TorchTabularConfig:
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 400
    batch_size: int = 64
    early_stopping_patience: int = 30
    seed: int = 42
    device: str = "auto"


class TorchTabularRegressor:
    """Simple MLP regressor for tabular rank targets with Huber loss."""

    def __init__(self, config: Optional[TorchTabularConfig] = None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        self.config = config or TorchTabularConfig()
        self.device_used = resolve_device(self.config.device)
        self.model: Optional[nn.Module] = None
        self.is_fitted = False

    @staticmethod
    def _to_float_array(X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        return np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _split_train_val(
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        if n <= 8:
            return X, y, X, y
        split = int(math.floor(n * (1.0 - val_ratio)))
        split = int(min(max(split, 4), n - 2))
        return X[:split], y[:split], X[split:], y[split:]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchTabularRegressor":
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        X_np = self._to_float_array(np.asarray(X))
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X_np.ndim != 2 or y_np.ndim != 2 or len(X_np) != len(y_np):
            raise ValueError("Invalid training matrix for TorchTabularRegressor")
        if len(X_np) == 0:
            raise ValueError("Empty training matrix")

        _seed_everything(int(self.config.seed))
        X_train, y_train, X_val, y_val = self._split_train_val(X_np, y_np)
        input_dim = int(X_train.shape[1])
        if input_dim <= 0:
            raise ValueError("Input feature dimension must be > 0")

        device = torch.device(self.device_used if self.device_used != "unavailable" else "cpu")
        model = _MLPRegressor(input_dim=input_dim, hidden_dims=self.config.hidden_dims, dropout=self.config.dropout).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        criterion = nn.HuberLoss(delta=1.0)

        train_ds = TensorDataset(
            torch.from_numpy(X_train).to(device),
            torch.from_numpy(y_train).to(device),
        )
        batch_size = int(max(8, self.config.batch_size))
        train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, drop_last=False)

        X_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)

        best_state: Optional[dict[str, torch.Tensor]] = None
        best_val_mae = float("inf")
        no_improve = 0
        max_epochs = int(max(20, self.config.epochs))
        patience = int(max(5, self.config.early_stopping_patience))

        for _ in range(max_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_mae = float(torch.mean(torch.abs(val_pred - y_val_t)).item())

            if val_mae + 1e-7 < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if torch is None or self.model is None or not self.is_fitted:
            raise RuntimeError("TorchTabularRegressor is not fitted")
        X_np = self._to_float_array(np.asarray(X))
        if X_np.ndim != 2:
            raise ValueError("Invalid inference matrix for TorchTabularRegressor")
        if len(X_np) == 0:
            return np.asarray([], dtype=float)
        device = torch.device(self.device_used if self.device_used != "unavailable" else "cpu")
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_np).to(device)
            pred = self.model(x_t).squeeze(-1)
        return pred.detach().cpu().numpy().astype(float)
