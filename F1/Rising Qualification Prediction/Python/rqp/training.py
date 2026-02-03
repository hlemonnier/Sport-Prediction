"""Model training utilities."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover - optional dependency
    Ridge = None


def train_model(train: pd.DataFrame, feature_cols: List[str]) -> Optional[object]:
    if train.empty:
        return None
    if Ridge is None:
        return None
    X = train[feature_cols].copy()
    y = train["target"]
    X = X.fillna(X.median(numeric_only=True))
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model
