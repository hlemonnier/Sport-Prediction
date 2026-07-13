"""Causal event-grouped learning-to-rank challengers for F1.

The optional XGBoost and LightGBM adapters consume contiguous event groups in
chronological order.  Their deterministic sklearn fallback is Bradley-Terry
pairwise logistic regression whose comparisons are constructed strictly inside
each event.  The fit result always contains a reproducible manifest, including
backend attempts and an explicit unavailable state; an unavailable native
backend is never silently relabelled as the requested model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from packages.f1.orchestration.model_runtime import inspect_optional_model_runtime


GROUPED_RANKING_BACKENDS: tuple[str, ...] = (
    "auto",
    "xgboost_lambdarank",
    "lightgbm_lambdarank",
    "sklearn_pairwise",
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_version(package: str) -> Optional[str]:
    try:
        return str(importlib.metadata.version(package))
    except Exception:
        return None


def _normalize_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "xgboost": "xgboost_lambdarank",
        "xgb": "xgboost_lambdarank",
        "xgb_lambdarank": "xgboost_lambdarank",
        "lightgbm": "lightgbm_lambdarank",
        "lgbm": "lightgbm_lambdarank",
        "lgbm_lambdarank": "lightgbm_lambdarank",
        "sklearn": "sklearn_pairwise",
        "pairwise": "sklearn_pairwise",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class GroupedRankingConfig:
    """Frozen model and grouping controls for one ranking experiment."""

    feature_columns: tuple[str, ...]
    backend: str = "auto"
    season_column: str = "season"
    group_column: str = "event_key"
    event_time_column: str = "event_as_of"
    item_column: str = "driver_id"
    target_column: str = "position"
    lower_target_is_better: bool = True
    random_state: int = 42
    n_estimators: int = 240
    learning_rate: float = 0.04
    max_depth: int = 4
    num_leaves: int = 15
    regularization_c: float = 0.5
    max_iterations: int = 1200
    minimum_training_events: int = 2
    allow_requested_backend_fallback: bool = False

    def __post_init__(self) -> None:
        backend = _normalize_backend(self.backend)
        if backend not in GROUPED_RANKING_BACKENDS:
            raise ValueError(f"backend must be one of {GROUPED_RANKING_BACKENDS}")
        if not self.feature_columns:
            raise ValueError("feature_columns must be an explicit non-empty allowlist")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        reserved = {
            self.season_column,
            self.group_column,
            self.event_time_column,
            self.item_column,
            self.target_column,
        }
        overlap = reserved.intersection(self.feature_columns)
        if overlap:
            raise ValueError(f"feature_columns contain reserved fields: {sorted(overlap)}")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if self.n_estimators < 1 or self.max_iterations < 1:
            raise ValueError("iteration counts must be positive")
        if self.learning_rate <= 0.0 or self.regularization_c <= 0.0:
            raise ValueError("learning_rate and regularization_c must be positive")
        if self.max_depth < 1 or self.num_leaves < 2:
            raise ValueError("tree depth/leaves must be positive")
        if self.minimum_training_events < 2:
            raise ValueError("minimum_training_events must be at least two")

    @property
    def normalized_backend(self) -> str:
        return _normalize_backend(self.backend)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend"] = self.normalized_backend
        payload["feature_columns"] = list(self.feature_columns)
        return payload

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_payload())


@dataclass(frozen=True)
class ChronologicalGroupPartition:
    """Disjoint early/late event blocks with a strict time boundary."""

    training: pd.DataFrame
    validation: pd.DataFrame
    training_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    training_max_time: str
    validation_min_time: str


@dataclass
class GroupedFeatureEncoder:
    """Deterministic mixed-type encoder shared by all ranking backends."""

    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    numeric_medians: dict[str, float] | None = None
    numeric_means: dict[str, float] | None = None
    numeric_scales: dict[str, float] | None = None
    categories: dict[str, tuple[str, ...]] | None = None
    feature_names_out: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> "GroupedFeatureEncoder":
        numeric: list[str] = []
        categorical: list[str] = []
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        categories: dict[str, tuple[str, ...]] = {}
        for column in feature_columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                values = pd.to_numeric(series, errors="coerce")
                finite = values[np.isfinite(values)]
                median = float(finite.median()) if len(finite) else 0.0
                filled = values.fillna(median).astype(float)
                mean = float(filled.mean()) if len(filled) else 0.0
                scale = float(filled.std(ddof=0)) if len(filled) else 1.0
                if not np.isfinite(scale) or scale <= 1e-12:
                    scale = 1.0
                medians[column] = median
                means[column] = mean
                scales[column] = scale
                numeric.append(column)
            else:
                values = series.astype("string").fillna("__missing__").astype(str)
                # Alphabetic categories make the schema invariant to row order.
                categories[column] = tuple(sorted(values.unique().tolist()))
                categorical.append(column)

        names: list[str] = list(numeric)
        for column in categorical:
            names.extend(f"{column}={category}" for category in categories[column])
        self.numeric_columns = tuple(numeric)
        self.categorical_columns = tuple(categorical)
        self.numeric_medians = medians
        self.numeric_means = means
        self.numeric_scales = scales
        self.categories = categories
        self.feature_names_out = tuple(names or ["__intercept__"])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if any(
            value is None
            for value in (self.numeric_medians, self.numeric_means, self.numeric_scales, self.categories)
        ):
            raise RuntimeError("GroupedFeatureEncoder is not fitted")
        parts: list[np.ndarray] = []
        assert self.numeric_medians is not None
        assert self.numeric_means is not None
        assert self.numeric_scales is not None
        assert self.categories is not None
        for column in self.numeric_columns:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            else:
                values = np.full(len(frame), np.nan, dtype=float)
            values = np.where(np.isfinite(values), values, self.numeric_medians[column])
            values = (values - self.numeric_means[column]) / self.numeric_scales[column]
            parts.append(values.reshape(-1, 1))
        for column in self.categorical_columns:
            if column in frame.columns:
                values = frame[column].astype("string").fillna("__missing__").astype(str).to_numpy()
            else:
                values = np.full(len(frame), "__missing__", dtype=object)
            for category in self.categories[column]:
                parts.append((values == category).astype(float).reshape(-1, 1))
        return (
            np.hstack(parts).astype(np.float32, copy=False)
            if parts
            else np.ones((len(frame), 1), dtype=np.float32)
        )


@dataclass(frozen=True)
class GroupedRankingDataset:
    """Chronologically ordered rows and contiguous listwise group sizes."""

    frame: pd.DataFrame
    values: np.ndarray
    relevance: np.ndarray
    group_sizes: tuple[int, ...]
    event_order: tuple[str, ...]
    event_times: tuple[str, ...]
    encoder: GroupedFeatureEncoder
    data_fingerprint: str


@dataclass(frozen=True)
class EventPurePairwiseDataset:
    values: np.ndarray
    labels: np.ndarray
    sample_weight: np.ndarray
    pair_rows_by_event: dict[str, int]
    undirected_pairs_by_event: dict[str, int]
    cross_event_pair_count: int = 0


def _event_metadata(frame: pd.DataFrame, config: GroupedRankingConfig) -> pd.DataFrame:
    required = {
        config.season_column,
        config.group_column,
        config.event_time_column,
        config.item_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"ranking data is missing required columns: {missing}")
    if (
        frame[config.season_column].isna().any()
        or frame[config.group_column].isna().any()
        or frame[config.item_column].isna().any()
    ):
        raise ValueError("season, event and item identifiers must not be missing")
    seasons = pd.to_numeric(frame[config.season_column], errors="coerce")
    if seasons.isna().any() or not np.equal(seasons, np.floor(seasons)).all():
        raise ValueError("season identifiers must be finite integers")
    times = pd.to_datetime(frame[config.event_time_column], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError("event_time_column must contain valid timestamps")
    work = pd.DataFrame(
        {
            "__season": seasons.astype(int),
            "__event": frame[config.group_column],
            "__event_token": [
                f"{int(season)}|{type(value).__name__}:{value}"
                for season, value in zip(seasons, frame[config.group_column])
            ],
            "__time": times,
        },
        index=frame.index,
    )
    per_event = work.groupby("__event_token", sort=False, dropna=False).agg(
        event=("__event", "first"),
        season=("__season", "first"),
        event_time=("__time", "first"),
        timestamp_count=("__time", "nunique"),
    )
    if (per_event["timestamp_count"] != 1).any():
        bad = per_event.index[per_event["timestamp_count"] != 1].tolist()
        raise ValueError(f"each event must have exactly one event timestamp: {bad}")
    per_event = per_event.reset_index().sort_values(
        ["event_time", "__event_token"], kind="mergesort"
    )
    return per_event


def _event_token_series(frame: pd.DataFrame, config: GroupedRankingConfig) -> pd.Series:
    seasons = pd.to_numeric(frame[config.season_column], errors="coerce")
    if seasons.isna().any():
        raise ValueError("season identifiers must be numeric")
    return pd.Series(
        [
            f"{int(season)}|{type(value).__name__}:{value}"
            for season, value in zip(seasons, frame[config.group_column])
        ],
        index=frame.index,
        dtype="string",
    )


def chronological_group_partition(
    records: pd.DataFrame,
    *,
    config: GroupedRankingConfig,
    validation_event_count: int | None = None,
    validation_fraction: float = 0.20,
) -> ChronologicalGroupPartition:
    """Split whole events into strict early-train and late-validation blocks."""

    frame = records.copy()
    events = _event_metadata(frame, config)
    event_count = len(events)
    if event_count < 2:
        raise ValueError("chronological partition requires at least two events")
    if validation_event_count is None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        validation_event_count = max(1, int(np.ceil(event_count * validation_fraction)))
    if validation_event_count < 1 or validation_event_count >= event_count:
        raise ValueError("validation_event_count must leave at least one training event")
    split = event_count - int(validation_event_count)
    train_events = events.iloc[:split]
    validation_events = events.iloc[split:]
    train_max = pd.Timestamp(train_events["event_time"].max())
    validation_min = pd.Timestamp(validation_events["event_time"].min())
    if train_max >= validation_min:
        raise ValueError("chronological split boundary must be strictly increasing")

    token = _event_token_series(frame, config)
    training_tokens = set(train_events["__event_token"].astype(str))
    validation_tokens = set(validation_events["__event_token"].astype(str))
    if training_tokens.intersection(validation_tokens):  # defensive assertion
        raise RuntimeError("event groups overlap across chronological partition")
    return ChronologicalGroupPartition(
        training=frame.loc[token.isin(training_tokens)].copy(),
        validation=frame.loc[token.isin(validation_tokens)].copy(),
        training_groups=tuple(train_events["__event_token"].astype(str)),
        validation_groups=tuple(validation_events["__event_token"].astype(str)),
        training_max_time=train_max.isoformat().replace("+00:00", "Z"),
        validation_min_time=validation_min.isoformat().replace("+00:00", "Z"),
    )


def _same_season_records(
    records: pd.DataFrame,
    config: GroupedRankingConfig,
    target_season: int | None,
) -> tuple[pd.DataFrame, int, int]:
    if config.season_column not in records.columns:
        raise ValueError(
            "same-season ranking requires an explicit season column; prior-season absolute pace is forbidden"
        )
    seasons = pd.to_numeric(records[config.season_column], errors="coerce")
    if seasons.isna().any() or not np.equal(seasons, np.floor(seasons)).all():
        raise ValueError("season identifiers must be finite integers")
    resolved = int(seasons.max()) if target_season is None else int(target_season)
    selected = records.loc[seasons.eq(float(resolved))].copy()
    if selected.empty:
        raise ValueError(f"no ranking rows are available for target season {resolved}")
    return selected, resolved, int(len(records) - len(selected))


def same_season_walk_forward_partitions(
    records: pd.DataFrame,
    *,
    config: GroupedRankingConfig,
    target_season: int | None = None,
    minimum_training_events: int | None = None,
) -> tuple[ChronologicalGroupPartition, ...]:
    """Build expanding same-season folds, one untouched target event at a time."""

    frame, _, _ = _same_season_records(records, config, target_season)
    events = _event_metadata(frame, config)
    minimum = (
        config.minimum_training_events
        if minimum_training_events is None
        else int(minimum_training_events)
    )
    if minimum < 1:
        raise ValueError("minimum_training_events must be positive")
    if len(events) <= minimum:
        raise ValueError("walk-forward history does not contain a target event after warm-up")
    token = _event_token_series(frame, config)
    folds: list[ChronologicalGroupPartition] = []
    for target_index in range(minimum, len(events)):
        train_events = events.iloc[:target_index]
        target_event = events.iloc[target_index : target_index + 1]
        train_tokens = set(train_events["__event_token"].astype(str))
        target_tokens = set(target_event["__event_token"].astype(str))
        train_max = pd.Timestamp(train_events["event_time"].max())
        target_min = pd.Timestamp(target_event["event_time"].min())
        if train_max >= target_min:
            raise ValueError("walk-forward event timestamps must be strictly increasing")
        folds.append(
            ChronologicalGroupPartition(
                training=frame.loc[token.isin(train_tokens)].copy(),
                validation=frame.loc[token.isin(target_tokens)].copy(),
                training_groups=tuple(train_events["__event_token"].astype(str)),
                validation_groups=tuple(target_event["__event_token"].astype(str)),
                training_max_time=train_max.isoformat().replace("+00:00", "Z"),
                validation_min_time=target_min.isoformat().replace("+00:00", "Z"),
            )
        )
    return tuple(folds)


def prepare_grouped_ranking_dataset(
    records: pd.DataFrame,
    *,
    config: GroupedRankingConfig,
    cutoff: object | None = None,
) -> GroupedRankingDataset:
    """Encode complete event groups in stable chronological order."""

    frame = records.copy()
    required = {
        config.group_column,
        config.season_column,
        config.event_time_column,
        config.item_column,
        config.target_column,
        *config.feature_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"ranking data is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("ranking data must not be empty")
    events = _event_metadata(frame, config)
    if cutoff is not None:
        cutoff_time = pd.to_datetime(cutoff, errors="coerce", utc=True)
        if pd.isna(cutoff_time):
            raise ValueError("cutoff must be a valid timestamp")
        keep_tokens = set(
            events.loc[events["event_time"] < cutoff_time, "__event_token"].astype(str)
        )
        token = _event_token_series(frame, config)
        frame = frame.loc[token.isin(keep_tokens)].copy()
        if frame.empty:
            raise ValueError("no complete ranking events precede cutoff")
        events = _event_metadata(frame, config)
    if len(events) < config.minimum_training_events:
        raise ValueError(
            f"ranking model needs at least {config.minimum_training_events} complete events"
        )
    duplicates = frame.duplicated(
        [config.season_column, config.group_column, config.item_column], keep=False
    )
    if duplicates.any():
        raise ValueError("each item may appear at most once inside an event group")
    target = pd.to_numeric(frame[config.target_column], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("ranking targets must be finite numeric values")

    order_lookup = dict(zip(events["__event_token"], range(len(events))))
    ordered = frame.copy()
    ordered["__event_token"] = _event_token_series(ordered, config)
    ordered["__event_order"] = ordered["__event_token"].map(order_lookup)
    ordered["__item_token"] = ordered[config.item_column].astype(str)
    ordered["__source_row"] = np.arange(len(ordered), dtype=int)
    ordered = ordered.sort_values(
        ["__event_order", "__item_token", "__source_row"], kind="mergesort"
    ).reset_index(drop=True)

    relevance_parts: list[np.ndarray] = []
    group_sizes: list[int] = []
    for _, group in ordered.groupby("__event_order", sort=True, dropna=False):
        values = pd.to_numeric(group[config.target_column], errors="raise")
        rank = values.rank(method="dense", ascending=config.lower_target_is_better)
        relevance = (rank.max() - rank).astype(int).to_numpy(dtype=np.int32)
        relevance_parts.append(relevance)
        group_sizes.append(len(group))
    relevance_values = np.concatenate(relevance_parts)
    if not any(np.unique(values).size > 1 for values in relevance_parts):
        raise ValueError("ranking history contains no within-event target ordering")

    encoder = GroupedFeatureEncoder().fit(ordered, config.feature_columns)
    matrix = encoder.transform(ordered)
    hash_columns = [
        config.group_column,
        config.season_column,
        config.event_time_column,
        config.item_column,
        config.target_column,
        *config.feature_columns,
    ]
    row_hashes = pd.util.hash_pandas_object(ordered[hash_columns], index=False, categorize=True)
    fingerprint = hashlib.sha256(row_hashes.to_numpy(dtype=np.uint64).tobytes()).hexdigest()
    event_times = tuple(
        pd.Timestamp(value).isoformat().replace("+00:00", "Z")
        for value in events["event_time"]
    )
    return GroupedRankingDataset(
        frame=ordered,
        values=matrix,
        relevance=relevance_values,
        group_sizes=tuple(group_sizes),
        event_order=tuple(events["__event_token"].astype(str)),
        event_times=event_times,
        encoder=encoder,
        data_fingerprint=fingerprint,
    )


def build_event_pure_pairwise_dataset(
    dataset: GroupedRankingDataset,
) -> EventPurePairwiseDataset:
    """Create balanced Bradley-Terry rows without cross-event comparisons."""

    pair_values: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    pair_rows: dict[str, int] = {}
    undirected: dict[str, int] = {}
    offset = 0
    for event, size in zip(dataset.event_order, dataset.group_sizes):
        local_values = dataset.values[offset : offset + size]
        local_relevance = dataset.relevance[offset : offset + size]
        local_deltas: list[tuple[np.ndarray, float]] = []
        for left in range(size):
            for right in range(left + 1, size):
                if local_relevance[left] == local_relevance[right]:
                    continue
                delta = local_values[left] - local_values[right]
                label = float(local_relevance[left] > local_relevance[right])
                local_deltas.append((delta, label))
        if local_deltas:
            row_weight = 1.0 / float(2 * len(local_deltas))
            for delta, label in local_deltas:
                pair_values.extend((delta, -delta))
                labels.extend((label, 1.0 - label))
                weights.extend((row_weight, row_weight))
        undirected[event] = len(local_deltas)
        pair_rows[event] = 2 * len(local_deltas)
        offset += size
    if not pair_values:
        raise ValueError("ranking history produced no within-event pairs")
    return EventPurePairwiseDataset(
        values=np.vstack(pair_values).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=float),
        sample_weight=np.asarray(weights, dtype=float),
        pair_rows_by_event=pair_rows,
        undirected_pairs_by_event=undirected,
        cross_event_pair_count=0,
    )


@dataclass
class _SklearnPairwiseRanker:
    config: GroupedRankingConfig
    estimator: Any = None
    pairwise_audit: EventPurePairwiseDataset | None = None

    def fit(self, dataset: GroupedRankingDataset) -> "_SklearnPairwiseRanker":
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            raise RuntimeError(f"sklearn pairwise fallback unavailable: {exc}") from exc
        pairs = build_event_pure_pairwise_dataset(dataset)
        estimator = LogisticRegression(
            C=float(self.config.regularization_c),
            fit_intercept=False,
            max_iter=int(self.config.max_iterations),
            random_state=int(self.config.random_state),
            solver="lbfgs",
        )
        estimator.fit(pairs.values, pairs.labels, sample_weight=pairs.sample_weight)
        self.estimator = estimator
        self.pairwise_audit = pairs
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("sklearn pairwise ranker is not fitted")
        return np.asarray(self.estimator.decision_function(values), dtype=float)


@dataclass
class GroupedRankingModel:
    config: GroupedRankingConfig
    backend_name: str
    estimator: Any
    encoder: GroupedFeatureEncoder
    training_event_order: tuple[str, ...]
    training_max_time: str
    training_season: int
    training_manifest: dict[str, Any]

    @property
    def model_name(self) -> str:
        return f"event_grouped_{self.backend_name}_v1"

    def predict(self, records: pd.DataFrame, *, enforce_future: bool = True) -> pd.DataFrame:
        frame = records.copy()
        required = {
            self.config.season_column,
            self.config.group_column,
            self.config.event_time_column,
            self.config.item_column,
            *self.config.feature_columns,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"ranking inference is missing required columns: {missing}")
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    self.config.group_column,
                    self.config.item_column,
                    "ranking_score",
                    "predicted_rank",
                    "ranking_model",
                ]
            )
        _event_metadata(frame, self.config)
        seasons = pd.to_numeric(frame[self.config.season_column], errors="coerce")
        if seasons.isna().any() or not seasons.eq(float(self.training_season)).all():
            raise ValueError(
                f"ranker trained for season {self.training_season}; inference may not mix seasons"
            )
        if frame.duplicated(
            [self.config.season_column, self.config.group_column, self.config.item_column]
        ).any():
            raise ValueError("each inference item may appear at most once inside an event")
        event_times = pd.to_datetime(frame[self.config.event_time_column], errors="coerce", utc=True)
        if enforce_future:
            training_max = pd.to_datetime(self.training_max_time, errors="raise", utc=True)
            if (event_times <= training_max).any():
                raise ValueError("inference events must be strictly later than all training events")

        values = self.encoder.transform(frame)
        if self.backend_name == "sklearn_pairwise":
            scores = self.estimator.predict(values)
        else:
            scores = np.asarray(self.estimator.predict(values), dtype=float)
        work = pd.DataFrame(
            {
                "__source_row": np.arange(len(frame), dtype=int),
                "__event": _event_token_series(frame, self.config).astype(str).to_numpy(),
                "__item": frame[self.config.item_column].astype(str).to_numpy(),
                "__score": scores,
            }
        )
        ranked = work.sort_values(
            ["__event", "__score", "__item", "__source_row"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        ranked["__rank"] = ranked.groupby("__event", sort=False).cumcount() + 1
        ranks = ranked.set_index("__source_row")["__rank"].reindex(range(len(frame))).to_numpy(dtype=int)
        output = pd.DataFrame(index=frame.index)
        output[self.config.group_column] = frame[self.config.group_column]
        output[self.config.item_column] = frame[self.config.item_column]
        output["ranking_score"] = scores
        output["predicted_rank"] = ranks
        output["ranking_model"] = self.model_name
        return output


@dataclass(frozen=True)
class GroupedRankingFitResult:
    """Fit outcome that remains serializable even when a backend is absent."""

    status: str
    model: GroupedRankingModel | None
    manifest: dict[str, Any]
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.status in {"available", "available_fallback"}

    def require_model(self) -> GroupedRankingModel:
        if self.model is None:
            raise RuntimeError(self.unavailable_reason or "grouped ranking model is unavailable")
        return self.model


def _backend_candidates(config: GroupedRankingConfig) -> tuple[str, ...]:
    requested = config.normalized_backend
    if requested == "auto":
        return ("xgboost_lambdarank", "lightgbm_lambdarank", "sklearn_pairwise")
    if config.allow_requested_backend_fallback and requested != "sklearn_pairwise":
        return (requested, "sklearn_pairwise")
    return (requested,)


def _fit_backend(
    backend: str,
    dataset: GroupedRankingDataset,
    config: GroupedRankingConfig,
) -> tuple[Any, dict[str, Any]]:
    if backend == "sklearn_pairwise":
        model = _SklearnPairwiseRanker(config).fit(dataset)
        assert model.pairwise_audit is not None
        return model, {
            "package": "scikit-learn",
            "version": _package_version("scikit-learn"),
            "pair_rows_by_event": model.pairwise_audit.pair_rows_by_event,
            "undirected_pairs_by_event": model.pairwise_audit.undirected_pairs_by_event,
            "cross_event_pair_count": model.pairwise_audit.cross_event_pair_count,
        }

    package = "xgboost" if backend == "xgboost_lambdarank" else "lightgbm"
    runtime = inspect_optional_model_runtime(package)
    if not runtime.available:
        raise RuntimeError(json.dumps(runtime.to_payload(), sort_keys=True))
    if backend == "xgboost_lambdarank":
        from xgboost import XGBRanker  # type: ignore

        model = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg",
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=int(config.random_state),
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(dataset.values, dataset.relevance, group=list(dataset.group_sizes), verbose=False)
    else:
        from lightgbm import LGBMRanker  # type: ignore

        max_relevance = int(dataset.relevance.max())
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            num_leaves=int(config.num_leaves),
            min_child_samples=1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            label_gain=list(range(max(2, max_relevance + 1))),
            random_state=int(config.random_state),
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(dataset.values, dataset.relevance, group=list(dataset.group_sizes))
    return model, {
        "package": package,
        "version": runtime.version,
        "runtime": runtime.to_payload(),
        "group_sizes": list(dataset.group_sizes),
        "cross_event_pair_count": 0,
    }


def _selection_history_manifest(
    selection_records: pd.DataFrame | None,
    *,
    training_dataset: GroupedRankingDataset,
    config: GroupedRankingConfig,
) -> dict[str, Any]:
    if selection_records is None:
        return {
            "status": "not_provided",
            "role": "hyperparameter_selection_only_not_model_fit",
            "event_count": 0,
        }
    selection = selection_records.copy()
    if selection.empty:
        raise ValueError("selection_records must be omitted rather than empty")
    required = {
        config.season_column,
        config.group_column,
        config.event_time_column,
        config.item_column,
        config.target_column,
        *config.feature_columns,
    }
    missing = sorted(required.difference(selection.columns))
    if missing:
        raise ValueError(f"selection history is missing required columns: {missing}")
    events = _event_metadata(selection, config)
    selection_tokens = set(events["__event_token"].astype(str))
    training_tokens = set(training_dataset.event_order)
    overlap = sorted(selection_tokens.intersection(training_tokens))
    if overlap:
        raise ValueError(f"selection and model-fit event groups must be disjoint: {overlap}")
    selection_max = pd.Timestamp(events["event_time"].max())
    training_min = pd.to_datetime(training_dataset.event_times[0], errors="raise", utc=True)
    if selection_max >= training_min:
        raise ValueError("hyperparameter-selection history must precede model-fit history")
    hash_columns = [
        config.season_column,
        config.group_column,
        config.event_time_column,
        config.item_column,
        config.target_column,
        *config.feature_columns,
    ]
    ordered = selection.copy()
    ordered["__event_token"] = _event_token_series(ordered, config)
    ordered["__item_token"] = ordered[config.item_column].astype(str)
    ordered = ordered.sort_values(["__event_token", "__item_token"], kind="mergesort")
    hashes = pd.util.hash_pandas_object(ordered[hash_columns], index=False, categorize=True)
    fingerprint = hashlib.sha256(hashes.to_numpy(dtype=np.uint64).tobytes()).hexdigest()
    return {
        "status": "provided_disjoint",
        "role": "hyperparameter_selection_only_not_model_fit",
        "rows": int(len(selection)),
        "event_count": int(len(events)),
        "seasons": sorted(pd.to_numeric(selection[config.season_column]).astype(int).unique().tolist()),
        "event_order": list(events["__event_token"].astype(str)),
        "max_time": selection_max.isoformat().replace("+00:00", "Z"),
        "data_sha256": fingerprint,
    }


def fit_grouped_ranking_challenger(
    records: pd.DataFrame,
    *,
    config: GroupedRankingConfig,
    cutoff: object | None = None,
    target_season: int | None = None,
    selection_records: pd.DataFrame | None = None,
) -> GroupedRankingFitResult:
    """Fit on same-season history or return an explicit unavailable result.

    ``selection_records`` may contain prior-season events used to freeze the
    hyperparameters, but those rows are audited as a separate disjoint history
    and are never passed to the ranker itself.
    """

    same_season, resolved_season, excluded_other_season_rows = _same_season_records(
        records, config, target_season
    )
    dataset = prepare_grouped_ranking_dataset(same_season, config=config, cutoff=cutoff)
    pair_audit = build_event_pure_pairwise_dataset(dataset)
    selection_manifest = _selection_history_manifest(
        selection_records,
        training_dataset=dataset,
        config=config,
    )
    base_manifest: dict[str, Any] = {
        "schema_version": "f1_grouped_ranking_manifest_v1",
        "status": "unavailable",
        "requested_backend": config.normalized_backend,
        "selected_backend": None,
        "fallback_used": False,
        "config": config.to_payload(),
        "config_sha256": config.fingerprint,
        "training_data_sha256": dataset.data_fingerprint,
        "training_rows": int(len(dataset.frame)),
        "training_event_count": int(len(dataset.event_order)),
        "training_season": resolved_season,
        "season_transfer_policy": "same_season_absolute_pace_only",
        "other_season_rows_excluded_from_fit": excluded_other_season_rows,
        "hyperparameter_selection_history": selection_manifest,
        "event_order": list(dataset.event_order),
        "event_times": list(dataset.event_times),
        "group_sizes": list(dataset.group_sizes),
        "feature_names": list(dataset.encoder.feature_names_out),
        "target_semantics": (
            "lower_target_is_better" if config.lower_target_is_better else "higher_target_is_better"
        ),
        "pairwise_audit": {
            "undirected_pairs_by_event": pair_audit.undirected_pairs_by_event,
            "pair_rows_by_event": pair_audit.pair_rows_by_event,
            "cross_event_pair_count": 0,
        },
        "backend_attempts": [],
    }
    attempts: list[dict[str, Any]] = []
    for backend in _backend_candidates(config):
        try:
            estimator, backend_manifest = _fit_backend(backend, dataset, config)
            attempts.append({"backend": backend, "status": "selected", **backend_manifest})
            fallback_used = backend != config.normalized_backend and config.normalized_backend != "auto"
            auto_fallback = config.normalized_backend == "auto" and backend == "sklearn_pairwise"
            status = "available_fallback" if fallback_used or auto_fallback else "available"
            manifest = dict(base_manifest)
            manifest.update(
                {
                    "status": status,
                    "selected_backend": backend,
                    "fallback_used": bool(fallback_used or auto_fallback),
                    "backend_attempts": attempts,
                }
            )
            model = GroupedRankingModel(
                config=config,
                backend_name=backend,
                estimator=estimator,
                encoder=dataset.encoder,
                training_event_order=dataset.event_order,
                training_max_time=dataset.event_times[-1],
                training_season=resolved_season,
                training_manifest=manifest,
            )
            return GroupedRankingFitResult(status=status, model=model, manifest=manifest)
        except Exception as exc:
            attempts.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}"[:3000],
                }
            )

    manifest = dict(base_manifest)
    manifest["backend_attempts"] = attempts
    reason = "; ".join(
        f"{attempt['backend']}: {attempt.get('reason', 'unavailable')}" for attempt in attempts
    )
    return GroupedRankingFitResult(
        status="unavailable",
        model=None,
        manifest=manifest,
        unavailable_reason=reason,
    )


__all__ = [
    "GROUPED_RANKING_BACKENDS",
    "ChronologicalGroupPartition",
    "EventPurePairwiseDataset",
    "GroupedFeatureEncoder",
    "GroupedRankingConfig",
    "GroupedRankingDataset",
    "GroupedRankingFitResult",
    "GroupedRankingModel",
    "build_event_pure_pairwise_dataset",
    "chronological_group_partition",
    "fit_grouped_ranking_challenger",
    "prepare_grouped_ranking_dataset",
    "same_season_walk_forward_partitions",
]
