"""Data preparation and feature assembly."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .providers import BaseProvider
from .utils import normalize_event_name, team_column


def _round_event_name(round_meta: dict[str, object], round_number: int) -> str:
    for key in ("event_name", "meeting_name", "country_name"):
        value = round_meta.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"Round {round_number}"


def _event_name_for_round(
    provider: BaseProvider,
    year: int,
    round_number: int,
    notes: List[str],
) -> str:
    try:
        rounds = provider.list_rounds(year)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec listing rounds {year}: {exc}")
        return f"Round {round_number}"
    for rnd in rounds:
        try:
            if int(rnd.get("round_number", 0)) == round_number:
                return _round_event_name(rnd, round_number)
        except Exception:
            continue
    return f"Round {round_number}"


def _ensure_fp_mean_delta(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "fp_mean_delta" not in out.columns:
        delta_cols = [c for c in out.columns if c.endswith("_delta")]
        if delta_cols:
            out["fp_mean_delta"] = out[delta_cols].apply(pd.to_numeric, errors="coerce").mean(
                axis=1,
                skipna=True,
            )
        else:
            out["fp_mean_delta"] = float("nan")
    else:
        out["fp_mean_delta"] = pd.to_numeric(out["fp_mean_delta"], errors="coerce")
    return out


def _add_temporal_features_train(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = _ensure_fp_mean_delta(frame)
    if "driver_id" in out.columns:
        out["driver_id"] = out["driver_id"].astype(str)
    out["event_name"] = out.get("event_name", pd.Series(index=out.index, dtype=object))
    out["event_name_norm"] = out["event_name"].map(normalize_event_name)

    sort_cols = [c for c in ["event_key", "event_year", "event_round", "driver_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    driver_group = out.groupby("driver_id", sort=False)["fp_mean_delta"] if "driver_id" in out.columns else None
    if driver_group is not None:
        out["driver_ewma_fp_mean_delta"] = driver_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["driver_form_3_fp_mean_delta"] = driver_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
        out["driver_form_5_fp_mean_delta"] = driver_group.transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean(),
        )
    else:
        out["driver_ewma_fp_mean_delta"] = float("nan")
        out["driver_form_3_fp_mean_delta"] = float("nan")
        out["driver_form_5_fp_mean_delta"] = float("nan")

    team_col = team_column(out)
    if team_col:
        out[team_col] = out[team_col].astype(str)
        team_group = out.groupby(team_col, sort=False)["fp_mean_delta"]
        out["team_ewma_fp_mean_delta"] = team_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["team_form_3_fp_mean_delta"] = team_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
        out["team_form_5_fp_mean_delta"] = team_group.transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean(),
        )
    else:
        out["team_ewma_fp_mean_delta"] = float("nan")
        out["team_form_3_fp_mean_delta"] = float("nan")
        out["team_form_5_fp_mean_delta"] = float("nan")

    if "driver_id" in out.columns:
        out["event_driver_hist_idx"] = out.groupby(
            ["event_name_norm", "driver_id"],
            sort=False,
        )["fp_mean_delta"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean(),
        )
    else:
        out["event_driver_hist_idx"] = float("nan")

    return out


def _attach_temporal_features_current(
    current: pd.DataFrame,
    history: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if current.empty:
        return current
    out = _ensure_fp_mean_delta(current)
    out["driver_id"] = out["driver_id"].astype(str)
    out["event_name"] = out.get("event_name", pd.Series(index=out.index, dtype=object))
    out["event_name_norm"] = out["event_name"].map(normalize_event_name)

    temporal_cols = [
        "driver_ewma_fp_mean_delta",
        "driver_form_3_fp_mean_delta",
        "driver_form_5_fp_mean_delta",
        "team_ewma_fp_mean_delta",
        "team_form_3_fp_mean_delta",
        "team_form_5_fp_mean_delta",
        "event_driver_hist_idx",
    ]
    for col in temporal_cols:
        out[col] = float("nan")

    if history is None or history.empty or "driver_id" not in history.columns:
        return out

    hist = _add_temporal_features_train(history)
    hist = hist.copy()
    hist["driver_id"] = hist["driver_id"].astype(str)
    sort_cols = [c for c in ["event_key", "event_year", "event_round"] if c in hist.columns]
    if sort_cols:
        hist = hist.sort_values(sort_cols, kind="mergesort")

    driver_last = hist.groupby("driver_id", sort=False).tail(1).set_index("driver_id")
    for col in [
        "driver_ewma_fp_mean_delta",
        "driver_form_3_fp_mean_delta",
        "driver_form_5_fp_mean_delta",
    ]:
        if col in driver_last.columns:
            out[col] = out["driver_id"].map(driver_last[col])

    hist_team_col = team_column(hist)
    current_team_col = team_column(out)
    if hist_team_col and current_team_col:
        hist[hist_team_col] = hist[hist_team_col].astype(str)
        out[current_team_col] = out[current_team_col].astype(str)
        team_last = hist.groupby(hist_team_col, sort=False).tail(1).set_index(hist_team_col)
        for col in [
            "team_ewma_fp_mean_delta",
            "team_form_3_fp_mean_delta",
            "team_form_5_fp_mean_delta",
        ]:
            if col in team_last.columns:
                out[col] = out[current_team_col].map(team_last[col])

    hist["event_name_norm"] = hist.get("event_name", pd.Series(index=hist.index, dtype=object)).map(
        normalize_event_name,
    )
    track_driver = hist.groupby(["event_name_norm", "driver_id"], sort=False)["fp_mean_delta"].mean()
    keys = list(zip(out["event_name_norm"], out["driver_id"]))
    out["event_driver_hist_idx"] = [track_driver.get(key, float("nan")) for key in keys]
    driver_mean = hist.groupby("driver_id", sort=False)["fp_mean_delta"].mean()
    out["event_driver_hist_idx"] = out["event_driver_hist_idx"].fillna(out["driver_id"].map(driver_mean))

    return out


def build_training_data(
    provider: BaseProvider,
    mode: str,
    train_seasons: List[int],
    target_year: int,
    target_round: int,
    include_standings: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[pd.DataFrame] = []
    notes: List[str] = []
    for year in train_seasons:
        try:
            rounds = provider.list_rounds(year)
        except (Exception, SystemExit) as exc:
            notes.append(f"Echec listing rounds {year}: {exc}")
            continue
        rounds_sorted = sorted(rounds, key=lambda r: int(r.get("round_number", 0)))
        for rnd in rounds_sorted:
            round_number = int(rnd.get("round_number", 0))
            event_name = _round_event_name(rnd, round_number)
            if year == target_year and round_number >= target_round:
                continue
            try:
                fp_features = provider.get_fp_features(year, round_number)
            except (Exception, SystemExit) as exc:
                notes.append(f"Echec FP {year} round {round_number}: {exc}")
                continue
            if fp_features.empty:
                continue
            fp_features = fp_features.copy()
            if "driver_id" in fp_features.columns:
                fp_features["driver_id"] = fp_features["driver_id"].astype(str)
            if mode == "qualifying":
                try:
                    qualy = provider.get_qualifying_results(year, round_number)
                except (Exception, SystemExit) as exc:
                    notes.append(f"Echec qualifs {year} round {round_number}: {exc}")
                    continue
                if qualy.empty or "q3_time" not in qualy.columns:
                    continue
                qualy = qualy.copy()
                qualy["driver_id"] = qualy["driver_id"].astype(str)
                qualy["q3_time"] = pd.to_numeric(qualy["q3_time"], errors="coerce")
                qualy = qualy.dropna(subset=["q3_time"])
                if qualy.empty:
                    continue
                best_q3 = qualy["q3_time"].min()
                qualy["target"] = qualy["q3_time"] - best_q3
                merged = fp_features.merge(qualy[["driver_id", "target"]], on="driver_id", how="inner")
                if merged.empty:
                    continue
                merged["event_name"] = event_name
                merged["event_year"] = year
                merged["event_round"] = round_number
                merged["event_key"] = (year * 100) + round_number
                rows.append(merged)
            else:
                try:
                    race = provider.get_race_results(year, round_number)
                    qualy = provider.get_qualifying_results(year, round_number)
                except (Exception, SystemExit) as exc:
                    notes.append(f"Echec race/qualifs {year} round {round_number}: {exc}")
                    continue
                if race.empty or qualy.empty:
                    continue
                qualy = qualy.copy()
                race = race.copy()
                qualy["driver_id"] = qualy["driver_id"].astype(str)
                race["driver_id"] = race["driver_id"].astype(str)
                qualy["position"] = pd.to_numeric(qualy["position"], errors="coerce")
                merged = fp_features.merge(qualy[["driver_id", "position"]], on="driver_id", how="inner")
                merged = merged.rename(columns={"position": "qualy_position"})
                merged = merged.merge(race[["driver_id", "position"]], on="driver_id", how="inner")
                merged = merged.rename(columns={"position": "target"})
                if include_standings:
                    try:
                        standings = provider.get_standings(year, round_number)
                    except (Exception, SystemExit) as exc:
                        notes.append(f"Echec standings {year} round {round_number}: {exc}")
                        standings = None
                    if standings is not None and not standings.empty:
                        standings = standings.copy()
                        standings["driver_id"] = standings["driver_id"].astype(str)
                        merged = merged.merge(
                            standings[["driver_id", "position_start"]],
                            on="driver_id",
                            how="left",
                        )
                if merged.empty:
                    continue
                merged["event_name"] = event_name
                merged["event_year"] = year
                merged["event_round"] = round_number
                merged["event_key"] = (year * 100) + round_number
                rows.append(merged)
    if not rows:
        notes.append("Pas assez de data historique: fallback heuristique.")
        return pd.DataFrame(), notes
    train = pd.concat(rows, ignore_index=True)
    train = _add_temporal_features_train(train)
    return train, notes


def build_current_features(
    provider: BaseProvider,
    mode: str,
    year: int,
    round_number: int,
    include_standings: bool,
    history: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    try:
        fp_features = provider.get_fp_features(year, round_number)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec recuperation FP: {exc}")
        return pd.DataFrame(), notes
    if fp_features.empty:
        notes.append("Aucune donnee FP disponible pour ce round.")
        return pd.DataFrame(), notes
    fp_features = fp_features.copy()
    fp_features["driver_id"] = fp_features["driver_id"].astype(str)
    event_name = _event_name_for_round(provider, year, round_number, notes)
    if mode == "qualifying":
        fp_features["event_name"] = event_name
        fp_features["event_year"] = year
        fp_features["event_round"] = round_number
        fp_features["event_key"] = (year * 100) + round_number
        current = _attach_temporal_features_current(fp_features, history)
        return current, notes
    try:
        qualy = provider.get_qualifying_results(year, round_number)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec recuperation qualifications: {exc}")
        return pd.DataFrame(), notes
    if qualy.empty:
        notes.append("Resultats qualifications indisponibles: impossible de predire la course.")
        return pd.DataFrame(), notes
    qualy = qualy.copy()
    qualy["driver_id"] = qualy["driver_id"].astype(str)
    qualy["position"] = pd.to_numeric(qualy["position"], errors="coerce")
    merged = fp_features.merge(qualy[["driver_id", "position"]], on="driver_id", how="inner")
    merged = merged.rename(columns={"position": "qualy_position"})
    if include_standings:
        try:
            standings = provider.get_standings(year, round_number)
        except (Exception, SystemExit) as exc:
            notes.append(f"Echec recuperation standings: {exc}")
            standings = None
        if standings is not None and not standings.empty:
            standings = standings.copy()
            standings["driver_id"] = standings["driver_id"].astype(str)
            merged = merged.merge(
                standings[["driver_id", "position_start"]],
                on="driver_id",
                how="left",
            )
    merged["event_name"] = event_name
    merged["event_year"] = year
    merged["event_round"] = round_number
    merged["event_key"] = (year * 100) + round_number
    current = _attach_temporal_features_current(merged, history)
    return current, notes
