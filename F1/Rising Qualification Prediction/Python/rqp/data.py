"""Data preparation and feature assembly."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .providers import BaseProvider


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
        rounds = provider.list_rounds(year)
        for rnd in rounds:
            round_number = int(rnd["round_number"])
            if year == target_year and round_number >= target_round:
                continue
            fp_features = provider.get_fp_features(year, round_number)
            if fp_features.empty:
                continue
            if mode == "qualifying":
                qualy = provider.get_qualifying_results(year, round_number)
                if qualy.empty or "q3_time" not in qualy.columns:
                    continue
                qualy = qualy.copy()
                qualy["q3_time"] = pd.to_numeric(qualy["q3_time"], errors="coerce")
                qualy = qualy.dropna(subset=["q3_time"])
                if qualy.empty:
                    continue
                best_q3 = qualy["q3_time"].min()
                qualy["target"] = qualy["q3_time"] - best_q3
                merged = fp_features.merge(qualy[["driver_id", "target"]], on="driver_id", how="inner")
                if merged.empty:
                    continue
                rows.append(merged)
            else:
                race = provider.get_race_results(year, round_number)
                qualy = provider.get_qualifying_results(year, round_number)
                if race.empty or qualy.empty:
                    continue
                qualy = qualy.copy()
                qualy["position"] = pd.to_numeric(qualy["position"], errors="coerce")
                merged = fp_features.merge(qualy[["driver_id", "position"]], on="driver_id", how="inner")
                merged = merged.rename(columns={"position": "qualy_position"})
                merged = merged.merge(race[["driver_id", "position"]], on="driver_id", how="inner")
                merged = merged.rename(columns={"position": "target"})
                if include_standings:
                    standings = provider.get_standings(year, round_number)
                    if standings is not None and not standings.empty:
                        merged = merged.merge(
                            standings[["driver_id", "position_start"]],
                            on="driver_id",
                            how="left",
                        )
                if merged.empty:
                    continue
                rows.append(merged)
    if not rows:
        notes.append("Pas assez de data historique: fallback heuristique.")
        return pd.DataFrame(), notes
    train = pd.concat(rows, ignore_index=True)
    return train, notes


def build_current_features(
    provider: BaseProvider,
    mode: str,
    year: int,
    round_number: int,
    include_standings: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    fp_features = provider.get_fp_features(year, round_number)
    if fp_features.empty:
        notes.append("Aucune donnee FP disponible pour ce round.")
        return pd.DataFrame(), notes
    if mode == "qualifying":
        return fp_features, notes
    qualy = provider.get_qualifying_results(year, round_number)
    if qualy.empty:
        notes.append("Resultats qualifications indisponibles: impossible de predire la course.")
        return pd.DataFrame(), notes
    qualy = qualy.copy()
    qualy["position"] = pd.to_numeric(qualy["position"], errors="coerce")
    merged = fp_features.merge(qualy[["driver_id", "position"]], on="driver_id", how="inner")
    merged = merged.rename(columns={"position": "qualy_position"})
    if include_standings:
        standings = provider.get_standings(year, round_number)
        if standings is not None and not standings.empty:
            merged = merged.merge(
                standings[["driver_id", "position_start"]],
                on="driver_id",
                how="left",
            )
    return merged, notes
