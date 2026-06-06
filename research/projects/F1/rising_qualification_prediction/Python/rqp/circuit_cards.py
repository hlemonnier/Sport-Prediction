"""Circuit-card priors and feature helpers for F1 track specificity.

The cards are deliberately numeric because the prediction stack consumes tabular
features. Static priors give every known circuit a usable shape before a weekend;
provider-derived track stats then refine the card when local historical data is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd

from .utils import normalize_event_name


def _clip01(value: object, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(max(0.0, min(1.0, float(numeric))))


def _key(value: object) -> str:
    text = normalize_event_name(value)
    if not text:
        return ""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalize_event_name(ascii_text)


def _slug(value: object) -> str:
    text = _key(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown_circuit"


@dataclass(frozen=True)
class CircuitCard:
    card_id: str
    canonical_name: str
    archetype: str
    aliases: tuple[str, ...]
    downforce_demand: float
    power_sensitivity: float
    low_speed_corner_demand: float
    high_speed_corner_demand: float
    traction_demand: float
    braking_demand: float
    tyre_degradation: float
    kerb_bump_penalty: float
    drs_effectiveness: float
    overtaking_difficulty: float
    qualifying_importance: float
    strategy_variance: float
    safety_car_probability: float
    data_reliability: float = 0.55

    def numeric_features(self) -> dict[str, float]:
        return {
            "circuit_downforce_demand": self.downforce_demand,
            "circuit_power_sensitivity": self.power_sensitivity,
            "circuit_low_speed_corner_demand": self.low_speed_corner_demand,
            "circuit_high_speed_corner_demand": self.high_speed_corner_demand,
            "circuit_traction_demand": self.traction_demand,
            "circuit_braking_demand": self.braking_demand,
            "circuit_tyre_degradation": self.tyre_degradation,
            "circuit_kerb_bump_penalty": self.kerb_bump_penalty,
            "circuit_drs_effectiveness": self.drs_effectiveness,
            "circuit_overtaking_difficulty": self.overtaking_difficulty,
            "circuit_qualifying_importance": self.qualifying_importance,
            "circuit_strategy_variance": self.strategy_variance,
            "circuit_safety_car_probability": self.safety_car_probability,
            "circuit_card_reliability": self.data_reliability,
        }

    def metadata(self) -> dict[str, str]:
        return {
            "circuit_card_id": self.card_id,
            "circuit_name": self.canonical_name,
            "circuit_archetype": self.archetype,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self.metadata(), **self.numeric_features()}


def _card(
    card_id: str,
    name: str,
    archetype: str,
    aliases: Iterable[str],
    values: tuple[float, float, float, float, float, float, float, float, float, float, float, float, float],
) -> CircuitCard:
    (
        downforce,
        power,
        low_speed,
        high_speed,
        traction,
        braking,
        tyre,
        kerb,
        drs,
        overtake_difficulty,
        quali_importance,
        strategy_variance,
        safety_car,
    ) = values
    return CircuitCard(
        card_id=card_id,
        canonical_name=name,
        archetype=archetype,
        aliases=tuple(_key(alias) for alias in aliases),
        downforce_demand=downforce,
        power_sensitivity=power,
        low_speed_corner_demand=low_speed,
        high_speed_corner_demand=high_speed,
        traction_demand=traction,
        braking_demand=braking,
        tyre_degradation=tyre,
        kerb_bump_penalty=kerb,
        drs_effectiveness=drs,
        overtaking_difficulty=overtake_difficulty,
        qualifying_importance=quali_importance,
        strategy_variance=strategy_variance,
        safety_car_probability=safety_car,
    )


CIRCUIT_CARDS: tuple[CircuitCard, ...] = (
    _card("bahrain", "Bahrain International Circuit", "traction_braking_tyre", ("bahrain grand prix", "bahrain gp", "sakhir grand prix"), (0.55, 0.72, 0.62, 0.42, 0.88, 0.82, 0.86, 0.45, 0.70, 0.38, 0.55, 0.70, 0.35)),
    _card("jeddah", "Jeddah Corniche Circuit", "high_speed_street_power", ("saudi arabian grand prix", "saudi arabia grand prix", "jeddah grand prix"), (0.52, 0.90, 0.30, 0.88, 0.46, 0.58, 0.45, 0.62, 0.75, 0.48, 0.62, 0.72, 0.70)),
    _card("albert_park", "Albert Park", "balanced_street", ("australian grand prix", "australia grand prix"), (0.55, 0.66, 0.48, 0.62, 0.55, 0.58, 0.46, 0.52, 0.64, 0.52, 0.62, 0.56, 0.50)),
    _card("suzuka", "Suzuka Circuit", "high_speed_aero", ("japanese grand prix", "japan grand prix"), (0.88, 0.62, 0.36, 0.96, 0.55, 0.50, 0.58, 0.58, 0.45, 0.66, 0.72, 0.46, 0.30)),
    _card("shanghai", "Shanghai International Circuit", "balanced_front_limited", ("chinese grand prix", "china grand prix"), (0.62, 0.68, 0.52, 0.60, 0.66, 0.70, 0.64, 0.35, 0.64, 0.44, 0.58, 0.58, 0.34)),
    _card("miami", "Miami International Autodrome", "power_traction", ("miami grand prix", "miami gp"), (0.50, 0.82, 0.58, 0.44, 0.72, 0.72, 0.56, 0.48, 0.72, 0.45, 0.56, 0.58, 0.38)),
    _card("imola", "Autodromo Enzo e Dino Ferrari", "old_school_aero_braking", ("emilia romagna grand prix", "imola grand prix", "san marino grand prix"), (0.76, 0.58, 0.58, 0.72, 0.62, 0.74, 0.50, 0.74, 0.42, 0.74, 0.78, 0.45, 0.36)),
    _card("monaco", "Circuit de Monaco", "street_max_downforce", ("monaco grand prix", "monaco gp"), (1.00, 0.12, 1.00, 0.22, 0.86, 0.52, 0.32, 0.95, 0.08, 1.00, 0.96, 0.48, 0.68)),
    _card("villeneuve", "Circuit Gilles Villeneuve", "power_braking_stop_go", ("canadian grand prix", "canada grand prix"), (0.38, 0.90, 0.48, 0.30, 0.72, 0.95, 0.52, 0.72, 0.78, 0.35, 0.52, 0.72, 0.55)),
    _card("barcelona", "Circuit de Barcelona-Catalunya", "balanced_aero_tyre", ("spanish grand prix", "spain grand prix"), (0.78, 0.58, 0.48, 0.78, 0.62, 0.54, 0.78, 0.45, 0.48, 0.68, 0.72, 0.50, 0.25)),
    _card("red_bull_ring", "Red Bull Ring", "power_braking_short_lap", ("austrian grand prix", "austria grand prix", "styrian grand prix"), (0.42, 0.86, 0.42, 0.56, 0.72, 0.82, 0.44, 0.44, 0.82, 0.32, 0.48, 0.54, 0.28)),
    _card("silverstone", "Silverstone Circuit", "high_speed_aero_power", ("british grand prix", "great britain grand prix", "70th anniversary grand prix"), (0.82, 0.78, 0.28, 0.96, 0.44, 0.42, 0.62, 0.36, 0.62, 0.46, 0.58, 0.50, 0.34)),
    _card("hungaroring", "Hungaroring", "high_downforce_low_overtake", ("hungarian grand prix", "hungary grand prix"), (0.92, 0.28, 0.92, 0.42, 0.76, 0.58, 0.56, 0.54, 0.28, 0.88, 0.88, 0.45, 0.32)),
    _card("spa", "Circuit de Spa-Francorchamps", "power_high_speed_variable", ("belgian grand prix", "belgium grand prix"), (0.70, 0.94, 0.24, 0.92, 0.44, 0.62, 0.44, 0.50, 0.78, 0.30, 0.48, 0.68, 0.45)),
    _card("zandvoort", "Circuit Zandvoort", "high_downforce_technical", ("dutch grand prix", "netherlands grand prix"), (0.90, 0.42, 0.74, 0.74, 0.66, 0.50, 0.50, 0.60, 0.32, 0.82, 0.84, 0.48, 0.40)),
    _card("monza", "Autodromo Nazionale Monza", "temple_of_speed", ("italian grand prix", "italy grand prix", "monza grand prix"), (0.12, 1.00, 0.22, 0.26, 0.38, 0.86, 0.42, 0.46, 0.94, 0.34, 0.50, 0.48, 0.30)),
    _card("baku", "Baku City Circuit", "street_power_braking", ("azerbaijan grand prix", "baku grand prix", "european grand prix"), (0.42, 0.94, 0.72, 0.42, 0.72, 0.82, 0.38, 0.82, 0.86, 0.42, 0.56, 0.82, 0.74)),
    _card("marina_bay", "Marina Bay Street Circuit", "street_high_downforce_chaos", ("singapore grand prix", "singapore gp"), (0.96, 0.26, 0.96, 0.30, 0.82, 0.74, 0.66, 0.88, 0.20, 0.86, 0.84, 0.86, 0.86)),
    _card("cota", "Circuit of the Americas", "balanced_aero_tyre", ("united states grand prix", "usa grand prix", "austin grand prix"), (0.72, 0.72, 0.54, 0.82, 0.64, 0.70, 0.72, 0.58, 0.68, 0.44, 0.56, 0.62, 0.34)),
    _card("mexico_city", "Autodromo Hermanos Rodriguez", "altitude_downforce_traction", ("mexico city grand prix", "mexican grand prix", "mexico grand prix"), (0.82, 0.70, 0.72, 0.48, 0.82, 0.74, 0.50, 0.62, 0.62, 0.48, 0.62, 0.62, 0.42)),
    _card("interlagos", "Interlagos", "balanced_bumpy_sprint", ("sao paulo grand prix", "brazilian grand prix", "brazil grand prix"), (0.66, 0.76, 0.58, 0.66, 0.72, 0.72, 0.62, 0.72, 0.72, 0.34, 0.50, 0.72, 0.50)),
    _card("las_vegas", "Las Vegas Strip Circuit", "low_grip_power_braking", ("las vegas grand prix", "vegas grand prix"), (0.28, 0.96, 0.54, 0.30, 0.80, 0.84, 0.38, 0.46, 0.92, 0.30, 0.48, 0.72, 0.55)),
    _card("losail", "Lusail International Circuit", "high_speed_tyre", ("qatar grand prix", "qatar gp"), (0.82, 0.66, 0.34, 0.90, 0.58, 0.48, 0.86, 0.40, 0.56, 0.62, 0.70, 0.58, 0.28)),
    _card("yas_marina", "Yas Marina Circuit", "balanced_traction_power", ("abu dhabi grand prix", "abu dhabi gp"), (0.58, 0.76, 0.62, 0.48, 0.76, 0.76, 0.52, 0.44, 0.78, 0.46, 0.58, 0.58, 0.34)),
    _card("paul_ricard", "Circuit Paul Ricard", "power_tyre_balanced", ("french grand prix", "france grand prix"), (0.58, 0.82, 0.42, 0.70, 0.58, 0.64, 0.68, 0.26, 0.76, 0.42, 0.54, 0.52, 0.24)),
    _card("portimao", "Autodromo Internacional do Algarve", "elevation_aero_tyre", ("portuguese grand prix", "portugal grand prix"), (0.72, 0.70, 0.52, 0.78, 0.64, 0.58, 0.66, 0.52, 0.62, 0.48, 0.58, 0.56, 0.30)),
    _card("istanbul", "Istanbul Park", "high_speed_aero_tyre", ("turkish grand prix", "turkey grand prix"), (0.78, 0.70, 0.46, 0.86, 0.62, 0.58, 0.72, 0.46, 0.62, 0.48, 0.58, 0.62, 0.34)),
    _card("sochi", "Sochi Autodrom", "power_street_smooth", ("russian grand prix", "russia grand prix"), (0.42, 0.86, 0.50, 0.44, 0.62, 0.70, 0.36, 0.30, 0.78, 0.42, 0.56, 0.50, 0.34)),
    _card("nurburgring", "Nurburgring GP-Strecke", "balanced_old_school", ("eifel grand prix", "german grand prix"), (0.66, 0.66, 0.62, 0.62, 0.64, 0.70, 0.58, 0.56, 0.58, 0.58, 0.64, 0.54, 0.36)),
    _card("mugello", "Mugello Circuit", "high_speed_aero", ("tuscan grand prix", "mugello grand prix"), (0.86, 0.70, 0.34, 0.94, 0.52, 0.50, 0.66, 0.60, 0.46, 0.70, 0.76, 0.56, 0.34)),
)


_ALIAS_INDEX: dict[str, CircuitCard] = {}
for _card_item in CIRCUIT_CARDS:
    _ALIAS_INDEX[_key(_card_item.canonical_name)] = _card_item
    _ALIAS_INDEX[_key(_card_item.card_id.replace("_", " "))] = _card_item
    for _alias in _card_item.aliases:
        _ALIAS_INDEX[_alias] = _card_item


def _lookup_static_card(event_name: object) -> Optional[CircuitCard]:
    normalized = _key(event_name)
    if not normalized:
        return None
    direct = _ALIAS_INDEX.get(normalized)
    if direct is not None:
        return direct
    for alias, card in _ALIAS_INDEX.items():
        if alias and (alias in normalized or normalized in alias):
            return card
    return None


def generic_circuit_card(event_name: object) -> CircuitCard:
    normalized = _key(event_name) or "unknown circuit"
    return CircuitCard(
        card_id=_slug(normalized),
        canonical_name=str(event_name or "Unknown circuit"),
        archetype="generic_balanced",
        aliases=(normalized,),
        downforce_demand=0.55,
        power_sensitivity=0.55,
        low_speed_corner_demand=0.55,
        high_speed_corner_demand=0.55,
        traction_demand=0.55,
        braking_demand=0.55,
        tyre_degradation=0.55,
        kerb_bump_penalty=0.45,
        drs_effectiveness=0.55,
        overtaking_difficulty=0.55,
        qualifying_importance=0.60,
        strategy_variance=0.55,
        safety_car_probability=0.35,
        data_reliability=0.20,
    )


def _blend(prior: float, observed: Optional[float], reliability: float, max_observed_weight: float = 0.65) -> float:
    if observed is None:
        return _clip01(prior, default=0.5)
    weight = _clip01(reliability, default=0.0) * float(max_observed_weight)
    return _clip01((float(prior) * (1.0 - weight)) + (float(observed) * weight), default=prior)


def _stat(stats: Optional[dict[str, object]], key: str) -> Optional[float]:
    if not isinstance(stats, dict):
        return None
    if key not in stats:
        return None
    value = pd.to_numeric(pd.Series([stats.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return _clip01(value, default=0.0)


def circuit_card_from_event(event_name: object, track_stats: Optional[dict[str, object]] = None) -> CircuitCard:
    base = _lookup_static_card(event_name) or generic_circuit_card(event_name)
    reliability = _stat(track_stats, "track_stats_reliability")
    if reliability is None:
        reliability = 0.0

    mobility = _stat(track_stats, "track_finish_order_mobility")
    if mobility is None:
        mobility = _stat(track_stats, "track_overtake_propensity")
    grid_stability = _stat(track_stats, "track_grid_stability")
    safety = _stat(track_stats, "track_safety_car_propensity")
    chaos = _stat(track_stats, "track_chaos_index")
    pit_intensity = _stat(track_stats, "track_pit_stop_intensity")

    observed_overtaking_difficulty = None if mobility is None else 1.0 - mobility
    observed_qualifying_importance = None
    if mobility is not None or grid_stability is not None:
        mobility_part = 1.0 - mobility if mobility is not None else base.overtaking_difficulty
        grid_part = grid_stability if grid_stability is not None else base.qualifying_importance
        observed_qualifying_importance = (0.55 * mobility_part) + (0.45 * grid_part)

    observed_tyre = None
    if pit_intensity is not None:
        observed_tyre = _clip01(pit_intensity / 2.5, default=base.tyre_degradation)

    return replace(
        base,
        overtaking_difficulty=_blend(base.overtaking_difficulty, observed_overtaking_difficulty, reliability),
        qualifying_importance=_blend(base.qualifying_importance, observed_qualifying_importance, reliability),
        safety_car_probability=_blend(base.safety_car_probability, safety, reliability),
        strategy_variance=_blend(base.strategy_variance, chaos, reliability),
        tyre_degradation=_blend(base.tyre_degradation, observed_tyre, reliability, max_observed_weight=0.35),
        data_reliability=max(float(base.data_reliability), _clip01(reliability, default=0.0)),
    )


def attach_circuit_card(frame: pd.DataFrame, event_name: object, track_stats: Optional[dict[str, object]]) -> pd.DataFrame:
    if frame.empty:
        return frame
    card = circuit_card_from_event(event_name, track_stats=track_stats)
    out = frame.copy()
    for key, value in card.metadata().items():
        out[key] = value
    for key, value in card.numeric_features().items():
        out[key] = float(value)
    return out


def circuit_card_payload_from_frame(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    first = frame.iloc[0]
    keys = [
        "circuit_card_id",
        "circuit_name",
        "circuit_archetype",
        *CIRCUIT_NUMERIC_FEATURES,
    ]
    payload: dict[str, object] = {}
    for key in keys:
        if key not in frame.columns:
            continue
        value = first.get(key)
        if pd.isna(value):
            continue
        if key in CIRCUIT_NUMERIC_FEATURES:
            payload[key] = float(value)
        else:
            payload[key] = str(value)
    return payload


CIRCUIT_NUMERIC_FEATURES = [
    "circuit_downforce_demand",
    "circuit_power_sensitivity",
    "circuit_low_speed_corner_demand",
    "circuit_high_speed_corner_demand",
    "circuit_traction_demand",
    "circuit_braking_demand",
    "circuit_tyre_degradation",
    "circuit_kerb_bump_penalty",
    "circuit_drs_effectiveness",
    "circuit_overtaking_difficulty",
    "circuit_qualifying_importance",
    "circuit_strategy_variance",
    "circuit_safety_car_probability",
    "circuit_card_reliability",
]


CIRCUIT_INTERACTION_FEATURES = [
    "fp_weighted_delta_downforce_adj",
    "fp_weighted_delta_power_adj",
    "fp_quali_sim_delta_downforce_adj",
    "fp_race_sim_delta_tyre_adj",
    "fp_race_sim_delta_power_adj",
    "qualy_position_circuit_importance_adj",
    "qualy_pred_position_circuit_importance_adj",
    "circuit_fit_index",
    "driver_archetype_form_3_fp_weighted_delta",
    "team_archetype_form_3_fp_weighted_delta",
    "driver_circuit_hist_fp_weighted_delta",
    "team_circuit_hist_fp_weighted_delta",
]
