"""Past-only quote admission and immutable reference extraction; no model fitting.

CSV/legacy JSON bytes physically contain outcomes. Metadata preparation never
interprets those fields. Training interprets them only after temporal admission;
external scoring labels require a closed, hash-bound complete forecast ledger.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping
from zoneinfo import ZoneInfo

from research.experiments.boundary_20260908.football_information.model import devig, probabilities

ROOT = Path(__file__).resolve().parents[4]
LEAGUES = {"E0": "Europe/London", "I1": "Europe/Rome", "SP1": "Europe/Madrid"}
INTERNAL = ("production_default_dc_auto", "dc365_elo50", "dc_180", "shot_strength_90d_ridge0.1")
CALIBRATION_SEASONS = (2019, 2020, 2021)
HISTORY_DAYS = 1095
HALF_LIFE = 365.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utc(value):
    """Old reference datetimes without offsets explicitly mean UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("An ISO datetime or datetime is required")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _delay(extra_days):
    if type(extra_days) is not int or extra_days not in (1, 7):
        raise ValueError("Only the frozen one or seven extra local days are allowed")
    return extra_days


def midnight(day, league):
    if not isinstance(day, date) or isinstance(day, datetime) or league not in LEAGUES:
        raise ValueError("Canonical date and known league are required")
    return datetime.combine(day, time(), ZoneInfo(LEAGUES[league])).astimezone(timezone.utc)


def availability(day, league, extra_days):
    return midnight(day + timedelta(days=1 + _delay(extra_days)), league)


@dataclass(frozen=True)
class RawFixture:
    match_id: str
    league: str
    season: int
    home: str
    away: str
    day: date
    payload: Mapping
    source_path: str
    source_sha256: str
    source_row_number: int
    source_row_hash: str


@dataclass
class Archive:
    rows: tuple[RawFixture, ...]
    bindings: list[dict] = field(default_factory=list)
    # Populated only after the caller's metadata/clock admission. Reuse never
    # bypasses admission, even if a later query populated the cache first.
    _quotes: dict = field(default_factory=dict, repr=False)
    _metadata: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        ids = [row.match_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate archive identity")
        self.by_id = {row.match_id: row for row in self.rows}


def _bound(binding, root=ROOT):
    path = Path(binding["path"])
    if not path.is_absolute():
        path = Path(root) / path
    if sha(path) != binding["sha256"] or ("bytes" in binding and path.stat().st_size != binding["bytes"]):
        raise ValueError(f"Input binding mismatch: {path}")
    return path


def _bound_json(binding, root):
    return json.loads(_bound(binding, root).read_text())


def _read_csv(path, binding):
    match = re.fullmatch(r"(?:transfer_)?(E0|I1|SP1)_(20\d\d)_(20\d\d)\.csv", path.name)
    if not match or int(match[3]) != int(match[2]) + 1:
        raise ValueError("Unrecognized provider filename")
    league, season = match[1], int(match[2])
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for number, raw in enumerate(csv.DictReader(stream), start=2):
            if not raw.get("Date") and not raw.get("HomeTeam") and not raw.get("AwayTeam"):
                continue
            home, away = raw.get("HomeTeam"), raw.get("AwayTeam")
            if raw.get("Div") != league or not home or not away or home == away:
                raise ValueError("Invalid fixture metadata")
            for fmt in ("%d/%m/%Y", "%d/%m/%y"):
                try:
                    day = datetime.strptime(raw["Date"], fmt).date()
                    break
                except ValueError:
                    pass
            else:
                raise ValueError("Invalid canonical provider date")
            # Hash raw strings as provenance, without parsing any numeric field.
            rows.append(RawFixture(f"{league}:{season}:{home}:{away}", league, season,
                                   home, away, day, raw, binding["path"], binding["sha256"],
                                   number, digest(raw)))
    if len(rows) != 380 or len({team for r in rows for team in (r.home, r.away)}) != 20:
        raise ValueError("Frozen provider season population differs")
    appearances = {team: sum(team in (r.home, r.away) for r in rows)
                   for r in rows for team in (r.home, r.away)}
    if set(appearances.values()) != {38}:
        raise ValueError("Provider season is not the fixed complete double round robin")
    return rows


def load_archive(spec, root=ROOT):
    """Verify all specified bytes and parse only IDs, teams, dates and raw strings."""
    entries = spec["inputs"]["cached_csv_files"]
    manifests = {entry["path"]: _bound_json(entry, root)
                 for entry in spec["inputs"]["acquisition_manifests"]}
    rows, cells = [], set()
    for entry in entries:
        path = _bound(entry, root)
        original = [row for row in manifests[entry["acquisition_manifest"]]["files"]
                    if row["name"] == path.name]
        if len(original) != 1 or any(original[0][key] != entry[key] for key in ("sha256", "bytes", "url")):
            raise ValueError("CSV differs from its original acquisition receipt")
        loaded = _read_csv(path, entry)
        cell = (loaded[0].league, loaded[0].season)
        if cell in cells:
            raise ValueError("Duplicate league-season file")
        cells.add(cell)
        rows.extend(loaded)
    expected = {(league, season) for league in spec["leagues"] for season in spec["input_season_start_years"]}
    if cells != expected or len(rows) != spec["quotes"]["verified_rows"]:
        raise ValueError("Archive differs from the fixed league-season population")
    return Archive(tuple(rows), copy.deepcopy(spec["inputs"]["acquisition_manifests"] + entries))


def _metadata(row):
    return {"match_id": row.match_id, "league": row.league, "season": row.season,
            "day": row.day.isoformat(), "home": row.home, "away": row.away,
            "forecast_cutoff_utc": midnight(row.day, row.league).isoformat(),
            "a0_utc": midnight(row.day + timedelta(days=1), row.league).isoformat(),
            "availability_by_delay": {str(d): availability(row.day, row.league, d).isoformat() for d in (1, 7)},
            "source_path": row.source_path, "source_sha256": row.source_sha256,
            "source_row_number": row.source_row_number, "source_row_hash": row.source_row_hash}


def metadata_rows(archive):
    return [_metadata(row) for row in archive.rows]


def validate_reference_metadata(archive, fixtures):
    """Exact canonical source-day join before building any forecast inputs."""
    ids = [row["match_id"] for row in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference identity")
    for fixture in fixtures:
        raw = archive.by_id.get(fixture["match_id"])
        if raw is None:
            raise ValueError("Reference is absent from the fixed provider archive")
        meta = _metadata(raw)
        if any(fixture[key] != meta[key] for key in ("match_id", "league", "season", "day", "home", "away")):
            raise ValueError("Reference canonical metadata differs from provider archive")
        if utc(fixture["forecast_cutoff_utc"]) != utc(meta["forecast_cutoff_utc"]):
            raise ValueError("Reference issuance differs from original source-day midnight")
        if "result_available_at" in fixture and utc(fixture["result_available_at"]) != utc(meta["a0_utc"]):
            raise ValueError("Reference result proxy differs from provider archive")


def _quote(archive, row):
    """Private: call only after the relevant temporal admission guard."""
    if row.match_id not in archive._quotes:
        columns = ("BbAvH", "BbAvD", "BbAvA") if row.season < 2019 else ("AvgH", "AvgD", "AvgA")
        raw = [row.payload.get(key) for key in columns]
        try:
            if any(isinstance(value, bool) for value in raw):
                raise ValueError("Boolean odds")
            odds = [float(value) for value in raw]
            if any(not math.isfinite(value) or value <= 1 for value in odds):
                raise ValueError("Invalid decimal odds")
        except (ValueError, TypeError, OverflowError):
            archive._quotes[row.match_id] = None
        else:
            power, exponent = devig(odds, "power")
            normalized, _ = devig(odds, "normalized")
            archive._quotes[row.match_id] = {"odds": odds, "q": probabilities([power])[0].tolist(),
                                              "q_normalized": normalized.tolist(), "power": exponent}
    return copy.deepcopy(archive._quotes[row.match_id])


def _outcome(row):
    values = [row.payload.get(key) for key in ("FTHG", "FTAG")]
    if any(isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)) for value in values):
        raise ValueError(f"Malformed admitted goals: {row.match_id}")
    home, away = map(int, values)
    outcome = 0 if home > away else 1 if home == away else 2
    if row.payload.get("FTR") != "HDA"[outcome]:
        raise ValueError(f"Admitted outcome disagrees with goals: {row.match_id}")
    return outcome, [home, away]


def quote_rows(archive, cutoff, extra_days):
    """Rolling shared quote population without reading any outcome field."""
    cutoff, extra_days = utc(cutoff), _delay(extra_days)
    local_days = {league: cutoff.astimezone(ZoneInfo(zone)).date() for league, zone in LEAGUES.items()}
    accepted, excluded = [], []
    for raw in archive.rows:
        day = local_days[raw.league]
        # No payload access or payload validation above either temporal gate.
        if not day - timedelta(days=HISTORY_DAYS) <= raw.day < day:
            continue
        available = availability(raw.day, raw.league, extra_days)
        if available >= cutoff:
            continue
        quote = _quote(archive, raw)
        if quote is None:
            excluded.append({"match_id": raw.match_id, "reason": "invalid_or_missing_average_quote"})
            continue
        if raw.match_id not in archive._metadata:
            archive._metadata[raw.match_id] = _metadata(raw)
        meta = copy.deepcopy(archive._metadata[raw.match_id])
        age = (cutoff - utc(meta["a0_utc"])).total_seconds() / 86400.
        if age <= 0:
            raise ValueError("Admitted outcome proxy must strictly precede cutoff")
        accepted.append({**meta, **quote, "quote_available_at_utc": available.isoformat(),
                         "cutoff_utc": cutoff.isoformat(), "weight": 2. ** (-age / HALF_LIFE)})
    accepted.sort(key=lambda row: (row["quote_available_at_utc"], row["match_id"]))
    sums = {league: math.fsum(r["weight"] for r in accepted if r["league"] == league) for league in LEAGUES}
    diagnostics = {"cutoff_utc": cutoff.isoformat(), "extra_days": extra_days,
                   "rows": len(accepted), "accepted_ids_sha256": digest([r["match_id"] for r in accepted]),
                   "weight_sums_by_league": sums, "excluded_missing_quotes": excluded,
                   "history_days": HISTORY_DAYS, "half_life_elapsed_days": HALF_LIFE,
                   "outcomes_interpreted": False}
    return accepted, diagnostics


def training_rows(archive, cutoff, extra_days):
    rows, diagnostics = quote_rows(archive, cutoff, extra_days)
    for row in rows:
        row["outcome"], _ = _outcome(archive.by_id[row["match_id"]])
    diagnostics["outcomes_interpreted"] = True
    return rows, diagnostics


def support_for(fixtures, admitted_rows):
    teams = {(r["league"], team) for r in admitted_rows for team in (r["home"], r["away"])}
    return [(r["league"], r["home"]) in teams and (r["league"], r["away"]) in teams for r in fixtures]


class EloCursor:
    """Cumulative, nondecreasing UTC queries and atomic equal-availability updates."""
    def __init__(self, archive, extra_days):
        self.archive, self.extra_days = archive, _delay(extra_days)
        self.events = sorted(((availability(r.day, r.league, extra_days), r.match_id, r) for r in archive.rows),
                             key=lambda item: (item[0], item[1]))
        self.index, self.last_cutoff = 0, None
        self.ratings, self.last_updates = {}, {}
        self.updates, self.missing = [], []
        self._updates_digest = None

    def query(self, fixtures, cutoff):
        cutoff = utc(cutoff)
        if self.last_cutoff is not None and cutoff < self.last_cutoff:
            raise ValueError("Elo queries cannot move backward")
        self.last_cutoff = cutoff
        while self.index < len(self.events) and self.events[self.index][0] < cutoff:
            clock = self.events[self.index][0]
            batch = []
            while self.index < len(self.events) and self.events[self.index][0] == clock:
                batch.append(self.events[self.index][2]); self.index += 1
            changes, accepted = {}, []
            for raw in batch:
                quote = _quote(self.archive, raw)
                if quote is None:
                    self.missing.append(raw.match_id)
                    continue
                home, away = (raw.league, raw.home), (raw.league, raw.away)
                rh, ra = self.ratings.get(home, 1000.), self.ratings.get(away, 1000.)
                z = math.log(10.) * (rh - ra + 80.) / 400.
                expected = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                q = quote["q_normalized"]
                delta = 175. * (q[0] + .5*q[1] - expected)
                changes.setdefault(home, []).append(delta)
                changes.setdefault(away, []).append(-delta)
                accepted.append({"match_id": raw.match_id, "source_sha256": raw.source_sha256,
                                 "source_row_hash": raw.source_row_hash, "q_normalized": q,
                                 "rating_home_before": rh, "rating_away_before": ra, "delta": delta})
            for team, deltas in changes.items():
                self.ratings[team] = self.ratings.get(team, 1000.) + math.fsum(deltas)
                self.last_updates[team] = clock.isoformat()
            self.updates.append({"quote_available_at_utc": clock.isoformat(), "rows": accepted})
            self._updates_digest = None
        # Several fixtures commonly share an issuance clock. Their state digest
        # is identical; do not repeatedly serialize the entire update journal.
        if self._updates_digest is None:
            self._updates_digest = digest(self.updates)
        x, queried = [], []
        for fixture in fixtures:
            league = fixture["league"]
            if league not in LEAGUES:
                raise ValueError("Unknown league")
            home, away = (league, fixture["home"]), (league, fixture["away"])
            rh, ra = self.ratings.get(home, 1000.), self.ratings.get(away, 1000.)
            x.append((rh - ra) / 400.)
            queried.append({"match_id": fixture["match_id"], "rating_home": rh, "rating_away": ra,
                            "home_last_update_utc": self.last_updates.get(home),
                            "away_last_update_utc": self.last_updates.get(away)})
        return {"x": x, "provenance": {"cutoff_utc": cutoff.isoformat(), "extra_days": self.extra_days,
                "processed_raw_rows": self.index, "update_batches": len(self.updates),
                "updates_sha256": self._updates_digest, "missing_quote_ids": list(self.missing),
                "fixtures": queried}}


def calibration_rows(archive, extra_days, cutoff):
    """Frozen 2019–2021 readout labels with causal original-midnight Elo inputs."""
    cutoff, extra_days = utc(cutoff), _delay(extra_days)
    cursor = EloCursor(archive, extra_days)
    candidates = sorted((r for r in archive.rows if r.season in CALIBRATION_SEASONS
                         and availability(r.day, r.league, extra_days) < cutoff),
                        key=lambda r: (midnight(r.day, r.league), r.match_id))
    x, y, records, skipped = [], [], [], []
    for raw in candidates:
        # Form the explanatory variable at its ORIGINAL issuance. The current
        # fixture's quote and outcome are interpreted only at calibration cutoff.
        issue = midnight(raw.day, raw.league)
        snapshot = cursor.query([_metadata(raw)], issue)
        quote = _quote(archive, raw)
        if quote is None:
            skipped.append(raw.match_id)
            continue
        outcome, _ = _outcome(raw)
        value = snapshot["x"][0]
        x.append(value); y.append(outcome)
        records.append({**_metadata(raw), "x": value, "y": outcome,
                        "quote_available_at_utc": availability(raw.day, raw.league, extra_days).isoformat(),
                        "rating_provenance": snapshot["provenance"]})
    return x, y, {"extra_days": extra_days, "calibration_cutoff_utc": cutoff.isoformat(),
                   "season_start_years": list(CALIBRATION_SEASONS), "rows": records,
                   "excluded_missing_quote_ids": skipped, "elo_updates": cursor.updates}


def _vector(vector):
    if not isinstance(vector, list) or len(vector) != 3:
        raise ValueError("Expected unchanged HDA vector")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 or v > 1 for v in vector):
        raise ValueError("Invalid reference probability")
    if abs(math.fsum(vector)-1.) > 1e-10:
        raise ValueError("Invalid reference simplex")
    return list(vector)  # No floor, normalization or arithmetic on the original bits.


def _extract_references(original, xg, phase, contract, source_binding, xg_binding=None):
    rows = []
    if len(original) != contract["rows"]:
        raise ValueError("Reference population count differs")
    ids = [row["match_id"] for row in original]
    if len(set(ids)) != len(ids) or digest(ids) != contract["ordered_original_match_ids_sha256"]:
        raise ValueError("Reference identities/order differ")
    if phase == "selection" and (xg is None or [r["match_id"] for r in xg] != ids):
        raise ValueError("xG safeguard identities/order differ")
    for number, raw in enumerate(original):
        league, season, home, away = (raw[key] for key in ("league", "season", "home", "away"))
        day = date.fromisoformat(raw["day"])
        if league not in LEAGUES or season not in contract["season_start_years"] or raw["match_id"] != f"{league}:{season}:{home}:{away}":
            raise ValueError("Reference fixture metadata differs")
        if phase == "selection" and league != contract["league"]:
            raise ValueError("Selection league differs")
        if utc(raw["forecast_cutoff_utc"]) != midnight(day, league) or utc(raw["result_available_at"]) != midnight(day+timedelta(days=1), league):
            raise ValueError("Reference source-day clock contract differs")
        row = {key: raw[key] for key in ("match_id", "league", "season", "day", "home", "away",
                                        "forecast_cutoff_utc", "fit_cutoff_utc", "fit_id", "result_available_at")}
        row["probabilities"] = {name: _vector(raw["probabilities"][name]) for name in INTERNAL}
        row["reference_source"] = {**source_binding, "row_index": number}
        if phase == "selection":
            saved = xg[number]
            if any(saved[key] != raw[key] for key in ("forecast_cutoff_utc", "fit_cutoff_utc")):
                raise ValueError("xG safeguard issuance/fit clocks differ")
            row["probabilities"]["xg_add90_selection_safeguard"] = _vector(saved["probabilities"]["xg_add90"])
            row["xg_reference_source"] = {**xg_binding, "row_index": number}
        rows.append(row)
    if len({utc(r["forecast_cutoff_utc"]) for r in rows}) != contract["unique_forecast_clocks"]:
        raise ValueError("Reference forecast clock count differs")
    expected = {(league, season) for league in ([contract["league"]] if phase == "selection" else LEAGUES)
                for season in contract["season_start_years"]}
    counts = {(r["league"], r["season"]): 0 for r in rows}
    for row in rows:
        counts[row["league"], row["season"]] += 1
    if set(counts) != expected or any(n != contract.get("rows_each_league_season", 380) for n in counts.values()):
        raise ValueError("Reference league-season population differs")
    return rows


def reference_rows(spec, phase, root=ROOT):
    if phase not in ("selection", "transfer"):
        raise ValueError("Unknown fixed population")
    inputs = spec["inputs"]
    verification = _bound_json(inputs["football_verification"], root)
    source = inputs["football_selection" if phase == "selection" else "football_transfer"]
    original = _bound_json(source, root)
    key = "selection_sha256" if phase == "selection" else "evaluation_sha256"
    if verification["status"] != "passed" or verification[key] != source["sha256"]:
        raise ValueError("Original reference verification does not bind these bytes")
    xg = None
    if phase == "selection":
        verified = _bound_json(inputs["xg_verification"], root)
        locked = _bound_json(inputs["xg_selection_lock"], root)
        if verified["status"] != "passed" or locked["selected"] != "xg_add90":
            raise ValueError("Invalid fixed xG safeguard selection")
        for name in ("xg_selection_lock", "xg_selection_result", "xg_selection_issued"):
            binding = inputs[name]
            _bound(binding, root)
            if verified["bindings"].get(binding["path"]) != binding["sha256"]:
                raise ValueError("xG verification does not bind the safeguard")
        xg = _bound_json(inputs["xg_selection_issued"], root)
    return _extract_references(original["predictions"], xg, phase, spec[phase], source,
                               inputs.get("xg_selection_issued"))


def attach_labels(issued_rows, archive, *, forecast_closure_path):
    """Scoring-only attachment after verifying the complete saved probability list."""
    closure = json.loads(Path(forecast_closure_path).read_text())
    if closure.get("labels_attached") is not False or closure.get("rows") != len(issued_rows):
        raise ValueError("A target-free full forecast closure is required")
    saved = _bound_json(closure["forecasts"], ROOT)
    if not isinstance(saved, list) or canonical(saved) != canonical(issued_rows):
        raise ValueError("Issued rows differ from the closed forecast ledger")
    ids = [r["match_id"] for r in issued_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate issued identity")
    validate_reference_metadata(archive, issued_rows)
    for issued in issued_rows:
        if any(key in issued for key in ("label", "outcome", "goals", "y")):
            raise ValueError("Forecast ledger already contains scoring labels")
        if not issued.get("probabilities"):
            raise ValueError("Every closed fixture requires probability forecasts")
        for vector in issued["probabilities"].values():
            _vector(vector)
    # Validate the entire population and all vectors before reading any label.
    result = []
    for issued in issued_rows:
        raw = archive.by_id[issued["match_id"]]
        outcome, goals = _outcome(raw)
        result.append({**copy.deepcopy(issued), "label": outcome, "goals": goals})
    return result
