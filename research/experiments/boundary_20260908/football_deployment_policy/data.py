"""Hash-bound, metadata-first inputs for the fixed deployment-policy proof.

Provider/legacy artifact bytes physically contain exposed outcomes. Loading
establishes their metadata only; numeric goals are interpreted after block
admission, or after the complete external forecast closure for scoring labels.
No fitting, forecasting, scoring or filesystem mutation occurs in this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping
from zoneinfo import ZoneInfo

from packages.football.mrp.data import MatchRecord, FixtureRecord
from packages.football.mrp.protocol import chronological_populations

ROOT = Path(__file__).resolve().parents[4]
LEAGUES = {"E0": "Europe/London", "I1": "Europe/Rome", "SP1": "Europe/Madrid"}
METADATA = ("match_id", "league", "season", "day", "home", "away",
            "forecast_cutoff_utc", "fit_cutoff_utc", "fit_id", "result_available_at")
REFERENCE = "production_default_dc_auto"
CANDIDATE = "dc_equal"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def utc(value):
    """Legacy naive clocks mean UTC, never the host's local timezone."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("A valid datetime is required")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def midnight(day, league):
    if type(day) is not date or league not in LEAGUES:
        raise ValueError("A local calendar date and declared league are required")
    return datetime.combine(day, time(), ZoneInfo(LEAGUES[league])).astimezone(timezone.utc)


def legacy_clock(value):
    return utc(value).replace(tzinfo=None).isoformat()


@dataclass(frozen=True)
class RawFixture:
    match_id: str
    league: str
    season: int
    day: date
    home: str
    away: str
    source_path: str
    source_sha256: str
    source_row_number: int
    source_row_hash: str

    def metadata(self):
        value = asdict(self)
        value["day"] = self.day.isoformat()
        value["forecast_cutoff_utc"] = legacy_clock(midnight(self.day, self.league))
        value["result_available_at"] = legacy_clock(midnight(self.day + timedelta(days=1), self.league))
        return value


@dataclass
class Inputs:
    spec: dict
    root: Path
    archive: tuple[RawFixture, ...]
    fixture_rows: list[dict]
    blocks: list[dict]
    block_inventory: list[dict]
    bindings: dict[str, str]
    inherited_sources: dict[str, str]
    _payloads: Mapping = field(repr=False)

    def __post_init__(self):
        self.by_id = {r.match_id: r for r in self.archive}
        if len(self.by_id) != len(self.archive):
            raise ValueError("Duplicate raw fixture identity")


def _bound(item, root, bindings):
    path = root / item["path"]
    actual = sha(path)
    if actual != item["sha256"] or ("bytes" in item and path.stat().st_size != item["bytes"]):
        raise ValueError("Closed input changed: " + item["path"])
    if item["path"] in bindings and bindings[item["path"]] != actual:
        raise ValueError("Conflicting input hashes")
    bindings[item["path"]] = actual
    return path


def _json(item, root, bindings):
    return json.loads(_bound(item, root, bindings).read_text())


def read_csv_metadata(path, binding, *, expected_rows=380, expected_teams=20):
    """Read raw strings and metadata, without interpreting goals or odds."""
    parsed = re.fullmatch(r"(?:transfer_)?(E0|I1|SP1)_(20\d\d)_(20\d\d)\.csv", Path(path).name)
    if parsed is None or int(parsed[3]) != int(parsed[2]) + 1:
        raise ValueError("Unexpected canonical provider filename")
    league, year = parsed[1], int(parsed[2])
    rows, payloads = [], {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for number, raw in enumerate(csv.DictReader(stream), start=2):
            if not any(raw.values()):
                continue
            if None in raw or raw.get("Div") != league:
                raise ValueError("Malformed provider schema or league")
            home, away = raw.get("HomeTeam", "").strip(), raw.get("AwayTeam", "").strip()
            if not home or not away or home == away:
                raise ValueError("Invalid provider teams")
            d = raw.get("Date", "")
            day = datetime.strptime(d, "%d/%m/%Y" if len(d.split("/")[-1]) == 4 else "%d/%m/%y").date()
            if not date(year, 7, 1) <= day <= date(year + 1, 8, 31):
                raise ValueError("Provider date outside its inherited season window")
            mid = f"{league}:{year}:{home}:{away}"
            if mid in payloads:
                raise ValueError("Duplicate source fixture identity")
            payloads[mid] = raw
            rows.append(RawFixture(mid, league, year, day, home, away, binding["path"],
                                   binding["sha256"], number, digest(raw)))
    counts = Counter(team for row in rows for team in (row.home, row.away))
    if len(rows) != expected_rows or len(counts) != expected_teams or set(counts.values()) != {2 * (expected_teams - 1)}:
        raise ValueError("Provider file is not the fixed complete season population")
    return rows, payloads


def _vector(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Expected an unchanged H/D/A triple")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in value):
        raise ValueError("H/D/A values must be positive finite numbers")
    if abs(math.fsum(value) - 1.) > 1e-10:
        raise ValueError("H/D/A probabilities must already be normalized")
    return list(value)  # Never normalize or transform an archived reference.


def fixture_row(raw):
    """Whitelist fields: legacy goals, labels and shot payloads are not read."""
    row = {key: raw[key] for key in METADATA}
    row["probabilities"] = {name: _vector(raw["probabilities"][name]) for name in (REFERENCE, CANDIDATE)}
    return row


def inventory_record(league, feature_path, index, fit, fixture_ids):
    base, pop = fit["base"], fit["production_default"]["populations"]
    return {
        "league": league, "feature_artifact": feature_path, "feature_fit_index": index,
        "fit_id": base["fit_id"], "fit_cutoff_utc": base["cutoff_utc"],
        "full_admitted_rows": len(base["fit_match_ids"]),
        "full_admitted_ids_sha256": base["fit_ids_sha256"],
        "full_admitted_content_sha256": base["fit_content_sha256"],
        "saved_dc_equal_parameters_sha256": digest(base["dixon_coles"][CANDIDATE]),
        "prefix_partition_sha256": digest(pop), "forecast_rows": len(fixture_ids),
        "ordered_forecast_ids_sha256": digest(fixture_ids),
        "prefix_fit_rows": pop["fit"]["sample_size"], "calibration_rows": pop["calibration"]["sample_size"],
        "historical_selection_rows": pop["selection"]["sample_size"],
        "historical_test_rows": pop["test"]["sample_size"],
    }


def _fixture_metadata_check(row, source):
    metadata = source.metadata()
    for key in ("match_id", "league", "season", "day", "home", "away", "forecast_cutoff_utc", "result_available_at"):
        if row[key] != metadata[key]:
            raise ValueError("Original fixture metadata differs from raw source: " + key)


def admitted_metadata(inputs, descriptor):
    """Decide the complete block population before touching any goal value."""
    league, cutoff = descriptor["league"], utc(descriptor["fit_cutoff_utc"])
    day = cutoff.astimezone(ZoneInfo(LEAGUES[league])).date()
    if midnight(day, league) != cutoff:
        raise ValueError("Block cutoff must be the inherited local midnight")
    lower = day - timedelta(days=inputs.spec["timing"]["history_days"])
    rows = [r for r in inputs.archive if r.league == league and lower <= r.day < day
            and midnight(r.day, league) < cutoff and midnight(r.day + timedelta(days=1), league) <= cutoff]
    return sorted(rows, key=lambda r: (r.day, r.match_id))


def load_inputs(spec, root=ROOT):
    """Validate the exact closed graph and reconstruct target-free metadata."""
    root, spec, bindings = Path(root), deepcopy(spec), {}
    inp = spec["inputs"]
    _bound(spec["proposal"], root, bindings)
    for group in ("canonical_global_caller_sources",):
        for item in inp[group]:
            _bound(item, root, bindings)
    for key in ("runtime_gap", "runtime_scoreline_diagnostic"):
        _bound(inp[key], root, bindings)
    evaluation = _json(inp["old_evaluation"], root, bindings)
    verification = _json(inp["old_verification"], root, bindings)
    design = _json(inp["old_design_lock"], root, bindings)
    sources = evaluation["source_files"]
    sm = inp["inherited_source_map"]
    if len(sources) != sm["entries"] or digest(sources) != sm["canonical_sha256"]:
        raise ValueError("Original source map differs")
    if verification["status"] != "passed" or verification["evaluation_sha256"] != inp["old_evaluation"]["sha256"] or verification["source_sha256"] != digest(sources):
        raise ValueError("Original evaluation lacks its exact passed verification")
    if evaluation["source_sha256"] != digest(sources) or evaluation["spec_sha256"] != design["spec_sha256"]:
        raise ValueError("Original evaluation source/design closure differs")
    for path, expected in sources.items():
        _bound({"path": path, "sha256": expected}, root, bindings)
    manifests = [_json(item, root, bindings) for item in inp["acquisition_manifests"]]
    acquired = {row["name"]: row for manifest in manifests for row in manifest["files"]}
    if len(acquired) != sum(len(m["files"]) for m in manifests):
        raise ValueError("Duplicate acquisition file")
    features = {league: _json(item, root, bindings) for league, item in inp["feature_artifacts"].items()}
    csv_files = {}
    for league, feature in features.items():
        path = inp["feature_artifacts"][league]["path"]
        maps = [x for x in inp["cached_csv_maps"] if x["artifact"] == path]
        if len(maps) != 1 or len(feature["input_hashes"]) != maps[0]["entries"] or digest(feature["input_hashes"]) != maps[0]["canonical_sha256"]:
            raise ValueError("Original provider hash map differs")
        if feature["source_files"] != sources or feature["source_sha256"] != digest(sources) or feature["spec_sha256"] != evaluation["spec_sha256"]:
            raise ValueError("Original feature source lineage differs")
        if evaluation["feature_artifacts_sha256"][league] != inp["feature_artifacts"][league]["sha256"]:
            raise ValueError("Feature artifact is not bound by the evaluation")
        for filename, expected in feature["input_hashes"].items():
            old = acquired.get(Path(filename).name)
            if old is None or old["sha256"] != expected or filename in csv_files:
                raise ValueError("CSV acquisition and feature hashes disagree")
            csv_files[filename] = {"path": filename, "sha256": expected, "bytes": old["bytes"]}
    if len(csv_files) != inp["cached_csv_total"]:
        raise ValueError("Incomplete raw file inventory")
    archive, payloads = [], {}
    for item in csv_files.values():
        rows, raw = read_csv_metadata(_bound(item, root, bindings), item)
        archive.extend(rows)
        if set(payloads).intersection(raw):
            raise ValueError("Duplicated archive identity")
        payloads.update(raw)
    cells = Counter((r.league, r.season) for r in archive)
    expected_cells = {(league, year) for league in LEAGUES for year in range(2017, 2026)}
    if set(cells) != expected_cells or set(cells.values()) != {380}:
        raise ValueError("The complete 27-season archive is required")
    fixture_rows = [fixture_row(row) for row in evaluation["predictions"]]
    ids = [r["match_id"] for r in fixture_rows]
    population = spec["population"]
    if len(ids) != population["rows"] or len(set(ids)) != len(ids) or digest(ids) != population["ordered_original_match_ids_sha256"]:
        raise ValueError("The complete ordered original fixture population is required")
    expected_fixtures = {(league, year) for league in population["countries"] for year in population["season_start_years"]}
    counts = Counter((r["league"], r["season"]) for r in fixture_rows)
    if set(counts) != expected_fixtures or set(counts.values()) != {population["rows_each_country_season"]}:
        raise ValueError("Original country-season fixture counts differ")
    by_id = {r["match_id"]: r for r in fixture_rows}
    blocks, inventory, seen = [], [], set()
    for league, feature in features.items():
        local_rows = [fixture_row(row) for row in feature["rows"]]
        for row in local_rows:
            if row["league"] != league or by_id.get(row["match_id"]) != row or row["match_id"] in seen:
                raise ValueError("Feature and evaluation fixture identity/vector differ")
            seen.add(row["match_id"])
        for index, fit in enumerate(feature["fits"]):
            base = fit["base"]
            batch = [row for row in local_rows if row["fit_id"] == base["fit_id"]]
            if not batch or any(row["fit_cutoff_utc"] != base["cutoff_utc"] or utc(row["forecast_cutoff_utc"]) < utc(base["cutoff_utc"]) for row in batch):
                raise ValueError("Invalid original block attribution or clock")
            if min(utc(row["forecast_cutoff_utc"]) for row in batch) != utc(base["cutoff_utc"]):
                raise ValueError("Original block does not start at its first fixture")
            record = inventory_record(league, inp["feature_artifacts"][league]["path"], index, fit, [r["match_id"] for r in batch])
            inventory.append(record)
            blocks.append({"block_id": f"{league}:{base['fit_id']}", "league": league, "fit_id": base["fit_id"],
                "fit_cutoff_utc": base["cutoff_utc"], "fixture_metadata": batch, "inventory": record,
                "saved_full_model": deepcopy(base["dixon_coles"][CANDIDATE]),
                "expected_partitions": deepcopy(fit["production_default"]["populations"]),
                "saved_full_lineage": {key: deepcopy(base[key]) for key in ("fit_match_ids", "fit_ids_sha256", "fit_content_sha256", "cutoff_utc", "latest_result_available_at")},
                "input_sources": deepcopy(feature["input_hashes"])})
            blocks[-1]["saved_full_lineage"].update(
                goal_state_sha256=record["saved_dc_equal_parameters_sha256"],
                source_artifact=inp["feature_artifacts"][league]["path"],
                source_artifact_sha256=inp["feature_artifacts"][league]["sha256"])
    key = lambda x: (x["fit_cutoff_utc"], x["league"], x["fit_id"])
    blocks.sort(key=key); inventory.sort(key=key)
    if seen != set(ids) or len(blocks) != population["fit_blocks"] or len({b["block_id"] for b in blocks}) != len(blocks):
        raise ValueError("Incomplete or duplicated block/fixture attribution")
    if Counter(b["league"] for b in blocks) != Counter({league: population["fit_blocks_each_country"] for league in population["countries"]}):
        raise ValueError("Each country must retain its fixed block count")
    if digest(inventory) != population["block_inventory_binding"]["complete_derived_inventory_sha256"]:
        raise ValueError("Frozen complete block inventory differs")
    result = Inputs(spec, root, tuple(archive), fixture_rows, blocks, inventory, bindings, dict(sources), payloads)
    for row in fixture_rows:
        _fixture_metadata_check(row, result.by_id[row["match_id"]])
    for block in blocks:
        _check_history_ids(block, admitted_metadata(result, block))
    return result


def _check_history_ids(descriptor, admitted):
    ids = [r.match_id for r in admitted]
    saved = descriptor["saved_full_lineage"]
    if saved["cutoff_utc"] != descriptor["fit_cutoff_utc"] or saved["goal_state_sha256"] != digest(descriptor["saved_full_model"]):
        raise ValueError("Saved goal-state or cutoff lineage differs")
    if ids != saved["fit_match_ids"] or digest(ids) != saved["fit_ids_sha256"]:
        raise ValueError("Independently reconstructed full history identities differ")
    if any(r["match_id"] in set(ids) for r in descriptor["fixture_metadata"]):
        raise ValueError("An issued fixture entered its block history")


def metadata_rows(inputs):
    return [r.metadata() for r in sorted(inputs.archive, key=lambda r: (r.league, r.day, r.match_id))]


def _goals(payload):
    values = []
    for key in ("FTHG", "FTAG"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Goal values must be nonnegative integers")
        if isinstance(value, str) and not re.fullmatch(r"\s*\+?\d+\s*", value):
            raise ValueError("Goal values must be nonnegative integers")
        value = int(value)
        if value < 0:
            raise ValueError("Negative goals")
        values.append(value)
    outcome = 0 if values[0] > values[1] else 1 if values[0] == values[1] else 2
    if payload["FTR"] != "HDA"[outcome]:
        raise ValueError("Provider result and goal values disagree")
    return values, outcome


def match_record(raw, goals):
    return MatchRecord(raw.match_id, midnight(raw.day, raw.league), raw.season,
                       "epl" if raw.league == "E0" else raw.league, None, raw.home, raw.away,
                       *goals, None, None, result_available_at=midnight(raw.day + timedelta(days=1), raw.league))


def records_sha256(matches):
    encoded = json.dumps([asdict(m) for m in matches], default=lambda value: value.isoformat(),
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_block(inputs, descriptor):
    """Materialize numeric training values only after exact metadata admission."""
    admitted = admitted_metadata(inputs, descriptor)
    _check_history_ids(descriptor, admitted)
    history, compact = [], []
    for raw in admitted:
        goals, _ = _goals(inputs._payloads[raw.match_id])
        history.append(match_record(raw, goals))
        compact.append([raw.match_id, raw.day.isoformat(), *goals])
    if digest(compact) != descriptor["saved_full_lineage"]["fit_content_sha256"]:
        raise ValueError("Admitted goal content differs from the closed original fit")
    if max((m.result_available_at.isoformat() for m in history), default=None) != descriptor["saved_full_lineage"]["latest_result_available_at"]:
        raise ValueError("Latest admitted result clock differs")
    if chronological_populations(history).metadata() != descriptor["expected_partitions"]:
        raise ValueError("Original prefix/calibration/selection/test partitions differ")
    block = deepcopy(descriptor)
    block["saved_full_lineage"]["match_records_sha256"] = records_sha256(history)
    block["history"] = history
    block["history_metadata"] = [r.metadata() for r in admitted]
    block["fixtures"] = [FixtureRecord(r["match_id"], utc(r["forecast_cutoff_utc"]), r["season"],
        "epl" if r["league"] == "E0" else r["league"], None, r["home"], r["away"]) for r in block["fixture_metadata"]]
    block["expected_reference_probabilities"] = [r["probabilities"][REFERENCE] for r in block["fixture_metadata"]]
    block["expected_candidate_probabilities"] = [r["probabilities"][CANDIDATE] for r in block["fixture_metadata"]]
    return block


def prepare_blocks(spec, root=ROOT, *, inputs=None):
    inputs = load_inputs(spec, root) if inputs is None else inputs
    return [prepare_block(inputs, block) for block in inputs.blocks]


def attach_labels(issued_rows, inputs, *, forecast_closure_path):
    """Attach targets only to an exact, closed full original forecast ledger."""
    closure = json.loads(Path(forecast_closure_path).read_text())
    if closure["labels_attached"] is not False or type(closure["rows"]) is not int or closure["rows"] != len(inputs.fixture_rows):
        raise ValueError("The complete target-free forecast closure is required")
    saved = _json(closure["forecasts"], inputs.root, {})
    ids = [r["match_id"] for r in inputs.fixture_rows]
    if saved != issued_rows or [r["match_id"] for r in issued_rows] != ids:
        raise ValueError("Closed forecasts do not preserve the original ordered population")
    for row, fixture in zip(issued_rows, inputs.fixture_rows, strict=True):
        if any(key in row for key in ("label", "goals", "outcome", "y")) or any(row[key] != fixture[key] for key in METADATA):
            raise ValueError("Forecast metadata or label-free status differs")
    labeled = deepcopy(issued_rows)
    for row in labeled:
        goals, label = _goals(inputs._payloads[row["match_id"]])
        row.update(goals=goals, label=label)
    return labeled
