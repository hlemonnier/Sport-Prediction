"""Prefix-causal raw completed-sector history; no canonical lap reconstruction.

The unit-increment packet closes the *old* epoch before creating the next one.
Only its S3/LastLapTime and previously received S1/S2 can support valid history.
All output snapshots are immutable with respect to later packets and revisions.
"""
from collections import Counter
from copy import deepcopy
import math

from research.experiments.boundary_20260908.sector_pilot.ledger import records

SUM_TOLERANCE_SECONDS = .003
FLOAT_SLACK_SECONDS = 1e-9
LATE_WINDOW_MS = 5000
STREAMS = ("SessionStatus", "TrackStatus", "TimingAppData")


def iter_packets(path, stats=None):
    """Immutable pilot metadata: sequence, recorded/effective ms, error, payload."""
    yield from records(path, Counter() if stats is None else stats)


def _items(value):
    if isinstance(value, dict):
        return value.items()
    return enumerate(value) if isinstance(value, list) else ()


def _duration(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        parts = str(value).split(":")
        if not 1 <= len(parts) <= 3:
            return None
        numbers = [float(p) for p in parts]
        if any(not math.isfinite(p) or p < 0 for p in numbers):
            return None
        if len(parts) > 1 and any(p >= 60 for p in numbers[1:]):
            return None
        result = 0.
        for number in numbers:
            result = 60 * result + number
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _counter(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or int(value) != value:
        return None
    return int(value)


def _source(value, packet):
    return {"value": value, "packet_sequence": packet["sequence"],
            "available_ms": packet["available_ms"], "recorded_ms": packet["recorded_ms"],
            "timestamp_regressed": packet["regressed"]}


def _sector_patch(patch):
    result = {}
    for key, value in _items(patch.get("Sectors", {})):
        if str(key) in ("0", "1", "2") and isinstance(value, dict) and "Value" in value:
            result[int(key) + 1] = value["Value"]
    return result


class _Auxiliary:
    def __init__(self, kind, path):
        self.kind = kind
        self.stats = Counter()
        self.iterator = iter(iter_packets(path, self.stats))
        self.next = next(self.iterator, None)
        self.current = None
        self.tyres = {}
        self.consumed_errors = 0

    def before(self, instant):
        # Unknown clocks have no cross-stream placement. Stop at them rather
        # than backdating errors or later values into an earlier forecast.
        while (self.next is not None and self.next["recorded_ms"] is not None
               and self.next["available_ms"] < instant):
            packet = self.next
            self.consumed_errors += int(packet["error"] is not None)
            if self.kind in ("SessionStatus", "TrackStatus"):
                if "Status" in packet["payload"]:
                    self.current = _source(str(packet["payload"]["Status"]), packet)
            else:
                for driver, patch in _items(packet["payload"].get("Lines", {})):
                    if not isinstance(patch, dict):
                        continue
                    stints = self.tyres.setdefault(str(driver), {})
                    for key, change in _items(patch.get("Stints", {})):
                        if not isinstance(change, dict) or not str(key).isdigit():
                            continue
                        state = stints.setdefault(int(key), {})
                        for field in ("Compound", "New", "StartLaps", "TotalLaps", "LapNumber"):
                            if field in change:
                                state[field] = _source(deepcopy(change[field]), packet)
            yield packet
            self.next = next(self.iterator, None)

    def tyre(self, driver):
        stints = self.tyres.get(driver, {})
        key = max(stints) if stints else None
        return {"highest_observed_stint_index": key,
                "fields": deepcopy(stints[key]) if key is not None else {},
                "active_stint_certified": False, "canonical_tyre_life_certified": False}


def _observe_control(epoch, kind, source):
    if source is None:
        epoch["quality"].add("session_status_unknown" if kind == "SessionStatus" else "track_status_unknown")
        return
    epoch["controls"][kind].append(deepcopy(source))
    if source["timestamp_regressed"]:
        epoch["ambiguity"].add("regressed_auxiliary_timestamp")
    value = source["value"]
    if kind == "SessionStatus":
        if value != "Started":
            epoch["quality"].add("not_started_during_epoch")
    elif value not in ("1", "2"):
        epoch["quality"].add("neutralization_observed" if any(s in value for s in "4567") else "track_status_unknown")


def _observe_pit(epoch, fields, changed=None):
    names = ("InPit", "PitOut", "Retired", "Stopped") if changed is None else changed
    for name in names:
        source = fields.get(name)
        if source is not None:
            epoch["pit_observations"].append({"field": name, **deepcopy(source)})
            if source["timestamp_regressed"]:
                epoch["ambiguity"].add("regressed_pit_timestamp")
        value = source["value"] if source is not None else None
        if name == "InPit":
            if value is True:
                epoch["quality"].add("pit_observed")
            elif value is not False:
                epoch["quality"].add("pit_state_unknown")
        elif value is True:
            epoch["quality"].add("pit_out_observed" if name == "PitOut" else "retirement_or_stoppage_observed")


def _new_epoch(state, packet, auxiliaries, reasons=()):
    epoch = {"generation": state["generation"], "counter": state["counter"],
             "start_ms": packet["available_ms"], "start_sequence": packet["sequence"],
             "boundary_known": state["counter"] is not None, "sectors": {}, "first": {},
             "quality": set(), "ambiguity": set(reasons), "repeats": Counter(), "revisions": Counter(),
             "clears": Counter(), "controls": {"SessionStatus": [], "TrackStatus": []},
             "pit_observations": [], "tyre_observations": []}
    if state["quarantine"]:
        epoch["ambiguity"].add("counter_quarantine")
    for name in ("SessionStatus", "TrackStatus"):
        _observe_control(epoch, name, auxiliaries[name].current)
    if any(a.consumed_errors for a in auxiliaries.values()):
        epoch["ambiguity"].add("auxiliary_parse_gap")
    _observe_pit(epoch, state["fields"], ("InPit", "Retired", "Stopped"))
    # PitOut is an observed event, not a forever-persistent contamination flag.
    # A fresh boundary-packet event conservatively belongs to both adjacent
    # epochs; an older event cannot taint every subsequent otherwise clean lap.
    pit_out = state["fields"].get("PitOut")
    if pit_out is not None and pit_out["packet_sequence"] == packet["sequence"]:
        _observe_pit(epoch, state["fields"], ("PitOut",))
    epoch["tyre_observations"].append(auxiliaries["TimingAppData"].tyre(state["driver"]))
    return epoch


def _update_sector(epoch, sector, raw, packet, *, collision=False):
    previous = epoch["sectors"].get(sector)
    value = _duration(raw)
    source = _source(value, packet)
    if raw in ("", None):
        epoch["clears"][str(sector)] += 1
        if previous is not None and previous["value"] is not None:
            epoch["ambiguity"].add("sector_cleared_after_positive")
    elif value is None:
        epoch["ambiguity"].add("invalid_sector_value")
    else:
        if previous is not None and previous["value"] is not None:
            if previous["value"] == value:
                epoch["repeats"][str(sector)] += 1
            else:
                epoch["revisions"][str(sector)] += 1
                epoch["ambiguity"].add("sector_revision_within_epoch")
        if sector not in epoch["first"]:
            epoch["first"][sector] = deepcopy(source)
        if not epoch["boundary_known"]:
            epoch["ambiguity"].add("unknown_counter")
        elif packet["available_ms"] - epoch["start_ms"] <= LATE_WINDOW_MS:
            epoch["ambiguity"].add("sector_within_five_seconds_of_boundary")
        if collision:
            epoch["ambiguity"].add("s1_s2_in_counter_boundary_packet")
    if packet["regressed"]:
        epoch["ambiguity"].add("raw_timestamp_regression")
    epoch["sectors"][sector] = source


def _complete_fields(record):
    values = record["sectors_seconds"]
    full = record["full_lap_seconds"]
    positive = all(v is not None and math.isfinite(v) and v > 0 for v in values)
    residual = math.fsum(values) - full if positive and full is not None else None
    coherent = residual is not None and abs(residual) <= SUM_TOLERANCE_SECONDS + FLOAT_SLACK_SECONDS
    record["sum_residual_seconds"] = residual
    record["support"]["all_three_positive_sectors"] = positive
    record["support"]["positive_full_lap"] = full is not None
    record["support"]["sector_sum_coherent"] = coherent
    reasons = set(record["quality_reasons"])
    if not positive:
        reasons.add("missing_positive_sector")
    if full is None:
        reasons.add("missing_positive_last_lap_time")
    if positive and full is not None and not coherent:
        reasons.add("sector_sum_inconsistent")
    record["quality_reasons"] = sorted(reasons)
    record["observed_valid_completed"] = (record["record_kind"] == "unit_counter_completion"
                                           and not record["quality_reasons"] and not record["ambiguity_reasons"])
    return record


def _record(event, driver, epoch, packet, previous, current, kind, sector3, lap, reasons=()):
    sources = [deepcopy(epoch["sectors"].get(s)) for s in (1, 2)]
    sources.append(_source(_duration(sector3), packet) if sector3 is not None else None)
    full = _duration(lap)
    ambiguity = set(epoch["ambiguity"]) | set(reasons)
    first = [epoch["first"].get(s) for s in (1, 2)]
    ordered = (all(s is not None and s["value"] is not None for s in sources[:2])
               and all(s is not None for s in first)
               and first[0]["packet_sequence"] < first[1]["packet_sequence"] < packet["sequence"])
    if not ordered:
        ambiguity.add("no_ordered_prepacket_s1_s2")
    if packet["regressed"]:
        ambiguity.add("raw_timestamp_regression")
    record = {"event_key": event, "driver": driver,
              "completion_id": f"{event}:{driver}:{packet['sequence']}:{kind}",
              "record_kind": kind, "available_ms": packet["available_ms"],
              "recorded_ms": packet["recorded_ms"], "packet_sequence": packet["sequence"],
              "previous_counter": previous, "new_counter": current,
              "observed_counter": current, "local_epoch": epoch["generation"],
              "epoch_start_ms": epoch["start_ms"], "epoch_start_sequence": epoch["start_sequence"],
              "sectors_seconds": [s["value"] if s is not None else None for s in sources],
              "sector_sources": sources, "full_lap_seconds": full,
              "full_lap_source": _source(full, packet) if lap is not None else None,
              "quality_reasons": sorted(epoch["quality"]), "ambiguity_reasons": sorted(ambiguity),
              "support": {"ordered_prepacket_s1_s2": bool(ordered),
                          "known_epoch_counter": epoch["counter"] is not None,
                          "unit_counter_advance": previous is not None and current == previous + 1,
                          "s3_same_boundary_packet": _duration(sector3) is not None,
                          "last_lap_same_boundary_packet": full is not None,
                          "sector_repeat_counts": dict(epoch["repeats"]),
                          "sector_revision_counts": dict(epoch["revisions"]),
                          "sector_clear_counts": dict(epoch["clears"])},
              "raw_control_observations": deepcopy(epoch["controls"]),
              "raw_pit_observations": deepcopy(epoch["pit_observations"]),
              "raw_tyre_support": deepcopy(epoch["tyre_observations"]),
              "revision_of": None, "canonical_lap_number_certified": False,
              "canonical_tyre_life_certified": False, "canonical_is_accurate_certified": False,
              "historical_client_receipt_certified": False,
              "quality_contract": "observed_raw_clean_proxy_not_final_IsAccurate"}
    return _complete_fields(record)


def _diagnostic(event, state, packet, sectors, lap):
    prior = state["closed"]
    late = (prior is not None and state["counter"] == prior["new_counter"]
            and 0 <= packet["available_ms"] - prior["available_ms"] <= LATE_WINDOW_MS)
    if not late:
        return _record(event, state["driver"], state["epoch"], packet, state["counter"],
                       state["counter"], "unassigned_completion_update", sectors.get(3), lap,
                       {"completion_update_without_counter_advance"})
    row = deepcopy(prior)
    row.update({"completion_id": f"{event}:{state['driver']}:{packet['sequence']}:late_completion_update",
                "record_kind": "late_completion_update", "available_ms": packet["available_ms"],
                "recorded_ms": packet["recorded_ms"], "packet_sequence": packet["sequence"],
                "revision_of": prior["completion_id"]})
    row["ambiguity_reasons"] = sorted(set(row["ambiguity_reasons"]) | {"late_completion_association_not_certified"})
    # Prior diagnostic missing-field flags are recomputed, but the immutable
    # prior row itself is untouched and this update can never become history.
    row["quality_reasons"] = [r for r in row["quality_reasons"] if r not in
                              {"missing_positive_sector", "missing_positive_last_lap_time", "sector_sum_inconsistent"}]
    if 3 in sectors:
        row["sectors_seconds"][2] = _duration(sectors[3])
        row["sector_sources"][2] = _source(row["sectors_seconds"][2], packet)
    if lap is not None:
        row["full_lap_seconds"] = _duration(lap)
        row["full_lap_source"] = _source(row["full_lap_seconds"], packet)
    row["support"]["s3_same_boundary_packet"] = False
    row["support"]["last_lap_same_boundary_packet"] = False
    return _complete_fields(row)


def iter_completed(paths, event_key, stats=None):
    """Yield immutable completion snapshots; never read a canonical final CSV.

    Callers must check stats['stream_input_valid'] after consuming the iterator.
    An unplaceable auxiliary timestamp is reported as invalid input and cannot
    certify source completeness, even when earlier rows have local support.
    History consumers additionally require available_ms < forecast checkpoint.
    """
    summary = {} if stats is None else stats
    auxiliaries = {name: _Auxiliary(name, paths[name]) for name in STREAMS}
    timing_stats, counts = Counter(), Counter()
    drivers = {}
    timing_gap = False
    for packet in iter_packets(paths["TimingData"], timing_stats):
        now = packet["available_ms"]
        for name, cursor in auxiliaries.items():
            for received in cursor.before(now):
                for state in drivers.values():
                    epoch = state["epoch"]
                    if received["error"] is not None:
                        epoch["ambiguity"].add("auxiliary_parse_gap")
                    elif name in ("SessionStatus", "TrackStatus") and "Status" in received["payload"]:
                        _observe_control(epoch, name, cursor.current)
                    elif name == "TimingAppData":
                        snapshot = cursor.tyre(state["driver"])
                        if snapshot != epoch["tyre_observations"][-1]:
                            epoch["tyre_observations"].append(snapshot)
        if packet["error"] is not None:
            timing_gap = True
            for state in drivers.values():
                state["epoch"]["ambiguity"].add("timing_parse_gap")
            continue
        patches = packet["payload"].get("Lines", {})
        for driver, patch in sorted(_items(patches), key=lambda pair: str(pair[0])):
            if not isinstance(patch, dict):
                continue
            driver = str(driver)
            if driver not in drivers:
                state = {"driver": driver, "counter": None, "highwater": None,
                         "quarantine": False, "generation": 0, "fields": {}, "closed": None}
                state["epoch"] = _new_epoch(state, packet, auxiliaries)
                drivers[driver] = state
            state = drivers[driver]
            epoch = state["epoch"]
            if timing_gap:
                epoch["ambiguity"].add("timing_parse_gap")
            if packet["regressed"]:
                epoch["ambiguity"].add("raw_timestamp_regression")
            changed_pit = []
            for field in ("InPit", "PitOut", "Retired", "Stopped"):
                if field in patch:
                    state["fields"][field] = _source(deepcopy(patch[field]), packet)
                    changed_pit.append(field)
            _observe_pit(epoch, state["fields"], changed_pit)
            sectors = _sector_patch(patch)
            lap_patch = patch.get("LastLapTime", {})
            lap = lap_patch.get("Value") if isinstance(lap_patch, dict) else None
            previous = state["counter"]
            current = _counter(patch["NumberOfLaps"]) if "NumberOfLaps" in patch else previous
            invalid_counter = "NumberOfLaps" in patch and current is None
            if invalid_counter:
                epoch["ambiguity"].add("invalid_counter")
                current = previous
                counts["invalid_counter_updates"] += 1
            boundary = "NumberOfLaps" in patch and not invalid_counter and current != previous
            output = None
            if boundary:
                reasons = set()
                kind = "unit_counter_completion"
                if previous is None:
                    kind = "counter_initialization"
                    reasons.add("unknown_previous_counter")
                elif current < previous:
                    kind = "counter_reversal"
                    reasons.add("counter_reversal")
                    state["quarantine"] = True
                elif current != previous + 1:
                    kind = "counter_jump"
                    reasons.add("counter_jump")
                if state["quarantine"]:
                    reasons.add("counter_quarantine")
                if any(_duration(sectors.get(s)) is not None for s in (1, 2)):
                    reasons.add("s1_s2_in_counter_boundary_packet")
                output = _record(event_key, driver, epoch, packet, previous, current,
                                 kind, sectors.get(3), lap, reasons)
                state["closed"] = deepcopy(output)
                high = state["highwater"]
                if state["quarantine"] and high is not None and current > high:
                    state["quarantine"] = False
                state["highwater"] = max(current, high if high is not None else current)
                state["counter"] = current
                state["generation"] += 1
                next_reasons = {r for r in reasons if r in {"counter_jump", "counter_reversal"}}
                state["epoch"] = _new_epoch(state, packet, auxiliaries, next_reasons)
                for sector in (1, 2):
                    if sector in sectors:
                        _update_sector(state["epoch"], sector, sectors[sector], packet, collision=True)
            else:
                for sector in (1, 2):
                    if sector in sectors:
                        _update_sector(epoch, sector, sectors[sector], packet)
                if _duration(sectors.get(3)) is not None or _duration(lap) is not None:
                    output = _diagnostic(event_key, state, packet, sectors, lap)
                if 3 in sectors:
                    # Display slot support only; never reuse it as a future
                    # boundary's same-packet S3.
                    epoch["sectors"][3] = _source(_duration(sectors[3]), packet)
            if output is not None:
                counts["records"] += 1
                counts[output["record_kind"]] += 1
                counts["observed_valid_completed"] += int(output["observed_valid_completed"])
                yield output
    # Only fully consumed/known clocks can assert input parsing integrity. A
    # parser failure discovered in a tail never mutates an already yielded row.
    aux_stats = {}
    for name, cursor in auxiliaries.items():
        for _ in cursor.before(float("inf")):
            pass
        aux_stats[name] = {**dict(cursor.stats), "consumed_errors": cursor.consumed_errors,
                           "unplaced_timestamp": cursor.next is not None and cursor.next["recorded_ms"] is None}
    summary.update({"event_key": event_key, "counts": dict(counts), "timing": dict(timing_stats),
                    "auxiliary": aux_stats, "drivers": len(drivers),
                    "stream_input_valid": not timing_stats["malformed_packets"] and all(
                        not s.get("malformed_packets", 0) and not s["unplaced_timestamp"] for s in aux_stats.values()),
                    "canonical_alignment_certified": False, "historical_client_receipt_certified": False})
