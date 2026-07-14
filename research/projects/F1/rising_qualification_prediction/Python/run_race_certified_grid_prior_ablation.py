#!/usr/bin/env python3
"""Certified-final-grid Race order ablation with prior-race state only.

This experiment intentionally does *not* call the Qualifying or practice
providers.  The current-event feature boundary is the FIA final-grid snapshot;
all learned corrections come from strictly earlier completed races in the same
season.  It exists to distinguish two very different evidence limitations:

* historical first-seen practice/Qualifying feeds cannot be recreated; and
* an FIA final-grid document can be certified retrospectively when its content
  hash and authoritative pre-race publication timestamp are preserved.

The protocol is fixed to R1-2 development, R3-4 selection, R5-6 locked
monitoring (called calibration for partition compatibility), and R7-9 audit.
No probability output is emitted, so the calibration block never fits or
changes a point-prediction parameter.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from capture_fia_final_grid_snapshot import (
    _pdf_text,
    parse_fia_grid_pdf_text,
)

SCHEMA_VERSION = "f1_race_certified_grid_prior_ablation_v1"
POST_GRID_PRE_RACE = "post_grid_pre_race"
MINIMUM_RELATIVE_SELECTION_GAIN = 0.05
GRID_CAPTURE_SCHEMA_VERSION = "f1_first_seen_post_grid_pre_race_v1"
WEEKEND_METADATA_SCHEMA_VERSION = "f1_weekend_snapshot_v2_point_in_time"
MAXIMUM_FIA_PUBLICATION_LAG_SECONDS = 5 * 60
PARTITIONS: Mapping[str, tuple[int, ...]] = {
    "development": (1, 2),
    "selection": (3, 4),
    "calibration": (5, 6),
    "audit": (7, 8, 9),
}
FIA_FINAL_GRID_DOCUMENTS: Mapping[int, Mapping[str, str]] = {
    1: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_01_australian_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_australian_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "354eabdb3a94fa7cfc302e42eba608b658d1dc6bb41cf22385b7483f1d8cffc9",
        "timezone": "Australia/Melbourne",
    },
    2: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_02_chinese_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_chinese_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "7eaf0e007bcd3b8674b5fb1ccc53c0af8d3a0dfd54f5fa875343b2720970c2f0",
        "timezone": "Asia/Shanghai",
    },
    3: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_03_japanese_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_japanese_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "c7d268b0ddf26caaecc6321552283737013bf7b57d56fe9ff66b6ab41a25f14f",
        "timezone": "Asia/Tokyo",
    },
    4: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_04_miami_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_miami_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "8a61b3b3292208cf07d9e82b848387ab9acd297e4fee638395dbd0001e6a1d60",
        "timezone": "America/New_York",
    },
    5: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_05_canadian_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_canadian_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "1a137eb47ad2588c6d8d0cbcc969cf793fd65f35e6516299082c9091cdf51988",
        "timezone": "America/Toronto",
    },
    6: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_06_monaco_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_monaco_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "eaf1bfe037be4a7ebb175cbf3f97b0fc996a8f594161e8ea7c2531a7cf958869",
        "timezone": "Europe/Monaco",
    },
    7: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_07_barcelona_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_barcelona-catalunya_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "823251318372a5b1fbc4505ee0022da49943509726b175c975c21647a865c108",
        "timezone": "Europe/Madrid",
    },
    8: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_08_austrian_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_austrian_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "675ba95ad8234513e1fa10c2a833a392eaf47ba95e6115044c3c80f084e94933",
        "timezone": "Europe/Vienna",
    },
    9: {
        "relative_path": (
            "data/f1/raw/weekends/2026/round_09_british_grand_prix/"
            "evidence/fia_final_starting_grid.pdf"
        ),
        "url": (
            "https://www.fia.com/system/files/decision-document/"
            "2026_british_grand_prix_-_final_starting_grid.pdf"
        ),
        "sha256": "e76cf4fee57cbb8ca843021ae7e7b66beaded99656e12dcff80a63004ce20f90",
        "timezone": "Europe/London",
    },
}


@dataclass(frozen=True)
class ResidualProfile:
    """Fixed exploratory empirical-Bayes correction to the legal grid order."""

    profile_id: str
    driver_prior_strength: float
    team_prior_strength: float
    driver_weight: float
    team_weight: float
    recency_decay: float


PROFILES: tuple[ResidualProfile, ...] = (
    ResidualProfile("d0_conservative_driver", 4.0, 16.0, 0.25, 0.0, 1.0),
    ResidualProfile("d1_moderate_driver", 2.0, 16.0, 0.50, 0.0, 0.75),
    ResidualProfile("d2_conservative_driver_team", 4.0, 16.0, 0.25, 0.25, 1.0),
    ResidualProfile("d3_recent_driver_team", 2.0, 8.0, 0.50, 0.25, 0.75),
)


@dataclass(frozen=True)
class EventData:
    event_key: int
    round_number: int
    event_name: str
    inference: pd.DataFrame
    target: pd.DataFrame
    input_paths: tuple[Path, ...]
    evidence: Mapping[str, Any]


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "").strip()) is not None


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a finite UTC timestamp")
    return timestamp


def _official_fia_https_url(value: object) -> bool:
    parsed = urlparse(str(value or "").strip())
    hostname = str(parsed.hostname or "").lower()
    return bool(
        parsed.scheme == "https"
        and parsed.path
        and (hostname == "fia.com" or hostname.endswith(".fia.com"))
    )


def _normalized_pdf_lines(pdf_text: str) -> list[str]:
    return [
        " ".join(line.replace("\u00a0", " ").split())
        for line in str(pdf_text).splitlines()
        if " ".join(line.replace("\u00a0", " ").split())
    ]


def _parse_fia_pdf_header(pdf_text: str) -> dict[str, str]:
    """Parse the FIA cover-page identity from text extracted from PDF bytes."""

    lines = _normalized_pdf_lines(pdf_text)

    def one(pattern: str, *, label: str) -> str:
        matches = [
            match.group(1).strip()
            for line in lines
            for match in [re.fullmatch(pattern, line)]
            if match
        ]
        if len(matches) != 1:
            raise ValueError(
                f"FIA PDF must contain exactly one {label} header; got {len(matches)}"
            )
        return matches[0]

    document_number = one(r"Document\s+(\d+)", label="document-number")
    document_date_text = one(r"Date\s+(.+)", label="document-date")
    document_time = one(r"Time\s+(\d{1,2}:\d{2})", label="document-time")
    document_title = one(r"Title\s+(.+)", label="document-title")
    document_date = pd.to_datetime(
        document_date_text,
        format="%d %B %Y",
        errors="coerce",
        utc=True,
    )
    if pd.isna(document_date):
        raise ValueError(f"FIA PDF document date is not parseable: {document_date_text!r}")
    return {
        "document_number": document_number,
        "document_date": document_date.strftime("%Y-%m-%d"),
        "document_time": document_time,
        "document_title": " ".join(document_title.split()),
    }


def _parse_capture_document_title(value: object) -> tuple[str, str]:
    match = re.fullmatch(
        r"Doc\s+(\d+)\s*-\s*(.+)",
        " ".join(str(value or "").replace("\u00a0", " ").split()),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("capture document_title must identify Doc N - title")
    return match.group(1), " ".join(match.group(2).split())


def _fia_document_header_timestamp_utc(
    header: Mapping[str, str],
    *,
    timezone_name: str,
) -> pd.Timestamp:
    """Convert the FIA document's printed venue-local timestamp to UTC."""

    try:
        venue_timezone = ZoneInfo(str(timezone_name))
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown pinned FIA document timezone: {timezone_name}") from exc
    try:
        local_timestamp = datetime.strptime(
            f"{header['document_date']} {header['document_time']}",
            "%Y-%m-%d %H:%M",
        ).replace(tzinfo=venue_timezone)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("FIA PDF document timestamp is incomplete or invalid") from exc
    return pd.Timestamp(local_timestamp).tz_convert("UTC")


def _canonical_grid_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int | None, str, str]]:
    canonical: dict[str, tuple[int | None, str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("FIA grid row must be an object")
        number = str(row.get("driver_number") or "").strip()
        if not number or number in canonical:
            raise ValueError("FIA PDF grid car numbers must be complete and unique")
        position = pd.to_numeric(row.get("position"), errors="coerce")
        canonical[number] = (
            None if pd.isna(position) else int(position),
            str(row.get("status") or "").strip().lower(),
            _normalize_driver_name(row.get("driver_name")),
        )
    return canonical


def _validate_local_fia_pdf(
    capture: Mapping[str, Any],
    *,
    source_document_path: Path,
    year: int,
    round_number: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Reopen and independently parse the pinned FIA source document bytes."""

    if int(year) != 2026 or int(round_number) not in FIA_FINAL_GRID_DOCUMENTS:
        raise ValueError(f"no pinned FIA final-grid document for {year} round {round_number}")
    pinned = FIA_FINAL_GRID_DOCUMENTS[int(round_number)]
    expected_path = (_root() / pinned["relative_path"]).resolve()
    actual_path = source_document_path.expanduser().resolve()
    if actual_path != expected_path:
        raise ValueError(
            f"{year} round {round_number} FIA PDF path is not the pinned weekend path"
        )
    if not actual_path.is_file():
        raise FileNotFoundError(
            f"{year} round {round_number} pinned FIA PDF is missing: {actual_path}"
        )

    raw_payload = capture.get("raw_payload")
    if not isinstance(raw_payload, Mapping):
        raise ValueError("certified FIA grid capture lacks raw payload object")
    raw_rows = raw_payload.get("grid_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("certified FIA source payload contains no official identities")

    actual_sha256 = _sha256(actual_path)
    extracted_text = _pdf_text(actual_path)
    actual_header = _parse_fia_pdf_header(extracted_text)
    captured_header = _parse_fia_pdf_header(
        str(raw_payload.get("pdf_extracted_text") or "")
    )
    captured_document_number, captured_document_title = _parse_capture_document_title(
        raw_payload.get("document_title")
    )
    parsed_rows = parse_fia_grid_pdf_text(
        extracted_text,
        expected_field_size=len(raw_rows),
    )
    parsed_rows_by_number = _canonical_grid_rows(parsed_rows)
    captured_rows_by_number = _canonical_grid_rows(raw_rows)
    first_published = _utc_timestamp(
        capture.get("first_published_at"), label="grid first_published_at"
    )
    document_timestamp_utc = _fia_document_header_timestamp_utc(
        actual_header,
        timezone_name=pinned["timezone"],
    )
    publication_lag_seconds = float(
        (first_published - document_timestamp_utc).total_seconds()
    )

    checks = {
        "local_pdf_path_is_pinned": actual_path == expected_path,
        "local_pdf_hash_matches_pinned_catalog": actual_sha256 == pinned["sha256"],
        "capture_url_matches_pinned_catalog": str(
            capture.get("source_document_url") or ""
        ).strip()
        == pinned["url"],
        "capture_hash_matches_pinned_catalog": str(
            capture.get("source_document_sha256") or ""
        ).strip()
        == pinned["sha256"],
        "raw_url_matches_pinned_catalog": str(
            raw_payload.get("document_url") or ""
        ).strip()
        == pinned["url"],
        "raw_hash_matches_pinned_catalog": str(
            raw_payload.get("document_sha256") or ""
        ).strip()
        == pinned["sha256"],
        "pdf_header_matches_captured_extraction": actual_header == captured_header,
        "pdf_document_number_matches_capture_title": (
            actual_header["document_number"] == captured_document_number
        ),
        "pdf_document_title_matches_capture_title": (
            actual_header["document_title"].casefold()
            == captured_document_title.casefold()
        ),
        "pdf_document_is_final_starting_grid": (
            actual_header["document_title"].casefold() == "final starting grid"
        ),
        "pdf_document_timestamp_matches_publication_time": (
            0.0
            <= publication_lag_seconds
            <= float(MAXIMUM_FIA_PUBLICATION_LAG_SECONDS)
        ),
        "pdf_rows_match_captured_rows": (
            parsed_rows_by_number == captured_rows_by_number
        ),
    }
    return checks, {
        "local_source_document_path": str(actual_path.relative_to(_root())),
        "local_source_document_sha256": actual_sha256,
        "parsed_document_header": actual_header,
        "pinned_document_timezone": pinned["timezone"],
        "parsed_document_timestamp_utc": document_timestamp_utc.isoformat(),
        "publication_lag_seconds": publication_lag_seconds,
        "parsed_grid_rows": len(parsed_rows_by_number),
    }


def _validate_capture_certification(
    capture: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    source_document_path: Path,
    year: int,
    round_number: int,
) -> dict[str, bool]:
    """Verify the evidence itself instead of trusting its certification flags."""

    snapshot = capture.get("snapshot")
    raw_payload = capture.get("raw_payload")
    if not isinstance(snapshot, Mapping) or not isinstance(raw_payload, Mapping):
        raise ValueError("certified FIA grid capture lacks snapshot/raw payload objects")
    entries = snapshot.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("certified FIA grid snapshot contains no entries")
    raw_rows = raw_payload.get("grid_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("certified FIA source payload contains no official identities")

    published = _utc_timestamp(
        capture.get("first_published_at"), label="grid first_published_at"
    )
    race_start = _utc_timestamp(
        capture.get("race_start_at"), label="grid race_start_at"
    )
    captured = _utc_timestamp(capture.get("captured_at"), label="grid captured_at")
    snapshot_prediction = _utc_timestamp(
        snapshot.get("prediction_as_of"), label="snapshot prediction_as_of"
    )
    snapshot_publication = _utc_timestamp(
        snapshot.get("publication_as_of"), label="snapshot publication_as_of"
    )
    raw_publication = _utc_timestamp(
        raw_payload.get("first_published_at"), label="raw first_published_at"
    )
    raw_race_start = _utc_timestamp(
        raw_payload.get("race_start_at"), label="raw race_start_at"
    )

    race_sessions = [
        row
        for row in metadata.get("sessions", [])
        if isinstance(row, Mapping)
        and str(row.get("session_type") or "").strip().lower() == "race"
    ]
    if len(race_sessions) != 1:
        raise ValueError("weekend metadata must contain exactly one Race session")
    scheduled_race_start = _utc_timestamp(
        race_sessions[0].get("scheduled_start_utc"),
        label="metadata Race scheduled_start_utc",
    )

    source_document_url = str(capture.get("source_document_url") or "").strip()
    source_endpoint = str(capture.get("source_endpoint") or "").strip()
    source_document_sha256 = str(
        capture.get("source_document_sha256") or ""
    ).strip()
    raw_document_url = str(raw_payload.get("document_url") or "").strip()
    raw_document_sha256 = str(raw_payload.get("document_sha256") or "").strip()
    raw_payload_sha256 = str(capture.get("raw_payload_sha256") or "").strip()
    raw_publication_page_url = str(
        raw_payload.get("publication_page_url") or ""
    ).strip()
    raw_publication_page_sha256 = str(
        raw_payload.get("publication_page_sha256") or ""
    ).strip()

    def structured_rows_match_raw_document() -> bool:
        if not all(isinstance(row, Mapping) for row in (*entries, *raw_rows)):
            return False
        raw_by_number: dict[str, tuple[int | None, str]] = {}
        for row in raw_rows:
            number = str(row.get("driver_number") or "").strip()
            if not number or number in raw_by_number:
                return False
            raw_position = pd.to_numeric(row.get("position"), errors="coerce")
            raw_by_number[number] = (
                None if pd.isna(raw_position) else int(raw_position),
                str(row.get("status") or "").strip().lower(),
            )
        structured_by_number: dict[str, tuple[int | None, str]] = {}
        for row in entries:
            number = str(row.get("driver_id") or "").strip()
            if not number or number in structured_by_number:
                return False
            structured_position = pd.to_numeric(
                row.get("grid_position"), errors="coerce"
            )
            structured_by_number[number] = (
                None if pd.isna(structured_position) else int(structured_position),
                str(row.get("status") or "").strip().lower(),
            )
        return structured_by_number == raw_by_number

    local_pdf_checks, _ = _validate_local_fia_pdf(
        capture,
        source_document_path=source_document_path,
        year=year,
        round_number=round_number,
    )
    checks = {
        "capture_schema_supported": capture.get("schema_version")
        == GRID_CAPTURE_SCHEMA_VERSION,
        "capture_year_matches": not isinstance(capture.get("year"), bool)
        and int(capture.get("year") or 0) == int(year),
        "capture_round_matches": not isinstance(capture.get("round_number"), bool)
        and int(capture.get("round_number") or 0) == int(round_number),
        "capture_id_present": bool(str(capture.get("capture_id") or "").strip()),
        "provider_is_fia": str(capture.get("provider") or "").strip().lower()
        == "fia",
        "publication_semantics_authoritative": str(
            capture.get("publication_time_semantics") or ""
        ).strip()
        == "authoritative_document_timestamp",
        "source_endpoint_is_official_fia_https": _official_fia_https_url(
            source_endpoint
        ),
        "source_document_is_official_fia_https": _official_fia_https_url(
            source_document_url
        ),
        "publication_page_is_official_fia_https": _official_fia_https_url(
            raw_publication_page_url
        ),
        "publication_page_matches_source_endpoint": raw_publication_page_url
        == source_endpoint,
        "source_document_hash_valid": _valid_sha256(source_document_sha256),
        "publication_page_hash_valid": _valid_sha256(
            raw_publication_page_sha256
        ),
        "raw_payload_hash_valid": _valid_sha256(raw_payload_sha256)
        and raw_payload_sha256 == _canonical_sha256(raw_payload),
        "source_document_matches_raw_payload": bool(source_document_sha256)
        and source_document_sha256 == raw_document_sha256
        and source_document_url == raw_document_url,
        "raw_document_is_final_starting_grid": "final starting grid"
        in " ".join(
            str(raw_payload.get("document_title") or "").lower().split()
        ),
        "raw_grid_rows_are_objects": all(
            isinstance(row, Mapping) for row in raw_rows
        ),
        "structured_snapshot_matches_raw_document_rows": (
            structured_rows_match_raw_document()
        ),
        "source_is_official_grid_revision": snapshot.get("source")
        == "official_grid_revision",
        "evidence_complete": snapshot.get("evidence_complete") is True
        and all(
            isinstance(row, Mapping) and row.get("evidence_complete") is True
            for row in entries
        ),
        "snapshot_available": snapshot.get("available") is True,
        "resolution_complete": snapshot.get("resolution_status") == "resolved",
        "publication_pre_race_verified": capture.get(
            "publication_pre_race_verified"
        )
        is True,
        "post_grid_pre_race_horizon": snapshot.get("horizon")
        == POST_GRID_PRE_RACE,
        "final_pre_race_revision": capture.get("revision_phase")
        == "final_pre_race",
        "publication_precedes_race": published < race_start,
        "publication_not_after_capture": published <= captured,
        "snapshot_times_match_publication": snapshot_prediction == published
        and snapshot_publication == published,
        "raw_times_match_capture": raw_publication == published
        and raw_race_start == race_start,
        "metadata_race_start_matches_capture": scheduled_race_start == race_start,
        "metadata_identity_matches": metadata.get("schema_version")
        == WEEKEND_METADATA_SCHEMA_VERSION
        and not isinstance(metadata.get("year"), bool)
        and int(metadata.get("year") or 0) == int(year)
        and not isinstance(metadata.get("round_number"), bool)
        and int(metadata.get("round_number") or 0) == int(round_number),
        **local_pdf_checks,
    }
    failed = sorted(name for name, valid in checks.items() if not valid)
    if failed:
        raise ValueError(
            f"{year} round {round_number} final grid is not horizon-certified: {failed}"
        )
    return checks


def _normalize_driver_name(value: object) -> str:
    text = " ".join(str(value or "").strip().upper().split())
    if not text or text in {"NAN", "NONE", "NULL"}:
        raise ValueError("FIA final grid contains an incomplete official driver name")
    return text


def _partition_for_round(round_number: int) -> str:
    matches = [
        name for name, rounds in PARTITIONS.items() if int(round_number) in rounds
    ]
    if len(matches) != 1:
        raise ValueError(f"round {round_number} is outside the fixed R1-R9 protocol")
    return matches[0]


def _round_directory(weekends_dir: Path, year: int, round_number: int) -> Path:
    matches = sorted((weekends_dir / str(year)).glob(f"round_{round_number:02d}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one local weekend for {year} round {round_number}, got {len(matches)}"
        )
    return matches[0]


def _resolve_reference(root: Path, weekend_dir: Path, reference: object) -> Path:
    text = str(reference or "").strip()
    if not text:
        raise ValueError(f"{weekend_dir} has no Race results reference")
    path = Path(text).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (root / path, weekend_dir / path.name, weekend_dir / path)
    )
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"could not resolve Race result reference {text!r}")
    return resolved.resolve()


def _race_result_path(root: Path, metadata_path: Path) -> Path:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    race_entries = [
        row
        for row in metadata.get("sessions", [])
        if str(row.get("session_type", "")).strip().lower() == "race"
    ]
    if len(race_entries) != 1:
        raise ValueError(f"{metadata_path} does not identify exactly one Race session")
    return _resolve_reference(
        root,
        metadata_path.parent,
        race_entries[0].get("results_path"),
    )


def _legal_grid_order(frame: pd.DataFrame) -> pd.Series:
    required = {"grid_position", "grid_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"final grid lacks legal-order fields: {missing}")
    status = frame["grid_status"].astype(str).str.strip().str.lower()
    invalid_status = ~status.isin({"grid", "pit_lane", "nonstarter"})
    if invalid_status.any():
        raise ValueError(
            "final grid contains unsupported statuses: "
            f"{sorted(status.loc[invalid_status].unique())}"
        )
    physical = pd.to_numeric(frame["grid_position"], errors="coerce")
    group = np.where(status.eq("grid"), 0, np.where(status.eq("pit_lane"), 1, 2))
    order = pd.DataFrame(
        {
            "group": group,
            "physical": physical.fillna(np.inf),
            "provider_order": np.arange(len(frame), dtype=int),
        },
        index=frame.index,
    ).sort_values(["group", "physical", "provider_order"], kind="mergesort")
    ranks = pd.Series(
        np.arange(1, len(order) + 1, dtype=int),
        index=order.index,
    ).reindex(frame.index)
    if sorted(ranks.astype(int).tolist()) != list(range(1, len(frame) + 1)):
        raise RuntimeError("legal final grid did not resolve to a complete permutation")
    return ranks


def _load_event(
    *,
    root: Path,
    weekends_dir: Path,
    year: int,
    round_number: int,
) -> EventData:
    """Read only the capture JSON, then open the Race target after frame freeze."""

    weekend_dir = _round_directory(weekends_dir, year, round_number)
    metadata_path = (weekend_dir / "weekend_metadata.json").resolve()
    capture_paths = sorted(
        (weekend_dir / "first_seen_grid_snapshots").glob("grid_*.json")
    )
    if len(capture_paths) != 1:
        raise ValueError(
            f"{year} round {round_number} must bind exactly one grid capture; "
            f"found {len(capture_paths)}"
        )
    capture_path = capture_paths[0].resolve()
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(capture, dict) or not isinstance(metadata, dict):
        raise ValueError("grid capture and weekend metadata must be JSON objects")
    snapshot = capture.get("snapshot")
    raw_payload = capture.get("raw_payload")
    if not isinstance(snapshot, dict) or not isinstance(raw_payload, dict):
        raise ValueError("certified FIA grid capture lacks snapshot/raw payload objects")
    entries = snapshot.get("entries")
    raw_rows = raw_payload.get("grid_rows")
    if not isinstance(entries, list) or not entries:
        raise ValueError("certified FIA grid snapshot contains no entries")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("certified FIA source payload contains no official identities")

    pinned_document = FIA_FINAL_GRID_DOCUMENTS[int(round_number)]
    source_document_path = (root / pinned_document["relative_path"]).resolve()
    certification_checks = _validate_capture_certification(
        capture,
        metadata,
        source_document_path=source_document_path,
        year=year,
        round_number=round_number,
    )
    source_document_sha256 = str(capture["source_document_sha256"])

    name_by_number: dict[str, str] = {}
    for row in raw_rows:
        number = str(row.get("driver_number") or "").strip()
        if not number or number in name_by_number:
            raise ValueError("FIA source grid car numbers must be complete and unique")
        name_by_number[number] = _normalize_driver_name(row.get("driver_name"))
    inference_records: list[dict[str, Any]] = []
    for provider_order, row in enumerate(entries):
        driver_id = str(row.get("driver_id") or "").strip()
        if driver_id not in name_by_number:
            raise ValueError(
                f"FIA structured grid identity {driver_id!r} lacks source-row identity"
            )
        status = str(row.get("status") or "").strip().lower()
        inference_records.append(
            {
                "driver_id": driver_id,
                "grid_official_driver_name": name_by_number[driver_id],
                "driver_key": name_by_number[driver_id],
                "grid_position": row.get("grid_position"),
                "grid_status": status,
                "grid_starter_eligible": bool(row.get("starter_eligible")),
                "grid_pit_lane_start": bool(row.get("pit_lane_start")),
                "provider_order": int(provider_order),
                "event_key": int(year) * 100 + int(round_number),
            }
        )
    inference = pd.DataFrame.from_records(inference_records)
    if inference["driver_id"].eq("").any() or inference["driver_id"].duplicated().any():
        raise ValueError("certified grid car-number identities must be complete and unique")
    if inference["driver_key"].duplicated().any():
        raise ValueError("certified grid official driver names must be unique")
    inference["grid_baseline_position"] = _legal_grid_order(inference).astype(int)
    # This immutable copy is the complete current-event feature surface.
    inference = inference.copy(deep=True)

    # Read only four target/prior-state columns. GridPosition, Qualifying, and
    # every practice field are impossible to enter through this usecols gate.
    race_result_path = _race_result_path(root, metadata_path)
    target = pd.read_csv(
        race_result_path,
        usecols=["DriverNumber", "Position", "TeamName", "Status"],
        dtype={"DriverNumber": str},
    ).rename(
        columns={
            "DriverNumber": "driver_id",
            "Position": "actual_position",
            "TeamName": "team_name",
            "Status": "race_status_raw",
        }
    )
    target["driver_id"] = target["driver_id"].astype(str).str.strip().map(
        lambda value: str(int(float(value))) if value.replace(".", "", 1).isdigit() else value
    )
    target["race_status_evidence_complete"] = (
        target["race_status_raw"].notna()
        & target["race_status_raw"].astype(str).str.strip().ne("")
    )
    if target["driver_id"].eq("").any() or target["driver_id"].duplicated().any():
        raise ValueError("Race target driver identities are not complete and unique")
    if set(target["driver_id"]) != set(inference["driver_id"]):
        raise ValueError("Race target roster differs from the certified FIA grid roster")
    target["actual_position"] = pd.to_numeric(
        target["actual_position"], errors="coerce"
    )
    if target["actual_position"].isna().any():
        raise ValueError("Race target does not contain a complete final classification")
    expected = list(range(1, len(target) + 1))
    if sorted(target["actual_position"].astype(int).tolist()) != expected:
        raise ValueError("Race target is not a complete final-position permutation")

    parsed_document_header = _parse_fia_pdf_header(
        str(raw_payload.get("pdf_extracted_text") or "")
    )
    pinned_document_timezone = FIA_FINAL_GRID_DOCUMENTS[int(round_number)]["timezone"]
    parsed_document_timestamp_utc = _fia_document_header_timestamp_utc(
        parsed_document_header,
        timezone_name=pinned_document_timezone,
    )
    publication_lag_seconds = float(
        (
            _utc_timestamp(
                capture.get("first_published_at"),
                label="grid first_published_at",
            )
            - parsed_document_timestamp_utc
        ).total_seconds()
    )
    evidence = {
        "capture_id": str(capture.get("capture_id") or ""),
        "capture_path": str(capture_path.relative_to(root)),
        "captured_at": str(capture.get("captured_at")),
        "first_published_at": str(capture.get("first_published_at")),
        "race_start_at": str(capture.get("race_start_at")),
        "publication_time_semantics": str(
            capture.get("publication_time_semantics")
        ),
        "local_source_document_path": str(source_document_path.relative_to(root)),
        "local_source_document_sha256": _sha256(source_document_path),
        "parsed_source_document_header": parsed_document_header,
        "pinned_document_timezone": pinned_document_timezone,
        "parsed_document_timestamp_utc": parsed_document_timestamp_utc.isoformat(),
        "publication_lag_seconds": publication_lag_seconds,
        "maximum_allowed_publication_lag_seconds": (
            MAXIMUM_FIA_PUBLICATION_LAG_SECONDS
        ),
        "parsed_source_document_grid_rows": len(raw_rows),
        "source_document_sha256": source_document_sha256,
        "local_capture_can_postdate_event": True,
        "logical_horizon_certified_by_authoritative_publication_time": bool(
            certification_checks[
                "pdf_document_timestamp_matches_publication_time"
            ]
            and certification_checks["publication_precedes_race"]
        ),
        "certification_checks": certification_checks,
    }
    return EventData(
        event_key=int(year) * 100 + int(round_number),
        round_number=int(round_number),
        event_name=str(metadata.get("event_name") or weekend_dir.name),
        inference=inference,
        target=target,
        input_paths=(metadata_path, capture_path, source_document_path, race_result_path),
        evidence=evidence,
    )


def _weighted_zero_shrunk_mean(
    rows: pd.DataFrame,
    *,
    value_column: str,
    current_event_key: int,
    prior_strength: float,
    recency_decay: float,
) -> float:
    if rows.empty:
        return 0.0
    if not (0.0 < float(recency_decay) <= 1.0):
        raise ValueError("recency_decay must be in (0, 1]")
    event_key = pd.to_numeric(rows["event_key"], errors="coerce").to_numpy(dtype=float)
    if not np.all(event_key < int(current_event_key)):
        raise ValueError("residual correction received non-prior event state")
    round_gap = int(current_event_key) - event_key
    weights = np.power(float(recency_decay), round_gap)
    values = pd.to_numeric(rows[value_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.any():
        return 0.0
    return float(
        np.dot(weights[finite], values[finite])
        / (float(prior_strength) + float(weights[finite].sum()))
    )


def _score_profile(
    inference: pd.DataFrame,
    history: pd.DataFrame,
    profile: ResidualProfile,
) -> pd.DataFrame:
    """Score one event without accepting any current-event target columns."""

    forbidden = {"actual_position", "team_name", "race_status_raw"}
    present = sorted(forbidden.intersection(inference.columns))
    if present:
        raise ValueError(f"current-event inference contains target-derived fields: {present}")
    event_keys = pd.to_numeric(inference["event_key"], errors="coerce").unique()
    if len(event_keys) != 1:
        raise ValueError("one score call must contain exactly one event")
    event_key = int(event_keys[0])
    prior = history.loc[
        pd.to_numeric(history.get("event_key"), errors="coerce").lt(event_key)
    ].copy() if not history.empty else history.copy()
    if not history.empty and len(prior) != len(history):
        raise ValueError("history contains current or future Race state")

    rows: list[dict[str, Any]] = []
    for row in inference.itertuples(index=False):
        driver_history = prior.loc[
            prior.get("driver_key", pd.Series(index=prior.index, dtype=object))
            .astype(str)
            .eq(str(row.driver_key))
        ]
        driver_effect = _weighted_zero_shrunk_mean(
            driver_history,
            value_column="grid_residual",
            current_event_key=event_key,
            prior_strength=profile.driver_prior_strength,
            recency_decay=profile.recency_decay,
        )
        prior_team: str | None = None
        if not driver_history.empty and "team_name" in driver_history.columns:
            last = driver_history.sort_values("event_key", kind="mergesort").iloc[-1]
            text = str(last.get("team_name") or "").strip()
            prior_team = text if text and text.lower() not in {"nan", "none", "null"} else None
        team_history = (
            prior.loc[prior["team_name"].astype(str).eq(prior_team)]
            if prior_team is not None and "team_name" in prior.columns
            else prior.iloc[:0]
        )
        team_effect = _weighted_zero_shrunk_mean(
            team_history,
            value_column="grid_residual",
            current_event_key=event_key,
            prior_strength=profile.team_prior_strength,
            recency_decay=profile.recency_decay,
        )
        correction = (
            profile.driver_weight * driver_effect
            + profile.team_weight * team_effect
        )
        rows.append(
            {
                "driver_id": str(row.driver_id),
                "driver_key": str(row.driver_key),
                "grid_baseline_position": int(row.grid_baseline_position),
                "provider_order": int(row.provider_order),
                "driver_residual_effect": float(driver_effect),
                "prior_team": prior_team,
                "team_residual_effect": float(team_effect),
                "candidate_score": float(row.grid_baseline_position + correction),
                "prior_driver_events": int(driver_history["event_key"].nunique())
                if not driver_history.empty
                else 0,
                "prior_team_events": int(team_history["event_key"].nunique())
                if not team_history.empty
                else 0,
            }
        )
    scored = pd.DataFrame.from_records(rows)
    ordered = scored.sort_values(
        ["candidate_score", "grid_baseline_position", "provider_order"],
        kind="mergesort",
    )
    scored["candidate_predicted_position"] = 0
    scored.loc[ordered.index, "candidate_predicted_position"] = np.arange(
        1, len(scored) + 1, dtype=int
    )
    expected = list(range(1, len(scored) + 1))
    if sorted(scored["candidate_predicted_position"].astype(int).tolist()) != expected:
        raise RuntimeError("certified-grid challenger did not emit a legal permutation")
    return scored


def _position_metrics(scored: pd.DataFrame) -> dict[str, float]:
    actual = pd.to_numeric(scored["actual_position"], errors="coerce")
    baseline = pd.to_numeric(scored["grid_baseline_position"], errors="coerce")
    candidate = pd.to_numeric(scored["candidate_predicted_position"], errors="coerce")
    return {
        "baseline_mae": float((baseline - actual).abs().mean()),
        "candidate_mae": float((candidate - actual).abs().mean()),
        "baseline_kendall": float(baseline.corr(actual, method="kendall")),
        "candidate_kendall": float(candidate.corr(actual, method="kendall")),
    }


def _select_challenger(selection_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_profile: list[dict[str, Any]] = []
    for profile in PROFILES:
        rows = [row for row in selection_rows if row["profile_id"] == profile.profile_id]
        if {int(row["round"]) for row in rows} != set(PARTITIONS["selection"]):
            raise ValueError(f"profile {profile.profile_id} lacks the full selection block")
        by_profile.append(
            {
                "profile_id": profile.profile_id,
                "mean_candidate_mae": float(
                    np.mean([float(row["candidate_mae"]) for row in rows])
                ),
                "mean_baseline_mae": float(
                    np.mean([float(row["baseline_mae"]) for row in rows])
                ),
                "event_keys": [int(row["event_key"]) for row in rows],
                "driver_weight": profile.driver_weight,
                "team_weight": profile.team_weight,
                "complexity": float(profile.driver_weight + profile.team_weight),
            }
        )
    chosen = min(
        by_profile,
        key=lambda row: (
            float(row["mean_candidate_mae"]),
            float(row["complexity"]),
            str(row["profile_id"]),
        ),
    )
    baseline = float(chosen["mean_baseline_mae"])
    candidate = float(chosen["mean_candidate_mae"])
    relative_gain = float((baseline - candidate) / baseline) if baseline > 0.0 else 0.0
    descriptive_material_gain = relative_gain >= MINIMUM_RELATIVE_SELECTION_GAIN
    return {
        "selected_challenger_profile_id": str(chosen["profile_id"]),
        # These profiles were fixed for this retrospective ablation after the
        # 2026 results existed.  Even a descriptive gain cannot become public
        # selection evidence; the alias remains the certified legal grid.
        "public_selected_model_id": "legal_grid_baseline",
        "challenger_selected_on_selection": False,
        "descriptive_candidate_would_clear_gain_threshold": descriptive_material_gain,
        "formal_selection_evidence": False,
        "profile_grid_evidence_role": (
            "posthoc_descriptive_fixed_for_this_run_not_prospective_selection"
        ),
        "selection_baseline_mae": baseline,
        "selection_challenger_mae": candidate,
        "selection_relative_gain": relative_gain,
        "minimum_relative_selection_gain": MINIMUM_RELATIVE_SELECTION_GAIN,
        "candidate_results": by_profile,
        "tie_break": "lower_complexity_then_profile_id",
    }


def _aggregate(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("cannot aggregate an empty event block")
    return {
        "events": len(events),
        "event_keys": [int(row["event_key"]) for row in events],
        "baseline_mean_mae": float(np.mean([float(row["baseline_mae"]) for row in events])),
        "candidate_mean_mae": float(np.mean([float(row["candidate_mae"]) for row in events])),
        "selected_mean_mae": float(np.mean([float(row["selected_mae"]) for row in events])),
        "baseline_mean_kendall": float(
            np.mean([float(row["baseline_kendall"]) for row in events])
        ),
        "candidate_mean_kendall": float(
            np.mean([float(row["candidate_kendall"]) for row in events])
        ),
        "candidate_minus_baseline_mae": float(
            np.mean([float(row["candidate_mae"]) - float(row["baseline_mae"]) for row in events])
        ),
    }


def run(
    *,
    weekends_dir: Path,
    year: int = 2026,
    generated_at: str = "2026-07-14T00:00:00Z",
) -> dict[str, Any]:
    root = _root()
    if int(year) != 2026:
        raise ValueError("v1 certified-grid ablation is locked to the 2026 R1-R9 evidence set")
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "repo_bootstrap.py",
        Path(__file__).resolve().parent / "capture_fia_final_grid_snapshot.py",
    )
    implementation_before = {
        str(path.relative_to(root)): _sha256(path) for path in implementation_paths
    }

    events = [
        _load_event(
            root=root,
            weekends_dir=weekends_dir,
            year=year,
            round_number=round_number,
        )
        for round_number in range(1, 10)
    ]
    input_paths = tuple(sorted({path for event in events for path in event.input_paths}))
    inputs_before = {str(path.relative_to(root)): _sha256(path) for path in input_paths}

    history = pd.DataFrame()
    development_profile_scores: dict[tuple[int, str], pd.DataFrame] = {}
    selection_rows: list[dict[str, Any]] = []
    locked_profile: ResidualProfile | None = None
    selection_lock: dict[str, Any] | None = None
    scored_events: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for event in events:
        partition = _partition_for_round(event.round_number)
        if event.round_number <= max(PARTITIONS["selection"]):
            for profile in PROFILES:
                profile_score = _score_profile(event.inference, history, profile).merge(
                    event.target,
                    on="driver_id",
                    validate="one_to_one",
                )
                development_profile_scores[(event.round_number, profile.profile_id)] = profile_score
                if partition == "selection":
                    metrics = _position_metrics(profile_score)
                    selection_rows.append(
                        {
                            "event_key": event.event_key,
                            "round": event.round_number,
                            "profile_id": profile.profile_id,
                            **metrics,
                        }
                    )
        if event.round_number == max(PARTITIONS["selection"]):
            selection_lock = _select_challenger(selection_rows)
            locked_profile = next(
                profile
                for profile in PROFILES
                if profile.profile_id == selection_lock["selected_challenger_profile_id"]
            )
        if event.round_number <= max(PARTITIONS["selection"]):
            # The final chosen challenger is retrieved only after R4 locks it.
            chosen_score = None
        else:
            if locked_profile is None or selection_lock is None:
                raise RuntimeError("selection parameters were not locked before calibration")
            chosen_score = _score_profile(event.inference, history, locked_profile).merge(
                event.target,
                on="driver_id",
                validate="one_to_one",
            )

        target_with_identity = event.inference[["driver_id", "driver_key"]].merge(
            event.target,
            on="driver_id",
            validate="one_to_one",
        )
        target_with_identity["event_key"] = event.event_key
        target_with_identity["grid_baseline_position"] = event.inference.set_index(
            "driver_id"
        ).loc[target_with_identity["driver_id"], "grid_baseline_position"].to_numpy()
        target_with_identity["grid_residual"] = (
            pd.to_numeric(target_with_identity["actual_position"], errors="coerce")
            - pd.to_numeric(
                target_with_identity["grid_baseline_position"], errors="coerce"
            )
        )
        history = pd.concat([history, target_with_identity], ignore_index=True)
        if chosen_score is not None:
            development_profile_scores[(event.round_number, locked_profile.profile_id)] = chosen_score

    if locked_profile is None or selection_lock is None:
        raise RuntimeError("selection did not produce a locked certified-grid challenger")

    for event in events:
        partition = _partition_for_round(event.round_number)
        scored = development_profile_scores[(event.round_number, locked_profile.profile_id)].copy()
        metrics = _position_metrics(scored)
        public_model = str(selection_lock["public_selected_model_id"])
        if public_model == "legal_grid_baseline":
            scored["selected_predicted_position"] = scored["grid_baseline_position"].astype(int)
        else:
            scored["selected_predicted_position"] = scored[
                "candidate_predicted_position"
            ].astype(int)
        scored["selected_model_id"] = public_model
        selected_mae = float(
            (
                pd.to_numeric(scored["selected_predicted_position"], errors="coerce")
                - pd.to_numeric(scored["actual_position"], errors="coerce")
            )
            .abs()
            .mean()
        )
        event_row = {
            "event_key": event.event_key,
            "year": year,
            "round": event.round_number,
            "event_name": event.event_name,
            "partition": partition,
            "information_horizon": POST_GRID_PRE_RACE,
            "training_event_keys": list(range(year * 100 + 1, event.event_key)),
            "training_events": event.round_number - 1,
            "locked_challenger_profile_id": locked_profile.profile_id,
            "selected_model_id": public_model,
            **metrics,
            "selected_mae": selected_mae,
            "candidate_minus_baseline_mae": float(
                metrics["candidate_mae"] - metrics["baseline_mae"]
            ),
            "legal_candidate_permutation": sorted(
                scored["candidate_predicted_position"].astype(int).tolist()
            )
            == list(range(1, len(scored) + 1)),
            "legal_selected_permutation": sorted(
                scored["selected_predicted_position"].astype(int).tolist()
            )
            == list(range(1, len(scored) + 1)),
            "certified_grid_evidence": dict(event.evidence),
        }
        scored_events.append(event_row)
        for row in scored.to_dict(orient="records"):
            prediction_rows.append(
                {
                    "event_key": event.event_key,
                    "round": event.round_number,
                    "event_name": event.event_name,
                    "partition": partition,
                    **row,
                }
            )

    implementation_after = {
        str(path.relative_to(root)): _sha256(path) for path in implementation_paths
    }
    inputs_after = {str(path.relative_to(root)): _sha256(path) for path in input_paths}
    if implementation_before != implementation_after:
        raise RuntimeError("certified-grid ablation implementation changed during evaluation")
    if inputs_before != inputs_after:
        raise RuntimeError("certified-grid ablation inputs changed during evaluation")

    partition_aggregates = {
        name: _aggregate(
            [row for row in scored_events if row["partition"] == name]
        )
        for name in PARTITIONS
    }
    audit = partition_aggregates["audit"]
    promotion_reasons = [
        "posthoc_profile_grid_not_prospective_selection_evidence",
        "fewer_than_four_independent_selection_events",
        "fewer_than_four_independent_calibration_events",
        "point_ablation_emits_no_status_or_position_probability_calibration",
    ]
    if not bool(
        selection_lock["descriptive_candidate_would_clear_gain_threshold"]
    ):
        promotion_reasons.append("challenger_failed_five_percent_selection_gain")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": str(generated_at),
        "mode": "race_final_position",
        "experiment_id": "certified_fia_grid_plus_prior_race_state_only",
        "target": "official_terminal_race_classification_position",
        "protocol": {
            "same_season_only": True,
            "year": year,
            "event_partitions": {
                name: [year * 100 + round_number for round_number in rounds]
                for name, rounds in PARTITIONS.items()
            },
            "current_event_inputs": [
                "certified_fia_final_grid_position_and_status",
                "fia_official_driver_identity",
            ],
            "prior_event_inputs": [
                "strictly_earlier_race_final_position",
                "strictly_earlier_race_team_identity",
            ],
            "explicitly_forbidden_inputs": [
                "qualifying_results_or_times",
                "practice_results_laps_or_telemetry",
                "current_event_race_team_identity",
                "current_or_future_race_target",
            ],
            "mathematics": {
                "residual": "race_final_position_minus_legal_grid_position",
                "driver_effect": "recency_weighted_residual_sum/(prior_strength+weight_sum)",
                "team_effect": "same formula using last-known prior-race team",
                "score": "legal_grid+driver_weight*driver_effect+team_weight*team_effect",
                "permutation": "stable sort by score, legal grid, provider row order",
            },
            "fixed_exploratory_profiles": [asdict(profile) for profile in PROFILES],
            "profile_grid_evidence_role": (
                "posthoc_descriptive_fixed_for_this_run_not_prospective_selection"
            ),
            "selection_lock": selection_lock,
            "calibration_partition_role": (
                "locked_post_selection_monitoring_only_no_parameter_or_probability_update"
            ),
            "audit_targets_read_for_selection_or_tuning": False,
            "qualifying_or_practice_provider_called": False,
            "grid_capture_semantics": (
                "local_download_may_be_retrospective_but_pinned_fia_pdf_bytes_"
                "parsed_rows_and_authoritative_publication_timestamp_certify_"
                "the_pre_race_horizon"
            ),
        },
        "aggregate": _aggregate(scored_events),
        "partition_aggregates": partition_aggregates,
        "audit_aggregate": audit,
        "promotion": {
            "promoted": False,
            "diagnostic_only": True,
            "reasons": promotion_reasons,
            "audit_events": int(audit["events"]),
            "selection_events": len(PARTITIONS["selection"]),
            "calibration_events": len(PARTITIONS["calibration"]),
        },
        "events": scored_events,
        "predictions": prediction_rows,
        "input_manifest": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(inputs_before.items())
        ],
        "implementation_manifest": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(implementation_before.items())
        ],
        "configuration_manifest": {
            "weekends_dir": str(weekends_dir.relative_to(root)),
            "year": year,
            "generated_at": str(generated_at),
            "minimum_relative_selection_gain": MINIMUM_RELATIVE_SELECTION_GAIN,
            "partitions": {name: list(rounds) for name, rounds in PARTITIONS.items()},
            "profiles": [asdict(profile) for profile in PROFILES],
            "pinned_fia_final_grid_documents": {
                str(round_number): dict(document)
                for round_number, document in sorted(FIA_FINAL_GRID_DOCUMENTS.items())
            },
        },
    }
    payload["manifest_hashes"] = {
        "input_manifest_sha256": _canonical_sha256(payload["input_manifest"]),
        "implementation_manifest_sha256": _canonical_sha256(
            payload["implementation_manifest"]
        ),
        "configuration_manifest_sha256": _canonical_sha256(
            payload["configuration_manifest"]
        ),
        "protocol_sha256": _canonical_sha256(payload["protocol"]),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekends-dir",
        type=Path,
        default=_root() / "data/f1/raw/weekends",
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--generated-at", default="2026-07-14T00:00:00Z")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _root()
            / "artifacts/backtests/f1/race_final_position/"
            "certified_grid_prior_state_ablation_v1_20260714.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root()
    weekends_dir = args.weekends_dir.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = root / output
    payload = run(
        weekends_dir=weekends_dir,
        year=args.year,
        generated_at=args.generated_at,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "selection_lock": payload["protocol"]["selection_lock"],
                "partition_aggregates": payload["partition_aggregates"],
                "promotion": payload["promotion"],
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: feat(f1-race): add certified-grid prior-state ablation
