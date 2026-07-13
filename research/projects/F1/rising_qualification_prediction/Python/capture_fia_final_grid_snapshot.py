#!/usr/bin/env python3
"""Capture an authoritative FIA starting-grid publication as immutable evidence.

The structured order comes from the FIA page's ``GRID OFFICIAL`` table.  The
published Final/Provisional Starting Grid PDF supplies the authoritative
publication timestamp, document URL, content hash, and a second independent
driver/order check.  No Race result or retrospective ``GridPosition`` field is
accepted by this command.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Iterable
import unicodedata
from urllib.parse import unquote, urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from packages.f1.domain.starting_grid import (
    GridAdjustmentKind,
    GridEntryStatus,
    GridRevisionPhase,
    OfficialGridDecision,
    OfficialGridEntry,
    OfficialGridRevision,
    build_race_grid_capture,
    persist_race_grid_capture,
)
from packages.f1.domain.weekend import build_weekend_contract


class _GridOfficialHTMLParser(HTMLParser):
    """Collect table cells only from the FIA ``GRID OFFICIAL`` section."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_h3 = False
        self._h3_data = ""
        self._h3_text: list[str] = []
        self._active = False
        self._table_depth = 0
        self._in_row = False
        self._in_cell = False
        self._cell: list[str] = []
        self._row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if normalized == "h3":
            self._in_h3 = True
            self._h3_data = attributes.get("data", "")
            self._h3_text = []
            return
        if not self._active:
            return
        if normalized == "table":
            self._table_depth += 1
        elif normalized == "tr" and self._table_depth > 0:
            self._in_row = True
            self._row = []
        elif normalized in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell = []
        elif normalized == "br" and self._in_cell:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "h3" and self._in_h3:
            marker = f"{self._h3_data} {' '.join(self._h3_text)}".upper()
            self._active = "GRID OFFICIAL" in marker
            self._in_h3 = False
            return
        if not self._active:
            return
        if normalized in {"td", "th"} and self._in_cell:
            value = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(value)
            self._in_cell = False
        elif normalized == "tr" and self._in_row:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = []
            self._in_row = False
        elif normalized == "table" and self._table_depth > 0:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_h3:
            self._h3_text.append(data)
        elif self._in_cell:
            self._cell.append(data)


class _AllTableHTMLParser(HTMLParser):
    """Small dependency-free table reader used only for the pre-race roster."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._table_depth = 0
        self._in_row = False
        self._in_cell = False
        self._row: list[str] = []
        self._cell: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "table":
            self._table_depth += 1
        elif normalized == "tr" and self._table_depth:
            self._in_row = True
            self._row = []
        elif normalized in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._in_cell:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._in_cell = False
        elif normalized == "tr" and self._in_row:
            if any(self._row):
                self.rows.append(self._row)
            self._in_row = False
        elif normalized == "table" and self._table_depth:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)


class _DecisionDocumentHTMLParser(HTMLParser):
    """Collect FIA decision-document anchors with their rendered text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._href: str | None = None
        self._depth = 0
        self._text: list[str] = []
        self.anchors: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a" and self._href is None:
            attributes = {str(key).lower(): str(value or "") for key, value in attrs}
            self._href = attributes.get("href") or None
            self._depth = 1
            self._text = []
        elif self._href is not None:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._href is None:
            return
        self._depth -= 1
        if self._depth <= 0:
            text = re.sub(r"\s+", " ", " ".join(self._text)).strip()
            self.anchors.append((self._href, text))
            self._href = None
            self._depth = 0
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)


@dataclass(frozen=True)
class FIADocumentPublication:
    document_url: str
    document_title: str
    first_published_at: str
    published_text: str
    evidence_text: str


def _canonical_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    host = parsed.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+", "/", unquote(parsed.path)).rstrip("/")
    return host, path


def _published_timestamp(value: str) -> tuple[str, str]:
    match = re.search(
        r"Published\s+on\s+(\d{2})\.(\d{2})\.(\d{2}|\d{4})\s+"
        r"(\d{2}):(\d{2})\s+(CET|CEST|UTC|GMT)\b",
        value,
        flags=re.I,
    )
    if match is None:
        raise ValueError("FIA document entry has no unambiguous publication timestamp")
    day, month, raw_year, hour, minute, zone = match.groups()
    year = int(raw_year)
    if year < 100:
        year += 2000
    zone_upper = zone.upper()
    publication_zone = (
        ZoneInfo("Europe/Paris")
        if zone_upper in {"CET", "CEST"}
        else timezone.utc
    )
    timestamp = datetime(
        year,
        int(month),
        int(day),
        int(hour),
        int(minute),
        tzinfo=publication_zone,
    )
    normalized = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return normalized, match.group(0)


def parse_fia_document_publication_html(
    html: str,
    *,
    page_url: str,
    expected_document_url: str | None,
    expected_document_title: str,
) -> FIADocumentPublication:
    """Bind one exact FIA PDF to the timestamp rendered in its page card."""

    parser = _DecisionDocumentHTMLParser()
    parser.feed(html)
    expected_url_key = (
        _canonical_url(expected_document_url) if expected_document_url else None
    )
    title_token = re.sub(r"\s+", " ", expected_document_title).strip().casefold()
    matches: list[tuple[str, str]] = []
    for href, evidence_text in parser.anchors:
        absolute_url = urljoin(page_url, href)
        if not absolute_url.lower().split("?", 1)[0].endswith(".pdf"):
            continue
        if expected_url_key is not None:
            if _canonical_url(absolute_url) != expected_url_key:
                continue
        elif title_token not in evidence_text.casefold():
            continue
        if title_token not in evidence_text.casefold():
            raise ValueError(
                "supplied FIA document URL does not match the expected document title"
            )
        matches.append((absolute_url, evidence_text))
    if len(matches) != 1:
        raise ValueError(
            "FIA publication page must resolve exactly one matching final-grid document; "
            f"matches={len(matches)}"
        )
    document_url, evidence_text = matches[0]
    first_published_at, published_text = _published_timestamp(evidence_text)
    title_text = re.sub(
        r"\s*Published\s+on\s+.*$", "", evidence_text, flags=re.I
    ).strip()
    return FIADocumentPublication(
        document_url=document_url,
        document_title=title_text,
        first_published_at=first_published_at,
        published_text=published_text,
        evidence_text=evidence_text,
    )


def _plain_token(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def _positive_int(value: object, maximum: int) -> int | None:
    text = re.sub(r"[^0-9]", "", str(value))
    if not text:
        return None
    number = int(text)
    return number if 1 <= number <= maximum else None


def parse_fia_grid_official_html(
    html: str,
    *,
    expected_field_size: int,
) -> list[dict[str, object]]:
    """Return strict position/car/name rows from the FIA official-grid table."""

    parser = _GridOfficialHTMLParser()
    parser.feed(html)
    records: list[dict[str, object]] = []
    for cells in parser.rows:
        compact_cells = [re.sub(r"\s+", " ", cell).strip() for cell in cells if cell.strip()]
        if not compact_cells:
            continue
        lowered = " ".join(compact_cells).lower()
        pit_lane = "pit lane" in lowered
        withdrawn = "withdraw" in lowered
        dns = "did not start" in lowered or re.search(r"\bdns\b", lowered) is not None
        disqualified = "disqual" in lowered or re.search(r"\bdsq\b", lowered) is not None

        position: int | None = None
        position_cell_index: int | None = None
        for index, cell in enumerate(compact_cells[:3]):
            candidate = _positive_int(cell, expected_field_size)
            if candidate is not None and re.fullmatch(r"\D*\d{1,2}\D*", cell):
                position = candidate
                position_cell_index = index
                break
        search_start = 0 if position_cell_index is None else position_cell_index + 1
        driver_number: int | None = None
        driver_cell_index: int | None = None
        for index in range(search_start, min(len(compact_cells), search_start + 4)):
            candidate = _positive_int(compact_cells[index], 999)
            if candidate is not None and re.fullmatch(r"\D*\d{1,3}\D*", compact_cells[index]):
                driver_number = candidate
                driver_cell_index = index
                break
        if driver_number is None:
            continue
        if position is None and not (pit_lane or withdrawn or dns or disqualified):
            continue
        name = ""
        if driver_cell_index is not None:
            for cell in compact_cells[driver_cell_index + 1 :]:
                if re.search(r"[A-Za-z]", cell) and not any(
                    token in cell.lower()
                    for token in ("team", "constructor", "chassis", "engine", "time")
                ):
                    name = cell
                    break
        if pit_lane:
            status = GridEntryStatus.PIT_LANE
            position = None
        elif withdrawn:
            status = GridEntryStatus.WITHDRAWN
            position = None
        elif dns:
            status = GridEntryStatus.DID_NOT_START
            position = None
        elif disqualified:
            status = GridEntryStatus.DISQUALIFIED
            position = None
        else:
            status = GridEntryStatus.GRID
        records.append(
            {
                "position": position,
                "driver_number": str(driver_number),
                "driver_name": name,
                "status": status.value,
                "source_cells": compact_cells,
            }
        )

    by_driver = {str(row["driver_number"]): row for row in records}
    records = list(by_driver.values())
    positions = [int(row["position"]) for row in records if row["position"] is not None]
    if len(records) != int(expected_field_size):
        raise ValueError(
            f"FIA GRID OFFICIAL roster has {len(records)} rows; expected {expected_field_size}"
        )
    if len(positions) != len(set(positions)):
        raise ValueError("FIA GRID OFFICIAL table has duplicate physical positions")
    return sorted(
        records,
        key=lambda row: (
            row["position"] is None,
            int(row["position"]) if row["position"] is not None else 10**6,
            str(row["driver_number"]),
        ),
    )


def parse_fia_qualifying_roster_html(
    html: str,
    *,
    expected_field_size: int,
) -> dict[str, str]:
    """Recover only driver number/name identity from FIA classification tables."""

    parser = _AllTableHTMLParser()
    parser.feed(html)
    roster: dict[str, str] = {}
    for raw_cells in parser.rows:
        cells = [re.sub(r"\s+", " ", cell).strip() for cell in raw_cells if cell.strip()]
        numeric_indexes = [
            index
            for index, cell in enumerate(cells[:6])
            if re.fullmatch(r"\D*\d{1,3}\D*", cell)
        ]
        if len(numeric_indexes) < 2:
            continue
        driver_index = numeric_indexes[1]
        driver = _positive_int(cells[driver_index], 999)
        if driver is None:
            continue
        name_candidates = [
            cell
            for cell in cells[driver_index + 1 : driver_index + 5]
            if re.search(r"[A-Za-z]", cell)
            and not any(
                token in cell.lower()
                for token in (
                    "position",
                    "driver",
                    "team",
                    "constructor",
                    "time",
                    "lap",
                    "tyre",
                )
            )
        ]
        if not name_candidates:
            continue
        # Names normally contain a space and are more informative than a
        # three-letter nationality cell.
        name = max(name_candidates, key=lambda value: (" " in value, len(value)))
        normalized_name = _plain_token(name)
        if len(normalized_name) < 4:
            continue
        existing = roster.get(str(driver))
        if existing is None or len(_plain_token(existing)) < len(normalized_name):
            roster[str(driver)] = name
    if len(roster) < int(expected_field_size):
        raise ValueError(
            f"FIA classification roster has {len(roster)} identities; expected {expected_field_size}"
        )
    # Some pages repeat historical/support-series tables.  The first complete
    # F1 roster is bounded by the expected field size; duplicate car numbers
    # with conflicting names already resolve to the most informative identity.
    return dict(list(sorted(roster.items(), key=lambda item: int(item[0])))[:expected_field_size])


def parse_fia_grid_pdf_text(
    pdf_text: str,
    *,
    roster: dict[str, str] | None = None,
    expected_field_size: int,
) -> list[dict[str, object]]:
    """Parse the self-contained FIA grid page, with roster fallback if needed.

    Current Final Starting Grid PDFs expose rows as ``position car_number
    Driver NAME``. Layout extraction may return the two page columns in either
    visual or column order; positions are therefore treated as keys, not row
    order. Older collapsed extraction can still be disambiguated against an
    optional pre-race roster.
    """

    # ``pypdf`` layout extraction preserves the FIA grid's two visual columns,
    # but the right-hand position is printed on the following team-name line.
    # Recover those cells before trying the simpler reading-order parser below.
    # This deliberately relies only on the signed FIA PDF: no qualifying or
    # race-result roster is needed to identify the drivers.
    layout_candidates: list[dict[str, object]] = []
    pending_right: dict[str, object] | None = None
    layout_status_section: GridEntryStatus | None = None
    driver_name_pattern = re.compile(
        r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*"
        r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*)*"
        r"\s+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ' -]+$"
    )

    def _layout_cells(line: str) -> list[tuple[int, str]]:
        return [
            (match.start(), match.group(0).strip())
            for match in re.finditer(r"\S+(?: \S+)*", line.rstrip())
        ]

    for line in pdf_text.splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip().upper()
        if "DRIVERS REQUIRED TO START FROM THE PIT LANE" in normalized_line:
            layout_status_section = GridEntryStatus.PIT_LANE
            pending_right = None
            continue
        if normalized_line.startswith(("* PENALTIES", "NOTES", "THE STEWARDS")):
            layout_status_section = None
        cells = _layout_cells(line)
        if not cells:
            continue
        resolved_pending = False

        # A pending right-column car/name tuple is followed by a line whose
        # first right-column cell is the grid position and whose next cell is
        # the team name. Match by horizontal location, not by visual row order.
        if pending_right is not None:
            pending_x = int(pending_right["_x"])
            position_cells = [
                (x, int(value))
                for x, value in cells
                if value.isdigit()
                and 1 <= int(value) <= int(expected_field_size)
                and abs(x - pending_x) <= 16
            ]
            if len(position_cells) == 1:
                pending_right.pop("_x", None)
                pending_right["position"] = position_cells[0][1]
                layout_candidates.append(pending_right)
                pending_right = None
                resolved_pending = True

        if resolved_pending:
            continue

        for index, (x, value) in enumerate(cells[:-1]):
            if not value.isdigit():
                continue
            driver_number = value
            driver_name = re.sub(r"\s+", " ", cells[index + 1][1]).strip()
            driver_name = re.sub(r"\s*\*+\s*$", "", driver_name).strip()
            if not driver_name_pattern.fullmatch(driver_name):
                continue
            record: dict[str, object] = {
                "position": None,
                "driver_number": driver_number,
                "driver_name": driver_name,
                "status": GridEntryStatus.GRID.value,
                "source_cells": [driver_number, driver_name],
            }
            if layout_status_section is not None:
                record["status"] = layout_status_section.value
                record["source_cells"] = [
                    layout_status_section.value,
                    driver_number,
                    driver_name,
                ]
                layout_candidates.append(record)
                continue
            if index > 0 and cells[index - 1][1].isdigit():
                position = int(cells[index - 1][1])
                if 1 <= position <= int(expected_field_size):
                    record["position"] = position
                    record["source_cells"] = [
                        str(position),
                        driver_number,
                        driver_name,
                    ]
                    layout_candidates.append(record)
                    continue
            # Only one unresolved right-column row may exist at a time in the
            # FIA layout. A second candidate means the extraction is ambiguous
            # and must fail closed instead of silently assigning a position.
            if pending_right is not None:
                layout_candidates = []
                pending_right = None
                break
            record["_x"] = x
            pending_right = record

    layout_positions = [
        int(row["position"])
        for row in layout_candidates
        if row.get("position") is not None
    ]
    layout_drivers = [str(row.get("driver_number")) for row in layout_candidates]
    layout_status_rows = [
        row for row in layout_candidates if row.get("status") != GridEntryStatus.GRID.value
    ]
    if (
        pending_right is None
        and len(layout_candidates) == int(expected_field_size)
        and set(layout_positions) == set(range(1, len(layout_positions) + 1))
        and all(row.get("position") is None for row in layout_status_rows)
        and len(layout_drivers) == len(set(layout_drivers))
    ):
        return sorted(
            layout_candidates,
            key=lambda row: (
                row.get("position") is None,
                int(row["position"]) if row.get("position") is not None else 10**6,
                str(row["driver_number"]),
            ),
        )

    direct_records: list[dict[str, object]] = []
    direct_drivers: set[str] = set()
    direct_positions: set[int] = set()
    status_pattern = re.compile(
        r"^(PIT\s*LANE|DID\s*NOT\s*START|DNS|WITHDRAWN|DISQUALIFIED)"
        r"\s+(\d{1,3})\s+(.+)$",
        flags=re.I,
    )
    row_pattern = re.compile(r"^(\d{1,2})\s+(\d{1,3})\s+(.+)$")
    for line in pdf_text.splitlines():
        for segment in re.split(r"\s{3,}", line.strip()):
            candidate = re.sub(r"\s+", " ", segment).strip(" |")
            if not candidate:
                continue
            status_match = status_pattern.match(candidate)
            if status_match is not None:
                status_text, driver, name = status_match.groups()
                status_key = re.sub(r"\s+", "", status_text).upper()
                status = {
                    "PITLANE": GridEntryStatus.PIT_LANE,
                    "DIDNOTSTART": GridEntryStatus.DID_NOT_START,
                    "DNS": GridEntryStatus.DID_NOT_START,
                    "WITHDRAWN": GridEntryStatus.WITHDRAWN,
                    "DISQUALIFIED": GridEntryStatus.DISQUALIFIED,
                }[status_key]
                if driver not in direct_drivers and re.search(r"[A-Za-z]", name):
                    direct_records.append(
                        {
                            "position": None,
                            "driver_number": driver,
                            "driver_name": name.strip(),
                            "status": status.value,
                            "source_cells": [status.value, driver, name.strip()],
                        }
                    )
                    direct_drivers.add(driver)
                continue
            row_match = row_pattern.match(candidate)
            if row_match is None:
                continue
            raw_position, driver, name = row_match.groups()
            position = int(raw_position)
            if not 1 <= position <= int(expected_field_size):
                continue
            # Reject headers, lap times, and footer prose masquerading as a row.
            normalized_name = re.sub(r"\s+", " ", name).strip(" |")
            if (
                not re.search(r"[A-Za-z]", normalized_name)
                or re.search(r"\b(?:LAP|TIME|SPEED|TEAM|FORMULA)\b", normalized_name, re.I)
            ):
                continue
            if position in direct_positions or driver in direct_drivers:
                continue
            direct_records.append(
                {
                    "position": position,
                    "driver_number": driver,
                    "driver_name": normalized_name,
                    "status": GridEntryStatus.GRID.value,
                    "source_cells": [str(position), driver, normalized_name],
                }
            )
            direct_positions.add(position)
            direct_drivers.add(driver)
    if len(direct_records) == int(expected_field_size):
        return sorted(
            direct_records,
            key=lambda row: (
                row["position"] is None,
                int(row["position"]) if row["position"] is not None else 10**6,
                str(row["driver_number"]),
            ),
        )
    if roster is None:
        raise ValueError(
            "FIA PDF layout did not expose a complete self-contained grid; "
            f"resolved={len(direct_records)} expected={expected_field_size}"
        )

    compact = _plain_token(pdf_text)
    assignments: dict[int, tuple[str, str]] = {}
    used_drivers: set[str] = set()
    for position in range(1, int(expected_field_size) + 1):
        matches: list[tuple[str, str]] = []
        for driver, name in roster.items():
            normalized_name = _plain_token(name)
            prefixes = (
                normalized_name,
                normalized_name[-min(10, len(normalized_name)) :],
            )
            numeric_prefix = f"{position}{driver}"
            cursor = compact.find(numeric_prefix)
            while cursor >= 0:
                window = compact[
                    cursor + len(numeric_prefix) : cursor + len(numeric_prefix) + 100
                ]
                if any(prefix and prefix in window for prefix in prefixes):
                    matches.append((driver, name))
                    break
                cursor = compact.find(numeric_prefix, cursor + 1)
        unique = {driver: name for driver, name in matches}
        if len(unique) == 1:
            driver, name = next(iter(unique.items()))
            if driver not in used_drivers:
                assignments[position] = (driver, name)
                used_drivers.add(driver)

    records = [
        {
            "position": position,
            "driver_number": driver,
            "driver_name": name,
            "status": GridEntryStatus.GRID.value,
            "source_cells": [str(position), driver, name],
        }
        for position, (driver, name) in sorted(assignments.items())
    ]
    remaining = {driver: name for driver, name in roster.items() if driver not in used_drivers}
    for driver, name in remaining.items():
        normalized_name = _plain_token(name)
        status: GridEntryStatus | None = None
        for token, candidate in (
            ("PITLANE", GridEntryStatus.PIT_LANE),
            ("DIDNOTSTART", GridEntryStatus.DID_NOT_START),
            ("DNS", GridEntryStatus.DID_NOT_START),
            ("WITHDRAWN", GridEntryStatus.WITHDRAWN),
            ("DISQUALIFIED", GridEntryStatus.DISQUALIFIED),
        ):
            marker = f"{token}{driver}"
            cursor = compact.find(marker)
            if cursor >= 0 and normalized_name[: min(8, len(normalized_name))] in compact[
                cursor : cursor + 140
            ]:
                status = candidate
                break
        if status is not None:
            records.append(
                {
                    "position": None,
                    "driver_number": driver,
                    "driver_name": name,
                    "status": status.value,
                    "source_cells": [status.value, driver, name],
                }
            )
    if len(records) != int(expected_field_size):
        missing = sorted(set(roster) - {str(row["driver_number"]) for row in records})
        raise ValueError(
            "FIA PDF layout parser did not resolve the complete field; "
            f"resolved={len(records)} expected={expected_field_size} missing={missing}"
        )
    if len(assignments) != len(set(assignments)):
        raise ValueError("FIA PDF layout parser emitted duplicate positions")
    return sorted(
        records,
        key=lambda row: (
            row["position"] is None,
            int(row["position"]) if row["position"] is not None else 10**6,
            str(row["driver_number"]),
        ),
    )


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional operational dependency
        raise RuntimeError(
            "FIA grid capture requires pypdf; install the optional fia capture dependency"
        ) from exc
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        # Preserve visual cells. Mixing reading-order and layout extraction in
        # one string can create a second, collapsed copy of every row and turn a
        # valid pit-lane entry into an ambiguous duplicate.
        layout = page.extract_text(extraction_mode="layout") or ""
        pages.append(layout or (page.extract_text() or ""))
    return "\n".join(pages)


def cross_check_grid_against_pdf(
    rows: Iterable[dict[str, object]],
    pdf_text: str,
) -> None:
    """Require every structured position/car/name tuple in the official PDF."""

    compact = _plain_token(pdf_text)
    failures: list[str] = []
    for row in rows:
        position = row.get("position")
        driver = str(row.get("driver_number"))
        name = _plain_token(row.get("driver_name"))
        if position is None:
            prefix_options = (f"PITLANE{driver}", f"DNS{driver}", driver)
        else:
            prefix_options = (f"{int(position)}{driver}",)
        found = False
        for prefix in prefix_options:
            cursor = compact.find(prefix)
            while cursor >= 0:
                window = compact[cursor : cursor + max(120, len(prefix) + len(name) + 30)]
                if not name or name[: min(8, len(name))] in window:
                    found = True
                    break
                cursor = compact.find(prefix, cursor + 1)
            if found:
                break
        if not found:
            failures.append(f"{position}:{driver}:{name}")
    if failures:
        raise ValueError(
            "FIA PDF does not confirm structured GRID OFFICIAL pairs: "
            + ",".join(failures[:5])
        )


def _decision_from_document_text(
    row: dict[str, object],
    pdf_text: str,
    *,
    evidence_id: str,
) -> tuple[OfficialGridDecision, ...]:
    status = GridEntryStatus(str(row["status"]))
    kind = {
        GridEntryStatus.PIT_LANE: GridAdjustmentKind.PIT_LANE_START,
        GridEntryStatus.WITHDRAWN: GridAdjustmentKind.WITHDRAWAL,
        GridEntryStatus.DISQUALIFIED: GridAdjustmentKind.DISQUALIFICATION,
    }.get(status)
    driver = re.escape(str(row["driver_number"]))
    sentence_match = re.search(
        rf"(?is)(?:car|driver)\s*(?:no\.?\s*)?{driver}\b(.{{0,220}}?)(?:\n|\.)",
        pdf_text,
    )
    reason = None
    places = None
    if sentence_match is not None:
        candidate = re.sub(r"\s+", " ", sentence_match.group(0)).strip()
        explicit = any(
            token in candidate.lower()
            for token in (
                "grid penalty",
                "back of the grid",
                "pit lane",
                "withdraw",
                "disqual",
                "did not start",
            )
        )
        if explicit:
            reason = candidate[:240] or None
            places_match = re.search(r"(\d+)\s*(?:place|position)", candidate, flags=re.I)
            if places_match:
                places = int(places_match.group(1))
            if kind is None and "back of the grid" in candidate.lower():
                kind = GridAdjustmentKind.BACK_OF_GRID
            elif kind is None and (places is not None or "grid penalty" in candidate.lower()):
                kind = GridAdjustmentKind.GRID_DROP
    if kind is None:
        return ()
    return (
        OfficialGridDecision(
            kind=kind,
            places=(places if kind in {GridAdjustmentKind.GRID_DROP, GridAdjustmentKind.BACK_OF_GRID} else None),
            reason=reason,
            evidence_id=evidence_id,
        ),
    )


def _read_or_fetch_bytes(path: Path | None, url: str | None, *, timeout: float) -> bytes:
    if path is not None:
        return path.read_bytes()
    if url is None:
        raise ValueError("a source URL or local file is required")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _read_or_fetch_pdf(
    path: Path | None,
    url: str,
    *,
    timeout: float,
    temporary_dir: Path,
) -> Path:
    if path is not None:
        return path
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    temporary_dir.mkdir(parents=True, exist_ok=True)
    target = temporary_dir / "fia_starting_grid_source.pdf"
    target.write_bytes(response.content)
    return target


def capture(
    *,
    year: int,
    round_number: int,
    classification_url: str | None,
    publication_page_url: str,
    document_url: str | None,
    first_published_at: str | None,
    race_start_at: str,
    output_dir: Path,
    document_title: str,
    html_file: Path | None = None,
    publication_html_file: Path | None = None,
    pdf_file: Path | None = None,
    captured_at: str | None = None,
    timeout: float = 30.0,
) -> tuple[Path, dict[str, object]]:
    contract = build_weekend_contract(int(year))
    classification_bytes = (
        _read_or_fetch_bytes(html_file, classification_url, timeout=timeout)
        if html_file is not None or classification_url is not None
        else b""
    )
    publication_page_bytes = _read_or_fetch_bytes(
        publication_html_file, publication_page_url, timeout=timeout
    )
    html = classification_bytes.decode("utf-8", errors="strict")
    publication_html = publication_page_bytes.decode("utf-8", errors="strict")
    publication = parse_fia_document_publication_html(
        publication_html,
        page_url=publication_page_url,
        expected_document_url=document_url,
        expected_document_title=document_title,
    )
    if first_published_at is not None:
        supplied = pd.to_datetime(first_published_at, errors="coerce", utc=True)
        authoritative = pd.to_datetime(
            publication.first_published_at, errors="coerce", utc=True
        )
        if pd.isna(supplied) or pd.isna(authoritative) or supplied != authoritative:
            raise ValueError(
                "supplied first_published_at does not equal the exact FIA page evidence"
            )
    authoritative_published_at = publication.first_published_at
    authoritative_document_url = publication.document_url
    classification_page_hash = hashlib.sha256(classification_bytes).hexdigest()
    publication_page_hash = hashlib.sha256(publication_page_bytes).hexdigest()
    capture_time = pd.to_datetime(
        captured_at or datetime.now(timezone.utc), errors="coerce", utc=True
    )
    if pd.isna(capture_time):
        raise ValueError("captured_at must be timezone-aware")
    pdf_path = _read_or_fetch_pdf(
        pdf_file,
        authoritative_document_url,
        timeout=timeout,
        temporary_dir=output_dir / ".capture_tmp",
    )
    document_bytes = pdf_path.read_bytes()
    document_hash = hashlib.sha256(document_bytes).hexdigest()
    extracted_text = _pdf_text(pdf_path)
    extraction_source = "fia_grid_official_html_cross_checked_by_pdf"
    try:
        if not html:
            raise ValueError("optional GRID OFFICIAL HTML not supplied")
        rows = parse_fia_grid_official_html(
            html,
            expected_field_size=contract.eligible_cars,
        )
        cross_check_grid_against_pdf(rows, extracted_text)
    except ValueError as html_error:
        try:
            rows = parse_fia_grid_pdf_text(
                extracted_text,
                expected_field_size=contract.eligible_cars,
            )
            extraction_source = (
                "fia_self_contained_pdf_layout_fallback:"
                f"{type(html_error).__name__}"
            )
        except ValueError as pdf_error:
            if not html:
                raise ValueError(
                    "FIA PDF failed self-contained parsing and no optional HTML roster exists"
                ) from pdf_error
            roster = parse_fia_qualifying_roster_html(
                html,
                expected_field_size=contract.eligible_cars,
            )
            rows = parse_fia_grid_pdf_text(
                extracted_text,
                roster=roster,
                expected_field_size=contract.eligible_cars,
            )
            extraction_source = (
                "fia_pdf_layout_with_optional_html_roster_fallback:"
                f"{type(html_error).__name__}:{type(pdf_error).__name__}"
            )
    title_lower = publication.document_title.strip().lower()
    phase = (
        GridRevisionPhase.FINAL_PRE_RACE
        if "final" in title_lower and "starting grid" in title_lower
        else GridRevisionPhase.PROVISIONAL_PRE_RACE
    )
    revision_id = f"fia:{year}:{round_number}:{document_hash[:20]}"
    entries: list[OfficialGridEntry] = []
    for row in rows:
        status = GridEntryStatus(str(row["status"]))
        entries.append(
            OfficialGridEntry(
                driver_id=str(row["driver_number"]),
                position=(None if row["position"] is None else int(row["position"])),
                status=status,
                decisions=_decision_from_document_text(
                    row,
                    extracted_text,
                    evidence_id=revision_id,
                ),
                evidence_complete=True,
            )
        )
    revision = OfficialGridRevision(
        revision_id=revision_id,
        phase=phase,
        entries=tuple(entries),
        as_of=authoritative_published_at,
        evidence_complete=True,
    )
    raw_payload = {
        "classification_url": classification_url,
        "classification_page_sha256": (
            classification_page_hash if classification_bytes else None
        ),
        "publication_page_url": publication_page_url,
        "publication_page_sha256": publication_page_hash,
        "publication_page_document_evidence": publication.evidence_text,
        "publication_page_timestamp_evidence": publication.published_text,
        "publication_page_timezone_interpretation": (
            "FIA CET display interpreted as Europe/Paris civil time with IANA DST rules"
        ),
        "document_url": authoritative_document_url,
        "document_title": publication.document_title,
        "document_sha256": document_hash,
        "first_published_at": authoritative_published_at,
        "supplied_first_published_at": first_published_at,
        "race_start_at": race_start_at,
        "grid_rows": rows,
        "structured_extraction_source": extraction_source,
        "pdf_extracted_text": extracted_text,
    }
    envelope = build_race_grid_capture(
        contract,
        year=int(year),
        round_number=int(round_number),
        provider="fia",
        source_endpoint=publication_page_url,
        captured_at=pd.Timestamp(capture_time).isoformat().replace("+00:00", "Z"),
        first_published_at=authoritative_published_at,
        race_start_at=race_start_at,
        publication_time_semantics="authoritative_document_timestamp",
        source_document_url=authoritative_document_url,
        source_document_sha256=document_hash,
        revision=revision,
        raw_payload=raw_payload,
    )
    output = persist_race_grid_capture(envelope, output_dir)
    if pdf_file is None:
        try:
            pdf_path.unlink()
            pdf_path.parent.rmdir()
        except OSError:
            pass
    return output, {
        "capture_id": envelope.capture_id,
        "snapshot_available": envelope.snapshot.available,
        "resolution_status": envelope.snapshot.resolution_status.value,
        "revision_phase": envelope.revision_phase.value,
        "first_published_at": envelope.first_published_at,
        "captured_at": envelope.captured_at,
        "document_sha256": document_hash,
        "publication_page_sha256": publication_page_hash,
        "classification_page_sha256": (
            classification_page_hash if classification_bytes else None
        ),
        "field_size": len(envelope.snapshot.entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument(
        "--classification-url",
        help="optional GRID OFFICIAL HTML cross-check; PDF parsing is self-contained",
    )
    parser.add_argument("--publication-page-url", required=True)
    parser.add_argument(
        "--document-url",
        help="optional exact PDF assertion; otherwise derived from the publication page",
    )
    parser.add_argument("--document-title", required=True)
    parser.add_argument(
        "--first-published-at",
        help="optional equality assertion; authoritative time is parsed from the FIA page",
    )
    parser.add_argument(
        "--race-start-at",
        required=True,
        help="scheduled Race start as timezone-aware ISO-8601",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html-file", type=Path)
    parser.add_argument("--publication-html-file", type=Path)
    parser.add_argument("--pdf-file", type=Path)
    parser.add_argument("--captured-at", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    output, summary = capture(
        year=args.year,
        round_number=args.round_number,
        classification_url=args.classification_url,
        publication_page_url=args.publication_page_url,
        document_url=args.document_url,
        document_title=args.document_title,
        first_published_at=args.first_published_at,
        race_start_at=args.race_start_at,
        output_dir=args.output_dir.expanduser().resolve(),
        html_file=(None if args.html_file is None else args.html_file.expanduser().resolve()),
        publication_html_file=(
            None
            if args.publication_html_file is None
            else args.publication_html_file.expanduser().resolve()
        ),
        pdf_file=(None if args.pdf_file is None else args.pdf_file.expanduser().resolve()),
        captured_at=args.captured_at,
        timeout=args.timeout,
    )
    print(json.dumps({"output": str(output), **summary}, indent=2, sort_keys=True))
    return 0 if bool(summary["snapshot_available"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: feat(f1-grid): capture authoritative FIA grid revisions
