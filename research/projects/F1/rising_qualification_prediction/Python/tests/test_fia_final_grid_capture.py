from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import capture_fia_final_grid_snapshot as capture_module  # noqa: E402


def _publication_html(timestamp: str, document_path: str) -> str:
    return f"""
    <ul><li class="document-row key-55"><a href="{document_path}">
      <div class="title">Doc 55 - Final Starting Grid</div>
      <div class="published">Published on <span>{timestamp}</span> CET</div>
    </a></li></ul>
    """


@pytest.mark.parametrize(
    ("display", "expected_utc"),
    [
        ("08.03.26 04:00", "2026-03-08T03:00:00Z"),
        # FIA labels this CET, but its cards use Paris civil time in summer.
        ("24.05.26 21:00", "2026-05-24T19:00:00Z"),
        # Europe/Paris has switched to summer time by 06:00 on 29 March.
        ("29.03.26 06:00", "2026-03-29T04:00:00Z"),
    ],
)
def test_publication_card_timestamp_uses_paris_dst_and_binds_exact_pdf(
    display: str,
    expected_utc: str,
) -> None:
    page_url = "https://www.fia.com/documents/event/Japanese%20Grand%20Prix"
    document_path = "/system/files/decision-document/final_starting_grid.pdf"
    evidence = capture_module.parse_fia_document_publication_html(
        _publication_html(display, document_path),
        page_url=page_url,
        expected_document_url=f"https://www.fia.com{document_path}",
        expected_document_title="Final Starting Grid",
    )

    assert evidence.document_url == f"https://www.fia.com{document_path}"
    assert evidence.first_published_at == expected_utc
    assert evidence.published_text == f"Published on {display} CET"


def test_publication_card_rejects_url_title_mismatch() -> None:
    with pytest.raises(ValueError, match="expected document title"):
        capture_module.parse_fia_document_publication_html(
            _publication_html(
                "29.03.26 06:00",
                "/system/files/decision-document/provisional_grid.pdf",
            ).replace("Final Starting Grid", "Provisional Starting Grid"),
            page_url="https://www.fia.com/documents/event/Japanese%20Grand%20Prix",
            expected_document_url=(
                "https://www.fia.com/system/files/decision-document/provisional_grid.pdf"
            ),
            expected_document_title="Final Starting Grid",
        )


def test_self_contained_two_column_pdf_layout_needs_no_html_roster() -> None:
    text = """
        2 4 Lando NORRIS                     1 63 George RUSSELL
        4 16 Charles LECLERC                 3 12 Kimi ANTONELLI
    """
    rows = capture_module.parse_fia_grid_pdf_text(
        text,
        expected_field_size=4,
    )

    assert [row["position"] for row in rows] == [1, 2, 3, 4]
    assert [row["driver_number"] for row in rows] == ["63", "4", "12", "16"]


def test_real_fia_layout_with_right_position_on_team_line_needs_no_roster() -> None:
    text = """
          1      63   George RUSSELL                                  1:18.518
                      Mercedes-AMG PETRONAS F1 Team                                                        12   Kimi ANTONELLI                                   1:18.811
                                                                                                     2          Mercedes-AMG PETRONAS F1 Team
          3       6   Isack HADJAR                                    1:19.303
                      Oracle Red Bull Racing                                                               16   Charles LECLERC                                  1:19.327
                                                                                                     4          Scuderia Ferrari HP
    """
    rows = capture_module.parse_fia_grid_pdf_text(
        text,
        expected_field_size=4,
    )

    assert [row["position"] for row in rows] == [1, 2, 3, 4]
    assert [row["driver_number"] for row in rows] == ["63", "12", "6", "16"]


def test_real_fia_layout_parses_penalty_marker_and_pit_lane_section() -> None:
    text = """
          1      63   George RUSSELL                                  1:18.518
                      Mercedes-AMG PETRONAS F1 Team                                                        12   Kimi ANTONELLI *                                 1:18.811
                                                                                                     2          Mercedes-AMG PETRONAS F1 Team
          3       6   Isack HADJAR                                    1:19.303
                      Oracle Red Bull Racing
              DRIVERS REQUIRED TO START FROM THE PIT LANE
                23    Alexander ALBON *                               1:20.100
                      Atlassian Williams F1 Team
    """
    rows = capture_module.parse_fia_grid_pdf_text(
        text,
        expected_field_size=4,
    )

    assert [row["driver_number"] for row in rows] == ["63", "12", "6", "23"]
    assert rows[-1]["position"] is None
    assert rows[-1]["status"] == "pit_lane"
    assert rows[-1]["driver_name"] == "Alexander ALBON"


def test_capture_derives_page_timestamp_and_persists_page_and_pdf_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = "/system/files/decision-document/2026_final_starting_grid.pdf"
    publication_file = tmp_path / "publication.html"
    publication_file.write_text(
        _publication_html("08.03.26 04:00", document_path),
        encoding="utf-8",
    )
    table_rows = "".join(
        f"<tr><td>{position}</td><td>{100 + position}</td>"
        f"<td>Driver {position}</td></tr>"
        for position in range(1, 23)
    )
    classification_file = tmp_path / "classification.html"
    classification_file.write_text(
        f'<h3 data="GRID OFFICIAL">GRID OFFICIAL</h3><table>{table_rows}</table>',
        encoding="utf-8",
    )
    pdf_file = tmp_path / "grid.pdf"
    pdf_file.write_bytes(b"synthetic-pdf-evidence")
    pdf_text = "\n".join(
        f"{position} {100 + position} Driver {position}"
        for position in range(1, 23)
    )
    monkeypatch.setattr(capture_module, "_pdf_text", lambda _: pdf_text)

    output, summary = capture_module.capture(
        year=2026,
        round_number=1,
        classification_url="https://www.fia.com/session-classifications",
        publication_page_url="https://www.fia.com/documents/event/Australia",
        document_url=f"https://www.fia.com{document_path}",
        first_published_at="2026-03-08T03:00:00Z",
        race_start_at="2026-03-08T04:00:00Z",
        output_dir=tmp_path / "captures",
        document_title="Final Starting Grid",
        html_file=classification_file,
        publication_html_file=publication_file,
        pdf_file=pdf_file,
        captured_at="2026-03-08T03:05:00Z",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    raw = payload["raw_payload"]
    assert summary["snapshot_available"] is True
    assert payload["first_published_at"] == "2026-03-08T03:00:00Z"
    assert raw["publication_page_timestamp_evidence"] == (
        "Published on 08.03.26 04:00 CET"
    )
    assert raw["publication_page_sha256"] == hashlib.sha256(
        publication_file.read_bytes()
    ).hexdigest()
    assert payload["source_document_sha256"] == hashlib.sha256(
        pdf_file.read_bytes()
    ).hexdigest()


# Suggested commit name: test(f1-grid): verify authoritative FIA publication capture
