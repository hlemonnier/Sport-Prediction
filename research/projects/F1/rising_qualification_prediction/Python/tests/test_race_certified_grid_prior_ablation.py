from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_race_certified_grid_prior_ablation as race_module  # noqa: E402
from run_race_certified_grid_prior_ablation import (  # noqa: E402
    FIA_FINAL_GRID_DOCUMENTS,
    MINIMUM_RELATIVE_SELECTION_GAIN,
    PARTITIONS,
    PROFILES,
    _canonical_sha256,
    _legal_grid_order,
    _partition_for_round,
    _root,
    _score_profile,
    _select_challenger,
    _validate_capture_certification,
    run,
)


def _certified_capture_fixture() -> tuple[
    dict[str, object], dict[str, object], Path
]:
    capture_path = sorted(
        (_root() / "data/f1/raw/weekends/2026").glob(
            "round_*/first_seen_grid_snapshots/grid_*.json"
        )
    )[0]
    metadata_path = capture_path.parents[1] / "weekend_metadata.json"
    return (
        json.loads(capture_path.read_text(encoding="utf-8")),
        json.loads(metadata_path.read_text(encoding="utf-8")),
        capture_path.parents[1] / "evidence/fia_final_starting_grid.pdf",
    )


def test_all_nine_certified_grid_captures_pass_reconstructed_evidence_gates() -> None:
    capture_paths = sorted(
        (_root() / "data/f1/raw/weekends/2026").glob(
            "round_*/first_seen_grid_snapshots/grid_*.json"
        )
    )
    assert len(capture_paths) == 9
    for capture_path in capture_paths:
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        metadata = json.loads(
            (capture_path.parents[1] / "weekend_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        checks = _validate_capture_certification(
            capture,
            metadata,
            source_document_path=(
                capture_path.parents[1] / "evidence/fia_final_starting_grid.pdf"
            ),
            year=int(capture["year"]),
            round_number=int(capture["round_number"]),
        )
        assert checks
        assert all(checks.values())


def test_artifact_manifest_binds_all_nine_parsed_pdf_inputs() -> None:
    payload = run(weekends_dir=_root() / "data/f1/raw/weekends")
    pdf_inputs = {
        row["path"]: row["sha256"]
        for row in payload["input_manifest"]
        if str(row["path"]).endswith("fia_final_starting_grid.pdf")
    }

    assert pdf_inputs == {
        document["relative_path"]: document["sha256"]
        for document in FIA_FINAL_GRID_DOCUMENTS.values()
    }
    assert payload["manifest_hashes"]["input_manifest_sha256"] == (
        _canonical_sha256(payload["input_manifest"])
    )
    assert all(
        all(event["certified_grid_evidence"]["certification_checks"].values())
        for event in payload["events"]
    )
    assert [
        event["certified_grid_evidence"]["publication_lag_seconds"]
        for event in payload["events"]
    ] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0]
    assert all(
        event["certified_grid_evidence"][
            "logical_horizon_certified_by_authoritative_publication_time"
        ]
        for event in payload["events"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "unsupported"),
        ("year", 2025),
        ("round_number", 99),
        ("provider", "self_asserted"),
        ("publication_time_semantics", "first_seen_upper_bound"),
        ("source_document_url", "https://example.com/grid.pdf"),
        ("source_document_sha256", "not-a-digest"),
        ("raw_payload_sha256", "0" * 64),
    ],
)
def test_certification_rejects_fabricated_top_level_assertions(
    field: str,
    value: object,
) -> None:
    capture, metadata, pdf_path = _certified_capture_fixture()
    capture[field] = value

    with pytest.raises(ValueError, match="not horizon-certified"):
        _validate_capture_certification(
            capture,
            metadata,
            source_document_path=pdf_path,
            year=2026,
            round_number=1,
        )


def test_certification_rejects_nested_time_and_metadata_mismatches() -> None:
    capture, metadata, pdf_path = _certified_capture_fixture()
    mutations = (
        ("snapshot_prediction", "2026-12-01T00:00:00Z"),
        ("raw_publication", "2026-12-01T00:00:00Z"),
        ("raw_race_start", "2026-12-01T00:00:00Z"),
        ("metadata_race_start", "2026-12-01T00:00:00Z"),
    )
    for mutation, value in mutations:
        candidate = copy.deepcopy(capture)
        candidate_metadata = copy.deepcopy(metadata)
        if mutation == "snapshot_prediction":
            candidate["snapshot"]["prediction_as_of"] = value
        elif mutation == "raw_publication":
            candidate["raw_payload"]["first_published_at"] = value
            candidate["raw_payload_sha256"] = (
                "0" * 64
            )
        elif mutation == "raw_race_start":
            candidate["raw_payload"]["race_start_at"] = value
            candidate["raw_payload_sha256"] = (
                "0" * 64
            )
        else:
            race = next(
                row
                for row in candidate_metadata["sessions"]
                if row["session_type"] == "race"
            )
            race["scheduled_start_utc"] = value

        with pytest.raises(ValueError, match="not horizon-certified"):
            _validate_capture_certification(
                candidate,
                candidate_metadata,
                source_document_path=pdf_path,
                year=2026,
                round_number=1,
            )


def test_certification_binds_claimed_publication_to_pdf_local_timestamp() -> None:
    capture, metadata, pdf_path = _certified_capture_fixture()
    candidate = copy.deepcopy(capture)
    forged_publication = "2026-03-08T00:00:00Z"
    candidate["first_published_at"] = forged_publication
    candidate["snapshot"]["prediction_as_of"] = forged_publication
    candidate["snapshot"]["publication_as_of"] = forged_publication
    candidate["raw_payload"]["first_published_at"] = forged_publication
    candidate["raw_payload_sha256"] = _canonical_sha256(candidate["raw_payload"])

    with pytest.raises(ValueError, match="not horizon-certified"):
        _validate_capture_certification(
            candidate,
            metadata,
            source_document_path=pdf_path,
            year=2026,
            round_number=1,
        )


@pytest.mark.parametrize("forgery", ["document_title", "document_time", "grid_rows"])
def test_certification_rejects_coherent_structured_forgery_against_pdf_bytes(
    forgery: str,
) -> None:
    capture, metadata, pdf_path = _certified_capture_fixture()
    candidate = copy.deepcopy(capture)
    raw = candidate["raw_payload"]
    assert isinstance(raw, dict)

    if forgery == "document_title":
        raw["document_title"] = "Doc 999 - Final Starting Grid"
    elif forgery == "document_time":
        extracted = str(raw["pdf_extracted_text"])
        mutated, replacements = re.subn(
            r"Time[\s\u00a0]+14:00", "Time 15:00", extracted, count=1
        )
        assert replacements == 1
        raw["pdf_extracted_text"] = mutated
    else:
        rows = raw["grid_rows"]
        entries = candidate["snapshot"]["entries"]
        assert isinstance(rows, list) and isinstance(entries, list)
        first_number = str(rows[0]["driver_number"])
        second_number = str(rows[1]["driver_number"])
        rows[0]["position"], rows[1]["position"] = (
            rows[1]["position"],
            rows[0]["position"],
        )
        entry_by_number = {str(row["driver_id"]): row for row in entries}
        entry_by_number[first_number]["grid_position"] = rows[0]["position"]
        entry_by_number[second_number]["grid_position"] = rows[1]["position"]

    candidate["raw_payload_sha256"] = _canonical_sha256(raw)
    with pytest.raises(ValueError, match="not horizon-certified"):
        _validate_capture_certification(
            candidate,
            metadata,
            source_document_path=pdf_path,
            year=2026,
            round_number=1,
        )


def test_pinned_catalog_rejects_pdf_and_capture_rehashed_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, metadata, original_pdf = _certified_capture_fixture()
    candidate = copy.deepcopy(capture)
    relative_path = Path(FIA_FINAL_GRID_DOCUMENTS[1]["relative_path"])
    forged_pdf = tmp_path / relative_path
    forged_pdf.parent.mkdir(parents=True)
    forged_pdf.write_bytes(original_pdf.read_bytes() + b"\n% coherent-forgery")
    forged_sha256 = hashlib.sha256(forged_pdf.read_bytes()).hexdigest()
    candidate["source_document_sha256"] = forged_sha256
    candidate["raw_payload"]["document_sha256"] = forged_sha256
    candidate["raw_payload_sha256"] = _canonical_sha256(candidate["raw_payload"])

    monkeypatch.setattr(race_module, "_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="not horizon-certified"):
        _validate_capture_certification(
            candidate,
            metadata,
            source_document_path=forged_pdf,
            year=2026,
            round_number=1,
        )


def test_certified_grid_protocol_has_disjoint_fixed_chronological_blocks() -> None:
    assert PARTITIONS == {
        "development": (1, 2),
        "selection": (3, 4),
        "calibration": (5, 6),
        "audit": (7, 8, 9),
    }
    assert [_partition_for_round(round_number) for round_number in range(1, 10)] == [
        "development",
        "development",
        "selection",
        "selection",
        "calibration",
        "calibration",
        "audit",
        "audit",
        "audit",
    ]
    with pytest.raises(ValueError, match="outside the fixed R1-R9 protocol"):
        _partition_for_round(10)


def test_legal_grid_orders_pit_lane_and_nonstarter_after_physical_grid() -> None:
    frame = pd.DataFrame(
        {
            "grid_position": [2.0, None, 1.0, None],
            "grid_status": ["grid", "nonstarter", "grid", "pit_lane"],
        }
    )

    assert _legal_grid_order(frame).tolist() == [2, 4, 1, 3]


def _inference(event_key: int = 202603) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["1", "2", "3"],
            "driver_key": ["DRIVER ONE", "DRIVER TWO", "DRIVER THREE"],
            "grid_baseline_position": [1, 2, 3],
            "provider_order": [0, 1, 2],
            "event_key": [event_key, event_key, event_key],
        }
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_key": ["DRIVER ONE", "DRIVER TWO", "DRIVER THREE"],
            "team_name": ["A", "A", "B"],
            "event_key": [202601, 202601, 202601],
            "grid_residual": [4.0, -2.0, 0.0],
        }
    )


def test_profile_score_uses_prior_state_and_emits_one_complete_permutation() -> None:
    scored = _score_profile(_inference(), _history(), PROFILES[1])

    assert sorted(scored["candidate_predicted_position"].tolist()) == [1, 2, 3]
    assert scored.set_index("driver_id").loc["1", "driver_residual_effect"] > 0.0
    assert scored.set_index("driver_id").loc["2", "driver_residual_effect"] < 0.0
    assert scored["prior_driver_events"].eq(1).all()


def test_profile_score_rejects_target_fields_and_non_prior_history() -> None:
    contaminated = _inference().assign(actual_position=[1, 2, 3])
    with pytest.raises(ValueError, match="target-derived fields"):
        _score_profile(contaminated, _history(), PROFILES[0])

    future = pd.concat(
        [
            _history(),
            pd.DataFrame(
                {
                    "driver_key": ["DRIVER ONE"],
                    "team_name": ["A"],
                    "event_key": [202603],
                    "grid_residual": [-10.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="current or future Race state"):
        _score_profile(_inference(), future, PROFILES[0])


def test_selection_tie_or_sub_five_percent_gain_retains_grid_policy() -> None:
    rows = []
    for profile_index, profile in enumerate(PROFILES):
        for round_number in PARTITIONS["selection"]:
            rows.append(
                {
                    "event_key": 202600 + round_number,
                    "round": round_number,
                    "profile_id": profile.profile_id,
                    "baseline_mae": 3.0,
                    "candidate_mae": 3.0 + 0.1 * profile_index,
                }
            )

    selected = _select_challenger(rows)

    assert MINIMUM_RELATIVE_SELECTION_GAIN == 0.05
    assert selected["selected_challenger_profile_id"] == PROFILES[0].profile_id
    assert selected["selection_relative_gain"] == 0.0
    assert selected["challenger_selected_on_selection"] is False
    assert selected["public_selected_model_id"] == "legal_grid_baseline"


def test_descriptive_material_gain_cannot_promote_posthoc_profile_grid() -> None:
    rows = []
    for profile_index, profile in enumerate(PROFILES):
        for round_number in PARTITIONS["selection"]:
            rows.append(
                {
                    "event_key": 202600 + round_number,
                    "round": round_number,
                    "profile_id": profile.profile_id,
                    "baseline_mae": 4.0,
                    "candidate_mae": 3.0 if profile_index == 1 else 4.5,
                }
            )

    selected = _select_challenger(rows)

    assert selected["selected_challenger_profile_id"] == PROFILES[1].profile_id
    assert selected["selection_relative_gain"] == pytest.approx(0.25)
    assert selected["descriptive_candidate_would_clear_gain_threshold"] is True
    assert selected["formal_selection_evidence"] is False
    assert selected["challenger_selected_on_selection"] is False
    assert selected["public_selected_model_id"] == "legal_grid_baseline"


# Suggested commit name: test(f1-race): verify certified-grid ablation boundaries
