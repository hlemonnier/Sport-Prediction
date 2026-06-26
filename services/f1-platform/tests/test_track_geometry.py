from f1_platform.fastf1_analysis import FastF1ArtifactStore
from f1_platform.track_geometry import (
    FastF1CenterlineProjector,
    centerline_from_rows,
    parse_session_aliases,
    project_location_to_centerline,
)


def test_project_location_to_centerline_interpolates_distance_and_progress():
    centerline = centerline_from_rows(
        [
            {"Distance": 0, "Progress": 0, "X": 0, "Y": 0, "Z": 0},
            {"Distance": 10, "Progress": 0.5, "X": 10, "Y": 0, "Z": 1},
            {"Distance": 20, "Progress": 1, "X": 20, "Y": 0, "Z": 2},
        ],
        session_key="fastf1:2026:austria:r",
        artifact_id="centerline",
        source="test",
    )

    projection = project_location_to_centerline(6, 3, centerline.points)

    assert projection is not None
    assert projection.distance == 6
    assert projection.progress == 0.3
    assert projection.x == 6
    assert projection.y == 0
    assert projection.error == 3


def test_fastf1_centerline_projector_uses_artifact_alias(tmp_path):
    store = FastF1ArtifactStore(tmp_path, allow_json_fallback=True)
    store.write_table(
        [
            {"Distance": 0, "Progress": 0, "X": 0, "Y": 0, "Z": 0},
            {"Distance": 100, "Progress": 1, "X": 100, "Y": 0, "Z": 0},
        ],
        "centerline/year=2026/event=austria/session=r/canonical",
        preferred_format="jsonl",
        metadata={"kind": "fastf1_centerline", "sessionKey": "fastf1:2026:austria:r"},
    )
    projector = FastF1CenterlineProjector(
        store,
        session_aliases={"openf1-9165": "fastf1:2026:austria:r"},
    )

    projection = projector.project("openf1-9165", {"x": 25, "y": 4})

    assert projection is not None
    assert projection.progress == 0.25
    assert projection.distance == 25
    assert projection.error == 4
    assert "centerline/year=2026/event=austria/session=r/canonical.jsonl" in projection.source


def test_parse_session_aliases_ignores_malformed_items():
    assert parse_session_aliases("9165=fastf1:2026:austria:r, bad, sample=fastf1:2024:monaco:r") == {
        "9165": "fastf1:2026:austria:r",
        "sample": "fastf1:2024:monaco:r",
    }
