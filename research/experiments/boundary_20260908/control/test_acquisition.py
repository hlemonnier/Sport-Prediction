import pytest

from research.experiments.boundary_20260908.control.acquire import parse


def test_ordered_same_clock_changes_are_preserved():
    frame = parse(b'00:01:00.000{"Status":"6","Message":"VSCDeployed"}\n'
                  b'00:01:00.000{"Status":"7","Message":"VSCEnding"}\n')
    assert frame.Time.tolist() == [60., 60.]
    assert frame.Status.tolist() == ["6", "7"]
    assert frame.sequence.tolist() == [0, 1]


def test_absent_and_explicit_unknown_remain_distinct_without_fill():
    frame = parse(b'00:00:10.000{"Status":"4"}\n00:00:20.000{"Message":"update"}\n'
                  b'00:00:30.000{"Status":null}\n00:00:40.000{"Status":"3"}\n')
    assert frame.Status.tolist() == ["4", "", "", "3"]
    assert frame.status_present.tolist() == [True, False, True, True]
    assert frame.Message.tolist() == ["", "update", "", ""]


@pytest.mark.parametrize("raw", [b'', b'00:01:00.000[]\n',
                                  b'00:02:00.000{"Status":"1"}\n00:01:00.000{"Status":"2"}\n'])
def test_invalid_or_decreasing_stream_rejected(raw):
    with pytest.raises((ValueError, KeyError)):
        parse(raw)
