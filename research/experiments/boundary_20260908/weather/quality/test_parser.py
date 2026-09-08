from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / 'acquire.py'
spec = importlib.util.spec_from_file_location('acquisition_under_review', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def line(clock, values): return (clock + json.dumps(values) + '\r\n').encode()


@pytest.mark.parametrize('value,expected', [('0', 0.), (0, 0.), (False, 0.), ('1', 1.), (1, 1.), (True, 1.)])
def test_rain_decoding_preserves_explicit_binary_values(value, expected):
    assert module.parse(line('00:01:02.345', {'Rainfall': value})).Rainfall.iloc[0] == expected


def test_missing_invalid_and_partial_fields_do_not_become_observed_zero():
    frame = module.parse(line('00:00:01.000', {'AirTemp': '25.0', 'Rainfall': None})
                         + line('00:00:02.000', {'WindSpeed': '1.5', 'Rainfall': 'not_a_binary_reading'}))
    assert frame.AirTemp.iloc[0] == 25 and np.isnan(frame.AirTemp.iloc[1])
    assert frame.Rainfall.isna().all() and frame.Humidity.isna().all()


def test_nonfinite_numeric_values_remain_missing():
    frame = module.parse(line('00:00:01.000', {'AirTemp': 'NaN', 'Humidity': 'Infinity', 'Pressure': None}))
    assert frame[['AirTemp', 'Humidity', 'Pressure']].isna().all().all()


def test_BOM_and_stream_origin_are_preserved_exactly():
    frame = module.parse(b'\xef\xbb\xbf' + line('01:02:03.456', {'Rainfall': '0'}))
    assert frame.Time.iloc[0] == 3723.456


def test_duplicate_or_reversed_timestamps_fail_explicitly():
    for second in ['00:00:02.000', '00:00:01.000']:
        with pytest.raises(ValueError, match='chronologically unique'):
            module.parse(line('00:00:02.000', {'Rainfall': '0'}) + line(second, {'Rainfall': '1'}))
