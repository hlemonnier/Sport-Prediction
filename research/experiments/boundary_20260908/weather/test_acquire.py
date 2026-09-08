import numpy as np
import pytest

from research.experiments.boundary_20260908.weather.acquire import parse, slug


def test_missing_weather_is_not_fabricated_as_zero_or_dry():
    raw = b'00:00:01.000{"AirTemp":"21.4","Rainfall":"1"}\r\n00:01:01.000{"TrackTemp":"bad","Rainfall":"0"}\r\n00:02:01.000{"Humidity":"inf"}'
    data = parse(raw)
    assert data.Time.tolist() == [1.,61.,121.]
    assert data.Rainfall.iloc[:2].tolist() == [1.,0.]
    assert np.isnan(data.Rainfall.iloc[2])
    assert data.TrackTemp.isna().all()
    assert data.Humidity.isna().all()


def test_bad_weather_chronology_is_rejected():
    with pytest.raises(ValueError,match="chronologically"):
        parse(b'00:01:01.000{}\n00:00:01.000{}')
    with pytest.raises(ValueError,match="chronologically"):
        parse(b'00:00:01.000{}\n00:00:01.000{}')
    assert slug("São Paulo Grand Prix") == "sao_paulo_grand_prix"
