"""Reusable sport venue/circuit weather locations."""

from __future__ import annotations

from .open_meteo import WeatherLocation


F1_CIRCUIT_LOCATIONS: dict[str, WeatherLocation] = {
    "bahrain": WeatherLocation("Bahrain International Circuit", 26.0325, 50.5106, "Asia/Bahrain"),
    "jeddah": WeatherLocation("Jeddah Corniche Circuit", 21.6319, 39.1044, "Asia/Riyadh"),
    "albert_park": WeatherLocation("Albert Park", -37.8497, 144.9680, "Australia/Melbourne"),
    "suzuka": WeatherLocation("Suzuka Circuit", 34.8431, 136.5410, "Asia/Tokyo"),
    "shanghai": WeatherLocation("Shanghai International Circuit", 31.3389, 121.2197, "Asia/Shanghai"),
    "miami": WeatherLocation("Miami International Autodrome", 25.9581, -80.2389, "America/New_York"),
    "imola": WeatherLocation("Autodromo Enzo e Dino Ferrari", 44.3439, 11.7167, "Europe/Rome"),
    "monaco": WeatherLocation("Circuit de Monaco", 43.7347, 7.4206, "Europe/Monaco"),
    "villeneuve": WeatherLocation("Circuit Gilles Villeneuve", 45.5000, -73.5228, "America/Toronto"),
    "barcelona": WeatherLocation("Circuit de Barcelona-Catalunya", 41.5700, 2.2611, "Europe/Madrid"),
    "red_bull_ring": WeatherLocation("Red Bull Ring", 47.2197, 14.7647, "Europe/Vienna"),
    "silverstone": WeatherLocation("Silverstone Circuit", 52.0786, -1.0169, "Europe/London"),
    "hungaroring": WeatherLocation("Hungaroring", 47.5789, 19.2486, "Europe/Budapest"),
    "spa": WeatherLocation("Circuit de Spa-Francorchamps", 50.4372, 5.9714, "Europe/Brussels"),
    "zandvoort": WeatherLocation("Circuit Zandvoort", 52.3888, 4.5409, "Europe/Amsterdam"),
    "monza": WeatherLocation("Autodromo Nazionale Monza", 45.6156, 9.2811, "Europe/Rome"),
    "baku": WeatherLocation("Baku City Circuit", 40.3725, 49.8533, "Asia/Baku"),
    "marina_bay": WeatherLocation("Marina Bay Street Circuit", 1.2914, 103.8640, "Asia/Singapore"),
    "cota": WeatherLocation("Circuit of the Americas", 30.1328, -97.6411, "America/Chicago"),
    "mexico_city": WeatherLocation("Autodromo Hermanos Rodriguez", 19.4042, -99.0907, "America/Mexico_City"),
    "interlagos": WeatherLocation("Interlagos", -23.7036, -46.6997, "America/Sao_Paulo"),
    "las_vegas": WeatherLocation("Las Vegas Strip Circuit", 36.1147, -115.1728, "America/Los_Angeles"),
    "losail": WeatherLocation("Lusail International Circuit", 25.4900, 51.4542, "Asia/Qatar"),
    "yas_marina": WeatherLocation("Yas Marina Circuit", 24.4672, 54.6031, "Asia/Dubai"),
    "paul_ricard": WeatherLocation("Circuit Paul Ricard", 43.2506, 5.7917, "Europe/Paris"),
    "portimao": WeatherLocation("Autodromo Internacional do Algarve", 37.2317, -8.6283, "Europe/Lisbon"),
    "istanbul": WeatherLocation("Istanbul Park", 40.9517, 29.4050, "Europe/Istanbul"),
    "sochi": WeatherLocation("Sochi Autodrom", 43.4057, 39.9578, "Europe/Moscow"),
    "nurburgring": WeatherLocation("Nurburgring GP-Strecke", 50.3356, 6.9475, "Europe/Berlin"),
    "mugello": WeatherLocation("Mugello Circuit", 43.9975, 11.3719, "Europe/Rome"),
}
