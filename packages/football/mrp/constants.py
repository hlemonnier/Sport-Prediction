"""Shared constants for football match result prediction."""

MODEL_VERSION = "0.3.0"

DATA_DIR_COMPONENTS = ("data", "football")
SUPPORTED_DATA_EXTENSIONS = ("parquet", "csv")
DATA_TABLES = ("teams", "matches", "fixtures")

TEAM_ID_COLUMNS = ("team_id", "id", "team", "team_code")
TEAM_NAME_COLUMNS = ("team_name", "name", "team")
LEAGUE_COLUMNS = ("league", "competition", "league_code")
SEASON_COLUMNS = ("season", "year")
ROUND_COLUMNS = ("round", "round_number", "matchday", "gameweek", "gw")
DATE_COLUMNS = ("date", "match_date", "kickoff", "kickoff_time", "datetime")
VENUE_COLUMNS = ("venue", "stadium", "ground", "venue_name")
VENUE_LATITUDE_COLUMNS = ("venue_latitude", "stadium_latitude", "lat", "latitude")
VENUE_LONGITUDE_COLUMNS = ("venue_longitude", "stadium_longitude", "lon", "lng", "longitude")
TIMEZONE_COLUMNS = ("timezone", "tz", "venue_timezone")

MATCH_ID_COLUMNS = ("match_id", "id", "fixture_id", "game_id")
HOME_TEAM_COLUMNS = ("home_team_id", "home_id", "home_team", "home")
AWAY_TEAM_COLUMNS = ("away_team_id", "away_id", "away_team", "away")
HOME_GOALS_COLUMNS = ("home_goals", "hg", "ft_home_goals", "goals_home")
AWAY_GOALS_COLUMNS = ("away_goals", "ag", "ft_away_goals", "goals_away")
HOME_XG_COLUMNS = ("home_xg", "xg_home")
AWAY_XG_COLUMNS = ("away_xg", "xg_away")

DEFAULT_HOME_GOALS_PRIOR = 1.35
DEFAULT_AWAY_GOALS_PRIOR = 1.05

DIXON_COLES_MAX_ITER = 260
DIXON_COLES_LEARNING_RATE = 0.05
DIXON_COLES_REGULARIZATION = 0.002
DIXON_COLES_TOLERANCE = 1e-6
DIXON_COLES_RATE_CLAMP_MIN = 0.05
DIXON_COLES_RATE_CLAMP_MAX = 6.0
DIXON_COLES_RHO_MIN = -0.2
DIXON_COLES_RHO_MAX = 0.2
DIXON_COLES_RHO_STEP = 0.005

OUTCOME_GOAL_GRID_MAX = 10
SCORELINE_GOAL_GRID_MAX = 6

HOME_WIN_CLASS = 0
DRAW_CLASS = 1
AWAY_WIN_CLASS = 2
OUTCOME_CLASSES = (HOME_WIN_CLASS, DRAW_CLASS, AWAY_WIN_CLASS)

CALIBRATION_MIN_SAMPLES = 30
CALIBRATION_MIN_CLASS_SAMPLES = 4

RECENT_FORM_WINDOW = 5
BENCHMARK_MIN_SAMPLES = 45
BENCHMARK_MIN_VALIDATION = 12
BENCHMARK_VALIDATION_FRACTION = 0.2

HYBRID_MIN_TRAIN_MATCHES = 45
HYBRID_MIN_TEAM_MATCHES = 3
HYBRID_MIN_VALIDATION_SIZE = 12
HYBRID_WEIGHT_GRID = tuple(step / 10.0 for step in range(11))

BASELINE_PSEUDOCOUNT = 1.0
BASELINE_SEASON_WINDOW = 1
BASELINE_TEAM_WEIGHT = 0.35

DIAGNOSTIC_ECE_BINS = 10
