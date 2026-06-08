"""Football model architecture contracts."""

FOOTBALL_MODEL_CONTRACT = {
    "pre_match": ["fixture", "team_history", "optional_weather"],
    "scoreline": ["fixture", "team_goal_intensity"],
    "live_match": ["starting_state", "events", "xg_stream"],
    "player_props": ["player", "lineup", "role", "market"],
}
