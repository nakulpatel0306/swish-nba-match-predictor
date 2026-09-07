"""Shared paths and constants for the Swish pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STANDINGS_DIR = RAW_DIR / "standings"
SCORES_DIR = RAW_DIR / "scores"
PROCESSED_DIR = DATA_DIR / "processed"
GAMES_CSV = PROCESSED_DIR / "nba_games.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
SELECTED_FEATURES_JSON = ARTIFACTS_DIR / "selected_features.json"

SEASONS = list(range(2016, 2025))

# Rolling form window, in games.
ROLLING_WINDOW = 10

# Number of features forward sequential selection keeps.
N_FEATURES = 30

# Columns that are labels, identifiers or bookkeeping -- never model inputs.
NON_FEATURE_COLUMNS = ["season", "date", "won", "target", "Team", "Team_opp"]

# Confidence tier cut points on |p - 0.5|.
TIER_EDGES = (0.05, 0.15)

RANDOM_STATE = 42
