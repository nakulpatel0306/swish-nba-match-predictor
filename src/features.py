"""Feature engineering for the Swish NBA predictor.

The raw dataset has one row per team per game (so two rows per game) holding that
game's own box score. Those numbers describe a game that has already happened, so
on their own they say very little about the *next* one. This module turns them
into what the model actually trains on:

1. a trailing ``ROLLING_WINDOW``-game mean of every stat, computed within each
   team-season so form never leaks across a season boundary;
2. the same rolling form for the opponent of the upcoming game, merged on;
3. the home/away flag for the upcoming game;
4. the target -- did the team win its *next* game.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

from .config import GAMES_CSV, NON_FEATURE_COLUMNS, ROLLING_WINDOW

# Duplicated / bookkeeping columns carried over from the scrape.
JUNK_COLUMNS = ["mp.1", "mp_opp.1", "index_opp"]


def load_games(path=GAMES_CSV) -> pd.DataFrame:
    """Load the scraped team-game rows, chronologically ordered."""
    df = pd.read_csv(path, index_col=0)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop(columns=[c for c in JUNK_COLUMNS if c in df.columns])
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row with whether that team won its *next* game.

    The last game of a team's history has no next game; those rows are marked ``2``
    so they can be carried through the pipeline and dropped at evaluation time.
    """
    df = df.copy()
    df["target"] = df.groupby("Team")["won"].shift(-1)
    df["target"] = df["target"].fillna(2).astype(int)
    return df


def drop_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with any missing values (a handful of unscraped advanced stats)."""
    null_counts = df.isnull().sum()
    return df[df.columns[null_counts == 0]].copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric per-game columns eligible to become model inputs."""
    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and is_numeric_dtype(df[c])
    ]


def add_rolling_form(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Append a trailing ``window``-game mean of every stat, per team-season.

    Grouping by ``(Team, season)`` -- not by team alone -- is what keeps a window
    from spanning two seasons. Rows before a team has ``window`` games in the
    current season come out null and are dropped by the caller.
    """
    cols = feature_columns(df)
    rolling = (
        df[cols + ["Team", "season"]]
        .groupby(["Team", "season"], group_keys=False)[cols]
        .rolling(window)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .sort_index()
    )
    rolling.columns = [f"{c}_{window}" for c in cols]
    return pd.concat([df, rolling], axis=1)


def add_next_game_context(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the identity, date and venue of each team's upcoming game."""
    df = df.copy()
    grouped = df.groupby("Team")
    df["home_next"] = grouped["Home"].shift(-1)
    df["Team_opp_next"] = grouped["Team_opp"].shift(-1)
    df["date_next"] = grouped["date"].shift(-1)
    return df


def merge_opponent_form(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Join each row against the rolling form of the team it is about to play.

    Row ``(team, date_next)`` is matched to the row where that same upcoming game
    is the opponent's next game -- i.e. ``(Team_opp_next, date_next)`` -- so the
    model sees both sides of the matchup before it is played.
    """
    rolling_cols = [c for c in df.columns if c.endswith(f"_{window}")]
    right = df[rolling_cols + ["Team_opp_next", "date_next", "Team"]]
    return df.merge(
        right,
        left_on=["Team", "date_next"],
        right_on=["Team_opp_next", "date_next"],
    )


def scale_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Min-max scale the given columns onto [0, 1]."""
    df = df.copy()
    df[columns] = MinMaxScaler().fit_transform(df[columns])
    return df


def build_feature_frame(
    path=GAMES_CSV, window: int = ROLLING_WINDOW, scale: bool = True
) -> pd.DataFrame:
    """Run the whole feature pipeline: raw team-game rows -> model-ready frame.

    Scaling happens *before* the rolling means, not after, and this ordering
    matters. A 10-game average varies far less than a single game does, so once
    the per-game stats are on [0, 1] the rolling columns land in a genuinely
    narrower band. Re-scaling them afterwards would stretch that band back out
    and erase the fact that an average is the steadier quantity -- which the
    shared L2 penalty would then read wrongly. Everything downstream is derived
    from these values, so no further scaling is needed.
    """
    df = load_games(path)
    df = add_target(df)
    df = drop_null_columns(df)
    if scale:
        df = scale_features(df, feature_columns(df))
    df = add_rolling_form(df, window=window)
    df = df.dropna().reset_index(drop=True)
    df = add_next_game_context(df)
    df = df.dropna(subset=["home_next", "Team_opp_next", "date_next"]).copy()
    return merge_opponent_form(df, window=window)


def model_columns(full: pd.DataFrame) -> list[str]:
    """Candidate predictors on the merged frame: everything numeric and non-label."""
    removed = set(NON_FEATURE_COLUMNS)
    return [
        c for c in full.columns if c not in removed and is_numeric_dtype(full[c])
    ]
