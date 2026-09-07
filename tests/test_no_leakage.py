"""Every walk-forward fold must train strictly in the past of what it is scored on.

Random cross-validation on time-ordered data leaks future information into
training and inflates accuracy. These tests pin the chronology down.
"""

import pandas as pd

from src.config import ROLLING_WINDOW
from src.predict import season_splits


def test_training_seasons_precede_test_season(feature_frame):
    seasons = sorted(feature_frame["season"].unique())
    folds = list(season_splits(seasons))

    assert folds, "walk-forward backtest produced no folds"

    for train_seasons, test_season in folds:
        assert max(train_seasons) < test_season


def test_training_dates_precede_test_dates(feature_frame):
    """The stronger check: max training date < min test date, fold by fold."""
    df = feature_frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    seasons = sorted(df["season"].unique())

    for train_seasons, test_season in season_splits(seasons):
        train = df[df["season"].isin(train_seasons)]
        test = df[df["season"] == test_season]
        assert not train.empty and not test.empty
        assert train["date"].max() < test["date"].min(), (
            f"fold testing on {test_season} trains on data at or after its start"
        )


def test_rolling_windows_respect_season_boundaries(feature_frame):
    """A team's form at the start of a season must not carry over from the last one.

    Rolling means are computed within ``(Team, season)``, so a team should never
    have more than ``games_in_season - ROLLING_WINDOW + 1`` usable rows: the first
    ``ROLLING_WINDOW - 1`` games of every season have an incomplete window and are
    dropped. If a window spanned seasons, those rows would have survived.
    """
    from src.features import (
        add_rolling_form,
        add_target,
        drop_null_columns,
        load_games,
    )

    base = drop_null_columns(add_target(load_games()))
    rolled = add_rolling_form(base)

    counts = base.groupby(["Team", "season"]).size()
    survivors = rolled.dropna().groupby(["Team", "season"]).size()

    for key, n_games in counts.items():
        n_survivors = survivors.get(key, 0)
        assert n_survivors == max(0, n_games - ROLLING_WINDOW + 1), (
            f"{key}: {n_survivors} rolling rows from {n_games} games -- "
            "a window appears to span a season boundary"
        )
