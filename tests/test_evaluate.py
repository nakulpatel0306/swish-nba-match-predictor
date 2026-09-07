"""Calibration and robustness — the claims a single accuracy number can't support."""

import numpy as np
import pandas as pd

from src.evaluate import (
    calibration_table,
    expected_calibration_error,
    season_breakdown,
)

# A model whose probabilities are honest should sit well under this. Anything
# above it means a stated 70% is not really a 70%.
MAX_CALIBRATION_ERROR = 0.05


def test_calibration_table_partitions_the_predictions(reconciled):
    table = calibration_table(reconciled)
    assert table["n"].sum() == len(reconciled)
    assert (table["n"] > 0).all()


def test_calibration_table_gap_is_predicted_minus_observed():
    df = pd.DataFrame({"prob": [0.9, 0.9, 0.9, 0.9], "actual": [1, 1, 1, 0]})
    table = calibration_table(df, n_bins=10)
    row = table.iloc[0]
    assert row["n"] == 4
    assert np.isclose(row["mean_predicted"], 0.9)
    assert np.isclose(row["observed"], 0.75)
    assert np.isclose(row["gap"], 0.15)


def test_perfect_predictions_have_zero_calibration_error():
    df = pd.DataFrame({"prob": [1.0] * 50 + [0.0] * 50, "actual": [1] * 50 + [0] * 50})
    assert expected_calibration_error(df) == 0.0


def test_model_probabilities_are_calibrated(reconciled):
    ece = expected_calibration_error(reconciled)
    assert ece < MAX_CALIBRATION_ERROR, (
        f"expected calibration error {ece:.4f} is too high to call the "
        "probabilities meaningful"
    )


def test_beats_home_court_in_every_held_out_season(reconciled):
    """Robustness: a 63% average carried by one lucky season is a different claim."""
    seasons = season_breakdown(reconciled)
    assert len(seasons) == 7

    losses = seasons[seasons["lift"] <= 0]
    assert losses.empty, (
        "model failed to beat the home-court baseline in "
        f"{sorted(losses['season'])}"
    )


def test_season_breakdown_covers_every_prediction(reconciled):
    seasons = season_breakdown(reconciled)
    assert seasons["n"].sum() == len(reconciled)
