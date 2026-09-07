"""Reconciliation must make the two views of a game agree, by construction."""

import numpy as np
import pandas as pd
import pytest

from src.predict import contradiction_rate, game_key, reconcile


def _fake_predictions(probs_a, probs_b):
    """Two rows per game: team A's view and team B's view of the same matchup."""
    rows = []
    for i, (pa, pb) in enumerate(zip(probs_a, probs_b)):
        date = f"2024-01-{i + 1:02d}"
        rows.append(("AAA", "BBB", date, pa, 1))
        rows.append(("BBB", "AAA", date, pb, 0))
    df = pd.DataFrame(
        rows, columns=["Team_x", "Team_opp_next_x", "date_next", "prob", "actual"]
    )
    df["prediction"] = (df["prob"] > 0.5).astype(int)
    return df


def test_game_key_is_symmetric():
    a = pd.Series(["LAL", "BOS"])
    b = pd.Series(["BOS", "LAL"])
    d = pd.Series(["2024-01-01", "2024-01-01"])
    keys = game_key(a, b, d)
    assert keys.iloc[0] == keys.iloc[1]


def test_paired_probabilities_sum_to_one():
    preds = _fake_predictions([0.7, 0.2, 0.55], [0.6, 0.9, 0.55])
    reconciled, unpaired = reconcile(preds)

    assert unpaired == 0
    sums = reconciled.groupby("game_key")["prob"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_reconciliation_eliminates_contradictions():
    # Both rows claim a win, then both claim a loss: two contradictory games.
    preds = _fake_predictions([0.7, 0.3], [0.7, 0.3])
    assert contradiction_rate(preds) == 1.0

    reconciled, _ = reconcile(preds)
    assert contradiction_rate(reconciled) == 0.0


def test_unpaired_rows_are_counted_not_silently_dropped():
    preds = _fake_predictions([0.7], [0.6])
    orphan = preds.iloc[[0]].copy()
    orphan["date_next"] = "2024-02-02"
    preds = pd.concat([preds, orphan], ignore_index=True)

    reconciled, unpaired = reconcile(preds)
    assert unpaired == 1
    assert len(reconciled) == 2


def test_reconciliation_requires_probabilities():
    preds = _fake_predictions([0.7], [0.6]).drop(columns=["prob"])
    with pytest.raises(ValueError):
        reconcile(preds)


def test_real_predictions_are_reconciled_and_contradiction_free(reconciled):
    sums = reconciled.groupby("game_key")["prob"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)
    assert contradiction_rate(reconciled) == 0.0
