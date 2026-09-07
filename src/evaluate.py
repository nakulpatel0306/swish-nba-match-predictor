"""Evaluation beyond a single accuracy number.

The project's headline claim is that its probabilities mean something -- that a
prediction of 0.75 wins about three quarters of the time. Accuracy cannot show
that, and Brier score only summarises it. These are the checks that actually
demonstrate it: a reliability table, expected calibration error, and a
season-by-season breakdown that shows the result is not carried by one lucky year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from .config import TIER_EDGES


def calibration_table(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions by predicted probability and compare to what happened.

    For a well-calibrated model, ``mean_predicted`` and ``observed`` track each
    other down every row: of the games it called at 0.70, roughly 70% are wins.
    ``gap`` is the signed error -- positive means the model was overconfident in
    that bucket.
    """
    df = predictions.copy()
    edges = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["prob"], bins=edges, include_lowest=True)

    grouped = df.groupby("bin", observed=True)
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean_predicted": grouped["prob"].mean(),
            "observed": grouped["actual"].mean(),
        }
    ).reset_index()
    table["gap"] = table["mean_predicted"] - table["observed"]
    return table


def expected_calibration_error(predictions: pd.DataFrame, n_bins: int = 10) -> float:
    """Mean |predicted - observed| across bins, weighted by bin size.

    One number for "how far off are the probabilities". 0.0 is perfect; anything
    under ~0.02 on this much data is a well-behaved forecaster.
    """
    table = calibration_table(predictions, n_bins=n_bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def season_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-season accuracy, AUC, Brier and home-court baseline.

    A model that averages 63% by going 70% in one season and 56% in the rest is
    a different, worse thing than one that is steady. This says which it is.
    """
    rows = []
    for season, group in predictions.groupby("season"):
        row = {
            "season": int(season),
            "n": int(len(group)),
            "accuracy": float(accuracy_score(group["actual"], group["prediction"])),
        }
        if "home_next" in group:
            row["home_court"] = float(
                accuracy_score(group["actual"], group["home_next"])
            )
            row["lift"] = row["accuracy"] - row["home_court"]
        if "prob" in group:
            row["auc"] = float(roc_auc_score(group["actual"], group["prob"]))
            row["brier"] = float(brier_score_loss(group["actual"], group["prob"]))
        rows.append(row)
    return pd.DataFrame(rows)


def tier_edges_label(edges: tuple[float, float] = TIER_EDGES) -> list[str]:
    """Short tier names, for plots and tables that cannot fit the definitions."""
    return ["Toss-up", "Lean", "Confident"]


def summarise(predictions: pd.DataFrame, n_bins: int = 10) -> dict:
    """Everything in this module at once, for the report and the plots."""
    return {
        "calibration": calibration_table(predictions, n_bins=n_bins),
        "ece": expected_calibration_error(predictions, n_bins=n_bins),
        "seasons": season_breakdown(predictions),
    }
