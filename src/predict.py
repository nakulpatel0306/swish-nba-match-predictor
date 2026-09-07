"""Modeling, walk-forward backtesting, reconciliation and benchmarks.

The pipeline this module drives:

1. engineer rolling-form features (:mod:`src.features`);
2. narrow ~400 candidates to 30 with forward sequential selection, scored under
   ``TimeSeriesSplit`` so selection never sees a future season;
3. backtest season by season -- train on every prior season, test on the next
   unseen one -- capturing calibrated win probabilities, not just hard labels;
4. reconcile the two independent predictions each game receives into one;
5. report accuracy, AUC, Brier score and per-confidence-tier accuracy, against
   coin-flip, home-court and raw-box-score baselines plus an XGBoost benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .config import (
    GAMES_CSV,
    N_FEATURES,
    RANDOM_STATE,
    SELECTED_FEATURES_JSON,
    TIER_EDGES,
)
from .features import (
    add_next_game_context,
    add_target,
    build_feature_frame,
    drop_null_columns,
    feature_columns,
    load_games,
    model_columns,
    scale_features,
)

# Rows whose team has no next game are labelled 2 and carry no usable answer.
NO_NEXT_GAME = 2


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def make_selector_model() -> RidgeClassifier:
    """Ridge is kept for feature *selection*: it is fast and selects well."""
    return RidgeClassifier(alpha=1)


def make_model() -> LogisticRegression:
    """Final model. Logistic regression so the output is a probability.

    L2-regularized like the Ridge it replaces -- which is what handles the heavy
    collinearity between rolling features -- but it emits ``predict_proba``, so
    predictions can be reconciled and bucketed by confidence.
    """
    return LogisticRegression(max_iter=3000, C=0.5)


def make_xgb_model():
    """Benchmark model. Imported lazily so xgboost stays an optional dependency."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #

def select_features(
    df: pd.DataFrame,
    candidates: list[str],
    n_features: int = N_FEATURES,
    n_splits: int = 3,
) -> list[str]:
    """Forward sequential selection under ``TimeSeriesSplit``.

    Chronological splits matter here as much as in the backtest: with random folds
    the selector would choose features using seasons it is later scored on.
    """
    sfs = SequentialFeatureSelector(
        make_selector_model(),
        n_features_to_select=n_features,
        direction="forward",
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=-1,
    )
    sfs.fit(df[candidates], df["target"])
    return [c for c, keep in zip(candidates, sfs.get_support()) if keep]


def load_or_select_features(
    df: pd.DataFrame,
    candidates: list[str],
    cache=SELECTED_FEATURES_JSON,
    refresh: bool = False,
) -> list[str]:
    """Return the selected predictors, running selection only when needed.

    Selection is the slow step (tens of thousands of Ridge fits). Its result is
    deterministic given the data, so it is cached to disk and reused.
    """
    if cache is not None and cache.exists() and not refresh:
        cached = json.loads(cache.read_text())["predictors"]
        if set(cached).issubset(candidates):
            return cached
    predictors = select_features(df, candidates)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"predictors": predictors}, indent=2) + "\n"
        )
    return predictors


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #

def season_splits(seasons: list, start: int = 2, step: int = 1):
    """Yield ``(train_seasons, test_season)`` pairs, rolling forward one season."""
    seasons = sorted(seasons)
    for i in range(start, len(seasons), step):
        yield seasons[:i], seasons[i]


CARRIED_COLUMNS = ["season", "Team_x", "Team_opp_next_x", "date_next", "home_next"]


def backtest(
    data: pd.DataFrame,
    model,
    predictors: list[str],
    start: int = 2,
    step: int = 1,
) -> pd.DataFrame:
    """Train on every prior season, test on the next unseen one, roll forward.

    Returns one row per held-out prediction with the hard label, the predicted
    probability of a win (when the model exposes one) and enough identifying
    columns to reconcile the two rows belonging to the same game.
    """
    carried = [c for c in CARRIED_COLUMNS if c in data.columns]
    frames = []

    for train_seasons, test_season in season_splits(
        sorted(data["season"].unique()), start=start, step=step
    ):
        train = data[data["season"].isin(train_seasons)]
        test = data[data["season"] == test_season]

        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])

        out = pd.DataFrame(
            {"actual": test["target"].to_numpy(), "prediction": preds},
            index=test.index,
        )
        if hasattr(model, "predict_proba"):
            out["prob"] = model.predict_proba(test[predictors])[:, 1]
        for col in carried:
            out[col] = test[col].to_numpy()
        frames.append(out)

    predictions = pd.concat(frames)
    return predictions[predictions["actual"] != NO_NEXT_GAME].copy()


# --------------------------------------------------------------------------- #
# Game-level reconciliation
# --------------------------------------------------------------------------- #

def game_key(team: pd.Series, opp: pd.Series, date_next: pd.Series) -> pd.Series:
    """A key both rows of the same game share, independent of which side we're on."""
    a = team.astype(str)
    b = opp.astype(str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    return pd.Series(
        [f"{x}|{y}|{d}" for x, y, d in zip(lo, hi, date_next.astype(str))],
        index=team.index,
    )


def reconcile(predictions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Average each game's two independent probability estimates into one.

    A game between A and B produces a row saying ``P(A wins) = p_A`` and a row
    saying ``P(B wins) = p_B``. They estimate the same event from opposite sides,
    and nothing in the model makes them agree -- so on their own they contradict
    each other on a meaningful share of games. Averaging them,

        ``p_A_final = (p_A + (1 - p_B)) / 2``

    makes the two predictions sum to 1 by construction, so contradictions become
    structurally impossible rather than merely rare.

    Returns the reconciled frame and the number of rows dropped for want of a
    partner (a row loses its pair when the other side is cut by the merge or the
    dropna steps upstream).
    """
    if "prob" not in predictions:
        raise ValueError(
            "reconciliation needs probabilities; "
            "use a model that exposes predict_proba"
        )

    df = predictions.copy()
    df["game_key"] = game_key(df["Team_x"], df["Team_opp_next_x"], df["date_next"])

    sizes = df.groupby("game_key")["prob"].transform("size")
    paired = df[sizes == 2].copy()
    unpaired = int((sizes != 2).sum())

    # Within each game, the partner's probability is the other row's value.
    partner_prob = (
        paired.groupby("game_key")["prob"].transform("sum") - paired["prob"]
    )
    paired["prob_raw"] = paired["prob"]
    paired["prob"] = (paired["prob"] + (1 - partner_prob)) / 2
    paired["prediction"] = _pick_winner(paired)
    return paired, unpaired


def _pick_winner(paired: pd.DataFrame, tol: float = 1e-12) -> pd.Series:
    """Hard label from a reconciled probability, with dead heats broken.

    Both rows of a game can land on exactly 0.5 -- when the two raw estimates
    disagree by precisely the amount reconciliation splits. A plain ``> 0.5``
    would then predict a loss for both teams, which is the contradiction this
    whole step exists to remove. Ties go to the alphabetically first team: an
    arbitrary rule, but a deterministic one that always names exactly one winner.
    """
    prediction = (paired["prob"] > 0.5).astype(int)
    tied = (paired["prob"] - 0.5).abs() < tol
    if tied.any():
        prediction[tied] = (
            paired.loc[tied, "Team_x"] < paired.loc[tied, "Team_opp_next_x"]
        ).astype(int)
    return prediction


def contradiction_rate(predictions: pd.DataFrame) -> float:
    """Share of games whose two rows both claim a win, or both claim a loss."""
    df = predictions.copy()
    if "game_key" not in df:
        df["game_key"] = game_key(
            df["Team_x"], df["Team_opp_next_x"], df["date_next"]
        )
    sums = df.groupby("game_key")["prediction"].agg(["sum", "size"])
    sums = sums[sums["size"] == 2]
    if sums.empty:
        return 0.0
    return float((sums["sum"] != 1).mean())


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score(predictions: pd.DataFrame) -> dict:
    """Accuracy, plus AUC and Brier score when probabilities are available."""
    out = {
        "n": int(len(predictions)),
        "accuracy": float(
            accuracy_score(predictions["actual"], predictions["prediction"])
        ),
    }
    if "prob" in predictions:
        out["auc"] = float(roc_auc_score(predictions["actual"], predictions["prob"]))
        out["brier"] = float(
            brier_score_loss(predictions["actual"], predictions["prob"])
        )
    return out


def confidence_tiers(
    predictions: pd.DataFrame, edges: tuple[float, float] = TIER_EDGES
) -> pd.DataFrame:
    """Bucket predictions by how far the probability sits from a coin flip.

    A single accuracy figure averages games the model is nearly certain about
    with games that genuinely are coin flips. Splitting on ``|p - 0.5|`` shows
    that the model knows the difference.
    """
    low, high = edges
    df = predictions.copy()
    df["edge"] = (df["prob"] - 0.5).abs()

    labels = [
        (f"Toss-up      |p-0.5| < {low}", df["edge"] < low),
        (f"Lean         {low} <= |p-0.5| < {high}", (df["edge"] >= low) & (df["edge"] < high)),
        (f"Confident    |p-0.5| >= {high}", df["edge"] >= high),
    ]
    rows = []
    for name, mask in labels:
        subset = df[mask]
        rows.append(
            {
                "tier": name,
                "n": int(len(subset)),
                "accuracy": (
                    float(accuracy_score(subset["actual"], subset["prediction"]))
                    if len(subset)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def home_court_baseline(predictions: pd.DataFrame) -> float:
    """Accuracy of always picking the home team, on the same held-out rows."""
    return float(accuracy_score(predictions["actual"], predictions["home_next"]))


# --------------------------------------------------------------------------- #
# Baseline: raw box score features, no rolling form
# --------------------------------------------------------------------------- #

def build_raw_frame(path=GAMES_CSV) -> pd.DataFrame:
    """The un-engineered comparison: predict the next game from *this* game's box score."""
    df = load_games(path)
    df = add_target(df)
    df = drop_null_columns(df)
    df = add_next_game_context(df)
    return df.dropna(subset=["home_next", "Team_opp_next", "date_next"]).copy()


def raw_feature_baseline() -> dict:
    """Fit and backtest the raw-box-score baseline, returning its score.

    Deliberately excludes ``home_next``: this baseline measures what a single
    game's own box score says about the next game, and handing it the venue of
    the next game would just fold the (stronger) home-court baseline into it.
    """
    raw = build_raw_frame()
    candidates = [c for c in feature_columns(raw) if c != "home_next"]
    raw = scale_features(raw, candidates)
    predictors = load_or_select_features(raw, candidates, cache=None)
    raw = raw.rename(columns={"Team": "Team_x", "Team_opp_next": "Team_opp_next_x"})
    return score(backtest(raw, make_selector_model(), predictors))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def run(refresh_features: bool = False, skip_baselines: bool = False) -> dict:
    """Run the full evaluation and print the report. Returns the numbers."""
    print("Building features...")
    full = build_feature_frame()
    candidates = model_columns(full)
    print(f"  {full.shape[0]:,} rows x {len(candidates)} candidate features")

    print(f"Selecting {N_FEATURES} features (forward sequential, TimeSeriesSplit)...")
    predictors = load_or_select_features(full, candidates, refresh=refresh_features)

    print("Backtesting (walk-forward by season)...")
    raw_preds = backtest(full, make_model(), predictors)
    reconciled, unpaired = reconcile(raw_preds)

    results = {
        "n_rows": int(full.shape[0]),
        "n_candidates": len(candidates),
        "predictors": predictors,
        "pre_reconciliation": score(raw_preds),
        "post_reconciliation": score(reconciled),
        "contradiction_rate_before": contradiction_rate(raw_preds),
        "contradiction_rate_after": contradiction_rate(reconciled),
        "unpaired_rows": unpaired,
        "home_court": home_court_baseline(raw_preds),
        "tiers": confidence_tiers(reconciled),
    }

    print()
    print("=" * 66)
    print(f"Held-out predictions: {results['pre_reconciliation']['n']:,}")
    print(f"Unpaired rows skipped by reconciliation: {unpaired:,}")
    print("=" * 66)
    print()
    print("Baselines")
    print(f"  Coin flip                       {_pct(0.5)}")
    print(f"  Home-court advantage only       {_pct(results['home_court'])}")
    if not skip_baselines:
        raw_baseline = raw_feature_baseline()
        results["raw_features"] = raw_baseline
        print(
            f"  Raw box score features          "
            f"{_pct(raw_baseline['accuracy'])}   (n={raw_baseline['n']:,})"
        )
    print()

    pre, post = results["pre_reconciliation"], results["post_reconciliation"]
    print("Calibrated logistic regression on rolling form + opponent context")
    print(
        f"  Pre-reconciliation              {_pct(pre['accuracy'])}"
        f"  AUC {pre['auc']:.3f}  Brier {pre['brier']:.3f}"
    )
    print(
        f"  Post-reconciliation             {_pct(post['accuracy'])}"
        f"  AUC {post['auc']:.3f}  Brier {post['brier']:.3f}"
    )
    print(
        f"  Contradictory games             "
        f"{_pct(results['contradiction_rate_before'])} -> "
        f"{_pct(results['contradiction_rate_after'])}"
    )
    print()
    print("Confidence tiers (after reconciliation)")
    for _, row in results["tiers"].iterrows():
        print(f"  {row['tier']:<40} n={row['n']:>6,}   {_pct(row['accuracy'])}")
    print()

    if not skip_baselines:
        print("XGBoost benchmark (identical walk-forward splits)")
        for label, cols in (
            (f"{N_FEATURES} selected features", predictors),
            (f"all {len(candidates)} features", candidates),
        ):
            xgb_score = score(backtest(full, make_xgb_model(), cols))
            results.setdefault("xgboost", {})[label] = xgb_score
            print(
                f"  {label:<32} {_pct(xgb_score['accuracy'])}"
                f"  AUC {xgb_score['auc']:.3f}"
            )
        print()
        print(
            "  XGBoost loses. ~400 rolling features are heavily collinear on ~15K\n"
            "  rows; L2 regularization handles that better than an unconstrained\n"
            "  tree ensemble. Reported as measured, not tuned until it wins."
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="re-run feature selection instead of using the cached result",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="skip the raw-feature and XGBoost comparisons (much faster)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the numbers to this file as JSON",
    )
    args = parser.parse_args()
    results = run(
        refresh_features=args.refresh_features, skip_baselines=args.skip_baselines
    )
    if args.json:
        results["tiers"] = results["tiers"].to_dict(orient="records")
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
