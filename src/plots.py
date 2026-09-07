"""Figures for the README.

Three charts, each rendered for a light and a dark surface so the README reads
correctly in either GitHub theme:

``calibration``  a reliability diagram -- the evidence that the probabilities
                 mean what they say, which no accuracy number can show;
``tiers``        accuracy by confidence tier, the headline result;
``seasons``      model against the home-court baseline, season by season, which
                 is what shows the result is not carried by one lucky year.

Every value plotted here also appears in a README table, so nothing is readable
only from the picture.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import ROOT, TIER_EDGES  # noqa: E402
from .evaluate import calibration_table, expected_calibration_error, season_breakdown  # noqa: E402

FIGURE_DIR = ROOT / "reports" / "figures"

# Validated against scripts/validate_palette.js: categorical slots 1-2 pass every
# check in both modes, and the three-step blue passes as an ordinal ramp.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
        "ordinal": ["#86b6ef", "#2a78d6", "#104281"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333331",
        "series_1": "#3987e5",
        "series_2": "#d95926",
        "ordinal": ["#184f95", "#3987e5", "#9ec5f4"],
    },
}


def _style(ax, theme, *, xlabel="", ylabel="", title=""):
    """Recessive chrome: hairline solid grid, no top/right spine, roomy padding."""
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(True, color=theme["grid"], linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=theme["muted"], fontsize=9.5, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme["muted"], fontsize=9.5, labelpad=10)
    if title:
        ax.set_title(
            title, color=theme["text"], fontsize=12.5, pad=16, loc="left",
        )


def _save(fig, name, mode):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}-{mode}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #

def plot_calibration(predictions, mode="light", n_bins=10, min_n=30):
    """Reliability diagram, with the bin counts as a companion strip below.

    Two panels rather than two y-scales on one: a second axis would invent a
    relationship between "how often" and "how accurate" that isn't in the data.

    Bins holding fewer than ``min_n`` predictions are left off the curve. The
    outermost bins here hold a single game each, and one game is either 0% or
    100% -- plotted, they read as a dramatic miscalibration at the extremes when
    they are just one coin landing. They stay in the histogram below, and in the
    ECE, where they are weighted by their size and contribute almost nothing.
    """
    theme = THEMES[mode]
    table = calibration_table(predictions, n_bins=n_bins)
    ece = expected_calibration_error(predictions, n_bins=n_bins)
    plotted = table[table["n"] >= min_n]

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(7, 6.4), height_ratios=[3, 1], sharex=True
    )

    ax.plot([0, 1], [0, 1], color=theme["muted"], linewidth=1.2, linestyle="--",
            zorder=1, alpha=0.55)
    ax.annotate(
        "perfect calibration", xy=(0.80, 0.80), xytext=(0.60, 0.90),
        color=theme["muted"], fontsize=9,
        arrowprops=dict(arrowstyle="-", color=theme["muted"], linewidth=0.8,
                        alpha=0.7),
    )
    ax.plot(
        plotted["mean_predicted"], plotted["observed"],
        color=theme["series_1"], linewidth=2, marker="o", markersize=8,
        markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(
        ax, theme, ylabel="Observed win rate",
        title="Predicted probabilities track observed outcomes",
    )
    ax.text(
        0.03, 0.95, f"Expected calibration error  {ece:.3f}",
        transform=ax.transAxes, color=theme["text"], fontsize=11, va="top",
    )
    ax.text(
        0.03, 0.88, f"bins under {min_n} predictions omitted from the curve",
        transform=ax.transAxes, color=theme["muted"], fontsize=8.5, va="top",
    )

    ax_n.bar(
        table["mean_predicted"], table["n"], width=0.085,
        color=theme["series_1"], alpha=0.55, zorder=2,
    )
    _style(ax_n, theme, xlabel="Predicted probability of a win",
           ylabel="Predictions")
    fig.subplots_adjust(hspace=0.12)
    return _save(fig, "calibration", mode)


def plot_tiers(tiers, baselines, mode="light"):
    """Accuracy by confidence tier, against the two baselines that matter."""
    theme = THEMES[mode]
    low, high = TIER_EDGES
    names = ["Toss-up", "Lean", "Confident"]
    subtitles = [f"|p\u22120.5| < {low}", f"{low} \u2013 {high}", f"\u2265 {high}"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = range(len(names))
    bars = ax.bar(
        x, 100 * tiers["accuracy"], width=0.56, color=theme["ordinal"], zorder=3,
    )

    # Reference lines stop short of the label gutter so the text never sits on
    # top of the rule it is naming.
    left, right = -0.62, 3.62
    ax.set_xlim(left, right)
    line_end = (2.55 - left) / (right - left)
    for label, value, color in (
        ("coin flip  50.0%", 50.0, theme["muted"]),
        (f"home court  {100 * baselines['home_court']:.1f}%",
         100 * baselines["home_court"], theme["series_2"]),
    ):
        ax.axhline(value, xmax=line_end, color=color, linewidth=1.2,
                   linestyle="--", alpha=0.8, zorder=2)
        ax.text(2.68, value, label, color=color, fontsize=9.5, va="center",
                ha="left")

    for bar, (_, row) in zip(bars, tiers.iterrows()):
        # The surface-coloured box lets a value label sit over a reference
        # rule without the dashes running through the digits.
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4,
            f"{100 * row['accuracy']:.1f}%",
            ha="center", color=theme["text"], fontsize=13, zorder=5,
            bbox=dict(facecolor=theme["surface"], edgecolor="none", pad=2.0),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [
            f"{n}\n{s}\nn = {int(r):,}"
            for n, s, r in zip(names, subtitles, tiers["n"])
        ],
        color=theme["text"], fontsize=10,
    )
    ax.set_ylim(0, 84)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    _style(ax, theme, ylabel="Accuracy",
           title="The model knows which games it knows")
    return _save(fig, "tiers", mode)


def plot_seasons(seasons, mode="light"):
    """Dumbbell: the gap between the baseline and the model, season by season.

    A grouped bar chart would put fourteen bars on screen to show seven gaps.
    The gap is the story, so the gap is the mark.
    """
    theme = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    y = list(range(len(seasons)))[::-1]

    for yi, (_, row) in zip(y, seasons.iterrows()):
        ax.plot(
            [100 * row["home_court"], 100 * row["accuracy"]], [yi, yi],
            color=theme["grid"], linewidth=2.5, zorder=2, solid_capstyle="round",
        )
        ax.scatter(100 * row["home_court"], yi, s=90, color=theme["series_2"],
                   edgecolor=theme["surface"], linewidth=2, zorder=3)
        ax.scatter(100 * row["accuracy"], yi, s=90, color=theme["series_1"],
                   edgecolor=theme["surface"], linewidth=2, zorder=4)
        ax.text(
            100 * row["accuracy"] + 1.4, yi, f"+{100 * row['lift']:.1f}",
            va="center", color=theme["muted"], fontsize=9.5,
        )

    ax.scatter([], [], s=90, color=theme["series_2"], label="Home court only")
    ax.scatter([], [], s=90, color=theme["series_1"], label="Model")
    legend = ax.legend(
        loc="lower left", frameon=False, fontsize=9.5, ncol=2,
        bbox_to_anchor=(0.0, -0.26),
    )
    for text in legend.get_texts():
        text.set_color(theme["muted"])

    ax.text(
        69.4, max(y) + 0.62, "lift, pts", color=theme["muted"], fontsize=9,
        va="center",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(s)) for s in seasons["season"]],
                       color=theme["text"], fontsize=10.5)
    ax.set_xlim(48, 71.5)
    ax.set_xticks([50, 55, 60, 65, 70])
    ax.set_xticklabels(["50%", "55%", "60%", "65%", "70%"])
    _style(ax, theme, xlabel="Accuracy",
           title="Beats the baseline in all seven held-out seasons")
    ax.grid(axis="y", visible=False)
    return _save(fig, "seasons", mode)


# --------------------------------------------------------------------------- #

def generate_all(predictions, tiers, baselines):
    """Render every figure in both themes. Returns the paths written."""
    seasons = season_breakdown(predictions)
    paths = []
    for mode in THEMES:
        paths.append(plot_calibration(predictions, mode=mode))
        paths.append(plot_tiers(tiers, baselines, mode=mode))
        paths.append(plot_seasons(seasons, mode=mode))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    from .features import build_feature_frame, model_columns
    from .predict import (
        backtest,
        confidence_tiers,
        home_court_baseline,
        load_or_select_features,
        make_model,
        reconcile,
    )

    print("Building features and backtesting...")
    full = build_feature_frame()
    candidates = model_columns(full)
    predictors = load_or_select_features(full, candidates)
    raw_preds = backtest(full, make_model(), predictors)
    reconciled, _ = reconcile(raw_preds)

    paths = generate_all(
        reconciled,
        confidence_tiers(reconciled),
        {"home_court": home_court_baseline(raw_preds)},
    )
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
