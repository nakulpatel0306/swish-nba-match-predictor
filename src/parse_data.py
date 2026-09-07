"""Parse scraped box score HTML into one structured row per team per game.

Each box score yields two rows -- one for each team -- combining basic and
advanced stats, as both team totals and per-game player maxima. The opponent's
stats are joined onto each row as ``_opp`` columns, so a single row already
describes both sides of the game it came from.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .config import GAMES_CSV, SCORES_DIR

# Columns kept from every box score, fixed by the first game parsed so every row
# lines up. Plus/minus columns are dropped: they are empty for most seasons.
DROPPED_STAT_PREFIXES = ("bpm",)


def parse_html(box_score: Path) -> BeautifulSoup:
    """Read one box score and strip the header rows that break ``read_html``."""
    soup = BeautifulSoup(Path(box_score).read_text(), "html.parser")
    for tag in soup.select("tr.over_header"):
        tag.decompose()
    for tag in soup.select("tr.thead"):
        tag.decompose()
    return soup


def read_line_score(soup: BeautifulSoup) -> pd.DataFrame:
    """Extract the two team codes and their final scores."""
    line_score = pd.read_html(str(soup), attrs={"id": "line_score"})[0]
    cols = list(line_score.columns)
    cols[0], cols[-1] = "Team", "Total"
    line_score.columns = cols
    return line_score[["Team", "Total"]]


def read_stats(soup: BeautifulSoup, team: str, stat: str) -> pd.DataFrame:
    """Read one team's ``basic`` or ``advanced`` table as numbers."""
    df = pd.read_html(
        str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0
    )[0]
    return df.apply(pd.to_numeric, errors="coerce")


def read_season_info(soup: BeautifulSoup) -> str:
    """Pull the season the game belongs to out of the page's bottom nav."""
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    return os.path.basename(hrefs[1]).split("_")[0]


def team_summary(soup: BeautifulSoup, team: str, base_cols: list[str] | None):
    """Build one team's stat row: totals plus per-player maxima, basic + advanced."""
    basic = read_stats(soup, team, "basic")
    advanced = read_stats(soup, team, "advanced")

    totals = pd.concat([basic.iloc[-1, :], advanced.iloc[-1, :]])
    totals.index = totals.index.str.lower()

    maxes = pd.concat([basic.iloc[:-1, :].max(), advanced.iloc[:-1, :].max()])
    maxes.index = maxes.index.str.lower() + "_max"

    summary = pd.concat([totals, maxes])

    if base_cols is None:
        base_cols = [
            c
            for c in summary.index.drop_duplicates(keep="first")
            if not c.startswith(DROPPED_STAT_PREFIXES)
        ]
    return summary[base_cols], base_cols


def parse_game(box_score: Path, base_cols: list[str] | None):
    """Turn one box score file into a two-row frame (one row per team)."""
    soup = parse_html(box_score)
    line_score = read_line_score(soup)
    teams = list(line_score["Team"])

    summaries = []
    for team in teams:
        summary, base_cols = team_summary(soup, team, base_cols)
        summaries.append(summary)

    summary = pd.concat(summaries, axis=1).T
    game = pd.concat([summary, line_score], axis=1)
    game["Home"] = [0, 1]

    # Flip the two rows and re-attach them so each team carries its opponent's line.
    game_opp = game.iloc[::-1].reset_index()
    game_opp.columns = [f"{c}_opp" for c in game_opp.columns]

    full_game = pd.concat([game, game_opp], axis=1)
    full_game["season"] = read_season_info(soup)
    full_game["date"] = pd.to_datetime(
        Path(box_score).name[:8], format="%Y%m%d"
    )
    full_game["won"] = full_game["Total"] > full_game["Total_opp"]
    return full_game, base_cols


def parse_all(scores_dir: Path = SCORES_DIR) -> pd.DataFrame:
    """Parse every cached box score into a single team-game frame."""
    box_scores = sorted(Path(scores_dir).glob("*.html"))
    if not box_scores:
        raise FileNotFoundError(
            f"No box scores in {scores_dir}. Run src/get_data.py first."
        )

    base_cols = None
    games = []
    for i, box_score in enumerate(box_scores, start=1):
        full_game, base_cols = parse_game(box_score, base_cols)
        games.append(full_game)
        if i % 100 == 0:
            print(f"{i} / {len(box_scores)}")

    return pd.concat(games, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--out", type=Path, default=GAMES_CSV)
    args = parser.parse_args()

    games = parse_all(args.scores_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.out)
    print(f"Wrote {len(games):,} team-game rows to {args.out}")


if __name__ == "__main__":
    main()
