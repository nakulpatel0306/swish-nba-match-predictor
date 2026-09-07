"""End-to-end orchestrator for the Swish pipeline.

By default this runs only the modeling stage, since ``data/processed/nba_games.csv``
is committed and a full re-scrape takes several hours. Pass ``--scrape`` to
rebuild the dataset from basketball-reference first.
"""

import argparse

from src import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="re-scrape and re-parse box scores before modeling (several hours)",
    )
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
    args = parser.parse_args()

    if args.scrape:
        # Imported here, not at module scope: the modeling path should not need
        # Playwright or BeautifulSoup installed just to run a backtest.
        import asyncio

        from src import get_data, parse_data

        print("[1/3] Scraping box scores...")
        asyncio.run(get_data.scrape_all())

        print("[2/3] Parsing box scores...")
        games = parse_data.parse_all()
        parse_data.GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(parse_data.GAMES_CSV)
    else:
        print("Using committed data/processed/nba_games.csv (pass --scrape to rebuild)")

    print("[3/3] Training and backtesting...")
    predict.run(
        refresh_features=args.refresh_features,
        skip_baselines=args.skip_baselines,
    )


if __name__ == "__main__":
    main()
