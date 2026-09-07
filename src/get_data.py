"""Scrape basketball-reference box scores for the configured seasons.

Two-stage crawl: fetch each season's schedule pages first, then every box score
they link to. Pages are cached to disk, so a re-run skips whatever it already
has and a full scrape only has to happen once.

A complete scrape takes several hours because of the rate-limit sleeps. The
processed dataset is committed at ``data/processed/nba_games.csv`` so the rest
of the pipeline can be run without repeating it.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from .config import SCORES_DIR, SEASONS, STANDINGS_DIR

BASE_URL = "https://www.basketball-reference.com"


async def get_html(url: str, selector: str, sleep: int = 5, retries: int = 3):
    """Fetch ``selector``'s inner HTML from ``url``, backing off on timeouts.

    The sleep grows with each attempt, which doubles as the crawl's rate limit --
    basketball-reference throttles aggressively without it.
    """
    html = None
    for attempt in range(1, retries + 1):
        time.sleep(sleep * attempt)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                html = await page.inner_html(selector)
                await browser.close()
        except PlaywrightTimeout:
            print(f"Timeout on {url} (attempt {attempt}/{retries})")
            continue
        else:
            break
    return html


async def scrape_season(season: int, standings_dir: Path = STANDINGS_DIR) -> None:
    """Download every monthly schedule page for one season."""
    standings_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    if not html:
        print(f"Could not load season index for {season}")
        return

    soup = BeautifulSoup(html, "html.parser")
    hrefs = [a["href"] for a in soup.find_all("a")]

    for href in hrefs:
        page_url = f"{BASE_URL}{href}"
        save_path = standings_dir / page_url.split("/")[-1]
        if save_path.exists():
            continue

        page_html = await get_html(page_url, "#all_schedule")
        if not page_html:
            continue
        save_path.write_text(page_html)


async def scrape_games(standings_file: Path, scores_dir: Path = SCORES_DIR) -> None:
    """Download every box score linked from one schedule page."""
    scores_dir.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(standings_file.read_text(), "html.parser")
    hrefs = [a.get("href") for a in soup.find_all("a")]
    box_scores = [h for h in hrefs if h and "boxscore" in h and h.endswith(".html")]

    for href in box_scores:
        url = f"{BASE_URL}{href}"
        save_path = scores_dir / url.split("/")[-1]
        if save_path.exists():
            continue

        html = await get_html(url, "#content")
        if not html:
            continue
        save_path.write_text(html)


async def scrape_all(seasons: list[int] = SEASONS) -> None:
    """Run both crawl stages for every season."""
    for season in seasons:
        print(f"Scraping schedule pages for {season}...")
        await scrape_season(season)

    standings_files = sorted(STANDINGS_DIR.glob("*.html"))
    for i, path in enumerate(standings_files, start=1):
        print(f"Scraping box scores [{i}/{len(standings_files)}] {path.name}")
        await scrape_games(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=SEASONS,
        help="seasons to scrape (default: %(default)s)",
    )
    args = parser.parse_args()
    asyncio.run(scrape_all(args.seasons))


if __name__ == "__main__":
    main()
