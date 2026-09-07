# 🏀 Swish — NBA Game Outcome Predictor

Predicts whether an NBA team wins their **next** game, using nine seasons of scraped box score data and a leak-free, walk-forward validated ML pipeline.

Built end-to-end in Python. Ships as both documented Jupyter notebooks (step-by-step workflow) and importable scripts (reproducible runs).

---

## 📊 Results

Evaluated on **11,371 held-out predictions** across seven seasons (2018–2024), using season-by-season walk-forward backtesting.

| Model | Accuracy | AUC | Brier |
|---|---|---|---|
| Coin flip (target is 50/50 balanced) | 50.0% | — | — |
| Raw box score features | 54.2% | — | — |
| Home-court advantage only | 56.8% | — | — |
| XGBoost, 30 selected features | 61.4% | 0.656 | — |
| XGBoost, all 397 features | 62.4% | 0.668 | — |
| Rolling form + opponent context | 62.6% | 0.676 | 0.227 |
| **↳ after game-level reconciliation** | **63.2%** | **0.678** | **0.226** |

Two things carry this project.

**Feature engineering is worth +9 points.** Raw single-game box score stats are barely predictive of the *next* game (54.2%). A team's recent form and their upcoming opponent's form are what actually carry signal.

**The model beats the home-court baseline by 6.4 points**, which is the bar that matters. Any model that can't clear "just pick the home team" isn't learning anything useful.

### 🎯 Confidence tiers — the most interesting result

A single accuracy figure averages games the model is nearly certain about with games that genuinely are coin flips. Bucketing by how far the predicted probability sits from 0.5 shows the model knows the difference:

| Tier | Definition | n | Accuracy |
|---|---|---|---|
| Toss-up | \|p − 0.5\| < 0.05 | 2,930 | 54.0% |
| Lean | 0.05 ≤ \|p − 0.5\| < 0.15 | 4,856 | 61.4% |
| **Confident** | \|p − 0.5\| ≥ 0.15 | 3,572 | **73.0%** |

The model is not uniformly 63% accurate. On the third of games it is most sure about, it is right nearly three times in four — and it tells you *which* third in advance.

---

## 🧠 How It Works

### 1. Data ingestion
Async scraper built on **Playwright** (headless Chromium) and **BeautifulSoup**, pulling every box score from basketball-reference for the 2016–2024 seasons.

- Retry logic with exponential backoff on timeouts
- Local HTML caching so re-runs skip already-scraped pages
- Two-stage crawl — season schedule pages first, then individual box scores

### 2. Feature extraction
Parses each box score into a team-game row combining **basic and advanced stats**, both team totals and per-game player maxima (`fg%`, `ts%`, `ortg`, `drtg`, `usg%`, `tov%`, and more).

Each game produces two rows, one per team, with the opponent's stats joined on as `_opp` columns.

**Output:** 17,534 team-game rows × 154 raw columns.

### 3. Feature engineering
This is where the accuracy comes from.

- **Rolling 10-game form** — every stat re-expressed as a trailing 10-game mean, computed *within* each team-season so form never leaks across seasons
- **Opponent-matchup merge** — each row is joined against the rolling form of the team it is about to play, so the model sees both sides of the upcoming matchup
- **Home/away flag** for the next game
- **Target** — the team's win/loss in their *next* game, shifted per team in chronological order

**Output:** 14,818 rows × 397 candidate features.

### 4. Feature selection
**Forward sequential feature selection** narrows 397 candidates down to 30, scored under `TimeSeriesSplit` so selection never peeks at future seasons.

`RidgeClassifier` drives the selection — it is fast and selects well. The result is cached in `artifacts/selected_features.json`, since selection is deterministic given the data and is by far the slowest step.

### 5. Modeling and validation
`LogisticRegression(max_iter=3000, C=0.5)` — L2-regularized, chosen for stability on a wide, highly collinear feature space (rolling stats are heavily correlated with each other), and because it emits **probabilities** rather than hard labels.

Validation is **walk-forward by season** — train on every prior season, test on the next unseen one, roll forward. No random shuffling, no future data in training. This is the correct protocol for time-ordered sports data and the main reason the number is trustworthy.

### 6. Game-level reconciliation
Every game appears in the data twice — once from each team's point of view — and the model predicts the two rows independently. Left alone, it contradicts itself: on **11.6% of games it predicted that both teams win, or that both lose.**

For a game between A and B, one row estimates `p_A = P(A wins)` and the other estimates `p_B = P(B wins)`. These are two estimates of the same event, from opposite sides. Averaging them:

```python
p_A_final = (p_A + (1 - p_B)) / 2
p_B_final = 1 - p_A_final
```

The two predictions now sum to 1 by construction, so contradictions are not merely rare — they are **structurally impossible**, and a test asserts it. Reconciliation is also worth about half a point of accuracy on its own, and it improves calibration.

Rows are grouped by a symmetric key (the two team names sorted, plus the game date). A handful of rows lose their partner in the merge and dropna steps upstream; those are skipped and **counted** (13 of 11,371), not dropped silently.

---

## 📂 Project Structure

```
.
├── notebooks/                 # Interactive, documented workflow
│   ├── getData.ipynb          # Async scrape of box scores
│   ├── parse_data.ipynb       # Parse HTML into structured features
│   └── predict.ipynb          # Engineer features, select, train, backtest
├── src/                       # Importable modules
│   ├── config.py              # Paths and constants
│   ├── features.py            # Feature engineering
│   ├── get_data.py            # Scraper
│   ├── parse_data.py          # HTML → structured rows
│   └── predict.py             # Model, backtest, reconciliation, benchmarks
├── tests/                     # Leakage, reconciliation and quality tests
├── data/
│   ├── raw/                   # Scraped HTML (gitignored)
│   └── processed/
│       └── nba_games.csv      # Cleaned dataset
├── artifacts/
│   └── selected_features.json # Cached feature selection result
├── requirements.txt
├── .gitignore
└── main.py                    # End-to-end orchestrator
```

---

## 🚀 Getting Started

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
playwright install chromium   # required by the scraper only
```

**3. Run or explore**

Modeling pipeline (uses the committed dataset):
```bash
python main.py
```

Faster — skips the raw-feature and XGBoost comparisons:
```bash
python main.py --skip-baselines
```

Re-scrape from scratch (several hours):
```bash
python main.py --scrape
```

Notebooks:
```bash
jupyter notebook
# open notebooks/ and run cells in order
```

Tests:
```bash
pytest
```

> **Note:** a full scrape takes several hours due to rate-limit sleeps. `data/processed/nba_games.csv` is committed so you can skip straight to `notebooks/predict.ipynb`.

---

## 🛠️ Tech Stack

**Python** · **pandas** · **scikit-learn** · **XGBoost** · **Playwright** · **BeautifulSoup** · **NumPy** · **Jupyter**

---

## 🔍 Design Notes

**Why a linear model over a boosted tree?** Because it was benchmarked, not assumed. XGBoost (`n_estimators=400, max_depth=4, learning_rate=0.05`) was run on identical walk-forward splits and **lost on both feature sets** — 61.4% on the 30 selected features and 62.4% on all 397, against 63.2% for the reconciled logistic regression.

The reason is the shape of the data. Roughly 400 rolling features are heavily collinear (a 10-game mean of `fg` and a 10-game mean of `fga` move together almost perfectly) across only ~15K rows. L2 regularization shrinks correlated coefficients together and degrades gracefully; an unconstrained tree ensemble keeps splitting on near-duplicate features and overfits at that feature-to-sample ratio. This result is reported as measured — XGBoost was not tuned until it won.

**Why logistic regression rather than ridge?** `RidgeClassifier` emits hard labels only, so there is no way to express confidence, no way to check calibration, and no probability to reconcile the two views of a game with. Logistic regression is the same kind of L2-regularized linear model with a probability attached. Ridge is still used for the feature *selection* step, where speed matters and probabilities do not.

**Why walk-forward instead of k-fold?** Random cross-validation on time-ordered data leaks future information into training and inflates accuracy. Every split here respects chronology, and a test asserts that every fold's maximum training date is strictly before its minimum test date.

**Why predict the *next* game?** Predicting the current game from its own box score is trivially easy and useless — the final score is right there in the features. Forecasting forward from prior form is the actual problem.

**Rejected: schedule features.** Rest days, back-to-back flags, and trailing-7-day game density were all built and evaluated. Combined lift was **+0.05 percentage points** — noise. They are deliberately not in the model. Shipping a feature that does nothing is worse than not shipping it, and a negative result is still a result.

**A note on the baselines' row counts.** The headline numbers are over 11,371 held-out predictions. The raw-box-score baseline is over 13,450: it needs no 10-game warm-up, so it keeps the first nine games of each team's season that the rolling pipeline necessarily drops.

---

## 🧪 Tests

`pytest` covers the four things most likely to be quietly wrong:

1. **No temporal leakage** — every walk-forward fold's maximum training date is strictly before its minimum test date.
2. **Rolling features respect season boundaries** — no window spans two seasons for the same team.
3. **Reconciliation is symmetric** — paired probabilities sum to 1.0, and the contradiction rate is exactly zero.
4. **Beats the home-court baseline** — overall accuracy > 56.8%.

---

## 🗺️ Roadmap

- [x] Benchmark gradient-boosted trees (XGBoost) against the linear baseline
- [x] Output calibrated win probabilities instead of hard labels
- [x] Reconcile the two per-game predictions into one
- [ ] Benchmark against the betting market — convert historical closing moneylines to vig-free implied probabilities, compare calibration, and compute flat-bet ROI restricted to the confident tier
- [ ] Track injury and lineup availability
- [ ] Expose predictions through a lightweight API and dashboard
