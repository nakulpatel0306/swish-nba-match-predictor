# NBA Basketball Match Predictor

Predict winners of NBA games using Python and ML. This repo includes both **Jupyter notebooks** for a step-by-step workflow and **Python scripts** for quick, reproducible runs.

## Project Structure
```
.
├── notebooks/                 # Interactive, documented workflow
│   ├── getData.ipynb          # Scrape/pull box score data
│   ├── parse_data.ipynb       # Clean/engineer features
│   └── predict.ipynb          # Train/evaluate model, make predictions
├── src/                       # Pure Python scripts generated from notebooks
│   ├── get_data.py
│   ├── parse_data.py
│   └── predict.py
├── data/
│   ├── raw/                   # Untracked by git (see .gitignore)
│   └── processed/
│       └── nba_games.csv      # Cleaned dataset (example)
├── requirements.txt
├── .gitignore
└── main.py                    # Orchestrates end-to-end run
```

## Getting Started

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3A) Run the **end-to-end** pipeline (scripts)
```bash
python main.py
```

### 3B) Explore the **notebooks**
```bash
# Launch Jupyter
jupyter notebook
# Then open files in notebooks/ and run cells in order.
```

## Reproducible Workflow
- **Data ingestion**: `src/get_data.py` (from `notebooks/getData.ipynb`)
- **Preprocessing**: `src/parse_data.py` (from `notebooks/parse_data.ipynb`)
- **Modeling**: `src/predict.py` (from `notebooks/predict.ipynb`)

The Python scripts are auto-generated from the notebooks by concatenating code cells in order. You can refine them into functions/classes as you productionize.

## Notes
- Large/secret data should go in `data/raw/` (ignored by git).
- Keep `data/processed/nba_games.csv` small if you plan to store it in the repo, or remove it and document how to regenerate.
- Consider pinning dependency versions for long-term reproducibility.
  
