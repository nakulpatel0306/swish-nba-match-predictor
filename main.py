"""
Simple orchestrator to run the NBA predictor pipeline end-to-end.

This assumes the generated scripts have runnable top-level code. As you refactor,
consider exposing functions like `fetch_data()`, `prepare_data()`, `train_and_eval()`
and import/call them here instead.
"""

import runpy
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"

print("[1/3] Running data ingestion (src/get_data.py)...")
runpy.run_path(str(SRC / "get_data.py"))

print("[2/3] Running preprocessing (src/parse_data.py)...")
runpy.run_path(str(SRC / "parse_data.py"))

print("[3/3] Running modeling (src/predict.py)...")
runpy.run_path(str(SRC / "predict.py"))

print("✅ Pipeline finished.")
