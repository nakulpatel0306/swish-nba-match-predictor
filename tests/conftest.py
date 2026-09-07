"""Shared fixtures. The feature frame and backtest are built once per session."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import build_feature_frame, model_columns  # noqa: E402
from src.predict import (  # noqa: E402
    backtest,
    load_or_select_features,
    make_model,
    reconcile,
)


@pytest.fixture(scope="session")
def feature_frame():
    return build_feature_frame()


@pytest.fixture(scope="session")
def predictions(feature_frame):
    """Walk-forward predictions from the cached predictor set."""
    full = feature_frame
    candidates = model_columns(full)
    predictors = load_or_select_features(full, candidates)
    return backtest(full, make_model(), predictors)


@pytest.fixture(scope="session")
def reconciled(predictions):
    return reconcile(predictions)[0]
