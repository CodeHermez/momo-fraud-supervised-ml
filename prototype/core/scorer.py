"""Loading the frozen bundle, once, for every page.

``FraudScorer.load`` reads the four JSON files in ``artifacts/`` -- the model,
the fitted preprocessor, the frozen thresholds and the model card. That is the
canonical path. ``artifacts/model.joblib`` exists beside them but is a
convenience pickle written by ``python -m momo_fraud.predict``, which recorded
``FraudScorer.__module__`` as ``__main__``; loading it here would need a
``sys.modules`` shim and would tie the prototype to the exact xgboost and numpy
builds that wrote it. ``predict.py`` says the JSON bundle is the source of
truth, so the prototype takes it at its word.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

from momo_fraud.predict import FraudScorer  # noqa: E402


@st.cache_resource(show_spinner="Loading the model bundle ...")
def get_scorer() -> FraudScorer:
    """The one scorer every page shares.

    ``cache_resource`` rather than ``cache_data``: this is a live object holding
    a booster and a SHAP explainer, not a value to be copied per session.
    """
    return FraudScorer.load(ARTIFACTS_DIR, with_explainer=True)


@lru_cache(maxsize=1)
def get_model_card() -> dict:
    """What the model was trained on, how it scored, and what it cannot do."""
    return json.loads((ARTIFACTS_DIR / "model_card.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_thresholds() -> dict:
    """The frozen decision threshold, severity ratio and band cutoffs."""
    return json.loads((ARTIFACTS_DIR / "thresholds.json").read_text(encoding="utf-8"))


def artifacts_present() -> bool:
    """Whether the bundle is on disk. Pages degrade rather than traceback."""
    return all(
        (ARTIFACTS_DIR / name).exists()
        for name in ("model.json", "preprocessor.json", "thresholds.json")
    )
